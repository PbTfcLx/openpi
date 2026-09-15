#!/usr/bin/env python
"""Offline diagnostics for "is this failure a vision problem?".

Both experiments run on the local GR00T-format LeRobot datasets -- no simulator, no
rollout videos -- so a conclusion can be reached in minutes instead of by eyeballing
failed episodes and guessing.

Experiment 1 -- counterfactual ablation (``--run-ablation``)
-----------------------------------------------------------
For every sampled frame we predict an action chunk once with the full observation and
once per *destroyed* input, reusing the exact same flow-matching noise for every variant
so the comparison carries no sampling noise. Reported per variant:

* how far the predicted chunk moves away from the baseline prediction, in physical action
  units, split by action group (``delta``); and
* the offline error against the recorded ground-truth action chunk (``err``), reported next
  to a trivial ``predict_mean_reference`` row. Absolute error numbers are only meaningful
  relative to that row: if the model is not clearly better than "always predict the training
  mean", a high error says nothing about vision.

Reading it:

* ``mask_all_images`` -> almost no action change **and** no error change means the policy
  is not using vision at all. No amount of "better visual features" fixes that: the
  problem is the action head / the shortcut through the proprioceptive state, which pi05
  puts into the prompt as discretized text.
* ``mask_all_images`` -> large change *and* a large error increase means vision is used
  but something about it is wrong; that is the case worth following up with attention maps
  and probes.
* ``swap_other_episode`` (same state, same prompt, images from a different episode of the
  same task) versus ``mask_all_images`` separates "the model needs *an* image" from "the
  model actually looks at *this* image". A policy can be very sensitive to a blank image
  while still ignoring the scene content.
* ``stale_image`` (images from ``t + offset``, state and prompt from ``t``) tests whether
  the images are used as *current* state or merely as a coarse context cue.
* ``blank_prompt`` and ``zero_state`` give the language and proprioception axes, so the
  vision verdict can be compared against how much the other modalities matter.

Experiment 2 -- vision feature inspection (``--run-features``)
-------------------------------------------------------------
Runs the SigLIP tower on the same frames for every checkpoint and reports, per layer:

* a PCA->RGB rendering of the 256 patch tokens next to the input frame: crisp object
  boundaries mean the features still carry spatial/object structure, a uniform wash means
  they collapsed. Note that PCA->RGB rescales each component to the full display range, so
  it *always* looks structured - treat it as a qualitative view and read the numbers below
  for the actual collapse test;
* ``within_frame_patch_sim``: mean pairwise cosine similarity of the 256 patch tokens of
  one frame. Near 1.0 means every patch looks alike, i.e. the spatial information is gone;
* ``same_episode_sim`` / ``cross_episode_sim`` / ``cross_task_sim``: similarity of the
  frame-level feature between frames of one episode versus unrelated episodes/tasks. A
  healthy encoder separates these; overlapping distributions mean the feature barely
  encodes the scene;
* ``effective_rank`` and ``dead_channel_frac`` for collapse detection;
* ``probe_r2_per_dim``: R^2 of a cosine k-NN decoder that predicts robot-frame quantities
  (joint angles, gripper opening) from the feature of a frame drawn from a *held-out*
  trajectory. This is the "is the information still there?" test: a collapsed layer scores
  around or below 0 (= no better than predicting the mean) on every dimension, while a
  healthy one still decodes the visually salient joints. With the default sampling this probe
  is on the weak side - raise ``--num-episodes-per-task`` / ``--num-frames-per-episode``
  (e.g. 3 x 15) if you want it to be conclusive, and read it per dimension rather than as an
  average;
* with ``--compare-checkpoint-dir`` (e.g. the pretrained ``pi05_base`` tower) also
  ``cka_to_<label>`` per pair of checkpoints, quantifying how far fine-tuning moved the
  representation. Note the reference checkpoint usually needs its own ``--compare-config-name``:
  the parameter tree has to match the variant the checkpoint was trained with (LoRA vs full
  fine-tune), so plain ``pi05_base`` weights cannot be loaded with a ``*_lora`` config.

Usage::

    uv run scripts/diagnose_vision.py \\
        --checkpoint-dir checkpoints/pi05_robocasa/test_new_lr/15000 \\
        --config-name pi05_robocasa \\
        --compare-checkpoint-dir openpi-assets/checkpoints/pi05_base \\
        --compare-label pi05_base \\
        --output-dir diagnostics/test_new_lr
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import os
import pathlib
from typing import Any, Sequence

# This script only runs a handful of forward passes, so it must not grab JAX's default 75%
# of the GPU the way a training run wants to. Must be set before jax is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import cv2  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import openpi.models.model as _model  # noqa: E402
import openpi.shared.nnx_utils as nnx_utils  # noqa: E402
import openpi.shared.normalize as _normalize  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402

logger = logging.getLogger("diagnose_vision")


# --------------------------------------------------------------------------------------
# Dataset layout constants.
#
# The key order below must stay identical to ``GrootOpenpiSingleDataset.__getitem__`` in
# ``src/openpi/training/groot_openpi_dataset.py``. That is the order the model was trained
# on, so the recorded ground-truth action has to be rebuilt the same way to be comparable
# with the model output.
# --------------------------------------------------------------------------------------

STATE_KEYS = (
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "base_position",
    "base_rotation",
    "gripper_qpos",
    "joint_position",
)
ACTION_KEYS = (
    "end_effector_position",
    "end_effector_rotation",
    "gripper_close",
    "base_motion",
    "control_mode",
)
# Proprioceptive channels appended to the action vector during training (the model predicts
# them, but ``RobocasaOutputs`` only returns the first 12 dims to the environment).
ACTION_EXTRA_STATE_KEYS = ("joint_velocity", "gripper_qvel")

# Quantities a linear probe should be able to decode from the vision features if the tower
# still carries task-relevant information: the arm's joint configuration and whether the
# gripper is shut. Deliberately robot-frame rather than world-frame (``joint_position``, not
# ``end_effector_position_absolute``): robocasa randomises the kitchen layout per episode, so
# world-frame targets are out of range for a held-out episode and would make any probe look
# broken. A low score here means the features lost the information (or the images never
# reached the tower at all).
PROBE_TARGET_KEYS = ("joint_position", "gripper_qpos")

# Policy image slot -> video key in ``meta/modality.json``.
CAMERA_VIEWS = {
    "base_0_rgb": "left_view",
    "base_1_rgb": "right_view",
    "left_wrist_0_rgb": "wrist_view",
}
# Policy image slot -> observation key consumed by ``RobocasaInputs``.
CAMERA_OBS_KEYS = {
    "base_0_rgb": "observation/image",
    "base_1_rgb": "observation/right_image",
    "left_wrist_0_rgb": "observation/wrist_image",
}
# Stable camera order used when stacking frames into a ``--dump-frames`` archive.
IMAGE_SLOTS = tuple(CAMERA_VIEWS)

# Action groups in *training* action order (see ``ACTION_KEYS``).
ACTION_GROUPS: dict[str, tuple[int, int]] = {
    "ee_pos": (0, 3),
    "ee_rot": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}
N_ENV_ACTION_DIM = 12  # action dim the environment consumes

ALL_VARIANTS = (
    "baseline",
    "mask_all_images",
    "mask_base",
    "mask_wrist",
    "swap_other_episode",
    "stale_image",
    "blank_prompt",
    "zero_state",
)

# Synthetic reference row in the error panel: what the error would be if the model simply
# predicted the training mean (zero in normalized space). An absolute error figure is only
# meaningful next to it.
PREDICT_MEAN_KEY = "predict_mean_reference"

DEFAULT_FEATURE_LAYERS = ("with_posemb", "block13", "block20", "encoded", "tokens")

DATASET_PREFIX = "single_panda_gripper."


# --------------------------------------------------------------------------------------
# Args
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    """Diagnose whether a policy actually uses its visual input."""

    # Training config the checkpoint belongs to (e.g. "pi05_robocasa").
    config_name: str = "pi05_robocasa"
    # Checkpoint directory containing ``params/`` and ``assets/norm_stats.json``. An
    # experiment directory with numeric step subdirectories is also accepted.
    checkpoint_dir: str = "checkpoints/pi05_robocasa/test_new_lr/15000"
    # Optional reference checkpoint to compare the vision tower against, e.g. the
    # pretrained ``openpi-assets/checkpoints/pi05_base``. Adds a second PCA row and CKA.
    compare_checkpoint_dir: str | None = None
    # Config used to instantiate the reference checkpoint. Defaults to ``config_name``.
    compare_config_name: str | None = None
    # Label for the reference checkpoint in figures/JSON.
    compare_label: str = "reference"

    # Where figures and JSON summaries are written.
    output_dir: str = "diagnostics/vision"
    # Dataset names to sample from. ``None`` uses every ``single_panda_gripper.*`` dataset
    # found in ``HF_LEROBOT_HOME``.
    tasks: list[str] | None = None
    # Episodes sampled per task, spread evenly across the dataset. Needs to be >= 2 for
    # ``cross_episode_sim`` to be populated in the feature report.
    num_episodes_per_task: int = 2
    # Frames sampled per episode, spread evenly over the episode.
    num_frames_per_episode: int = 6
    # Frame shift used by the ``stale_image`` variant.
    stale_offset: int = 20

    # Data-preparation-only mode: sample the frames, write them to this ``.npz`` and exit
    # WITHOUT loading a model. Reading the parquet/mp4 files and packing the raw inputs is
    # pure CPU work, so this can run on a machine with no free GPU. The archive holds raw
    # uint8 frames, the float32 state and the prompt text (never normalized tensors), so a
    # reader still applies its own checkpoint's ``Normalize`` and the file stays valid for
    # any checkpoint trained on the same data.
    dump_frames: str | None = None
    # Read the frames from an ``.npz`` written by ``--dump-frames`` instead of from the
    # LeRobot datasets. Requires no dataset access at all (``$HF_LEROBOT_HOME`` is then
    # unused) and guarantees every checkpoint sees bit-identical inputs, which is what makes
    # two runs comparable even when the datasets differ. ``--tasks``,
    # ``--num-episodes-per-task``, ``--num-frames-per-episode`` and ``--stale-offset`` are
    # ignored: the sampling decisions are already baked into the archive.
    frames_from: str | None = None

    # Which experiments to run.
    run_ablation: bool = True
    run_features: bool = True
    # Occlusion sensitivity: hide one image region at a time and measure how much the
    # predicted action moves. Produces a heatmap that can be judged by eye.
    run_occlusion: bool = False
    # Occlusion grid resolution (``grid`` x ``grid`` regions per image).
    occlusion_grid: int = 6
    # How many frames to build occlusion maps for.
    occlusion_frames: int = 8
    # Which cameras to occlude: "camera" (only --camera), "base", "wrist", or "all".
    occlusion_target: str = "all"
    # Upper end of the (absolute) colour scale, in units of the baseline action spread.
    # Keep it fixed rather than per-figure: an auto-scaled colour map stretches a flat, dead
    # map to the full colour range and makes it look structured.
    occlusion_vmax: float = 0.5

    # Ablation variants to run, a subset of ALL_VARIANTS. Empty means all of them.
    variants: list[str] = dataclasses.field(default_factory=list)
    # Frames pushed through the model per jit call.
    batch_size: int = 8
    # Flow-matching denoising steps.
    num_steps: int = 10
    # Seed for the fixed noise shared by all ablation variants.
    seed: int = 0

    # Camera slot whose SigLIP features are visualised, one of CAMERA_VIEWS.
    camera: str = "base_0_rgb"
    # Layers to visualise, e.g. "stem", "with_posemb", "block13", "encoded", "tokens".
    feature_layers: list[str] = dataclasses.field(default_factory=list)
    # Maximum number of frames per feature figure.
    max_frames_in_figure: int = 10

    def __post_init__(self) -> None:
        if not self.variants:
            self.variants = list(ALL_VARIANTS)
        if not self.feature_layers:
            self.feature_layers = list(DEFAULT_FEATURE_LAYERS)
        if self.camera not in CAMERA_VIEWS:
            raise ValueError(f"Unknown camera {self.camera!r}; expected one of {list(CAMERA_VIEWS)}")
        unknown = [v for v in self.variants if v not in ALL_VARIANTS]
        if unknown:
            raise ValueError(f"Unknown variants {unknown}; expected a subset of {list(ALL_VARIANTS)}")
        if "baseline" not in self.variants:
            self.variants = ["baseline", *self.variants]
        if self.occlusion_grid < 2:
            raise ValueError("--occlusion-grid must be at least 2")
        if self.occlusion_target not in ("camera", "base", "wrist", "all"):
            raise ValueError("--occlusion-target must be one of camera/base/wrist/all")


# --------------------------------------------------------------------------------------
# Dataset reading
# --------------------------------------------------------------------------------------


def dataset_root() -> pathlib.Path:
    home = os.environ.get("HF_LEROBOT_HOME")
    if not home:
        raise RuntimeError("HF_LEROBOT_HOME is not set; cannot locate the LeRobot datasets.")
    return pathlib.Path(home)


def discover_tasks() -> list[str]:
    root = dataset_root()
    return sorted(
        p.name[len(DATASET_PREFIX) :]
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith(DATASET_PREFIX)
    )


class TaskDataset:
    """Direct reader for one GR00T-format LeRobot v2 dataset directory.

    Deliberately bypasses ``lerobot``/``LeRobotSingleDataset`` so the diagnostic can sample
    arbitrary (episode, step) pairs and read the raw frames that go into the policy without
    any training-time sampling machinery in the way.
    """

    def __init__(self, root: pathlib.Path):
        self.root = root
        self.task_name = root.name.split(".", 1)[1]
        self.info = json.loads((root / "meta" / "info.json").read_text())
        self.modality = json.loads((root / "meta" / "modality.json").read_text())
        self.tasks = [
            json.loads(line)["task"]
            for line in (root / "meta" / "tasks.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.episodes = [
            json.loads(line)
            for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self._chunks_size = int(self.info.get("chunks_size", 1000))
        self._episode_cache: dict[int, dict[str, Any]] = {}
        self._video_cache: dict[tuple[int, str], np.ndarray] = {}
        self._state_slices = self._modality_slices("state", STATE_KEYS)
        self._action_slices = self._modality_slices("action", ACTION_KEYS)
        self._extra_slices = self._modality_slices("state", ACTION_EXTRA_STATE_KEYS)
        self._probe_slices = self._modality_slices("state", PROBE_TARGET_KEYS)

    # -- paths -------------------------------------------------------------------------
    def _parquet_path(self, ep: int) -> pathlib.Path:
        return self.root / self.info["data_path"].format(
            episode_chunk=ep // self._chunks_size, episode_index=ep
        )

    def _video_path(self, ep: int, view: str) -> pathlib.Path:
        return self.root / self.info["video_path"].format(
            episode_chunk=ep // self._chunks_size,
            video_key=f"observation.images.{view}",
            episode_index=ep,
        )

    def _modality_slices(self, group: str, keys: Sequence[str]) -> list[tuple[str, int, int]]:
        table = self.modality[group]
        missing = [k for k in keys if k not in table]
        if missing:
            raise KeyError(f"{self.root.name}: modality.json[{group!r}] is missing {missing}")
        return [(k, int(table[k]["start"]), int(table[k]["end"])) for k in keys]

    # -- data --------------------------------------------------------------------------
    def episode(self, ep: int) -> dict[str, Any]:
        if ep in self._episode_cache:
            return self._episode_cache[ep]
        import pyarrow.parquet as pq

        table = pq.read_table(self._parquet_path(ep))

        def column(name: str) -> np.ndarray:
            return np.asarray(table.column(name).to_pylist())

        episode = {
            "state": column("observation.state"),
            "action": column("action"),
            "description_index": column("annotation.human.action.task_description").astype(int),
        }
        episode["length"] = int(episode["state"].shape[0])
        self._episode_cache[ep] = episode
        return episode

    def frames(self, ep: int, view: str) -> np.ndarray:
        """Decodes and caches every frame of one camera of one episode as RGB uint8."""
        key = (ep, view)
        if key in self._video_cache:
            return self._video_cache[key]
        path = self._video_path(ep, view)
        if not path.exists():
            raise FileNotFoundError(f"Missing video for view {view!r}: {path}")
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame[:, :, ::-1])  # cv2 decodes BGR; the policy was trained on RGB
        cap.release()
        if not frames:
            raise RuntimeError(f"Could not decode any frame from {path}")
        arr = np.stack(frames)
        self._video_cache[key] = arr
        return arr

    # -- convenience -------------------------------------------------------------------
    def has_view(self, view: str) -> bool:
        return view in self.modality.get("video", {})

    def first_view(self) -> str:
        return next(view for view in CAMERA_VIEWS.values() if self.has_view(view))

    def usable_length(self, ep: int) -> int:
        """Frames available from both the parquet and every decoded video."""
        lengths = [self.episode(ep)["length"]]
        lengths += [self.frames(ep, view).shape[0] for view in CAMERA_VIEWS.values() if self.has_view(view)]
        return int(min(lengths))

    def released_steps(self, ep: int, horizon: int, count: int) -> np.ndarray:
        """Steps of ``ep`` whose ground-truth action chunk fits inside the episode."""
        last = self.usable_length(ep) - horizon - 1
        if last <= 1:
            return np.array([], dtype=int)
        first = max(1, int(0.03 * last))
        return np.unique(np.linspace(first, last, count).astype(int))

    def policy_state(self, ep: int, step: int) -> np.ndarray:
        state = self.episode(ep)["state"][step]
        return np.concatenate([state[s:e] for _, s, e in self._state_slices]).astype(np.float32)

    def gt_action_chunk(self, ep: int, step: int, horizon: int) -> np.ndarray:
        """Ground-truth action chunk in training order, shape ``(horizon, 21)``."""
        episode = self.episode(ep)
        action = episode["action"][step : step + horizon]
        state = episode["state"][step : step + horizon]
        parts = [action[:, s:e] for _, s, e in self._action_slices]
        parts += [state[:, s:e] for _, s, e in self._extra_slices]
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def prompt(self, ep: int, step: int) -> str:
        return self.tasks[int(self.episode(ep)["description_index"][step])]

    def probe_target(self, ep: int, step: int) -> np.ndarray:
        """Low-dimensional state used as a linear-probe label, see ``PROBE_TARGET_KEYS``."""
        state = self.episode(ep)["state"][step]
        return np.concatenate([state[s:e] for _, s, e in self._probe_slices]).astype(np.float32)

    def episode_description(self, ep: int) -> str:
        """Prompt of an episode, from ``episodes.jsonl`` (no parquet read needed)."""
        tasks = self.episodes[ep].get("tasks") or []
        return str(tasks[0]) if tasks else ""

    def different_episode_same_task(self, ep: int) -> int:
        """Another episode with the same instruction, so a scene swap stays well posed."""
        description = self.episode_description(ep)
        candidates = [
            i for i in range(len(self.episodes)) if i != ep and self.episode_description(i) == description
        ]
        if candidates:
            return candidates[len(candidates) // 2]
        return (ep + 1) % len(self.episodes)

    def free(self) -> None:
        self._episode_cache.clear()
        self._video_cache.clear()


# --------------------------------------------------------------------------------------
# Sample collection
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class SamplePoint:
    """One evaluation frame plus the pre-rendered observations of the raw-level variants."""

    task: str
    episode: int
    step: int
    gt_chunk: np.ndarray  # (horizon, 21), training action order
    probe: np.ndarray  # low-dimensional state used as a linear-probe label
    probe_names: list[str]
    obs: dict[str, Any]  # raw observation for ``RobocasaInputs``
    variant_obs: dict[str, dict[str, Any]]
    key: str


def images_at(dataset: TaskDataset, ep: int, step: int) -> dict[str, np.ndarray]:
    """RGB frames of all cameras at one (episode, step), clamped into range."""
    step = int(np.clip(step, 0, dataset.usable_length(ep) - 1))
    return {
        slot: dataset.frames(ep, view)[step] if dataset.has_view(view) else np.zeros((224, 224, 3), np.uint8)
        for slot, view in CAMERA_VIEWS.items()
    }


def pack_raw_obs(state: np.ndarray, images: dict[str, np.ndarray], prompt: str) -> dict[str, Any]:
    """Raw observation dict consumed by ``RobocasaInputs``.

    Kept dataset-free so a ``--dump-frames`` archive can rebuild exactly the same structure
    that ``collect_samples`` produced from the parquet/video files.
    """
    obs: dict[str, Any] = {"prompt": prompt, "observation/state": state}
    for slot, obs_key in CAMERA_OBS_KEYS.items():
        obs[obs_key] = images[slot]
    return obs


def build_raw_obs(
    dataset: TaskDataset, ep: int, step: int, images: dict[str, np.ndarray], prompt: str
) -> dict[str, Any]:
    return pack_raw_obs(dataset.policy_state(ep, step), images, prompt)


def collect_samples(args: Args, horizon: int) -> list[SamplePoint]:
    tasks = args.tasks or discover_tasks()
    root = dataset_root()
    samples: list[SamplePoint] = []

    for task in tasks:
        dataset = TaskDataset(root / f"{DATASET_PREFIX}{task}")
        n_episodes = len(dataset.episodes)
        episode_ids = [
            int(e) for e in np.linspace(0, n_episodes - 1, min(args.num_episodes_per_task, n_episodes))
        ]
        n_task = 0

        for ep in episode_ids:
            steps = dataset.released_steps(ep, horizon, args.num_frames_per_episode)
            if len(steps) == 0:
                logger.warning("%s episode %d too short for horizon %d; skipping", task, ep, horizon)
                continue
            swap_ep = dataset.different_episode_same_task(ep)
            probe_names = [
                f"{key}[{i}]" for key, start, end in dataset._probe_slices for i in range(end - start)  # noqa: SLF001
            ]

            for step in steps:
                step = int(step)
                images = images_at(dataset, ep, step)
                prompt = dataset.prompt(ep, step)
                obs = build_raw_obs(dataset, ep, step, images, prompt)

                # ``swap_other_episode`` isolates the scene content: state and prompt stay
                # this frame's, only the pixels come from another episode of the same task.
                last_step = max(1, dataset.usable_length(ep) - horizon - 1)
                progress = step / last_step
                swap_step = int(round(progress * last_step))
                stale_step = min(dataset.usable_length(ep) - 1, step + args.stale_offset)

                variant_obs = {
                    "blank_prompt": build_raw_obs(dataset, ep, step, images, ""),
                    "swap_other_episode": build_raw_obs(
                        dataset, ep, step, images_at(dataset, swap_ep, swap_step), prompt
                    ),
                    "stale_image": build_raw_obs(
                        dataset, ep, step, images_at(dataset, ep, stale_step), prompt
                    ),
                }
                samples.append(
                    SamplePoint(
                        task=task,
                        episode=ep,
                        step=step,
                        gt_chunk=dataset.gt_action_chunk(ep, step, horizon),
                        probe=dataset.probe_target(ep, step),
                        probe_names=probe_names,
                        obs=obs,
                        variant_obs=variant_obs,
                        key=f"{task}/ep{ep}/t{step}",
                    )
                )
                n_task += 1
        dataset.free()
        logger.info("sampled %s: %d frames", task, n_task)
    return samples


# --------------------------------------------------------------------------------------
# Frame archives (run the ablation on a machine that has no dataset)
# --------------------------------------------------------------------------------------


def resolve_horizon(config_name: str) -> int:
    """Action horizon of a config, without instantiating the model (so: no GPU, no weights)."""
    return int(_config.get_config(config_name).model.action_horizon)


def _archive_meta(args: Args, samples: Sequence[SamplePoint]) -> dict[str, Any]:
    return {
        "created_by": "scripts/diagnose_vision.py --dump-frames",
        "config_name": args.config_name,
        "checkpoint_dir": args.checkpoint_dir,
        "tasks": sorted({s.task for s in samples}),
        "num_frames": len(samples),
        "episodes": {t: sorted({s.episode for s in samples if s.task == t}) for t in sorted({s.task for s in samples})},
        "num_episodes_per_task": args.num_episodes_per_task,
        "num_frames_per_episode": args.num_frames_per_episode,
        "stale_offset": args.stale_offset,
        "image_slots": list(IMAGE_SLOTS),
    }


def dump_samples(samples: Sequence[SamplePoint], path: str, args: Args) -> None:
    """Write the sampled frames to an ``.npz`` so another machine needs no dataset.

    Stores inputs only (uint8 frames, float32 state, prompt text) for the three raw-level
    observation variants, plus the ground-truth chunk and the probe target. The remaining
    variants (``mask_*``, ``zero_state``, occlusion) are applied to the normalized data at
    run time, so they keep working and new ones can still be added later.
    """

    def slots_of(obs: dict[str, Any]) -> list[np.ndarray]:
        return [np.ascontiguousarray(obs[CAMERA_OBS_KEYS[slot]]) for slot in IMAGE_SLOTS]

    baseline = [s.obs for s in samples]
    logger.info("stacking %d frames...", len(samples))
    np.savez_compressed(
        pathlib.Path(path),
        key=np.array([s.key for s in samples]),
        task=np.array([s.task for s in samples]),
        episode=np.array([s.episode for s in samples]),
        step=np.array([s.step for s in samples]),
        prompt=np.array([str(o["prompt"]) for o in baseline]),
        state=np.stack([np.asarray(o["observation/state"], dtype=np.float32) for o in baseline]),
        gt_chunk=np.stack([np.asarray(s.gt_chunk, dtype=np.float32) for s in samples]),
        probe=np.stack([np.asarray(s.probe, dtype=np.float32) for s in samples]),
        probe_names=np.array([str(n) for n in samples[0].probe_names]),
        images=np.stack([slots_of(o) for o in baseline]),
        swap_images=np.stack([slots_of(s.variant_obs["swap_other_episode"]) for s in samples]),
        stale_images=np.stack([slots_of(s.variant_obs["stale_image"]) for s in samples]),
        meta_json=np.array(json.dumps(_archive_meta(args, samples), indent=2)),
    )
    out = pathlib.Path(path)
    logger.info("dumped %d frames to %s (%.1f MiB)", len(samples), out.resolve(), out.stat().st_size / 2**20)


def load_samples(path: str) -> list[SamplePoint]:
    """Rebuild ``SamplePoint``s from an archive written by ``--dump-frames``."""
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta_json"]))
        logger.info("frame archive: %s", json.dumps(meta, indent=2))
        slots = tuple(meta["image_slots"])
        if set(slots) != set(IMAGE_SLOTS):
            raise ValueError(f"archive image slots {slots} do not match this script's {IMAGE_SLOTS}")
        keys, tasks, episodes, steps = archive["key"], archive["task"], archive["episode"], archive["step"]
        prompts, states = archive["prompt"], archive["state"]
        gt_chunks, probes, probe_names = archive["gt_chunk"], archive["probe"], archive["probe_names"]
        images, swap_images, stale_images = archive["images"], archive["swap_images"], archive["stale_images"]
        names = [str(n) for n in probe_names]

        samples: list[SamplePoint] = []
        for i in range(len(keys)):
            state, prompt = states[i], str(prompts[i])
            base = {slot: images[i, j] for j, slot in enumerate(slots)}
            swap = {slot: swap_images[i, j] for j, slot in enumerate(slots)}
            stale = {slot: stale_images[i, j] for j, slot in enumerate(slots)}
            samples.append(
                SamplePoint(
                    task=str(tasks[i]),
                    episode=int(episodes[i]),
                    step=int(steps[i]),
                    gt_chunk=gt_chunks[i],
                    probe=probes[i],
                    probe_names=list(names),
                    obs=pack_raw_obs(state, base, prompt),
                    variant_obs={
                        "blank_prompt": pack_raw_obs(state, base, ""),
                        "swap_other_episode": pack_raw_obs(state, swap, prompt),
                        "stale_image": pack_raw_obs(state, stale, prompt),
                    },
                    key=str(keys[i]),
                )
            )
    return samples


# --------------------------------------------------------------------------------------
# Transform pipeline
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class VariantTransform(_transforms.DataTransformFn):
    """Counterfactual input edit applied to already-normalized data.

    Sits between ``Normalize`` and the model transforms, the one place where ``state`` is
    the normalized proprioceptive vector (so zeroing it means "the training mean") and the
    images are still uint8 HWC frames whose ``image_mask`` can be switched off. Masking an
    image removes its 256 patch tokens from attention entirely (``make_attn_mask`` requires
    the key token to be a valid input), which is a stronger ablation than zeroing pixels.

    ``occlusion`` additionally blanks one rectangular region (of a ``grid`` x ``grid`` tiling)
    in the selected cameras. Note this changes the pixels but keeps the token mask on: the
    model sees a black patch, it does not lose the tokens. The images are copied before
    editing because ``RobocasaInputs`` hands out the very arrays held by the sample.
    """

    variant: str = "baseline"
    occlusion: tuple[int, int, int] | None = None
    occlusion_slots: tuple[str, ...] = ()

    def __call__(self, data: dict) -> dict:
        if self.occlusion is not None:
            self._occlude(data, *self.occlusion)
        match self.variant:
            case "mask_all_images":
                self._mask(data, tuple(CAMERA_VIEWS))
            case "mask_base":
                self._mask(data, ("base_0_rgb", "base_1_rgb"))
            case "mask_wrist":
                self._mask(data, ("left_wrist_0_rgb",))
            case "zero_state":
                data["state"] = np.zeros_like(data["state"])
            case _:
                pass
        return data

    def _occlude(self, data: dict, row: int, col: int, grid: int) -> None:
        for slot in self.occlusion_slots:
            image = data["image"][slot]
            height, width = image.shape[:2]
            y0, y1 = int(round(row * height / grid)), int(round((row + 1) * height / grid))
            x0, x1 = int(round(col * width / grid)), int(round((col + 1) * width / grid))
            blanked = np.array(image, copy=True)
            blanked[y0:y1, x0:x1] = 0
            data["image"][slot] = blanked

    @staticmethod
    def _mask(data: dict, slots: Sequence[str]) -> None:
        for slot in slots:
            data["image"][slot] = np.zeros_like(data["image"][slot])
            data["image_mask"][slot] = np.False_


def make_input_transform(
    data_config: _config.DataConfig,
    norm_stats,
    variant: str,
    occlusion: tuple[int, int, int] | None = None,
    occlusion_slots: Sequence[str] | None = None,
):
    """Rebuilds the input transform ``Policy`` uses, with an ablation hook inserted.

    Deliberately mirrors ``create_trained_policy``'s list, which means none of the
    training-only regularisers apply here: ``RandomPromptDrop`` and
    ``InterpolatedStateNoise`` are injected by the data loader (see
    ``training/data_loader.py``), never by the policy, so inference is unaffected by
    ``prompt_drop_p`` / ``state_noise`` in the config.
    """
    slots = tuple(CAMERA_VIEWS) if occlusion_slots is None else tuple(occlusion_slots)
    return _transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            _transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats),
            VariantTransform(variant, occlusion=occlusion, occlusion_slots=slots),
            *data_config.model_transforms.inputs,
        ]
    )


def make_output_transform(data_config: _config.DataConfig, norm_stats):
    return _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats),
            *data_config.data_transforms.outputs,
            *data_config.repack_transforms.outputs,
        ]
    )


def to_batch(transformed: Sequence[dict]) -> dict:
    return jax.tree.map(lambda *xs: np.stack(xs, axis=0), *transformed)


# --------------------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class LoadedModel:
    label: str
    checkpoint_dir: pathlib.Path
    config: _config.TrainConfig
    model: Any
    data_config: _config.DataConfig
    norm_stats: Any | None


def _data_config_for(config: _config.TrainConfig) -> _config.DataConfig:
    if isinstance(config.data, _config.LeRobotRobocasaDataConfig):
        return config.data.create(config.assets_dirs, config.model, load_norm_stats=False)
    return config.data.create(config.assets_dirs, config.model)


def resolve_checkpoint_dir(checkpoint_dir: str) -> pathlib.Path:
    """Resolves a checkpoint path, also against ``$OPENPI_DATA_HOME``.

    ``OPENPI_DATA_HOME`` is where ``download.maybe_download`` maps the ``gs://openpi-assets``
    paths used in the training configs, so ``openpi-assets/checkpoints/pi05_base`` works
    even though it does not live next to the repository.
    """
    candidate = pathlib.Path(checkpoint_dir).expanduser()
    if candidate.exists():
        return candidate
    data_home = os.environ.get("OPENPI_DATA_HOME")
    if data_home:
        alternative = pathlib.Path(data_home).expanduser() / checkpoint_dir
        if alternative.exists():
            return alternative
    raise FileNotFoundError(
        f"Checkpoint directory {checkpoint_dir!r} does not exist "
        f"(also tried {'$OPENPI_DATA_HOME/' + checkpoint_dir if data_home else 'no OPENPI_DATA_HOME'})"
    )


def load_model(checkpoint_dir: str, config_name: str, label: str) -> LoadedModel:
    train_config = _config.get_config(config_name)
    ckpt = resolve_checkpoint_dir(checkpoint_dir)
    if not (ckpt / "params").is_dir():
        steps = [(int(p.name), p) for p in ckpt.iterdir() if p.is_dir() and p.name.isdigit()]
        if not steps:
            raise FileNotFoundError(f"No params/ and no numeric step directories under {ckpt}")
        ckpt = max(steps)[1]
        logger.info("[%s] resolved %s to step %s", label, checkpoint_dir, ckpt.name)

    logger.info("[%s] loading params from %s", label, ckpt / "params")
    try:
        model = train_config.model.load(_model.restore_params(ckpt / "params", dtype=jnp.bfloat16))
    except ValueError as exc:
        raise ValueError(
            f"[{label}] the parameter tree of {ckpt / 'params'} does not match config "
            f"{config_name!r}.\n"
            f"{exc}\n\n"
            "The config's model variant has to match the variant the checkpoint was trained "
            "with. Mixing a LoRA config with a full-fine-tune checkpoint (or the other way "
            "round, e.g. loading the plain pi05_base weights with a *_lora config) produces a "
            "structural mismatch like the one above. Pass --compare-config-name to give the "
            "reference checkpoint its own config."
        ) from exc

    norm_stats = None
    if (ckpt / "assets").is_dir():
        try:
            norm_stats = _normalize.load(ckpt / "assets")
            logger.info("[%s] loaded norm stats from %s", label, ckpt / "assets")
        except FileNotFoundError:
            logger.warning("[%s] no norm_stats.json under %s", label, ckpt / "assets")
    else:
        logger.warning("[%s] no assets/ directory; normalization will be a no-op", label)

    return LoadedModel(
        label=label,
        checkpoint_dir=ckpt,
        config=train_config,
        model=model,
        data_config=_data_config_for(train_config),
        norm_stats=norm_stats,
    )


# --------------------------------------------------------------------------------------
# Experiment 1: counterfactual ablation
# --------------------------------------------------------------------------------------


def noise_for(key: str, horizon: int, action_dim: int, seed: int) -> np.ndarray:
    """Deterministic per-frame flow-matching noise, identical across every variant."""
    digest = jax.random.key(seed)
    for byte in key.encode():
        digest = jax.random.fold_in(digest, int(byte))
    return np.asarray(jax.random.normal(digest, (horizon, action_dim)), dtype=np.float32)


def group_means(per_dim: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | None]:
    """Mean of ``per_dim`` per action group; ``None`` for a fully excluded group."""
    out: dict[str, float | None] = {}
    for name, (start, end) in ACTION_GROUPS.items():
        if mask is None:
            out[name] = float(np.mean(per_dim[start:end]))
            continue
        selected = mask[start:end]
        out[name] = float(np.mean(per_dim[start:end][selected])) if selected.any() else None
    return out


def run_ablation(args: Args, loaded: LoadedModel, samples: Sequence[SamplePoint]) -> dict[str, Any]:
    horizon = loaded.config.model.action_horizon
    action_dim = loaded.config.model.action_dim
    sample_actions = nnx_utils.module_jit(loaded.model.sample_actions)
    rng = jax.random.key(args.seed)

    transforms = {
        variant: make_input_transform(loaded.data_config, loaded.norm_stats, variant)
        for variant in args.variants
    }
    output_transform = make_output_transform(loaded.data_config, loaded.norm_stats)
    noise = np.stack([noise_for(s.key, horizon, action_dim, args.seed) for s in samples])

    if loaded.norm_stats is not None:
        action_mean = np.asarray(loaded.norm_stats["actions"].mean)
        action_std = np.asarray(loaded.norm_stats["actions"].std)
    else:
        action_mean = np.zeros(action_dim, dtype=np.float32)
        action_std = np.ones(action_dim, dtype=np.float32)

    # Dimensions whose normalization std is zero carry no signal at all: robocasa's fixed
    # robot base has `base_motion` and `control_mode` identically zero in every dataset, so
    # the normalizer maps them to 0/(0 + 1e-6) = 0. Comparing the model's raw output for
    # those dims against 0 in normalized space is meaningless (any output maps back to ~0 in
    # physical units), and including them would dominate the aggregate error.
    informative = action_std[:N_ENV_ACTION_DIM] > 1e-6
    excluded_dims = np.flatnonzero(~informative).tolist()

    n_gt = samples[0].gt_chunk.shape[-1]
    gt_norm = np.stack(
        [
            (s.gt_chunk - action_mean[: s.gt_chunk.shape[-1]])
            / (action_std[: s.gt_chunk.shape[-1]] + 1e-6)
            for s in samples
        ]
    )

    preds: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    phys: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    for start in range(0, len(samples), args.batch_size):
        chunk = samples[start : start + args.batch_size]
        for variant in args.variants:
            raw = [s.variant_obs.get(variant, s.obs) for s in chunk]
            batch = to_batch([transforms[variant](obs) for obs in raw])
            observation = _model.Observation.from_dict(batch)
            actions = sample_actions(
                rng,
                observation,
                noise=jnp.asarray(noise[start : start + len(chunk)]),
                num_steps=args.num_steps,
            )
            pred = np.asarray(actions, dtype=np.float32)
            preds[variant].append(pred)
            dummy_state = np.zeros(pred.shape[:2] + (action_dim,), dtype=np.float32)
            out = output_transform({"state": dummy_state, "actions": pred})
            phys[variant].append(np.asarray(out["actions"], dtype=np.float32))
        logger.info("  ablation %d/%d frames", min(start + len(chunk), len(samples)), len(samples))

    predictions = {k: np.concatenate(v, axis=0) for k, v in preds.items()}
    physical = {k: np.concatenate(v, axis=0) for k, v in phys.items()}

    base_norm = predictions["baseline"]
    base_phys = physical["baseline"]

    # Natural spread of the baseline predictions: if a counterfactual moves the action by
    # less than this, the change is within the policy's own frame-to-frame variability.
    spread = np.std(base_phys.reshape(-1, base_phys.shape[-1]), axis=0)

    results: dict[str, Any] = {}
    for variant in args.variants:
        delta_phys = np.abs(physical[variant] - base_phys).mean(axis=(0, 1))
        err_norm = np.abs(predictions[variant][..., :n_gt] - gt_norm).mean(axis=(0, 1))
        # A change is only meaningful relative to how much the policy's own output moves
        # across frames, so express the delta in units of that natural spread.
        spread_sel = spread[:N_ENV_ACTION_DIM]
        ratio = np.zeros(N_ENV_ACTION_DIM, dtype=np.float64)
        ratio[informative] = delta_phys[:N_ENV_ACTION_DIM][informative] / (
            spread_sel[informative] + 1e-9
        )
        results[variant] = {
            "delta_phys_per_dim": delta_phys.tolist(),
            "delta_phys_per_group": group_means(delta_phys, informative),
            "delta_phys_mean": float(np.mean(delta_phys[:N_ENV_ACTION_DIM][informative])),
            "delta_phys_over_spread": float(np.mean(ratio)),
            "delta_norm_mean": float(np.abs(predictions[variant][..., :n_gt] - base_norm[..., :n_gt]).mean()),
            "err_norm_per_dim": err_norm.tolist(),
            "err_norm_per_group": group_means(err_norm, informative),
            "err_norm_mean": float(np.mean(err_norm[:N_ENV_ACTION_DIM][informative])),
        }

    # What the error would be for a model that always predicts the training mean. High
    # absolute errors have to be read against this: "error 0.55" means very different things
    # depending on whether the trivial predictor scores 0.61 or 0.20.
    mean_err_norm = np.abs(gt_norm).mean(axis=(0, 1))
    results[PREDICT_MEAN_KEY] = {
        "delta_phys_per_dim": [0.0] * len(spread),
        "delta_phys_per_group": group_means(np.zeros_like(spread), informative),
        "delta_phys_mean": 0.0,
        "delta_phys_over_spread": 0.0,
        "delta_norm_mean": 0.0,
        "err_norm_per_dim": mean_err_norm.tolist(),
        "err_norm_per_group": group_means(mean_err_norm, informative),
        "err_norm_mean": float(np.mean(mean_err_norm[:N_ENV_ACTION_DIM][informative])),
    }

    base_err = results["baseline"]["err_norm_mean"]
    for variant in args.variants:
        results[variant]["err_norm_over_baseline"] = results[variant]["err_norm_mean"] / (base_err + 1e-9)

    summary = {
        "num_frames": len(samples),
        "action_horizon": horizon,
        "num_steps": args.num_steps,
        "scored_action_dims": np.flatnonzero(informative).tolist(),
        "excluded_action_dims": excluded_dims,
        "excluded_action_dims_reason": (
            "normalization std is zero (these channels are identically zero in the training "
            "data), so the normalized error is not defined for them"
        ),
        "baseline_action_spread": spread.tolist(),
        "baseline_err_norm_mean": base_err,
        "predict_mean_err_norm_mean": results[PREDICT_MEAN_KEY]["err_norm_mean"],
        "baseline_error_vs_predict_mean": base_err
        / (results[PREDICT_MEAN_KEY]["err_norm_mean"] + 1e-9),
        "variants": results,
    }
    summary["verdict"] = ablation_verdict(results)
    return summary


def ablation_verdict(results: dict[str, Any]) -> dict[str, Any]:
    """Turns the ablation numbers into a decision-relevant reading."""
    pairs = (
        ("vision", "mask_all_images"),
        ("base_camera", "mask_base"),
        ("wrist_camera", "mask_wrist"),
        ("scene_content", "swap_other_episode"),
        ("temporal_alignment", "stale_image"),
        ("language", "blank_prompt"),
        ("proprioception", "zero_state"),
    )
    reliance = {k: results[v]["delta_phys_over_spread"] for k, v in pairs if v in results}
    verdict: dict[str, Any] = {"reliance_over_baseline_spread": reliance}

    if "baseline" in results and PREDICT_MEAN_KEY in results:
        ratio = results["baseline"]["err_norm_mean"] / (results[PREDICT_MEAN_KEY]["err_norm_mean"] + 1e-9)
        verdict["baseline_error_vs_predict_mean"] = ratio
        if ratio > 0.95:
            detail = (
                "That is no better than always predicting the training mean, so the action "
                "head has not even fitted the recorded behaviour. Any 'is it vision?' "
                "question is premature until the fit itself improves."
            )
        elif ratio > 0.7:
            detail = (
                "The fit is only slightly better than a constant, so expect the modality "
                "verdicts below to be noisy."
            )
        else:
            detail = (
                "The fit is meaningfully better than a constant, so the action head is "
                "learning something and the modality verdicts below are trustworthy."
            )
        verdict["fit_quality"] = (
            f"The model's offline error is {ratio:.2f}x the error of predicting the training mean. "
            + detail
        )

    vision = reliance.get("vision")
    if vision is not None:
        if vision < 0.15:
            verdict["vision"] = (
                "UNUSED: removing the images barely moves the action, so this is not a "
                "visual-feature problem - the policy is effectively proprioception-only. Check "
                "the action head, the delta-action parameterisation and the state-in-prompt "
                "shortcut before touching the vision tower."
            )
        elif vision < 0.5:
            verdict["vision"] = (
                "WEAK: the images influence the action, but much less than the natural spread "
                "of the predictions. Vision is at best a minor correction term."
            )
        else:
            verdict["vision"] = (
                "USED: the images strongly influence the action. If the policy still fails, the "
                "problem is *what* it looks at or *what* it extracts, not whether it looks - "
                "follow up with the feature PCA and the attention maps."
            )

    if "mask_all_images" in results:
        ratio = results["mask_all_images"]["err_norm_over_baseline"]
        verdict["error_cost_of_masking_vision"] = ratio
        if ratio < 1.05:
            verdict["vision_error"] = (
                f"Masking the images does not worsen the offline action error (ratio {ratio:.2f}). "
                "The model reproduces the recorded actions without vision, which is the clearest "
                "possible sign of a shortcut rather than a representation problem."
            )
        elif ratio > 1.5:
            verdict["vision_error"] = (
                f"Masking the images makes the offline action error much worse (ratio {ratio:.2f}), "
                "so the policy does rely on the images to fit the data."
            )

    if "swap_other_episode" in results and vision is not None:
        scene = results["swap_other_episode"]["delta_phys_over_spread"]
        verdict["scene_content_ratio"] = scene
        if vision > 0.5 and scene < 0.2 * vision:
            verdict["scene_content"] = (
                "The policy reacts to the presence of an image but barely reacts to *which* scene "
                "it is, so the visual content is largely being ignored."
            )
    return verdict


# --------------------------------------------------------------------------------------
# Experiment 1b: occlusion sensitivity (the qualitative one)
# --------------------------------------------------------------------------------------


def occlusion_slots(target: str, camera: str) -> tuple[str, ...]:
    match target:
        case "camera":
            return (camera,)
        case "base":
            return ("base_0_rgb", "base_1_rgb")
        case "wrist":
            return ("left_wrist_0_rgb",)
        case _:
            return tuple(CAMERA_VIEWS)


def run_occlusion(args: Args, loaded: LoadedModel, samples: Sequence[SamplePoint]) -> dict[str, Any]:
    """Hides one tile of the image at a time and measures how far the action moves.

    This is the qualitative counterpart to the ablation: instead of one number for "are the
    images used at all", it produces a spatial map of *which* part of the image the predicted
    action depends on, which can be judged by eye in one glance (a hot spot on the object /
    gripper / handle versus uniform noise).

    Caveat: a black tile is out of distribution, so a large response can mean either "the
    policy read information from this region" or "the policy was disturbed by an unnatural
    patch". Read it as an upper bound on spatial reliance, and prefer comparing two
    checkpoints on the same frames over reading a single map in absolute terms.
    """
    grid = args.occlusion_grid
    slots = occlusion_slots(args.occlusion_target, args.camera)
    horizon = loaded.config.model.action_horizon
    action_dim = loaded.config.model.action_dim
    sample_actions = nnx_utils.module_jit(loaded.model.sample_actions)
    rng = jax.random.key(args.seed)

    output_transform = make_output_transform(loaded.data_config, loaded.norm_stats)
    base_transform = make_input_transform(loaded.data_config, loaded.norm_stats, "baseline")
    cell_transforms = {
        (row, col): make_input_transform(
            loaded.data_config, loaded.norm_stats, "baseline", occlusion=(row, col, grid), occlusion_slots=slots
        )
        for row in range(grid)
        for col in range(grid)
    }

    picks = np.linspace(0, len(samples) - 1, min(args.occlusion_frames, len(samples))).astype(int)
    frames = [samples[int(i)] for i in picks]

    maps: list[np.ndarray] = []
    images: list[np.ndarray] = []
    per_frame: list[dict[str, Any]] = []
    baselines: list[np.ndarray] = []
    for sample in frames:
        # ``noise_for`` returns (horizon, action_dim); the model wants a batch dimension.
        noise = jnp.asarray(noise_for(sample.key, horizon, action_dim, args.seed))[None]
        batch = to_batch([base_transform(sample.obs)])
        base_pred = np.asarray(
            sample_actions(rng, _model.Observation.from_dict(batch), noise=noise, num_steps=args.num_steps),
            dtype=np.float32,
        )
        base_phys = _physical(base_pred, output_transform, action_dim)[0]
        baselines.append(base_phys)

        sensitivities = np.zeros((grid, grid), dtype=np.float64)
        for (row, col), transform in cell_transforms.items():
            cell_batch = to_batch([transform(sample.obs)])
            pred = np.asarray(
                sample_actions(
                    rng, _model.Observation.from_dict(cell_batch), noise=noise, num_steps=args.num_steps
                ),
                dtype=np.float32,
            )
            phys = _physical(pred, output_transform, action_dim)[0]
            delta = np.abs(phys - base_phys).mean(axis=0)[:N_ENV_ACTION_DIM]
            sensitivities[row, col] = float(delta.mean())
        maps.append(sensitivities)
        # Raw RGB frame (uint8 HWC) held by the sample, for the background of the heatmap.
        images.append(np.asarray(sample.obs[CAMERA_OBS_KEYS[args.camera]], dtype=np.uint8))

        flat = int(np.argmax(sensitivities))
        per_frame.append(
            {
                "key": sample.key,
                "map": sensitivities.tolist(),
                "max_cell": [int(flat // grid), int(flat % grid)],
                "mean_sensitivity": float(sensitivities.mean()),
            }
        )
        logger.info("  occlusion %d/%d frames", len(maps), len(frames))

    # Normalise by how much the predicted action naturally varies across the sampled frames,
    # so the map is in "fraction of the policy's own variability" units.
    stacked = np.concatenate(baselines, axis=0)
    baseline_spread = float(stacked[:, :N_ENV_ACTION_DIM].std(axis=0).mean())
    normalized = np.stack(maps) / (baseline_spread + 1e-9)
    summary = {
        "num_frames": len(frames),
        "grid": grid,
        "occluded_slots": list(slots),
        "background_camera": args.camera,
        "baseline_action_spread": baseline_spread,
        "mean_sensitivity": float(normalized.mean()),
        "max_sensitivity": float(normalized.max()),
        "per_frame": per_frame,
        "maps_normalized": normalized.tolist(),
    }
    summary["verdict"] = occlusion_verdict(summary)
    summary["_images"] = images
    return summary


def _physical(pred: np.ndarray, output_transform, action_dim: int) -> np.ndarray:
    dummy = np.zeros(pred.shape[:2] + (action_dim,), dtype=np.float32)
    return np.asarray(output_transform({"state": dummy, "actions": pred})["actions"], dtype=np.float32)


def occlusion_verdict(summary: dict[str, Any]) -> dict[str, Any]:
    """Reads the map by its *peak*, not its mean.

    Hiding one tile out of 36 will never move the action much on average - most tiles are
    background. What distinguishes a policy that reads the image from one that does not is
    whether *some* tile matters, and whether that tile stands out from the rest.
    """
    mean = summary["mean_sensitivity"]
    peak = summary["max_sensitivity"]
    concentration = peak / (mean + 1e-9)
    verdict: dict[str, Any] = {
        "mean_sensitivity": mean,
        "peak_sensitivity": peak,
        "peak_over_mean": concentration,
    }

    if peak < 0.05:
        verdict["reading"] = (
            "No single region moves the action by even 5% of its natural spread, so the policy "
            "is not reading the image anywhere. Consistent with 'vision unused': stop debugging "
            "with visualisation and change the training setup (see the state-noise result)."
        )
    elif peak < 0.2:
        verdict["reading"] = (
            "The strongest single region moves the action by only a modest fraction of the "
            "natural spread - the image is at best a weak global cue."
        )
    elif concentration < 3.0:
        verdict["reading"] = (
            "Occluding almost any region disturbs the action about equally, so the image enters "
            "as a diffuse global cue rather than through a specific object."
        )
    else:
        verdict["reading"] = (
            f"A single region dominates: hiding it moves the action by {peak:.2f}x the natural "
            f"spread, {concentration:.1f}x the average tile. The policy is genuinely reading a "
            "specific part of the image - check in the figure whether the hot spot sits on the "
            "object / handle / gripper being manipulated (expected) or somewhere unrelated."
        )
    return verdict


def plot_occlusion(
    summary: dict[str, Any],
    images: Sequence[np.ndarray],
    out_path: pathlib.Path,
    label: str,
    vmax: float = 0.5,
) -> None:
    maps = [np.asarray(m, dtype=np.float64) for m in summary["maps_normalized"]]
    n = len(maps)
    grid = summary["grid"]
    fig, axes = plt.subplots(n, 2, figsize=(7.0, 2.0 * n), squeeze=False)
    for i, (sensitivity, image) in enumerate(zip(maps, images, strict=False)):
        axes[i][0].imshow(image)
        axes[i][0].set_title("input (base camera)", fontsize=8)
        upsampled = np.kron(np.asarray(sensitivity), np.ones((image.shape[0] // grid, image.shape[1] // grid)))
        upsampled = upsampled[: image.shape[0], : image.shape[1]]
        axes[i][1].imshow(image)
        image_artist = axes[i][1].imshow(upsampled, cmap="jet", alpha=0.55, vmin=0.0, vmax=vmax)
        axes[i][1].set_title(f"occlusion sensitivity (frame max {sensitivity.max():.3f})", fontsize=8)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    # Absolute colour scale, so a dead map stays dark instead of being stretched to full colour.
    bar = fig.colorbar(image_artist, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    bar.set_label("|delta action| / baseline spread", fontsize=8)
    fig.suptitle(
        f"Occlusion sensitivity - {label}\n"
        f"occluded: {', '.join(summary['occluded_slots'])} | absolute scale 0-{vmax:g} | "
        f"mean {summary['mean_sensitivity']:.4f}, peak {summary['max_sensitivity']:.4f}",
        fontsize=9,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(summary: dict[str, Any], out_path: pathlib.Path, label: str) -> None:
    results = summary["variants"]
    variants = list(results)
    # The baseline is the reference the deltas are measured against, so it would be an
    # all-zero bar in the left panel; the right panel needs it plus the mean predictor.
    delta_variants = [v for v in variants if v not in ("baseline", PREDICT_MEAN_KEY)] or variants
    # Action groups whose every dimension has a zero normalization std carry no signal
    # (e.g. base_motion for a fixed robot base) and are dropped from the figure.
    groups = [
        name
        for name in ACTION_GROUPS
        if any(results[v]["err_norm_per_group"].get(name) is not None for v in variants)
    ]
    dropped = [name for name in ACTION_GROUPS if name not in groups]

    fig, axes = plt.subplots(1, 2, figsize=(max(10.0, 1.3 * len(variants) + 6), 5.5))
    width = 0.8 / max(1, len(groups))

    for ax, key, panel, title, ylabel in (
        (
            axes[0],
            "delta_phys_per_group",
            delta_variants,
            "Action change when an input is destroyed",
            "mean |delta action| (physical units)",
        ),
        (
            axes[1],
            "err_norm_per_group",
            variants,
            "Offline error vs the recorded action",
            "mean |error| (normalized units)",
        ),
    ):
        x = np.arange(len(panel))
        for gi, group in enumerate(groups):
            ax.bar(
                x + gi * width - 0.4 + width / 2,
                [results[v][key].get(group) or 0.0 for v in panel],
                width,
                label=group,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(panel, rotation=35, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

    axes[1].axhline(
        results["baseline"]["err_norm_mean"], color="black", ls="--", lw=1, label="baseline"
    )
    for ax in axes:
        ax.legend(fontsize=8, ncols=2)
    title = f"Counterfactual ablation - {label} ({summary['num_frames']} frames)"
    if dropped:
        title += (
            f"\nconstant action dims ignored: {', '.join(dropped)} "
            f"(idx {summary['excluded_action_dims']}, zero normalization std)"
        )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Experiment 2: feature extraction and representation health
# --------------------------------------------------------------------------------------


def siglip_features(model: Any, images_bhwc: jax.Array, layers: Sequence[str]) -> dict[str, np.ndarray]:
    """Runs the SigLIP tower and returns ``{layer: (B, 256, C)}`` for the wanted layers.

    pi0/pi05 build the image branch with ``pool_type="none"`` and
    ``num_classes=paligemma_width``, so the first return value is the projected patch
    sequence the LLM actually consumes (``tokens``), while ``out["encoded"]`` is the raw
    SigLIP output before that projection.

    ``scan=True`` is used for the encoder, so ``out["encoder"]["blockNN"]`` is the whole
    dict of block internals; the block's output activation is its ``"+mlp"`` entry.
    """
    tokens, out = model.PaliGemma.img(images_bhwc, train=False)
    available: dict[str, Any] = {
        "tokens": tokens,
        "encoded": out["encoded"],
        "with_posemb": out["with_posemb"],
        "stem": out["stem"].reshape(tokens.shape[0], -1, tokens.shape[-1]),
    }
    for name, value in out["encoder"].items():
        if name.startswith("block"):
            available[name] = value["+mlp"] if isinstance(value, dict) else value
    missing = [layer for layer in layers if layer not in available]
    if missing:
        raise KeyError(f"Unknown feature layers {missing}; available: {sorted(available)}")
    return {layer: np.asarray(available[layer], dtype=np.float32) for layer in layers}


def pca_basis(features: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(mean, basis)`` for a ``(N, 256, C)`` feature array, on a capped subsample."""
    x = features.reshape(-1, features.shape[-1]).astype(np.float64)
    if x.shape[0] > 20000:
        x = x[np.linspace(0, x.shape[0] - 1, 20000).astype(int)]
    mean = x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return mean, vt[:n_components]


def pca_rgb(features: np.ndarray, basis: dict[str, Any], side: int) -> np.ndarray:
    """Projects ``(B, 256, C)`` features to ``(B, side, side, 3)`` uint8 with shared scaling."""
    proj = (
        features.reshape(features.shape[0], features.shape[1], -1).astype(np.float64) - basis["mean"]
    ) @ basis["basis"].T
    proj = np.clip((proj - basis["lo"]) / (basis["hi"] - basis["lo"] + 1e-9), 0.0, 1.0)
    return (proj.reshape(features.shape[0], side, side, 3) * 255).astype(np.uint8)


def effective_rank(features: np.ndarray) -> float:
    """Participation-ratio effective rank of the centered patch features."""
    x = features.reshape(-1, features.shape[-1]).astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False) ** 2
    if s.sum() <= 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def dead_channel_frac(features: np.ndarray) -> float:
    return float(np.mean(features.reshape(-1, features.shape[-1]).std(axis=0) < 1e-3))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def within_frame_patch_sim(features: np.ndarray, max_frames: int = 32) -> float:
    """Mean pairwise cosine similarity of the patch tokens inside a single frame."""
    sims = []
    for frame in features[:max_frames]:
        normed = frame / (np.linalg.norm(frame, axis=-1, keepdims=True) + 1e-9)
        gram = normed @ normed.T
        n = gram.shape[0]
        sims.append(float((gram.sum() - np.trace(gram)) / (n * (n - 1))))
    return float(np.mean(sims))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(-1, x.shape[-1]).astype(np.float64)
    y = y.reshape(-1, y.shape[-1]).astype(np.float64)
    if x.shape[0] > 20000:
        idx = np.linspace(0, x.shape[0] - 1, 20000).astype(int)
        x, y = x[idx], y[idx]
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    num = float(np.linalg.norm(x.T @ y, "fro") ** 2)
    den = float(np.linalg.norm(x.T @ x, "fro")) * float(np.linalg.norm(y.T @ y, "fro"))
    return num / den if den > 0 else 0.0


def knn_probe_r2(
    features: np.ndarray, targets: np.ndarray, groups: np.ndarray, k: int = 5
) -> np.ndarray:
    """Per-target R^2 of a cosine k-NN decoder from ``features`` to ``targets``.

    Answers "does a visually similar frame have a similar robot pose?" - if the information
    is present in the representation, the nearest neighbours in feature space must have
    similar labels. A cosine k-NN readout is used instead of a linear probe because the
    feature dimension (1152-2048) far exceeds the number of sampled frames (~80), which makes
    any regression on the raw features hopelessly ill-conditioned and prone to reporting
    nonsense. Neighbours from the same task/episode as the query are excluded, so the decoder
    is evaluated on a trajectory it has never seen. R^2 near 0 means no better than predicting
    the mean, i.e. the information is not decodable.
    """
    x = features.astype(np.float64)
    x = x - x.mean(axis=0)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    similarity = x @ x.T
    # Never let a frame decode itself or a near-duplicate frame from its own episode.
    similarity[groups[:, None] == groups[None, :]] = -np.inf

    n_neighbours = int(min(k, max(1, len(x) - 1)))
    order = np.argsort(-similarity, axis=1)[:, :n_neighbours]

    r2 = np.full(targets.shape[1], np.nan, dtype=np.float64)
    for t in range(targets.shape[1]):
        y = targets[:, t].astype(np.float64)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        if ss_tot <= 1e-12:
            continue  # constant target: a score would be meaningless
        pred = y[order].mean(axis=1)
        r2[t] = 1.0 - float(((y - pred) ** 2).sum()) / ss_tot
    return r2


def run_features(args: Args, models: Sequence[LoadedModel], samples: Sequence[SamplePoint]) -> dict[str, Any]:
    layers = list(args.feature_layers)
    # Build the model input once and reuse it for every checkpoint so the comparison is exact.
    transform = make_input_transform(models[0].data_config, models[0].norm_stats, "baseline")
    chunks = []
    for start in range(0, len(samples), args.batch_size):
        batch = to_batch([transform(s.obs) for s in samples[start : start + args.batch_size]])
        observation = _model.preprocess_observation(None, _model.Observation.from_dict(batch), train=False)
        chunks.append(np.asarray(observation.images[args.camera], dtype=np.float32))
    camera_images = np.concatenate(chunks, axis=0)  # (N, 224, 224, 3), values in [-1, 1]
    side = camera_images.shape[1] // 14

    raw: dict[str, dict[str, np.ndarray]] = {}
    health: dict[str, dict[str, Any]] = {}
    episode_ids = np.array([s.episode for s in samples])
    task_ids = np.array([s.task for s in samples])
    # Group the probe CV by episode *within* a task, so the held-out fold is a trajectory the
    # probe has never seen rather than a neighbouring frame of one it trained on.
    probe_groups = np.array([f"{s.task}/{s.episode}" for s in samples])
    probe_targets = np.stack([s.probe for s in samples])
    probe_names = list(samples[0].probe_names)

    for loaded in models:
        per_layer: dict[str, list[np.ndarray]] = collections.defaultdict(list)
        for start in range(0, len(camera_images), args.batch_size):
            part = siglip_features(
                loaded.model, jnp.asarray(camera_images[start : start + args.batch_size]), layers
            )
            for name, value in part.items():
                per_layer[name].append(value)
        feats = {name: np.concatenate(value, axis=0) for name, value in per_layer.items()}
        raw[loaded.label] = feats

        stats: dict[str, Any] = {}
        for layer in layers:
            f = feats[layer]
            pooled = f.mean(axis=1)
            same_episode, cross_episode, cross_task = [], [], []
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    sim = cosine(pooled[i], pooled[j])
                    if task_ids[i] != task_ids[j]:
                        cross_task.append(sim)
                    elif episode_ids[i] == episode_ids[j]:
                        same_episode.append(sim)
                    else:
                        cross_episode.append(sim)
            probe_r2 = knn_probe_r2(pooled, probe_targets, probe_groups)
            stats[layer] = {
                "dim": int(f.shape[-1]),
                "within_frame_patch_sim": within_frame_patch_sim(f),
                "same_episode_sim": float(np.mean(same_episode)) if same_episode else None,
                "cross_episode_sim": float(np.mean(cross_episode)) if cross_episode else None,
                "cross_task_sim": float(np.mean(cross_task)) if cross_task else None,
                "effective_rank": effective_rank(f),
                "dead_channel_frac": dead_channel_frac(f),
                "probe_targets": probe_names,
                "probe_r2_per_dim": probe_r2.tolist(),
                "probe_r2_mean": float(np.nanmean(probe_r2)),
            }
        health[loaded.label] = stats
        logger.info("[%s] extracted %d layers for %d frames", loaded.label, len(layers), len(samples))

    # Linear CKA between every pair of checkpoints, recorded on both sides with an explicit
    # label (it is symmetric, so one value serves both directions).
    for i, first in enumerate(models):
        for second in models[i + 1 :]:
            if first.label == second.label:
                continue
            for layer in layers:
                value = linear_cka(raw[first.label][layer], raw[second.label][layer])
                health[first.label][layer][f"cka_to_{second.label}"] = value
                health[second.label][layer][f"cka_to_{first.label}"] = value

    # PCA bases are fitted on the first checkpoint and reused for the others, so a colour
    # means the same thing in every row of the figure.
    primary = models[0].label
    bases: dict[str, dict[str, Any]] = {}
    for layer in layers:
        mean, basis = pca_basis(raw[primary][layer])
        proj = (
            raw[primary][layer].reshape(-1, raw[primary][layer].shape[-1]).astype(np.float64) - mean
        ) @ basis.T
        bases[layer] = {
            "mean": mean,
            "basis": basis,
            "lo": np.percentile(proj, 1, axis=0),
            "hi": np.percentile(proj, 99, axis=0),
        }

    return {
        "camera": args.camera,
        "num_frames": len(samples),
        "side": side,
        "camera_images": camera_images,
        "raw": raw,
        "bases": bases,
        "health": health,
        "checkpoint_dirs": {m.label: str(m.checkpoint_dir) for m in models},
    }


def plot_features(args: Args, features: dict[str, Any], out_dir: pathlib.Path) -> None:
    images = features["camera_images"]
    labels = list(features["raw"])
    n_show = min(args.max_frames_in_figure, len(images))
    idx = np.linspace(0, len(images) - 1, n_show).astype(int)
    side = features["side"]
    upscale = max(1, 224 // side)

    for layer in args.feature_layers:
        n_rows = 1 + len(labels)
        fig, axes = plt.subplots(n_rows, n_show, figsize=(1.6 * n_show, 1.7 * n_rows), squeeze=False)
        for col, i in enumerate(idx):
            axes[0][col].imshow(((images[i] + 1.0) * 127.5).astype(np.uint8))
            axes[0][col].set_title(f"frame {i}", fontsize=7)
            for row, label in enumerate(labels, start=1):
                rgb = pca_rgb(features["raw"][label][layer][i : i + 1], features["bases"][layer], side)[0]
                axes[row][col].imshow(np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1))
            for row in range(n_rows):
                axes[row][col].set_xticks([])
                axes[row][col].set_yticks([])
        for row, label in enumerate(["input image", *labels]):
            axes[row][0].set_ylabel(label, fontsize=9)
        fig.suptitle(f"PCA of SigLIP patch features - layer '{layer}' ({args.camera})")
        fig.tight_layout()
        fig.savefig(out_dir / f"features_pca_{layer}.png", dpi=130)
        plt.close(fig)

    layers = [layer for layer in args.feature_layers if layer in features["health"][labels[0]]]
    if not layers:
        return
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for label in labels:
        stats = features["health"][label]
        xs = list(range(len(layers)))
        axes[0][0].plot(xs, [stats[l]["within_frame_patch_sim"] for l in layers], "o-", label=label)
        axes[0][1].plot(
            xs,
            [
                (stats[l]["same_episode_sim"] or 0.0) - (stats[l]["cross_task_sim"] or 0.0)
                for l in layers
            ],
            "o-",
            label=label,
        )
        axes[0][2].plot(xs, [stats[l]["probe_r2_mean"] for l in layers], "o-", label=label)
        axes[1][0].plot(xs, [stats[l]["effective_rank"] for l in layers], "o-", label=label)
        axes[1][1].plot(xs, [stats[l]["dead_channel_frac"] for l in layers], "o-", label=label)
        if any(key.startswith("cka_to_") for key in stats[layers[0]]):
            cka_key = next(key for key in stats[layers[0]] if key.startswith("cka_to_"))
            axes[1][2].plot(
                xs,
                [stats[l].get(cka_key, float("nan")) for l in layers],
                "o-",
                label=label,
            )
    axes[0][0].set_title("Within-frame patch similarity\n(1.0 = every patch identical = no spatial info)")
    axes[0][1].set_title("Scene separability\n(same-episode sim - cross-task sim; higher is better)")
    axes[0][2].set_title("k-NN cosine decode R^2\n(robot pose / gripper from the features)")
    axes[1][0].set_title("Effective rank (participation ratio)")
    axes[1][1].set_title("Fraction of dead channels")
    axes[1][2].set_title("Linear CKA between checkpoints\n(how far fine-tuning moved the representation)")
    for ax in axes.ravel():
        ax.set_xticks(list(range(len(layers))))
        ax.set_xticklabels(layers, rotation=30, ha="right")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_health.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    # Data-preparation-only mode: sample the frames, write an archive, stop. No model, no
    # checkpoint and no GPU -- see ``Args.dump_frames``.
    if args.dump_frames:
        logger.info("dumping frames for %s (no model will be loaded)", args.config_name)
        samples = collect_samples(args, resolve_horizon(args.config_name))
        if not samples:
            raise RuntimeError("No samples collected; check --tasks and the dataset lengths.")
        dump_samples(samples, args.dump_frames, args)
        return

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_model(args.checkpoint_dir, args.config_name, "finetuned")
    horizon = loaded.config.model.action_horizon

    models = [loaded]
    if args.compare_checkpoint_dir:
        models.append(
            load_model(
                args.compare_checkpoint_dir,
                args.compare_config_name or args.config_name,
                args.compare_label,
            )
        )

    if args.frames_from:
        logger.warning(
            "--frames-from is set; --tasks/--num-episodes-per-task/--num-frames-per-episode/"
            "--stale-offset are ignored, the sampling is already baked into the archive"
        )
        samples = load_samples(args.frames_from)
    else:
        logger.info("tasks: %s", args.tasks or discover_tasks())
        samples = collect_samples(args, horizon)
    if not samples:
        raise RuntimeError("No samples collected; check --tasks and the dataset lengths.")
    logger.info("collected %d frames", len(samples))

    summary: dict[str, Any] = {
        "config_name": args.config_name,
        "checkpoint_dir": str(loaded.checkpoint_dir),
        "compare_checkpoint_dir": args.compare_checkpoint_dir,
        "tasks": sorted({s.task for s in samples}),
        "num_frames": len(samples),
    }

    if args.run_ablation:
        logger.info("running counterfactual ablation...")
        ablation = run_ablation(args, loaded, samples)
        (out_dir / "ablation.json").write_text(json.dumps(ablation, indent=2))
        plot_ablation(ablation, out_dir / "ablation.png", loaded.label)
        summary["ablation_verdict"] = ablation["verdict"]
        logger.info("ablation verdict:\n%s", json.dumps(ablation["verdict"], indent=2))

    if args.run_occlusion:
        logger.info("running occlusion sensitivity...")
        occlusion = run_occlusion(args, loaded, samples)
        images = occlusion.pop("_images")
        plot_occlusion(occlusion, images, out_dir / "occlusion.png", loaded.label, args.occlusion_vmax)
        (out_dir / "occlusion.json").write_text(json.dumps(occlusion, indent=2))
        summary["occlusion_verdict"] = occlusion["verdict"]
        logger.info("occlusion verdict:\n%s", json.dumps(occlusion["verdict"], indent=2))

    if args.run_features:
        logger.info("running feature inspection...")
        features = run_features(args, models, samples)
        plot_features(args, features, out_dir)
        (out_dir / "features.json").write_text(
            json.dumps(
                {
                    "camera": features["camera"],
                    "num_frames": features["num_frames"],
                    "checkpoint_dirs": features["checkpoint_dirs"],
                    "health": features["health"],
                },
                indent=2,
            )
        )
        summary["feature_health"] = features["health"]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote diagnostics to %s", out_dir.resolve())


if __name__ == "__main__":
    main(tyro.cli(Args))
