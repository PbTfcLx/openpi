#!/usr/bin/env python
"""Print everything needed to ablate one checkpoint, so the run conditions cannot be guessed.

Three things have to be right before an ablation is meaningful, and all three are properties
of the *checkpoint* rather than of the current code:

1. **Training config** - the model variant (full fine-tune vs LoRA) must match, otherwise the
   parameter tree does not load at all.
2. **Action horizon** - a checkpoint trained with 50 predicted steps has to be evaluated at
   50: the predicted chunk length enters every per-step metric, and evaluating at the wrong
   horizon visibly degrades the fit (measured: err/mean 1.03 -> 1.00 for the same checkpoint).
3. **State layout** - a checkpoint trained without some proprioceptive key expects the padding
   identity (mean 0, std 1) in that slot. Feeding the real value hands it an input it never
   saw, so the slot has to be zeroed with ``--zero-state-keys``.

The script reads the run's wandb config (``wandb/run-*/files/config.yaml``, located through
``<exp>/wandb_id.txt``) and the persisted ``assets/norm_stats.json``, then prints a ready
``diagnose_vision.py`` command.

Usage::

    uv run .github/skills/reliance-eval/scripts/ckpt_info.py --checkpoint checkpoints/pi05_robocasa/fix_checkpoint/15000
    uv run .github/skills/reliance-eval/scripts/ckpt_info.py --checkpoint <ckpt> --zero-state-keys joint_position
"""

from __future__ import annotations

import dataclasses
import glob
import json
import pathlib
import subprocess

import numpy as np
import tyro

# Order and widths of the state vector as ``groot_openpi_dataset`` assembles it. The last two
# are padding to the model's 32-dim action/state width.
STATE_LAYOUT = (
    ("end_effector_position_relative", 3),
    ("end_effector_rotation_relative", 4),
    ("base_position", 3),
    ("base_rotation", 4),
    ("gripper_qpos", 2),
    ("joint_position", 7),
)
# Keys of the wandb run config worth reporting, in report order.
INTERESTING_CONFIG = (
    "exp_name",
    "action_horizon",
    "max_token_len",
    "action_dim",
    "num_train_steps",
    "batch_size",
    "save_interval",
    "extra_delta_transform",
    "state_noise",
    "state_noise_beta_a",
    "state_noise_beta_b",
    "state_noise_p",
    "prompt_drop_p",
    "paligemma_variant",
    "action_expert_variant",
)


@dataclasses.dataclass
class Args:
    # Checkpoint step directory (containing ``params/``), or the experiment directory whose
    # numeric step subdirectories are listed.
    checkpoint: str
    # Diagnostic run: print the command with these state keys zeroed.
    zero_state_keys: list[str] = dataclasses.field(default_factory=list)
    # Configs to probe when the directory name cannot be mapped to one.
    config_candidates: list[str] = dataclasses.field(
        default_factory=lambda: ["pi05_robocasa", "pi05_robocasa_state_noise", "pi05_robocasa_low_mem"]
    )


def unwrap(node):
    """wandb writes ``{key: {value: ...}}``; strip the wrapper, tolerating both spellings."""
    seen = 0
    while isinstance(node, dict) and "value" in node and len(node) == 1 and seen < 4:
        node, seen = node["value"], seen + 1
    return node


def run_config(exp_dir: pathlib.Path, repo_root: pathlib.Path) -> dict | None:
    """The wandb config dict of the run that produced this experiment directory."""
    wandb_id_file = exp_dir / "wandb_id.txt"
    if not wandb_id_file.exists():
        return None
    wandb_id = wandb_id_file.read_text().strip()
    matches = sorted(glob.glob(str(repo_root / "wandb" / f"run-*{wandb_id}" / "files" / "config.yaml")))
    if not matches:
        return None
    try:
        import yaml
    except ImportError:
        return None
    return unwrap(yaml.safe_load(pathlib.Path(matches[-1]).read_text())) or {}


def flattened(config: dict) -> dict:
    """One level of nesting flattened: ``action_horizon`` lives under ``model`` in the run config."""
    out: dict = {}
    for key, node in config.items():
        value = unwrap(node)
        if isinstance(value, dict):
            for sub_key, sub_node in value.items():
                out[f"{key}.{sub_key}"] = unwrap(sub_node)
        else:
            out[key] = value
    return out


def report_config(config: dict | None, exp_dir: pathlib.Path) -> None:
    print("training run config")
    if config is None:
        print("  (not found: no wandb_id.txt, no matching wandb/run-*/files/config.yaml, or pyyaml missing)")
        return
    flat = flattened(config)
    for key in INTERESTING_CONFIG:
        for candidate in (key, f"model.{key}"):
            if candidate in flat:
                print(f"  {key:<26} {flat[candidate]}")
                break
    data = unwrap(config.get("data"))
    if isinstance(data, dict):
        dirs = unwrap(data.get("data_dirs")) or []
        tasks = [unwrap(d).get("task") for d in dirs if isinstance(d, dict)]
        print(f"  {'trained tasks':<26} {tasks}")
        print(f"  {'dataset_weights':<26} {unwrap(data.get('dataset_weights'))}")
        print(f"  {'repo_id':<26} {unwrap(data.get('repo_id'))}")


def report_norm_stats(step_dir: pathlib.Path) -> int | None:
    """State statistics: which slots carry signal, and which hold the padding identity."""
    path = step_dir / "assets" / "norm_stats.json"
    if not path.exists():
        print("norm stats                  (none: assets/norm_stats.json missing)")
        return None
    stats = json.loads(path.read_text())["norm_stats"]
    mean = np.asarray(stats["state"]["mean"], dtype=float)
    std = np.asarray(stats["state"]["std"], dtype=float)
    padding = np.flatnonzero((np.abs(mean) < 1e-9) & (np.abs(std - 1.0) < 1e-9))
    print(f"norm stats                  state {mean.shape[0]} dims, actions {len(stats['actions']['mean'])} dims")
    print(f"  padding identity slots     {padding.tolist()}")
    real = [d for d in range(len(mean)) if d not in set(padding.tolist())]
    print(f"  slots carrying signal      {real}")

    # Map the layout onto the slots that are actually read: Normalize only uses the stats that
    # line up with the state vector the transforms produce, so trailing padding is invisible.
    start = 0
    lines = []
    for key, width in STATE_LAYOUT:
        flags = []
        for i in range(start, start + width):
            if i in set(padding.tolist()):
                flags.append("padding")
        lines.append(f"    {key:<32} {start:>2}..{start + width - 1:<2} {'PADDING (identity)' if len(flags) == width else ''}")
        start += width
    print("  layout misalignment (a key marked PADDING is one the model never saw):")
    print("\n".join(lines))
    return int(padding[0]) if len(padding) else len(mean)


def free_vram_gib() -> float | None:
    """Free GPU memory, only to tell the reader what the run will size itself against.

    The rule itself lives in ``scripts/diagnose_vision.py::auto_batch_size`` (single source of
    truth) and is applied by that script at start-up, when the memory situation is current.
    """
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip().isdigit()]
    return max(float(v) for v in values) / 1024.0 if values else None


def main(args: Args) -> None:
    exp_dir = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"{exp_dir} does not exist")
    repo_root = pathlib.Path(__file__).resolve().parents[4]

    steps = sorted((p for p in exp_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    if steps:
        print(f"experiment dir              {exp_dir}")
        print(f"  available steps           {[int(p.name) for p in steps]}")
        step_dir = steps[-1]
        print(f"  latest step used below    {step_dir.name}")
    else:
        step_dir = exp_dir
        print(f"checkpoint step dir         {exp_dir}")

    # ``checkpoints/<config>/<exp>[/<step>]`` is the layout the training script writes.
    parts = exp_dir.parts
    config_guess = None
    if "checkpoints" in parts and len(parts) > parts.index("checkpoints") + 2:
        config_guess = parts[parts.index("checkpoints") + 1]
    print(f"training config (from path) {config_guess or '(unknown; pass a config explicitly)'}")
    if config_guess is None:
        print(f"  candidates to try         {args.config_candidates}")

    print()
    report_config(run_config(exp_dir, repo_root), exp_dir)

    print()
    first_padding = report_norm_stats(step_dir)

    print()
    free = free_vram_gib()
    print(
        "free VRAM now               "
        + (f"{free:.1f} GiB (the run sizes its own batch from this)" if free is not None else "(nvidia-smi unavailable)")
    )

    flat = flattened(run_config(exp_dir, repo_root) or {})
    horizon = flat.get("action_horizon", flat.get("model.action_horizon")) or "<from the run config; must equal the training value>"
    print()
    print("ready-to-run command")
    config = config_guess or "<config>"
    suffix = " ".join(f"--zero-state-keys {k}" for k in args.zero_state_keys)
    if first_padding is not None and args.zero_state_keys:
        print(f"  # slots {first_padding}.. are padding identity in this checkpoint, so it was trained")
        print(f"  # without those keys; use --state-dim-limit {first_padding} for the exact match.")
    print(
        f"  cd <repo-root> && XLA_PYTHON_CLIENT_PREALLOCATE=false HF_LEROBOT_HOME=/nonexistent \\\n"
        f"    uv run scripts/diagnose_vision.py \\\n"
        f"      --config-name {config} --checkpoint-dir {step_dir} \\\n"
        f"      --action-horizon {horizon} {suffix} \\\n"
        f"      --frames-from diagnostics/frames_robocasa_7tasks_84frames.npz \\\n"
        f"      --batch-size 0 --no-run-features --no-run-occlusion \\\n"
        f"      --output-dir diagnostics/<label>"
    )
    print("  # --batch-size 0 = auto: derived from the free VRAM at start-up, and halved")
    print("  # automatically if the GPU still runs out, so no hand-tuning next to a training job.")


if __name__ == "__main__":
    main(tyro.cli(Args))
