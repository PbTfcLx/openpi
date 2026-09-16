# ruff: noqa: E402 - the thread limits below must be applied before numpy/MuJoCo import.
import collections
import dataclasses
import faulthandler
import logging
import multiprocessing
import os
import pathlib
import queue
import signal
import time

# /etc/profile.d/autodl.env.sh exports OMP_NUM_THREADS=MKL_NUM_THREADS=$(nproc) (=25
# here). Every spawned worker inherits it, so --num_envs 12 would try to run 12 * 25
# OpenMP threads on 25 cores. MuJoCo's physics and the offscreen renderer are
# OpenMP-based, and that much oversubscription is a known cause of random crashes
# (and of large slowdowns). Cap the pools per process before they get initialized;
# override with OPENPI_EVAL_THREADS=n.
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = os.environ.get("OPENPI_EVAL_THREADS", "1")

import cv2
import imageio
import robocasa  # noqa: F401
import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
from robocasa.utils.env_utils import convert_action
import gymnasium as gym
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    env_name: str = "robocasa_panda_omron/OpenDrawer_PandaOmron_Env"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 100  # Number of rollouts per task
    max_steps: int = 720

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/robocasa/pi05_base/videos/opendrawer"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)
    num_envs : int = 12

    #################################################################################################################
    # Multi-env / video recording
    #################################################################################################################
    save_video: bool = True  # Whether to save per-episode replay videos
    steps_per_render: int = 1  # Record a video frame every N env steps
    num_retries: int = 3  # Re-run an episode this many times if it hits NaN
    max_episode_restarts: int = 3  # Re-queue an episode in a fresh worker if its worker died or raised; after this
    # many losses the episode is counted as a failure instead of aborting the eval
    max_worker_restarts: int = 3  # How often a worker slot may be restarted after crashing; guards against a crash loop
    one_episode_per_worker: bool = True  # Retire the worker process + env after every episode (~15% slower). Needed for
    # reproducible runs: a reused env makes an episode's outcome depend on which episodes ran before it in that process

    #################################################################################################################
    # Determinism / reproducibility
    #################################################################################################################
    deterministic: bool = True  # Pin action noise to (episode seed, replan index) so runs are reproducible and different checkpoints can be compared on identical scenes
    results_out_path: str = ""  # Optional TSV to dump per-episode results (episode_idx, seed, success, length); keep the same seed/num_trials when comparing checkpoints
    run_start_epoch: float = 0.0  # Internal: wall-clock start of this run, set by eval_robocasa; used to tell this run's videos from leftovers of earlier runs

def get_robocasa_env_fn(
    env_name: str,
    robocasa_split: str = "",
):
    def env_fn():
        kwargs = {}
        if env_name.startswith("robocasa365_panda_omron/"):
            import gr00t.eval.sim.robocasa365.gymnasium_groot  # noqa: F401

            if robocasa_split:
                kwargs["split"] = robocasa_split
        else:
            import robocasa  # noqa: F401
            import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

        return gym.make(env_name, enable_render=True, **kwargs)

    return env_fn


class _FlattenActionEnv(gym.Wrapper):
    """Expose the robocasa Dict action as a flat Box(12) action.

    The policy outputs a flat 12-dim action. This wrapper converts it back to the
    robocasa dict action format (``action.gripper_close``, ``action.control_mode``,
    etc.) before passing it to the underlying env, so the rollout loop only ever
    deals with flat actions.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

    def step(self, action):
        return self.env.step(convert_action(action))


def _to_bool(value) -> bool:
    """Coerce robocasa's success flag (bool / np.bool_ / list / ndarray / int) to bool."""
    if isinstance(value, np.ndarray):
        value = value.item() if value.size == 1 else np.any(value)
    if isinstance(value, (list, tuple)):
        return any(_to_bool(v) for v in value)
    return bool(value)


# Styling for the prompt banner drawn under each replay frame.
_BANNER_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BANNER_SCALE = 0.5
_BANNER_THICKNESS = 1
_BANNER_PAD = 6
_BANNER_BG = (0, 0, 0)  # black
_BANNER_FG = (255, 255, 255)  # white


def _wrap_banner_text(text: str, max_width: int) -> list[str]:
    """Greedily wrap ``text`` into lines that fit within ``max_width`` pixels."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        width = cv2.getTextSize(
            candidate, _BANNER_FONT, _BANNER_SCALE, _BANNER_THICKNESS
        )[0][0]
        # Always keep at least one word per line, even if it overflows.
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_prompt_banner(frame: np.ndarray, prompt: str) -> np.ndarray:
    """Append a black banner showing the task prompt at the bottom of ``frame``.

    The text is wrapped to the frame width and the banner grows by as many lines
    as needed, so long task descriptions stay fully readable.
    """
    if not prompt:
        return frame

    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    max_width = max(frame.shape[1] - 2 * _BANNER_PAD, 1)
    lines = _wrap_banner_text(str(prompt), max_width)

    (_, text_h), baseline = cv2.getTextSize(
        "Ag", _BANNER_FONT, _BANNER_SCALE, _BANNER_THICKNESS
    )
    line_step = text_h + baseline + 4
    # The cameras are 256px tall (a multiple of 16), so keeping the banner a
    # multiple of 16 as well keeps the video height encoder-friendly - otherwise
    # ffmpeg rescales every frame (macro_block_size warning) on write.
    banner_h = -(-(2 * _BANNER_PAD + line_step * len(lines)) // 16) * 16
    banner = np.full((banner_h, frame.shape[1], 3), _BANNER_BG, dtype=np.uint8)

    text_block_h = line_step * (len(lines) - 1) + text_h
    y = (banner_h - text_block_h) // 2 + text_h
    for line in lines:
        cv2.putText(
            banner,
            line,
            (_BANNER_PAD, y),
            _BANNER_FONT,
            _BANNER_SCALE,
            _BANNER_FG,
            _BANNER_THICKNESS,
            cv2.LINE_AA,
        )
        y += line_step

    return np.concatenate((frame, banner), axis=0)


def _make_replay_frame(obs, prompt: str = "") -> np.ndarray:
    """Tile the three camera views into one wide frame for the replay video.

    The order matches the policy input: side_0, side_1, wrist. ``prompt`` (the
    task description) is drawn on a banner below the tiled cameras.
    """
    frame = np.concatenate(
        (
            np.ascontiguousarray(obs["video.res256_image_side_0"]),
            np.ascontiguousarray(obs["video.res256_image_side_1"]),
            np.ascontiguousarray(obs["video.res256_image_wrist_0"]),
        ),
        axis=1,
    )
    return _draw_prompt_banner(frame, prompt)


def _has_nan(obs) -> bool:
    """True if the observation's proprioceptive state contains NaN.

    A NaN state means the simulator has entered an invalid configuration, which
    can also make the offscreen renderer start producing blank frames.
    """
    return any(
        np.isnan(obs[key]).any()
        for key in (
            "state.end_effector_position_relative",
            "state.end_effector_rotation_relative",
            "state.base_position",
            "state.base_rotation",
            "state.gripper_qpos",
            "state.joint_position",
        )
    )


# The three camera views fed to the policy and used for the replay video.
_CAMERA_KEYS = (
    "video.res256_image_side_0",
    "video.res256_image_side_1",
    "video.res256_image_wrist_0",
)


def _renderer_dead(obs, prev_images) -> bool:
    """True if the offscreen renderer stopped producing real frames.

    Robocasa images come straight from robosuite's MuJoCo offscreen renderer
    (``*_image`` obs). If that renderer dies mid-episode the physics state stays
    perfectly finite, so ``_has_nan`` never fires - the failure is purely visual
    and has to be detected from the images themselves. The tell-tale signature
    is that *all three* cameras (which always see different things in a live
    scene) simultaneously start returning the same dead buffer. We flag it when
    every camera is:

      * frozen: byte-identical to the previous step's image (a live renderer
        always moves at least a little because the arm is in motion), and/or
      * blank:  (near-)uniform frame with essentially no contrast, and/or
      * all three cameras (near-)identical to each other - a fallback that also
        holds on slightly noisy frames, since different viewpoints can never
        produce the same image in a real scene.

    Returns True only when *every* camera is affected, so a legitimately dark or
    static view (e.g. the wrist cam pressed against an object) is not flagged.
    """
    if prev_images is None:
        # Nothing to compare on the very first step.
        return False
    imgs = [np.asarray(obs[k], dtype=np.uint8) for k in _CAMERA_KEYS]

    # Byte-level gates first (each is a fast memcmp on 256x256x3).
    # A live renderer always moves between steps because the arm is in motion,
    # so all three cameras frozen means it stopped producing new frames...
    frozen = [np.array_equal(imgs[i], prev_images[i]) for i in range(3)]
    if all(frozen):
        return True
    # ...and two cameras that should see different viewpoints being
    # byte-identical means they are both returning the same dead buffer.
    pair_identical = (
        np.array_equal(imgs[0], imgs[1]),
        np.array_equal(imgs[0], imgs[2]),
        np.array_equal(imgs[1], imgs[2]),
    )
    if all(pair_identical):
        return True

    # Otherwise every camera must independently look dead: frozen or
    # (near-)uniform with no contrast. Std is computed on a 4x4-subsampled
    # luma - ~free per step, and it drops high-frequency encode noise that
    # would otherwise hide a truly flat (cleared) buffer.
    def _is_flat(img) -> bool:
        luma = img[::4, ::4].astype(np.float32).mean(axis=2)
        return float(luma.std()) < 12.0

    every_dead = all((frozen[i] or _is_flat(imgs[i])) for i in range(3))
    if every_dead:
        return True

    # Fallback - different viewpoints can never be near-identical in a live
    # scene, so if all three are, the renderer is returning one dead buffer
    # (this still holds on slightly noisy, non-byte-identical frames).
    return all(
        float(
            np.mean(
                np.abs(imgs[i].astype(np.int16) - imgs[j].astype(np.int16))
            )
        )
        < 3.0
        for i in range(3)
        for j in range(i + 1, 3)
    )


def _run_episode(
    env: gym.Env,
    client: _websocket_client_policy.WebsocketClientPolicy,
    args: Args,
    seed: int | None,
    episode_idx: int,
    attempt: int = 0,
) -> tuple[bool, int, bool, bool]:
    """Run one attempt of an episode.

    Returns ``(success, num_env_steps, hit_nan, render_dead)``. ``hit_nan`` is
    True when the simulator entered an invalid (NaN) state; ``render_dead`` is
    True when the offscreen renderer started returning blank/frozen frames even
    though the state stayed finite. Either may be retried by the caller.
    """
    if seed is not None:
        _reseed_scene_rng(env, seed)
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()

    task_description = str(obs["annotation.human.action.task_description"])
    action_plan = collections.deque()
    replay_images = []
    t = 0
    replan_idx = 0
    done = False
    hit_nan = False
    render_dead = False
    prev_images = None

    while True:
        # Preprocess the three camera views.
        img = np.ascontiguousarray(obs["video.res256_image_side_0"])
        wrist_img = np.ascontiguousarray(obs["video.res256_image_wrist_0"])
        img_right = np.ascontiguousarray(obs["video.res256_image_side_1"])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
        )
        img_right = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img_right, args.resize_size, args.resize_size)
        )

        if not action_plan:
            state = np.concatenate(
                (
                    obs["state.end_effector_position_relative"],
                    obs["state.end_effector_rotation_relative"],
                    obs["state.base_position"],
                    obs["state.base_rotation"],
                    obs["state.gripper_qpos"],
                    obs["state.joint_position"],
                ),
                axis=0,
            )
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/right_image": img_right,
                "observation/state": state,
                "prompt": task_description,
            }
            if args.deterministic and seed is not None:
                # Pin sampling noise to (episode seed, replan index) so the
                # policy's stochasticity is reproducible and identical across
                # checkpoints evaluated on the same scene.
                element["eval/episode_seed"] = int(seed)
                element["eval/replan_idx"] = int(replan_idx)
            action_chunk = client.infer(element)["actions"]
            replan_idx += 1
            assert len(action_chunk) >= args.replan_steps, (
                f"We want to replan every {args.replan_steps} steps, but policy "
                f"only predicts {len(action_chunk)} steps."
            )
            action_plan.extend(action_chunk[: args.replan_steps])

        action = action_plan.popleft()
        obs, _reward, terminated, truncated, info = env.step(
            np.asarray(action, dtype=np.float32)
        )

        # Robocasa never sets `terminated=True` (its robosuite wrapper always
        # returns done=False). Success is only reported via info["success"], so
        # we must break on it explicitly instead of waiting for termination.
        success = _to_bool(info.get("success", False))

        # If the simulator entered an invalid (NaN) state, the renderer can start
        # producing blank frames for the rest of the episode. Abort early so the
        # caller can retry instead of wasting the remaining max_steps.
        if not success and _has_nan(obs):
            hit_nan = True
            break

        # The reverse also happens: the renderer dies while the state stays
        # finite (NaN checks see nothing). All cameras then return the same
        # frozen/blank buffer, which would silently poison both the policy input
        # and the replay video for the rest of the episode - abort and retry.
        if not success and _renderer_dead(obs, prev_images):
            render_dead = True
            break

        if args.save_video and (
            t % args.steps_per_render == 0 or success or terminated or truncated
        ):
            replay_images.append(_make_replay_frame(obs, task_description))

        if success or terminated or truncated:
            done = success
            break
        t += 1

        # Remember this step's camera images so the next step can detect a
        # renderer freeze (images identical across consecutive env steps).
        prev_images = tuple(np.asarray(obs[k]) for k in _CAMERA_KEYS)

    if args.save_video and replay_images and not (hit_nan or render_dead):
        suffix = "success" if done else "failure"
        video_path = (
            pathlib.Path(args.video_out_path)
            / f"rollout_{episode_idx:04d}_{suffix}.mp4"
        )
        # The file name encodes the outcome, so an episode that flipped between two
        # runs leaves *both* `..._success.mp4` and `..._failure.mp4` behind and the
        # directory looks like one episode produced two contradictory videos. Drop
        # the other outcome if it was written by an earlier run; a re-run inside the
        # current run (worker died -> episode re-queued) keeps both, by design.
        for other in ("success", "failure"):
            stale = video_path.with_name(f"rollout_{episode_idx:04d}_{other}.mp4")
            if stale != video_path and stale.exists() and stale.stat().st_mtime < args.run_start_epoch:
                try:
                    stale.unlink()
                    logging.info(f"Removed stale {stale.name} (written by an earlier run)")
                except OSError as e:  # noqa: BLE001 - cleanup is best-effort
                    logging.warning(f"Could not remove stale {stale}: {e}")
        try:
            imageio.mimwrite(video_path, replay_images, fps=40, codec="libx264")
        except Exception as e:  # noqa: BLE001 - video is best-effort
            logging.error(f"Failed to save video {video_path}: {e}")

    return done, t + 1, hit_nan, render_dead


def _run_worker(
    worker_id: int,
    task_queue,
    args: Args,
    result_queue,
) -> None:
    """Worker process: owns one env and one policy client.

    The parent hands this worker one episode at a time through its private
    ``task_queue`` and the worker reports back with
    ``(worker_id, episode_idx, success, length)``. Episode indices map to
    deterministic seeds via ``seed + episode_idx``.

    Every per-episode failure is *reported* instead of silently killing the
    process, so the parent can re-queue that one episode. The only failure the
    parent cannot be told about is a hard crash (segfault in MuJoCo/EGL, OOM
    kill): there faulthandler dumps the C-level traceback to stderr, and the
    parent notices the missing report and reschedules the episode.
    """
    # A crash that takes the process down (corrupt EGL context, driver bug, OOM
    # kill) never reaches Python's exception handling, so the log would otherwise
    # end with nothing but "worker died". This prints the actual faulting stack.
    faulthandler.enable()

    env = None
    try:
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        env = _make_env(args.env_name, args.max_steps)

        while True:
            episode_idx = task_queue.get()
            if episode_idx is None:
                # Parent is shutting this worker down.
                break

            seed = args.seed + episode_idx if args.seed is not None else None
            success, length = False, 0
            try:
                for attempt in range(args.num_retries):
                    success, length, hit_nan, render_dead = _run_episode(
                        env, client, args, seed, episode_idx, attempt
                    )
                    if not (hit_nan or render_dead):
                        break
                    if attempt + 1 < args.num_retries:
                        if render_dead:
                            # A dead renderer usually means this worker's offscreen
                            # EGL/GL context is corrupted. Reusing the same env will
                            # most likely stay broken, so build a fresh one (and with
                            # it a fresh render context).
                            try:
                                env.close()
                            except Exception:  # noqa: BLE001
                                pass
                            env = _make_env(args.env_name, args.max_steps)
                        reason = "NaN state" if hit_nan else "dead renderer"
                        logging.warning(
                            f"Episode {episode_idx} attempt {attempt + 1}/{args.num_retries} "
                            f"hit {reason}, retrying..."
                        )
                else:
                    reason = "NaN state" if hit_nan else "dead renderer"
                    # Retry budget exhausted and the episode never recovered.
                    logging.error(
                        f"[ALARM] Episode {episode_idx}: {reason} persisted after "
                        f"{args.num_retries} retries, marking as failure"
                    )
            except Exception as e:  # noqa: BLE001 - report per-episode failures
                logging.error(f"Worker {worker_id} failed on episode {episode_idx}: {e}")
                result_queue.put((worker_id, episode_idx, None, str(e)))
                # The env is in an undefined state now, so exit instead of
                # poisoning the following episodes; the parent re-queues this
                # episode and starts a fresh worker.
                return

            result_queue.put((worker_id, episode_idx, bool(success), length))
    except Exception as e:  # noqa: BLE001 - report setup failures to the parent
        logging.error(f"Worker {worker_id} failed: {e}")
        result_queue.put((worker_id, -1, None, f"worker setup failed: {e}"))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass


def _make_env(env_name: str, max_steps: int) -> gym.Env:
    """Create one wrapped robocasa env for a vector-env worker."""
    env = get_robocasa_env_fn(env_name)()
    env = _FlattenActionEnv(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    return env


def _reseed_scene_rng(env: gym.Env, seed: int) -> None:
    """Make the next ``env.reset()`` scene a pure function of ``seed``.

    Robocasa samples the kitchen layout/style/object placement/initial pose from
    the robosuite environment's internal ``rng`` (an ``np.random.default_rng``
    created with ``seed=None`` at env construction). ``RoboCasaEnv.reset`` only
    reseeds the *global* numpy RNG, which robocasa ignores - so without this, two
    runs (or two checkpoints) given the same episode seed still get *different*
    scenes. Reseeding the inner RNG pins the scene to ``seed``.
    """
    # env.unwrapped -> RoboCasaEnv (gym.Env); its .env is the robosuite env.
    env.unwrapped.env.rng = np.random.default_rng(seed)


def eval_robocasa(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    # Remember when this run started: the video writer needs it to distinguish
    # leftovers of an earlier run from videos of the current one.
    args.run_start_epoch = time.time()
    leftovers = sorted(pathlib.Path(args.video_out_path).glob("rollout_*.mp4"))
    if leftovers:
        logging.warning(
            f"{args.video_out_path} already contains {len(leftovers)} rollout_*.mp4 from earlier runs. "
            f"They will be replaced as this run's episodes finish; pass a fresh --args.video-out-path "
            f"(or clear the directory) if you want each run kept separate."
        )

    results_file = None
    if args.results_out_path:
        pathlib.Path(args.results_out_path).parent.mkdir(parents=True, exist_ok=True)
        results_file = open(args.results_out_path, "a")
        if results_file.tell() == 0:
            results_file.write("episode_idx\tseed\tsuccess\tlength\n")

    n_envs = max(int(args.num_envs), 1)
    total_episodes = int(args.num_trials_per_task)
    n_workers = min(n_envs, total_episodes)

    # The parent is the only scheduler: it hands each worker exactly one episode
    # at a time through that worker's own queue and only re-fills the slot once
    # the worker reports back. Nothing races for work, so the parent always knows
    # which episode a worker had in flight - including when the worker is killed
    # outright (segfault / OOM kill) and never reports anything at all. Losing a
    # worker therefore costs at most that one episode, which gets re-queued into a
    # fresh worker instead of ending the whole evaluation.
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    # One private task queue per *process*, re-created on every spawn, so a stale
    # shutdown sentinel can never be read by the wrong worker.
    task_queues: dict[int, object] = {}

    pending = collections.deque(range(total_episodes))  # episodes not claimed yet
    assigned: dict[int, int | None] = dict.fromkeys(range(n_workers))
    live: dict[int, multiprocessing.process.BaseProcess] = {}
    recycled: set[int] = set()  # slots whose worker is exiting on purpose after one episode
    completed: dict[int, tuple[bool, int]] = {}
    abandoned: dict[int, str] = {}
    episode_restarts: collections.Counter = collections.Counter()
    worker_restarts: collections.Counter = collections.Counter()

    pbar = tqdm.tqdm(total=total_episodes, desc="Episodes")

    def _write_result(episode_idx: int, success: bool, length: int) -> None:
        if results_file is None:
            return
        seed = args.seed + episode_idx if args.seed is not None else -1
        results_file.write(f"{episode_idx}\t{seed}\t{int(success)}\t{length}\n")
        results_file.flush()

    def _abandon(episode_idx: int, reason: str) -> None:
        """Give up on one episode without giving up on the whole evaluation."""
        abandoned[episode_idx] = reason
        pbar.update(1)
        logging.error(f"[ALARM] Episode {episode_idx} abandoned ({reason}); counted as a failure")
        _write_result(episode_idx, False, 0)

    def _requeue(episode_idx: int, reason: str) -> None:
        episode_restarts[episode_idx] += 1
        if episode_restarts[episode_idx] > args.max_episode_restarts:
            _abandon(episode_idx, f"{reason}; already restarted {episode_restarts[episode_idx] - 1} time(s)")
        else:
            logging.warning(f"Re-queuing episode {episode_idx}: {reason}")
            pending.append(episode_idx)

    def _assign(worker_id: int) -> None:
        """Hand the next pending episode to an idle, live worker."""
        if assigned[worker_id] is not None or worker_id not in live or not pending:
            return
        episode_idx = pending.popleft()
        assigned[worker_id] = episode_idx
        task_queues[worker_id].put(episode_idx)

    def _spawn(worker_id: int) -> bool:
        if worker_restarts[worker_id] > args.max_worker_restarts:
            logging.error(
                f"Worker slot {worker_id} crashed {worker_restarts[worker_id] - 1} times, not restarting it"
            )
            return False
        worker_restarts[worker_id] += 1
        task_queues[worker_id] = ctx.Queue()
        proc = ctx.Process(
            target=_run_worker,
            args=(worker_id, task_queues[worker_id], args, result_queue),
        )
        proc.start()
        live[worker_id] = proc
        return True

    def _describe_exit(proc) -> str:
        code = proc.exitcode
        if code is not None and code < 0:
            try:
                return f"killed by {signal.Signals(-code).name} (exitcode {code})"
            except ValueError:
                return f"exitcode {code}"
        return f"exitcode {code}"

    def _handle(worker_id: int, episode_idx: int, success: bool | None, length) -> None:
        """Process one worker report and immediately refill that worker's slot."""
        if episode_idx < 0:
            # The worker could not even build its env / reach the policy server,
            # so there is no episode to re-queue - the reap logic restarts it.
            logging.error(f"Worker {worker_id} could not start: {length}")
            return
        assigned[worker_id] = None
        if episode_idx in completed or episode_idx in abandoned:
            # Duplicate report: an episode re-queued after a worker died also
            # made it back from the original worker. Keep the first result.
            return
        if success is None:
            _requeue(episode_idx, f"worker {worker_id} raised: {length}")
            return
        completed[episode_idx] = (bool(success), int(length))
        pbar.update(1)
        _write_result(episode_idx, bool(success), int(length))
        if args.one_episode_per_worker:
            # Retire this worker together with its env; the reap loop starts a
            # pristine process+env for the next episode. The sentinel makes the
            # worker close its env and exit cleanly (no leaked EGL context).
            recycled.add(worker_id)
            task_queues[worker_id].put(None)
        else:
            _assign(worker_id)

    for worker_id in range(n_workers):
        if _spawn(worker_id):
            _assign(worker_id)

    try:
        while len(completed) + len(abandoned) < total_episodes:
            # 1. Drain every report that is already waiting *before* looking for
            #    dead workers, so a worker that reported and then exited is not
            #    mistaken for one that lost its episode to a crash.
            try:
                _handle(*result_queue.get(timeout=2.0))
            except queue.Empty:
                pass
            else:
                while True:
                    try:
                        _handle(*result_queue.get_nowait())
                    except queue.Empty:
                        break

            # 2. Reap the workers that are gone. Because the parent owns every
            #    assignment, a worker that died without reporting still has its
            #    episode recorded in ``assigned`` - that is the episode to
            #    rescue, and it is the whole point of this loop.
            for worker_id in list(live):
                proc = live[worker_id]
                if proc.is_alive():
                    continue
                proc.join()
                del live[worker_id]
                orphan = assigned[worker_id]
                assigned[worker_id] = None
                if worker_id in recycled:
                    # Retired after one episode on purpose, so this is not a crash:
                    # give the slot back its full restart budget.
                    recycled.discard(worker_id)
                    worker_restarts[worker_id] = 0
                elif orphan is None:
                    logging.warning(
                        f"Worker {worker_id} exited ({_describe_exit(proc)}) with no episode in flight"
                    )
                else:
                    logging.error(
                        f"Worker {worker_id} died ({_describe_exit(proc)}) while running episode {orphan}"
                    )
                    _requeue(orphan, f"worker {worker_id} died ({_describe_exit(proc)})")
                if pending and _spawn(worker_id):
                    _assign(worker_id)

            # 3. No process left that could make progress: report what did finish
            #    instead of throwing the whole run away. A hard crash loop that
            #    exhausts the restart budgets ends up here.
            if not live:
                if not completed:
                    raise RuntimeError(
                        f"All {n_workers} worker(s) died before completing a single episode - see the "
                        f"exit codes logged above (a dead offscreen renderer or an unreachable policy "
                        f"server is the usual cause)."
                    )
                while pending:
                    _abandon(pending.popleft(), "no worker left to run it")
    finally:
        pbar.close()
        if results_file is not None:
            results_file.close()
        for proc in live.values():
            proc.terminate()
        for proc in live.values():
            proc.join()

    n_done = len(completed) + len(abandoned)
    n_success = sum(1 for success, _ in completed.values() if success)
    logging.info(f"Total episodes: {n_done}")
    logging.info(f"Total success rate: {n_success / n_done * 100:.1f}% ({n_success}/{n_done})")
    if abandoned:
        logging.error(
            f"{len(abandoned)} episode(s) were abandoned and counted as failures: {sorted(abandoned)} "
            f"(lower --num_envs if this keeps happening)"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_robocasa)
