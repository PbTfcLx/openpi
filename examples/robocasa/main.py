import collections
import dataclasses
import logging
import multiprocessing
import pathlib
import queue

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
    # API key of the policy server, if it was started with one (see deploy/autodl/).
    # For a remote server reachable over the public internet, pass e.g.
    # `--host wss://uXXXX-xxx.bjb1.seetacloud.com --port 8443`.
    api_key: str | None = None
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    env_name: str = "robocasa_panda_omron/CloseDoubleDoor_PandaOmron_Env"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 100  # Number of rollouts per task
    max_steps: int = 720

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/robocasa/test_token_len/checkpoint-21000/videos/closedoubledoor"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)
    num_envs : int = 3

    #################################################################################################################
    # Multi-env / video recording
    #################################################################################################################
    save_video: bool = True  # Whether to save per-episode replay videos
    steps_per_render: int = 1  # Record a video frame every N env steps
    num_retries: int = 3  # Re-run an episode this many times if it hits NaN

    #################################################################################################################
    # Determinism / reproducibility
    #################################################################################################################
    deterministic: bool = True  # Pin action noise to (episode seed, replan index) so runs are reproducible and different checkpoints can be compared on identical scenes
    results_out_path: str = ""  # Optional TSV to dump per-episode results (episode_idx, seed, success, length); keep the same seed/num_trials when comparing checkpoints

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

    Pulls the next episode index from the shared ``task_queue`` as soon as the
    previous episode finishes, so a fast worker keeps picking up work instead of
    idling behind a slow worker. Episode indices still map to deterministic seeds
    via ``seed + episode_idx``.
    """
    env = None
    try:
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port, api_key=args.api_key)
        env = _make_env(args.env_name, args.max_steps)

        while True:
            episode_idx = task_queue.get()
            if episode_idx is None:
                # All episodes have already been claimed by other workers.
                break

            seed = args.seed + episode_idx if args.seed is not None else None
            success, length = False, 0
            for attempt in range(args.num_retries):
                try:
                    success, length, hit_nan, render_dead = _run_episode(
                        env, client, args, seed, episode_idx, attempt
                    )
                except Exception as e:  # noqa: BLE001 - report per-episode failures
                    logging.error(f"Worker {worker_id} failed on episode {episode_idx}: {e}")
                    result_queue.put((episode_idx, None, str(e)))
                    return
                if not (hit_nan or render_dead):
                    break
                if attempt < args.num_retries:
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
            result_queue.put((episode_idx, bool(success), length))
    except Exception as e:  # noqa: BLE001 - report setup failures to the parent
        logging.error(f"Worker {worker_id} failed: {e}")
        result_queue.put((worker_id, None, f"worker setup failed: {e}"))
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

    results_file = None
    if args.results_out_path:
        pathlib.Path(args.results_out_path).parent.mkdir(parents=True, exist_ok=True)
        results_file = open(args.results_out_path, "a")
        if results_file.tell() == 0:
            results_file.write("episode_idx\tseed\tsuccess\tlength\n")

    n_envs = max(int(args.num_envs), 1)
    total_episodes = int(args.num_trials_per_task)
    n_workers = min(n_envs, total_episodes)

    # Each worker owns its own env and policy client. Workers pull the next
    # episode index from a shared queue, so a fast worker immediately starts a
    # new episode instead of idling behind a slow one.
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    task_queue = ctx.Queue()
    for episode_idx in range(total_episodes):
        task_queue.put(episode_idx)
    for _ in range(n_workers):
        task_queue.put(None)  # one termination sentinel per worker

    workers = [
        ctx.Process(target=_run_worker, args=(worker_id, task_queue, args, result_queue))
        for worker_id in range(n_workers)
    ]

    for worker in workers:
        worker.start()

    pbar = tqdm.tqdm(total=total_episodes, desc="Episodes")
    successes: list[bool] = []
    try:
        while len(successes) < total_episodes:
            try:
                episode_idx, success, length = result_queue.get(timeout=2.0)
            except queue.Empty:
                if all(not worker.is_alive() for worker in workers):
                    raise RuntimeError(
                        "All worker processes died before completing all episodes."
                    )
                continue

            if success is None:
                raise RuntimeError(f"Episode {episode_idx} failed in a worker: {length}")

            successes.append(bool(success))
            pbar.update(1)
            if results_file is not None:
                seed = args.seed + episode_idx if args.seed is not None else -1
                results_file.write(f"{episode_idx}\t{seed}\t{int(bool(success))}\t{length}\n")
                results_file.flush()
    finally:
        pbar.close()
        if results_file is not None:
            results_file.close()
        for worker in workers:
            worker.terminate()
        for worker in workers:
            worker.join()

    n_done = len(successes)
    n_success = int(sum(successes))
    logging.info(f"Total episodes: {n_done}")
    logging.info(
        f"Total success rate: {n_success / n_done * 100:.1f}% ({n_success}/{n_done})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_robocasa)
