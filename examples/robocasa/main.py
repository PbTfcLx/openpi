import collections
import dataclasses
import logging
import multiprocessing
import pathlib
import queue

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
    video_out_path: str = "data/robocasa/checkpoint-20000/videos/opendrawer"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)
    num_envs : int = 12

    #################################################################################################################
    # Multi-env / video recording
    #################################################################################################################
    save_video: bool = True  # Whether to save per-episode replay videos
    steps_per_render: int = 1  # Record a video frame every N env steps
    num_retries: int = 3  # Re-run an episode this many times if it hits NaN

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


def _make_replay_frame(obs) -> np.ndarray:
    """Tile the three camera views into one wide frame for the replay video.

    The order matches the policy input: side_0, side_1, wrist.
    """
    return np.concatenate(
        (
            np.ascontiguousarray(obs["video.res256_image_side_0"]),
            np.ascontiguousarray(obs["video.res256_image_side_1"]),
            np.ascontiguousarray(obs["video.res256_image_wrist_0"]),
        ),
        axis=1,
    )


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
        )
    )


def _run_episode(
    env: gym.Env,
    client: _websocket_client_policy.WebsocketClientPolicy,
    args: Args,
    seed: int | None,
    episode_idx: int,
    attempt: int = 0,
) -> tuple[bool, int, bool]:
    """Run one attempt of an episode.

    Returns ``(success, num_env_steps, hit_nan)``. ``hit_nan`` is True when the
    simulator entered an invalid (NaN) state, which the caller may retry.
    """
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()

    task_description = str(obs["annotation.human.action.task_description"])
    action_plan = collections.deque()
    replay_images = []
    t = 0
    done = False
    hit_nan = False

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
            action_chunk = client.infer(element)["actions"]
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

        if args.save_video and (
            t % args.steps_per_render == 0 or success or terminated or truncated
        ):
            replay_images.append(_make_replay_frame(obs))

        if success or terminated or truncated:
            done = success
            break
        t += 1

    if args.save_video and replay_images and not hit_nan:
        suffix = "success" if done else "failure"
        video_path = (
            pathlib.Path(args.video_out_path)
            / f"rollout_{episode_idx:04d}_{suffix}.mp4"
        )
        try:
            imageio.mimwrite(video_path, replay_images, fps=40, codec="libx264")
        except Exception as e:  # noqa: BLE001 - video is best-effort
            logging.error(f"Failed to save video {video_path}: {e}")

    return done, t + 1, hit_nan


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
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
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
                    success, length, hit_nan = _run_episode(
                        env, client, args, seed, episode_idx, attempt
                    )
                except Exception as e:  # noqa: BLE001 - report per-episode failures
                    logging.error(f"Worker {worker_id} failed on episode {episode_idx}: {e}")
                    result_queue.put((episode_idx, None, str(e)))
                    return
                if not hit_nan:
                    break
                if attempt < args.num_retries:
                    logging.warning(
                        f"Episode {episode_idx} attempt {attempt + 1}/{args.num_retries} "
                        f"hit NaN, retrying..."
                    )
            else:
                # Retry budget exhausted and still NaN.
                logging.error(
                    f"[ALARM] Episode {episode_idx}: NaN persisted after "
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


def eval_robocasa(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

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
            # logging.info(
            #     f"Episode {episode_idx}: success={bool(success)}, length={length}"
            # )
    finally:
        pbar.close()
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
