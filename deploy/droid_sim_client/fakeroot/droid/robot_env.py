"""模拟的 droid.robot_env：只提供 main.py 需要的接口，发随机图像/状态，用于无硬件联调。

动作不会被执行，只打印前几步，方便确认"客户端取到动作块 -> 逐条下发"的流程。
"""

import numpy as np


class RobotEnv:
    def __init__(self, action_space=None, gripper_action_space=None):
        print(
            f"[sim droid] RobotEnv(action_space={action_space!r}, gripper_action_space={gripper_action_space!r})",
            flush=True,
        )
        self.step_count = 0

    def get_observation(self):
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        return {
            "image": {
                # 键名需同时包含相机 ID 和 "left"（与真实 DROID 相机键规则一致）
                "left_cam-left": img,
                "right_cam-left": img,
                "wrist_cam-left": img,
            },
            "robot_state": {
                "cartesian_position": np.zeros(6, dtype=np.float32),
                "joint_positions": np.random.rand(7).astype(np.float32),
                "gripper_position": 0.5,
            },
        }

    def step(self, action):
        self.step_count += 1
        if self.step_count <= 3 or self.step_count % 8 == 0:
            print(f"[sim droid] step #{self.step_count} action={np.round(np.asarray(action), 3)}", flush=True)

    def reset(self):
        self.step_count = 0
