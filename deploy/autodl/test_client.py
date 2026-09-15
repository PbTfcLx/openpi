#!/usr/bin/env python3
"""连通性自检：向 openpi policy server 发一次随机观测，打印动作 chunk 与耗时。

用法：
    # 实例内部自测
    python deploy/autodl/test_client.py --host 127.0.0.1 --port 8000 --api-key <key>

    # 通过 AutoDL 自定义服务（6006）从公网测试
    python deploy/autodl/test_client.py --uri wss://uXXXX-xxx.bjb1.seetacloud.com:8443 --api-key <key>

注意：观测的键必须与服务端 policy 的输入匹配，这里内置了几个常用预设（--preset）。
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from openpi_client import websocket_client_policy


def _rand_image(size: int) -> np.ndarray:
    return np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)


def make_observation(preset: str, size: int, prompt: str) -> dict:
    """构造各环境的最小合法观测（键名与对应 policy 的输入一致）。"""
    if preset == "robocasa":
        # RobocasaInputs 会读取 right_image，这里用全零占位（真实 eval 中来自 gym 环境）
        return {
            "observation/state": np.random.rand(8).astype(np.float32),
            "observation/image": _rand_image(size),
            "observation/wrist_image": _rand_image(size),
            "observation/right_image": np.zeros((size, size, 3), dtype=np.uint8),
            "prompt": prompt,
        }
    if preset == "libero":
        return {
            "observation/state": np.random.rand(8).astype(np.float32),
            "observation/image": _rand_image(size),
            "observation/wrist_image": _rand_image(size),
            "prompt": prompt,
        }
    if preset == "droid":
        return {
            "observation/exterior_image_1_left": _rand_image(size),
            "observation/wrist_image_left": _rand_image(size),
            "observation/joint_position": np.random.rand(7).astype(np.float32),
            "observation/gripper_position": np.random.rand(1).astype(np.float32),
            "prompt": prompt,
        }
    if preset == "aloha":
        return {
            "state": np.ones((14,), dtype=np.float32),
            "images": {
                "cam_high": _rand_image(size).transpose(2, 0, 1),
                "cam_left_wrist": _rand_image(size).transpose(2, 0, 1),
                "cam_right_wrist": _rand_image(size).transpose(2, 0, 1),
            },
            "prompt": prompt,
        }
    raise ValueError(f"未知 preset: {preset}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uri", default=None, help="完整地址，如 ws://127.0.0.1:8000 或 wss://xxx:8443（优先于 --host/--port）")
    parser.add_argument("--host", default="127.0.0.1", help="服务端地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8000, help="服务端端口（默认 8000）")
    parser.add_argument("--api-key", default=None, help="服务端启用了 --api-key 时必填")
    parser.add_argument("--preset", default="robocasa", choices=["robocasa", "libero", "droid", "aloha"])
    parser.add_argument("--image-size", type=int, default=224, help="输入图像边长（默认 224）")
    parser.add_argument("--prompt", default="do something", help="语言指令")
    parser.add_argument("--num-steps", type=int, default=3, help="重复推理次数（第一次会触发 JIT 编译，较慢）")
    args = parser.parse_args()

    if args.uri:
        client = websocket_client_policy.WebsocketClientPolicy(host=args.uri, api_key=args.api_key)
        target = args.uri
    else:
        client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port, api_key=args.api_key
        )
        target = f"{args.host}:{args.port}"

    print(f"已连接 {target}")
    print(f"服务端 metadata: {client.get_server_metadata()}")

    observation = make_observation(args.preset, args.image_size, args.prompt)
    for step in range(args.num_steps):
        start = time.monotonic()
        result = client.infer(observation)
        elapsed = (time.monotonic() - start) * 1000
        actions = np.asarray(result["actions"])
        timing = result.get("server_timing", {})
        infer_ms = timing.get("infer_ms", float("nan"))
        print(
            f"[{step + 1}/{args.num_steps}] actions.shape={actions.shape} dtype={actions.dtype} "
            f"客户端往返={elapsed:.1f}ms 服务端推理={infer_ms:.1f}ms"
        )

    print("✅ 服务可正常调用")


if __name__ == "__main__":
    main()
