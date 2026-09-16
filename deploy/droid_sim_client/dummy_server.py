"""本地假 policy 服务：用 openpi 自带的 WebsocketPolicyServer + 固定动作块，用于 DROID 客户端联调。

启动：/root/autodl-tmp/openpi/.venv/bin/python deploy/droid_sim_client/dummy_server.py
"""

import numpy as np

from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class DummyDroidPolicy:
    def __init__(self):
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        print(f"[dummy server] infer #{self.calls}: keys={sorted(obs.keys())}", flush=True)
        if self.calls == 1:
            for k in [
                "observation/exterior_image_1_left",
                "observation/wrist_image_left",
                "observation/joint_position",
                "observation/gripper_position",
            ]:
                v = np.asarray(obs[k])
                print(f"[dummy server]   {k}: shape={v.shape} dtype={v.dtype}", flush=True)
            print(f"[dummy server]   prompt={obs.get('prompt')!r}", flush=True)
        return {"actions": np.zeros((10, 8), dtype=np.float32)}

    def reset(self):
        pass


if __name__ == "__main__":
    server = WebsocketPolicyServer(
        DummyDroidPolicy(),
        host="127.0.0.1",
        port=8765,
        api_key="test-key-123",
        metadata={"policy": "dummy-droid", "action_horizon": 10, "action_dim": 8},
    )
    print("[dummy server] listening on ws://127.0.0.1:8765 (api-key=test-key-123)", flush=True)
    server.serve_forever()
