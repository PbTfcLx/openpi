# DROID 客户端无硬件联调环境

本目录用于在**没有 Franka + ZED 相机的机器**（如本 AutoDL 实例）上把 **DROID 客户端**跑起来，
验证「客户端 ↔ openpi policy 服务端」的完整链路：可以连你们远端已经起好的服务端，也可以连本目录的假服务端。

> 真机（DROID 控制笔记本）请直接使用仓库根目录的 `droid_client_remote.sh`，本目录不参与。

## 组成

| 文件 | 作用 |
| --- | --- |
| `droid_client_sim.sh` | 入口：给 `droid_client_remote.sh` 注入模拟机器人（fakeroot）后运行 |
| `fakeroot/droid/` | 模拟 `droid.robot_env.RobotEnv`（发随机图像/关节状态，动作用打印代替执行） |
| `fakeroot/moviepy/` | 假 `moviepy`（不真正写视频文件） |
| `dummy_server.py` | 本地假 policy 服务（openpi 自带 `WebsocketPolicyServer`，返回全零动作块，带 API key 鉴权） |

## 用法

### 0. 统一入口（推荐）

模式自动识别已经收进仓库根目录的 `droid_client_remote.sh`：
**Python 环境里没有 `droid` 包时会自动走模拟模式**，用法和 robocasa 的 `client_remote.sh` 一致：

```bash
cd /root/autodl-tmp/openpi
bash droid_client_remote.sh \
    --host wss://uXXXX-xxx.bjb1.seetacloud.com --port 8443 --api-key <KEY>
```

默认自动补 `--external_camera left` 与占位相机 ID；可用 `--sim` / `--real` 强制模式。

### 1. 连远端服务端（本目录入口，等价于上面的 `--sim`）

```bash
cd /root/autodl-tmp/openpi
bash deploy/droid_sim_client/droid_client_sim.sh \
    --host wss://uXXXX-xxx.bjb1.seetacloud.com --port 8443 \
    --api-key <KEY>
```

- 相机 ID 与 `--external_camera` 无需指定（模拟模式会自动补 `left` / `left_cam` 等默认值）。
- 启动后输入语言指令回车，进入 20/600 步的执行循环；按 Ctrl+C 可提前结束。

### 2. 纯本地自检（假服务端）

```bash
# 终端 A：起假服务
.venv/bin/python deploy/droid_sim_client/dummy_server.py

# 终端 B：跑客户端
bash droid_client_remote.sh --sim --host 127.0.0.1 --port 8765 \
    --api-key test-key-123 --max_timesteps 20
```

## 说明与限制

- 模拟机器人只验证 **观测格式、鉴权、链路、动作块处理**，不反映真机行为（图像是随机噪声）。
- 动作块长度 T 由服务端 policy 决定（pi0_droid=10、pi05_droid=15、finetune 可能为 16 等），
  客户端只要求形状为 `(T, 8)`；若 T < `--open_loop_horizon`（默认 8），请把 `--open_loop_horizon` 调小。
- 依赖：直接用仓库自带的 `.venv`（含 `openpi_client`、`tyro`、`pandas`、`Pillow`、`websockets`），
  不需要安装 `droid`/`moviepy`（本目录的 fakeroot 已提供）。
- 真机环境（控制笔记本）需要：DROID 的 conda 环境（含 `droid` 包、`pyzed`）+ `pip install -e packages/openpi-client`
  + `tyro`（以及 `moviepy`、`pandas` 等示例依赖）。
