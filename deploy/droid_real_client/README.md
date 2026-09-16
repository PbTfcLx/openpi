# DROID 真机客户端（控制笔记本）配置

本目录用于在 **DROID 控制笔记本**（连接 Franka 机械臂 + ZED 相机的真机）上配置并运行
`examples/droid/main.py`：观测从真机本地采集（`RobotEnv`），推理在远端 GPU 的 openpi policy
服务端完成（websocket），返回的动作块再下发给真机执行。

> 无硬件联调（如本 AutoDL 实例）请用 `deploy/droid_sim_client/`，与本目录互不依赖。

## 前置要求

1. **DROID 官方环境**（conda，含 `droid` 包与 ZED SDK）已装好 —— 参考 <https://github.com/droid-dataset/droid>；
2. openpi 仓库已放到控制笔记本（本目录脚本需要 `packages/openpi-client` 与 `examples/droid/main.py`）；
3. 控制笔记本可访问服务端（我们的服务端在 AutoDL 公网，形如 `wss://xxxx.bjb1.seetacloud.com:8443`）。

## 步骤

```bash
# 0) 激活 DROID 环境
conda activate <你的 DROID 环境名>

# 1) 在 openpi 仓库根目录检查/安装依赖（droid 包必须由 DROID 官方环境提供）
bash deploy/droid_real_client/setup_droid_env.sh --check   # 先预检
bash deploy/droid_real_client/setup_droid_env.sh           # 再安装缺失的 pip 依赖

# 2) 用 ZED_Explorer 查看相机 ID（左/右外部相机、腕部相机）
ZED_Explorer

# 3) 运行（推荐入口：自动探测 conda Python、自动带 API key、参数自动转换）
bash droid_client_remote.sh --real \
    --host wss://<服务端地址> --port 8443 --api-key <KEY> \
    --external_camera left \
    --left_camera_id <左相机ID> --right_camera_id <右相机ID> --wrist_camera_id <腕部ID>
```

说明：

- `--left_camera_id / --right_camera_id / --wrist_camera_id / --external_camera` 会原样透传给
  `main.py`（tyro 参数），**无需修改源码**；不传相机 ID 时只适用于模拟模式，真机必须传真实 ID；
- 服务端 API key 不要写入任何仓库文件，用运行参数或环境变量 `OPENPI_API_KEY` 传入即可；
- 首次推理约 30s（服务端冷启动/JIT），之后每个动作块约 100ms，属正常现象。

## 手工方式（等价，不依赖入口脚本）

```bash
python3 examples/droid/main.py \
    --remote_host=<服务端地址> --remote_port=8443 --api_key=<KEY> \
    --external_camera=left \
    --left_camera_id=<左相机ID> --right_camera_id=<右相机ID> --wrist_camera_id=<腕部ID>
```

（官方 README 的“复制 main.py 到 `$DROID_ROOT/scripts` 并改相机 ID”步骤在本仓库中可省略，
直接传命令行参数即可。）

## main.py 依赖清单

| 包 | 来源 |
| --- | --- |
| `droid`（+ ZED SDK） | DROID 官方环境，本目录脚本不安装 |
| `openpi_client` | `pip install -e packages/openpi-client`（自动带 dm-tree/msgpack/numpy/pillow/websockets） |
| `tyro`、`tqdm`、`pandas` | pip |
| `moviepy<2` | pip；**必须 <2**：main.py 使用 `moviepy.editor`，moviepy 2.x 已移除该模块 |

## 故障排查

| 现象 | 处理 |
| --- | --- |
| `ModuleNotFoundError: No module named 'droid'` | 没有激活 DROID conda 环境（或环境安装不完整） |
| `ModuleNotFoundError: No module named 'moviepy.editor'` | 误装了 moviepy 2.x → `pip install "moviepy<2"` |
| 服务端返回 `401` | API key 不对；确认 `--api-key` 与服务端启动时一致 |
| 首次推理很慢（≈30s） | 服务端冷启动/JIT，后续恢复正常，非故障 |
| 首次连接 `TimeoutError`（握手超时） | 偶发网络抖动，重试即可 |
| 找不到相机 / 图像全黑 | 用 `ZED_Explorer` 确认连接与视角，必要时重插相机 |
