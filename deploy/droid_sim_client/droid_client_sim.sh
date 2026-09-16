#!/usr/bin/env bash
# 无硬件联调入口（兼容保留）：等价于 `bash droid_client_remote.sh --sim`。
# 模式识别、默认参数（--external_camera left / 占位相机 ID）与 fakeroot 注入都在
# droid_client_remote.sh 里，本脚本只是薄封装，方便老命令/文档继续可用。
#
# 用法（在 openpi 仓库根目录）:
#   bash deploy/droid_sim_client/droid_client_sim.sh \
#       --host wss://uXXXX-xxx.bjb1.seetacloud.com --port 8443 --api-key <KEY>
#   bash deploy/droid_sim_client/droid_client_sim.sh \
#       --host 127.0.0.1 --port 8765 --api-key test-key-123 --max_timesteps 20
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SIM_DIR/../.."  # openpi 根目录

exec bash droid_client_remote.sh --sim "$@"
