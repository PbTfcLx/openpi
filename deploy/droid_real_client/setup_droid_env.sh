#!/usr/bin/env bash
# 真机（DROID 控制笔记本）环境配置 / 检查脚本。
#
# 运行前提：
#   1. 已激活 DROID 官方 conda 环境（含 droid 包与 ZED SDK，见 https://github.com/droid-dataset/droid）；
#   2. 在 openpi 仓库根目录下运行本脚本（需要 packages/openpi-client 与 examples/droid/main.py）。
#
# 用法：
#   bash deploy/droid_real_client/setup_droid_env.sh          # 检查并用 pip 安装缺失依赖
#   bash deploy/droid_real_client/setup_droid_env.sh --check  # 只检查不安装（预检）
#
# main.py 依赖 = droid（DROID 环境自带）+ openpi-client（dm-tree/msgpack/numpy/pillow/websockets）
#   + tyro + tqdm + pandas + moviepy(<2)。
# 注意 moviepy 必须 <2：main.py 使用 `from moviepy.editor import ...`，moviepy 2.x 已移除该模块。
set -euo pipefail

cd "$(dirname "$0")/../.." # openpi 仓库根目录

usage() {
  cat <<'EOF'
用法: bash deploy/droid_real_client/setup_droid_env.sh [--check]

  （默认）检查并用 pip 安装缺失的依赖
  --check 只检查、不安装（预检用）
运行前提：已激活 DROID 官方 conda 环境（含 droid 包、ZED SDK）。
EOF
}

CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -d packages/openpi-client || ! -f examples/droid/main.py ]]; then
  echo "❌ 请在 openpi 仓库根目录运行（当前缺少 packages/openpi-client 或 examples/droid/main.py）。" >&2
  exit 1
fi

# ---- 探测 Python ----
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  echo "❌ 找不到 python3/python，请先激活 DROID 的 conda 环境（conda activate <环境名>）。" >&2
  exit 1
fi

if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "❌ main.py 需要 Python ≥ 3.10（使用了 str | None 类型语法）。" >&2
  exit 1
fi

echo "🐍 Python: $PY （$( "$PY" -c 'import sys; print(sys.version.split()[0])' )）"

MISSING=0

check() { # $1=名称 $2=import 语句；失败时打印并计数
  if "$PY" -c "$2" >/dev/null 2>&1; then
    echo "✅ $1"
    return 0
  fi
  echo "❌ 缺 $1"
  MISSING=$((MISSING + 1))
  return 1
}

pip_missing() { # $1=名称 $2=import 语句 $3=pip 包规格
  local name="$1" imp="$2" spec="$3"
  if check "$name" "$imp"; then return 0; fi
  if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "   （将执行: $PY -m pip install \"$spec\"）"
    return 0
  fi
  echo "→ $PY -m pip install \"$spec\""
  if "$PY" -m pip install "$spec" && "$PY" -c "$imp" >/dev/null 2>&1; then
    echo "✅ $name（安装成功）"
    MISSING=$((MISSING - 1))
  else
    echo "⚠️  $name 安装失败，请手动处理" >&2
  fi
}

echo "---- 依赖检查 ----"

# droid 包（前置，由 DROID 官方环境提供，本脚本不安装）
check "droid 包（DROID 官方环境）" "from droid.robot_env import RobotEnv" ||
  echo "   → 请先按 DROID 官方指引安装并激活环境: https://github.com/droid-dataset/droid"

# openpi-client（从本仓库 editable 安装）
if ! check "openpi_client（websocket 客户端）" "from openpi_client import websocket_client_policy"; then
  if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "   （将执行: $PY -m pip install -e packages/openpi-client）"
  else
    echo "→ $PY -m pip install -e packages/openpi-client"
    if "$PY" -m pip install -e packages/openpi-client &&
      "$PY" -c 'from openpi_client import websocket_client_policy' >/dev/null 2>&1; then
      echo "✅ openpi_client（安装成功）"
      MISSING=$((MISSING - 1))
    else
      echo "⚠️  openpi_client 安装失败，请手动处理" >&2
    fi
  fi
fi

pip_missing "tyro（命令行解析）" "import tyro" "tyro"
pip_missing "tqdm（进度条）" "import tqdm" "tqdm"
pip_missing "pandas（结果表）" "import pandas" "pandas"
pip_missing "moviepy.editor（录像保存）" "from moviepy.editor import ImageSequenceClip" "moviepy<2"

echo "-----------------------------------"
if [[ $MISSING -eq 0 ]]; then
  echo "🎉 依赖就绪（真机模式）。"
  echo
  echo "下一步："
  echo "  1) 用 ZED_Explorer 查看三路相机 ID（左/右外部相机 + 腕部相机）"
  echo "  2) 在 openpi 仓库根目录运行（DROID 环境激活状态下）："
  echo "     bash droid_client_remote.sh --real \\"
  echo "         --host wss://<服务端地址> --port 8443 --api-key <KEY> \\"
  echo "         --external_camera left \\"
  echo "         --left_camera_id <左相机ID> --right_camera_id <右相机ID> --wrist_camera_id <腕部ID>"
  echo "     或手工方式（官方 README 风格）："
  echo "     python3 examples/droid/main.py --remote_host=<服务端地址> --remote_port=8443 --api_key=<KEY> \\"
  echo "         --external_camera=left \\"
  echo "         --left_camera_id=<左相机ID> --right_camera_id=<右相机ID> --wrist_camera_id=<腕部ID>"
  exit 0
else
  echo "⚠️  仍有 $MISSING 项未就绪（见上）。"
  if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "   去掉 --check 重跑可自动安装 pip 依赖。"
  fi
  exit 1
fi
