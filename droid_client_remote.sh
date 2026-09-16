#!/usr/bin/env bash
# 连接 openpi policy server 运行 DROID 客户端（examples/droid/main.py）。
#
# 与真机直跑的区别：
#   1. 自动带上 API key（优先 OPENPI_API_KEY 环境变量，其次 logs/api_key），否则公网服务端会返回 401；
#   2. 可以直接指定远端服务地址（--host/--port），不必手工拼 tyro 参数；
#   3. 真机 / 无硬件模拟自动识别：Python 环境里能 `import droid` 走真机模式，
#      否则（如本 AutoDL 实例）自动注入 deploy/droid_sim_client/fakeroot（模拟 droid + moviepy），
#      并补齐 --external_camera left 与占位相机 ID；可用 --sim / --real 强制模式。
#
# 用法（在 openpi 仓库根目录）：
#   bash droid_client_remote.sh --host wss://uXXXX-xxx.bjb1.seetacloud.com --port 8443 --api-key <KEY>
#   bash droid_client_remote.sh --sim --host 127.0.0.1 --port 8765 --api-key test-key-123 --max_timesteps 20
#
# 说明：examples/droid/main.py 由 tyro.cli(Args) 驱动；本脚本的 --host/--port/--api-key
# 会被转换成 --remote_host / --remote_port / --api_key，其余参数原样透传（如 --max_timesteps 600）。
# Python 解释器探测顺序：OPENPI_PYTHON > OPENPI_VENV > 已激活的 python > 仓库 .venv/bin/python > uv run python。
set -euo pipefail

cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
用法: bash droid_client_remote.sh [--sim|--real] [--host HOST] [--port PORT] [--api-key KEY] [main.py 的其他参数...]

选项:
  --sim           强制模拟模式（无 DROID 硬件，自动注入 fakeroot）
  --real          强制真机模式（需要真实 droid 包）
  --host HOST     服务端地址；公网 wss 需带 wss:// 前缀（默认：main.py 的 0.0.0.0）
  --port PORT     服务端端口（默认：main.py 的 8000）
  --api-key KEY   服务端 API key；默认取环境变量 OPENPI_API_KEY，再取 logs/api_key
  -h, --help      显示帮助

示例:
  bash droid_client_remote.sh --host wss://uXXXX-xxx.bjb1.seetacloud.com --port 8443 --api-key <KEY>
  bash droid_client_remote.sh --sim --host 127.0.0.1 --port 8765 --api-key test-key-123 --max_timesteps 20
EOF
}

MODE="auto"
HOST=""
PORT=""
API_KEY="${OPENPI_API_KEY:-}"
PASSTHRU=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sim) MODE="sim"; shift ;;
    --real) MODE="real"; shift ;;
    --host) HOST="${2:?缺少 --host 的值}"; shift 2 ;;
    --port) PORT="${2:?缺少 --port 的值}"; shift 2 ;;
    --api-key) API_KEY="${2:?缺少 --api-key 的值}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) PASSTHRU+=("$1"); shift ;;
  esac
done

# ---- 探测 Python 解释器 ----
PY_CMD=()
if [[ -n "${OPENPI_PYTHON:-}" ]]; then
  PY_CMD=("$OPENPI_PYTHON")
elif [[ -n "${OPENPI_VENV:-}" && -x "${OPENPI_VENV}/bin/python" ]]; then
  PY_CMD=("${OPENPI_VENV}/bin/python")
elif command -v python >/dev/null 2>&1; then
  PY_CMD=("$(command -v python)")
elif [[ -x .venv/bin/python ]]; then
  PY_CMD=("$PWD/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PY_CMD=(uv run python)
else
  echo "❌ 找不到可用的 Python 解释器（可设置 OPENPI_PYTHON=/path/to/python）" >&2
  exit 1
fi

# ---- 模式自动识别 ----
if [[ "$MODE" == "auto" ]]; then
  if "${PY_CMD[@]}" -c 'import droid' >/dev/null 2>&1; then
    MODE="real"
  else
    MODE="sim"
  fi
fi

# ---- 组装参数（tyro 风格）----
has_flag() {
  local flag="$1" a
  for a in ${PASSTHRU[@]+"${PASSTHRU[@]}"}; do
    if [[ "$a" == "$flag" || "$a" == "$flag="* ]]; then return 0; fi
  done
  return 1
}

# 没显式给 key 时，读取 start_server.sh 生成的 key
if [[ -z "$API_KEY" && -f logs/api_key ]]; then
  API_KEY="$(<logs/api_key)"
fi

ARGS=()
if [[ -n "$HOST" ]]; then ARGS+=(--remote_host "$HOST"); fi
if [[ -n "$PORT" ]]; then ARGS+=(--remote_port "$PORT"); fi
if [[ -n "$API_KEY" ]]; then
  ARGS+=(--api_key "$API_KEY")
else
  echo "⚠️  未找到 API key（logs/api_key 不存在且未设置 OPENPI_API_KEY）；若服务端启用了鉴权会返回 401" >&2
fi

if [[ "$MODE" == "sim" ]]; then
  # 用 fakeroot 里的模拟 droid / moviepy 顶替真实依赖（本机无 DROID 硬件）
  export PYTHONPATH="$PWD/deploy/droid_sim_client/fakeroot:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
  if ! has_flag --external_camera; then ARGS+=(--external_camera left); fi
  if ! has_flag --left_camera_id; then ARGS+=(--left_camera_id left_cam); fi
  if ! has_flag --right_camera_id; then ARGS+=(--right_camera_id right_cam); fi
  if ! has_flag --wrist_camera_id; then ARGS+=(--wrist_camera_id wrist_cam); fi
  echo "ℹ️  模拟模式（无 DROID 硬件）：注入 fakeroot，external_camera=left，camera_id=left_cam/right_cam/wrist_cam" >&2
else
  if ! "${PY_CMD[@]}" -c 'import droid' >/dev/null 2>&1; then
    echo "❌ --real 模式需要当前 Python 环境中有 droid 包（DROID 官方环境）。" >&2
    echo "   请先激活 DROID 的 conda 环境；或用 --sim 走无硬件模拟模式。" >&2
    echo "   环境配置可运行: bash deploy/droid_real_client/setup_droid_env.sh" >&2
    exit 1
  fi
  export PYTHONPATH="$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
  echo "ℹ️  真机模式：使用真实 droid 包" >&2
fi

echo "连接服务端: ${HOST:-0.0.0.0}:${PORT:-8000}" >&2
exec "${PY_CMD[@]}" examples/droid/main.py ${ARGS[@]+"${ARGS[@]}"} ${PASSTHRU[@]+"${PASSTHRU[@]}"}
