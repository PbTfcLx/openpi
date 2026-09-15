#!/usr/bin/env bash
# 连接 openpi policy server 运行 robocasa 仿真评估（客户端）。
#
# 与 client.sh 的区别：
#   1. 自动带上 API key（优先 OPENPI_API_KEY 环境变量，其次 logs/api_key），否则服务端会返回 HTTP 401；
#   2. 可以直接指定远端服务地址（--host/--port），不必手工拼 tyro 的 --args.* 参数；
#   3. 其余参数原样透传给 examples/robocasa/main.py。
#
# 用法:
#   bash client_remote.sh                                   # 连本机 127.0.0.1:8000
#   bash client_remote.sh --host wss://uXXX-xxx.bjb1.seetacloud.com --port 8443
#   bash client_remote.sh --num_envs 1 --args.num_trials_per_task 1 --args.max_steps 60 --args.no-save-video
#
# 说明：examples/robocasa/main.py 由 tyro.cli(eval_robocasa) 驱动，模型服务相关参数带
# --args. 前缀（如 --args.num_envs），布尔开关写 --args.no-xxx；本脚本的 --host/--port/--api-key
# 会被转换成对应的 --args.host / --args.port / --args.api-key。
set -euo pipefail

cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
用法: bash client_remote.sh [--host HOST] [--port PORT] [--api-key KEY] [main.py 的其他参数...]

选项:
  --host HOST     服务端地址；公网 wss 需带 wss:// 前缀（默认：main.py 的 0.0.0.0）
  --port PORT     服务端端口（默认：main.py 的 8000）
  --api-key KEY   服务端 API key；默认取环境变量 OPENPI_API_KEY，再取 logs/api_key
  -h, --help      显示帮助

示例:
  bash client_remote.sh --host wss://uXXXX-xxx.bjb1.seetacloud.com --port 8443
  bash client_remote.sh --num_envs 1 --args.num_trials_per_task 1 --args.max_steps 60 --args.no-save-video
EOF
}

HOST=""
PORT=""
API_KEY="${OPENPI_API_KEY:-}"
PASSTHRU=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="${2:?缺少 --host 的值}"; shift 2 ;;
    --port) PORT="${2:?缺少 --port 的值}"; shift 2 ;;
    --api-key) API_KEY="${2:?缺少 --api-key 的值}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) PASSTHRU+=("$1"); shift ;;
  esac
done

# 没显式给 key 时，读取 start_server.sh 生成的 key
if [[ -z "$API_KEY" && -f logs/api_key ]]; then
  API_KEY="$(<logs/api_key)"
fi

ARGS=()
if [[ -n "$HOST" ]]; then ARGS+=(--args.host "$HOST"); fi
if [[ -n "$PORT" ]]; then ARGS+=(--args.port "$PORT"); fi
if [[ -n "$API_KEY" ]]; then
  ARGS+=(--args.api-key "$API_KEY")
else
  echo "⚠️  未找到 API key（logs/api_key 不存在且未设置 OPENPI_API_KEY）；若服务端启用了鉴权会返回 401" >&2
fi

source ../Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/activate
export PYTHONPATH=packages/openpi-client/src/

echo "连接服务端: ${HOST:-0.0.0.0}:${PORT:-8000}"
python examples/robocasa/main.py "${ARGS[@]}" ${PASSTHRU[@]+"${PASSTHRU[@]}"}
