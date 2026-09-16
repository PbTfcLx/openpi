#!/usr/bin/env bash
# 在 AutoDL 实例内以后台方式启动 openpi policy server（websocket 服务）。
#
# 常用命令：
#   bash deploy/autodl/start_server.sh                       # 默认 pi05_robocasa + checkpoint/6000，监听 8000
#   bash deploy/autodl/start_server.sh --port 8000 --api-key my-secret
#   bash deploy/autodl/start_server.sh --config pi05_libero --dir gs://openpi-assets/checkpoints/pi05_libero
#   bash deploy/autodl/stop_server.sh --port 8000            # 停止服务
#
# 启动后：
#   * 服务监听 0.0.0.0:<port>，健康检查 http://127.0.0.1:<port>/healthz
#   * 日志 logs/serve_<port>.log，PID 文件 logs/serve_<port>.pid
#   * 外部访问方式见 deploy/autodl/README.md（SSH 隧道 / AutoDL 自定义服务）
set -euo pipefail

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$OPENPI_ROOT"

CONFIG="${OPENPI_CONFIG:-pi05_robocasa}"
CKPT_DIR="${OPENPI_CHECKPOINT_DIR:-checkpoint/4999}"
PORT="${OPENPI_PORT:-8000}"
API_KEY="${OPENPI_API_KEY:-}"
DEFAULT_PROMPT="${OPENPI_DEFAULT_PROMPT:-}"
GPU="${CUDA_VISIBLE_DEVICES:-}"
HEALTH_TIMEOUT="${OPENPI_HEALTH_TIMEOUT:-900}"

usage() {
  cat <<'EOF'
用法: bash deploy/autodl/start_server.sh [选项]

选项:
  --config <name>         训练配置名，如 pi05_robocasa / pi05_libero / pi05_droid（默认 pi05_robocasa）
  --dir <path>            checkpoint 目录，如 checkpoint/6000 或 gs://openpi-assets/...（默认 checkpoint/6000）
  --port <int>            监听端口（默认 8000）
  --api-key <key>         客户端鉴权密钥；缺省时读取环境变量 OPENPI_API_KEY，
                          若都没有则自动生成并保存到 logs/api_key
  --default-prompt <str>  观测里没有 prompt 时使用的默认指令
  --gpu <id>              指定 GPU，如 0（等价于设置 CUDA_VISIBLE_DEVICES）
  -h, --help              显示帮助

环境变量:
  OPENPI_CONFIG / OPENPI_CHECKPOINT_DIR / OPENPI_PORT / OPENPI_API_KEY / OPENPI_DEFAULT_PROMPT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --dir|--policy-dir) CKPT_DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --default-prompt) DEFAULT_PROMPT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

LOG_DIR="$OPENPI_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/serve_${PORT}.log"
PID_FILE="$LOG_DIR/serve_${PORT}.pid"

# ---------------- 选择 python 解释器 ----------------
if [[ -x "$OPENPI_ROOT/.venv/bin/python" ]]; then
  PY_CMD=("$OPENPI_ROOT/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PY_CMD=(uv run python)
else
  echo "找不到 .venv/bin/python，也找不到 uv；请先按 README 安装依赖（uv sync）" >&2
  exit 1
fi

# ---------------- API key ----------------
if [[ -z "$API_KEY" ]]; then
  if [[ -f "$LOG_DIR/api_key" ]]; then
    API_KEY="$(<"$LOG_DIR/api_key")"
  else
    API_KEY="$("${PY_CMD[@]}" -c 'import secrets; print(secrets.token_hex(16))')"
    (umask 077 && printf '%s' "$API_KEY" > "$LOG_DIR/api_key")
    echo "已自动生成 API key 并保存到 $LOG_DIR/api_key"
  fi
fi

# ---------------- 停掉同一端口上的旧服务 ----------------
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(<"$PID_FILE")"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "停止旧进程 PID=$OLD_PID ..."
    kill "$OLD_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$OLD_PID" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$OLD_PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi

# ---------------- 启动 ----------------
echo "启动 openpi policy server:"
echo "  config     = $CONFIG"
echo "  checkpoint = $CKPT_DIR"
echo "  port       = $PORT"
echo "  log        = $LOG_FILE"

ENV_VARS=(
  # 推理场景不要让 JAX 预占全部显存
  "XLA_PYTHON_CLIENT_PREALLOCATE=false"
)
[[ -n "$GPU" ]] && ENV_VARS+=("CUDA_VISIBLE_DEVICES=$GPU")

SERVE_ARGS=(
  scripts/serve_policy.py
  "--port=$PORT"
  "--api-key=$API_KEY"
  policy:checkpoint
  "--policy.config=$CONFIG"
  "--policy.dir=$CKPT_DIR"
)
[[ -n "$DEFAULT_PROMPT" ]] && SERVE_ARGS+=("--default-prompt=$DEFAULT_PROMPT")

nohup env "${ENV_VARS[@]}" "${PY_CMD[@]}" "${SERVE_ARGS[@]}" >>"$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo "  pid        = $NEW_PID"

# ---------------- 等待就绪 ----------------
echo "等待服务就绪（模型加载可能需要几分钟，首次推理还会触发 JIT 编译）..."
READY=0
if command -v curl >/dev/null 2>&1; then
  HEALTH_CMD=(curl -fsS -m 2 "http://127.0.0.1:$PORT/healthz")
else
  HEALTH_CMD=("${PY_CMD[@]}" -c "import sys,urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/healthz',timeout=2)")
fi

for _ in $(seq 1 "$HEALTH_TIMEOUT"); do
  if ! kill -0 "$NEW_PID" 2>/dev/null; then
    echo "进程已退出，日志末尾：" >&2
    tail -n 30 "$LOG_FILE" >&2
    exit 1
  fi
  if "${HEALTH_CMD[@]}" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "$READY" != 1 ]]; then
  echo "服务在 ${HEALTH_TIMEOUT}s 内没有就绪，请查看日志：$LOG_FILE" >&2
  exit 1
fi

# ---------------- 打印访问信息 ----------------
PUBLIC_HOST="${AutoDLService6006URL:-}"
PUBLIC_HOST="${PUBLIC_HOST#https://}"
PUBLIC_HOST="${PUBLIC_HOST#http://}"

echo
echo "✅ 服务已就绪"
echo "   本地:      ws://127.0.0.1:$PORT   (健康检查 http://127.0.0.1:$PORT/healthz)"
echo "   API key:   $API_KEY"
echo "   日志:      tail -f $LOG_FILE"
if [[ -n "$PUBLIC_HOST" ]]; then
  echo "   AutoDL 自定义服务(6006) 公网地址: wss://$PUBLIC_HOST"
  echo "     → 先把 6006 转发到本服务: ${PY_CMD[*]} deploy/autodl/forward_to_6006.py --target-port=$PORT"
fi
echo "   SSH 隧道（在你自己的电脑上执行）:"
echo "     ssh -CNg -L $PORT:127.0.0.1:$PORT root@<实例的region地址> -p <实例的SSH端口>"
echo "   客户端测试: ${PY_CMD[*]} deploy/autodl/test_client.py --host 127.0.0.1 --port $PORT --api-key $API_KEY"
