#!/usr/bin/env bash
# 停止由 deploy/autodl/start_server.sh 启动的 openpi policy server。
#
# 用法: bash deploy/autodl/stop_server.sh [--port 8000]
set -euo pipefail

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${OPENPI_PORT:-8000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    -h|--help) echo "用法: bash deploy/autodl/stop_server.sh [--port 8000]"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

PID_FILE="$OPENPI_ROOT/logs/serve_${PORT}.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "没有找到 PID 文件 $PID_FILE（服务可能未通过 start_server.sh 启动）"
  exit 0
fi

PID="$(<"$PID_FILE")"
if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
  echo "停止 PID=$PID ..."
  kill "$PID" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  kill -9 "$PID" 2>/dev/null || true
  echo "已停止。"
else
  echo "PID=$PID 已不在运行。"
fi
rm -f "$PID_FILE"
