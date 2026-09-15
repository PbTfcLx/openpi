#!/usr/bin/env bash
# 一键验证 openpi 服务：进程 → 本地健康检查 → 6006 转发 → 公网地址 → 鉴权 → 端到端推理。
#
# 用法:
#   bash deploy/autodl/verify.sh                    # 全量验证（含一次真实推理）
#   bash deploy/autodl/verify.sh --skip-inference   # 只做连通性检查，不加载模型推理
#   bash deploy/autodl/verify.sh --port 8000 --api-key <key>
set -uo pipefail

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$OPENPI_ROOT"

PORT="${OPENPI_PORT:-8000}"
API_KEY=""
SKIP_INFERENCE=0
PRESET="${OPENPI_PRESET:-robocasa}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --skip-inference) SKIP_INFERENCE=1; shift ;;
    -h|--help) echo "用法: bash deploy/autodl/verify.sh [--port 8000] [--api-key <key>] [--preset robocasa] [--skip-inference]"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$API_KEY" && -f "$OPENPI_ROOT/logs/api_key" ]]; then
  API_KEY="$(<"$OPENPI_ROOT/logs/api_key")"
fi

if [[ -x "$OPENPI_ROOT/.venv/bin/python" ]]; then
  PY="$OPENPI_ROOT/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

PASS=0
FAIL=0
ok()   { echo "  ✅ $1"; PASS=$((PASS + 1)); }
bad()  { echo "  ❌ $1"; FAIL=$((FAIL + 1)); }
info() { echo "  ℹ️  $1"; }

check_url() {  # $1=描述 $2=url
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$2" 2>/dev/null)"
  if [[ "$code" == "200" ]]; then ok "$1 ($2) → HTTP 200"; else bad "$1 ($2) → HTTP ${code:-超时}"; fi
}

echo "== 1. 服务进程 =="
if pgrep -f "serve_policy.py --port=$PORT" >/dev/null; then
  ok "policy server 正在运行 (PID: $(pgrep -f "serve_policy.py --port=$PORT" | tr '\n' ' '))"
else
  bad "没有找到 serve_policy.py --port=$PORT 进程；请先执行 bash deploy/autodl/start_server.sh --port $PORT"
fi

echo "== 2. 本地健康检查（服务本身） =="
check_url "本地 /healthz" "http://127.0.0.1:$PORT/healthz"

echo "== 3. AutoDL 自定义服务 6006 转发 =="
if pgrep -f "forward_to_6006.py" >/dev/null; then
  check_url "6006 转发 /healthz" "http://127.0.0.1:6006/healthz"
else
  info "未运行 forward_to_6006.py（如需公网访问请执行：nohup .venv/bin/python deploy/autodl/forward_to_6006.py --target-port=$PORT > logs/forward_6006.log 2>&1 &）"
fi

echo "== 4. 公网地址（AutoDL 反向代理） =="
PUBLIC_HOST="${AutoDLService6006URL:-}"
PUBLIC_HOST="${PUBLIC_HOST#https://}"
PUBLIC_HOST="${PUBLIC_HOST#http://}"
if [[ -n "$PUBLIC_HOST" ]]; then
  check_url "公网 /healthz" "https://$PUBLIC_HOST/healthz"
else
  info "环境变量 AutoDLService6006URL 为空，跳过"
fi

echo "== 5. 鉴权 =="
if [[ -z "$API_KEY" ]]; then
  info "未配置/未找到 API key，跳过鉴权检查"
else
  out="$("$PY" deploy/autodl/test_client.py --host 127.0.0.1 --port "$PORT" --api-key "definitely-wrong" --num-steps 1 2>&1)"
  if grep -qiE "401|rejected|Unauthorized" <<<"$out"; then
    ok "错误 API key 被拒绝（401）"
  else
    bad "错误 API key 竟然没有被拒绝：$(tail -n 1 <<<"$out")"
  fi
fi

echo "== 6. 端到端推理 =="
if [[ "$SKIP_INFERENCE" == 1 ]]; then
  info "已跳过（--skip-inference）"
elif [[ -z "$API_KEY" ]]; then
  info "无 API key，跳过；可手动指定 --api-key"
else
  start=$(date +%s)
  out="$("$PY" deploy/autodl/test_client.py --host 127.0.0.1 --port "$PORT" --api-key "$API_KEY" \
        --preset "$PRESET" --num-steps 1 2>&1)"
  if grep -q "✅" <<<"$out"; then
    ok "实例内 ws://127.0.0.1:$PORT 推理成功（$(grep -o 'actions.shape=[^ ]*' <<<"$out" | head -1)，耗时 $(( $(date +%s) - start ))s，首次含 JIT 编译）"
  else
    bad "推理失败：$(tail -n 3 <<<"$out")"
  fi

  if [[ -n "$PUBLIC_HOST" ]]; then
    out="$("$PY" deploy/autodl/test_client.py --uri "wss://$PUBLIC_HOST" --api-key "$API_KEY" \
          --preset "$PRESET" --num-steps 1 2>&1)"
    if grep -q "✅" <<<"$out"; then
      ok "公网 wss://$PUBLIC_HOST 推理成功（$(grep -o '客户端往返=[0-9.]*ms' <<<"$out" | head -1)）"
    else
      bad "公网推理失败：$(tail -n 3 <<<"$out")"
    fi
  fi
fi

echo
echo "验证结果：$PASS 项通过，$FAIL 项失败"
[[ "$FAIL" == 0 ]]
