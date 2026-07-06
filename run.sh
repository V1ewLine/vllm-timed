#!/usr/bin/env bash
set -euo pipefail

# 变电部署的智能体模型推理优化
# vLLM timed_trace benchmark script

# =========================
# Required / common configs
# =========================
MODEL="${MODEL:-${1:-/loong/workspace/resources/models/qwen3-30A3-inst}}"
TRACE="${TRACE:-${2:-/loong/workspace/resources/data/Grafana/grafana-export/trace/vllm_timed_trace.jsonl}}"

export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-30A3}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

# 每个 hash_id 代表多少 input token。
# 你的样例 input_length=201, len(hash_ids)=13, ceil(201/16)=13，所以默认 16。
TIMED_TRACE_CHUNK_HASH_SIZE="${TIMED_TRACE_CHUNK_HASH_SIZE:-16}"

# timestamp 缩放系数：
# 1   = 按原始 trace 节奏 replay
# 0.5 = 请求间隔减半，约 2 倍流量
# 2   = 请求间隔翻倍，约 0.5 倍流量
TIMED_TRACE_SEC_MULTIPLIER="${TIMED_TRACE_SEC_MULTIPLIER:-1}"

# 推荐默认使用 OpenAI-compatible completions endpoint。
# 如果要测 chat:
#   BACKEND=openai-chat
#   ENDPOINT=/v1/chat/completions
BACKEND="${BACKEND:-openai}"
ENDPOINT="${ENDPOINT:-/v1/completions}"

RESULT_DIR="${RESULT_DIR:-./bench_results}"
RESULT_FILENAME="${RESULT_FILENAME:-timed_trace_result.json}"

SERVER_LOG="${SERVER_LOG:-./vllm_server.log}"
BENCH_LOG="${BENCH_LOG:-./vllm_bench.log}"

# 是否跑完后保留 server：
#   KEEP_SERVER=1 ./run_vllm_timed_trace_bench.sh
KEEP_SERVER="${KEEP_SERVER:-0}"

# Extra args:
#   VLLM_SERVER_EXTRA_ARGS="--some-server-arg ..."
#   BENCH_EXTRA_ARGS="--some-bench-arg ..."
VLLM_SERVER_EXTRA_ARGS="${VLLM_SERVER_EXTRA_ARGS:-}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"

# =========================
# Basic checks
# =========================
if [[ -z "${MODEL}" ]]; then
  echo "ERROR: MODEL is required."
  echo "Usage:"
  echo "  MODEL=/path/to/model TRACE=/path/to/vllm_timed_trace.jsonl bash $0"
  echo "or:"
  echo "  bash $0 /path/to/model /path/to/vllm_timed_trace.jsonl"
  exit 1
fi

if [[ ! -e "${MODEL}" ]]; then
  echo "WARN: MODEL path does not exist locally: ${MODEL}"
  echo "      If this is a HuggingFace model id, this warning can be ignored."
fi

if [[ ! -f "${TRACE}" ]]; then
  echo "ERROR: trace file not found: ${TRACE}"
  exit 1
fi

mkdir -p "${RESULT_DIR}"

NUM_PROMPTS="$(grep -cve '^[[:space:]]*$' "${TRACE}")"
if [[ "${NUM_PROMPTS}" -le 0 ]]; then
  echo "ERROR: trace file has no non-empty lines: ${TRACE}"
  exit 1
fi

echo "========================="
echo "Benchmark configuration"
echo "========================="
echo "MODEL=${MODEL}"
echo "TRACE=${TRACE}"
echo "NUM_PROMPTS=${NUM_PROMPTS}"
echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "BASE_URL=${BASE_URL}"
echo "BACKEND=${BACKEND}"
echo "ENDPOINT=${ENDPOINT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "TIMED_TRACE_CHUNK_HASH_SIZE=${TIMED_TRACE_CHUNK_HASH_SIZE}"
echo "TIMED_TRACE_SEC_MULTIPLIER=${TIMED_TRACE_SEC_MULTIPLIER}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "RESULT_FILENAME=${RESULT_FILENAME}"
echo "SERVER_LOG=${SERVER_LOG}"
echo "BENCH_LOG=${BENCH_LOG}"
echo "KEEP_SERVER=${KEEP_SERVER}"
echo

# =========================
# Validate timed_trace
# =========================
echo "Validating timed_trace file..."

TRACE_PATH="${TRACE}" \
TRACE_CHUNK_HASH_SIZE="${TIMED_TRACE_CHUNK_HASH_SIZE}" \
TRACE_SEC_MULTIPLIER="${TIMED_TRACE_SEC_MULTIPLIER}" \
MAX_MODEL_LEN_FOR_CHECK="${MAX_MODEL_LEN}" \
uv run python - <<'PY'
import json
import math
import os
import sys

path = os.environ["TRACE_PATH"]
chunk = int(os.environ["TRACE_CHUNK_HASH_SIZE"])
sec_multiplier = float(os.environ["TRACE_SEC_MULTIPLIER"])
max_model_len = int(os.environ["MAX_MODEL_LEN_FOR_CHECK"])

n = 0
bad = 0
prev_ts = None
first_ts = None
last_ts = None
max_input_len = 0
max_output_len = 0
max_ctx = 0
bad_examples = []

with open(path, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        try:
            r = json.loads(line)
        except Exception as e:
            bad += 1
            bad_examples.append(f"line {line_no}: bad json: {e}")
            continue

        required = ["hash_ids", "input_length", "output_length", "timestamp"]
        missing = [k for k in required if k not in r]
        if missing:
            bad += 1
            bad_examples.append(f"line {line_no}: missing fields: {missing}")
            continue

        try:
            hash_ids = r["hash_ids"]
            input_len = int(r["input_length"])
            output_len = int(r["output_length"])
            ts = float(r["timestamp"])
        except Exception as e:
            bad += 1
            bad_examples.append(f"line {line_no}: invalid field type: {e}")
            continue

        if not isinstance(hash_ids, list):
            bad += 1
            bad_examples.append(f"line {line_no}: hash_ids is not a list")
            continue

        if input_len <= 0 or output_len <= 0:
            bad += 1
            bad_examples.append(
                f"line {line_no}: input_length/output_length must be positive"
            )
            continue

        need_hashes = math.ceil(input_len / chunk)
        if len(hash_ids) < need_hashes:
            bad += 1
            bad_examples.append(
                f"line {line_no}: hash_ids too short; "
                f"input_length={input_len}, chunk={chunk}, "
                f"need>={need_hashes}, got={len(hash_ids)}"
            )
            continue

        if prev_ts is not None and ts < prev_ts:
            bad += 1
            bad_examples.append(
                f"line {line_no}: timestamp is not non-decreasing; "
                f"prev={prev_ts}, current={ts}"
            )
            continue

        if first_ts is None:
            first_ts = ts
        last_ts = ts
        prev_ts = ts

        max_input_len = max(max_input_len, input_len)
        max_output_len = max(max_output_len, output_len)
        max_ctx = max(max_ctx, input_len + output_len)

        n += 1

print(f"TRACE_STATS num_requests={n}")
print(f"TRACE_STATS first_timestamp={first_ts}")
print(f"TRACE_STATS last_timestamp={last_ts}")

if first_ts is not None and last_ts is not None:
    raw_duration = last_ts - first_ts
    scaled_duration = raw_duration * sec_multiplier
    avg_qps = n / scaled_duration if scaled_duration > 0 else float("inf")
    print(f"TRACE_STATS duration_raw_sec={raw_duration:.6f}")
    print(f"TRACE_STATS duration_after_multiplier_sec={scaled_duration:.6f}")
    print(f"TRACE_STATS approx_avg_arrival_qps={avg_qps:.6f}")

print(f"TRACE_STATS max_input_length={max_input_len}")
print(f"TRACE_STATS max_output_length={max_output_len}")
print(f"TRACE_STATS max_input_plus_output={max_ctx}")
print(f"TRACE_STATS bad_lines={bad}")

if max_ctx > max_model_len:
    print(
        f"ERROR: max input+output length {max_ctx} exceeds "
        f"MAX_MODEL_LEN {max_model_len}",
        file=sys.stderr,
    )
    sys.exit(2)

if bad:
    print("ERROR: trace validation failed. First bad examples:", file=sys.stderr)
    for x in bad_examples[:20]:
        print(f"  {x}", file=sys.stderr)
    sys.exit(2)

if n == 0:
    print("ERROR: no valid requests in trace", file=sys.stderr)
    sys.exit(2)

if n <= 10:
    print(
        "WARN: trace has very few requests; benchmark may finish almost immediately.",
        file=sys.stderr,
    )
PY

echo "Trace validation OK."
echo

# =========================
# Cleanup server on exit
# =========================
SERVER_PID=""

cleanup() {
  if [[ "${KEEP_SERVER}" == "1" ]]; then
    echo "KEEP_SERVER=1, leaving vLLM server running. pid=${SERVER_PID}"
    return 0
  fi

  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "Stopping vLLM server, pid=${SERVER_PID}"
    kill "${SERVER_PID}" >/dev/null 2>&1 || true

    for _ in $(seq 1 20); do
      if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        break
      fi
      sleep 0.5
    done

    if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "Force killing vLLM server, pid=${SERVER_PID}"
      kill -9 "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
  fi
}

trap cleanup EXIT INT TERM

# =========================
# Start vLLM server
# =========================
echo "Starting vLLM server..."
rm -f "${SERVER_LOG}"

# shellcheck disable=SC2086
uv run vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --seed 42 \
  --attention-backend FLASH_ATTN \
  --generation-config vllm \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  ${VLLM_SERVER_EXTRA_ARGS} \
  > "${SERVER_LOG}" 2>&1 &

SERVER_PID="$!"

echo "vLLM server pid=${SERVER_PID}"
echo "server log: ${SERVER_LOG}"
echo

# =========================
# Wait for server readiness
# =========================
echo "Waiting for server to become ready..."

READY=0
for i in $(seq 1 600); do
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "ERROR: vLLM server exited early. Last 200 server log lines:"
    tail -n 200 "${SERVER_LOG}" || true
    exit 1
  fi

  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    READY=1
    break
  fi

  if (( i % 20 == 0 )); then
    echo "Still waiting... ${i}s elapsed"
  fi

  sleep 1
done

if [[ "${READY}" != "1" ]]; then
  echo "ERROR: server did not become ready in time. Last 200 server log lines:"
  tail -n 200 "${SERVER_LOG}" || true
  exit 1
fi

echo "Server is ready at $(date '+%F %T')."
echo

# =========================
# Warmup
# =========================
echo "Running a small warmup request..."

if [[ "${BACKEND}" == "openai-chat" ]]; then
  curl -fsS "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${SERVED_MODEL_NAME}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"hello\"}],
      \"max_tokens\": 1,
      \"temperature\": 0
    }" >/dev/null
else
  curl -fsS "${BASE_URL}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${SERVED_MODEL_NAME}\",
      \"prompt\": \"hello\",
      \"max_tokens\": 1,
      \"temperature\": 0
    }" >/dev/null
fi

echo "Warmup OK at $(date '+%F %T')."
echo

# =========================
# Run benchmark
# =========================
echo "Running timed_trace benchmark at $(date '+%F %T')..."
echo "BENCH_LOG=${BENCH_LOG}"

set +e

# shellcheck disable=SC2086
uv run vllm bench serve \
  --backend "${BACKEND}" \
  --base-url "${BASE_URL}" \
  --endpoint "${ENDPOINT}" \
  --model "${SERVED_MODEL_NAME}" \
  --tokenizer "${MODEL}" \
  --dataset-name timed_trace \
  --dataset-path "${TRACE}" \
  --num-prompts "${NUM_PROMPTS}" \
  --no-oversample \
  --self-timed \
  --timed-trace-chunk-hash-size "${TIMED_TRACE_CHUNK_HASH_SIZE}" \
  --timed-trace-sec-multiplier "${TIMED_TRACE_SEC_MULTIPLIER}" \
  --save-result \
  --save-detailed \
  --plot-timeline \
  --plot-dataset-stats \
  --result-dir "${RESULT_DIR}" \
  --result-filename "${RESULT_FILENAME}" \
  --max-concurrency 32 \
  --ignore-eos \
  ${BENCH_EXTRA_ARGS} \
  > "${BENCH_LOG}" 2>&1

BENCH_EXIT_CODE=$?
set -e

echo
echo "Benchmark finished at $(date '+%F %T')."
echo "Benchmark exit code: ${BENCH_EXIT_CODE}"
echo

echo "Last 120 lines of benchmark log:"
echo "--------------------------------"
tail -n 120 "${BENCH_LOG}" || true
echo "--------------------------------"
echo

if [[ "${BENCH_EXIT_CODE}" != "0" ]]; then
  echo "ERROR: benchmark failed."
  echo "Full benchmark log: ${BENCH_LOG}"
  echo "Server log: ${SERVER_LOG}"
  exit "${BENCH_EXIT_CODE}"
fi

echo "Benchmark succeeded."
echo "Result JSON: ${RESULT_DIR}/${RESULT_FILENAME}"
echo "Benchmark log: ${BENCH_LOG}"
echo "Server log: ${SERVER_LOG}"

if [[ "${KEEP_SERVER}" == "1" ]]; then
  echo "vLLM server is still running. pid=${SERVER_PID}"
else
  echo "vLLM server will be stopped by cleanup."
fi
