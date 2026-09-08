#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR=""
OUTPUT_DIR=""
SUBMIT_TEMPLATE=""
MODEL="qwen3.6-plus"
WORKERS="6"
PYTHON_BIN=""
CHECK_RUNTIME="0"

usage() {
  echo "Usage: $0 --input INPUT_DIR --output OUTPUT_DIR [--submit-template FILE] [--model qwen3.5/3.6/3.7...] [--workers N] [--python PYTHON]" >&2
  echo "       $0 --check-runtime [--python PYTHON]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT_DIR="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --submit-template) SUBMIT_TEMPLATE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --check-runtime) CHECK_RUNTIME="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$RUNTIME_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$RUNTIME_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

# Offline relocation smoke test. It hashes the declared runtime and exits
# before reading a key or touching the output directory.
if [[ "$CHECK_RUNTIME" == "1" ]]; then
  (
  cd "$RUNTIME_DIR"
  export PYTHONPATH="$RUNTIME_DIR"
  "$PYTHON_BIN" - <<'PY'
from agent.repro import build_runtime_manifest

manifest = build_runtime_manifest()
if not manifest.get("files"):
    raise SystemExit("runtime manifest is empty")
print(f"runtime check OK: {len(manifest['files'])} files")
PY
  )
  exit 0
fi

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
  usage
  exit 2
fi
if [[ ! "$MODEL" =~ ^qwen3\.(5|6|7)([-._][a-z0-9][a-z0-9._-]*)?$ ]]; then
  echo "Supported models are Qwen3.5, Qwen3.6, and Qwen3.7: $MODEL" >&2
  exit 2
fi
if [[ -z "${DASHSCOPE_API_KEY:-}" && ! -f "$RUNTIME_DIR/.env" ]]; then
  echo "DASHSCOPE_API_KEY is required" >&2
  exit 2
fi
if [[ ! -d "$RUNTIME_DIR/processed_data" ]]; then
  echo "Missing processed_data under $RUNTIME_DIR" >&2
  exit 2
fi

if compgen -G "$INPUT_DIR/*.json" > /dev/null || \
   compgen -G "$INPUT_DIR/*.jsonl" > /dev/null; then
  QUESTION_DIR="$INPUT_DIR"
elif [[ -d "$INPUT_DIR/questions" ]]; then
  QUESTION_DIR="$INPUT_DIR/questions"
else
  echo "No question JSON/JSONL found; pass the question directory with --input" >&2
  exit 2
fi
if [[ ! -d "$QUESTION_DIR" ]]; then
  echo "Question directory not found: $QUESTION_DIR" >&2
  exit 2
fi

if [[ -z "$SUBMIT_TEMPLATE" ]]; then
  if [[ -f "$INPUT_DIR/submit.csv" ]]; then
    SUBMIT_TEMPLATE="$INPUT_DIR/submit.csv"
  elif [[ -f "$(dirname "$QUESTION_DIR")/submit.csv" ]]; then
    SUBMIT_TEMPLATE="$(dirname "$QUESTION_DIR")/submit.csv"
  else
    echo "submit.csv not found; pass --submit-template" >&2
    exit 2
  fi
fi

set -a
# shellcheck source=/dev/null
# Remove inherited tuning flags so this entrypoint has one declared profile.
while IFS= read -r NAME; do
  unset "$NAME"
done < <(compgen -A variable AFAC_ || true)
. "$RUNTIME_DIR/config/runtime.env"
set +a
export PYTHONPATH="$RUNTIME_DIR"

"$PYTHON_BIN" "$RUNTIME_DIR/agent/run.py" \
  --output-dir "$OUTPUT_DIR" \
  --qdir "$QUESTION_DIR" \
  --submit-template "$SUBMIT_TEMPLATE" \
  --model "$MODEL" \
  --verify-model "$MODEL" \
  --workers "$WORKERS"

"$PYTHON_BIN" "$RUNTIME_DIR/script/build_evidence.py" "$OUTPUT_DIR" \
  --qdir "$QUESTION_DIR"
"$PYTHON_BIN" "$RUNTIME_DIR/script/check_reproduction.py" "$OUTPUT_DIR" \
  --qdir "$QUESTION_DIR" \
  --submit-template "$SUBMIT_TEMPLATE"
