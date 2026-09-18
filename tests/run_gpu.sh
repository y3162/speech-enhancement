#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

TMPDIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

export SPEECH_DB_ROOT_DIR="$TMPDIR/db"
export SPEECH_CORPORA_ROOT_DIR="$TMPDIR/corpora"
mkdir -p "$SPEECH_DB_ROOT_DIR" "$SPEECH_CORPORA_ROOT_DIR"

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "CUDA is required for GPU tests" >&2
  exit 1
fi

"$PYTHON" -m unittest discover -s tests/se -t "$ROOT" -p 'gpu_test_*.py' -v
"$PYTHON" -m unittest discover -s tests/asr -t "$ROOT" -p 'test_*.py' -v
