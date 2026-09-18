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
"$PYTHON" -m unittest discover -s tests/se -t "$ROOT" -p 'test_*.py' -v
"$PYTHON" -m unittest discover -s tests/data -t "$ROOT" -p 'test_*.py' -v
