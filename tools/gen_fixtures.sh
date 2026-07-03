#!/usr/bin/env bash
# Run LTX-2.3 fixture generators against the diffusers reference checkout.
# Usage: tools/gen_fixtures.sh [script.py ...]   (default: all gen_fixtures_*.py)
#
# Uses uv with a CPU-only torch index; never touches the system Python.
set -euo pipefail
cd "$(dirname "$0")/.."

scripts=("$@")
if [ ${#scripts[@]} -eq 0 ]; then
  scripts=(tools/gen_fixtures_*.py)
fi

for s in "${scripts[@]}"; do
  echo "== $s"
  uv run --no-project \
    --index https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    --with torch \
    --with numpy \
    --with safetensors \
    --with huggingface_hub \
    --with packaging \
    --with filelock \
    --with regex \
    --with requests \
    --with tqdm \
    --with pillow \
    python "$s"
done
