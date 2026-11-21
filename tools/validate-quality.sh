#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-python3}"
if [[ -z "${PYTHON:-}" && -x ".venv/bin/python" ]]; then
  python_bin=".venv/bin/python"
fi

"$python_bin" -m json.tool CMakePresets.json >/dev/null
"$python_bin" -m json.tool vcpkg.json >/dev/null
"$python_bin" -m json.tool engine/data/nnue/task-027-smoke-manifest.json >/dev/null
"$python_bin" tools/validate_nnue_manifest.py engine/data/nnue/task-027-smoke-manifest.json
"$python_bin" tools/validate_nnue_manifest_test.py
"$python_bin" -m pip install -e trainer
PYTHONPATH=trainer/src "$python_bin" -m unittest discover -s trainer/tests
"$python_bin" tools/validate_trainer_smoke.py --device cpu --log build/trainer/validate-quality-smoke.jsonl
"$python_bin" tools/validate_nnue_loader.py \
  --dataset engine/data/nnue/task-027-smoke.binpack \
  --manifest engine/data/nnue/task-027-smoke-manifest.json \
  --batch-size 4 \
  --log build/trainer/validate-quality-loader.jsonl
"$python_bin" tools/validate_nnue_training.py \
  --dataset engine/data/nnue/task-027-smoke.binpack \
  --manifest engine/data/nnue/task-027-smoke-manifest.json \
  --architecture small \
  --device cpu \
  --epochs 16 \
  --batch-size 4 \
  --checkpoint-dir build/trainer/validate-quality-training \
  --log build/trainer/validate-quality-training.jsonl
"$python_bin" tools/validate_nnue_export.py \
  --checkpoint build/trainer/validate-quality-training/run-1/checkpoint-epoch-0016.pt \
  --dataset engine/data/nnue/task-027-smoke.binpack \
  --output build/trainer/validate-quality-network.nnue \
  --log build/trainer/validate-quality-nnue-export.jsonl
"$python_bin" tools/run_match_parser_test.py
"$python_bin" tools/run_validation_gate_test.py
cmake --list-presets all

cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure

cmake --preset lint
cmake --build --preset lint

cmake --preset asan
cmake --build --preset asan
ctest --preset asan --output-on-failure

cmake --preset ubsan
cmake --build --preset ubsan
ctest --preset ubsan --output-on-failure

cmake --preset tsan
cmake --build --preset tsan
ctest --preset tsan --output-on-failure

cmake --preset fuzz
cmake --build --preset fuzz --target fuzz-smoke

cmake --preset release
cmake --build --preset release
printf 'quit\n' | ./build/release/engine/localchessengine
