# LocalChessEngine Trainer

This directory contains the Python/PyTorch trainer for the NNUE pipeline.

## Setup

From the repository root:

```sh
source .venv/bin/activate
python3 -m pip install -e trainer
```

FEN-pool export from Hugging Face is optional because normal local validation must not download
data:

```sh
python3 -m pip install -e 'trainer[fen-pool]'
```

`trainer/pyproject.toml` declares the runtime dependency on PyTorch and the build dependency on
pybind11 for the native binpack loader. PyTorch is used for training only; the C++ engine core
remains independent of PyTorch, Qt, and GUI code.

## Binpack Loader

Validate the current smoke dataset loader:

```sh
python3 tools/validate_nnue_loader.py \
  --dataset engine/data/nnue/task-027-smoke.binpack \
  --manifest engine/data/nnue/task-027-smoke-manifest.json \
  --batch-size 4 \
  --log build/trainer/task-029-loader-smoke.jsonl
```

The loader lives at `localchessengine_trainer.binpack`, wraps the native
`localchessengine_trainer._binpack_native` extension, and returns sparse NumPy arrays suitable for
PyTorch `EmbeddingBag` input. It supports only the LocalChessEngine binpack v1 format from
`TASK-027`; Stockfish/Bullet-compatible external formats are out of scope.

## Smoke Train

Run the TASK-028 smoke train:

```sh
python3 -m localchessengine_trainer.smoke_train \
  --device auto \
  --epochs 2 \
  --batch-size 128 \
  --hidden-size 32 \
  --features-per-position 16 \
  --log build/trainer/task-028-smoke.jsonl
```

The smoke uses synthetic sparse features to validate the environment, MPS/CPU device selection,
`EmbeddingBag` execution, Adam, gradient clipping, and JSONL metrics logging. It does not train a
real chess network and is not strength evidence.

Use `--device mps` to require MPS, `--device cpu` for deterministic CPU validation, or `--device
auto` to prefer MPS and fall back to CPU when MPS is unavailable or fails during the smoke.

## NNUE Training

Run the TASK-030 trainer against the accepted local smoke dataset:

```sh
python3 -m localchessengine_trainer.train \
  --dataset engine/data/nnue/task-027-smoke.binpack \
  --manifest engine/data/nnue/task-027-smoke-manifest.json \
  --architecture small \
  --device cpu \
  --epochs 16 \
  --batch-size 4 \
  --checkpoint-dir build/trainer/task-030-small-checkpoints \
  --log build/trainer/task-030-small.jsonl
```

The `small` architecture is `(768->512)x2->1` with CReLU. The `v1` architecture is
`(768->1024)x2->1` with SCReLU. Both use two perspective `EmbeddingBag` transformers, Adam,
cosine learning-rate annealing, gradient clipping, and the interpolated eval/WDL loss from the
project guide. Targets are white-relative to match the accepted binpack manifest.

For the reproducible validation smoke, use:

```sh
python3 tools/validate_nnue_training.py \
  --dataset engine/data/nnue/task-027-smoke.binpack \
  --manifest engine/data/nnue/task-027-smoke-manifest.json \
  --architecture small \
  --device cpu \
  --epochs 16 \
  --batch-size 4 \
  --checkpoint-dir build/trainer/task-030-validation-small \
  --log build/trainer/task-030-validation-small.jsonl
```

This validator runs two identical CPU trainings and compares the final model/checkpoint digests.
MPS execution can be smoke-tested with `--device auto`, but deterministic checkpoint comparison is
only required for CPU.

## NNUE Export

Export a TASK-030 checkpoint to the TASK-031 quantized `.nnue` format:

```sh
python3 -m localchessengine_trainer.serialize \
  --checkpoint build/trainer/task-030-validation-small/run-1/checkpoint-epoch-0016.pt \
  --dataset engine/data/nnue/task-027-smoke.binpack \
  --output build/trainer/task-031-smoke-network.nnue \
  --log build/trainer/task-031-export.jsonl
```

The exporter validates the checkpoint schema, model-state digest and tensor shapes, quantizes with
`QA=255` and `QB=64`, writes a versioned header/checksum/metadata block, and logs
`nnue.validation.result`. The exported smoke network validates serialization only; C++ runtime
loading/inference is covered separately and the smoke network is not strength evidence.

## Small NNUE Gate

`TASK-034` validates the complete small-network path before scaling to `v1`:

```sh
python3 tools/validate_small_nnue.py \
  --out-dir build/nnue/task-034-small/current
```

The staged default uses only local engine/HCE labels, mixed self-play playouts, per-game sampling,
separate train/holdout generation, and local EPD openings generated from the holdout preview. It
trains `small` with CReLU, feature-transform biases and side-to-move-relative `us/them` output
ordering, exports `.nnue` v2, checks Python-vs-C++ quantized evaluation checksums over the
generated binpack, reports fit and holdout metrics, and runs HCE vs NNUE through Cute Chess with
`Use NNUE=true`, `EvalFile=<network>`, `NNUE Backend=Scalar`, `-openings policy=default`, and
`-repeat 2`. The validator parses the PGN and requires enough unique FENs before accepting match
evidence. `+50 Elo` evidence is required before continuing to the large network; otherwise the gate
records an inconclusive or rejected decision for data/training follow-up.

## Local Relabel From Lichess FEN Pool

For `TASK-034` recovery, use Lichess only as a FEN/diversity pool while regenerating labels with
LocalChessEngine. Direct Stockfish/Lichess labelled training is not supported in the promoted
release path. The relabel
manifests keep `external_data=false`, record `external_position_source=true`, and preserve the CC0
Hugging Face provenance.

```sh
localchessengine-export-lichess-fens \
  --output build/nnue/local-relabel/stage1/fen-pool.txt \
  --manifest build/nnue/local-relabel/stage1/fen-pool-manifest.json \
  --dataset-id lce-task-034-fen-pool-stage1 \
  --date 2026-05-12T00:00:00Z \
  --positions 800000 \
  --shuffle-buffer 500000 \
  --shuffle-seed 34034

./build/release/localchessengine_nnue_data relabel-fens \
  --input build/nnue/local-relabel/stage1/fen-pool.txt \
  --train-output build/nnue/local-relabel/stage1/train.binpack \
  --holdout-output build/nnue/local-relabel/stage1/holdout.binpack \
  --match-openings-output build/nnue/local-relabel/stage1/match-openings.epd \
  --manifest build/nnue/local-relabel/stage1/manifest.json \
  --train-manifest build/nnue/local-relabel/stage1/train-manifest.json \
  --holdout-manifest build/nnue/local-relabel/stage1/holdout-manifest.json \
  --train-preview build/nnue/local-relabel/stage1/train-preview.csv \
  --holdout-preview build/nnue/local-relabel/stage1/holdout-preview.csv \
  --match-preview build/nnue/local-relabel/stage1/match-preview.csv \
  --dataset-id lce-task-034-local-relabel-stage1 \
  --date 2026-05-12T00:00:00Z \
  --train-positions 500000 \
  --holdout-positions 100000 \
  --match-openings 512 \
  --depth 3
```

Use `--stability-depth-delta 1 --max-stability-delta-cp 150` on `relabel-fens` for the stricter
Stage-B corpus. The split manifests are accepted by `localchessengine-train`; manifests marked
`external_data=true` are rejected explicitly so external labels cannot silently replace local
engine/HCE labels.
