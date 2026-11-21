#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

from localchessengine_trainer.binpack import BinpackDataset


def _checksum_hex(value: int) -> str:
    return f"0x{value:016x}"


def _parse_checksum(value: str) -> int:
    return int(value, 16 if value.startswith("0x") else 10)


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        record = {"event": event, "task": "TASK-029", **payload}
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("event") != "trainer.dataset.manifest":
        raise ValueError("manifest event must be trainer.dataset.manifest")
    return payload


def _validate_manifest(manifest: dict[str, Any], positions: int, checksum: int) -> None:
    expected_positions = int(manifest["positions"])
    expected_checksum = _parse_checksum(str(manifest["checksum"]))
    if expected_positions != positions:
        raise ValueError(
            f"manifest positions {expected_positions} != dataset positions {positions}"
        )
    if expected_checksum != checksum:
        raise ValueError(
            f"manifest checksum {_checksum_hex(expected_checksum)} != dataset checksum {_checksum_hex(checksum)}"
        )


def run(dataset_path: Path, manifest_path: Path, batch_size: int, log_path: Path) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")

    manifest = _load_manifest(manifest_path)
    dataset = BinpackDataset(dataset_path)
    metadata = dataset.metadata()
    validation = dataset.validate()
    _validate_manifest(manifest, validation.record_count, validation.checksum)
    if metadata.record_count != validation.record_count or metadata.checksum != validation.checksum:
        raise ValueError("metadata and validation results disagree")

    batches = 0
    positions = 0
    feature_entries = 0
    start = time.perf_counter()
    with JsonlLogger(log_path) as logger:
        logger.emit(
            "trainer.loader.metadata",
            {
                "dataset": str(dataset_path),
                "manifest": str(manifest_path),
                "dataset_id": manifest["dataset_id"],
                "record_count": metadata.record_count,
                "checksum": metadata.checksum_hex,
                "batch_size": batch_size,
            },
        )

        for batch in dataset.iter_batches(batch_size):
            white = torch.as_tensor(batch.white_indices)
            black = torch.as_tensor(batch.black_indices)
            offsets = torch.as_tensor(batch.offsets)
            eval_cp = torch.as_tensor(batch.eval_cp)
            wdl = torch.as_tensor(batch.wdl)
            if (
                white.dtype != torch.int64
                or black.dtype != torch.int64
                or offsets.dtype != torch.int64
            ):
                raise ValueError("feature tensors must use torch.int64")
            if eval_cp.dtype != torch.float32 or wdl.dtype != torch.float32:
                raise ValueError("target tensors must use torch.float32")
            if white.numel() != black.numel():
                raise ValueError("white/black feature tensor lengths differ")

            batches += 1
            positions += int(eval_cp.numel())
            feature_entries += int(white.numel())

        elapsed = max(time.perf_counter() - start, 1e-9)
        result = {
            "status": "pass",
            "dataset": str(dataset_path),
            "dataset_id": manifest["dataset_id"],
            "positions": positions,
            "batches": batches,
            "feature_entries": feature_entries,
            "checksum": _checksum_hex(validation.checksum),
            "throughput_positions_per_second": positions / elapsed,
        }
        logger.emit("trainer.loader.result", result)
        return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the LocalChessEngine NNUE binpack loader."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--log", type=Path, default=Path("build/trainer/task-029-loader-smoke.jsonl"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args.dataset, args.manifest, args.batch_size, args.log)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
