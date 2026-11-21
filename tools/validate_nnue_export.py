#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from localchessengine_trainer.serialize import ExportConfig, export_network, read_network_info


def run(args: argparse.Namespace) -> dict[str, Any]:
    result = export_network(
        ExportConfig(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            dataset_path=args.dataset,
            batch_size=args.batch_size,
            log_path=args.log,
        )
    )
    info = read_network_info(args.output)
    if result["network_id"] != info["network_id"]:
        raise ValueError("export result network_id does not match parsed network")
    if result["checksum64"] != info["checksum64"]:
        raise ValueError("export result checksum does not match parsed network")

    quant_error = result["quant_error_cp"]
    if float(quant_error["mean_abs_cp"]) > args.max_mean_error_cp:
        raise ValueError(
            f"mean quantization error {quant_error['mean_abs_cp']} exceeds "
            f"{args.max_mean_error_cp}"
        )
    if float(quant_error["max_abs_cp"]) > args.max_error_cp:
        raise ValueError(
            f"max quantization error {quant_error['max_abs_cp']} exceeds {args.max_error_cp}"
        )

    if args.log is not None:
        records = [
            json.loads(line)
            for line in args.log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not any(record.get("event") == "nnue.validation.result" for record in records):
            raise ValueError("export log is missing nnue.validation.result")

    return {
        "status": "pass",
        "network": str(args.output),
        "network_id": result["network_id"],
        "format_version": result["format_version"],
        "architecture": result["architecture"],
        "feature_bias": result["feature_bias"],
        "side_to_move_ordering": result["side_to_move_ordering"],
        "checksum64": result["checksum64"],
        "quant_error_cp": quant_error,
        "metadata_len": info["metadata_len"],
        "tensor_bytes": info["tensor_bytes"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate TASK-031 NNUE checkpoint quantization/export."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-mean-error-cp", type=float, default=1.0)
    parser.add_argument("--max-error-cp", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"FAIL NNUE export validation: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
