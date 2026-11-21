#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import torch

from localchessengine_trainer.train import (
    CHECKPOINT_SCHEMA,
    EPOCH_METRICS_EVENT,
    TRAINING_RESULT_EVENT,
    TrainingConfig,
    run_training,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _validate_log(path: Path) -> None:
    records = _load_jsonl(path)
    events = {record.get("event") for record in records}
    required = {
        "trainer.dependency.check",
        "trainer.dataset.metadata",
        EPOCH_METRICS_EVENT,
        TRAINING_RESULT_EVENT,
    }
    missing = required - events
    if missing:
        raise ValueError(f"training log is missing events: {sorted(missing)}")
    result = next(record for record in records if record.get("event") == TRAINING_RESULT_EVENT)
    if result.get("status") != "pass":
        raise ValueError("training result did not pass")
    epoch_metrics = [record for record in records if record.get("event") == EPOCH_METRICS_EVENT]
    if not epoch_metrics:
        raise ValueError("training log has no epoch metrics")
    for metric in epoch_metrics:
        for field in ("dataset_id", "loss", "eval_loss", "wdl_loss", "learning_rate", "device"):
            if field not in metric:
                raise ValueError(f"epoch metric is missing {field}")


def _validate_checkpoint(path: Path, architecture: str, dataset_id: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    if checkpoint.get("architecture", {}).get("name") != architecture:
        raise ValueError("checkpoint architecture mismatch")
    if checkpoint.get("dataset", {}).get("dataset_id") != dataset_id:
        raise ValueError("checkpoint dataset_id mismatch")
    if not checkpoint.get("model_state_sha256"):
        raise ValueError("checkpoint has no model_state_sha256")
    if "model_state" not in checkpoint:
        raise ValueError("checkpoint has no model_state")
    if "optimizer_state" not in checkpoint or "scheduler_state" not in checkpoint:
        raise ValueError("checkpoint has no optimizer/scheduler state")
    return checkpoint


def _run_once(args: argparse.Namespace, suffix: str) -> dict[str, Any]:
    checkpoint_dir = args.checkpoint_dir / suffix
    log_path = args.log.with_name(f"{args.log.stem}-{suffix}{args.log.suffix}")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    config = TrainingConfig(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        architecture=args.architecture,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eta_min=args.eta_min,
        lambda_eval=args.lambda_eval,
        clip_norm=args.clip_norm,
        seed=args.seed,
        checkpoint_dir=checkpoint_dir,
        log_path=log_path,
    )
    result = run_training(config)
    _validate_log(log_path)
    checkpoint = _validate_checkpoint(
        Path(str(result["checkpoint"])),
        args.architecture,
        str(result["dataset_id"]),
    )
    if checkpoint["model_state_sha256"] != result["model_state_sha256"]:
        raise ValueError("result/checkpoint model digest mismatch")
    return {**result, "log": str(log_path)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.determinism_runs <= 0:
        raise ValueError("determinism-runs must be positive")
    results = [_run_once(args, f"run-{index + 1}") for index in range(args.determinism_runs)]
    first = results[0]
    if args.require_loss_drop and float(first["final_loss"]) >= float(first["initial_loss"]):
        raise ValueError(
            f"final loss {first['final_loss']} did not improve initial loss {first['initial_loss']}"
        )
    if args.determinism_runs > 1:
        for result in results[1:]:
            if result["model_state_sha256"] != first["model_state_sha256"]:
                raise ValueError("model_state_sha256 changed across deterministic CPU runs")
            if result["checkpoint_sha256"] != first["checkpoint_sha256"]:
                raise ValueError("checkpoint_sha256 changed across deterministic CPU runs")
            if result["final_loss"] != first["final_loss"]:
                raise ValueError("final loss changed across deterministic CPU runs")

    return {
        "status": "pass",
        "runs": args.determinism_runs,
        "architecture": args.architecture,
        "device": first["device"],
        "dataset_id": first["dataset_id"],
        "initial_loss": first["initial_loss"],
        "final_loss": first["final_loss"],
        "model_state_sha256": first["model_state_sha256"],
        "checkpoint_sha256": first["checkpoint_sha256"],
        "checkpoint": first["checkpoint"],
        "logs": [result["log"] for result in results],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the TASK-030 LocalChessEngine NNUE trainer."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--architecture", choices=["small", "v1"], default="small")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="cpu")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8.75e-4)
    parser.add_argument("--eta-min", type=float, default=5.0e-5)
    parser.add_argument("--lambda-eval", type=float, default=0.7)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=30030)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("build/trainer/task-030-validation-checkpoints"),
    )
    parser.add_argument("--log", type=Path, default=Path("build/trainer/task-030-validation.jsonl"))
    parser.add_argument("--determinism-runs", type=int, default=2)
    parser.add_argument(
        "--no-require-loss-drop",
        action="store_false",
        dest="require_loss_drop",
        help="Allow smoke runs that only validate model execution, not loss improvement.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"FAIL NNUE training validation: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
