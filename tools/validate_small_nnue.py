#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER_SRC = REPO_ROOT / "trainer" / "src"
if str(TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINER_SRC))

from localchessengine_trainer.binpack import BinpackDataset  # noqa: E402
from localchessengine_trainer.model import PerspectiveNnueModel  # noqa: E402
from localchessengine_trainer.serialize import (  # noqa: E402
    _load_checkpoint,
    _quantized_batch_prediction,
    quantize_state,
)


DEFAULT_OUT_DIR = REPO_ROOT / "build" / "nnue" / "task-034-small"
DEFAULT_GENERATOR = REPO_ROOT / "build" / "release" / "localchessengine_nnue_data"
DEFAULT_NNUE_BENCH = REPO_ROOT / "build" / "release" / "localchessengine_nnue_bench"
DEFAULT_ENGINE = REPO_ROOT / "build" / "release" / "engine" / "localchessengine"
CHECKSUM_SEED = 14695981039346656037
CHECKSUM_MIX = 0x9E3779B97F4A7C15
CHECKSUM_MASK = (1 << 64) - 1
CHECKSUM_SCORE_OFFSET = 32768
EXPECTED_RELEASE_BENCH_NODES = 18417


@dataclass(frozen=True)
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    output: str
    elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return self.returncode == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "command_text": shlex.join(self.command),
            "returncode": self.returncode,
            "pass": self.passed,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "output_tail": self.output[-12000:],
        }


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def run_command(name: str, command: list[str]) -> CommandResult:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(TRAINER_SRC)
        if not existing_pythonpath
        else f"{TRAINER_SRC}{os.pathsep}{existing_pythonpath}"
    )
    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.perf_counter() - started
    return CommandResult(
        name=name,
        command=command,
        returncode=completed.returncode,
        output=completed.stdout,
        elapsed_seconds=elapsed,
    )


def json_from_stdout(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        text = line.strip()
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
    raise ValueError("command output did not contain a JSON object")


def mix_checksum(checksum: int, score: int) -> int:
    value = (score + CHECKSUM_SCORE_OFFSET) & 0xFFFFFFFF
    mixed = (value + CHECKSUM_MIX + ((checksum << 6) & CHECKSUM_MASK) + (checksum >> 2)) & CHECKSUM_MASK
    return (checksum ^ mixed) & CHECKSUM_MASK


def round_cp(value: float) -> int:
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


def python_quantized_checksum(checkpoint: Path, dataset: Path, batch_size: int) -> dict[str, Any]:
    _, state, architecture = _load_checkpoint(checkpoint)
    tensors = quantize_state(state)
    model = PerspectiveNnueModel(architecture)
    model.load_state_dict(state)

    checksum = CHECKSUM_SEED
    positions = 0
    for batch in BinpackDataset(dataset).iter_batches(batch_size):
        predictions = _quantized_batch_prediction(
            batch,
            tensors,
            architecture.activation,
            architecture.side_to_move_ordering,
        )
        for prediction in predictions:
            checksum = mix_checksum(checksum, round_cp(float(prediction)))
        positions += batch.positions
    return {"positions": positions, "checksum": checksum}


def parse_cpp_dataset_checksum(output: str) -> dict[str, Any]:
    records_match = re.search(r"^dataset records:\s+(?P<records>\d+)\s*$", output, re.MULTILINE)
    checksum_match = re.search(
        r"backend:\s+scalar\s+eval cp:\s+[-+]?\d+\s+checksum:\s+(?P<checksum>\d+)",
        output,
        re.MULTILINE,
    )
    if records_match is None or checksum_match is None:
        raise ValueError("C++ NNUE dataset bench output is missing scalar records/checksum")
    return {
        "positions": int(records_match.group("records")),
        "checksum": int(checksum_match.group("checksum")),
    }


def parse_engine_bench_nodes(output: str) -> int:
    match = re.search(r"(?P<nodes>\d+)\s+nodes\s+\d+\s+nps", output)
    if match is None:
        raise ValueError("engine bench output did not contain final nodes/nps line")
    return int(match.group("nodes"))


def write_openings_from_preview(
    preview: Path,
    output: Path,
    *,
    max_openings: int,
    max_abs_eval: int,
) -> dict[str, Any]:
    all_rows: list[dict[str, str]] = []
    with preview.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            all_rows.append(row)
    rows: list[dict[str, str]] = []
    for row in all_rows:
        try:
            if abs(int(row["eval_cp"])) <= max_abs_eval:
                rows.append(row)
        except (KeyError, ValueError):
            continue
    if len(rows) < max_openings:
        rows.extend(all_rows)

    rows.sort(key=lambda row: (abs(int(row.get("eval_cp", "0"))), int(row.get("ply", "0"))))
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    side_counts = {"w": 0, "b": 0}
    for row in rows:
        fen = row.get("fen", "")
        parts = fen.split()
        if len(parts) < 4 or fen in seen:
            continue
        side = parts[1]
        if side not in side_counts:
            continue
        other = "b" if side == "w" else "w"
        if side_counts[side] > side_counts[other] + 1:
            continue
        seen.add(fen)
        selected.append(row)
        side_counts[side] += 1
        if len(selected) >= max_openings:
            break
    if len(selected) < max_openings:
        for row in rows:
            fen = row.get("fen", "")
            parts = fen.split()
            if len(parts) < 4 or fen in seen:
                continue
            side = parts[1]
            if side not in side_counts:
                continue
            seen.add(fen)
            selected.append(row)
            side_counts[side] += 1
            if len(selected) >= max_openings:
                break

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(selected, start=1):
            fen = row["fen"]
            epd = " ".join(fen.split()[:4])
            bestmove = row.get("bestmove", "0000")
            handle.write(f'{epd} bm {bestmove}; id "task034-{index:04d}";\n')

    return {
        "path": str(output),
        "source_preview": str(preview),
        "count": len(selected),
        "max_abs_eval_cp": max_abs_eval,
        "side_to_move": side_counts,
    }


def parse_match_opening_diversity(
    pgn: Path,
    *,
    generated_openings: int,
    completed_games: int,
    repeat: int,
) -> dict[str, Any]:
    fen_pattern = re.compile(r'^\[FEN\s+"(?P<fen>[^"]+)"\]$')
    counts: Counter[str] = Counter()
    if pgn.exists():
        with pgn.open(encoding="utf-8") as handle:
            for line in handle:
                match = fen_pattern.match(line.strip())
                if match is not None:
                    counts[match.group("fen")] += 1

    expected_pairs = completed_games // max(repeat, 1)
    required_unique = min(generated_openings, max(1, expected_pairs)) if completed_games > 0 else 0
    unique_fens = len(counts)
    repeated_startpos_only = completed_games > 0 and unique_fens == 0
    passed = unique_fens >= required_unique and not repeated_startpos_only
    return {
        "pgn": str(pgn),
        "generated_openings": generated_openings,
        "completed_games": completed_games,
        "repeat": repeat,
        "expected_opening_pairs": expected_pairs,
        "required_unique_fens": required_unique,
        "unique_fens": unique_fens,
        "pass": passed,
        "fen_counts": [
            {"fen": fen, "games": games}
            for fen, games in counts.most_common(min(len(counts), 32))
        ],
    }


def prediction_metrics(checkpoint: Path, dataset: Path, batch_size: int) -> dict[str, Any]:
    _, state, architecture = _load_checkpoint(checkpoint)
    tensors = quantize_state(state)
    predictions: list[float] = []
    targets: list[float] = []
    for batch in BinpackDataset(dataset).iter_batches(batch_size):
        observed = _quantized_batch_prediction(
            batch,
            tensors,
            architecture.activation,
            architecture.side_to_move_ordering,
        )
        predictions.extend(float(value) for value in observed)
        if architecture.side_to_move_ordering:
            targets.extend(
                float(-eval_cp if side == 1 else eval_cp)
                for eval_cp, side in zip(batch.eval_cp, batch.side_to_move, strict=True)
            )
        else:
            targets.extend(float(value) for value in batch.eval_cp)

    if not predictions:
        raise ValueError("prediction metrics require at least one dataset position")

    count = len(predictions)
    errors = [prediction - target for prediction, target in zip(predictions, targets, strict=True)]
    abs_errors = [abs(error) for error in errors]
    pred_mean = sum(predictions) / count
    target_mean = sum(targets) / count
    pred_var = sum((value - pred_mean) ** 2 for value in predictions) / count
    target_var = sum((value - target_mean) ** 2 for value in targets) / count
    covariance = sum(
        (prediction - pred_mean) * (target - target_mean)
        for prediction, target in zip(predictions, targets, strict=True)
    ) / count
    correlation = (
        covariance / math.sqrt(pred_var * target_var)
        if pred_var > 0.0 and target_var > 0.0
        else 0.0
    )
    saturated = sum(1 for value in predictions if abs(value) >= 2500.0)
    return {
        "positions": count,
        "mae_cp": sum(abs_errors) / count,
        "rms_cp": math.sqrt(sum(error * error for error in errors) / count),
        "max_abs_cp": max(abs_errors),
        "prediction_mean_cp": pred_mean,
        "prediction_stdev_cp": math.sqrt(pred_var),
        "target_mean_cp": target_mean,
        "target_stdev_cp": math.sqrt(target_var),
        "correlation": correlation,
        "saturated_predictions": saturated,
    }


def classify_match(match_result: dict[str, Any] | None, minimum_games: int) -> dict[str, Any]:
    if match_result is None:
        return {
            "classification": "not_run",
            "promotes_strength": False,
            "games_completed": 0,
            "minimum_games": minimum_games,
        }

    match = match_result.get("match", {})
    sprt = match_result.get("sprt", {})
    output = sprt.get("output", {}) if isinstance(sprt, dict) else {}
    llr = output.get("llr") if isinstance(output, dict) else None
    lower = output.get("lower_bound") if isinstance(output, dict) else None
    upper = output.get("upper_bound") if isinstance(output, dict) else None
    games_completed = int(match.get("games_completed") or 0)

    classification = "missing_sprt_output"
    if isinstance(llr, (int, float)) and isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
        if llr <= lower:
            classification = "accept_h0"
        elif llr >= upper:
            classification = "accept_h1"
        else:
            classification = "inconclusive"

    promotes_strength = games_completed >= minimum_games and classification == "accept_h1"
    return {
        "classification": classification,
        "promotes_strength": promotes_strength,
        "games_completed": games_completed,
        "minimum_games": minimum_games,
        "llr": llr,
        "lower_bound": lower,
        "upper_bound": upper,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def command_or_fail(commands: list[CommandResult], name: str, command: list[str]) -> CommandResult:
    result = run_command(name, command)
    commands.append(result)
    if not result.passed:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = out_dir / "task-034-small.binpack"
    manifest = out_dir / "task-034-small-manifest.json"
    preview = out_dir / "task-034-small-preview.csv"
    holdout_dataset = out_dir / "task-034-small-holdout.binpack"
    holdout_manifest = out_dir / "task-034-small-holdout-manifest.json"
    holdout_preview = out_dir / "task-034-small-holdout-preview.csv"
    openings = out_dir / "task-034-local-openings.epd"
    checkpoint_dir = out_dir / "checkpoints"
    training_log = out_dir / "training.jsonl"
    network = out_dir / "task-034-small.nnue"
    export_log = out_dir / "export.jsonl"
    result_json = out_dir / "result.json"
    validation_log = out_dir / "validation.jsonl"

    generator = resolve_path(args.generator)
    nnue_bench = resolve_path(args.nnue_bench)
    engine = resolve_path(args.engine)
    cutechess = resolve_path(args.cutechess)
    python = sys.executable
    commands: list[CommandResult] = []
    failures: list[str] = []
    started = time.perf_counter()

    try:
        bench = command_or_fail(
            commands,
            "release engine bench",
            [str(engine), "bench"],
        )
        bench_nodes = parse_engine_bench_nodes(bench.output)
        if bench_nodes != EXPECTED_RELEASE_BENCH_NODES:
            raise RuntimeError(
                f"release bench changed: expected {EXPECTED_RELEASE_BENCH_NODES}, got {bench_nodes}"
            )
        command_or_fail(
            commands,
            "generate staged NNUE dataset",
            [
                str(generator),
                "generate",
                "--output",
                str(dataset),
                "--manifest",
                str(manifest),
                "--preview",
                str(preview),
                "--positions",
                str(args.positions),
                "--games",
                str(args.games),
                "--max-plies",
                str(args.max_plies),
                "--depth",
                str(args.depth),
                "--seed",
                str(args.seed),
                "--playout-policy",
                args.playout_policy,
                "--random-playout-permille",
                str(args.random_playout_permille),
                "--top-n-playout-permille",
                str(args.top_n_playout_permille),
                "--top-n",
                str(args.top_n),
                "--max-records-per-game",
                str(args.max_records_per_game),
                "--accepted-sample-stride",
                str(args.accepted_sample_stride),
                "--dataset-id",
                args.dataset_id,
                "--source-task",
                "TASK-034",
                "--notes",
                "TASK-034 staged local small-network dataset; no external data or Stockfish labels.",
                "--date",
                args.date,
            ],
        )
        command_or_fail(
            commands,
            "validate staged dataset manifest",
            [python, "tools/validate_nnue_manifest.py", str(manifest)],
        )
        if args.holdout_positions > 0:
            command_or_fail(
                commands,
                "generate holdout NNUE dataset",
                [
                    str(generator),
                    "generate",
                    "--output",
                    str(holdout_dataset),
                    "--manifest",
                    str(holdout_manifest),
                    "--preview",
                    str(holdout_preview),
                    "--positions",
                    str(args.holdout_positions),
                    "--games",
                    str(args.holdout_games),
                    "--max-plies",
                    str(args.max_plies),
                    "--depth",
                    str(args.depth),
                    "--seed",
                    str(args.seed + 1),
                    "--playout-policy",
                    args.playout_policy,
                    "--random-playout-permille",
                    str(args.random_playout_permille),
                    "--top-n-playout-permille",
                    str(args.top_n_playout_permille),
                    "--top-n",
                    str(args.top_n),
                    "--max-records-per-game",
                    str(args.max_records_per_game),
                    "--accepted-sample-stride",
                    str(args.accepted_sample_stride),
                    "--dataset-id",
                    f"{args.dataset_id}-holdout",
                    "--source-task",
                    "TASK-034",
                    "--notes",
                    "TASK-034 local holdout dataset; no external data or Stockfish labels.",
                    "--date",
                    args.date,
                ],
            )
            command_or_fail(
                commands,
                "validate holdout dataset manifest",
                [python, "tools/validate_nnue_manifest.py", str(holdout_manifest)],
            )
        opening_source_preview = holdout_preview if args.holdout_positions > 0 else preview
        opening_summary = write_openings_from_preview(
            opening_source_preview,
            openings,
            max_openings=args.opening_count,
            max_abs_eval=args.opening_max_abs_eval,
        )
        if opening_summary["count"] == 0:
            raise RuntimeError("opening generation produced no usable local EPD positions")
        training = command_or_fail(
            commands,
            "train small NNUE",
            [
                python,
                "tools/validate_nnue_training.py",
                "--dataset",
                str(dataset),
                "--manifest",
                str(manifest),
                "--architecture",
                "small",
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--eta-min",
                str(args.eta_min),
                "--lambda-eval",
                str(args.lambda_eval),
                "--clip-norm",
                str(args.clip_norm),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--log",
                str(training_log),
                "--determinism-runs",
                "1",
            ],
        )
        training_result = json_from_stdout(training.output)
        checkpoint = Path(str(training_result["checkpoint"]))
        export = command_or_fail(
            commands,
            "export small NNUE",
            [
                python,
                "tools/validate_nnue_export.py",
                "--checkpoint",
                str(checkpoint),
                "--dataset",
                str(dataset),
                "--output",
                str(network),
                "--log",
                str(export_log),
                "--batch-size",
                str(args.batch_size),
            ],
        )
        export_result = json_from_stdout(export.output)
        fit_metrics = prediction_metrics(checkpoint, dataset, args.batch_size)
        holdout_metrics = (
            prediction_metrics(checkpoint, holdout_dataset, args.batch_size)
            if args.holdout_positions > 0
            else None
        )

        python_checksum = python_quantized_checksum(checkpoint, dataset, args.batch_size)
        dataset_bench = command_or_fail(
            commands,
            "C++ dataset NNUE equivalence",
            [str(nnue_bench), "--network", str(network), "--dataset", str(dataset)],
        )
        cpp_checksum = parse_cpp_dataset_checksum(dataset_bench.output)
        checksum_match = python_checksum == cpp_checksum
        if not checksum_match:
            raise RuntimeError(
                f"Python/C++ checksum mismatch: python={python_checksum}, cpp={cpp_checksum}"
            )

        command_or_fail(
            commands,
            "C++ fixed NNUE bench",
            [str(nnue_bench), "--network", str(network), "--iterations", str(args.bench_iterations)],
        )

        match_result: dict[str, Any] | None = None
        match_openings: dict[str, Any] | None = None
        if args.skip_match:
            failures.append("match skipped; no +50 Elo evidence")
        else:
            match_dir = out_dir / "match"
            match = command_or_fail(
                commands,
                "HCE vs small NNUE match",
                [
                    python,
                    "tools/run_match.py",
                    "--preset",
                    "sprt-small-nnue",
                    "--base-engine",
                    str(engine),
                    "--test-engine",
                    str(engine),
                    "--base-name",
                    "LCE-HCE",
                    "--test-name",
                    "LCE-SmallNNUE",
                    "--test-option",
                    "Use NNUE=true",
                    "--test-option",
                    f"EvalFile={network}",
                    "--test-option",
                    "NNUE Backend=Scalar",
                    "--games",
                    str(args.match_games),
                    "--rounds",
                    str(args.match_rounds),
                    "--concurrency",
                    str(args.match_concurrency),
                    "--time-control",
                    args.match_time_control,
                    "--maxmoves",
                    str(args.match_maxmoves),
                    "--openings",
                    str(openings),
                    "--opening-format",
                    "epd",
                    "--opening-order",
                    args.opening_order,
                    "--opening-policy",
                    args.opening_policy,
                    "--repeat",
                    str(args.opening_repeat),
                    "--cutechess",
                    str(cutechess),
                    "--out-dir",
                    str(match_dir),
                ],
            )
            del match
            match_result = json.loads((match_dir / "result.json").read_text(encoding="utf-8"))

        match_classification = classify_match(match_result, args.minimum_match_games)
        if match_result is not None:
            match_openings = parse_match_opening_diversity(
                out_dir / "match" / "match.pgn",
                generated_openings=int(opening_summary["count"]),
                completed_games=int(match_classification["games_completed"]),
                repeat=args.opening_repeat,
            )
            if not bool(match_openings["pass"]):
                failures.append(
                    "match opening diversity insufficient "
                    f"({match_openings['unique_fens']}/"
                    f"{match_openings['required_unique_fens']} required unique FENs)"
                )
        if not match_classification["promotes_strength"]:
            failures.append(
                "small NNUE did not provide meaningful +50 Elo evidence "
                f"({match_classification['classification']}, "
                f"{match_classification['games_completed']}/"
                f"{match_classification['minimum_games']} games)"
            )

        if not failures:
            status = "pass"
        elif match_openings is not None and not bool(match_openings["pass"]):
            status = "fail"
        elif match_classification.get("classification") == "accept_h0":
            status = "rejected"
        else:
            status = "inconclusive"
        result = {
            "event": "nnue.validation.result",
            "task": "TASK-034",
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "status": status,
            "decision": "promote-small-network-pipeline"
            if status == "pass"
            else (
                "pipeline-failed"
                if status == "fail"
                else (
                    "reject-small-network-strength"
                    if status == "rejected"
                    else "needs-more-strength-evidence"
                )
            ),
            "dataset": str(dataset),
            "manifest": str(manifest),
            "holdout_dataset": str(holdout_dataset) if args.holdout_positions > 0 else None,
            "holdout_manifest": str(holdout_manifest) if args.holdout_positions > 0 else None,
            "openings": opening_summary,
            "match_openings": match_openings,
            "network": str(network),
            "training": training_result,
            "export": export_result,
            "bench_nodes": bench_nodes,
            "fit_metrics": fit_metrics,
            "holdout_metrics": holdout_metrics,
            "python_quantized_checksum": python_checksum,
            "cpp_quantized_checksum": cpp_checksum,
            "checksum_match": checksum_match,
            "match": match_result,
            "match_classification": match_classification,
            "commands": [command.as_dict() for command in commands],
            "failure_reasons": failures,
        }
    except Exception as exc:
        failures.append(str(exc))
        result = {
            "event": "nnue.validation.result",
            "task": "TASK-034",
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "status": "fail",
            "decision": "pipeline-failed",
            "commands": [command.as_dict() for command in commands],
            "failure_reasons": failures,
        }

    write_json(result_json, result)
    append_jsonl(validation_log, result)
    print(json.dumps(result, sort_keys=True))

    if result["status"] == "fail":
        return result
    if args.require_promotion and result["status"] != "pass":
        raise RuntimeError("small NNUE promotion evidence was not proven")
    return result


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def lambda_float(value: str) -> float:
    parsed = nonnegative_float(value)
    if parsed > 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TASK-034 small NNUE staged pipeline.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR / timestamp())
    parser.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)
    parser.add_argument("--nnue-bench", type=Path, default=DEFAULT_NNUE_BENCH)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--cutechess", type=Path, default=REPO_ROOT / ".deps/cutechess-build/cutechess-cli")
    parser.add_argument("--positions", type=positive_int, default=8192)
    parser.add_argument("--games", type=positive_int, default=768)
    parser.add_argument("--max-plies", type=positive_int, default=96)
    parser.add_argument("--depth", type=positive_int, default=2)
    parser.add_argument("--seed", type=int, default=34034)
    parser.add_argument("--playout-policy", choices=("best", "random", "mixed"), default="mixed")
    parser.add_argument("--random-playout-permille", type=nonnegative_int, default=250)
    parser.add_argument("--top-n-playout-permille", type=nonnegative_int, default=350)
    parser.add_argument("--top-n", type=positive_int, default=6)
    parser.add_argument("--max-records-per-game", type=nonnegative_int, default=8)
    parser.add_argument("--accepted-sample-stride", type=positive_int, default=1)
    parser.add_argument("--holdout-positions", type=nonnegative_int, default=8192)
    parser.add_argument("--holdout-games", type=positive_int, default=1024)
    parser.add_argument("--opening-count", type=positive_int, default=256)
    parser.add_argument("--opening-max-abs-eval", type=nonnegative_int, default=300)
    parser.add_argument("--opening-order", choices=("random", "sequential"), default="sequential")
    parser.add_argument("--opening-policy", choices=("default", "encounter", "round"), default="default")
    parser.add_argument("--opening-repeat", type=positive_int, default=2)
    parser.add_argument("--dataset-id", default="lce-task-034-small-staged")
    parser.add_argument("--date", default="2026-05-11T00:00:00Z")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--epochs", type=positive_int, default=64)
    parser.add_argument("--batch-size", type=positive_int, default=512)
    parser.add_argument("--learning-rate", type=positive_float, default=0.1)
    parser.add_argument("--eta-min", type=nonnegative_float, default=0.001)
    parser.add_argument("--lambda-eval", type=lambda_float, default=1.0)
    parser.add_argument("--clip-norm", type=positive_float, default=10.0)
    parser.add_argument("--bench-iterations", type=positive_int, default=1000)
    parser.add_argument("--match-games", type=positive_int, default=400)
    parser.add_argument("--match-rounds", type=positive_int, default=1)
    parser.add_argument("--match-concurrency", type=positive_int, default=4)
    parser.add_argument("--match-time-control", default="1+0.01")
    parser.add_argument("--match-maxmoves", type=nonnegative_int, default=0)
    parser.add_argument("--minimum-match-games", type=positive_int, default=400)
    parser.add_argument("--skip-match", action="store_true")
    parser.add_argument("--require-promotion", action="store_true")
    args = parser.parse_args(argv)
    if args.random_playout_permille + args.top_n_playout_permille > 1000:
        parser.error("--random-playout-permille + --top-n-playout-permille must be <= 1000")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"FAIL TASK-034 small NNUE validation: {exc}", file=sys.stderr)
        return 1
    return 0 if result["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
