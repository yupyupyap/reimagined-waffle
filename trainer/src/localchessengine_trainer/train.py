from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy
import torch
from torch import nn
import torch.nn.functional as functional

from localchessengine_trainer.binpack import BinpackDataset
from localchessengine_trainer.device import DeviceRequest, DeviceSelection, select_device
from localchessengine_trainer.model import PerspectiveNnueModel, resolve_architecture


CHECKPOINT_SCHEMA = "localchessengine.nnue.checkpoint.v1"
TRAINING_RESULT_EVENT = "trainer.training.result"
EPOCH_METRICS_EVENT = "trainer.epoch.metrics"
DATASET_METADATA_EVENT = "trainer.dataset.metadata"
DEPENDENCY_EVENT = "trainer.dependency.check"


@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: Path
    manifest_path: Path
    architecture: str = "small"
    device: DeviceRequest = "auto"
    epochs: int = 16
    batch_size: int = 4
    learning_rate: float = 8.75e-4
    eta_min: float = 5.0e-5
    lambda_eval: float = 0.7
    clip_norm: float = 1.0
    seed: int = 30030
    checkpoint_dir: Path = Path("build/trainer/task-030-checkpoints")
    log_path: Path = Path("build/trainer/task-030-train.jsonl")
    eval_scale: float = 400.0


@dataclass(frozen=True)
class LossBreakdown:
    loss: torch.Tensor
    eval_loss: torch.Tensor
    wdl_loss: torch.Tensor


@dataclass(frozen=True)
class BatchTensors:
    white_indices: torch.Tensor
    black_indices: torch.Tensor
    offsets: torch.Tensor
    side_to_move: torch.Tensor
    eval_cp: torch.Tensor
    wdl: torch.Tensor

    @property
    def positions(self) -> int:
        return int(self.eval_cp.numel())


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._handle.close()

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        record = {"event": event, "task": "TASK-030", **payload}
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _checksum_hex(value: int) -> str:
    return f"0x{value:016x}"


def _parse_checksum(value: str) -> int:
    return int(value, 16 if value.startswith("0x") else 10)


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("event") != "trainer.dataset.manifest":
        raise ValueError("manifest event must be trainer.dataset.manifest")
    return payload


def _validate_manifest_contract(manifest: dict[str, Any], positions: int, checksum: int) -> None:
    external_data = bool(manifest.get("external_data", False))
    if external_data:
        raise ValueError(
            "external NNUE label datasets are not supported in the promoted release path; "
            "use a FEN pool plus C++ local relabeling instead"
        )
    if int(manifest["positions"]) != positions:
        raise ValueError(
            f"manifest positions {manifest['positions']} != dataset positions {positions}"
        )
    expected_checksum = _parse_checksum(str(manifest["checksum"]))
    if expected_checksum != checksum:
        raise ValueError(
            "manifest checksum "
            f"{_checksum_hex(expected_checksum)} != dataset checksum {_checksum_hex(checksum)}"
        )

    filters = manifest.get("filters")
    if not isinstance(filters, dict):
        raise ValueError("manifest must include NNUE filter metadata")
    source = manifest.get("source")
    is_local_relabel = (
        isinstance(source, dict)
        and source.get("kind") == "local_relabel_from_lichess_fen_pool"
        and source.get("stockfish_labels_used") is False
    )
    if is_local_relabel:
        if bool(manifest.get("external_position_source", False)) is not True:
            raise ValueError("local relabel manifest must record external_position_source=true")
        if not isinstance(manifest.get("source_engine"), dict):
            raise ValueError("local relabel manifest must include source_engine metadata")
        if int(filters.get("max_abs_eval_cp", 0)) > 2500:
            raise ValueError("local relabel manifest max_abs_eval_cp must be <= 2500")
        if filters.get("skip_in_check") is not True:
            raise ValueError("local relabel manifest must record skip_in_check=true")
        if filters.get("skip_best_move_capture") is not True:
            raise ValueError("local relabel manifest must record skip_best_move_capture=true")
        if filters.get("skip_terminal_or_no_bestmove") is not True:
            raise ValueError("local relabel manifest must record terminal/no-bestmove filtering")
        wdl = manifest.get("wdl")
        if not isinstance(wdl, dict) or wdl.get("orientation") != "white_relative":
            raise ValueError("manifest WDL orientation must be white_relative")
        return

    if int(filters.get("min_ply", -1)) != 8:
        raise ValueError("manifest min_ply must be 8 for TASK-030 training")
    if int(filters.get("max_abs_eval_cp", 0)) > 2500:
        raise ValueError("manifest max_abs_eval_cp must be <= 2500")
    if filters.get("skip_in_check") is not True:
        raise ValueError("manifest must record skip_in_check=true")
    if filters.get("skip_best_move_capture") is not True:
        raise ValueError("manifest must record skip_best_move_capture=true")

    wdl = manifest.get("wdl")
    if not isinstance(wdl, dict) or wdl.get("orientation") != "white_relative":
        raise ValueError("manifest WDL orientation must be white_relative")


def load_dataset_metadata(dataset_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    dataset = BinpackDataset(dataset_path)
    metadata = dataset.metadata()
    validation = dataset.validate()
    if metadata.record_count != validation.record_count or metadata.checksum != validation.checksum:
        raise ValueError("dataset metadata and checksum validation disagree")
    _validate_manifest_contract(manifest, validation.record_count, validation.checksum)
    return {
        "dataset_id": str(manifest["dataset_id"]),
        "dataset": str(dataset_path),
        "manifest": str(manifest_path),
        "format": metadata.format,
        "version": metadata.version,
        "positions": validation.record_count,
        "checksum": _checksum_hex(validation.checksum),
        "source_engine": manifest.get("source_engine", {}),
        "source": manifest.get("source", {}),
        "external_data": bool(manifest.get("external_data", False)),
        "filters": manifest["filters"],
        "wdl": manifest["wdl"],
    }


def _dependency_payload(selection: DeviceSelection) -> dict[str, Any]:
    return {
        **selection.as_dict(),
        "dependency": "torch",
        "dependency_license": "BSD-3-Clause",
        "numpy_version": numpy.__version__,
        "numpy_license": "BSD-3-Clause",
    }


def _validate_config(config: TrainingConfig) -> None:
    resolve_architecture(config.architecture)
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if config.eta_min < 0.0:
        raise ValueError("eta-min must be non-negative")
    if config.eta_min > config.learning_rate:
        raise ValueError("eta-min must not exceed learning-rate")
    if not 0.0 <= config.lambda_eval <= 1.0:
        raise ValueError("lambda-eval must be in [0, 1]")
    if config.clip_norm <= 0.0:
        raise ValueError("clip-norm must be positive")
    if config.eval_scale <= 0.0:
        raise ValueError("eval-scale must be positive")


def _seed_everything(seed: int, *, deterministic_cpu: bool) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic_cpu:
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)


def batch_to_tensors(batch: Any, device: torch.device) -> BatchTensors:
    return BatchTensors(
        white_indices=torch.as_tensor(batch.white_indices, dtype=torch.long, device=device),
        black_indices=torch.as_tensor(batch.black_indices, dtype=torch.long, device=device),
        offsets=torch.as_tensor(batch.offsets, dtype=torch.long, device=device),
        side_to_move=torch.as_tensor(batch.side_to_move, dtype=torch.long, device=device),
        eval_cp=torch.as_tensor(batch.eval_cp, dtype=torch.float32, device=device),
        wdl=torch.as_tensor(batch.wdl, dtype=torch.float32, device=device),
    )


def interpolated_loss(
    prediction_cp: torch.Tensor,
    target_eval_cp: torch.Tensor,
    target_wdl: torch.Tensor,
    *,
    lambda_eval: float,
    eval_scale: float = 400.0,
) -> LossBreakdown:
    prediction_probability = torch.sigmoid(prediction_cp / eval_scale)
    target_eval_probability = torch.sigmoid(target_eval_cp / eval_scale)
    eval_loss = functional.mse_loss(prediction_probability, target_eval_probability)
    wdl_loss = functional.mse_loss(prediction_probability, target_wdl)
    loss = (lambda_eval * eval_loss) + ((1.0 - lambda_eval) * wdl_loss)
    return LossBreakdown(loss=loss, eval_loss=eval_loss, wdl_loss=wdl_loss)


def model_prediction(model: PerspectiveNnueModel, tensors: BatchTensors) -> torch.Tensor:
    side_to_move = tensors.side_to_move if model.architecture.side_to_move_ordering else None
    return model(tensors.white_indices, tensors.black_indices, tensors.offsets, side_to_move)


def loss_targets(
    model: PerspectiveNnueModel,
    tensors: BatchTensors,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not model.architecture.side_to_move_ordering:
        return tensors.eval_cp, tensors.wdl
    black_to_move = tensors.side_to_move == 1
    target_eval = torch.where(black_to_move, -tensors.eval_cp, tensors.eval_cp)
    target_wdl = torch.where(black_to_move, 1.0 - tensors.wdl, tensors.wdl)
    return target_eval, target_wdl


def _empty_metric_totals() -> dict[str, float]:
    return {"loss": 0.0, "eval_loss": 0.0, "wdl_loss": 0.0}


def _weighted_average(totals: dict[str, float], positions: int) -> dict[str, float]:
    divisor = max(positions, 1)
    return {key: value / divisor for key, value in totals.items()}


def evaluate_dataset(
    model: PerspectiveNnueModel,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    totals = _empty_metric_totals()
    positions = 0
    batches = 0
    with torch.no_grad():
        for batch in BinpackDataset(config.dataset_path).iter_batches(config.batch_size):
            tensors = batch_to_tensors(batch, device)
            target_eval, target_wdl = loss_targets(model, tensors)
            breakdown = interpolated_loss(
                model_prediction(model, tensors),
                target_eval,
                target_wdl,
                lambda_eval=config.lambda_eval,
                eval_scale=config.eval_scale,
            )
            count = tensors.positions
            totals["loss"] += float(breakdown.loss.detach().cpu().item()) * count
            totals["eval_loss"] += float(breakdown.eval_loss.detach().cpu().item()) * count
            totals["wdl_loss"] += float(breakdown.wdl_loss.detach().cpu().item()) * count
            positions += count
            batches += 1

    return {"positions": positions, "batches": batches, **_weighted_average(totals, positions)}


def train_one_epoch(
    model: PerspectiveNnueModel,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    totals = _empty_metric_totals()
    positions = 0
    batches = 0
    grad_norm = 0.0
    start = time.perf_counter()

    for batch in BinpackDataset(config.dataset_path).iter_batches(config.batch_size):
        tensors = batch_to_tensors(batch, device)
        optimizer.zero_grad(set_to_none=True)
        target_eval, target_wdl = loss_targets(model, tensors)
        breakdown = interpolated_loss(
            model_prediction(model, tensors),
            target_eval,
            target_wdl,
            lambda_eval=config.lambda_eval,
            eval_scale=config.eval_scale,
        )
        breakdown.loss.backward()
        clipped_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_norm)
        optimizer.step()

        count = tensors.positions
        totals["loss"] += float(breakdown.loss.detach().cpu().item()) * count
        totals["eval_loss"] += float(breakdown.eval_loss.detach().cpu().item()) * count
        totals["wdl_loss"] += float(breakdown.wdl_loss.detach().cpu().item()) * count
        grad_norm = float(clipped_norm.detach().cpu().item())
        positions += count
        batches += 1

    elapsed = max(time.perf_counter() - start, 1e-9)
    return {
        "positions": positions,
        "batches": batches,
        "throughput_positions_per_second": positions / elapsed,
        "grad_norm": grad_norm,
        **_weighted_average(totals, positions),
    }


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().cpu().contiguous().numpy()
    return array.tobytes(order="C")


def model_state_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def save_checkpoint(
    *,
    config: TrainingConfig,
    model: PerspectiveNnueModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataset_metadata: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
) -> dict[str, str]:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.checkpoint_dir / f"checkpoint-epoch-{epoch:04d}.pt"
    model_state = _to_cpu(model.state_dict())
    model_digest = model_state_sha256(model_state)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "task": "TASK-030",
        "architecture": model.architecture.as_dict(),
        "dataset": dataset_metadata,
        "epoch": epoch,
        "epochs": config.epochs,
        "seed": config.seed,
        "lambda_eval": config.lambda_eval,
        "eval_scale": config.eval_scale,
        "metrics": metrics,
        "model_state_sha256": model_digest,
        "model_state": model_state,
        "optimizer_state": _to_cpu(optimizer.state_dict()),
        "scheduler_state": _to_cpu(scheduler.state_dict()),
    }
    torch.save(checkpoint, checkpoint_path)
    return {
        "checkpoint": str(checkpoint_path),
        "model_state_sha256": model_digest,
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }


def run_training(config: TrainingConfig) -> dict[str, Any]:
    _validate_config(config)
    dataset_metadata = load_dataset_metadata(config.dataset_path, config.manifest_path)
    selection = select_device(config.device)
    deterministic_cpu = selection.device_type == "cpu"
    _seed_everything(config.seed, deterministic_cpu=deterministic_cpu)

    architecture = resolve_architecture(config.architecture)
    device = selection.torch_device
    model = PerspectiveNnueModel(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.eta_min,
    )

    with JsonlLogger(config.log_path) as logger:
        logger.emit(DEPENDENCY_EVENT, _dependency_payload(selection))
        logger.emit(DATASET_METADATA_EVENT, dataset_metadata)
        initial = evaluate_dataset(model, config, device)

        final_metrics = initial
        checkpoint_info: dict[str, str] = {}
        for epoch in range(1, config.epochs + 1):
            lr = float(optimizer.param_groups[0]["lr"])
            train_metrics = train_one_epoch(model, optimizer, config, device)
            final_metrics = evaluate_dataset(model, config, device)
            scheduler.step()
            metric_payload = {
                "dataset_id": dataset_metadata["dataset_id"],
                "architecture": architecture.name,
                "activation": architecture.activation,
                "hidden_size": architecture.hidden_size,
                "feature_bias": architecture.feature_bias,
                "side_to_move_ordering": architecture.side_to_move_ordering,
                "epoch": epoch,
                "epochs": config.epochs,
                "device": selection.device_type,
                "batch_size": config.batch_size,
                "learning_rate": lr,
                "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "lambda_eval": config.lambda_eval,
                "clip_norm": config.clip_norm,
                "train_loss": train_metrics["loss"],
                "train_eval_loss": train_metrics["eval_loss"],
                "train_wdl_loss": train_metrics["wdl_loss"],
                "loss": final_metrics["loss"],
                "eval_loss": final_metrics["eval_loss"],
                "wdl_loss": final_metrics["wdl_loss"],
                "initial_loss": initial["loss"],
                "throughput_positions_per_second": train_metrics[
                    "throughput_positions_per_second"
                ],
                "positions": train_metrics["positions"],
                "batches": train_metrics["batches"],
                "grad_norm": train_metrics["grad_norm"],
            }
            checkpoint_metrics = {
                key: value
                for key, value in metric_payload.items()
                if key != "throughput_positions_per_second"
            }
            checkpoint_info = save_checkpoint(
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataset_metadata=dataset_metadata,
                epoch=epoch,
                metrics=checkpoint_metrics,
            )
            logger.emit(
                "trainer.checkpoint.saved",
                {
                    "dataset_id": dataset_metadata["dataset_id"],
                    "epoch": epoch,
                    **checkpoint_info,
                },
            )
            logger.emit(EPOCH_METRICS_EVENT, {**metric_payload, **checkpoint_info})

        result = {
            "status": "pass",
            "dataset_id": dataset_metadata["dataset_id"],
            "architecture": architecture.name,
            "activation": architecture.activation,
            "hidden_size": architecture.hidden_size,
            "feature_bias": architecture.feature_bias,
            "side_to_move_ordering": architecture.side_to_move_ordering,
            "device": selection.device_type,
            "fallback_used": selection.fallback_used,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "initial_loss": initial["loss"],
            "final_loss": final_metrics["loss"],
            "final_eval_loss": final_metrics["eval_loss"],
            "final_wdl_loss": final_metrics["wdl_loss"],
            **checkpoint_info,
        }
        logger.emit(TRAINING_RESULT_EVENT, result)
        return result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _lambda_value(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed


def parse_args(argv: list[str] | None = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train a LocalChessEngine NNUE checkpoint.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--architecture", choices=["small", "v1"], default="small")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--epochs", type=_positive_int, default=16)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--learning-rate", type=_positive_float, default=8.75e-4)
    parser.add_argument("--eta-min", type=_non_negative_float, default=5.0e-5)
    parser.add_argument("--lambda-eval", type=_lambda_value, default=0.7)
    parser.add_argument("--clip-norm", type=_positive_float, default=1.0)
    parser.add_argument("--seed", type=int, default=30030)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("build/trainer/task-030-checkpoints"),
    )
    parser.add_argument("--log", type=Path, default=Path("build/trainer/task-030-train.jsonl"))
    args = parser.parse_args(argv)

    return TrainingConfig(
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
        checkpoint_dir=args.checkpoint_dir,
        log_path=args.log,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        config = parse_args(argv)
        result = run_training(config)
    except Exception as exc:  # pragma: no cover - keeps CLI failure readable.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
