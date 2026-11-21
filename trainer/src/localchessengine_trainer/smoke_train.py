from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy
import torch
from torch import nn
import torch.nn.functional as functional

from localchessengine_trainer.device import DeviceRequest, DeviceSelection, select_device


@dataclass(frozen=True)
class SmokeTrainConfig:
    device: DeviceRequest = "auto"
    log_path: Path = Path("build/trainer/task-028-smoke.jsonl")
    dataset_id: str = "synthetic-task-028-smoke"
    epochs: int = 2
    batch_size: int = 128
    input_features: int = 768
    features_per_position: int = 16
    hidden_size: int = 32
    learning_rate: float = 8.75e-4
    lambda_eval: float = 0.5
    clip_norm: float = 1.0
    seed: int = 28028


class SmokeNnueModel(nn.Module):
    def __init__(self, input_features: int, hidden_size: int) -> None:
        super().__init__()
        self.white = nn.EmbeddingBag(input_features, hidden_size, mode="sum")
        self.black = nn.EmbeddingBag(input_features, hidden_size, mode="sum")
        self.output = nn.Linear(hidden_size * 2, 1)

    @staticmethod
    def screlu(values: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(values, min=0.0, max=1.0)
        return clipped * clipped

    def forward(
        self,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        white = self.screlu(self.white(white_indices, offsets))
        black = self.screlu(self.black(black_indices, offsets))
        return self.output(torch.cat((white, black), dim=1)).squeeze(1)


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._handle.close()

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        record = {"event": event, "task": "TASK-028", **payload}
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _validate_config(config: SmokeTrainConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if config.input_features <= 0:
        raise ValueError("input-features must be positive")
    if config.features_per_position <= 0:
        raise ValueError("features-per-position must be positive")
    if config.hidden_size <= 0:
        raise ValueError("hidden-size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if not 0.0 <= config.lambda_eval <= 1.0:
        raise ValueError("lambda-eval must be in [0, 1]")
    if config.clip_norm <= 0.0:
        raise ValueError("clip-norm must be positive")


def _make_synthetic_batch(
    config: SmokeTrainConfig,
    device: torch.device,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + epoch)
    total_indices = config.batch_size * config.features_per_position
    white_cpu = torch.randint(
        low=0,
        high=config.input_features,
        size=(total_indices,),
        generator=generator,
        dtype=torch.long,
    )
    black_cpu = (white_cpu + (config.input_features // 2)) % config.input_features
    offsets_cpu = torch.arange(0, total_indices, config.features_per_position, dtype=torch.long)
    eval_cpu = torch.linspace(-300.0, 300.0, config.batch_size, dtype=torch.float32)
    wdl_cpu = torch.sigmoid(eval_cpu / 400.0)

    return (
        white_cpu.to(device),
        black_cpu.to(device),
        offsets_cpu.to(device),
        eval_cpu.to(device),
        wdl_cpu.to(device),
    )


def _dependency_payload(selection: DeviceSelection) -> dict[str, Any]:
    return {
        **selection.as_dict(),
        "dependency": "torch",
        "dependency_license": "BSD-3-Clause",
        "numpy_version": numpy.__version__,
        "numpy_license": "BSD-3-Clause",
    }


def _train_on_device(
    config: SmokeTrainConfig,
    selection: DeviceSelection,
    logger: JsonlLogger,
) -> dict[str, Any]:
    device = selection.torch_device
    torch.manual_seed(config.seed)
    model = SmokeNnueModel(config.input_features, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    final_loss = 0.0

    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        white, black, offsets, target_eval_cp, target_wdl = _make_synthetic_batch(
            config, device, epoch
        )

        optimizer.zero_grad(set_to_none=True)
        prediction_cp = model(white, black, offsets)
        prediction_prob = torch.sigmoid(prediction_cp / 400.0)
        target_eval_prob = torch.sigmoid(target_eval_cp / 400.0)
        eval_loss = functional.mse_loss(prediction_prob, target_eval_prob)
        wdl_loss = functional.mse_loss(prediction_prob, target_wdl)
        loss = (config.lambda_eval * eval_loss) + ((1.0 - config.lambda_eval) * wdl_loss)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_norm)
        optimizer.step()

        elapsed = max(time.perf_counter() - start, 1e-9)
        final_loss = float(loss.detach().cpu().item())
        logger.emit(
            "trainer.epoch.metrics",
            {
                "dataset_id": config.dataset_id,
                "epoch": epoch,
                "epochs": config.epochs,
                "loss": final_loss,
                "eval_loss": float(eval_loss.detach().cpu().item()),
                "wdl_loss": float(wdl_loss.detach().cpu().item()),
                "lr": optimizer.param_groups[0]["lr"],
                "throughput_positions_per_second": config.batch_size / elapsed,
                "device": selection.device_type,
                "batch_size": config.batch_size,
                "hidden_size": config.hidden_size,
                "features_per_position": config.features_per_position,
                "grad_norm": float(grad_norm.detach().cpu().item()),
            },
        )

    return {
        "status": "pass",
        "dataset_id": config.dataset_id,
        "device": selection.device_type,
        "fallback_used": selection.fallback_used,
        "final_loss": final_loss,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
    }


def run_smoke_training(config: SmokeTrainConfig) -> dict[str, Any]:
    _validate_config(config)
    with JsonlLogger(config.log_path) as logger:
        selection = select_device(config.device)
        logger.emit("trainer.dependency.check", _dependency_payload(selection))
        try:
            result = _train_on_device(config, selection, logger)
        except RuntimeError as exc:
            if config.device != "auto" or selection.device_type != "mps":
                raise
            fallback = DeviceSelection(
                requested="auto",
                device_type="cpu",
                torch_version=selection.torch_version,
                mps_built=selection.mps_built,
                mps_available=selection.mps_available,
                fallback_used=True,
                reason=f"MPS smoke train failed: {exc}",
                python_version=selection.python_version,
                platform=selection.platform,
                pytorch_enable_mps_fallback=selection.pytorch_enable_mps_fallback,
            )
            logger.emit("trainer.device.fallback", _dependency_payload(fallback))
            result = _train_on_device(config, fallback, logger)

        logger.emit("trainer.smoke_train.result", result)
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


def _lambda_value(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed


def parse_args(argv: list[str] | None = None) -> SmokeTrainConfig:
    parser = argparse.ArgumentParser(description="Run the LocalChessEngine NNUE trainer smoke.")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--log", type=Path, default=Path("build/trainer/task-028-smoke.jsonl"))
    parser.add_argument("--dataset-id", default="synthetic-task-028-smoke")
    parser.add_argument("--epochs", type=_positive_int, default=2)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--input-features", type=_positive_int, default=768)
    parser.add_argument("--features-per-position", type=_positive_int, default=16)
    parser.add_argument("--hidden-size", type=_positive_int, default=32)
    parser.add_argument("--learning-rate", type=_positive_float, default=8.75e-4)
    parser.add_argument("--lambda-eval", type=_lambda_value, default=0.5)
    parser.add_argument("--clip-norm", type=_positive_float, default=1.0)
    parser.add_argument("--seed", type=int, default=28028)
    args = parser.parse_args(argv)

    return SmokeTrainConfig(
        device=args.device,
        log_path=args.log,
        dataset_id=args.dataset_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_features=args.input_features,
        features_per_position=args.features_per_position,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        lambda_eval=args.lambda_eval,
        clip_norm=args.clip_norm,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        config = parse_args(argv)
        result = run_smoke_training(config)
    except Exception as exc:  # pragma: no cover - keeps CLI failure readable.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
