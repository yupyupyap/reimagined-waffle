from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Any

import numpy
import torch

from localchessengine_trainer.binpack import BinpackDataset
from localchessengine_trainer.model import PerspectiveNnueModel, resolve_architecture
from localchessengine_trainer.train import CHECKPOINT_SCHEMA, file_sha256, model_state_sha256


NETWORK_SCHEMA = "localchessengine.nnue.network.v2"
LEGACY_NETWORK_SCHEMA = "localchessengine.nnue.network.v1"
VALIDATION_EVENT = "nnue.validation.result"
NETWORK_MAGIC = b"LCENNUEWEIGHTS1\0"
NETWORK_VERSION = 2
LEGACY_NETWORK_VERSION = 1
QA = 255
QB = 64
MAX_METADATA_BYTES = 1024 * 1024
FNV_OFFSET_BASIS = 14695981039346656037
FNV_PRIME = 1099511628211
HEADER_STRUCT = struct.Struct("<16sIIQIIIIIIIQ")
HEADER_TAIL_STRUCT = struct.Struct("<IIIIIIIQ")
CHECKSUM_TAIL_OFFSET = 16 + 4 + 4 + 8
ARCH_IDS = {"small": 1, "v1": 2}
ACTIVATION_IDS = {"crelu": 1, "screlu": 2}
INT16_MIN = -(1 << 15)
INT16_MAX = (1 << 15) - 1
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class ExportConfig:
    checkpoint_path: Path
    output_path: Path
    dataset_path: Path
    batch_size: int = 4
    log_path: Path | None = None


@dataclass(frozen=True)
class QuantizedTensors:
    white_weight: numpy.ndarray
    black_weight: numpy.ndarray
    white_bias: numpy.ndarray
    black_bias: numpy.ndarray
    output_weight: numpy.ndarray
    output_bias: numpy.ndarray

    def tensor_bytes_for_version(self, version: int) -> bytes:
        chunks = [
            self.white_weight.tobytes(order="C"),
            self.black_weight.tobytes(order="C"),
        ]
        if version >= NETWORK_VERSION:
            chunks.extend(
                [
                    self.white_bias.tobytes(order="C"),
                    self.black_bias.tobytes(order="C"),
                ]
            )
        chunks.extend(
            [
                self.output_weight.tobytes(order="C"),
                self.output_bias.tobytes(order="C"),
            ]
        )
        return b"".join(chunks)

    @property
    def tensor_bytes(self) -> bytes:
        return self.tensor_bytes_for_version(NETWORK_VERSION)


def fnv1a64(chunks: list[bytes]) -> int:
    value = FNV_OFFSET_BASIS
    for chunk in chunks:
        for byte in chunk:
            value ^= byte
            value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def _schema_for_version(version: int) -> str:
    if version == LEGACY_NETWORK_VERSION:
        return LEGACY_NETWORK_SCHEMA
    if version == NETWORK_VERSION:
        return NETWORK_SCHEMA
    raise ValueError("network version mismatch")


def _network_version_for_architecture(architecture: Any) -> int:
    if bool(architecture.feature_bias) or bool(architecture.side_to_move_ordering):
        return NETWORK_VERSION
    return LEGACY_NETWORK_VERSION


def _architecture_from_payload(payload: dict[str, Any]) -> Any:
    architecture = resolve_architecture(str(payload.get("name")))
    has_v2_fields = "feature_bias" in payload or "side_to_move_ordering" in payload
    if not has_v2_fields:
        return replace(architecture, feature_bias=False, side_to_move_ordering=False)
    return replace(
        architecture,
        feature_bias=bool(payload.get("feature_bias", architecture.feature_bias)),
        side_to_move_ordering=bool(
            payload.get("side_to_move_ordering", architecture.side_to_move_ordering)
        ),
    )


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor], Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    architecture_payload = checkpoint.get("architecture")
    if not isinstance(architecture_payload, dict):
        raise ValueError("checkpoint has no architecture metadata")
    architecture = _architecture_from_payload(architecture_payload)
    if int(architecture_payload.get("input_features", -1)) != architecture.input_features:
        raise ValueError("checkpoint input feature count does not match architecture")
    if int(architecture_payload.get("hidden_size", -1)) != architecture.hidden_size:
        raise ValueError("checkpoint hidden size does not match architecture")
    if str(architecture_payload.get("activation")) != architecture.activation:
        raise ValueError("checkpoint activation does not match architecture")

    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model_state")

    expected_shapes = {
        "white.weight": (architecture.input_features, architecture.hidden_size),
        "black.weight": (architecture.input_features, architecture.hidden_size),
        "output.weight": (1, architecture.hidden_size * 2),
        "output.bias": (1,),
    }
    if architecture.feature_bias:
        expected_shapes |= {
            "white_bias": (architecture.hidden_size,),
            "black_bias": (architecture.hidden_size,),
        }
    for name, shape in expected_shapes.items():
        tensor = state.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"checkpoint model_state is missing tensor {name}")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"checkpoint tensor {name} has shape {tuple(tensor.shape)}, expected {shape}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"checkpoint tensor {name} must be float32")

    expected_digest = checkpoint.get("model_state_sha256")
    if not isinstance(expected_digest, str) or not expected_digest:
        raise ValueError("checkpoint has no model_state_sha256")
    observed_digest = model_state_sha256(state)
    if observed_digest != expected_digest:
        raise ValueError("checkpoint model_state_sha256 mismatch")

    normalized_state = dict(state)
    if "white_bias" not in normalized_state:
        normalized_state["white_bias"] = torch.zeros(architecture.hidden_size, dtype=torch.float32)
    if "black_bias" not in normalized_state:
        normalized_state["black_bias"] = torch.zeros(architecture.hidden_size, dtype=torch.float32)

    return checkpoint, normalized_state, architecture


def _quantize_array(name: str, tensor: torch.Tensor, scale: int, dtype: str) -> numpy.ndarray:
    rounded = torch.round(tensor.detach().cpu().contiguous() * scale)
    if dtype == "int16":
        minimum = INT16_MIN
        maximum = INT16_MAX
        numpy_dtype = "<i2"
        torch_dtype = torch.int16
    elif dtype == "int32":
        minimum = INT32_MIN
        maximum = INT32_MAX
        numpy_dtype = "<i4"
        torch_dtype = torch.int32
    else:
        raise ValueError(f"unsupported quantized dtype {dtype}")

    low = float(rounded.min().item())
    high = float(rounded.max().item())
    if low < minimum or high > maximum:
        raise ValueError(
            f"quantizing {name} would saturate {dtype}: observed range [{low}, {high}]"
        )
    return rounded.to(torch_dtype).numpy().astype(numpy_dtype, copy=False)


def quantize_state(state: dict[str, torch.Tensor]) -> QuantizedTensors:
    hidden_size = int(state["white.weight"].shape[1])
    white_bias = state.get("white_bias", torch.zeros(hidden_size, dtype=torch.float32))
    black_bias = state.get("black_bias", torch.zeros(hidden_size, dtype=torch.float32))
    return QuantizedTensors(
        white_weight=_quantize_array("white.weight", state["white.weight"], QA, "int16"),
        black_weight=_quantize_array("black.weight", state["black.weight"], QA, "int16"),
        white_bias=_quantize_array("white_bias", white_bias, QA, "int32"),
        black_bias=_quantize_array("black_bias", black_bias, QA, "int32"),
        output_weight=_quantize_array(
            "output.weight",
            state["output.weight"].reshape(-1),
            QB,
            "int16",
        ),
        output_bias=_quantize_array("output.bias", state["output.bias"], QA * QB, "int32"),
    )


def _quantized_batch_prediction(
    batch: Any,
    tensors: QuantizedTensors,
    activation: str,
    side_to_move_ordering: bool = False,
) -> numpy.ndarray:
    offsets = batch.offsets.astype(numpy.int64, copy=False)
    end_offsets = numpy.concatenate(
        (offsets[1:], numpy.array([batch.white_indices.shape[0]], dtype=numpy.int64))
    )
    predictions: list[float] = []
    output_weight = tensors.output_weight.astype(numpy.int64, copy=False)
    output_bias = int(tensors.output_bias[0])
    white_bias = tensors.white_bias.astype(numpy.int64, copy=False)
    black_bias = tensors.black_bias.astype(numpy.int64, copy=False)
    side_to_move = batch.side_to_move.astype(numpy.int64, copy=False)

    for row, (start, end) in enumerate(zip(offsets, end_offsets, strict=True)):
        white_acc = tensors.white_weight[batch.white_indices[start:end]].sum(
            axis=0,
            dtype=numpy.int64,
        ) + white_bias
        black_acc = tensors.black_weight[batch.black_indices[start:end]].sum(
            axis=0,
            dtype=numpy.int64,
        ) + black_bias
        white = numpy.clip(white_acc, 0, QA)
        black = numpy.clip(black_acc, 0, QA)
        if activation == "screlu":
            white = ((white * white) + (QA // 2)) // QA
            black = ((black * black) + (QA // 2)) // QA
        elif activation != "crelu":
            raise ValueError(f"unsupported activation {activation}")

        if side_to_move_ordering and int(side_to_move[row]) == 1:
            activated = numpy.concatenate((black, white)).astype(numpy.int64, copy=False)
        else:
            activated = numpy.concatenate((white, black)).astype(numpy.int64, copy=False)
        raw = int(numpy.dot(activated, output_weight)) + output_bias
        predictions.append(raw / float(QA * QB))

    return numpy.asarray(predictions, dtype=numpy.float64)


def compute_quantization_error(
    *,
    model: PerspectiveNnueModel,
    tensors: QuantizedTensors,
    architecture: Any,
    dataset_path: Path,
    batch_size: int,
) -> dict[str, float | int]:
    model.eval()
    errors: list[float] = []
    positions = 0
    with torch.no_grad():
        for batch in BinpackDataset(dataset_path).iter_batches(batch_size):
            white_indices = torch.as_tensor(batch.white_indices, dtype=torch.long)
            black_indices = torch.as_tensor(batch.black_indices, dtype=torch.long)
            offsets = torch.as_tensor(batch.offsets, dtype=torch.long)
            side_to_move = torch.as_tensor(batch.side_to_move, dtype=torch.long)
            model_side = side_to_move if architecture.side_to_move_ordering else None
            expected = model(white_indices, black_indices, offsets, model_side).detach().cpu().numpy()
            observed = _quantized_batch_prediction(
                batch,
                tensors,
                architecture.activation,
                architecture.side_to_move_ordering,
            )
            errors.extend(numpy.abs(observed - expected).astype(numpy.float64).tolist())
            positions += batch.positions

    if not errors:
        raise ValueError("quantization error requires at least one dataset position")

    error_array = numpy.asarray(errors, dtype=numpy.float64)
    return {
        "positions": positions,
        "mean_abs_cp": float(error_array.mean()),
        "max_abs_cp": float(error_array.max()),
        "rms_cp": float(numpy.sqrt(numpy.mean(error_array * error_array))),
    }


def _expected_tensor_bytes(
    input_features: int,
    hidden_size: int,
    version: int = NETWORK_VERSION,
) -> int:
    feature_bytes = 2 * input_features * hidden_size * numpy.dtype("<i2").itemsize
    hidden_bias_bytes = (
        2 * hidden_size * numpy.dtype("<i4").itemsize if version >= NETWORK_VERSION else 0
    )
    output_weight_bytes = 2 * hidden_size * numpy.dtype("<i2").itemsize
    output_bias_bytes = numpy.dtype("<i4").itemsize
    return feature_bytes + hidden_bias_bytes + output_weight_bytes + output_bias_bytes


def _metadata_bytes(payload: dict[str, Any]) -> bytes:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(encoded) > MAX_METADATA_BYTES:
        raise ValueError("network metadata exceeds maximum size")
    return encoded


def _pack_header(
    *,
    checksum: int,
    version: int,
    architecture_name: str,
    input_features: int,
    hidden_size: int,
    activation: str,
    metadata_len: int,
    tensor_bytes: int,
) -> bytes:
    return HEADER_STRUCT.pack(
        NETWORK_MAGIC,
        version,
        HEADER_STRUCT.size,
        checksum,
        ARCH_IDS[architecture_name],
        input_features,
        hidden_size,
        ACTIVATION_IDS[activation],
        QA,
        QB,
        metadata_len,
        tensor_bytes,
    )


def _pack_header_tail(
    *,
    architecture_name: str,
    input_features: int,
    hidden_size: int,
    activation: str,
    metadata_len: int,
    tensor_bytes: int,
) -> bytes:
    return HEADER_TAIL_STRUCT.pack(
        ARCH_IDS[architecture_name],
        input_features,
        hidden_size,
        ACTIVATION_IDS[activation],
        QA,
        QB,
        metadata_len,
        tensor_bytes,
    )


def read_network_info(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < HEADER_STRUCT.size:
        raise ValueError("network file is truncated")
    header = data[: HEADER_STRUCT.size]
    (
        magic,
        version,
        header_size,
        checksum,
        arch_id,
        input_features,
        hidden_size,
        activation_id,
        qa,
        qb,
        metadata_len,
        tensor_bytes,
    ) = HEADER_STRUCT.unpack(header)

    if magic != NETWORK_MAGIC:
        raise ValueError("network magic mismatch")
    if version not in (LEGACY_NETWORK_VERSION, NETWORK_VERSION):
        raise ValueError("network version mismatch")
    if header_size != HEADER_STRUCT.size:
        raise ValueError("network header size mismatch")
    if qa != QA or qb != QB:
        raise ValueError("network quantization constants mismatch")
    if metadata_len > MAX_METADATA_BYTES:
        raise ValueError("network metadata is too large")

    architectures_by_id = {value: key for key, value in ARCH_IDS.items()}
    activations_by_id = {value: key for key, value in ACTIVATION_IDS.items()}
    if arch_id not in architectures_by_id or activation_id not in activations_by_id:
        raise ValueError("network architecture metadata is unsupported")
    architecture = resolve_architecture(architectures_by_id[arch_id])
    if input_features != architecture.input_features:
        raise ValueError("network input feature count mismatch")
    if hidden_size != architecture.hidden_size:
        raise ValueError("network hidden size mismatch")
    if activations_by_id[activation_id] != architecture.activation:
        raise ValueError("network activation mismatch")
    if tensor_bytes != _expected_tensor_bytes(input_features, hidden_size, version):
        raise ValueError("network tensor byte count mismatch")

    expected_size = HEADER_STRUCT.size + metadata_len + tensor_bytes
    if len(data) != expected_size:
        raise ValueError("network file size does not match header")

    metadata_start = HEADER_STRUCT.size
    metadata_end = metadata_start + metadata_len
    metadata_raw = data[metadata_start:metadata_end]
    tensors = data[metadata_end:]
    observed_checksum = fnv1a64([header[CHECKSUM_TAIL_OFFSET:], metadata_raw, tensors])
    if observed_checksum != checksum:
        raise ValueError("network checksum mismatch")
    metadata = json.loads(metadata_raw.decode("utf-8"))
    if metadata.get("schema") != _schema_for_version(version):
        raise ValueError("network metadata schema mismatch")

    return {
        "path": str(path),
        "schema": metadata["schema"],
        "version": version,
        "architecture": architecture.name,
        "activation": architecture.activation,
        "input_features": input_features,
        "hidden_size": hidden_size,
        "qa": qa,
        "qb": qb,
        "metadata_len": metadata_len,
        "tensor_bytes": tensor_bytes,
        "checksum64": f"0x{checksum:016x}",
        "network_id": file_sha256(path),
        "metadata": metadata,
    }


def export_network(config: ExportConfig) -> dict[str, Any]:
    if config.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    checkpoint, state, architecture = _load_checkpoint(config.checkpoint_path)
    tensors = quantize_state(state)
    network_version = _network_version_for_architecture(architecture)
    network_schema = _schema_for_version(network_version)
    tensor_payload = tensors.tensor_bytes_for_version(network_version)
    expected_bytes = _expected_tensor_bytes(
        architecture.input_features,
        architecture.hidden_size,
        network_version,
    )
    if len(tensor_payload) != expected_bytes:
        raise ValueError("internal tensor payload size mismatch")

    model = PerspectiveNnueModel(architecture)
    model.load_state_dict(state)
    quant_error = compute_quantization_error(
        model=model,
        tensors=tensors,
        architecture=architecture,
        dataset_path=config.dataset_path,
        batch_size=config.batch_size,
    )

    metadata = {
        "schema": network_schema,
        "format_version": network_version,
        "source_checkpoint": str(config.checkpoint_path),
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_sha256": file_sha256(config.checkpoint_path),
        "model_state_sha256": checkpoint["model_state_sha256"],
        "dataset": checkpoint.get("dataset", {}),
        "architecture": architecture.as_dict(),
        "quantization": {
            "qa": QA,
            "qb": QB,
            "feature_dtype": "int16",
            "feature_bias_dtype": "int32" if network_version >= NETWORK_VERSION else None,
            "output_weight_dtype": "int16",
            "output_bias_dtype": "int32",
            "tensor_order": (
                [
                    "white.weight",
                    "black.weight",
                    "white_bias",
                    "black_bias",
                    "output.weight",
                    "output.bias",
                ]
                if network_version >= NETWORK_VERSION
                else [
                    "white.weight",
                    "black.weight",
                    "output.weight",
                    "output.bias",
                ]
            ),
        },
        "quant_error_cp": quant_error,
    }
    metadata_payload = _metadata_bytes(metadata)
    header_tail = _pack_header_tail(
        architecture_name=architecture.name,
        input_features=architecture.input_features,
        hidden_size=architecture.hidden_size,
        activation=architecture.activation,
        metadata_len=len(metadata_payload),
        tensor_bytes=len(tensor_payload),
    )
    checksum = fnv1a64([header_tail, metadata_payload, tensor_payload])
    header = _pack_header(
        checksum=checksum,
        version=network_version,
        architecture_name=architecture.name,
        input_features=architecture.input_features,
        hidden_size=architecture.hidden_size,
        activation=architecture.activation,
        metadata_len=len(metadata_payload),
        tensor_bytes=len(tensor_payload),
    )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_bytes(header + metadata_payload + tensor_payload)
    info = read_network_info(config.output_path)
    result = {
        "status": "pass",
        "network": str(config.output_path),
        "network_id": info["network_id"],
        "format_version": network_version,
        "architecture": architecture.name,
        "activation": architecture.activation,
        "hidden_size": architecture.hidden_size,
        "feature_bias": architecture.feature_bias,
        "side_to_move_ordering": architecture.side_to_move_ordering,
        "qa": QA,
        "qb": QB,
        "checksum64": info["checksum64"],
        "metadata_len": info["metadata_len"],
        "tensor_bytes": info["tensor_bytes"],
        "quant_error_cp": quant_error,
        "checkpoint": str(config.checkpoint_path),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "dataset_id": metadata["dataset"].get("dataset_id"),
    }
    if config.log_path is not None:
        config.log_path.parent.mkdir(parents=True, exist_ok=True)
        config.log_path.write_text(
            json.dumps({"event": VALIDATION_EVENT, "task": "TASK-031", **result}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> ExportConfig:
    parser = argparse.ArgumentParser(
        description="Export a LocalChessEngine NNUE checkpoint to quantized .nnue format."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args(argv)
    return ExportConfig(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        log_path=args.log,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        result = export_network(parse_args(argv))
    except Exception as exc:  # pragma: no cover - keeps CLI failure readable.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
