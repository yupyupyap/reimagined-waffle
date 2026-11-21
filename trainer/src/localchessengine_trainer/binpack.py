from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy

from localchessengine_trainer import _binpack_native


BinpackError = _binpack_native.BinpackError


@dataclass(frozen=True)
class BinpackMetadata:
    path: Path
    format: str
    version: int
    record_count: int
    checksum: int

    @classmethod
    def from_native(cls, payload: dict[str, object]) -> "BinpackMetadata":
        return cls(
            path=Path(str(payload["path"])),
            format=str(payload["format"]),
            version=int(payload["version"]),
            record_count=int(payload["record_count"]),
            checksum=int(payload["checksum"]),
        )

    @property
    def checksum_hex(self) -> str:
        return f"0x{self.checksum:016x}"


@dataclass(frozen=True)
class BinpackValidation:
    path: Path
    record_count: int
    checksum: int

    @classmethod
    def from_native(cls, payload: dict[str, object]) -> "BinpackValidation":
        return cls(
            path=Path(str(payload["path"])),
            record_count=int(payload["record_count"]),
            checksum=int(payload["checksum"]),
        )

    @property
    def checksum_hex(self) -> str:
        return f"0x{self.checksum:016x}"


@dataclass(frozen=True)
class BinpackBatch:
    white_indices: numpy.ndarray
    black_indices: numpy.ndarray
    offsets: numpy.ndarray
    side_to_move: numpy.ndarray
    eval_cp: numpy.ndarray
    wdl: numpy.ndarray

    @classmethod
    def from_native(cls, payload: dict[str, object]) -> "BinpackBatch":
        return cls(
            white_indices=numpy.asarray(payload["white_indices"], dtype=numpy.int64),
            black_indices=numpy.asarray(payload["black_indices"], dtype=numpy.int64),
            offsets=numpy.asarray(payload["offsets"], dtype=numpy.int64),
            side_to_move=numpy.asarray(payload["side_to_move"], dtype=numpy.int64),
            eval_cp=numpy.asarray(payload["eval_cp"], dtype=numpy.float32),
            wdl=numpy.asarray(payload["wdl"], dtype=numpy.float32),
        )

    @property
    def positions(self) -> int:
        return int(self.eval_cp.shape[0])

    @property
    def feature_entries(self) -> int:
        return int(self.white_indices.shape[0])


class BinpackDataset:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def metadata(self) -> BinpackMetadata:
        return read_metadata(self.path)

    def validate(self) -> BinpackValidation:
        return validate_dataset(self.path)

    def iter_batches(self, batch_size: int) -> Iterator[BinpackBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for payload in _binpack_native.BatchIterator(str(self.path), int(batch_size)):
            yield BinpackBatch.from_native(payload)


def read_metadata(path: str | Path) -> BinpackMetadata:
    return BinpackMetadata.from_native(_binpack_native.metadata(str(path)))


def validate_dataset(path: str | Path) -> BinpackValidation:
    return BinpackValidation.from_native(_binpack_native.validate(str(path)))


def iter_batches(path: str | Path, batch_size: int) -> Iterator[BinpackBatch]:
    return BinpackDataset(path).iter_batches(batch_size)


__all__ = [
    "BinpackBatch",
    "BinpackDataset",
    "BinpackError",
    "BinpackMetadata",
    "BinpackValidation",
    "iter_batches",
    "read_metadata",
    "validate_dataset",
]
