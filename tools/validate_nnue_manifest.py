#!/usr/bin/env python3
"""Validate LocalChessEngine NNUE dataset manifest artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


EVENT = "trainer.dataset.manifest"
UTC_RFC3339_SECONDS = re.compile(
    r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])T([01]\d|2[0-3]):[0-5]\d:[0-5]\dZ$"
)
CHECKSUM_HEX = re.compile(r"^0x[0-9a-f]{16}$")


def _is_object(value: Any) -> bool:
    return isinstance(value, dict)


def validate_manifest(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    required = ["dataset_id", "positions", "filters", "source_engine", "checksum", "date"]
    for key in required:
        if key not in data:
            errors.append(f"missing required field: {key}")

    if data.get("event") != EVENT:
        errors.append(f"event must be {EVENT!r}")
    if not isinstance(data.get("dataset_id"), str) or not data.get("dataset_id"):
        errors.append("dataset_id must be a non-empty string")
    if not isinstance(data.get("positions"), int) or data.get("positions", 0) < 0:
        errors.append("positions must be a non-negative integer")
    if not isinstance(data.get("date"), str) or not UTC_RFC3339_SECONDS.fullmatch(data["date"]):
        errors.append("date must use UTC RFC3339 seconds format YYYY-MM-DDTHH:MM:SSZ")
    if not isinstance(data.get("checksum"), str) or not CHECKSUM_HEX.fullmatch(data["checksum"]):
        errors.append("checksum must be a 16-digit lowercase hex string with 0x prefix")
    if not isinstance(data.get("dataset_checksum"), str) or data.get("dataset_checksum") != data.get(
        "checksum"
    ):
        errors.append("dataset_checksum must match checksum")
    if data.get("external_data") is not False:
        errors.append("external_data must be false")
    if data.get("strength_evidence") is not False:
        errors.append("strength_evidence must be false")

    generation = data.get("generation")
    if not _is_object(generation):
        errors.append("generation must be an object")
    elif data.get("positions") != generation.get("positions_accepted"):
        errors.append("positions must equal generation.positions_accepted")

    if not _is_object(data.get("filters")):
        errors.append("filters must be an object")
    if not _is_object(data.get("source_engine")):
        errors.append("source_engine must be an object")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    errors = validate_manifest(data)
    if errors:
        for error in errors:
            print(f"{args.manifest}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
