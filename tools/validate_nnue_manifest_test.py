#!/usr/bin/env python3
"""Unit tests for the NNUE manifest validator."""

from __future__ import annotations

import unittest

from validate_nnue_manifest import validate_manifest


def valid_manifest() -> dict:
    return {
        "event": "trainer.dataset.manifest",
        "dataset_id": "lce-task-027-smoke-d5d6c290bdd640da",
        "positions": 8,
        "filters": {"reasons": {}},
        "source_engine": {"name": "LocalChessEngine"},
        "checksum": "0xd5d6c290bdd640da",
        "dataset_checksum": "0xd5d6c290bdd640da",
        "date": "2026-05-11T00:00:00Z",
        "external_data": False,
        "strength_evidence": False,
        "generation": {"positions_accepted": 8},
    }


class ManifestValidatorTest(unittest.TestCase):
    def test_valid_manifest_passes(self) -> None:
        self.assertEqual(validate_manifest(valid_manifest()), [])

    def test_required_an006_fields_are_enforced(self) -> None:
        manifest = valid_manifest()
        del manifest["dataset_id"]
        del manifest["date"]
        errors = validate_manifest(manifest)
        self.assertIn("missing required field: dataset_id", errors)
        self.assertIn("missing required field: date", errors)

    def test_positions_must_match_generation_count(self) -> None:
        manifest = valid_manifest()
        manifest["positions"] = 7
        self.assertIn("positions must equal generation.positions_accepted", validate_manifest(manifest))

    def test_checksum_alias_must_match_dataset_checksum(self) -> None:
        manifest = valid_manifest()
        manifest["dataset_checksum"] = "0x0000000000000000"
        self.assertIn("dataset_checksum must match checksum", validate_manifest(manifest))

    def test_manifest_date_is_utc_rfc3339_seconds(self) -> None:
        manifest = valid_manifest()
        manifest["date"] = "2026-05-11 00:00:00"
        self.assertIn(
            "date must use UTC RFC3339 seconds format YYYY-MM-DDTHH:MM:SSZ",
            validate_manifest(manifest),
        )


if __name__ == "__main__":
    unittest.main()
