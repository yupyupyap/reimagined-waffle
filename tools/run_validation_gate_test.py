#!/usr/bin/env python3
"""Parser and policy tests for tools/run_validation_gate.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def load_gate_module():
    module_path = Path(__file__).with_name("run_validation_gate.py")
    spec = importlib.util.spec_from_file_location("run_validation_gate", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = load_gate_module()


class RunValidationGateTest(unittest.TestCase):
    def test_parses_last_bench_line(self) -> None:
        output = "\n".join(
            [
                "info string warmup",
                "10 nodes 20 nps",
                "18417 nodes 2302125 nps",
            ]
        )

        parsed = GATE.parse_bench_output(output)

        self.assertEqual(parsed, {"nodes": 18417, "nps": 2302125})

    def test_parses_perft_nodes_line(self) -> None:
        output = "\n".join(
            [
                "fen: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "depth: 4",
                "nodes: 197281",
            ]
        )

        self.assertEqual(GATE.parse_perft_nodes(output), 197281)

    def test_classifies_sprt_bounds(self) -> None:
        base_result = {
            "match": {"games_completed": 500},
            "sprt": {
                "enabled": True,
                "bounds": [0.0, 5.0],
                "output": {"llr": 0.0, "lower_bound": -2.94, "upper_bound": 2.94},
            },
        }

        inconclusive = GATE.classify_sprt(base_result, minimum_games=400)
        self.assertEqual(inconclusive["classification"], "inconclusive")
        self.assertEqual(inconclusive["bounds_type"], "gainer")
        self.assertEqual(inconclusive["evidence_kind"], "meaningful")
        self.assertFalse(inconclusive["promotes_strength"])

        h1_result = dict(base_result)
        h1_result["sprt"] = {
            "enabled": True,
            "bounds": [0.0, 5.0],
            "output": {"llr": 3.0, "lower_bound": -2.94, "upper_bound": 2.94},
        }
        self.assertEqual(GATE.classify_sprt(h1_result, minimum_games=400)["classification"], "accept_h1")

        h0_result = dict(base_result)
        h0_result["sprt"] = {
            "enabled": True,
            "bounds": [0.0, 5.0],
            "output": {"llr": -3.0, "lower_bound": -2.94, "upper_bound": 2.94},
        }
        self.assertEqual(GATE.classify_sprt(h0_result, minimum_games=400)["classification"], "accept_h0")

    def test_short_sprt_is_wiring_smoke_not_promotion_evidence(self) -> None:
        result = {
            "match": {"games_completed": 2},
            "sprt": {
                "enabled": True,
                "bounds": [0.0, 5.0],
                "output": {"llr": 3.0, "lower_bound": -2.94, "upper_bound": 2.94},
            },
        }

        classification = GATE.classify_sprt(result, minimum_games=400)

        self.assertEqual(classification["classification"], "accept_h1")
        self.assertEqual(classification["evidence_kind"], "wiring_smoke")
        self.assertFalse(classification["promotes_strength"])

    def test_strength_evidence_does_not_block_smoke_gate_pass(self) -> None:
        result = {
            "commands": [{"name": "cmd", "required": True, "pass": True}],
            "bench": {"pass": True},
            "perft": [{"pass": True}],
            "match": {"pass": True},
            "sprt_classification": {
                "classification": "inconclusive",
                "evidence_kind": "wiring_smoke",
                "promotes_strength": False,
                "blocks_regression": False,
            },
        }

        checklist = GATE.build_promotion_checklist(result)
        hard_gate_items = [
            item for item in checklist if item["name"] not in GATE.STATISTICAL_CHECK_NAMES
        ]

        self.assertTrue(all(item["pass"] for item in hard_gate_items))
        self.assertFalse(next(item for item in checklist if item["name"] == "Strength promotion evidence")["pass"])
        self.assertFalse(next(item for item in checklist if item["name"] == "Regression evidence")["pass"])

    def test_accept_h0_is_not_strength_promotion_evidence(self) -> None:
        result = {
            "commands": [{"name": "cmd", "required": True, "pass": True}],
            "bench": {"pass": True},
            "perft": [{"pass": True}],
            "match": {"pass": True},
            "sprt_classification": {
                "classification": "accept_h0",
                "bounds_type": "gainer",
                "evidence_kind": "meaningful",
                "promotes_strength": False,
                "blocks_regression": False,
            },
        }

        checklist = GATE.build_promotion_checklist(result)

        self.assertFalse(next(item for item in checklist if item["name"] == "Strength promotion evidence")["pass"])
        self.assertFalse(next(item for item in checklist if item["name"] == "Regression evidence")["pass"])

    def test_meaningful_inconclusive_sprt_is_not_promotion_or_regression_evidence(self) -> None:
        result = {
            "match": {"games_completed": 500},
            "sprt": {
                "enabled": True,
                "bounds": [-3.0, 0.0],
                "output": {"llr": 0.0, "lower_bound": -2.94, "upper_bound": 2.94},
            },
        }

        classification = GATE.classify_sprt(result, minimum_games=400)

        self.assertEqual(classification["classification"], "inconclusive")
        self.assertEqual(classification["bounds_type"], "non_regression")
        self.assertEqual(classification["evidence_kind"], "meaningful")
        self.assertFalse(classification["promotes_strength"])
        self.assertFalse(classification["blocks_regression"])

    def test_accept_h1_gainer_is_strength_promotion_evidence(self) -> None:
        result = {
            "match": {"games_completed": 500},
            "sprt": {
                "enabled": True,
                "bounds": [0.0, 5.0],
                "output": {"llr": 3.0, "lower_bound": -2.94, "upper_bound": 2.94},
            },
        }

        classification = GATE.classify_sprt(result, minimum_games=400)

        self.assertEqual(classification["bounds_type"], "gainer")
        self.assertTrue(classification["promotes_strength"])
        self.assertFalse(classification["blocks_regression"])

    def test_accept_h1_non_regression_is_regression_evidence(self) -> None:
        result = {
            "match": {"games_completed": 500},
            "sprt": {
                "enabled": True,
                "bounds": [-3.0, 0.0],
                "output": {"llr": 3.0, "lower_bound": -2.94, "upper_bound": 2.94},
            },
        }

        classification = GATE.classify_sprt(result, minimum_games=400)

        self.assertEqual(classification["bounds_type"], "non_regression")
        self.assertFalse(classification["promotes_strength"])
        self.assertTrue(classification["blocks_regression"])

    def test_missing_match_fails_cutechess_gate_and_collects_failure(self) -> None:
        result = {
            "commands": [{"name": "cmd", "required": True, "pass": True}],
            "bench": {"pass": True},
            "perft": [{"pass": True}],
            "match": None,
            "sprt_classification": None,
        }

        checklist = GATE.build_promotion_checklist(result)

        self.assertFalse(next(item for item in checklist if item["name"] == "Cute Chess failure gate")["pass"])
        self.assertIn("missing match evidence", GATE.collect_failures(result))


if __name__ == "__main__":
    unittest.main()
