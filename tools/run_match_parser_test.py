#!/usr/bin/env python3
"""Parser regression tests for tools/run_match.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from argparse import Namespace
from pathlib import Path


def load_run_match_module():
    module_path = Path(__file__).with_name("run_match.py")
    spec = importlib.util.spec_from_file_location("run_match", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUN_MATCH = load_run_match_module()


class RunMatchParserTest(unittest.TestCase):
    def test_clean_cutechess_output_has_no_failures(self) -> None:
        log_text = "\n".join(
            [
                "Started game 1 of 2 (LCE-Test vs LCE-Base)",
                "Finished game 1 (LCE-Test vs LCE-Base): 1/2-1/2 {Draw by adjudication}",
                "Score of LCE-Test vs LCE-Base: 0 - 0 - 1  [0.500] 1",
            ]
        )

        self.assertEqual(RUN_MATCH.detect_failures(log_text), [])

    def test_detects_cutechess_verbose_failure_results(self) -> None:
        cases = {
            "Finished game 1 (A vs B): 0-1 {Black loses on time}": "time forfeit",
            "Finished game 1 (A vs B): 1/2-1/2 {Draw by timeout}": "timeout",
            "Finished game 1 (A vs B): 1-0 {Black makes an illegal move}": "illegal move",
            "Finished game 1 (A vs B): 1-0 {Black disconnects}": "disconnect",
            "Finished game 1 (A vs B): 1-0 {Black's connection stalls}": "stalled connection",
            "Finished game 1 (A vs B): * {No result}": "unterminated game",
            "A forfeits by invalid result claim: 1-0 {Invalid result claim}": "invalid result claim",
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertIn(expected, RUN_MATCH.detect_failures(text))

    def test_detects_pgn_termination_failures(self) -> None:
        pgn_text = "\n".join(
            [
                '[Event "LocalChessEngine smoke"]',
                '[Termination "time forfeit"]',
                "",
                '[Event "LocalChessEngine smoke"]',
                '[Termination "stalled connection"]',
                "",
                '[Event "LocalChessEngine smoke"]',
                '[Termination "abandoned"]',
                "",
                '[Event "LocalChessEngine smoke"]',
                '[Termination "illegal move"]',
            ]
        )

        failures = RUN_MATCH.detect_failures("", pgn_text)

        self.assertIn("time forfeit", failures)
        self.assertIn("stalled connection", failures)
        self.assertIn("disconnect", failures)
        self.assertIn("illegal move", failures)

    def test_parses_score_elo_and_sprt_lines(self) -> None:
        log_text = "\n".join(
            [
                "Score of LCE-Test vs LCE-Base: 1 - 0 - 1  [0.750] 2",
                "Elo difference: 190.8 +/- 580.4, LOS: 75.0 %, DrawRatio: 50.0 %",
                "SPRT: llr -0.00127 (-0.0%), lbound -2.94, ubound 2.94",
            ]
        )

        score = RUN_MATCH.parse_score(log_text)
        elo = RUN_MATCH.parse_elo(log_text)
        sprt = RUN_MATCH.parse_sprt(log_text)

        self.assertEqual(score["wins"], 1)
        self.assertEqual(score["losses"], 0)
        self.assertEqual(score["draws"], 1)
        self.assertEqual(score["games_completed"], 2)
        self.assertEqual(elo["difference"], 190.8)
        self.assertEqual(elo["draw_ratio_percent"], 50.0)
        self.assertEqual(sprt["llr"], -0.00127)
        self.assertEqual(sprt["lower_bound"], -2.94)
        self.assertEqual(sprt["upper_bound"], 2.94)
        self.assertFalse(RUN_MATCH.sprt_has_decision(sprt))

    def test_sprt_decision_allows_pending_concurrent_games(self) -> None:
        sprt = RUN_MATCH.parse_sprt("SPRT: llr -3.1 (-105.4%), lbound -2.94, ubound 2.94")

        self.assertTrue(RUN_MATCH.sprt_has_decision(sprt))

    def test_build_command_adds_per_engine_and_shared_options(self) -> None:
        args = Namespace(
            cutechess=Path("cutechess-cli"),
            base_engine=Path("base"),
            test_engine=Path("test"),
            base_name="Base",
            test_name="Test",
            time_control=None,
            games=None,
            rounds=None,
            concurrency=1,
            timemargin_ms=250,
            hash_mb=None,
            threads=None,
            base_option=["Null Move Pruning=false"],
            test_option=["Null Move Pruning=true"],
            both_option=["Hash=8", "SEE Pruning=false"],
            maxmoves=None,
            seed=123,
            openings=None,
            opening_format=None,
            opening_plies=None,
            opening_order="sequential",
            opening_policy="default",
            repeat=None,
            preset="smoke",
            debug=False,
        )

        command = RUN_MATCH.build_command(
            args, RUN_MATCH.PRESETS["smoke"], Path("match.pgn")
        )

        self.assertIn("option.Null Move Pruning=true", command)
        self.assertIn("option.Null Move Pruning=false", command)
        self.assertEqual(command.count("option.Hash=8"), 2)
        self.assertEqual(command.count("option.SEE Pruning=false"), 2)

    def test_build_command_adds_openings_and_repeat(self) -> None:
        args = Namespace(
            cutechess=Path("cutechess-cli"),
            base_engine=Path("base"),
            test_engine=Path("test"),
            base_name="Base",
            test_name="Test",
            time_control=None,
            games=None,
            rounds=None,
            concurrency=1,
            timemargin_ms=250,
            hash_mb=None,
            threads=None,
            base_option=[],
            test_option=[],
            both_option=[],
            maxmoves=None,
            seed=123,
            openings=Path("local-openings.epd"),
            opening_format=None,
            opening_plies=8,
            opening_order="random",
            opening_policy="round",
            repeat=2,
            preset="sprt-small-nnue",
            debug=False,
        )

        command = RUN_MATCH.build_command(
            args, RUN_MATCH.PRESETS["sprt-small-nnue"], Path("match.pgn")
        )

        self.assertIn("-openings", command)
        self.assertIn("file=local-openings.epd", command)
        self.assertIn("format=epd", command)
        self.assertIn("plies=8", command)
        self.assertIn("order=random", command)
        self.assertIn("policy=round", command)
        self.assertIn("-repeat", command)


if __name__ == "__main__":
    unittest.main()
