from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from localchessengine_trainer.lichess_fen_pool import clean_fen_row, write_fen_pool


class LichessFenPoolTest(unittest.TestCase):
    def test_clean_fen_row_normalizes_without_using_stockfish_label_as_target(self) -> None:
        fen, reason = clean_fen_row(
            {
                "fen": "8/8/8/8/8/8/4K3/4k3 w - -",
                "line": "e2e4",
                "depth": 30,
                "knodes": 10,
                "cp": 123,
                "mate": None,
            },
            min_source_depth=1,
            min_source_knodes=1,
            source_max_abs_cp=2500,
            skip_mates=True,
        )
        self.assertIsNone(reason)
        self.assertEqual(fen, "8/8/8/8/8/8/4K3/4k3 w - - 0 1")

    def test_write_fen_pool_deduplicates_and_writes_plain_fens(self) -> None:
        rows = [
            {"fen": "8/8/8/8/8/8/4K3/4k3 w - -", "depth": 30, "knodes": 10, "cp": 0},
            {"fen": "8/8/8/8/8/8/4K3/4k3 w - -", "depth": 30, "knodes": 10, "cp": 0},
            {"fen": "8/8/8/8/8/8/4K3/3k4 b - -", "depth": 30, "knodes": 10, "cp": 0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.jsonl"
            output = tmp_path / "pool.fens"
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            args = Namespace(
                input_jsonl=source,
                output=output,
                positions=2,
                format="fen",
                min_source_depth=1,
                min_source_knodes=1,
                source_max_abs_cp=2500,
                skip_mates=True,
                max_source_rows=0,
                sample_mod=1,
                sample_remainder=0,
            )
            stats = write_fen_pool(args)
            self.assertEqual(stats.accepted, 2)
            self.assertEqual(stats.duplicate_fen, 1)
            self.assertEqual(len(output.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
