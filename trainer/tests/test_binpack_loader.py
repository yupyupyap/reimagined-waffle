from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy

from localchessengine_trainer.binpack import (
    BinpackDataset,
    BinpackError,
    read_metadata,
    validate_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DATASET = REPO_ROOT / "engine/data/nnue/task-027-smoke.binpack"
SMOKE_CHECKSUM = 0xD5D6C290BDD640DA


def _piece_count(fen: str) -> int:
    placement = fen.split(" ", 1)[0]
    return sum(1 for symbol in placement if symbol.isalpha())


class BinpackLoaderTest(unittest.TestCase):
    def test_metadata_reads_header_without_full_load(self) -> None:
        metadata = read_metadata(SMOKE_DATASET)

        self.assertEqual(metadata.format, "localchessengine.nnue.dataset.v1")
        self.assertEqual(metadata.version, 1)
        self.assertEqual(metadata.record_count, 8)
        self.assertEqual(metadata.checksum, SMOKE_CHECKSUM)
        self.assertEqual(metadata.checksum_hex, "0xd5d6c290bdd640da")

    def test_validate_streams_records_and_confirms_checksum(self) -> None:
        validation = validate_dataset(SMOKE_DATASET)

        self.assertEqual(validation.record_count, 8)
        self.assertEqual(validation.checksum, SMOKE_CHECKSUM)

    def test_batches_have_embedding_bag_shapes_and_dtypes(self) -> None:
        dataset = BinpackDataset(SMOKE_DATASET)
        batch = next(dataset.iter_batches(batch_size=4))

        self.assertEqual(batch.positions, 4)
        self.assertEqual(batch.white_indices.dtype, numpy.int64)
        self.assertEqual(batch.black_indices.dtype, numpy.int64)
        self.assertEqual(batch.offsets.dtype, numpy.int64)
        self.assertEqual(batch.side_to_move.dtype, numpy.int64)
        self.assertEqual(batch.side_to_move.shape, (4,))
        self.assertEqual(batch.eval_cp.dtype, numpy.float32)
        self.assertEqual(batch.wdl.dtype, numpy.float32)
        self.assertEqual(batch.eval_cp.tolist(), [44.0, -14.0, 20.0, -37.0])
        self.assertEqual(batch.wdl.tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(batch.offsets[0], 0)

        first_fen = (
            "rnbqkb1r/1pp2ppp/p2pp2n/8/P7/5R1P/1PPPPPP1/1NBQKBNR w Kkq - 0 5"
        )
        first_piece_count = _piece_count(first_fen)
        self.assertEqual(batch.offsets[1], first_piece_count)

        black_rook_white_perspective = (6 + 3) * 64 + 56
        black_rook_black_perspective = 3 * 64 + 0
        self.assertEqual(batch.white_indices[0], black_rook_white_perspective)
        self.assertEqual(batch.black_indices[0], black_rook_black_perspective)

    def test_iteration_covers_all_records(self) -> None:
        dataset = BinpackDataset(SMOKE_DATASET)
        batches = list(dataset.iter_batches(batch_size=3))

        self.assertEqual([batch.positions for batch in batches], [3, 3, 2])
        self.assertEqual(sum(batch.positions for batch in batches), 8)
        self.assertEqual(
            sum(batch.white_indices.shape[0] for batch in batches),
            sum(batch.black_indices.shape[0] for batch in batches),
        )

    def test_rejects_bad_magic_truncation_and_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            valid_copy = Path(tmp) / "valid.binpack"
            bad_magic = Path(tmp) / "bad-magic.binpack"
            truncated = Path(tmp) / "truncated.binpack"
            bad_checksum = Path(tmp) / "bad-checksum.binpack"
            shutil.copyfile(SMOKE_DATASET, valid_copy)
            data = bytearray(valid_copy.read_bytes())

            bad_magic.write_bytes(bytes([ord("X")]) + bytes(data[1:]))
            truncated.write_bytes(bytes(data[:10]))

            checksum_offset = 16 + 4 + 8
            data[checksum_offset] ^= 0xFF
            bad_checksum.write_bytes(bytes(data))

            with self.assertRaisesRegex(BinpackError, "magic"):
                read_metadata(bad_magic)
            with self.assertRaisesRegex(BinpackError, "truncated"):
                validate_dataset(truncated)
            with self.assertRaisesRegex(BinpackError, "checksum"):
                validate_dataset(bad_checksum)


if __name__ == "__main__":
    unittest.main()
