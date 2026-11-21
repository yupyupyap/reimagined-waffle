from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from localchessengine_trainer.model import PerspectiveNnueModel
from localchessengine_trainer.serialize import (
    ExportConfig,
    export_network,
    read_network_info,
)
from localchessengine_trainer.train import CHECKPOINT_SCHEMA, model_state_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DATASET = REPO_ROOT / "engine/data/nnue/task-027-smoke.binpack"


def _checkpoint_payload(model: PerspectiveNnueModel) -> dict[str, object]:
    state = model.state_dict()
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task": "TASK-030",
        "architecture": model.architecture.as_dict(),
        "dataset": {
            "dataset_id": "task-027-smoke",
            "positions": 8,
            "checksum": "0xd5d6c290bdd640da",
        },
        "epoch": 1,
        "epochs": 1,
        "seed": 31,
        "lambda_eval": 0.7,
        "eval_scale": 400.0,
        "metrics": {"loss": 0.1},
        "model_state_sha256": model_state_sha256(state),
        "model_state": state,
    }


def _write_checkpoint(path: Path, payload: dict[str, object]) -> None:
    torch.save(payload, path)


class NnueSerializeTest(unittest.TestCase):
    def test_export_roundtrip_reports_quantization_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.manual_seed(31)
            model = PerspectiveNnueModel("small")
            checkpoint = root / "checkpoint.pt"
            network = root / "network.nnue"
            log = root / "export.jsonl"
            _write_checkpoint(checkpoint, _checkpoint_payload(model))

            result = export_network(
                ExportConfig(
                    checkpoint_path=checkpoint,
                    output_path=network,
                    dataset_path=SMOKE_DATASET,
                    batch_size=4,
                    log_path=log,
                )
            )
            info = read_network_info(network)

            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["network_id"], info["network_id"])
            self.assertEqual(info["schema"], "localchessengine.nnue.network.v2")
            self.assertEqual(result["format_version"], 2)
            self.assertEqual(info["architecture"], "small")
            self.assertTrue(result["feature_bias"])
            self.assertTrue(result["side_to_move_ordering"])
            self.assertEqual(info["qa"], 255)
            self.assertEqual(info["qb"], 64)
            self.assertGreater(info["tensor_bytes"], 0)
            self.assertEqual(info["metadata"]["dataset"]["dataset_id"], "task-027-smoke")
            self.assertNotIn("network_id", info["metadata"])
            self.assertEqual(result["quant_error_cp"]["positions"], 8)
            self.assertLess(result["quant_error_cp"]["max_abs_cp"], 5.0)
            self.assertIn("nnue.validation.result", log.read_text(encoding="utf-8"))

    def test_rejects_bad_checkpoint_digest_shape_and_saturation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = PerspectiveNnueModel("small")
            payload = _checkpoint_payload(model)

            bad_digest = dict(payload)
            bad_digest["model_state_sha256"] = "bad"
            bad_digest_path = root / "bad-digest.pt"
            _write_checkpoint(bad_digest_path, bad_digest)
            with self.assertRaisesRegex(ValueError, "model_state_sha256"):
                export_network(
                    ExportConfig(
                        checkpoint_path=bad_digest_path,
                        output_path=root / "bad-digest.nnue",
                        dataset_path=SMOKE_DATASET,
                    )
                )

            bad_shape = dict(payload)
            bad_shape_state = dict(payload["model_state"])  # type: ignore[arg-type]
            bad_shape_state["output.weight"] = torch.zeros((1, 3), dtype=torch.float32)
            bad_shape["model_state"] = bad_shape_state
            bad_shape["model_state_sha256"] = model_state_sha256(bad_shape_state)
            bad_shape_path = root / "bad-shape.pt"
            _write_checkpoint(bad_shape_path, bad_shape)
            with self.assertRaisesRegex(ValueError, "shape"):
                export_network(
                    ExportConfig(
                        checkpoint_path=bad_shape_path,
                        output_path=root / "bad-shape.nnue",
                        dataset_path=SMOKE_DATASET,
                    )
                )

            saturated = dict(payload)
            saturated_state = {
                key: value.detach().clone()
                for key, value in payload["model_state"].items()  # type: ignore[union-attr]
            }
            saturated_state["white.weight"].fill_(200.0)
            saturated["model_state"] = saturated_state
            saturated["model_state_sha256"] = model_state_sha256(saturated_state)
            saturated_path = root / "saturated.pt"
            _write_checkpoint(saturated_path, saturated)
            with self.assertRaisesRegex(ValueError, "saturate"):
                export_network(
                    ExportConfig(
                        checkpoint_path=saturated_path,
                        output_path=root / "saturated.nnue",
                        dataset_path=SMOKE_DATASET,
                    )
                )

    def test_network_reader_rejects_magic_truncation_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = PerspectiveNnueModel("small")
            checkpoint = root / "checkpoint.pt"
            network = root / "network.nnue"
            _write_checkpoint(checkpoint, _checkpoint_payload(model))
            export_network(
                ExportConfig(
                    checkpoint_path=checkpoint,
                    output_path=network,
                    dataset_path=SMOKE_DATASET,
                )
            )
            payload = bytearray(network.read_bytes())

            bad_magic = root / "bad-magic.nnue"
            bad_magic.write_bytes(bytes([ord("X")]) + bytes(payload[1:]))
            with self.assertRaisesRegex(ValueError, "magic"):
                read_network_info(bad_magic)

            truncated = root / "truncated.nnue"
            truncated.write_bytes(bytes(payload[:10]))
            with self.assertRaisesRegex(ValueError, "truncated"):
                read_network_info(truncated)

            bad_checksum = root / "bad-checksum.nnue"
            payload[-1] ^= 0xFF
            bad_checksum.write_bytes(bytes(payload))
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_network_info(bad_checksum)


if __name__ == "__main__":
    unittest.main()
