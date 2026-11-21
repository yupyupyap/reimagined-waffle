from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from localchessengine_trainer.model import PerspectiveNnueModel
from localchessengine_trainer.train import (
    CHECKPOINT_SCHEMA,
    TrainingConfig,
    interpolated_loss,
    load_dataset_metadata,
    model_state_sha256,
    parse_args,
    run_training,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DATASET = REPO_ROOT / "engine/data/nnue/task-027-smoke.binpack"
SMOKE_MANIFEST = REPO_ROOT / "engine/data/nnue/task-027-smoke-manifest.json"


class NnueTrainTest(unittest.TestCase):
    def test_interpolated_loss_honors_lambda(self) -> None:
        prediction = torch.tensor([0.0, 100.0])
        target_eval = torch.tensor([200.0, -100.0])
        target_wdl = torch.tensor([1.0, 0.0])

        eval_only = interpolated_loss(
            prediction,
            target_eval,
            target_wdl,
            lambda_eval=1.0,
        )
        wdl_only = interpolated_loss(
            prediction,
            target_eval,
            target_wdl,
            lambda_eval=0.0,
        )
        mixed = interpolated_loss(
            prediction,
            target_eval,
            target_wdl,
            lambda_eval=0.25,
        )

        expected = (0.25 * eval_only.eval_loss) + (0.75 * wdl_only.wdl_loss)
        self.assertTrue(torch.allclose(eval_only.loss, eval_only.eval_loss))
        self.assertTrue(torch.allclose(wdl_only.loss, wdl_only.wdl_loss))
        self.assertTrue(torch.allclose(mixed.loss, expected))

    def test_model_state_digest_is_stable(self) -> None:
        torch.manual_seed(123)
        model = PerspectiveNnueModel("small")

        self.assertEqual(model_state_sha256(model.state_dict()), model_state_sha256(model.state_dict()))

    def test_parse_args_maps_cli_to_config(self) -> None:
        config = parse_args(
            [
                "--dataset",
                str(SMOKE_DATASET),
                "--manifest",
                str(SMOKE_MANIFEST),
                "--architecture",
                "v1",
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--batch-size",
                "4",
            ]
        )

        self.assertEqual(config.architecture, "v1")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.epochs, 1)
        self.assertEqual(config.batch_size, 4)

    def test_cpu_training_writes_metrics_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_training(
                TrainingConfig(
                    dataset_path=SMOKE_DATASET,
                    manifest_path=SMOKE_MANIFEST,
                    architecture="small",
                    device="cpu",
                    epochs=2,
                    batch_size=4,
                    learning_rate=0.01,
                    checkpoint_dir=root / "checkpoints",
                    log_path=root / "train.jsonl",
                    seed=7,
                )
            )

            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["architecture"], "small")
            self.assertLess(result["final_loss"], result["initial_loss"])
            self.assertTrue(Path(str(result["checkpoint"])).exists())

            records = [
                json.loads(line)
                for line in (root / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            events = {record["event"] for record in records}
            self.assertIn("trainer.epoch.metrics", events)
            self.assertIn("trainer.training.result", events)

            checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["schema"], CHECKPOINT_SCHEMA)
            self.assertEqual(checkpoint["architecture"]["name"], "small")
            self.assertEqual(checkpoint["model_state_sha256"], result["model_state_sha256"])

    def test_external_label_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.loads(SMOKE_MANIFEST.read_text(encoding="utf-8"))
            manifest["external_data"] = True
            path = Path(tmp) / "external-manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "external NNUE label datasets"):
                load_dataset_metadata(SMOKE_DATASET, path)


if __name__ == "__main__":
    unittest.main()
