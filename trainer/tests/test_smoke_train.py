from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from localchessengine_trainer.smoke_train import SmokeNnueModel, SmokeTrainConfig, run_smoke_training


class SmokeTrainTest(unittest.TestCase):
    def test_screlu_clamps_and_squares(self) -> None:
        values = torch.tensor([-1.0, 0.5, 2.0])
        observed = SmokeNnueModel.screlu(values)
        expected = torch.tensor([0.0, 0.25, 1.0])
        self.assertTrue(torch.allclose(observed, expected))

    def test_cpu_smoke_train_writes_required_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "smoke.jsonl"
            result = run_smoke_training(
                SmokeTrainConfig(
                    device="cpu",
                    log_path=log_path,
                    epochs=1,
                    batch_size=8,
                    features_per_position=4,
                    hidden_size=4,
                    seed=1,
                )
            )

            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["device"], "cpu")
            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            events = {record["event"] for record in records}
            self.assertIn("trainer.dependency.check", events)
            self.assertIn("trainer.epoch.metrics", events)
            self.assertIn("trainer.smoke_train.result", events)
            dependency = next(record for record in records if record["event"] == "trainer.dependency.check")
            self.assertEqual(dependency["dependency"], "torch")
            self.assertEqual(dependency["dependency_license"], "BSD-3-Clause")
            self.assertTrue(dependency["numpy_version"])
            self.assertEqual(dependency["numpy_license"], "BSD-3-Clause")


if __name__ == "__main__":
    unittest.main()
