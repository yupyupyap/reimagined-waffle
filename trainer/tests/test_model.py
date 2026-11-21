from __future__ import annotations

import unittest

import torch

from localchessengine_trainer.model import (
    PerspectiveNnueModel,
    crelu,
    resolve_architecture,
    screlu,
)


class NnueModelTest(unittest.TestCase):
    def test_crelu_clamps(self) -> None:
        values = torch.tensor([-1.0, 0.5, 2.0])
        expected = torch.tensor([0.0, 0.5, 1.0])

        self.assertTrue(torch.allclose(crelu(values), expected))

    def test_screlu_clamps_and_squares(self) -> None:
        values = torch.tensor([-1.0, 0.5, 2.0])
        expected = torch.tensor([0.0, 0.25, 1.0])

        self.assertTrue(torch.allclose(screlu(values), expected))

    def test_small_and_v1_output_shapes(self) -> None:
        indices = torch.tensor([0, 1, 64, 65], dtype=torch.long)
        offsets = torch.tensor([0, 2], dtype=torch.long)
        for name in ("small", "v1"):
            model = PerspectiveNnueModel(name)
            observed = model(indices, indices, offsets)
            architecture = resolve_architecture(name)

            self.assertEqual(observed.shape, (2,))
            self.assertEqual(model.white.embedding_dim, architecture.hidden_size)
            self.assertEqual(model.black.embedding_dim, architecture.hidden_size)


if __name__ == "__main__":
    unittest.main()
