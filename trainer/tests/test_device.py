from __future__ import annotations

import unittest

from localchessengine_trainer.device import select_device


class DeviceSelectionTest(unittest.TestCase):
    def test_cpu_can_be_selected_explicitly(self) -> None:
        selection = select_device("cpu")
        self.assertEqual(selection.device_type, "cpu")
        self.assertFalse(selection.fallback_used)
        self.assertTrue(selection.torch_version)

    def test_auto_selects_a_supported_device(self) -> None:
        selection = select_device("auto")
        self.assertIn(selection.device_type, {"cpu", "mps"})
        if selection.device_type == "cpu":
            self.assertTrue(selection.reason)


if __name__ == "__main__":
    unittest.main()
