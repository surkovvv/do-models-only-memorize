#!/usr/bin/env python3
"""Tests for typed experiment configuration."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from omegaconf.errors import ConfigKeyError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_config import ExperimentConfig, load_experiment_config  # noqa: E402


class ExperimentConfigTests(unittest.TestCase):
    def test_loads_repository_config_as_dataclasses(self) -> None:
        config = load_experiment_config(ROOT / "configs/sft_smoke.yaml")

        self.assertIsInstance(config, ExperimentConfig)
        self.assertEqual(".models/Qwen3-0.6B", config.model.name_or_path)
        self.assertEqual(16, config.training.batch_size)
        self.assertEqual(10, config.training.smoke_steps)
        self.assertIsNone(config.runtime.prediction_log_limit)
        self.assertFalse(config.tracking.enabled)

    def test_applies_typed_command_line_overrides(self) -> None:
        config = load_experiment_config(
            ROOT / "configs/sft_smoke.yaml",
            ["training.batch_size=8", "runtime.debug=true"],
        )

        self.assertEqual(8, config.training.batch_size)
        self.assertTrue(config.runtime.debug)

    def test_rejects_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "invalid.yaml"
            path.write_text("unknown_option: true\n", encoding="utf-8")

            with self.assertRaises(ConfigKeyError):
                load_experiment_config(path)

    def test_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            load_experiment_config(
                ROOT / "configs/sft_smoke.yaml",
                ["training.batch_size=0"],
            )

        with self.assertRaisesRegex(ValueError, "smoke_steps must be positive or null"):
            load_experiment_config(
                ROOT / "configs/sft_smoke.yaml",
                ["training.smoke_steps=0"],
            )


if __name__ == "__main__":
    unittest.main()
