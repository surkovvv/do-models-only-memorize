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
        self.assertEqual(64, config.evaluation.batch_size)
        self.assertEqual(10, config.training.smoke_steps)
        self.assertIsNone(config.runtime.prediction_log_limit)
        self.assertTrue(config.runtime.save_model)
        self.assertFalse(config.tracking.enabled)
        self.assertEqual(1, config.tracking.log_interval_steps)

    def test_applies_typed_command_line_overrides(self) -> None:
        config = load_experiment_config(
            ROOT / "configs/sft_smoke.yaml",
            [
                "training.batch_size=8",
                "evaluation.batch_size=32",
                "runtime.debug=true",
                "runtime.save_model=false",
            ],
        )

        self.assertEqual(8, config.training.batch_size)
        self.assertEqual(32, config.evaluation.batch_size)
        self.assertTrue(config.runtime.debug)
        self.assertFalse(config.runtime.save_model)

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

        with self.assertRaisesRegex(ValueError, "log_interval_steps must be positive"):
            load_experiment_config(
                ROOT / "configs/sft_smoke.yaml",
                ["tracking.log_interval_steps=0"],
            )

        with self.assertRaisesRegex(ValueError, "evaluation.batch_size must be positive"):
            load_experiment_config(
                ROOT / "configs/sft_smoke.yaml",
                ["evaluation.batch_size=0"],
            )


if __name__ == "__main__":
    unittest.main()
