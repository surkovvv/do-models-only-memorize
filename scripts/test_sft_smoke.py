#!/usr/bin/env python3
"""Regression tests for the smoke-training loop."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sft_smoke import (  # noqa: E402
    Example,
    exact_match_accuracy,
    log_predictions,
    select_smoke_examples,
    train,
)


def make_example(question: str, answer: str) -> Example:
    return Example(
        example_id="example",
        world_id="world",
        world_seed="seed",
        person_id="person",
        person_name="Person",
        fact_id="fact",
        relation_id="relation",
        fact_value=answer,
        operation_id="operation",
        template_family_id="family",
        template_id="template",
        rendered_question=question,
        canonical_answer=answer,
        answer_format_id="format",
        split="train",
    )


class SmokeTrainingTests(unittest.TestCase):
    def test_selects_a_reproducible_random_smoke_batch(self) -> None:
        examples = [make_example(f"q{index}", str(index)) for index in range(10)]

        first = select_smoke_examples(examples, batch_size=4, seed=42)
        second = select_smoke_examples(examples, batch_size=4, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(4, len(first))
        self.assertEqual(4, len(set(first)))
        self.assertNotEqual(examples[:4], first)

    def test_repeats_only_the_first_batch(self) -> None:
        first_batch = {"input_ids": torch.tensor([[1]])}
        second_batch = {"input_ids": torch.tensor([[2]])}
        model = Mock()
        optimizer = Mock()
        tokenizer = Mock()
        device = torch.device("cpu")
        smoke_examples = [make_example("question", "answer")]

        with (
            patch("scripts.sft_smoke.train_step", return_value=0.5) as train_step,
            patch(
                "scripts.sft_smoke.generate_answers",
                return_value=["answer"],
            ) as generate_answers,
        ):
            train(
                model=model,
                dataloader=[first_batch, second_batch],  # type: ignore[arg-type]
                optimizer=optimizer,
                device=device,
                epochs=3,
                smoke_steps=10,
                tokenizer=tokenizer,
                smoke_examples=smoke_examples,
                prediction_log_limit=None,
            )

        self.assertEqual(10, train_step.call_count)
        self.assertEqual(10, generate_answers.call_count)
        for call in train_step.call_args_list:
            self.assertIs(first_batch, call.args[1])
        for call in generate_answers.call_args_list:
            self.assertIs(smoke_examples, call.args[2])

    def test_rejects_an_empty_smoke_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty dataset"):
            train(
                model=Mock(),
                dataloader=[],  # type: ignore[arg-type]
                optimizer=Mock(),
                device=torch.device("cpu"),
                epochs=1,
                smoke_steps=1,
                tokenizer=Mock(),
                smoke_examples=[make_example("question", "answer")],
                prediction_log_limit=None,
            )

    def test_exact_match_strips_outer_whitespace(self) -> None:
        examples = [
            make_example("q1", "Paris"),
            make_example("q2", "Doctor"),
            make_example("q3", "Berlin"),
        ]

        self.assertEqual(
            2 / 3,
            exact_match_accuracy([" Paris ", "doctor", "Berlin"], examples),
        )

    def test_exact_match_rejects_misaligned_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            exact_match_accuracy(
                ["one"],
                [make_example("q1", "one"), make_example("q2", "two")],
            )

    def test_prediction_logging_can_be_limited(self) -> None:
        examples = [make_example("q1", "one"), make_example("q2", "two")]

        with patch("builtins.print") as print_line:
            log_predictions(["one", "wrong"], examples, limit=1)

        self.assertEqual(2, print_line.call_count)
        self.assertIn("prediction[00] match=true", print_line.call_args_list[0].args[0])
        self.assertEqual("predictions_omitted=1", print_line.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
