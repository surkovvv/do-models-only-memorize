#!/usr/bin/env python3
"""Regression tests for the smoke-training loop."""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import Mock, patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sft_smoke import (  # noqa: E402
    EvaluationLoggers,
    Example,
    StepMetrics,
    TrainingResult,
    cosine_learning_rate_multiplier,
    evaluate_splits,
    exact_match_accuracy,
    generate_answers_batched,
    group_evaluation_examples,
    log_predictions,
    make_metric_logger,
    report_final_metrics,
    report_metric_to_clearml,
    resolve_warmup_steps,
    save_final_evaluation_summary,
    select_smoke_examples,
    train,
)


def make_example(question: str, answer: str, *, split: str = "train") -> Example:
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
        split=split,
    )


class SmokeTrainingTests(unittest.TestCase):
    def test_groups_the_combined_test_file_by_scientific_slice(self) -> None:
        examples = [
            make_example(split, "answer", split=split)
            for split in (
                "eval_exact_recall",
                "eval_seen_family_new_template",
                "eval_heldout_family",
            )
        ]

        groups = group_evaluation_examples(examples)

        self.assertEqual(
            [
                "eval_exact_recall",
                "eval_seen_family_new_template",
                "eval_heldout_family",
            ],
            list(groups),
        )
        self.assertTrue(all(len(split_examples) == 1 for split_examples in groups.values()))

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
            patch(
                "scripts.sft_smoke.train_step",
                return_value=StepMetrics(
                    loss=0.5,
                    token_accuracy=0.75,
                    target_tokens=4,
                    learning_rate=2e-5,
                ),
            ) as train_step,
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

    def test_smoke_logs_first_interval_and_final_steps(self) -> None:
        batch = {"input_ids": torch.tensor([[1]])}
        smoke_examples = [make_example("question", "answer")]
        step_metrics = StepMetrics(0.5, 0.75, 4, 2e-5)

        with (
            patch("scripts.sft_smoke.train_step", return_value=step_metrics),
            patch("scripts.sft_smoke.generate_answers", return_value=["answer"]) as generate,
        ):
            result = train(
                model=Mock(),
                dataloader=[batch],  # type: ignore[arg-type]
                optimizer=Mock(),
                device=torch.device("cpu"),
                epochs=1,
                smoke_steps=10,
                tokenizer=Mock(),
                smoke_examples=smoke_examples,
                prediction_log_limit=1,
                log_interval_steps=4,
            )

        self.assertEqual([1, 4, 8, 10], [metric["step"] for metric in result.metrics])
        self.assertEqual(4, generate.call_count)
        self.assertEqual(0.75, result.metrics[-1]["token_accuracy"])

    def test_full_training_evaluates_before_training_and_after_each_epoch(self) -> None:
        batch = {"input_ids": torch.tensor([[1]])}
        step_metrics = StepMetrics(0.5, 0.75, 4, 2e-5)
        evaluation_splits = {
            split: [make_example(f"{split} question", "answer", split=split)]
            for split in (
                "eval_exact_recall",
                "eval_seen_family_new_template",
                "eval_heldout_family",
            )
        }

        with (
            patch("scripts.sft_smoke.train_step", return_value=step_metrics),
            patch(
                "scripts.sft_smoke.generate_answers",
                side_effect=lambda _model, _tokenizer, examples, _device, **_kwargs: [
                    example.canonical_answer for example in examples
                ],
            ) as generate,
        ):
            result = train(
                model=Mock(),
                dataloader=[batch],  # type: ignore[arg-type]
                optimizer=Mock(),
                device=torch.device("cpu"),
                epochs=2,
                smoke_steps=None,
                tokenizer=Mock(),
                smoke_examples=None,
                prediction_log_limit=None,
                evaluation_splits=evaluation_splits,
                evaluation_batch_size=3,
            )

        eval_metrics = [metric for metric in result.metrics if metric["mode"] == "eval"]
        self.assertEqual(
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [metric["epoch"] for metric in eval_metrics],
        )
        self.assertEqual(
            [
                "eval_exact_recall",
                "eval_seen_family_new_template",
                "eval_heldout_family",
            ]
            * 3,
            [metric["split"] for metric in eval_metrics],
        )
        self.assertTrue(all(metric["exact_match_accuracy"] == 1.0 for metric in eval_metrics))
        self.assertEqual(9, generate.call_count)
        self.assertIsNotNone(result.final_evaluation_summary)
        assert result.final_evaluation_summary is not None
        self.assertEqual(
            {
                "eval_exact_recall": 1.0,
                "eval_seen_family_new_template": 1.0,
                "eval_heldout_family": 1.0,
            },
            result.final_evaluation_summary["exact_match_accuracy"],
        )

    def test_full_training_evaluates_baseline_before_accumulated_optimizer_steps(self) -> None:
        batches = [{"input_ids": torch.tensor([[index]])} for index in range(5)]
        evaluation_splits = {
            split: [make_example(f"{split} question", "answer", split=split)]
            for split in (
                "eval_exact_recall",
                "eval_seen_family_new_template",
                "eval_heldout_family",
            )
        }
        events: list[str] = []

        def generated_answers(_model, _tokenizer, examples, _device, **_kwargs):
            events.append("eval")
            return [example.canonical_answer for example in examples]

        def trained_step(*_args, **_kwargs):
            events.append("train")
            return StepMetrics(0.5, 0.75, 4, 5e-5)

        with (
            patch("scripts.sft_smoke.train_step", side_effect=trained_step) as train_step,
            patch("scripts.sft_smoke.generate_answers", side_effect=generated_answers),
        ):
            result = train(
                model=Mock(),
                dataloader=batches,  # type: ignore[arg-type]
                optimizer=Mock(),
                device=torch.device("cpu"),
                epochs=1,
                smoke_steps=None,
                tokenizer=Mock(),
                smoke_examples=None,
                prediction_log_limit=None,
                evaluation_splits=evaluation_splits,
                evaluation_batch_size=8,
                gradient_accumulation_steps=2,
            )

        self.assertEqual(["eval", "eval", "eval", "train"], events[:4])
        self.assertEqual(5, train_step.call_count)
        self.assertEqual(
            [2, 2, 2, 2, 1],
            [call.kwargs["gradient_accumulation_divisor"] for call in train_step.call_args_list],
        )
        self.assertEqual(
            [True, False, True, False, True],
            [call.kwargs["zero_grad"] for call in train_step.call_args_list],
        )
        self.assertEqual(
            [False, True, False, True, True],
            [call.kwargs["perform_optimizer_step"] for call in train_step.call_args_list],
        )
        train_metrics = [metric for metric in result.metrics if metric["mode"] == "train"]
        self.assertEqual([1, 2, 3], [metric["step"] for metric in train_metrics])
        final_eval_metrics = [
            metric
            for metric in result.metrics
            if metric["mode"] == "eval" and metric["epoch"] == 1
        ]
        self.assertTrue(all(metric["step"] == 3 for metric in final_eval_metrics))

    def test_point_six_billion_schedule_has_twenty_warmup_steps_and_ten_percent_floor(
        self,
    ) -> None:
        total_steps = 282
        warmup_steps = resolve_warmup_steps(total_steps, 0.05, 20)

        self.assertEqual(20, warmup_steps)
        self.assertEqual(
            0.05,
            cosine_learning_rate_multiplier(
                0,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                final_learning_rate_ratio=0.1,
            ),
        )
        self.assertEqual(
            1.0,
            cosine_learning_rate_multiplier(
                19,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                final_learning_rate_ratio=0.1,
            ),
        )
        self.assertAlmostEqual(
            0.1,
            cosine_learning_rate_multiplier(
                total_steps - 1,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                final_learning_rate_ratio=0.1,
            ),
        )

    def test_generated_eval_uses_bounded_batches(self) -> None:
        examples = [make_example(f"q{index}", str(index)) for index in range(5)]

        with patch(
            "scripts.sft_smoke.generate_answers",
            side_effect=lambda _model, _tokenizer, batch, _device, **_kwargs: [
                example.canonical_answer for example in batch
            ],
        ) as generate:
            predictions = generate_answers_batched(
                Mock(),
                Mock(),
                examples,
                torch.device("cpu"),
                batch_size=2,
            )

        self.assertEqual(["0", "1", "2", "3", "4"], predictions)
        self.assertEqual([2, 2, 1], [len(call.args[2]) for call in generate.call_args_list])

    def test_eval_persists_full_traces_aggregates_and_paired_gaps(self) -> None:
        exact = replace(
            make_example("exact question", "answer", split="eval_exact_recall"),
            template_family_id="direct_question",
            template_id="direct_001",
        )
        seen = replace(
            exact,
            example_id="seen_example",
            split="eval_seen_family_new_template",
            rendered_question="seen-family question",
            template_id="direct_002",
        )
        heldout = replace(
            exact,
            example_id="heldout_example",
            split="eval_heldout_family",
            rendered_question="heldout-family question",
            template_family_id="nominal_attribute",
            template_id="nominal_001",
        )
        splits = {
            "eval_exact_recall": [exact],
            "eval_seen_family_new_template": [seen],
            "eval_heldout_family": [heldout],
        }
        traces: list[Mapping[str, Any]] = []
        aggregates: list[Mapping[str, Any]] = []
        pairs: list[Mapping[str, Any]] = []
        pair_aggregates: list[Mapping[str, Any]] = []
        summaries: list[Mapping[str, Any]] = []
        answers = {
            "eval_exact_recall": " answer\n",
            "eval_seen_family_new_template": "wrong",
            "eval_heldout_family": "answer",
        }

        with (
            patch(
                "scripts.sft_smoke.generate_answers",
                side_effect=lambda _model, _tokenizer, examples, _device, **_kwargs: [
                    answers[example.split] for example in examples
                ],
            ),
            patch("builtins.print") as print_line,
        ):
            result = evaluate_splits(
                Mock(),
                Mock(),
                splits,
                torch.device("cpu"),
                batch_size=1,
                epoch=2,
                step=17,
                loggers=EvaluationLoggers(
                    trace=traces.extend,
                    aggregate=aggregates.extend,
                    pair=pairs.extend,
                    pair_aggregate=pair_aggregates.extend,
                    summary=summaries.extend,
                ),
            )

        self.assertEqual(3, len(traces))
        self.assertEqual(" answer\n", traces[0]["prediction_raw"])
        self.assertEqual("answer", traces[0]["prediction_normalized"])
        self.assertEqual("exact question", traces[0]["rendered_question"])
        self.assertEqual("direct_001", traces[0]["template_id"])
        self.assertTrue(traces[0]["exact_match"])
        self.assertTrue(
            any(
                row["group_by"] == "template_id"
                and row["group_value"] == "direct_002"
                and row["exact_match_accuracy"] == 0.0
                for row in aggregates
            )
        )
        exact_to_seen = next(
            row
            for row in pairs
            if row["comparison"] == "exact_recall_minus_seen_family_new_template"
        )
        self.assertEqual("left_only", exact_to_seen["transition"])
        self.assertEqual(1, exact_to_seen["correctness_delta"])
        overall_gap = next(
            row
            for row in pair_aggregates
            if row["comparison"] == "exact_recall_minus_seen_family_new_template"
            and row["group_by"] == "overall"
        )
        self.assertEqual(1.0, overall_gap["exact_match_gap"])
        self.assertEqual(
            {
                "eval_exact_recall": 1.0,
                "eval_seen_family_new_template": 0.0,
                "eval_heldout_family": 1.0,
            },
            result.summary["exact_match_accuracy"],
        )
        self.assertEqual([result.summary], summaries)
        self.assertTrue(
            any(
                call.args[0].startswith("eval_gap epoch=2 step=17")
                for call in print_line.call_args_list
            )
        )

    def test_eval_series_include_the_split_name(self) -> None:
        logger = Mock()

        report_metric_to_clearml(
            logger,
            {
                "mode": "eval",
                "split": "test",
                "epoch": 0,
                "step": 0,
                "exact_match_accuracy": 0.25,
                "examples": 4,
            },
        )

        logger.report_scalar.assert_called_once_with(
            title="accuracy",
            series="eval/test/answer_exact_match",
            value=0.25,
            iteration=0,
        )

    def test_reports_structured_scalars_to_clearml(self) -> None:
        logger = Mock()

        report_metric_to_clearml(
            logger,
            {
                "mode": "smoke",
                "step": 3,
                "loss": 0.25,
                "token_accuracy": 0.75,
                "exact_match_accuracy": 0.5,
                "learning_rate": 2e-5,
                "examples_seen": 12,
            },
        )

        self.assertEqual(5, logger.report_scalar.call_count)
        logger.report_scalar.assert_any_call(title="loss", series="smoke", value=0.25, iteration=3)
        logger.report_scalar.assert_any_call(
            title="accuracy", series="smoke/token", value=0.75, iteration=3
        )
        logger.report_scalar.assert_any_call(
            title="accuracy", series="smoke/answer_exact_match", value=0.5, iteration=3
        )

    def test_metric_logger_fans_out_to_jsonl_and_clearml(self) -> None:
        clearml_logger = Mock()
        with patch("scripts.sft_smoke.append_jsonl") as append:
            metric_logger = make_metric_logger(Path("metrics.jsonl"), clearml_logger)
            self.assertIsNotNone(metric_logger)
            assert metric_logger is not None
            metric = {"mode": "train", "step": 1, "loss": 0.5}
            metric_logger(metric)

        append.assert_called_once_with(Path("metrics.jsonl"), metric)
        clearml_logger.report_scalar.assert_called_once_with(
            title="loss", series="train", value=0.5, iteration=1
        )

    def test_reports_final_metrics_as_summary_values(self) -> None:
        logger = Mock()
        result = TrainingResult(
            metrics=[
                {
                    "loss": 0.1,
                    "token_accuracy": 1.0,
                    "exact_match_accuracy": 0.75,
                }
            ],
            predictions=None,
        )

        report_final_metrics(logger, result)

        logger.report_single_value.assert_any_call(name="final/loss", value=0.1)
        logger.report_single_value.assert_any_call(name="final/token_accuracy", value=1.0)
        logger.report_single_value.assert_any_call(name="final/exact_match_accuracy", value=0.75)

    def test_reports_all_three_final_eval_scores_and_gaps(self) -> None:
        logger = Mock()
        summary: dict[str, Any] = {
            "mode": "eval_summary",
            "epoch": 1,
            "step": 4,
            "exact_match_accuracy": {
                "eval_exact_recall": 0.9,
                "eval_seen_family_new_template": 0.8,
                "eval_heldout_family": 0.6,
            },
            "paired_exact_match_gap": {
                "exact_recall_minus_seen_family_new_template": 0.1,
                "seen_family_new_template_minus_heldout_family": 0.2,
                "exact_recall_minus_heldout_family": 0.3,
            },
        }
        result = TrainingResult(
            metrics=[{"mode": "eval", "step": 4, "exact_match_accuracy": 0.6}],
            predictions=None,
            final_evaluation_summary=summary,
        )

        report_final_metrics(logger, result)

        for split_name, value in summary["exact_match_accuracy"].items():
            logger.report_single_value.assert_any_call(
                name=f"final/{split_name}/exact_match_accuracy",
                value=value,
            )
        logger.report_single_value.assert_any_call(
            name="final/gap/exact_recall_minus_seen_family_new_template",
            value=0.1,
        )

    def test_saves_final_eval_summary_as_readable_json(self) -> None:
        summary = {
            "mode": "eval_summary",
            "epoch": 1,
            "step": 4,
            "exact_match_accuracy": {"eval_exact_recall": 1.0},
            "paired_exact_match_gap": {},
        }
        result = TrainingResult([], None, final_evaluation_summary=summary)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            save_final_evaluation_summary(output_dir, result)
            rendered = (output_dir / "final_summary.json").read_text(encoding="utf-8")

        self.assertIn('"eval_exact_recall": 1.0', rendered)

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
