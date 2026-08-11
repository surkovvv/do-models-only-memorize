from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, cast

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_config import ExperimentConfig, config_from_mapping, load_experiment_config  # noqa: E402
from dataset import EVALUATION_SPLITS  # noqa: E402

if TYPE_CHECKING:
    from clearml import Logger, Task


EVALUATION_GROUP_FIELDS = (
    "world_id",
    "person_id",
    "operation_id",
    "template_family_id",
    "template_id",
    "fact_value",
)
EVALUATION_ARTIFACT_FILES = {
    "trace": "eval_predictions.jsonl",
    "aggregate": "eval_aggregates.jsonl",
    "pair": "eval_pairs.jsonl",
    "pair_aggregate": "eval_pair_aggregates.jsonl",
    "summary": "eval_summaries.jsonl",
}
PAIR_COMPARISONS = (
    (
        "exact_recall_minus_seen_family_new_template",
        "eval_exact_recall",
        "eval_seen_family_new_template",
    ),
    (
        "seen_family_new_template_minus_heldout_family",
        "eval_seen_family_new_template",
        "eval_heldout_family",
    ),
    (
        "exact_recall_minus_heldout_family",
        "eval_exact_recall",
        "eval_heldout_family",
    ),
)
ArtifactLogger = Callable[[Sequence[Mapping[str, Any]]], None]


# Одна запись из train.jsonl.
@dataclass(frozen=True, slots=True)
class Example:
    example_id: str
    world_id: str
    world_seed: str
    person_id: str
    person_name: str
    fact_id: str
    relation_id: str
    fact_value: str
    operation_id: str
    template_family_id: str
    template_id: str
    rendered_question: str
    canonical_answer: str
    answer_format_id: str
    split: str


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Metrics and final predictions produced by one training run."""

    metrics: list[dict[str, Any]]
    predictions: list[str] | None
    final_evaluation_summary: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EvaluationLoggers:
    """Optional durable sinks for every analyzable evaluation artifact."""

    trace: ArtifactLogger | None = None
    aggregate: ArtifactLogger | None = None
    pair: ArtifactLogger | None = None
    pair_aggregate: ArtifactLogger | None = None
    summary: ArtifactLogger | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Scalar metrics and final compact summary from one evaluation checkpoint."""

    metrics: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Optimization metrics available from a single causal-LM forward pass."""

    loss: float
    token_accuracy: float
    target_tokens: int
    learning_rate: float
    correct_tokens: int | None = None
    grad_norm: float | None = None


# датасет
class FactDataset(Dataset):
    def __init__(self, data: list[Example]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Example:
        return self.data[idx]


def select_smoke_examples(
    examples: Sequence[Example],
    batch_size: int,
    seed: int,
) -> list[Example]:
    if not examples:
        raise ValueError("cannot sample a smoke batch from an empty dataset")
    if batch_size <= 0:
        raise ValueError("smoke batch size must be positive")

    sample_size = min(batch_size, len(examples))
    return random.Random(seed).sample(examples, sample_size)


def group_evaluation_examples(
    examples: Sequence[Example],
) -> dict[str, list[Example]]:
    """Partition the combined evaluation file by its scientific slice."""

    groups: dict[str, list[Example]] = {split: [] for split in EVALUATION_SPLITS}
    unexpected_splits = sorted({example.split for example in examples} - set(groups))
    if unexpected_splits:
        raise ValueError(f"unexpected evaluation splits: {unexpected_splits}")
    for example in examples:
        groups[example.split].append(example)
    empty_splits = [split for split, split_examples in groups.items() if not split_examples]
    if empty_splits:
        raise ValueError(f"empty evaluation splits: {empty_splits}")
    return groups


# датадоудер, как оказалось, не особо нужен. а вот колейтор(то, что собирает батч) - нужен
class SFTCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_sequence_length: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.tokenizer.padding_side = "right"

    def __call__(self, data: list[Example]) -> dict[str, torch.Tensor]:
        conversations = [
            [
                {"role": "system", "content": "Answer with the value only."},
                {"role": "user", "content": elem.rendered_question},
                {"role": "assistant", "content": elem.canonical_answer},
            ]
            for elem in data
        ]
        length_options: dict[str, Any] = {"truncation": self.max_sequence_length is not None}
        if self.max_sequence_length is not None:
            length_options["max_length"] = self.max_sequence_length
        batch = cast(
            BatchEncoding,
            self.tokenizer.apply_chat_template(
                conversations,
                return_tensors="pt",
                padding=True,
                return_dict=True,
                enable_thinking=False,
                **length_options,
            ),
        )
        labels = batch["input_ids"].clone()

        # Padding
        labels[batch["attention_mask"] == 0] = -100

        prompt_token_ids = cast(
            list[list[int]],
            self.tokenizer.apply_chat_template(
                # without last assistant message
                [conversation[:-1] for conversation in conversations],
                # adding <|assistant|> token before actual assistant answer
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
                **length_options,
            ),
        )
        for row, prompt_ids in enumerate(prompt_token_ids):
            sequence_length = int(batch["attention_mask"][row].sum())
            if len(prompt_ids) > sequence_length:
                raise ValueError("prompt was truncated before the assistant answer")

            padding_length = batch["input_ids"].shape[1] - sequence_length
            prompt_start = padding_length if self.tokenizer.padding_side == "left" else 0
            actual_prompt_ids = batch["input_ids"][
                row, prompt_start : prompt_start + len(prompt_ids)
            ].tolist()
            if actual_prompt_ids != prompt_ids:
                raise ValueError("prompt tokenization is not a prefix of the full conversation!")

            labels[row, prompt_start : prompt_start + len(prompt_ids)] = -100

        batch["labels"] = labels
        return cast(dict[str, torch.Tensor], dict(batch))


# тест колатора
def inspect_sft_batch(
    tokenizer: PreTrainedTokenizerBase,
    batch: dict[str, torch.Tensor],
) -> None:
    for row in range(batch["input_ids"].shape[0]):
        input_ids = batch["input_ids"][row]
        labels = batch["labels"][row]
        attention_mask = batch["attention_mask"][row].bool()

        target_mask = labels != -100

        print(f"\n=== Example {row} ===")
        print("FULL:")
        print(tokenizer.decode(input_ids[attention_mask], skip_special_tokens=False))

        print("\nTRAINED TARGET:")
        print(tokenizer.decode(labels[target_mask], skip_special_tokens=False))

        print("\nTOKENS:")
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        for index, (token, label) in enumerate(zip(tokens, labels.tolist(), strict=True)):
            state = "MASK" if label == -100 else "LOSS"
            print(f"{index:4} {state:4} {token!r}")


def resolve_warmup_steps(
    total_steps: int,
    warmup_ratio: float,
    warmup_min_steps: int,
) -> int:
    """Resolve ratio-based warmup with a lower bound and a finite-run clamp."""

    if total_steps <= 0:
        raise ValueError("total optimizer steps must be positive")
    requested = max(math.ceil(total_steps * warmup_ratio), warmup_min_steps)
    return min(total_steps, requested)


def cosine_learning_rate_multiplier(
    current_step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    final_learning_rate_ratio: float,
) -> float:
    """Warm up linearly, then decay with cosine to the configured LR floor."""

    if warmup_steps and current_step < warmup_steps:
        return (current_step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return final_learning_rate_ratio
    progress = min(1.0, (current_step - warmup_steps) / (decay_steps - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_learning_rate_ratio + (1.0 - final_learning_rate_ratio) * cosine


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str,
    total_steps: int,
    warmup_ratio: float,
    warmup_min_steps: int,
    final_learning_rate_ratio: float,
) -> tuple[Any | None, int]:
    """Build the optimizer-step scheduler and return its resolved warmup length."""

    if scheduler_type == "constant":
        return None, 0
    if scheduler_type != "cosine":
        raise ValueError(f"unsupported learning-rate scheduler: {scheduler_type!r}")
    warmup_steps = resolve_warmup_steps(total_steps, warmup_ratio, warmup_min_steps)

    from torch.optim.lr_scheduler import LambdaLR

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda current_step: cosine_learning_rate_multiplier(
            current_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            final_learning_rate_ratio=final_learning_rate_ratio,
        ),
    )
    return scheduler, warmup_steps


# трейн шаг
def train_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str = "fp32",
    *,
    gradient_accumulation_divisor: int = 1,
    zero_grad: bool = True,
    perform_optimizer_step: bool = True,
    max_grad_norm: float | None = None,
    lr_scheduler: Any | None = None,
) -> StepMetrics:
    """Run one micro-step and optionally finish its accumulated optimizer step."""

    if gradient_accumulation_divisor <= 0:
        raise ValueError("gradient accumulation divisor must be positive")
    model.train()
    if zero_grad:
        optimizer.zero_grad()
    for k, v in batch.items():
        batch[k] = v.to(device)

    use_bf16 = precision == "bf16"
    if use_bf16 and device.type != "cuda":
        raise ValueError("bf16 training currently requires a CUDA device")

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_bf16,
    ):
        outputs = model(**batch)
    loss = outputs.loss

    shifted_labels = batch["labels"][..., 1:]
    target_mask = shifted_labels != -100
    target_tokens = int(target_mask.sum().item())
    if target_tokens == 0:
        raise ValueError("training batch contains no target tokens")
    with torch.no_grad():
        shifted_predictions = outputs.logits[..., :-1, :].argmax(dim=-1)
        correct_tokens = int(
            (shifted_predictions[target_mask] == shifted_labels[target_mask]).sum().item()
        )

    learning_rate = float(optimizer.param_groups[0]["lr"])
    (loss / gradient_accumulation_divisor).backward()
    grad_norm: float | None = None
    if perform_optimizer_step:
        if max_grad_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norm = float(norm.item())
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

    return StepMetrics(
        loss=loss.item(),
        token_accuracy=correct_tokens / target_tokens,
        target_tokens=target_tokens,
        learning_rate=learning_rate,
        correct_tokens=correct_tokens,
        grad_norm=grad_norm,
    )


# трейн луп
def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    smoke_steps: int | None,
    tokenizer: PreTrainedTokenizerBase,
    smoke_examples: Sequence[Example] | None,
    prediction_log_limit: int | None,
    precision: str = "fp32",
    metric_logger: Callable[[Mapping[str, Any]], None] | None = None,
    log_interval_steps: int = 1,
    evaluation_splits: Mapping[str, Sequence[Example]] | None = None,
    evaluation_batch_size: int = 64,
    evaluation_loggers: EvaluationLoggers | None = None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float | None = None,
    lr_scheduler: Any | None = None,
    max_sequence_length: int | None = None,
) -> TrainingResult:
    if log_interval_steps <= 0:
        raise ValueError("log_interval_steps must be positive")
    if evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    metrics: list[dict[str, Any]] = []
    final_predictions: list[str] | None = None

    def record_metric(metric: dict[str, Any]) -> None:
        metrics.append(metric)
        if metric_logger is not None:
            metric_logger(metric)

    if smoke_steps is not None:
        if not smoke_examples:
            raise ValueError("smoke examples must not be empty")
        try:
            first_batch = next(iter(dataloader))
        except StopIteration as error:
            raise ValueError("cannot run smoke training on an empty dataset") from error
        if first_batch["input_ids"].shape[0] != len(smoke_examples):
            raise ValueError("smoke examples do not match the first training batch")

        for step in range(smoke_steps):
            step_metrics = train_step(
                model,
                first_batch,
                optimizer,
                device,
                precision=precision,
                max_grad_norm=max_grad_norm,
                lr_scheduler=lr_scheduler,
            )
            global_step = step + 1
            should_log = (
                global_step == 1
                or global_step % log_interval_steps == 0
                or global_step == smoke_steps
            )
            if should_log:
                predictions = generate_answers(
                    model,
                    tokenizer,
                    smoke_examples,
                    device,
                    max_sequence_length=max_sequence_length,
                )
                final_predictions = predictions
                accuracy = exact_match_accuracy(predictions, smoke_examples)
                record_metric(
                    {
                        "mode": "smoke",
                        "step": global_step,
                        "loss": step_metrics.loss,
                        "token_accuracy": step_metrics.token_accuracy,
                        "exact_match_accuracy": accuracy,
                        "learning_rate": step_metrics.learning_rate,
                        "target_tokens": step_metrics.target_tokens,
                        "examples_seen": global_step * len(smoke_examples),
                    }
                )
                print(
                    f"smoke_step={global_step}/{smoke_steps} "
                    f"loss={step_metrics.loss:.6f} "
                    f"token_accuracy={step_metrics.token_accuracy:.4f} "
                    f"exact_match_accuracy={accuracy:.4f}"
                )
                log_predictions(predictions, smoke_examples, prediction_log_limit)
        return TrainingResult(metrics=metrics, predictions=final_predictions)

    evaluation_splits = evaluation_splits or {}
    step = 0
    examples_seen = 0
    optimizer_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    total_steps = epochs * optimizer_steps_per_epoch
    micro_step = 0
    evaluation_result = evaluate_splits(
        model,
        tokenizer,
        evaluation_splits,
        device,
        batch_size=evaluation_batch_size,
        epoch=0,
        step=0,
        loggers=evaluation_loggers,
        max_sequence_length=max_sequence_length,
    )
    for metric in evaluation_result.metrics:
        record_metric(metric)
    final_evaluation_summary = evaluation_result.summary

    for epoch in range(epochs):
        window_metrics: list[StepMetrics] = []
        window_examples = 0
        for batch_index, batch in enumerate(dataloader):
            batch_size = int(batch["input_ids"].shape[0])
            window_start = (batch_index // gradient_accumulation_steps) * (
                gradient_accumulation_steps
            )
            window_end = min(window_start + gradient_accumulation_steps, len(dataloader))
            window_size = window_end - window_start
            is_window_start = batch_index == window_start
            is_window_end = batch_index + 1 == window_end
            step_metrics = train_step(
                model,
                batch,
                optimizer,
                device,
                precision=precision,
                gradient_accumulation_divisor=window_size,
                zero_grad=is_window_start,
                perform_optimizer_step=is_window_end,
                max_grad_norm=max_grad_norm,
                lr_scheduler=lr_scheduler,
            )
            window_metrics.append(step_metrics)
            window_examples += batch_size
            micro_step += 1
            examples_seen += batch_size
            if not is_window_end:
                continue

            step += 1
            target_tokens = sum(metric.target_tokens for metric in window_metrics)
            correct_tokens = sum(
                metric.correct_tokens
                if metric.correct_tokens is not None
                else round(metric.token_accuracy * metric.target_tokens)
                for metric in window_metrics
            )
            loss = sum(
                metric.loss * metric.target_tokens for metric in window_metrics
            ) / target_tokens
            token_accuracy = correct_tokens / target_tokens
            learning_rate = window_metrics[-1].learning_rate
            grad_norm = window_metrics[-1].grad_norm
            if step == 1 or step % log_interval_steps == 0 or step == total_steps:
                train_metric: dict[str, Any] = {
                    "mode": "train",
                    "epoch": epoch + 1,
                    "step": step,
                    "micro_step": micro_step,
                    "loss": loss,
                    "token_accuracy": token_accuracy,
                    "learning_rate": learning_rate,
                    "target_tokens": target_tokens,
                    "examples_in_optimizer_step": window_examples,
                    "examples_seen": examples_seen,
                }
                if grad_norm is not None:
                    train_metric["grad_norm"] = grad_norm
                record_metric(train_metric)
                grad_norm_log = "" if grad_norm is None else f" grad_norm={grad_norm:.4f}"
                print(
                    f"epoch={epoch + 1} optimizer_step={step}/{total_steps} "
                    f"micro_step={micro_step} loss={loss:.6f} "
                    f"token_accuracy={token_accuracy:.4f} "
                    f"learning_rate={learning_rate:.8g}{grad_norm_log}"
                )
            window_metrics = []
            window_examples = 0

        evaluation_result = evaluate_splits(
            model,
            tokenizer,
            evaluation_splits,
            device,
            batch_size=evaluation_batch_size,
            epoch=epoch + 1,
            step=step,
            loggers=evaluation_loggers,
            max_sequence_length=max_sequence_length,
        )
        for metric in evaluation_result.metrics:
            record_metric(metric)
        final_evaluation_summary = evaluation_result.summary

    return TrainingResult(
        metrics=metrics,
        predictions=None,
        final_evaluation_summary=final_evaluation_summary,
    )


# эвал луп?
def generate_answers(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[Example],
    device: torch.device,
    *,
    max_sequence_length: int | None = None,
) -> list[str]:
    tokenizer.padding_side = "left"

    conversations = [
        [
            {"role": "system", "content": "Answer with the value only."},
            {"role": "user", "content": example.rendered_question},
        ]
        for example in examples
    ]

    length_options: dict[str, Any] = {"truncation": max_sequence_length is not None}
    if max_sequence_length is not None:
        length_options["max_length"] = max_sequence_length
    batch = cast(
        BatchEncoding,
        tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
            **length_options,
        ),
    )

    prompt_width = batch["input_ids"].shape[1]
    batch = batch.to(device)

    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=32,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[:, prompt_width:]
    return tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )


def normalize_answer(value: str) -> str:
    return value.strip()


def exact_match_accuracy(
    predictions: Sequence[str],
    examples: Sequence[Example],
) -> float:
    if len(predictions) != len(examples):
        raise ValueError("predictions and examples must have the same length")
    if not examples:
        raise ValueError("cannot compute exact-match accuracy for an empty batch")

    correct = sum(
        normalize_answer(prediction) == normalize_answer(example.canonical_answer)
        for prediction, example in zip(predictions, examples, strict=True)
    )
    return correct / len(examples)


def generate_answers_batched(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[Example],
    device: torch.device,
    batch_size: int,
    *,
    max_sequence_length: int | None = None,
) -> list[str]:
    """Generate answers without materializing an entire evaluation split on device."""

    if batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    predictions: list[str] = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        predictions.extend(
            generate_answers(
                model,
                tokenizer,
                batch_examples,
                device,
                max_sequence_length=max_sequence_length,
            )
        )
    return predictions


def make_evaluation_trace(
    example: Example,
    prediction: str,
    *,
    epoch: int,
    step: int,
) -> dict[str, Any]:
    """Combine a full source example with its lossless generated answer."""

    normalized_prediction = normalize_answer(prediction)
    normalized_target = normalize_answer(example.canonical_answer)
    return {
        "epoch": epoch,
        "step": step,
        **asdict(example),
        "prediction_raw": prediction,
        "prediction_normalized": normalized_prediction,
        "target_normalized": normalized_target,
        "exact_match": normalized_prediction == normalized_target,
    }


def _accuracy_record(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    group_by: str,
    group_value: str,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    correct = sum(bool(row["exact_match"]) for row in rows)
    return {
        "mode": mode,
        "epoch": int(rows[0]["epoch"]),
        "step": int(rows[0]["step"]),
        **extra,
        "group_by": group_by,
        "group_value": group_value,
        "correct": correct,
        "examples": len(rows),
        "exact_match_accuracy": correct / len(rows),
    }


def aggregate_evaluation_traces(
    traces: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create tidy exact-match aggregates for every requested analysis dimension."""

    if not traces:
        raise ValueError("cannot aggregate empty evaluation traces")
    aggregates: list[dict[str, Any]] = []
    split_order = [
        split
        for split in EVALUATION_SPLITS
        if any(row["split"] == split for row in traces)
    ]
    for split_name in split_order:
        split_rows = [row for row in traces if row["split"] == split_name]
        aggregates.append(
            _accuracy_record(
                split_rows,
                mode="eval_aggregate",
                group_by="overall",
                group_value="all",
                extra={"split": split_name},
            )
        )
        for field in EVALUATION_GROUP_FIELDS:
            values = sorted({str(row[field]) for row in split_rows})
            for value in values:
                group_rows = [row for row in split_rows if str(row[field]) == value]
                aggregates.append(
                    _accuracy_record(
                        group_rows,
                        mode="eval_aggregate",
                        group_by=field,
                        group_value=value,
                        extra={"split": split_name, field: value},
                    )
                )
    return aggregates


def _pair_transition(left_correct: bool, right_correct: bool) -> str:
    if left_correct and right_correct:
        return "both_correct"
    if left_correct:
        return "left_only"
    if right_correct:
        return "right_only"
    return "neither_correct"


def build_paired_evaluation_rows(
    traces: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Align slice results by fact so template-transfer gaps stay paired."""

    rows_by_split: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    for split_name in EVALUATION_SPLITS:
        split_rows = [row for row in traces if row["split"] == split_name]
        index = {(str(row["world_id"]), str(row["fact_id"])): row for row in split_rows}
        if len(index) != len(split_rows):
            raise ValueError(f"duplicate world/fact rows in evaluation split {split_name!r}")
        rows_by_split[split_name] = index

    fact_keys = set(rows_by_split[EVALUATION_SPLITS[0]])
    for split_name, index in rows_by_split.items():
        if set(index) != fact_keys:
            raise ValueError(f"evaluation split {split_name!r} is not fact-paired")

    paired_rows: list[dict[str, Any]] = []
    for comparison, left_split, right_split in PAIR_COMPARISONS:
        for fact_key in sorted(fact_keys):
            left = rows_by_split[left_split][fact_key]
            right = rows_by_split[right_split][fact_key]
            for field in (
                "world_id",
                "world_seed",
                "person_id",
                "person_name",
                "fact_id",
                "relation_id",
                "fact_value",
                "operation_id",
                "canonical_answer",
            ):
                if left[field] != right[field]:
                    raise ValueError(
                        f"paired evaluation rows disagree on {field!r} for {fact_key!r}"
                    )
            left_correct = bool(left["exact_match"])
            right_correct = bool(right["exact_match"])
            paired_rows.append(
                {
                    "mode": "eval_pair",
                    "epoch": int(left["epoch"]),
                    "step": int(left["step"]),
                    "comparison": comparison,
                    "left_split": left_split,
                    "right_split": right_split,
                    "world_id": left["world_id"],
                    "world_seed": left["world_seed"],
                    "person_id": left["person_id"],
                    "person_name": left["person_name"],
                    "fact_id": left["fact_id"],
                    "relation_id": left["relation_id"],
                    "fact_value": left["fact_value"],
                    "operation_id": left["operation_id"],
                    "canonical_answer": left["canonical_answer"],
                    "left_template_family_id": left["template_family_id"],
                    "right_template_family_id": right["template_family_id"],
                    "left_template_id": left["template_id"],
                    "right_template_id": right["template_id"],
                    "left_question": left["rendered_question"],
                    "right_question": right["rendered_question"],
                    "left_prediction_raw": left["prediction_raw"],
                    "right_prediction_raw": right["prediction_raw"],
                    "left_exact_match": left_correct,
                    "right_exact_match": right_correct,
                    "correctness_delta": int(left_correct) - int(right_correct),
                    "transition": _pair_transition(left_correct, right_correct),
                }
            )
    return paired_rows


def _paired_aggregate_record(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_by: str,
    group_value: str,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    left_correct = sum(bool(row["left_exact_match"]) for row in rows)
    right_correct = sum(bool(row["right_exact_match"]) for row in rows)
    transitions = {
        transition: sum(row["transition"] == transition for row in rows)
        for transition in ("both_correct", "left_only", "right_only", "neither_correct")
    }
    return {
        "mode": "eval_pair_aggregate",
        "epoch": int(rows[0]["epoch"]),
        "step": int(rows[0]["step"]),
        **extra,
        "group_by": group_by,
        "group_value": group_value,
        "examples": len(rows),
        "left_correct": left_correct,
        "right_correct": right_correct,
        "left_exact_match_accuracy": left_correct / len(rows),
        "right_exact_match_accuracy": right_correct / len(rows),
        "exact_match_gap": (left_correct - right_correct) / len(rows),
        **transitions,
    }


def aggregate_paired_evaluation_rows(
    paired_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize paired correctness transitions overall and by stable dimensions."""

    if not paired_rows:
        raise ValueError("cannot aggregate empty paired evaluation rows")
    aggregates: list[dict[str, Any]] = []
    for comparison, _, _ in PAIR_COMPARISONS:
        comparison_rows = [row for row in paired_rows if row["comparison"] == comparison]
        aggregates.append(
            _paired_aggregate_record(
                comparison_rows,
                group_by="overall",
                group_value="all",
                extra={"comparison": comparison},
            )
        )
        for field in (
            "world_id",
            "person_id",
            "operation_id",
            "left_template_family_id",
            "right_template_family_id",
        ):
            for value in sorted({str(row[field]) for row in comparison_rows}):
                group_rows = [row for row in comparison_rows if str(row[field]) == value]
                aggregates.append(
                    _paired_aggregate_record(
                        group_rows,
                        group_by=field,
                        group_value=value,
                        extra={"comparison": comparison, field: value},
                    )
                )
    return aggregates


def evaluation_summary(
    aggregates: Sequence[Mapping[str, Any]],
    pair_aggregates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the three primary EM values and their paired transfer gaps."""

    overall = {
        str(row["split"]): float(row["exact_match_accuracy"])
        for row in aggregates
        if row["group_by"] == "overall"
    }
    gaps = {
        str(row["comparison"]): float(row["exact_match_gap"])
        for row in pair_aggregates
        if row["group_by"] == "overall"
    }
    if set(overall) != set(EVALUATION_SPLITS):
        raise ValueError("evaluation summary requires all scientific slices")
    if set(gaps) != {comparison for comparison, _, _ in PAIR_COMPARISONS}:
        raise ValueError("evaluation summary requires all paired comparisons")
    first = aggregates[0]
    return {
        "mode": "eval_summary",
        "epoch": int(first["epoch"]),
        "step": int(first["step"]),
        "exact_match_accuracy": overall,
        "paired_exact_match_gap": gaps,
    }


def evaluate_splits(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    splits: Mapping[str, Sequence[Example]],
    device: torch.device,
    *,
    batch_size: int,
    epoch: int,
    step: int,
    loggers: EvaluationLoggers | None = None,
    max_sequence_length: int | None = None,
) -> EvaluationResult:
    """Generate, persist, and aggregate all three paired evaluation slices."""

    if batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    if set(splits) != set(EVALUATION_SPLITS):
        raise ValueError("evaluation requires exactly the three scientific slices")
    loggers = loggers or EvaluationLoggers()
    traces: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for split_name in EVALUATION_SPLITS:
        examples = splits[split_name]
        if not examples:
            raise ValueError(f"evaluation split {split_name!r} must not be empty")
        split_traces: list[dict[str, Any]] = []
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            predictions = generate_answers(
                model,
                tokenizer,
                batch_examples,
                device,
                max_sequence_length=max_sequence_length,
            )
            batch_traces = [
                make_evaluation_trace(example, prediction, epoch=epoch, step=step)
                for prediction, example in zip(predictions, batch_examples, strict=True)
            ]
            split_traces.extend(batch_traces)
            traces.extend(batch_traces)
            if loggers.trace is not None:
                loggers.trace(batch_traces)

        accuracy = sum(row["exact_match"] for row in split_traces) / len(split_traces)
        metric = {
            "mode": "eval",
            "split": split_name,
            "epoch": epoch,
            "step": step,
            "exact_match_accuracy": accuracy,
            "examples": len(examples),
        }
        metrics.append(metric)
        print(
            f"eval_split={split_name} epoch={epoch} step={step} "
            f"examples={len(examples)} exact_match_accuracy={accuracy:.4f}"
        )

    aggregates = aggregate_evaluation_traces(traces)
    paired_rows = build_paired_evaluation_rows(traces)
    pair_aggregates = aggregate_paired_evaluation_rows(paired_rows)
    summary = evaluation_summary(aggregates, pair_aggregates)
    if loggers.aggregate is not None:
        loggers.aggregate(aggregates)
    if loggers.pair is not None:
        loggers.pair(paired_rows)
    if loggers.pair_aggregate is not None:
        loggers.pair_aggregate(pair_aggregates)
    if loggers.summary is not None:
        loggers.summary([summary])

    exact_match = summary["exact_match_accuracy"]
    print(
        f"eval_em epoch={epoch} step={step} "
        f"exact_recall={exact_match['eval_exact_recall']:.4f} "
        f"seen_family_new_template={exact_match['eval_seen_family_new_template']:.4f} "
        f"heldout_family={exact_match['eval_heldout_family']:.4f}"
    )
    gaps = summary["paired_exact_match_gap"]
    print(
        f"eval_gap epoch={epoch} step={step} "
        f"template={gaps['exact_recall_minus_seen_family_new_template']:+.4f} "
        f"family={gaps['seen_family_new_template_minus_heldout_family']:+.4f} "
        f"total={gaps['exact_recall_minus_heldout_family']:+.4f}"
    )
    for comparison, gap in gaps.items():
        metrics.append(
            {
                "mode": "eval_gap",
                "comparison": comparison,
                "epoch": epoch,
                "step": step,
                "exact_match_gap": gap,
                "examples": next(
                    row["examples"]
                    for row in pair_aggregates
                    if row["comparison"] == comparison and row["group_by"] == "overall"
                ),
            }
        )
    return EvaluationResult(metrics=metrics, summary=summary)


def log_predictions(
    predictions: Sequence[str],
    examples: Sequence[Example],
    limit: int | None,
) -> None:
    shown = len(examples) if limit is None else min(limit, len(examples))
    for index, (prediction, example) in enumerate(
        zip(predictions[:shown], examples[:shown], strict=True)
    ):
        is_match = normalize_answer(prediction) == normalize_answer(example.canonical_answer)
        print(
            f"prediction[{index:02d}] match={str(is_match).lower()} "
            f"question={example.rendered_question!r} prediction={prediction!r} "
            f"target={example.canonical_answer!r}"
        )

    omitted = len(examples) - shown
    if omitted:
        print(f"predictions_omitted={omitted}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the small H1 SFT experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sft_smoke.yaml",
        help="path to an experiment YAML file",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="SECTION.KEY=VALUE",
        help="typed configuration overrides, for example training.batch_size=4",
    )
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_tracking(config: ExperimentConfig) -> tuple[Task | None, ExperimentConfig]:
    if not config.tracking.enabled:
        return None, config

    from clearml import Task

    project_name = config.tracking.project_name
    if config.tracking.project_id is not None:
        from clearml.backend_api.session.client import APIClient

        project = APIClient().projects.get_by_id(config.tracking.project_id)
        project_name = str(project.name)

    task: Task = Task.init(
        project_name=project_name,
        task_name=(
            config.tracking.task_name
            or os.environ.get("EXPERIMENT_RUN_ID")
            or config.experiment_name
        ),
        output_uri=config.tracking.output_uri,
        tags=config.tracking.tags,
        reuse_last_task_id=False,
    )
    connected_config = task.connect(config.to_dict(), name="config")
    to_dict = getattr(connected_config, "to_dict", None)
    if callable(to_dict):
        connected_config = to_dict()
    elif isinstance(connected_config, Mapping):
        connected_config = dict(connected_config)
    else:  # pragma: no cover - defensive guard around an external SDK contract
        raise TypeError("ClearML connected config is not a mapping")
    return task, config_from_mapping(connected_config)


def resolve_project_path(value: str) -> Path:
    """Resolve a configured path without coupling callers to the repository location."""

    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def prepare_output_dir(config: ExperimentConfig) -> Path | None:
    """Create a run output directory and persist its resolved scientific config."""

    if config.runtime.output_dir is None:
        return None

    output_dir = resolve_project_path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.resolved.yaml").write_text(
        OmegaConf.to_yaml(config.to_dict(), resolve=True, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    if config.training.smoke_steps is None:
        for filename in EVALUATION_ARTIFACT_FILES.values():
            (output_dir / filename).write_text("", encoding="utf-8")
    return output_dir


def write_environment_metadata(output_dir: Path, device: torch.device) -> None:
    """Record the runtime facts needed to diagnose or reproduce a remote run."""

    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "hostname": os.environ.get("HOSTNAME"),
    }
    if device.type == "cuda":
        device_index = device.index or 0
        properties = torch.cuda.get_device_properties(device_index)
        metadata["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(device_index)),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }

    (output_dir / "environment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(dict(value), ensure_ascii=False) + "\n")


def append_jsonl_many(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    """Append a batch atomically enough to avoid per-example file opens."""

    with path.open("a", encoding="utf-8") as output:
        for value in values:
            output.write(json.dumps(dict(value), ensure_ascii=False) + "\n")


def make_evaluation_loggers(output_dir: Path | None) -> EvaluationLoggers | None:
    """Create batched JSONL sinks for full evaluation when artifacts are enabled."""

    if output_dir is None:
        return None

    def sink(name: str) -> ArtifactLogger:
        path = output_dir / EVALUATION_ARTIFACT_FILES[name]
        return lambda values: append_jsonl_many(path, values)

    return EvaluationLoggers(
        trace=sink("trace"),
        aggregate=sink("aggregate"),
        pair=sink("pair"),
        pair_aggregate=sink("pair_aggregate"),
        summary=sink("summary"),
    )


def report_metric_to_clearml(logger: Logger, metric: Mapping[str, Any]) -> None:
    """Publish one trainer metric record using stable ClearML chart names."""

    iteration = int(metric["step"])
    mode = str(metric["mode"])
    if mode == "eval_gap":
        logger.report_scalar(
            title="accuracy_gap",
            series=f"eval/{metric['comparison']}",
            value=float(metric["exact_match_gap"]),
            iteration=iteration,
        )
        return
    split = metric.get("split")
    series_prefix = mode if split is None else f"{mode}/{split}"
    scalar_series = {
        "loss": ("loss", series_prefix),
        "token_accuracy": ("accuracy", f"{series_prefix}/token"),
        "exact_match_accuracy": ("accuracy", f"{series_prefix}/answer_exact_match"),
        "learning_rate": ("optimization", f"{series_prefix}/learning_rate"),
        "grad_norm": ("optimization", f"{series_prefix}/grad_norm"),
        "examples_seen": ("progress", f"{series_prefix}/examples_seen"),
    }
    for key, (title, series) in scalar_series.items():
        value = metric.get(key)
        if value is not None:
            logger.report_scalar(
                title=title,
                series=series,
                value=float(value),
                iteration=iteration,
            )


def make_metric_logger(
    metrics_path: Path | None,
    clearml_logger: Logger | None,
) -> Callable[[Mapping[str, Any]], None] | None:
    """Fan each metric out to the local artifact and optional ClearML task."""

    if metrics_path is None and clearml_logger is None:
        return None

    def report(metric: Mapping[str, Any]) -> None:
        if metrics_path is not None:
            append_jsonl(metrics_path, metric)
        if clearml_logger is not None:
            report_metric_to_clearml(clearml_logger, metric)

    return report


def report_final_metrics(logger: Logger, result: TrainingResult) -> None:
    """Expose the final point as searchable ClearML summary values."""

    if not result.metrics:
        return
    for key in ("loss", "token_accuracy"):
        value = next(
            (metric[key] for metric in reversed(result.metrics) if metric.get(key) is not None),
            None,
        )
        if value is not None:
            logger.report_single_value(name=f"final/{key}", value=float(value))

    if result.final_evaluation_summary is None:
        value = next(
            (
                metric["exact_match_accuracy"]
                for metric in reversed(result.metrics)
                if metric.get("exact_match_accuracy") is not None
            ),
            None,
        )
        if value is not None:
            logger.report_single_value(name="final/exact_match_accuracy", value=float(value))
        return

    exact_match = result.final_evaluation_summary["exact_match_accuracy"]
    for split_name in EVALUATION_SPLITS:
        logger.report_single_value(
            name=f"final/{split_name}/exact_match_accuracy",
            value=float(exact_match[split_name]),
        )
    for comparison, gap in result.final_evaluation_summary["paired_exact_match_gap"].items():
        logger.report_single_value(
            name=f"final/gap/{comparison}",
            value=float(gap),
        )


def save_final_evaluation_summary(output_dir: Path, result: TrainingResult) -> None:
    """Persist the final searchable triplet separately from checkpoint history."""

    if result.final_evaluation_summary is None:
        return
    (output_dir / "final_summary.json").write_text(
        json.dumps(result.final_evaluation_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def upload_evaluation_artifacts(task: Task, output_dir: Path) -> None:
    """Attach completed evaluation tables to the ClearML task when enabled."""

    artifact_paths = [
        *(output_dir / filename for filename in EVALUATION_ARTIFACT_FILES.values()),
        output_dir / "final_summary.json",
    ]
    for path in artifact_paths:
        if path.is_file() and path.stat().st_size:
            task.upload_artifact(
                name=f"evaluation/{path.stem}",
                artifact_object=str(path),
            )


def save_smoke_predictions(
    output_dir: Path,
    predictions: Sequence[str],
    examples: Sequence[Example],
) -> None:
    path = output_dir / "predictions.jsonl"
    with path.open("w", encoding="utf-8") as output:
        for prediction, example in zip(predictions, examples, strict=True):
            value = {
                "example_id": example.example_id,
                "question": example.rendered_question,
                "prediction": prediction,
                "target": example.canonical_answer,
                "exact_match": normalize_answer(prediction)
                == normalize_answer(example.canonical_answer),
            }
            output.write(json.dumps(value, ensure_ascii=False) + "\n")


def run(config: ExperimentConfig, task: Task | None = None) -> TrainingResult:
    output_dir = prepare_output_dir(config)
    path_to_data = resolve_project_path(config.data.train_path)

    with path_to_data.open(encoding="utf-8") as f:
        data = [Example(**json.loads(line)) for line in f]

    evaluation_splits: Mapping[str, Sequence[Example]] | None = None
    if config.training.smoke_steps is None:
        path_to_test_data = resolve_project_path(config.data.test_path)
        with path_to_test_data.open(encoding="utf-8") as f:
            test_data = [Example(**json.loads(line)) for line in f]
        evaluation_splits = group_evaluation_examples(test_data)

    smoke_examples: list[Example] | None = None
    training_data = data
    if config.training.smoke_steps is not None:
        smoke_examples = select_smoke_examples(
            data,
            batch_size=config.training.batch_size,
            seed=config.training.seed,
        )
        training_data = smoke_examples

    dataset = FactDataset(training_data)
    model_name_or_path = config.model.name_or_path
    configured_model_path = Path(model_name_or_path)
    if not configured_model_path.is_absolute() and (ROOT / configured_model_path).exists():
        model_name_or_path = str(ROOT / configured_model_path)

    pretrained_kwargs: dict[str, Any] = {
        "local_files_only": config.model.local_files_only,
    }
    if config.model.revision is not None and not Path(model_name_or_path).is_dir():
        pretrained_kwargs["revision"] = config.model.revision

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        **pretrained_kwargs,
    )

    if config.runtime.debug:
        batch = SFTCollator(
            tokenizer,
            max_sequence_length=config.training.max_sequence_length,
        )([dataset[0], dataset[1]])
        inspect_sft_batch(tokenizer, batch)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.smoke_steps is None,
        collate_fn=SFTCollator(
            tokenizer,
            max_sequence_length=config.training.max_sequence_length,
        ),
    )

    model: nn.Module = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **pretrained_kwargs,
    )
    device = resolve_device(config.runtime.device)
    if config.training.precision == "bf16":
        if device.type != "cuda":
            raise ValueError("training.precision=bf16 requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("the selected CUDA device does not support bf16")
    model = model.to(device)

    if output_dir is not None:
        write_environment_metadata(output_dir, device)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        eps=config.training.adam_epsilon,
        weight_decay=config.training.weight_decay,
    )
    total_optimizer_steps = (
        config.training.smoke_steps
        if config.training.smoke_steps is not None
        else config.training.epochs
        * math.ceil(len(dataloader) / config.training.gradient_accumulation_steps)
    )
    lr_scheduler, warmup_steps = create_lr_scheduler(
        optimizer,
        scheduler_type=config.training.lr_scheduler,
        total_steps=total_optimizer_steps,
        warmup_ratio=config.training.warmup_ratio,
        warmup_min_steps=config.training.warmup_min_steps,
        final_learning_rate_ratio=config.training.final_learning_rate_ratio,
    )
    print(
        f"training_protocol optimizer_steps={total_optimizer_steps} "
        f"micro_batch_size={config.training.batch_size} "
        f"gradient_accumulation_steps={config.training.gradient_accumulation_steps} "
        f"effective_batch_size="
        f"{config.training.batch_size * config.training.gradient_accumulation_steps} "
        f"peak_learning_rate={config.training.learning_rate:.8g} "
        f"lr_scheduler={config.training.lr_scheduler} warmup_steps={warmup_steps} "
        f"max_sequence_length={config.training.max_sequence_length}"
    )

    metrics_path = output_dir / "metrics.jsonl" if output_dir is not None else None
    clearml_logger = task.get_logger() if task is not None else None
    if config.training.smoke_steps is None and output_dir is None:
        print("eval_artifacts=disabled reason=runtime.output_dir_is_null")
    result = train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        epochs=config.training.epochs,
        smoke_steps=config.training.smoke_steps,
        tokenizer=tokenizer,
        smoke_examples=smoke_examples,
        prediction_log_limit=config.runtime.prediction_log_limit,
        precision=config.training.precision,
        metric_logger=make_metric_logger(metrics_path, clearml_logger),
        log_interval_steps=config.tracking.log_interval_steps,
        evaluation_splits=evaluation_splits,
        evaluation_batch_size=config.evaluation.batch_size,
        evaluation_loggers=make_evaluation_loggers(output_dir),
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=config.training.max_grad_norm,
        lr_scheduler=lr_scheduler,
        max_sequence_length=config.training.max_sequence_length,
    )

    if clearml_logger is not None:
        report_final_metrics(clearml_logger, result)

    if output_dir is not None:
        save_final_evaluation_summary(output_dir, result)
        if result.predictions is not None and smoke_examples is not None:
            save_smoke_predictions(output_dir, result.predictions, smoke_examples)
        if (
            task is not None
            and result.final_evaluation_summary is not None
            and config.tracking.output_uri not in (None, False)
        ):
            upload_evaluation_artifacts(task, output_dir)
    if output_dir is not None and config.runtime.save_model:
        model_output_dir = output_dir / "final_model"
        cast(Any, model).save_pretrained(model_output_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_output_dir)
    return result


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, args.overrides)
    task, config = init_tracking(config)
    try:
        # ClearML Agent/UI may have overridden the connected configuration.
        random.seed(config.training.seed)
        torch.manual_seed(config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training.seed)
        run(config, task=task)
    finally:
        if task is not None:
            task.close()


if __name__ == "__main__":
    main()
