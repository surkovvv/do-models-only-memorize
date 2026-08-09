from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Optimization metrics available from a single causal-LM forward pass."""

    loss: float
    token_accuracy: float
    target_tokens: int
    learning_rate: float


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

    groups = {split: [] for split in EVALUATION_SPLITS}
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
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
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
        batch = cast(
            BatchEncoding,
            self.tokenizer.apply_chat_template(
                conversations,
                return_tensors="pt",
                padding=True,
                truncation=False,  # for smoke purpose ok
                return_dict=True,
                enable_thinking=False,
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


# трейн шаг
def train_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str = "fp32",
) -> StepMetrics:
    """Run one optimizer step and measure accuracy on non-masked target tokens."""
    model.train()
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

    loss.backward()
    optimizer.step()

    return StepMetrics(
        loss=loss.item(),
        token_accuracy=correct_tokens / target_tokens,
        target_tokens=target_tokens,
        learning_rate=float(optimizer.param_groups[0]["lr"]),
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
) -> TrainingResult:
    if log_interval_steps <= 0:
        raise ValueError("log_interval_steps must be positive")
    if evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be positive")

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
            )
            global_step = step + 1
            should_log = (
                global_step == 1
                or global_step % log_interval_steps == 0
                or global_step == smoke_steps
            )
            if should_log:
                predictions = generate_answers(model, tokenizer, smoke_examples, device)
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
    total_steps = epochs * len(dataloader)
    for metric in evaluate_splits(
        model,
        tokenizer,
        evaluation_splits,
        device,
        batch_size=evaluation_batch_size,
        epoch=0,
        step=0,
    ):
        record_metric(metric)

    for epoch in range(epochs):
        for batch in dataloader:
            batch_size = int(batch["input_ids"].shape[0])
            step_metrics = train_step(
                model,
                batch,
                optimizer,
                device,
                precision=precision,
            )
            step += 1
            examples_seen += batch_size
            if step == 1 or step % log_interval_steps == 0 or step == total_steps:
                record_metric(
                    {
                        "mode": "train",
                        "epoch": epoch + 1,
                        "step": step,
                        "loss": step_metrics.loss,
                        "token_accuracy": step_metrics.token_accuracy,
                        "learning_rate": step_metrics.learning_rate,
                        "target_tokens": step_metrics.target_tokens,
                        "examples_seen": examples_seen,
                    }
                )
                print(
                    f"epoch={epoch + 1} step={step} "
                    f"loss={step_metrics.loss:.6f} "
                    f"token_accuracy={step_metrics.token_accuracy:.4f}"
                )

        for metric in evaluate_splits(
            model,
            tokenizer,
            evaluation_splits,
            device,
            batch_size=evaluation_batch_size,
            epoch=epoch + 1,
            step=step,
        ):
            record_metric(metric)

    return TrainingResult(metrics=metrics, predictions=None)


# эвал луп?
def generate_answers(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[Example],
    device: torch.device,
) -> list[str]:
    tokenizer.padding_side = "left"

    conversations = [
        [
            {"role": "system", "content": "Answer with the value only."},
            {"role": "user", "content": example.rendered_question},
        ]
        for example in examples
    ]

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
) -> list[str]:
    """Generate answers without materializing an entire evaluation split on device."""

    if batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    predictions: list[str] = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        predictions.extend(generate_answers(model, tokenizer, batch_examples, device))
    return predictions


def evaluate_splits(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    splits: Mapping[str, Sequence[Example]],
    device: torch.device,
    *,
    batch_size: int,
    epoch: int,
    step: int,
) -> list[dict[str, Any]]:
    """Measure generated-answer exact match for each named evaluation split."""

    metrics: list[dict[str, Any]] = []
    for split_name, examples in splits.items():
        if not examples:
            raise ValueError(f"evaluation split {split_name!r} must not be empty")
        predictions = generate_answers_batched(
            model,
            tokenizer,
            examples,
            device,
            batch_size,
        )
        accuracy = exact_match_accuracy(predictions, examples)
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
    return metrics


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


def report_metric_to_clearml(logger: Logger, metric: Mapping[str, Any]) -> None:
    """Publish one trainer metric record using stable ClearML chart names."""

    iteration = int(metric["step"])
    mode = str(metric["mode"])
    split = metric.get("split")
    series_prefix = mode if split is None else f"{mode}/{split}"
    scalar_series = {
        "loss": ("loss", series_prefix),
        "token_accuracy": ("accuracy", f"{series_prefix}/token"),
        "exact_match_accuracy": ("accuracy", f"{series_prefix}/answer_exact_match"),
        "learning_rate": ("optimization", f"{series_prefix}/learning_rate"),
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
    final_metric = result.metrics[-1]
    for key in ("loss", "token_accuracy", "exact_match_accuracy"):
        value = final_metric.get(key)
        if value is not None:
            logger.report_single_value(name=f"final/{key}", value=float(value))


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

    evaluation_splits: dict[str, Sequence[Example]] | None = None
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
        batch = SFTCollator(tokenizer)([dataset[0], dataset[1]])
        inspect_sft_batch(tokenizer, batch)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.smoke_steps is None,
        collate_fn=SFTCollator(tokenizer),
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
    )

    metrics_path = output_dir / "metrics.jsonl" if output_dir is not None else None
    clearml_logger = task.get_logger() if task is not None else None
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
    )

    if clearml_logger is not None:
        report_final_metrics(clearml_logger, result)

    if output_dir is not None and config.runtime.save_model:
        model_output_dir = output_dir / "final_model"
        cast(Any, model).save_pretrained(model_output_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_output_dir)
    if output_dir is not None:
        if result.predictions is not None and smoke_examples is not None:
            save_smoke_predictions(output_dir, result.predictions, smoke_examples)
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
