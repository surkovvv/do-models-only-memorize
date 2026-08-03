from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, cast

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_config import ExperimentConfig, config_from_mapping, load_experiment_config  # noqa: E402

if TYPE_CHECKING:
    from clearml import Task


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
                enable_thinking=False
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
                enable_thinking=False
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
) -> float:
    """v0: stupid forward backward, no accumulation etc"""
    model.train()
    optimizer.zero_grad()
    for k, v in batch.items():
        batch[k] = v.to(device)

    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    return loss.item()


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
) -> None:
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
            batch_loss = train_step(model, first_batch, optimizer, device)
            predictions = generate_answers(model, tokenizer, smoke_examples, device)
            accuracy = exact_match_accuracy(predictions, smoke_examples)
            print(
                f"smoke_step={step + 1}/{smoke_steps} "
                f"loss={batch_loss:.6f} exact_match_accuracy={accuracy:.4f}"
            )
            log_predictions(predictions, smoke_examples, prediction_log_limit)
        return

    step = 0
    for epoch in range(epochs):
        for batch in dataloader:
            batch_loss = train_step(model, batch, optimizer, device)
            print(f"epoch={epoch + 1} step={step} loss={batch_loss:.6f}")
            step += 1


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

    task: Task = Task.init(
        project_name=config.tracking.project_name,
        task_name=config.tracking.task_name or config.experiment_name,
        output_uri=config.tracking.output_uri,
        tags=config.tracking.tags,
        reuse_last_task_id=False,
    )
    connected_config = task.connect(config.to_dict(), name="config")
    return task, config_from_mapping(connected_config)


def run(config: ExperimentConfig) -> None:
    path_to_data = ROOT / config.data.train_path
    if Path(config.data.train_path).is_absolute():
        path_to_data = Path(config.data.train_path)

    with path_to_data.open(encoding="utf-8") as f:
        data = [Example(**json.loads(line)) for line in f]

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

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=config.model.local_files_only,
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
        local_files_only=config.model.local_files_only,
    )
    device = resolve_device(config.runtime.device)
    model = model.to(device)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.training.learning_rate,
    )

    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        epochs=config.training.epochs,
        smoke_steps=config.training.smoke_steps,
        tokenizer=tokenizer,
        smoke_examples=smoke_examples,
        prediction_log_limit=config.runtime.prediction_log_limit,
    )


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config, args.overrides)
    task, config = init_tracking(config)
    try:
        # ClearML Agent/UI may have overridden the connected configuration.
        torch.manual_seed(config.training.seed)
        run(config)
    finally:
        if task is not None:
            task.close()


if __name__ == "__main__":
    main()
