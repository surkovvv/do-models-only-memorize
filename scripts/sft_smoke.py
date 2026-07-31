# тут загрузить зависимости
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

import torch
import json


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

# датадоудер, как оказалось, не особо нужен. а вот колейтор(то, что собирает батч) - нужен
class SFTCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data: list[Example]) -> dict[str, torch.Tensor]:
        conversations = [
            [
                {"role": "system", "content": "..."},
                {"role": "user", "content": elem.rendered_question},
                {"role": "assistant", "content": elem.canonical_answer},
            ]
            for elem in data
        ]
        batch = self.tokenizer.apply_chat_template(
            conversations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_dict=True,
        )
        labels = batch["input_ids"].clone()

        # Padding не участвует в loss.
        labels[batch["attention_mask"] == 0] = -100

        # Qwen не возвращает корректную assistant mask для своего chat template,
        # поэтому определяем границу ответа по длине prompt с assistant header.
        prompt_token_ids: list[list[int]] = self.tokenizer.apply_chat_template(
            [conversation[:-1] for conversation in conversations],
            add_generation_prompt=True,
            tokenize=True,
        )
        for row, prompt_ids in enumerate(prompt_token_ids):
            sequence_length = int(batch["attention_mask"][row].sum())
            if len(prompt_ids) > sequence_length:
                raise ValueError("prompt was truncated before the assistant answer")

            padding_length = batch["input_ids"].shape[1] - sequence_length
            prompt_start = padding_length if self.tokenizer.padding_side == "left" else 0
            labels[row, prompt_start : prompt_start + len(prompt_ids)] = -100

        batch["labels"] = labels
        return batch

        

# трейн луп
# эвал луп?

# cli main - загрузка модели, установление всяких оптимизаторов, запуск обучения
def main():
    path_to_data = (
        Path(__file__).resolve().parents[1]
        / "data/generated/h1/seed_30072026/train.jsonl"
    )
    with path_to_data.open(encoding="utf-8") as f:
        data = [Example(**json.loads(line)) for line in f]

    dataset = FactDataset(data)
    print(dataset[0])

if __name__ == "__main__":
    main()
