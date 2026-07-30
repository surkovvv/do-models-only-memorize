"""Build provenance-rich H1 examples from a generated micro-world."""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from world import World


OPERATION_TO_RELATION = {
    "GET_BIRTH_CITY": "birth_city",
    "GET_BIRTH_DATE": "birth_date",
    "GET_RESIDENCE_CITY": "residence_city",
    "GET_OCCUPATION": "occupation",
}
EXAMPLE_FIELDS = {
    "example_id",
    "world_id",
    "world_seed",
    "person_id",
    "person_name",
    "fact_id",
    "relation_id",
    "fact_value",
    "operation_id",
    "template_family_id",
    "template_id",
    "rendered_question",
    "canonical_answer",
    "answer_format_id",
    "split",
}
SOURCE_TEMPLATE_FILES = (
    "born_in_city.jsonl",
    "born_in_date.jsonl",
    "lives_in_city.jsonl",
    "profession.jsonl",
)


@dataclass(frozen=True, slots=True)
class QueryTemplate:
    operation_id: str
    template_family_id: str
    template_id: str
    answer_format_id: str
    template: str

    @classmethod
    def from_dict(cls, record: dict[str, object]) -> "QueryTemplate":
        try:
            values = {
                field: record[field]
                for field in (
                    "operation_id",
                    "template_family_id",
                    "template_id",
                    "answer_format_id",
                    "template",
                )
            }
        except KeyError as exc:
            raise ValueError(f"template is missing field {exc.args[0]!r}") from exc
        if not all(isinstance(value, str) and value for value in values.values()):
            raise ValueError("all template fields must be non-empty strings")
        if values["operation_id"] not in OPERATION_TO_RELATION:
            raise ValueError(f"unknown operation {values['operation_id']!r}")
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class DatasetExample:
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

    def to_dict(self) -> dict[str, str]:
        return {
            field: getattr(self, field)
            for field in (
                "example_id",
                "world_id",
                "world_seed",
                "person_id",
                "person_name",
                "fact_id",
                "relation_id",
                "fact_value",
                "operation_id",
                "template_family_id",
                "template_id",
                "rendered_question",
                "canonical_answer",
                "answer_format_id",
                "split",
            )
        }


def generate_h1_examples(
    world: World,
    template_records: Iterable[dict[str, object]],
    *,
    train_family_ids: set[str],
    test_family_ids: set[str],
) -> tuple[tuple[DatasetExample, ...], tuple[DatasetExample, ...]]:
    overlap = train_family_ids & test_family_ids
    if overlap:
        raise ValueError(f"template families leak across splits: {sorted(overlap)}")

    templates = tuple(QueryTemplate.from_dict(record) for record in template_records)
    assigned_families = train_family_ids | test_family_ids
    registry_families = {template.template_family_id for template in templates}
    if assigned_families != registry_families:
        raise ValueError(
            "family assignment does not match registry: "
            f"assigned={sorted(assigned_families)}, "
            f"registry={sorted(registry_families)}"
        )

    facts = {
        (fact.subject_id, fact.relation): fact
        for fact in world.facts
    }
    train: list[DatasetExample] = []
    test: list[DatasetExample] = []

    for person in world.people:
        for template in templates:
            relation_id = OPERATION_TO_RELATION[template.operation_id]
            fact = facts[(person.id, relation_id)]
            split = (
                "train"
                if template.template_family_id in train_family_ids
                else "test"
            )
            example = DatasetExample(
                example_id=f"{world.id}.{fact.id}.{template.template_id}",
                world_id=world.id,
                world_seed=world.seed.value,
                person_id=person.id,
                person_name=person.name,
                fact_id=fact.id,
                relation_id=relation_id,
                fact_value=fact.value,
                operation_id=template.operation_id,
                template_family_id=template.template_family_id,
                template_id=template.template_id,
                rendered_question=template.template.format(person=person.name),
                canonical_answer=fact.value,
                answer_format_id=template.answer_format_id,
                split=split,
            )
            (train if split == "train" else test).append(example)

    return tuple(train), tuple(test)


def write_jsonl(path: Path, examples: Iterable[DatasetExample]) -> None:
    rendered = "".join(
        json.dumps(example.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
        for example in examples
    )
    path.write_text(rendered, encoding="utf-8")


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            errors.append(f"{path.name}:{line_number}: blank line")
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(record, dict):
            errors.append(f"{path.name}:{line_number}: expected JSON object")
            continue
        records.append(record)
    return records, errors


def validate_example_records(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    *,
    train_family_ids: set[str],
    test_family_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    all_records = train_records + test_records

    for expected_split, records in (("train", train_records), ("test", test_records)):
        for index, record in enumerate(records, 1):
            if set(record) != EXAMPLE_FIELDS:
                errors.append(
                    f"{expected_split}.jsonl:{index}: example schema mismatch"
                )
                continue
            if not all(
                isinstance(record[field], str) and record[field]
                for field in EXAMPLE_FIELDS
            ):
                errors.append(
                    f"{expected_split}.jsonl:{index}: all fields must be "
                    "non-empty strings"
                )
            if record["split"] != expected_split:
                errors.append(
                    f"{expected_split}.jsonl:{index}: split field is "
                    f"{record['split']!r}"
                )
            if record["fact_id"] != (
                f"{record['person_id']}.{record['relation_id']}"
            ):
                errors.append(
                    f"{expected_split}.jsonl:{index}: fact_id does not match "
                    "person_id + relation_id"
                )
            if record["canonical_answer"] != record["fact_value"]:
                errors.append(
                    f"{expected_split}.jsonl:{index}: answer differs from fact value"
                )
            expected_relation = OPERATION_TO_RELATION.get(record["operation_id"])
            if record["relation_id"] != expected_relation:
                errors.append(
                    f"{expected_split}.jsonl:{index}: relation does not match operation"
                )

    train_families = {
        value
        for record in train_records
        if isinstance((value := record.get("template_family_id")), str)
    }
    test_families = {
        value
        for record in test_records
        if isinstance((value := record.get("template_family_id")), str)
    }
    if train_families != train_family_ids:
        errors.append(
            f"train families differ from manifest: {sorted(train_families)}"
        )
    if test_families != test_family_ids:
        errors.append(f"test families differ from manifest: {sorted(test_families)}")
    leaked = train_families & test_families
    if leaked:
        errors.append(f"template families leak across train and test: {sorted(leaked)}")

    example_ids = [record.get("example_id") for record in all_records]
    if len(example_ids) != len(set(example_ids)):
        errors.append("example_id values are not globally unique")

    fact_template_pairs = [
        (record.get("world_id"), record.get("fact_id"), record.get("template_id"))
        for record in all_records
    ]
    if len(fact_template_pairs) != len(set(fact_template_pairs)):
        errors.append("a fact/template pair occurs more than once")

    fact_definitions: dict[tuple[object, object], tuple[object, object, object]] = {}
    for record in all_records:
        fact_key = (record.get("world_id"), record.get("fact_id"))
        definition = (
            record.get("person_id"),
            record.get("relation_id"),
            record.get("fact_value"),
        )
        previous = fact_definitions.setdefault(fact_key, definition)
        if previous != definition:
            errors.append(f"inconsistent definition of fact {fact_key!r}")

    train_fact_ids = {
        (record.get("world_id"), record.get("fact_id")) for record in train_records
    }
    test_fact_ids = {
        (record.get("world_id"), record.get("fact_id")) for record in test_records
    }
    if train_fact_ids != test_fact_ids:
        errors.append("train and test do not cover the same facts")

    return errors


def counts_by_operation(
    train_records: Iterable[dict[str, Any] | DatasetExample],
    test_records: Iterable[dict[str, Any] | DatasetExample],
) -> dict[str, dict[str, int]]:
    def operation_id(record: dict[str, Any] | DatasetExample) -> str:
        if isinstance(record, DatasetExample):
            return record.operation_id
        return str(record.get("operation_id"))

    return {
        "train": dict(
            sorted(Counter(operation_id(record) for record in train_records).items())
        ),
        "test": dict(
            sorted(Counter(operation_id(record) for record in test_records).items())
        ),
    }


def input_hashes(root: Path, split_manifest: Path) -> dict[str, str]:
    paths = [
        root / "data" / "strange_names.txt",
        root / "data" / "dates.txt",
        root / "data" / "professions.txt",
        root / "data" / "towns.txt",
        *(
            root / "data" / "templates" / filename
            for filename in SOURCE_TEMPLATE_FILES
        ),
        split_manifest,
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
