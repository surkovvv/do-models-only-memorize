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
EVAL_EXACT_RECALL = "eval_exact_recall"
EVAL_SEEN_FAMILY_NEW_TEMPLATE = "eval_seen_family_new_template"
EVAL_HELDOUT_FAMILY = "eval_heldout_family"
EVALUATION_SPLITS = (
    EVAL_EXACT_RECALL,
    EVAL_SEEN_FAMILY_NEW_TEMPLATE,
    EVAL_HELDOUT_FAMILY,
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
    sft_template_suffix: str,
    seen_family_eval_template_suffix: str,
    evaluation_people_count: int,
    evaluation_sample_seed: str,
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
    if not sft_template_suffix or not seen_family_eval_template_suffix:
        raise ValueError("template suffixes must be non-empty")
    if sft_template_suffix == seen_family_eval_template_suffix:
        raise ValueError("SFT and seen-family eval template suffixes must differ")
    if not 0 < evaluation_people_count <= len(world.people):
        raise ValueError(
            "evaluation_people_count must be positive and no larger than the world"
        )
    if not evaluation_sample_seed:
        raise ValueError("evaluation_sample_seed must be non-empty")

    operations = tuple(OPERATION_TO_RELATION)
    train_families = tuple(sorted(train_family_ids))
    sft_templates: dict[tuple[str, str], QueryTemplate] = {}
    seen_eval_templates: dict[tuple[str, str], QueryTemplate] = {}
    for operation_id in operations:
        for family_id in train_families:
            family_templates = [
                template
                for template in templates
                if template.operation_id == operation_id
                and template.template_family_id == family_id
            ]
            sft_matches = [
                template
                for template in family_templates
                if template.template_id.endswith(sft_template_suffix)
            ]
            seen_eval_matches = [
                template
                for template in family_templates
                if template.template_id.endswith(seen_family_eval_template_suffix)
            ]
            if len(sft_matches) != 1 or len(seen_eval_matches) != 1:
                raise ValueError(
                    f"{operation_id}/{family_id} must have exactly one SFT and "
                    "one seen-family eval template"
                )
            sft_templates[(operation_id, family_id)] = sft_matches[0]
            seen_eval_templates[(operation_id, family_id)] = seen_eval_matches[0]

    heldout_templates: dict[str, tuple[QueryTemplate, ...]] = {}
    for operation_id in operations:
        matches = tuple(
            sorted(
                (
                    template
                    for template in templates
                    if template.operation_id == operation_id
                    and template.template_family_id in test_family_ids
                ),
                key=lambda template: template.template_id,
            )
        )
        if not matches:
            raise ValueError(f"{operation_id} has no held-out-family templates")
        heldout_templates[operation_id] = matches

    facts = {
        (fact.subject_id, fact.relation): fact
        for fact in world.facts
    }
    def render_example(person_id: str, template: QueryTemplate, split: str) -> DatasetExample:
        person = world.person(person_id)
        relation_id = OPERATION_TO_RELATION[template.operation_id]
        fact = facts[(person.id, relation_id)]
        return DatasetExample(
            example_id=f"{world.id}.{split}.{fact.id}.{template.template_id}",
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

    train = [
        render_example(person.id, sft_templates[(operation_id, family_id)], "train")
        for person in world.people
        for operation_id in operations
        for family_id in train_families
    ]

    evaluation_person_ids = world.seed.ordered(
        f"h1-evaluation-people\0{evaluation_sample_seed}",
        tuple(person.id for person in world.people),
    )[:evaluation_people_count]
    exact_recall: list[DatasetExample] = []
    seen_family_new_template: list[DatasetExample] = []
    heldout_family: list[DatasetExample] = []
    for person_index, person_id in enumerate(evaluation_person_ids):
        for operation_index, operation_id in enumerate(operations):
            variant_index = person_index + operation_index
            family_id = train_families[variant_index % len(train_families)]
            exact_recall.append(
                render_example(
                    person_id,
                    sft_templates[(operation_id, family_id)],
                    EVAL_EXACT_RECALL,
                )
            )
            seen_family_new_template.append(
                render_example(
                    person_id,
                    seen_eval_templates[(operation_id, family_id)],
                    EVAL_SEEN_FAMILY_NEW_TEMPLATE,
                )
            )
            operation_heldout_templates = heldout_templates[operation_id]
            heldout_family.append(
                render_example(
                    person_id,
                    operation_heldout_templates[
                        variant_index % len(operation_heldout_templates)
                    ],
                    EVAL_HELDOUT_FAMILY,
                )
            )

    test = exact_recall + seen_family_new_template + heldout_family

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
    sft_template_suffix: str,
    seen_family_eval_template_suffix: str,
    evaluation_people_count: int,
) -> list[str]:
    errors: list[str] = []
    all_records = train_records + test_records
    records_by_split = {
        "train": train_records,
        **{
            split: [record for record in test_records if record.get("split") == split]
            for split in EVALUATION_SPLITS
        },
    }

    for expected_file, records in (("train", train_records), ("test", test_records)):
        for index, record in enumerate(records, 1):
            if set(record) != EXAMPLE_FIELDS:
                errors.append(
                    f"{expected_file}.jsonl:{index}: example schema mismatch"
                )
                continue
            if not all(
                isinstance(record[field], str) and record[field]
                for field in EXAMPLE_FIELDS
            ):
                errors.append(
                    f"{expected_file}.jsonl:{index}: all fields must be "
                    "non-empty strings"
                )
            split = record["split"]
            valid_split = (
                split == "train"
                if expected_file == "train"
                else split in EVALUATION_SPLITS
            )
            if not valid_split:
                errors.append(
                    f"{expected_file}.jsonl:{index}: split field is {split!r}"
                )
            if record["fact_id"] != (
                f"{record['person_id']}.{record['relation_id']}"
            ):
                errors.append(
                    f"{expected_file}.jsonl:{index}: fact_id does not match "
                    "person_id + relation_id"
                )
            if record["canonical_answer"] != record["fact_value"]:
                errors.append(
                    f"{expected_file}.jsonl:{index}: answer differs from fact value"
                )
            expected_relation = OPERATION_TO_RELATION.get(record["operation_id"])
            if record["relation_id"] != expected_relation:
                errors.append(
                    f"{expected_file}.jsonl:{index}: relation does not match operation"
                )
            expected_example_id = (
                f"{record['world_id']}.{split}.{record['fact_id']}.{record['template_id']}"
            )
            if record["example_id"] != expected_example_id:
                errors.append(
                    f"{expected_file}.jsonl:{index}: example_id does not match provenance"
                )

    for split in ("train", EVAL_EXACT_RECALL, EVAL_SEEN_FAMILY_NEW_TEMPLATE):
        families = {record.get("template_family_id") for record in records_by_split[split]}
        if families != train_family_ids:
            errors.append(f"{split} families differ from manifest: {sorted(families)}")
    heldout_families = {
        record.get("template_family_id")
        for record in records_by_split[EVAL_HELDOUT_FAMILY]
    }
    if heldout_families != test_family_ids:
        errors.append(
            "eval_heldout_family families differ from manifest: "
            f"{sorted(heldout_families)}"
        )

    for split in ("train", EVAL_EXACT_RECALL):
        if any(
            not str(record.get("template_id", "")).endswith(sft_template_suffix)
            for record in records_by_split[split]
        ):
            errors.append(f"{split} contains a non-SFT template variant")
    if any(
        not str(record.get("template_id", "")).endswith(
            seen_family_eval_template_suffix
        )
        for record in records_by_split[EVAL_SEEN_FAMILY_NEW_TEMPLATE]
    ):
        errors.append("eval_seen_family_new_template contains an SFT template variant")

    example_ids = [record.get("example_id") for record in all_records]
    if len(example_ids) != len(set(example_ids)):
        errors.append("example_id values are not globally unique")

    for split, records in records_by_split.items():
        fact_template_pairs = [
            (record.get("world_id"), record.get("fact_id"), record.get("template_id"))
            for record in records
        ]
        if len(fact_template_pairs) != len(set(fact_template_pairs)):
            errors.append(f"a fact/template pair occurs more than once in {split}")

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

    fact_ids_by_split = {
        split: {(record.get("world_id"), record.get("fact_id")) for record in records}
        for split, records in records_by_split.items()
    }
    evaluation_fact_ids = fact_ids_by_split[EVAL_EXACT_RECALL]
    if any(
        fact_ids_by_split[split] != evaluation_fact_ids
        for split in EVALUATION_SPLITS[1:]
    ):
        errors.append("evaluation slices do not cover the same facts")
    if not evaluation_fact_ids <= fact_ids_by_split["train"]:
        errors.append("evaluation contains facts absent from train")

    evaluation_person_ids = {
        record.get("person_id") for record in records_by_split[EVAL_EXACT_RECALL]
    }
    if len(evaluation_person_ids) != evaluation_people_count:
        errors.append("evaluation person count differs from the manifest")
    expected_operations = set(OPERATION_TO_RELATION)
    for split in EVALUATION_SPLITS:
        operations_by_person: dict[object, set[object]] = {}
        for record in records_by_split[split]:
            operations_by_person.setdefault(record.get("person_id"), set()).add(
                record.get("operation_id")
            )
        if any(operations != expected_operations for operations in operations_by_person.values()):
            errors.append(f"{split} does not cover every operation per person")

    train_pairs = {
        (record.get("world_id"), record.get("fact_id"), record.get("template_id"))
        for record in train_records
    }
    exact_pairs = {
        (record.get("world_id"), record.get("fact_id"), record.get("template_id"))
        for record in records_by_split[EVAL_EXACT_RECALL]
    }
    if not exact_pairs <= train_pairs:
        errors.append("exact-recall examples are not present in SFT")
    train_template_ids = {record.get("template_id") for record in train_records}
    seen_eval_template_ids = {
        record.get("template_id")
        for record in records_by_split[EVAL_SEEN_FAMILY_NEW_TEMPLATE]
    }
    if train_template_ids & seen_eval_template_ids:
        errors.append("seen-family eval template IDs leak into SFT")

    exact_by_fact = {
        (record.get("world_id"), record.get("fact_id")): record
        for record in records_by_split[EVAL_EXACT_RECALL]
    }
    seen_by_fact = {
        (record.get("world_id"), record.get("fact_id")): record
        for record in records_by_split[EVAL_SEEN_FAMILY_NEW_TEMPLATE]
    }
    for fact_key, exact_record in exact_by_fact.items():
        seen_record = seen_by_fact.get(fact_key)
        if seen_record is not None and (
            exact_record.get("template_family_id")
            != seen_record.get("template_family_id")
        ):
            errors.append(f"exact/seen-family pair differs in family for {fact_key!r}")

    return errors


def counts_by_operation(
    *record_groups: Iterable[dict[str, Any] | DatasetExample],
) -> dict[str, dict[str, int]]:
    def operation_id(record: dict[str, Any] | DatasetExample) -> str:
        if isinstance(record, DatasetExample):
            return record.operation_id
        return str(record.get("operation_id"))

    def split_id(record: dict[str, Any] | DatasetExample) -> str:
        if isinstance(record, DatasetExample):
            return record.split
        return str(record.get("split"))

    counts: dict[str, Counter[str]] = {}
    for records in record_groups:
        for record in records:
            counts.setdefault(split_id(record), Counter())[operation_id(record)] += 1
    return {
        split: dict(sorted(operation_counts.items()))
        for split, operation_counts in sorted(counts.items())
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
