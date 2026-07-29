#!/usr/bin/env python3
"""Validate the curated query-template registry."""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "data" / "templates"
TEMPLATE_FILES = {
    "born_in_city.jsonl": "GET_BIRTH_CITY",
    "born_in_date.jsonl": "GET_BIRTH_DATE",
    "lives_in_city.jsonl": "GET_RESIDENCE_CITY",
    "profession.jsonl": "GET_OCCUPATION",
}
REQUIRED_FIELDS = {
    "operation_id",
    "template_family_id",
    "template_id",
    "answer_format_id",
    "template",
}
EXPECTED_FAMILIES = {
    "direct_question",
    "nominal_attribute",
    "imperative",
    "profile_field",
}
EXPECTED_TEMPLATES_PER_FAMILY = 2
EXPECTED_ANSWER_FORMAT = "canonical_value"
SEQUENCE_SIMILARITY_LIMIT = 0.88
COMBINED_SEQUENCE_FLOOR = 0.65
TOKEN_JACCARD_LIMIT = 0.80
TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    text = text.lower().replace("{person}", "person")
    return " ".join(TOKEN_RE.findall(text))


def token_set(text: str) -> set[str]:
    return set(normalize(text).split())


def token_jaccard(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    union = left_tokens | right_tokens
    if not union:
        return 1.0
    return len(left_tokens & right_tokens) / len(union)


def similarity_scores(left: str, right: str) -> tuple[float, float]:
    return (
        SequenceMatcher(None, normalize(left), normalize(right)).ratio(),
        token_jaccard(left, right),
    )


def is_near_duplicate(left: str, right: str) -> bool:
    sequence_score, jaccard_score = similarity_scores(left, right)
    return sequence_score >= SEQUENCE_SIMILARITY_LIMIT or (
        sequence_score >= COMBINED_SEQUENCE_FLOOR
        and jaccard_score >= TOKEN_JACCARD_LIMIT
    )


def load_jsonl(path: Path) -> tuple[list[tuple[int, dict[str, object]]], list[str]]:
    records: list[tuple[int, dict[str, object]]] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            errors.append(f"{path.name}:{line_number}: blank lines are not allowed")
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(record, dict):
            errors.append(f"{path.name}:{line_number}: expected a JSON object")
            continue
        records.append((line_number, record))
    return records, errors


def validate_file(path: Path, expected_operation: str) -> tuple[list[str], list[str]]:
    records, errors = load_jsonl(path)
    warnings: list[str] = []

    template_ids: set[str] = set()
    family_counts: dict[str, int] = {}
    templates: list[tuple[str, str, str]] = []

    for line_number, record in records:
        fields = set(record)
        if fields != REQUIRED_FIELDS:
            missing = REQUIRED_FIELDS - fields
            extra = fields - REQUIRED_FIELDS
            details = []
            if missing:
                details.append(f"missing={sorted(missing)}")
            if extra:
                details.append(f"extra={sorted(extra)}")
            errors.append(f"{path.name}:{line_number}: schema mismatch ({', '.join(details)})")
            continue

        operation_id = record["operation_id"]
        family_id = record["template_family_id"]
        template_id = record["template_id"]
        answer_format_id = record["answer_format_id"]
        template = record["template"]
        values = (operation_id, family_id, template_id, answer_format_id, template)
        if not all(isinstance(value, str) and value for value in values):
            errors.append(f"{path.name}:{line_number}: all fields must be non-empty strings")
            continue
        if operation_id != expected_operation:
            errors.append(
                f"{path.name}:{line_number}: expected operation {expected_operation!r}, "
                f"found {operation_id!r}"
            )
        if family_id not in EXPECTED_FAMILIES:
            errors.append(f"{path.name}:{line_number}: unknown family id {family_id!r}")
        if answer_format_id != EXPECTED_ANSWER_FORMAT:
            errors.append(
                f"{path.name}:{line_number}: expected answer format "
                f"{EXPECTED_ANSWER_FORMAT!r}, found {answer_format_id!r}"
            )
        if template_id in template_ids:
            errors.append(f"{path.name}:{line_number}: duplicate template id {template_id!r}")
        if template.count("{person}") != 1:
            errors.append(
                f"{path.name}:{line_number}: template must contain exactly one {{person}} slot"
            )

        template_ids.add(template_id)
        family_counts[family_id] = family_counts.get(family_id, 0) + 1
        templates.append((family_id, template_id, template))

    if set(family_counts) != EXPECTED_FAMILIES:
        errors.append(
            f"{path.name}: expected families {sorted(EXPECTED_FAMILIES)}, "
            f"found {sorted(family_counts)}"
        )
    for family_id in EXPECTED_FAMILIES:
        count = family_counts.get(family_id, 0)
        if count != EXPECTED_TEMPLATES_PER_FAMILY:
            errors.append(
                f"{path.name}: family {family_id!r} must contain exactly "
                f"{EXPECTED_TEMPLATES_PER_FAMILY} templates, found {count}"
            )

    for (left_family, left_id, left), (right_family, right_id, right) in combinations(
        templates, 2
    ):
        sequence_score, jaccard_score = similarity_scores(left, right)
        if normalize(left) == normalize(right):
            errors.append(
                f"{path.name}: duplicate normalized templates {left_id!r} and {right_id!r}"
            )
            continue
        if left_family == right_family:
            continue
        if is_near_duplicate(left, right):
            errors.append(
                f"{path.name}: near-duplicate candidates {left_id!r} and {right_id!r} "
                f"(sequence={sequence_score:.2f}, token_jaccard={jaccard_score:.2f})"
            )
        elif sequence_score >= 0.72 and jaccard_score >= 0.60:
            warnings.append(
                f"{path.name}: review pair {left_id!r} and {right_id!r} "
                f"(sequence={sequence_score:.2f}, token_jaccard={jaccard_score:.2f})"
            )

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat review-threshold similarity warnings as errors.",
    )
    args = parser.parse_args()

    all_errors: list[str] = []
    all_warnings: list[str] = []
    global_template_ids: set[str] = set()

    for filename, expected_operation in TEMPLATE_FILES.items():
        path = TEMPLATE_DIR / filename
        if not path.exists():
            all_errors.append(f"{filename}: missing template file")
            continue
        errors, warnings = validate_file(path, expected_operation)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

        records, _ = load_jsonl(path)
        for _, record in records:
            template_id = record.get("template_id")
            if isinstance(template_id, str):
                if template_id in global_template_ids:
                    all_errors.append(f"duplicate global template id {template_id!r}")
                global_template_ids.add(template_id)

    for warning in all_warnings:
        print(f"WARNING: {warning}")
    for error in all_errors:
        print(f"ERROR: {error}")

    if all_errors or (args.strict and all_warnings):
        return 1

    print(
        f"Validated {len(global_template_ids)} templates in "
        f"{len(TEMPLATE_FILES)} operation files."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
