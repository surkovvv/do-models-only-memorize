#!/usr/bin/env python3
"""Validate the global template-family split used to test H1."""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from validate_templates import TEMPLATE_DIR, TEMPLATE_FILES, load_jsonl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_PATH = ROOT / "data" / "splits" / "h1_template_families.json"
TARGET_COUNTS = {8: (6, 2)}
REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "hypothesis",
    "split_unit",
    "train_family_ids",
    "test_family_ids",
    "sft_template_suffix",
    "seen_family_eval_template_suffix",
    "evaluation_people_count",
    "evaluation_sample_seed",
}


@dataclass(frozen=True, slots=True)
class SplitSummary:
    operation_id: str
    total_count: int
    train_count: int
    test_count: int


def load_manifest(path: Path = DEFAULT_SPLIT_PATH) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("split manifest must contain a JSON object")
    return manifest


def load_registry() -> tuple[list[dict[str, object]], list[str]]:
    records: list[dict[str, object]] = []
    errors: list[str] = []
    for filename in TEMPLATE_FILES:
        file_records, file_errors = load_jsonl(TEMPLATE_DIR / filename)
        errors.extend(file_errors)
        records.extend(record for _, record in file_records)
    return records, errors


def validate_h1_split(
    records: Iterable[dict[str, object]],
    manifest: dict[str, object],
) -> tuple[list[SplitSummary], list[str]]:
    records = list(records)
    errors: list[str] = []

    fields = set(manifest)
    if fields != REQUIRED_MANIFEST_FIELDS:
        missing = REQUIRED_MANIFEST_FIELDS - fields
        extra = fields - REQUIRED_MANIFEST_FIELDS
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"extra={sorted(extra)}")
        return [], [f"manifest schema mismatch ({', '.join(details)})"]

    if manifest["schema_version"] != 2:
        errors.append("schema_version must be 2")
    if manifest["hypothesis"] != "H1":
        errors.append("hypothesis must be 'H1'")
    if manifest["split_unit"] != "template_family_id_and_template_id":
        errors.append("split_unit must be 'template_family_id_and_template_id'")

    sft_suffix = _non_empty_string(
        manifest["sft_template_suffix"], "sft_template_suffix", errors
    )
    seen_eval_suffix = _non_empty_string(
        manifest["seen_family_eval_template_suffix"],
        "seen_family_eval_template_suffix",
        errors,
    )
    if sft_suffix == seen_eval_suffix:
        errors.append("SFT and seen-family eval template suffixes must differ")
    evaluation_people_count = manifest["evaluation_people_count"]
    if not isinstance(evaluation_people_count, int) or evaluation_people_count <= 0:
        errors.append("evaluation_people_count must be a positive integer")
    _non_empty_string(
        manifest["evaluation_sample_seed"], "evaluation_sample_seed", errors
    )

    train_families = _string_set(
        manifest["train_family_ids"], "train_family_ids", errors
    )
    test_families = _string_set(
        manifest["test_family_ids"], "test_family_ids", errors
    )

    leaked_families = train_families & test_families
    if leaked_families:
        errors.append(
            "template families leak across train and test: "
            f"{sorted(leaked_families)}"
        )

    registry_families = {
        record.get("template_family_id")
        for record in records
        if isinstance(record.get("template_family_id"), str)
    }
    assigned_families = train_families | test_families
    if assigned_families != registry_families:
        missing = registry_families - assigned_families
        unknown = assigned_families - registry_families
        if missing:
            errors.append(f"unassigned registry families: {sorted(missing)}")
        if unknown:
            errors.append(f"manifest contains unknown families: {sorted(unknown)}")

    template_ids = [
        record.get("template_id")
        for record in records
        if isinstance(record.get("template_id"), str)
    ]
    duplicate_ids = sorted(
        template_id
        for template_id, count in Counter(template_ids).items()
        if count > 1
    )
    if duplicate_ids:
        errors.append(f"duplicate template ids: {duplicate_ids}")

    summaries: list[SplitSummary] = []
    for operation_id in TEMPLATE_FILES.values():
        operation_records = [
            record
            for record in records
            if record.get("operation_id") == operation_id
        ]
        train_count = sum(
            record.get("template_family_id") in train_families
            for record in operation_records
        )
        test_count = sum(
            record.get("template_family_id") in test_families
            for record in operation_records
        )
        summary = SplitSummary(
            operation_id=operation_id,
            total_count=len(operation_records),
            train_count=train_count,
            test_count=test_count,
        )
        summaries.append(summary)

        if train_count + test_count != len(operation_records):
            errors.append(
                f"{operation_id}: split does not cover every template "
                f"({train_count} train + {test_count} test != "
                f"{len(operation_records)} total)"
            )

        expected = TARGET_COUNTS.get(len(operation_records))
        if expected is None:
            errors.append(
                f"{operation_id}: no H1 ratio is defined for "
                f"{len(operation_records)} templates"
            )
        elif (train_count, test_count) != expected:
            errors.append(
                f"{operation_id}: expected {expected[0]} train / "
                f"{expected[1]} test, found {train_count} / {test_count}"
            )

        for family_id in train_families:
            family_records = [
                record
                for record in operation_records
                if record.get("template_family_id") == family_id
            ]
            sft_records = [
                record
                for record in family_records
                if isinstance(record.get("template_id"), str)
                and str(record["template_id"]).endswith(sft_suffix)
            ]
            seen_eval_records = [
                record
                for record in family_records
                if isinstance(record.get("template_id"), str)
                and str(record["template_id"]).endswith(seen_eval_suffix)
            ]
            if len(sft_records) != 1 or len(seen_eval_records) != 1:
                errors.append(
                    f"{operation_id}/{family_id}: expected one SFT template ending "
                    f"in {sft_suffix!r} and one seen-family eval template ending "
                    f"in {seen_eval_suffix!r}"
                )

    return summaries, errors


def _string_set(value: object, field: str, errors: list[str]) -> set[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        errors.append(f"{field} must be a list of non-empty strings")
        return set()
    if len(value) != len(set(value)):
        errors.append(f"{field} contains duplicate values")
    return set(value)


def _non_empty_string(value: object, field: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a non-empty string")
        return ""
    return value


def main() -> int:
    try:
        manifest = load_manifest()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: could not load H1 split manifest: {exc}")
        return 1

    records, load_errors = load_registry()
    summaries, validation_errors = validate_h1_split(records, manifest)
    errors = load_errors + validation_errors

    for summary in summaries:
        print(
            f"{summary.operation_id}: {summary.train_count} train / "
            f"{summary.test_count} test ({summary.total_count} total)"
        )
    for error in errors:
        print(f"ERROR: {error}")

    if errors:
        return 1

    print(
        "Validated H1 split: no template_family_id leakage between train and test."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
