#!/usr/bin/env python3
"""Check that H1 family folds differ only in template-family assignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import EVALUATION_SPLITS, read_jsonl  # noqa: E402


COMPARABLE_METADATA_FIELDS = (
    "world_id",
    "world_seed",
    "people_count",
    "fact_count",
    "sft_template_suffix",
    "seen_family_eval_template_suffix",
    "evaluation_people_count",
    "evaluation_sample_seed",
    "counts",
    "counts_by_split",
    "counts_by_operation",
)


def _fact_definitions(
    records: list[dict[str, Any]],
) -> dict[str, dict[tuple[object, object], tuple[object, ...]]]:
    return {
        split: {
            (record.get("world_id"), record.get("fact_id")): (
                record.get("person_id"),
                record.get("person_name"),
                record.get("relation_id"),
                record.get("fact_value"),
                record.get("operation_id"),
                record.get("canonical_answer"),
            )
            for record in records
            if record.get("split") == split
        }
        for split in EVALUATION_SPLITS
    }


def _source_hashes(metadata: dict[str, Any]) -> dict[str, object]:
    hashes = metadata.get("input_sha256")
    if not isinstance(hashes, dict):
        return {}
    return {
        str(path): digest
        for path, digest in hashes.items()
        if not str(path).startswith("data/splits/")
    }


def validate_family_folds(dataset_dirs: list[Path]) -> list[str]:
    """Require identical worlds/facts and distinct singleton held-out families."""

    errors: list[str] = []
    baseline_dir: Path | None = None
    baseline_metadata: dict[str, Any] | None = None
    baseline_world_sha256: str | None = None
    baseline_definitions: dict[
        str, dict[tuple[object, object], tuple[object, ...]]
    ] | None = None
    heldout_to_dir: dict[str, Path] = {}

    if len(dataset_dirs) < 2:
        return ["at least two family-fold dataset directories are required"]

    for dataset_dir in dataset_dirs:
        metadata_path = dataset_dir / "metadata.json"
        world_path = dataset_dir / "world.json"
        test_path = dataset_dir / "test.jsonl"
        missing = [
            path.name
            for path in (metadata_path, world_path, test_path)
            if not path.is_file()
        ]
        if missing:
            errors.append(f"{dataset_dir}: missing files {missing}")
            continue

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{metadata_path}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(metadata, dict):
            errors.append(f"{metadata_path}: expected JSON object")
            continue

        heldout_families = metadata.get("test_family_ids")
        if not (
            isinstance(heldout_families, list)
            and len(heldout_families) == 1
            and isinstance(heldout_families[0], str)
        ):
            errors.append(f"{metadata_path}: expected one held-out family")
        else:
            heldout_family = heldout_families[0]
            previous_dir = heldout_to_dir.setdefault(heldout_family, dataset_dir)
            if previous_dir != dataset_dir:
                errors.append(
                    f"held-out family {heldout_family!r} is repeated in "
                    f"{previous_dir} and {dataset_dir}"
                )

        records, record_errors = read_jsonl(test_path)
        errors.extend(f"{dataset_dir}: {error}" for error in record_errors)
        definitions = _fact_definitions(records)
        intra_fold_fact_sets = {
            split: set(split_definitions)
            for split, split_definitions in definitions.items()
        }
        if len({frozenset(keys) for keys in intra_fold_fact_sets.values()}) != 1:
            errors.append(
                f"{dataset_dir}: evaluation slices do not contain the same facts"
            )

        world_sha256 = hashlib.sha256(world_path.read_bytes()).hexdigest()
        if baseline_metadata is None:
            baseline_dir = dataset_dir
            baseline_metadata = metadata
            baseline_world_sha256 = world_sha256
            baseline_definitions = definitions
            continue

        assert baseline_dir is not None
        assert baseline_definitions is not None
        for field in COMPARABLE_METADATA_FIELDS:
            if metadata.get(field) != baseline_metadata.get(field):
                errors.append(
                    f"{dataset_dir}: metadata {field} differs from {baseline_dir}"
                )
        if _source_hashes(metadata) != _source_hashes(baseline_metadata):
            errors.append(
                f"{dataset_dir}: non-manifest source hashes differ from {baseline_dir}"
            )
        if world_sha256 != baseline_world_sha256:
            errors.append(f"{dataset_dir}: world.json differs from {baseline_dir}")
        for split in EVALUATION_SPLITS:
            if definitions[split] != baseline_definitions[split]:
                errors.append(
                    f"{dataset_dir}: fact definitions in {split} differ from "
                    f"{baseline_dir}"
                )

    if len(heldout_to_dir) != len(dataset_dirs):
        errors.append("family-fold datasets do not have distinct held-out families")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dirs", type=Path, nargs="+")
    args = parser.parse_args()

    errors = validate_family_folds(args.dataset_dirs)
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1

    heldout_families = []
    for dataset_dir in args.dataset_dirs:
        metadata = json.loads(
            (dataset_dir / "metadata.json").read_text(encoding="utf-8")
        )
        heldout_families.extend(metadata["test_family_ids"])
    print(
        "Validated comparable H1 family folds with held-out families: "
        + ", ".join(heldout_families)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
