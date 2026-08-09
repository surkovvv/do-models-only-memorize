#!/usr/bin/env python3
"""Strictly validate a generated H1 train/test dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import (  # noqa: E402
    counts_by_operation,
    generate_h1_examples,
    input_hashes,
    read_jsonl,
    validate_example_records,
)
from validate_h1_split import (  # noqa: E402
    DEFAULT_SPLIT_PATH,
    load_manifest,
    load_registry,
    validate_h1_split,
)
from world import WorldGenerator  # noqa: E402


def validate_dataset(dataset_dir: Path) -> list[str]:
    errors: list[str] = []
    metadata_path = dataset_dir / "metadata.json"
    world_path = dataset_dir / "world.json"
    train_path = dataset_dir / "train.jsonl"
    test_path = dataset_dir / "test.jsonl"

    for path in (metadata_path, world_path, train_path, test_path):
        if not path.exists():
            errors.append(f"missing file: {path.name}")
    if errors:
        return errors

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"metadata.json: invalid JSON: {exc.msg}"]
    if not isinstance(metadata, dict):
        return ["metadata.json: expected JSON object"]

    required_metadata = {
        "schema_version",
        "dataset_id",
        "hypothesis",
        "world_id",
        "world_seed",
        "people_count",
        "fact_count",
        "train_family_ids",
        "test_family_ids",
        "sft_template_suffix",
        "seen_family_eval_template_suffix",
        "evaluation_people_count",
        "evaluation_sample_seed",
        "counts",
        "counts_by_split",
        "counts_by_operation",
        "input_sha256",
    }
    if set(metadata) != required_metadata:
        errors.append("metadata.json: schema mismatch")
        return errors

    try:
        people_count = int(metadata["people_count"])
        world = WorldGenerator().generate(
            seed=str(metadata["world_seed"]),
            people_count=people_count,
        )
    except (TypeError, ValueError) as exc:
        return [f"metadata.json: invalid world parameters: {exc}"]

    try:
        stored_world = json.loads(world_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"world.json: invalid JSON: {exc.msg}")
        stored_world = None
    if stored_world != world.to_dict():
        errors.append("world.json does not match the recorded seed and source data")

    template_records, registry_errors = load_registry()
    errors.extend(registry_errors)
    manifest = load_manifest()
    _, split_errors = validate_h1_split(template_records, manifest)
    errors.extend(split_errors)

    train_family_ids = set(manifest["train_family_ids"])
    test_family_ids = set(manifest["test_family_ids"])
    if sorted(train_family_ids) != metadata["train_family_ids"]:
        errors.append("metadata train families differ from current H1 manifest")
    if sorted(test_family_ids) != metadata["test_family_ids"]:
        errors.append("metadata test families differ from current H1 manifest")
    for field in (
        "sft_template_suffix",
        "seen_family_eval_template_suffix",
        "evaluation_people_count",
        "evaluation_sample_seed",
    ):
        if metadata[field] != manifest[field]:
            errors.append(f"metadata {field} differs from current H1 manifest")

    train_records, train_errors = read_jsonl(train_path)
    test_records, test_errors = read_jsonl(test_path)
    errors.extend(train_errors)
    errors.extend(test_errors)
    errors.extend(
        validate_example_records(
            train_records,
            test_records,
            train_family_ids=train_family_ids,
            test_family_ids=test_family_ids,
            sft_template_suffix=str(manifest["sft_template_suffix"]),
            seen_family_eval_template_suffix=str(
                manifest["seen_family_eval_template_suffix"]
            ),
            evaluation_people_count=int(manifest["evaluation_people_count"]),
        )
    )

    expected_train, expected_test = generate_h1_examples(
        world,
        template_records,
        train_family_ids=train_family_ids,
        test_family_ids=test_family_ids,
        sft_template_suffix=str(manifest["sft_template_suffix"]),
        seen_family_eval_template_suffix=str(
            manifest["seen_family_eval_template_suffix"]
        ),
        evaluation_people_count=int(manifest["evaluation_people_count"]),
        evaluation_sample_seed=str(manifest["evaluation_sample_seed"]),
    )
    expected_train_records = [example.to_dict() for example in expected_train]
    expected_test_records = [example.to_dict() for example in expected_test]
    if train_records != expected_train_records:
        errors.append("train.jsonl differs from the complete expected Cartesian product")
    if test_records != expected_test_records:
        errors.append("test.jsonl differs from the complete expected Cartesian product")

    expected_counts = {
        "train": len(expected_train),
        "test": len(expected_test),
        "total": len(expected_train) + len(expected_test),
    }
    if metadata["counts"] != expected_counts:
        errors.append("metadata counts do not match generated examples")
    expected_counts_by_split: dict[str, int] = {}
    for example in (*expected_train, *expected_test):
        expected_counts_by_split[example.split] = (
            expected_counts_by_split.get(example.split, 0) + 1
        )
    if metadata["counts_by_split"] != dict(sorted(expected_counts_by_split.items())):
        errors.append("metadata split counts do not match generated examples")
    expected_operation_counts = counts_by_operation(expected_train, expected_test)
    if metadata["counts_by_operation"] != expected_operation_counts:
        errors.append("metadata operation counts do not match generated examples")
    if metadata["input_sha256"] != input_hashes(ROOT, DEFAULT_SPLIT_PATH):
        errors.append("metadata input hashes differ from current source files")
    if metadata["schema_version"] != 2:
        errors.append("metadata schema_version must be 2")
    if metadata["hypothesis"] != "H1":
        errors.append("metadata hypothesis must be H1")
    if metadata["dataset_id"] != f"h1_{world.id}":
        errors.append("metadata dataset_id does not match world_id")
    if metadata["world_id"] != world.id:
        errors.append("metadata world_id does not match the world seed")
    if metadata["fact_count"] != len(world.facts):
        errors.append("metadata fact_count does not match world.json")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()

    errors = validate_dataset(args.dataset_dir)
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1

    metadata = json.loads(
        (args.dataset_dir / "metadata.json").read_text(encoding="utf-8")
    )
    print(
        f"Validated {metadata['counts']['train']} SFT and "
        f"{metadata['counts']['test']} evaluation examples across three slices: "
        "no family or template leakage."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
