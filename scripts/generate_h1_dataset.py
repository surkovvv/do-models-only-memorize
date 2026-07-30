#!/usr/bin/env python3
"""Generate train/test JSONL data for the first H1 template split."""

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
    write_jsonl,
)
from validate_h1_split import (  # noqa: E402
    DEFAULT_SPLIT_PATH,
    load_manifest,
    load_registry,
    validate_h1_split,
)
from world import WorldGenerator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, help="World seed.")
    parser.add_argument("--people", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
    )
    args = parser.parse_args()

    manifest = load_manifest(args.split_manifest)
    template_records, load_errors = load_registry()
    _, split_errors = validate_h1_split(template_records, manifest)
    errors = load_errors + split_errors
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    train_family_ids = set(manifest["train_family_ids"])
    test_family_ids = set(manifest["test_family_ids"])
    world = WorldGenerator().generate(seed=args.seed, people_count=args.people)
    train, test = generate_h1_examples(
        world,
        template_records,
        train_family_ids=train_family_ids,
        test_family_ids=test_family_ids,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "world.json").write_text(
        world.to_json() + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "test.jsonl", test)

    metadata = {
        "schema_version": 1,
        "dataset_id": f"h1_{world.id}",
        "hypothesis": "H1",
        "world_id": world.id,
        "world_seed": world.seed.value,
        "people_count": len(world.people),
        "fact_count": len(world.facts),
        "train_family_ids": sorted(train_family_ids),
        "test_family_ids": sorted(test_family_ids),
        "counts": {
            "train": len(train),
            "test": len(test),
            "total": len(train) + len(test),
        },
        "counts_by_operation": counts_by_operation(train, test),
        "input_sha256": input_hashes(ROOT, args.split_manifest),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Generated {len(train)} train and {len(test)} test examples "
        f"for {len(world.people)} people in {args.output_dir}."
    )
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
