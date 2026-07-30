#!/usr/bin/env python3
"""Generate a deterministic micro-world as JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from world import WorldGenerator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, help="Integer or text world seed.")
    parser.add_argument(
        "--people",
        type=int,
        required=True,
        help="Number of people to generate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this path instead of stdout.",
    )
    args = parser.parse_args()

    try:
        world = WorldGenerator().generate(seed=args.seed, people_count=args.people)
    except ValueError as exc:
        parser.error(str(exc))

    rendered = world.to_json() + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
