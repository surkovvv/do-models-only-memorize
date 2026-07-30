#!/usr/bin/env python3
"""Tests for deterministic micro-world generation."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from world import Seed, WorldGenerator, WorldSourceData  # noqa: E402


SOURCE_DATA = WorldSourceData(
    names=tuple(f"Name-{index}" for index in range(30)),
    dates=(date(1990, 1, 1), date(2000, 2, 2)),
    professions=("engineer", "teacher"),
    towns=("Town-A", "Town-B", "Town-C"),
)


class SeedTests(unittest.TestCase):
    def test_keyed_choices_do_not_depend_on_call_order(self) -> None:
        seed = Seed(42)
        first = seed.choice("person.name", ("A", "B", "C"))
        seed.choice("an.unrelated.new_field", ("x", "y"))
        self.assertEqual(first, seed.choice("person.name", ("A", "B", "C")))


class WorldGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = WorldGenerator(SOURCE_DATA)

    def test_same_seed_produces_identical_world(self) -> None:
        first = self.generator.generate(seed="experiment-1", people_count=20)
        second = self.generator.generate(seed="experiment-1", people_count=20)
        self.assertEqual(first, second)
        self.assertEqual(first.to_json(), second.to_json())

    def test_different_seed_changes_world(self) -> None:
        first = self.generator.generate(seed=1, people_count=20)
        second = self.generator.generate(seed=2, people_count=20)
        self.assertNotEqual(first.people, second.people)

    def test_names_are_unique_and_person_ids_are_stable(self) -> None:
        world = self.generator.generate(seed=1, people_count=10)
        self.assertEqual(10, len({person.name for person in world.people}))
        self.assertEqual("person_0001", world.people[0].id)
        self.assertEqual("person_0010", world.people[-1].id)

    def test_movers_always_live_outside_their_birth_city(self) -> None:
        world = self.generator.generate(seed=7, people_count=30)
        for person in world.people:
            self.assertEqual(
                person.moved,
                person.birth_city != person.residence_city,
            )

    def test_world_exposes_four_facts_per_person(self) -> None:
        world = self.generator.generate(seed=1, people_count=3)
        self.assertEqual(12, len(world.facts))
        self.assertEqual(
            {
                "birth_date",
                "occupation",
                "birth_city",
                "residence_city",
            },
            {fact.relation for fact in world.facts},
        )

    def test_rejects_more_people_than_unique_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique names"):
            self.generator.generate(seed=1, people_count=31)

    def test_rejects_negative_people_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.generator.generate(seed=1, people_count=-1)


if __name__ == "__main__":
    unittest.main()
