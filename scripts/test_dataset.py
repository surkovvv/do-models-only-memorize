#!/usr/bin/env python3
"""Regression tests for H1 dataset generation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import generate_h1_examples, validate_example_records  # noqa: E402
from validate_h1_split import load_manifest, load_registry  # noqa: E402
from world import WorldGenerator  # noqa: E402


class DatasetGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records, errors = load_registry()
        if errors:
            raise AssertionError(errors)
        cls.manifest = load_manifest()
        cls.train_families = set(cls.manifest["train_family_ids"])
        cls.test_families = set(cls.manifest["test_family_ids"])
        cls.world = WorldGenerator().generate(seed=123, people_count=3)
        cls.train, cls.test = generate_h1_examples(
            cls.world,
            cls.records,
            train_family_ids=cls.train_families,
            test_family_ids=cls.test_families,
        )

    def test_generates_full_cartesian_product(self) -> None:
        self.assertEqual(54, len(self.train))
        self.assertEqual(24, len(self.test))

    def test_fact_identity_is_shared_across_templates(self) -> None:
        fact_id = "person_0001.birth_city"
        examples = [
            example
            for example in (*self.train, *self.test)
            if example.fact_id == fact_id
        ]
        self.assertEqual(7, len(examples))
        self.assertEqual(1, len({example.fact_value for example in examples}))
        self.assertEqual(7, len({example.example_id for example in examples}))

    def test_train_and_test_cover_the_same_atomic_facts(self) -> None:
        self.assertEqual(
            {example.fact_id for example in self.train},
            {example.fact_id for example in self.test},
        )
        self.assertEqual(12, len({example.fact_id for example in self.train}))

    def test_only_test_family_is_held_out(self) -> None:
        self.assertEqual(
            self.train_families,
            {example.template_family_id for example in self.train},
        )
        self.assertEqual(
            self.test_families,
            {example.template_family_id for example in self.test},
        )

    def test_generated_records_pass_semantic_validation(self) -> None:
        errors = validate_example_records(
            [example.to_dict() for example in self.train],
            [example.to_dict() for example in self.test],
            train_family_ids=self.train_families,
            test_family_ids=self.test_families,
        )
        self.assertEqual([], errors)

    def test_validator_rejects_changed_fact_answer(self) -> None:
        train_records = [example.to_dict() for example in self.train]
        test_records = [example.to_dict() for example in self.test]
        test_records[0]["canonical_answer"] = "wrong"
        errors = validate_example_records(
            train_records,
            test_records,
            train_family_ids=self.train_families,
            test_family_ids=self.test_families,
        )
        self.assertTrue(any("answer differs from fact value" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
