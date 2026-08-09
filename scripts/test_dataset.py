#!/usr/bin/env python3
"""Regression tests for H1 dataset generation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import (  # noqa: E402
    EVAL_EXACT_RECALL,
    EVAL_HELDOUT_FAMILY,
    EVAL_SEEN_FAMILY_NEW_TEMPLATE,
    generate_h1_examples,
    validate_example_records,
)
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
            sft_template_suffix=str(cls.manifest["sft_template_suffix"]),
            seen_family_eval_template_suffix=str(
                cls.manifest["seen_family_eval_template_suffix"]
            ),
            evaluation_people_count=2,
            evaluation_sample_seed=str(cls.manifest["evaluation_sample_seed"]),
        )

    def test_generates_sft_and_three_paired_evaluation_slices(self) -> None:
        self.assertEqual(36, len(self.train))
        self.assertEqual(24, len(self.test))
        self.assertEqual(
            {
                EVAL_EXACT_RECALL: 8,
                EVAL_SEEN_FAMILY_NEW_TEMPLATE: 8,
                EVAL_HELDOUT_FAMILY: 8,
            },
            {
                split: sum(example.split == split for example in self.test)
                for split in {
                    EVAL_EXACT_RECALL,
                    EVAL_SEEN_FAMILY_NEW_TEMPLATE,
                    EVAL_HELDOUT_FAMILY,
                }
            },
        )

    def test_fact_identity_is_shared_across_templates(self) -> None:
        fact_id = self.test[0].fact_id
        examples = [
            example
            for example in (*self.train, *self.test)
            if example.fact_id == fact_id
        ]
        self.assertEqual(6, len(examples))
        self.assertEqual(1, len({example.fact_value for example in examples}))
        self.assertEqual(6, len({example.example_id for example in examples}))

    def test_evaluation_facts_are_a_subset_of_sft_facts(self) -> None:
        train_fact_ids = {example.fact_id for example in self.train}
        test_fact_ids = {example.fact_id for example in self.test}
        self.assertEqual(12, len(train_fact_ids))
        self.assertEqual(8, len(test_fact_ids))
        self.assertLessEqual(test_fact_ids, train_fact_ids)

    def test_only_heldout_slice_uses_the_test_family(self) -> None:
        self.assertEqual(
            self.train_families,
            {example.template_family_id for example in self.train},
        )
        self.assertEqual(
            self.test_families,
            {
                example.template_family_id
                for example in self.test
                if example.split == EVAL_HELDOUT_FAMILY
            },
        )

    def test_seen_family_eval_template_ids_are_globally_absent_from_sft(self) -> None:
        train_template_ids = {example.template_id for example in self.train}
        seen_eval_template_ids = {
            example.template_id
            for example in self.test
            if example.split == EVAL_SEEN_FAMILY_NEW_TEMPLATE
        }
        self.assertTrue(train_template_ids.isdisjoint(seen_eval_template_ids))

    def test_exact_recall_fact_template_pairs_occur_in_sft(self) -> None:
        train_pairs = {(example.fact_id, example.template_id) for example in self.train}
        exact_pairs = {
            (example.fact_id, example.template_id)
            for example in self.test
            if example.split == EVAL_EXACT_RECALL
        }
        self.assertLessEqual(exact_pairs, train_pairs)

    def test_generated_records_pass_semantic_validation(self) -> None:
        errors = validate_example_records(
            [example.to_dict() for example in self.train],
            [example.to_dict() for example in self.test],
            train_family_ids=self.train_families,
            test_family_ids=self.test_families,
            sft_template_suffix=str(self.manifest["sft_template_suffix"]),
            seen_family_eval_template_suffix=str(
                self.manifest["seen_family_eval_template_suffix"]
            ),
            evaluation_people_count=2,
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
            sft_template_suffix=str(self.manifest["sft_template_suffix"]),
            seen_family_eval_template_suffix=str(
                self.manifest["seen_family_eval_template_suffix"]
            ),
            evaluation_people_count=2,
        )
        self.assertTrue(any("answer differs from fact value" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
