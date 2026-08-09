#!/usr/bin/env python3
"""Regression tests for the H1 template-family split."""

from __future__ import annotations

import unittest

from validate_h1_split import load_manifest, load_registry, validate_h1_split


class H1SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records, self.load_errors = load_registry()
        self.manifest = load_manifest()

    def test_current_split_has_no_leakage_and_matches_target_counts(self) -> None:
        self.assertEqual([], self.load_errors)
        summaries, errors = validate_h1_split(self.records, self.manifest)
        self.assertEqual([], errors)
        self.assertEqual(
            [
                ("GET_BIRTH_CITY", 6, 2),
                ("GET_BIRTH_DATE", 6, 2),
                ("GET_RESIDENCE_CITY", 6, 2),
                ("GET_OCCUPATION", 6, 2),
            ],
            [
                (summary.operation_id, summary.train_count, summary.test_count)
                for summary in summaries
            ],
        )

    def test_rejects_family_assigned_to_both_splits(self) -> None:
        manifest = dict(self.manifest)
        manifest["test_family_ids"] = [
            *self.manifest["test_family_ids"],
            "direct_question",
        ]
        _, errors = validate_h1_split(self.records, manifest)
        self.assertTrue(any("leak across train and test" in error for error in errors))

    def test_rejects_unassigned_family(self) -> None:
        manifest = dict(self.manifest)
        manifest["train_family_ids"] = [
            family
            for family in self.manifest["train_family_ids"]
            if family != "imperative"
        ]
        _, errors = validate_h1_split(self.records, manifest)
        self.assertTrue(
            any("unassigned registry families" in error for error in errors)
        )

    def test_rejects_identical_sft_and_seen_eval_suffixes(self) -> None:
        manifest = dict(self.manifest)
        manifest["seen_family_eval_template_suffix"] = manifest[
            "sft_template_suffix"
        ]
        _, errors = validate_h1_split(self.records, manifest)
        self.assertTrue(any("suffixes must differ" in error for error in errors))

    def test_rejects_non_positive_evaluation_people_count(self) -> None:
        manifest = dict(self.manifest)
        manifest["evaluation_people_count"] = 0
        _, errors = validate_h1_split(self.records, manifest)
        self.assertTrue(
            any("evaluation_people_count must be" in error for error in errors)
        )


if __name__ == "__main__":
    unittest.main()
