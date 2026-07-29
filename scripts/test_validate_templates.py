#!/usr/bin/env python3
"""Regression tests for template-registry validation."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from validate_templates import (
    TEMPLATE_DIR,
    TEMPLATE_FILES,
    is_near_duplicate,
    load_jsonl,
    validate_file,
)


class SimilarityTests(unittest.TestCase):
    def test_rejects_possessive_of_genitive_pseudo_family(self) -> None:
        left = "What is {person}'s city of birth?"
        right = "What is the city of {person}'s birth?"
        self.assertTrue(is_near_duplicate(left, right))

    def test_allows_question_to_imperative_contrast(self) -> None:
        left = "In which city was {person} born?"
        right = "Name the city where {person} was born."
        self.assertFalse(is_near_duplicate(left, right))


class RegistryTests(unittest.TestCase):
    def test_registry_passes_strict_validation(self) -> None:
        for filename, operation_id in TEMPLATE_FILES.items():
            with self.subTest(filename=filename):
                errors, warnings = validate_file(
                    TEMPLATE_DIR / filename,
                    operation_id,
                )
                self.assertEqual([], errors)
                self.assertEqual([], warnings)

    def test_jsonl_loader_preserves_source_line_numbers(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            path.write_text('{"value": 1}\nnot-json\n{"value": 2}\n', encoding="utf-8")
            records, errors = load_jsonl(path)
        self.assertEqual([1, 3], [line_number for line_number, _ in records])
        self.assertEqual(1, len(errors))
        self.assertIn("sample.jsonl:2:", errors[0])


if __name__ == "__main__":
    unittest.main()
