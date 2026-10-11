"""Check that benchmark matching and synthetic holdouts do not inflate results."""

from __future__ import annotations

import itertools
import random
import tempfile
import unittest
from pathlib import Path

from foveate_tools.synthetic import HELDOUT_FAMILIES, generate_workloads
from scripts.benchmark_autostructure import maximum_weight_assignment, score_boxes


class BenchmarkScorerTests(unittest.TestCase):
    def test_assignment_agrees_with_exhaustive_optimum(self) -> None:
        rng = random.Random(71)
        for _ in range(30):
            weights = [[rng.random() * 5 for _ in range(4)] for _ in range(4)]
            expected = max(
                sum(weights[row][col] for row, col in enumerate(order))
                for order in itertools.permutations(range(4))
            )
            actual = sum(
                weights[row][col] for row, col in maximum_weight_assignment(weights)
            )
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            set(maximum_weight_assignment([[4, 3], [3, 0]])), {(0, 1), (1, 0)}
        )

    def test_duplicate_predictions_reduce_precision(self) -> None:
        score = score_boxes([[0, 0, 100, 100], [1, 1, 101, 101]], ((0, 0, 100, 100),))
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (1, 1, 0))
        self.assertEqual(score["precision"], 0.5)
        self.assertEqual(score["recall"], 1)

    def test_match_threshold_and_clipping_are_independent(self) -> None:
        score = score_boxes([[0, 0, 75, 100]], ((0, 0, 100, 100),))
        self.assertEqual(score["f1"], 1)
        self.assertEqual(score["matched_geometry_clipped_fraction"], 1)
        rejected = score_boxes([[0, 0, 49, 100]], ((0, 0, 100, 100),))
        self.assertEqual((rejected["tp"], rejected["fn"]), (0, 1))
        self.assertIsNone(rejected["mean_iou"])

    def test_empty_predictions_preserve_false_negatives(self) -> None:
        score = score_boxes([], ((0, 0, 100, 100),))
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (0, 0, 1))
        self.assertEqual(score_boxes([[0, 0, 100, 100]], ())["fp"], 1)

    def test_holdout_corpus_seed_and_template_separation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cases = generate_workloads(Path(directory), repeats=1)
            development = [case for case in cases if case.split == "development"]
            heldout = [case for case in cases if case.split != "development"]
            self.assertTrue(
                {c.seed for c in development}.isdisjoint(c.seed for c in heldout)
            )
            dev_families = {case.family for case in development}
            self.assertTrue(dev_families.isdisjoint(HELDOUT_FAMILIES))
            self.assertEqual(
                {case.family for case in heldout if case.split == "heldout-template"},
                set(HELDOUT_FAMILIES),
            )
            for case in cases:
                self.assertEqual(case.negative, not case.boxes)
            self.assertTrue((Path(directory) / "corpus.json").exists())


if __name__ == "__main__":
    unittest.main()
