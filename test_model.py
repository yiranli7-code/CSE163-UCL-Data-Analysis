"""Small, deterministic smoke tests for the optional modeling challenge.

Authors: Rodrigo Li and Larry Zhang
Course: CSE 163, Autumn 2026
Section: AC (Rodrigo Li)
"""

from __future__ import annotations

import unittest

import pandas as pd

from model import FEATURES, get_partitions, make_estimator, scores


class ModelSmokeTests(unittest.TestCase):
    """Check model training without making claims about real test metrics."""

    def test_temporal_partitions(self: ModelSmokeTests) -> None:
        """Verify that later seasons are excluded from earlier splits."""
        frame = pd.DataFrame({'season': ['2011-12', '2019-20',
                                         '2020-21', '2021-22',
                                         '2022-23', '2023-24']})
        train, validation, test = get_partitions(frame)
        self.assertEqual(train['season'].tolist(), ['2011-12', '2019-20'])
        self.assertEqual(validation['season'].tolist(),
                         ['2020-21', '2021-22'])
        self.assertEqual(test['season'].tolist(), ['2022-23', '2023-24'])

    def test_model_fits_small_data(self: ModelSmokeTests) -> None:
        """Both estimators accept the exact proposed pre-match fields."""
        example = pd.DataFrame({
            'home_gf_prior': [0, 1, 2, 3, 1, 0, 3, 2],
            'home_ga_prior': [1, 2, 1, 0, 1, 0, 2, 3],
            'away_gf_prior': [0, 2, 1, 3, 1, 0, 3, 2],
            'away_ga_prior': [2, 0, 1, 1, 2, 1, 0, 1],
            'stage': ['Group', 'Group', 'Knockout', 'Group',
                      'Knockout', 'Group', 'Group', 'Knockout'],
            'high_scoring': [0, 0, 1, 1, 0, 1, 1, 0],
        })
        for name, hyperparam in [('logistic', 1.0), ('forest', 4)]:
            with self.subTest(name=name):
                estimator = make_estimator(name, hyperparam)
                estimator.fit(example[FEATURES], example['high_scoring'])
                result = scores(estimator, example)
                self.assertTrue(0 <= result['f1'] <= 1)
                self.assertTrue(0 <= result['roc_auc'] <= 1)

    def test_unknown_model_rejected(self: ModelSmokeTests) -> None:
        """A misspelled model name cannot silently select a model."""
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            make_estimator('wrong-model', 1)


if __name__ == '__main__':
    unittest.main()
