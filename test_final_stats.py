"""Regression, boundary and synthetic-data tests for Part 3 statistics.

Authors: Rodrigo Li and Larry Zhang
Course: CSE 163, Autumn 2026
"""

from __future__ import annotations

import unittest

import pandas as pd

from final_stats import (player_table, stage_effect, trend_analysis,
                         validate_full_matches, validate_seasons)


class FinalStatisticsTests(unittest.TestCase):
    """Test identities, missingness, inference and fixture reconciliation."""

    def setUp(self) -> None:
        """Create a small correctly reconciled three-season data fixture."""
        self.seasons = pd.DataFrame({
            'season': ['2011-12', '2012-13', '2013-14'],
            'n': [2, 2, 2], 'goals': [2, 4, 6],
            'mean_goals': [1., 2., 3.],
            'high_scoring': [0, 1, 1],
            'high_rate': [0., .5, .5]})

    def test_season_identities(self) -> None:
        """Counts, means and high-scoring rates remain consistent."""
        self.assertEqual(validate_seasons(self.seasons)['n'].sum(), 6)

    def test_seasons_sorted(self) -> None:
        """Season order is deterministic even for unsorted input."""
        reversed_data = self.seasons.iloc[::-1]
        self.assertEqual(validate_seasons(reversed_data)['season'][0],
                         '2011-12')

    def test_duplicate_season_fails(self) -> None:
        """Duplicated seasons are rejected rather than double-counted."""
        dup = pd.concat([self.seasons, self.seasons.iloc[:1]])
        with self.assertRaises(ValueError):
            validate_seasons(dup)

    def test_rate_mismatch_fails(self) -> None:
        """Incorrect tabulated rates cannot silently enter a figure."""
        wrong = self.seasons.copy()
        wrong.loc[0, 'high_rate'] = .99
        with self.assertRaisesRegex(ValueError, 'rate'):
            validate_seasons(wrong)

    def test_mean_mismatch_fails(self) -> None:
        """A wrong goal average is caught by a count identity."""
        wrong = self.seasons.copy()
        wrong.loc[0, 'mean_goals'] = 1.8
        with self.assertRaisesRegex(ValueError, 'Mean'):
            validate_seasons(wrong)

    def test_player_missingness_preserved(self) -> None:
        """No rate is guessed for a player with unknown appearances."""
        frame = pd.DataFrame({'player': ['A', 'B'], 'goals': [2, 4],
                              'assists': [1, 1],
                              'appearances': [2., None]})
        result = player_table(frame).set_index('player')
        self.assertAlmostEqual(
            result.loc['A', 'contributions_per_appearance'], 1.5)
        self.assertTrue(
            pd.isna(result.loc['B', 'contributions_per_appearance']))

    def test_no_zero_denominator(self) -> None:
        """An appearance denominator of zero raises an error."""
        frame = pd.DataFrame({'player': ['A'], 'goals': [1],
                              'assists': [0], 'appearances': [0]})
        with self.assertRaises(ValueError):
            player_table(frame)

    def test_stage_rates(self) -> None:
        """The stage difference is knockout minus group, not reverse."""
        result = stage_effect({'group': {'n': 10, 'three_plus': 8},
                               'knockout': {'n': 10, 'three_plus': 3}})
        self.assertAlmostEqual(result['risk_difference'], -.5)
        self.assertGreaterEqual(result['fisher_p'], 0.)

    def test_bad_stage_counts(self) -> None:
        """Successes cannot exceed the number of matches."""
        with self.assertRaises(ValueError):
            stage_effect({'group': {'n': 3, 'three_plus': 4},
                          'knockout': {'n': 3, 'three_plus': 2}})

    def test_trend_permutation_reproducible(self) -> None:
        """A fixed seed yields identical randomization diagnostics."""
        a = trend_analysis(self.seasons, iterations=200, seed=19)
        b = trend_analysis(self.seasons, iterations=200, seed=19)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a['slope_goals_per_game_per_season'], 1.)

    def test_negative_iterations(self) -> None:
        """Reject an invalid randomization sample count."""
        with self.assertRaises(ValueError):
            trend_analysis(self.seasons, iterations=0)

    def test_fixture_reconciliation(self) -> None:
        """Independent raw fixture totals match the summary tables."""
        rows = []
        for idx, season in enumerate(self.seasons['season']):
            rows.extend([{'season': season, 'home_goals': idx + 1,
                          'away_goals': 0, 'stage': 'Group'},
                         {'season': season, 'home_goals': idx + 1,
                          'away_goals': 0, 'stage': 'Group'}])
        validate_full_matches(pd.DataFrame(rows), self.seasons)

    def test_fixture_mismatch(self) -> None:
        """Unexpected match score changes cause a hard audit failure."""
        matches = pd.DataFrame({'season': ['2011-12'],
                                'home_goals': [3], 'away_goals': [0],
                                'stage': ['Group']})
        with self.assertRaises(ValueError):
            validate_full_matches(matches, self.seasons)


if __name__ == '__main__':
    unittest.main()
