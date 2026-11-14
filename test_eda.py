"""Unit and integration checks for CSE 163 final-project EDA.

Run: python -m unittest -v test_eda.py
All tests use deterministic source-derived samples or small fake fixtures.

Authors: Rodrigo Li and Larry Zhang
Course: CSE 163, Autumn 2026
Section: AC (Rodrigo Li)
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from collect_openfootball import parse_cl, parse_score
from eda import (add_prematch_form, bootstrap_seasons,
                 bootstrap_stage_difference, make_analysis,
                 prepare_matches, seven_number, stage_test,
                 validate_matches)


class ProjectTests(unittest.TestCase):
    """Check source parsing, arithmetic, quality, and information leakage."""

    def setUp(self: ProjectTests) -> None:
        """Make a minimal two-team season and independent fixtures."""
        self.fixtures: pd.DataFrame = pd.DataFrame([
            ('2023-24', '2023-09-19', 'Group', 'A', 'B', 2, 0),
            ('2023-24', '2023-09-19', 'Group', 'C', 'D', 1, 1),
            ('2023-24', '2023-10-03', 'Group', 'A', 'C', 1, 3),
            ('2023-24', '2023-10-03', 'Group', 'B', 'D', 0, 0),
            ('2023-24', '2023-11-01', 'Knockout', 'D', 'A', 4, 1),
            ('2023-24', '2023-11-01', 'Knockout', 'B', 'C', 2, 3),
        ], columns=['season', 'date', 'stage', 'home_team',
                    'away_team', 'home_goals', 'away_goals'])

    def test_seven_number_summary(self: ProjectTests) -> None:
        """Quartiles and sample standard deviation are computed correctly."""
        result = seven_number(pd.Series([0, 1, 2, 3, 4, np.nan]))
        self.assertEqual(result['mean'], 2.0)
        self.assertEqual(result['median'], 2.0)
        self.assertEqual(result['q1'], 1.0)
        self.assertEqual(result['q3'], 3.0)
        self.assertAlmostEqual(result['std'], np.sqrt(2.5))

    def test_zero_observed_raises(self: ProjectTests) -> None:
        """An empty quantitative series must not produce invented stats."""
        with self.assertRaises(ValueError):
            seven_number(pd.Series([np.nan]))

    def test_match_targets(self: ProjectTests) -> None:
        """Threshold is three combined goals, including exactly three."""
        finished = prepare_matches(self.fixtures)
        target = finished.loc[finished['home_team'] == 'A']
        self.assertEqual(target['total_goals'].tolist(), [2, 4])
        self.assertEqual(target['high_scoring'].tolist(), [0, 1])

    def test_duplicate_rejected(self: ProjectTests) -> None:
        """Duplicate fixtures cannot silently inflate counts."""
        duplicate = pd.concat([self.fixtures, self.fixtures.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_matches(duplicate)

    def test_missing_goals_rejected(self: ProjectTests) -> None:
        """Unknown outcomes must not be zero-filled."""
        damaged = self.fixtures.copy()
        damaged.loc[0, 'home_goals'] = np.nan
        with self.assertRaisesRegex(ValueError, 'missing'):
            validate_matches(damaged)

    def test_negative_goals_rejected(self: ProjectTests) -> None:
        """Impossible scores must raise a validation error."""
        damaged = self.fixtures.copy()
        damaged.loc[0, 'home_goals'] = -1
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            validate_matches(damaged)

    def test_same_day_no_leakage(self: ProjectTests) -> None:
        """All first-date fixtures have missing prior means."""
        forms = add_prematch_form(self.fixtures)
        first = forms[forms['date'] == '2023-09-19']
        self.assertTrue(first['home_gf_prior'].isna().all())
        self.assertTrue(first['away_ga_prior'].isna().all())

    def test_form_from_only_previous_days(self: ProjectTests) -> None:
        """A's first result determines its form in its second fixture."""
        forms = add_prematch_form(self.fixtures)
        target = forms[(forms['home_team'] == 'A') &
                       (forms['date'] == '2023-10-03')].iloc[0]
        self.assertEqual(target['home_games_prior'], 1)
        self.assertEqual(target['home_gf_prior'], 2.0)
        self.assertEqual(target['home_ga_prior'], 0.0)
        self.assertEqual(target['away_gf_prior'], 1.0)
        self.assertEqual(target['away_ga_prior'], 1.0)

    def test_season_reset(self: ProjectTests) -> None:
        """A new season never inherits prior-season goals."""
        again = self.fixtures.iloc[[0]].copy()
        again['season'] = '2024-25'
        test = pd.concat([self.fixtures, again], ignore_index=True)
        with_form = add_prematch_form(test)
        row = with_form[with_form['season'] == '2024-25'].iloc[0]
        self.assertTrue(np.isnan(row['home_gf_prior']))

    def test_stage_counts_and_null(self: ProjectTests) -> None:
        """Malformed empty outcome margin prevents chi-square inference."""
        all_low = prepare_matches(self.fixtures)
        all_low['high_scoring'] = 0
        self.assertFalse(stage_test(all_low)['valid'])

    def test_bootstrap_reproducible(self: ProjectTests) -> None:
        """The bootstrap interval is deterministic with a fixed seed."""
        data = prepare_matches(self.fixtures)
        left = bootstrap_stage_difference(data, iterations=200, seed=12)
        right = bootstrap_stage_difference(data, iterations=200, seed=12)
        self.assertEqual(left, right)

    def test_penalty_excluded(self: ProjectTests) -> None:
        """Penalties do not become scored match goals."""
        self.assertEqual(parse_score('4-2', ' pen. 1-0 a.e.t. (1-0)'),
                         (1, 0))
        self.assertEqual(parse_score('3-4', ' pen. 1-1 a.e.t. (1-1)'),
                         (1, 1))

    def test_parser_source_count(self: ProjectTests) -> None:
        """Parser accepts a small realistic source excerpt and checks count."""
        raw = ('= UEFA Champions League 2023/24\n'
               '# Matches    2\n'
               '▪ Group, Matchday 1\n'
               '  Tue Sep 19 2023\n'
               '    18:45 AC Milan (ITA)  v Newcastle FC (ENG)  0-0\n'
               '▪ Finals, Round of 16\n'
               '  Tue Mar 12 2024\n'
               '    21:00 Arsenal FC (ENG)  v FC Porto (POR)  '
               '4-2 pen. 1-0 a.e.t. (1-0)\n')
        frame = parse_cl(raw, '2023-24')
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame['stage'].tolist(), ['Group', 'Knockout'])
        self.assertEqual(frame.iloc[1]['away_goals'], 0)
        self.assertEqual(frame.iloc[1]['home_goals'], 1)

    def test_parser_rejects_partial_source(self: ProjectTests) -> None:
        """An incomplete season cannot generate apparently complete data."""
        snippet = ('# Matches 125\n▪ Group A\n  Wed Sep 14 2011\n'
                   '    20:45 A (ENG)  v B (ESP)  1-1\n')
        with self.assertRaisesRegex(ValueError, 'parsed 1'):
            parse_cl(snippet, '2011-12')

    def test_published_2023_24_snapshot(self: ProjectTests) -> None:
        """Compare result counts with OpenFootball's official header."""
        from pathlib import Path
        root = Path(__file__).resolve().parent
        source = pd.read_csv(root / 'data/matches_2023_24.csv')
        result = prepare_matches(source)
        self.assertEqual(len(result), 125)
        self.assertEqual((result['stage'] == 'Group').sum(), 96)
        self.assertEqual((result['stage'] == 'Knockout').sum(), 29)
        self.assertEqual(result['total_goals'].sum(), 375)
        self.assertEqual(result['high_scoring'].sum(), 71)
        self.assertEqual(result['stage'].isna().sum(), 0)

    def test_localized_group_heading(self: ProjectTests) -> None:
        """The German 'Gruppe' headings must remain group fixtures."""
        raw = ('# Matches 2\n▪ Gruppe G\n  Tue Sep 15 2020\n'
               '    21:00 A (ITA)  v B (ENG)  1-0\n'
               '▪ Gruppe H\n  Wed Sep 16\n'
               '    21:00 C (FRA)  v D (ESP)  1-1\n')
        result = parse_cl(raw, '2020-21')
        self.assertEqual(result['stage'].tolist(), ['Group', 'Group'])

    def test_2020_delayed_knockout_dates(self: ProjectTests) -> None:
        """August postponed knockouts occur in 2020, not August 2019."""
        raw = ('# Matches 2\n▪ Group A\n  Tue Sep 17 2019\n'
               '    21:00 A (ITA)  v B (ENG)  0-0\n'
               '▪ Quarterfinals\n  Fri Aug 14\n'
               '    21:00 C (ESP)  v D (GER)  2-8\n')
        parsed = parse_cl(raw, '2019-20')
        self.assertEqual(parsed['date'].tolist(),
                         ['2019-09-17', '2020-08-14'])

    def test_all_13_seasons_have_correct_groups(self: ProjectTests) -> None:
        """Count fixtures and stages for the complete uploaded archive."""
        from pathlib import Path
        root = Path(__file__).resolve().parent
        matches = pd.read_csv(root / 'data/matches_2011_2024.csv')
        self.assertEqual(len(matches), 1619)
        counts = matches.groupby('season').size()
        self.assertEqual(len(counts), 13)
        self.assertEqual(counts.loc['2019-20'], 119)
        self.assertTrue((counts.drop('2019-20') == 125).all())
        group = matches.loc[matches['stage'] == 'Group']
        self.assertTrue((group.groupby('season').size() == 96).all())
        self.assertEqual(len(group), 1248)
        self.assertEqual(len(matches) - len(group), 371)

    def test_cluster_bootstrap_reproducible(self: ProjectTests) -> None:
        """Resampling seasons is reproducible and retains stage totals."""
        fixtures = pd.concat([self.fixtures,
                              self.fixtures.assign(season='2022-23')])
        prepared = prepare_matches(fixtures)
        left = bootstrap_seasons(prepared, iterations=100, seed=11)
        right = bootstrap_seasons(prepared, iterations=100, seed=11)
        self.assertEqual(left, right)
        self.assertEqual(left['clusters'], 2)

    def test_match_archive_source_crosscheck(self: ProjectTests) -> None:
        """Direct parser output agrees with the original one-year snapshot."""
        from pathlib import Path
        root = Path(__file__).resolve().parent
        source = root / 'source_openfootball/2023-24/cl.txt'
        parsed = parse_cl(source.read_text(encoding='utf-8'),
                          '2023-24')
        archive = pd.read_csv(root / 'data/matches_2011_2024.csv')
        subset = archive.loc[archive['season'] == '2023-24']
        keys = ['date', 'home_team', 'away_team', 'stage',
                'home_goals', 'away_goals']
        self.assertEqual(len(parsed), 125)
        self.assertEqual(len(parsed[keys].merge(subset[keys],
                         how='inner')), 125)

    def test_full_pipeline(self: ProjectTests) -> None:
        """A complete deterministic run writes its metric files and plots."""
        from pathlib import Path
        root = Path(__file__).resolve().parent
        metrics = make_analysis(root / 'data')
        self.assertEqual(metrics['matches']['rows'], 1619)
        self.assertEqual(metrics['form_rows_eligible'], 1411)
        self.assertEqual(metrics['stage_summary']['Group']['n'], 1248)
        self.assertEqual(metrics['stage_summary']['Knockout']['n'], 371)
        self.assertTrue((root / 'figures/fig6_stage_high_scoring.png')
                        .exists())


if __name__ == '__main__':
    unittest.main()
