"""Auditable supplementary statistics for the CSE 163 final report.

Inputs are the team's existing historical tables or their exact checked-in
small reference snapshots. No player-level missing values are imputed.

Authors: Rodrigo Li and Larry Zhang
Course: CSE 163, Autumn 2026
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, linregress


REQUIRED_SEASON_COLUMNS = ('season', 'n', 'goals', 'mean_goals',
                           'high_scoring', 'high_rate')


def validate_seasons(seasons: pd.DataFrame) -> pd.DataFrame:
    """Validate counts and rates; return chronologically sorted data."""
    missing = set(REQUIRED_SEASON_COLUMNS).difference(seasons.columns)
    if missing:
        raise ValueError(f'Missing season columns: {sorted(missing)}')
    if seasons.empty or seasons['season'].duplicated().any():
        raise ValueError('Require unique, nonempty season records')
    out = seasons.copy().sort_values('season').reset_index(drop=True)
    if ((out['n'] <= 0) | (out['goals'] < 0) |
            (out['high_scoring'] < 0) |
            (out['high_scoring'] > out['n'])).any():
        raise ValueError('Invalid season-level counts')
    if not np.allclose(out['mean_goals'], out['goals'] / out['n']):
        raise ValueError('Mean goals does not match goal count / games')
    if not np.allclose(out['high_rate'],
                       out['high_scoring'] / out['n']):
        raise ValueError('High-scoring rate does not match counts')
    return out


def player_table(players: pd.DataFrame) -> pd.DataFrame:
    """Calculate exact available rates; retain NA for missing denominators."""
    cols = {'player', 'goals', 'assists', 'appearances'}
    if not cols.issubset(players.columns):
        raise ValueError('Player data missing required columns')
    if players['player'].duplicated().any() or players.empty:
        raise ValueError('Require one row per player')
    frame = players.copy()
    if ((frame['goals'] < 0) | (frame['assists'] < 0)).any():
        raise ValueError('Goals and assists must be nonnegative')
    known = frame['appearances'].notna()
    if (frame.loc[known, 'appearances'] <= 0).any():
        raise ValueError('Known appearance totals must be positive')
    frame['contributions'] = frame['goals'] + frame['assists']
    frame['goals_per_appearance'] = (
        frame['goals'] / frame['appearances'])
    frame['contributions_per_appearance'] = (
        frame['contributions'] / frame['appearances'])
    return frame.sort_values('contributions', ascending=False)


def stage_effect(stage: dict[str, Any]) -> dict[str, float]:
    """Compute stage risk gap, odds ratio and an exact 2x2 Fisher p-value."""
    g = stage['group']
    k = stage['knockout']
    counts = [g['n'], k['n'], g['three_plus'], k['three_plus']]
    if (min(counts) < 0 or min(g['n'], k['n']) == 0 or
            g['three_plus'] > g['n'] or k['three_plus'] > k['n']):
        raise ValueError('Invalid stage contingency table')
    success_k = k['three_plus']
    failure_k = k['n'] - success_k
    success_g = g['three_plus']
    failure_g = g['n'] - success_g
    odds, p_value = fisher_exact([[success_k, failure_k],
                                  [success_g, failure_g]])
    return {'group_rate': success_g / g['n'],
            'knockout_rate': success_k / k['n'],
            'risk_difference': success_k / k['n'] - success_g / g['n'],
            'odds_ratio': float(odds),
            'fisher_p': float(p_value)}


def trend_analysis(seasons: pd.DataFrame, iterations: int = 20000,
                   seed: int = 163) -> dict[str, Any]:
    """Fit a descriptive season trend with a two-sided permutation check.

    Shuffles season-mean outcomes among fixed year labels. This is an
    exchangeability-based diagnostic, not a causal test or forecast.
    """
    if iterations <= 0:
        raise ValueError('Permutation iterations must be positive')
    data = validate_seasons(seasons)
    if len(data) < 3:
        raise ValueError('At least three seasons are required')
    x = np.arange(len(data), dtype=float)
    y = data['mean_goals'].to_numpy(dtype=float)
    regression = linregress(x, y)
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))
    observed = abs(float(regression.slope))
    rng = np.random.default_rng(seed)
    more_extreme = 0
    for _ in range(iterations):
        slope = float(np.dot(x_centered, rng.permutation(y)) / denom)
        more_extreme += abs(slope) >= observed - 1e-12
    leave_one_out = []
    for omitted in range(len(data)):
        keep = np.arange(len(data)) != omitted
        leave_one_out.append(float(linregress(x[keep], y[keep]).slope))
    non_pandemic = data['season'] != '2019-20'
    slope_excluding = float(linregress(
        x[non_pandemic], y[non_pandemic]).slope)
    return {'seasons': int(len(data)),
            'weighted_goals_per_match': float(data['goals'].sum() /
                                              data['n'].sum()),
            'overall_three_plus_rate': float(data['high_scoring'].sum() /
                                             data['n'].sum()),
            'total_games': int(data['n'].sum()),
            'total_goals': int(data['goals'].sum()),
            'slope_goals_per_game_per_season': float(regression.slope),
            'ols_p': float(regression.pvalue),
            'ols_r_squared': float(regression.rvalue ** 2),
            'permutation_p': (more_extreme + 1) / (iterations + 1),
            'min_leave_one_out_slope': min(leave_one_out),
            'max_leave_one_out_slope': max(leave_one_out),
            'slope_excluding_2019_20': slope_excluding}


def validate_full_matches(matches: pd.DataFrame,
                          seasons: pd.DataFrame) -> None:
    """Reconcile optional original fixture data with saved season summaries."""
    required = {'season', 'home_goals', 'away_goals', 'stage'}
    if not required.issubset(matches.columns):
        raise ValueError('Fixture file missing needed fields')
    if matches[['home_goals', 'away_goals']].isna().any().any():
        raise ValueError('Fixture scores cannot be missing')
    if ((matches['home_goals'] < 0) |
            (matches['away_goals'] < 0)).any():
        raise ValueError('Negative scores are invalid')
    derived = matches.assign(
        goals=matches['home_goals'] + matches['away_goals'])
    by_season = derived.groupby('season').agg(
        n=('goals', 'size'), goals=('goals', 'sum'))
    expected = validate_seasons(seasons).set_index('season')
    if not by_season.equals(expected[['n', 'goals']]):
        raise ValueError('Fixture counts/goals disagree with season summary')
