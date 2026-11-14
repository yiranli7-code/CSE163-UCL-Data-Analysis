"""Exploratory analysis of UEFA player rankings and OpenFootball fixtures.

Run: python eda.py
Dependencies: pandas, numpy, matplotlib, scipy.

Authors: Rodrigo Li and Larry Zhang
Course: CSE 163, Autumn 2026
Section: AC (Rodrigo Li)
"""

from __future__ import annotations
from typing import Any

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUTPUT = ROOT / 'outputs'
FIGURES = ROOT / 'figures'
COLORS = ['#153b64', '#128a8a', '#e2a23a', '#9966a7', '#cc6554']


def seven_number(series: pd.Series) -> dict[str, float | None]:
    """Compute mean, SD, minimum, quartiles, and maximum for series.
    
        Ignore missing observations and raise ValueError for an empty sample.
        Return seven named numeric results; SD is None for a single value."""
    numeric = pd.to_numeric(series, errors='coerce').dropna()
    if numeric.empty:
        raise ValueError('Quantitative variable has no observed values')
    return {
        'mean': float(numeric.mean()),
        'std': float(numeric.std(ddof=1)) if len(numeric) > 1 else None,
        'min': float(numeric.min()),
        'q1': float(numeric.quantile(0.25)),
        'median': float(numeric.median()),
        'q3': float(numeric.quantile(0.75)),
        'max': float(numeric.max()),
    }


def validate_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns, goals, stages, dates and fixture IDs.
    
        Return a copy of matches; raise ValueError for malformed records."""
    required = ['season', 'date', 'stage', 'home_team', 'away_team',
                'home_goals', 'away_goals']
    missing = set(required) - set(matches.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if matches[required].isna().any().any():
        raise ValueError('Required match values are missing')
    if not matches['stage'].isin(['Group', 'Knockout']).all():
        raise ValueError('Stage contains an unsupported value')
    if (matches['home_team'] == matches['away_team']).any():
        raise ValueError('Team cannot play against itself')
    for field in ('home_goals', 'away_goals'):
        values = matches[field]
        if (values < 0).any() or (values != values.astype(int)).any():
            raise ValueError(f'Invalid goals in {field}')
    identifiers = ['season', 'date', 'home_team', 'away_team']
    if matches.duplicated(identifiers).any():
        raise ValueError('Duplicate match records')
    pd.to_datetime(matches['date'], errors='raise')
    return matches.copy()


def prepare_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Return validated matches with ordered dates and goal labels.
    
        Add total_goals and a binary high_scoring (three-plus goals) column."""
    result = validate_matches(matches)
    result['date'] = pd.to_datetime(result['date'])
    result = result.sort_values(['season', 'date', 'home_team',
                                 'away_team']).reset_index(drop=True)
    result['total_goals'] = result['home_goals'] + result['away_goals']
    result['high_scoring'] = (result['total_goals'] >= 3).astype(int)
    return result


def add_prematch_form(matches: pd.DataFrame) -> pd.DataFrame:
    """Return matches with prior scoring and conceding averages.
    
        For each team within each season, use only matches before its date.
        Same-day fixtures cannot enter each other's predictor values.
        Unknown opening-match averages remain missing."""
    ordered = prepare_matches(matches)
    state = {}
    records = []
    for (season, date), batch in ordered.groupby(['season', 'date'],
                                                 sort=True):
        todays_updates = []
        for _, game in batch.iterrows():
            row = game.to_dict()
            home = (season, row['home_team'])
            away = (season, row['away_team'])
            for prefix, team in [('home', home), ('away', away)]:
                played, scored, conceded = state.get(team, (0, 0, 0))
                row[f'{prefix}_games_prior'] = played
                row[f'{prefix}_gf_prior'] = (scored / played
                                             if played else np.nan)
                row[f'{prefix}_ga_prior'] = (conceded / played
                                             if played else np.nan)
            records.append(row)
            todays_updates.append((home, away, row['home_goals'],
                                   row['away_goals']))
        for home, away, hg, ag in todays_updates:
            for team, gf, ga in [(home, hg, ag), (away, ag, hg)]:
                played, scored, conceded = state.get(team, (0, 0, 0))
                state[team] = (played + 1, scored + gf, conceded + ga)
    return pd.DataFrame(records)


def stage_test(matches: pd.DataFrame) -> dict[str, Any]:
    """Compare match stage with the high_scoring outcome in matches.
    
        Return observed/expected counts, chi-square results and diagnostics.
        Flag invalid inference if a margin is empty or counts are too small."""
    contingency = pd.crosstab(matches['stage'], matches['high_scoring'])
    contingency = contingency.reindex(index=['Group', 'Knockout'],
                                      columns=[0, 1], fill_value=0)
    observed = contingency.to_numpy()
    if np.any(observed.sum(axis=0) == 0):
        return {'valid': False, 'reason': 'Outcome has a zero margin'}
    chi2, p, _, expected = chi2_contingency(observed, correction=False)
    adequate = bool(np.all(expected >= 5))
    total = int(observed.sum())
    effect = np.sqrt(chi2 / total) if total else np.nan
    return {'valid': adequate, 'chi2': float(chi2), 'p': float(p),
            'cramers_v': float(effect),
            'min_expected': float(expected.min()),
            'observed': contingency.to_dict(orient='index'),
            'expected': expected.tolist(),
            'warning': (None if adequate else
                        'Expected counts below five; no chi-square inference')}


def bootstrap_stage_difference(matches: pd.DataFrame,
                               iterations: int = 3000,
                               seed: int = 163) -> dict[str, float]:
    """Bootstrap the knockout-minus-group high-scoring difference.
    
        Use iterations resamples and the given random seed; return the
        observed rate difference and an exploratory 95% iid interval.
        Raise ValueError if either stage has no matches."""
    rng = np.random.default_rng(seed)
    group = matches.loc[matches['stage'] == 'Group',
                        'high_scoring'].to_numpy()
    knockout = matches.loc[matches['stage'] == 'Knockout',
                           'high_scoring'].to_numpy()
    if not len(group) or not len(knockout):
        raise ValueError('Both stages are required for comparison')
    estimates = np.empty(iterations)
    for index in range(iterations):
        g = rng.choice(group, size=len(group), replace=True).mean()
        k = rng.choice(knockout, size=len(knockout), replace=True).mean()
        estimates[index] = k - g
    return {'difference': float(knockout.mean() - group.mean()),
            'ci_low': float(np.quantile(estimates, 0.025)),
            'ci_high': float(np.quantile(estimates, 0.975))}


def bootstrap_seasons(matches: pd.DataFrame,
                      iterations: int = 3000,
                      seed: int = 163) -> dict[str, float | int]:
    """Bootstrap full seasons for a stage-rate difference interval.
    
        Use iterations resamples and seed; return the observed gap, 95%
        interval and cluster count. Require both stages in multiple seasons.
        This does not fully remove repeated-team dependence within seasons."""
    if matches['season'].nunique() < 2:
        raise ValueError('Need multiple seasons for cluster bootstrap')
    summary = matches.groupby(['season', 'stage'])['high_scoring'].agg(
        ['size', 'sum']).unstack('stage')
    summary = summary.reindex(columns=['Group', 'Knockout'], level=1)
    if summary.isna().any().any():
        raise ValueError('Every season needs both competition stages')
    group_count = summary[('size', 'Group')].to_numpy()
    knock_count = summary[('size', 'Knockout')].to_numpy()
    group_hits = summary[('sum', 'Group')].to_numpy()
    knock_hits = summary[('sum', 'Knockout')].to_numpy()
    n = len(summary)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(iterations, n))
    g = group_hits[indices].sum(axis=1) / group_count[indices].sum(axis=1)
    k = knock_hits[indices].sum(axis=1) / knock_count[indices].sum(axis=1)
    return {'difference': float(knock_hits.sum() / knock_count.sum() -
                                group_hits.sum() / group_count.sum()),
            'ci_low': float(np.quantile(k - g, 0.025)),
            'ci_high': float(np.quantile(k - g, 0.975)),
            'clusters': n}


def describe_dataframe(frame: pd.DataFrame,
                       quantitative: list[str],
                       categorical: list[str]) -> dict[str, Any]:
    """Describe columns and missingness in frame.
    
        Summarize quantitative columns by seven-number statistics and
        categorical columns by complete value counts; return a mapping."""
    return {
        'rows': int(len(frame)), 'columns': int(len(frame.columns)),
        'missing_by_column': {
            column: int(value)
            for column, value in frame.isna().sum().items()
        },
        'quantitative': {
            column: seven_number(frame[column]) for column in quantitative
        },
        'categorical': {
            column: {
                str(label): int(count) for label, count in
                frame[column].value_counts(dropna=False).items()
            } for column in categorical
        },
    }


def save_figure(path: Path) -> None:
    """Save the current matplotlib plot to path, then close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=175, bbox_inches='tight', facecolor='white')
    plt.close()


def player_graphs(players: pd.DataFrame) -> None:
    """Save career totals and per-appearance charts from players.
    
        Players without appearance counts are excluded from the rate chart."""
    ordered = players.sort_values('goals', ascending=False)
    labels = ordered['player'].str.replace('Cristiano ', 'C. ', regex=False)
    labels = labels.str.replace('Robert ', 'R. ', regex=False)
    labels = labels.str.replace('Karim ', 'K. ', regex=False)
    labels = labels.str.replace('Lionel ', 'L. ', regex=False)
    labels = labels.str.replace('Kylian ', 'K. ', regex=False)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    x = np.arange(len(ordered))
    ax.bar(x, ordered['goals'], label='Goals', color=COLORS[0])
    ax.bar(x, ordered['assists'], bottom=ordered['goals'],
           label='Assists', color=COLORS[1])
    ax.set_xticks(x, labels, rotation=20, ha='right')
    ax.set_ylabel('UEFA-listed career contributions')
    ax.set_title('Career goals and assists: five selected scorers')
    ax.legend(frameon=False)
    save_figure(FIGURES / 'fig1_career_contributions.png')

    present = players.dropna(subset=['appearances']).copy()
    present['goals_per_match'] = present['goals'] / present['appearances']
    present['ga_per_match'] = (
        (present['goals'] + present['assists']) / present['appearances']
    )
    present = present.sort_values('goals_per_match', ascending=False)
    x = np.arange(len(present))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.bar(x - width / 2, present['goals_per_match'], width,
           label='Goals / appearance', color=COLORS[0])
    ax.bar(x + width / 2, present['ga_per_match'], width,
           label='(Goals + assists) / appearance', color=COLORS[1])
    ax.set_xticks(x, present['player'], rotation=15, ha='right')
    ax.set_ylabel('Per UEFA-listed appearance')
    ax.set_title('Available exposure-adjusted performance (n = 3)')
    ax.legend(frameon=False, loc='upper right')
    save_figure(FIGURES / 'fig2_rates_verified.png')


def season_graphs(seasons: pd.DataFrame) -> None:
    """Save selected-season comparisons from the seasons table.
    
        The selected observations do not form a full career time series."""
    ordered = seasons.sort_values('goals', ascending=True)
    names = ordered['player'].str.split().str[-1] + '  ' + ordered['season']
    fig, ax = plt.subplots(figsize=(8, 5.0))
    ax.barh(names, ordered['goals'], color=COLORS[0])
    ax.set_xlabel('UEFA-listed goals in selected season')
    ax.set_title('Verified selected high-scoring UCL seasons')
    save_figure(FIGURES / 'fig3_selected_seasons.png')

    subset = seasons.loc[seasons['player'] == 'Cristiano Ronaldo'].copy()
    subset = subset.sort_values('season')
    fig, ax = plt.subplots(figsize=(8, 4.1))
    ax.plot(subset['season'], subset['goals'], marker='o', markersize=7,
            linewidth=2.5, color=COLORS[0])
    ax.set_ylim(0, max(subset['goals']) + 3)
    ax.set_ylabel('Goals')
    ax.set_xlabel('Selected Ronaldo seasons (not consecutive)')
    ax.set_title('Ronaldo selected campaign peaks; uneven year gaps')
    ax.grid(axis='y', alpha=0.25)
    save_figure(FIGURES / 'fig4_ronaldo_peak_seasons.png')


def match_graphs(matches: pd.DataFrame) -> None:
    """Save four match-level and seasonal scoring figures.
    
        Use matches with total_goals and high_scoring already computed."""
    values = matches['total_goals'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.1))
    ax.bar(values.index, values.values, color=COLORS[0])
    ax.axvline(2.5, color=COLORS[2], linestyle='--', linewidth=1.6,
               label='High-scoring threshold: >= 3')
    ax.set_xlabel('Goals by both teams (excluding penalty shootouts)')
    ax.set_ylabel('Number of fixtures')
    ax.set_title(f'Goal totals, 2011/12-2023/24 (n={len(matches):,})')
    ax.legend(frameon=False)
    save_figure(FIGURES / 'fig5_match_goal_distribution.png')

    stages = matches.groupby('stage')['high_scoring'].agg(['mean', 'size'])
    stages = stages.reindex(['Group', 'Knockout'])
    fig, ax = plt.subplots(figsize=(8, 3.9))
    bars = ax.bar(stages.index, 100 * stages['mean'],
                  color=[COLORS[0], COLORS[1]], width=0.55)
    for bar, size in zip(bars, stages['size']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5, f'n = {size}', ha='center')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Share with >= 3 match goals (%)')
    ax.set_title('High-scoring match frequencies by stage, 13 seasons')
    save_figure(FIGURES / 'fig6_stage_high_scoring.png')

    seasonal = matches.groupby('season').agg(
        match_count=('total_goals', 'size'),
        mean_goals=('total_goals', 'mean'),
        high_rate=('high_scoring', 'mean')).reset_index()
    fig, ax = plt.subplots(figsize=(8, 3.7))
    ax.plot(seasonal['season'], seasonal['mean_goals'], marker='o',
            color=COLORS[0], linewidth=2)
    ax.set_ylim(2.3, 3.65)
    ax.set_xticks(range(len(seasonal)))
    ax.set_xticklabels(seasonal['season'], rotation=45, ha='right')
    ax.set_ylabel('Mean goals per match')
    ax.set_title('Scoring by Champions League season')
    ax.grid(axis='y', alpha=0.2)
    save_figure(FIGURES / 'fig7_season_goal_trend.png')

    rate = matches.groupby(['season', 'stage'])['high_scoring'].mean()
    both = rate.unstack('stage').reindex(columns=['Group', 'Knockout'])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for stage, color in [('Group', COLORS[0]), ('Knockout', COLORS[1])]:
        ax.plot(both.index, 100 * both[stage], marker='o',
                color=color, label=stage, linewidth=1.9)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Share with >= 3 goals (%)')
    ax.set_xticks(range(len(both)))
    ax.set_xticklabels(both.index, rotation=45, ha='right')
    ax.set_title('Season-specific high-scoring rates by stage')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.2)
    save_figure(FIGURES / 'fig8_stage_rates_by_season.png')


def make_analysis(data_dir: Path = DATA) -> dict[str, Any]:
    """Generate EDA outputs from the CSV files under data_dir.
    
        Save figures, cleaned tables and JSON metrics; return the metrics."""
    plt.switch_backend('Agg')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    players = pd.read_csv(data_dir / 'players_uefa_2026_11_12.csv')
    seasons = pd.read_csv(data_dir / 'verified_peak_seasons.csv')
    matches_raw = pd.read_csv(data_dir / 'matches_2011_2024.csv')
    matches = prepare_matches(matches_raw)
    features = add_prematch_form(matches_raw)
    players['goal_contributions'] = players['goals'] + players['assists']
    players['goals_per_appearance'] = (
        players['goals'] / players['appearances']
    )
    players['contributions_per_appearance'] = (
        players['goal_contributions'] / players['appearances']
    )
    metrics = {
        'snapshot_date': '2026-11-12',
        'match_range': '2011-12 to 2023-24',
        'players': describe_dataframe(
            players, ['goals', 'assists', 'appearances', 'goal_contributions',
                      'goals_per_appearance',
                      'contributions_per_appearance'], ['player']),
        'peak_seasons': describe_dataframe(seasons, ['goals'],
                                           ['player', 'season']),
        'matches': describe_dataframe(
            matches, ['home_goals', 'away_goals', 'total_goals',
                      'high_scoring'], ['stage', 'season', 'home_team',
                                        'away_team']),
        'prematch': describe_dataframe(
            features, ['home_gf_prior', 'home_ga_prior',
                       'away_gf_prior', 'away_ga_prior'],
            ['stage', 'season']),
        'chi_square': stage_test(matches),
        'bootstrap': bootstrap_stage_difference(matches),
        'season_bootstrap': bootstrap_seasons(matches),
        'season_summary': {
            str(index): {'n': int(row['n']),
                         'goals': int(row['goals']),
                         'avg_goals': float(row['avg_goals']),
                         'high_scoring': int(row['high_scoring']),
                         'high_rate': float(row['high_rate'])}
            for index, row in matches.groupby('season').agg(
                n=('total_goals', 'size'),
                goals=('total_goals', 'sum'),
                avg_goals=('total_goals', 'mean'),
                high_scoring=('high_scoring', 'sum'),
                high_rate=('high_scoring', 'mean')).iterrows()
        },
        'stage_summary': {
            stage: {'n': int(group.shape[0]),
                    'high_scoring': int(group['high_scoring'].sum()),
                    'high_rate': float(group['high_scoring'].mean())}
            for stage, group in matches.groupby('stage')
        },
        'total_goals': int(matches['total_goals'].sum()),
        'form_rows_eligible': int(features.dropna(subset=[
            'home_gf_prior', 'home_ga_prior', 'away_gf_prior',
            'away_ga_prior']).shape[0]),
        'total_high_scoring': int(matches['high_scoring'].sum()),
        'source_match_counts': matches['stage'].value_counts().to_dict(),
    }
    player_graphs(players)
    season_graphs(seasons)
    match_graphs(matches)
    matches.to_csv(OUTPUT / 'matches_clean.csv', index=False)
    matches.groupby('season').agg(
        n=('total_goals', 'size'),
        goals=('total_goals', 'sum'),
        mean_goals=('total_goals', 'mean'),
        high_scoring=('high_scoring', 'sum'),
        high_rate=('high_scoring', 'mean')).to_csv(
            OUTPUT / 'season_summary.csv')
    features.to_csv(OUTPUT / 'prematch_features.csv', index=False)
    players.to_csv(OUTPUT / 'players_clean.csv', index=False)
    with (OUTPUT / 'eda_metrics.json').open('w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    print('Processed:', len(players), 'players;', len(seasons),
          'selected player-seasons;', len(matches), 'matches.')
    print('Group/knockout: ', metrics['source_match_counts'])
    print('Chi-square:', metrics['chi_square'])
    print('Form-eligible fixtures:', metrics['form_rows_eligible'])
    return metrics


def main() -> None:
    """Execute the complete EDA using checked-in source data."""
    make_analysis()


if __name__ == '__main__':
    main()
