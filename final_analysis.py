"""Generate final-report figures and supplementary diagnostics.

Run from any directory: python final_analysis.py
When installed in the team repository, prefer its existing data/ and
outputs/ artifacts; otherwise use the bundled, audited small snapshots.

Authors: Rodrigo Li and Larry Zhang
Course: CSE 163, Autumn 2026
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_stats import (player_table, stage_effect, trend_analysis,
                         validate_full_matches, validate_seasons)

plt.switch_backend('Agg')

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / 'final_figures'
OUTPUTS = ROOT / 'final_outputs'
BLUE = '#17345B'
TEAL = '#087F83'
GOLD = '#DC9C3D'
GRAY = '#7C8795'
BG = '#F8FAFC'


def first_existing(*paths: str) -> Path:
    """Select the first existing supported source path; fail transparently."""
    for relative in paths:
        result = ROOT / relative
        if result.is_file():
            return result
    raise FileNotFoundError('No source found among: ' + ', '.join(paths))


def set_axes(ax: Any, title: str, ylabel: str = '') -> None:
    """Apply a restrained, consistent figure design."""
    ax.set_title(title, fontsize=12, loc='left', weight='bold', pad=13)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#DCE3EA', alpha=.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=8)


def export(name: str) -> None:
    """Write the currently active figure with reproducible rendering."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=1.4)
    plt.savefig(FIGURES / name, dpi=205, facecolor='white',
                bbox_inches='tight')
    plt.close()


def plot_players(players: pd.DataFrame) -> None:
    """Show observed career counts and rates only where available."""
    frame = player_table(players)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(frame))
    ax.barh(y, frame['goals'], color=BLUE, label='Goals')
    ax.barh(y, frame['assists'], left=frame['goals'], color=GOLD,
            label='Assists')
    ax.set_yticks(y, frame['player'])
    ax.invert_yaxis()
    set_axes(ax, 'Career goal contributions | five-player UEFA snapshot',
             'Goals plus assists')
    ax.set_ylabel('')
    ax.set_xlabel('UEFA-listed career events', fontsize=9)
    ax.legend(frameon=False, loc='lower right')
    for i, number in enumerate(frame['contributions']):
        ax.text(number + 2, i, str(int(number)), va='center', fontsize=9)
    ax.set_xlim(0, max(frame['contributions']) + 29)
    export('01_career_contributions.png')

    known = frame.dropna(subset=['appearances']).copy()
    known = known.sort_values('contributions_per_appearance')
    fig, ax = plt.subplots(figsize=(9, 4.0))
    y = np.arange(len(known))
    width = .33
    ax.barh(y - width / 2, known['goals_per_appearance'], width,
            color=TEAL, label='Goals / appearance')
    ax.barh(y + width / 2,
            known['contributions_per_appearance'], width,
            color=GOLD, label='(Goals + assists) / appearance')
    ax.set_yticks(y, known['player'])
    ax.set_xlabel('Rate per listed appearance, not per 90 minutes',
                  fontsize=9)
    set_axes(ax, 'Opportunity-adjusted comparison | available n = 3')
    ax.legend(loc='lower right', frameon=False)
    ax.set_xlim(0, 1.35)
    export('02_appearance_rates.png')


def plot_peaks(peaks: pd.DataFrame) -> None:
    """Plot selected audited peak records without implying complete seasons."""
    chosen = peaks.sort_values(['goals', 'player']).copy()
    chosen['label'] = chosen['player'].str.split().str[-1] + ' | ' + \
        chosen['season']
    fig, ax = plt.subplots(figsize=(9, 4.7))
    y = np.arange(len(chosen))
    colors = [BLUE if n == 'Cristiano Ronaldo' else GRAY
              for n in chosen['player']]
    ax.barh(y, chosen['goals'], color=colors)
    ax.set_yticks(y, chosen['label'])
    ax.set_xlim(0, 20)
    ax.set_xlabel('Goals in selected documented UCL season', fontsize=9)
    set_axes(ax, 'Documented elite seasons | intentionally selective')
    for i, value in enumerate(chosen['goals']):
        ax.text(value + .2, i, str(int(value)), va='center', fontsize=8)
    export('03_selected_peak_seasons.png')


def plot_seasons(seasons: pd.DataFrame, slope: float) -> None:
    """Plot observed goals-per-match means with a descriptive linear fit."""
    seasons = validate_seasons(seasons)
    x = np.arange(len(seasons))
    y = seasons['mean_goals'].to_numpy()
    line = y.mean() + slope * (x - x.mean())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, y, color=TEAL, marker='o', markersize=5, linewidth=2.1,
            label='Observed goal rate')
    ax.plot(x, line, '--', color=GOLD, linewidth=1.9,
            label='Exploratory linear trend')
    ax.set_xticks(x, seasons['season'].str.slice(0, 4), rotation=45)
    ax.set_ylim(2.55, 3.50)
    set_axes(ax, 'UCL goals per match | 2011/12–2023/24',
             'Match goals (season mean)')
    ax.legend(frameon=False, loc='upper left')
    export('04_season_goal_trend.png')


def plot_stage(stats: dict[str, float]) -> None:
    """Plot observed high-scoring proportions for both competition stages."""
    fig, ax = plt.subplots(figsize=(9, 3.9))
    vals = [stats['group_rate'] * 100, stats['knockout_rate'] * 100]
    bars = ax.bar(['Group stage', 'Knockout stage'], vals,
                  color=[BLUE, TEAL], width=.47)
    for rect, value in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, value + .7,
                f'{value:.1f}%', ha='center', fontsize=11, weight='bold')
    ax.set_ylim(0, 70)
    set_axes(ax, 'Three-plus-goal match frequency | by stage',
             'Share of games (%)')
    export('05_stage_goal_frequency.png')


def plot_model(model: dict[str, Any]) -> None:
    """Make discrimination and baseline comparisons visually explicit."""
    names = ['Accuracy', 'F1']
    fitted = [model['test_accuracy'], model['test_f1']]
    baseline = [model['baseline_accuracy'], model['baseline_f1']]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 3.9))
    w = .28
    ax.bar(x - w / 2, fitted, w, color=BLUE,
           label=model.get('model_label', 'Selected model'))
    ax.bar(x + w / 2, baseline, w, color=GOLD,
           label='Majority-class baseline')
    ax.set_xticks(x, names)
    ax.set_ylim(0, .9)
    set_axes(ax, 'Held-out performance | baseline comparison',
             'Evaluation metric')
    ax.legend(frameon=False)
    ax.text(.02, .92, f"ROC-AUC: {model['test_roc_auc']:.3f} "
            '(near chance)', transform=ax.transAxes, fontsize=10,
            weight='bold')
    export('06_model_baseline.png')


def load_stage_metrics() -> dict[str, Any]:
    """Prefer regenerated repository metrics over the bundled snapshot."""
    original = [ROOT / 'outputs/eda_metrics.json',
                ROOT / 'eda_metrics.json']
    for path in original:
        if path.is_file():
            source = json.loads(path.read_text(encoding='utf-8'))
            stage = source['stage_summary']
            return {
                'group': {'n': stage['Group']['n'],
                          'three_plus': stage['Group']['high_scoring']},
                'knockout': {'n': stage['Knockout']['n'],
                             'three_plus': stage['Knockout']['high_scoring']},
                'season_cluster_bootstrap_95': [
                    source['season_bootstrap']['ci_low'],
                    source['season_bootstrap']['ci_high']],
                'chi_square': source['chi_square']['chi2'],
                'chi_square_p': source['chi_square']['p']}
    path = first_existing('supporting_data/stage_summary.json')
    return json.loads(path.read_text(encoding='utf-8'))


def load_model_metrics() -> dict[str, Any]:
    """Normalize the existing full team JSON or its exact small snapshot."""
    original = [ROOT / 'outputs/model_metrics.json',
                ROOT / 'model_metrics.json']
    for path in original:
        if path.is_file():
            data = json.loads(path.read_text(encoding='utf-8'))
            test = data['test_results']
            return {'model': str(data['selected']),
                    'model_label': {'logistic': 'Logistic regression',
                                    'forest': 'Random forest'}.get(
                        data['selected']['name'], 'Selected model'),
                    'train': data['sample_sizes']['train'],
                    'validation': data['sample_sizes']['validation'],
                    'test': data['sample_sizes']['test'],
                    'test_accuracy': test['accuracy'],
                    'test_precision': test['precision'],
                    'test_recall': test['recall'],
                    'test_f1': test['f1'],
                    'test_roc_auc': test['roc_auc'],
                    'baseline_accuracy': test['baseline_majority_accuracy'],
                    'baseline_f1': test['baseline_majority_f1']}
    path = first_existing('supporting_data/model_metrics.json')
    return json.loads(path.read_text(encoding='utf-8'))


def create_all() -> dict[str, Any]:
    """Run checks, save figures and write a machine-readable results file."""
    seasons = pd.read_csv(first_existing('outputs/season_summary.csv',
                                         'season_summary.csv',
                                         'supporting_data/season_summary.csv'))
    players = pd.read_csv(first_existing(
        'data/players_uefa_2026_11_12.csv',
        'players_uefa_2026_11_12.csv',
        'supporting_data/players_uefa_2026_11_12.csv'))
    peaks = pd.read_csv(first_existing(
        'data/verified_peak_seasons.csv',
        'verified_peak_seasons.csv',
        'supporting_data/verified_peak_seasons.csv'))
    stage = load_stage_metrics()
    model = load_model_metrics()
    cleaned = validate_seasons(seasons)
    fixture_paths = [ROOT / 'data/matches_2011_2024.csv',
                     ROOT / 'matches_2011_2024.csv']
    raw_present = False
    for path in fixture_paths:
        if path.exists():
            validate_full_matches(pd.read_csv(path), cleaned)
            raw_present = True
            break
    stage_stats = stage_effect(stage)
    trend = trend_analysis(cleaned)
    plot_players(players)
    plot_peaks(peaks)
    plot_seasons(cleaned, trend['slope_goals_per_game_per_season'])
    plot_stage(stage_stats)
    plot_model(model)
    report = {'fixture_level_reconciled': raw_present,
              'data_source': 'project repository or exact small snapshots',
              'stage': stage_stats, 'trend': trend,
              'model': model,
              'players': player_table(players).replace(
                  {np.nan: None}).to_dict(orient='records')}
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / 'final_results.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding='utf-8')
    print(json.dumps({'fixture_level_reconciled': raw_present,
                      'stage': stage_stats, 'trend': trend}, indent=2))
    return report


def main() -> None:
    """Regenerate the six report figures and audited numerical results."""
    create_all()


if __name__ == '__main__':
    main()
