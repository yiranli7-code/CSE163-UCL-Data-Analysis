"""Season-held-out UCL high-scoring-match modeling (challenge goal).

Run after collecting all 13 seasons: python model.py
No test-set evaluation occurs until train/validation choices are fixed.

Authors: Rodrigo Li and Larry Zhang
Course: CSE 163, Autumn 2026
Section: AC (Rodrigo Li)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from eda import add_prematch_form


ROOT = Path(__file__).resolve().parent
FEATURES = ['home_gf_prior', 'home_ga_prior', 'away_gf_prior',
            'away_ga_prior', 'stage']
NUMERIC = FEATURES[:-1]


def get_partitions(frame: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Split frame into chronological train, validation and test sets.

        Return three season-disjoint DataFrames; reject missing partitions."""
    earlier = frame[frame['season'] <= '2019-20']
    validation = frame[(frame['season'] >= '2020-21') &
                       (frame['season'] <= '2021-22')]
    test = frame[frame['season'] >= '2022-23']
    if min(len(earlier), len(validation), len(test)) == 0:
        raise ValueError('Need 2011/12-2023/24 data; run '
                         'collect_openfootball.py first')
    return earlier, validation, test


def make_estimator(name: str, parameter: float | int | None) -> Pipeline:
    """Construct a pipeline for model name and its hyperparameter.

        Use parameter as logistic C or forest max depth; return a
        preprocessing-plus-classifier Pipeline, or reject unknown names."""
    preprocess = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC),
        ('stage', OneHotEncoder(handle_unknown='ignore'), ['stage'])
    ])
    if name == 'logistic':
        classifier = LogisticRegression(C=parameter, max_iter=2000,
                                        random_state=163)
    elif name == 'forest':
        classifier = RandomForestClassifier(n_estimators=200,
                                            max_depth=parameter,
                                            min_samples_leaf=3,
                                            random_state=163,
                                            n_jobs=-1)
    else:
        raise ValueError('Unknown model')
    return Pipeline([('preprocess', preprocess), ('model', classifier)])


def scores(fitted: Pipeline, data: pd.DataFrame) -> dict[str, float | None]:
    """Evaluate fitted classification pipeline on labeled data.

        Return accuracy, precision, recall, F1 and ROC-AUC; the AUC is None
        if the evaluation outcome contains only one class."""
    actual = data['high_scoring']
    predicted = fitted.predict(data[FEATURES])
    probabilities = fitted.predict_proba(data[FEATURES])[:, 1]
    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    result = {'accuracy': float(accuracy_score(actual, predicted)),
              'precision': float(precision),
              'recall': float(recall),
              'f1': float(f1_score(actual, predicted, zero_division=0))}
    result['roc_auc'] = (float(roc_auc_score(actual, probabilities))
                         if actual.nunique() == 2 else None)
    return result


def run_model() -> dict[str, Any]:
    """Select a model on validation seasons and evaluate it on test.

        Read match data, save outputs/model_metrics.json and return
        selection details, validation results and held-out metrics."""
    path = ROOT / 'data/matches_2011_2024.csv'
    if not path.exists():
        raise FileNotFoundError('Run collect_openfootball.py before model.py')
    data = pd.read_csv(path)
    features = add_prematch_form(data)
    features = features.dropna(subset=NUMERIC).copy()
    train, valid, test = get_partitions(features)
    choices = [('logistic', 0.1), ('logistic', 1.0),
               ('logistic', 10.0), ('forest', 4),
               ('forest', 8), ('forest', None)]
    best = None
    best_f1 = -1.0
    validations = []
    for name, parameter in choices:
        estimator = make_estimator(name, parameter)
        estimator.fit(train[FEATURES], train['high_scoring'])
        measures = scores(estimator, valid)
        validations.append({'model': name, 'parameter': parameter,
                            **measures})
        if measures['f1'] > best_f1:
            best_f1 = measures['f1']
            best = (name, parameter)
    name, parameter = best
    combined = pd.concat([train, valid], ignore_index=True)
    fitted = make_estimator(name, parameter)
    fitted.fit(combined[FEATURES], combined['high_scoring'])
    test_scores = scores(fitted, test)
    baseline_class = int(combined['high_scoring'].mean() >= 0.5)
    baseline = np.full(len(test), baseline_class)
    test_scores['baseline_majority_accuracy'] = float(
        accuracy_score(test['high_scoring'], baseline))
    test_scores['baseline_majority_f1'] = float(
        f1_score(test['high_scoring'], baseline, zero_division=0))
    result = {'selected': {'name': name, 'parameter': parameter},
              'validation_results': validations, 'test_results': test_scores,
              'sample_sizes': {'train': len(train), 'validation': len(valid),
                               'test': len(test)}}
    out = ROOT / 'outputs/model_metrics.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    """Run the multi-season machine-learning evaluation."""
    run_model()


if __name__ == '__main__':
    main()
