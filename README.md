# CSE 163 — Cristiano Ronaldo's Champions League Legacy

**Part 3 review package | Autumn 2026**  
**Authors:** Rodrigo Li and Larry Zhang  
**Public team repository:** https://github.com/yiranli7-code/CSE163-UCL-Data-Analysis  
**Course deadline:** Friday, December 11, 2026, 11:59 p.m. Pacific  
**Submission format:** Export the reviewed Word report to PDF; include the public GitHub repository link in the PDF. The code stays in GitHub, not as a Gradescope code upload for Part 3.

## Scope and integrity

This folder is a **Part 3 extension** designed for review and incorporation into the existing team GitHub repository; it is **not** a reconstructed clone. Do not delete or replace the existing project's `collect_openfootball.py`, `eda.py`, `model.py`, `test_eda.py`, `test_model.py`, the original 13 `cl.txt` inputs, or its data files. The existing repository is the authoritative source for the full 1,619-match dataset. The supplied `supporting_data/` tables are small exact snapshots of checked-in team outputs read on November 23, 2026. They permit regenerating the new figures offline, but do not by themselves certify original match-by-match parsing. `final_results.json` records whether the full fixture-level reconciliation actually ran.

**Dataset distinction:** player-level career goals, assists and appearances are *UEFA-derived snapshots*; match-level season, scoring and classifier results are *OpenFootball-derived*. They must not be merged by matching on player: the match dataset does not contain historic player-level minutes or assists. The machine-learning classifier predicts whether a full UCL match has 3+ goals, **not Ronaldo's next-season goal count**.

## Contents of this Part 3 supplement

- The editable report draft is distributed separately; review all 12 required rubric sections before PDF submission.
- `requirements.txt`: dependencies for the supplement and full repository pipeline.
- `final_stats.py`: pure validation/statistics functions: season identities, player exposure metrics, stage effect and Fisher test, permutation-checked seasonal trend, and raw-fixture reconciliation.
- `final_analysis.py`: runnable report pipeline; writes six titled figures and numeric results using existing team tables when present.
- `test_final_stats.py`: 13 deterministic unit tests for statistical formulas, missing-data handling, input validation and fixture reconciliation.
- `supporting_data/season_summary.csv`: exact 13-season team output snapshot.
- `supporting_data/players_uefa_2026_11_12.csv`: exact five-player team snapshot.
- `supporting_data/verified_peak_seasons.csv`: exact selected player-season records from the team's Part 2 work.
- `supporting_data/stage_summary.json`: exact aggregate table and uncertainty output values copied from the team's EDA artifacts.
- `supporting_data/model_metrics.json`: exact held-out model results copied from the team's model artifact.
- `final_figures/`: six figures generated programmatically and embedded in the Word draft.
- `final_outputs/final_results.json`: final numeric artifact; `fixture_level_reconciled` is false in the standalone package and automatically true if the original fixture file is present and passes identity checks.

## Reproduce this Part 3 extension offline

Use Python 3.10 or newer. From the folder containing `final_analysis.py`:

```bash
python -m pip install -r requirements.txt
python final_analysis.py
python -m unittest test_final_stats -v
python -m flake8 final_stats.py final_analysis.py test_final_stats.py --max-line-length 79
```

The script reads its data from paths relative to the **script file**, so it runs from any working directory. Figures appear in `final_figures/`, results in `final_outputs/`. A fixed `numpy` random seed makes the 20,000-permutation sensitivity check reproducible. The test file does not access the internet or require original UCL fixtures.

## Full repository reproduction and additional fixture-level audit

The `part3-final` branch includes the new scripts and small reference tables alongside the original Part 2 code and data. The reproduction commands generate `final_figures/` and `final_outputs/` locally. The dependency list includes the original repository requirements. From that repository root:

```bash
python -m pip install -r requirements.txt
python collect_openfootball.py --local-root source_openfootball
python eda.py
python model.py
python final_analysis.py
python -m unittest discover -p 'test_*.py' -v
python -m flake8 collect_openfootball.py eda.py model.py test_eda.py test_model.py make_report.py final_stats.py final_analysis.py test_final_stats.py --max-line-length 100
```

When `data/matches_2011_2024.csv` exists, `final_analysis.py` independently reconciles fixture counts and goal totals with all 13 season summaries; it raises an error if they disagree. Check that `final_outputs/final_results.json` contains `"fixture_level_reconciled": true` in the final repository run.

## Original repository modules (preserve these)

- `collect_openfootball.py`: parses the 13 original OpenFootball season files, verifies fixture counts, and writes the match CSV.
- `eda.py`: prepares and validates player and match data, builds pre-match features, computes exploratory statistics, and generates the original EDA figures and outputs.
- `model.py`: fits logistic regression and random forest with chronological training, validation and test partitions, and writes the model metrics.
- `test_eda.py`: original parser and EDA unit/integration tests.
- `test_model.py`: original chronological partition and model smoke tests.
- `make_report.py`: regenerates the historical Part 2 EDA PDF; not needed to generate the final Part 3 figures or to export the reviewed Word report.

The original `data/`, `source_openfootball/`, `outputs/`, and `figures/` directories should be retained. Run the full reproduction commands before submission. The standalone Part 3 package passed all 13 tests, figure generation, syntax compilation, and flake8 with the default 79-character limit on the author's Mac (November 23, 2026). The complete integrated repository run remains an author check. The supplemental scripts are formatted to a maximum of 79 characters per line; the full-repository example retains the earlier 100-character setting. Use the course's required lint configuration if it differs.

## Figures and reproducibility reference

1. `01_career_contributions.png` — selected career goals and assists; no claim of post-2011-only totals.
2. `02_appearance_rates.png` — per-appearance rates for only the three players with denominators; no per-90 estimate.
3. `03_selected_peak_seasons.png` — selected documented player seasons, including two outside the match sample; not a full census.
4. `04_season_goal_trend.png` — 13-season average goals/game with exploratory fitted line.
5. `05_stage_goal_frequency.png` — observed percentage of games with 3+ goals by stage.
6. `06_model_baseline.png` — test-sample accuracy/F1 for random forest vs majority baseline; AUC reported separately.

**Quality controls:** Each figure's title, caption, source and interpretation appear in the report. No unknown appearance denominator is imputed. Season-level permutation testing is a sensitivity diagnostic, not proof of independent seasons or a causal trend. The existing project tests are not duplicated; they must still be run in the original repository.

## Before submission (author review required)

1. Confirm each UEFA-listed player total and historical peak-season line against the original cited UEFA pages on the final submission date.
2. Have Rodrigo and Larry verify the original proposal wording, actual time/work allocation, any TA feedback, and the scope of consultations/AI assistance. Do not invent contributions or hours.
3. Run the **full** commands above inside the original repository, not only against the small snapshots. Resolve `flake8` findings there; the original repo previously noted flake8 was unavailable during its build.
4. Read the Word report and adjust any highlighted author-review fields. Export the final reviewed `.docx` to PDF. Verify 11-point or larger fonts, non-single spacing, page numbering, the 15-page report-body limit, title/caption for each visual, and the public GitHub link.
5. Commit the additional `.py` modules and generated figures to the public team repository only after both teammates review them. Gradescope requires the **PDF** final report.

## References

- Team repository: https://github.com/yiranli7-code/CSE163-UCL-Data-Analysis
- OpenFootball Champions League (CC0): https://github.com/openfootball/champions-league
- UEFA historical rankings: https://www.uefa.com/uefachampionsleague/history/rankings/players/goals_scored/
- Dixon, M. J., & Coles, S. G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*, Journal of the Royal Statistical Society C, 46(2), 265–280. https://doi.org/10.1111/1467-9876.00065
