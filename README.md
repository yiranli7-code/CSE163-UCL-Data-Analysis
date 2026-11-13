# CSE 163 — Cristiano Ronaldo's Champions League Legacy

**Part 2: Exploratory Data Analysis | November 12, 2026**  
**Authors:** Rodrigo Li and Larry Zhang  
**Public team repository:** https://github.com/yiranli7-code/CSE163-UCL-Data-Analysis  
**Required Gradescope report:** `eda.pdf`

## What is in this project

- `eda.pdf` — 8-page numbered report, with seven-number summaries, categorical counts, eight titled/captioned plots, hypothesis check, and challenge-goal updates.
- `eda.py` — Python EDA and plotting script. Generates all eight figures and numerical outputs.
- `test_eda.py` — 21 deterministic EDA/parser unit and integration tests.
- `collect_openfootball.py` — parser for the 13 official `cl.txt` source files. Checks each file's `# Matches` header AND 96 group games per season; handles `Gruppe`, penalty shootouts, and August 2020 pandemic knockouts.
- `model.py` — regularized logistic regression and random forest, with season-based train/validation/test and an explicit majority baseline.
- `test_model.py` — 3 modeling tests.
- `make_report.py` — regenerates PDF using the actual computed results and public team URL.
- `data/` — two UEFA-derived player tables, the combined **1,619-row** parsed match CSV, an original 2023/24 single-season transcription used as an independent test reference, and provenance notes.
- `source_openfootball/` — 13 raw `cl.txt` files (2011/12–2023/24) as supplied by the team, together with the source project's CC0 license.
- `figures/` — eight plotted PNG figures.
- `outputs/` — cleaned datasets, leakage-safe pre-match features, season summary and numeric JSON results.

## Reproduce the analysis

Use Python 3.10+ from the project directory:

```bash
python -m pip install -r requirements.txt
python collect_openfootball.py --local-root source_openfootball
python eda.py
python model.py
python -m unittest discover -p "test_*.py" -v
python make_report.py
python -m flake8 collect_openfootball.py eda.py model.py test_eda.py test_model.py make_report.py --max-line-length 100
```

**Note:** `flake8` is included in `requirements.txt` but was unavailable to install in the build environment. All 24 unit tests passed, all Python files passed AST syntax validation, and no file has a line exceeding 100 characters. **Please run flake8 locally before submitting.**

You can also check source data without downloading anything: the exact 13 season text files are included. No web access is needed for reproduction. The analysis script does not modify source tables.

## Results (all from real data)

- **13 seasons, 1,619 matches** (12 years ×125, plus 119 games in 2019/20); **1,248 group +371 knockout** fixtures.
- **4,809 match goals**; **916 matches (56.6%)** have at least three goals.
- 1,411 of 1,619 matches have four available pre-match team-form predictors; 208 opening games are excluded from ML.
- Chi-square comparison of stage vs 3+ match goals: χ²=0.889, p=0.346, Cramér's V=0.023; min expected cell=161.1. The season-cluster bootstrap includes zero. The test's independent-match assumption is not completely satisfied due to repeated teams.
- Selected random forest (max depth 4) held-out test: accuracy 0.546, F1 0.703, ROC-AUC 0.473, versus majority-class baseline F1 0.699. The model has little practical discrimination despite a marginal F1 improvement.
- All 24 unittest cases pass on supplied data; all 8 figure PNGs are generated from CSVs using `eda.py`.

## Data limitations and methodology

The 13-season OpenFootball source contains **match fixtures and scores**, not historical per-player assists or playing minutes. The player analyses (RQ1–RQ3) continue to use the separate UEFA-derived snapshots from the proposal; historical ranking scope and missing appearance denominators are disclosed in the PDF. Do not interpret team-level model results as a causal measure of Ronaldo's impact.

Special source handling:

1. Some 2020/21–2022/23 group headings read `Gruppe G/H`; these must be **Group**, not Knockout.
2. 2019/20 UEFA knockout matches postponed to August occurred in **2020**, not August 2019, even when source date lines omit the year.
3. For score text such as `4-2 pen. 1-0 a.e.t.`, use **1-0** (the match result before penalties), not 4-2.
4. Include all completed main-draw fixtures in each of 2011/12–2023/24. Do not include qualifying games or the league-phase format introduced after 2023/24.

See `data/SOURCE_NOTES.md` for original links, audit notes and reproducibility checks.

## Publish to team GitHub and submit

1. Open your public repo: https://github.com/yiranli7-code/CSE163-UCL-Data-Analysis
2. In Finder, unzip the **updated** project package. On the repository page select **Add file → Upload files**; drag/upload the contents of the extracted project folder, preserving its subfolders. Alternatively, use GitHub Desktop or the Git CLI. Do **not** upload only the ZIP — staff should be able to browse the `.py`, data and figures directly.
3. Confirm the GitHub files open publicly. Invite Larry via **Settings → Collaborators** if needed.
4. Run flake8 locally and review the report with Larry. Personal hours and any additional consultation facts must be verified by the authors rather than invented. Confirm acceptable AI-assistance disclosure under the course's policy.
5. Submit **`eda.pdf`** and at minimum **`eda.py`** and **`test_eda.py`** to Gradescope, plus any other required code/data files. The public repository should contain the complete code and data dependencies. Due: **November 15, 2026, 11:59 p.m. PST**.

This repository is prepared for upload, but no claim is made that any GitHub files were uploaded from ChatGPT or that the Gradescope submission was completed.
