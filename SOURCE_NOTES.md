# Data provenance and audit — November 12, 2026

## Match data

Original: https://github.com/openfootball/champions-league (CC0 licensed).

A complete user-supplied `champions-league-master.zip` archive, dated July 2, 2026 in ZIP entries, contained the season source files. We copied only the original Champions League `2011-12/cl.txt` through `2023-24/cl.txt` files into `source_openfootball/`, along with the project license. We did not use Europa League, Conference League, qualifiers, or 2024/25 onwards.

The parser `collect_openfootball.py` takes these raw files and writes `matches_2011_2024.csv`. It confirms each season has the source's declared number of matches and 96 group matches. Individual files contain 125 completed games except 2019/20 (119 due to COVID format). Total: **1,619 matches, 1,248 group, 371 knockout**, 4,809 match goals, 916 games with 3+ goals. Pre-match variables reset each season and use only earlier dates.

Three notable traps caught during verification:

- German `▪ Gruppe G/H` header lines in 2020/21, 2021/22 and 2022/23 denote **group matches**. An English-only parser would misclassify 72 games as knockout.
- Postponed 2019/20 August knockouts are dated **August 2020** even though lines do not repeat the year. Incorrectly placing them in August 2019 creates data leakage in "prior-match" features.
- `pen.` shootout conversions are not match score goals; use extra-time score `a.e.t.` for those lines.

`matches_2023_24.csv` is a separately transcribed earlier pilot snapshot; a test verifies that its 125 rows match the complete source parser output. The main EDA and model use `matches_2011_2024.csv`, not this pilot.

## UEFA player data

`players_uefa_2026_11_12.csv` and `verified_peak_seasons.csv` are manually recorded, source-specific subsets from the previous project proposal, which are **not** present in OpenFootball's `cl.txt`. Data references:

- https://www.uefa.com/uefachampionsleague/history/rankings/players/goals_scored/
- https://www.uefa.com/uefachampionsleague/history/rankings/players/assists/
- https://www.uefa.com/uefachampionsleague/history/rankings/players/matches_played/
- https://www.uefa.com/uefachampionsleague/history/rankings/players/season_goals_scored/

Two players lack confirmed appearance denominators in the snapshot, so those values are kept as NA. UEFA rankings may not count the same competition stages as other UEFA historical articles. Selected season figures are not a comprehensive all-season dataset.

## Integrity and current limitations

- Parser matches each raw season header, but a declared match count alone is not independent certification that every fixture score is correct. Source score formats and match date chronology were specifically inspected; sample matches were cross-checked.
- The full chi-square test's expected counts exceed five, but independence is imperfect because clubs have repeated observations.
- Season-block bootstrap partially reflects across-season variability, not all within-season dependence.
- Holdout seasons were not used to select the model. The resulting near-chance ROC-AUC is explicitly reported.
- The uploaded source ZIP and generated CSV checksums are recorded in `SOURCE_HASHES.txt`.
