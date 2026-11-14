"""Download and strictly parse OpenFootball's 2011/12-2023/24 CL files.

Run with internet: python collect_openfootball.py --fetch
Or use a local clone: python collect_openfootball.py --local-root /path/to/repo
A season is rejected if its parsed count differs from its source header.

Authors: Rodrigo Li and Larry Zhang
Course: CSE 163, Autumn 2026
Section: AC (Rodrigo Li)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_URL = ('https://raw.githubusercontent.com/openfootball/'
           'champions-league/master/{season}/cl.txt')
DATE_PATTERN = re.compile(
    r'^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+'
    r'([A-Za-z]{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$'
)
GAME_PATTERN = re.compile(
    r'^\s*(?:\d{1,2}:\d{2}\s+)?(.+?)\s+v\s+(.+?)\s{2,}'
    r'(\d+-\d+)(.*)$'
)
COUNT_PATTERN = re.compile(r'^# Matches\s+(\d+)')
SCORE_PATTERN = re.compile(r'(\d+)-(\d+)')
AET_PATTERN = re.compile(r'(\d+)-(\d+)\s+a\.e\.t\.')


def parse_score(initial_score: str, suffix: str) -> tuple[int, int]:
    """Return home and away goals from a source scoreline.

        Read an extra-time score from suffix when supplied and exclude
        penalty-shootout conversions. Raise ValueError on unusable scores."""
    overtime = AET_PATTERN.search(initial_score + suffix)
    if overtime:
        return int(overtime.group(1)), int(overtime.group(2))
    match = SCORE_PATTERN.fullmatch(initial_score)
    if match is None:
        raise ValueError('Malformed score')
    if 'pen.' in suffix:
        raise ValueError('Penalty line lacks extra-time score')
    return int(match.group(1)), int(match.group(2))


def parse_cl(text: str, season: str) -> pd.DataFrame:
    """Parse main-draw games from the given season's source text.

        Return a match DataFrame; reject incomplete counts and invalid
        group-stage totals, including localized source headings."""
    stage = None
    date = None
    expected = None
    result = []
    start_year = int(season[:4])
    for line in text.splitlines():
        count = COUNT_PATTERN.match(line.strip())
        if count:
            expected = int(count.group(1))
        if line.strip().startswith('▪'):
            stage = ('Group' if re.search(r'\b(?:Group|Gruppe)\b', line)
                     else 'Knockout')
            continue
        date_match = DATE_PATTERN.match(line)
        if date_match:
            month, day, year = date_match.groups()
            month_number = datetime.strptime(month, '%b').month
            inferred_year = (start_year if month_number >= 8
                             else start_year + 1)
            # The pandemic postponed 2019/20 knockouts to August 2020;
            # those dates omit their year in the source text.
            if (season == '2019-20' and stage == 'Knockout' and
                    month_number == 8):
                inferred_year = start_year + 1
            year = int(year) if year else inferred_year
            date = datetime(year, month_number, int(day)).date().isoformat()
            continue
        match = GAME_PATTERN.match(line)
        if match and stage and date:
            home, away, score, suffix = match.groups()
            hg, ag = parse_score(score, suffix)
            result.append({'season': season, 'date': date, 'stage': stage,
                           'home_team': home.strip(),
                           'away_team': away.strip(),
                           'home_goals': hg, 'away_goals': ag,
                           'source': RAW_URL.format(season=season)})
        elif (stage and date and ' v ' in line and
              re.search(r'\d+-\d+', line)):
            raise ValueError(f'Unparsed fixture in {season}: {line!r}')
    if expected is None:
        raise ValueError(f'{season}: source has no match-count header')
    if len(result) != expected:
        raise ValueError(f'{season}: parsed {len(result)}; '
                         f'source declares {expected}')
    # All 2011/12-2023/24 main-draw seasons had 96 group games.
    # Some 2020-23 files use the German heading 'Gruppe G/H'.
    group_count = sum(row['stage'] == 'Group' for row in result)
    if expected in (119, 125) and group_count != 96:
        raise ValueError(f'{season}: expected 96 group matches, '
                         f'found {group_count}')
    return pd.DataFrame(result)


def get_season(season: str, local_root: Path | None = None) -> str:
    """Read the text of season from local_root or the source URL.

        Return the UTF-8 season file; fetch online only without local_root."""
    if local_root is not None:
        path = local_root / season / 'cl.txt'
        return path.read_text(encoding='utf-8')
    url = RAW_URL.format(season=season)
    request = Request(url, headers={'User-Agent': 'CSE163-EDA/1.0'})
    with urlopen(request, timeout=30) as response:
        return response.read().decode('utf-8')


def collect(local_root: Path | None = None) -> pd.DataFrame:
    """Collect the 13 main-competition seasons from local_root or web.

        Validate and save the combined match CSV; return its DataFrame."""
    combined = []
    for year in range(2011, 2024):
        season = f'{year}-{str(year + 1)[-2:]}'
        text = get_season(season, local_root=local_root)
        parsed = parse_cl(text, season)
        combined.append(parsed)
        print(season, len(parsed), 'checked against source header')
    all_matches = pd.concat(combined, ignore_index=True)
    all_matches.to_csv(ROOT / 'data/matches_2011_2024.csv', index=False)
    print('Saved', len(all_matches), 'season-specific match records')
    return all_matches


def main() -> None:
    """Parse command-line source options and collect the match records."""
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument('--fetch', action='store_true')
    options.add_argument('--local-root', type=Path)
    arguments = parser.parse_args()
    collect(local_root=arguments.local_root)


if __name__ == '__main__':
    main()
