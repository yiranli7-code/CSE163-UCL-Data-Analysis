"""Create the reproducible, numbered CSE 163 Part 2 EDA PDF.

Run: python make_report.py
The report is generated from outputs/eda_metrics.json and model_metrics.json.

Authors: Rodrigo Li and Larry Zhang
Course: CSE 163, Autumn 2026
Section: AC (Rodrigo Li)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
URL = 'https://github.com/yiranli7-code/CSE163-UCL-Data-Analysis'
FONT_DIR = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
pdfmetrics.registerFont(TTFont('DejaVu', str(FONT_DIR / 'DejaVuSans.ttf')))
pdfmetrics.registerFont(
    TTFont('DejaVuBold', str(FONT_DIR / 'DejaVuSans-Bold.ttf'))
)
NAVY = colors.HexColor('#153b64')
TEAL = colors.HexColor('#107b81')
TEXT = colors.HexColor('#263846')

STYLES = {
    'title': ParagraphStyle('TitleCustom', fontName='DejaVuBold',
                            fontSize=15.5, leading=22, textColor=NAVY,
                            spaceAfter=8),
    'subtitle': ParagraphStyle('SubtitleCustom', fontName='DejaVu',
                               fontSize=11, leading=16.3,
                               textColor=TEXT, spaceAfter=8),
    'h': ParagraphStyle('HeadingCustom', fontName='DejaVuBold',
                        fontSize=12, leading=16.5,
                        textColor=NAVY, spaceBefore=9, spaceAfter=5),
    'subh': ParagraphStyle('SubheadingCustom', fontName='DejaVuBold',
                           fontSize=11.1, leading=16.0,
                           textColor=TEAL, spaceBefore=6, spaceAfter=3),
    'body': ParagraphStyle('BodyCustom', fontName='DejaVu',
                           fontSize=11, leading=15.8,
                           textColor=TEXT, spaceAfter=7),
    'caption': ParagraphStyle('CaptionCustom', fontName='DejaVu',
                              fontSize=11, leading=15.1,
                              textColor=TEXT, spaceAfter=5),
    'cell': ParagraphStyle('CellCustom', fontName='DejaVu',
                           fontSize=11, leading=14.2, textColor=TEXT),
    'head': ParagraphStyle('HeadCustom', fontName='DejaVuBold',
                           fontSize=11, leading=14.2,
                           textColor=colors.white),
}


def para(string: str, style: str = 'body') -> Paragraph:
    """Return a paragraph from string in the named report style."""
    return Paragraph(string, STYLES[style])


def table(rows: list[list[object]], widths: list[int],
          first_header: bool = True) -> Table:
    """Return a styled report table using rows, widths and header flag."""
    cell_rows = []
    for index, values in enumerate(rows):
        style = 'head' if index == 0 and first_header else 'cell'
        cell_rows.append([para(str(value), style) for value in values])
    result = Table(cell_rows, colWidths=widths, hAlign='LEFT',
                   repeatRows=1 if first_header else 0)
    result.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#f0f5f8')]),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return result


def seven_table(info: dict[str, Any],
                cols: list[tuple[str, str, int]]) -> Table:
    """Return a statistics table for info and labeled cols.

        Each cols tuple gives a displayed label, metric key and sample size."""
    out = [['Variable', 'n', 'Mean', 'SD', 'Min', 'Q1', 'Med', 'Q3', 'Max']]
    for label, key, n in cols:
        values = info[key]
        stats = []
        for stat in ['mean', 'std', 'min', 'q1', 'median', 'q3', 'max']:
            val = values[stat]
            stats.append('—' if val is None else f'{val:.2f}')
        out.append([label, str(n)] + stats)
    return table(out, [91, 44, 51, 51, 49, 49, 48, 48, 49])


def figure(filename: str, caption: str, interpretation: str,
           height: int = 171) -> KeepTogether:
    """Return a plot with its caption and interpretation grouped.

        Use filename under figures/ and cap the plot height in points."""
    from PIL import Image as PILImage
    path = ROOT / 'figures' / filename
    with PILImage.open(path) as original:
        width_px, height_px = original.size
    scale = min(458 / width_px, height / height_px)
    image = Image(str(path), width=width_px * scale,
                  height=height_px * scale)
    image.hAlign = 'CENTER'
    return KeepTogether([image, Spacer(1, 4), para(caption, 'caption'),
                         para(interpretation, 'body'), Spacer(1, 7)])


def footer(canvas: Canvas, document: BaseDocTemplate) -> None:
    """Draw the numbered footer on canvas for the given document page."""
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#e3eaf0'))
    canvas.line(54, 43, 558, 43)
    canvas.setFillColor(TEXT)
    canvas.setFont('DejaVu', 11)
    canvas.drawString(54, 29, 'CSE 163  |  Final Project Part 2 — EDA')
    canvas.drawRightString(558, 29, str(document.page))
    canvas.restoreState()


def make_report() -> None:
    """Build a report from stored metrics and plotted figures.

        Overwrites eda.pdf; do not run after manually editing that PDF."""
    summary = json.loads((ROOT / 'outputs/eda_metrics.json').read_text())
    model = json.loads((ROOT / 'outputs/model_metrics.json').read_text())
    doc = BaseDocTemplate(str(ROOT / 'eda.pdf'), pagesize=letter,
                          leftMargin=54, rightMargin=54,
                          topMargin=49, bottomMargin=55,
                          title='Champions League Legacy - CSE 163 EDA',
                          author='Rodrigo Li and Larry Zhang')
    frame = Frame(54, 55, 504, 688, id='normal',
                  leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id='page', frames=[frame],
                                      onPage=footer))
    work = []
    add = work.append

    def section(num: int, text: str) -> None:
        """Append a numbered section heading with num and text."""
        add(para(f'{num}. {text}', 'h'))

    def page() -> None:
        """Append a page break to the report's flowables."""
        add(PageBreak())

    # Page 1 - Research framing, scope, and method.
    add(para('Cristiano Ronaldo’s Champions League Legacy:<br/>'
             'Exploratory Data Analysis', 'title'))
    add(para('<b>CSE 163 | Project Part 2 | November 12, 2026</b><br/>'
             '<b>Authors:</b> Rodrigo Li and Larry Zhang<br/>'
             '<b>Public team GitHub:</b> <link color="#107b81" href="' +
             escape(URL) + '">' + escape(URL) + '</link>', 'subtitle'))
    section(1, 'Summary of Research Questions')
    add(para('<b>RQ1 — Career contributions:</b> Compare Cristiano Ronaldo '
             'with Lionel Messi, Robert Lewandowski, Karim Benzema and Kylian '
             'Mbappé using UEFA-listed goals, assists and G+A totals. '
             '<b>RQ2 — Efficiency:</b> Examine goals and goal contributions '
             'per appearance only when compatible appearance denominators '
             'are available.'))
    add(para('<b>RQ3 — Seasonal peaks:</b> Compare four selected Ronaldo '
             'seasons with standout seasons from the other players; avoid '
             'making up historical assists. <b>RQ4 — Match prediction:</b> '
             'Use regularized logistic regression and tuned random forests '
             'to predict whether a match has at least three total goals '
             'from competition stage and each side’s pre-match scoring form.'))
    section(2, 'Motivation')
    add(para('Career accumulation, exposure-adjusted rates and exceptional '
             'seasons express different kinds of football performance. '
             'Our goal is a transparent comparison, not a predetermined '
             '“greatest player” conclusion. The match-level model is a '
             'separate statistical-learning application, not evidence of '
             'Ronaldo’s individual causal impact on Real Madrid outcomes.'))
    section(3, 'Datasets and Provenance')
    add(para('<b>UEFA career snapshot:</b> five selected players × four '
             'source attributes (player, goals, assists, appearances). '
             '<b>UEFA selected peaks:</b> nine player-season entries × '
             'four fields (player, season, goals, source_note). '
             '<b>OpenFootball:</b> 13 raw <i>cl.txt</i> season files '
             '(2011/12–2023/24) from the uploaded CC0 GitHub archive, '
             'converted into 1,619 main-competition matches × eight source '
             'fields. The OpenFootball match records do not contain '
             'complete player assists or minutes and are not substituted '
             'for UEFA’s player ranking tables.'))
    section(4, 'Method')
    add(para('We parsed source scorelines and stage headings, excluding '
             'qualifying fixtures and shootout conversion scores while '
             'retaining extra-time goals. Every season’s record count was '
             'checked against its own <i># Matches</i> header. The parser '
             'also recognizes <i>Gruppe</i> headings in three seasons. '
             'Pandas then validated dates and unique fixtures, calculated '
             'total goals and a binary ≥3-goal indicator, and constructed '
             'prior-date goals-for/against averages separately for each '
             'club and season; same-day scores cannot leak into features. '
             'Summary statistics, eight figures, hypothesis tests and '
             'machine-learning evaluations were computed in Python.'))
    page()

    # Page 2 - All quantitative summaries and categorical frequencies.
    section(5, 'EDA Results: Size, Missingness and Variables')
    add(para('<b>Career table:</b> 5 × 4 raw and 5 × 7 with derived '
             'G+A and rate variables. The five player names are categorical '
             '(each appears once). Two appearance entries are missing '
             '(Lewandowski and Mbappé; 40%), and therefore two observations '
             'are missing in each per-appearance field (40%). Goals, '
             'assists and contributions have no missing values.'))
    add(seven_table(summary['players']['quantitative'], [
        ('Goals', 'goals', 5), ('Assists', 'assists', 5),
        ('Appearances', 'appearances', 3),
        ('G+A', 'goal_contributions', 5),
        ('Goals/app', 'goals_per_appearance', 3),
        ('G+A/app', 'contributions_per_appearance', 3)]))
    add(Spacer(1, 8))
    add(para('<b>Peak-season table:</b> 9 × 4, with no missing data '
             'by <i>isna().sum()</i>. Player category counts: Ronaldo 4; '
             'Mbappé 2; Messi, Benzema and Lewandowski 1 each. The nine '
             'season labels are 2007/08, 2011/12, 2013/14, 2015/16, '
             '2016/17, 2019/20, 2021/22, 2023/24 and 2025/26 '
             '(one record each). This is a purposely selected set.'))
    add(seven_table(summary['peak_seasons']['quantitative'], [
        ('Season goals', 'goals', 9)]))
    add(Spacer(1, 8))
    add(para('<b>Match table:</b> 1,619 × 8 parsed source columns; '
             '1,619 × 10 after total_goals and high_scoring. '
             'Each row represents one completed UCL match. '
             'Stage categories: Group 1,248; Knockout 371. '
             'There are zero missing source and derived match values '
             'according to <i>isna().sum()</i>, while 208 season-opening '
             'matches (12.85%) lack prior-game averages by design; '
             '1,411 matches are eligible for modeling.'))
    add(seven_table(summary['matches']['quantitative'], [
        ('Home goals', 'home_goals', 1619),
        ('Away goals', 'away_goals', 1619),
        ('Total goals', 'total_goals', 1619),
        ('High (0/1)', 'high_scoring', 1619)]))
    add(para('<b>Season category counts:</b> 2011/12, 2012/13, '
             '2013/14, 2014/15, 2015/16, 2016/17, 2017/18, '
             '2018/19 (125 each); 2019/20 (119); 2020/21, 2021/22, '
             '2022/23 and 2023/24 (125 each). Team labels are '
             'identifiers, not model features.', 'body'))
    page()

    # Page 3 - Historical player comparisons.
    add(para('5.1 Career Contributions and Efficiency (RQ1–RQ2)', 'h'))
    add(figure('fig1_career_contributions.png',
               '<b>Figure 1.</b> Ronaldo leads the five-player UEFA snapshot '
               'in listed career goals and observed goals plus assists.',
               'The stacked bar plot separates goals and assists: Ronaldo has '
               '141 + 42 = 183 listed contributions, compared with Messi’s '
               '129 + 40 = 169. The comparison '
               'uses one UEFA ranking convention '
               'but is not a claim about all '
               'competition-stage definitions.', 194))
    add(figure('fig2_rates_verified.png',
               '<b>Figure 2.</b> Messi has the highest goals per UEFA-listed '
               'appearance among the three '
               'players with observed denominators.',
               'The grouped bars show goals/appearance and G+A/appearance. '
               'Messi has 129/163 ≈ 0.79 goals per appearance, versus '
               'Ronaldo 141/187 ≈ 0.75 and Benzema 90/152 ≈ 0.59. '
               'Missing denominators for Lewandowski '
               'and Mbappé are not zero-filled.',
               194))
    page()

    # Page 4 - Selected seasons.
    add(para('5.2 Selected Seasonal Peaks (RQ3)', 'h'))
    add(figure('fig3_selected_seasons.png',
               '<b>Figure 3.</b> Ronaldo’s 2013/14 total of 17 goals '
               'is the largest among the nine selected player-seasons.',
               'A horizontal bar plot allows direct comparison of the source '
               'season-goal totals, including Ronaldo 2015/16 (16) and '
               'Benzema 2021/22 (15). The entries were deliberately selected, '
               'so this is not a distribution '
               'of all UCL player-seasons.', 201))
    add(figure('fig4_ronaldo_peak_seasons.png',
               '<b>Figure 4.</b> Ronaldo’s four highlighted seasons scored '
               '8, 17, 16 and 12 goals, respectively.',
               'The line connects only preselected 2007/08, 2013/14, '
               '2015/16 and 2016/17 campaigns. Uneven gaps and excluded '
               'seasons prevent a continuous career-trend interpretation; '
               'unavailable assists are not assumed to be zero.', 186))
    page()

    # Page 5 - 13-season match distributions.
    add(para('5.3 Thirteen-Season Match Results (RQ4)', 'h'))
    add(figure('fig5_match_goal_distribution.png',
               '<b>Figure 5.</b> Across 1,619 fixtures, mean scoring was '
               '2.97 goals per match and 916 '
               'games (56.6%) reached three goals.',
               'This discrete goal-count plot '
               'gives the distribution underlying '
               'the binary target; totals range from 0 to 12. The dashed '
               'threshold marks three goals, and extra-time goals count but '
               'shootout conversions do not.', 192))
    add(figure('fig6_stage_high_scoring.png',
               '<b>Figure 6.</b> High-scoring proportions were 714/1,248 '
               '(57.2%) for group fixtures and '
               '202/371 (54.4%) for knockout games.',
               'The bar plot compares proportions rather than raw counts '
               'because the stages have unequal numbers of fixtures. '
               'The 2.76-percentage-point knockout-minus-group difference '
               'is small and descriptive; team '
               'strength and year may confound it.',
               193))
    page()

    # Page 6 - Across-season variation.
    add(para('5.4 Seasonal Variation and Source Audit (RQ4)', 'h'))
    add(figure('fig7_season_goal_trend.png',
               '<b>Figure 7.</b> Seasonal average goals varied from '
               '2.76 (2011/12) to 3.24 (2019/'
               '20), without a monotonic pattern.',
               'The line chart highlights between-season variation in scoring '
               'opportunity and fixtures. The 2019/20 season had 119 games '
               'rather than 125, reflecting its altered knockout schedule; '
               'per-game averages remain interpretable.', 130))
    add(figure('fig8_stage_rates_by_season.png',
               '<b>Figure 8.</b> The high-scoring-rate gap between group and '
               'knockout games changes direction across individual seasons.',
               'The two lines show stage-specific yearly proportions instead '
               'of obscuring heterogeneity in the pooled comparison. Their '
               'variation motivates season-level uncertainty checks. '
               'This is descriptive, not evidence '
               'that stage causes more goals.',
               130))
    add(Spacer(1, 7))
    add(para('<b>Pre-match predictors:</b> The four team form variables '
             'represent goals scored (GF) or conceded (GA) per prior '
             'match, home and away. Each is missing for 208 opening '
             'fixtures and available for 1,411 matches each; '
             '1,411 matches have all four predictors.'))
    add(seven_table(summary['prematch']['quantitative'], [
        ('Home GF', 'home_gf_prior', 1411),
        ('Home GA', 'home_ga_prior', 1411),
        ('Away GF', 'away_gf_prior', 1411),
        ('Away GA', 'away_ga_prior', 1411)]))
    page()

    # Page 7 - Challenges, hypotheses, ML performance.
    section(6, 'Result Validity Challenge: Hypothesis and Assumptions')
    add(para('<b>H0:</b> Competition stage and a match’s ≥3-goal '
             'indicator are statistically independent. '
             '<b>H1:</b> They are associated. '
             'The planned Pearson chi-square test uses the 2 × 2 '
             'stage-by-target table. Observed rows [0–2, ≥3 goals] '
             'are Group [534, 714] and Knockout [169, 202].'))
    test = summary['chi_square']
    boot = summary['season_bootstrap']
    add(para('<b>Assumption check:</b> The smallest expected cell '
             'count is ' + f"{test['min_expected']:.1f}" +
             ' (≥5), so the large-sample count criterion is met. '
             f"χ²(1) = {test['chi2']:.3f}, p = {test['p']:.3f}, " +
             f"Cramér’s V = {test['cramers_v']:.3f}." +
             ' A two-sided 95% season-cluster bootstrap (3,000 draws; '
             'seed 163) estimates knockout-minus-group difference '
             f"{boot['difference'] * 100:.2f} percentage points; " +
             f"interval [{boot['ci_low'] * 100:.2f}, " +
             f"{boot['ci_high'] * 100:.2f}] points. " +
             'The interval includes zero.'))
    add(para('The strict independent-match assumption is questionable: '
             'clubs recur, matchups are strategic, and 2019/20 changed '
             'format. Resampling whole seasons partly addresses '
             'between-season heterogeneity but not within-season '
             'repeated-team dependence. Thus, the p-value is '
             'exploratory, not causal evidence or proof of equality.'))
    section(7, 'Challenge Goals Update: Machine Learning')
    add(para('The four numerical predictors were pre-match goals scored '
             'and conceded averages for each team; stage was one-hot '
             'encoded. We excluded 208 opening fixtures with no '
             'within-season history. Train: 2011/12–2019/20 (n=975); '
             'validation: 2020/21–2021/22 (n=218); held-out test: '
             '2022/23–2023/24 (n=218). The validation set selected '
             'random forest maximum depth 4; final fitting combined '
             'train and validation without using the test outcomes.'))
    ts = model['test_results']
    add(table([
        ['Held-out model', 'Accuracy', 'Precision', 'Recall', 'F1',
         'ROC-AUC'],
        ['Selected forest', f"{ts['accuracy']:.3f}",
         f"{ts['precision']:.3f}", f"{ts['recall']:.3f}",
         f"{ts['f1']:.3f}", f"{ts['roc_auc']:.3f}"],
        ['Majority baseline', f"{ts['baseline_majority_accuracy']:.3f}",
         '—', '—', f"{ts['baseline_majority_f1']:.3f}", '—'],
    ], [137, 74, 78, 64, 62, 89]))
    add(Spacer(1, 7))
    add(para('The tuned forest’s F1 (0.703) only narrowly exceeds the '
             'always-majority-class baseline (0.699). Its ROC-AUC '
             '(0.473) is approximately chance, so these predictors '
             'have little discriminative value in the tested period. '
             'We do not claim meaningful forecasting success.'))
    page()

    # Page 8 - Planning/testing/reproducibility and references.
    section(8, 'Work Plan Evaluation')
    add(para('Our proposal allocated 44 person-hours (22 per author): '
             '4 setup, 8 cleaning/parsing, 8 historical figures, '
             '10 modeling, 8 robustness and 6 final deliverables. '
             'The local-source parsing and validation tasks required '
             'more care than anticipated: 2019/20 had 119 games, '
             '2020/21–2022/23 included non-English <i>Gruppe</i> '
             'headings, and penalty scores required special handling. '
             'Actual personal time logs were not provided; we cannot '
             'claim the original hours were met or exceeded. '
             'Estimated remaining effort: 2–3 hours each for '
             'joint source checks, interpretive review, repository '
             'publication and final presentation planning.'))
    section(9, 'Testing and Trustworthiness')
    add(para('All 24 Python unittest checks pass. Tests cover '
             'seven-number summaries, missing/negative values, '
             'duplicate fixtures, expected header counts, localized '
             'stage labels, extra-time versus shootout scores, '
             'same-date no-leakage features, season resets, both '
             'bootstraps, and estimator fit/temporal partitions. '
             'We checked 12 × 125 + 119 = 1,619 games, '
             '13 × 96 = 1,248 group games, 4,809 match goals '
             'and 916 positive target labels. All raw season files '
             'and the reproducible parser accompany this report. '
             'Run <i>python eda.py</i>, '
             '<i>python model.py</i> and '
             '<i>python -m unittest discover -p "test_*.py"</i>. '
             'Final flake8 verification is requested on a local '
             'environment where flake8 is installed.'))
    section(10, 'Collaboration and Consulted Resources')
    add(para('Authors: Rodrigo Li and Larry Zhang. As planned, '
             'Rodrigo’s focus is RQ1–RQ3 while Larry’s is RQ4; '
             'both should review interpretations and tests. '
             'Sources: UEFA historical ranking pages, OpenFootball’s '
             'CC0 repository, Python scientific-library documentation. '
             'ChatGPT was used to help draft and check code and text; '
             'both authors must verify results and follow the '
             'course’s AI disclosure requirements. No additional '
             'human collaborators are claimed.'))
    section(11, 'Limitations and Next Steps')
    add(para('UEFA ranking counts can have a different competition '
             'scope from other historical articles; two player '
             'appearance denominators remain unverified. Selected '
             'player-seasons are not a full historical census. '
             'Pre-match scoring features do not fully describe '
             'opponents, lineups or team quality; repeated teams '
             'limit simple statistical independence. Next: review '
             'the raw records and league-stage format differences, '
             'check the final models jointly, and publish this '
             'version to the public team GitHub repository.'))
    add(para('<b>Data sources:</b> <link color="#107b81" '
             'href="https://github.com/openfootball/champions-league">'
             'OpenFootball Champions League (CC0)</link> | '
             '<link color="#107b81" '
             'href="https://www.uefa.com/uefachampionsleague/history/'
             'rankings/players/goals_scored/">UEFA goals</link> | '
             '<link color="#107b81" '
             'href="https://www.uefa.com/uefachampionsleague/history/'
             'rankings/players/assists/">UEFA assists</link> | '
             '<link color="#107b81" '
             'href="https://www.uefa.com/uefachampionsleague/history/'
             'rankings/players/matches_played/">UEFA appearances</link> | '
             '<link color="#107b81" '
             'href="https://www.uefa.com/uefachampionsleague/history/'
             'rankings/players/season_goals_scored/'
             '">UEFA season goals</link>'))

    doc.build(work)
    print('Generated:', ROOT / 'eda.pdf')


def main() -> None:
    """Generate the PDF report from saved outputs."""
    make_report()


if __name__ == '__main__':
    main()
