"""
College draft-class layer: lookup, leaderboards, and usage-vs-efficiency
framing over the 2026 draft class, from draft_class_2026.csv
(pull_2026_draft_class.py).

Unlike the NBA per-game-rate categories in compute_defense.py/compute_offense.py,
this data has no thin-sample risk to guard against: it's one row per player,
one full college season each, already resolved to a fixed 60-player list. No
percentile-derived qualification floor is needed or applied here.

status == "international" rows (6 of 60) have no NCAA data source and are
excluded from leaderboards/efficiency views, same as any player who doesn't
clear a qualification floor elsewhere in this project — but unlike those
cases, every leaderboard/efficiency answer here surfaces that exclusion
explicitly, per this project's own scope-honesty convention (see README
Design Principle 4).
"""

from __future__ import annotations

import pandas as pd

_COLLEGE_CSV = "draft_class_2026.csv"

_LEADERBOARD_LABEL = {
    "PTS": "scoring (PTS)",
    "TRB": "rebounding (TRB)",
    "AST": "assists (AST)",
    "USG%": "usage rate (USG%)",
    "TS%": "true shooting (TS%)",
    "BPM": "box plus-minus (BPM)",
    "PER": "PER",
    "AST%": "assist rate (AST%)",
}

_N_INTERNATIONAL = 6
_INTERNATIONAL_NOTE = (
    f"NOTE: {_N_INTERNATIONAL} international picks in this draft class have no NCAA data "
    "source and are excluded from this ranking."
)


def _load() -> pd.DataFrame:
    return pd.read_csv(_COLLEGE_CSV)


def college_player_lookup(name: str) -> pd.Series | None:
    """Case-insensitive substring match on player name (handles 'Jr.'/'Sr.'/
    'II'/'III' suffixes naturally since they're part of the same string).
    Returns the player's row (whatever status — 'ok' or 'international'),
    or None if no name in the draft class matches. Raises ValueError on an
    ambiguous substring match against more than one player — silently
    returning the first match risks the exact silent-wrong-answer failure
    mode this project's README (Design Principle 5) calls out as more
    dangerous than a crash. Not a concern with the current 60-player list
    (no in-list collisions as of this dataset), but the check costs nothing
    and the list is not guaranteed to stay collision-free if it ever grows
    (e.g. next year's draft class appended, or a similar surname added).
    """
    df = _load()
    name_lower = name.lower().strip()
    matches = df[df["name"].str.lower().str.contains(name_lower, regex=False)]
    if matches.empty:
        return None
    if len(matches) > 1:
        candidates = ", ".join(matches["name"].tolist())
        raise ValueError(
            f"'{name}' matches multiple draft-class players ({candidates}) — "
            f"ambiguous lookup, provide a more specific name."
        )
    return matches.iloc[0]


def college_leaderboard(metric: str, ascending: bool = False) -> pd.DataFrame:
    """Rank the 2026 draft class by a single college stat column.

    Only status == 'ok' rows are ranked (54 of 60) — international picks
    have no NCAA data and are never included in the returned DataFrame.
    """
    if metric not in _LEADERBOARD_LABEL:
        raise ValueError(f"Unsupported college leaderboard metric: {metric!r}")

    df = _load()
    qualified = df[df["status"] == "ok"].copy()
    return (
        qualified.sort_values(metric, ascending=ascending)
        .reset_index(drop=True)
    )


_YOUTH_CLASSES = {"FR", "SO"}

# NOTE ON WHAT THIS IS NOT: "youth_adjusted" names the flag, not a validated
# adjustment. There is no historical multi-class dataset here (this project
# has exactly one draft class, 2026) to establish what a class-year-adjusted
# baseline should even look like -- no age curve, no cohort comparison
# across years, nothing to regress against. The flag is a plain descriptive
# fact within this single class only: "this FR/SO ranks in the top half of
# THIS 54-player pool on THIS metric," not "this player is outperforming
# what a freshman/sophomore is expected to produce" in any calibrated sense.
# Framed explicitly as such in the docstring and the returned column name
# (OUTPERFORMING_UPPERCLASSMEN, not e.g. AGE_ADJUSTED_SCORE) so a caller
# can't mistake a within-class-only observation for a validated formula.
def youth_adjusted_leaderboard(metric: str) -> pd.DataFrame:
    """Rank the 2026 draft class by a single stat, same as college_leaderboard,
    annotated with class_year and a top-half-of-pool flag for underclassmen.

    NOT an age-adjusted or class-year-normalized formula. This dataset is a
    single 60-pick draft class with no historical baseline to validate what
    "adjusted for youth" should mean quantitatively. All this does is state
    a plain fact within this one pool: whether a FR/SO ranks in the top half
    of the 54 qualified players on the given metric. That's a real,
    checkable observation, not a claim about age-normalized performance.

    Parameters
    ----------
    metric : one of the same metrics college_leaderboard supports

    Returns
    -------
    DataFrame sorted descending by metric, same columns as college_leaderboard
    plus:
        OUTPERFORMING_UPPERCLASSMEN : bool, True only for FR/SO rows in the
                                       top half (by rank) of the qualified pool
    """
    ranked = college_leaderboard(metric)
    halfway = len(ranked) // 2
    ranked["OUTPERFORMING_UPPERCLASSMEN"] = (
        ranked["class_year"].isin(_YOUTH_CLASSES) & (ranked.index < halfway)
    )
    return ranked


def _school_display(school) -> str:
    """school is a real NaN (float) for Jayden Quaintance (no school listed
    in the source draft list — see pull_2026_draft_class.py's own docstring
    on this), not an empty string. An f-string would print the literal text
    "nan" without this guard."""
    return school if isinstance(school, str) else "no school listed"


def format_youth_adjusted_leaderboard_answer(row: pd.Series, metric: str) -> str:
    """Format a one-sentence answer for the top youth_adjusted_leaderboard
    result. Always states the class-year-observation framing explicitly
    (see youth_adjusted_leaderboard's own docstring) rather than letting the
    flag imply a validated adjustment it isn't."""
    label = _LEADERBOARD_LABEL.get(metric, metric)
    school = _school_display(row["school"])
    base = (
        f"{row['name']} ({school}, {row['class_year']}) leads the 2026 draft class in "
        f"{label} with {row[metric]}. {_INTERNATIONAL_NOTE}"
    )
    if row["OUTPERFORMING_UPPERCLASSMEN"]:
        base += (
            f" NOTE: {row['class_year']} ranks in the top half of this specific "
            f"draft class on {label} — this is a within-class observation, not a "
            f"validated age-adjusted formula (no historical baseline exists in "
            f"this dataset to support that stronger claim)."
        )
    return base


# Below this USG%, a player isn't really a "high-usage prospect" in the
# first place — ranking them on TS% here would answer "who's the most
# efficient scorer" (already covered by a plain TS%/PER leaderboard), not
# the actual question this view exists for: who's both a primary offensive
# option AND efficient with it. Set at the qualified pool's own median
# USG% (54 players, median 24.5%) rather than an arbitrary round number,
# same p-derived-from-real-distribution convention used throughout this
# project (see compute_offense.py's _MIN_DRIVES_PER_GAME for the fullest
# writeup of why a guessed threshold isn't good enough here).
_HIGH_USAGE_FLOOR = 24.5


def college_efficiency_volume() -> pd.DataFrame:
    """USG% vs. TS% for the 2026 draft class — the volume-vs-outcome framing
    this project applies everywhere else (USG% on offense plays the role
    contest volume plays on defense; TS% is whether it actually worked).

    Ranks TS% only among players who clear _HIGH_USAGE_FLOOR — without that
    filter this just reproduces a plain TS% leaderboard (already answered
    by a different question) and can surface a below-median-usage player as
    the top "high-usage, efficient" result, which misstates what the number
    actually shows.

    Only status == 'ok' rows (54 of 60); sorted by TS% descending.
    """
    df = _load()
    qualified = df[df["status"] == "ok"].copy()
    high_usage = qualified[qualified["USG%"] >= _HIGH_USAGE_FLOOR].copy()
    return (
        high_usage.sort_values("TS%", ascending=False)
        .reset_index(drop=True)
    )


def format_college_lookup_answer(row: pd.Series) -> str:
    """Format a one-sentence answer for a single college player lookup."""
    if row["status"] == "international":
        return (
            f"{row['name']} (pick #{int(row['pick_number'])}) played internationally, "
            f"not in the NCAA — no college stats available for this player in this dataset."
        )

    return (
        f"{row['name']} ({row['school']}, {row['class_year']}, pick #{int(row['pick_number'])}) "
        f"averaged {row['PTS']} PTS, {row['TRB']} TRB, {row['AST']} AST on "
        f"{row['FG%']}% FG / {row['3P%']}% 3P ({row['eFG%']}% eFG), "
        f"with {row['USG%']}% usage and {row['TS%']}% true shooting "
        f"(BPM {row['BPM']}, PER {row['PER']}) in his final college season."
    )


def format_college_leaderboard_answer(row: pd.Series, metric: str) -> str:
    """Format a one-sentence answer for the top college leaderboard result."""
    label = _LEADERBOARD_LABEL.get(metric, metric)
    return (
        f"{row['name']} ({row['school']}) leads the 2026 draft class in {label} "
        f"with {row[metric]}. {_INTERNATIONAL_NOTE}"
    )


# Real gap distribution across the 28-player qualified pool (2026 draft
# class, checked directly rather than guessed): 27 consecutive-rank TS%
# margins range 0.0-1.7 points, median 0.3. A margin at or above 1.0 sits
# above all but 2 of those 27 gaps -- i.e. genuinely unusual spacing, not
# the pool's typical tightness. Below 1.0 (the actual #1->#2 gap in this
# data is 0.3) is well within normal noise for this stat, which is why the
# caveat exists: presenting a 0.3-point TS% edge with the same confidence
# as a real gap would overstate how decisive "leads in true shooting"
# actually is.
_TS_MARGIN_THIN_THRESHOLD = 1.0

# A player "leads decisively" on BPM/PER only if the edge is large enough
# to matter, not any nonzero difference -- same intent as _DRIVE_DIVERGENCE_GAP
# in compute_offense.py. Checked directly against this pool's real
# consecutive-rank gap distributions (28 players, own BPM/PER sort order,
# 27 gaps each), the same way _TS_MARGIN_THIN_THRESHOLD was validated above:
#   BPM gaps: min=0.0  p25=0.10  median=0.30  p75=0.50  p90=1.00  max=4.20
#   PER gaps: min=0.0  p25=0.15  median=0.30  p75=0.75  p90=1.32  max=2.60
# An initial guess of 2.0/3.0 was checked against this data and rejected:
# 2.0 cleared only 1 of 27 BPM gaps (96th percentile, too strict to ever
# realistically fire), and 3.0 exceeded the PER distribution's own max gap
# (2.6) entirely -- unreachable regardless of how decisive a real gap was.
# Set at ~p90 instead (BPM 1.0, 4/27 gaps clear it; PER 1.3, 3/27 gaps
# clear it) -- selective without being impossible, matching the rarity
# TS%'s 1.0 threshold achieves (2/27) on its own distribution.
_BPM_DECISIVE_GAP = 1.0
_PER_DECISIVE_GAP = 1.3


def format_college_efficiency_volume_answer(ranked: pd.DataFrame) -> str:
    """Format a one-sentence answer for the top usage-vs-efficiency result.

    Parameters
    ----------
    ranked : full result of college_efficiency_volume() (not just the top
             row) -- the #2 player's BPM/PER are needed to check whether a
             thin TS% margin is offset by a clearer lead on those metrics.
    """
    row = ranked.iloc[0]
    base = (
        f"{row['name']} ({row['school']}) leads the 2026 draft class in true shooting "
        f"among high-usage prospects (≥{_HIGH_USAGE_FLOOR}% USG, the qualified pool's own "
        f"median), at {row['TS%']}% TS on {row['USG%']}% usage. {_INTERNATIONAL_NOTE}"
    )

    if len(ranked) < 2:
        return base

    runner_up = ranked.iloc[1]
    ts_margin = row["TS%"] - runner_up["TS%"]
    if ts_margin >= _TS_MARGIN_THIN_THRESHOLD:
        return base

    bpm_gap = runner_up["BPM"] - row["BPM"]
    per_gap = runner_up["PER"] - row["PER"]
    if bpm_gap >= _BPM_DECISIVE_GAP and per_gap >= _PER_DECISIVE_GAP:
        base += (
            f" NOTE: TS% margin over the next-closest qualifier is thin ({ts_margin:.1f} "
            f"points) — {runner_up['name']} leads decisively on BPM ({runner_up['BPM']} vs. "
            f"{row['BPM']}) and PER ({runner_up['PER']} vs. {row['PER']}) instead, worth "
            f"weighing alongside pure shooting efficiency."
        )

    return base
