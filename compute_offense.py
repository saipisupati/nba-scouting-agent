from __future__ import annotations

import pandas as pd

# ── play type CSV paths ───────────────────────────────────────────────────────

_PLAYTYPE_CSV = {
    "Isolation":    "playtype_offense_isolation_2025_26.csv",
    "PRBallHandler":"playtype_offense_prballhandler_2025_26.csv",
    "PRRollman":    "playtype_offense_prrollman_2025_26.csv",
    "Postup":       "playtype_offense_postup_2025_26.csv",
    "Spotup":       "playtype_offense_spotup_2025_26.csv",
    "Handoff":      "playtype_offense_handoff_2025_26.csv",
    "Cut":          "playtype_offense_cut_2025_26.csv",
    "OffScreen":    "playtype_offense_offscreen_2025_26.csv",
    "Transition":   "playtype_offense_transition_2025_26.csv",
}

# Thresholds derived from each category's actual POSS/g distribution.
# Target: p25 for high-volume categories (Spotup, Transition, PRBallHandler);
# p35-ish (midpoint of p25/p50) for mid-volume; p25 rounded for sparse ones.
# Rationale per category:
#   Isolation:    p25=0.40 → 0.5  (round up slightly; 0.4 is genuine micro-sample)
#   PRBallHandler:p25=0.75 → 0.8  (high-volume PG action, p25 is reasonable floor)
#   PRRollman:    p25=0.30 → 0.5  (most bigs are 0.1–0.5; need some intent)
#   Postup:       p25=0.40 → 0.5  (only 149 players; raise slightly vs isolation)
#   Spotup:       p25=1.90 → 1.0  (p25 is already high; lower to keep more guards)
#   Handoff:      p25=0.40 → 0.5  (sparse action, keep threshold modest)
#   Cut:          p25=0.40 → 0.5  (finishing action; 0.4 is genuine usage)
#   OffScreen:    p25=0.30 → 0.5  (very skewed; 0.3 is noise)
#   Transition:   p25=1.10 → 1.0  (round down slightly from p25 to widen pool)
#
# NOTE on OffScreen: at p75 (1.05 poss/g) Jokić ranks #1; below 0.5 he falls
# to #11. His 1.2 poss/g is real usage, not micro-sample, but results are
# threshold-sensitive — the docstring calls this out explicitly.
_PLAYTYPE_DEFAULT_MIN_POSS: dict[str, float] = {
    "Isolation":    0.5,
    "PRBallHandler":0.8,
    "PRRollman":    0.5,
    "Postup":       0.5,
    "Spotup":       1.0,
    "Handoff":      0.5,
    "Cut":          0.5,
    "OffScreen":    0.5,
    "Transition":   1.0,
}

_RETURN_COLS = ["PLAYER_NAME", "TEAM_ABBREVIATION", "POSS", "PPP", "FG_PCT", "PERCENTILE"]

_VALID_PLAY_TYPES = set(_PLAYTYPE_CSV)

# Minimum total possessions (POSS/g × GP) to require meaningful sample size.
# Filters out players like Josh Minott (16 GP, 14 total poss) who top per-game
# efficiency lists purely on tiny samples. 30 is the same threshold used for
# shot suppression and playtype_defense in this project.
_MIN_TOTAL_POSS = 30

# Cut is sorted by POSS (volume) not PPP because finishing efficiency on cuts
# is uniformly high across all qualifiers (~1.3–1.7 PPP, 65–85% FG) — the
# range is too compressed to meaningfully rank players by efficiency. The real
# signal is who gets the most cuts, not who finishes them slightly better.
_SORT_BY_VOLUME = {"Cut"}

# OffScreen: total-poss threshold below which the answer notes the player's
# cut usage is moderate-volume and the ranking is somewhat threshold-sensitive.
# Derived from the actual distribution: Murray (57 total poss) and Jokić
# (76 total poss) are the top qualifiers at the default floor — both are real
# samples but modest for a full-season action. 100 total possessions is a
# clean round number that sits between "micro-sample" and "primary action".
_OFFSCREEN_MODERATE_POSS_THRESHOLD = 100


def playtype_offense(play_type: str, min_poss: float | None = None) -> pd.DataFrame:
    """Return offensive efficiency rankings for a Synergy play type.

    Ranks DESCENDING by PPP — higher PPP means more efficient scorer.
    This is the OPPOSITE direction from playtype_defense and shot_suppression,
    which both rank ascending (lower = better defender).

    Exception — Cut: sorted DESCENDING by POSS (volume) instead of PPP.
    Cutting efficiency is uniformly high (~1.3–1.7 PPP) across all qualifiers;
    PPP doesn't meaningfully differentiate cutters. Volume of cuts is the real
    signal for who uses this action as a meaningful part of their game.

    All 9 play types have real data on the offensive side (including Cut and
    Transition, which returned empty on the defensive pull).

    OffScreen note: top qualifiers at the default threshold have 57–76 total
    possessions — real usage, but moderate volume for a full season. Rankings
    are somewhat threshold-sensitive (Jokić moves from #11 → #2 → absent as
    the floor tightens from 0.3 → 0.5 → 1.5 poss/g). format_playtype_offense_answer()
    adds a caveat when a player's total possessions fall below
    _OFFSCREEN_MODERATE_POSS_THRESHOLD (100).

    Parameters
    ----------
    play_type : one of Isolation / PRBallHandler / PRRollman / Postup / Spotup /
                Handoff / Cut / OffScreen / Transition
    min_poss  : minimum possessions per game; defaults to per-category threshold
                derived from the actual POSS distribution

    Returns
    -------
    DataFrame sorted descending by POSS (Cut) or PPP (all others), columns:
        PLAYER_NAME, TEAM_ABBREVIATION, POSS, PPP, FG_PCT, PERCENTILE
    """
    if play_type not in _VALID_PLAY_TYPES:
        raise ValueError(
            f"play_type must be one of {sorted(_VALID_PLAY_TYPES)}, got {play_type!r}"
        )

    floor = min_poss if min_poss is not None else _PLAYTYPE_DEFAULT_MIN_POSS[play_type]
    df = pd.read_csv(_PLAYTYPE_CSV[play_type])
    filtered = df[
        (df["POSS"] >= floor) &
        (df["POSS"] * df["GP"] >= _MIN_TOTAL_POSS)
    ].copy()

    sort_col = "POSS" if play_type in _SORT_BY_VOLUME else "PPP"
    return (
        filtered[_RETURN_COLS]
        .sort_values(sort_col, ascending=False)
        .reset_index(drop=True)
    )


# ── Answer formatting ─────────────────────────────────────────────────────────

_PLAYTYPE_OFFENSE_LABEL = {
    "Isolation":    "isolation offense",
    "PRBallHandler":"pick-and-roll ball-handler offense",
    "PRRollman":    "pick-and-roll roll-man offense",
    "Postup":       "post-up offense",
    "Spotup":       "spot-up / catch-and-shoot offense",
    "Handoff":      "dribble handoff offense",
    "Cut":          "cutting",
    "OffScreen":    "off-screen offense",
    "Transition":   "transition offense",
}

_OFFSCREEN_CAVEAT = (
    "NOTE: Off-screen is a moderate-volume action for most players in this ranking — "
    "the top qualifiers have roughly 57–100 total possessions on the season, which is "
    "real usage but not a primary action. Rankings are somewhat threshold-sensitive: "
    "small changes to the minimum-possession floor shift who qualifies and in what order. "
    "Pair with POSS/g volume when comparing players here."
)

_CUT_VOLUME_NOTE = (
    "NOTE: Sorted by cuts per game (volume), not PPP. "
    "Finishing efficiency on cuts is uniformly high across all qualifiers "
    "(roughly 1.3–1.7 PPP, 65–85% FG%) — the range is too compressed to meaningfully "
    "rank players by efficiency. Who gets the most cuts is the real signal."
)


def format_playtype_offense_answer(
    row: pd.Series,
    play_type: str,
    season_label: str,
    total_poss: float | None = None,
) -> str:
    """Format a one-sentence answer for the top offensive play-type result.

    Parameters
    ----------
    row          : top row from playtype_offense(play_type)
    play_type    : play type string
    season_label : e.g. '2025-26'
    total_poss   : POSS/g × GP for the top player; used for OffScreen caveat.
                   Pass None to skip the threshold check.
    """
    label = _PLAYTYPE_OFFENSE_LABEL.get(play_type, play_type)
    player = f"{row['PLAYER_NAME']} ({row['TEAM_ABBREVIATION']})"

    if play_type == "Cut":
        base = (
            f"{player} leads in {label} with {row['POSS']:.1f} cuts per game "
            f"({row['PPP']} PPP, {row['FG_PCT']:.1%} FG%) [{season_label}]. "
            f"{_CUT_VOLUME_NOTE}"
        )
        return base

    base = (
        f"{player} leads in {label} with {row['PPP']} PPP "
        f"({row['FG_PCT']:.1%} FG%, {row['POSS']:.1f} poss/g, "
        f"{row['PERCENTILE']:.0%} percentile) [{season_label}]."
    )

    if play_type == "OffScreen":
        moderate = (
            total_poss is not None and total_poss < _OFFSCREEN_MODERATE_POSS_THRESHOLD
        )
        if moderate:
            return f"{base} {_OFFSCREEN_CAVEAT}"

    return base
