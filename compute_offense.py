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


def playtype_offense(play_type: str, min_poss: float | None = None) -> pd.DataFrame:
    """Return offensive efficiency rankings for a Synergy play type.

    Ranks DESCENDING by PPP — higher PPP means more efficient scorer.
    This is the OPPOSITE direction from playtype_defense and shot_suppression,
    which both rank ascending (lower = better defender).

    All 9 play types have real data on the offensive side (including Cut and
    Transition, which returned empty on the defensive pull).

    OffScreen note: Jokić ranks #1 at the default threshold (0.5 poss/g), but
    his ranking is threshold-sensitive — he has 1.2 poss/g (real usage) but
    sits just above the p25 floor. At thresholds above 1.5 poss/g he drops out
    of the pool entirely. Pair with POSS volume when interpreting OffScreen results.

    Parameters
    ----------
    play_type : one of Isolation / PRBallHandler / PRRollman / Postup / Spotup /
                Handoff / Cut / OffScreen / Transition
    min_poss  : minimum possessions per game; defaults to per-category threshold
                derived from the actual POSS distribution

    Returns
    -------
    DataFrame sorted descending by PPP with columns:
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

    return (
        filtered[_RETURN_COLS]
        .sort_values("PPP", ascending=False)
        .reset_index(drop=True)
    )
