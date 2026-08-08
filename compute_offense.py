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


# ── drive efficiency (LeagueDashPtStats, pt_measure_type='Drives') ────────────

_DRIVES_CSV = "drives_2025_26.csv"

# DRIVES in this CSV is already a per-game rate (confirmed against known
# volume drivers: SGA shows DRIVES=18.8 across GP=68, consistent with his
# real ~19 drives/game reputation — a season total at that magnitude would
# be implausible).
#
# Floors were originally set at p25 (1.0/game, 30 total), matching this
# project's default convention for high-volume categories. That floor was
# too loose for this metric specifically: at n=582, PTS_PER_DRIVE among a
# 30-total-drive qualifier pool was dominated by low-volume bigs (Jalen
# Duren at 3.6 drives/g / 252 total, Jarrett Allen at 1.6/g / 89.6 total)
# who finish point-blank rolls/dives rather than guards driving through
# contact. Raising the total-drives floor alone (tested at 30/46/60) did
# NOT dislodge them — corr(DRIVES per-game, PTS_PER_DRIVE) is only 0.163
# among a TOTAL_DRIVES>=60 pool, and bucketed means by drives-per-game are
# flat-to-rising with volume, not falling. That rules out small-sample
# noise as the cause: it's a real category-conflation, the same shape as
# the Cut volume-vs-PPP decision above (a mechanically easy, high-FG%
# finishing action mixed in with harder, contested drives).
#
# Floors raised to p40 (this project's convention for sparse/skewed
# categories, see compute_defense.py's Postup/Handoff/OffScreen precedent)
# on both dimensions of the actual distribution (n=582, 2025-26):
#   DRIVES per-game   p25=1.0  p35=1.4  p40=1.7  p50=2.4  p75=5.3
#   TOTAL_DRIVES       p25=20.3 p35=46.1 p40=60.3 p50=96.6 p75=292.6
# 309 of 582 players (53%) still qualify at this floor — this is not a
# thin-sample cutoff, it excludes low-intent/marginal drivers while
# keeping the metric an honest "efficient among real drivers" ranking.
_MIN_DRIVES_PER_GAME = 1.7

# Total-drives floor (DRIVES × GP), same role as _MIN_TOTAL_POSS above: a
# per-game rate alone lets single-game/small-sample players qualify (e.g.
# a player with GP=1, DRIVES=1.7 clears the per-game floor on two drives
# taken all season). See comment above _MIN_DRIVES_PER_GAME for how 60 was
# derived (p40 of TOTAL_DRIVES).
_MIN_TOTAL_DRIVES = 60


def drive_efficiency(min_drives_per_game: float | None = None) -> pd.DataFrame:
    """Return drive volume and efficiency for qualifying players, from
    LeagueDashPtStats (pt_measure_type='Drives').

    Ranks DESCENDING by PTS_PER_DRIVE (points scored per drive — DRIVE_PTS
    divided by DRIVES — not DRIVE_FG_PCT alone). Points-per-drive captures
    value from drawn free throws as well as makes, the same "points per
    action" convention as PPP elsewhere in this project; DRIVE_FG_PCT and
    PTS_PER_DRIVE correlate at only ~0.59 across qualifiers, so they are
    not interchangeable — a player drawing fouls on drives can be a more
    efficient scorer than his raw FG% on drives alone suggests.

    Also surfaces DRIVE_PASSES_PCT and DRIVE_AST_PCT so volume and
    efficiency can be read alongside playmaking: a high-volume driver who
    also passes/creates at a high rate is a different profile than one who
    drives mostly to shoot himself, and neither PTS_PER_DRIVE nor DRIVES
    alone captures that distinction.

    Parameters
    ----------
    min_drives_per_game : minimum DRIVES per game to qualify; defaults to
                           _MIN_DRIVES_PER_GAME (the p40 floor derived from
                           the actual distribution — see the comment above
                           _MIN_DRIVES_PER_GAME for why p25 was too loose
                           for this specific metric). A total-drives floor
                           (_MIN_TOTAL_DRIVES) is always applied on top of
                           this, regardless of what's passed in, to exclude
                           small-sample players who clear the per-game rate
                           on a handful of games.

    Returns
    -------
    DataFrame sorted descending by PTS_PER_DRIVE, columns:
        PLAYER_NAME, TEAM_ABBREVIATION, GP, DRIVES, PTS_PER_DRIVE,
        DRIVE_FG_PCT, DRIVE_PASSES_PCT, DRIVE_AST_PCT, DRIVE_TOV_PCT
    """
    floor = min_drives_per_game if min_drives_per_game is not None else _MIN_DRIVES_PER_GAME
    df = pd.read_csv(_DRIVES_CSV)

    df = df.copy()
    df["TOTAL_DRIVES"] = df["DRIVES"] * df["GP"]
    df["PTS_PER_DRIVE"] = df["DRIVE_PTS"] / df["DRIVES"]

    filtered = df[
        (df["DRIVES"] >= floor) &
        (df["TOTAL_DRIVES"] >= _MIN_TOTAL_DRIVES)
    ].copy()

    filtered["PTS_PER_DRIVE"] = filtered["PTS_PER_DRIVE"].round(3)
    filtered["DRIVE_FG_PCT"] = filtered["DRIVE_FG_PCT"].round(3)
    filtered["DRIVE_PASSES_PCT"] = filtered["DRIVE_PASSES_PCT"].round(3)
    filtered["DRIVE_AST_PCT"] = filtered["DRIVE_AST_PCT"].round(3)
    filtered["DRIVE_TOV_PCT"] = filtered["DRIVE_TOV_PCT"].round(3)

    keep = ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "DRIVES", "PTS_PER_DRIVE",
            "DRIVE_FG_PCT", "DRIVE_PASSES_PCT", "DRIVE_AST_PCT", "DRIVE_TOV_PCT"]
    return (
        filtered[keep]
        .sort_values("PTS_PER_DRIVE", ascending=False)
        .reset_index(drop=True)
    )


# PTS_PER_DRIVE and DRIVE_FG_PCT correlate at only ~0.59 (see drive_efficiency
# docstring) — a player can lead in one without leading in the other, most
# often because of drawn fouls/free throws inflating points per drive
# relative to raw shooting percentage on the drive itself.
_DRIVE_DIVERGENCE_GAP = 0.10

# See the archetype-conflation analysis above _MIN_DRIVES_PER_GAME: even
# after raising the qualification floor to p40 on both dimensions,
# PTS_PER_DRIVE stays dominated by low-volume bigs finishing short
# rolls/dives rather than by high-volume guards driving into a set defense
# (corr(DRIVES, PTS_PER_DRIVE) = 0.163 among well-sampled qualifiers — a
# real category-conflation, not a small-sample artifact). Always attached,
# the same way _CUT_VOLUME_NOTE is always attached to the Cut answer,
# since it applies to every drive_efficiency ranking, not just edge cases.
_DRIVE_ARCHETYPE_CAVEAT = (
    "NOTE: PTS_PER_DRIVE mixes fundamentally different shot types — short-roll/rim "
    "finishes for bigs who drive rarely but efficiently, vs. contested, defense-set "
    "drives for high-volume guards — so a raw efficiency ranking here shouldn't be "
    "read as \"best offensive driver\" without that context. For evaluating "
    "high-volume creators specifically, pair this with DRIVES (volume) and "
    "DRIVE_AST_PCT/DRIVE_PASSES_PCT (playmaking) — PTS_PER_DRIVE alone favors a "
    "different archetype."
)


def format_drive_efficiency_answer(row: pd.Series, season_label: str) -> str:
    """Format a one-sentence answer for the top drive-efficiency result."""
    player = f"{row['PLAYER_NAME']} ({row['TEAM_ABBREVIATION']})"
    base = (
        f"{player} leads in points per drive with {row['PTS_PER_DRIVE']:.2f} "
        f"({row['DRIVE_FG_PCT']:.1%} FG% on drives, {row['DRIVES']:.1f} drives/g, "
        f"{row['DRIVE_AST_PCT']:.1%} assist rate) [{season_label}]. "
        f"{_DRIVE_ARCHETYPE_CAVEAT}"
    )

    # rough FG%-equivalent of PTS_PER_DRIVE (2pt scale) to flag when scoring
    # efficiency is meaningfully propped up by fouls drawn rather than raw shooting
    if row["PTS_PER_DRIVE"] / 2 - row["DRIVE_FG_PCT"] > _DRIVE_DIVERGENCE_GAP:
        base += (
            " NOTE: Points-per-drive and raw drive FG% diverge meaningfully here — "
            "this player's scoring value on drives is boosted substantially by drawn "
            "fouls/free throws, not just made field goals."
        )

    return base


# ── signature play type ────────────────────────────────────────────────────

# Cut is deliberately excluded from signature detection. It's the one
# category playtype_offense() itself sorts by volume rather than PPP,
# because per that function's own docstring, cut finishing efficiency is
# uniformly high (~1.3-1.7 PPP) across all qualifiers -- the PERCENTILE
# column exists and is populated for Cut, but this project's own established
# reasoning is that it isn't a meaningful efficiency signal for this
# category. Including it here would treat "he cuts efficiently" as
# comparable to "he's an elite isolation scorer," which contradicts that
# reasoning rather than extending it.
_SIGNATURE_EXCLUDED_TYPES = {"Cut"}

# Checked directly against the real cross-category percentile-gap
# distribution (377 players who qualify in >=2 offensive play-type
# categories, 2025-26): top1-vs-top2 gap has min=0.001, p10=0.020,
# p25=0.051, median=0.139, p75=0.251, max=0.869 (PERCENTILE is stored as a
# 0-1 fraction, not 0-100). A margin of 0.04 (4 percentile points) flags
# 71 of 377 players (18.8%) as having a genuinely close multi-category
# profile -- a real minority, not near-zero or near-everyone, which is
# what "genuine tie" should look like against this data.
_SIGNATURE_TIE_MARGIN = 0.04

# A player's OWN best qualifying-category percentile also needs to clear a
# floor before it's called a "signature" at all -- otherwise a role player
# whose best category is, say, the 45th percentile gets a "signature play
# type" that isn't actually a strength, just whichever category happened to
# rank highest among several mediocre ones. Checked against the real
# distribution of each player's own best percentile (356 multi-category
# players, 2025-26): min=0.075, p10=0.434, p25=0.609, median=0.783,
# p75=0.890 -- the median best-category percentile is already ~78th, so
# p25 (~0.60) is a genuine below-average bar, not an arbitrary round number.
_SIGNATURE_MIN_PERCENTILE = 0.60


# The full-roster reference used to distinguish "this name doesn't match any
# NBA player" from "this player exists but has no qualifying play-type data" --
# hustle_stats includes every rostered player regardless of play-type
# qualification (581 rows, 2025-26), unlike the union of playtype_offense's
# own per-category CSVs, which only contains players who cleared some
# category's possession floor somewhere (438 rows) -- a real but thin-data
# player like Thanasis Antetokounmpo (5.2 min/g) exists in the former but not
# the latter, which is exactly the distinction this needs.
_ROSTER_CSV = "hustle_stats_2025_26.csv"


def resolve_player_name(name: str) -> str | None:
    """Case-insensitive substring match against the full NBA roster, same
    matching convention as compute_college.college_player_lookup (reused
    deliberately, not reimplemented, so the two lookups can't silently
    drift into different matching behavior). Returns the canonical
    PLAYER_NAME on a single match, or None if no player matches.

    Raises ValueError on an ambiguous match against more than one player,
    same rationale as college_player_lookup: silently picking the first
    match risks a silent-wrong-answer, which this project's README calls
    out as more dangerous than a crash.
    """
    roster = pd.read_csv(_ROSTER_CSV)
    name_lower = name.lower().strip()
    matches = roster[roster["PLAYER_NAME"].str.lower().str.contains(name_lower, regex=False)]
    if matches.empty:
        return None
    resolved = matches["PLAYER_NAME"].unique()
    if len(resolved) > 1:
        candidates = ", ".join(resolved)
        raise ValueError(
            f"'{name}' matches multiple NBA players ({candidates}) — "
            f"ambiguous lookup, provide a more specific name."
        )
    return resolved[0]


def signature_play_type(player_name: str) -> dict:
    """Identify a player's standout offensive play type: the qualifying
    category (min-possession floor already enforced by playtype_offense())
    where they rank highest by PERCENTILE, not just their highest raw PPP.

    The player name is resolved against the full NBA roster first (see
    resolve_player_name) so that "this player doesn't exist / didn't match"
    (player_found=False) is distinguishable from "this real player has no
    qualifying play-type category" (player_found=True, categories=[]) --
    previously both cases collapsed into the same empty-categories result
    and the same "doesn't qualify" answer text, which was actively wrong
    for a real player looked up under a nickname or minor misspelling
    (e.g. "Steph Curry" vs. the data's "Stephen Curry").

    Returns a dict:
        {
            "player_name": str,  -- the RESOLVED canonical name when
                                  player_found is True (not the raw input
                                  the caller passed), so a caller can't
                                  display a corrected answer under an
                                  uncorrected name
            "player_found": bool,
            "categories": [{"play_type": str, "percentile": float, "poss": float}, ...]
                          -- every category the player qualifies for, sorted
                          descending by percentile (Cut excluded, see
                          _SIGNATURE_EXCLUDED_TYPES)
            "signature": [str, ...] or None
                          -- the top category name, or multiple names if two
                          or more are within _SIGNATURE_TIE_MARGIN of each
                          other (a genuine multi-category strength rather
                          than an arbitrary single pick), or None if the
                          player doesn't qualify for any category
        }
    """
    resolved_name = resolve_player_name(player_name)
    if resolved_name is None:
        return {
            "player_name": player_name,
            "player_found": False,
            "categories": [],
            "signature": None,
        }

    categories = []
    for play_type in sorted(_VALID_PLAY_TYPES - _SIGNATURE_EXCLUDED_TYPES):
        df = playtype_offense(play_type)
        row = df[df["PLAYER_NAME"] == resolved_name]
        if row.empty:
            continue
        categories.append({
            "play_type": play_type,
            "percentile": float(row.iloc[0]["PERCENTILE"]),
            "poss": float(row.iloc[0]["POSS"]),
        })

    categories.sort(key=lambda c: c["percentile"], reverse=True)

    if not categories:
        return {
            "player_name": resolved_name,
            "player_found": True,
            "categories": [],
            "signature": None,
        }

    top_percentile = categories[0]["percentile"]

    # the player's own best category has to clear _SIGNATURE_MIN_PERCENTILE
    # before anything is called a signature at all -- ranking #1 among your
    # own mediocre categories doesn't make a strength.
    if top_percentile < _SIGNATURE_MIN_PERCENTILE:
        signature = None
    else:
        signature = [
            c["play_type"] for c in categories
            if top_percentile - c["percentile"] <= _SIGNATURE_TIE_MARGIN
        ]

    return {
        "player_name": resolved_name,
        "player_found": True,
        "categories": categories,
        "signature": signature,
    }


def format_signature_play_type_answer(result: dict) -> str:
    """Format a one-sentence answer for signature_play_type()'s result."""
    player = result["player_name"]

    if not result.get("player_found", True):
        return f"{player} isn't in this season's NBA player data — check the spelling, or the name may not match a current roster player."

    if not result["categories"]:
        return f"{player} doesn't qualify for any offensive play-type category this season."

    signature = result["signature"]
    top = result["categories"][0]

    if signature is None:
        return (
            f"No clear signature play type for {player} this season — their best "
            f"qualifying category ({_PLAYTYPE_OFFENSE_LABEL.get(top['play_type'], top['play_type'])}) "
            f"is only in the {top['percentile']:.1%} percentile."
        )

    if len(signature) == 1:
        label = _PLAYTYPE_OFFENSE_LABEL.get(top["play_type"], top["play_type"])
        return (
            f"{player}'s signature play type is {label} "
            f"({top['percentile']:.1%} percentile, {top['poss']:.1f} poss/g)."
        )

    cats_str = " and ".join(
        _PLAYTYPE_OFFENSE_LABEL.get(pt, pt) for pt in signature
    )
    return (
        f"{player} doesn't have a single standout — {cats_str} are both genuine "
        f"strengths this season, within {_SIGNATURE_TIE_MARGIN:.0%} percentile points "
        f"of each other (top: {top['percentile']:.1%})."
    )
