"""
Fixture-based unit tests: small, hand-built DataFrames with known inputs
and known expected outputs, independent of the live CSVs. Complements
test_router.py/test_report.py (which exercise real data end-to-end) by
locking in the exact bug classes this project has actually hit this
weekend -- sign-convention errors, threshold-boundary errors, silent
wrong-match errors -- so they can't silently regress.

Same print-and-assert convention as test_router.py/test_report.py:
prints "ok"/"FAIL" per check, exits 1 if anything failed.
"""

import sys
from unittest.mock import patch

import pandas as pd

from compute_defense import deflections_per36
from compute_offense import resolve_player_name
from report import _player_row, _shot_suppression_section_data

failures = []


def check(condition: bool, message: str):
    if not condition:
        failures.append(message)
        print(f"  FAIL: {message}")
    else:
        print(f"  ok: {message}")


# ── 1. deflections_per36: ranking + per-36 math on a tiny known fixture ──────
print("=" * 70)
print("1. deflections_per36 — tiny fixture, known math")
print("=" * 70)

fixture = pd.DataFrame([
    # 30 deflections in 20 min -> 30/20*36 = 54.0 per 36 (highest)
    {"PLAYER_NAME": "Fixture A", "TEAM_ABBREVIATION": "AAA", "G": 40, "MIN": 20, "DEFLECTIONS": 30},
    # 10 deflections in 30 min -> 10/30*36 = 12.0 per 36 (lowest)
    {"PLAYER_NAME": "Fixture B", "TEAM_ABBREVIATION": "BBB", "G": 40, "MIN": 30, "DEFLECTIONS": 10},
    # 18 deflections in 18 min -> 18/18*36 = 36.0 per 36 (middle)
    {"PLAYER_NAME": "Fixture C", "TEAM_ABBREVIATION": "CCC", "G": 40, "MIN": 18, "DEFLECTIONS": 18},
    # below min_games=40 -> must be excluded regardless of rate
    {"PLAYER_NAME": "Fixture D (thin)", "TEAM_ABBREVIATION": "DDD", "G": 5, "MIN": 20, "DEFLECTIONS": 100},
])

result = deflections_per36(fixture)
check(len(result) == 3, f"thin-sample player (G=5) correctly excluded, 3 of 4 rows qualify: got {len(result)}")
check(list(result["PLAYER_NAME"]) == ["Fixture A", "Fixture C", "Fixture B"],
      f"ranked descending by DEFLECTIONS_PER36: got {list(result['PLAYER_NAME'])}")
check(result.iloc[0]["DEFLECTIONS_PER36"] == 54.0, f"Fixture A math: 30/20*36 = 54.0, got {result.iloc[0]['DEFLECTIONS_PER36']}")
check(result.iloc[1]["DEFLECTIONS_PER36"] == 36.0, f"Fixture C math: 18/18*36 = 36.0, got {result.iloc[1]['DEFLECTIONS_PER36']}")
check(result.iloc[2]["DEFLECTIONS_PER36"] == 12.0, f"Fixture B math: 10/30*36 = 12.0, got {result.iloc[2]['DEFLECTIONS_PER36']}")


# ── 2. PCT_PLUSMINUS sign convention in report.py's suppression formatting ──
# This is the exact bug class this project already hit once: PCT_PLUSMINUS
# negative = opponents shoot WORSE than normal = player suppresses shooting.
# Positive = opponents shoot BETTER than normal = player does NOT suppress.
# Tested via report._shot_suppression_section_data (where the "suppresses"/
# "does not suppress" wording actually lives), not compute_defense.py itself.
print()
print("=" * 70)
print("2. PCT_PLUSMINUS sign convention (report._shot_suppression_section_data)")
print("=" * 70)

_overall_fixture = pd.DataFrame([
    # negative PCT_PLUSMINUS -> opponents shoot BELOW normal -> "suppresses"
    {"PLAYER_NAME": "Suppressor", "PLAYER_LAST_TEAM_ABBREVIATION": "SUP", "G": 60,
     "D_FGA": 200, "D_FG_PCT": 0.40, "NORMAL_FG_PCT": 0.47, "PCT_PLUSMINUS": -0.07},
    # positive PCT_PLUSMINUS -> opponents shoot ABOVE normal -> "does not suppress"
    {"PLAYER_NAME": "Non-Suppressor", "PLAYER_LAST_TEAM_ABBREVIATION": "NON", "G": 60,
     "D_FGA": 200, "D_FG_PCT": 0.55, "NORMAL_FG_PCT": 0.47, "PCT_PLUSMINUS": 0.08},
])
_threept_fixture = pd.DataFrame([
    {"PLAYER_NAME": "Suppressor", "PLAYER_LAST_TEAM_ABBREVIATION": "SUP", "G": 60,
     "FG3A": 200, "FG3_PCT": 0.30, "NS_FG3_PCT": 0.36, "PLUSMINUS": -0.06},
    {"PLAYER_NAME": "Non-Suppressor", "PLAYER_LAST_TEAM_ABBREVIATION": "NON", "G": 60,
     "FG3A": 200, "FG3_PCT": 0.40, "NS_FG3_PCT": 0.36, "PLUSMINUS": 0.04},
])
_rim_fixture = pd.DataFrame([
    {"PLAYER_NAME": "Suppressor", "PLAYER_LAST_TEAM_ABBREVIATION": "SUP", "G": 60,
     "FGA_LT_06": 200, "LT_06_PCT": 0.55, "NS_LT_06_PCT": 0.63, "PLUSMINUS": -0.08},
    {"PLAYER_NAME": "Non-Suppressor", "PLAYER_LAST_TEAM_ABBREVIATION": "NON", "G": 60,
     "FGA_LT_06": 200, "LT_06_PCT": 0.70, "NS_LT_06_PCT": 0.63, "PLUSMINUS": 0.07},
])

with patch("report.SHOT_DEFENSE_CSV", {"2025-26": {"Overall": "x", "3 Pointers": "y", "Less Than 6Ft": "z"}}), \
     patch("report.pd.read_csv", side_effect=lambda path: (
         _overall_fixture if path == "x" else _threept_fixture if path == "y" else _rim_fixture
     )):
    section = _shot_suppression_section_data("Suppressor", "2025-26")
    overall_row = next(r for r in section["rows"] if r["label"] == "Overall")
    check("this player suppresses shooting efficiency here" in overall_row["text"],
          f"negative PCT_PLUSMINUS (-0.07) reads as 'suppresses': {overall_row['text']!r}")

    section2 = _shot_suppression_section_data("Non-Suppressor", "2025-26")
    overall_row2 = next(r for r in section2["rows"] if r["label"] == "Overall")
    check("this player does not suppress shooting efficiency here" in overall_row2["text"],
          f"positive PCT_PLUSMINUS (+0.08) reads as 'does not suppress': {overall_row2['text']!r}")


# ── 3. Total-possession qualification boundary (playtype_defense/offense) ───
# playtype_defense/playtype_offense both read their own CSV internally
# (pd.read_csv(_PLAYTYPE_CSV[play_type])) rather than accepting a df
# parameter, so a fixture test has to patch that read rather than pass
# a DataFrame directly. Testing playtype_defense's boundary: threshold
# is POSS >= min_poss (a single per-game-rate floor, no separate
# total-possession check at this layer -- that combined check exists in
# playtype_offense instead).
print()
print("=" * 70)
print("3. playtype_defense — POSS threshold boundary (below / at / above)")
print("=" * 70)

_threshold = 25.0  # Isolation's real default per _PLAYTYPE_DEFAULT_MIN_POSS
_boundary_fixture = pd.DataFrame([
    {"PLAYER_NAME": "Below", "TEAM_ABBREVIATION": "BEL", "POSS": _threshold - 1, "PPP": 0.90, "FG_PCT": 0.40},
    {"PLAYER_NAME": "AtThreshold", "TEAM_ABBREVIATION": "ATT", "POSS": _threshold, "PPP": 0.85, "FG_PCT": 0.42},
    {"PLAYER_NAME": "Above", "TEAM_ABBREVIATION": "ABV", "POSS": _threshold + 1, "PPP": 0.80, "FG_PCT": 0.44},
])

from compute_defense import playtype_defense

with patch("compute_defense.pd.read_csv", return_value=_boundary_fixture):
    result = playtype_defense("Isolation", min_poss=_threshold)
    names = set(result["PLAYER_NAME"])
    check("Below" not in names, f"player 1 POSS below threshold ({_threshold - 1} < {_threshold}) correctly excluded")
    check("AtThreshold" in names, f"player exactly at threshold ({_threshold}) correctly included (>=, not >)")
    check("Above" in names, f"player above threshold ({_threshold + 1}) correctly included")
    check(len(result) == 2, f"exactly 2 of 3 fixture rows qualify: got {len(result)}")


# ── 4. signature_play_type's 0.04 tie-margin boundary ────────────────────────
# signature_play_type() itself calls playtype_offense() internally for 8
# categories plus resolve_player_name() -- rather than mocking 9 CSV reads
# to exercise it indirectly, this replicates its exact, real tie-margin
# arithmetic (verbatim from compute_offense.py's own signature_play_type
# body: "signature = [c for c in categories if top_percentile - c['percentile']
# <= _SIGNATURE_TIE_MARGIN]") against hand-built category fixtures, so the
# boundary condition itself -- not the I/O around it -- is what's tested.
print()
print("=" * 70)
print("4. signature_play_type — 0.04 tie-margin boundary")
print("=" * 70)

from compute_offense import _SIGNATURE_TIE_MARGIN, _SIGNATURE_MIN_PERCENTILE

check(_SIGNATURE_TIE_MARGIN == 0.04, f"tie margin constant is still 0.04 (guards against silent threshold drift): got {_SIGNATURE_TIE_MARGIN}")


def _signature_from_categories(categories):
    """Reproduces signature_play_type()'s own tie-margin/floor logic verbatim."""
    categories = sorted(categories, key=lambda c: c["percentile"], reverse=True)
    top_percentile = categories[0]["percentile"]
    if top_percentile < _SIGNATURE_MIN_PERCENTILE:
        return None
    return [c["play_type"] for c in categories if top_percentile - c["percentile"] <= _SIGNATURE_TIE_MARGIN]


# Right at the margin -> both should tie (the real code uses "<=", not
# "<"). IEEE-754 float subtraction is not perfectly invertible in general
# -- (top - (top - 0.04)) does not equal exactly 0.04 for most decimal
# values of top (e.g. top=0.90 gives a gap of 0.040000000000000036, which
# fails <= 0.04, even though "the gap is 0.04" in decimal terms). That's a
# float-representation artifact, not a real "exactly 4.00 percentile
# points, to 17 decimal places" scenario a caller could ever construct from
# real ranked data anyway. top=0.60/bottom=0.56 was found by direct search
# to be one of the pairs (above the 0.60 floor, so this exercises the tie
# margin specifically and not the separate floor check) where the computed
# gap lands at or under 0.04 in this Python's float representation --
# confirmed by the assertion immediately below before relying on it.
_top, _bottom_at_margin = 0.60, 0.56
_measured_gap = _top - _bottom_at_margin
check(_measured_gap <= _SIGNATURE_TIE_MARGIN,
      f"fixture gap ({_measured_gap!r}) is <= the real margin constant, confirming this pair exercises the boundary")
at_margin = [{"play_type": "A", "percentile": _top}, {"play_type": "B", "percentile": _bottom_at_margin}]
sig = _signature_from_categories(at_margin)
check(sig == ["A", "B"], f"gap at the tie-margin boundary ties (uses <=, not <): got {sig}")

# unambiguously outside the margin (0.10 gap, well past 0.04) -> only the
# top category, no tie -- no float-boundary ambiguity here at all
just_outside = [{"play_type": "A", "percentile": 0.90}, {"play_type": "B", "percentile": 0.80}]
sig2 = _signature_from_categories(just_outside)
check(sig2 == ["A"], f"gap of 0.10 (well over the 0.04 margin) does NOT tie: got {sig2}")

# top category below the 0.60 floor -> no signature at all, regardless of ties
below_floor = [{"play_type": "A", "percentile": 0.55}, {"play_type": "B", "percentile": 0.52}]
sig3 = _signature_from_categories(below_floor)
check(sig3 is None, f"top category below 0.60 floor -> no signature even though within tie margin: got {sig3}")


# ── 5. resolve_player_name — exact, substring, typo, and ambiguity cases ────
print()
print("=" * 70)
print("5. resolve_player_name — exact / substring / typo / ambiguity")
print("=" * 70)

_roster_fixture = pd.DataFrame({
    "PLAYER_NAME": ["Alex Caruso", "Cameron Boozer", "Cameron Carr", "Nikola Jokić"],
})

with patch("compute_offense.pd.read_csv", return_value=_roster_fixture):
    check(resolve_player_name("Alex Caruso") == "Alex Caruso", "exact name match resolves correctly")
    check(resolve_player_name("Caruso") == "Alex Caruso", "substring match ('Caruso') resolves to full name")
    check(resolve_player_name("Karuso") is None, "typo ('Karuso', not a substring of 'Caruso') does NOT resolve -- returns None, not a wrong guess")

    # Accent-insensitive fallback: unaccented input against a roster name that
    # carries real diacritics (the NBA API's actual PLAYER_NAME values do, e.g.
    # "Nikola Jokić", "Kristaps Porziņģis") -- a typed or LLM-extracted question
    # typically won't include the accent. Regression case for the bug found via
    # feature_vector.py's reference-player lookup silently dropping Jokić.
    check(resolve_player_name("Nikola Jokic") == "Nikola Jokić", "unaccented input resolves to the accented canonical roster name")
    check(resolve_player_name("Jokic") == "Nikola Jokić", "unaccented substring match also resolves via the accent-insensitive fallback")
    check(resolve_player_name("Nikola Jokić") == "Nikola Jokić", "already-accented exact input still resolves via the normal (non-fallback) path")

    try:
        resolve_player_name("Cameron")
        check(False, "ambiguous name ('Cameron' matches 2 players) should raise ValueError, not silently pick one")
    except ValueError as e:
        check("Cameron Boozer" in str(e) and "Cameron Carr" in str(e),
              f"ambiguous name raises ValueError naming both candidates: {e}")


# ── 6. Empty qualifying pool — explicit message, not a crash ─────────────────
print()
print("=" * 70)
print("6. Empty qualifying pool — _player_row on a filtered-out fixture")
print("=" * 70)

_empty_after_filter = pd.DataFrame(columns=["PLAYER_NAME", "PPP", "POSS"])  # no rows qualify
row = _player_row(_empty_after_filter, "Anyone")
check(row is None, "empty qualifying pool returns None (the 'insufficient sample' signal), not a crash or a misleading row")

_nonempty_fixture = pd.DataFrame([{"PLAYER_NAME": "Real Player", "PPP": 1.0, "POSS": 5.0}])
row2 = _player_row(_nonempty_fixture, "Nonexistent Player")
check(row2 is None, "player genuinely absent from a non-empty pool also returns None, not a wrong row")

row3 = _player_row(_nonempty_fixture, "Real Player")
check(row3 is not None and row3["PPP"] == 1.0, "player present in the pool returns their real row, not None")


# ── Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 70)
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
