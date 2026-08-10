"""
Required-column schemas for every CSV this app reads, derived directly
from what the compute functions in compute_defense.py / compute_offense.py
/ compute_college.py / report.py actually access -- not guessed, not
"every column that happens to exist in the current pull". A column listed
here is one a real function will KeyError on if it's missing; a column NOT
listed here may exist in the CSV but nothing in this codebase currently
depends on it.

Used by api.py's startup schema check (validate_startup_schema) and by
refresh_data.sh's manifest generation.
"""

from __future__ import annotations

# ── hustle_stats_2025_26.csv / hustle_stats_2024_25.csv ──────────────────────
# Read by: deflections_per36, contest_profile_per36, boxout_conversion,
# hustle_iq_composite, hustle_vs_suppression_gap (all in compute_defense.py),
# and resolve_player_name's roster check (compute_offense.py, current
# season only).
HUSTLE_COLUMNS = {
    "PLAYER_NAME", "TEAM_ABBREVIATION", "G", "MIN",
    "CONTESTED_SHOTS", "CONTESTED_SHOTS_2PT", "CONTESTED_SHOTS_3PT",
    "DEFLECTIONS", "CHARGES_DRAWN",
    "DEF_LOOSE_BALLS_RECOVERED",
    "BOX_OUTS", "BOX_OUT_PLAYER_REBS",
}

# ── shot_defense_{overall,3pt,2pt,rim}_2025_26.csv ────────────────────────────
# Read by: shot_suppression, hustle_vs_suppression_gap (compute_defense.py).
# Column NAMES differ per category (_DEFEND_SCHEMA in compute_defense.py) --
# each category's required set reflects its own specific columns, not a
# shared superset (e.g. 3PT/2PT/rim use "PLUSMINUS", Overall uses
# "PCT_PLUSMINUS" directly).
SHOT_DEFENSE_COLUMNS = {
    "Overall": {
        "PLAYER_NAME", "PLAYER_LAST_TEAM_ABBREVIATION", "PLAYER_POSITION", "G",
        "D_FGA", "D_FG_PCT", "NORMAL_FG_PCT", "PCT_PLUSMINUS",
    },
    "3 Pointers": {
        "PLAYER_NAME", "PLAYER_LAST_TEAM_ABBREVIATION", "PLAYER_POSITION", "G",
        "FG3A", "FG3_PCT", "NS_FG3_PCT", "PLUSMINUS",
    },
    "2 Pointers": {
        "PLAYER_NAME", "PLAYER_LAST_TEAM_ABBREVIATION", "PLAYER_POSITION", "G",
        "FG2A", "FG2_PCT", "NS_FG2_PCT", "PLUSMINUS",
    },
    "Less Than 6Ft": {
        "PLAYER_NAME", "PLAYER_LAST_TEAM_ABBREVIATION", "PLAYER_POSITION", "G",
        "FGA_LT_06", "LT_06_PCT", "NS_LT_06_PCT", "PLUSMINUS",
    },
}

# ── playtype_defense_*.csv / playtype_offense_*.csv (one file per category) ──
# Read by: playtype_defense, playtype_offense (both compute_ modules).
# Same required column set across every category file -- the play-type
# name changes the filename, not the schema.
PLAYTYPE_COLUMNS = {
    "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "POSS", "PPP", "FG_PCT", "PERCENTILE",
}

# ── drives_2025_26.csv ────────────────────────────────────────────────────────
# Read by: drive_efficiency (compute_offense.py).
DRIVES_COLUMNS = {
    "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "DRIVES", "DRIVE_PTS",
    "DRIVE_FG_PCT", "DRIVE_PASSES_PCT", "DRIVE_AST_PCT", "DRIVE_TOV_PCT",
}

# ── draft_class_2026.csv ──────────────────────────────────────────────────────
# Read by: college_player_lookup, college_leaderboard, college_efficiency_volume,
# youth_adjusted_leaderboard (compute_college.py).
DRAFT_CLASS_COLUMNS = {
    "pick_number", "name", "school", "class_year", "status",
    "PTS", "TRB", "AST", "USG%", "TS%", "BPM", "PER",
}


def missing_columns(df_columns, required: set[str]) -> set[str]:
    """Columns in `required` that are absent from df_columns. df_columns can
    be a pandas Index, a list, or any iterable of column names."""
    return required - set(df_columns)
