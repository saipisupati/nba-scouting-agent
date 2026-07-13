"""
Full player scouting report: assembles every relevant existing compute
function into one profile for a single player.

Pure deterministic assembly and formatting of numbers already computed by
compute_defense.py / compute_offense.py. No LLM involvement anywhere in this
file — a report combines many numbers at once, and any LLM involvement in
assembling them risks silently misstating one of them. Every caveat string
is imported verbatim from the module that owns it (query_router.py for the
PRRollman caveat, compute_offense.py for the OffScreen/Cut notes) rather than
retyped, so this file cannot drift out of sync with the wording users already
see from the router.

Section-by-section qualification: each section re-derives the player's
qualification against that metric's own existing filters by calling the real
compute function and checking whether the player's row survives the filter.
There is no separate "does X qualify" helper anywhere in the codebase (see
hustle_stats.py's own row lookup pattern) — calling the function and checking
membership in the result is the established pattern, reused here.

generate_scouting_report_data() builds one structured representation (a list
of section dicts: title, qualified rows, caveats) that both the plain-text
CLI report (generate_scouting_report) and the /report API endpoint render
from — the qualification logic lives in exactly one place.
"""

from __future__ import annotations

import pandas as pd

from compute_defense import (
    deflections_per36,
    contest_profile_per36,
    boxout_conversion,
    hustle_iq_composite,
    shot_suppression,
    hustle_vs_suppression_gap,
    playtype_defense,
    year_over_year_delta,
    _PLAYTYPE_CSV as _DEF_PLAYTYPE_CSV,
)
from compute_offense import (
    playtype_offense,
    format_playtype_offense_answer,
    _PLAYTYPE_CSV as _OFF_PLAYTYPE_CSV,
)
from query_router import _PRROLLMAN_CAVEAT

HUSTLE_CSV = {
    "2025-26": "hustle_stats_2025_26.csv",
    "2024-25": "hustle_stats_2024_25.csv",
}
SHOT_DEFENSE_CSV = {
    "2025-26": {
        "Overall":       "shot_defense_overall_2025_26.csv",
        "3 Pointers":    "shot_defense_3pt_2025_26.csv",
        "Less Than 6Ft": "shot_defense_rim_2025_26.csv",
    },
}

_DEF_PLAYTYPE_CATEGORIES = list(_DEF_PLAYTYPE_CSV)   # 7 categories, no Cut/Transition
_OFF_PLAYTYPE_CATEGORIES = list(_OFF_PLAYTYPE_CSV)   # 9 categories


def _player_row(df: pd.DataFrame, player_name: str) -> pd.Series | None:
    match = df[df["PLAYER_NAME"] == player_name]
    return None if match.empty else match.iloc[0]


def _rank_of(ranked: pd.DataFrame, player_name: str) -> int:
    return int(ranked.index[ranked["PLAYER_NAME"] == player_name][0]) + 1


def _strip_note_prefix(caveat: str) -> str:
    """Some caveat constants (e.g. _PRROLLMAN_CAVEAT, the offense-side
    OffScreen/Cut notes) already embed a leading 'NOTE: ' since that's how
    they render inline in query_router's plain-text answers. The report's
    "caveats" list is meant to hold bare caveat text that any renderer can
    label consistently (the CLI prefixes with "NOTE:", the UI with a "Note"
    badge) — normalize here so callers never double-label."""
    return caveat[len("NOTE: "):] if caveat.startswith("NOTE: ") else caveat


# ── structured section builders ───────────────────────────────────────────────
# Each returns {"title": str, "rows": [ {label, qualified, text, caveats: []} ]}
# so the API can render distinct rows per metric and the CLI can flatten them
# into plain text — same underlying data, two presentations.

def _hustle_section_data(player_name: str, hustle_df: pd.DataFrame) -> dict:
    checks = [
        ("Deflections per 36",     deflections_per36(hustle_df),      "DEFLECTIONS_PER36",  []),
        ("Contest volume per 36",  contest_profile_per36(hustle_df),  "TOTAL_CONTESTED_PER36", []),
        ("Boxout conversion rate", boxout_conversion(hustle_df),      "BOXOUT_CONV_RATE", []),
        ("Hustle IQ composite",    hustle_iq_composite(hustle_df),    "HUSTLE_IQ_COMPOSITE",
         ["NOT an official NBA stat — weighted z-score of def. loose balls + charges drawn."]),
    ]

    rows = []
    for label, ranked, col, caveats in checks:
        row = _player_row(ranked, player_name)
        if row is None:
            rows.append({
                "label": label, "qualified": False,
                "text": "Insufficient sample this season (does not clear the qualification floor).",
                "caveats": [], "value": None, "better": None,
            })
            continue
        rank = _rank_of(ranked, player_name)
        rows.append({
            "label": label, "qualified": True,
            "text": f"{row[col]} (rank #{rank} of {len(ranked)} qualified players)",
            "caveats": caveats,
            "value": float(row[col]), "better": "higher",
        })

    return {"title": "Hustle / Activity Profile", "rows": rows}


def _shot_suppression_section_data(player_name: str, season: str) -> dict:
    csv_map = SHOT_DEFENSE_CSV.get(season)
    if csv_map is None:
        return {"title": "Shot Suppression", "rows": [{
            "label": "Shot suppression", "qualified": False,
            "text": f"No shot-defense data file mapping available for season {season}.",
            "caveats": [], "value": None, "better": None,
        }]}

    rows = []
    for category in ("Overall", "3 Pointers", "Less Than 6Ft"):
        df = pd.read_csv(csv_map[category])
        ranked = shot_suppression(df, category=category)
        row = _player_row(ranked, player_name)
        label = "Rim" if category == "Less Than 6Ft" else category
        if row is None:
            rows.append({
                "label": label, "qualified": False,
                "text": "Insufficient sample this season (below minimum defended FGA).",
                "caveats": [], "value": None, "better": None,
            })
            continue
        rank = _rank_of(ranked, player_name)
        pm = row["PCT_PLUSMINUS"]
        direction = "suppresses" if pm < 0 else "does not suppress"
        rows.append({
            "label": label, "qualified": True,
            "text": (
                f"{pm:+.3f} PCT_PLUSMINUS (rank #{rank} of {len(ranked)} qualified players) — "
                f"opponents shoot {row['DEF_FG_PCT']:.1%} vs. their normal {row['NORMAL_FG_PCT']:.1%}, "
                f"this player {direction} shooting efficiency here"
            ),
            "caveats": [],
            # lower PCT_PLUSMINUS = opponents shoot worse than normal = better defense
            "value": float(pm), "better": "lower",
        })

    return {"title": "Shot Suppression", "rows": rows}


def _defense_playtype_section_data(player_name: str, season: str) -> dict:
    rows = []
    for category in _DEF_PLAYTYPE_CATEGORIES:
        ranked = playtype_defense(category)
        row = _player_row(ranked, player_name)
        if row is None:
            rows.append({
                "label": category, "qualified": False,
                "text": f"Insufficient sample for {category} defense this season.",
                "caveats": [], "value": None, "better": None,
            })
            continue
        rank = _rank_of(ranked, player_name)
        poss = int(row["POSS"])
        caveats = [_strip_note_prefix(_PRROLLMAN_CAVEAT)] if category == "PRRollman" else []
        rows.append({
            "label": category, "qualified": True,
            "text": (
                f"{row['PPP']} PPP allowed ({row['FG_PCT']:.1%} FG%) on {poss} possessions "
                f"(rank #{rank} of {len(ranked)} qualified players)"
            ),
            "caveats": caveats,
            # lower PPP allowed = better defense
            "value": float(row["PPP"]), "better": "lower",
        })

    return {"title": "Defensive Play-Type Profile", "rows": rows}


def _offense_playtype_section_data(player_name: str, season: str) -> dict:
    rows = []
    for category in _OFF_PLAYTYPE_CATEGORIES:
        ranked = playtype_offense(category)
        row = _player_row(ranked, player_name)
        if row is None:
            rows.append({
                "label": category, "qualified": False,
                "text": f"Insufficient sample for {category} offense this season.",
                "caveats": [], "value": None, "better": None,
            })
            continue
        rank = _rank_of(ranked, player_name)
        poss_per_game = row["POSS"]

        # playtype_offense's POSS column is possessions PER GAME (unlike
        # playtype_defense, where POSS is already a season total) — recover
        # the season total from GP in the raw CSV for display and for the
        # OffScreen caveat threshold, which is defined in season-total terms.
        raw = pd.read_csv(_OFF_PLAYTYPE_CSV[category])
        raw_row = _player_row(raw, player_name)
        total_poss = round(raw_row["POSS"] * raw_row["GP"]) if raw_row is not None else None
        total_poss_str = f"~{total_poss}" if total_poss is not None else "unknown"

        sentence = format_playtype_offense_answer(row, category, season, total_poss=total_poss)
        sort_note = " [sorted by volume, not PPP, for this category]" if category == "Cut" else ""

        caveats = []
        if "NOTE:" in sentence:
            caveats.append(_strip_note_prefix(sentence[sentence.index("NOTE:"):]))

        # Cut is ranked by volume, not efficiency (compute_offense._SORT_BY_VOLUME) —
        # per that module's own reasoning, PPP is too compressed across qualifiers
        # to mean "better" for this category, so it's excluded from highlighting
        # rather than comparing on a metric the codebase itself says isn't
        # meaningful for ranking quality.
        better = None if category == "Cut" else "higher"

        rows.append({
            "label": category, "qualified": True,
            "text": (
                f"{row['PPP']} PPP ({row['FG_PCT']:.1%} FG%), {poss_per_game} poss/game "
                f"({total_poss_str} total this season) (rank #{rank} of {len(ranked)} qualified players){sort_note}"
            ),
            "caveats": caveats,
            "value": float(row["PPP"]), "better": better,
        })

    return {"title": "Offensive Play-Type Profile", "rows": rows}


def _gap_section_data(player_name: str, hustle_df: pd.DataFrame, season: str) -> dict:
    csv_map = SHOT_DEFENSE_CSV.get(season)
    if csv_map is None:
        return {"title": "Hustle-vs-Suppression Gap", "rows": [{
            "label": "Gap", "qualified": False,
            "text": f"No shot-defense data available for season {season}.",
            "caveats": [], "value": None, "better": None,
        }]}

    defend_df = pd.read_csv(csv_map["Overall"])
    ranked = hustle_vs_suppression_gap(hustle_df, defend_df)
    row = _player_row(ranked, player_name)

    if row is None:
        return {"title": "Hustle-vs-Suppression Gap", "rows": [{
            "label": "Gap", "qualified": False,
            "text": (
                "Insufficient sample for the hustle-vs-suppression gap this season "
                "(does not clear both the hustle and shot-suppression qualification floors)."
            ),
            "caveats": [], "value": None, "better": None,
        }]}

    gap = row["GAP"]
    position = row["PLAYER_POSITION"]
    group_size = int((ranked["PLAYER_POSITION"] == position).sum())
    label = "quiet but effective" if gap > 0 else "busy but not impactful"

    # HUSTLE_ACTIVITY_RANK and SUPPRESSION_RANK here are position-grouped ranks
    # computed inside hustle_vs_suppression_gap itself (a blend of
    # DEFLECTIONS_PER36 + TOTAL_CONTESTED_PER36 rank, averaged, ranked only
    # against same-position players) — a different ranking system from the
    # league-wide, single-metric ranks in the Hustle/Activity section. Both
    # are real and correctly computed, but they answer different questions.
    text = (
        f"GAP: {gap:+.1f} -> {label}. "
        f"Hustle activity rank: #{row['HUSTLE_ACTIVITY_RANK']:.0f} of {group_size} "
        f"(position group: {position}) | Suppression rank: #{row['SUPPRESSION_RANK']:.0f} of {group_size} "
        f"(position group: {position}). "
        f"Deflections/36: {row['DEFLECTIONS_PER36']} | Contested shots/36: {row['TOTAL_CONTESTED_PER36']} | "
        f"Suppression PCT_PLUSMINUS: {row['PCT_PLUSMINUS']:+.3f}"
    )

    caveats = [
        "GAP is a custom rank-difference metric, not an official NBA stat. Hustle activity rank "
        "here is a blend of deflections and contest volume, ranked only within this player's "
        "position group — it will not match the league-wide, single-metric ranks shown in the "
        "Hustle/Activity section, which measure different things.",
        "This GAP is calculated from Overall shot suppression, not the 3PT-specific category — "
        "a player can suppress efficiency overall while being weaker from a specific zone. See "
        "the Shot Suppression section for the category breakdown; the same player's 3PT "
        "suppression figure is repeated below for direct contrast with the Overall number above.",
    ]

    threept_csv = csv_map.get("3 Pointers")
    if threept_csv is not None:
        threept_ranked = shot_suppression(pd.read_csv(threept_csv), category="3 Pointers")
        threept_row = _player_row(threept_ranked, player_name)
        if threept_row is not None:
            threept_rank = _rank_of(threept_ranked, player_name)
            text += (
                f". 3PT suppression (for contrast): {threept_row['PCT_PLUSMINUS']:+.3f} PCT_PLUSMINUS "
                f"(rank #{threept_rank} of {len(threept_ranked)} qualified players)"
            )
        else:
            text += ". 3PT suppression (for contrast): insufficient sample this season (below minimum defended FGA)."

    # GAP is not treated as a competitive "better/worse" metric for highlighting —
    # positive vs. negative describes two different defensive profiles (quiet-but-
    # effective vs. busy-but-not-impactful), not a quality ranking one can win.
    return {"title": "Hustle-vs-Suppression Gap", "rows": [{
        "label": "Gap", "qualified": True, "text": text, "caveats": caveats,
        "value": float(gap), "better": None,
    }]}


def _yoy_section_data(player_name: str, current_df: pd.DataFrame, prior_df: pd.DataFrame | None) -> dict:
    if prior_df is None:
        return {"title": "Year-Over-Year Trend", "rows": [{
            "label": "Deflections/36 trend", "qualified": False,
            "text": "No prior-season data file available for comparison.",
            "caveats": [], "value": None, "better": None,
        }]}

    ranked = year_over_year_delta(current_df, prior_df, metric="deflections_per36")
    row = _player_row(ranked, player_name)

    if row is None:
        return {"title": "Year-Over-Year Trend", "rows": [{
            "label": "Deflections/36 trend", "qualified": False,
            "text": (
                "Insufficient data for year-over-year comparison (player must qualify for "
                "deflections_per36 in both seasons; a missing prior season or a season that "
                "didn't clear the minutes/games floor both show up as no match here)."
            ),
            "caveats": [], "value": None, "better": None,
        }]}

    delta = row["DELTA"]
    direction = "up" if delta > 0 else "down" if delta < 0 else "unchanged"
    return {"title": "Year-Over-Year Trend", "rows": [{
        "label": "Deflections/36 trend", "qualified": True,
        "text": (
            f"Deflections/36: {row['DEFLECTIONS_PER36_PRIOR']} -> {row['DEFLECTIONS_PER36_CUR']} "
            f"({delta:+.2f}, trending {direction})"
        ),
        "caveats": [],
        # bigger improvement (more positive DELTA) is unambiguously "better" here
        "value": float(delta), "better": "higher",
    }]}


def generate_scouting_report_data(player_name: str, season: str = "2025-26") -> dict:
    """Assemble a full scouting report as structured data: a dict with
    player_name, season, and a list of section dicts, each with a title and
    a list of row dicts ({label, qualified, text, caveats}).

    This is the single source of truth for report content — both the
    plain-text CLI report and the JSON API response render from this
    structure, so qualification logic and caveat text live in exactly one
    place.
    """
    hustle_df = pd.read_csv(HUSTLE_CSV[season])

    prior_season = "2024-25" if season == "2025-26" else None
    prior_df = pd.read_csv(HUSTLE_CSV[prior_season]) if prior_season and prior_season in HUSTLE_CSV else None

    sections = [
        _hustle_section_data(player_name, hustle_df),
        _shot_suppression_section_data(player_name, season),
        _defense_playtype_section_data(player_name, season),
        _offense_playtype_section_data(player_name, season),
        _gap_section_data(player_name, hustle_df, season),
        _yoy_section_data(player_name, hustle_df, prior_df),
    ]

    return {"player_name": player_name, "season": season, "sections": sections}


def _row_winner(row_a: dict, row_b: dict) -> str | None:
    """Return 'a', 'b', or None (no highlight) for a pair of aligned rows.

    Only compares when both players qualify, both have a numeric value, and
    the metric has a defined "better" direction — a row with better=None
    (GAP, Cut) is shown side by side without a winner, since those aren't
    single-axis competitive comparisons. A tie (equal values) also yields
    no highlight rather than an arbitrary pick.
    """
    if not (row_a["qualified"] and row_b["qualified"]):
        return None
    if row_a["value"] is None or row_b["value"] is None:
        return None
    if row_a["better"] != row_b["better"] or row_a["better"] is None:
        return None

    if row_a["value"] == row_b["value"]:
        return None
    if row_a["better"] == "higher":
        return "a" if row_a["value"] > row_b["value"] else "b"
    return "a" if row_a["value"] < row_b["value"] else "b"


def compare_players_data(player_a: str, player_b: str, season: str = "2025-26") -> dict:
    """Head-to-head comparison of two players, reusing
    generate_scouting_report_data() for each rather than any new assembly
    logic. Sections and row order are deterministic and identical for any
    player (the same fixed category lists are iterated in the same order
    by generate_scouting_report_data()), so sections/rows can be safely
    zipped by index across the two players' reports.

    Returns {player_a, player_b, season, sections}, where each section has
    {title, rows}, and each row has {label, a: {...row...}, b: {...row...},
    winner: 'a' | 'b' | None} — 'winner' is the player with the better
    number for that specific metric, using each metric's own better-direction
    convention (e.g. lower is better for shot-suppression PCT_PLUSMINUS,
    higher is better for PPP and hustle metrics), or None when the metric
    isn't a meaningful single-axis comparison (GAP, Cut) or either player
    doesn't qualify.
    """
    data_a = generate_scouting_report_data(player_a, season)
    data_b = generate_scouting_report_data(player_b, season)

    sections = []
    for section_a, section_b in zip(data_a["sections"], data_b["sections"]):
        rows = []
        for row_a, row_b in zip(section_a["rows"], section_b["rows"]):
            rows.append({
                "label": row_a["label"],
                "a": row_a,
                "b": row_b,
                "winner": _row_winner(row_a, row_b),
            })
        sections.append({"title": section_a["title"], "rows": rows})

    return {
        "player_a": player_a, "player_b": player_b,
        "season": season, "sections": sections,
    }


def _render_section_as_text(section: dict, number: int) -> str:
    title = f"{number}. {section['title']}"
    bar = "-" * len(title)
    lines = [title, bar, ""]
    any_qualified = False
    for row in section["rows"]:
        any_qualified = any_qualified or row["qualified"]
        lines.append(f"  {row['label']}: {row['text']}" if section["title"] not in
                      ("Hustle-vs-Suppression Gap", "Year-Over-Year Trend") else f"  {row['text']}")
        for caveat in row["caveats"]:
            lines.append(f"    NOTE: {caveat}")
    if not any_qualified and len(section["rows"]) > 1:
        lines.append("")
        lines.append(f"  No {section['title'].lower()} category qualifies for this player this season.")
    return "\n".join(lines)


def generate_scouting_report(player_name: str, season: str = "2025-26") -> str:
    """Assemble a full scouting report for one player from every relevant
    existing compute function, as a formatted plain-text string. Each
    section independently checks the player's qualification against that
    metric's own sample-size floor; a section the player doesn't qualify
    for is reported explicitly ("Insufficient sample for ...") rather than
    silently omitted, and one section failing to qualify never prevents
    the others from running.

    No LLM involvement — every number and every caveat string here comes
    directly from the existing compute_defense.py / compute_offense.py /
    query_router.py functions and constants.
    """
    data = generate_scouting_report_data(player_name, season)

    header = f"SCOUTING REPORT: {player_name} ({season})"
    parts = [header, "=" * len(header), ""]
    for i, section in enumerate(data["sections"], start=1):
        parts.append(_render_section_as_text(section, i))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


if __name__ == "__main__":
    for name in ["Alex Caruso", "Donovan Clingan", "Nikola Jokić"]:
        print(generate_scouting_report(name))
        print("\n")
