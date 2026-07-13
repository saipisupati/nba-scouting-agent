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
    _PLAYTYPE_DEFAULT_MIN_POSS as _DEF_PLAYTYPE_MIN_POSS,
)
from compute_offense import (
    playtype_offense,
    format_playtype_offense_answer,
    _PLAYTYPE_CSV as _OFF_PLAYTYPE_CSV,
    _OFFSCREEN_MODERATE_POSS_THRESHOLD,
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


def _section_header(title: str) -> str:
    bar = "-" * len(title)
    return f"{title}\n{bar}"


def _build_hustle_section(player_name: str, hustle_df: pd.DataFrame) -> str:
    lines = [_section_header("1. Hustle / Activity Profile"), ""]

    checks = [
        ("Deflections per 36",     deflections_per36(hustle_df),      "DEFLECTIONS_PER36",  ""),
        ("Contest volume per 36",  contest_profile_per36(hustle_df),  "TOTAL_CONTESTED_PER36", ""),
        ("Boxout conversion rate", boxout_conversion(hustle_df),      "BOXOUT_CONV_RATE", ""),
        ("Hustle IQ composite",    hustle_iq_composite(hustle_df),    "HUSTLE_IQ_COMPOSITE",
         " (NOT an official NBA stat — weighted z-score of def. loose balls + charges drawn)"),
    ]

    any_qualified = False
    for label, ranked, col, note in checks:
        row = _player_row(ranked, player_name)
        if row is None:
            lines.append(f"  {label}: insufficient sample this season (does not clear the qualification floor).")
            continue
        any_qualified = True
        rank = ranked.index[ranked["PLAYER_NAME"] == player_name][0] + 1
        lines.append(f"  {label}: {row[col]} (rank #{rank} of {len(ranked)} qualified players){note}")

    if not any_qualified:
        lines.append("")
        lines.append("  No hustle/activity metric qualifies for this player this season.")

    return "\n".join(lines)


def _build_shot_suppression_section(player_name: str, season: str) -> str:
    lines = [_section_header("2. Shot Suppression"), ""]

    csv_map = SHOT_DEFENSE_CSV.get(season)
    if csv_map is None:
        lines.append(f"  No shot-defense data file mapping available for season {season}.")
        return "\n".join(lines)

    any_qualified = False
    for category in ("Overall", "3 Pointers", "Less Than 6Ft"):
        df = pd.read_csv(csv_map[category])
        ranked = shot_suppression(df, category=category)
        row = _player_row(ranked, player_name)
        label = "Rim" if category == "Less Than 6Ft" else category
        if row is None:
            lines.append(f"  {label}: insufficient sample this season (below minimum defended FGA).")
            continue
        any_qualified = True
        rank = ranked.index[ranked["PLAYER_NAME"] == player_name][0] + 1
        pm = row["PCT_PLUSMINUS"]
        direction = "suppresses" if pm < 0 else "does not suppress"
        lines.append(
            f"  {label}: {pm:+.3f} PCT_PLUSMINUS (rank #{rank} of {len(ranked)} qualified players) "
            f"[opponents shoot {row['DEF_FG_PCT']:.1%} vs. their normal {row['NORMAL_FG_PCT']:.1%} — "
            f"this player {direction} shooting efficiency here]"
        )

    if not any_qualified:
        lines.append("")
        lines.append("  No shot-suppression category qualifies for this player this season.")

    return "\n".join(lines)


def _build_defense_playtype_section(player_name: str, season: str) -> str:
    lines = [_section_header("3. Defensive Play-Type Profile"), ""]

    any_qualified = False
    for category in _DEF_PLAYTYPE_CATEGORIES:
        ranked = playtype_defense(category)
        row = _player_row(ranked, player_name)
        if row is None:
            lines.append(f"  Insufficient sample for {category} defense this season.")
            continue
        any_qualified = True
        rank = ranked.index[ranked["PLAYER_NAME"] == player_name][0] + 1
        poss = int(row["POSS"])
        line = (
            f"  {category}: {row['PPP']} PPP allowed ({row['FG_PCT']:.1%} FG%) on "
            f"{poss} possessions (rank #{rank} of {len(ranked)} qualified players)"
        )
        lines.append(line)
        if category == "PRRollman":
            lines.append(f"    {_PRROLLMAN_CAVEAT}")

    if not any_qualified:
        lines.append("")
        lines.append("  No defensive play-type category qualifies for this player this season.")

    return "\n".join(lines)


def _build_offense_playtype_section(player_name: str, season: str) -> str:
    lines = [_section_header("4. Offensive Play-Type Profile"), ""]

    any_qualified = False
    for category in _OFF_PLAYTYPE_CATEGORIES:
        ranked = playtype_offense(category)
        row = _player_row(ranked, player_name)
        if row is None:
            lines.append(f"  Insufficient sample for {category} offense this season.")
            continue
        any_qualified = True
        rank = ranked.index[ranked["PLAYER_NAME"] == player_name][0] + 1
        poss_per_game = row["POSS"]

        # playtype_offense's POSS column is possessions PER GAME (unlike
        # playtype_defense, where POSS is already a season total) — recover
        # the season total from GP in the raw CSV for display and for the
        # OffScreen caveat threshold, which is defined in season-total terms.
        raw = pd.read_csv(_OFF_PLAYTYPE_CSV[category])
        raw_row = _player_row(raw, player_name)
        total_poss = round(raw_row["POSS"] * raw_row["GP"]) if raw_row is not None else None

        sentence = format_playtype_offense_answer(row, category, season, total_poss=total_poss)
        sort_note = " [sorted by volume, not PPP, for this category]" if category == "Cut" else ""
        total_poss_str = f"~{total_poss}" if total_poss is not None else "unknown"
        lines.append(
            f"  {category}: {row['PPP']} PPP ({row['FG_PCT']:.1%} FG%), {poss_per_game} poss/game "
            f"({total_poss_str} total this season) (rank #{rank} of {len(ranked)} qualified players){sort_note}"
        )
        # format_playtype_offense_answer already carries the category's caveat
        # text (Cut volume note / OffScreen possession caveat) when applicable;
        # surface it verbatim rather than re-deriving the condition here.
        if "NOTE:" in sentence:
            lines.append(f"    {sentence[sentence.index('NOTE:'):]}")

    if not any_qualified:
        lines.append("")
        lines.append("  No offensive play-type category qualifies for this player this season.")

    return "\n".join(lines)


def _build_gap_section(player_name: str, hustle_df: pd.DataFrame, season: str) -> str:
    lines = [_section_header("5. Hustle-vs-Suppression Gap"), ""]

    csv_map = SHOT_DEFENSE_CSV.get(season)
    if csv_map is None:
        lines.append(f"  No shot-defense data available for season {season}.")
        return "\n".join(lines)

    defend_df = pd.read_csv(csv_map["Overall"])
    ranked = hustle_vs_suppression_gap(hustle_df, defend_df)
    row = _player_row(ranked, player_name)

    if row is None:
        lines.append(
            "  Insufficient sample for the hustle-vs-suppression gap this season "
            "(does not clear both the hustle and shot-suppression qualification floors)."
        )
        return "\n".join(lines)

    gap = row["GAP"]
    position = row["PLAYER_POSITION"]
    group_size = int((ranked["PLAYER_POSITION"] == position).sum())
    label = "quiet but effective" if gap > 0 else "busy but not impactful"

    # HUSTLE_ACTIVITY_RANK and SUPPRESSION_RANK here are position-grouped ranks
    # computed inside hustle_vs_suppression_gap itself (a blend of
    # DEFLECTIONS_PER36 + TOTAL_CONTESTED_PER36 rank, averaged, ranked only
    # against same-position players) — a different ranking system from the
    # league-wide, single-metric ranks shown in Section 1. Both are real and
    # both are correctly computed, but they answer different questions, so
    # this section shows its own constituent ranks rather than leaning on
    # Section 1's numbers to narrate the gap.
    lines.append(
        f"  GAP: {gap:+.1f} -> {label}"
    )
    lines.append(
        f"    Hustle activity rank: #{row['HUSTLE_ACTIVITY_RANK']:.0f} of {group_size} "
        f"(position group: {position}) | Suppression rank: #{row['SUPPRESSION_RANK']:.0f} of {group_size} "
        f"(position group: {position})"
    )
    lines.append(
        f"    Deflections/36: {row['DEFLECTIONS_PER36']} | Contested shots/36: {row['TOTAL_CONTESTED_PER36']} | "
        f"Suppression PCT_PLUSMINUS: {row['PCT_PLUSMINUS']:+.3f}"
    )
    lines.append(
        "    NOTE: GAP is a custom rank-difference metric, not an official NBA stat. "
        "Hustle activity rank here is a blend of deflections and contest volume, ranked only "
        "within this player's position group — it will not match the league-wide, single-metric "
        "ranks shown in Section 1, which measure different things."
    )
    lines.append(
        "    NOTE: This GAP is calculated from Overall shot suppression, not the 3PT-specific "
        "category — a player can suppress efficiency overall while being weaker from a specific "
        "zone. See Section 2 for the category breakdown; the same player's 3PT suppression figure "
        "is repeated below for direct contrast with the Overall number above."
    )

    threept_csv = csv_map.get("3 Pointers")
    if threept_csv is not None:
        threept_ranked = shot_suppression(pd.read_csv(threept_csv), category="3 Pointers")
        threept_row = _player_row(threept_ranked, player_name)
        if threept_row is not None:
            threept_rank = threept_ranked.index[threept_ranked["PLAYER_NAME"] == player_name][0] + 1
            lines.append(
                f"    3PT suppression (for contrast): {threept_row['PCT_PLUSMINUS']:+.3f} PCT_PLUSMINUS "
                f"(rank #{threept_rank} of {len(threept_ranked)} qualified players)"
            )
        else:
            lines.append("    3PT suppression (for contrast): insufficient sample this season (below minimum defended FGA).")

    return "\n".join(lines)


def _build_yoy_section(player_name: str, current_df: pd.DataFrame, prior_df: pd.DataFrame) -> str:
    lines = [_section_header("6. Year-Over-Year Trend"), ""]

    ranked = year_over_year_delta(current_df, prior_df, metric="deflections_per36")
    row = _player_row(ranked, player_name)

    if row is None:
        lines.append(
            "  Insufficient data for year-over-year comparison (player must qualify "
            "for deflections_per36 in both seasons; a missing prior season or a season "
            "that didn't clear the minutes/games floor both show up as no match here)."
        )
        return "\n".join(lines)

    delta = row["DELTA"]
    direction = "up" if delta > 0 else "down" if delta < 0 else "unchanged"
    lines.append(
        f"  Deflections/36: {row['DEFLECTIONS_PER36_PRIOR']} -> {row['DEFLECTIONS_PER36_CUR']} "
        f"({delta:+.2f}, trending {direction})"
    )

    return "\n".join(lines)


def generate_scouting_report(player_name: str, season: str = "2025-26") -> str:
    """Assemble a full scouting report for one player from every relevant
    existing compute function. Each section independently checks the
    player's qualification against that metric's own sample-size floor;
    a section the player doesn't qualify for is reported explicitly
    ("Insufficient sample for ...") rather than silently omitted, and one
    section failing to qualify never prevents the others from running.

    No LLM involvement — every number and every caveat string here comes
    directly from the existing compute_defense.py / compute_offense.py /
    query_router.py functions and constants.
    """
    hustle_df = pd.read_csv(HUSTLE_CSV[season])

    sections = [
        f"SCOUTING REPORT: {player_name} ({season})",
        "=" * (len(player_name) + len(season) + 20),
        "",
        _build_hustle_section(player_name, hustle_df),
        "",
        _build_shot_suppression_section(player_name, season),
        "",
        _build_defense_playtype_section(player_name, season),
        "",
        _build_offense_playtype_section(player_name, season),
        "",
        _build_gap_section(player_name, hustle_df, season),
        "",
    ]

    prior_season = "2024-25" if season == "2025-26" else None
    if prior_season and prior_season in HUSTLE_CSV:
        prior_df = pd.read_csv(HUSTLE_CSV[prior_season])
        sections.append(_build_yoy_section(player_name, hustle_df, prior_df))
    else:
        sections.append(_section_header("6. Year-Over-Year Trend") + "\n\n  No prior-season data file available for comparison.")

    return "\n".join(sections)


if __name__ == "__main__":
    for name in ["Alex Caruso", "Donovan Clingan", "Nikola Jokić"]:
        print(generate_scouting_report(name))
        print("\n\n")
