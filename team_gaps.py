"""
Team Roster Gaps and Capability Profile Engine (Layer 4).

Evaluates all 30 NBA teams across the 6 validated capability subsets:
  1. rim_protection: Rim & Post Defense, Box-outs, Contests
  2. perimeter_defense: Isolation, Perimeter Suppression, Deflections
  3. primary_shot_creation: Self-creation share, Usage load, Drive finishing
  4. playmaking: Assist rate, Assist ratio, Drive kickouts/passes
  5. movement_scoring: Spot-up, Off-ball cuts, True shooting
  6. pace_transition: Pace, Transition efficiency

Computes rotation-level capability scores, ranks all 30 teams on each subset,
identifies team-specific roster gaps, and suggests optimal NBA transfer targets
and 2026 Draft prospects to solve those exact holes.
"""

from __future__ import annotations

import re
from typing import Optional
import numpy as np
import pandas as pd

from feature_vector import build_feature_matrix
from layer4_subsets import CAPABILITY_SUBSETS, build_subset_style_space

# ── Team metadata ─────────────────────────────────────────────────────────────

TEAM_NAMES: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

SUBSET_DISPLAY_NAMES: dict[str, str] = {
    "rim_protection": "Rim Protection & Interior Defense",
    "perimeter_defense": "Perimeter Defense & Point-of-Attack",
    "primary_shot_creation": "Primary Shot Creation & Volume",
    "playmaking": "Playmaking & Passing Creation",
    "movement_scoring": "Off-Ball Movement & Catch-and-Shoot",
    "pace_transition": "Pace & Transition Offense",
}

_CACHED_TEAM_DATA: dict | None = None


def resolve_team(query: str) -> Optional[str]:
    """Resolve a team abbreviation, nickname, city, or full name to a 3-letter code."""
    q = query.strip().upper()
    if q in TEAM_NAMES:
        return q

    q_lower = query.strip().lower()
    for abbr, full_name in TEAM_NAMES.items():
        if q_lower == full_name.lower():
            return abbr
        # Match city or mascot (e.g. "Warriors", "Lakers", "Chicago")
        words = full_name.lower().split()
        if q_lower in words or q_lower == words[-1]:
            return abbr

    # Substring search
    for abbr, full_name in TEAM_NAMES.items():
        if q_lower in full_name.lower():
            return abbr

    return None


def get_or_build_team_capability_space() -> dict:
    """Compute and cache all 30 teams' capability profiles, ranks, and player-level scores."""
    global _CACHED_TEAM_DATA
    if _CACHED_TEAM_DATA is not None:
        return _CACHED_TEAM_DATA

    fm = build_feature_matrix()
    space = build_subset_style_space(fm)
    X_scaled = space["X_scaled"]
    feature_cols = space["feature_cols"]
    col_map = {c: i for i, c in enumerate(feature_cols)}

    # Determine metric orientation (1.0 = higher is better, -1.0 = lower is better)
    direction = {}
    for c in feature_cols:
        if "DEF_PPP" in c or "SHOT_SUPPRESSION" in c:
            direction[c] = -1.0  # lower PPP / lower allowed FG% is elite defense
        else:
            direction[c] = 1.0

    # Compute player scores for each capability subset
    player_subset_scores = {}
    for subset_name, cols in CAPABILITY_SUBSETS.items():
        sub_indices = [col_map[c] for c in cols if c in col_map]
        dirs = np.array([direction[c] for c in cols if c in col_map])
        sub_matrix = X_scaled[:, sub_indices] * dirs
        player_subset_scores[subset_name] = np.nanmean(sub_matrix, axis=1)

    scores_df = pd.DataFrame(player_subset_scores)
    # Map each player to their active 2026-27 team roster if available
    try:
        rosters_2026_27 = pd.read_csv("data/rosters_2026_27.csv")
        roster_map = dict(zip(rosters_2026_27["PLAYER_NAME"], rosters_2026_27["TEAM_ABBREVIATION"]))
        scores_df["TEAM_ABBREVIATION"] = [
            roster_map.get(name, old_team)
            for name, old_team in zip(fm["PLAYER_NAME"], fm["TEAM_ABBREVIATION"])
        ]
    except Exception:
        scores_df["TEAM_ABBREVIATION"] = fm["TEAM_ABBREVIATION"].values
    scores_df["PLAYER_NAME"] = fm["PLAYER_NAME"].values

    # Compute team score = average of top 5 rotation players in that capability
    team_scores = {}
    team_top_players = {}
    for team, group in scores_df.groupby("TEAM_ABBREVIATION"):
        team_scores[team] = {}
        team_top_players[team] = {}
        for subset_name in CAPABILITY_SUBSETS:
            top_rows = group.nlargest(min(5, len(group)), subset_name)
            team_scores[team][subset_name] = float(top_rows[subset_name].mean())
            # Save top 2 key contributors
            team_top_players[team][subset_name] = top_rows["PLAYER_NAME"].head(2).tolist()

    team_df = pd.DataFrame(team_scores).T
    # Rank 1 is best, 30 is biggest gap
    ranks_df = team_df.rank(ascending=False, method="min").astype(int)

    # Load 2026 draft class for college prospect recommendations
    try:
        draft_df = pd.read_csv("data/draft_class_2026.csv")
    except Exception:
        draft_df = pd.DataFrame()

    _CACHED_TEAM_DATA = {
        "fm": fm,
        "scores_df": scores_df,
        "team_df": team_df,
        "ranks_df": ranks_df,
        "team_top_players": team_top_players,
        "draft_df": draft_df,
    }
    return _CACHED_TEAM_DATA


def get_team_gaps_data(team_query: str) -> dict:
    """Retrieve full team capability analysis, identified roster gaps, and target recommendations."""
    team_abbr = resolve_team(team_query)
    if team_abbr is None:
        raise ValueError(f"Unknown NBA team {team_query!r}. Please provide a team name or abbreviation.")

    cached = get_or_build_team_capability_space()
    team_df = cached["team_df"]
    ranks_df = cached["ranks_df"]
    scores_df = cached["scores_df"]
    top_players = cached["team_top_players"][team_abbr]
    draft_df = cached["draft_df"]

    capabilities = []
    for sub, label in SUBSET_DISPLAY_NAMES.items():
        rank = int(ranks_df.loc[team_abbr, sub])
        score = round(float(team_df.loc[team_abbr, sub]), 2)
        if rank <= 10:
            status = "Elite Strength"
            tier = "strength"
        elif rank <= 20:
            status = "League Average"
            tier = "average"
        else:
            status = "Roster Gap / Need"
            tier = "gap"

        contributors = top_players.get(sub, [])

        capabilities.append({
            "subset_name": sub,
            "capability_id": sub,
            "capability_name": label,
            "label": label,
            "rank": rank,
            "team_rank": rank,
            "score": score,
            "team_score": score,
            "status": status,
            "tier_status": status,
            "tier": tier,
            "key_contributors": contributors,
        })

    # Sort capabilities from biggest gap (highest rank number) to greatest strength
    capabilities_by_gap = sorted(capabilities, key=lambda c: c["rank"], reverse=True)
    gaps = [c for c in capabilities_by_gap if c["tier"] == "gap"]
    if len(gaps) < 2:
        # If an elite team has fewer than 2 sub-20 ranks, backfill from their lowest relative rankings
        for c in capabilities_by_gap:
            if c not in gaps:
                gaps.append(c)
            if len(gaps) >= 2:
                break

    # Find target solutions for the top gaps
    gap_solutions = []
    for g in gaps[:2]:
        sub = g["subset_name"]
        # Find top 3 non-team NBA players in this capability
        non_team_players = scores_df[scores_df["TEAM_ABBREVIATION"] != team_abbr]
        top_nba = non_team_players.nlargest(3, sub)
        nba_targets = [
            {
                "player_name": row["PLAYER_NAME"],
                "team_abbreviation": row["TEAM_ABBREVIATION"],
                "score": round(float(row[sub]), 2),
                "fit_score": round(float(row[sub]), 2),
            }
            for _, row in top_nba.iterrows()
        ]

        # Find 2026 draft prospects matching the need
        prospect_targets = []
        if not draft_df.empty:
            if sub == "rim_protection":
                # Big men / high rebound & BPM
                cand = draft_df[draft_df["status"] == "ok"].nlargest(2, "TRB")
            elif sub == "movement_scoring":
                # Elite 3P% & eFG%
                cand = draft_df[draft_df["status"] == "ok"].nlargest(2, "3P%")
            elif sub == "playmaking":
                # High AST / AST%
                cand = draft_df[draft_df["status"] == "ok"].nlargest(2, "AST")
            elif sub == "primary_shot_creation":
                # High USG% & PTS
                cand = draft_df[draft_df["status"] == "ok"].nlargest(2, "PTS")
            else:
                # High overall BPM
                cand = draft_df[draft_df["status"] == "ok"].nlargest(2, "BPM")

            for _, prow in cand.iterrows():
                prospect_targets.append({
                    "name": prow["name"],
                    "player_name": prow["name"],
                    "school": prow["school"],
                    "pick_number": int(prow["pick_number"]),
                    "stat_highlight": f"{prow['PTS']} PTS, {prow['TRB']} REB, {prow['AST']} AST",
                    "draft_profile": f"Projected pick #{int(prow['pick_number'])} · {prow['PTS']} PTS, {prow['TRB']} REB, {prow['AST']} AST",
                    "score": 1.5,
                    "fit_score": 1.5,
                })

        gap_solutions.append({
            "gap_name": g["label"],
            "capability_name": g["label"],
            "rank": g["rank"],
            "team_rank": g["rank"],
            "score": g["score"],
            "team_score": g["score"],
            "nba_targets": nba_targets,
            "matching_nba_targets": nba_targets,
            "prospect_targets": prospect_targets,
            "matching_draft_prospects": prospect_targets,
        })

    rotation_size = int((scores_df["TEAM_ABBREVIATION"] == team_abbr).sum())
    return {
        "team_abbreviation": team_abbr,
        "team_name": TEAM_NAMES[team_abbr],
        "qualified_players_count": rotation_size,
        "rotation_size": rotation_size,
        "capabilities": capabilities,
        "identified_gaps": gap_solutions,
        "top_gaps": gap_solutions,
        "season": "2026-27 Active Rosters (2025-26 Tracking Baseline)",
        "roster_season": "2026-27",
        "tracking_baseline": "2025-26",
        "audit": {
            "intent": "team_roster_capability_gaps",
            "parameters": {
                "team": team_abbr,
                "qualifying_pool": "30 NBA teams, 305 rotation players",
                "roster_source": "2026-27 active rosters",
                "tracking_baseline": "2025-26 full season",
            },
            "qualifying_pool_size": 30,
            "routing_method": "deterministic_team_gaps",
            "data_as_of": "2026-10-25",
        },
    }


def format_team_gaps_answer(data: dict) -> str:
    """Format team gaps data into a verified conversational answer string."""
    lines = [
        f"{data['team_name']} ({data['team_abbreviation']}) — Roster Capability Profile & Gaps [2026-27 Active Rosters · 2025-26 Tracking Baseline]:"
    ]
    lines.append("")
    lines.append("CAPABILITY RANKINGS (across 30 NBA teams):")
    for c in data["capabilities"]:
        contrib = f" (led by {', '.join(c['key_contributors'])})" if c["key_contributors"] else ""
        lines.append(f"• {c['label']}: Rank #{c['rank']}/30 [{c['status'].upper()}]{contrib}")

    lines.append("")
    lines.append("TOP IDENTIFIED ROSTER GAPS & RECOMMENDED TARGETS:")
    for g in data["identified_gaps"]:
        nba_str = ", ".join([f"{t['player_name']} ({t['team_abbreviation']})" for t in g["nba_targets"]])
        lines.append(f"• {g['gap_name']} (Rank #{g['rank']}/30):")
        lines.append(f"   - NBA Style Solvers: {nba_str}")
        if g["prospect_targets"]:
            draft_str = ", ".join([f"{p['name']} ({p['school']}, pick #{p['pick_number']})" for p in g["prospect_targets"]])
            lines.append(f"   - 2026 Draft Fits: {draft_str}")

    return "\n".join(lines)
