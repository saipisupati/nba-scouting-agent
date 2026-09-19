"""
League-scale verification of Layer 4's percentile-space coverage metric
(layer4_percentile.py), built in response to a direct challenge: the
prior CLE-vs-CHI validation was 5 hand-picked reference players against
one hand-picked pairwise team comparison ("maximally obvious" is a
judgment call, not evidence) -- not a real verification.

This module builds an INDEPENDENT, real, checkable ground-truth stat --
TEAM_RIM_DEFENSE_REAL, a volume-weighted average of qualified rostered
players' real PLUSMINUS on rim shot defense (from
data/shot_defense_rim_2025_26.csv, the same source feature_vector.py
already reads for SHOT_SUPPRESSION_LESS_THAN_6FT) -- and correlates it
against every one of the 30 real NBA teams' Layer-4 percentile-space
coverage score for a fixed rim-protector reference player.

This is NOT circular: TEAM_RIM_DEFENSE_REAL is computed directly from
real box-tracking PLUSMINUS values, aggregated to team level. The Layer 4
coverage score is computed from nearest-rostered-player percentile
distance across a 4-feature style subset (RIM_PROTECTION_TRIMMED, which
DOES include SHOT_SUPPRESSION_LESS_THAN_6FT as one of its 4 inputs, so
this check is not fully independent of that one input feature -- flagged
explicitly in the report below, not hidden). It IS independent of the
nearest-neighbor/percentile-transform MECHANISM itself, which is the
actual thing under test here.

Team abbreviation is taken from feature_vector.py's own
build_feature_matrix() (_apply_current_team-resolved, the same current-
roster assignment every other Layer 4 module already uses for roster
subspaces) rather than the raw CSV's PLAYER_LAST_TEAM_ABBREVIATION --
those two disagree for at least one player found during this check
(Donovan Clingan: POR in the raw 2025-26 shot-defense CSV, used
elsewhere in this project as a CLE/CHI reference from an earlier,
staler snapshot) -- joining both sides of the correlation on the SAME
current team assignment avoids re-introducing the stale-team-abbreviation
bug this project already found and fixed once (see git commit a64562e).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_vector import MIN_GAMES, MIN_MINUTES, build_feature_matrix
from layer4_subsets import RIM_PROTECTION_TRIMMED
from layer4_percentile import build_percentile_matrix, get_percentile_coverage

_SHOT_DEFENSE_RIM_CSV = "data/shot_defense_rim_2025_26.csv"

# Reference players for the league-wide check. Holmgren is the one
# player that passed reliably across the prior 7-point spectrum test
# (layer4_middleband.py / layer4_percentile.py); Clingan is kept as a
# documented known-hard outlier case, run in parallel rather than
# silently dropped, per this project's standing rule against silently
# narrowing an inconvenient result.
REFERENCE_PLAYERS = ["Chet Holmgren", "Donovan Clingan"]

WEAK_CORRELATION_THRESHOLD = 0.05  # same bar layer4_subsets.py already uses


def build_team_rim_defense_groundtruth(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Volume-weighted (by FGA_LT_06) mean PLUSMINUS across each team's
    qualified rostered players (the same MIN>=15/GP>=40 pool every other
    Layer 4 module uses), joined onto feature_matrix's own
    TEAM_ABBREVIATION (current-roster, _apply_current_team-resolved) --
    not the raw CSV's PLAYER_LAST_TEAM_ABBREVIATION, which can be stale
    for a recently-traded player.

    Negative PLUSMINUS = opponents shoot BELOW their normal rate at the
    rim against this player = good defense (same sign convention as
    shot_defense_overall_2025_26.csv's PCT_PLUSMINUS, documented in
    feature_vector.py's Axis 5 / this project's README). So a team with
    good real rim defense has NEGATIVE TEAM_RIM_DEFENSE_REAL.

    Raises if a qualified rostered player is missing from the rim
    shot-defense CSV entirely -- silently dropping a player would quietly
    bias the team average, and this ground-truth stat only has value if
    it's not silently incomplete.
    """
    rim_defense = pd.read_csv(_SHOT_DEFENSE_RIM_CSV)
    dupes = rim_defense["PLAYER_NAME"].value_counts()
    dupes = dupes[dupes > 1]
    if len(dupes) > 0:
        raise ValueError(
            f"{len(dupes)} player(s) have duplicate rows in {_SHOT_DEFENSE_RIM_CSV}: "
            f"{dupes.index.tolist()}"
        )

    rim_by_name = rim_defense.set_index("PLAYER_NAME")[["FGA_LT_06", "PLUSMINUS"]]
    missing = set(feature_matrix["PLAYER_NAME"]) - set(rim_by_name.index)
    if missing:
        raise ValueError(
            f"{len(missing)} qualified player(s) not found in {_SHOT_DEFENSE_RIM_CSV}: "
            f"{sorted(missing)}"
        )

    joined = feature_matrix[["PLAYER_NAME", "TEAM_ABBREVIATION"]].merge(
        rim_by_name, left_on="PLAYER_NAME", right_index=True, how="left"
    )

    def _weighted_mean(group: pd.DataFrame) -> float:
        weights = group["FGA_LT_06"]
        if weights.sum() == 0:
            return float("nan")
        return float(np.average(group["PLUSMINUS"], weights=weights))

    team_stat = (
        joined.groupby("TEAM_ABBREVIATION")
        .apply(_weighted_mean, include_groups=False)
        .rename("TEAM_RIM_DEFENSE_REAL")
        .reset_index()
    )
    return team_stat.sort_values("TEAM_RIM_DEFENSE_REAL")


def print_groundtruth_ranking(team_stat: pd.DataFrame) -> None:
    print("=" * 70)
    print("GROUND TRUTH — TEAM_RIM_DEFENSE_REAL")
    print("(volume-weighted mean PLUSMINUS across qualified rostered players,")
    print(f" from real box-tracking data in {_SHOT_DEFENSE_RIM_CSV})")
    print("=" * 70)
    print("  Negative = opponents shoot BELOW normal rate at the rim = good defense.\n")
    for _, row in team_stat.iterrows():
        print(f"    {row['TEAM_ABBREVIATION']:<5} {row['TEAM_RIM_DEFENSE_REAL']:+.4f}")
    print()
    print("  Sanity check (hand-verifiable against well-known rosters):")
    for team in ["CLE", "MIN", "CHI"]:
        if team in team_stat["TEAM_ABBREVIATION"].values:
            val = team_stat.loc[team_stat["TEAM_ABBREVIATION"] == team, "TEAM_RIM_DEFENSE_REAL"].iloc[0]
            print(f"    {team}: {val:+.4f}")
    print()


def _current_team_of(feature_matrix: pd.DataFrame, player_name: str) -> str | None:
    row = feature_matrix.loc[feature_matrix["PLAYER_NAME"] == player_name, "TEAM_ABBREVIATION"]
    return row.iloc[0] if not row.empty else None


def run_league_wide_correlation(feature_matrix: pd.DataFrame, team_stat: pd.DataFrame) -> None:
    cols = RIM_PROTECTION_TRIMMED
    pct_matrix = build_percentile_matrix(feature_matrix, cols)
    all_teams = sorted(feature_matrix["TEAM_ABBREVIATION"].unique())

    for ref_player in REFERENCE_PLAYERS:
        print("=" * 70)
        print(f"LEAGUE-WIDE CHECK — {ref_player}, all {len(all_teams)} teams")
        print("=" * 70)

        coverage_rows = []
        for team in all_teams:
            result = get_percentile_coverage(pct_matrix, team, ref_player, cols)
            if result is None:
                continue
            nearest, pct_covered = result
            coverage_rows.append({"TEAM_ABBREVIATION": team, "PCT_COVERED": pct_covered})

        if not coverage_rows:
            print(f"  [{ref_player} could not be resolved / scored against any team]\n")
            continue

        coverage_df = pd.DataFrame(coverage_rows)
        merged = coverage_df.merge(team_stat, on="TEAM_ABBREVIATION", how="inner")
        n = len(merged)

        print(f"  {n} of {len(all_teams)} teams scored (missing teams: "
              f"{sorted(set(all_teams) - set(merged['TEAM_ABBREVIATION']))})\n")

        for _, row in merged.sort_values("PCT_COVERED", ascending=False).iterrows():
            print(f"    {row['TEAM_ABBREVIATION']:<5} coverage={row['PCT_COVERED']:6.2f}%   "
                  f"real_rim_defense={row['TEAM_RIM_DEFENSE_REAL']:+.4f}")

        # Expected direction: teams with real GOOD rim defense (negative
        # TEAM_RIM_DEFENSE_REAL) should show HIGHER coverage of an elite
        # rim-protector reference -- i.e. coverage and TEAM_RIM_DEFENSE_REAL
        # should be NEGATIVELY correlated. Reported without flipping the
        # sign, so the printed r is exactly what pandas' .corr() computes,
        # not a pre-interpreted number.
        r = merged["PCT_COVERED"].corr(merged["TEAM_RIM_DEFENSE_REAL"])
        print(f"\n  Pearson r (coverage vs. TEAM_RIM_DEFENSE_REAL): {r:+.4f}  (n={n})")
        print(f"  Expected direction: NEGATIVE (good real rim defense = negative "
              f"TEAM_RIM_DEFENSE_REAL = should predict HIGHER coverage).")

        # Robustness check: the reference player's own team scores a
        # trivial 100% coverage by construction (they're literally on
        # that roster) -- rerun the correlation excluding that one team,
        # so the headline r isn't propped up by a single mechanical
        # data point.
        own_team = _current_team_of(feature_matrix, ref_player)
        excl_df = merged[merged["TEAM_ABBREVIATION"] != own_team]
        if len(excl_df) < len(merged):
            r_excl = excl_df["PCT_COVERED"].corr(excl_df["TEAM_RIM_DEFENSE_REAL"])
            print(f"  Robustness check (excluding {ref_player}'s own roster, the one "
                  f"trivial-100%-by-construction data point): r={r_excl:+.4f} (n={len(excl_df)})")

        is_weak = abs(r) < WEAK_CORRELATION_THRESHOLD
        is_expected_direction = r < 0
        if is_weak:
            print(f"  WEAK: |r|={abs(r):.4f} is below the {WEAK_CORRELATION_THRESHOLD} "
                  "threshold this project already uses to flag a near-zero relationship "
                  "(layer4_subsets.py's WEAK_CORRELATION_THRESHOLD). No real evidence the "
                  f"coverage metric tracks real rim-defense outcomes for {ref_player} as "
                  "the reference, at league scale.")
        elif is_expected_direction:
            print(f"  Correlation is in the EXPECTED direction and non-trivial "
                  f"(r={r:+.4f}). This is real, citable evidence -- computed against "
                  f"independent real box-tracking data across all {n} scored teams, not "
                  "5 hand-picked anecdotes -- that the coverage metric tracks something "
                  "real about team rim defense, at least for this reference player and "
                  "this one capability subset.")
        else:
            print(f"  Correlation is non-trivial but in the WRONG direction "
                  f"(r={r:+.4f} — positive, meaning worse real rim defense predicts "
                  f"HIGHER coverage). This is a real, citable finding that the metric is "
                  f"unreliable at league scale for {ref_player} as the reference, not "
                  "just in the earlier CLE/CHI pairwise case.")
        print()


def main() -> None:
    print("Building feature matrix from Layer 1 axes "
          f"(qualification floor: MIN>={MIN_MINUTES}, GP>={MIN_GAMES})...")
    feature_matrix = build_feature_matrix()
    print(f"Feature matrix: {feature_matrix.shape[0]} qualified players\n")

    team_stat = build_team_rim_defense_groundtruth(feature_matrix)
    print_groundtruth_ranking(team_stat)

    print("NOTE -- this check is not fully independent of the coverage metric's")
    print("inputs: TEAM_RIM_DEFENSE_REAL and one of RIM_PROTECTION_TRIMMED's 4")
    print("features (SHOT_SUPPRESSION_LESS_THAN_6FT) are both built from the same")
    print(f"underlying {_SHOT_DEFENSE_RIM_CSV} PLUSMINUS values -- at the PLAYER")
    print("level for the coverage metric's inputs, aggregated to TEAM level here.")
    print("This checks the nearest-neighbor/percentile-transform MECHANISM (does")
    print("team-level aggregation of real data agree with the metric's own")
    print("player-level nearest-neighbor logic), not full external validation")
    print("against an unrelated outcome variable like NET_RATING.\n")

    run_league_wide_correlation(feature_matrix, team_stat)
    print("Done.")


if __name__ == "__main__":
    main()
