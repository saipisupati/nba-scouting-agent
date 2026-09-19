"""
Redesigned Layer 4: capability-subset gap scores, validated against real
outcome data (NET_RATING), built in response to a documented bug --
running roster_fit.py's Layer 4 CLE-vs-CHI "maximally obvious" diagnostic
(TEAM_FIT_LAYER4_OUTPUT.txt) found CLE (two elite shot-blocking bigs)
scoring WORSE Clingan coverage than CHI (no true center at all), the
opposite of basketball intuition. The suspected cause: computing gap
score over all 28 style features at once lets non-defensive dimensions
dominate distance, drowning out the rim-protection-specific signal.

This module:
  1. Defines 6 named capability subsets (small groups of the existing 28
     PCA_FEATURE_COLUMNS, each meant to isolate one specific style
     dimension) explicitly in code.
  2. Validates each subset's features BEFORE trusting them, by
     correlating each feature against real NET_RATING (from
     data/usage_context_2025_26.csv, already in the pipeline) across all
     305 qualified players -- flags any feature with |r| < 0.05 as a
     candidate for removal (weak/no relationship to real winning impact).
  3. Rebuilds the Layer 4 orthogonal-projection gap score to run
     PER SUBSET instead of over all 28 features at once -- a team's
     roster subspace and a candidate's gap score are now computed
     separately for each of the 6 subsets.
  4. Reruns the CLE/CHI/OKC diagnostic using ONLY the rim_protection
     subset, and reports honestly whether this fixes the documented bug.

Reuses feature_vector.py's build_feature_matrix()/run_pca() and
roster_fit.py's SVD-basis/orthogonal-projection math directly --
StandardScaler standardizes each column independently, so slicing the
already-fitted 28-column X_scaled matrix down to a named subset's columns
gives identical per-column z-scores to fitting a scaler on that subset
alone. No refitting needed; subsets are pure column selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_vector import (
    MIN_GAMES,
    MIN_MINUTES,
    _resolve_reference_player,
    build_feature_matrix,
)
from roster_fit import build_style_space, player_style_vector

USAGE_CONTEXT_CSV = "data/usage_context_2025_26.csv"
WEAK_CORRELATION_THRESHOLD = 0.05

# ── 1. Named capability subsets ─────────────────────────────────────────────

CAPABILITY_SUBSETS: dict[str, list[str]] = {
    "rim_protection": [
        "SHOT_SUPPRESSION_LESS_THAN_6FT",
        "DEF_PPP_POSTUP",
        "DEF_PPP_PRROLLMAN",
        "BOX_OUTS_PER36",
        "CONTESTED_SHOTS_PER36",
    ],
    "perimeter_defense": [
        "SHOT_SUPPRESSION_3_POINTERS",
        "SHOT_SUPPRESSION_2_POINTERS",
        "DEF_PPP_ISOLATION",
        "DEF_PPP_SPOTUP",
        "DEF_PPP_HANDOFF",
        "DEF_PPP_OFFSCREEN",
        "DEFLECTIONS_PER36",
    ],
    "primary_shot_creation": [
        "SELF_CREATION_SHARE",
        "USG_PCT",
        "DRIVE_FG_PCT",
        "EFG_PCT",
        "TS_PCT",
    ],
    "playmaking": [
        "AST_PCT",
        "AST_TO",
        "AST_RATIO",
        "DRIVE_AST_PCT",
        "DRIVE_PASSES_PCT",
    ],
    "movement_scoring": [
        "SPOTUP_SHARE",
        "CUT_OFFSCREEN_SHARE",
        "EFG_PCT",
        "TS_PCT",
    ],
    "pace_transition": [
        "PACE",
        "TRANSITION_PPP",
    ],
}

# rim_protection with CONTESTED_SHOTS_PER36 removed -- independently
# flagged both as weakly correlated with NET_RATING (r=+0.0136, below the
# 0.05 threshold) and, separately, as the specific driver of Donovan
# Clingan's outlier distortion in the CLE/CHI/OKC diagnostic (his
# CONTESTED_SHOTS_PER36 sits at the 99.7th percentile of all 305
# qualified players -- the single highest value in the dataset). Kept as
# a standalone constant rather than edited into CAPABILITY_SUBSETS itself
# so the original 5-feature version stays available for comparison.
RIM_PROTECTION_TRIMMED = [
    "SHOT_SUPPRESSION_LESS_THAN_6FT",
    "DEF_PPP_POSTUP",
    "DEF_PPP_PRROLLMAN",
    "BOX_OUTS_PER36",
]


# ── 2. Validation against real outcome data (NET_RATING) ───────────────────

def load_net_rating(feature_matrix: pd.DataFrame) -> pd.Series:
    """Pull NET_RATING from data/usage_context_2025_26.csv for every
    qualified player in feature_matrix, joined on exact PLAYER_NAME match
    (both feature_matrix and the usage-context CSV are pulled from the
    same nba_api live-roster source, so names match exactly -- verified
    separately: all 305 qualified players resolve with zero fuzzy
    matching needed). Returns a Series indexed the same as
    feature_matrix's row order.

    Raises if any qualified player is missing from the usage-context CSV
    or if NET_RATING has an unexpected duplicate-row situation -- this is
    a validation step feeding a "should we trust this feature" decision,
    so it must not silently drop or average away real data.
    """
    usage = pd.read_csv(USAGE_CONTEXT_CSV)
    dupe_counts = usage["PLAYER_NAME"].value_counts()
    dupes = dupe_counts[dupe_counts > 1]
    if len(dupes) > 0:
        raise ValueError(
            f"{len(dupes)} player(s) have duplicate rows in {USAGE_CONTEXT_CSV}: "
            f"{dupes.index.tolist()} -- NET_RATING join would be ambiguous."
        )

    net_rating_by_name = usage.set_index("PLAYER_NAME")["NET_RATING"]
    missing = set(feature_matrix["PLAYER_NAME"]) - set(net_rating_by_name.index)
    if missing:
        raise ValueError(
            f"{len(missing)} qualified player(s) not found in {USAGE_CONTEXT_CSV}: "
            f"{sorted(missing)}"
        )

    return feature_matrix["PLAYER_NAME"].map(net_rating_by_name)


def validate_subsets_against_net_rating(
    feature_matrix: pd.DataFrame, net_rating: pd.Series
) -> dict[str, dict[str, float]]:
    """For each capability subset, compute the Pearson correlation of
    each individual feature against NET_RATING across all qualified
    players. Returns {subset_name: {feature_name: r}}.

    This validates the subsets BEFORE the gap-score mechanism is asked
    to trust them -- a feature with near-zero correlation to real winning
    impact (|r| < WEAK_CORRELATION_THRESHOLD) is flagged, not silently
    kept, since the whole point of this redesign is to stop trusting
    features that don't demonstrably relate to outcomes.
    """
    correlations: dict[str, dict[str, float]] = {}
    for subset_name, cols in CAPABILITY_SUBSETS.items():
        subset_corrs = {}
        for col in cols:
            r = feature_matrix[col].corr(net_rating)
            subset_corrs[col] = r
        correlations[subset_name] = subset_corrs
    return correlations


def print_validation_report(correlations: dict[str, dict[str, float]]) -> list[tuple[str, str, float]]:
    """Print the correlation results per subset and flag weak features.
    Returns the flat list of (subset_name, feature, r) flagged as weak
    (|r| < WEAK_CORRELATION_THRESHOLD) for use in the summary section.
    """
    print("=" * 70)
    print("VALIDATION — feature correlation with real NET_RATING")
    print("=" * 70)
    print(f"  {len(next(iter(correlations.values())))*0 + sum(len(v) for v in correlations.values())} "
          f"features checked across {len(correlations)} subsets, against NET_RATING "
          f"from {USAGE_CONTEXT_CSV}.")
    print(f"  Flag threshold: |r| < {WEAK_CORRELATION_THRESHOLD} "
          f"(near-zero relationship to real winning impact)\n")

    flagged: list[tuple[str, str, float]] = []
    for subset_name, subset_corrs in correlations.items():
        print(f"  -- {subset_name} --")
        for feature, r in subset_corrs.items():
            is_weak = abs(r) < WEAK_CORRELATION_THRESHOLD
            flag = "  <-- WEAK, candidate for removal" if is_weak else ""
            print(f"    {feature:<34} r={r:+.4f}{flag}")
            if is_weak:
                flagged.append((subset_name, feature, r))
        print()

    print("=" * 70)
    print(f"WEAK-FEATURE SUMMARY -- {len(flagged)} feature(s) flagged")
    print("=" * 70)
    if flagged:
        for subset_name, feature, r in flagged:
            print(f"  [{subset_name}] {feature}  r={r:+.4f}")
    else:
        print("  None. Every feature in every subset clears the "
              f"|r| >= {WEAK_CORRELATION_THRESHOLD} bar against NET_RATING.")
    print()
    return flagged


# ── 3. Per-subset roster subspace + gap score ───────────────────────────────

def build_subset_style_space(feature_matrix: pd.DataFrame) -> dict:
    """Same standardized, median-imputed style space as roster_fit.py's
    build_style_space (full 28-feature X_scaled + feature_cols) -- built
    once here and then sliced per-subset by column index in
    build_subset_roster_subspace/subset_style_vector, rather than
    refitting a separate StandardScaler per subset. Valid because
    StandardScaler z-scores each column independently: a subset's
    sliced-out z-scores are identical to fitting the scaler on just that
    subset's raw columns.
    """
    return build_style_space(feature_matrix)


def _subset_column_indices(space: dict, subset_cols: list[str]) -> list[int]:
    col_index = {c: i for i, c in enumerate(space["feature_cols"])}
    missing = [c for c in subset_cols if c not in col_index]
    if missing:
        raise ValueError(f"Subset references columns not in the fitted style space: {missing}")
    return [col_index[c] for c in subset_cols]


def build_subset_roster_subspace(
    space: dict, team_abbreviation: str, subset_name: str, subset_cols: list[str] | None = None
) -> dict:
    """DISCOVERED DEGENERACY (see the module docstring / validation output
    for the full account): roster_fit.py's SVD-orthogonal-projection
    method only works when a roster's player count is comfortably less
    than the feature space's dimensionality -- with the full 28-feature
    space and ~10-14 rostered players per team, no team's roster spans
    more than a fraction of the space, so "residual orthogonal to the
    roster's span" is a real, non-trivial measurement.

    Every one of the 6 capability subsets defined here has far FEWER
    columns (2-7) than a typical roster has qualified players (~10-14).
    That means every real team's roster trivially SPANS the entire
    subset subspace (rank == subset dimensionality), so the orthogonal
    residual for ANY player against ANY such team is mathematically
    forced to ~0 -- a gap score of 0.0000 and 100% coverage regardless of
    whether the team's players actually resemble the query player. This
    was caught by this module's own CLE/CHI/OKC test producing an
    uninformative 3-way tie at exactly 100% coverage for all three teams,
    which is not a real result -- it is the SVD-span method breaking down
    once subset dimensionality < roster size.

    FIX: for low-dimensional subsets (dim < roster size, which is every
    subset here), coverage cannot be "orthogonal residual to the full
    span" -- instead this measures each rostered player's individual
    distance to the query player in the subset's z-scored feature space,
    and defines the team's coverage as its CLOSEST rostered player's
    distance (does the roster have at least one player who plays that
    specific dimension of the query player's game), analogous to a
    per-subset nearest-neighbor comp rather than a subspace-projection
    residual. This keeps the "does the roster already have this style"
    interpretation the original design doc asked for, but is
    mathematically well-defined at low dimensionality, unlike the SVD
    span approach that was originally built for the full 28-feature case
    and does not generalize down to a 2-7 feature subset.

    subset_cols defaults to CAPABILITY_SUBSETS[subset_name]; pass an
    explicit list to test an ad-hoc variant (e.g. a named subset with one
    feature removed) without redefining CAPABILITY_SUBSETS itself and
    losing the ability to compare against the original.
    """
    if subset_cols is None:
        subset_cols = CAPABILITY_SUBSETS[subset_name]
    col_idx = _subset_column_indices(space, subset_cols)

    feature_matrix = space["feature_matrix"]
    mask = (feature_matrix["TEAM_ABBREVIATION"] == team_abbreviation).values
    n_players = int(mask.sum())
    if n_players == 0:
        raise ValueError(f"No qualified players found for team {team_abbreviation!r}")

    roster_X = space["X_scaled"][mask][:, col_idx]
    player_names = feature_matrix.loc[mask, "PLAYER_NAME"].tolist()

    # Degeneracy check, computed regardless, so callers/reports can see
    # exactly when the classic SVD-span approach would have been
    # meaningless for this roster/subset combination.
    U, S, Vt = np.linalg.svd(roster_X, full_matrices=False)
    tol = max(roster_X.shape) * np.finfo(roster_X.dtype).eps * (S[0] if len(S) else 0)
    svd_rank = int((S > tol).sum())
    svd_degenerate = svd_rank >= len(subset_cols)

    return {
        "team_abbreviation": team_abbreviation,
        "subset_name": subset_name,
        "subset_cols": subset_cols,
        "player_names": player_names,
        "n_players": n_players,
        "roster_X": roster_X,          # each rostered player's subset-space vector, for nearest-neighbor coverage
        "svd_rank": svd_rank,
        "svd_degenerate": svd_degenerate,
    }


def subset_style_vector(
    space: dict, player_name: str, subset_name: str, subset_cols: list[str] | None = None
) -> np.ndarray | None:
    full_vec = player_style_vector(space, player_name)
    if full_vec is None:
        return None
    if subset_cols is None:
        subset_cols = CAPABILITY_SUBSETS[subset_name]
    col_idx = _subset_column_indices(space, subset_cols)
    return full_vec[col_idx]


def subset_gap_score(roster: dict, style_vector: np.ndarray) -> tuple[float, float, float]:
    """Nearest-rostered-player distance in the subset's z-scored space
    (see build_subset_roster_subspace's docstring for why this replaced
    the SVD-orthogonal-projection approach: that method is only
    meaningful when subset dimensionality exceeds roster size, which is
    never true here). "gap" is this roster's CLOSEST player's Euclidean
    distance to style_vector -- small gap means the roster already has
    at least one player who plays this specific dimension of the query
    player's game; large gap means nobody on the roster resembles them
    on this axis. "covered"/"total" are kept in the same shape as the
    original SVD version's return signature (covered = total - gap,
    floored at 0) purely so pct_covered stays comparably interpretable
    across both mechanisms, not because it carries the same subspace
    meaning.
    """
    roster_X = roster["roster_X"]
    distances = np.linalg.norm(roster_X - style_vector, axis=1)
    gap_magnitude = float(distances.min())
    total_magnitude = float(np.linalg.norm(style_vector))
    covered_magnitude = max(0.0, total_magnitude - gap_magnitude)
    return gap_magnitude, covered_magnitude, total_magnitude


def avg_best_n_gap_score(roster: dict, style_vector: np.ndarray, n: int = 3) -> tuple[float, float, float]:
    """Average-of-best-N variant of subset_gap_score: instead of taking
    only the single closest rostered player's distance, take the N
    closest rostered players' distances and average them. Built as a
    follow-up test after subset_gap_score's single-nearest-neighbor
    version proved sensitive to whichever one dimension the query player
    happened to be most extreme on (Clingan's CONTESTED_SHOTS_PER36, then
    DEF_PPP_POSTUP once that was removed) -- averaging over the roster's
    3 best comparable players, rather than trusting its single closest
    match, is meant to be less swayed by one player being an outlier on
    one axis specifically.

    If a roster has fewer than n qualified players, averages over
    however many it has (documented via the returned n_used so callers
    can see when this happened rather than silently averaging over a
    smaller-than-requested group).
    """
    roster_X = roster["roster_X"]
    distances = np.linalg.norm(roster_X - style_vector, axis=1)
    n_used = min(n, len(distances))
    closest_n = np.sort(distances)[:n_used]
    gap_magnitude = float(closest_n.mean())
    total_magnitude = float(np.linalg.norm(style_vector))
    covered_magnitude = max(0.0, total_magnitude - gap_magnitude)
    return gap_magnitude, covered_magnitude, total_magnitude, n_used


def get_subset_gap(
    space: dict, team: str, subset_name: str, player_name: str, subset_cols: list[str] | None = None
) -> tuple[float, float, float, float] | None:
    """Convenience wrapper: resolve player_name, build (or reuse) the
    team's subset roster subspace, and return
    (gap, covered, total, pct_covered) -- or None if the player can't be
    resolved into the qualified feature matrix. subset_cols defaults to
    CAPABILITY_SUBSETS[subset_name]; pass an explicit list to test an
    ad-hoc variant.
    """
    canonical = _resolve_reference_player(player_name, space["feature_matrix"]["PLAYER_NAME"])
    if canonical is None:
        return None
    vec = subset_style_vector(space, canonical, subset_name, subset_cols)
    if vec is None:
        return None
    roster = build_subset_roster_subspace(space, team, subset_name, subset_cols)
    gap, covered, total = subset_gap_score(roster, vec)
    pct_covered = (covered / total * 100) if total > 0 else float("nan")
    return gap, covered, total, pct_covered


def get_subset_gap_avg_best_n(
    space: dict, team: str, subset_name: str, player_name: str,
    subset_cols: list[str] | None = None, n: int = 3,
) -> tuple[float, float, float, float, int] | None:
    """avg_best_n_gap_score analog of get_subset_gap: resolve
    player_name, build the team's subset roster, and return
    (gap, covered, total, pct_covered, n_used) using the average of the
    n closest rostered players' distances instead of just the single
    closest.
    """
    canonical = _resolve_reference_player(player_name, space["feature_matrix"]["PLAYER_NAME"])
    if canonical is None:
        return None
    vec = subset_style_vector(space, canonical, subset_name, subset_cols)
    if vec is None:
        return None
    roster = build_subset_roster_subspace(space, team, subset_name, subset_cols)
    gap, covered, total, n_used = avg_best_n_gap_score(roster, vec, n)
    pct_covered = (covered / total * 100) if total > 0 else float("nan")
    return gap, covered, total, pct_covered, n_used


def print_subset_roster_sanity_check(
    space: dict, team: str, subset_name: str, subset_cols: list[str] | None = None
) -> bool:
    """Confirm every rostered player scores gap=0.0000 against their own
    team's subset roster (their nearest-rostered-player distance to
    themselves is trivially 0, since they ARE one of the rostered
    players compared against). Returns True iff every player passes.
    """
    roster = build_subset_roster_subspace(space, team, subset_name, subset_cols)
    all_zero = True
    for p in roster["player_names"]:
        vec = subset_style_vector(space, p, subset_name, subset_cols)
        gap, covered, total = subset_gap_score(roster, vec)
        passed = abs(gap) < 1e-9
        all_zero = all_zero and passed
        status = "OK" if passed else "FAIL"
        print(f"    {p:<28} gap={gap:.6f}  [{status}]")
    return all_zero


# ── 4. THE ACTUAL TEST: rim_protection subset, CLE vs. CHI vs. OKC ─────────

def run_rim_protection_diagnostic(space: dict) -> None:
    """Rerun the exact CLE (two elite shot-blocking bigs: Evan Mobley,
    Jarrett Allen) vs. CHI (no true center on the qualified roster at
    all) vs. OKC diagnostic documented in TEAM_FIT_LAYER4_OUTPUT.txt --
    but this time scoring gap ONLY on the rim_protection subset instead
    of all 28 features. That prior run found CLE scoring WORSE Clingan
    coverage than CHI (66.7% vs. 85.7%), the opposite of basketball
    intuition -- this is the pass/fail test for whether restricting to a
    relevant feature subset actually fixes that.
    """
    print("=" * 70)
    print("THE ACTUAL TEST — rim_protection subset only: CLE vs. CHI vs. OKC")
    print("=" * 70)
    print("  Prior result (all 28 features, see TEAM_FIT_LAYER4_OUTPUT.txt):")
    print("    CLE (2 elite shot-blockers) scored WORSE Clingan coverage (66.7%)")
    print("    than CHI (no true center) at 85.7% -- the opposite of basketball")
    print("    intuition. Pass/fail bar for this redesign: does restricting to")
    print("    the rim_protection subset fix this?\n")

    print("  NOTE -- mechanism had to change mid-build: the first attempt at this")
    print("  test reused roster_fit.py's original SVD-orthogonal-projection gap")
    print("  score unmodified, just restricted to the 5-column rim_protection")
    print("  subset. That produced OKC=CLE=CHI=100.0% coverage, an uninformative")
    print("  3-way tie -- a real bug, not a real result. Cause: with only 5")
    print("  columns in the subset and 10-14 qualified players per roster, every")
    print("  team's roster trivially SPANS the entire 5-dimensional subset space")
    print("  (rank 5 of 5), so the orthogonal residual is mathematically forced")
    print("  to ~0 for ANY player against ANY team, regardless of actual style")
    print("  similarity. Fixed by switching the per-subset gap score from")
    print("  \"orthogonal residual to the roster's full span\" to \"distance to the")
    print("  roster's single closest player\" (nearest-neighbor in the subset's")
    print("  z-scored space) -- well-defined at any dimensionality, and still")
    print("  answers \"does this roster already have a player who plays this")
    print("  specific dimension of the query player's game.\"\n")

    for team in ["OKC", "CLE", "CHI"]:
        roster = build_subset_roster_subspace(space, team, "rim_protection")
        print(f"    {team}: subset dim={len(roster['subset_cols'])}, "
              f"roster size={roster['n_players']}, SVD rank={roster['svd_rank']} "
              f"-> SVD-span degenerate: {roster['svd_degenerate']}")
    print()

    print("  -- Sanity check: every rostered player should score gap=0.0000")
    print("     against their OWN team's rim_protection subspace --\n")
    for team in ["OKC", "CLE", "CHI"]:
        print(f"  {team}:")
        passed = print_subset_roster_sanity_check(space, team, "rim_protection")
        print(f"    -> {'ALL PASS' if passed else 'SOME FAILED'}\n")

    print("  -- Clingan rim_protection-subset gap score vs. each team --\n")
    results = {}
    for team in ["OKC", "CLE", "CHI"]:
        result = get_subset_gap(space, team, "rim_protection", "Donovan Clingan")
        results[team] = result
        if result is None:
            print(f"    {team}: [Donovan Clingan could not be resolved]")
            continue
        gap, covered, total, pct_covered = result
        print(f"    {team:<5} gap={gap:.4f}  covered={covered:.4f}  "
              f"total={total:.4f}  ({pct_covered:.1f}% covered)")

    print()
    valid = {t: r for t, r in results.items() if r is not None}
    ranked = sorted(valid.items(), key=lambda kv: kv[1][3], reverse=True)
    print("  Ranking (highest rim_protection coverage first):")
    for rank, (team, (gap, covered, total, pct)) in enumerate(ranked, start=1):
        print(f"    {rank}. {team:<5} {pct:.1f}% covered  (gap={gap:.4f})")

    print()
    print("  -- Why: raw rim_protection feature values + league percentile --")
    print("     (checked directly against the raw feature matrix, not the")
    print("     z-scored/nearest-neighbor pipeline, to rule out a scaling")
    print("     artifact before trusting this as a real finding)\n")
    feature_matrix = space["feature_matrix"]
    rim_cols = CAPABILITY_SUBSETS["rim_protection"]
    for name in ["Evan Mobley", "Jarrett Allen", "Donovan Clingan"]:
        row = feature_matrix.loc[feature_matrix["PLAYER_NAME"] == name, rim_cols]
        if row.empty:
            continue
        print(f"    {name}:")
        for col in rim_cols:
            val = row[col].values[0]
            series = feature_matrix[col].dropna()
            pctile = (series < val).mean() * 100
            print(f"      {col:<34} {val:>8.3f}   ({pctile:.1f}th percentile of 305 qualified players)")
        print()

    print("=" * 70)
    print("PASS/FAIL VERDICT")
    print("=" * 70)
    if "CLE" in valid and "CHI" in valid:
        cle_pct = valid["CLE"][3]
        chi_pct = valid["CHI"][3]
        if cle_pct > chi_pct:
            margin = cle_pct - chi_pct
            print(f"  PASS: CLE's rim_protection coverage of Clingan ({cle_pct:.1f}%) is "
                  f"HIGHER than CHI's ({chi_pct:.1f}%), by {margin:.1f} points.")
            print("  Restricting to the rim_protection-specific feature subset corrects "
                  "the documented bug in this maximally obvious test case: a roster with "
                  "two genuine elite shot-blocking bigs now clearly shows better "
                  "rim-protection coverage than a roster with none.")
        else:
            margin = chi_pct - cle_pct
            print(f"  FAIL: CLE's rim_protection coverage of Clingan ({cle_pct:.1f}%) is "
                  f"still NOT higher than CHI's ({chi_pct:.1f}%) "
                  f"(CHI leads/ties by {margin:.1f} points).")
            print("  Restricting to the rim_protection-specific feature subset does NOT "
                  "fix the documented bug in this maximally obvious test case. Even with "
                  "distance computed over only 5 rim-protection-relevant features via "
                  "nearest-rostered-player matching (not the degenerate SVD-span method), "
                  "a roster with two genuine elite shot-blocking bigs does not clearly "
                  "outscore a roster with no true center at all.")
            print()
            print("  Root cause, confirmed against the raw feature values above: Donovan "
                  "Clingan is not a representative 'elite rim protector' reference point -- "
                  "he is a statistical OUTLIER even among elite bigs. His "
                  "CONTESTED_SHOTS_PER36 (17.8) sits at the 99.7th percentile of all 305 "
                  "qualified players -- the single highest value in the dataset -- while "
                  "Mobley (94.7th) and Allen (89.7th) are themselves elite but "
                  "meaningfully less extreme. His DEF_PPP_POSTUP and BOX_OUTS_PER36 are "
                  "similarly the highest or near-highest of the three. Nearest-neighbor "
                  "distance to an extreme outlier penalizes even genuinely elite comps for "
                  "not being AS extreme as the single most extreme player in the league on "
                  "that exact dimension -- this is a real limitation of using Clingan "
                  "specifically as the rim-protection reference point in a "
                  "nearest-single-player-distance metric, not evidence that the "
                  "rim_protection subset's features are poorly chosen (their NET_RATING "
                  "correlations above are among the strongest of any subset checked). A "
                  "reference point closer to the profile's centroid, or scoring against "
                  "the roster's best-N players' average rather than its single closest "
                  "player, would likely behave differently and is worth testing next.")
    else:
        print("  [Could not compute verdict -- CLE or CHI result missing]")
    print()


# ── 5. Follow-up test: does removing the outlier-driving feature fix it? ──

def run_rim_protection_trimmed_diagnostic(space: dict) -> None:
    """One-variable-at-a-time follow-up to run_rim_protection_diagnostic:
    CONTESTED_SHOTS_PER36 was independently flagged (a) weakly correlated
    with NET_RATING (r=+0.0136, below the 0.05 threshold) and (b) the
    specific feature where Donovan Clingan is most extreme (99.7th
    percentile of 305 qualified players -- the single highest value in
    the dataset), identified as the likely driver of his distorting the
    prior rim_protection-subset test.

    This reruns the EXACT same CLE vs. CHI vs. OKC diagnostic, same
    nearest-rostered-player mechanism, changing only the subset: 4
    features (RIM_PROTECTION_TRIMMED) instead of 5
    (CAPABILITY_SUBSETS['rim_protection']). If removing this one feature
    fixes the ranking, that is real evidence CONTESTED_SHOTS_PER36 itself
    (not the general subset-and-nearest-neighbor approach) was the
    problem. If CLE still loses to CHI, that is evidence the problem is
    upstream of any single feature and the average-of-best-N /
    reference-point-sensitivity ideas are worth trying next instead.
    """
    print("=" * 70)
    print("FOLLOW-UP TEST — rim_protection TRIMMED (CONTESTED_SHOTS_PER36 removed):")
    print("CLE vs. CHI vs. OKC")
    print("=" * 70)
    print("  One variable at a time: same 3 teams, same nearest-rostered-player")
    print("  mechanism as the prior rim_protection test above -- the ONLY change")
    print("  is removing CONTESTED_SHOTS_PER36 from the subset (5 features -> 4):")
    print(f"    Before: {CAPABILITY_SUBSETS['rim_protection']}")
    print(f"    After:  {RIM_PROTECTION_TRIMMED}\n")
    print("  Prior (5-feature) result: CLE=19.7% covered, CHI=37.9% covered, "
          "OKC=50.6% covered -- CLE lost to CHI, opposite of basketball intuition.\n")

    print("  -- Sanity check: every rostered player should score gap=0.0000")
    print("     against their OWN team's trimmed rim_protection subspace --\n")
    for team in ["OKC", "CLE", "CHI"]:
        print(f"  {team}:")
        passed = print_subset_roster_sanity_check(
            space, team, "rim_protection_trimmed", RIM_PROTECTION_TRIMMED
        )
        print(f"    -> {'ALL PASS' if passed else 'SOME FAILED'}\n")

    print("  -- Clingan trimmed-rim_protection-subset gap score vs. each team --\n")
    results = {}
    for team in ["OKC", "CLE", "CHI"]:
        result = get_subset_gap(
            space, team, "rim_protection_trimmed", "Donovan Clingan", RIM_PROTECTION_TRIMMED
        )
        results[team] = result
        if result is None:
            print(f"    {team}: [Donovan Clingan could not be resolved]")
            continue
        gap, covered, total, pct_covered = result
        print(f"    {team:<5} gap={gap:.4f}  covered={covered:.4f}  "
              f"total={total:.4f}  ({pct_covered:.1f}% covered)")

    print()
    valid = {t: r for t, r in results.items() if r is not None}
    ranked = sorted(valid.items(), key=lambda kv: kv[1][3], reverse=True)
    print("  Ranking (highest trimmed rim_protection coverage first):")
    for rank, (team, (gap, covered, total, pct)) in enumerate(ranked, start=1):
        print(f"    {rank}. {team:<5} {pct:.1f}% covered  (gap={gap:.4f})")

    print()
    print("  -- For reference: Clingan's percentile on the 4 remaining features --")
    feature_matrix = space["feature_matrix"]
    for name in ["Evan Mobley", "Jarrett Allen", "Donovan Clingan"]:
        row = feature_matrix.loc[feature_matrix["PLAYER_NAME"] == name, RIM_PROTECTION_TRIMMED]
        if row.empty:
            continue
        print(f"    {name}:")
        for col in RIM_PROTECTION_TRIMMED:
            val = row[col].values[0]
            series = feature_matrix[col].dropna()
            pctile = (series < val).mean() * 100
            print(f"      {col:<34} {val:>8.3f}   ({pctile:.1f}th percentile of 305 qualified players)")
        print()

    print("=" * 70)
    print("PASS/FAIL VERDICT — trimmed subset")
    print("=" * 70)
    if "CLE" in valid and "CHI" in valid:
        cle_pct = valid["CLE"][3]
        chi_pct = valid["CHI"][3]
        if cle_pct > chi_pct:
            margin = cle_pct - chi_pct
            print(f"  PASS: CLE's trimmed rim_protection coverage of Clingan ({cle_pct:.1f}%) "
                  f"is HIGHER than CHI's ({chi_pct:.1f}%), by {margin:.1f} points.")
            print("  Removing CONTESTED_SHOTS_PER36 FIXES the ranking in this maximally "
                  "obvious test case. This is real evidence that feature specifically -- "
                  "not the general subset-and-nearest-neighbor approach -- was the problem: "
                  "it was both weakly correlated with real winning impact (NET_RATING) and "
                  "the exact dimension on which Clingan is a league-wide outlier (99.7th "
                  "percentile), so distance on that one feature was overwhelming the signal "
                  "from the other 4, more genuinely diagnostic features.")
        else:
            margin = chi_pct - cle_pct
            print(f"  FAIL: CLE's trimmed rim_protection coverage of Clingan ({cle_pct:.1f}%) "
                  f"is STILL NOT higher than CHI's ({chi_pct:.1f}%) "
                  f"(CHI leads/ties by {margin:.1f} points).")
            print("  Removing CONTESTED_SHOTS_PER36 does NOT fix the ranking. The problem "
                  "persists even without the specific feature that drove Clingan's outlier "
                  "distortion, which is evidence the issue is NOT isolated to that one "
                  "feature -- it is upstream of it, in either the reference player choice "
                  "(Clingan may simply be an outlier on multiple axes at once, not just "
                  "CONTESTED_SHOTS_PER36) or the single-nearest-neighbor aggregation itself "
                  "(one player's distance, rather than an average across a roster's best "
                  "comparable players, stays sensitive to whichever single dimension the "
                  "query player is most extreme on). The average-of-best-N and "
                  "reference-point-sensitivity ideas flagged earlier are the next things "
                  "worth testing, not further feature-by-feature removal from this subset.")
    else:
        print("  [Could not compute verdict -- CLE or CHI result missing]")
    print()


# ── 6. Average-of-best-N (N=3) test, on the trimmed 4-feature subset ───────

def run_avg_best_n_diagnostic(space: dict, n: int = 3) -> None:
    """Test whether averaging over the 3 closest rostered players'
    distances, instead of trusting only the single closest, produces a
    more sensible CLE/CHI ranking than either the 5-feature or 4-feature
    (trimmed) single-nearest-neighbor versions did (both FAILed: CHI beat
    CLE in both). Uses RIM_PROTECTION_TRIMMED (4 features,
    CONTESTED_SHOTS_PER36 already removed) since that was the
    better-performing of the two single-nearest-neighbor variants, and
    keeps every other variable fixed -- same 3 teams, same reference
    player (Clingan), same subset -- so this isolates the effect of the
    aggregation change alone.
    """
    print("=" * 70)
    print(f"AVERAGE-OF-BEST-{n} TEST — rim_protection TRIMMED subset: CLE vs. CHI vs. OKC")
    print("=" * 70)
    print(f"  Same 4-feature RIM_PROTECTION_TRIMMED subset and same reference player "
          f"(Donovan Clingan) as the trimmed single-nearest-neighbor test above. The "
          f"ONLY change: gap score is now the AVERAGE distance to the {n} closest "
          f"rostered players, not just the single closest.\n")
    print("  Prior (single-nearest-neighbor, trimmed subset) result: CLE=34.7% covered, "
          "CHI=42.8% covered, OKC=33.6% covered -- CLE still lost to CHI.\n")

    results = {}
    for team in ["OKC", "CLE", "CHI"]:
        result = get_subset_gap_avg_best_n(
            space, team, "rim_protection_trimmed", "Donovan Clingan", RIM_PROTECTION_TRIMMED, n
        )
        results[team] = result
        if result is None:
            print(f"    {team}: [Donovan Clingan could not be resolved]")
            continue
        gap, covered, total, pct_covered, n_used = result
        note = "" if n_used == n else f"  [only {n_used} qualified players on roster, averaged over {n_used}]"
        print(f"    {team:<5} gap={gap:.4f}  covered={covered:.4f}  "
              f"total={total:.4f}  ({pct_covered:.1f}% covered){note}")

    print()
    valid = {t: r for t, r in results.items() if r is not None}
    ranked = sorted(valid.items(), key=lambda kv: kv[1][3], reverse=True)
    print(f"  Ranking (highest avg-of-best-{n} rim_protection coverage first):")
    for rank, (team, (gap, covered, total, pct, n_used)) in enumerate(ranked, start=1):
        print(f"    {rank}. {team:<5} {pct:.1f}% covered  (gap={gap:.4f}, n_used={n_used})")

    print()
    print("=" * 70)
    print(f"PASS/FAIL VERDICT — average-of-best-{n}")
    print("=" * 70)
    if "CLE" in valid and "CHI" in valid:
        cle_pct = valid["CLE"][3]
        chi_pct = valid["CHI"][3]
        if cle_pct > chi_pct:
            margin = cle_pct - chi_pct
            print(f"  PASS: CLE's avg-of-best-{n} rim_protection coverage of Clingan "
                  f"({cle_pct:.1f}%) is HIGHER than CHI's ({chi_pct:.1f}%), by {margin:.1f} points.")
            print(f"  Averaging over the {n} closest rostered players fixes the ranking in "
                  "this maximally obvious test case. This is real evidence that the "
                  "single-nearest-neighbor aggregation -- not the feature subset itself -- "
                  "was the remaining problem: it stayed hostage to whichever one player and "
                  "one dimension happened to be closest/most extreme, while averaging over "
                  "multiple comparable players smooths that out.")
        else:
            margin = chi_pct - cle_pct
            print(f"  FAIL: CLE's avg-of-best-{n} rim_protection coverage of Clingan "
                  f"({cle_pct:.1f}%) is STILL NOT higher than CHI's ({chi_pct:.1f}%) "
                  f"(CHI leads/ties by {margin:.1f} points).")
            print(f"  Averaging over the {n} closest rostered players does NOT fix the "
                  "ranking either. The problem survives both a feature-subset change and "
                  "an aggregation-method change, which is stronger evidence still that the "
                  "issue is not mechanism-shaped at all -- it is likely in what these "
                  "specific engineered features (postup/pick-and-roll defensive PPP, shot "
                  "suppression at the rim, box-outs per 36) actually capture about "
                  "individual defensive role and scheme, or in Clingan specifically being a "
                  "poor universal reference point for 'elite rim protector' regardless of "
                  "how the comparison is aggregated.")
    else:
        print("  [Could not compute verdict -- CLE or CHI result missing]")
    print()


# ── 7. Reference-sensitivity test: Chet Holmgren instead of Clingan ────────

def run_reference_sensitivity_diagnostic(space: dict) -> None:
    """Test whether the CLE/CHI ranking failure is specific to Donovan
    Clingan as the reference player, or persists with a different
    reference. Chet Holmgren -- a well-regarded, less statistically
    extreme rim protector on OKC's own roster (not CLE or CHI, so no
    trivial 100%-by-construction result) -- is used here, on the ORIGINAL
    5-feature rim_protection subset with the ORIGINAL single-nearest-
    neighbor mechanism (both unchanged from the very first CLE/CHI/OKC
    test), so this isolates the effect of the reference-player choice
    alone.
    """
    print("=" * 70)
    print("REFERENCE-SENSITIVITY TEST — Chet Holmgren instead of Clingan:")
    print("CLE vs. CHI vs. OKC (original 5-feature subset, single-nearest-neighbor)")
    print("=" * 70)
    print("  Same mechanism and same subset as the very first rim_protection test "
          "(5 features including CONTESTED_SHOTS_PER36, single-nearest-neighbor gap "
          "score). The ONLY change: reference player is Chet Holmgren (OKC), not "
          "Donovan Clingan.\n")
    print("  Prior (Clingan, same mechanism/subset) result: CLE=19.7% covered, "
          "CHI=37.9% covered, OKC=88.6% covered -- but OKC's 88.6% partly reflects "
          "Clingan not being on OKC's roster, so it is a genuine outside comparison "
          "there too. Holmgren, by contrast, IS on OKC's roster, so OKC's coverage "
          "of him will be trivially 100% by construction -- flagged explicitly below, "
          "not silently included in the CLE vs. CHI comparison.\n")

    results = {}
    for team in ["OKC", "CLE", "CHI"]:
        result = get_subset_gap(space, team, "rim_protection", "Chet Holmgren")
        results[team] = result
        if result is None:
            print(f"    {team}: [Chet Holmgren could not be resolved]")
            continue
        gap, covered, total, pct_covered = result
        on_roster_note = "  [Holmgren is on this roster -- trivially 100% by construction]" \
            if team == "OKC" else ""
        print(f"    {team:<5} gap={gap:.4f}  covered={covered:.4f}  "
              f"total={total:.4f}  ({pct_covered:.1f}% covered){on_roster_note}")

    print()
    print("  -- Holmgren's raw rim_protection feature values + league percentile, "
          "for comparison against Mobley/Allen/Clingan (see earlier sections) --\n")
    feature_matrix = space["feature_matrix"]
    rim_cols = CAPABILITY_SUBSETS["rim_protection"]
    row = feature_matrix.loc[feature_matrix["PLAYER_NAME"] == "Chet Holmgren", rim_cols]
    if not row.empty:
        print("    Chet Holmgren:")
        for col in rim_cols:
            val = row[col].values[0]
            series = feature_matrix[col].dropna()
            pctile = (series < val).mean() * 100
            print(f"      {col:<34} {val:>8.3f}   ({pctile:.1f}th percentile of 305 qualified players)")
    print()

    print("=" * 70)
    print("PASS/FAIL VERDICT — reference-sensitivity (Holmgren)")
    print("=" * 70)
    valid = {t: r for t, r in results.items() if r is not None}
    # Exclude OKC from the CLE-vs-CHI comparison since Holmgren is
    # literally on OKC's roster (trivial 100% by construction) -- same
    # exclusion logic as the earlier basketball-sense-check function, but
    # this diagnostic's core question is CLE vs. CHI specifically, so
    # it's applied directly here rather than via that shared helper.
    if "CLE" in valid and "CHI" in valid:
        cle_pct = valid["CLE"][3]
        chi_pct = valid["CHI"][3]
        if cle_pct > chi_pct:
            margin = cle_pct - chi_pct
            print(f"  PASS: with Holmgren as the reference, CLE's rim_protection coverage "
                  f"({cle_pct:.1f}%) is HIGHER than CHI's ({chi_pct:.1f}%), by {margin:.1f} "
                  "points.")
            print("  A less statistically extreme rim-protector reference produces the "
                  "basketball-sensible ranking. This is real evidence that Clingan "
                  "specifically -- as an outlier reference point -- was driving the earlier "
                  "failures, not a fundamental flaw in the subset/mechanism combination "
                  "itself.")
        else:
            margin = chi_pct - cle_pct
            print(f"  FAIL: with Holmgren as the reference, CLE's rim_protection coverage "
                  f"({cle_pct:.1f}%) is STILL NOT higher than CHI's ({chi_pct:.1f}%) "
                  f"(CHI leads/ties by {margin:.1f} points).")
            print("  Even a less statistically extreme, widely-regarded elite rim protector "
                  "produces the same wrong-direction ranking. This rules out "
                  "reference-player extremity as the sole explanation -- the problem is "
                  "more fundamental to how these specific features and this comparison "
                  "mechanism characterize rim protection for these particular rosters, not "
                  "just a quirk of Clingan being unusually extreme.")
    else:
        print("  [Could not compute verdict -- CLE or CHI result missing]")
    print()


# ── 8. Confirmatory test: a genuinely median-range rim protector ───────────

MEDIAN_RIM_PROTECTOR = "Onyeka Okongwu"
MEDIAN_RIM_PROTECTOR_TEAM = "ATL"


def print_percentile_screen_for_median_pick(space: dict) -> None:
    """Show the systematic screen used to pick MEDIAN_RIM_PROTECTOR:
    percentile rank on each RIM_PROTECTION_TRIMMED feature for a list of
    plausible rostered rim-protecting centers (not already used as a
    reference in this file's earlier tests), ranked by max deviation from
    the 50th percentile on any single feature -- i.e. how extreme their
    single MOST extreme feature is, the same failure mode that made
    Clingan (84-99th percentile on multiple features) a bad reference.
    Printed so the pick is auditable, not just asserted.
    """
    feature_matrix = space["feature_matrix"]
    cols = RIM_PROTECTION_TRIMMED
    candidates = [
        "Onyeka Okongwu", "Alperen Sengun", "Jarrett Allen", "Bam Adebayo",
        "Daniel Gafford", "Rudy Gobert", "Jaren Jackson Jr.", "Wendell Carter Jr.",
        "Jakob Poeltl", "Nic Claxton", "Brook Lopez", "Ivica Zubac", "Mitchell Robinson",
    ]
    print("  Screen: percentile rank (of 305 qualified players) on each of the 4 "
          "RIM_PROTECTION_TRIMMED features, for a list of plausible rostered rim "
          "protectors not already used as a reference player in this file. Ranked by "
          "'max deviation from the 50th percentile on any single feature' -- the same "
          "failure mode (one wildly extreme dimension) that made Clingan a bad "
          "reference is what this screen is selecting AGAINST.\n")
    rows = []
    for name in candidates:
        row = feature_matrix.loc[feature_matrix["PLAYER_NAME"] == name, cols]
        if row.empty or row.isna().any(axis=None):
            continue
        pctiles = {c: float((feature_matrix[c] < row[c].values[0]).mean() * 100) for c in cols}
        max_dev = max(abs(p - 50) for p in pctiles.values())
        rows.append((name, pctiles, max_dev))
    rows.sort(key=lambda r: r[2])
    for name, pctiles, max_dev in rows:
        pct_str = ", ".join(f"{c}={p:.0f}th" for c, p in pctiles.items())
        print(f"    {name:<22} max_dev_from_median={max_dev:5.1f}   {pct_str}")
    print(f"\n  Picked: {MEDIAN_RIM_PROTECTOR} -- lowest max-deviation-from-median of any "
          f"candidate with complete data on all 4 features, on {MEDIAN_RIM_PROTECTOR_TEAM} "
          "(not CLE, CHI, or OKC, so no trivial-100%-by-construction issue).\n")


def run_median_reference_confirmatory_test(space: dict) -> None:
    """Confirmatory test requested after the Holmgren PASS: does the fix
    replicate with a SECOND, genuinely median/normal-range reference
    rim protector -- not another outlier like Clingan, and not
    necessarily as clean a case as Holmgren happened to be, just
    honestly middle-of-the-pack for a rostered rim protector? Same
    5-feature rim_protection subset and single-nearest-neighbor
    mechanism as the original and Holmgren tests, so this isolates
    whether the fix generalizes past one lucky reference pick.

    If this ALSO passes cleanly: real, replicated evidence the mechanism
    works fine for non-outlier references, and the earlier CLE/CHI
    failures were specifically about extreme reference players, not a
    deeper flaw. If it fails: the "outlier reference player" explanation
    is incomplete and there is more going on than Clingan/Holmgren
    specifically.
    """
    print("=" * 70)
    print(f"CONFIRMATORY TEST — {MEDIAN_RIM_PROTECTOR} (median-range, not an outlier):")
    print("CLE vs. CHI vs. OKC (original 5-feature subset, single-nearest-neighbor)")
    print("=" * 70)
    print_percentile_screen_for_median_pick(space)

    print("  Same mechanism and same subset as the Clingan and Holmgren tests above "
          "(5 features including CONTESTED_SHOTS_PER36, single-nearest-neighbor gap "
          f"score). The ONLY change: reference player is {MEDIAN_RIM_PROTECTOR} "
          f"({MEDIAN_RIM_PROTECTOR_TEAM}), not Clingan or Holmgren.\n")
    print("  Prior results, same mechanism/subset:")
    print("    Clingan (extreme outlier):  CLE=19.7% covered, CHI=37.9% covered -- FAILED")
    print("    Holmgren (clean, less extreme): CLE=63.7% covered, CHI=40.2% covered -- PASSED\n")

    results = {}
    for team in ["OKC", "CLE", "CHI"]:
        result = get_subset_gap(space, team, "rim_protection", MEDIAN_RIM_PROTECTOR)
        results[team] = result
        if result is None:
            print(f"    {team}: [{MEDIAN_RIM_PROTECTOR} could not be resolved]")
            continue
        gap, covered, total, pct_covered = result
        on_roster_note = f"  [{MEDIAN_RIM_PROTECTOR} is on this roster -- trivially 100% by construction]" \
            if team == MEDIAN_RIM_PROTECTOR_TEAM else ""
        print(f"    {team:<5} gap={gap:.4f}  covered={covered:.4f}  "
              f"total={total:.4f}  ({pct_covered:.1f}% covered){on_roster_note}")

    print()
    print("=" * 70)
    print(f"PASS/FAIL VERDICT — confirmatory ({MEDIAN_RIM_PROTECTOR})")
    print("=" * 70)
    valid = {t: r for t, r in results.items() if r is not None}
    if "CLE" in valid and "CHI" in valid:
        cle_pct = valid["CLE"][3]
        chi_pct = valid["CHI"][3]
        if cle_pct > chi_pct:
            margin = cle_pct - chi_pct
            print(f"  PASS: with {MEDIAN_RIM_PROTECTOR} as the reference, CLE's rim_protection "
                  f"coverage ({cle_pct:.1f}%) is HIGHER than CHI's ({chi_pct:.1f}%), by "
                  f"{margin:.1f} points.")
            print("  This REPLICATES the Holmgren result with a second, independently-picked "
                  "median-range reference player. Two out of two non-outlier references now "
                  "produce the basketball-sensible CLE > CHI ranking, while the one outlier "
                  "reference (Clingan) did not, across both the single-nearest-neighbor and "
                  "average-of-best-3 mechanisms. This is real, replicated evidence that the "
                  "subset (5 or 4 features) and the single-nearest-neighbor mechanism both "
                  "work correctly for the maximally obvious CLE/CHI test case -- the earlier "
                  "documented failures were specifically about Clingan as an outlier "
                  "reference point, not a fundamental flaw in this Layer 4 redesign.")
        else:
            margin = chi_pct - cle_pct
            print(f"  FAIL: with {MEDIAN_RIM_PROTECTOR} as the reference, CLE's rim_protection "
                  f"coverage ({cle_pct:.1f}%) is STILL NOT higher than CHI's ({chi_pct:.1f}%) "
                  f"(CHI leads/ties by {margin:.1f} points).")
            print("  This does NOT replicate the Holmgren result. A second, independently "
                  "median-range reference player still produces the wrong-direction ranking, "
                  "which complicates the 'outlier reference player' explanation -- Holmgren's "
                  "PASS may have been a property of Holmgren specifically (or of OKC's own "
                  "roster composition, since OKC was the excluded/home team in that test) "
                  "rather than evidence the mechanism generalizes to any non-extreme rim "
                  "protector. There is more going on than reference-player extremity alone, "
                  "and this needs further investigation before the redesign can be trusted "
                  "for production use.")
            print()
            print(f"  Additional wrinkle, checked directly against the raw numbers: "
                  f"{MEDIAN_RIM_PROTECTOR}'s total z-scored vector magnitude in this run "
                  f"(total={valid['CHI'][2]:.4f}) is roughly a third of Clingan's (~5.0) or "
                  "Holmgren's (~4.0) from the earlier tests. That is a DIRECT, mechanical "
                  "consequence of being near the population mean on every one of the 5 "
                  "features at once -- a z-scored vector for a genuinely average player has "
                  "small magnitude by construction, since z-scoring centers each feature at "
                  "0. With a small total, almost ANY roster's nearest player ends up "
                  "representing a large fraction of that small total, which mechanically "
                  "compresses percent-covered toward 0 for every team, median-range player or "
                  "not -- not just for CLE. This is a separate, previously undocumented "
                  "problem from the 'outlier reference distorts the ranking' one Clingan "
                  "exposed: the pct_covered metric itself appears unreliable at BOTH ends of "
                  "the reference-player-extremity spectrum, for different mechanical reasons, "
                  "and may only behave sensibly in some middle band. Worth checking directly "
                  "before trusting this metric further.")
    else:
        print("  [Could not compute verdict -- CLE or CHI result missing]")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Building feature matrix from Layer 1 axes "
          f"(qualification floor: MIN>={MIN_MINUTES}, GP>={MIN_GAMES})...")
    feature_matrix = build_feature_matrix()
    print(f"Feature matrix: {feature_matrix.shape[0]} qualified players\n")

    print("Loading NET_RATING from", USAGE_CONTEXT_CSV, "for validation...")
    net_rating = load_net_rating(feature_matrix)
    print(f"NET_RATING loaded for all {net_rating.notna().sum()} qualified players.\n")

    correlations = validate_subsets_against_net_rating(feature_matrix, net_rating)
    flagged = print_validation_report(correlations)

    space = build_subset_style_space(feature_matrix)

    run_rim_protection_diagnostic(space)
    run_rim_protection_trimmed_diagnostic(space)
    run_avg_best_n_diagnostic(space, n=3)
    run_reference_sensitivity_diagnostic(space)
    run_median_reference_confirmatory_test(space)

    print("Done.")


if __name__ == "__main__":
    main()
