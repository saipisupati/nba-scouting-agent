"""
Percentile-space redesign of Layer 4's coverage metric, built in response
to the middle-band diagnostic (layer4_middleband.py) rejecting "pick a
better-behaved reference player" as a fix for pct_covered's unreliability:
2 of 7 tested points passed the CLE-vs-CHI basketball-sense check, and
they didn't cluster by z-scored vector magnitude (2.75 and 3.99 passed,
separated by two failing points, 3.42 and 4.16, in between). That result
means the defect is in normalizing nearest-neighbor distance by the
reference player's own raw z-scored magnitude -- a quantity whose scale
is not comparable across features of different variance and ties the
score to how statistically extreme that specific player happens to be.

Fix tested here: convert every feature to a percentile rank (0-100
across the 305 qualified players) BEFORE computing any distance or
coverage number, using the exact
`(feature_matrix[col] < val).mean() * 100` formula layer4_subsets.py
already uses four times over for its own percentile printouts -- not a
new statistical method, just applying that established, already-trusted
transform to the whole feature matrix instead of one value at a time,
and using percentile-space distance in place of z-scored distance.

Percentile rank is bounded (0-100) and comparable by construction across
features regardless of each feature's raw variance, and matches this
project's existing trust-vocabulary: signature_play_type already ranks
players by PERCENTILE, and every earlier Layer 4 diagnostic already
explains its findings in percentile terms because that's the unit a
scout can act on ("this roster's closest comp sits 35 percentile points
below the query player"), unlike a z-scored Euclidean ratio.

This module reruns the EXACT SAME validation gate as
layer4_middleband.py -- same 7 reference players spanning the prior
magnitude spectrum, same RIM_PROTECTION_TRIMMED subset, same CLE/CHI/OKC
basketball-sense check -- so the percentile-space redesign is held to
the identical bar the z-scored version failed, not a looser one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_vector import MIN_GAMES, MIN_MINUTES, _resolve_reference_player, build_feature_matrix
from layer4_subsets import RIM_PROTECTION_TRIMMED

TEAMS = ["OKC", "CLE", "CHI"]

# The exact 7 reference players layer4_middleband.py tested, spanning its
# full documented magnitude spectrum from near-zero to outlier -- rerun
# here unchanged so the percentile-space result is directly comparable
# point-for-point against the z-scored result it's meant to replace.
SPECTRUM_PLAYERS = [
    "Onyeka Okongwu",
    "Nique Clifford",
    "John Konchar",
    "Anthony Gill",
    "Chet Holmgren",
    "Jusuf Nurkić",
    "Donovan Clingan",
]

PRIOR_ZSCORE_VERDICT = {
    "Onyeka Okongwu": "FAIL",
    "Nique Clifford": "FAIL",
    "John Konchar": "PASS",
    "Anthony Gill": "FAIL",
    "Chet Holmgren": "PASS",
    "Jusuf Nurkić": "FAIL",
    "Donovan Clingan": "FAIL",
}


# ── percentile transform ─────────────────────────────────────────────────

def build_percentile_matrix(feature_matrix: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert each of `cols` to a 0-100 percentile rank across all rows
    of feature_matrix, using the same `(series < val).mean() * 100`
    formula layer4_subsets.py already uses for its printed percentile
    diagnostics. Returns a new DataFrame (PLAYER_NAME/TEAM_ABBREVIATION
    plus the percentile-transformed columns), leaving feature_matrix
    itself untouched -- same non-mutating convention as
    feature_vector.py/layer4_subsets.py's median-imputation-on-a-copy
    disclosure.

    NaN feature values (player didn't qualify for that specific
    sub-metric) stay NaN after transform -- `series < val` is False for
    every comparison against a NaN val's row, and pandas' `.mean()`
    already excludes NaN rows within `series`, but the percentile for a
    NaN val itself is computed as NaN explicitly below so a missing
    feature never silently reports a fabricated percentile.
    """
    result = feature_matrix[["PLAYER_NAME", "TEAM_ABBREVIATION"]].copy()
    for col in cols:
        series = feature_matrix[col]
        pct_col = series.apply(
            lambda val: float((series < val).mean() * 100) if pd.notna(val) else np.nan
        )
        result[col] = pct_col
    return result


def player_percentile_vector(pct_matrix: pd.DataFrame, player_name: str, cols: list[str]) -> np.ndarray | None:
    row = pct_matrix.loc[pct_matrix["PLAYER_NAME"] == player_name, cols]
    if row.empty:
        return None
    vec = row.values[0].astype(float)
    if np.isnan(vec).any():
        return None
    return vec


# ── roster subspace + coverage, in percentile space ─────────────────────

def build_percentile_roster(pct_matrix: pd.DataFrame, team: str, cols: list[str]) -> dict:
    """Same nearest-rostered-player structure as
    layer4_subsets.build_subset_roster_subspace, but operating on
    percentile-space vectors instead of z-scored ones -- no SVD-span
    degeneracy check needed here since this module never uses the SVD
    orthogonal-projection method at all (that method, and its
    degeneracy at low subset dimensionality, is what
    layer4_subsets.py's own docstring already replaced with
    nearest-neighbor distance; this module inherits that replacement
    directly rather than reintroducing the degenerate method).
    """
    mask = (pct_matrix["TEAM_ABBREVIATION"] == team).values
    n_players = int(mask.sum())
    if n_players == 0:
        raise ValueError(f"No qualified players found for team {team!r}")

    roster_rows = pct_matrix.loc[mask, cols]
    complete = ~roster_rows.isna().any(axis=1)
    roster_pct = roster_rows[complete].values.astype(float)
    player_names = pct_matrix.loc[mask, "PLAYER_NAME"][complete].tolist()

    return {
        "team": team,
        "player_names": player_names,
        "n_players": len(player_names),
        "roster_pct": roster_pct,
    }


def percentile_coverage(roster: dict, player_vec: np.ndarray) -> tuple[float, float]:
    """Nearest-rostered-player distance in percentile space (each axis
    0-100, so max possible per-axis gap is 100, unlike z-scored space
    where a feature's spread sets its own scale). Returns
    (nearest_distance, pct_covered) where pct_covered is expressed
    relative to the maximum possible distance in this space
    (100 * sqrt(n_features)), NOT relative to the reference player's own
    vector magnitude -- this is the specific change from the z-scored
    version's pct_covered, which is what tied the old metric to how
    extreme the reference player happened to be. A player near the
    percentile-space origin (all near-0th-percentile) and a player at
    the far corner (all near-100th-percentile) are now scored on the
    SAME fixed scale, not two different ones.
    """
    roster_pct = roster["roster_pct"]
    distances = np.linalg.norm(roster_pct - player_vec, axis=1)
    nearest = float(distances.min())
    max_possible = 100.0 * np.sqrt(len(player_vec))
    pct_covered = (1 - nearest / max_possible) * 100
    return nearest, pct_covered


def get_percentile_coverage(
    pct_matrix: pd.DataFrame, team: str, player_name: str, cols: list[str]
) -> tuple[float, float] | None:
    canonical = _resolve_reference_player(player_name, pct_matrix["PLAYER_NAME"])
    if canonical is None:
        return None
    vec = player_percentile_vector(pct_matrix, canonical, cols)
    if vec is None:
        return None
    roster = build_percentile_roster(pct_matrix, team, cols)
    return percentile_coverage(roster, vec)


# ── validation gate: rerun the exact middle-band spectrum ───────────────

def run_spectrum_validation(feature_matrix: pd.DataFrame) -> None:
    cols = RIM_PROTECTION_TRIMMED
    pct_matrix = build_percentile_matrix(feature_matrix, cols)

    print("=" * 70)
    print("PERCENTILE-SPACE VALIDATION — same 7-point spectrum as layer4_middleband.py")
    print("=" * 70)
    print(f"  Subset: {cols}")
    print("  Same reference players, same CLE/CHI/OKC test, same pass bar")
    print("  (CLE > CHI on rim-protection coverage) as the z-scored diagnostic")
    print("  this is meant to replace. Coverage is now nearest-rostered-player")
    print("  percentile-space distance, expressed as a fraction of the maximum")
    print("  possible distance in that space -- NOT normalized by the reference")
    print("  player's own vector magnitude, which is the specific defect the")
    print("  middle-band test traced the z-scored failures to.\n")

    spectrum = []
    for name in SPECTRUM_PLAYERS:
        print("-" * 70)
        results = {}
        for team in TEAMS:
            r = get_percentile_coverage(pct_matrix, team, name, cols)
            results[team] = r
        valid = {t: r for t, r in results.items() if r is not None}

        if len(valid) < len(TEAMS):
            missing = [t for t in TEAMS if t not in valid]
            print(f"{name}  [could not compute for: {missing}]")
            continue

        print(f"{name}")
        for team in TEAMS:
            nearest, pct = valid[team]
            print(f"    {team:<5} nearest_dist={nearest:6.2f}  ({pct:.1f}% covered)")

        cle_pct, chi_pct = valid["CLE"][1], valid["CHI"][1]
        passed = cle_pct > chi_pct
        verdict = "PASS" if passed else "FAIL"
        margin = abs(cle_pct - chi_pct)
        prior = PRIOR_ZSCORE_VERDICT.get(name, "?")
        changed = "" if verdict == prior else f"  [CHANGED from z-scored: was {prior}]"
        print(f"  {verdict}: CLE {'>' if passed else '<='} CHI ({cle_pct:.1f}% vs {chi_pct:.1f}%, "
              f"margin {margin:.1f} pts)  [prior z-scored verdict: {prior}]{changed}")
        spectrum.append((name, verdict, prior))

    print()
    print("=" * 70)
    print("SUMMARY — percentile-space vs. prior z-scored verdicts")
    print("=" * 70)
    passes = sum(1 for _, v, _ in spectrum if v == "PASS")
    total = len(spectrum)
    for name, verdict, prior in spectrum:
        flag = "" if verdict == prior else "  <-- changed"
        print(f"    {name:<20} percentile-space={verdict:<4}  z-scored={prior:<4}{flag}")
    print(f"\n  {passes} of {total} points PASS in percentile space "
          f"(z-scored space: {sum(1 for _,p in PRIOR_ZSCORE_VERDICT.items() if p=='PASS')} of "
          f"{len(PRIOR_ZSCORE_VERDICT)}).")

    if passes == total:
        print("  ALL points now PASS. This is strong evidence the percentile-space "
              "redesign fixes the reliability problem the z-scored middle-band test "
              "found -- every reference player across the full documented magnitude "
              "spectrum now produces the basketball-sensible CLE > CHI ranking.")
    elif passes > sum(1 for p in PRIOR_ZSCORE_VERDICT.values() if p == "PASS"):
        print("  More points PASS than under the z-scored metric, but not all -- a "
              "real improvement, not a full fix. The remaining failures should be "
              "inspected individually (raw percentile values, which specific team's "
              "nearest player is closest) before this metric is trusted for "
              "production use.")
    else:
        print("  NOT an improvement over the z-scored metric. Percentile-space "
              "normalization does not fix the underlying reliability problem -- the "
              "issue is not (only) about z-scored magnitude sensitivity, and a "
              "different mechanism (not just a different feature transform) is "
              "needed.")
    print()


def main() -> None:
    print("Building feature matrix from Layer 1 axes "
          f"(qualification floor: MIN>={MIN_MINUTES}, GP>={MIN_GAMES})...")
    feature_matrix = build_feature_matrix()
    print(f"Feature matrix: {feature_matrix.shape[0]} qualified players\n")

    run_spectrum_validation(feature_matrix)
    print("Done.")


if __name__ == "__main__":
    main()
