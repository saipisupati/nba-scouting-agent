"""
Follow-up diagnostic to layer4_subsets.py's confirmatory test
(LAYER4_SUBSET_REDESIGN_OUTPUT.txt): that run flagged, but did not test,
the hypothesis that pct_covered is unreliable at BOTH extremes of
reference-player z-scored vector magnitude for different mechanical
reasons --

  - an OUTLIER reference (Clingan, total ~5.0) distorts nearest-neighbor
    distance: distance to an extreme point overwhelms genuinely elite
    comps for merely being less extreme.
  - a NEAR-MEDIAN reference (Okongwu, total ~1.5) mechanically compresses
    pct_covered toward 0 for every team, because a z-scored vector near
    the population mean has small magnitude by construction (z-scoring
    centers each feature at 0), so almost any nearest player's distance
    is a large fraction of that small total.

This module tests whether a genuine MIDDLE BAND exists: reference
players whose total z-scored magnitude sits between the near-zero and
outlier extremes, where pct_covered might behave sensibly. Same
rim_protection subset (RIM_PROTECTION_TRIMMED, 4 features,
CONTESTED_SHOTS_PER36 already excluded per layer4_subsets.py's finding),
same CLE/CHI/OKC basketball-sense test, same single-nearest-neighbor
mechanism -- the only new thing is deliberately screening for and testing
multiple reference players across the magnitude spectrum, instead of
picking one player per test as the prior diagnostics did.

This is a diagnosis, not a redesign: the goal is to find out whether
pct_covered is salvageable in some magnitude band, not to invent a new
metric. If no band works, that is itself the answer -- Layer 4 needs a
different metric, not just a better-chosen reference player.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_vector import MIN_GAMES, MIN_MINUTES, build_feature_matrix
from layer4_subsets import (
    RIM_PROTECTION_TRIMMED,
    build_subset_style_space,
    get_subset_gap,
    subset_style_vector,
)

TEAMS = ["OKC", "CLE", "CHI"]

# Prior results, for direct comparison in the printed report.
PRIOR_RESULTS = {
    "Donovan Clingan": {"CLE": 34.7, "CHI": 42.8, "OKC": 33.6, "total_mag": 5.0, "verdict": "FAIL (outlier)"},
    "Chet Holmgren":   {"CLE": 63.7, "CHI": 40.2, "OKC": None, "total_mag": 3.99, "verdict": "PASS"},
    "Onyeka Okongwu":  {"CLE": 0.0,  "CHI": 5.5,  "OKC": 0.0,  "total_mag": 1.46, "verdict": "FAIL (near-zero)"},
}


def total_magnitude(space: dict, player_name: str) -> float | None:
    vec = subset_style_vector(space, player_name, "rim_protection_trimmed", RIM_PROTECTION_TRIMMED)
    if vec is None:
        return None
    return float(np.linalg.norm(vec))


def screen_candidates_by_magnitude(space: dict) -> list[tuple[str, float]]:
    """Compute RIM_PROTECTION_TRIMMED total z-scored magnitude for every
    qualified player, so candidates can be picked to deliberately span
    the range between Okongwu's ~1.5 and Clingan's ~5.0, rather than
    guessing at names. Excludes players already used as a reference in
    prior tests (Clingan, Holmgren, Okongwu) and excludes CLE/CHI/OKC
    rostered players (a rostered player scores trivially against their
    own team, contaminating the CLE-vs-CHI comparison the same way
    Holmgren-on-OKC did in the reference-sensitivity test).
    """
    feature_matrix = space["feature_matrix"]
    already_used = {"Donovan Clingan", "Chet Holmgren", "Onyeka Okongwu"}
    excluded_teams = set(TEAMS)

    rows = []
    for _, row in feature_matrix.iterrows():
        name = row["PLAYER_NAME"]
        if name in already_used or row["TEAM_ABBREVIATION"] in excluded_teams:
            continue
        vec = subset_style_vector(space, name, "rim_protection_trimmed", RIM_PROTECTION_TRIMMED)
        if vec is None or np.isnan(vec).any():
            continue
        rows.append((name, float(np.linalg.norm(vec))))
    rows.sort(key=lambda r: r[1])
    return rows


def print_magnitude_screen(space: dict) -> list[str]:
    """Print the full magnitude spectrum and pick 4 candidates that
    deliberately span between the two failed extremes (~1.5 and ~5.0):
    evenly-spaced target magnitudes at roughly 2.0, 2.75, 3.5, 4.25 --
    the nearest available real player to each target, so the middle-band
    test isn't just re-testing near-zero or near-outlier cases under a
    different name.
    """
    rows = screen_candidates_by_magnitude(space)
    print("=" * 70)
    print("MAGNITUDE SCREEN — RIM_PROTECTION_TRIMMED total z-scored magnitude")
    print("=" * 70)
    print(f"  {len(rows)} candidates screened (qualified players, excluding prior")
    print("  references and CLE/CHI/OKC rostered players).")
    print(f"  Range: {rows[0][1]:.3f} ({rows[0][0]}) to {rows[-1][1]:.3f} ({rows[-1][0]})")
    print(f"  Prior failed extremes: Okongwu=1.462 (near-zero, FAIL), "
          f"Clingan=~5.0 (outlier, FAIL). Holmgren=3.99 (PASS, unreplicated).\n")

    targets = [2.0, 2.75, 3.5, 4.25]
    picked = []
    for target in targets:
        closest = min(rows, key=lambda r: abs(r[1] - target))
        if closest[0] not in picked:
            picked.append(closest[0])
    print(f"  Middle-band candidates picked (nearest real player to targets "
          f"{targets}, spanning between the two failed extremes):")
    for name in picked:
        mag = dict(rows)[name]
        print(f"    {name:<28} total_mag={mag:.3f}")
    print()
    return picked


def run_middleband_test(space: dict, player_name: str) -> dict:
    results = {}
    for team in TEAMS:
        result = get_subset_gap(space, team, "rim_protection_trimmed", player_name, RIM_PROTECTION_TRIMMED)
        results[team] = result
    return results


def print_middleband_report(space: dict, player_name: str) -> tuple[float | None, bool | None]:
    print("-" * 70)
    mag = total_magnitude(space, player_name)
    print(f"{player_name}  (total_mag={mag:.3f})" if mag is not None else f"{player_name}  (could not resolve)")
    print("-" * 70)
    if mag is None:
        return None, None

    results = run_middleband_test(space, player_name)
    valid = {t: r for t, r in results.items() if r is not None}
    for team in TEAMS:
        r = results[team]
        if r is None:
            print(f"    {team:<5} [could not compute]")
            continue
        gap, covered, total, pct = r
        print(f"    {team:<5} gap={gap:.4f}  covered={covered:.4f}  total={total:.4f}  ({pct:.1f}% covered)")

    if "CLE" not in valid or "CHI" not in valid:
        print("  [insufficient data for CLE-vs-CHI verdict]\n")
        return mag, None

    cle_pct, chi_pct = valid["CLE"][3], valid["CHI"][3]
    passed = cle_pct > chi_pct
    verdict = "PASS" if passed else "FAIL"
    margin = abs(cle_pct - chi_pct)
    print(f"  {verdict}: CLE {'>' if passed else '<='} CHI on rim-protection coverage "
          f"({cle_pct:.1f}% vs {chi_pct:.1f}%, margin {margin:.1f} pts)")
    print()
    return mag, passed


def main() -> None:
    print("Building feature matrix from Layer 1 axes "
          f"(qualification floor: MIN>={MIN_MINUTES}, GP>={MIN_GAMES})...")
    feature_matrix = build_feature_matrix()
    print(f"Feature matrix: {feature_matrix.shape[0]} qualified players\n")

    space = build_subset_style_space(feature_matrix)

    print("=" * 70)
    print("MIDDLE-BAND DIAGNOSTIC")
    print("=" * 70)
    print("  Question: does pct_covered behave sensibly (CLE > CHI on a")
    print("  rim-protection reference) for SOME magnitude band between the")
    print("  two documented failures (near-zero ~1.5, outlier ~5.0)? Same")
    print("  RIM_PROTECTION_TRIMMED subset, same single-nearest-neighbor")
    print("  mechanism, same CLE/CHI/OKC test as every prior diagnostic in")
    print("  layer4_subsets.py -- only the reference player changes.\n")

    print("  Prior results for context:")
    for name, r in PRIOR_RESULTS.items():
        okc_str = f"{r['OKC']:.1f}%" if r["OKC"] is not None else "n/a (on roster)"
        print(f"    {name:<20} total_mag={r['total_mag']:.2f}  CLE={r['CLE']:.1f}%  "
              f"CHI={r['CHI']:.1f}%  OKC={okc_str}  -> {r['verdict']}")
    print()

    candidates = print_magnitude_screen(space)

    spectrum: list[tuple[float, bool | None, str]] = []
    for name, mag in PRIOR_RESULTS.items():
        r = PRIOR_RESULTS[name]
        passed = r["CLE"] > r["CHI"]
        spectrum.append((r["total_mag"], passed, name))

    for name in candidates:
        mag, passed = print_middleband_report(space, name)
        if mag is not None:
            spectrum.append((mag, passed, name))

    spectrum.sort(key=lambda r: r[0])
    print("=" * 70)
    print("FULL MAGNITUDE SPECTRUM — CLE-vs-CHI verdict at each tested point")
    print("=" * 70)
    for mag, passed, name in spectrum:
        verdict = "PASS" if passed else ("FAIL" if passed is False else "n/a")
        print(f"    mag={mag:5.2f}   {verdict:<4}   {name}")
    print()

    passes = [m for m, p, n in spectrum if p is True]
    fails = [m for m, p, n in spectrum if p is False]
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if len(passes) >= 2:
        print(f"  {len(passes)} of {len(passes) + len(fails)} tested points PASS. "
              f"PASS magnitudes: {sorted(round(m, 2) for m in passes)}. "
              f"FAIL magnitudes: {sorted(round(m, 2) for m in fails)}.")
        print("  If PASS points cluster together and FAIL points sit outside that "
              "cluster, that supports a real middle band where pct_covered is "
              "trustworthy -- state that band explicitly before using this metric "
              "for any other reference player.")
        print("  If PASS/FAIL are interleaved with no clean separation by magnitude, "
              "magnitude alone does not explain the failures, and pct_covered should "
              "be treated as unreliable in general, not just at the two tested "
              "extremes -- a different metric is needed, not a restricted range.")
    else:
        print(f"  Only {len(passes)} of {len(passes) + len(fails)} tested points PASS "
              "(at most the original unreplicated Holmgren case). No middle band "
              "was found where pct_covered reliably produces the basketball-sensible "
              "CLE > CHI ranking. This is evidence pct_covered is not a trustworthy "
              "metric at any magnitude tested so far -- the fix belongs in the metric "
              "itself (e.g. a percentile-rank-based coverage score, or raw untransformed "
              "distance without a magnitude-relative percentage), not in picking a "
              "better-behaved reference player or restricting to a safe magnitude range.")
    print("\nDone.")


if __name__ == "__main__":
    main()
