"""
Layer 4 of the player-team fit model (docs/fit-model-design.md): represent
a team's current roster as a subspace of style-space, spanned by its
qualified rostered players' standardized style vectors (Layers 1-2), then
score any candidate player by the magnitude of the component of their
style vector that is *orthogonal* to that subspace -- i.e. how much of
what they do is not already covered by someone on the roster.

This is a structural gap measurement, not a performance prediction: it
answers "does this roster already have a player who moves like this," not
"how would this player perform here." Per the design doc's core
constraint, nothing here outputs a projected stat line or a fit score
framed as a forecast.

Reuses feature_vector.py's build_feature_matrix() and run_pca() directly
for the feature matrix and its standardized/imputed form -- no
reimplementation of that pipeline here, and the gap score is computed in
the same standardized feature space Layer 3's Mahalanobis comps use, for
consistency across layers.

Run directly to build OKC's roster subspace and print gap scores for a
handful of players, testing whether the orthogonal-gap score behaves
sensibly (a player who looks like the roster's existing core gets a small
gap; a player with an uncovered style gets a large one).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_vector import (
    MIN_GAMES,
    MIN_MINUTES,
    _resolve_reference_player,
    build_feature_matrix,
    run_pca,
)

# ── roster subspace ──────────────────────────────────────────────────────────

def build_roster_subspace(space: dict, team_abbreviation: str) -> dict:
    """A team's roster subspace is the span of its qualified rostered
    players' standardized style vectors (X_scaled rows for that team) --
    unweighted by minutes/usage: every qualified player on the roster
    contributes equally to what style-directions the team "already has,"
    regardless of how much they play. This is a deliberate simplification
    (not a "who the team actually plays most" measure) consistent with
    every other qualification-floor-based cut in this project: a player
    either clears the MIN>=15/GP>=40 floor and counts, or doesn't and is
    absent entirely.

    The subspace basis is computed via SVD of the team's (n_players x
    n_features) standardized matrix: the right singular vectors (V rows)
    with non-negligible singular values span the same subspace as the
    raw player vectors but are orthonormal, which is what the orthogonal-
    projection gap score in gap_score() needs.

    Returns a dict with: team_abbreviation, player_names (players
    contributing to the subspace), n_players, basis (orthonormal basis
    vectors, shape (rank, n_features)), rank (numerical rank of the
    subspace -- can be less than n_players if two rostered players have
    near-identical style vectors).
    """
    feature_matrix = space["feature_matrix"]
    mask = (feature_matrix["TEAM_ABBREVIATION"] == team_abbreviation).values
    n_players = int(mask.sum())
    if n_players == 0:
        raise ValueError(f"No qualified players found for team {team_abbreviation!r}")

    roster_X = space["X_scaled"][mask]
    player_names = feature_matrix.loc[mask, "PLAYER_NAME"].tolist()

    # SVD: roster_X = U @ diag(S) @ Vt. Vt's rows spanning non-negligible
    # singular values ARE an orthonormal basis for the row space of
    # roster_X (the subspace the roster's style vectors span).
    U, S, Vt = np.linalg.svd(roster_X, full_matrices=False)
    tol = max(roster_X.shape) * np.finfo(roster_X.dtype).eps * (S[0] if len(S) else 0)
    rank = int((S > tol).sum())
    basis = Vt[:rank]

    return {
        "team_abbreviation": team_abbreviation,
        "player_names": player_names,
        "n_players": n_players,
        "basis": basis,
        "rank": rank,
    }


def gap_score(roster: dict, style_vector: np.ndarray) -> tuple[float, float, float]:
    """The orthogonal-projection gap score: given a player's (or
    synthetic point's) standardized style_vector, project it onto the
    roster's subspace and take the magnitude of the residual (the
    orthogonal component) -- how much of this style is NOT already
    covered by someone on the roster.

    Returns (gap_magnitude, covered_magnitude, total_magnitude) where
    total_magnitude = ||style_vector|| and
    total_magnitude^2 ~= covered_magnitude^2 + gap_magnitude^2
    (exact when the basis is orthonormal, which build_roster_subspace
    guarantees via SVD). covered_magnitude is included so callers can see
    the split, not just the gap number in isolation -- a large gap on a
    player with a small total_magnitude (a fairly average, unremarkable
    style vector) means something different than a large gap on a player
    with an extreme style vector.
    """
    basis = roster["basis"]
    projection_coords = basis @ style_vector          # coordinates in the roster subspace
    projected = basis.T @ projection_coords            # projection back into full feature space
    residual = style_vector - projected

    gap_magnitude = float(np.linalg.norm(residual))
    covered_magnitude = float(np.linalg.norm(projected))
    total_magnitude = float(np.linalg.norm(style_vector))
    return gap_magnitude, covered_magnitude, total_magnitude


# ── shared setup (mirrors comps.py's build_mahalanobis_space) ──────────────────

def build_style_space(feature_matrix: pd.DataFrame) -> dict:
    """Same standardized, median-imputed feature space as comps.py's
    build_mahalanobis_space -- reused here via run_pca directly rather
    than rebuilt, so Layer 3 and Layer 4 operate on identical vectors.

    Missing-feature handling: identical median-imputation disclosure as
    feature_vector.run_pca / comps.py -- median-imputed on a copy for
    this statistical step only, raw feature_matrix values untouched.
    """
    _, _, X_scaled, feature_cols = run_pca(feature_matrix)
    X_raw = feature_matrix[feature_cols]
    return {
        "feature_matrix": feature_matrix,
        "feature_cols": feature_cols,
        "X_raw": X_raw,
        "X_scaled": X_scaled,
    }


def player_style_vector(space: dict, player_name: str) -> np.ndarray | None:
    names = space["feature_matrix"]["PLAYER_NAME"].reset_index(drop=True)
    idx = names[names == player_name].index
    if len(idx) == 0:
        return None
    return space["X_scaled"][idx[0]]


# ── Reporting ────────────────────────────────────────────────────────────────

def print_roster_subspace_summary(roster: dict) -> None:
    print("=" * 70)
    print(f"ROSTER SUBSPACE — {roster['team_abbreviation']}")
    print("=" * 70)
    print(f"  Qualified rostered players ({roster['n_players']}): "
          + ", ".join(roster["player_names"]))
    print(f"  Numerical rank of subspace: {roster['rank']} "
          f"(out of {roster['n_players']} contributing player vectors)")
    if roster["rank"] < roster["n_players"]:
        print(f"  Rank < player count: some rostered players' style vectors are "
              f"near-linearly-dependent on others' (very similar styles), so they "
              f"don't add a fully new dimension to the roster's covered style-space.")
    print()


def print_gap_score(space: dict, roster: dict, player_name: str) -> None:
    canonical = _resolve_reference_player(player_name, space["feature_matrix"]["PLAYER_NAME"])
    if canonical is None:
        print(f"  [{player_name}] — could not resolve to a roster player, skipping]")
        return
    vec = player_style_vector(space, canonical)
    if vec is None:
        print(f"  [{canonical}] — resolved but not in the qualified feature matrix, skipping]")
        return

    on_roster = canonical in roster["player_names"]
    gap, covered, total = gap_score(roster, vec)
    pct_covered = (covered / total * 100) if total > 0 else float("nan")

    tag = "  [already on this roster]" if on_roster else ""
    print(f"  {canonical:<28} gap={gap:.4f}  covered={covered:.4f}  "
          f"total={total:.4f}  ({pct_covered:.1f}% covered){tag}")


REFERENCE_PLAYERS = [
    "Stephen Curry",
    "Nikola Jokic",
    "Alex Caruso",
    "Donovan Clingan",
    "Shai Gilgeous-Alexander",
]


def get_gap_scores(space: dict, roster: dict, player_names: list[str]) -> dict[str, tuple]:
    """Resolve and score each reference player against a roster subspace,
    returning {canonical_or_input_name: (gap, covered, total, pct_covered)}
    for players that could be resolved and are in the qualified feature
    matrix -- unresolvable names are omitted rather than raising, since
    this is used for a fixed reference list that may not always overlap
    with a given team's own roster.
    """
    results = {}
    for name in player_names:
        canonical = _resolve_reference_player(name, space["feature_matrix"]["PLAYER_NAME"])
        if canonical is None:
            continue
        vec = player_style_vector(space, canonical)
        if vec is None:
            continue
        gap, covered, total = gap_score(roster, vec)
        pct_covered = (covered / total * 100) if total > 0 else float("nan")
        results[canonical] = (gap, covered, total, pct_covered)
    return results


def run_team_analysis(space: dict, team: str, roster_note: str = "") -> dict:
    """Build the roster subspace for `team`, print the same
    subspace-summary / rostered-player-gap / reference-player-gap report
    format used for every team, and return the roster dict plus the
    reference-player gap scores so callers can do cross-team comparison
    or sanity-check narration afterward.
    """
    roster = build_roster_subspace(space, team)
    if roster_note:
        print(f"  ({roster_note})")
    print_roster_subspace_summary(roster)

    print("=" * 70)
    print(f"GAP SCORES — {team}")
    print("=" * 70)
    print(f"  Sanity check: a {team} player already on the roster should score a")
    print("  near-zero gap (they ARE part of the subspace). A stylistically")
    print("  distinct outside player should score a larger gap.")
    print(f"\n  Caveat: this roster's subspace has rank {roster['rank']} within a "
          f"{len(space['feature_cols'])}-dimensional feature space. {roster['n_players']} "
          f"players cannot span more than {roster['n_players']} dimensions regardless of "
          f"who they are, so even a high 'percent covered' figure below reflects coverage "
          f"relative to what a {roster['n_players']}-player roster can possibly cover, not "
          f"coverage of the full style space.\n")

    print(f"  -- Players already on the {team} roster --")
    for p in roster["player_names"]:
        print_gap_score(space, roster, p)

    print(f"\n  -- Reference players from earlier layers, not necessarily on {team} --")
    ref_scores = get_gap_scores(space, roster, REFERENCE_PLAYERS)
    for name, (gap, covered, total, pct_covered) in ref_scores.items():
        on_roster = name in roster["player_names"]
        tag = "  [already on this roster]" if on_roster else ""
        print(f"  {name:<28} gap={gap:.4f}  covered={covered:.4f}  "
              f"total={total:.4f}  ({pct_covered:.1f}% covered){tag}")

    print()
    return {"roster": roster, "ref_scores": ref_scores}


def print_basketball_sense_check(
    team: str,
    hypothesis: str,
    results: dict,
    ref_player: str,
    all_results: dict[str, dict],
    expect_high_coverage: bool,
) -> None:
    """Print an explicit narrative for one team's gap score against one
    reference player, stating the hypothesis being tested and then the
    ACTUAL result and whether it confirms or contradicts that hypothesis
    -- judged by comparing this team's coverage of ref_player against the
    other teams' coverage of the same player (a relative, cross-team
    comparison, since raw coverage % has no inherent "high" or "low"
    threshold on its own). This intentionally does not force the printed
    verdict to match the hypothesis -- if the numbers disagree with the
    basketball intuition, that's reported as a real, notable finding,
    not smoothed over.
    """
    ref_scores = results["ref_scores"]
    print("=" * 70)
    print(f"BASKETBALL-SENSE CHECK — {team}")
    print("=" * 70)
    print(f"  Hypothesis: {hypothesis}")
    if ref_player not in ref_scores:
        print(f"  [{ref_player} could not be resolved / is not in the qualified "
              f"feature matrix -- cannot check]\n")
        return

    gap, covered, total, pct_covered = ref_scores[ref_player]

    # Exclude any comparison team where ref_player is literally on that
    # team's own roster: their coverage is then mechanically 100% (gap=0)
    # by construction -- they're part of their own subspace basis, not a
    # real signal about whether that roster's style covers this player's
    # profile. Including those would rig the "highest/lowest of the
    # teams compared" comparison toward whichever team happens to roster
    # the reference player.
    excluded = [
        other_team for other_team, r in all_results.items()
        if other_team != team and ref_player in r["ref_scores"]
        and ref_player in r["roster"]["player_names"]
    ]
    other_pcts = {
        other_team: r["ref_scores"][ref_player][3]
        for other_team, r in all_results.items()
        if other_team != team and ref_player in r["ref_scores"]
        and ref_player not in r["roster"]["player_names"]
    }

    print(f"  Result: {ref_player} vs. {team} roster subspace: "
          f"gap={gap:.4f}  ({pct_covered:.1f}% covered)")
    if excluded:
        print(f"  Excluded from comparison: {', '.join(excluded)} -- "
              f"{ref_player} is literally on that roster, so their coverage "
              f"there is a trivial 100% by construction, not a real signal.")
    if other_pcts:
        comparison = ", ".join(f"{t}={p:.1f}%" for t, p in other_pcts.items())
        print(f"  For comparison, {ref_player}'s coverage against the other teams "
              f"in this run: {comparison}")
        is_highest = pct_covered >= max(other_pcts.values())
        is_lowest = pct_covered <= min(other_pcts.values())
        matches = (expect_high_coverage and is_highest) or (not expect_high_coverage and is_lowest)
        direction = "highest" if is_highest else ("lowest" if is_lowest else "in between")
        verdict = "CONFIRMS" if matches else "DOES NOT CONFIRM"
        print(f"  {team}'s coverage of {ref_player} is the {direction} of the "
              f"teams compared. Verdict: this {verdict} the hypothesis.")
    print()


def main() -> None:
    print("Building feature matrix from Layer 1 axes "
          f"(qualification floor: MIN>={MIN_MINUTES}, GP>={MIN_GAMES})...")
    feature_matrix = build_feature_matrix()
    print(f"Feature matrix: {feature_matrix.shape[0]} qualified players\n")

    space = build_style_space(feature_matrix)

    print("#" * 70)
    print("# TEAM 1 OF 3: OKC -- balanced, defensively versatile roster")
    print("# (baseline case from the original Layer 4 run)")
    print("#" * 70 + "\n")
    okc_results = run_team_analysis(space, "OKC")

    print("#" * 70)
    print("# TEAM 2 OF 3: GSW -- lacks an elite rim-protecting center")
    print("# (Horford is a stretch-5, Draymond is undersized at center;")
    print("#  no traditional rim-protector on the qualified roster)")
    print("#" * 70 + "\n")
    gsw_results = run_team_analysis(space, "GSW")

    print("#" * 70)
    print("# TEAM 3 OF 5: NYK -- guard-heavy, loaded with ball-dominant")
    print("# perimeter players (Brunson, Clarkson, Alvarado, McBride, Hart)")
    print("#" * 70 + "\n")
    nyk_results = run_team_analysis(space, "NYK")

    print("#" * 70)
    print("# TEAM 4 OF 5: CLE -- maximally obvious rim-protection-rich case:")
    print("# TWO genuine elite shot-blocking bigs (Evan Mobley, Jarrett")
    print("# Allen) on the roster at once. If the gap-score mechanism means")
    print("# anything, this team should show the HIGHEST Clingan coverage")
    print("# of every team tested.")
    print("#" * 70 + "\n")
    cle_results = run_team_analysis(space, "CLE")

    print("#" * 70)
    print("# TEAM 5 OF 5: CHI -- maximally obvious rim-protection-poor case:")
    print("# unambiguously undersized/guard-and-wing-only frontcourt, no")
    print("# true center on the qualified roster at all (Yabusele, Jalen")
    print("# Smith, Leonard Miller, Patrick Williams, Buzelis are combo")
    print("# forwards, not rim protectors). Should show the LOWEST Clingan")
    print("# coverage of every team tested.")
    print("#" * 70 + "\n")
    chi_results = run_team_analysis(space, "CHI")

    print("#" * 70)
    print("# EXPLICIT BASKETBALL-SENSE CHECKS")
    print("#" * 70 + "\n")

    all_results = {
        "OKC": okc_results,
        "GSW": gsw_results,
        "NYK": nyk_results,
        "CLE": cle_results,
        "CHI": chi_results,
    }

    print_basketball_sense_check(
        "GSW",
        "GSW lacks an elite rim-protecting center, so it should show LOW "
        "coverage / HIGH gap for a rim-protector profile like Donovan "
        "Clingan relative to the other teams in this run -- that would be "
        "a real, missing piece of this roster.",
        gsw_results,
        "Donovan Clingan",
        all_results,
        expect_high_coverage=False,
    )
    print_basketball_sense_check(
        "NYK",
        "NYK is guard-heavy and already loaded with ball-dominant "
        "perimeter players, so it should show HIGH coverage / LOW gap for "
        "a ball-dominant guard profile like Shai Gilgeous-Alexander "
        "relative to the other teams in this run -- this roster likely "
        "already has that style covered.",
        nyk_results,
        "Shai Gilgeous-Alexander",
        all_results,
        expect_high_coverage=True,
    )
    print_basketball_sense_check(
        "CLE",
        "MAXIMALLY OBVIOUS CASE: CLE rosters TWO genuine elite "
        "shot-blocking bigs at once (Evan Mobley, Jarrett Allen). If the "
        "gap-score mechanism captures rim-protection style at all, this "
        "team should show the HIGHEST Clingan coverage of every team "
        "tested -- if it doesn't, that's real evidence the mechanism "
        "itself has a problem, not just noise from an ambiguous case.",
        cle_results,
        "Donovan Clingan",
        all_results,
        expect_high_coverage=True,
    )
    print_basketball_sense_check(
        "CHI",
        "MAXIMALLY OBVIOUS CASE: CHI has no true center on the qualified "
        "roster at all -- an unambiguously undersized, guard-and-wing-only "
        "frontcourt. This team should show the LOWEST Clingan coverage of "
        "every team tested -- if it doesn't, that's real evidence the "
        "mechanism itself has a problem, not just noise from an ambiguous "
        "case.",
        chi_results,
        "Donovan Clingan",
        all_results,
        expect_high_coverage=False,
    )

    print("=" * 70)
    print("CROSS-TEAM COMPARISON -- same 5 reference players, all 5 rosters")
    print("=" * 70)
    team_order = ["OKC", "GSW", "NYK", "CLE", "CHI"]
    header = "  " + f"{'Reference player':<28}" + "".join(f"{t:>16}" for t in team_order)
    print(header)
    for name in REFERENCE_PLAYERS:
        canonical = _resolve_reference_player(name, feature_matrix["PLAYER_NAME"]) or name
        row_cells = []
        for team in team_order:
            results = all_results[team]
            if canonical in results["ref_scores"]:
                gap, _, _, pct = results["ref_scores"][canonical]
                row_cells.append(f"{gap:.2f} ({pct:.0f}%)")
            else:
                row_cells.append("n/a")
        row = "  " + f"{canonical:<28}" + "".join(f"{c:>16}" for c in row_cells)
        print(row)

    print("\n  -- Clingan-specific ranking across all 5 teams (highest coverage first) --")
    clingan_canonical = _resolve_reference_player("Donovan Clingan", feature_matrix["PLAYER_NAME"])
    clingan_rank = []
    for team in team_order:
        results = all_results[team]
        if clingan_canonical in results["ref_scores"]:
            gap, _, _, pct = results["ref_scores"][clingan_canonical]
            clingan_rank.append((team, pct, gap))
    clingan_rank.sort(key=lambda row: row[1], reverse=True)
    for rank, (team, pct, gap) in enumerate(clingan_rank, start=1):
        print(f"    {rank}. {team:<5} {pct:.1f}% covered  (gap={gap:.4f})")

    if clingan_rank:
        top_team = clingan_rank[0][0]
        bottom_team = clingan_rank[-1][0]
        if top_team == "CLE" and bottom_team == "CHI":
            print(f"\n  CLE ranks highest and CHI ranks lowest, as the maximally obvious "
                  f"case predicts -- this supports the mechanism actually tracking "
                  f"rim-protection style, and suggests the earlier GSW/NYK results "
                  f"were closer to genuine subtlety in the data than a broken method.")
        else:
            print(f"\n  Top-ranked team is {top_team}, bottom-ranked is {bottom_team} -- "
                  f"NOT the CLE-highest/CHI-lowest pattern the maximally obvious case "
                  f"predicts. Even in this deliberately unambiguous test, coverage does "
                  f"not clearly favor the size-heavy team. This is evidence the "
                  f"gap-score mechanism itself may not be reliably tracking the "
                  f"rim-protection dimension of style-space, rather than the GSW/NYK "
                  f"result being explainable as noise from an ambiguous middle case.")

    print("\nDone.")


if __name__ == "__main__":
    main()
