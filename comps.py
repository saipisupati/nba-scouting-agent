"""
Layer 3 of the player-team fit model (docs/fit-model-design.md): find a
given player's nearest real comps using Mahalanobis distance over the
standardized Layer 1 feature space, correcting for feature covariance
(e.g. usage and assist rate moving together, drive volume and rim
pressure overlapping) so a comp isn't double-weighted on two features
that are really measuring one underlying thing.

Reuses feature_vector.py's build_feature_matrix() and run_pca() directly
for the feature matrix and its standardized/imputed form -- no
reimplementation of that pipeline here. Mahalanobis distance is computed
over the full standardized feature space (the same X_scaled run_pca()
already builds), not the PCA-reduced space -- Layer 3 as designed
corrects for covariance among "the feature axes above" (the raw Layer 1
axes), which is exactly what the inverse covariance matrix over X_scaled
does; reducing to PCA components first would throw away the very
covariance structure Mahalanobis distance is being used to correct for.

Run directly to print real comps for the same five reference players
used throughout this project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

from feature_vector import (
    MIN_GAMES,
    MIN_MINUTES,
    REFERENCE_PLAYERS,
    _resolve_reference_player,
    build_feature_matrix,
    run_pca,
)

# ── covariance / inverse-covariance ─────────────────────────────────────────

# Below this many effective samples per feature, the sample covariance
# matrix is at meaningful risk of being singular or near-singular (37
# features on 305 players, many correlated, is within this risk zone) --
# used only to decide whether to log a near-singularity note, not as a
# hard gate.
_SINGULARITY_CONDITION_NUMBER_THRESHOLD = 1e10


def compute_inverse_covariance(X_scaled: np.ndarray) -> tuple[np.ndarray, str]:
    """Compute the inverse of the covariance matrix of X_scaled (rows =
    players, columns = standardized features), which is what Mahalanobis
    distance actually needs.

    X_scaled has 305 players and up to 27 features here -- more samples
    than features, so the sample covariance matrix is not guaranteed
    singular by construction, but real correlation between many of the
    Layer 1 features (documented in the design doc: usage/assist rate,
    drive volume/rim pressure) can still make it ill-conditioned enough
    that direct inversion (np.linalg.inv) is numerically unstable or
    fails outright.

    Tries direct inversion first. If that fails (LinAlgError) OR the
    condition number of the covariance matrix exceeds
    _SINGULARITY_CONDITION_NUMBER_THRESHOLD (a near-singular matrix that
    inverts "successfully" but produces a numerically garbage inverse),
    falls back to the Moore-Penrose pseudo-inverse (np.linalg.pinv),
    which is the standard, named regularized approach for exactly this
    case -- not a silent, ad hoc substitute. Returns (inv_cov, method)
    where method is "direct inverse" or "pseudo-inverse (near-singular
    covariance matrix)" so callers can disclose which path was taken
    rather than the caller having to infer it.
    """
    cov = np.cov(X_scaled, rowvar=False)
    condition_number = np.linalg.cond(cov)

    if condition_number < _SINGULARITY_CONDITION_NUMBER_THRESHOLD:
        try:
            inv_cov = np.linalg.inv(cov)
            return inv_cov, "direct inverse"
        except np.linalg.LinAlgError:
            pass

    inv_cov = np.linalg.pinv(cov)
    return inv_cov, (
        f"pseudo-inverse (near-singular covariance matrix, "
        f"condition number={condition_number:.3e})"
    )


# ── confidence tiers ─────────────────────────────────────────────────────────

# Tiered on TOTAL imputed features across the pair (query player's missing
# count + comp's missing count) -- either side being heavily imputed
# degrades the comp equally, since the distance calculation pulls whichever
# side is imputed toward the population median regardless of which player
# it is.
_HIGH_CONFIDENCE_MAX_TOTAL_IMPUTED = 2
_MODERATE_CONFIDENCE_MAX_TOTAL_IMPUTED = 6


def confidence_tier(total_imputed: int) -> str:
    if total_imputed <= _HIGH_CONFIDENCE_MAX_TOTAL_IMPUTED:
        return "high confidence"
    if total_imputed <= _MODERATE_CONFIDENCE_MAX_TOTAL_IMPUTED:
        return "moderate confidence"
    return "low confidence"


# ── Mahalanobis comps ────────────────────────────────────────────────────────

def build_mahalanobis_space(feature_matrix: pd.DataFrame) -> dict:
    """One-time setup shared by every comp lookup: the standardized,
    median-imputed feature matrix (identical construction to
    feature_vector.run_pca, reused directly rather than rebuilt) and its
    inverse covariance matrix.

    Missing-feature handling (same disclosure as feature_vector.py):
    StandardScaler/covariance cannot run on NaN. Missing values are
    median-imputed on a copy made for this statistical step ONLY -- the
    raw feature_matrix passed in keeps its true NaNs untouched, and this
    is the same deliberate, documented, isolated median-imputation
    exception feature_vector.run_pca uses, applied consistently here via
    the same function rather than a second reimplementation. A player
    with several imputed (originally-NaN) features has those features
    silently pulled toward the population median in the distance
    calculation -- this can make them look artificially CLOSER to
    typical players than their true (unmeasured) profile would, since an
    imputed feature contributes ~0 to the standardized distance in every
    direction. This is flagged per-player at report time (imputed
    feature count is shown for both the query player and the returned
    comps) rather than left as a silent artifact.

    Returns a dict with: feature_matrix, feature_cols, X (pre-impute, for
    counting NaNs), X_scaled (standardized+imputed, the space distance is
    computed in), inv_cov, inv_cov_method.
    """
    _, _, X_scaled, feature_cols = run_pca(feature_matrix)
    X_raw = feature_matrix[feature_cols]
    inv_cov, inv_cov_method = compute_inverse_covariance(X_scaled)

    return {
        "feature_matrix": feature_matrix,
        "feature_cols": feature_cols,
        "X_raw": X_raw,
        "X_scaled": X_scaled,
        "inv_cov": inv_cov,
        "inv_cov_method": inv_cov_method,
    }


def find_comps(
    space: dict,
    player_name: str,
    n: int = 5,
    max_candidate_imputed: int | None = None,
) -> pd.DataFrame:
    """Mahalanobis distance from player_name to every other qualified
    player in the standardized feature space, returning the n closest.

    player_name must already be the canonical name as it appears in
    feature_matrix['PLAYER_NAME'] (resolve via
    feature_vector._resolve_reference_player or
    feature_vector.resolve_player_name before calling, same as every
    other reference-player lookup in this project).

    max_candidate_imputed, if given, restricts CANDIDATES (not the query
    player) to those with at most that many imputed features before
    ranking -- e.g. max_candidate_imputed=2 answers "if we only trust
    lightly-imputed comps, who's closest?" This does not touch the query
    player's own imputed-feature count, which is a separate, unavoidable
    fact about how well-measured that specific player is.

    Returns a DataFrame with PLAYER_NAME, TEAM_ABBREVIATION,
    MAHALANOBIS_DISTANCE, N_IMPUTED_FEATURES (count of features that were
    NaN in the raw feature matrix and therefore median-imputed for this
    player before the distance calculation -- see
    build_mahalanobis_space's docstring), QUERY_N_IMPUTED,
    TOTAL_IMPUTED (query + candidate combined), and CONFIDENCE_TIER
    (see confidence_tier()), sorted closest-first, excluding the query
    player itself.
    """
    names = space["feature_matrix"]["PLAYER_NAME"].reset_index(drop=True)
    match_idx = names[names == player_name].index
    if len(match_idx) == 0:
        raise ValueError(f"{player_name!r} not found in the qualified feature matrix")
    query_idx = match_idx[0]

    X_scaled = space["X_scaled"]
    inv_cov = space["inv_cov"]
    query_vec = X_scaled[query_idx]

    n_imputed = space["X_raw"].isna().sum(axis=1).reset_index(drop=True)
    query_n_imputed = int(n_imputed.iloc[query_idx])

    candidate_mask = pd.Series(True, index=names.index)
    candidate_mask.iloc[query_idx] = False
    if max_candidate_imputed is not None:
        candidate_mask &= n_imputed <= max_candidate_imputed
    candidate_indices = names.index[candidate_mask]

    distances = np.array([
        mahalanobis(query_vec, X_scaled[i], inv_cov) for i in candidate_indices
    ])

    result = pd.DataFrame({
        "PLAYER_NAME": names.iloc[candidate_indices].values,
        "TEAM_ABBREVIATION": space["feature_matrix"]["TEAM_ABBREVIATION"].reset_index(drop=True).iloc[candidate_indices].values,
        "MAHALANOBIS_DISTANCE": distances,
        "N_IMPUTED_FEATURES": n_imputed.iloc[candidate_indices].values,
    })
    result["QUERY_N_IMPUTED"] = query_n_imputed
    result["TOTAL_IMPUTED"] = result["N_IMPUTED_FEATURES"] + query_n_imputed
    result["CONFIDENCE_TIER"] = result["TOTAL_IMPUTED"].apply(confidence_tier)

    result = result.sort_values("MAHALANOBIS_DISTANCE")
    return result.head(n).reset_index(drop=True)


# ── Reporting ────────────────────────────────────────────────────────────────

def print_comps_for_player(space: dict, player_name: str, n: int = 5) -> pd.DataFrame | None:
    roster_names = space["feature_matrix"]["PLAYER_NAME"]
    canonical = _resolve_reference_player(player_name, roster_names)
    if canonical is None:
        print(f"\n[{player_name}] — could not resolve to a roster player, skipping]")
        return None

    n_imputed = space["X_raw"].isna().sum(axis=1)
    names = space["feature_matrix"]["PLAYER_NAME"].reset_index(drop=True)
    query_idx = names[names == canonical].index
    query_n_imputed = int(n_imputed.iloc[query_idx[0]]) if len(query_idx) else None

    print("=" * 70)
    print(f"{canonical}  ({space['feature_matrix'].loc[space['feature_matrix']['PLAYER_NAME'] == canonical, 'TEAM_ABBREVIATION'].iloc[0]})")
    print("=" * 70)
    if query_n_imputed:
        print(f"  [{query_n_imputed} of {len(space['feature_cols'])} features were missing for "
              f"{canonical} and median-imputed before this distance calculation -- "
              f"treat this comp list with extra caution.]")

    comps = find_comps(space, canonical, n=n)
    print(f"  {n} closest comps by Mahalanobis distance:")
    for rank, row in enumerate(comps.itertuples(index=False), start=1):
        print(f"    {rank}. {row.PLAYER_NAME:<28} ({row.TEAM_ABBREVIATION})  "
              f"dist={row.MAHALANOBIS_DISTANCE:.4f}  [{row.CONFIDENCE_TIER.upper()} -- "
              f"{query_n_imputed} (query) + {row.N_IMPUTED_FEATURES} (comp) = "
              f"{row.TOTAL_IMPUTED} total imputed of {2 * len(space['feature_cols'])}]")
    print()
    return comps


def print_restricted_comparison(space: dict, player_name: str, n: int = 5, max_candidate_imputed: int = 2) -> None:
    """Re-run find_comps restricted to lightly-imputed candidates only
    (<=max_candidate_imputed) and diff against the unrestricted top-n:
    does a more reliable comp exist further down the ranked list that the
    unrestricted output isn't surfacing?"""
    roster_names = space["feature_matrix"]["PLAYER_NAME"]
    canonical = _resolve_reference_player(player_name, roster_names)
    if canonical is None:
        return

    unrestricted = find_comps(space, canonical, n=n)
    restricted = find_comps(space, canonical, n=n, max_candidate_imputed=max_candidate_imputed)

    print(f"  {canonical} -- restricted to candidates with <= {max_candidate_imputed} imputed features:")
    if restricted.empty:
        print(f"    No qualified candidates have <= {max_candidate_imputed} imputed features. "
              f"No restricted comp list is possible at this threshold.")
        print()
        return

    for rank, row in enumerate(restricted.itertuples(index=False), start=1):
        print(f"    {rank}. {row.PLAYER_NAME:<28} ({row.TEAM_ABBREVIATION})  "
              f"dist={row.MAHALANOBIS_DISTANCE:.4f}  [{row.CONFIDENCE_TIER.upper()} -- "
              f"{row.QUERY_N_IMPUTED} (query) + {row.N_IMPUTED_FEATURES} (comp) = "
              f"{row.TOTAL_IMPUTED} total imputed]")

    unrestricted_names = set(unrestricted["PLAYER_NAME"])
    restricted_names = set(restricted["PLAYER_NAME"])
    new_names = restricted_names - unrestricted_names
    if new_names:
        print(f"\n    Change vs. unrestricted top-{n}: {', '.join(sorted(new_names))} "
              f"now appear(s) -- a more lightly-imputed comp exists further down the "
              f"ranked list than the unrestricted output surfaced.")
    else:
        print(f"\n    Change vs. unrestricted top-{n}: none -- the same players lead "
              f"the ranking even after excluding heavily-imputed candidates, so "
              f"restricting the candidate pool doesn't surface a materially different "
              f"or more reliable comp list here.")
    print()

_CACHED_SPACE: dict | None = None


def get_or_build_mahalanobis_space() -> dict:
    """Lazily build and cache the standardized feature space and inverse
    covariance matrix so repeated API requests run in milliseconds."""
    global _CACHED_SPACE
    if _CACHED_SPACE is None:
        fm = build_feature_matrix()
        _CACHED_SPACE = build_mahalanobis_space(fm)
    return _CACHED_SPACE


FEATURE_LABELS = {
    "SELF_CREATION_SHARE": "Self-creation share",
    "AST_PCT": "Assist rate",
    "AST_TO": "Assist-to-turnover ratio",
    "AST_RATIO": "Assist ratio",
    "DRIVE_AST_PCT": "Drive kickout rate",
    "DRIVE_PASSES_PCT": "Drive pass rate",
    "DRIVE_FG_PCT": "Finishing on drives",
    "SPOTUP_SHARE": "Spot-up / catch-and-shoot volume",
    "CUT_OFFSCREEN_SHARE": "Off-ball movement & cuts",
    "DEF_PPP_ISOLATION": "Isolation defense",
    "DEF_PPP_PRBALLHANDLER": "Pick-and-roll ballhandler defense",
    "DEF_PPP_PRROLLMAN": "Roll man defense",
    "DEF_PPP_POSTUP": "Post-up defense",
    "DEF_PPP_SPOTUP": "Closeout / spot-up defense",
    "DEF_PPP_HANDOFF": "Dribble handoff defense",
    "DEF_PPP_OFFSCREEN": "Off-screen defense",
    "SHOT_SUPPRESSION_OVERALL": "Overall shot suppression",
    "SHOT_SUPPRESSION_3_POINTERS": "Perimeter shot suppression",
    "SHOT_SUPPRESSION_2_POINTERS": "Midrange shot suppression",
    "SHOT_SUPPRESSION_LESS_THAN_6FT": "Rim shot suppression",
    "DEFLECTIONS_PER36": "Deflection activity (per 36)",
    "CONTESTED_SHOTS_PER36": "Contest volume (per 36)",
    "BOX_OUTS_PER36": "Box-out frequency (per 36)",
    "TS_PCT": "True shooting efficiency",
    "EFG_PCT": "Effective field goal %",
    "USG_PCT": "Usage load",
    "PACE": "Transition pace",
    "TRANSITION_PPP": "Transition efficiency",
}


def get_comps_data(player_name: str, n: int = 5, space: dict | None = None) -> dict:
    """Retrieve Mahalanobis-distance stylistic comps for a player, returning
    a structured dictionary suitable for API and UI rendering."""
    if space is None:
        space = get_or_build_mahalanobis_space()

    roster_names = space["feature_matrix"]["PLAYER_NAME"]
    canonical = _resolve_reference_player(player_name, roster_names)
    if canonical is None:
        raise ValueError(
            f"Player {player_name!r} not found in the qualified feature matrix "
            f"(minimum {MIN_MINUTES} MPG, {MIN_GAMES} GP)."
        )

    fm = space["feature_matrix"]
    player_row = fm[fm["PLAYER_NAME"] == canonical].iloc[0]
    team = str(player_row["TEAM_ABBREVIATION"])

    n_imputed = space["X_raw"].isna().sum(axis=1)
    names = fm["PLAYER_NAME"].reset_index(drop=True)
    query_idx = names[names == canonical].index
    query_n_imputed = int(n_imputed.iloc[query_idx[0]]) if len(query_idx) else 0

    comps_df = find_comps(space, canonical, n=n)

    X_scaled = space["X_scaled"]
    feature_cols = space["feature_cols"]
    query_row_idx = query_idx[0]
    query_vec = X_scaled[query_row_idx]

    comp_items = []
    for rank, row in enumerate(comps_df.itertuples(index=False), start=1):
        dist = float(row.MAHALANOBIS_DISTANCE)
        sim_pct = max(15.0, min(98.5, round(100.0 - (dist - 2.5) * 8.5, 1)))
        tier = "Near Twin" if dist <= 4.8 else ("High Overlap" if dist <= 5.8 else "Moderate Overlap")

        comp_row_matches = names[names == row.PLAYER_NAME].index
        if len(comp_row_matches):
            comp_row_idx = comp_row_matches[0]
            comp_vec = X_scaled[comp_row_idx]
            diffs = np.abs(query_vec - comp_vec)
            sorted_asc = np.argsort(diffs)
            sorted_desc = np.argsort(diffs)[::-1]
            shared_traits = [FEATURE_LABELS.get(feature_cols[i], feature_cols[i]) for i in sorted_asc[:2]]
            divergent_traits = [FEATURE_LABELS.get(feature_cols[i], feature_cols[i]) for i in sorted_desc[:2]]
        else:
            shared_traits = []
            divergent_traits = []

        comp_items.append({
            "rank": rank,
            "player_name": str(row.PLAYER_NAME),
            "team_abbreviation": str(row.TEAM_ABBREVIATION),
            "mahalanobis_distance": round(dist, 3),
            "similarity_pct": sim_pct,
            "similarity_tier": tier,
            "confidence_tier": str(row.CONFIDENCE_TIER),
            "shared_archetype_features": shared_traits,
            "divergent_features": divergent_traits,
            "n_imputed_features": int(row.N_IMPUTED_FEATURES),
            "query_n_imputed": int(row.QUERY_N_IMPUTED),
            "total_imputed": int(row.TOTAL_IMPUTED),
        })

    notes = (
        f"Evaluated across {len(feature_cols)} style dimensions with inverted covariance correction. "
        + (f"{query_n_imputed} features median-imputed for {canonical} due to limited play-type sample." if query_n_imputed > 0 else "0 query features imputed.")
    )

    return {
        "player_name": canonical,
        "team_abbreviation": team,
        "query_n_imputed": query_n_imputed,
        "total_features": len(feature_cols),
        "comps": comp_items,
        "notes": notes,
        "season": "2025-26",
    }


def format_comps_answer(data: dict) -> str:
    """Format comps data into a verified natural language answer string."""
    lines = [
        f"{data['player_name']} ({data['team_abbreviation']}) — closest stylistic twins by covariance-corrected Mahalanobis distance across {data['total_features']} style axes [2025-26]:"
    ]
    for c in data["comps"]:
        lines.append(
            f"• #{c['rank']} {c['player_name']} ({c['team_abbreviation']}) — distance: {c['mahalanobis_distance']:.2f} [{c['confidence_tier'].upper()}]"
        )
    if data["query_n_imputed"] > 0:
        lines.append(
            f"NOTE: {data['query_n_imputed']} of {data['total_features']} style features were median-imputed for {data['player_name']} due to limited qualifying sample in certain play types."
        )
    lines.append(
        "NOTE: Mahalanobis distance corrects for feature covariance so comps are not double-weighted on correlated metrics (e.g. usage and assist rate moving together)."
    )
    return "\n".join(lines)


def main() -> None:
    print("Building feature matrix from Layer 1 axes "
          f"(qualification floor: MIN>={MIN_MINUTES}, GP>={MIN_GAMES})...")
    feature_matrix = build_feature_matrix()
    print(f"Feature matrix: {feature_matrix.shape[0]} qualified players\n")

    space = build_mahalanobis_space(feature_matrix)
    print("=" * 70)
    print("INVERSE COVARIANCE MATRIX")
    print("=" * 70)
    print(f"  Features used: {len(space['feature_cols'])}")
    print(f"  Method: {space['inv_cov_method']}")
    print()

    print("=" * 70)
    print("REAL COMPS — Mahalanobis distance, 5 closest per reference player")
    print("=" * 70)

    all_comps = {}
    for ref_name in REFERENCE_PLAYERS:
        comps = print_comps_for_player(space, ref_name, n=5)
        if comps is not None:
            canonical = _resolve_reference_player(ref_name, feature_matrix["PLAYER_NAME"])
            all_comps[canonical] = comps

    # Caruso and Clingan both had heavily-imputed comp lists in the
    # unrestricted output above -- check whether a more reliable
    # (lightly-imputed) comp exists further down the ranking that the
    # unrestricted top-5 isn't surfacing.
    print("=" * 70)
    print("RESTRICTED-CANDIDATE COMPARISON — candidates limited to <= 2 imputed features")
    print("=" * 70)
    for ref_name in ["Alex Caruso", "Donovan Clingan"]:
        print_restricted_comparison(space, ref_name, n=5, max_candidate_imputed=2)

    # Explicit Jokic/SGA cross-check, given they landed in the same
    # k-means cluster in the earlier clustering run (KMEANS_CLUSTERING_OUTPUT.txt).
    print("=" * 70)
    print("JOKIC / SGA CROSS-CHECK")
    print("=" * 70)
    jokic_canonical = _resolve_reference_player("Nikola Jokic", feature_matrix["PLAYER_NAME"])
    sga_canonical = _resolve_reference_player("Shai Gilgeous-Alexander", feature_matrix["PLAYER_NAME"])
    if jokic_canonical in all_comps and sga_canonical in all_comps:
        jokic_has_sga = sga_canonical in all_comps[jokic_canonical]["PLAYER_NAME"].values
        sga_has_jokic = jokic_canonical in all_comps[sga_canonical]["PLAYER_NAME"].values
        print(f"  Does {jokic_canonical}'s comp list include {sga_canonical}? {jokic_has_sga}")
        print(f"  Does {sga_canonical}'s comp list include {jokic_canonical}? {sga_has_jokic}")
        if not jokic_has_sga and not sga_has_jokic:
            print("  Neither appears in the other's top-5 -- despite landing in the same "
                  "k-means cluster (k=3), Mahalanobis distance over the full covariance-"
                  "corrected feature space finds closer real comps for each of them than "
                  "each other. This is not a contradiction: k-means at k=3 was a coarse, "
                  "weakly-supported split (silhouette <=0.096, see KMEANS_CLUSTERING_OUTPUT.txt) "
                  "grouping both into one large 108-player bucket, while Mahalanobis distance "
                  "is a much finer-grained, continuous measure within that same space.")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
