"""
Layer 1 of the player-team fit model (docs/fit-model-design.md): construct
a per-player archetype feature vector from the seven documented axes, using
only existing, already-pulled CSVs -- no new data pull.

Qualification floor: MIN >= 15, GP >= 40, this project's standard threshold
(see compute_defense.py's deflections_per36 et al.) applied to the base
player pool. Individual play-type axes additionally require that player to
clear THAT category's own possession floor (compute_offense.py's
_PLAYTYPE_DEFAULT_MIN_POSS / compute_defense.py's playtype defense floors) --
a qualified player who doesn't run enough of a given action has that
column left as NaN, never zeroed. Zero would claim "this player does this
0% of the time," which is a different (and false) statement from
"insufficient sample to measure this for this player."

Run directly to build the feature matrix, run PCA, and print the
diagnostics documented in the design doc: explained variance, top-component
loadings, and where five reference players land in the reduced space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from compute_offense import (
    _PLAYTYPE_CSV as _OFF_PLAYTYPE_CSV,
    _PLAYTYPE_DEFAULT_MIN_POSS as _OFF_PLAYTYPE_MIN_POSS,
    _MIN_TOTAL_POSS as _OFF_MIN_TOTAL_POSS,
    _apply_current_team,
    resolve_player_name,
)
from compute_defense import (
    _PLAYTYPE_CSV as _DEF_PLAYTYPE_CSV,
    _PLAYTYPE_DEFAULT_MIN_POSS as _DEF_PLAYTYPE_MIN_POSS,
)

# ── qualification thresholds ────────────────────────────────────────────────

MIN_MINUTES = 15
MIN_GAMES = 40

_USAGE_CONTEXT_CSV = "data/usage_context_2025_26.csv"
_DRIVES_CSV = "data/drives_2025_26.csv"
_HUSTLE_CSV = "data/hustle_stats_2025_26.csv"

# Drive qualification floor, same rationale/values as compute_offense.drive_efficiency
_MIN_DRIVES_PER_GAME = 1.7
_MIN_TOTAL_DRIVES = 60

_SHOT_DEFENSE_CSV = {
    "Overall":         ("data/shot_defense_overall_2025_26.csv", "D_FGA", "PCT_PLUSMINUS"),
    "3 Pointers":      ("data/shot_defense_3pt_2025_26.csv", "FG3A", "PLUSMINUS"),
    "2 Pointers":      ("data/shot_defense_2pt_2025_26.csv", "FG2A", "PLUSMINUS"),
    "Less Than 6Ft":   ("data/shot_defense_rim_2025_26.csv", "FGA_LT_06", "PLUSMINUS"),
}
_MIN_DEF_FGA = 100

# Self-creation vs. assisted grouping, exactly as specified in
# docs/fit-model-design.md's Layer 1 table. Transition and PRRollman are
# deliberately excluded -- neither is a "self-creation" or "assisted" shot
# for the primary ball-handler in the way the other seven categories are.
_SELF_CREATION_TYPES = ["Isolation", "PRBallHandler", "Postup"]
_ASSISTED_TYPES = ["Spotup", "Cut", "Handoff", "OffScreen"]

# Off-ball movement profile axis: same categories as above, different framing.
_OFFBALL_SPOTUP = ["Spotup"]
_OFFBALL_CUT_OFFSCREEN = ["Cut", "OffScreen"]


# ── mid-season trade dedup ──────────────────────────────────────────────────

def _combine_traded_stints(df: pd.DataFrame) -> pd.DataFrame:
    """Synergy play-type CSVs (offense: all 9 categories; defense: Postup,
    Handoff, OffScreen) carry one row PER TEAM STINT for a player traded
    mid-season, not one row per player -- e.g. James Harden appears twice
    in playtype_offense_isolation with LAC (GP=41) and CLE (GP=26). Indexing
    these files by PLAYER_NAME (as every axis function below does) requires
    exactly one row per player, so stints are combined here: POSS/GP/PTS/
    FGM/FGA are summed (real, additive totals across the season), and
    PPP/FG_PCT are RE-DERIVED from those summed totals (PTS/POSS,
    FGM/FGA) rather than averaged -- averaging two PPP values directly
    would weight a 5-possession stint equally with an 80-possession one,
    which is wrong. TEAM_ABBREVIATION is set to the stint with more POSS
    (the team the player did more of this action for); it is not used
    downstream anyway since _apply_current_team overwrites team elsewhere.
    Players with only one stint pass through unchanged."""
    if not df.duplicated("PLAYER_NAME", keep=False).any():
        return df

    df = df.copy()
    agg = {"POSS": "sum", "GP": "sum", "PTS": "sum", "FGM": "sum", "FGA": "sum"}
    agg = {k: v for k, v in agg.items() if k in df.columns}

    combined = df.groupby("PLAYER_NAME", as_index=False).agg(agg)
    if "PTS" in combined.columns:
        combined["PPP"] = (combined["PTS"] / combined["POSS"]).round(3)
    if "FGM" in combined.columns and "FGA" in combined.columns:
        combined["FG_PCT"] = (combined["FGM"] / combined["FGA"]).round(3)

    # team = stint with the most POSS for this play type
    team_col = "TEAM_ABBREVIATION" if "TEAM_ABBREVIATION" in df.columns else None
    if team_col:
        primary_team = (
            df.sort_values("POSS", ascending=False)
            .drop_duplicates("PLAYER_NAME")[["PLAYER_NAME", team_col]]
        )
        combined = combined.merge(primary_team, on="PLAYER_NAME", how="left")

    return combined


# ── qualified player base ───────────────────────────────────────────────────

def _qualified_player_base() -> pd.DataFrame:
    """MIN >= 15, GP >= 40 pool from usage_context_2025_26.csv, this
    project's standard qualification threshold. usage_context's MIN/GP are
    the same PerGame values as hustle_stats's MIN/G (verified: identical
    for every checked player, 582 vs 581 total rows, one extra in
    usage_context) -- usage_context is used as the base here rather than
    hustle_stats because it's the direct source for axes 2, 6, and 7
    (AST_PCT/AST_TO/AST_RATIO, TS_PCT/EFG_PCT/USG_PCT, PACE)."""
    df = pd.read_csv(_USAGE_CONTEXT_CSV)
    qualified = df[(df["MIN"] >= MIN_MINUTES) & (df["GP"] >= MIN_GAMES)].copy()
    qualified = qualified[["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN"]]
    return _apply_current_team(qualified).reset_index(drop=True)


# ── Axis 1: shot-creation style ─────────────────────────────────────────────

def _axis_shot_creation() -> pd.DataFrame:
    """SELF_CREATION_POSS_SHARE = self-creation POSS / (self-creation +
    assisted POSS), from the same 7 offensive play-type categories the
    design doc lists. A player missing from ALL 7 category files (i.e. no
    qualifying possessions in any of them) gets NaN, not 0/0 -- pandas
    naturally produces NaN for a 0/0 division here, which is the correct
    behavior for "no measurable sample" rather than "0% self-creation."""
    poss_by_type: dict[str, pd.Series] = {}
    for play_type in _SELF_CREATION_TYPES + _ASSISTED_TYPES:
        floor = _OFF_PLAYTYPE_MIN_POSS[play_type]
        df = _combine_traded_stints(pd.read_csv(_OFF_PLAYTYPE_CSV[play_type]))
        qualified = df[
            (df["POSS"] >= floor) & (df["POSS"] * df["GP"] >= _OFF_MIN_TOTAL_POSS)
        ]
        poss_by_type[play_type] = qualified.set_index("PLAYER_NAME")["POSS"]

    wide = pd.DataFrame(poss_by_type)  # NaN where a player didn't qualify for that category
    self_poss = wide[_SELF_CREATION_TYPES].sum(axis=1, min_count=1)
    assisted_poss = wide[_ASSISTED_TYPES].sum(axis=1, min_count=1)
    total = self_poss.add(assisted_poss, fill_value=0)

    # min_count=1 above: sum() of an all-NaN row returns NaN (not 0), so a
    # player with zero qualifying categories on one side stays NaN there
    # rather than being silently treated as 0 possessions in that bucket.
    result = pd.DataFrame({
        "SELF_CREATION_POSS": self_poss,
        "ASSISTED_POSS": assisted_poss,
        "SELF_CREATION_SHARE": self_poss / total,
    })
    return result


# ── Axis 2: playmaking role ─────────────────────────────────────────────────

def _axis_playmaking() -> pd.DataFrame:
    """AST_PCT/AST_TO/AST_RATIO from usage_context (Advanced measure type);
    DRIVE_AST_PCT/DRIVE_PASSES_PCT from drives. usage_context is already
    filtered to the qualified base upstream of the merge in
    build_feature_matrix, so no additional qualification filter is applied
    here beyond drives' own drive-volume floor -- a qualified player who
    doesn't clear the drive-volume floor gets NaN on the two DRIVE_* columns
    only, not on AST_PCT/AST_TO/AST_RATIO (those come from a different,
    unrelated qualification: every base-qualified player has them)."""
    usage = pd.read_csv(_USAGE_CONTEXT_CSV).set_index("PLAYER_NAME")
    ast_cols = usage[["AST_PCT", "AST_TO", "AST_RATIO"]]

    drives = pd.read_csv(_DRIVES_CSV)
    drives = drives.copy()
    drives["TOTAL_DRIVES"] = drives["DRIVES"] * drives["GP"]
    qualified_drives = drives[
        (drives["DRIVES"] >= _MIN_DRIVES_PER_GAME) &
        (drives["TOTAL_DRIVES"] >= _MIN_TOTAL_DRIVES)
    ].set_index("PLAYER_NAME")
    drive_cols = qualified_drives[["DRIVE_AST_PCT", "DRIVE_PASSES_PCT"]]

    return ast_cols.join(drive_cols, how="left")


# ── Axis 3: finishing / rim pressure ────────────────────────────────────────

def _axis_finishing() -> pd.DataFrame:
    """DRIVES, DRIVE_PTS, DRIVE_FG_PCT for players who clear the drive
    qualification floor (same floor as compute_offense.drive_efficiency);
    NaN for players who don't -- a low-drive player isn't "0 rim pressure,"
    they're simply not a high-enough-volume driver to measure this on."""
    drives = pd.read_csv(_DRIVES_CSV)
    drives = drives.copy()
    drives["TOTAL_DRIVES"] = drives["DRIVES"] * drives["GP"]
    qualified = drives[
        (drives["DRIVES"] >= _MIN_DRIVES_PER_GAME) &
        (drives["TOTAL_DRIVES"] >= _MIN_TOTAL_DRIVES)
    ].set_index("PLAYER_NAME")
    return qualified[["DRIVES", "DRIVE_PTS", "DRIVE_FG_PCT"]]


# ── Axis 4: off-ball movement profile ───────────────────────────────────────

def _axis_offball_movement() -> pd.DataFrame:
    """SPOTUP_SHARE vs. CUT_OFFSCREEN_SHARE of combined POSS across
    Spotup/Cut/OffScreen -- reuses the same per-category possession floors
    and qualification logic as Axis 1's playtype pull (kept as a separate
    read here since the two axes group the same three underlying
    categories differently: Axis 1 treats Spotup as "assisted" alongside
    Cut/Handoff/OffScreen; this axis treats Spotup vs. Cut+OffScreen as the
    contrast, per the design doc's own framing, and does not include
    Handoff)."""
    poss_by_type: dict[str, pd.Series] = {}
    for play_type in _OFFBALL_SPOTUP + _OFFBALL_CUT_OFFSCREEN:
        floor = _OFF_PLAYTYPE_MIN_POSS[play_type]
        df = _combine_traded_stints(pd.read_csv(_OFF_PLAYTYPE_CSV[play_type]))
        qualified = df[
            (df["POSS"] >= floor) & (df["POSS"] * df["GP"] >= _OFF_MIN_TOTAL_POSS)
        ]
        poss_by_type[play_type] = qualified.set_index("PLAYER_NAME")["POSS"]

    wide = pd.DataFrame(poss_by_type)
    spotup_poss = wide[_OFFBALL_SPOTUP].sum(axis=1, min_count=1)
    cut_offscreen_poss = wide[_OFFBALL_CUT_OFFSCREEN].sum(axis=1, min_count=1)
    total = spotup_poss.add(cut_offscreen_poss, fill_value=0)

    return pd.DataFrame({
        "SPOTUP_SHARE": spotup_poss / total,
        "CUT_OFFSCREEN_SHARE": cut_offscreen_poss / total,
    })


# ── Axis 5: defensive role ──────────────────────────────────────────────────

def _axis_defense() -> pd.DataFrame:
    """Defensive playtype PPP by category (qualified categories only, NaN
    elsewhere), shot suppression PCT_PLUSMINUS by zone (qualified zones
    only), and hustle rate stats (deflections/contests/boxouts per 36).

    The hustle columns apply hustle_stats_2025_26.csv's OWN MIN>=15/G>=40
    filter, independently of the base pool's usage_context-derived
    qualification -- these are two separately-pulled NBA API sources for
    the same season and do not always report identical G/GP for a given
    player (confirmed case: Cam Thomas shows G=39 in hustle_stats vs.
    GP=42 in usage_context, likely a stint-counting difference around his
    in-season trade). A handful of base-qualified players therefore end up
    NaN on the three hustle columns specifically -- this is real
    cross-source disagreement at the qualification boundary, not a bug,
    and is left as NaN rather than resolved by preferring one source's
    game count over the other's."""
    parts: dict[str, pd.Series] = {}

    for play_type, csv_path in _DEF_PLAYTYPE_CSV.items():
        floor = _DEF_PLAYTYPE_MIN_POSS[play_type]
        df = _combine_traded_stints(pd.read_csv(csv_path))
        qualified = df[df["POSS"] >= floor].set_index("PLAYER_NAME")
        parts[f"DEF_PPP_{play_type.upper()}"] = qualified["PPP"]

    for zone, (csv_path, fga_col, pm_col) in _SHOT_DEFENSE_CSV.items():
        df = pd.read_csv(csv_path)
        qualified = df[df[fga_col] >= _MIN_DEF_FGA].set_index("PLAYER_NAME")
        zone_key = zone.replace(" ", "_").upper()
        parts[f"SHOT_SUPPRESSION_{zone_key}"] = qualified[pm_col]

    hustle = pd.read_csv(_HUSTLE_CSV)
    hustle_qualified = hustle[
        (hustle["MIN"] >= MIN_MINUTES) & (hustle["G"] >= MIN_GAMES)
    ].set_index("PLAYER_NAME").copy()
    parts["DEFLECTIONS_PER36"] = (hustle_qualified["DEFLECTIONS"] / hustle_qualified["MIN"] * 36)
    parts["CONTESTED_SHOTS_PER36"] = (hustle_qualified["CONTESTED_SHOTS"] / hustle_qualified["MIN"] * 36)
    parts["BOX_OUTS_PER36"] = (hustle_qualified["BOX_OUTS"] / hustle_qualified["MIN"] * 36)

    return pd.DataFrame(parts)


# ── Axis 6: efficiency-under-volume ─────────────────────────────────────────

def _axis_efficiency_under_volume() -> pd.DataFrame:
    """TS_PCT, EFG_PCT, USG_PCT directly from usage_context -- every
    base-qualified player has these (no secondary qualification floor)."""
    usage = pd.read_csv(_USAGE_CONTEXT_CSV).set_index("PLAYER_NAME")
    return usage[["TS_PCT", "EFG_PCT", "USG_PCT"]]


# ── Axis 7: pace / tempo fit ────────────────────────────────────────────────

def _axis_pace() -> pd.DataFrame:
    """PACE from usage_context (every base-qualified player has this);
    Transition POSS/PPP from the offensive play-type pull, NaN for players
    who don't clear Transition's own possession floor."""
    usage = pd.read_csv(_USAGE_CONTEXT_CSV).set_index("PLAYER_NAME")
    pace = usage[["PACE"]]

    trans = _combine_traded_stints(pd.read_csv(_OFF_PLAYTYPE_CSV["Transition"]))
    floor = _OFF_PLAYTYPE_MIN_POSS["Transition"]
    trans_qualified = trans[
        (trans["POSS"] >= floor) & (trans["POSS"] * trans["GP"] >= _OFF_MIN_TOTAL_POSS)
    ].set_index("PLAYER_NAME")

    return pace.join(trans_qualified[["POSS", "PPP"]].rename(columns={
        "POSS": "TRANSITION_POSS", "PPP": "TRANSITION_PPP",
    }), how="left")


# ── Assemble the full feature matrix ────────────────────────────────────────

# Columns fed into PCA -- every numeric axis column above except raw
# volume/count columns whose scale is dominated by playing time rather than
# style (DRIVES, DRIVE_PTS, SELF_CREATION_POSS, ASSISTED_POSS,
# TRANSITION_POSS are volume; the _SHARE/_PCT/rate versions of the same
# information are what's kept for PCA, consistent with the design doc's
# "efficiency-under-volume" framing being about RATES, not totals).
PCA_FEATURE_COLUMNS = [
    "SELF_CREATION_SHARE",
    "AST_PCT", "AST_TO", "AST_RATIO", "DRIVE_AST_PCT", "DRIVE_PASSES_PCT",
    "DRIVE_FG_PCT",
    "SPOTUP_SHARE", "CUT_OFFSCREEN_SHARE",
    "DEF_PPP_ISOLATION", "DEF_PPP_PRBALLHANDLER", "DEF_PPP_PRROLLMAN",
    "DEF_PPP_POSTUP", "DEF_PPP_SPOTUP", "DEF_PPP_HANDOFF", "DEF_PPP_OFFSCREEN",
    "SHOT_SUPPRESSION_OVERALL", "SHOT_SUPPRESSION_3_POINTERS",
    "SHOT_SUPPRESSION_2_POINTERS", "SHOT_SUPPRESSION_LESS_THAN_6FT",
    "DEFLECTIONS_PER36", "CONTESTED_SHOTS_PER36", "BOX_OUTS_PER36",
    "TS_PCT", "EFG_PCT", "USG_PCT",
    "PACE", "TRANSITION_PPP",
]


def build_feature_matrix() -> pd.DataFrame:
    """Left-merge all seven axes onto the qualified player base. A player
    absent from a given axis source (didn't clear that axis's own
    qualification floor) gets NaN in that axis's columns -- this is a plain
    pandas left-join, so no fillna/zero-fill happens anywhere in this
    function. Returns one row per base-qualified player (MIN>=15, GP>=40),
    indexed by PLAYER_NAME, with TEAM_ABBREVIATION plus every raw feature
    column from all seven axes."""
    base = _qualified_player_base().set_index("PLAYER_NAME")

    axes = [
        _axis_shot_creation(),
        _axis_playmaking(),
        _axis_finishing(),
        _axis_offball_movement(),
        _axis_defense(),
        _axis_efficiency_under_volume(),
        _axis_pace(),
    ]

    result = base
    for axis_df in axes:
        result = result.join(axis_df, how="left")

    return result.reset_index()


# ── PCA ──────────────────────────────────────────────────────────────────────

REFERENCE_PLAYERS = [
    "Stephen Curry",
    "Nikola Jokic",
    "Alex Caruso",
    "Donovan Clingan",
    "Shai Gilgeous-Alexander",
]


def run_pca(feature_matrix: pd.DataFrame, n_components: int = 8):
    """Standardize PCA_FEATURE_COLUMNS (z-score: mean 0, std 1 -- required
    because these columns are on wildly different native scales, e.g.
    USG_PCT ~0.05-0.35 vs. DEFLECTIONS_PER36 ~0.5-6.5) and fit PCA.

    StandardScaler/PCA cannot run on NaN. Missing values in the columns fed
    into PCA are median-imputed HERE ONLY, on a copy made for this
    statistical step -- the raw feature_matrix returned by
    build_feature_matrix() and printed/saved elsewhere keeps its true NaNs
    untouched. This is a deliberate, documented, isolated exception to the
    "never silently zero missing data" rule: median imputation for PCA
    input is a standard, named statistical technique (not a silent zero),
    it is scoped to a copy, and it is disclosed here rather than buried.

    Returns (pca, scaler, X_scaled, feature_cols_used) where
    feature_cols_used is PCA_FEATURE_COLUMNS filtered to columns that
    actually have at least one non-null value in feature_matrix (guards
    against a fully-empty column breaking the median imputation).
    """
    feature_cols_used = [
        c for c in PCA_FEATURE_COLUMNS
        if c in feature_matrix.columns and feature_matrix[c].notna().any()
    ]
    X = feature_matrix[feature_cols_used].copy()
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    return pca, scaler, X_scaled, feature_cols_used


# ── K-means clustering ──────────────────────────────────────────────────────

HYPOTHESIZED_ARCHETYPES = [
    "movement shooter",
    "primary shot creator",
    "offensive hub",
    "rim-running finisher",
    "3-and-D wing",
    "high-activity defensive specialist",
    "two-way engine",
    "traditional post scorer",
]


def run_kmeans_sweep(X_scaled: np.ndarray, k_range: range = range(3, 13)) -> pd.DataFrame:
    """Fit k-means at each k in k_range on the same standardized,
    median-imputed matrix used for PCA. Returns a DataFrame with inertia
    (within-cluster sum of squares, the elbow-method statistic) and mean
    silhouette score at each k. n_init=10 and a fixed random_state make the
    sweep reproducible run to run."""
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def print_kmeans_sweep(sweep: pd.DataFrame) -> None:
    print("=" * 70)
    print("K-MEANS SWEEP — elbow (inertia/WCSS) and silhouette score per k")
    print("=" * 70)
    print(f"  {'k':>3}  {'inertia (WCSS)':>16}  {'silhouette':>11}")
    for _, row in sweep.iterrows():
        print(f"  {int(row['k']):>3}  {row['inertia']:>16.2f}  {row['silhouette']:>11.4f}")
    print()


def select_k(sweep: pd.DataFrame) -> int:
    """Pick k by highest mean silhouette score -- the standard data-driven
    criterion for "how many clusters does this data actually support,"
    independent of any hypothesized archetype count."""
    return int(sweep.loc[sweep["silhouette"].idxmax(), "k"])


def run_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X_scaled)
    return model


def print_cluster_profiles(
    feature_matrix: pd.DataFrame,
    model: KMeans,
    X_scaled: np.ndarray,
    feature_cols: list[str],
    n_representatives: int = 5,
    n_distinctive_features: int = 6,
) -> None:
    """For each cluster: size, the n_representatives players closest to the
    cluster centroid (Euclidean distance in the same standardized space
    k-means was fit on), and the features whose cluster mean deviates most
    (in standard-deviation units, i.e. z-score of the cluster mean against
    the overall population mean/std) from the overall population -- both
    highest and lowest ends of that ranking, since a cluster can be
    distinctive either by being unusually high OR unusually low on a
    feature."""
    print("=" * 70)
    print(f"CLUSTER PROFILES — k={model.n_clusters}")
    print("=" * 70)

    names = feature_matrix["PLAYER_NAME"].reset_index(drop=True)
    labels = model.labels_
    overall_mean = X_scaled.mean(axis=0)
    overall_std = X_scaled.std(axis=0)
    overall_std[overall_std == 0] = 1.0  # guard, shouldn't occur post-StandardScaler

    for cluster_id in range(model.n_clusters):
        mask = labels == cluster_id
        size = mask.sum()
        centroid = model.cluster_centers_[cluster_id]

        print(f"\n{'─'*70}")
        print(f"Cluster {cluster_id}  —  {size} players")
        print(f"{'─'*70}")

        cluster_points = X_scaled[mask]
        cluster_names = names[mask].reset_index(drop=True)
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        order = np.argsort(dists)[:n_representatives]
        print("  Representative players (closest to centroid):")
        for rank, idx in enumerate(order, start=1):
            print(f"    {rank}. {cluster_names.iloc[idx]}  (dist={dists[idx]:.3f})")

        cluster_mean = cluster_points.mean(axis=0)
        z = (cluster_mean - overall_mean) / overall_std
        z_series = pd.Series(z, index=feature_cols).sort_values(ascending=False)

        print(f"\n  Most distinctive features (cluster mean vs. population, in std units):")
        print("    Highest:")
        for feat, val in z_series.head(n_distinctive_features).items():
            print(f"      {feat:<32} {val:+.2f}")
        print("    Lowest:")
        for feat, val in z_series.tail(n_distinctive_features)[::-1].items():
            print(f"      {feat:<32} {val:+.2f}")
    print()


# ── Diagnostics / reporting ─────────────────────────────────────────────────

def print_explained_variance(pca: PCA) -> None:
    print("=" * 70)
    print("EXPLAINED VARIANCE RATIO PER PRINCIPAL COMPONENT")
    print("=" * 70)
    cumulative = 0.0
    for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        cumulative += ratio
        print(f"  PC{i}: {ratio:.4f}   (cumulative: {cumulative:.4f})")
    print()


def print_loadings(pca: PCA, feature_cols: list[str], n_components: int = 4, top_n: int = 8) -> None:
    print("=" * 70)
    print(f"TOP FEATURE LOADINGS — first {n_components} components")
    print("=" * 70)
    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    for pc in loadings.columns:
        print(f"\n  {pc}:")
        ranked = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index)
        for feat, val in ranked.head(top_n).items():
            print(f"    {feat:<32} {val:+.3f}")
    print()


def _resolve_reference_player(name: str, roster_names: pd.Series) -> str | None:
    """resolve_player_name() first (exact substring match against the live
    roster) since that's the shared, canonical lookup used everywhere else
    in this project. It fails on unaccented input against accented roster
    names (e.g. "Nikola Jokic" vs. the roster's "Nikola Jokić") because it
    does plain substring matching with no accent-folding. Falling back to
    the same accent-tolerant first+last-name match already used in
    scripts/explore_offense.py's PROBE_PLAYERS matching (match first name,
    and last name with any trailing "c" stripped) rather than silently
    dropping the player from the report -- this fallback is local to this
    reference-player lookup, not a change to resolve_player_name itself."""
    canonical = resolve_player_name(name)
    if canonical is not None:
        return canonical

    parts = name.split()
    first, last = parts[0], parts[-1].rstrip("c")
    match = roster_names[roster_names.apply(lambda n: first in n and last in n)]
    if len(match) == 1:
        return match.iloc[0]
    return None


def print_reference_players(
    feature_matrix: pd.DataFrame,
    pca: PCA,
    scaler: StandardScaler,
    feature_cols: list[str],
    n_components_shown: int = 4,
) -> None:
    print("=" * 70)
    print("REFERENCE PLAYERS — raw feature vector + position in reduced space")
    print("=" * 70)

    roster_names = feature_matrix["PLAYER_NAME"]
    for ref_name in REFERENCE_PLAYERS:
        canonical = _resolve_reference_player(ref_name, roster_names)
        if canonical is None:
            print(f"\n  [{ref_name}] — could not resolve to a roster player, skipping]")
            continue

        row = feature_matrix[feature_matrix["PLAYER_NAME"] == canonical]
        if row.empty:
            print(f"\n  [{canonical}] — resolved but not in the qualified feature matrix "
                  f"(does not clear MIN>={MIN_MINUTES}/GP>={MIN_GAMES}), skipping]")
            continue

        print(f"\n{'─'*70}")
        print(f"{canonical}  ({row.iloc[0]['TEAM_ABBREVIATION']})")
        print(f"{'─'*70}")

        print("  Raw feature vector:")
        for col in PCA_FEATURE_COLUMNS:
            if col not in row.columns:
                continue
            val = row.iloc[0][col]
            shown = "NaN (insufficient qualifying sample)" if pd.isna(val) else f"{val:.4f}"
            print(f"    {col:<32} {shown}")

        X_row = row[feature_cols].copy()
        X_row = X_row.fillna(feature_matrix[feature_cols].median(numeric_only=True))
        X_scaled_row = scaler.transform(X_row)
        pcs = pca.transform(X_scaled_row)[0]

        print("\n  Position in reduced space:")
        for i in range(n_components_shown):
            print(f"    PC{i+1}: {pcs[i]:+.3f}")


def print_pc3_separation_check(
    feature_matrix: pd.DataFrame,
    model: KMeans,
    pca: PCA,
    scaler: StandardScaler,
    feature_cols: list[str],
) -> None:
    """PC3 in the PCA run blended 'primary shot creator' and 'offensive
    hub' -- both Jokic and SGA scored high on it despite different styles.
    Check explicitly whether k-means on the full feature space (not just
    the top PCs) puts them in the same or different clusters."""
    print("=" * 70)
    print("PC3 BLEND CHECK — Jokic vs. SGA cluster assignment")
    print("=" * 70)

    names = feature_matrix["PLAYER_NAME"].reset_index(drop=True)
    roster_names = feature_matrix["PLAYER_NAME"]

    for label in ["Nikola Jokic", "Shai Gilgeous-Alexander"]:
        canonical = _resolve_reference_player(label, roster_names)
        if canonical is None:
            print(f"  [{label}] — could not resolve to a roster player, skipping]")
            continue
        idx = names[names == canonical].index
        if len(idx) == 0:
            print(f"  [{canonical}] — not in the qualified feature matrix, skipping]")
            continue
        cluster_id = model.labels_[idx[0]]
        print(f"  {canonical:<28} -> cluster {cluster_id}")

    jokic_idx = names[names == _resolve_reference_player("Nikola Jokic", roster_names)].index
    sga_idx = names[names == _resolve_reference_player("Shai Gilgeous-Alexander", roster_names)].index
    if len(jokic_idx) and len(sga_idx):
        same = model.labels_[jokic_idx[0]] == model.labels_[sga_idx[0]]
        verdict = "SAME cluster -- clustering on the full feature space does NOT separate them either" \
            if same else \
            "DIFFERENT clusters -- the fuller feature space separates them where PC3 alone did not"
        print(f"\n  Verdict: {verdict}")
    print()


def print_archetype_comparison(
    feature_matrix: pd.DataFrame,
    model: KMeans,
) -> None:
    print("=" * 70)
    print("HYPOTHESIZED ARCHETYPES vs. K-MEANS OUTPUT")
    print("=" * 70)
    print(f"  Hypothesized ({len(HYPOTHESIZED_ARCHETYPES)}): " + ", ".join(HYPOTHESIZED_ARCHETYPES))
    print(f"  Data-supported k: {model.n_clusters}")
    print(f"\n  {model.n_clusters} clusters were found vs. {len(HYPOTHESIZED_ARCHETYPES)} hypothesized "
          "archetypes. Cross-reference each cluster's representative players and distinctive\n"
          "  features (printed above) against the hypothesis list by hand -- this function "
          "intentionally does not\n  auto-label clusters, since forcing a label onto a cluster is "
          "exactly the kind of unjustified\n  assumption this analysis is trying to avoid.")
    print()


def main() -> None:
    print("Building feature matrix from Layer 1 axes "
          f"(qualification floor: MIN>={MIN_MINUTES}, GP>={MIN_GAMES})...")
    feature_matrix = build_feature_matrix()
    print(f"Feature matrix: {feature_matrix.shape[0]} qualified players, "
          f"{feature_matrix.shape[1]} columns\n")

    pca, scaler, X_scaled, feature_cols = run_pca(feature_matrix)

    print_explained_variance(pca)
    print_loadings(pca, feature_cols, n_components=4)
    print_reference_players(feature_matrix, pca, scaler, feature_cols, n_components_shown=4)

    sweep = run_kmeans_sweep(X_scaled, range(3, 13))
    print_kmeans_sweep(sweep)

    k = select_k(sweep)
    print(f"Data-supported k (highest silhouette score): {k}\n")

    model = run_kmeans(X_scaled, k)
    print_cluster_profiles(feature_matrix, model, X_scaled, feature_cols)

    print_pc3_separation_check(feature_matrix, model, pca, scaler, feature_cols)
    print_archetype_comparison(feature_matrix, model)

    print("Done.")


if __name__ == "__main__":
    main()
