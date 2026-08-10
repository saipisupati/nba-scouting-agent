import pandas as pd


def deflections_per36(df: pd.DataFrame, min_minutes: float = 15, min_games: int = 40) -> pd.DataFrame:
    d = df[(df["MIN"] >= min_minutes) & (df["G"] >= min_games)].copy()
    d["DEFLECTIONS_PER36"] = (d["DEFLECTIONS"] / d["MIN"] * 36).round(2)
    return (
        d[["PLAYER_NAME", "TEAM_ABBREVIATION", "G", "MIN", "DEFLECTIONS_PER36"]]
        .sort_values("DEFLECTIONS_PER36", ascending=False)
        .reset_index(drop=True)
    )


def contest_profile_per36(df: pd.DataFrame, min_minutes: float = 15, min_games: int = 40) -> pd.DataFrame:
    d = df[(df["MIN"] >= min_minutes) & (df["G"] >= min_games)].copy()
    d["CONTESTED_2PT_PER36"] = (d["CONTESTED_SHOTS_2PT"] / d["MIN"] * 36).round(2)
    d["CONTESTED_3PT_PER36"] = (d["CONTESTED_SHOTS_3PT"] / d["MIN"] * 36).round(2)
    d["TOTAL_CONTESTED_PER36"] = (d["CONTESTED_SHOTS"] / d["MIN"] * 36).round(2)
    return (
        d[
            [
                "PLAYER_NAME",
                "TEAM_ABBREVIATION",
                "G",
                "MIN",
                "CONTESTED_2PT_PER36",
                "CONTESTED_3PT_PER36",
                "TOTAL_CONTESTED_PER36",
            ]
        ]
        .sort_values("TOTAL_CONTESTED_PER36", ascending=False)
        .reset_index(drop=True)
    )


def boxout_conversion(df: pd.DataFrame, min_boxouts: int = 20, min_games: int = 40) -> pd.DataFrame:
    d = df[(df["BOX_OUTS"] * df["G"] >= min_boxouts) & (df["G"] >= min_games)].copy()
    d["BOXOUT_CONV_RATE"] = (d["BOX_OUT_PLAYER_REBS"] / d["BOX_OUTS"]).round(4)
    return (
        d[
            [
                "PLAYER_NAME",
                "TEAM_ABBREVIATION",
                "G",
                "BOX_OUTS",
                "BOX_OUT_PLAYER_REBS",
                "BOXOUT_CONV_RATE",
                "PCT_BOX_OUTS_REB",
            ]
        ]
        .sort_values("BOXOUT_CONV_RATE", ascending=False)
        .reset_index(drop=True)
    )


def hustle_iq_composite(df: pd.DataFrame, min_minutes: float = 15, min_games: int = 40) -> pd.DataFrame:
    d = df[(df["MIN"] >= min_minutes) & (df["G"] >= min_games)].copy()
    d["DEF_LOOSE_BALLS_PER36"] = (d["DEF_LOOSE_BALLS_RECOVERED"] / d["MIN"] * 36).round(3)
    d["CHARGES_PER36"] = (d["CHARGES_DRAWN"] / d["MIN"] * 36).round(3)

    for col in ("DEF_LOOSE_BALLS_PER36", "CHARGES_PER36"):
        std = d[col].std()
        mean = d[col].mean()
        d[f"_z_{col}"] = (d[col] - mean) / std if std > 0 else 0.0

    d["HUSTLE_IQ_COMPOSITE"] = (
        0.6 * d["_z_DEF_LOOSE_BALLS_PER36"] + 0.4 * d["_z_CHARGES_PER36"]
    ).round(3)

    return (
        d[
            [
                "PLAYER_NAME",
                "TEAM_ABBREVIATION",
                "G",
                "MIN",
                "DEF_LOOSE_BALLS_PER36",
                "CHARGES_PER36",
                "HUSTLE_IQ_COMPOSITE",
            ]
        ]
        .sort_values("HUSTLE_IQ_COMPOSITE", ascending=False)
        .reset_index(drop=True)
    )


_DEFEND_SCHEMA = {
    "Overall":        ("D_FGA",     "D_FG_PCT",  "NORMAL_FG_PCT", "PCT_PLUSMINUS"),
    "3 Pointers":     ("FG3A",      "FG3_PCT",   "NS_FG3_PCT",    "PLUSMINUS"),
    "2 Pointers":     ("FG2A",      "FG2_PCT",   "NS_FG2_PCT",    "PLUSMINUS"),
    "Less Than 6Ft":  ("FGA_LT_06", "LT_06_PCT", "NS_LT_06_PCT",  "PLUSMINUS"),
}


def shot_suppression(
    df: pd.DataFrame,
    category: str = "Overall",
    min_def_fga: int = 100,
) -> pd.DataFrame:
    if category not in _DEFEND_SCHEMA:
        raise ValueError(f"category must be one of {list(_DEFEND_SCHEMA)}")
    fga_col, fgpct_col, normal_col, pm_col = _DEFEND_SCHEMA[category]

    d = df[df[fga_col] >= min_def_fga].copy()
    d = d.rename(columns={fgpct_col: "DEF_FG_PCT", normal_col: "NORMAL_FG_PCT", pm_col: "PCT_PLUSMINUS"})
    d["DEF_FG_PCT"]    = d["DEF_FG_PCT"].round(3)
    d["NORMAL_FG_PCT"] = d["NORMAL_FG_PCT"].round(3)
    d["PCT_PLUSMINUS"] = d["PCT_PLUSMINUS"].round(3)

    keep = ["PLAYER_NAME", "PLAYER_LAST_TEAM_ABBREVIATION", "G", fga_col,
            "DEF_FG_PCT", "NORMAL_FG_PCT", "PCT_PLUSMINUS"]
    return (
        d[keep]
        .sort_values("PCT_PLUSMINUS", ascending=True)
        .reset_index(drop=True)
    )


def hustle_vs_suppression_gap(
    hustle_df: pd.DataFrame,
    defend_df: pd.DataFrame,
    min_minutes: float = 15,
    min_def_fga: int = 100,
    min_games: int = 40,
) -> pd.DataFrame:
    fga_col, _, _, pm_col = _DEFEND_SCHEMA["Overall"]

    h = hustle_df[(hustle_df["MIN"] >= min_minutes) & (hustle_df["G"] >= min_games)].copy()
    h["DEFLECTIONS_PER36"]     = (h["DEFLECTIONS"] / h["MIN"] * 36).round(2)
    h["TOTAL_CONTESTED_PER36"] = (h["CONTESTED_SHOTS"] / h["MIN"] * 36).round(2)

    s = defend_df[(defend_df[fga_col] >= min_def_fga) & (defend_df["G"] >= min_games)].copy()
    s = s.rename(columns={pm_col: "PCT_PLUSMINUS"})

    # inner join — players who pass ALL filters; bring in PLAYER_POSITION from defend_df
    merged = h.merge(
        s[["PLAYER_NAME", "PCT_PLUSMINUS", "PLAYER_POSITION"]],
        on="PLAYER_NAME",
        how="inner",
    )

    # rank within each position group so guards aren't penalized for low FGA counts
    # and bigs aren't penalized for low deflection rates
    grp = merged.groupby("PLAYER_POSITION")
    merged["HUSTLE_ACTIVITY_RANK"] = (
        grp["DEFLECTIONS_PER36"].rank(ascending=False) +
        grp["TOTAL_CONTESTED_PER36"].rank(ascending=False)
    ) / 2
    merged["SUPPRESSION_RANK"] = grp["PCT_PLUSMINUS"].rank(ascending=True)

    # GAP = HUSTLE_ACTIVITY_RANK − SUPPRESSION_RANK (within position group)
    # rank=1 means best in both dimensions, so:
    # positive GAP: hustle rank > suppression rank → low activity, good outcomes (quiet but effective)
    # negative GAP: hustle rank < suppression rank → high activity, poor outcomes (busy but not impactful)
    merged["GAP"] = (merged["HUSTLE_ACTIVITY_RANK"] - merged["SUPPRESSION_RANK"]).round(1)

    return (
        merged[[
            "PLAYER_NAME", "TEAM_ABBREVIATION", "PLAYER_POSITION",
            "DEFLECTIONS_PER36", "TOTAL_CONTESTED_PER36",
            "HUSTLE_ACTIVITY_RANK", "PCT_PLUSMINUS", "SUPPRESSION_RANK", "GAP",
        ]]
        .sort_values("GAP", ascending=False)
        .reset_index(drop=True)
    )


_PLAYTYPE_CSV = {
    "Isolation":    "data/playtype_defense_isolation_2025_26.csv",
    "PRBallHandler":"data/playtype_defense_prballhandler_2025_26.csv",
    "PRRollman":    "data/playtype_defense_prrollman_2025_26.csv",
    "Postup":       "data/playtype_defense_postup_2025_26.csv",
    "Spotup":       "data/playtype_defense_spotup_2025_26.csv",
    "Handoff":      "data/playtype_defense_handoff_2025_26.csv",
    "OffScreen":    "data/playtype_defense_offscreen_2025_26.csv",
}

# POSS distributions (2025-26, all qualified players):
#   Isolation     n=394  min=10  p25=25  p35=30  p40=33  median=40  p75=58   max=156
#   PRBallHandler n=415  min=13  p25=89  p35=112 p40=124 median=156 p75=226  max=806
#   PRRollman     n=386  min=10  p25=27  p35=32  p40=35  median=41  p75=59   max=214
#   Postup        n=329  min=10  p25=16  p35=18  p40=19  median=22  p75=31   max=64
#   Spotup        n=419  min=11  p25=76  p35=97  p40=107 median=128 p75=198  max=378
#   Handoff       n=313  min=10  p25=19  p35=22  p40=24  median=31  p75=45   max=98
#   OffScreen     n=317  min=10  p25=16  p35=18  p40=20  median=24  p75=36   max=85
# Isolation/PRBallHandler/PRRollman/Spotup: p25 (sanity-checked, retained as-is).
# Postup/Handoff/OffScreen raised to p40 — sparse distributions where p25 let through
# genuine single-game samples (e.g. 16 poss for Raynaud in OffScreen leading the list).
_PLAYTYPE_DEFAULT_MIN_POSS = {
    "Isolation":    25,
    "PRBallHandler": 90,
    "PRRollman":    27,
    "Postup":       19,
    "Spotup":       76,
    "Handoff":      24,
    "OffScreen":    20,
}

# Below this possession count the answer text flags "small sample".
# Set at 30: comfortably above the p40 floors for sparse types, and meaningful
# even in high-volume types (e.g. an Isolation result with 27 poss is still thin).
SMALL_SAMPLE_THRESHOLD = 30

_NO_DATA_TYPES = {"Cut", "Transition"}


def playtype_defense(play_type: str, min_poss: int = None) -> pd.DataFrame:
    """Return players ranked by PPP allowed (ascending) for a given Synergy play type.

    Lower PPP = better defender — the player allowed fewer points per possession
    when the offense ran this play type at them.

    Parameters
    ----------
    play_type : str
        One of: Isolation, PRBallHandler, PRRollman, Postup, Spotup, Handoff, OffScreen.
        Cut and Transition have no player-level data in the Synergy feed and will raise ValueError.

        PRRollman caveat: PPP allowed on logged roll-man possessions reflects both
        defensive quality AND selection effects. Elite rim protectors may show middling
        numbers here because opponents avoid attacking them in this play type — only the
        possessions where the offense chose to attack get logged. Do not read this
        category as "best rim protector." Pair it with shot_suppression('Less Than 6Ft')
        for a fuller picture of interior defense.
    min_poss : int, optional
        Minimum possessions to qualify. Defaults to the ~p25 of each play type's
        distribution (see _PLAYTYPE_DEFAULT_MIN_POSS), which retains ~75% of players
        while filtering out true single-game samples.

    Returns
    -------
    DataFrame with columns: PLAYER_NAME, TEAM_ABBREVIATION, POSS, PPP, FG_PCT
        sorted ascending by PPP (best defenders first).
    """
    if play_type in _NO_DATA_TYPES:
        raise ValueError(
            f"'{play_type}' has no player-level defensive data in the Synergy feed. "
            f"Valid play types: {sorted(_PLAYTYPE_CSV)}"
        )
    if play_type not in _PLAYTYPE_CSV:
        raise ValueError(
            f"Unknown play type '{play_type}'. "
            f"Valid: {sorted(_PLAYTYPE_CSV)} (Cut and Transition have no data)."
        )

    threshold = min_poss if min_poss is not None else _PLAYTYPE_DEFAULT_MIN_POSS[play_type]
    df = pd.read_csv(_PLAYTYPE_CSV[play_type])
    d = df[df["POSS"] >= threshold].copy()
    return (
        d[["PLAYER_NAME", "TEAM_ABBREVIATION", "POSS", "PPP", "FG_PCT"]]
        .sort_values("PPP", ascending=True)
        .reset_index(drop=True)
    )


# Maps metric name → (function, primary metric column to diff)
# contest_profile_per36 has three rate columns; TOTAL_CONTESTED_PER36 is the
# most natural single-number summary for year-over-year comparison.
_YOY_METRIC_MAP = {
    "deflections_per36":   (deflections_per36,   "DEFLECTIONS_PER36"),
    "contest_profile_per36": (contest_profile_per36, "TOTAL_CONTESTED_PER36"),
    "boxout_conversion":   (boxout_conversion,   "BOXOUT_CONV_RATE"),
    "hustle_iq_composite": (hustle_iq_composite, "HUSTLE_IQ_COMPOSITE"),
}


def year_over_year_delta(
    current_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    metric: str = "deflections_per36",
    min_minutes: float = 15,
    min_games: int = 40,
) -> pd.DataFrame:
    """Compare a hustle metric between two seasons and rank by change.

    Returns players sorted by DELTA descending (biggest improvers first).
    Negative DELTA = declined. Only players who pass the min_minutes / min_games
    filter in BOTH seasons are included — a player with 10 prior-season games
    is not a meaningful comparison point and will be excluded.

    Parameters
    ----------
    current_df : DataFrame from the current season's hustle CSV.
    prior_df   : DataFrame from the prior season's hustle CSV.
    metric     : one of 'deflections_per36', 'contest_profile_per36',
                 'boxout_conversion', 'hustle_iq_composite'.
    min_minutes, min_games : qualification thresholds applied to BOTH seasons.

    Returns
    -------
    DataFrame with columns:
        PLAYER_NAME, TEAM_CUR, G_CUR, G_PRIOR, <metric_col>_CUR,
        <metric_col>_PRIOR, DELTA
    sorted by DELTA descending.
    """
    if metric not in _YOY_METRIC_MAP:
        raise ValueError(f"metric must be one of {sorted(_YOY_METRIC_MAP)}")

    fn, metric_col = _YOY_METRIC_MAP[metric]

    # boxout_conversion uses a different filter signature; pass through cleanly
    if metric == "boxout_conversion":
        cur = fn(current_df)
        pri = fn(prior_df)
    else:
        cur = fn(current_df, min_minutes=min_minutes, min_games=min_games)
        pri = fn(prior_df,   min_minutes=min_minutes, min_games=min_games)

    merged = cur.merge(
        pri[["PLAYER_NAME", "G", metric_col]],
        on="PLAYER_NAME",
        how="inner",
        suffixes=("_CUR", "_PRIOR"),
    )

    merged["DELTA"] = (merged[f"{metric_col}_CUR"] - merged[f"{metric_col}_PRIOR"]).round(3)

    keep = ["PLAYER_NAME", "TEAM_ABBREVIATION", f"G_CUR", f"G_PRIOR",
            f"{metric_col}_CUR", f"{metric_col}_PRIOR", "DELTA"]
    return (
        merged[keep]
        .rename(columns={"TEAM_ABBREVIATION": "TEAM_CUR"})
        .sort_values("DELTA", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    hustle = pd.read_csv("data/hustle_stats_2025_26.csv")
    defend_overall = pd.read_csv("data/shot_defense_overall_2025_26.csv")
    defend_rim     = pd.read_csv("data/shot_defense_rim_2025_26.csv")

    sections = [
        ("Deflections per 36", deflections_per36(hustle)),
        ("Contest Profile per 36", contest_profile_per36(hustle)),
        ("Box-Out Conversion (min 20 box outs this season)", boxout_conversion(hustle)),
        ("Hustle IQ Composite", hustle_iq_composite(hustle)),
    ]

    for label, result in sections:
        print(f"\n--- {label} (Top 10) ---")
        if label.startswith("Hustle IQ"):
            print("  NOTE: HUSTLE_IQ_COMPOSITE is a weighted z-score composite")
            print("  (60% def loose balls per 36 + 40% charges drawn per 36).")
            print("  This is NOT an official NBA stat.\n")
        print(result.head(10).to_string(index=False))

    print("\n\n--- Shot Suppression: Overall (Top 10, min 100 defended FGA) ---")
    print("  (PCT_PLUSMINUS < 0 = shooter performs worse vs. this defender)\n")
    print(shot_suppression(defend_overall, category="Overall").head(10).to_string(index=False))

    print("\n\n--- Shot Suppression: Rim <6ft (Top 10, min 100 defended FGA) ---")
    print("  (PCT_PLUSMINUS < 0 = rim attempts go in less often vs. this defender)\n")
    print(shot_suppression(defend_rim, category="Less Than 6Ft").head(10).to_string(index=False))

    print("\n\n--- Hustle vs. Suppression Gap ---")
    print("  GAP > 0: low hustle activity, good shot suppression (quiet but effective)")
    print("  GAP < 0: high hustle activity, poor shot suppression (busy but not impactful)\n")
    gap_df = hustle_vs_suppression_gap(hustle, defend_overall)
    n = len(gap_df)
    print("  Top 10 POSITIVE gap (low hustle, good suppression — quiet but effective):")
    print(gap_df.head(10).to_string(index=False))
    print("\n  Top 10 NEGATIVE gap (high hustle, poor suppression — busy but not impactful):")
    print(gap_df.tail(10).sort_values("GAP").to_string(index=False))
