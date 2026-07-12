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


if __name__ == "__main__":
    hustle = pd.read_csv("hustle_stats_2025_26.csv")
    defend_overall = pd.read_csv("shot_defense_overall_2025_26.csv")
    defend_rim     = pd.read_csv("shot_defense_rim_2025_26.csv")

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
