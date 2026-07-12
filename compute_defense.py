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


if __name__ == "__main__":
    df = pd.read_csv("hustle_stats_2025_26.csv")

    sections = [
        ("Deflections per 36", deflections_per36(df)),
        ("Contest Profile per 36", contest_profile_per36(df)),
        ("Box-Out Conversion (min 20 box outs this season)", boxout_conversion(df)),
        ("Hustle IQ Composite", hustle_iq_composite(df)),
    ]

    for label, result in sections:
        print(f"\n--- {label} (Top 10) ---")
        if label.startswith("Hustle IQ"):
            print("  NOTE: HUSTLE_IQ_COMPOSITE is a weighted z-score composite")
            print("  (60% def loose balls per 36 + 40% charges drawn per 36).")
            print("  This is NOT an official NBA stat.\n")
        print(result.head(10).to_string(index=False))
