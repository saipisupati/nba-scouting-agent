import time
import pandas as pd
from nba_api.stats.endpoints import LeagueDashPtDefend

SEASON = "2025-26"
PLAYERS = ["Rudy Gobert", "Draymond Green", "Alex Caruso", "Herbert Jones"]

# (category, csv_path, defender_fgpct_col, normal_fgpct_col, plusminus_col)
CATEGORIES = [
    ("Overall",       "data/shot_defense_overall_2025_26.csv", "D_FG_PCT",   "NORMAL_FG_PCT", "PCT_PLUSMINUS"),
    ("3 Pointers",    "data/shot_defense_3pt_2025_26.csv",     "FG3_PCT",    "NS_FG3_PCT",    "PLUSMINUS"),
    ("2 Pointers",    "data/shot_defense_2pt_2025_26.csv",     "FG2_PCT",    "NS_FG2_PCT",    "PLUSMINUS"),
    ("Less Than 6Ft", "data/shot_defense_rim_2025_26.csv",     "LT_06_PCT",  "NS_LT_06_PCT",  "PLUSMINUS"),
]

for category, csv_path, fgpct_col, normal_col, diff_col in CATEGORIES:
    print(f"\n{'='*60}")
    print(f"  Category: {category}")
    print(f"{'='*60}")

    try:
        result = LeagueDashPtDefend(
            season=SEASON,
            season_type_all_star="Regular Season",
            per_mode_simple="Totals",
            defense_category=category,
        )
        df = result.get_data_frames()[0]
    except Exception as e:
        print(f"  Error fetching {category}: {e}")
        time.sleep(2)
        continue

    if df.empty:
        print(f"  No data returned.")
        time.sleep(2)
        continue

    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} rows → {csv_path}")
    print(f"\n  Columns: {list(df.columns)}")

    print(f"\n  --- Signal breakdown ({fgpct_col} / {normal_col} / {diff_col}) ---")
    print(f"  (PLUSMINUS < 0 = shooter performs worse than normal = good defense)\n")

    for player in PLAYERS:
        row = df[df["PLAYER_NAME"] == player]
        if row.empty:
            print(f"  {player}: not found in dataset")
            continue
        r = row.iloc[0]
        d_fgpct      = f"{r[fgpct_col]:.3f}"  if fgpct_col  in df.columns else "N/A"
        normal_fgpct = f"{r[normal_col]:.3f}" if normal_col  in df.columns else "N/A"
        plusminus    = f"{r[diff_col]:+.3f}"  if diff_col    in df.columns else "N/A"
        print(f"  {player:<20}  DEF_FG%={d_fgpct}  NORMAL={normal_fgpct}  DIFF={plusminus}")

    time.sleep(2)

print("\nDone.")
