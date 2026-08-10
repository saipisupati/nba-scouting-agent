import time
import pandas as pd
from nba_api.stats.endpoints import LeagueHustleStatsPlayer

SEASONS = {
    "2025-26": "data/hustle_stats_2025_26.csv",
    "2024-25": "data/hustle_stats_2024_25.csv",
}

PLAYERS = ["Rudy Gobert", "Draymond Green", "Alex Caruso", "Herbert Jones"]

frames = {}

for season, csv_path in SEASONS.items():
    print(f"\nFetching {season}...")
    try:
        result = LeagueHustleStatsPlayer(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_time="PerGame",
        )
        df = result.get_data_frames()[0]
        if df.empty:
            print(f"  No data returned for {season}.")
            frames[season] = pd.DataFrame()
        else:
            df.to_csv(csv_path, index=False)
            print(f"  Saved {len(df)} rows → {csv_path}")
            frames[season] = df
    except Exception as e:
        print(f"  Error fetching {season}: {e}")
        frames[season] = pd.DataFrame()
    time.sleep(1)  # be polite to the NBA API

# Print columns from whichever season returned data
sample = next((f for f in frames.values() if not f.empty), None)
if sample is not None:
    print("\n--- Available columns ---")
    for col in sample.columns:
        print(f"  {col}")

# Side-by-side comparison
STAT_COLS = [
    "PLAYER_NAME",
    "CONTESTED_SHOTS",
    "CONTESTED_SHOTS_2PT",
    "CONTESTED_SHOTS_3PT",
    "DEFLECTIONS",
    "CHARGES_DRAWN",
    "SCREEN_ASSISTS",
    "SCREEN_AST_PTS",
    "BOX_OUTS",
    "OFF_BOXOUTS",
    "DEF_BOXOUTS",
    "BOX_OUT_PLAYER_REBS",
    "LOOSE_BALLS_RECOVERED",
]

print("\n\n--- Year-over-year hustle stats ---")
for player in PLAYERS:
    print(f"\n{'='*60}")
    print(f"  {player}")
    print(f"{'='*60}")

    for season, df in frames.items():
        if df.empty:
            print(f"  [{season}] No data available")
            continue
        row = df[df["PLAYER_NAME"] == player]
        if row.empty:
            print(f"  [{season}] Player not found in dataset")
            continue

        print(f"\n  [{season}]")
        existing_cols = [c for c in STAT_COLS if c in row.columns and c != "PLAYER_NAME"]
        for col in existing_cols:
            val = row[col].values[0]
            print(f"    {col:<30} {val}")
