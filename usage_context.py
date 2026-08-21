"""
Pull usage-rate context for the 2025-26 season:
  LeagueDashPlayerStats, measure_type_detailed_defense='Advanced'
  — USG_PCT (usage rate) and TEAM_ABBREVIATION for every player.

This is the "who's the ISO-heavy, ball-dominant guy on this team" context --
used to identify each roster's primary usage player relative to a given
reference player (e.g. Caruso vs. SGA on OKC).

CSV saved:
  data/usage_context_2025_26.csv
"""

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

SEASON = "2025-26"
CSV_PATH = "data/usage_context_2025_26.csv"

df = leaguedashplayerstats.LeagueDashPlayerStats(
    season=SEASON,
    measure_type_detailed_defense="Advanced",
    per_mode_detailed="PerGame",
).get_data_frames()[0]

if df.empty:
    print("*** EMPTY — LeagueDashPlayerStats (Advanced) returned no data ***")
else:
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df)} rows → {CSV_PATH}")
    print(f"Columns: {list(df.columns)}")
