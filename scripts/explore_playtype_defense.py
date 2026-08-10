"""
Pull SynergyPlayTypes defensive data (PlayerOrTeam='P') for the 2025-26 season.
One CSV per play type; prints PPP, FG_PCT, POSS for four reference players.

Confirmed: PlayerOrTeam='P' (player_or_team_abbreviation='P') returns player-level
rows with PLAYER_ID and PLAYER_NAME — not team-level aggregates.
"""

import time
import pandas as pd
from nba_api.stats.endpoints import synergyplaytypes

SEASON = "2025-26"
PLAY_TYPES = [
    "Isolation",
    "PRBallHandler",
    "PRRollman",
    "Postup",
    "Spotup",
    "Handoff",
    "Cut",
    "OffScreen",
    "Transition",
]
PROBE_PLAYERS = ["Rudy Gobert", "Draymond Green", "Alex Caruso", "Herbert Jones"]
SHOW_COLS = ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "POSS", "PPP", "FG_PCT", "PERCENTILE"]


def pull_playtype(play_type: str) -> pd.DataFrame:
    r = synergyplaytypes.SynergyPlayTypes(
        player_or_team_abbreviation="P",
        play_type_nullable=play_type,
        type_grouping_nullable="Defensive",
        season=SEASON,
        per_mode_simple="Totals",
    )
    return r.get_data_frames()[0]


for i, play_type in enumerate(PLAY_TYPES):
    print(f"\n{'='*70}")
    print(f"PLAY TYPE: {play_type}  (Defensive, Player-level, {SEASON})")
    print(f"{'='*70}")

    df = pull_playtype(play_type)

    csv_name = f"data/playtype_defense_{play_type.lower()}_{SEASON.replace('-', '_')}.csv"
    df.to_csv(csv_name, index=False)

    print(f"Shape: {df.shape}   |   Saved → {csv_name}")
    print(f"Columns: {list(df.columns)}")

    # Lower PPP = better defender (fewer points allowed per possession)
    # Sort ascending so best defenders appear first
    probe = df[df["PLAYER_NAME"].isin(PROBE_PLAYERS)].copy()
    if probe.empty:
        print("\n  [none of the four probe players appear in this play type's data]")
    else:
        available_cols = [c for c in SHOW_COLS if c in probe.columns]
        print(f"\n  Reference players (lower PPP = better defender):")
        print(
            probe[available_cols]
            .sort_values("PPP")
            .to_string(index=False)
        )

    if i < len(PLAY_TYPES) - 1:
        time.sleep(1.5)

print("\n\nDone. All 9 play-type CSVs saved.")
