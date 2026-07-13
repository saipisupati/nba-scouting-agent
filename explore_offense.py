"""
Pull offensive data for the 2025-26 season:
  1. SynergyPlayTypes, TypeGrouping='Offensive', PlayerOrTeam='P'
     — all 9 play types including Cut and Transition (both should have real data
       on the offensive side unlike the defensive pull)
  2. LeagueDashPtStats pt_measure_type='Drives'
     — drive frequency and efficiency per player

CSVs saved:
  playtype_offense_{type}_2025_26.csv  (one per play type)
  drives_2025_26.csv
"""

import time
import pandas as pd
from nba_api.stats.endpoints import synergyplaytypes, leaguedashptstats

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

PROBE_PLAYERS = [
    "Nikola Jokic",
    "Stephen Curry",
    "Shai Gilgeous-Alexander",
    "Jalen Brunson",
]

SHOW_COLS = ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "POSS", "PPP", "FG_PCT", "PERCENTILE"]


# ── Part 1: Offensive play types ──────────────────────────────────────────────

print("=" * 70)
print("PART 1 — SynergyPlayTypes: Offensive, Player-level, 2025-26")
print("=" * 70)

for i, play_type in enumerate(PLAY_TYPES):
    print(f"\n{'─'*70}")
    print(f"PLAY TYPE: {play_type}")
    print(f"{'─'*70}")

    df = synergyplaytypes.SynergyPlayTypes(
        player_or_team_abbreviation="P",
        play_type_nullable=play_type,
        type_grouping_nullable="Offensive",
        season=SEASON,
        per_mode_simple="PerGame",
    ).get_data_frames()[0]

    csv_name = f"playtype_offense_{play_type.lower()}_2025_26.csv"
    df.to_csv(csv_name, index=False)

    if df.empty:
        print(f"  *** EMPTY — no data returned for {play_type} (unexpected on offensive side) ***")
        if i < len(PLAY_TYPES) - 1:
            time.sleep(1)
        continue

    print(f"Shape : {df.shape}   |   Saved → {csv_name}")

    # Confirm key columns present
    expected = {"PPP", "POSS", "FG_PCT"}
    missing = expected - set(df.columns)
    if missing:
        print(f"  *** MISSING EXPECTED COLUMNS: {missing} ***")
    else:
        print(f"Columns confirmed: PPP, POSS, FG_PCT all present")
    print(f"All columns: {list(df.columns)}")

    # Flag Cut / Transition explicitly
    if play_type in ("Cut", "Transition"):
        print(f"  >>> {play_type.upper()} HAS DATA — {len(df)} player rows (unlike the defensive pull)")

    # Reference player rows
    # NBA API uses accented "Nikola Jokić" — match on first + last with accent-tolerant substring
    probe_mask = df["PLAYER_NAME"].apply(
        lambda n: any(ref.split()[0] in n and ref.split()[-1].rstrip("c") in n
                      for ref in PROBE_PLAYERS)
    )
    probe = df[probe_mask].copy()

    if probe.empty:
        print("\n  [none of the four probe players appear in this category]")
    else:
        available = [c for c in SHOW_COLS if c in probe.columns]
        print(f"\n  Reference players (higher PPP = more efficient scorer):")
        print(probe[available].sort_values("PPP", ascending=False).to_string(index=False))

    if i < len(PLAY_TYPES) - 1:
        time.sleep(1.5)


# ── Part 2: Drives ────────────────────────────────────────────────────────────

print(f"\n\n{'='*70}")
print("PART 2 — LeagueDashPtStats: Drives, Player-level, 2025-26")
print("=" * 70)

time.sleep(1.5)

drives_df = leaguedashptstats.LeagueDashPtStats(
    season=SEASON,
    pt_measure_type="Drives",
    player_or_team="Player",
    per_mode_simple="PerGame",
).get_data_frames()[0]

drives_csv = "drives_2025_26.csv"
drives_df.to_csv(drives_csv, index=False)

if drives_df.empty:
    print("*** EMPTY — drives endpoint returned no data ***")
else:
    print(f"Shape : {drives_df.shape}   |   Saved → {drives_csv}")
    print(f"All columns: {list(drives_df.columns)}")

    # Identify key drive efficiency columns
    key_cols = [c for c in drives_df.columns if any(k in c for k in
                ("DRIVE", "PTS", "FG_PCT", "PASS", "TOV", "PF", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP"))]
    print(f"\nDrive-relevant columns: {key_cols}")

    # Reference players
    probe_mask2 = drives_df["PLAYER_NAME"].apply(
        lambda n: any(ref.split()[0] in n and ref.split()[-1].rstrip("c") in n
                      for ref in PROBE_PLAYERS)
    )
    probe2 = drives_df[probe_mask2].copy()

    if probe2.empty:
        print("\n  [none of the four probe players appear in drives data]")
    else:
        print(f"\n  Reference players — drives data:")
        show_drive_cols = [c for c in ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP"] + key_cols
                           if c in probe2.columns and c not in ("PLAYER_NAME",)]
        show_drive_cols = ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP"] + \
                          [c for c in key_cols if c not in ("PLAYER_NAME", "TEAM_ABBREVIATION", "GP")]
        show_drive_cols = list(dict.fromkeys(show_drive_cols))  # dedupe, preserve order
        print(probe2[[c for c in show_drive_cols if c in probe2.columns]]
              .sort_values("DRIVES" if "DRIVES" in probe2.columns else probe2.columns[0], ascending=False)
              .to_string(index=False))

print("\n\nDone.")
