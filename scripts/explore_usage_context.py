"""
Pull usage-rate context for the 2025-26 season:
  LeagueDashPlayerStats, measure_type_detailed_defense='Advanced'
  — USG_PCT (usage rate) and TEAM_ABBREVIATION for every player

This is the "who's the ISO-heavy, ball-dominant guy on this team" context —
per-team USG_PCT rankings to identify each roster's primary usage player.
"""

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

SEASON = "2025-26"

REFERENCE_PLAYERS = [
    "Alex Caruso",
    "Donovan Clingan",
    "Stephen Curry",
    "Nikola Jokic",
    "Shai Gilgeous-Alexander",
]


# ── Pull ────────────────────────────────────────────────────────────────────

print("=" * 70)
print("LeagueDashPlayerStats: Advanced, Player-level, 2025-26")
print("=" * 70)

df = leaguedashplayerstats.LeagueDashPlayerStats(
    season=SEASON,
    measure_type_detailed_defense="Advanced",
    per_mode_detailed="PerGame",
).get_data_frames()[0]

print(f"\nShape: {df.shape}")
print(f"All columns: {list(df.columns)}")

if "USG_PCT" in df.columns:
    print("\nUSG_PCT confirmed present.")
else:
    print("\n*** USG_PCT NOT FOUND IN RESPONSE ***")
    raise SystemExit(1)


# ── Reference players ───────────────────────────────────────────────────────

# NBA API uses accented "Nikola Jokić" — match on first + last with accent-tolerant substring
probe_mask = df["PLAYER_NAME"].apply(
    lambda n: any(ref.split()[0] in n and ref.split()[-1].rstrip("c") in n
                  for ref in REFERENCE_PLAYERS)
)
probe = df[probe_mask].copy()

show_cols = ["PLAYER_NAME", "TEAM_ABBREVIATION", "USG_PCT"]

print(f"\n{'─'*70}")
print("Reference players — USG_PCT and team")
print(f"{'─'*70}")
if probe.empty:
    print("[none of the reference players found]")
else:
    print(probe[show_cols].sort_values("USG_PCT", ascending=False).to_string(index=False))


# ── Per-team top-3 USG_PCT ──────────────────────────────────────────────────

teams = sorted(probe["TEAM_ABBREVIATION"].dropna().unique())

print(f"\n{'='*70}")
print("Top-3 highest-USG_PCT players per reference player's team")
print(f"{'='*70}")

for team in teams:
    team_df = df[df["TEAM_ABBREVIATION"] == team].sort_values("USG_PCT", ascending=False)
    top3 = team_df.head(3)

    print(f"\n{'─'*70}")
    print(f"TEAM: {team}")
    print(f"{'─'*70}")
    print(top3[show_cols].to_string(index=False))

    ref_names_on_team = probe[probe["TEAM_ABBREVIATION"] == team]["PLAYER_NAME"].tolist()
    for ref_name in ref_names_on_team:
        rank = (team_df["PLAYER_NAME"] == ref_name).values
        if rank.any():
            position = team_df.reset_index(drop=True)
            idx = position.index[position["PLAYER_NAME"] == ref_name][0]
            usg = position.loc[idx, "USG_PCT"]
            print(f"  -> {ref_name}: rank #{idx + 1} on {team} by USG_PCT ({usg:.3f})")


# ── Sanity check: OKC / SGA vs. Caruso ──────────────────────────────────────

print(f"\n{'='*70}")
print("SANITY CHECK — OKC: SGA should be clear top-usage player, Caruso much lower")
print(f"{'='*70}")

okc_df = df[df["TEAM_ABBREVIATION"] == "OKC"].sort_values("USG_PCT", ascending=False).reset_index(drop=True)
sga_row = okc_df[okc_df["PLAYER_NAME"].str.contains("Gilgeous-Alexander", na=False)]
caruso_row = okc_df[okc_df["PLAYER_NAME"].str.contains("Caruso", na=False)]

if sga_row.empty or caruso_row.empty:
    print("*** could not locate SGA and/or Caruso in OKC roster — sanity check inconclusive ***")
else:
    sga_idx = sga_row.index[0]
    caruso_idx = caruso_row.index[0]
    sga_usg = sga_row.iloc[0]["USG_PCT"]
    caruso_usg = caruso_row.iloc[0]["USG_PCT"]

    print(f"SGA:    rank #{sga_idx + 1} on OKC, USG_PCT = {sga_usg:.3f}")
    print(f"Caruso: rank #{caruso_idx + 1} on OKC, USG_PCT = {caruso_usg:.3f}")

    if sga_idx == 0 and caruso_idx > sga_idx:
        print("\nPATTERN CONFIRMED: SGA is the clear top-usage player on OKC, Caruso ranks much lower.")
    else:
        print("\n*** PATTERN DID NOT HOLD — investigate before relying on this context ***")

print("\nDone.")
