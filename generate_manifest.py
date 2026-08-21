"""
Generates data_manifest.json: for every CSV this app depends on, records
the source endpoint, season, extraction timestamp (this script's own run
time -- the actual moment the file was last (re)written to disk, which is
what "when was this data last pulled" actually means, not a guess), and
row count.

Run by refresh_data.sh after all the pull scripts finish -- this script
itself does not pull anything, it just inventories what's already on disk
at the moment it runs, so the manifest reflects the state of the CSVs as
of the most recent refresh, whatever that was.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

_MANIFEST_PATH = "data/data_manifest.json"

# (file, source endpoint, season) for every CSV this app depends on.
_SOURCES = [
    ("data/hustle_stats_2025_26.csv", "LeagueHustleStatsPlayer", "2025-26"),
    ("data/hustle_stats_2024_25.csv", "LeagueHustleStatsPlayer", "2024-25"),
    ("data/shot_defense_overall_2025_26.csv", "LeagueDashPtDefend (Overall)", "2025-26"),
    ("data/shot_defense_3pt_2025_26.csv", "LeagueDashPtDefend (3 Pointers)", "2025-26"),
    ("data/shot_defense_2pt_2025_26.csv", "LeagueDashPtDefend (2 Pointers)", "2025-26"),
    ("data/shot_defense_rim_2025_26.csv", "LeagueDashPtDefend (Less Than 6Ft)", "2025-26"),
    ("data/drives_2025_26.csv", "LeagueDashPtStats (Drives)", "2025-26"),
    ("data/usage_context_2025_26.csv", "LeagueDashPlayerStats (Advanced)", "2025-26"),
    ("data/draft_class_2026.csv", "sports-reference.com/cbb (via pull_2026_draft_class.py)", "2025-26 college season"),
]

_PLAYTYPE_FILES = ["isolation", "prballhandler", "prrollman", "postup",
                    "spotup", "handoff", "cut", "offscreen", "transition"]
for suffix in _PLAYTYPE_FILES:
    _SOURCES.append((f"data/playtype_defense_{suffix}_2025_26.csv", "SynergyPlayTypes (Defensive)", "2025-26"))
    _SOURCES.append((f"data/playtype_offense_{suffix}_2025_26.csv", "SynergyPlayTypes (Offensive)", "2025-26"))


def build_manifest() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    files = {}
    for path, endpoint, season in _SOURCES:
        try:
            df = pd.read_csv(path)
            row_count = len(df)
        except FileNotFoundError:
            row_count = None
        files[path] = {
            "source_endpoint": endpoint,
            "season": season,
            "extracted_at": now,
            "row_count": row_count,
        }
    return {"generated_at": now, "files": files}


if __name__ == "__main__":
    manifest = build_manifest()
    with open(_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    n_files = len(manifest["files"])
    n_missing = sum(1 for v in manifest["files"].values() if v["row_count"] is None)
    print(f"Wrote {_MANIFEST_PATH}: {n_files} files recorded"
          + (f" ({n_missing} missing on disk)" if n_missing else ""))
