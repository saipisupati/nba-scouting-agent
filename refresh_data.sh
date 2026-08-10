#!/usr/bin/env bash
# Refresh all data CSVs used by the app, in one command.
#
# Runs each pull/exploration script as a subprocess (not an import) since
# none of them are written as importable modules — each is a standalone
# script that fetches its own endpoint(s) and writes its own CSV(s) on
# execution. Order doesn't matter functionally (each script is independent),
# but hustle/shot-defense/play-type run first since they're fast and reliable
# against nba_api; the college draft-class pull runs last since it's slow
# (~8-10+ minutes) and rate-limit sensitive against sports-reference.com.
#
# Usage:
#   ./refresh_data.sh              # refresh everything
#   ./refresh_data.sh --skip-college   # skip the slow college draft-class pull

set -euo pipefail
cd "$(dirname "$0")"

SKIP_COLLEGE=false
for arg in "$@"; do
  case "$arg" in
    --skip-college) SKIP_COLLEGE=true ;;
  esac
done

run() {
  echo ""
  echo "=================================================================="
  echo "Running: $1"
  echo "=================================================================="
  python3 "$1"
}

run hustle_stats.py
run scripts/explore_shot_defense.py
run scripts/explore_playtype_defense.py
run scripts/explore_offense.py

if [ "$SKIP_COLLEGE" = false ]; then
  echo ""
  echo "=================================================================="
  echo "Running: pull_2026_draft_class.py (slow — ~8-10+ min, rate-limited)"
  echo "=================================================================="
  python3 scripts/pull_2026_draft_class.py
else
  echo ""
  echo "Skipping pull_2026_draft_class.py (--skip-college)"
fi

echo ""
echo "=================================================================="
echo "Generating data_manifest.json"
echo "=================================================================="
python3 generate_manifest.py

echo ""
echo "Data refresh complete."
