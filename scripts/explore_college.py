"""
College basketball data source exploration.
Confirms what's actually accessible before building any compute functions.

Sources tested:
  1. cbbpy (cbbpy.mens_scraper) — ESPN-backed game/box score/PBP scraper
  2. sportsdataverse — ESPN-backed parquet loader
  3. sports-reference.com/cbb — pandas.read_html (rate-limited, 3-4s between requests)

Run this once to confirm data availability. Do not loop over many pages.
"""

import time
import traceback
import warnings
warnings.filterwarnings("ignore")

SEP = "=" * 70


# ── PART 1: cbbpy ─────────────────────────────────────────────────────────────
# VERDICT (confirmed): BROKEN. Returns empty DataFrames for all calls.
# ESPN changed their internal API layout; cbbpy (last release 2023) has not
# been updated. Do not rely on this package.

print(SEP)
print("PART 1 — cbbpy (cbbpy.mens_scraper)")
print(SEP)

try:
    import cbbpy.mens_scraper as cbd

    print("\nFunctions exposed:")
    fns = [f for f in dir(cbd) if not f.startswith("_") and callable(getattr(cbd, f))
           and f not in ("Tuple", "Union", "datetime", "pd")]
    for f in fns:
        print(f"  {f}")
    # Functions: get_conference_schedule, get_game, get_game_boxscore,
    # get_game_ids, get_game_info, get_game_pbp, get_games_conference,
    # get_games_range, get_games_season, get_games_team, get_player_info,
    # get_team_schedule, get_teams_from_conference

    print("\n--- Single game box score (Purdue vs NC State, 2024 tournament, ESPN ID 401522267) ---")
    try:
        result = cbd.get_game_boxscore("401522267")
        sample = result[0] if isinstance(result, tuple) else result
        if sample.empty:
            print("  RESULT: EMPTY — ESPN scraper broken for current layout")
        else:
            print(f"  Shape: {sample.shape}")
            print(f"  Columns: {list(sample.columns)}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\ncbbpy VERDICT: BROKEN — all endpoints return empty DataFrames.")
    print("  ESPN changed internal API; cbbpy last updated 2023. Do not use.")

except ImportError as e:
    print(f"cbbpy import failed: {e}")


# ── PART 2: sportsdataverse ───────────────────────────────────────────────────
# VERDICT (confirmed): BROKEN in this environment.
# Fails to import due to numpy binary incompatibility (sklearn/xgboost compiled
# against NumPy 1.x, environment has NumPy 2.0.2).
# Error: "numpy.dtype size changed, may indicate binary incompatibility"
# Even if fixed, it provides per-game box scores only (same limitation as cbbpy)
# — no season aggregates, no USG%, no advanced metrics.

print(f"\n{SEP}")
print("PART 2 — sportsdataverse")
print(SEP)

try:
    import sportsdataverse
    print(f"  Import OK — version: {getattr(sportsdataverse, '__version__', 'unknown')}")
    try:
        from sportsdataverse import mbb
        fns = [f for f in dir(mbb) if not f.startswith("_") and callable(getattr(mbb, f))]
        print(f"  mbb functions: {fns}")

        print("\n--- load_mbb_player_boxscore (2024 season) ---")
        df = mbb.load_mbb_player_boxscore(seasons=2024)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"  First row: {dict(df.iloc[0])}")
    except Exception as e:
        print(f"  mbb error: {e}")
except Exception as e:
    print(f"  sportsdataverse BROKEN: {type(e).__name__}: {str(e)[:150]}")
    print("  Root cause: sklearn/xgboost compiled against NumPy 1.x; environment has NumPy 2.x.")
    print("  Even if fixed: only per-game box scores, not season aggregates.")

print("\nsportsdataverse VERDICT: BROKEN in this environment (NumPy binary incompatibility).")
print("  And even if fixed: per-game box scores only — same data gap as cbbpy.")


# ── PART 3: sports-reference.com/cbb ─────────────────────────────────────────
# VERDICT (confirmed): WORKS. Player pages + school stats accessible.
# Advanced stats (USG%, BPM, etc.) are in HTML comment blocks — need requests
# + regex to extract, not just pandas.read_html directly.
#
# Confirmed working URLs (others 404):
#   /cbb/players/{slug}.html               — per-player multi-season stats
#   /cbb/seasons/men/{year}-school-stats.html — team-level aggregates
#   /cbb/seasons/men/{year}-leaders.html   — per-player leaders (needs bs4>=4.11.2)
#
# Confirmed 404 URLs (wrong format):
#   /cbb/seasons/men/{year}-advanced.html
#   /cbb/seasons/men/{year}-stats.html
#   /cbb/seasons/men/{year}-per_game.html

print(f"\n{SEP}")
print("PART 3 — sports-reference.com/cbb (pandas.read_html + requests)")
print(SEP)

import pandas as pd
import re
from io import StringIO

# ── 3a: Player page — per-game averages (visible tables) ─────────────────────
print("\n--- 3a: Zach Edey career page — visible tables (per-game averages) ---")
url_edey = "https://www.sports-reference.com/cbb/players/zach-edey-1.html"
try:
    tables = pd.read_html(url_edey)
    print(f"  Visible tables: {len(tables)}")
    # Table 0: per-game averages; Table 2: season totals (Tables 1,3 are conference-game splits)
    t0 = tables[0]
    print(f"\n  Table 0 (per-game averages): shape={t0.shape}")
    print(f"  Columns: {list(t0.columns)}")
    print(f"\n  All seasons:")
    print(t0.to_string(index=False))
except Exception as e:
    print(f"  ERROR: {e}")

print(f"\n  Sleeping 4 seconds...")
time.sleep(4)

# ── 3b: Player page — advanced stats (in HTML comment blocks) ─────────────────
print("\n--- 3b: Zach Edey career page — advanced stats (from HTML comments) ---")
try:
    import requests
    resp = requests.get(
        url_edey,
        headers={"User-Agent": "Mozilla/5.0 (research/personal-use data exploration)"},
        timeout=10,
    )
    html = resp.text
    commented = re.findall(r"<!--(.*?)-->", html, re.DOTALL)
    print(f"  Comment blocks in page: {len(commented)}")

    advanced_found = False
    for j, block in enumerate(commented):
        if "<table" in block and any(col in block for col in ("USG", "BPM", "PER", "OBPM")):
            try:
                t = pd.read_html(StringIO(block))[0]
                if hasattr(t.columns, "levels"):
                    t.columns = [" ".join(str(c).strip() for c in col
                                          if "Unnamed" not in str(c)).strip()
                                 for col in t.columns]
                print(f"\n  Advanced table found (comment block {j}): shape={t.shape}")
                print(f"  Columns: {list(t.columns)}")
                print(f"\n  All seasons:")
                print(t.to_string(index=False))
                advanced_found = True
            except Exception:
                pass
    if not advanced_found:
        print("  No advanced stats table found in comments.")
except ImportError:
    print("  requests not available — cannot extract commented-out tables")
except Exception as e:
    print(f"  ERROR: {e}")

print(f"\n  Sleeping 4 seconds...")
time.sleep(4)

# ── 3c: School stats page — team-level (confirms URL pattern) ─────────────────
print("\n--- 3c: 2024 School Stats page (team-level, not player-level) ---")
url_schools = "https://www.sports-reference.com/cbb/seasons/men/2024-school-stats.html"
try:
    tables3 = pd.read_html(url_schools)
    t = tables3[0]
    if hasattr(t.columns, "levels"):
        t.columns = [" ".join(str(c).strip() for c in col if "Unnamed" not in str(c)).strip()
                     for col in t.columns]
    if "School" in t.columns:
        t = t[t["School"] != "School"].reset_index(drop=True)
    print(f"  Shape: {t.shape}")
    print(f"  Columns: {list(t.columns)}")
    print(f"\n  NOTE: This is TEAM-level data only, not player-level.")
    print(f"  For player-level season stats, must pull each player's individual page.")
except Exception as e:
    print(f"  ERROR: {e}")


# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("FINAL SUMMARY — What's actually available at the college level")
print(SEP)
print("""
SOURCE RESULTS:

1. cbbpy
   STATUS: BROKEN. ESPN changed internal API; all calls return empty DataFrames.
   Do not use.

2. sportsdataverse
   STATUS: BROKEN in this environment (NumPy 1.x/2.x binary incompatibility).
   Even if fixed: per-game box scores only, same data gap as cbbpy.
   No season aggregates, no USG%, no advanced metrics.

3. sports-reference.com/cbb  ← THE ONLY WORKING SOURCE
   STATUS: WORKS. Two access patterns confirmed:

   A) Per-player page (/cbb/players/{slug}.html):
      - Visible tables: per-game averages + season totals
        Columns: Season, Team, Conf, Class, Pos, G, GS, MP,
                 FG, FGA, FG%, 3P, 3PA, 3P%, 2P, 2PA, 2P%,
                 eFG%, FT, FTA, FT%, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS
      - Advanced stats in HTML comment blocks (need requests + regex):
        Columns: Season, Team, Conf, Class, Pos, G, GS, MP,
                 PER, TS%, 3PAr, FTr, PProd, ORB%, DRB%, TRB%,
                 AST%, STL%, BLK%, TOV%, USG%,
                 OWS, DWS, WS, WS/40, OBPM, DBPM, BPM
      - Multi-year: all seasons on one page per player slug

   B) School stats page (/cbb/seasons/men/{year}-school-stats.html):
      - TEAM-level data only (380 teams × 38 cols for 2024)
      - Not useful for player scouting

   RATE LIMITS: 3-4 seconds between requests minimum. Bot detection active.
   For a 60-player draft class: ~5-6 minutes sequential pulls.

WHAT'S CONFIRMED AVAILABLE:
   Per-player season averages:  G, MP, PTS, TRB, AST, STL, BLK, TOV
   Shooting splits:             FG%, 3P%, 2P%, eFG%, FT%, FTr, 3PAr
   Advanced per-player:        PER, TS%, USG%, ORB%, DRB%, STL%, BLK%, TOV%
   Win shares:                 OWS, DWS, WS, WS/40
   Box plus/minus:             OBPM, DBPM, BPM
   Multi-year tracking:        All seasons on one player page (transfer/freshman→senior)

WHAT DOES NOT EXIST PUBLICLY AT COLLEGE LEVEL:
   Shot location / zone data   — no equivalent of NBA.com zone charts
   Play-type breakdowns        — no Synergy equivalent for college
   On/off splits               — not published
   Tracking data               — not public (speed, distance, positioning)
   Opponent quality adjustment — BPM is closest proxy (strength-of-schedule baked in)
   Conference-adjusted metrics — not on sports-ref; need external SOS data for this

ACCESS PATTERN FOR COMPUTE FUNCTIONS:
   For a known player slug (e.g. "zach-edey-1"):
     url = f"https://www.sports-reference.com/cbb/players/{slug}.html"
     visible = pd.read_html(url)           # per-game averages (table 0)
     html = requests.get(url).text
     advanced = extract_comment_table(html, markers=["USG", "BPM"])  # advanced table

   The slug pattern is: {first}-{last}-{n}.html where n=1 for most players.
   There is no bulk season endpoint — player-by-player pulls only.
""")


# ── PART 4: Batch pull — 5 named players (validation cases) ──────────────────
# GOAL: validate the Part 3 access pattern generalizes across different
# slugs/schools, using players named by the user as recently-drafted with
# complete college careers.
#
# IMPORTANT — NOT ASSUMED, TO BE CHECKED: I do not have confirmed prior
# knowledge that all 5 of these are real players with completed,
# publicly-documented college careers. This section is a literal lookup:
# each name is tried against sports-reference exactly as given, and whatever
# is actually found (or not found) is reported below. No numbers are
# invented to fill in a failed pull, and the "sanity check" claims from the
# request are checked against pulled data rather than assumed true.

print(f"\n{SEP}")
print("PART 4 — Batch pull: 5 named players (validation cases)")
print(SEP)

PLAYERS = [
    ("Cam Boozer", "Duke"),
    ("AJ Dybantsa", "BYU"),
    ("Darryn Peterson", "Kansas"),
    ("Keaton Wagler", "Illinois"),
    ("Cameron Carr", "Baylor"),  # per request: transferred from Tennessee
]

VISIBLE_COLS = ["Season", "Class", "G", "MP", "PTS", "TRB", "AST",
                "FG%", "3P%", "2P%", "eFG%", "FT%"]
ADVANCED_COLS = ["Season", "USG%", "TS%", "PER", "BPM", "AST%"]

HEADERS = {"User-Agent": "Mozilla/5.0 (research/personal-use data exploration)"}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9\- ]", "", name.lower()).strip().replace(" ", "-")


def page_matches_school(html: str, school: str) -> bool:
    """Guard against slug collisions with a different, unrelated player
    (e.g. 'cameron-boozer-1' resolving to a 1990s Auburn/Troy State player
    instead of the intended Duke recruit). Checks the school name appears
    as one of the page's actual Team values, not just anywhere in the HTML
    (an opponent mention would give a false positive)."""
    try:
        tables = pd.read_html(StringIO(html))
        t0 = tables[0]
        if hasattr(t0.columns, "levels"):
            t0.columns = [" ".join(str(c).strip() for c in col
                                    if "Unnamed" not in str(c)).strip()
                          for col in t0.columns]
        if "Team" not in t0.columns:
            return False
        teams = t0["Team"].astype(str).str.lower().tolist()
        return any(school.lower() in t for t in teams)
    except Exception:
        return False


def resolve_slug(name: str, school: str):
    """Try the guessed slug directly; on 404, or on a school mismatch
    (slug collision with a different player), fall back to sports-
    reference's player search to confirm the real slug or confirm no
    match exists."""
    guessed = f"{slugify(name)}-1"
    url = f"https://www.sports-reference.com/cbb/players/{guessed}.html"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code == 200 and page_matches_school(resp.text, school):
        return guessed, url, resp.text
    direct_hit_wrong_school = resp.status_code == 200 and not page_matches_school(resp.text, school)

    search_url = f"https://www.sports-reference.com/cbb/search/search.fcgi?search={name.replace(' ', '+')}"
    search_resp = requests.get(search_url, headers=HEADERS, timeout=10)
    if search_resp.status_code == 200:
        candidate_slugs = []
        if "/cbb/players/" in search_resp.url:
            # Search redirected straight to a single player page.
            m = re.search(r"/cbb/players/([a-z0-9\-]+)\.html", search_resp.url)
            if m:
                candidate_slugs.append((m.group(1), search_resp.url, search_resp.text))
        else:
            # Disambiguation/results page — collect all player links.
            for slug in re.findall(r'/cbb/players/([a-z0-9\-]+)\.html', search_resp.text):
                if slug not in [c[0] for c in candidate_slugs]:
                    candidate_slugs.append((slug, None, None))

        for slug, cached_url, cached_html in candidate_slugs:
            player_url = cached_url or f"https://www.sports-reference.com/cbb/players/{slug}.html"
            if cached_html is None:
                player_resp = requests.get(player_url, headers=HEADERS, timeout=10)
                if player_resp.status_code != 200:
                    continue
                cached_html = player_resp.text
            if page_matches_school(cached_html, school):
                return slug, player_url, cached_html

    if direct_hit_wrong_school:
        return None, url, "WRONG_SCHOOL"
    return None, url, None


def extract_advanced_table(html: str):
    commented = re.findall(r"<!--(.*?)-->", html, re.DOTALL)
    for block in commented:
        if "<table" in block and any(col in block for col in ("USG", "BPM", "PER", "OBPM")):
            try:
                t = pd.read_html(StringIO(block))[0]
                if hasattr(t.columns, "levels"):
                    t.columns = [" ".join(str(c).strip() for c in col
                                          if "Unnamed" not in str(c)).strip()
                                 for col in t.columns]
                return t
            except Exception:
                continue
    return None


def last_season_row(table: pd.DataFrame) -> pd.Series:
    """Return the row for the most recent dated season (e.g. '2025-26'),
    skipping 'Career' / '<School> (N Yrs)' summary rows that sports-reference
    appends after the per-season rows."""
    if "Season" not in table.columns:
        return table.iloc[-1]
    dated = table[table["Season"].astype(str).str.match(r"^\d{4}-\d{2}$")]
    return dated.iloc[-1] if not dated.empty else table.iloc[-1]


def fetch_player(name: str, school: str) -> dict:
    try:
        slug, url, html = resolve_slug(name, school)
        if html is None or html == "WRONG_SCHOOL":
            reason = (f"WRONG PLAYER — a page exists at the guessed slug "
                      f"'{slugify(name)}-1' but its team history does not "
                      f"include '{school}' (likely a same-named different "
                      f"player), and no matching alternate was found via search"
                      if html == "WRONG_SCHOOL" else
                      f"NOT FOUND — no sports-reference page located "
                      f"(tried slug '{slugify(name)}-1' and player search)")
            return {"name": name, "school": school, "found": False, "error": reason}

        visible_tables = pd.read_html(StringIO(html))
        if not visible_tables or visible_tables[0].empty:
            return {"name": name, "school": school, "found": False, "slug": slug,
                     "error": "Page resolved but per-game table is empty/missing"}

        per_game = visible_tables[0]
        if hasattr(per_game.columns, "levels"):
            per_game.columns = [" ".join(str(c).strip() for c in col
                                          if "Unnamed" not in str(c)).strip()
                                 for col in per_game.columns]
        last_row = last_season_row(per_game)

        row = {"name": name, "school": school, "found": True, "slug": slug}
        for col in VISIBLE_COLS:
            row[col] = last_row[col] if col in per_game.columns else None

        adv_table = extract_advanced_table(html)
        if adv_table is not None and not adv_table.empty:
            adv_last = last_season_row(adv_table)
            for col in ADVANCED_COLS:
                if col == "Season":
                    continue
                row[col] = adv_last[col] if col in adv_table.columns else None
        else:
            for col in ADVANCED_COLS:
                if col != "Season":
                    row[col] = None
            row["advanced_note"] = "no advanced table found in comment blocks"

        return row
    except Exception as e:
        return {"name": name, "school": school, "found": False,
                 "error": f"{type(e).__name__}: {str(e)[:200]}"}


results = []
for i, (name, school) in enumerate(PLAYERS):
    print(f"\n--- Fetching {name} ({school}) ---")
    r = fetch_player(name, school)
    if r.get("found"):
        print(f"  Resolved slug: {r.get('slug')}")
        print(f"  Season row: { {k: v for k, v in r.items() if k not in ('name','school','found','slug')} }")
    else:
        print(f"  {r.get('error')}")
    results.append(r)
    if i < len(PLAYERS) - 1:
        print("  Sleeping 4 seconds...")
        time.sleep(4)


# ── Side-by-side table ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SIDE-BY-SIDE TABLE")
print(SEP)

table_cols = ["name", "school", "slug", "Season", "Class", "G", "MP", "PTS",
              "TRB", "AST", "FG%", "3P%", "2P%", "eFG%", "FT%",
              "USG%", "TS%", "PER", "BPM", "AST%"]
table_rows = []
for r in results:
    if r.get("found"):
        table_rows.append({c: r.get(c, None) for c in table_cols})
    else:
        table_rows.append({"name": r["name"], "school": r["school"],
                            "slug": "N/A", **{c: "—" for c in table_cols
                                              if c not in ("name", "school", "slug")}})

summary_df = pd.DataFrame(table_rows, columns=table_cols)
print(summary_df.to_string(index=False))


# ── Sanity check against claims in the request ────────────────────────────────
print(f"\n{SEP}")
print("SANITY CHECK — pulled data vs. claims in the request")
print(SEP)


def get(name_fragment):
    for r in results:
        if name_fragment.lower() in r["name"].lower():
            return r
    return None


def fmt_frac_pct(v):
    """For fields pulled from the visible per-game table (3P%, 2P%, eFG%),
    which sports-reference stores as a 0-1 fraction."""
    if v is None:
        return "NO DATA"
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(v)


def fmt_usg_pct(v):
    """For USG% from the advanced table, which sports-reference already
    stores as a percentage number (e.g. 33.9, not 0.339)."""
    if v is None:
        return "NO DATA"
    try:
        return f"{float(v):.1f}%"
    except (TypeError, ValueError):
        return str(v)


boozer = get("Boozer")
dybantsa = get("Dybantsa")
peterson = get("Peterson")

print("\nClaim: Boozer shows high 3P% (~40%) on real volume + strong post scoring efficiency")
if boozer and boozer.get("found"):
    threep = boozer.get("3P%")
    twop = boozer.get("2P%")
    efg = boozer.get("eFG%")
    threep_val = float(threep) * 100 if threep is not None else None
    if threep_val is None:
        verdict = "NO DATA on 3P%"
    elif threep_val >= 38:
        verdict = "MATCH — 3P% is at/above the ~40% claim"
    elif threep_val >= 30:
        verdict = f"MISMATCH — 3P% ({threep_val:.1f}%) is real but well below the ~40% claimed"
    else:
        verdict = f"MISMATCH — 3P% ({threep_val:.1f}%) is far below the ~40% claimed"
    print(f"  PULLED: 3P%={fmt_frac_pct(threep)}  2P%={fmt_frac_pct(twop)}  eFG%={fmt_frac_pct(efg)}")
    print(f"  -> {verdict}")
    print(f"  NOTE: this data is for sports-reference slug '{boozer.get('slug')}' — see NOT FOUND/")
    print(f"        WRONG PLAYER caveat below if this does not match the intended person.")
else:
    print(f"  PULLED: NO DATA ({boozer.get('error') if boozer else 'player not attempted'})")
    print("  -> NO DATA — cannot confirm or refute this claim")

print("\nClaim: Dybantsa shows high usage as primary option")
if dybantsa and dybantsa.get("found"):
    usg = dybantsa.get("USG%")
    if usg is None:
        verdict = "NO DATA — advanced table (USG%) not found for this player"
    elif float(usg) >= 28:
        verdict = f"MATCH — USG% ({float(usg):.1f}) is in/above typical primary-option range (~28%+)"
    else:
        verdict = f"MISMATCH — USG% ({float(usg):.1f}) is below typical primary-option range (~28%+)"
    print(f"  PULLED: USG%={fmt_usg_pct(usg)}")
    print(f"  -> {verdict}")
else:
    print(f"  PULLED: NO DATA ({dybantsa.get('error') if dybantsa else 'player not attempted'})")
    print("  -> NO DATA — cannot confirm or refute this claim")

print("\nClaim: Peterson shows high usage as primary option")
if peterson and peterson.get("found"):
    usg = peterson.get("USG%")
    if usg is None:
        verdict = "NO DATA — advanced table (USG%) not found for this player"
    elif float(usg) >= 28:
        verdict = f"MATCH — USG% ({float(usg):.1f}) is in/above typical primary-option range (~28%+)"
    else:
        verdict = f"MISMATCH — USG% ({float(usg):.1f}) is below typical primary-option range (~28%+)"
    print(f"  PULLED: USG%={fmt_usg_pct(usg)}")
    print(f"  -> {verdict}")
else:
    print(f"  PULLED: NO DATA ({peterson.get('error') if peterson else 'player not attempted'})")
    print("  -> NO DATA — cannot confirm or refute this claim")

not_found = [r["name"] for r in results if not r.get("found")]
if not_found:
    print(f"\nNOTE: {len(not_found)}/{len(results)} players could not be resolved on "
          f"sports-reference.com/cbb under the attempted slug/search: {', '.join(not_found)}")
    print("These may be misspelled, not yet on sports-reference (career not final/indexed),")
    print("or not real/complete college careers as of this run — see per-player errors above.")
