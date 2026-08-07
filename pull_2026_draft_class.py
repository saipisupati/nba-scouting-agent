"""
Batch pull of final college-season stats for the 2026 NBA Draft class
(60 picks, drafted June 23-24, 2026), using the sports-reference.com/cbb
access pattern proven in explore_college.py Part 4.

NOTE ON THE DRAFT LIST: the pick/round numbers and school assignments below
are exactly as supplied by the user (sourced to NBA.com). This script does
not independently verify draft outcomes -- it only verifies that a given
name's sports-reference page exists and matches the claimed school. Treat
pick_number/round as row labels carried through from the input list, not as
something this script confirms.

Handling rules:
  1. International picks (6 known: no NCAA data source) are logged directly
     as status="international" without attempting a scrape.
  2. Jayden Quaintance (pick 20, no school given) is resolved via the
     sports-reference search fallback only; if no confident match is found,
     logged as status="unresolved" rather than guessing a school.
  3. Reuses page_matches_school to guard against slug collisions -- this
     class has a known collision risk (Cameron Boozer has a twin brother,
     Cayden Boozer, also a college player) and likely other common-surname
     collisions across 54 NCAA names.
  4. 5-6 second delay between every player pull, chunked into 3 groups of
     ~18 players with a 60-90s pause between chunks (~54 scrape targets =>
     budget 8-10+ minutes). Sequential only, no parallelization.
  5. Results written to CSV incrementally (after every player, not just at
     the end) so a mid-batch stop still leaves completed rows on disk.
  6. If a 429 survives the in-request retry/backoff, the batch stops
     immediately and reports state rather than continuing past a live
     rate-limit signal.
"""

import csv
import random
import re
import socket
import time
import warnings
from io import StringIO

import pandas as pd
import requests

warnings.filterwarnings("ignore")

# requests' own per-call timeout does not reliably bound slow DNS/connect
# phases on this network (observed: multi-minute hang in getaddrinfo/socket
# connect despite timeout=10 on every call). A hard global socket timeout
# is the actual backstop.
socket.setdefaulttimeout(12)

REQUEST_TIMEOUT = 10
MAX_RETRIES = 2

SEP = "=" * 70
HEADERS = {"User-Agent": "Mozilla/5.0 (research/personal-use data exploration)"}

VISIBLE_COLS = ["Season", "Class", "G", "MP", "PTS", "TRB", "AST",
                "FG%", "3P%", "2P%", "eFG%", "FT%"]
ADVANCED_COLS = ["Season", "USG%", "TS%", "PER", "BPM", "AST%"]

CSV_COLS = ["pick_number", "name", "school", "round", "class_year",
            "PTS", "TRB", "AST", "FG%", "3P%", "eFG%", "USG%", "TS%",
            "BPM", "PER", "AST%", "status"]


# ── Draft list, exactly as supplied by the user ───────────────────────────────
# (pick_number, name, school_or_None, round, is_international)
DRAFT_CLASS = [
    (1, "AJ Dybantsa", "BYU", 1, False),
    (2, "Darryn Peterson", "Kansas", 1, False),
    (3, "Cameron Boozer", "Duke", 1, False),
    (4, "Caleb Wilson", "North Carolina", 1, False),
    (5, "Keaton Wagler", "Illinois", 1, False),
    (6, "Mikel Brown Jr.", "Louisville", 1, False),
    (7, "Darius Acuff Jr.", "Arkansas", 1, False),
    (8, "Kingston Flemings", "Houston", 1, False),
    (9, "Morez Johnson Jr.", "Michigan", 1, False),
    (10, "Brayden Burries", "Arizona", 1, False),
    (11, "Yaxel Lendeborg", "Michigan", 1, False),
    (12, "Aday Mara", "Michigan", 1, False),
    (13, "Nate Ament", "Tennessee", 1, False),
    (14, "Hannes Steinbach", "Washington", 1, False),
    (15, "Dailyn Swain", "Texas", 1, False),
    (16, "Bennett Stirtz", "Iowa", 1, False),
    (17, "Ebuka Okorie", "Stanford", 1, False),
    (18, "Christian Anderson", "Texas Tech", 1, False),
    (19, "Allen Graves", "Santa Clara", 1, False),
    (20, "Jayden Quaintance", None, 1, False),
    (21, "Karim Lopez", "New Zealand Breakers", 1, True),
    (22, "Labaron Philon Jr.", "Alabama", 1, False),
    (23, "Zuby Ejiofor", "St. John's", 1, False),
    (24, "Cameron Carr", "Baylor", 1, False),
    (25, "Sergio De Larrea", "Valencia", 1, True),
    (26, "Tarris Reed Jr.", "Connecticut", 1, False),
    (27, "Chris Cenac Jr.", "Houston", 1, False),
    (28, "Joshua Jefferson", "Iowa State", 1, False),
    (29, "Alex Karaban", "Connecticut", 1, False),
    (30, "Koa Peat", "Arizona", 1, False),
    (31, "Bruce Thornton", "Ohio State", 2, False),
    (32, "Richie Saunders", "BYU", 2, False),
    (33, "Isaiah Evans", "Duke", 2, False),
    (34, "Meleek Thomas", "Arkansas", 2, False),
    (35, "Trevon Brazile", "Arkansas", 2, False),
    (36, "Baba Miller", "Cincinnati", 2, False),
    (37, "Ryan Conwell", "Louisville", 2, False),
    (38, "Braden Smith", "Purdue", 2, False),
    (39, "Jack Kayil", "Alba Berlin", 2, True),
    (40, "Dillon Mitchell", "St. John's", 2, False),
    (41, "Otega Oweh", "Kentucky", 2, False),
    (42, "Ja'Kobi Gillespie", "Tennessee", 2, False),
    (43, "Tyler Bilodeau", "UCLA", 2, False),
    (44, "Maliq Brown", "Duke", 2, False),
    (45, "Emanuel Sharp", "Houston", 2, False),
    (46, "Felix Okpara", "Tennessee", 2, False),
    (47, "Tyler Nickel", "Vanderbilt", 2, False),
    (48, "Tobi Lawal", "Virginia Tech", 2, False),
    (49, "Bryce Hopkins", "St. John's", 2, False),
    (50, "Jaden Bradley", "Arizona", 2, False),
    (51, "Izaiyah Nelson", "South Florida", 2, False),
    (52, "Henri Veesaar", "North Carolina", 2, False),
    (53, "Ugonna Onyenso", "Virginia", 2, False),
    (54, "Lajae Jones", "Florida State", 2, False),
    (55, "Nick Martinelli", "Northwestern", 2, False),
    (56, "Vsevolod Ishchenko", "Lokomotiv Kuban", 2, True),
    (57, "Narcisse Ngoy", "Poitiers", 2, True),
    (58, "Jaron Pierre Jr.", "Southern Methodist", 2, False),
    (59, "Trey Kaufman-Renn", "Purdue", 2, False),
    (60, "Malique Lewis", "South East Melbourne", 2, True),
]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9\- ]", "", name.lower()).strip().replace(" ", "-")


# Sports-reference's Team column often uses a nickname/abbreviation that
# doesn't contain the school's full name as a substring -- confirmed
# directly: Alex Karaban's real, correct page lists Team="UConn" (draft
# list says "Connecticut"), and Caleb Wilson's real, correct page lists
# Team="UNC" (draft list says "North Carolina"). Both were misreported as
# not_found/WRONG_SCHOOL before this alias map existed, even though the
# page substring-matching logic itself was working as designed -- the
# draft list's school name just wasn't literally what sports-reference
# prints. Not an exhaustive list; add entries here as more mismatches are
# confirmed rather than guessing at aliases that haven't been verified.
_SCHOOL_ALIASES = {
    "connecticut": {"uconn"},
    "north carolina": {"unc"},
    "ohio state": {"ohio st."},
    "southern methodist": {"smu"},
}


def page_matches_school(html: str, school: str) -> bool:
    """Guard against slug collisions with a different, unrelated player.
    Checks the school name (or a known alias of it) appears as one of the
    page's actual Team values, not just anywhere in the HTML (an opponent
    mention would false-positive).
    """
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
        school_lower = school.lower()
        candidates = {school_lower} | _SCHOOL_ALIASES.get(school_lower, set())
        return any(alias in t for alias in candidates for t in teams)
    except Exception:
        return False


def safe_get(url: str):
    """requests.get with a hard retry/backoff wrapper. A single slow or
    hanging DNS/connect on one player must not stall the whole 60-pick
    batch, and a 429 (rate limited) must not be misread as "player not
    found" -- it gets a long backoff and retry instead. Returns a
    Response, or None if all attempts failed/timed out/stayed rate-limited."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                if attempt < MAX_RETRIES:
                    wait = 20 * (attempt + 1)
                    print(f"    (429 rate-limited on attempt {attempt + 1} — backing off {wait}s)")
                    time.sleep(wait)
                    continue
                print(f"    (still 429 after {MAX_RETRIES + 1} attempts — giving up on this URL)")
                return resp
            return resp
        except (requests.exceptions.RequestException, socket.timeout, OSError) as e:
            if attempt < MAX_RETRIES:
                print(f"    (request error on attempt {attempt + 1}: {type(e).__name__} — retrying)")
                time.sleep(2)
            else:
                print(f"    (request failed after {MAX_RETRIES + 1} attempts: {type(e).__name__}: {str(e)[:150]})")
    return None


_NAME_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}


def _strip_name_suffix(name: str) -> str:
    """Sports-reference's own search.fcgi does not reliably match on
    'Jr.'/'Sr.'/'II' etc. -- confirmed directly: searching "Labaron Philon
    Jr." returned five unrelated players (search noise), while "Labaron
    Philon" (suffix stripped) redirected straight to the correct, single
    matching player. Slugs themselves still include the suffix where
    sports-reference uses one (handled separately by slugify) -- this only
    affects the search query text."""
    parts = name.split()
    if parts and parts[-1].strip(".").lower() in _NAME_SUFFIXES:
        return " ".join(parts[:-1])
    return name


def search_candidates(name: str):
    """Query sports-reference player search; return list of
    (slug, url, html) candidates found, fetching each candidate page.

    Returns the string "RATE_LIMITED" instead of a list if the search
    itself, or any candidate-page fetch, hit a 429 that survived
    safe_get's retry/backoff. A 429 must never be read as "confirmed zero
    candidates" -- that previously caused a false WRONG_SCHOOL/not_found
    result for at least one real player (Alex Karaban) when the search
    fallback got rate-limited mid-batch instead of genuinely finding no
    match. Any other non-200/non-429 outcome (404, network failure) is
    still treated as a real empty result, since those are conclusive.
    """
    search_name = _strip_name_suffix(name)
    search_url = f"https://www.sports-reference.com/cbb/search/search.fcgi?search={search_name.replace(' ', '+')}"
    search_resp = safe_get(search_url)
    if search_resp is not None and search_resp.status_code == 429:
        return "RATE_LIMITED"

    candidates = []
    if search_resp is None or search_resp.status_code != 200:
        return candidates

    if "/cbb/players/" in search_resp.url:
        m = re.search(r"/cbb/players/([a-z0-9\-]+)\.html", search_resp.url)
        if m:
            candidates.append((m.group(1), search_resp.url, search_resp.text))
        return candidates

    seen = set()
    for slug in re.findall(r'/cbb/players/([a-z0-9\-]+)\.html', search_resp.text):
        if slug in seen:
            continue
        seen.add(slug)
        player_url = f"https://www.sports-reference.com/cbb/players/{slug}.html"
        player_resp = safe_get(player_url)
        if player_resp is not None and player_resp.status_code == 429:
            return "RATE_LIMITED"
        if player_resp is not None and player_resp.status_code == 200:
            candidates.append((slug, player_url, player_resp.text))
    return candidates


def resolve_slug(name: str, school: str):
    """Try the guessed slug directly; on 404, or on a school mismatch
    (slug collision with a different/same-surname player), fall back to
    sports-reference's player search to confirm the real slug or confirm
    no match exists. Returns (slug, url, html) or (None, url, sentinel)."""
    guessed = f"{slugify(name)}-1"
    url = f"https://www.sports-reference.com/cbb/players/{guessed}.html"
    resp = safe_get(url)
    if resp is not None and resp.status_code == 429:
        return None, url, "RATE_LIMITED"
    if resp is not None and resp.status_code == 200 and page_matches_school(resp.text, school):
        return guessed, url, resp.text
    direct_hit_wrong_school = (resp is not None and resp.status_code == 200
                               and not page_matches_school(resp.text, school))

    candidates = search_candidates(name)
    if candidates == "RATE_LIMITED":
        return None, url, "RATE_LIMITED"

    for slug, player_url, html in candidates:
        if page_matches_school(html, school):
            return slug, player_url, html

    if direct_hit_wrong_school:
        return None, url, "WRONG_SCHOOL"
    return None, url, None


def resolve_slug_no_school(name: str):
    """For players with no school listed (Jayden Quaintance): only trust
    a search result if it resolves to exactly one candidate. Multiple
    candidates or zero candidates -> unresolved rather than guessing."""
    guessed = f"{slugify(name)}-1"
    url = f"https://www.sports-reference.com/cbb/players/{guessed}.html"

    candidates = search_candidates(name)
    if candidates == "RATE_LIMITED":
        return None, url, "RATE_LIMITED"

    if len(candidates) == 1:
        slug, player_url, html = candidates[0]
        return slug, player_url, html
    return None, url, f"AMBIGUOUS({len(candidates)})" if candidates else None


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
    skipping 'Career' / '<School> (N Yrs)' summary rows."""
    if "Season" not in table.columns:
        return table.iloc[-1]
    dated = table[table["Season"].astype(str).str.match(r"^\d{4}-\d{2}$")]
    return dated.iloc[-1] if not dated.empty else table.iloc[-1]


def fetch_player(name: str, school):
    """Returns a dict with at least {name, school, status, error?}.
    status in {"ok", "unresolved", "not_found"}."""
    try:
        if school is None:
            slug, url, html = resolve_slug_no_school(name)
            if html == "RATE_LIMITED":
                return {"name": name, "school": None, "status": "rate_limited",
                         "error": "Sports-reference returned 429 (rate limited) even "
                                  "after retry/backoff — not a real not-found, retry later."}
            if html is None or (isinstance(html, str) and html.startswith("AMBIGUOUS")):
                note = html if isinstance(html, str) and html.startswith("AMBIGUOUS") else "no search match"
                return {"name": name, "school": None, "status": "unresolved",
                         "error": f"No school given; search result {note} — "
                                  f"not resolved confidently, not guessing."}
        else:
            slug, url, html = resolve_slug(name, school)
            if html == "RATE_LIMITED":
                return {"name": name, "school": school, "status": "rate_limited",
                         "error": "Sports-reference returned 429 (rate limited) even "
                                  "after retry/backoff — not a real not-found, retry later."}
            if html is None or html == "WRONG_SCHOOL":
                reason = (f"WRONG PLAYER — page at guessed slug '{slugify(name)}-1' "
                          f"exists but team history doesn't include '{school}' "
                          f"(likely a same-named different player), and no "
                          f"matching alternate found via search"
                          if html == "WRONG_SCHOOL" else
                          f"NOT FOUND — no sports-reference page located "
                          f"(tried slug '{slugify(name)}-1' and player search)")
                return {"name": name, "school": school, "status": "not_found", "error": reason}

        visible_tables = pd.read_html(StringIO(html))
        if not visible_tables or visible_tables[0].empty:
            return {"name": name, "school": school, "status": "not_found", "slug": slug,
                     "error": "Page resolved but per-game table is empty/missing"}

        per_game = visible_tables[0]
        if hasattr(per_game.columns, "levels"):
            per_game.columns = [" ".join(str(c).strip() for c in col
                                          if "Unnamed" not in str(c)).strip()
                                 for col in per_game.columns]
        last_row = last_season_row(per_game)

        row = {"name": name, "school": school, "status": "ok", "slug": slug}
        for col in VISIBLE_COLS:
            row[col] = last_row[col] if col in per_game.columns else None

        adv_table = extract_advanced_table(html)
        if adv_table is not None and not adv_table.empty:
            adv_last = last_season_row(adv_table)
            for col in ADVANCED_COLS:
                if col != "Season":
                    row[col] = adv_last[col] if col in adv_table.columns else None
        else:
            for col in ADVANCED_COLS:
                if col != "Season":
                    row[col] = None
            row["advanced_note"] = "no advanced table found in comment blocks"

        return row
    except Exception as e:
        return {"name": name, "school": school, "status": "not_found",
                 "error": f"{type(e).__name__}: {str(e)[:200]}"}


def frac_to_pct_str(v):
    if v is None:
        return None
    try:
        return round(float(v) * 100, 1)
    except (TypeError, ValueError):
        return None


def as_pct_str(v):
    if v is None:
        return None
    try:
        return round(float(v), 1)
    except (TypeError, ValueError):
        return None


# ── Main pull loop ─────────────────────────────────────────────────────────────
CSV_PATH = "draft_class_2026.csv"
CHUNK_SIZE = 15
CHUNK_PAUSE_RANGE = (60, 90)
REQUEST_DELAY_RANGE = (5.0, 6.0)


def blank_row(pick_number, name, school, rnd, status):
    return {
        "pick_number": pick_number, "name": name, "school": school, "round": rnd,
        "class_year": None, "PTS": None, "TRB": None, "AST": None, "FG%": None,
        "3P%": None, "eFG%": None, "USG%": None, "TS%": None, "BPM": None,
        "PER": None, "AST%": None, "status": status,
    }


def ok_row(pick_number, name, school, rnd, result):
    return {
        "pick_number": pick_number, "name": name, "school": school, "round": rnd,
        "class_year": result.get("Class"),
        "PTS": result.get("PTS"), "TRB": result.get("TRB"), "AST": result.get("AST"),
        "FG%": frac_to_pct_str(result.get("FG%")),
        "3P%": frac_to_pct_str(result.get("3P%")),
        "eFG%": frac_to_pct_str(result.get("eFG%")),
        "USG%": as_pct_str(result.get("USG%")),
        "TS%": frac_to_pct_str(result.get("TS%")),
        "BPM": as_pct_str(result.get("BPM")),
        "PER": as_pct_str(result.get("PER")),
        "AST%": as_pct_str(result.get("AST%")),
        "status": "ok",
    }


def write_csv(rows):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_existing_csv():
    """Load already-resolved rows from CSV_PATH, if present, keyed by
    pick_number (as int). Used to resume a stopped batch without losing
    or overwriting previously completed picks."""
    try:
        with open(CSV_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return {}
    numeric_cols = ("PTS", "TRB", "AST", "FG%", "3P%", "eFG%", "USG%", "TS%", "BPM", "PER", "AST%")
    by_pick = {}
    for r in rows:
        r["pick_number"] = int(r["pick_number"])
        r["round"] = int(r["round"]) if r["round"] not in (None, "") else None
        for col in numeric_cols:
            r[col] = float(r[col]) if r.get(col) not in (None, "") else None
        if r.get("class_year") == "":
            r["class_year"] = None
        if r.get("school") == "":
            r["school"] = None
        by_pick[r["pick_number"]] = r
    return by_pick


def chunk(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def run_batch(resume_from_pick: int = 1):
    """Run the chunked batch pull, starting at resume_from_pick (inclusive).
    Existing rows already saved in CSV_PATH for picks before resume_from_pick
    are preserved and merged into the final summary/trend check rather than
    discarded -- write_csv always writes the full merged set, so a resumed
    run never overwrites previously completed picks with blanks.

    Guarded by __main__ so this module can also be imported (e.g. to
    re-resolve a handful of flagged players with fetch_player() directly)
    without re-triggering the whole batch as a side effect of import.
    """
    existing_by_pick = load_existing_csv()
    targets = [p for p in DRAFT_CLASS if p[0] >= resume_from_pick]

    print(SEP)
    print("2026 NBA DRAFT CLASS — batch pull of final college season stats")
    if resume_from_pick > 1:
        print(f"RESUMING from pick #{resume_from_pick}. "
              f"{len(existing_by_pick)} already-resolved picks preserved from {CSV_PATH}.")
    n_scrape_targets = sum(1 for p in targets if not p[4])
    print(f"{len(targets)} picks to process this run ({n_scrape_targets} to scrape, "
          f"{len(targets) - n_scrape_targets} international logged without a request).")
    print(f"5-6s delay between requests, chunked into groups of {CHUNK_SIZE} with "
          f"{CHUNK_PAUSE_RANGE[0]}-{CHUNK_PAUSE_RANGE[1]}s pauses between chunks.")
    print(SEP)

    all_rows = dict(existing_by_pick)  # pick_number -> row, merged as we go
    chunks = chunk(targets, CHUNK_SIZE)
    stopped_early = False

    def merged_rows():
        return [all_rows[p] for p in sorted(all_rows)]

    for chunk_idx, batch in enumerate(chunks):
        print(f"\n{SEP}\nCHUNK {chunk_idx + 1}/{len(chunks)} ({len(batch)} picks)\n{SEP}")

        for i, (pick_number, name, school, rnd, is_intl) in enumerate(batch):
            print(f"\n--- Pick #{pick_number}: {name} ({school or 'no school listed'}) ---")

            if is_intl:
                print("  INTERNATIONAL — no NCAA data source. Logged, no scrape attempted.")
                all_rows[pick_number] = blank_row(pick_number, name, school, rnd, "international")
                write_csv(merged_rows())
                continue

            result = fetch_player(name, school)
            status = result.get("status")

            if status == "rate_limited":
                print(f"  {result.get('error')}")
                print(f"\n{SEP}")
                print("STOPPING EARLY — 429 rate-limit persisted through in-request retry/backoff.")
                print(f"Completed {len(all_rows)} of {len(DRAFT_CLASS)} picks total before stopping "
                      f"(including {len(existing_by_pick)} preserved from before this run).")
                print(f"Partial results already saved to {CSV_PATH} (written incrementally).")
                print(f"{SEP}")
                all_rows[pick_number] = blank_row(pick_number, name, school, rnd, "rate_limited")
                write_csv(merged_rows())
                stopped_early = True
                break

            if status == "ok":
                print(f"  Resolved slug: {result.get('slug')}  Season: {result.get('Season')}  Class: {result.get('Class')}")
                all_rows[pick_number] = ok_row(pick_number, name, school, rnd, result)
            else:
                print(f"  {result.get('error')}")
                all_rows[pick_number] = blank_row(pick_number, name, school, rnd, status)

            write_csv(merged_rows())

            is_last_in_batch = i == len(batch) - 1
            if not is_last_in_batch:
                delay = random.uniform(*REQUEST_DELAY_RANGE)
                print(f"  Sleeping {delay:.1f}s...")
                time.sleep(delay)

        if stopped_early:
            break

        is_last_chunk = chunk_idx == len(chunks) - 1
        if not is_last_chunk:
            pause = random.uniform(*CHUNK_PAUSE_RANGE)
            print(f"\n--- End of chunk {chunk_idx + 1}/{len(chunks)}. Pausing {pause:.0f}s before next chunk... ---")
            time.sleep(pause)

    final_rows = merged_rows()

    print(f"\n{SEP}")
    if stopped_early:
        print(f"BATCH STOPPED EARLY due to persistent rate-limiting. "
              f"{len(final_rows)} of {len(DRAFT_CLASS)} picks processed in total.")
    else:
        print(f"Batch complete. Saved {len(final_rows)} rows to {CSV_PATH}")
    print(SEP)

    # ── Summary (across the full merged set: previously-resolved + this run) ──
    n_ok = sum(1 for r in final_rows if r["status"] == "ok")
    n_intl = sum(1 for r in final_rows if r["status"] == "international")
    n_unresolved = sum(1 for r in final_rows if r["status"] == "unresolved")
    n_not_found = sum(1 for r in final_rows if r["status"] == "not_found")
    n_rate_limited = sum(1 for r in final_rows if r["status"] == "rate_limited")

    print(f"\nSUMMARY (full draft class, all picks processed to date)")
    if stopped_early:
        print(f"  NOTE: this run stopped early due to persistent rate-limiting — "
              f"summary covers the {len(final_rows)} of {len(DRAFT_CLASS)} "
              f"picks processed so far across all runs.")
    print(f"  Total picks processed: {len(final_rows)} of {len(DRAFT_CLASS)}")
    print(f"  Resolved (ok):      {n_ok}")
    print(f"  International:      {n_intl}")
    print(f"  Unresolved:         {n_unresolved}")
    print(f"  Not found:          {n_not_found}")
    print(f"  Still rate-limited: {n_rate_limited}")

    if n_not_found > 0:
        print(f"\n  NOT FOUND / WRONG_SCHOOL rows (verify individually before trusting -- "
              f"three prior not_found results this project turned out to be false negatives "
              f"from bugs, not genuine misses):")
        for r in final_rows:
            if r["status"] == "not_found":
                print(f"    Pick #{r['pick_number']}: {r['name']} ({r['school']})")

    ok_rows = [r for r in final_rows if r["status"] == "ok"]
    if ok_rows:
        ok_rows_sorted = sorted(ok_rows, key=lambda r: r["pick_number"])
        n = len(ok_rows_sorted)
        first_half = ok_rows_sorted[: n // 2]
        second_half = ok_rows_sorted[n // 2:]

        def avg(rows, key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        usg_first = avg(first_half, "USG%")
        usg_second = avg(second_half, "USG%")
        bpm_first = avg(first_half, "BPM")
        bpm_second = avg(second_half, "BPM")

        print(f"\nPICK-NUMBER TREND CHECK (does USG%/BPM trend downward as pick # rises?)")
        print(f"  Resolved players split into pick-order halves: first {len(first_half)}, second {len(second_half)}")
        print(f"  Avg USG%  — first half: {usg_first:.1f}   second half: {usg_second:.1f}"
              if usg_first is not None and usg_second is not None else "  Avg USG% — insufficient data")
        print(f"  Avg BPM   — first half: {bpm_first:.1f}   second half: {bpm_second:.1f}"
              if bpm_first is not None and bpm_second is not None else "  Avg BPM  — insufficient data")

        if usg_first is not None and usg_second is not None:
            print(f"  -> USG% {'DECREASES' if usg_second < usg_first else 'DOES NOT DECREASE'} from first half to second half of resolved picks")
        if bpm_first is not None and bpm_second is not None:
            print(f"  -> BPM {'DECREASES' if bpm_second < bpm_first else 'DOES NOT DECREASE'} from first half to second half of resolved picks")
    else:
        print("\nNo resolved players — cannot compute pick-number trend.")


if __name__ == "__main__":
    import sys
    resume_from = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_batch(resume_from_pick=resume_from)
