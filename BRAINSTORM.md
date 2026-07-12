# NBA Scouting Agent — Improvement Brainstorm

## Current State (as of 2026-07-12)

### What's built
- `hustle_stats.py` — pulls LeagueHustleStatsPlayer for 2025-26 and 2024-25
- `compute_defense.py` — four ranked-output functions:
  - `deflections_per36` — active hands, pass disruptions
  - `contest_profile_per36` — rim contests (2PT) and perimeter closeouts (3PT)
  - `boxout_conversion` — box-out-to-rebound conversion rate
  - `hustle_iq_composite` — weighted z-score: def loose balls + charges drawn per 36
- `query_router.py` — deterministic regex router + Groq/Llama-3.3-70b LLM fallback
- `test_router.py` — 13-question test suite (7 deterministic, 6 llm_fallback), all passing

### Current coverage
Only hustle/effort stats: deflections, contested shots, box outs, loose balls, charges.
Does NOT cover: on-ball defense quality, switchability, advanced metrics, offense, physical data, 
multi-season trajectories, or opponent/context adjustments.

---

## Research Findings

### What real scouts and analytics teams actually use

**The layered scouting framework (validated across analytics-first teams):**
1. **Effort/hustle layer** — what we have: deflections, contested shots, box outs, charges
2. **Shot defense quality layer** — opponent FG% when guarded, by shot zone (rim, midrange, 3PT)
3. **Play-type defense layer** — how a player defends specific scenarios: ISO, P&R ball handler, P&R roll man, post-up, spot-up
4. **Plus-minus layer** — on/off defensive rating (noisy, needs 3+ seasons to stabilize), DRPM/RAPM
5. **Context/scheme layer** — who they play with, what defensive scheme the team runs, opponent quality
6. **Physical profile** — wingspan, height, lateral quickness (combine data), age/development curve

**What analytically elite teams target:**
- **OKC Thunder** — length + switchability, high block% + steal% combos, young players with tracking upside
- **Boston Celtics** — versatile defenders who can switch 1-5, prioritize opponent FG% suppression
- **Denver Nuggets** — two-way wings with positive on/off, de-emphasize raw block/steal counts
- **Milwaukee Bucks (Antetokounmpo era)** — rim protection + fast closeout ability, zone-hybrid coverage

**Key validated finding:** The combination of **block% + steal% from college** (e.g., >5% blocks + >4% steals) is one of the strongest early predictors of NBA defensive impact. Very few players achieve both — it signals elite athleticism AND defensive IQ.

**Hustle stats vs. outcome metrics:**
- Hustle stats (what we have) measure *effort and activity* — good for identifying high-motor players
- They do NOT measure whether that effort translates to good outcomes (a player can contest shots and still give up high FG%)
- Analytically, the strongest signal is **opponent FG% when defending** combined with **defensive play-type efficiency** (PPP allowed by scenario)
- Hustle stats are *inputs*; shot defense quality is *output* — you want both

---

## Available Data Sources (Public, nba_api-accessible)

### New nba_api endpoints we can add

| Endpoint | What it gives us | Key fields |
|---|---|---|
| `LeagueDashPtDefend` | Opponent FG% by zone when player is nearest defender | D_FGM, D_FGA, D_FG_PCT, NORMAL_FG_PCT — filterable by DefenseCategory: Overall, 3PT, 2PT, Less Than 6Ft, Less Than 10Ft, Greater Than 15Ft |
| `DefenseHub` | Comprehensive defensive dashboard | THREEP_DFGPCT, TWOP_DFGPCT, rim defense %, overall plus-minus |
| `SynergyPlayTypes` | Defense by play type | PPP allowed, FG%, EFG%, POSS% for ISO, PRBallHandler, PRRollman, Postup, Spotup, etc. |
| `LeagueDashPtStats` (defensive filter) | Physical effort metrics | DIST_MILES_DEF (miles run on defense), AVG_SPEED_DEF |
| `DraftCombineStats` / `DraftCombineNonStationaryShooting` | Physical measurements | Height, wingspan, standing reach, vertical jump, lane agility, sprint time |
| `LeagueSeasonMatchups` | Head-to-head matchup data | Who guarded whom, outcomes |

### Third-party public sources (scraping or direct)
- **EPM (dunksandthrees.com/epm)** — public, 1997-2026, combines box + play-by-play into OEPM/DEPM
- **RAPTOR (github.com/fivethirtyeight/nba-player-advanced-metrics)** — public CSV, 1997-2026
- **nbarapm.com** — RAPM, DARKO, LEBRON estimates; public
- **Basketball Reference** — scrapable via `basketball-reference-scraper` package

### Known reliability issues (important)
- NBA.com has Cloudflare rate limiting — must add `time.sleep()` between API calls
- `PlayByPlayV2` and `ScoreboardV2` are deprecated for 2025-26 — use V3
- `ShotChartDetail` has reliability issues (missing coordinates, restricted time windows)
- nba_api can break without warning when NBA.com updates endpoints; pin to known-good version

---

## Improvement Opportunities (Ranked by Impact vs. Effort)

### Tier 1 — High impact, directly buildable with existing patterns

**A. Shot defense quality layer (`LeagueDashPtDefend`)**
- Add `opponent_fg_pct_allowed(df, zone='overall')` to `compute_defense.py`
- Filterable by zone: rim (<6ft), paint (<10ft), mid-range, 3PT
- This is the *most requested* metric in real scouting: "how often do shots go in when this player is the nearest defender?"
- Answers questions like: "Who is the best rim defender by actual shot suppression?"

**B. Play-type defensive efficiency (`SynergyPlayTypes`)**
- Add `defensive_playtype_profile(player_id)` — PPP allowed defending ISO, P&R ball handler, P&R roll man, post-up
- Identifies *specialists*: some players are elite P&R defenders but bad in ISO, etc.
- Directly maps to scouting reports: "can he guard the ball handler in pick-and-roll?"

**C. Multi-season trend view**
- We already pull 2024-25 and 2025-26 — but the agent only surfaces 2025-26
- Add a `year_over_year_delta()` function that computes improvement/decline per metric
- Answers: "Is this player trending up or down on hustle?"

### Tier 2 — Medium impact, worth adding

**D. Physical profile integration (`DraftCombineStats`)**
- Pull wingspan, height, lane agility for young players
- Add a `physical_profile(player_name)` lookup
- Combine with defensive stats: "players with 7ft+ wingspan who also lead in deflections"

**E. Defensive speed/distance (`LeagueDashPtStats` with defensive filters)**
- DIST_MILES_DEF and AVG_SPEED_DEF: quantifies how much ground a player covers defensively
- Complements hustle stats: high deflections + high defensive distance = genuinely active defender
- Answers: "Who covers the most ground defensively?"

**F. EPM/RAPTOR integration (third-party)**
- Pull from public GitHub CSV or scrape dunksandthrees.com
- Adds a blended estimate of overall defensive impact (captures what box score misses)
- Useful as a sanity check: "does this player's hustle profile match their overall defensive impact?"

### Tier 3 — Nice to have, more complex

**G. Context-adjusted metrics**
- Adjust stats for opponent quality (easy to deflect against bad offenses)
- On/off defensive rating — requires lineup data, noisy without 3+ seasons
- Scheme tagging — zone vs. man, switching vs. drop coverage

**H. Prospect/college scouting extension**
- Block% + steal% from college (Basketball Reference)
- Age at draft + draft position as predictive inputs
- Combine measurements for draft class evaluation

---

## Recommended Next Step

The single highest-value addition is **Tier 1A: `LeagueDashPtDefend`** — opponent FG% allowed by zone.

It directly answers the question hustle stats can't: "Does this player's defense actually work?"
It's one nba_api call, same structure as our existing pulls, and maps cleanly to new `compute_defense.py` functions like:
- `rim_defense_fg_pct(df)` — who suppresses shots at the rim most
- `perimeter_defense_fg_pct(df)` — who holds 3PT shooters to the lowest FG%

This is the natural next layer on top of what we have.
