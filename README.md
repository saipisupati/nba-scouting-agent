# NBA Scouting Agent

A natural-language scouting analytics tool built on public NBA data. Ask a defensive or offensive question in plain English, get a ranked answer backed by real numbers — no hallucinated statistics, no vague summaries.

Live at `http://localhost:8000` when the server is running.

---

## What This Is

The agent accepts free-text scouting questions and routes them to the right analytic function. Questions like "who contests the most shots at the rim?" or "which players have improved the most in deflections this year?" resolve directly to ranked player tables pulled from `nba_api` data. The interface is a single-page chat UI with suggestion chips, a stat card for the top result, and a ranked mini-list that's always visible.

The routing architecture is deterministic-first: a regex rule set handles the majority of recognizable question patterns without touching a language model. The LLM (Groq / Llama-3.3-70b) is used only for structured classification decisions — routing ambiguous questions to the right function, or returning a `needs_clarification` response when a question is genuinely underspecified. The LLM never generates statistics, rankings, or player names. Every number in every answer comes directly from the underlying data.

---

## What This Is Not

This does not compete with what NBA front offices actually use for decisions.

NBA teams have access to:
- **Second Spectrum** player tracking data — precise on-ball defensive positioning, coverage assignments, foot speed, help rotations, and more, derived from arena camera systems
- **Synergy full subscriptions** — play-by-play tagging with clip access and opponent-adjusted context
- **Medical and biometric data** — injury history, load management metrics, sleep and recovery tracking
- **Proprietary draft intelligence** — physical testing results, psychological profiling, private workouts
- **Lineup-level context** — on/off defensive rating, scheme tagging (zone vs. man, switching vs. drop), opponent quality adjustments

This tool is built entirely on public `nba_api` endpoints. It covers player hustle activity, shot suppression by zone, Synergy play-type efficiency (offensive and defensive), and year-over-year trends. That is a meaningful slice of what matters, but it is not the whole picture, and the tool says so explicitly when questions land outside its data coverage.

---

## Why It Exists

The value is not the data — it's the interface.

In most sports organizations, analytics analysts are the bottleneck between a question and an answer. A coach or scout has a question at 10pm; the analyst who can run the query isn't available until morning. A simple question — "who in this draft class has the best pick-and-roll defense?" — might take 20 minutes for an analyst to pull and format correctly, and zero seconds to ask out loud.

A trustworthy natural-language interface over an analytics dataset changes that calculus. The key word is trustworthy. An interface that sometimes returns hallucinated statistics or quietly misroutes questions to the wrong function is worse than no interface at all, because it erodes confidence in the data itself.

This project is a demonstration of how to build that interface correctly: with honest scope boundaries, explicit caveats where the data has known limitations, and a routing architecture that fails loudly rather than silently.

---

## Architecture

```
User question
     │
     ▼
Deterministic regex router (_RULES, ~20 rules)
     │                    │
     │ match              │ no match
     │                    ▼
     │           LLM structured classifier (Groq)
     │                    │
     │         ┌──────────┼──────────┐
     │         │          │          │
     │    function    out_of_scope  needs_clarification
     │         │
     └────►  compute function
               │
               ▼
         ranked DataFrame → JSON → chat UI
```

**Backend:** FastAPI (`api.py`), serving both the `/query` endpoint and the static frontend. Two DataFrames (current + prior season) are loaded once at startup via FastAPI's lifespan context manager and passed to every query.

**Compute layer:** `compute_defense.py` and `compute_offense.py` contain pure functions that take DataFrames and return ranked DataFrames. No side effects, no network calls, no LLM involvement.

**Router:** `query_router.py` handles routing, formatting, and answer construction. The LLM is called only when the regex rules produce no match.

**Frontend:** Single-file vanilla JS chat UI (`chat/index.html`). No framework, no build step. Suggestion chips, stat card with mini top-5 always visible, follow-up chips, full-table toggle.

---

## Current Data Coverage

| Layer | Source | Functions |
|---|---|---|
| Hustle activity | `LeagueHustleStatsPlayer` | `deflections_per36`, `contest_profile_per36`, `boxout_conversion`, `hustle_iq_composite` |
| Shot suppression | `LeagueDashPtDefend` | `shot_suppression` (Overall, 2PT, 3PT, rim) |
| Activity vs. outcome gap | derived | `hustle_vs_suppression_gap` |
| Defensive play-type efficiency | `SynergyPlayTypes` (Defensive) | `playtype_defense` (7 categories) |
| Offensive play-type efficiency | `SynergyPlayTypes` (Offensive) | `playtype_offense` (9 categories, including Cut + Transition) |
| Drive efficiency | `LeagueDashPtStats` | `drives_2025_26.csv` (exploration complete, routing TBD) |
| Year-over-year trends | derived from 2024-25 + 2025-26 | `year_over_year_delta` |

Seasons covered: 2025-26 (current), 2024-25 (prior, for YoY comparisons).

---

## Design Principles

These aren't aspirational guidelines — they're conclusions from building this specific tool and finding out what broke.

### 1. Deterministic routing first; LLM only for structured decisions

The LLM never touches a number. It classifies which function to call (or returns `out_of_scope` / `needs_clarification`). Every statistic, ranking, and player name in an answer is computed directly from the underlying data.

This means the tool can be tested exhaustively. `test_router.py` runs 23 questions and checks that each routes to the right function with the right parameters. That test suite is the correctness guarantee — not the LLM.

### 2. Activity metrics and outcome metrics diverge more than they agree

The most consistent finding across building this tool: metrics that measure how much a player *does* something frequently disagree with metrics that measure whether doing it *works*.

Concrete examples found in this data:

- **Alex Caruso** leads the league in deflections per 36 minutes. His shot suppression numbers are unremarkable — he creates turnovers, but doesn't hold opponents to below-average FG% when defending shots directly.
- **Donovan Clingan** is #1 in 3PT contest volume (4.82 contested 3PT per 36) but ranks #279 in 3PT shot suppression — opponents shoot *above* their normal rate against him on perimeter contests. He's closing out but not affecting the shot.
- The same gap exists at the rim: Clingan is #1 in 2PT contest volume but #21 in rim suppression. Gobert is #10 in 2PT contest volume but #29 in suppression.
- **PRRollman selection effect**: elite rim protectors like Gobert show misleading PPP-allowed numbers in pick-and-roll defense because opposing offenses *avoid attacking them*. Only the possessions where the offense chose to attack get logged — which systematically underrepresents the best defenders.

Every contest-volume answer in the tool includes a caveat pointing to the corresponding suppression function. Every PRRollman answer includes the selection-effect caveat. These aren't disclaimers added as an afterthought — they're the correct answer to the question.

### 3. Sample-size floors should be per-category, not one flat number

A blanket `min_games=40` threshold doesn't work across play types with wildly different volume distributions. Postup has 149 players; Spotup has 400. PRRollman possessions per game peak at 5.2; Spotup peaks at 6.6. The right floor for each category is derived from its actual possession distribution (p25–p35 of `POSS`, combined with a minimum total-possessions check of 30).

The total-possessions check (`POSS/g × GP ≥ 30`) catches a class of errors the per-game floor misses: players like Josh Minott (16 GP, 0.9 poss/g = 14 total possessions) who top efficiency lists on micro-samples and would otherwise appear as the #1 handoff scorer in the league.

### 4. `needs_clarification` is not the same as `out_of_scope`

"Who's trending up defensively this year?" is in scope — the tool has year-over-year data. But it's underspecified: trending up in *which* metric, and which direction? Returning `out_of_scope` would be a lie. Returning a generic answer by guessing a metric would be worse.

The router has a third response type — `needs_clarification` — with its own distinct UI treatment (purple badge, separate message style). Questions that are in-scope but ambiguous get an honest "I need more information" response rather than a silent best-guess or a false rejection. Conflating underspecified with out-of-scope trains users to distrust the tool's scope claims.

### 5. AI-assisted development requires the same verification discipline as any other output

Several consequential bugs in this project were introduced during AI-assisted development and caught only through explicit verification:

- **The fabricated test pass**: An LLM-assisted code change reported all 23 routing tests passing. Manual re-run showed Q16 ("which player has good hustle numbers that don't translate to results?") was routing to `hustle_iq_composite` instead of `hustle_vs_suppression_gap`. The regex rule ordering had been silently changed, and the test result had been described rather than actually run.
- **The rank-pool mismatch**: The `hustle_vs_suppression_gap` function was described in comments as "positive GAP = high hustle activity + poor suppression." The actual arithmetic was the opposite. The gap had the right magnitude but the wrong semantic direction — answers about "quiet but effective" defenders were returning the opposite set of players. Caught by manually checking five known players against the output.
- **The sign-convention bug**: `PCT_PLUSMINUS` in `shot_suppression` is negative when the defender is good (opponents shoot below their normal rate). Early formatting code was displaying this without a sign, making elite defenders appear to have a negative metric rather than a favorable one. Caught during the first manual review of formatted output.

The pattern in all three: the code ran without errors, the outputs looked plausible, and the mistake was only visible when the output was checked against known ground truth. Automated tests catch regressions; they don't catch plausible-looking wrong answers. The verification step is not optional.

---

## Running the Tool

**Requirements:** Python 3.9+, `fastapi`, `uvicorn`, `pandas`, `requests`, `nba_api`. Set `GROQ_API_KEY` in environment for LLM fallback (free tier sufficient).

```bash
# Pull fresh data (only needed once, or to update)
python hustle_stats.py

# Start the server
GROQ_API_KEY=your_key uvicorn api:app --port 8000

# Open in browser
open http://localhost:8000
```

The CSVs are not committed to the repo — they're generated locally by the exploration and data-pull scripts. The `explore_*.py` scripts document exactly which endpoints were called and what was found.

---

## Future Direction

The most natural extension is **prospect scouting across college and international data**, which is where even NBA teams' proprietary systems have the least coverage.

Second Spectrum and Synergy track NBA games comprehensively. College tracking is inconsistent across conferences. International leagues (EuroLeague, Liga ACB, NBL) have even less. The public data gap is largest exactly where the scouting judgment call is hardest: evaluating a 20-year-old playing in the Turkish BSL or the French Pro A.

Public data sources for this layer include Basketball Reference (college box scores back to 1993), Sports Reference (international stats), and the NBA draft combine measurements via `nba_api`. The block% + steal% combination from college remains one of the most predictive signals of NBA defensive impact from publicly available data — a finding that holds up across multiple published analyses and that this tool's architecture is well-positioned to surface.

The same routing architecture applies: deterministic rules for well-defined questions, LLM classification for ambiguous ones, explicit scope limits for questions that require data this tool doesn't have. The honesty about scope is a feature, not a limitation — it's what makes the tool usable for decisions that actually matter.
