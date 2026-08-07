import json
import os
import re
import time
from typing import Optional
import pandas as pd
import requests
from compute_defense import (
    deflections_per36,
    contest_profile_per36,
    boxout_conversion,
    hustle_iq_composite,
    shot_suppression,
    hustle_vs_suppression_gap,
    playtype_defense,
    year_over_year_delta,
    _YOY_METRIC_MAP,
    SMALL_SAMPLE_THRESHOLD,
)
from compute_offense import (
    playtype_offense,
    format_playtype_offense_answer,
    drive_efficiency,
    format_drive_efficiency_answer,
)

OUT_OF_SCOPE_MSG = (
    "I don't have data to answer that — this tool covers "
    "deflections, rim/perimeter contests, box-out efficiency, "
    "hustle-play composites, shot suppression (opponent FG%), "
    "hustle-vs-suppression gap analysis, Synergy play-type defense "
    "(Isolation, PRBallHandler, PRRollman, Postup, Spotup, Handoff, OffScreen), "
    "Synergy play-type offense "
    "(Isolation, PRBallHandler, PRRollman, Postup, Spotup, Handoff, Cut, OffScreen, Transition), "
    "and drive efficiency (points per drive, drive FG%, playmaking on drives)."
)

_NEEDS_CLARIFICATION_MSG = (
    "I can track year-over-year trends, but I need to know which stat "
    "(deflections, contests, box-outs, or hustle IQ) and which direction "
    "(improving or declining) you're asking about."
)

_NO_PLAYTYPE_DATA_MSG = (
    "Synergy doesn't publish player-level defensive data for this play type. "
    "Cut and Transition defense are not available at the individual player level — "
    "only team-level data exists for these categories in the Synergy feed."
)

# ── Year-over-year routing ────────────────────────────────────────────────────
# Phase 1: detect a clear trend direction (improve OR decline).
# Both must be present for deterministic routing; ambiguous questions fall
# through to the LLM.
_YOY_IMPROVE_PATTERN = re.compile(
    r"improv|trending up|getting better|stepped up|better than last|risen|"
    r"leap|breakout|jump.*stat|stat.*jump",
    re.IGNORECASE,
)
_YOY_DECLINE_PATTERN = re.compile(
    r"declin|dropped? off|regress|trending down|getting worse|fallen off|"
    r"worse than last|slump|lost a step|fall.*off",
    re.IGNORECASE,
)

# Phase 2: detect WHICH of the four YoY-supported metrics the question is about.
# Keyword subsets are drawn directly from the corresponding _RULES patterns so
# the same phrasing resolves to the same metric in both the base and YoY paths.
_YOY_METRIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    # boxout before hustle_iq so "box out" doesn't accidentally hit hustle
    (re.compile(r"box.?out|boxes out|rebound.*position|boxing out|screen.*out", re.IGNORECASE),
     "boxout_conversion"),
    (re.compile(r"deflect|active hand|tip.*pass|pass.*lane", re.IGNORECASE),
     "deflections_per36"),
    (re.compile(r"contest|shot.*contest|rim.*contest|perimeter.*contest|closeout", re.IGNORECASE),
     "contest_profile_per36"),
    (re.compile(r"hustle|loose ball|draw.*charge|charge.*draw|charges|motor|defensive iq", re.IGNORECASE),
     "hustle_iq_composite"),
]


def _yoy_route(question: str) -> Optional[dict]:
    """Return a YoY routing dict if question has both a clear direction AND a
    detectable metric; return None otherwise (falls through to LLM)."""
    improve = bool(_YOY_IMPROVE_PATTERN.search(question))
    decline = bool(_YOY_DECLINE_PATTERN.search(question))

    # Require exactly one direction signal; both present → ambiguous → LLM
    if improve == decline:
        return None

    metric = None
    for pattern, name in _YOY_METRIC_PATTERNS:
        if pattern.search(question):
            metric = name
            break

    if metric is None:
        return None  # no detectable metric → LLM

    direction = "improve" if improve else "decline"
    return {"function": "year_over_year_delta", "sort_col": f"{metric}:{direction}"}


# ─────────────────────────────────────────────────────────────────────────────
_RULES = [
    (
        re.compile(
            r"deflect|active hand|steal.*pass|pass.*lane|tip.*pass",
            re.IGNORECASE,
        ),
        "deflections_per36",
        None,
    ),
    (
        re.compile(
            r"rim protector|shot blocker|contest.*rim|block.*shot|protect.*rim|alter.*shot",
            re.IGNORECASE,
        ),
        "contest_profile_per36",
        "2pt",
    ),
    (
        re.compile(
            r"perimeter defender|closeout|close.*out|contest.*three|contest.*shooter|"
            r"contest.*perimeter|challenge.*shooter",
            re.IGNORECASE,
        ),
        "contest_profile_per36",
        "3pt",
    ),
    (
        re.compile(
            r"contest|challenge.*shot|shot.*challeng",
            re.IGNORECASE,
        ),
        "contest_profile_per36",
        None,
    ),
    (
        re.compile(
            r"box.?out|boxes out|rebound.*position|position.*rebound|boxing out|screen.*out",
            re.IGNORECASE,
        ),
        "boxout_conversion",
        None,
    ),
    # hustle_vs_suppression_gap — checked BEFORE hustle_iq_composite so that
    # translate/effort-vs-results phrasing wins over the generic "hustle" keyword
    # negative gap: high hustle activity, poor suppression (busy but not impactful)
    (
        re.compile(
            r"busy but not impactful|hustle.*don.t translate|hustle.*not.*translate|"
            r"numbers.*don.t translate|numbers.*not.*translate|don.t translate|"
            r"don.t.*translate|not.*translate|hustle.*translate|translate.*hustle|"
            r"numbers.*translate|effort.*result|hustle.*result|hustle.*outcome|"
            r"result.*hustle",
            re.IGNORECASE,
        ),
        "hustle_vs_suppression_gap",
        "negative",
    ),
    # positive gap: low hustle activity, good suppression (quiet but effective)
    (
        re.compile(
            r"quiet but effective|underrated defender|doesn.t show up|"
            r"box score.*defend|translate.*result|translate.*outcome|doesn.t translate",
            re.IGNORECASE,
        ),
        "hustle_vs_suppression_gap",
        "positive",
    ),
    (
        re.compile(
            r"hustle|loose ball|draw.*charge|charge.*draw|high motor|high.?iq|"
            r"defensive iq|motor|scrappy|reads.*game|read.*game|game.*read|"
            r"awareness|instinct|anticipat",
            re.IGNORECASE,
        ),
        "hustle_iq_composite",
        None,
    ),
    # shot_suppression — rim-specific
    (
        re.compile(
            r"suppress.*rim|rim.*suppress|defend.*rim|rim.*defend|"
            r"shot.*block.*pct|opponent.*fg.*rim|rim.*fg|at the rim",
            re.IGNORECASE,
        ),
        "shot_suppression",
        "Less Than 6Ft",
    ),
    # shot_suppression — perimeter/3PT-specific
    (
        re.compile(
            r"suppress.*three|three.*suppress|suppress.*perimeter|perimeter.*suppress|"
            r"opponent.*three.*pct|three.*point.*defend|hold.*three|hold.*shooter",
            re.IGNORECASE,
        ),
        "shot_suppression",
        "3 Pointers",
    ),
    # shot_suppression — overall
    (
        re.compile(
            r"shot suppression|suppress.*shoot|makes.*shoot.*worse|"
            r"opponent.*field goal|opponent.*fg|holds.*shoot.*below|"
            r"defend.*fg|fg.*allow|field goal.*allow",
            re.IGNORECASE,
        ),
        "shot_suppression",
        "Overall",
    ),
    # playtype_defense — cut/transition have no player-level data; checked before
    # generic playtype rules so the specific message fires instead of a miss
    (
        re.compile(r"cut.*defense|defend.*cut|cut.*basket|cutting.*lane", re.IGNORECASE),
        "playtype_defense",
        "Cut",
    ),
    (
        re.compile(r"transition.*defense|defend.*transition|defend.*fast.?break|fast.?break.*defense", re.IGNORECASE),
        "playtype_defense",
        "Transition",
    ),
    # playtype_defense — 7 valid play types
    (
        re.compile(r"iso.*defense|isolation.*defense|defend.*iso|one.on.one.*defense|1.on.1.*defense", re.IGNORECASE),
        "playtype_defense",
        "Isolation",
    ),
    (
        re.compile(
            r"^(?!.*(?:scor|finish|offens|efficien)).*"
            r"(?:pick.*roll.*defen|p.?r.*defen|pick.*roll.*ball.?handler|"
            r"defend.*ball.?handler|guard.*ball.?handler|ball.?handler.*defen|"
            r"defend.*pick.*roll|guard.*pick.*roll)",
            re.IGNORECASE | re.DOTALL,
        ),
        "playtype_defense",
        "PRBallHandler",
    ),
    (
        re.compile(
            r"roll.?man.*defense|defend.*roll.?man|defend.*the.*roll|"
            r"pick.*roll.*big|roll.*man.*defend|roll.*big.*defend",
            re.IGNORECASE,
        ),
        "playtype_defense",
        "PRRollman",
    ),
    (
        re.compile(r"post.*defense|post.?up.*defense|defend.*post|defending.*post", re.IGNORECASE),
        "playtype_defense",
        "Postup",
    ),
    (
        re.compile(
            r"spot.?up.*defense|defend.*spot.?up|defend.*shooter|defending.*shooter|"
            r"clos.*out.*spot|spot.*up.*defend",
            re.IGNORECASE,
        ),
        "playtype_defense",
        "Spotup",
    ),
    (
        re.compile(r"handoff.*defense|defend.*handoff|defending.*handoff|hand.?off.*defend", re.IGNORECASE),
        "playtype_defense",
        "Handoff",
    ),
    (
        re.compile(
            r"off.?screen.*defense|defend.*off.?screen|navigat.*screen|"
            r"chas.*shooter|screen.*defense|off.screen.*defend",
            re.IGNORECASE,
        ),
        "playtype_defense",
        "OffScreen",
    ),
    # ── playtype_offense — all rules require explicit scoring/offensive language
    # so they cannot collide with defensive rules above. The anchor group
    # (scor|finish|efficient|shoot) must appear somewhere in the question.
    # Defensive questions ("defend the iso", "guard the pick and roll") will
    # never contain those terms and fall through to the defensive rules.
    (
        re.compile(
            r"(?=.*(?:scor|finish|efficient|shoot|best at|good at))"
            r"(?:iso.*scor|scor.*iso|best in isolation|one.on.one.*scor|"
            r"iso.*finish|isolation.*offens|isolation.*efficien)",
            re.IGNORECASE,
        ),
        "playtype_offense",
        "Isolation",
    ),
    (
        re.compile(
            r"(?:pick.*roll.*(?:ball.?handler|scor|finish|creat)|"
            r"ball.?handler.*(?:scor|efficient)|"
            r"p\.?r.*ball.?handler.*(?:scor|offens)|"
            r"scor.*(?:pick.*roll|ball.?handler)|"
            r"efficien.*ball.?handler|"
            r"pick.*roll.*offens)",
            re.IGNORECASE,
        ),
        "playtype_offense",
        "PRBallHandler",
    ),
    (
        re.compile(
            r"(?=.*(?:scor|finish|roll))"
            r"(?:roll.?man.*(?:scor|finish|offens)|"
            r"(?:scor|finish).*roll.?man|"
            r"pick.*roll.*finish|roll.*big.*(?:scor|finish)|"
            r"finishing.*(?:roll|the roll)|roll.*man.*offens)",
            re.IGNORECASE,
        ),
        "playtype_offense",
        "PRRollman",
    ),
    (
        re.compile(
            r"(?=.*(?:scor|finish|offens|back.to.the.basket|post))"
            r"(?:post.*scor|scor.*post|back.to.the.basket|"
            r"post.*offens|post.?up.*(?:scor|finish|offens)|"
            r"(?:scor|finish).*post)",
            re.IGNORECASE,
        ),
        "playtype_offense",
        "Postup",
    ),
    (
        re.compile(
            r"(?:spot.?up.*shoot|catch.*shoot|shoot.*off.*catch|"
            r"catch.and.shoot|spot.?up.*(?:scor|offens|efficien)|"
            r"stand.*still.*shoot|stationar.*shoot)",
            re.IGNORECASE,
        ),
        "playtype_offense",
        "Spotup",
    ),
    (
        re.compile(
            r"(?=.*(?:scor|shoot|finish|efficient))"
            r"(?:handoff.*(?:scor|shoot|finish)|"
            r"(?:scor|shoot|finish).*handoff|"
            r"dribble.*handoff|hand.?off.*offens|"
            r"scor.*off.*handoff|handoff.*efficien)",
            re.IGNORECASE,
        ),
        "playtype_offense",
        "Handoff",
    ),
    (
        re.compile(
            r"(?:who cuts the most|best cutter|most cuts|"
            r"cut.*(?:most|best|frequen|offens)|"
            r"cutter.*(?:league|best|most)|"
            r"(?:most|best).*cutter)",
            re.IGNORECASE,
        ),
        "playtype_offense",
        "Cut",
    ),
    (
        re.compile(
            r"(?=.*(?:scor|shoot|offens|efficien))"
            r"(?:off.?screen.*(?:scor|shoot|offens)|"
            r"(?:scor|shoot).*off.*screen|"
            r"movement.*shoot|coming.*off.*screen|"
            r"off.screen.*efficien|screen.*(?:shoot|scor).*offens)",
            re.IGNORECASE,
        ),
        "playtype_offense",
        "OffScreen",
    ),
    (
        re.compile(
            r"(?:transition.*(?:scor|offens|finish|efficien)|"
            r"fast.?break.*(?:scor|offens|finish)|"
            r"(?:scor|finish).*transition|"
            r"(?:scor|finish).*fast.?break|"
            r"run.*(?:floor|break).*scor|open.*floor.*scor)",
            re.IGNORECASE,
        ),
        "playtype_offense",
        "Transition",
    ),
    # ── drive_efficiency (LeagueDashPtStats, Drives) ──────────────────────────
    (
        re.compile(
            r"(?:drive.*(?:efficien|scor|point)|point.*per.*drive|"
            r"most efficient driver|best driver.*basket|"
            r"driving.*(?:efficien|scor)|attack.*(?:basket|rim).*(?:efficien|scor)|"
            r"slash.*(?:efficien|scor)|off.the.dribble.*(?:efficien|scor))",
            re.IGNORECASE,
        ),
        "drive_efficiency",
        None,
    ),
]

_SYSTEM_PROMPT = """You are a routing assistant for an NBA scouting tool.
You have access to exactly ten statistical functions:

1. deflections_per36(df, min_minutes=15, min_games=40)
   Measures: deflections per 36 minutes.
   Use for: questions about active hands, disrupting passes, tipping the ball.

2. contest_profile_per36(df, min_minutes=15, min_games=40)
   Measures: contested 2PT shots per 36, contested 3PT shots per 36, total contested per 36.
   Use for: questions about shot contesting, rim protection (2PT focus), perimeter closeouts (3PT focus).
   Sort column hint: use "2pt" for rim-protector questions, "3pt" for perimeter/closeout questions.

3. boxout_conversion(df, min_boxouts=20, min_games=40)
   Measures: % of box outs that result in a personal rebound (BOXOUT_CONV_RATE).
   Use for: questions about boxing out, rebounding position, screen-out efficiency.

4. hustle_iq_composite(df, min_minutes=15, min_games=40)
   Measures: weighted z-score composite of def_loose_balls_recovered per 36 and charges_drawn per 36.
   Use for: questions about hustle, loose balls, drawing charges, motor, defensive IQ.

5. shot_suppression(df, category='Overall', min_def_fga=100)
   Measures: opponent FG% when this player is the nearest defender vs. their normal FG% (PCT_PLUSMINUS).
   Negative PCT_PLUSMINUS = shooters perform worse vs. this defender = good defense.
   Use for: questions about shot suppression, opponent field goal percentage, making shooters worse.
   Sort column hint: use "Less Than 6Ft" for rim defense, "3 Pointers" for perimeter, "Overall" otherwise.

6. hustle_vs_suppression_gap(hustle_df, defend_df, min_minutes=15, min_def_fga=100, min_games=40)
   Measures: gap between hustle activity rank and shot suppression rank, within position group.
   Positive GAP = low hustle activity but good shot suppression (quiet but effective defender).
   Negative GAP = high hustle activity but poor shot suppression (busy but not impactful).
   Use for: questions about underrated defenders, effort vs. results, hustle that doesn't translate.
   Sort column hint: use "positive" for quiet/underrated questions, "negative" for hustle-doesn't-translate questions.

7. playtype_defense(play_type, min_poss=<per-type default>)
   Measures: PPP (points per possession) ALLOWED by play type. Lower PPP = better defense.
   ONLY for DEFENSIVE questions — "defending", "guarding", "stopping", "contesting", "how does he defend".
   Valid play types: "Isolation", "PRBallHandler", "PRRollman", "Postup", "Spotup", "Handoff", "OffScreen".
   NOT available: Cut, Transition (no player-level defensive data for these).
   Sort column hint: the play type name exactly (e.g. "Isolation", "PRBallHandler").

8. playtype_offense(play_type, min_poss=<per-type default>)
   Measures: PPP (points per possession) SCORED by play type. Higher PPP = better offense.
   ONLY for OFFENSIVE questions — "scoring", "finishing", "shooting", "efficient at", "best at scoring".
   This is the OPPOSITE of playtype_defense — do not confuse them.
   Valid play types: "Isolation", "PRBallHandler", "PRRollman", "Postup", "Spotup", "Handoff",
                     "Cut", "OffScreen", "Transition".
   Cut and Transition ARE available offensively (unlike defensively).
   Special: Cut is ranked by possession volume (who cuts most), not PPP — cutting efficiency is
   uniformly high and doesn't meaningfully differentiate players.
   Sort column hint: the play type name exactly (e.g. "Isolation", "Cut", "Transition").

   DISAMBIGUATION — offense vs. defense:
   - "How does he defend the pick and roll?" → playtype_defense, PRBallHandler
   - "How does he score in the pick and roll?" → playtype_offense, PRBallHandler
   - "Who's the best isolation defender?" → playtype_defense, Isolation
   - "Who's the best isolation scorer?" → playtype_offense, Isolation
   The key signal is whether the question is about allowing points (defense) or scoring points (offense).

9. year_over_year_delta(current_df, prior_df, metric)
   Measures: season-over-season change in a hustle metric.
   Use for: improvement, decline, trending up/down in a defensive metric.
   Supported metrics: deflections_per36, contest_profile_per36, boxout_conversion, hustle_iq_composite.
   Sort column hint: "<metric>:improve" or "<metric>:decline".

10. drive_efficiency(min_drives_per_game=None)
    Measures: points scored per drive (PTS_PER_DRIVE), drive FG%, and drive playmaking
    (DRIVE_PASSES_PCT, DRIVE_AST_PCT, DRIVE_TOV_PCT).
    Use for: questions about driving to the basket, attacking off the dribble, slashing efficiency.
    Sort column hint: not used, leave null.

None of these functions cover: salary, trade value, draft grades, injuries, assists, or anything outside
hustle/defense/play-type offense/drive efficiency as described above.

Respond ONLY with a JSON object — no prose, no markdown, no explanation. Three possible responses:
- If the question maps to one of the ten functions:
  {"function": "<function_name>", "sort_col": "<hint>"}
- If the question is about a stat this tool tracks but is too vague or underspecified:
  {"needs_clarification": true}
- If the question is genuinely outside what this tool covers:
  {"out_of_scope": true}"""


class MissingGroqKeyError(RuntimeError):
    """Raised when the LLM fallback is needed but GROQ_API_KEY isn't set."""


def _llm_route(question: str) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise MissingGroqKeyError(
            "GROQ_API_KEY is not set — cannot use the LLM fallback for this question."
        )

    payload = {
        "model": "llama-3.3-70b-versatile",
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    delay = 4
    for attempt in range(4):
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        if resp.status_code == 429:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        if not raw:
            time.sleep(delay)
            delay *= 2
            continue
        raw = raw.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    raise RuntimeError(f"Groq rate-limited after 4 attempts: {resp.text[:200]}")


def _deterministic_route(question: str) -> Optional[dict]:
    # YoY checked first — a trend question with a clear direction+metric wins
    # before any base-metric rule can claim the "improve/decline" phrasing.
    yoy = _yoy_route(question)
    if yoy is not None:
        return yoy
    for pattern, func_name, hint in _RULES:
        if pattern.search(question):
            return {"function": func_name, "sort_col": hint}
    return None


def _format_deflections(row: pd.Series, season_label: str) -> str:
    return (
        f"{row['PLAYER_NAME']} ({row['TEAM_ABBREVIATION']}) leads in deflections per 36 "
        f"with {row['DEFLECTIONS_PER36']} in {row['G']} games "
        f"averaging {row['MIN']} minutes [{season_label}]."
    )


def _format_contest(row: pd.Series, sort_col: Optional[str], season_label: str) -> str:
    if sort_col == "2pt":
        return (
            f"{row['PLAYER_NAME']} ({row['TEAM_ABBREVIATION']}) leads in rim contests "
            f"with {row['CONTESTED_2PT_PER36']} 2PT contests per 36 "
            f"({row['TOTAL_CONTESTED_PER36']} total) in {row['G']} games [{season_label}]. "
            f"NOTE: This measures contest volume at the rim, not shot suppression — contesting "
            f"frequently does not mean opponents miss. The volume leaders and the suppression "
            f"leaders differ significantly at the rim (9 of the top-10 in contest volume are "
            f"not in the top-10 in rim suppression; Clingan is #1 in volume but only #21 in "
            f"suppression, Gobert is #10 in volume but #29 in suppression). "
            f"For true rim shot suppression, see shot_suppression('Less Than 6Ft')."
        )
    elif sort_col == "3pt":
        return (
            f"{row['PLAYER_NAME']} ({row['TEAM_ABBREVIATION']}) leads in perimeter "
            f"closeout contests with {row['CONTESTED_3PT_PER36']} 3PT contests per 36 "
            f"({row['TOTAL_CONTESTED_PER36']} total) in {row['G']} games [{season_label}]. "
            f"NOTE: This measures contest volume, not shot suppression — high contest counts "
            f"do not guarantee shooters miss. Volume leaders and suppression leaders differ "
            f"significantly at the perimeter (e.g. Clingan leads in 3PT contest volume but "
            f"opponents shoot above their normal rate against him). "
            f"For true perimeter suppression, see shot_suppression('3 Pointers')."
        )
    return (
        f"{row['PLAYER_NAME']} ({row['TEAM_ABBREVIATION']}) leads in total shot contests "
        f"per 36 with {row['TOTAL_CONTESTED_PER36']} "
        f"({row['CONTESTED_2PT_PER36']} at rim, {row['CONTESTED_3PT_PER36']} on perimeter) "
        f"in {row['G']} games [{season_label}]."
    )


def _format_boxout(row: pd.Series, season_label: str) -> str:
    pct = round(row["BOXOUT_CONV_RATE"] * 100, 1)
    return (
        f"{row['PLAYER_NAME']} ({row['TEAM_ABBREVIATION']}) has the highest box-out "
        f"conversion rate at {pct}% ({row['BOX_OUT_PLAYER_REBS']} rebounds / "
        f"{row['BOX_OUTS']} box outs per game) in {row['G']} games [{season_label}]."
    )


def _format_shot_suppression(row: pd.Series, category: str, season_label: str) -> str:
    diff = row["PCT_PLUSMINUS"]
    direction = "worse" if diff < 0 else "better"
    return (
        f"{row['PLAYER_NAME']} ({row['PLAYER_LAST_TEAM_ABBREVIATION']}) leads in shot suppression "
        f"({category}) with opponents shooting {row['DEF_FG_PCT']:.1%} vs. their normal "
        f"{row['NORMAL_FG_PCT']:.1%} — {abs(diff):.1%} {direction} than average "
        f"({row['D_FGA'] if 'D_FGA' in row.index else row.get('FGA_LT_06', row.get('FG3A', row.get('FG2A', '?')))} FGA defended) "
        f"in {row['G']} games [{season_label}]."
    )


def _format_gap(row: pd.Series, season_label: str) -> str:
    gap_sign = "positive" if row["GAP"] > 0 else "negative"
    label = "quiet but effective" if row["GAP"] > 0 else "busy but not impactful"
    return (
        f"{row['PLAYER_NAME']} ({row['TEAM_ABBREVIATION']}, {row['PLAYER_POSITION']}) "
        f"has the largest {gap_sign} hustle-vs-suppression gap (GAP={row['GAP']:+.1f}) — "
        f"{label}. Hustle activity rank: {row['HUSTLE_ACTIVITY_RANK']:.0f}, "
        f"shot suppression rank: {row['SUPPRESSION_RANK']:.0f} (within position group). "
        f"PCT_PLUSMINUS: {row['PCT_PLUSMINUS']:+.3f} [{season_label}]. "
        f"NOTE: GAP is a custom rank-difference metric, not an official NBA stat."
    )


def _format_hustle_iq(row: pd.Series, season_label: str) -> str:
    return (
        f"{row['PLAYER_NAME']} ({row['TEAM_ABBREVIATION']}) ranks highest on the "
        f"Hustle IQ Composite (score: {row['HUSTLE_IQ_COMPOSITE']}) — "
        f"{row['DEF_LOOSE_BALLS_PER36']} def loose balls/36 and "
        f"{row['CHARGES_PER36']} charges drawn/36 in {row['G']} games [{season_label}]. "
        f"NOTE: Hustle IQ Composite is a weighted z-score (60% def loose balls + "
        f"40% charges drawn per 36). This is NOT an official NBA stat."
    )


_PLAYTYPE_LABEL = {
    "Isolation":    "isolation defense",
    "PRBallHandler":"pick-and-roll ball-handler defense",
    "PRRollman":    "pick-and-roll roll-man defense",
    "Postup":       "post-up defense",
    "Spotup":       "spot-up defense",
    "Handoff":      "handoff defense",
    "OffScreen":    "off-screen defense",
}

_PRROLLMAN_CAVEAT = (
    "NOTE: This reflects PPP allowed on possessions the offense chose to attack this player "
    "in the pick-and-roll. Elite rim protectors may rank lower here because opponents avoid "
    "attacking them — only the possessions where the offense attacked get logged. "
    "Do not read this as 'best rim protector.' Pair with shot suppression (rim category) "
    "for a fuller picture of interior defense."
)


def _format_playtype(row: pd.Series, play_type: str, season_label: str) -> str:
    label = _PLAYTYPE_LABEL.get(play_type, play_type)
    poss = int(row["POSS"])
    sample_flag = f" (small sample — {poss} possessions)" if poss < SMALL_SAMPLE_THRESHOLD else ""
    base = (
        f"{row['PLAYER_NAME']} ({row['TEAM_ABBREVIATION']}) leads in {label} "
        f"with {row['PPP']} PPP allowed ({row['FG_PCT']:.1%} FG%) on "
        f"{poss} possessions{sample_flag} [{season_label}]."
    )
    if play_type == "PRRollman":
        return f"{base} {_PRROLLMAN_CAVEAT}"
    return base


_YOY_METRIC_LABEL = {
    "deflections_per36":    "deflections per 36",
    "contest_profile_per36": "total contested shots per 36",
    "boxout_conversion":    "box-out conversion rate",
    "hustle_iq_composite":  "Hustle IQ Composite",
}
_YOY_METRIC_COL = {k: v for k, (_, v) in _YOY_METRIC_MAP.items()}


def _format_yoy(row: pd.Series, metric: str, direction: str, season_label: str, prior_label: str) -> str:
    metric_col = _YOY_METRIC_COL[metric]
    label = _YOY_METRIC_LABEL[metric]
    cur_val = row[f"{metric_col}_CUR"]
    prior_val = row[f"{metric_col}_PRIOR"]
    delta = row["DELTA"]
    verb = "improved the most" if direction == "improve" else "declined the most"

    if metric == "boxout_conversion":
        cur_str = f"{cur_val * 100:.1f}%"
        prior_str = f"{prior_val * 100:.1f}%"
        delta_str = f"{delta * 100:+.1f}pp"
    elif metric == "hustle_iq_composite":
        cur_str = f"{cur_val:.3f}"
        prior_str = f"{prior_val:.3f}"
        delta_str = f"{delta:+.3f}"
    else:
        cur_str = f"{cur_val:.2f}"
        prior_str = f"{prior_val:.2f}"
        delta_str = f"{delta:+.2f}"

    return (
        f"{row['PLAYER_NAME']} ({row['TEAM_CUR']}) has {verb} in {label} "
        f"({prior_label}: {prior_str} → {season_label}: {cur_str}, DELTA={delta_str}) "
        f"in {int(row['G_CUR'])} games this season vs. {int(row['G_PRIOR'])} last season."
    )


def route(
    question: str,
    df: pd.DataFrame,
    season_label: str = "2025-26",
    prior_df: Optional[pd.DataFrame] = None,
    prior_label: str = "2024-25",
) -> dict:
    routing = _deterministic_route(question)
    method = "deterministic"

    if routing is None:
        try:
            routing = _llm_route(question)
            method = "llm_fallback"
        except Exception as e:
            return {
                "question": question,
                "method": "error",
                "function_matched": None,
                "answer": f"Routing error: {e}",
            }

    if routing.get("out_of_scope"):
        return {
            "question": question,
            "method": method,
            "function_matched": "out_of_scope",
            "answer": OUT_OF_SCOPE_MSG,
        }

    if routing.get("needs_clarification"):
        return {
            "question": question,
            "method": method,
            "function_matched": "needs_clarification",
            "answer": _NEEDS_CLARIFICATION_MSG,
        }

    func_name = routing.get("function")
    sort_col = routing.get("sort_col")

    try:
        if func_name == "deflections_per36":
            result = deflections_per36(df)
            top = result.iloc[0]
            answer = _format_deflections(top, season_label)

        elif func_name == "contest_profile_per36":
            result = contest_profile_per36(df)
            if sort_col == "3pt":
                result = result.sort_values("CONTESTED_3PT_PER36", ascending=False).reset_index(drop=True)
            elif sort_col == "2pt":
                result = result.sort_values("CONTESTED_2PT_PER36", ascending=False).reset_index(drop=True)
            top = result.iloc[0]
            answer = _format_contest(top, sort_col, season_label)

        elif func_name == "boxout_conversion":
            result = boxout_conversion(df)
            top = result.iloc[0]
            answer = _format_boxout(top, season_label)

        elif func_name == "hustle_iq_composite":
            result = hustle_iq_composite(df)
            top = result.iloc[0]
            answer = _format_hustle_iq(top, season_label)

        elif func_name == "shot_suppression":
            category = sort_col if sort_col in ("Overall", "3 Pointers", "2 Pointers", "Less Than 6Ft") else "Overall"
            csv_map = {
                "Overall":        "shot_defense_overall_2025_26.csv",
                "3 Pointers":     "shot_defense_3pt_2025_26.csv",
                "2 Pointers":     "shot_defense_2pt_2025_26.csv",
                "Less Than 6Ft":  "shot_defense_rim_2025_26.csv",
            }
            defend_df = pd.read_csv(csv_map[category])
            result = shot_suppression(defend_df, category=category)
            top = result.iloc[0]
            answer = _format_shot_suppression(top, category, season_label)

        elif func_name == "hustle_vs_suppression_gap":
            defend_df = pd.read_csv("shot_defense_overall_2025_26.csv")
            result = hustle_vs_suppression_gap(df, defend_df)
            if sort_col == "negative":
                result = result.sort_values("GAP", ascending=True).reset_index(drop=True)
            top = result.iloc[0]
            answer = _format_gap(top, season_label)

        elif func_name == "playtype_defense":
            play_type = sort_col  # sort_col carries the play_type name for this function
            if play_type in ("Cut", "Transition"):
                return {
                    "question": question,
                    "method": method,
                    "function_matched": f"playtype_defense:{play_type}",
                    "answer": _NO_PLAYTYPE_DATA_MSG,
                }
            result = playtype_defense(play_type)
            top = result.iloc[0]
            answer = _format_playtype(top, play_type, season_label)

        elif func_name == "playtype_offense":
            play_type = sort_col  # sort_col carries the play_type name
            result = playtype_offense(play_type)
            top = result.iloc[0]
            # compute total_poss for OffScreen caveat threshold check
            raw_csv = f"playtype_offense_{play_type.lower()}_2025_26.csv"
            _raw = pd.read_csv(raw_csv)
            _gp_rows = _raw[_raw["PLAYER_NAME"] == top["PLAYER_NAME"]]["GP"]
            total_poss = float(top["POSS"] * _gp_rows.values[0]) if not _gp_rows.empty else None
            answer = format_playtype_offense_answer(top, play_type, season_label, total_poss=total_poss)

        elif func_name == "drive_efficiency":
            result = drive_efficiency()
            top = result.iloc[0]
            answer = format_drive_efficiency_answer(top, season_label)

        elif func_name == "year_over_year_delta":
            # sort_col encodes "metric:direction" e.g. "deflections_per36:decline"
            if prior_df is None:
                return {
                    "question": question,
                    "method": method,
                    "function_matched": func_name,
                    "answer": "Year-over-year comparison requires a prior season dataframe. Pass prior_df to route().",
                }
            if not sort_col or ":" not in sort_col:
                return {
                    "question": question,
                    "method": method,
                    "function_matched": "needs_clarification",
                    "answer": _NEEDS_CLARIFICATION_MSG,
                }
            metric, direction = sort_col.rsplit(":", 1)
            if metric not in _YOY_METRIC_MAP or direction not in ("improve", "decline"):
                return {
                    "question": question,
                    "method": method,
                    "function_matched": "needs_clarification",
                    "answer": _NEEDS_CLARIFICATION_MSG,
                }
            result = year_over_year_delta(df, prior_df, metric=metric)
            if direction == "decline":
                result = result.sort_values("DELTA", ascending=True).reset_index(drop=True)
            top = result.iloc[0]
            answer = _format_yoy(top, metric, direction, season_label, prior_label)

        else:
            return {
                "question": question,
                "method": method,
                "function_matched": func_name,
                "answer": OUT_OF_SCOPE_MSG,
            }

    except Exception as e:
        return {
            "question": question,
            "method": method,
            "function_matched": func_name,
            "answer": f"Function execution error: {e}",
        }

    return {
        "question": question,
        "method": method,
        "function_matched": func_name,
        "answer": answer,
        "table": result.head(25).to_dict(orient="records"),
    }
