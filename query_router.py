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
    SMALL_SAMPLE_THRESHOLD,
)

OUT_OF_SCOPE_MSG = (
    "I don't have data to answer that — this tool only covers "
    "deflections, rim/perimeter contests, box-out efficiency, "
    "hustle-play composites, shot suppression (opponent FG%), "
    "hustle-vs-suppression gap analysis, and Synergy play-type defense "
    "(Isolation, PRBallHandler, PRRollman, Postup, Spotup, Handoff, OffScreen)."
)

_NO_PLAYTYPE_DATA_MSG = (
    "Synergy doesn't publish player-level defensive data for this play type. "
    "Cut and Transition defense are not available at the individual player level — "
    "only team-level data exists for these categories in the Synergy feed."
)

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
            r"pick.*roll.*defense|p.?r.*defense|pick.*roll.*ball.?handler|"
            r"defend.*ball.?handler|ball.?handler.*defense|defend.*pick.*roll",
            re.IGNORECASE,
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
]

_SYSTEM_PROMPT = """You are a routing assistant for an NBA scouting tool.
You have access to exactly seven statistical functions:

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
   Sort column hint: use "positive" for quiet/underrated/effort-pays-off questions, "negative" for hustle-doesn't-translate/busy-but-not-impactful questions, null if unclear.

7. playtype_defense(play_type, min_poss=<per-type default>)
   Measures: PPP (points per possession) allowed by play type from Synergy data. Lower PPP = better defense.
   Valid play types and when to use them:
     "Isolation"    — iso defense, one-on-one defense, defending iso situations
     "PRBallHandler"— pick-and-roll defense (ball handler side), defending the ball handler in P&R
     "PRRollman"    — defending the roll man / pick-and-roll big
     "Postup"       — post defense, defending post-ups
     "Spotup"       — spot-up defense, defending shooters catching and shooting
     "Handoff"      — defending handoffs
     "OffScreen"    — off-screen defense, chasing shooters off screens, navigating screens
   NOT available (no player-level data): Cut, Transition — return out_of_scope for these.
   Sort column hint: use the play type name exactly as listed above (e.g. "Isolation", "PRBallHandler").

None of these functions cover: scoring, shooting efficiency, assists, salary, trade value, draft grades, or anything unrelated to the seven hustle/defense categories above.

Respond ONLY with a JSON object — no prose, no markdown, no explanation:
- If the question maps to one of the seven functions:
  {"function": "<function_name>", "sort_col": "<hint: '2pt', '3pt', 'Overall', 'Less Than 6Ft', 'positive', 'negative', play type name, or null>"}
- If the question is outside the scope of all seven functions:
  {"out_of_scope": true}"""


def _llm_route(question: str) -> dict:
    payload = {
        "model": "llama-3.3-70b-versatile",
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    }
    headers = {
        "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
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
            f"({row['TOTAL_CONTESTED_PER36']} total) in {row['G']} games [{season_label}]."
        )
    elif sort_col == "3pt":
        return (
            f"{row['PLAYER_NAME']} ({row['TEAM_ABBREVIATION']}) leads in perimeter "
            f"closeout contests with {row['CONTESTED_3PT_PER36']} 3PT contests per 36 "
            f"({row['TOTAL_CONTESTED_PER36']} total) in {row['G']} games [{season_label}]."
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


def route(question: str, df: pd.DataFrame, season_label: str = "2025-26") -> dict:
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
    }
