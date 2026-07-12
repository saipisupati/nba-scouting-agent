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
)

OUT_OF_SCOPE_MSG = (
    "I don't have data to answer that — this tool only covers "
    "deflections, rim/perimeter contests, box-out efficiency, "
    "and hustle-play composites."
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
]

_SYSTEM_PROMPT = """You are a routing assistant for an NBA scouting tool.
You have access to exactly four statistical functions:

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

None of these functions cover: scoring, shooting efficiency, assists, salary, trade value, draft grades, or anything unrelated to the four hustle/defense categories above.

Respond ONLY with a JSON object — no prose, no markdown, no explanation:
- If the question maps to one of the four functions:
  {"function": "<function_name>", "sort_col": "<optional hint: '2pt' or '3pt' or null>"}
- If the question is outside the scope of all four functions:
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


def _format_hustle_iq(row: pd.Series, season_label: str) -> str:
    return (
        f"{row['PLAYER_NAME']} ({row['TEAM_ABBREVIATION']}) ranks highest on the "
        f"Hustle IQ Composite (score: {row['HUSTLE_IQ_COMPOSITE']}) — "
        f"{row['DEF_LOOSE_BALLS_PER36']} def loose balls/36 and "
        f"{row['CHARGES_PER36']} charges drawn/36 in {row['G']} games [{season_label}]. "
        f"NOTE: Hustle IQ Composite is a weighted z-score (60% def loose balls + "
        f"40% charges drawn per 36). This is NOT an official NBA stat."
    )


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
