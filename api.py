from __future__ import annotations

import os
import math
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from query_router import route
from report import generate_scouting_report_data, compare_players_data


# ── startup: load dataframes once ────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["df"] = pd.read_csv("hustle_stats_2025_26.csv")
        _state["prior_df"] = pd.read_csv("hustle_stats_2024_25.csv")
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Missing required data file: {e.filename}. "
            "Run ./refresh_data.sh to pull data, or check that CSVs were "
            "included in this deployment."
        ) from e
    yield
    _state.clear()


app = FastAPI(title="NBA Scouting Agent", lifespan=lifespan)

_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allowed_origins == "*" else [o.strip() for o in _allowed_origins.split(",")],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    response_type: str   # "answer" | "out_of_scope" | "needs_clarification" | "error"
    answer: str
    function_matched: Optional[str]
    method: Optional[str]
    table: Optional[list[dict]]


class ReportRequest(BaseModel):
    player_name: str
    season: str = "2025-26"


class ReportSectionRow(BaseModel):
    label: str
    qualified: bool
    text: str
    caveats: list[str]
    value: Optional[float] = None
    better: Optional[str] = None   # "higher" | "lower" | None


class ReportSection(BaseModel):
    title: str
    rows: list[ReportSectionRow]


class ReportResponse(BaseModel):
    player_name: str
    season: str
    sections: list[ReportSection]


class CompareRequest(BaseModel):
    player_a: str
    player_b: str
    season: str = "2025-26"


class CompareRow(BaseModel):
    label: str
    a: ReportSectionRow
    b: ReportSectionRow
    winner: Optional[str] = None   # "a" | "b" | None


class CompareSection(BaseModel):
    title: str
    rows: list[CompareRow]


class CompareResponse(BaseModel):
    player_a: str
    player_b: str
    season: str
    sections: list[CompareSection]


# ── helpers ───────────────────────────────────────────────────────────────────

def _sanitize(records: list[dict]) -> list[dict]:
    """Replace NaN/Inf with None so JSON serialization never breaks."""
    out = []
    for row in records:
        clean = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        out.append(clean)
    return out


def _response_type(result: dict) -> str:
    fm = result.get("function_matched") or ""
    if fm == "out_of_scope" or result.get("method") == "error":
        return "out_of_scope"
    if fm == "needs_clarification":
        return "needs_clarification"
    return "answer"


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "current_season_rows": len(_state["df"]),
        "prior_season_rows": len(_state["prior_df"]),
        "groq_key_configured": bool(os.environ.get("GROQ_API_KEY")),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    result = route(
        req.question,
        _state["df"],
        prior_df=_state["prior_df"],
    )

    table = _sanitize(result["table"]) if result.get("table") else None

    return QueryResponse(
        question=result["question"],
        response_type=_response_type(result),
        answer=result["answer"],
        function_matched=result.get("function_matched"),
        method=result.get("method"),
        table=table,
    )


@app.post("/report", response_model=ReportResponse)
def report(req: ReportRequest):
    data = generate_scouting_report_data(req.player_name, req.season)
    return ReportResponse(**data)


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    data = compare_players_data(req.player_a, req.player_b, req.season)
    return CompareResponse(**data)


# ── serve frontend ────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory="chat", html=True), name="frontend")
