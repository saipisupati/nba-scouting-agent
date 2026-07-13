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


# ── startup: load dataframes once ────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["df"] = pd.read_csv("hustle_stats_2025_26.csv")
    _state["prior_df"] = pd.read_csv("hustle_stats_2024_25.csv")
    yield
    _state.clear()


app = FastAPI(title="NBA Scouting Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


# ── serve frontend ────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory="chat", html=True), name="frontend")
