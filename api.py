from __future__ import annotations

import os
import math
from contextlib import asynccontextmanager
from typing import Optional

import json
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from query_router import route
from report import generate_scouting_report_data, compare_players_data
from data_schema import (
    HUSTLE_COLUMNS, SHOT_DEFENSE_COLUMNS, PLAYTYPE_COLUMNS,
    DRIVES_COLUMNS, DRAFT_CLASS_COLUMNS, missing_columns,
)


# ── startup: load dataframes once, validate every data file's schema ─────────

_state: dict = {}

_MANIFEST_PATH = "data/data_manifest.json"

# (file path, required columns) for every CSV the app depends on -- checked
# at startup so a broken/stale file fails loudly here, with the exact file
# and exact missing column named, rather than surfacing later as a KeyError
# deep inside whichever compute function happens to touch it first (or
# worse, silently producing a wrong answer if the missing column wasn't
# strictly required by pandas' own indexing but changed the data's meaning).
_PLAYTYPE_FILES = {
    "Isolation": "isolation", "PRBallHandler": "prballhandler", "PRRollman": "prrollman",
    "Postup": "postup", "Spotup": "spotup", "Handoff": "handoff",
    "Cut": "cut", "OffScreen": "offscreen", "Transition": "transition",
}


def _schema_check_targets() -> list[tuple[str, set[str]]]:
    targets = [
        ("data/hustle_stats_2025_26.csv", HUSTLE_COLUMNS),
        ("data/hustle_stats_2024_25.csv", HUSTLE_COLUMNS),
        ("data/shot_defense_overall_2025_26.csv", SHOT_DEFENSE_COLUMNS["Overall"]),
        ("data/shot_defense_3pt_2025_26.csv", SHOT_DEFENSE_COLUMNS["3 Pointers"]),
        ("data/shot_defense_2pt_2025_26.csv", SHOT_DEFENSE_COLUMNS["2 Pointers"]),
        ("data/shot_defense_rim_2025_26.csv", SHOT_DEFENSE_COLUMNS["Less Than 6Ft"]),
        ("data/drives_2025_26.csv", DRIVES_COLUMNS),
        ("data/draft_class_2026.csv", DRAFT_CLASS_COLUMNS),
    ]
    for suffix in _PLAYTYPE_FILES.values():
        targets.append((f"data/playtype_defense_{suffix}_2025_26.csv", PLAYTYPE_COLUMNS))
        targets.append((f"data/playtype_offense_{suffix}_2025_26.csv", PLAYTYPE_COLUMNS))
    return targets


def validate_startup_schema() -> None:
    """Read every CSV this app depends on and confirm each has its required
    columns. Raises RuntimeError naming the exact file and exact missing
    column(s) on the first problem found, rather than a generic exception
    or a silent partial load. Cut/Transition playtype_defense files are
    near-empty by design (no player-level Synergy data for those categories
    on defense -- see README) and are skipped here for that reason, not
    because their schema doesn't matter."""
    for path, required in _schema_check_targets():
        if path in ("data/playtype_defense_cut_2025_26.csv", "data/playtype_defense_transition_2025_26.csv"):
            continue
        try:
            df = pd.read_csv(path)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Startup schema check failed: missing required data file {path!r}. "
                "Run ./refresh_data.sh to pull data, or check that CSVs were "
                "included in this deployment."
            ) from e

        missing = missing_columns(df.columns, required)
        if missing:
            raise RuntimeError(
                f"Startup schema check failed: {path!r} is missing required "
                f"column(s) {sorted(missing)}. This file exists but does not "
                f"match the schema this app depends on -- it may be stale, "
                f"corrupted, or pulled from a changed upstream endpoint. "
                f"Re-run ./refresh_data.sh to regenerate it."
            )


def _load_manifest() -> dict:
    try:
        with open(_MANIFEST_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_startup_schema()

    try:
        _state["df"] = pd.read_csv("data/hustle_stats_2025_26.csv")
        _state["prior_df"] = pd.read_csv("data/hustle_stats_2024_25.csv")
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Missing required data file: {e.filename}. "
            "Run ./refresh_data.sh to pull data, or check that CSVs were "
            "included in this deployment."
        ) from e

    _state["manifest"] = _load_manifest()
    yield
    _state.clear()


def _data_as_of() -> str | None:
    """Pulled from data_manifest.json (see refresh_data.sh) -- the most
    recent extraction timestamp across all recorded files, or None if no
    manifest exists (e.g. a fresh clone with committed CSVs but no
    refresh_data.sh run yet). None is surfaced as null in API responses
    rather than a guessed/fabricated date."""
    manifest = _state.get("manifest") or {}
    files = manifest.get("files", {})
    timestamps = [f["extracted_at"] for f in files.values() if "extracted_at" in f]
    return max(timestamps) if timestamps else None


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


class AuditInfo(BaseModel):
    intent: Optional[str] = None
    parameters: dict = {}
    qualifying_pool_size: Optional[int] = None
    routing_method: Optional[str] = None
    data_as_of: Optional[str] = None
    matched_text: Optional[str] = None
    matched_pattern: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    response_type: str   # "answer" | "out_of_scope" | "needs_clarification" | "error"
    answer: str
    function_matched: Optional[str]
    method: Optional[str]
    table: Optional[list[dict]]
    data_as_of: Optional[str] = None
    audit: Optional[AuditInfo] = None


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
    audit: list[Optional[AuditInfo]] = []


class ReportResponse(BaseModel):
    player_name: str
    season: str
    sections: list[ReportSection]
    data_as_of: Optional[str] = None


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
    audit: list[Optional[AuditInfo]] = []


class CompareResponse(BaseModel):
    player_a: str
    player_b: str
    season: str
    sections: list[CompareSection]
    data_as_of: Optional[str] = None


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


def _with_data_as_of(audit: Optional[dict], data_as_of: Optional[str]) -> Optional[dict]:
    """Stamp data_as_of onto an audit dict built by query_router/report --
    those modules have no knowledge of data_manifest.json (that's api.py's
    concern alone, same as the existing top-level data_as_of field), so the
    freshness timestamp is merged in here rather than threaded through
    every compute/router function signature."""
    if audit is None:
        return None
    return {**audit, "data_as_of": data_as_of}


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
    data_as_of = _data_as_of()

    return QueryResponse(
        question=result["question"],
        response_type=_response_type(result),
        answer=result["answer"],
        function_matched=result.get("function_matched"),
        method=result.get("method"),
        table=table,
        data_as_of=data_as_of,
        audit=_with_data_as_of(result.get("audit"), data_as_of),
    )


@app.post("/report", response_model=ReportResponse)
def report(req: ReportRequest):
    data = generate_scouting_report_data(req.player_name, req.season)
    data_as_of = _data_as_of()
    for section in data["sections"]:
        section["audit"] = [_with_data_as_of(a, data_as_of) for a in section.get("audit", [])]
    return ReportResponse(**data, data_as_of=data_as_of)


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    data = compare_players_data(req.player_a, req.player_b, req.season)
    data_as_of = _data_as_of()
    for section in data["sections"]:
        section["audit"] = [_with_data_as_of(a, data_as_of) for a in section.get("audit", [])]
    return CompareResponse(**data, data_as_of=data_as_of)


# ── serve frontend ────────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory="chat", html=True), name="frontend")
