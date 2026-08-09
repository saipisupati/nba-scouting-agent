"""
Lightweight smoke test for api.py: confirms the FastAPI app actually starts
(runs the real lifespan startup -- loads the CSVs, not just a bare import)
and responds on /health.

This exists because a plain `import api` only catches import-time errors
(bad imports, syntax errors) -- it would NOT catch a startup failure like a
missing data CSV, since that only happens inside the lifespan context
manager, which only runs when the app actually starts. Using TestClient as
a context manager triggers real startup/shutdown, same as this project's
own earlier Docker-container verification.
"""

from fastapi.testclient import TestClient

from api import app

with TestClient(app) as client:
    resp = client.get("/health")
    assert resp.status_code == 200, f"/health returned {resp.status_code}: {resp.text}"

    data = resp.json()
    assert data["status"] == "ok", f"unexpected /health payload: {data}"
    assert data["current_season_rows"] > 0, "current-season data failed to load"
    assert data["prior_season_rows"] > 0, "prior-season data failed to load"

    print(f"Smoke test passed: /health returned {data}")
