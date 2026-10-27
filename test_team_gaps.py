"""Tests for team capability space and roster gaps analysis."""
from team_gaps import (
    resolve_team,
    get_or_build_team_capability_space,
    get_team_gaps_data,
    format_team_gaps_answer,
    TEAM_NAMES,
)
from fastapi.testclient import TestClient
from api import app


def test_resolve_team():
    assert resolve_team("Warriors") == "GSW"
    assert resolve_team("GSW") == "GSW"
    assert resolve_team("Golden State") == "GSW"
    assert resolve_team("Bulls") == "CHI"
    assert resolve_team("Thunder") == "OKC"
    assert resolve_team("Lakers") == "LAL"
    assert resolve_team("Boston") == "BOS"
    assert resolve_team("InvalidTeamXYZ") is None


def test_capability_space_rankings():
    space = get_or_build_team_capability_space()
    assert "scores_df" in space
    assert "ranks_df" in space
    ranks = space["ranks_df"]

    # All 30 teams must be present and ranked 1 through 30
    assert len(ranks) == 30
    for sub in ["rim_protection", "perimeter_defense", "primary_shot_creation", "playmaking", "movement_scoring", "pace_transition"]:
        rank_vals = sorted(ranks[sub].tolist())
        assert rank_vals == list(range(1, 31)), f"Ranks for {sub} must be a permutation of 1..30"


def test_get_team_gaps_warriors():
    data = get_team_gaps_data("Warriors")
    assert data["team_abbreviation"] == "GSW"
    assert data["team_name"] == "Golden State Warriors"
    assert data["qualified_players_count"] >= 8
    assert len(data["capabilities"]) == 6
    assert len(data["identified_gaps"]) == 2

    # Check that answer formats cleanly
    answer = format_team_gaps_answer(data)
    assert "Golden State Warriors (GSW)" in answer
    assert "CAPABILITY RANKINGS" in answer
    assert "TOP IDENTIFIED ROSTER GAPS" in answer


def test_api_team_gaps_endpoint():
    with TestClient(app) as client:
        # Valid team
        res = client.post("/team-gaps", json={"team": "Thunder"})
        assert res.status_code == 200
        payload = res.json()
        assert payload["team_abbreviation"] == "OKC"
        assert payload["rotation_size"] >= 8
        assert len(payload["capabilities"]) == 6
        assert len(payload["top_gaps"]) == 2

        # Invalid team
        bad_res = client.post("/team-gaps", json={"team": "NonExistentTeam"})
        assert bad_res.status_code == 404


def test_api_query_team_gaps_routing():
    with TestClient(app) as client:
        res = client.post("/query", json={"question": "What are the biggest roster gaps for the Warriors?"})
        assert res.status_code == 200
        payload = res.json()
        assert payload["function_matched"] == "team_gaps"
        assert payload["team_gaps_data"] is not None
        assert payload["team_gaps_data"]["team_abbreviation"] == "GSW"


if __name__ == "__main__":
    print("Running test_resolve_team...")
    test_resolve_team()
    print("Running test_capability_space_rankings...")
    test_capability_space_rankings()
    print("Running test_get_team_gaps_warriors...")
    test_get_team_gaps_warriors()
    print("Running test_api_team_gaps_endpoint...")
    test_api_team_gaps_endpoint()
    print("Running test_api_query_team_gaps_routing...")
    test_api_query_team_gaps_routing()
    print("All team gaps tests passed successfully!")
