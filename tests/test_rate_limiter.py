"""Unit and API tests for SaaS Rate Limiting and Monetization."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.rate_limiter import rate_limiter

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_limiter():
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def test_free_tier_rate_limit_enforcement():
    session_headers = {"X-Session-ID": "test_user_free_123", "X-Subscription-Tier": "free"}

    # 1-5 requests should succeed
    for i in range(5):
        res = client.post("/api/query", json={"query": f"question {i}"}, headers=session_headers)
        assert res.status_code == 200

    # 6th request must trigger HTTP 429 Too Many Requests
    res_blocked = client.post("/api/query", json={"query": "question 6"}, headers=session_headers)
    assert res_blocked.status_code == 429
    assert res_blocked.headers.get("X-RateLimit-Remaining") == "0"
    assert "upgrade" in res_blocked.json()["detail"]["message"].lower()


def test_pro_tier_unlimited_access():
    pro_headers = {"X-Session-ID": "test_user_pro_999", "X-Subscription-Tier": "pro"}

    # Run 8 requests consecutively (exceeding free limit of 5)
    for i in range(8):
        res = client.post("/api/query", json={"query": f"pro query {i}"}, headers=pro_headers)
        assert res.status_code == 200


def test_usage_telemetry_endpoint():
    headers = {"X-Session-ID": "telemetry_session_42", "X-Subscription-Tier": "free"}

    # Query 2 times
    client.post("/api/query", json={"query": "first query"}, headers=headers)
    client.post("/api/query", json={"query": "second query"}, headers=headers)

    res = client.get("/api/usage", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["tier"] == "free"
    assert data["queries_used"] == 2
    assert data["queries_remaining"] == 3
    assert data["is_rate_limited"] is False
