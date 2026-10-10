from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from searchpilot.config import clear_settings_cache


def test_live_stays_open_and_other_routes_require_token(
    monkeypatch: pytest.MonkeyPatch, make_client
) -> None:
    monkeypatch.setenv("SEARCHPILOT_API_TOKEN", "secret-token")
    clear_settings_cache()
    try:
        client = make_client()
        assert client.get("/health/live").status_code == 200
        denied = client.get("/health/ready")
        assert denied.status_code == 401
        assert denied.json()["error"]["code"] == "UNAUTHORIZED"
        assert "secret-token" not in denied.text
        allowed = client.get("/health/ready", headers={"Authorization": "Bearer secret-token"})
        assert allowed.status_code != 401
    finally:
        clear_settings_cache()


def test_feedback_rejects_a_different_user(
    monkeypatch: pytest.MonkeyPatch, make_client, fakes
) -> None:
    monkeypatch.setenv("SEARCHPILOT_API_TOKEN", "secret-token")
    clear_settings_cache()
    try:
        client: TestClient = make_client(feedback=fakes.FakeFeedbackStore(), items=None)
        headers = {"Authorization": "Bearer secret-token", "X-Actor": "U1"}
        mismatch = client.post(
            "/feedback", headers=headers, json=fakes.feedback_payload(user_id="U2")
        )
        assert mismatch.status_code == 403
        assert mismatch.json()["error"]["code"] == "FORBIDDEN"
        matched = client.post(
            "/feedback", headers=headers, json=fakes.feedback_payload(user_id="U1")
        )
        assert matched.status_code == 201
    finally:
        clear_settings_cache()
