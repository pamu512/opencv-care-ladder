"""Task 3: schedule-learning badge, explain line, demo-safe controls."""

from __future__ import annotations

from fastapi.testclient import TestClient

from care_ladder.api.app import create_app
from care_ladder.audit.store import AuditStore
from care_ladder.learning.store import RoutineProfileStore


def _app(tmp_path):
    return create_app(store=AuditStore(), profile_store=RoutineProfileStore(tmp_path))


def test_ui_contains_badge_copy_and_no_clinical_claims():
    with TestClient(create_app(store=AuditStore())) as client:
        resp = client.get("/ui/")
    assert resp.status_code == 200
    text = resp.text
    assert "Learning schedule" in text
    assert "Schedule settled" in text
    assert "Learning frozen" in text
    assert "risk score" not in text.lower()
    assert "911" not in text


def test_learning_status_starts_rapid_and_controls_work(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        status = client.get("/learning")
        assert status.status_code == 200
        body = status.json()
        assert body["learning_phase"] == "rapid"
        assert body["frozen"] is False
        assert body["badge"] == "Learning schedule"
        assert "risk" not in body.get("explain", "").lower()

        run = client.post("/demo/run", json={"fixture": "no_movement_ok"})
        assert run.status_code == 200
        inc = client.get(f"/incidents/{run.json()['incident_id']}").json()
        cue = inc["events"][0]
        assert cue["detail"]["learning_phase"] == "rapid"
        assert "explain" in cue["detail"]
        assert any(e["tool"] == "routine_profile_update" for e in inc["events"])

        frozen = client.post("/learning/freeze").json()
        assert frozen["frozen"] is True
        assert client.get("/learning").json()["badge"] == "Learning frozen"

        reset = client.post("/learning/reset").json()
        assert reset["learning_phase"] == "rapid"
        assert reset["frozen"] is False
        assert reset["confirmed_ok_days"] == 0

        settled = client.post("/learning/mark-settled").json()
        assert settled["learning_phase"] == "settled"
        assert client.get("/learning").json()["badge"] == "Schedule settled"


def test_learning_controls_never_enable_emergency(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        for path in ("/learning/freeze", "/learning/reset", "/learning/mark-settled"):
            resp = client.post(path)
            assert resp.status_code == 200
            body = resp.json()
            blob = str(body).lower()
            assert "emergency" not in blob
            assert "911" not in blob
            assert "risk" not in blob
