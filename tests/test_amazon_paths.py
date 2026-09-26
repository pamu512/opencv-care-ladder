"""Amazon demo fixtures: Path A (stillness) and Path B (occlusion, never distress)."""

import json

from fastapi.testclient import TestClient

from care_ladder.api.app import create_app
from care_ladder.audit.store import AuditStore


def test_alexa_path_a_full_ladder():
    with TestClient(create_app(store=AuditStore())) as client:
        r = client.post("/demo/run", json={"fixture": "alexa_path_a"})
        assert r.status_code == 200
        inc = client.get(f"/incidents/{r.json()['incident_id']}").json()
        tools = [e["tool"] for e in inc["events"]]
        assert "alexa_checkin" in tools
        assert "wait_window" in tools
        assert "notify_caretaker" in tools
        assert "request_call" in tools
        assert "dial_contact" not in tools  # Amazon path never dials for real
        assert inc["household_id"] == "amazon-demo-1"


def test_alexa_path_b_never_claims_distress():
    with TestClient(create_app(store=AuditStore())) as client:
        r = client.post("/demo/run", json={"fixture": "alexa_path_b"})
        assert r.status_code == 200
        inc = client.get(f"/incidents/{r.json()['incident_id']}").json()
        assert inc["cue"]["kind"] == "no_visibility"
        blob = json.dumps(inc)
        assert "distress_heuristic" not in blob
        notify = next(e for e in inc["events"] if e["tool"] == "notify_caretaker")
        assert notify["detail"]["basis"] == "camera_health_inform"
        # occlusion prompt was the blanket ask, not a distress prompt
        checkins = [e for e in inc["events"] if e["tool"] == "alexa_checkin"]
        assert any("blanket" in e["detail"]["prompt"] for e in checkins)


def test_amazon_plan_not_the_opencv_default():
    # the OpenCV fixtures still use demo_home.yaml (Alex/Sam household)
    with TestClient(create_app(store=AuditStore())) as client:
        r = client.post("/demo/run", json={"fixture": "no_movement_silence"})
        inc = client.get(f"/incidents/{r.json()['incident_id']}").json()
        assert inc["household_id"] == "demo-home-1"
        tools = [e["tool"] for e in inc["events"]]
        assert "speaker_prompt" in tools and "alexa_checkin" not in tools
