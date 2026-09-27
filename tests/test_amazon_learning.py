"""Amazon track: RoutineProfile on Alexa+ fixtures + Fire TV short badges."""

from __future__ import annotations

from fastapi.testclient import TestClient

from care_ladder.api.app import create_app
from care_ladder.audit.store import AuditStore
from care_ladder.learning.profile import effective_no_movement_timeout_sec, new_profile
from care_ladder.learning.store import RoutineProfileStore
from care_ladder.plan_loader import load_care_plan
from pathlib import Path


def test_amazon_plan_rapid_timeout_is_visibly_shorter():
    plan = load_care_plan(Path("configs/amazon_demo_home.yaml"))
    assert plan.triggers.no_movement.timeout_sec == 240
    assert plan.learning.enabled is True
    # 240 * 0.4 = 96 → clamp to min 120, never above plan
    assert effective_no_movement_timeout_sec(plan, new_profile(plan.household_id)) == 120


def test_alexa_path_a_audits_learning_update(tmp_path):
    app = create_app(store=AuditStore(), profile_store=RoutineProfileStore(tmp_path))
    with TestClient(app) as client:
        r = client.post("/demo/run", json={"fixture": "alexa_path_a"})
        assert r.status_code == 200
        inc = client.get(f"/incidents/{r.json()['incident_id']}").json()
        tools = [e["tool"] for e in inc["events"]]
        assert "alexa_checkin" in tools
        assert "routine_profile_update" in tools
        cue = inc["events"][0]
        assert cue["detail"]["learning_phase"] == "rapid"
        assert cue["detail"]["effective_timeout_sec"] == 120
        assert "risk" not in cue["detail"]["explain"].lower()
        assert "911" not in str(inc).lower()
        assert inc["household_id"] == "amazon-demo-1"


def test_firetv_html_has_short_learning_labels():
    with TestClient(create_app(store=AuditStore())) as client:
        html = client.get("/firetv/").text
        assert "Learning" in html
        assert "Settled" in html
        assert "Frozen" in html
        assert "learnBadge" in html
        assert "/learning" in html
        assert "risk score" not in html.lower()


def test_learning_badge_short_for_amazon_household(tmp_path):
    app = create_app(store=AuditStore(), profile_store=RoutineProfileStore(tmp_path))
    with TestClient(app) as client:
        body = client.get("/learning").json()
        assert body["subject_key"] == "amazon-demo-1"
        assert body["badge"] == "Learning schedule"
        assert body["badge_short"] == "Learning"
        settled = client.post("/learning/mark-settled").json()
        assert settled["badge_short"] == "Settled"
        frozen = client.post("/learning/freeze").json()
        assert frozen["badge_short"] == "Frozen"
