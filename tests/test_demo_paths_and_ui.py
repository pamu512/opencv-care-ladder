"""Spec §10 Path A: verbal OK resolves without any dial (judge demo path)."""

from fastapi.testclient import TestClient

from care_ladder.api.app import create_app
from care_ladder.audit.store import AuditStore


def _run(client: TestClient, fixture: str) -> dict:
    resp = client.post("/demo/run", json={"fixture": fixture})
    assert resp.status_code == 200, resp.text
    inc_id = resp.json()["incident_id"]
    detail = client.get(f"/incidents/{inc_id}")
    assert detail.status_code == 200, detail.text
    return detail.json()


def test_no_movement_ok_resolves_without_dial():
    app = create_app(store=AuditStore())
    with TestClient(app) as client:
        inc = _run(client, "no_movement_ok")
    tools = [e["tool"] for e in inc["events"]]
    assert "speaker_prompt" in tools
    assert inc["status"] == "resolved"
    assert "dial_contact" not in tools
    assert "emergency" not in tools
    # reply captured in audit trace
    speaker = next(e for e in inc["events"] if e["tool"] == "speaker_prompt")
    assert speaker["detail"]["reply_kind"] == "ok"


def test_unknown_fixture_rejected_400():
    app = create_app(store=AuditStore())
    with TestClient(app) as client:
        resp = client.post("/demo/run", json={"fixture": "nonsense"})
    assert resp.status_code == 400


def test_ui_served_with_caregiver_console():
    app = create_app(store=AuditStore())
    with TestClient(app) as client:
        resp = client.get("/ui/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Caregiver" in resp.text
    assert "Path A" in resp.text
    assert "Demo scenarios" in resp.text
    assert "No reading · lens covered" in resp.text
    assert "path_b_inflight" in resp.text
    assert "opencv_occlusion" in resp.text
    assert "response_intent" in resp.text
    assert "clear_ok" in resp.text
    assert "Alexa+" not in resp.text
    assert "Ambient Hearth" not in resp.text


def test_ui_setup_archive_and_family_chat_mirror():
    html = TestClient(create_app(store=AuditStore())).get("/ui/").text
    assert "Setup + archive" in html
    assert "Family chat (mirror)" in html
    assert "Acknowledge" in html
    assert 'class="demo-strip" open' not in html
    assert 'class="demo-strip"' in html
    assert "/family/runtime" in html
    assert "Alexa+" not in html


def test_family_runtime_is_honest_stub_without_token():
    with TestClient(create_app(store=AuditStore())) as client:
        r = client.get("/family/runtime")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source"] in {"stub", "demo_fixture"}
        assert body["state"] in {
            "idle",
            "speaker_window",
            "family_paged",
            "pressure",
            "calling_1",
            "calling_2",
            "closed",
        }
        assert "ok" not in body or body.get("configured") is False


def test_family_runtime_mirrors_family_paged_fixture():
    with TestClient(create_app(store=AuditStore())) as client:
        run = client.post("/demo/run", json={"fixture": "family_paged_inflight"})
        assert run.status_code == 200, run.text
        iid = run.json()["incident_id"]
        body = client.get("/family/runtime").json()
        assert body["source"] in {"stub", "demo_fixture"}
        assert body["state"] == "family_paged"
        assert body["incident_id"] == iid
        assert body.get("inform_text")
        assert "did not answer" in body["inform_text"]

def test_ui_served_with_learning_badge():
    from fastapi.testclient import TestClient
    from care_ladder.api.app import create_app
    from care_ladder.audit.store import AuditStore

    with TestClient(create_app(store=AuditStore())) as client:
        r = client.get('/ui/')
        assert r.status_code == 200
        html = r.text
        assert 'learning-badge' in html
        assert 'Learning schedule' in html and 'Schedule settled' in html and 'Learning frozen' in html
        assert '/learning/demo-home-1' in html
