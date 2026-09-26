"""Fire TV caregiver app (slice 4): served HTML anchors + full API-driven flow."""

from fastapi.testclient import TestClient

from care_ladder.api.app import create_app
from care_ladder.audit.store import AuditStore


def test_firetv_served_with_calm_care_tech_tokens():
    with TestClient(create_app(store=AuditStore())) as client:
        r = client.get("/firetv/")
        assert r.status_code == 200
        html = r.text
        # design-system lock anchors
        for token in ("#24221E", "#A8B5A0", "#D9A05B", "#D98873", "#9DB0C7",
                      "JetBrains Mono", "Instrument Sans"):
            assert token in html, token
        # d-pad focus machinery
        assert "spatialMove" in html and "ArrowLeft" in html
        # states + copy anchors (prototype strings kept verbatim per Q5)
        assert "Meera's home · Fire TV" in html
        assert "(555) 010-2276" in html
        assert "Wellness ladder — not a medical device" in html
        assert "Emergency dial off by default" in html
        # demo console drives the real API
        assert 'data-fixture="alexa_path_a"' in html
        assert 'data-fixture="alexa_path_b"' in html
        assert "/demo/run" in html and "/incidents" in html
        # emergency gate present + hard-locked copy
        assert "gateHold" in html and "Hard-locked in this build" in html
        # audit trail section
        assert "Audit trail" in html
        # privacy: silhouette only, no video element
        assert "<video" not in html


def test_firetv_flow_path_a_then_ack():
    with TestClient(create_app(store=AuditStore())) as client:
        # fire Path A through the same endpoint the TV console uses
        r = client.post("/demo/run", json={"fixture": "alexa_path_a"})
        assert r.status_code == 200
        iid = r.json()["incident_id"]

        # TV poll: household-filtered incident list finds it
        listing = client.get("/incidents").json()
        mine = [i for i in listing if i["household_id"] == "amazon-demo-1"]
        assert any(i["id"] == iid for i in mine)

        # full incident carries the trail the TV renders
        inc = client.get(f"/incidents/{iid}").json()
        tools = [e["tool"] for e in inc["events"]]
        assert "alexa_checkin" in tools and "notify_caretaker" in tools

        # ack from the TV closes the loop
        ack = client.post(f"/incidents/{iid}/ack",
                          json={"contact": "Anoop", "via": "fire_tv"})
        assert ack.status_code == 200
        inc2 = client.get(f"/incidents/{iid}").json()
        assert inc2["acked_by"] == "Anoop"
        assert any(e["tool"] == "notify" and e["detail"].get("action") == "caregiver_ack"
                   for e in inc2["events"])


def test_firetv_flow_path_b_occlusion_copy():
    with TestClient(create_app(store=AuditStore())) as client:
        r = client.post("/demo/run", json={"fixture": "alexa_path_b"})
        iid = r.json()["incident_id"]
        inc = client.get(f"/incidents/{iid}").json()
        assert inc["cue"]["kind"] == "no_visibility"
        # the TV renders "never claims distress" from this data
        notify = next(e for e in inc["events"] if e["tool"] == "notify_caretaker")
        assert notify["detail"]["basis"] == "camera_health_inform"
