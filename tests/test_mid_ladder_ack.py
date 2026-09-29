"""Mid-ladder Acknowledge stands the escalation down."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from care_ladder.api.app import create_app
from care_ladder.audit.store import AuditStore
from care_ladder.channels.dial import StubDialer
from care_ladder.channels.speaker import SpeakerSimulator
from care_ladder.ladder.orchestrator import run_incident
from care_ladder.models import AuditEvent, CueEvent
from care_ladder.plan_loader import load_care_plan

DAYTIME = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)


def test_ack_during_notify_wait_resolves_caregiver_ack():
    store = AuditStore()
    plan = load_care_plan(Path("configs/demo_home.yaml"))
    cue = CueEvent(kind="no_movement", confidence=0.9, detail={"fixture": "ack"})

    async def ack_when_open() -> None:
        for _ in range(80):
            found = [i for i in store.list_incidents() if i.status == "open"]
            if found:
                inc = found[0]
                inc.acked_by = "caregiver"
                inc.acked_at = datetime.now(timezone.utc)
                inc.events.append(
                    AuditEvent(
                        tool="notify",
                        cue_kind=inc.cue.kind,
                        detail={"action": "caregiver_ack", "contact": "caregiver"},
                    )
                )
                store.save(inc)
                return
            await asyncio.sleep(0.02)
        raise AssertionError("incident never appeared as open")

    async def both():
        return await asyncio.gather(
            run_incident(
                cue=cue,
                plan=plan,
                speaker=SpeakerSimulator(scripted=[""]),
                dialer=StubDialer(behavior={"caregiver": "answered"}),
                pre_event_frames=[],
                store=store,
                now=DAYTIME,
                max_wait_sec=1.5,
            ),
            ack_when_open(),
        )

    incident, _ = asyncio.run(both())
    assert incident.status == "resolved"
    reasons = [e.detail.get("reason") for e in incident.events if e.tool == "resolve"]
    assert "caregiver_ack" in reasons
    assert "dial_contact" not in [e.tool for e in incident.events]


def test_path_b_inflight_ack_via_api_stands_down():
    with TestClient(create_app(store=AuditStore())) as client:
        r = client.post("/demo/run", json={"fixture": "path_b_inflight"})
        assert r.status_code == 200, r.text
        iid = r.json()["incident_id"]
        opened = client.get(f"/incidents/{iid}").json()
        assert opened["status"] == "open"
        ack = client.post(f"/incidents/{iid}/ack", json={"contact": "caregiver", "note": "on my way"})
        assert ack.status_code == 200
        closed = client.get(f"/incidents/{iid}").json()
        assert closed["status"] == "resolved"
        assert closed["acked_by"] == "caregiver"
        reasons = [e["detail"].get("reason") for e in closed["events"] if e["tool"] == "resolve"]
        assert "caregiver_ack" in reasons
        assert "dial_contact" not in [e["tool"] for e in closed["events"]]
        explain = closed.get("explain") or {}
        assert explain.get("acked_by") == "caregiver"
        assert explain.get("resolve_reason") == "caregiver_ack"
