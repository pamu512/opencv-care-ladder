"""POST /telegram/webhook + family_paged_inflight fixture (stub mode)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from care_ladder.api.app import create_app
from care_ladder.audit.store import AuditStore


def test_webhook_unconfigured_is_honest_no_retry_storm():
    with TestClient(create_app(store=AuditStore())) as client:
        r = client.post("/telegram/webhook", json={"message": {"text": "1"}})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert body["reason"] == "telegram_not_configured"


def test_family_paged_inflight_leaves_thread_open():
    with TestClient(create_app(store=AuditStore())) as client:
        r = client.post("/demo/run", json={"fixture": "family_paged_inflight"})
        assert r.status_code == 200, r.text
        iid = r.json()["incident_id"]
        opened = client.get(f"/incidents/{iid}").json()
        assert opened["status"] == "open"
        tools = [e["tool"] for e in opened["events"]]
        assert "bot" in tools
        states = [e["detail"].get("state") for e in opened["events"] if e["tool"] == "bot"]
        assert "family_paged" in states
        assert "calling_1" not in states
        assert all(e.get("at") for e in opened["events"] if e["tool"] == "bot")


def test_webhook_callback_acks_family_paged_and_prevents_dial():
    store = AuditStore()
    with TestClient(create_app(store=store)) as client:
        r = client.post("/demo/run", json={"fixture": "family_paged_inflight"})
        assert r.status_code == 200, r.text
        iid = r.json()["incident_id"]
        w = client.post(
            "/telegram/webhook",
            json={"callback_query": {"data": "ack:1", "message": {"chat": {"id": 1}}}},
        )
        assert w.status_code == 200
        body = w.json()
        assert body["ok"] is True
        assert body.get("applied") is True
        assert body.get("incident_id") == iid

        closed = None
        for _ in range(80):
            closed = client.get(f"/incidents/{iid}").json()
            if closed["status"] != "open":
                break
            time.sleep(0.02)
        assert closed is not None
        assert closed["status"] == "resolved"
        reasons = [e["detail"].get("reason") for e in closed["events"] if e["tool"] == "resolve"]
        assert "family_ack" in reasons or "caregiver_ack" in reasons
        assert "dial_contact" not in [e["tool"] for e in closed["events"]]
