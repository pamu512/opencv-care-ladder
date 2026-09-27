"""Galuxium hooks: same RoutineProfile API, tenant subject keys, JSON until Postgres."""

from __future__ import annotations

from fastapi.testclient import TestClient

from care_ladder.api.app import create_app
from care_ladder.audit.store import AuditStore
from care_ladder.learning.profile import subject_key
from care_ladder.learning.store import RoutineProfileStore


def test_subject_key_household_or_tenant():
    assert subject_key(household_id="demo-home-1") == "demo-home-1"
    assert subject_key(tenant_id="fac-1") == "fac-1"
    assert subject_key(tenant_id="fac-1", monitored_id="room-12") == "fac-1:room-12"


def test_learning_api_scopes_by_tenant_id(tmp_path):
    app = create_app(store=AuditStore(), profile_store=RoutineProfileStore(tmp_path))
    with TestClient(app) as client:
        home = client.get("/learning").json()
        assert home["subject_key"] == "amazon-demo-1"
        assert home["frozen"] is False

        frozen = client.post("/learning/freeze", params={"tenant_id": "fac-1"}).json()
        assert frozen["subject_key"] == "fac-1"
        assert frozen["frozen"] is True
        assert frozen["badge"] == "Learning frozen"

        # household profile is a different key
        assert client.get("/learning").json()["frozen"] is False
        tenant = client.get("/learning", params={"tenant_id": "fac-1"}).json()
        assert tenant["frozen"] is True
        assert tenant["subject_key"] == "fac-1"


def test_tenant_plus_monitored_is_own_profile(tmp_path):
    app = create_app(store=AuditStore(), profile_store=RoutineProfileStore(tmp_path))
    with TestClient(app) as client:
        client.post(
            "/learning/mark-settled",
            params={"tenant_id": "fac-1", "monitored_id": "room-12"},
        )
        room = client.get(
            "/learning", params={"tenant_id": "fac-1", "monitored_id": "room-12"}
        ).json()
        facility = client.get("/learning", params={"tenant_id": "fac-1"}).json()
        assert room["learning_phase"] == "settled"
        assert facility["learning_phase"] == "rapid"
