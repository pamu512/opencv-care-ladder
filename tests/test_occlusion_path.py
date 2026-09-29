"""Occlusion Path B: covered camera is camera-health, never distress."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from care_ladder.api.app import create_app
from care_ladder.audit.store import AuditStore
from care_ladder.channels.dial import StubDialer
from care_ladder.channels.speaker import SpeakerSimulator
from care_ladder.ladder.orchestrator import run_incident
from care_ladder.models import CueEvent
from care_ladder.plan_loader import load_care_plan
from care_ladder.vision.cues import CueDetector

DAYTIME = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)


def _covered(w=160, h=120, value=4):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_detector_emits_camera_occlusion_on_covered_lens():
    det = CueDetector(
        no_movement_timeout_sec=99.0,
        zone=((0, 0), (160, 0), (160, 120), (0, 120)),
    )
    frame = _covered()
    assert det.observe(frame, t=0.0) is None  # sustain gate
    cue = det.observe(frame, t=0.4)
    assert cue is not None
    assert cue.kind == "camera_occlusion"
    assert cue.detail.get("reason") == "lens_covered"
    assert cue.detail.get("distress_claimed") is False


def test_detector_stillness_blob_is_not_occlusion():
    det = CueDetector(
        no_movement_timeout_sec=3.0,
        zone=((0, 0), (160, 0), (160, 120), (0, 120)),
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[40:80, 60:100] = 200
    assert det.observe(frame, t=0.0) is None
    assert det.observe(frame, t=0.5) is None
    cue = det.observe(frame, t=3.5)
    assert cue is not None
    assert cue.kind == "no_movement"


def test_orchestrator_occlusion_silence_informs_without_distress_or_dial():
    plan = load_care_plan(Path("configs/demo_home.yaml"))
    cue = CueEvent(
        kind="camera_occlusion",
        confidence=0.9,
        detail={"reason": "lens_covered", "distress_claimed": False},
    )
    incident = asyncio.run(
        run_incident(
            cue=cue,
            plan=plan,
            speaker=SpeakerSimulator(scripted=[]),
            dialer=StubDialer(behavior={"caregiver": "answered"}),
            pre_event_frames=[],
            now=DAYTIME,
        )
    )
    tools = [e.tool for e in incident.events]
    assert "speaker_prompt" in tools
    assert "notify_caretaker" in tools
    assert "dial_contact" not in tools
    speaker = next(e for e in incident.events if e.tool == "speaker_prompt")
    assert "clear" in speaker.detail["text"].lower() or "lens" in speaker.detail["text"].lower()
    assert speaker.detail.get("response_intent") == "unclear"
    notify = next(e for e in incident.events if e.tool == "notify_caretaker")
    assert notify.detail.get("basis") == "camera_health_inform"
    assert notify.detail.get("distress_claimed") is False
    assert incident.status == "resolved"
    resolve = next(e for e in incident.events if e.tool == "resolve")
    assert resolve.detail.get("reason") == "camera_health_informed"
    blob = json.dumps(incident.model_dump(), default=str)
    assert "distress_heuristic" not in blob
    assert "fall signature" not in blob.lower()


def test_soft_ok_intent_resolves_without_dial():
    plan = load_care_plan(Path("configs/demo_home.yaml"))
    cue = CueEvent(kind="no_movement", confidence=0.9, detail={})
    incident = asyncio.run(
        run_incident(
            cue=cue,
            plan=plan,
            speaker=SpeakerSimulator(scripted=["don't worry"]),
            dialer=StubDialer(behavior={"caregiver": "answered"}),
            pre_event_frames=[],
            now=DAYTIME,
        )
    )
    tools = [e.tool for e in incident.events]
    speaker = next(e for e in incident.events if e.tool == "speaker_prompt")
    assert speaker.detail.get("response_intent") == "clear_ok"
    assert incident.status == "resolved"
    assert "dial_contact" not in tools


def test_needs_human_intent_jumps_to_human_rung():
    plan = load_care_plan(Path("configs/demo_home.yaml"))
    cue = CueEvent(kind="no_movement", confidence=0.9, detail={})
    incident = asyncio.run(
        run_incident(
            cue=cue,
            plan=plan,
            speaker=SpeakerSimulator(scripted=["I'm ok but I hurt my hip"]),
            dialer=StubDialer(behavior={"caregiver": "answered"}),
            pre_event_frames=[],
            now=DAYTIME,
        )
    )
    speaker = next(e for e in incident.events if e.tool == "speaker_prompt")
    assert speaker.detail.get("response_intent") == "needs_human"
    tools = [e.tool for e in incident.events]
    assert "notify_caretaker" in tools or "dial_contact" in tools
    assert incident.status == "resolved"


def test_opencv_occlusion_fixture_distinct_tool_path():
    with TestClient(create_app(store=AuditStore())) as client:
        r = client.post("/demo/run", json={"fixture": "opencv_occlusion"})
        assert r.status_code == 200, r.text
        inc = client.get(f"/incidents/{r.json()['incident_id']}").json()
    assert inc["cue"]["kind"] == "camera_occlusion"
    assert inc["cue"]["detail"].get("source") == "opencv_cue_detector"
    tools = [e["tool"] for e in inc["events"]]
    assert "notify_caretaker" in tools
    assert "dial_contact" not in tools
    assert inc["status"] == "resolved"
    blob = json.dumps(inc)
    assert "distress_heuristic" not in blob
    speaker = next(e for e in inc["events"] if e["tool"] == "speaker_prompt")
    assert speaker["detail"].get("response_intent") in {"unclear", "clear_ok", "needs_human"}
    duration = (inc.get("explain") or {}).get("duration_sec")
    assert duration is not None and duration < 60, duration


def test_soft_ok_and_needs_human_fixtures():
    with TestClient(create_app(store=AuditStore())) as client:
        ok = client.post("/demo/run", json={"fixture": "speaker_soft_ok"})
        assert ok.status_code == 200, ok.text
        ok_inc = client.get(f"/incidents/{ok.json()['incident_id']}").json()
        human = client.post("/demo/run", json={"fixture": "speaker_needs_human"})
        assert human.status_code == 200, human.text
        human_inc = client.get(f"/incidents/{human.json()['incident_id']}").json()
    ok_sp = next(e for e in ok_inc["events"] if e["tool"] == "speaker_prompt")
    assert ok_sp["detail"]["response_intent"] == "clear_ok"
    assert ok_inc["status"] == "resolved"
    assert "dial_contact" not in [e["tool"] for e in ok_inc["events"]]
    human_sp = next(e for e in human_inc["events"] if e["tool"] == "speaker_prompt")
    assert human_sp["detail"]["response_intent"] == "needs_human"
    assert "notify_caretaker" in [e["tool"] for e in human_inc["events"]] or "dial_contact" in [
        e["tool"] for e in human_inc["events"]
    ]
