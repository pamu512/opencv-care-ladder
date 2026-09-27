"""Task 2: care-plan learning block + orchestrator timeout / audit wiring."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from care_ladder.channels.dial import StubDialer
from care_ladder.channels.speaker import SpeakerSimulator
from care_ladder.ladder.orchestrator import run_incident
from care_ladder.learning.profile import LearningConfig, new_profile
from care_ladder.learning.store import RoutineProfileStore
from care_ladder.models import CueEvent
from care_ladder.plan_loader import load_care_plan

DAYTIME = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)


def _run(plan, *, speaker, store, now=DAYTIME, **kwargs):
    cue = CueEvent(kind="no_movement", confidence=0.9, detail={})
    return asyncio.run(
        run_incident(
            cue=cue,
            plan=plan,
            speaker=speaker,
            dialer=StubDialer(behavior={"caregiver": "answered"}),
            pre_event_frames=[],
            now=now,
            profile_store=store,
            **kwargs,
        )
    )


def test_demo_plan_enables_learning_with_defaults():
    plan = load_care_plan(Path("configs/demo_home.yaml"))
    assert plan.learning is not None
    assert plan.learning.enabled is True
    assert plan.learning.settled_after_days == 10
    assert plan.learning.min_no_movement_timeout_sec == 120
    assert plan.learning.rapid_timeout_factor == 0.4


def test_missing_learning_block_defaults_enabled(tmp_path: Path):
    yaml_text = """
household_id: t
caregiver:
  display_name: Alex
  phone_e164: "+12125550101"
monitored:
  display_name: Pat
triggers:
  no_movement:
    enabled: true
    timeout_sec: 900
rungs:
  - id: ask
    tool: speaker_prompt
    params: { text: "ok?" }
"""
    p = tmp_path / "no-learning.yaml"
    p.write_text(yaml_text)
    plan = load_care_plan(p)
    assert plan.learning.enabled is True
    assert plan.learning.rapid_timeout_factor == 0.4


def test_learning_block_can_disable(tmp_path: Path):
    yaml_text = """
household_id: t
caregiver:
  display_name: Alex
  phone_e164: "+12125550101"
monitored:
  display_name: Pat
triggers:
  no_movement:
    enabled: true
    timeout_sec: 900
learning:
  enabled: false
rungs:
  - id: ask
    tool: speaker_prompt
    params: { text: "ok?" }
"""
    p = tmp_path / "off.yaml"
    p.write_text(yaml_text)
    plan = load_care_plan(p)
    assert plan.learning.enabled is False


def test_incident_stashes_effective_timeout_and_audits_update(tmp_path: Path):
    plan = load_care_plan(Path("configs/demo_home.yaml"))
    store = RoutineProfileStore(tmp_path)
    incident = _run(plan, speaker=SpeakerSimulator(scripted=["I'm fine"]), store=store)
    cue_ev = incident.events[0]
    assert cue_ev.tool == "cue"
    assert cue_ev.detail["learning_phase"] == "rapid"
    assert cue_ev.detail["effective_timeout_sec"] == 360
    assert "explain" in cue_ev.detail
    assert "risk" not in cue_ev.detail["explain"].lower()

    updates = [e for e in incident.events if e.tool == "routine_profile_update"]
    assert len(updates) == 1
    detail = updates[0].detail
    assert detail["learning_phase"] == "rapid"
    assert detail["suggested_timeout_sec"] == 360
    assert "explain" in detail
    assert incident.status == "resolved"

    saved = store.load(plan.household_id)
    assert saved.still_hour_hist[12] == 1
    assert saved.confirmed_ok_days == 1


def test_path_a_still_resolves_without_dial_when_learning_on(tmp_path: Path):
    plan = load_care_plan(Path("configs/demo_home.yaml"))
    store = RoutineProfileStore(tmp_path)
    incident = _run(plan, speaker=SpeakerSimulator(scripted=["I'm fine"]), store=store)
    tools = [e.tool for e in incident.events]
    assert incident.status == "resolved"
    assert "dial_contact" not in tools
    assert "emergency" not in tools
    assert "routine_profile_update" in tools


def test_disabled_learning_uses_plan_timeout(tmp_path: Path):
    plan = load_care_plan(Path("configs/demo_home.yaml"))
    plan.learning = LearningConfig(enabled=False)
    store = RoutineProfileStore(tmp_path)
    incident = _run(plan, speaker=SpeakerSimulator(scripted=["I'm fine"]), store=store)
    assert incident.events[0].detail["effective_timeout_sec"] == 900
    assert incident.events[0].detail["learning_phase"] == "off"
    assert not any(e.tool == "routine_profile_update" for e in incident.events)
    assert store.load(plan.household_id).confirmed_ok_days == 0


def test_learning_never_enables_emergency(tmp_path: Path):
    plan = load_care_plan(Path("configs/demo_home.yaml"))
    store = RoutineProfileStore(tmp_path)
    incident = _run(plan, speaker=SpeakerSimulator(scripted=[]), store=store)
    emergency = next(r for r in plan.rungs if r.tool == "emergency")
    assert emergency.params.get("enabled") is False
    assert "emergency" not in [e.tool for e in incident.events]
    assert all("911" not in str(e.detail) for e in incident.events)


def test_seeded_settled_profile_uses_plan_timeout(tmp_path: Path):
    plan = load_care_plan(Path("configs/demo_home.yaml"))
    store = RoutineProfileStore(tmp_path)
    store.save(new_profile(plan.household_id).model_copy(update={"learning_phase": "settled"}))
    incident = _run(plan, speaker=SpeakerSimulator(scripted=["I'm fine"]), store=store)
    assert incident.events[0].detail["learning_phase"] == "settled"
    assert incident.events[0].detail["effective_timeout_sec"] == 900
