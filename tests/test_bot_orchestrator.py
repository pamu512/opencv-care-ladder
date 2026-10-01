"""notify_caretaker drives BotThread + FakeTelegram (Phase 2)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from care_ladder.audit.store import AuditStore
from care_ladder.channels.bot import BotRegistry, BotState
from care_ladder.channels.dial import StubDialer
from care_ladder.channels.speaker import SpeakerSimulator
from care_ladder.channels.telegram_adapter import FakeTelegram
from care_ladder.ladder.orchestrator import run_incident
from care_ladder.models import AuditEvent, CueEvent
from care_ladder.plan_loader import load_care_plan

DAYTIME = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)


def _plan(*, countdown: float = 30, pressure_remaining: float = 15):
    plan = load_care_plan(Path("configs/demo_home.yaml"))
    notify = next(r for r in plan.rungs if r.tool == "notify_caretaker")
    notify.params["channels"] = ["telegram", "console"]
    notify.params["family_countdown_sec"] = countdown
    notify.params["pressure_at_remaining_sec"] = pressure_remaining
    notify.params["await_sec"] = countdown
    return plan


def _bot_states(incident) -> list[str]:
    return [e.detail.get("state") for e in incident.events if e.tool == "bot"]


def test_family_ack_prevents_dial():
    store = AuditStore()
    telegram = FakeTelegram()
    cue = CueEvent(kind="no_movement", confidence=0.9, detail={"fixture": "family_ack"})

    async def ack_when_paged() -> None:
        for _ in range(80):
            found = [i for i in store.list_incidents() if i.status == "open"]
            if found and telegram.sent:
                telegram.inject_update({"callback_query": {"data": "ack:1"}})
                return
            await asyncio.sleep(0.02)
        raise AssertionError("family page never sent")

    async def both():
        return await asyncio.gather(
            run_incident(
                cue=cue,
                plan=_plan(),
                speaker=SpeakerSimulator(scripted=[""]),
                dialer=StubDialer(behavior={"caregiver": "answered"}),
                pre_event_frames=[],
                store=store,
                now=DAYTIME,
                max_wait_sec=1.5,
                telegram=telegram,
            ),
            ack_when_paged(),
        )

    incident, _ = asyncio.run(both())
    assert incident.status == "resolved"
    reasons = [e.detail.get("reason") for e in incident.events if e.tool == "resolve"]
    assert "family_ack" in reasons or "caregiver_ack" in reasons
    assert "dial_contact" not in [e.tool for e in incident.events]
    assert "family_paged" in _bot_states(incident)
    assert "calling_1" not in _bot_states(incident)
    assert telegram.sent
    text = telegram.sent[0]["text"]
    assert "did not answer the spoken check-in" in text
    assert telegram.sent[0]["reply_markup"]["inline_keyboard"]
    bot_events = [e for e in incident.events if e.tool == "bot"]
    assert bot_events
    assert all(e.at is not None for e in bot_events)


def test_pressure_then_dial_when_no_ack():
    telegram = FakeTelegram()
    incident = asyncio.run(
        run_incident(
            cue=CueEvent(kind="no_movement", confidence=0.9, detail={"fixture": "pressure"}),
            plan=_plan(countdown=0.2, pressure_remaining=0.1),
            speaker=SpeakerSimulator(scripted=[""]),
            dialer=StubDialer(behavior={"caregiver": "answered"}),
            pre_event_frames=[],
            now=DAYTIME,
            max_wait_sec=1.0,
            telegram=telegram,
        )
    )
    states = _bot_states(incident)
    assert states[:3] == ["family_paged", "pressure", "calling_1"]
    assert "dial_contact" in [e.tool for e in incident.events]
    assert incident.status == "resolved"
    assert any("pressure" in (p.get("text") or "").lower() or p.get("kind") == "pressure" for p in telegram.sent)
    bot_events = [e for e in incident.events if e.tool == "bot"]
    assert all(e.at is not None for e in bot_events)


def test_clear_ok_skips_bot_thread():
    telegram = FakeTelegram()
    incident = asyncio.run(
        run_incident(
            cue=CueEvent(kind="no_movement", confidence=0.9, detail={}),
            plan=_plan(),
            speaker=SpeakerSimulator(scripted=["don't worry"]),
            dialer=StubDialer(behavior={"caregiver": "answered"}),
            pre_event_frames=[],
            now=DAYTIME,
            telegram=telegram,
        )
    )
    assert incident.status == "resolved"
    reasons = [e.detail.get("reason") for e in incident.events if e.tool == "resolve"]
    assert "speaker_ok" in reasons
    assert "family_paged" not in _bot_states(incident)
    assert "notify_caretaker" not in [e.tool for e in incident.events]
    assert telegram.sent == []


def test_console_ack_still_stops_bot_wait():
    store = AuditStore()
    telegram = FakeTelegram()
    cue = CueEvent(kind="no_movement", confidence=0.9, detail={"fixture": "console_ack"})

    async def ack_when_open() -> None:
        for _ in range(80):
            found = [i for i in store.list_incidents() if i.status == "open"]
            if found and any(e.tool == "bot" for e in found[0].events):
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
        raise AssertionError("incident never reached family page")

    async def both():
        return await asyncio.gather(
            run_incident(
                cue=cue,
                plan=_plan(),
                speaker=SpeakerSimulator(scripted=[""]),
                dialer=StubDialer(behavior={"caregiver": "answered"}),
                pre_event_frames=[],
                store=store,
                now=DAYTIME,
                max_wait_sec=1.5,
                telegram=telegram,
            ),
            ack_when_open(),
        )

    incident, _ = asyncio.run(both())
    assert incident.status == "resolved"
    reasons = [e.detail.get("reason") for e in incident.events if e.tool == "resolve"]
    assert "caregiver_ack" in reasons
    assert "dial_contact" not in [e.tool for e in incident.events]


def test_occlusion_pages_health_but_never_dials():
    telegram = FakeTelegram()
    incident = asyncio.run(
        run_incident(
            cue=CueEvent(
                kind="camera_occlusion",
                confidence=0.9,
                detail={"reason": "lens_covered", "distress_claimed": False},
            ),
            plan=_plan(countdown=0.1, pressure_remaining=0.05),
            speaker=SpeakerSimulator(scripted=[]),
            dialer=StubDialer(behavior={"caregiver": "answered"}),
            pre_event_frames=[],
            now=DAYTIME,
            max_wait_sec=0.2,
            telegram=telegram,
        )
    )
    tools = [e.tool for e in incident.events]
    assert "notify_caretaker" in tools
    assert "dial_contact" not in tools
    assert "calling_1" not in _bot_states(incident)
    resolve = next(e for e in incident.events if e.tool == "resolve")
    assert resolve.detail.get("reason") == "camera_health_informed"
    assert resolve.detail.get("distress_claimed") is False


def test_dial_answered_and_exhausted_close_calling_thread():
    """/family/runtime must not stick on calling_1 after dial resolve or exhaust."""

    def _run(behavior: dict[str, str]):
        store = AuditStore()
        registry = BotRegistry()
        incident = asyncio.run(
            run_incident(
                cue=CueEvent(kind="no_movement", confidence=0.9, detail={"fixture": "dial_close"}),
                plan=_plan(countdown=0.2, pressure_remaining=0.1),
                speaker=SpeakerSimulator(scripted=[""]),
                dialer=StubDialer(behavior=behavior),
                pre_event_frames=[],
                store=store,
                now=DAYTIME,
                max_wait_sec=1.0,
                telegram=FakeTelegram(),
                bot_registry=registry,
            )
        )
        thread = registry.get(incident.id)
        assert thread is not None
        assert thread.state is BotState.closed
        assert _bot_states(incident)[-1] == "closed"
        return incident, thread

    answered, answered_thread = _run({"caregiver": "answered"})
    assert answered.status == "resolved"
    assert answered_thread.close_reason == "dial_answered"
    reasons = [e.detail.get("reason") for e in answered.events if e.tool == "resolve"]
    assert "dial_answered" in reasons

    exhausted, exhausted_thread = _run(
        {"caregiver": "no_answer", "secondary": "no_answer"}
    )
    assert exhausted.status == "exhausted"
    assert exhausted_thread.close_reason == "dial_exhausted"
