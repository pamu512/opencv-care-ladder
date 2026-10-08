"""Care ladder agentic loop: cue → rungs → audit trail."""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, time, timezone
from typing import Any

from care_ladder.audit.store import AuditStore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from care_ladder.learning.profile import RoutineProfile
    from care_ladder.vision.cues import CueDetector

from care_ladder.channels.bot import BotRegistry, BotThread, InformCard
from care_ladder.channels.dial import StubDialer, next_rung_after_no_answer
from care_ladder.channels.telegram_adapter import AckIntent, FakeTelegram, TelegramAdapter
from care_ladder.learning.profile import (
    effective_no_movement_timeout_sec,
)
from care_ladder.learning.profile import explain as _learning_explain
from care_ladder.channels.response_intent import classify_response_intent
from care_ladder.channels.speaker import SpeakerChannel
from care_ladder.models import AuditEvent, CarePlan, CueEvent, Incident, PrivacyMode, Rung
from care_ladder.privacy import blur_faces, to_silhouette

_OCCLUSION_PROMPT = "The camera looks covered. Could you clear the lens?"


def _contact_for_role(plan: CarePlan, role: str):
    if role == "caregiver":
        return plan.caregiver
    if role == "secondary":
        if plan.secondary is None:
            raise ValueError("plan has no secondary contact")
        return plan.secondary
    if role == "monitored":
        return plan.monitored
    raise ValueError(f"unknown contact role: {role!r}")


def _append(
    events: list[AuditEvent],
    *,
    tool: str,
    cue_kind: str | None = None,
    rung_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    events.append(
        AuditEvent(
            tool=tool,
            cue_kind=cue_kind,
            rung_id=rung_id,
            detail=detail or {},
            at=datetime.now(timezone.utc),
        )
    )


def _log_jump(
    events: list[AuditEvent],
    rung: Rung,
    *,
    reason: str,
    from_index: int,
    to_index: int | None,
    **extra: Any,
) -> None:
    detail: dict[str, Any] = {
        "reason": reason,
        "skipped_tool": rung.tool,
        "skipped_rung_id": rung.id,
        "from_index": from_index,
        "to_index": to_index,
        **extra,
    }
    if rung.tool == "emergency":
        detail["emergency_enabled"] = rung.params.get("enabled")
    _append(
        events,
        tool="jump",
        rung_id=rung.id,
        detail=detail,
    )


def _find_dial_primary_index(plan: CarePlan) -> int:
    for i, rung in enumerate(plan.rungs):
        if rung.tool == "dial_contact" and rung.params.get("contact") == "caregiver":
            return i
    for i, rung in enumerate(plan.rungs):
        if rung.tool == "dial_contact":
            return i
    raise ValueError("care plan has no dial_contact rung")


def _find_human_index(plan: CarePlan) -> int:
    """Prefer notify-before-dial when a caregiver stand-down rung exists."""
    for i, rung in enumerate(plan.rungs):
        if rung.tool == "notify_caretaker":
            return i
    return _find_dial_primary_index(plan)


def _is_occlusion(cue: CueEvent) -> bool:
    return cue.kind == "camera_occlusion" or bool(
        (cue.detail or {}).get("reason") == "lens_covered"
    )


def _refresh_ack(incident: Incident, store: AuditStore | None) -> bool:
    if incident.acked_by:
        return True
    if store is None:
        return False
    latest = store.get(incident.id)
    if latest is not None and latest.acked_by:
        incident.acked_by = latest.acked_by
        incident.acked_at = latest.acked_at
        return True
    return False


def _already_resolved_ack(events: list[AuditEvent]) -> bool:
    return any(
        e.tool == "resolve" and e.detail.get("reason") in {"caregiver_ack", "family_ack"}
        for e in events
    )


def _resolve_caregiver_ack(
    incident: Incident,
    events: list[AuditEvent],
    cue: CueEvent,
) -> None:
    incident.status = "resolved"
    if _already_resolved_ack(events):
        return
    _append(
        events,
        tool="resolve",
        cue_kind=cue.kind,
        detail={"reason": "caregiver_ack", "contact": incident.acked_by or "caregiver"},
    )


_CUE_CARD_LABEL = {
    "no_movement": "Stillness",
    "no_visibility": "No visibility",
    "distress_heuristic": "Distress check",
    "camera_occlusion": "Camera health",
}


def _close_calling_thread(incident: Incident, events: list[AuditEvent], cue: CueEvent) -> None:
    """Close a BotThread left on calling_N after dial resolve or exhaust.

    ``/family/runtime`` mirrors the latest thread. Without this, a resolved
    dial stays on ``calling_1``.
    """
    thread = incident.__dict__.get("_bot_thread")
    if thread is None or not str(thread.state.value).startswith("calling_"):
        return
    reason = "dial_answered" if incident.status == "resolved" else "dial_exhausted"
    seen = len(thread.audit_events())
    thread.close(reason)
    _flush_bot_audit(thread, events, cue, seen)


def _flush_bot_audit(thread: BotThread, events: list[AuditEvent], cue: CueEvent, seen: int) -> int:
    raw = thread.audit_events()
    for item in raw[seen:]:
        at = item.get("at")
        events.append(
            AuditEvent(
                tool="bot",
                cue_kind=cue.kind,
                detail=dict(item.get("detail") or {}),
                at=at if isinstance(at, datetime) else datetime.now(timezone.utc),
            )
        )
    return len(raw)


def _pop_telegram_ack(adapter: TelegramAdapter | None) -> AckIntent | None:
    if adapter is None:
        return None
    inbox = getattr(adapter, "inbox", None)
    if not inbox:
        return None
    while inbox:
        update = inbox.pop(0)
        intent = adapter.parse_update(update)
        if intent is not None:
            return intent
    return None


def _resolve_family_ack(
    incident: Incident,
    events: list[AuditEvent],
    cue: CueEvent,
    *,
    source: str,
    choice: str | None,
) -> None:
    incident.status = "resolved"
    incident.acked_by = incident.acked_by or "family"
    incident.acked_at = incident.acked_at or datetime.now(timezone.utc)
    if _already_resolved_ack(events):
        return
    detail: dict[str, Any] = {
        "reason": "family_ack",
        "contact": incident.acked_by,
        "source": source,
    }
    if choice is not None:
        detail["choice"] = choice
    _append(events, tool="resolve", cue_kind=cue.kind, detail=detail)


async def _poll_family_ack(
    incident: Incident,
    store: AuditStore | None,
    adapter: TelegramAdapter | None,
    sec: float,
) -> tuple[str | None, AckIntent | None, float]:
    """Poll store ack + Telegram inbox. Returns (kind, intent, slept)."""
    duration = max(0.0, float(sec))
    slept = 0.0
    step = 0.05
    while True:
        if _refresh_ack(incident, store) or incident.status == "resolved":
            return "caregiver_ack", None, slept
        intent = _pop_telegram_ack(adapter)
        if intent is not None:
            return "family_ack", intent, slept
        if slept >= duration:
            return None, None, slept
        chunk = min(step, duration - slept)
        await asyncio.sleep(chunk)
        slept += chunk


async def _run_family_page(
    incident: Incident,
    store: AuditStore | None,
    events: list[AuditEvent],
    cue: CueEvent,
    plan: CarePlan,
    adapter: TelegramAdapter,
    registry: BotRegistry,
    *,
    family_countdown_sec: float,
    pressure_at_remaining_sec: float,
    max_wait_sec: float,
    inform_only: bool,
) -> str:
    """Page family via BotThread. Returns ack | inform_only | timeout."""
    thread = BotThread(
        incident_id=incident.id,
        family_countdown_sec=family_countdown_sec,
        pressure_at_remaining_sec=pressure_at_remaining_sec,
    )
    registry.attach(thread)
    incident.__dict__["_bot_thread"] = thread

    remaining = max(1, int(round(family_countdown_sec)))
    card = InformCard(
        cue_text=_CUE_CARD_LABEL.get(cue.kind, cue.kind.replace("_", " ")),
        actions=("1", "2", "3"),
        remaining_sec=remaining,
        monitored_name=plan.monitored.display_name or "Mom",
    )
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip() or "stub-family"
    thread.enter_family_paged(card)
    adapter.send_inform(chat_id, card)
    seen = _flush_bot_audit(thread, events, cue, 0)
    if store is not None:
        store.save(incident)

    total = max(0.0, float(family_countdown_sec))
    pressure_rem = max(0.0, float(pressure_at_remaining_sec))
    pressure_delay = max(0.0, total - pressure_rem)
    budget = total if max_wait_sec < 0 else min(total, float(max_wait_sec))
    first = min(pressure_delay, budget)
    second = min(pressure_rem, max(0.0, budget - first))

    def _apply(kind: str, intent: AckIntent | None) -> None:
        nonlocal seen
        if kind == "family_ack" and intent is not None:
            thread.ack(source=intent.source, choice=intent.choice)
            seen = _flush_bot_audit(thread, events, cue, seen)
            _resolve_family_ack(
                incident, events, cue, source=intent.source, choice=intent.choice
            )
            return
        thread.ack(source="console", choice=None)
        seen = _flush_bot_audit(thread, events, cue, seen)
        _resolve_caregiver_ack(incident, events, cue)

    wait_first = total if inform_only else first
    if inform_only and max_wait_sec >= 0:
        wait_first = min(wait_first, float(max_wait_sec))
    kind, intent, _ = await _poll_family_ack(incident, store, adapter, wait_first)
    if kind is not None:
        _apply(kind, intent)
        if store is not None:
            store.save(incident)
        return "ack"
    if inform_only:
        return "inform_only"

    thread.enter_pressure()
    adapter.send_pressure_warn(chat_id, pressure_rem)
    seen = _flush_bot_audit(thread, events, cue, seen)
    if store is not None:
        store.save(incident)

    kind, intent, _ = await _poll_family_ack(incident, store, adapter, second)
    if kind is not None:
        _apply(kind, intent)
        if store is not None:
            store.save(incident)
        return "ack"

    thread.enter_calling(1)
    seen = _flush_bot_audit(thread, events, cue, seen)
    if store is not None:
        store.save(incident)
    return "timeout"


async def _sleep_or_ack(
    incident: Incident,
    store: AuditStore | None,
    sec: float,
    max_wait_sec: float,
) -> float:
    """Bound a wait and stand down immediately if a caregiver acks."""
    duration = max(0.0, float(sec))
    if max_wait_sec >= 0:
        duration = min(duration, float(max_wait_sec))
    slept = 0.0
    step = 0.05
    while slept < duration:
        if _refresh_ack(incident, store) or incident.status == "resolved":
            return slept
        chunk = min(step, duration - slept)
        await asyncio.sleep(chunk)
        slept += chunk
    return slept


async def _bounded_sleep(sec: float, max_wait_sec: float) -> float:
    """Sleep up to ``max_wait_sec`` (demo/tests stay snappy; full waits optional)."""
    duration = max(0.0, float(sec))
    if max_wait_sec >= 0:
        duration = min(duration, float(max_wait_sec))
    await asyncio.sleep(duration)
    return duration


def _parse_hhmm(value: str) -> time:
    hour_s, minute_s = value.strip().split(":", 1)
    return time(hour=int(hour_s), minute=int(minute_s))


def _in_quiet_hours(now: datetime, start_s: str, end_s: str) -> bool:
    """True if ``now.time()`` falls in [start, end) (supports overnight windows)."""
    start = _parse_hhmm(start_s)
    end = _parse_hhmm(end_s)
    t = now.timetz().replace(tzinfo=None) if now.tzinfo else now.time()
    # Compare as naive local clock components from the provided datetime.
    t = time(hour=t.hour, minute=t.minute, second=t.second)
    if start <= end:
        return start <= t < end
    # Overnight e.g. 22:00 → 07:00
    return t >= start or t < end


def _apply_privacy(
    frames: list[Any],
    mode: PrivacyMode,
) -> tuple[list[Any], PrivacyMode | None, int]:
    """Map frames through blur/silhouette. Non-zero count only with a privacy mode."""
    if not frames:
        return [], None, 0
    if mode == "silhouette":
        transformed = [to_silhouette(f) for f in frames]
    else:
        transformed = [blur_faces(f) for f in frames]
        mode = "blur"
    return transformed, mode, len(transformed)


def _learning_enabled(plan: CarePlan) -> bool:
    """Learning block absent -> enabled defaults (spec: missing = enabled)."""
    cfg = plan.learning
    return cfg is None or cfg.enabled


def _annotate_detection_frame(frame, cue) -> "np.ndarray | None":
    """Draw the detector's view: bounding box (if carried in cue.detail) plus
    key telemetry, so the console can show WHY the cue fired."""
    if frame is None or cue is None:
        return None
    import cv2

    out = frame.copy()
    d = cue.detail or {}
    h = out.shape[0]
    # bbox fields written by CueDetector for DNN/blob paths
    box = d.get("bbox") or d.get("box")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        x, y, w, hh = [int(v) for v in box]
        cv2.rectangle(out, (x, y), (x + w, y + hh), (66, 215, 255), 2)
    lines = [cue.kind]
    if d.get("confidence") is not None:
        lines.append(f"conf {d['confidence']}")
    if d.get("motion_mean") is not None:
        lines.append(f"motion {d['motion_mean']}")
    pose = d.get("pose")
    if isinstance(pose, dict):
        if pose.get("torso_angle_deg") is not None:
            lines.append(f"torso {pose['torso_angle_deg']}")
        if pose.get("hip_y_ratio") is not None:
            lines.append(f"hip {pose['hip_y_ratio']}")
    if d.get("torso_angle_deg") is not None:
        lines.append(f"torso {d['torso_angle_deg']}")
    if d.get("hip_y_ratio") is not None:
        lines.append(f"hip {d['hip_y_ratio']}")
    if d.get("pattern"):
        lines.append(str(d["pattern"]))
    y0 = 20
    for ln in lines[:5]:
        cv2.putText(out, ln, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (66, 215, 255), 1, cv2.LINE_AA)
        y0 += 18
    return out


async def run_incident(
    cue: CueEvent,
    plan: CarePlan,
    speaker: SpeakerChannel,
    dialer: StubDialer,
    pre_event_frames: list[Any] | None = None,
    routine_profile: "RoutineProfile | None" = None,
    *,
    store: AuditStore | None = None,
    detector: "CueDetector | None" = None,
    max_wait_sec: float = 0.05,
    privacy_mode: PrivacyMode = "blur",
    now: datetime | None = None,
    telegram: TelegramAdapter | None = None,
    bot_registry: BotRegistry | None = None,
) -> Incident:
    """Run the care-plan rung loop for one cue; return an Incident with audit events.

    Rung outcomes:
    - speaker ``clear_ok`` / ``ok`` → resolve
    - speaker ``needs_human`` / ``call_caregiver`` → jump to notify (or dial)
    - speaker ``unclear`` / ``silence`` → continue
    - ``notify_caretaker`` + caregiver ack → resolve ``caregiver_ack``
    - occlusion + silence after notify → inform-only resolve (never distress)
    - dial ``answered`` → resolve
    - dial ``no_answer`` → ``next_rung_after_no_answer`` (log jumps over skipped rungs)
    - ``emergency`` with ``enabled`` not True → fail-closed skip (never real 911)

    Pre-event frames are privacy-transformed (default blur) before attach count.

    ``detector`` (optional): the CueDetector that produced the cue. When both
    the detector and pre-event frames are available, the ``reperceive`` rung
    re-observes the buffered frames through a cold copy of the detector and
    logs a second CueEvent per cue kind that re-fires. Without it the rung is
    honestly labeled ``stub_ok`` / ``no_frames_buffered`` instead of claiming
    a re-check that did not happen.
    """
    # Adaptive schedule learning (spec 2026-09-27): resolve effective stillness
    # timeout from the household's RoutineProfile BEFORE building the incident,
    # and record the explain string on the cue detail for the timeline/UI.
    learning_detail: dict[str, Any] | None = None
    if routine_profile is not None and _learning_enabled(plan):
        plan_timeout = int(plan.triggers.no_movement.timeout_sec)
        eff = effective_no_movement_timeout_sec(plan, routine_profile)
        learning_detail = {
            "learning_phase": routine_profile.learning_phase,
            "effective_timeout_sec": eff,
            "plan_timeout_sec": plan_timeout,
            "explain": _learning_explain(
                routine_profile, plan_timeout_sec=plan_timeout, effective_sec=eff
            ),
        }
        cue = cue.model_copy(update={"detail": {**cue.detail, "learning": learning_detail}})

    frames_in = list(pre_event_frames or [])
    # Detection frame (P1.1): annotate the cue's box/telemetry on a copy of the
    # raw frame, THEN privacy-transform it like every other frame - judges see
    # what the detector saw (bbox + numbers) without identifying pixels.
    detection_frame = _annotate_detection_frame(frames_in[-1] if frames_in else None, cue)
    if detection_frame is not None:
        detection_frame, _, _ = _apply_privacy([detection_frame], privacy_mode)

    private_frames, privacy, frame_count = _apply_privacy(frames_in, privacy_mode)
    # Refuse non-zero attach without a privacy transform flag.
    if frame_count > 0 and privacy not in {"blur", "silhouette"}:
        private_frames, privacy, frame_count = [], None, 0

    clock = now or datetime.now(timezone.utc)
    incident = Incident(
        id=uuid.uuid4().hex,
        household_id=plan.household_id,
        cue=cue,
        events=[],
        status="open",
        created_at=clock,
        pre_event_frame_count=frame_count,
        privacy=privacy,
    )
    # Keep transformed frames available to callers that need a clip snapshot
    # without serializing numpy into the pydantic model / JSON timeline.
    incident.__dict__["_private_pre_event_frames"] = private_frames
    incident.__dict__["_private_detection_frame"] = (
        detection_frame[0] if detection_frame else None
    )

    events = incident.events
    cue_detail: dict[str, Any] = {
        "confidence": cue.confidence,
        "detail": cue.detail,
    }
    if privacy is not None:
        cue_detail["privacy"] = privacy
        cue_detail["pre_event_frame_count"] = frame_count
    _append(
        events,
        tool="cue",
        cue_kind=cue.kind,
        detail=cue_detail,
    )
    if store is not None:
        store.save(incident)

    if (
        plan.quiet_hours is not None
        and plan.quiet_hours.policy == "soft_suppress_non_distress"
        and cue.kind != "distress_heuristic"
        and _in_quiet_hours(clock, plan.quiet_hours.start, plan.quiet_hours.end)
    ):
        _append(
            events,
            tool="suppress",
            cue_kind=cue.kind,
            detail={
                "reason": "quiet_hours",
                "policy": plan.quiet_hours.policy,
                "quiet_hours_start": plan.quiet_hours.start,
                "quiet_hours_end": plan.quiet_hours.end,
            },
        )
        incident.status = "suppressed"
        if store is not None:
            store.save(incident)
        return incident

    idx = 0
    n = len(plan.rungs)
    skip_next_wait = False
    last_intent = "unclear"
    occlusion = _is_occlusion(cue)
    while idx < n:
        if _refresh_ack(incident, store) or incident.status == "resolved":
            _resolve_caregiver_ack(incident, events, cue)
            break
        rung = plan.rungs[idx]
        tool = rung.tool

        if tool == "reperceive":
            detail: dict[str, Any] = {"params": dict(rung.params)}
            second_cues: list[CueEvent] = []
            if detector is not None and frames_in:
                # Real re-observation: a cold detector re-runs CueDetector
                # over the buffered pre-event frames; every cue kind that
                # re-fires is logged as a second CueEvent (spec: reperceive
                # must confirm the cue, not just assert it).
                second_cues, telemetry = detector.reobserve(frames_in)
                detail["result"] = "reobserved"
                detail["second_opinion"] = telemetry
                if second_cues:
                    detail["confirmed"] = second_cues[0].kind == cue.kind
                    detail["confirmed_kind"] = second_cues[0].kind
            else:
                # No detector or no buffered frames (fixture-injected cues):
                # say so honestly instead of claiming a re-check happened.
                detail["result"] = (
                    "no_frames_buffered" if detector is not None else "stub_ok"
                )
            if occlusion:
                detail.update(
                    {
                        "holding": True,
                        "basis": "camera_health",
                        "note": "No reading · lens covered · not distress",
                    }
                )
            _append(
                events,
                tool="reperceive",
                cue_kind=cue.kind,
                rung_id=rung.id,
                detail=detail,
            )
            for ev in second_cues:
                events.append(
                    AuditEvent(
                        tool="cue",
                        cue_kind=ev.kind,
                        detail={
                            "confidence": ev.confidence,
                            "detail": ev.detail,
                            "second_opinion": True,
                            "source": "reperceive_cue_detector",
                        },
                        at=datetime.now(timezone.utc),
                    )
                )
            if store is not None:
                store.save(incident)
            idx += 1
            continue

        if tool == "speaker_prompt":
            text = str(rung.params.get("text", "Are you okay?"))
            if occlusion:
                text = str(rung.params.get("occlusion_text") or _OCCLUSION_PROMPT)
            wait_sec = float(rung.params.get("wait_sec", 0))
            consumed_follow_wait = False
            # Prefer following wait rung as listen window when speaker has no wait_sec.
            if "wait_sec" not in rung.params and idx + 1 < n and plan.rungs[idx + 1].tool == "wait":
                wait_sec = float(plan.rungs[idx + 1].params.get("sec", wait_sec))
                consumed_follow_wait = True
            # Bound the speaker listen window for demo/tests.
            listen = wait_sec
            if max_wait_sec >= 0:
                listen = min(listen, float(max_wait_sec))
            reply = await speaker.prompt(text, listen)
            intent = getattr(reply, "intent", None) or classify_response_intent(reply.raw)
            last_intent = intent
            _append(
                events,
                tool="speaker_prompt",
                cue_kind=cue.kind,
                rung_id=rung.id,
                detail={
                    "text": text,
                    "reply_kind": reply.kind,
                    "reply_raw": reply.raw,
                    "response_intent": intent,
                    "wait_sec": listen,
                },
            )
            if store is not None:
                store.save(incident)
            if _refresh_ack(incident, store):
                _resolve_caregiver_ack(incident, events, cue)
                break
            if intent == "clear_ok" or reply.kind == "ok":
                incident.status = "resolved"
                _append(
                    events,
                    tool="resolve",
                    cue_kind=cue.kind,
                    detail={"reason": "speaker_ok", "response_intent": intent},
                )
                break
            if intent == "needs_human" or reply.kind == "call_caregiver":
                target = _find_human_index(plan)
                for k in range(idx + 1, target):
                    _log_jump(
                        events,
                        plan.rungs[k],
                        reason="needs_human",
                        from_index=idx,
                        to_index=target,
                    )
                idx = target
                continue
            # unclear / silence → continue; skip wait if it was already the listen window
            if consumed_follow_wait:
                skip_next_wait = True
            idx += 1
            continue

        if tool == "wait":
            if skip_next_wait:
                _log_jump(
                    events,
                    rung,
                    reason="listen_window_already_consumed",
                    from_index=idx,
                    to_index=idx + 1 if idx + 1 < n else None,
                )
                skip_next_wait = False
                idx += 1
                continue
            sec = float(rung.params.get("sec", 0))
            slept = await _sleep_or_ack(incident, store, sec, max_wait_sec)
            _append(
                events,
                tool="wait",
                cue_kind=cue.kind,
                rung_id=rung.id,
                detail={"sec": sec, "slept_sec": slept},
            )
            if _refresh_ack(incident, store) or incident.status == "resolved":
                _resolve_caregiver_ack(incident, events, cue)
                break
            idx += 1
            continue

        if tool == "notify_caretaker":
            role = str(rung.params.get("contact") or rung.params.get("role") or "caregiver")
            contact = _contact_for_role(plan, role)
            channels = list(rung.params.get("channels") or ["console"])
            inform_only = occlusion and last_intent != "needs_human"
            _append(
                events,
                tool="notify_caretaker",
                cue_kind=cue.kind,
                rung_id=rung.id,
                detail={
                    "contact": role,
                    "contact_id": contact.display_name,
                    "channels": channels,
                    "basis": "camera_health_inform" if inform_only else "no_response_escalation",
                    "distress_claimed": False if inform_only else None,
                    "note": (
                        "lens covered · no reading · not claiming distress"
                        if inform_only
                        else "awaiting caregiver"
                    ),
                },
            )
            if store is not None:
                store.save(incident)
            await_sec = float(rung.params.get("await_sec", 0) or 0)
            family_countdown = float(
                rung.params.get("family_countdown_sec", await_sec) or await_sec
            )
            pressure_remaining = float(rung.params.get("pressure_at_remaining_sec", 0) or 0)
            adapter = telegram or (getattr(store, "telegram", None) if store else None)
            registry = bot_registry or (getattr(store, "bot_registry", None) if store else None)
            use_bot = adapter is not None or "telegram" in channels
            if use_bot:
                if adapter is None:
                    adapter = FakeTelegram()
                if registry is None:
                    registry = BotRegistry()
                if store is not None:
                    store.telegram = adapter
                    store.bot_registry = registry
                outcome = await _run_family_page(
                    incident,
                    store,
                    events,
                    cue,
                    plan,
                    adapter,
                    registry,
                    family_countdown_sec=family_countdown,
                    pressure_at_remaining_sec=pressure_remaining,
                    max_wait_sec=max_wait_sec,
                    inform_only=inform_only,
                )
                if outcome == "ack":
                    break
                if inform_only or outcome == "inform_only":
                    incident.status = "resolved"
                    _append(
                        events,
                        tool="resolve",
                        cue_kind=cue.kind,
                        detail={
                            "reason": "camera_health_informed",
                            "distress_claimed": False,
                        },
                    )
                    break
                idx += 1
                continue
            await _sleep_or_ack(incident, store, await_sec, max_wait_sec)
            if _refresh_ack(incident, store) or incident.status == "resolved":
                _resolve_caregiver_ack(incident, events, cue)
                break
            if inform_only:
                incident.status = "resolved"
                _append(
                    events,
                    tool="resolve",
                    cue_kind=cue.kind,
                    detail={
                        "reason": "camera_health_informed",
                        "distress_claimed": False,
                    },
                )
                break
            idx += 1
            continue

        if tool == "dial_contact":
            role = str(rung.params.get("contact", "caregiver"))
            contact = _contact_for_role(plan, role)
            ring_sec = float(rung.params.get("ring_sec", 1.0))
            if max_wait_sec >= 0:
                ring_sec = min(ring_sec, float(max_wait_sec))
            result = dialer.dial(contact, ring_sec=ring_sec)
            _append(
                events,
                tool="dial_contact",
                cue_kind=cue.kind,
                rung_id=rung.id,
                detail={
                    "contact": role,
                    "contact_id": result.contact_id,
                    "status": result.status,
                    "phone_e164": contact.phone_e164,
                },
            )
            if result.status == "answered":
                incident.status = "resolved"
                _append(
                    events,
                    tool="resolve",
                    cue_kind=cue.kind,
                    detail={"reason": "dial_answered", "contact_id": result.contact_id},
                )
                break
            if result.status in {"no_answer", "skipped"}:
                nxt = next_rung_after_no_answer(plan, idx)
                if nxt is None:
                    for k in range(idx + 1, n):
                        _log_jump(
                            events,
                            plan.rungs[k],
                            reason="no_answer_fail_closed_or_exhausted",
                            from_index=idx,
                            to_index=None,
                        )
                    incident.status = "exhausted"
                    break
                for k in range(idx + 1, nxt):
                    _log_jump(
                        events,
                        plan.rungs[k],
                        reason="no_answer_escalation",
                        from_index=idx,
                        to_index=nxt,
                    )
                idx = nxt
                continue
            idx += 1
            continue

        if tool == "emergency":
            enabled = rung.params.get("enabled") is True
            if not enabled:
                _log_jump(
                    events,
                    rung,
                    reason="emergency_disabled_fail_closed",
                    from_index=idx,
                    to_index=idx + 1 if idx + 1 < n else None,
                )
                idx += 1
                continue
            # Enabled emergency: audit only in demo. Never place a real 911 call.
            _append(
                events,
                tool="emergency",
                cue_kind=cue.kind,
                rung_id=rung.id,
                detail={
                    "executed": False,
                    "enabled": True,
                    "note": "demo_stub_no_real_911",
                },
            )
            incident.status = "exhausted"
            break

        # Unknown tool: log and continue
        _append(
            events,
            tool=tool,
            cue_kind=cue.kind,
            rung_id=rung.id,
            detail={"params": dict(rung.params), "result": "unknown_tool_skipped"},
        )
        idx += 1

    if incident.status == "open":
        incident.status = "exhausted"
    _close_calling_thread(incident, events, cue)

    # Adaptive schedule learning (spec 2026-09-27 section 5): record outcome
    # on every closed incident and audit the profile state. Learning never
    # touches rung shape or emergency; it only moves timing.
    if routine_profile is not None and _learning_enabled(plan):
        from care_ladder.learning.profile import record_incident_outcome

        start_hour = (now or datetime.now(timezone.utc)).hour
        if incident.created_at is not None:
            start_hour = incident.created_at.hour
        resolved_ok = incident.status == "resolved"
        ok_day = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
        record_incident_outcome(
            routine_profile,
            cue_kind=cue.kind,
            cue_start_hour=start_hour,
            resolved_ok=resolved_ok,
            ok_day=ok_day,
        )
        _append(
            events,
            tool="routine_profile_update",
            cue_kind=cue.kind,
            detail={
                "learning_phase": routine_profile.learning_phase,
                "confirmed_ok_days": routine_profile.confirmed_ok_days,
                "suggested_timeout_sec": routine_profile.suggested_no_movement_timeout_sec,
                "explain": (cue.detail.get("learning") or {}).get("explain", ""),
                "frozen": routine_profile.frozen,
            },
        )

    if store is not None:
        store.save(incident)
    return incident
