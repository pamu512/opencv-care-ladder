"""Self-hosted MCP server exposing Care Ladder care-flow tools over
Streamable HTTP (MCP spec 2025-11-25+), for the Alexa+ agent path.

Mounted as an ASGI sub-app inside the existing FastAPI app: one container,
one port. The Alexa+ agent (or the in-repo simulator, alexa_sim.py) calls
these tools at runtime to drive an incident through its rungs.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from care_ladder.channels.dial import StubDialer
from care_ladder.channels.speaker import SpeakerSimulator
from care_ladder.ladder.orchestrator import run_incident
from care_ladder.models import CueEvent
from care_ladder.plan_loader import load_care_plan

# Midday UTC clock so quiet-hours soft-suppress never hides the demo ladder.
from datetime import datetime, timezone

_DEMO_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AMAZON_PLAN = _REPO_ROOT / "configs" / "amazon_demo_home.yaml"

mcp: MCPServer = MCPServer("Care Ladder")

# In-memory incident sessions keyed by household+incident id (the AuditStore
# remains the durable record; this registry maps MCP sessions to incidents).
_SESSIONS: dict[str, dict[str, Any]] = {}
_PENDING_ANSWERS: dict[str, str] = {}


def _session_key(household_id: str, incident_id: str) -> str:
    return f"{household_id}:{incident_id}"


def _current_rung(sess: dict[str, Any] | None) -> str | None:
    if sess is None:
        return None
    inc = sess.get("incident")
    if inc is None:
        return None
    if inc.status == "resolved":
        return "resolve"
    tools = [e.tool for e in inc.events]
    last = None
    for rung in sess.get("rungs") or []:
        if rung in tools:
            last = rung
    if last:
        return last
    return tools[-1] if tools else None


def _rung_index(sess: dict[str, Any] | None, rung: str | None) -> int | None:
    if sess is None or not rung:
        return None
    rungs = sess.get("rungs") or []
    try:
        return rungs.index(rung) + 1
    except ValueError:
        return None


def session_snapshot(
    household_id: str,
    incident_id: str,
    sess: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One agent-memory blob reused on every MCP tool return."""
    if sess is None and household_id and incident_id:
        sess = _SESSIONS.get(_session_key(household_id, incident_id))
    tools = [e.tool for e in sess["incident"].events] if sess else []
    status = sess["incident"].status if sess else "unknown"
    rung = _current_rung(sess)
    return {
        "household_id": household_id,
        "incident_id": incident_id,
        "rung": rung,
        "rung_index": _rung_index(sess, rung),
        "status": status,
        "tools": tools,
    }


def _attach_session(
    payload: dict[str, Any],
    household_id: str,
    incident_id: str,
    sess: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snap = session_snapshot(household_id, incident_id, sess)
    if "status" in payload:
        snap = {**snap, "status": payload["status"]}
    if "tools" in payload and isinstance(payload["tools"], list):
        snap = {**snap, "tools": payload["tools"]}
    return {
        **payload,
        "household_id": household_id,
        "incident_id": incident_id,
        "rung": snap["rung"],
        "rung_index": snap["rung_index"],
        "status": snap["status"],
        "tools": snap["tools"],
        "session_snapshot": snap,
    }


@mcp.tool()
def start_or_resume_incident(
    cue_kind: str,
    household_id: str = "amazon-demo-1",
    confidence: float = 0.9,
    incident_id: str | None = None,
) -> dict[str, Any]:
    """Start (or resume) a care-ladder incident for a household.

    cue_kind: no_movement | no_visibility | distress_heuristic.
    Returns the incident id plus the plan's rung sequence.
    """
    if incident_id and _session_key(household_id, incident_id) in _SESSIONS:
        sess = _SESSIONS[_session_key(household_id, incident_id)]
        return _attach_session(
            {"incident_id": sess["incident"].id, "resumed": True,
             "status": sess["incident"].status, "rungs": sess["rungs"]},
            household_id, sess["incident"].id, sess,
        )
    if cue_kind not in {"no_movement", "no_visibility", "distress_heuristic"}:
        return _attach_session(
            {"error": "invalid cue_kind", "cue_kind": cue_kind},
            household_id, incident_id or "",
        )
    plan = load_care_plan(_AMAZON_PLAN)
    iid = incident_id or uuid.uuid4().hex
    cue = CueEvent(
        kind=cue_kind,  # type: ignore[arg-type]  # validated above
        confidence=confidence,
        detail={"via": "mcp", "incident_id": iid},
    )
    # The orchestrator runs async; run it to completion here (demo scale).
    incident = asyncio.run(
        run_incident(
            cue=cue,
            plan=plan,
            speaker=SpeakerSimulator(scripted=[]),
            dialer=StubDialer(behavior={}),
            pre_event_frames=[],
            now=_DEMO_NOW,
        )
    )
    # Keep the caller-visible id stable even though run_incident mints its own.
    incident.id = iid
    _SESSIONS[_session_key(household_id, iid)] = {
        "incident": incident,
        "rungs": [r.tool for r in plan.rungs],
    }
    return _attach_session(
        {"incident_id": iid, "resumed": False, "status": incident.status,
         "rungs": [r.tool for r in plan.rungs]},
        household_id, iid,
    )


@mcp.tool()
def check_in_prompt(household_id: str, incident_id: str, utterance: str) -> dict[str, Any]:
    """Record the monitored person's utterance from a voice check-in.

    The Alexa+ agent calls this after each TTS attempt. Fail-closed intent
    (clear_ok / needs_human / unclear) is the gate; reply_kind stays as the
    legacy ok / call_caregiver / silence mapping.
    """
    _PENDING_ANSWERS[_session_key(household_id, incident_id)] = utterance
    from care_ladder.channels.response_intent import (
        classify_response_intent,
        intent_label,
        intent_to_reply_kind,
    )

    intent = classify_response_intent(utterance)
    kind = intent_to_reply_kind(intent, utterance)
    return _attach_session(
        {
            "incident_id": incident_id,
            "reply_kind": kind,
            "response_intent": intent,
            "intent_label": intent_label(intent),
            "raw": utterance,
        },
        household_id,
        incident_id,
    )


@mcp.tool()
def advance_rung(household_id: str, incident_id: str) -> dict[str, Any]:
    """Advance the incident to its next rung (silence / no-answer path)."""
    sess = _SESSIONS.get(_session_key(household_id, incident_id))
    if sess is None:
        return _attach_session(
            {"error": "unknown incident", "incident_id": incident_id},
            household_id, incident_id,
        )
    tools = [e.tool for e in sess["incident"].events]
    return _attach_session(
        {"incident_id": incident_id, "advanced_to": sess["rungs"][-1],
         "events_so_far": tools},
        household_id, incident_id, sess,
    )


@mcp.tool()
def resolve_incident(
    household_id: str, incident_id: str, reason: str = "voice_ok"
) -> dict[str, Any]:
    """Resolve an incident (voice OK or caretaker acknowledge)."""
    sess = _SESSIONS.get(_session_key(household_id, incident_id))
    if sess is None:
        return _attach_session(
            {"error": "unknown incident", "incident_id": incident_id},
            household_id, incident_id,
        )
    if sess["incident"].status == "resolved":
        return _attach_session(
            {"error": "already_resolved", "incident_id": incident_id},
            household_id, incident_id, sess,
        )
    sess["incident"].status = "resolved"
    sess["incident"].events.append(
        type(sess["incident"].events[0])(
            tool="resolve", cue_kind=sess["incident"].cue.kind,
            detail={"reason": reason, "via": "mcp"},
        )
    ) if sess["incident"].events else None
    return _attach_session(
        {"incident_id": incident_id, "status": "resolved", "reason": reason},
        household_id, incident_id, sess,
    )


@mcp.tool()
def get_incident_status(household_id: str, incident_id: str) -> dict[str, Any]:
    """Return the incident's status, cue, and audit-trail tool sequence."""
    sess = _SESSIONS.get(_session_key(household_id, incident_id))
    if sess is None:
        return _attach_session(
            {"error": "unknown incident", "incident_id": incident_id},
            household_id, incident_id,
        )
    inc = sess["incident"]
    return _attach_session(
        {
            "incident_id": inc.id,
            "status": inc.status,
            "cue_kind": inc.cue.kind,
            "tools": [e.tool for e in inc.events],
            "rungs": sess["rungs"],
        },
        household_id, inc.id, sess,
    )


@mcp.tool()
def notify_caretaker(household_id: str, incident_id: str) -> dict[str, Any]:
    """Notify the caretaker (push mock + Fire TV). Demo-simulated, no real push."""
    sess = _SESSIONS.get(_session_key(household_id, incident_id))
    if sess is None:
        return _attach_session(
            {"error": "unknown incident", "incident_id": incident_id},
            household_id, incident_id,
        )
    inc = sess["incident"]
    inc.events.append(
        type(inc.events[0])(
            tool="notify_caretaker", cue_kind=inc.cue.kind,
            detail={"channels": ["push_mock", "fire_tv"], "simulated": True, "via": "mcp"},
        )
    ) if inc.events else None
    return _attach_session(
        {"incident_id": inc.id, "notified": True, "channels": ["push_mock", "fire_tv"],
         "simulated": True},
        household_id, inc.id, sess,
    )


@mcp.tool()
def request_call(household_id: str, incident_id: str) -> dict[str, Any]:
    """Request a call to the caregiver's reserved fictional number. Simulated only."""
    sess = _SESSIONS.get(_session_key(household_id, incident_id))
    if sess is None:
        return _attach_session(
            {"error": "unknown incident", "incident_id": incident_id},
            household_id, incident_id,
        )
    plan = load_care_plan(_AMAZON_PLAN)
    return _attach_session(
        {
            "incident_id": incident_id,
            "phone_e164": plan.caregiver.phone_e164,
            "simulated": True,
            "note": "demo_stub_no_real_dial",
        },
        household_id, incident_id, sess,
    )


def mount_path() -> str:
    """ASGI sub-app for FastAPI mounting at /mcp."""
    return "/mcp"


__all__ = ["mcp", "mount_path", "session_snapshot"]
