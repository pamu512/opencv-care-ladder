"""FastAPI incident timeline + demo trigger (demo fixtures only; no live camera)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import asyncio

from fastapi import FastAPI, HTTPException, Response, UploadFile
from uuid import uuid4
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from care_ladder.audit.store import AuditStore
from care_ladder.channels.dial import StubDialer
from care_ladder.cloud.sinks import CloudSinks
from care_ladder.channels.speaker import SpeakerSimulator
from care_ladder.ladder.orchestrator import run_incident
from care_ladder.learning.profile import (
    LearningConfig,
    RoutineProfile,
    freeze as _freeze_learning,
    mark_settled as _mark_settled,
    reset as _reset_learning,
)
from care_ladder.learning.store import RoutineProfileJSONStore
from care_ladder.models import AuditEvent, CueEvent
from care_ladder.plan_loader import load_care_plan
from care_ladder.vision.cues import CueDetector
from care_ladder.vision.ingest import ingest_video, save_upload

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEMO_PLAN_PATH = _REPO_ROOT / "configs" / "demo_home.yaml"
_LEARNING_STORE = RoutineProfileJSONStore(_REPO_ROOT / "data" / "routine_profiles")

SUPPORTED_FIXTURES = frozenset(
    {
        "quiet_hours_suppressed",
        "no_visibility",
        "no_movement_ok",
        "no_movement_silence",
        "opencv_stillness",
        "opencv_dnn_person",
        "opencv_pose_person",
        "opencv_occlusion",
        "speaker_soft_ok",
        "speaker_needs_human",
        "path_b_inflight",
    }
)

_TOOL_LABELS = {
    "cue": "OpenCV cue",
    "reperceive": "Re-check the room",
    "speaker_prompt": "Spoken check-in",
    "wait": "Listen window",
    "notify_caretaker": "Notify caregiver",
    "notify": "Caregiver note",
    "dial_contact": "Dial",
    "emergency": "Emergency (fail-closed)",
    "resolve": "Resolve",
    "jump": "Skip (logged)",
    "suppress": "Quiet hours",
    "routine_profile_update": "Learning update",
}

_VISION_LABELS = {
    "no_movement": "Stillness in zone",
    "no_visibility": "Person left zone",
    "distress_heuristic": "Fall signature (non-clinical)",
    "camera_occlusion": "No reading · lens covered",
}
_POSE_MODEL_PATH = _REPO_ROOT / "models" / "pose_estimation_mediapipe_2023mar.onnx"
_MODEL_PATH = _REPO_ROOT / "models" / "person_detection_mediapipe_2023mar.onnx"
# Midday UTC so quiet_hours soft-suppress does not hide the judge demo ladder.
DEMO_NOW = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)


class DemoRunRequest(BaseModel):
    fixture: str = Field(
        ...,
        description='Demo fixture id, e.g. "no_movement_silence" or "opencv_stillness"',
    )


class DemoRunResponse(BaseModel):
    incident_id: str


def _incident_summary(incident) -> dict[str, Any]:
    return {
        "id": incident.id,
        "household_id": incident.household_id,
        "status": incident.status,
        "cue": incident.cue.model_dump(),
        "event_count": len(incident.events),
        "acked_by": incident.acked_by,
    }


def incident_explain(incident) -> dict[str, Any]:
    """Management-grade explain surface: what vision knew, what we did, how long, who acked."""
    events = incident.events
    speaker = next((e for e in events if e.tool == "speaker_prompt"), None)
    resolve = next((e for e in events if e.tool == "resolve"), None)
    notify = next((e for e in events if e.tool == "notify_caretaker"), None)
    stamped = [e.at for e in events if e.at is not None]
    start = incident.created_at or (stamped[0] if stamped else None)
    end = stamped[-1] if stamped else None
    duration = None
    if start is not None and end is not None:
        duration = max(0.0, (end - start).total_seconds())
    actions = []
    prev = start
    for e in events:
        if e.tool in {"cue", "routine_profile_update"}:
            prev = e.at or prev
            continue
        delta = None
        if e.at is not None and prev is not None:
            delta = max(0.0, (e.at - prev).total_seconds())
        actions.append(
            {
                "tool": e.tool,
                "label": _TOOL_LABELS.get(e.tool, e.tool.replace("_", " ")),
                "delta_sec": delta,
                "rung_id": e.rung_id,
            }
        )
        prev = e.at or prev
    return {
        "vision": _VISION_LABELS.get(incident.cue.kind, incident.cue.kind),
        "vision_note": (incident.cue.detail or {}).get("note"),
        "intent": (speaker.detail or {}).get("response_intent") if speaker else None,
        "reply_quote": (speaker.detail or {}).get("reply_raw") if speaker else None,
        "checkin_prompt": (speaker.detail or {}).get("text") if speaker else None,
        "actions": actions,
        "duration_sec": duration,
        "acked_by": incident.acked_by,
        "resolve_reason": (resolve.detail or {}).get("reason") if resolve else None,
        "notify_basis": (notify.detail or {}).get("basis") if notify else None,
        "distress_claimed": False if incident.cue.kind == "camera_occlusion" else None,
    }


async def _run_quiet_hours_suppressed(store: AuditStore):
    """Fixture: non-distress cue inside quiet hours -> suppressed (audited)."""
    plan = load_care_plan(_DEMO_PLAN_PATH)
    cue = CueEvent(kind="no_movement", confidence=0.9, detail={"fixture": "quiet_hours_suppressed"})
    speaker = SpeakerSimulator(scripted=["I'm fine"])
    dialer = StubDialer(behavior={})
    # DEMO_NOW is 12:00; quiet hours window 22:00-07:00 -> pass a 23:30 clock
    from datetime import datetime, timezone as tz

    night = datetime(2026, 9, 11, 23, 30, tzinfo=tz.utc)
    incident = await run_incident(
        cue=cue, plan=plan, speaker=speaker, dialer=dialer,
        pre_event_frames=[], store=store, now=night,
    )
    return incident


async def _run_no_visibility(store: AuditStore):
    """Fixture: person left the monitored zone -> no_visibility cue."""
    plan = load_care_plan(_DEMO_PLAN_PATH)
    cue = CueEvent(kind="no_visibility", confidence=0.85, detail={"fixture": "no_visibility"})
    speaker = SpeakerSimulator(scripted=["I'm fine"])
    dialer = StubDialer(behavior={})
    incident = await run_incident(
        cue=cue, plan=plan, speaker=speaker, dialer=dialer,
        pre_event_frames=[], store=store, now=DEMO_NOW,
    )
    return incident


def _demo_profile():
    """RoutineProfile for the OpenCV demo household (JSON store, gitignored)."""
    return _LEARNING_STORE.get_or_create("demo-home-1")


def _save_profile(profile):
    _LEARNING_STORE.save(profile)


async def _run_no_movement_ok(store: AuditStore):
    """Fixture: no_movement cue + verbal OK → resolve without dial (spec §10 Path A)."""
    plan = load_care_plan(_DEMO_PLAN_PATH)
    cue = CueEvent(kind="no_movement", confidence=0.9, detail={"fixture": "no_movement_ok"})
    speaker = SpeakerSimulator(scripted=["I'm fine"])
    dialer = StubDialer(behavior={})  # never reached on this path
    profile = _demo_profile()
    incident = await run_incident(
        cue=cue,
        plan=plan,
        speaker=speaker,
        dialer=dialer,
        pre_event_frames=[],
        routine_profile=profile,
        store=store,
        now=DEMO_NOW,
    )
    _save_profile(profile)  # persist learning update (spec section 5)
    return incident


async def _run_speaker_soft_ok(store: AuditStore):
    """Stillness + soft OK ('don't worry') → clear_ok, no dial."""
    plan = load_care_plan(_DEMO_PLAN_PATH)
    cue = CueEvent(kind="no_movement", confidence=0.9, detail={"fixture": "speaker_soft_ok"})
    speaker = SpeakerSimulator(scripted=["don't worry"])
    dialer = StubDialer(behavior={})
    profile = _demo_profile()
    incident = await run_incident(
        cue=cue, plan=plan, speaker=speaker, dialer=dialer,
        pre_event_frames=[], routine_profile=profile, store=store, now=DEMO_NOW,
    )
    _save_profile(profile)
    return incident


async def _run_speaker_needs_human(store: AuditStore):
    """Stillness + mixed hurt never invents OK → needs_human → notify/dial."""
    plan = load_care_plan(_DEMO_PLAN_PATH)
    cue = CueEvent(kind="no_movement", confidence=0.9, detail={"fixture": "speaker_needs_human"})
    speaker = SpeakerSimulator(scripted=["I'm ok but I hurt my hip"])
    dialer = StubDialer(behavior={"caregiver": "answered"})
    profile = _demo_profile()
    incident = await run_incident(
        cue=cue, plan=plan, speaker=speaker, dialer=dialer,
        pre_event_frames=[], routine_profile=profile, store=store, now=DEMO_NOW,
    )
    _save_profile(profile)
    return incident


def _synthetic_occlusion_frames(
    *,
    width: int = 640,
    height: int = 480,
) -> list[np.ndarray]:
    """Uniform near-black frames: covered / unreadable lens."""
    return [np.full((height, width, 3), 4, dtype=np.uint8) for _ in range(4)]


async def _run_opencv_occlusion(store: AuditStore):
    """Fixture: covered-lens frames → CueDetector camera_occlusion → inform-only Path B."""
    plan = load_care_plan(_DEMO_PLAN_PATH)
    detector = CueDetector.from_plan(plan, zone_id="living_room")
    frames = _synthetic_occlusion_frames()
    cue: CueEvent | None = None
    for frame, t in zip(frames, [0.0, 0.4, 0.8, 1.2], strict=True):
        cue = detector.observe(frame, t=t)
        if cue is not None:
            break
    if cue is None:
        raise RuntimeError("opencv_occlusion fixture: CueDetector did not emit camera_occlusion")
    cue.detail = {
        **cue.detail,
        "source": "opencv_cue_detector",
        "fixture": "opencv_occlusion",
        "distress_claimed": False,
        "note": "No reading · lens covered",
    }
    speaker = SpeakerSimulator(scripted=[])
    dialer = StubDialer(behavior={"caregiver": "answered"})
    return await run_incident(
        cue=cue,
        plan=plan,
        speaker=speaker,
        dialer=dialer,
        pre_event_frames=frames[-2:],
        store=store,
        privacy_mode="silhouette",
        now=DEMO_NOW,
    )


async def _run_path_b_inflight(store: AuditStore, *, max_wait_sec: float = 25.0):
    """Path B left open on the notify rung so a caregiver can ack mid-ladder."""
    plan = load_care_plan(_DEMO_PLAN_PATH)
    cue = CueEvent(kind="no_movement", confidence=0.9, detail={"fixture": "path_b_inflight"})
    # Immediate silence (empty utterance) so the interruptible window is notify+await.
    speaker = SpeakerSimulator(scripted=[""])
    dialer = StubDialer(behavior={"caregiver": "no_answer", "secondary": "answered"})
    return await run_incident(
        cue=cue,
        plan=plan,
        speaker=speaker,
        dialer=dialer,
        pre_event_frames=[],
        store=store,
        now=DEMO_NOW,
        max_wait_sec=max_wait_sec,
    )


async def _run_no_movement_silence(store: AuditStore):
    """Fixture: no_movement cue + speaker silence → escalate via stub dialer."""
    plan = load_care_plan(_DEMO_PLAN_PATH)
    cue = CueEvent(kind="no_movement", confidence=0.9, detail={"fixture": "no_movement_silence"})
    # Silence path: empty script → escalate; secondary answers so incident resolves.
    speaker = SpeakerSimulator(scripted=[])
    dialer = StubDialer(behavior={"caregiver": "no_answer", "secondary": "answered"})
    profile = _demo_profile()
    incident = await run_incident(
        cue=cue,
        plan=plan,
        speaker=speaker,
        dialer=dialer,
        pre_event_frames=[],
        routine_profile=profile,
        store=store,
        now=DEMO_NOW,
    )
    _save_profile(profile)  # persist learning update (spec section 5)
    return incident


def _synthetic_stillness_frames(
    *,
    width: int = 160,
    height: int = 120,
) -> list[np.ndarray]:
    """Build a short synthetic still-person sequence for CueDetector (no live camera)."""
    frames: list[np.ndarray] = []
    for _ in range(4):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[40:80, 60:100] = 200
        frames.append(frame)
    return frames


async def _run_opencv_stillness(store: AuditStore):
    """Fixture: synthetic frames → CueDetector.observe → run_incident (OpenCV path)."""
    plan = load_care_plan(_DEMO_PLAN_PATH)
    # Short timeout so demo/tests emit no_movement without waiting plan's 900s.
    plan.triggers.no_movement.timeout_sec = 2
    # Fixture pins the demo clock (2s): spec section 6 - an explicit fixture
    # timeout overrides learning, so no profile here.
    detector = CueDetector.from_plan(plan, zone_id="living_room")
    # Zone in demo YAML is 640x480; use matching canvas so blob sits in-zone.
    frames = _synthetic_stillness_frames(width=640, height=480)
    # Place blob well inside living_room polygon
    for f in frames:
        f[:, :] = 0
        f[200:280, 300:380] = 200

    cue: CueEvent | None = None
    times = [0.0, 1.0, 2.5, 3.0]
    for frame, t in zip(frames, times, strict=True):
        cue = detector.observe(frame, t=t)
        if cue is not None:
            break
    if cue is None:
        raise RuntimeError("opencv_stillness fixture: CueDetector did not emit a cue")

    cue.detail = {
        **cue.detail,
        "source": "opencv_cue_detector",
        "fixture": "opencv_stillness",
    }

    speaker = SpeakerSimulator(scripted=[])
    dialer = StubDialer(behavior={"caregiver": "no_answer", "secondary": "answered"})
    # Attach last frames (privacy-blurred inside run_incident).
    incident = await run_incident(
        cue=cue,
        plan=plan,
        speaker=speaker,
        dialer=dialer,
        pre_event_frames=frames[-2:],
        store=store,
        privacy_mode="blur",
        now=DEMO_NOW,
    )
    return incident


async def _run_opencv_dnn_person(store: AuditStore):
    """Fixture: real photo → DNN person detector → zone check → ladder.

    Uses tests/fixtures/basketball1.png (OpenCV sample image with a person).
    Requires models/person_detection_mediapipe_2023mar.onnx (scripts/download_models.sh).
    The DNN localizes the person (in-zone) every frame; identical frames → motion stays
    ~0 → `no_movement` cue → ladder.
    """
    import cv2

    photo = cv2.imread(str(_REPO_ROOT / "tests" / "fixtures" / "basketball1.png"))
    if photo is None:
        raise RuntimeError("opencv_dnn_person fixture: sample photo missing")
    if not _MODEL_PATH.exists():
        raise RuntimeError(
            "opencv_dnn_person fixture: run scripts/download_models.sh first"
        )

    from care_ladder.vision.mppersondet import MPPersonDet

    plan = load_care_plan(_DEMO_PLAN_PATH)
    plan.triggers.no_movement.timeout_sec = 2  # demo clock, not plan's 900s
    detector = CueDetector.from_plan(plan, zone_id="living_room")
    detector.person_detector = MPPersonDet(str(_MODEL_PATH), scoreThreshold=0.3)

    cue: CueEvent | None = None
    times = [0.0, 1.0, 2.5, 3.0]
    for t in times:
        cue = detector.observe(photo, t=t)
        if cue is not None:
            break
    if cue is None:
        raise RuntimeError("opencv_dnn_person fixture: detector emitted no cue")

    cue.detail = {
        **cue.detail,
        "source": "opencv_dnn_person_detector",
        "detector": "mediapipe_persondet_2023mar (OpenCV 5 DNN)",
        "fixture": "opencv_dnn_person",
    }

    speaker = SpeakerSimulator(scripted=["I'm fine, thanks"])
    dialer = StubDialer(behavior={})
    incident = await run_incident(
        cue=cue,
        plan=plan,
        speaker=speaker,
        dialer=dialer,
        pre_event_frames=[photo],
        store=store,
        privacy_mode="silhouette",
        now=DEMO_NOW,
    )
    return incident


def _default_store() -> AuditStore:
    """dynamodb when CARE_LADDER_STORE=dynamodb (and boto3 present), else memory."""
    if os.environ.get("CARE_LADDER_STORE", "").lower() == "dynamodb":
        try:
            from care_ladder.audit.dynamo_store import DynamoAuditStore

            return DynamoAuditStore()
        except Exception as exc:  # pragma: no cover - env-dependent
            print(f"WARNING: CARE_LADDER_STORE=dynamodb failed ({exc}); using memory")
    return AuditStore()


async def _publish_cloud(incident) -> None:
    """Best-effort S3 clip upload + EventBridge cue emission (no-ops locally).

    Never fails the incident: cloud sink errors are logged and swallowed.
    """
    try:
        sinks = CloudSinks()
        if not sinks.enabled:
            return
        frames = incident.__dict__.get("_private_pre_event_frames") or []
        uris = sinks.upload_clip_frames(incident.id, frames, incident.privacy)
        sinks.emit_cue(incident, uris)
    except Exception as exc:  # pragma: no cover - env-dependent
        print(f"WARNING: cloud publish failed for {incident.id}: {exc}")


async def _run_opencv_pose_person(store: AuditStore):
    """Fixture: real photo → person ONNX → pose ONNX → torso metrics, no distress.

    Standing person: pose runs, torso angle computed, no distress cue; the
    incident demonstrates the pose path end-to-end with `source: pose_heuristics`
    context in the cue detail (pattern telemetry, not a fall).
    """
    import cv2

    photo = cv2.imread(str(_REPO_ROOT / "tests" / "fixtures" / "basketball1.png"))
    if photo is None:
        raise RuntimeError("opencv_pose_person fixture: sample photo missing")
    if not _MODEL_PATH.exists() or not _POSE_MODEL_PATH.exists():
        raise RuntimeError("opencv_pose_person fixture: run scripts/download_models.sh first")

    from care_ladder.vision.mppersondet import MPPersonDet
    from care_ladder.vision.mppose import MPPose

    plan = load_care_plan(_DEMO_PLAN_PATH)
    plan.triggers.no_movement.timeout_sec = 2
    # Fixture pins the demo clock (2s): spec section 6 - an explicit fixture
    # timeout overrides learning, so no profile here.
    detector = CueDetector.from_plan(plan, zone_id="living_room")
    detector.person_detector = MPPersonDet(str(_MODEL_PATH), scoreThreshold=0.3)
    detector.pose_model = MPPose(str(_POSE_MODEL_PATH), confThreshold=0.5)

    cue = None
    for t in [0.0, 0.5, 1.0, 2.5, 3.0]:
        cue = detector.observe(photo, t=t)
        if cue is not None:
            break
    if cue is None:
        raise RuntimeError("opencv_pose_person fixture: detector emitted no cue")

    cue.detail = {
        **cue.detail,
        "source": "opencv_pose_path",
        "pose": getattr(detector, "last_pose_metrics", None),
        "models": ["person_detection_mediapipe_2023mar", "pose_estimation_mediapipe_2023mar"],
        "fixture": "opencv_pose_person",
    }
    speaker = SpeakerSimulator(scripted=["I'm fine"])
    dialer = StubDialer(behavior={})
    return await run_incident(
        cue=cue, plan=plan, speaker=speaker, dialer=dialer,
        pre_event_frames=[photo], store=store, privacy_mode="silhouette", now=DEMO_NOW,
    )


# In-flight upload analysis jobs: job_id -> {status, filename, incident_id, error}
_UPLOAD_JOBS: dict[str, dict[str, Any]] = {}


def create_app(store: AuditStore | None = None) -> FastAPI:
    """Build FastAPI app with injectable store (tests inject a fresh memory store)."""
    audit = store if store is not None else _default_store()
    application = FastAPI(
        title="Care Ladder",
        description="Incident timeline + demo trigger (reserved phones; emergency fail-closed).",
        version="0.1.0",
    )
    application.state.store = audit

    static_dir = Path(__file__).resolve().parent / "static"
    application.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")

    @application.get("/learning/{household_id}")
    def get_learning(household_id: str) -> dict[str, Any]:
        """RoutineProfile for the UI badge: phase, days, explain, controls state."""
        profile = _LEARNING_STORE.get_or_create(household_id)
        plan = load_care_plan(_DEMO_PLAN_PATH)
        from care_ladder.learning.profile import (
            effective_no_movement_timeout_sec,
            explain as _learning_explain,
        )

        eff = effective_no_movement_timeout_sec(plan, profile)

        return {
            "household_id": household_id,
            "learning_phase": profile.learning_phase,
            "frozen": profile.frozen,
            "confirmed_ok_days": profile.confirmed_ok_days,
            "settled_after_days": profile.settled_after_days,
            "effective_no_movement_timeout_sec": eff,
            "plan_timeout_sec": plan.triggers.no_movement.timeout_sec,
            "usual_still_end_hour": profile.usual_still_end_hour,
            "explain": _learning_explain(
                profile,
                plan_timeout_sec=plan.triggers.no_movement.timeout_sec,
                effective_sec=eff,
            ),
        }

    @application.post("/learning/{household_id}/freeze")
    def learning_freeze(household_id: str) -> dict[str, Any]:
        profile = _LEARNING_STORE.get_or_create(household_id)
        _freeze_learning(profile)
        _LEARNING_STORE.save(profile)
        return {"household_id": household_id, "frozen": True}

    @application.post("/learning/{household_id}/reset")
    def learning_reset(household_id: str) -> dict[str, Any]:
        profile = _reset_learning(_LEARNING_STORE.get_or_create(household_id))
        _LEARNING_STORE.save(profile)
        return {"household_id": household_id, "learning_phase": "rapid"}

    @application.post("/learning/{household_id}/settle")
    def learning_settle(household_id: str) -> dict[str, Any]:
        profile = _mark_settled(_LEARNING_STORE.get_or_create(household_id))
        _LEARNING_STORE.save(profile)
        return {"household_id": household_id, "learning_phase": "settled"}

    @application.get("/plan")
    def get_plan() -> dict[str, Any]:
        """Current demo care plan (phone numbers redacted) for the console's
        full-ladder rail: shows never-reached rungs (e.g. emergency fail-closed)."""
        plan = load_care_plan(_DEMO_PLAN_PATH)

        def _redact(phone: str | None) -> str | None:
            if not phone:
                return phone
            return f"****{phone[-2:]}" if len(phone) >= 2 else "****"

        data = plan.model_dump()
        for key in ("caregiver", "secondary", "monitored"):
            contact = data.get(key)
            if contact and contact.get("phone_e164"):
                contact["phone_e164"] = _redact(contact["phone_e164"])
        return data

    @application.get("/incidents")
    def list_incidents() -> list[dict[str, Any]]:
        return [_incident_summary(i) for i in application.state.store.list_incidents()]

    @application.get("/incidents/{incident_id}")
    def get_incident(incident_id: str) -> dict[str, Any]:
        incident = application.state.store.get(incident_id)
        if incident is None:
            raise HTTPException(status_code=404, detail="incident not found")
        # Full timeline JSON plus a management-grade explain surface.
        payload = incident.model_dump()
        payload["explain"] = incident_explain(incident)
        return payload

    @application.post("/incidents/{incident_id}/ack")
    def ack_incident(incident_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Caregiver acknowledgement: a human confirmed they saw this incident.

        Stamps ``acked_by`` / ``acked_at`` and appends a ``caregiver_ack``
        audit event. If the ladder is still open (mid-flight notify/wait),
        ack resolves the incident with reason ``caregiver_ack`` so escalation
        stands down. Already-closed incidents keep their status.
        """
        incident = application.state.store.get(incident_id)
        if incident is None:
            raise HTTPException(status_code=404, detail="incident not found")
        if incident.acked_by is not None:
            raise HTTPException(status_code=409, detail="incident already acknowledged")
        body = body or {}
        contact = str(body.get("contact", "caregiver"))
        note = str(body.get("note") or "")[:200]
        now = datetime.now(timezone.utc)
        incident.acked_by = contact
        incident.acked_at = now
        incident.events.append(
            AuditEvent(
                tool="notify",
                cue_kind=incident.cue.kind,
                detail={"action": "caregiver_ack", "contact": contact, "note": note},
                at=now,
            )
        )
        if incident.status == "open":
            incident.status = "resolved"
            incident.events.append(
                AuditEvent(
                    tool="resolve",
                    cue_kind=incident.cue.kind,
                    detail={"reason": "caregiver_ack", "contact": contact, "note": note},
                    at=now,
                )
            )
        application.state.store.save(incident)
        return {
            "incident_id": incident.id,
            "acked_by": contact,
            "acked_at": now.isoformat(),
            "status": incident.status,
        }

    @application.get("/incidents/{incident_id}/frames/{index}", response_class=Response)
    def get_incident_frame(incident_id: str, index: int):
        """Serve a privacy-transformed pre-event frame as PNG.

        Frames reaching this endpoint already went through blur/silhouette in
        run_incident; raw identifiable pixels never enter the store.
        """
        incident = application.state.store.get(incident_id)
        if incident is None:
            raise HTTPException(status_code=404, detail="incident not found")
        frames = incident.__dict__.get("_private_pre_event_frames") or []
        if not (0 <= index < len(frames)):
            raise HTTPException(
                status_code=404,
                detail=f"frame index {index} out of range (0..{max(len(frames)-1, 0)})",
            )
        ok, buf = cv2.imencode(".png", frames[index])
        if not ok:
            raise HTTPException(status_code=500, detail="png encode failed")
        return Response(content=buf.tobytes(), media_type="image/png")

    @application.get("/incidents/{incident_id}/detection_frame", response_class=Response)
    def get_detection_frame(incident_id: str):
        """The frame the detector saw at cue time, annotated (bbox + telemetry)
        and privacy-transformed. Answers 'why did this cue fire' visually."""
        incident = application.state.store.get(incident_id)
        if incident is None:
            raise HTTPException(status_code=404, detail="incident not found")
        frame = incident.__dict__.get("_private_detection_frame")
        if frame is None:
            raise HTTPException(status_code=404, detail="no detection frame for this incident")
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            raise HTTPException(status_code=500, detail="png encode failed")
        return Response(content=buf.tobytes(), media_type="image/png")

    @application.get("/incidents/{incident_id}/frames")
    def list_incident_frames(incident_id: str) -> dict[str, Any]:
        incident = application.state.store.get(incident_id)
        if incident is None:
            raise HTTPException(status_code=404, detail="incident not found")
        frames = incident.__dict__.get("_private_pre_event_frames") or []
        return {
            "privacy": incident.privacy,
            "count": len(frames),
            "frame_urls": [f"/incidents/{incident_id}/frames/{i}" for i in range(len(frames))],
        }

    @application.post("/demo/upload", response_model=DemoRunResponse)
    async def demo_upload(file: UploadFile) -> DemoRunResponse:
        """Real video ingest: upload a clip → OpenCV decode → CueDetector → ladder.

        Returns 202 immediately with a job id; the CPU-heavy decode+DNN scan
        runs in a worker thread (a 28 MB / 100+ s clip takes minutes on a
        0.5-vCPU Fargate task — far past gateway timeouts — and would block
        the event loop if awaited inline). Poll GET /demo/upload/{job_id} for
        the resulting incident.
        """
        data = await file.read()
        if len(data) > 64 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="clip too large (max 64 MB)")
        suffix = Path(file.filename or "clip.mp4").suffix.lower() or ".mp4"
        if suffix not in {".mp4", ".mov", ".avi", ".m4v", ".webm"}:
            raise HTTPException(status_code=400, detail=f"unsupported type {suffix!r}")

        spilled = save_upload(data, suffix=suffix)
        job_id = uuid4().hex
        _UPLOAD_JOBS[job_id] = {
            "status": "processing",
            "filename": file.filename,
            "incident_id": None,
            "error": None,
        }

        def _process() -> None:
            entry = _UPLOAD_JOBS[job_id]
            try:
                entry["incident_id"] = _run_upload_sync(spilled, application.state.store)
                entry["status"] = "done"
            except HTTPException as exc:
                entry["status"] = "error"
                entry["error"] = exc.detail
            except Exception as exc:  # pragma: no cover - env-dependent
                entry["status"] = "error"
                entry["error"] = f"analysis failed: {exc}"
            finally:
                Path(spilled).unlink(missing_ok=True)

        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _process)
        return DemoRunResponse(incident_id=job_id)

    @application.get("/demo/upload/{job_id}")
    def upload_job_status(job_id: str) -> dict[str, Any]:
        job = _UPLOAD_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job id")
        return job

    def _run_upload_sync(spilled: Path, store: AuditStore) -> str:
        """CPU-heavy upload analysis (runs in executor thread): decode →
        detectors → ladder. Returns incident id; raises HTTPException on
        undecodable/no-cue input."""
        plan = load_care_plan(_DEMO_PLAN_PATH)
        plan.triggers.no_movement.timeout_sec = 2  # demo clock
        detector = CueDetector.from_plan(plan, zone_id="living_room", profile=_demo_profile())
        if _MODEL_PATH.exists():
            from care_ladder.vision.mppersondet import MPPersonDet
            from care_ladder.vision.mppose import MPPose

            detector.person_detector = MPPersonDet(str(_MODEL_PATH), scoreThreshold=0.3)
            if _POSE_MODEL_PATH.exists():
                detector.pose_model = MPPose(str(_POSE_MODEL_PATH), confThreshold=0.5)

        # Clip resolution may differ from plan zone canvas; use full-frame zone.
        probe = cv2.VideoCapture(str(spilled))
        ok, first = probe.read()
        probe.release()
        if not ok:
            raise HTTPException(
                status_code=422,
                detail="clip could not be decoded — is it a valid video file?",
            )
        h, w = first.shape[:2]
        detector.zone = np.asarray(
            [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
        )

        result = ingest_video(spilled, detector, sample_hz=5.0, prefer_distress=True, max_seconds=180.0)
        if result.cue is None and detector.person_detector is not None:
            # DNN found nobody in the whole clip (stylized/low-res footage):
            # fall back to the contour-blob path and re-run once.
            detector.person_detector = None
            detector.pose_model = None
            result = ingest_video(spilled, detector)

        if result.cue is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"no cue emitted from clip ({result.frame_count} frames, "
                    f"{result.duration_sec}s) — try a clip with a still person, "
                    "someone leaving frame, or lying on the floor"
                ),
            )

        result.cue.detail = {
            **result.cue.detail,
            # keep the more specific pose label when the pose path fired
            "source": result.cue.detail.get("source", f"uploaded_clip:{result.detector_source}"),
            "clip_frames": result.frame_count,
            "clip_fps": result.fps,
            "clip_duration_sec": result.duration_sec,
        }
        speaker = SpeakerSimulator(scripted=[])
        dialer = StubDialer(behavior={"caregiver": "no_answer", "secondary": "answered"})
        incident = asyncio.run(
            run_incident(
                cue=result.cue,
                plan=plan,
                speaker=speaker,
                dialer=dialer,
                pre_event_frames=result.pre_event_frames[-4:],
                store=store,
                privacy_mode="blur",
                now=DEMO_NOW,
            )
        )
        asyncio.run(_publish_cloud(incident))
        return incident.id

    @application.post("/demo/camera/start")
    async def camera_start(body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Live camera demo: open a device index or RTSP/HTTP URL, run the same
        CueDetector at ~5 Hz, auto-create an incident on the first cue.

        Guardrails: demo plan only, hard 10-minute cap, one session per process,
        privacy transform before persist. A visible "LIVE" banner is shown in
        the console while a session runs.
        """
        from care_ladder.vision import camera as cam

        body = body or {}
        source = body.get("source", 0)
        if isinstance(source, str) and not source.startswith(("rtsp://", "http://", "https://")):
            raise HTTPException(status_code=400, detail="source must be a device index or rtsp/http URL")

        plan = load_care_plan(_DEMO_PLAN_PATH)
        plan.triggers.no_movement.timeout_sec = min(plan.triggers.no_movement.timeout_sec, 30)
        detector = CueDetector.from_plan(plan, zone_id="living_room", profile=_demo_profile())
        if _MODEL_PATH.exists():
            from care_ladder.vision.mppersondet import MPPersonDet

            detector.person_detector = MPPersonDet(str(_MODEL_PATH), scoreThreshold=0.3)

        async def _on_cue(cue, t: float):
            speaker = SpeakerSimulator(scripted=[])
            dialer = StubDialer(behavior={"caregiver": "no_answer", "secondary": "answered"})
            incident = await run_incident(
                cue=cue,
                plan=plan,
                speaker=speaker,
                dialer=dialer,
                pre_event_frames=[],
                store=application.state.store,
                privacy_mode="silhouette",
                now=datetime.now(timezone.utc),
            )
            await _publish_cloud(incident)

        try:
            session = cam.start_session(
                source=source, detector=detector, on_incident=_on_cue, max_seconds=600.0
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"status": "started", "source": source, "label": session.label}

    @application.post("/demo/camera/stop")
    def camera_stop() -> dict[str, Any]:
        from care_ladder.vision import camera as cam

        stopped = cam.stop_session()
        return {"status": "stopped" if stopped else "not_running"}

    @application.get("/demo/camera/status")
    def camera_status() -> dict[str, Any]:
        from care_ladder.vision import camera as cam

        session = cam.active_session()
        if session is None:
            return {"running": False}
        return session.status()

    @application.post("/demo/run", response_model=DemoRunResponse)
    async def demo_run(body: DemoRunRequest) -> DemoRunResponse:
        if body.fixture not in SUPPORTED_FIXTURES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown fixture {body.fixture!r}; supported: {sorted(SUPPORTED_FIXTURES)}",
            )
        if body.fixture == "quiet_hours_suppressed":
            incident = await _run_quiet_hours_suppressed(application.state.store)
        elif body.fixture == "no_visibility":
            incident = await _run_no_visibility(application.state.store)
        elif body.fixture == "no_movement_ok":
            incident = await _run_no_movement_ok(application.state.store)
        elif body.fixture == "no_movement_silence":
            incident = await _run_no_movement_silence(application.state.store)
        elif body.fixture == "opencv_stillness":
            incident = await _run_opencv_stillness(application.state.store)
        elif body.fixture == "opencv_dnn_person":
            incident = await _run_opencv_dnn_person(application.state.store)
        elif body.fixture == "opencv_pose_person":
            incident = await _run_opencv_pose_person(application.state.store)
        elif body.fixture == "opencv_occlusion":
            incident = await _run_opencv_occlusion(application.state.store)
        elif body.fixture == "speaker_soft_ok":
            incident = await _run_speaker_soft_ok(application.state.store)
        elif body.fixture == "speaker_needs_human":
            incident = await _run_speaker_needs_human(application.state.store)
        elif body.fixture == "path_b_inflight":
            task = asyncio.create_task(
                _run_path_b_inflight(application.state.store)
            )
            tasks = getattr(application.state, "inflight_tasks", None)
            if tasks is None:
                application.state.inflight_tasks = []
                tasks = application.state.inflight_tasks
            tasks.append(task)
            incident = None
            for _ in range(250):
                await asyncio.sleep(0.02)
                opens = [
                    i
                    for i in application.state.store.list_incidents()
                    if i.status == "open"
                    and (i.cue.detail or {}).get("fixture") == "path_b_inflight"
                ]
                if opens:
                    incident = opens[-1]
                    break
            if incident is None:
                raise HTTPException(status_code=500, detail="path_b_inflight failed to open")
            return DemoRunResponse(incident_id=incident.id)
        else:  # pragma: no cover - guarded by SUPPORTED_FIXTURES
            raise HTTPException(status_code=400, detail="unsupported fixture")
        await _publish_cloud(incident)
        return DemoRunResponse(incident_id=incident.id)

    return application


# Default ASGI app for `uvicorn care_ladder.api.app:app`
app = create_app()
