"""RoutineProfile model and deterministic rapid → settled rules."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

LearningPhase = Literal["rapid", "settled"]


class LearningConfig(BaseModel):
    """Optional care-plan block. Missing block → these defaults (learning on)."""

    enabled: bool = True
    settled_after_days: int = 10
    min_no_movement_timeout_sec: int = 120
    rapid_timeout_factor: float = 0.4


class RoutineProfile(BaseModel):
    subject_key: str
    learning_phase: LearningPhase = "rapid"
    frozen: bool = False
    confirmed_ok_days: int = 0
    settled_after_days: int = 10
    still_hour_hist: list[int] = Field(default_factory=lambda: [0] * 24)
    leave_zone_hour_hist: list[int] = Field(default_factory=lambda: [0] * 24)
    ok_resolve_hour_hist: list[int] = Field(default_factory=lambda: [0] * 24)
    suggested_no_movement_timeout_sec: int = 360
    usual_still_end_hour: int | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_confirmed_ok_date: date | None = None


def _empty_hist() -> list[int]:
    return [0] * 24


def new_profile(subject_key: str, *, settled_after_days: int = 10) -> RoutineProfile:
    return RoutineProfile(
        subject_key=subject_key,
        settled_after_days=settled_after_days,
    )


def learning_config_from_plan(plan: Any) -> LearningConfig:
    cfg = getattr(plan, "learning", None)
    if cfg is None:
        return LearningConfig()
    if isinstance(cfg, LearningConfig):
        return cfg
    if isinstance(cfg, dict):
        return LearningConfig.model_validate(cfg)
    return LearningConfig.model_validate(cfg)


def _plan_timeout_sec(plan: Any) -> int:
    return int(plan.triggers.no_movement.timeout_sec)


def effective_no_movement_timeout_sec(plan: Any, profile: RoutineProfile) -> int:
    """Adaptive stillness timeout. Never exceeds the plan timeout."""
    plan_timeout = _plan_timeout_sec(plan)
    cfg = learning_config_from_plan(plan)
    if not cfg.enabled:
        return plan_timeout
    if profile.learning_phase == "settled":
        raw = plan_timeout
    else:
        raw = int(plan_timeout * cfg.rapid_timeout_factor)
    return min(plan_timeout, max(cfg.min_no_movement_timeout_sec, raw))


def _usual_still_end_hour(hist: list[int]) -> int | None:
    total = sum(hist)
    if total <= 0:
        return None
    threshold = total * 0.8
    acc = 0
    for hour, count in enumerate(hist):
        acc += count
        if acc >= threshold:
            return hour
    return 23


def _bump_hour(hist: list[int], hour: int) -> list[int]:
    if not 0 <= hour <= 23:
        return hist
    out = list(hist)
    out[hour] += 1
    return out


def record_incident_outcome(
    profile: RoutineProfile,
    *,
    cue_kind: str,
    hour: int,
    resolved_ok: bool,
    now: datetime,
    plan: Any | None = None,
) -> RoutineProfile:
    """Update histograms / OK-day count. Frozen profiles are left unchanged."""
    if profile.frozen:
        return profile

    still = list(profile.still_hour_hist)
    leave = list(profile.leave_zone_hour_hist)
    ok_hist = list(profile.ok_resolve_hour_hist)
    if cue_kind == "no_movement":
        still = _bump_hour(still, hour)
    elif cue_kind == "no_visibility":
        leave = _bump_hour(leave, hour)

    confirmed = profile.confirmed_ok_days
    last_ok = profile.last_confirmed_ok_date
    if resolved_ok:
        ok_hist = _bump_hour(ok_hist, hour)
        local_day = now.date()
        if last_ok != local_day:
            confirmed += 1
            last_ok = local_day

    phase: LearningPhase = profile.learning_phase
    if confirmed >= profile.settled_after_days:
        phase = "settled"

    updated = profile.model_copy(
        update={
            "still_hour_hist": still,
            "leave_zone_hour_hist": leave,
            "ok_resolve_hour_hist": ok_hist,
            "confirmed_ok_days": confirmed,
            "last_confirmed_ok_date": last_ok,
            "learning_phase": phase,
            "usual_still_end_hour": _usual_still_end_hour(still),
            "updated_at": now,
        }
    )
    if plan is not None:
        updated = updated.model_copy(
            update={
                "suggested_no_movement_timeout_sec": effective_no_movement_timeout_sec(
                    plan, updated
                )
            }
        )
    return updated


def freeze_learning(profile: RoutineProfile) -> RoutineProfile:
    return profile.model_copy(update={"frozen": True})


def reset_learning(profile: RoutineProfile) -> RoutineProfile:
    return new_profile(
        profile.subject_key, settled_after_days=profile.settled_after_days
    )


def mark_settled(profile: RoutineProfile) -> RoutineProfile:
    return profile.model_copy(update={"learning_phase": "settled"})


def explain_schedule(
    profile: RoutineProfile,
    *,
    effective_timeout_sec: int,
    still_since: datetime | None = None,
) -> str:
    if profile.usual_still_end_hour is None:
        usual = "Usual still window not mapped yet"
    else:
        usual = f"Usual still often ends by {profile.usual_still_end_hour:02d}:00"
    if still_since is None:
        today = "today still"
    else:
        today = f"today still since {still_since.strftime('%H:%M')}"
    minutes = max(1, round(effective_timeout_sec / 60))
    if profile.frozen:
        phase = "frozen"
    elif profile.learning_phase == "settled":
        phase = "settled"
    else:
        phase = "learning"
    return f"{usual}; {today} · timeout {minutes}m ({phase})"
