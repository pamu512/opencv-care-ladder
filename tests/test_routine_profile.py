"""RoutineProfile: rapid/settled histograms, freeze, explain (no ML)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from care_ladder.learning.profile import (
    LearningConfig,
    RoutineProfile,
    effective_no_movement_timeout_sec,
    explain_schedule,
    freeze_learning,
    mark_settled,
    new_profile,
    record_incident_outcome,
    reset_learning,
)
from care_ladder.learning.store import RoutineProfileStore
from care_ladder.models import CarePlan, Contact, NoMovementTrigger, Triggers


def _plan(timeout_sec: int = 900, learning: LearningConfig | None = None) -> CarePlan:
    kwargs = dict(
        household_id="demo-home-1",
        caregiver=Contact(display_name="Alex"),
        monitored=Contact(display_name="Pat"),
        triggers=Triggers(no_movement=NoMovementTrigger(timeout_sec=timeout_sec)),
        rungs=[],
    )
    if learning is not None:
        kwargs["learning"] = learning
    return CarePlan(**kwargs)


def test_new_profile_starts_rapid_and_unfrozen():
    profile = new_profile("demo-home-1")
    assert profile.subject_key == "demo-home-1"
    assert profile.learning_phase == "rapid"
    assert profile.frozen is False
    assert profile.confirmed_ok_days == 0
    assert profile.settled_after_days == 10
    assert profile.still_hour_hist == [0] * 24
    assert profile.leave_zone_hour_hist == [0] * 24
    assert profile.ok_resolve_hour_hist == [0] * 24


def test_rapid_timeout_uses_factor_and_clamps():
    profile = new_profile("h")
    # 900 * 0.4 = 360, inside [120, 900]
    assert effective_no_movement_timeout_sec(_plan(900), profile) == 360
    # floor: 100 * 0.4 = 40 → clamp to min 120, but also never above plan
    assert effective_no_movement_timeout_sec(_plan(100), profile) == 100
    # tiny plan stays at plan (never above plan timeout)
    assert effective_no_movement_timeout_sec(_plan(60), profile) == 60


def test_record_buckets_cue_hour_and_ok_resolve():
    profile = new_profile("h")
    now = datetime(2026, 9, 11, 7, 10, tzinfo=timezone.utc)
    profile = record_incident_outcome(
        profile,
        cue_kind="no_movement",
        hour=7,
        resolved_ok=True,
        now=now,
        plan=_plan(900),
    )
    assert profile.still_hour_hist[7] == 1
    assert profile.ok_resolve_hour_hist[7] == 1
    assert profile.confirmed_ok_days == 1
    assert profile.usual_still_end_hour == 7


def test_leave_zone_buckets_separately():
    profile = new_profile("h")
    now = datetime(2026, 9, 11, 14, 0, tzinfo=timezone.utc)
    profile = record_incident_outcome(
        profile,
        cue_kind="no_visibility",
        hour=14,
        resolved_ok=False,
        now=now,
        plan=_plan(900),
    )
    assert profile.leave_zone_hour_hist[14] == 1
    assert profile.still_hour_hist[14] == 0
    assert profile.confirmed_ok_days == 0


def test_ok_days_bump_once_per_local_calendar_day():
    profile = new_profile("h")
    day = datetime(2026, 9, 11, 8, 0, tzinfo=timezone.utc)
    later = datetime(2026, 9, 11, 18, 0, tzinfo=timezone.utc)
    nxt = datetime(2026, 9, 12, 8, 0, tzinfo=timezone.utc)
    profile = record_incident_outcome(
        profile, cue_kind="no_movement", hour=8, resolved_ok=True, now=day, plan=_plan()
    )
    profile = record_incident_outcome(
        profile, cue_kind="no_movement", hour=18, resolved_ok=True, now=later, plan=_plan()
    )
    assert profile.confirmed_ok_days == 1
    profile = record_incident_outcome(
        profile, cue_kind="no_movement", hour=8, resolved_ok=True, now=nxt, plan=_plan()
    )
    assert profile.confirmed_ok_days == 2


def test_promotes_to_settled_after_n_ok_days():
    profile = new_profile("h", settled_after_days=2)
    d1 = datetime(2026, 9, 11, 9, 0, tzinfo=timezone.utc)
    d2 = datetime(2026, 9, 12, 9, 0, tzinfo=timezone.utc)
    profile = record_incident_outcome(
        profile, cue_kind="no_movement", hour=9, resolved_ok=True, now=d1, plan=_plan()
    )
    assert profile.learning_phase == "rapid"
    profile = record_incident_outcome(
        profile, cue_kind="no_movement", hour=9, resolved_ok=True, now=d2, plan=_plan()
    )
    assert profile.learning_phase == "settled"
    assert profile.confirmed_ok_days == 2
    # settled uses plan timeout (less sensitive than rapid 360)
    assert effective_no_movement_timeout_sec(_plan(900), profile) == 900


def test_freeze_skips_histogram_updates():
    profile = freeze_learning(new_profile("h"))
    assert profile.frozen is True
    before = profile.model_copy(deep=True)
    profile = record_incident_outcome(
        profile,
        cue_kind="no_movement",
        hour=9,
        resolved_ok=True,
        now=datetime(2026, 9, 11, 9, 0, tzinfo=timezone.utc),
        plan=_plan(),
    )
    assert profile.still_hour_hist == before.still_hour_hist
    assert profile.confirmed_ok_days == 0
    assert profile.learning_phase == "rapid"
    # frozen still serves a timeout
    assert effective_no_movement_timeout_sec(_plan(900), profile) == 360


def test_reset_returns_to_rapid_and_clears_histogram():
    profile = new_profile("h")
    profile = record_incident_outcome(
        profile,
        cue_kind="no_movement",
        hour=9,
        resolved_ok=True,
        now=datetime(2026, 9, 11, 9, 0, tzinfo=timezone.utc),
        plan=_plan(),
    )
    profile = mark_settled(profile)
    profile = reset_learning(profile)
    assert profile.learning_phase == "rapid"
    assert profile.frozen is False
    assert profile.confirmed_ok_days == 0
    assert profile.still_hour_hist == [0] * 24
    assert profile.usual_still_end_hour is None


def test_mark_settled_sets_phase_without_clearing_hist():
    profile = new_profile("h")
    profile = record_incident_outcome(
        profile,
        cue_kind="no_movement",
        hour=9,
        resolved_ok=True,
        now=datetime(2026, 9, 11, 9, 0, tzinfo=timezone.utc),
        plan=_plan(),
    )
    profile = mark_settled(profile)
    assert profile.learning_phase == "settled"
    assert profile.still_hour_hist[9] == 1


def test_explain_string_names_usual_window_and_timeout():
    profile = new_profile("h")
    profile = record_incident_outcome(
        profile,
        cue_kind="no_movement",
        hour=9,
        resolved_ok=True,
        now=datetime(2026, 9, 11, 7, 10, tzinfo=timezone.utc),
        plan=_plan(900),
    )
    text = explain_schedule(
        profile,
        effective_timeout_sec=360,
        still_since=datetime(2026, 9, 11, 7, 10),
    )
    assert "Usual still often ends by 09:00" in text
    assert "today still since 07:10" in text
    assert "timeout 6m" in text
    assert "learning" in text
    assert "risk" not in text.lower()
    assert "911" not in text


def test_json_store_roundtrip(tmp_path: Path):
    store = RoutineProfileStore(tmp_path)
    profile = new_profile("demo-home-1")
    profile = record_incident_outcome(
        profile,
        cue_kind="no_movement",
        hour=8,
        resolved_ok=True,
        now=datetime(2026, 9, 11, 8, 0, tzinfo=timezone.utc),
        plan=_plan(),
    )
    store.save(profile)
    loaded = store.load("demo-home-1")
    assert loaded.still_hour_hist[8] == 1
    assert loaded.confirmed_ok_days == 1
    missing = store.load("brand-new")
    assert missing.learning_phase == "rapid"
    assert missing.subject_key == "brand-new"


def test_learning_never_mentions_emergency_or_risk():
    profile = mark_settled(freeze_learning(new_profile("h")))
    blob = profile.model_dump_json()
    assert "emergency" not in blob
    assert "risk" not in blob
    assert "911" not in blob
