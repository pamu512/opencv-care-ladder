"""BotThread FSM: family page hops after the speaker window."""

from datetime import datetime

from care_ladder.channels.bot import BotState, BotThread, InformCard


def test_silence_path_reaches_family_paged_then_ack_closes():
    t = BotThread(incident_id="i1", family_countdown_sec=300, pressure_at_remaining_sec=180)
    t.enter_speaker_window()
    t.enter_family_paged(InformCard(cue_text="Stillness", actions=("1", "2", "3")))
    assert t.state == BotState.family_paged
    t.ack(source="telegram_callback", choice="1")
    assert t.state == BotState.closed
    assert t.close_reason == "family_ack"


def test_silence_path_reaches_pressure_then_calling_1():
    t = BotThread(incident_id="i1", family_countdown_sec=300, pressure_at_remaining_sec=180)
    t.enter_speaker_window()
    t.enter_family_paged(InformCard(cue_text="Stillness", actions=("1", "2", "3")))
    t.enter_pressure()
    assert t.state == BotState.pressure
    t.enter_calling(1)
    assert t.state == BotState.calling_1


def test_ack_from_pressure_closes():
    t = BotThread(incident_id="i1", family_countdown_sec=300, pressure_at_remaining_sec=180)
    t.enter_speaker_window()
    t.enter_family_paged(InformCard(cue_text="Stillness", actions=("1", "2", "3")))
    t.enter_pressure()
    t.ack(source="telegram_text", choice="2")
    assert t.state == BotState.closed
    assert t.close_reason == "family_ack"


def test_path_a_never_enters_family_paged():
    t = BotThread(incident_id="i1", family_countdown_sec=300, pressure_at_remaining_sec=180)
    t.enter_speaker_window()
    assert t.state == BotState.speaker_window
    t.on_cue_handled()
    assert t.state == BotState.closed
    assert t.close_reason == "speaker_ok"
    states = [e["detail"]["state"] for e in t.audit_events()]
    assert "family_paged" not in states
    assert "pressure" not in states


def test_every_hop_audited_with_timestamp():
    t = BotThread(incident_id="i1", family_countdown_sec=300, pressure_at_remaining_sec=180)
    t.enter_speaker_window()
    t.enter_family_paged(InformCard(cue_text="Stillness", actions=("1", "2", "3")))
    t.enter_pressure()
    t.enter_calling(1)
    t.close(reason="exhausted")
    events = t.audit_events()
    assert events
    seen = []
    for event in events:
        assert event["tool"] == "bot"
        assert event["at"] is not None
        assert isinstance(event["at"], datetime)
        seen.append(event["detail"]["state"])
    assert seen == [
        "speaker_window",
        "family_paged",
        "pressure",
        "calling_1",
        "closed",
    ]
    assert t.close_reason == "exhausted"


def test_ack_is_idempotent_once_closed():
    t = BotThread(incident_id="i1", family_countdown_sec=300, pressure_at_remaining_sec=180)
    t.enter_speaker_window()
    t.enter_family_paged(InformCard(cue_text="Stillness", actions=("1", "2", "3")))
    t.ack(source="console", choice=None)
    hops = len(t.audit_events())
    t.ack(source="telegram_callback", choice="3")
    assert t.state == BotState.closed
    assert t.close_reason == "family_ack"
    assert len(t.audit_events()) == hops


def test_thread_exposes_deadlines_and_starts_idle():
    t = BotThread(incident_id="i1", family_countdown_sec=300, pressure_at_remaining_sec=180)
    assert t.state == BotState.idle
    assert t.incident_id == "i1"
    assert t.started_at is not None
    assert t.deadline_at is not None
    assert t.pressure_at is not None
    remaining_to_deadline = (t.deadline_at - t.started_at).total_seconds()
    remaining_at_pressure = (t.deadline_at - t.pressure_at).total_seconds()
    assert remaining_to_deadline == 300
    assert remaining_at_pressure == 180
