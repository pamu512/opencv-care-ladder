"""Family-channel BotThread FSM after the speaker window.

States: idle | speaker_window | family_paged | pressure | calling_N | closed.
Ack from family_paged or pressure closes. Path A (speaker OK) never pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


class BotState(str, Enum):
    idle = "idle"
    speaker_window = "speaker_window"
    family_paged = "family_paged"
    pressure = "pressure"
    calling_1 = "calling_1"
    calling_2 = "calling_2"
    closed = "closed"


@dataclass(frozen=True)
class InformCard:
    cue_text: str
    actions: tuple[str, ...] = ("1", "2", "3")
    remaining_sec: int | None = None
    monitored_name: str = "Mom"


class BotThread:
    """Deterministic family-page thread. Side-map first; orchestrator wiring is later."""

    def __init__(
        self,
        incident_id: str,
        family_countdown_sec: float,
        pressure_at_remaining_sec: float,
        *,
        now: datetime | None = None,
    ) -> None:
        if family_countdown_sec < 0:
            raise ValueError("family_countdown_sec must be >= 0")
        if pressure_at_remaining_sec < 0:
            raise ValueError("pressure_at_remaining_sec must be >= 0")
        self.incident_id = incident_id
        self.state = BotState.idle
        self.close_reason: str | None = None
        self.inform: InformCard | None = None
        self.started_at = now or datetime.now(timezone.utc)
        self.deadline_at = self.started_at + timedelta(seconds=family_countdown_sec)
        self.pressure_at = self.deadline_at - timedelta(seconds=pressure_at_remaining_sec)
        self._events: list[dict[str, Any]] = []

    def _hop(self, state: BotState, **extra: Any) -> None:
        self.state = state
        detail: dict[str, Any] = {"state": state.value, "incident_id": self.incident_id}
        detail.update(extra)
        self._events.append(
            {
                "tool": "bot",
                "detail": detail,
                "at": datetime.now(timezone.utc),
            }
        )

    def enter_speaker_window(self) -> None:
        self._hop(BotState.speaker_window)

    def enter_family_paged(self, inform: InformCard) -> None:
        self.inform = inform
        self._hop(BotState.family_paged, cue_text=inform.cue_text)

    def enter_pressure(self) -> None:
        self._hop(BotState.pressure)

    def enter_calling(self, n: int) -> None:
        try:
            state = BotState(f"calling_{n}")
        except ValueError as exc:
            raise ValueError(f"unsupported calling index: {n}") from exc
        self._hop(state, n=n)

    def on_cue_handled(self) -> None:
        if self.state is BotState.closed:
            return
        self.close_reason = "speaker_ok"
        self._hop(BotState.closed, reason="speaker_ok")

    def ack(self, source: str, choice: str | None) -> None:
        if self.state is BotState.closed:
            return
        self.close_reason = "family_ack"
        extra: dict[str, Any] = {"reason": "family_ack", "source": source}
        if choice is not None:
            extra["choice"] = choice
        self._hop(BotState.closed, **extra)

    def close(self, reason: str) -> None:
        if self.state is BotState.closed:
            return
        self.close_reason = reason
        self._hop(BotState.closed, reason=reason)

    def audit_events(self) -> list[dict[str, Any]]:
        return list(self._events)


class BotRegistry:
    """Process-local incident_id → BotThread side-map (no Incident model change)."""

    def __init__(self) -> None:
        self._threads: dict[str, BotThread] = {}

    def attach(self, thread: BotThread) -> None:
        self._threads[thread.incident_id] = thread

    def get(self, incident_id: str) -> BotThread | None:
        return self._threads.get(incident_id)

    def open_page(self) -> BotThread | None:
        for thread in reversed(list(self._threads.values())):
            if thread.state in {BotState.family_paged, BotState.pressure}:
                return thread
        return None
