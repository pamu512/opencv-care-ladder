"""Thin Telegram Bot API client plus FakeTelegram for tests.

Live HTTP (httpx2) runs only when a bot token is set. Tests use FakeTelegram
or TelegramAdapter(token=None) and never touch api.telegram.org.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx2 as httpx

from care_ladder.channels.bot import InformCard

_API_ROOT = "https://api.telegram.org"
_CHOICES = frozenset({"1", "2", "3"})


@dataclass(frozen=True)
class AckIntent:
    choice: str
    source: str


class _HttpClient(Protocol):
    def post(self, url: str, json: dict[str, Any] | None = None, **kwargs: Any) -> Any: ...


def _countdown(remaining_sec: int | None) -> str:
    total = max(0, int(remaining_sec or 0))
    minutes, seconds = divmod(total, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _action_label(action: str, monitored_name: str) -> str:
    if action == "1":
        return "1 · I'm on it - I'll call"
    if action == "2":
        return f"2 · Call {monitored_name} now (care ladder dial)"
    if action == "3":
        return "3 · Can't talk - continue to next contact"
    return action


def inform_text(card: InformCard) -> str:
    cue = (card.cue_text or "care").strip().casefold()
    name = card.monitored_name or "Mom"
    lines = [
        f"Care Ladder · {cue} cue",
        f"{name} did not answer the spoken check-in.",
        f"You have {_countdown(card.remaining_sec)} to decide.",
    ]
    for action in card.actions:
        lines.append(_action_label(action, name))
    return "\n".join(lines)


def inform_payload(chat_id: str | int, card: InformCard) -> dict[str, Any]:
    name = card.monitored_name or "Mom"
    keyboard = [
        [{"text": _action_label(action, name), "callback_data": f"ack:{action}"}]
        for action in card.actions
    ]
    return {
        "chat_id": chat_id,
        "text": inform_text(card),
        "reply_markup": {"inline_keyboard": keyboard},
    }


def parse_update(update: dict[str, Any] | None) -> AckIntent | None:
    if not isinstance(update, dict):
        return None
    callback = update.get("callback_query")
    if isinstance(callback, dict):
        data = str(callback.get("data") or "")
        if data.startswith("ack:"):
            choice = data.split(":", 1)[1].strip()
            if choice in _CHOICES:
                return AckIntent(choice=choice, source="telegram_callback")
        return None
    message = update.get("message")
    if isinstance(message, dict):
        text = str(message.get("text") or "").strip()
        if text in _CHOICES:
            return AckIntent(choice=text, source="telegram_text")
    return None


class TelegramAdapter:
    """Bot API send + update parse. Missing token → in-memory stub (no HTTP)."""

    def __init__(
        self,
        token: str | None = None,
        *,
        client: _HttpClient | None = None,
    ) -> None:
        # Explicit token=None stays stub so tests never pick up a host TELEGRAM_BOT_TOKEN.
        self.token = (token or "").strip() or None
        self._client = client
        self.sent: list[dict[str, Any]] = []

    @staticmethod
    def parse_update(update: dict[str, Any] | None) -> AckIntent | None:
        return parse_update(update)

    def send_inform(self, chat_id: str | int, card: InformCard) -> int:
        payload = inform_payload(chat_id, card)
        if self.token is None:
            self.sent.append(payload)
            return len(self.sent)
        return self._send_live(payload)

    def send_pressure_warn(self, chat_id: str | int, remaining_sec: float) -> int:
        total = max(0, int(remaining_sec))
        minutes, seconds = divmod(total, 60)
        text = (
            f"Pressure: still no reply. {minutes:02d}:{seconds:02d} left "
            "before we call the next contact."
        )
        recorded = {"chat_id": chat_id, "text": text, "kind": "pressure"}
        if self.token is None:
            self.sent.append(recorded)
            return len(self.sent)
        return self._send_live({"chat_id": chat_id, "text": text})

    def _send_live(self, payload: dict[str, Any]) -> int:
        url = f"{_API_ROOT}/bot{self.token}/sendMessage"
        client = self._client
        if client is None:
            with httpx.Client(timeout=10.0) as owned:
                return self._post_send_message(owned, url, payload)
        return self._post_send_message(client, url, payload)

    def _post_send_message(
        self, client: _HttpClient, url: str, payload: dict[str, Any]
    ) -> int:
        response = client.post(url, json=payload)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        data = response.json() if hasattr(response, "json") else None
        if not isinstance(data, dict):
            raise RuntimeError("telegram sendMessage returned a non-object body")
        result = data.get("result")
        if not isinstance(result, dict) or "message_id" not in result:
            raise RuntimeError("telegram sendMessage missing result.message_id")
        return int(result["message_id"])


class FakeTelegram(TelegramAdapter):
    """Records send payloads and injected updates. Never opens a socket."""

    def __init__(self) -> None:
        super().__init__(token=None)
        self.inbox: list[dict[str, Any]] = []

    def inject_update(self, update: dict[str, Any]) -> AckIntent | None:
        self.inbox.append(update)
        return self.parse_update(update)
