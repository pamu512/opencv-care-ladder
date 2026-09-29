"""Telegram adapter + FakeTelegram: inform card send and ack parse. No live network."""

from __future__ import annotations

from care_ladder.channels.bot import InformCard
from care_ladder.channels.telegram_adapter import AckIntent, FakeTelegram, TelegramAdapter


def _stillness_card() -> InformCard:
    return InformCard(cue_text="Stillness", actions=("1", "2", "3"), remaining_sec=300)


def test_fake_telegram_records_inform_text_and_inline_actions():
    fake = FakeTelegram()
    message_id = fake.send_inform(chat_id=123, card=_stillness_card())
    assert isinstance(message_id, int)
    assert fake.sent
    payload = fake.sent[-1]
    text = payload["text"]
    assert "Care Ladder" in text
    assert "stillness cue" in text.lower()
    assert "did not answer the spoken check-in" in text
    assert "05:00" in text
    assert "to decide" in text
    assert "I'm on it" in text
    assert "Call Mom now" in text
    assert "Can't talk" in text
    assert "—" not in text

    keyboard = payload["reply_markup"]["inline_keyboard"]
    callbacks = [btn["callback_data"] for row in keyboard for btn in row]
    assert callbacks == ["ack:1", "ack:2", "ack:3"]
    labels = [btn["text"] for row in keyboard for btn in row]
    assert labels[0].startswith("1")
    assert labels[1].startswith("2")
    assert labels[2].startswith("3")


def test_parse_callback_ack_n():
    intent = TelegramAdapter.parse_update(
        {"callback_query": {"data": "ack:1", "message": {"chat": {"id": 123}}}}
    )
    assert intent == AckIntent(choice="1", source="telegram_callback")


def test_parse_text_digits_1_2_3():
    for choice in ("1", "2", "3"):
        intent = TelegramAdapter.parse_update(
            {"message": {"text": choice, "chat": {"id": 123}}}
        )
        assert intent == AckIntent(choice=choice, source="telegram_text"), choice


def test_parse_unknown_text_is_none():
    assert TelegramAdapter.parse_update({"message": {"text": "hello", "chat": {"id": 1}}}) is None
    assert TelegramAdapter.parse_update({"message": {"text": "ack:1", "chat": {"id": 1}}}) is None
    assert TelegramAdapter.parse_update({}) is None
    assert TelegramAdapter.parse_update({"callback_query": {"data": "noop"}}) is None
    assert TelegramAdapter.parse_update({"callback_query": {"data": "ack:9"}}) is None


def test_fake_inject_update_parses_callback():
    fake = FakeTelegram()
    intent = fake.inject_update({"callback_query": {"data": "ack:3"}})
    assert intent == AckIntent(choice="3", source="telegram_callback")
    assert fake.inbox[-1]["callback_query"]["data"] == "ack:3"


class _BoomClient:
    def post(self, *args, **kwargs):
        raise AssertionError("tests must never hit the network")


def test_adapter_without_token_never_uses_http_client():
    adapter = TelegramAdapter(token=None, client=_BoomClient())
    message_id = adapter.send_inform(chat_id="chat-1", card=_stillness_card())
    assert isinstance(message_id, int)
    assert adapter.sent


def test_live_path_posts_send_message_via_injected_client():
    recorded: list[tuple[str, dict]] = []

    class _Capture:
        def post(self, url, json=None, **kwargs):
            recorded.append((url, json or {}))
            return _FakeResponse({"ok": True, "result": {"message_id": 77}})

    adapter = TelegramAdapter(token="test-token", client=_Capture())
    message_id = adapter.send_inform(chat_id=99, card=_stillness_card())
    assert message_id == 77
    assert recorded
    url, body = recorded[0]
    assert url == "https://api.telegram.org/bottest-token/sendMessage"
    assert body["chat_id"] == 99
    assert body["reply_markup"]["inline_keyboard"]


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload
