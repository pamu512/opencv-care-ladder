"""Fail-closed check-in response intent (deterministic; LLM stays off)."""

from care_ladder.channels.response_intent import (
    classify_response_intent,
    intent_llm_enabled,
    intent_to_reply_kind,
)


def test_soft_ok_dont_worry_is_clear_ok():
    assert classify_response_intent("don't worry") == "clear_ok"
    assert classify_response_intent("Dont worry, I'm fine") == "clear_ok"
    assert intent_to_reply_kind("clear_ok") == "ok"


def test_plain_ok_variants_are_clear_ok():
    for phrase in ("ok", "I'm fine", "yes I'm okay", "all good"):
        assert classify_response_intent(phrase) == "clear_ok", phrase


def test_mixed_hurt_never_invents_ok():
    assert classify_response_intent("I'm ok but I hurt my hip") == "needs_human"
    assert classify_response_intent("fine, but I'm in pain") == "needs_human"
    assert intent_to_reply_kind("needs_human") == "call_caregiver"


def test_explicit_help_is_needs_human():
    for phrase in ("please come", "call Jamie", "yes call", "I need help"):
        assert classify_response_intent(phrase) == "needs_human", phrase


def test_groan_and_empty_stay_unclear():
    assert classify_response_intent("") == "unclear"
    assert classify_response_intent("   ") == "unclear"
    assert classify_response_intent("nnngh") == "unclear"
    assert classify_response_intent("ugh") == "unclear"
    assert classify_response_intent("groan") == "unclear"
    assert intent_to_reply_kind("unclear") == "silence"


def test_intent_llm_off_by_default(monkeypatch):
    monkeypatch.delenv("CARE_LADDER_INTENT_LLM", raising=False)
    assert intent_llm_enabled() is False
    monkeypatch.setenv("CARE_LADDER_INTENT_LLM", "1")
    assert classify_response_intent("don't worry") == "clear_ok"


def test_negation_of_ok_is_not_clear_ok():
    negated = (
        "I'm not ok",
        "im not okay",
        "I'm not fine",
        "no I'm not fine",
        "nothing is ok",
        "I am not okay",
        "isn't ok",
        "not fine",
        "never okay",
        "wasn't fine",
        "I am not fine at all",
        "no ok",
    )
    assert len(negated) >= 8
    for phrase in negated:
        assert classify_response_intent(phrase) != "clear_ok", phrase
    # Negation after the OK token, and soft-OK phrases that contain "not"/"n't".
    for phrase in (
        "I'm ok",
        "I'm fine",
        "don't worry",
        "do not worry",
        "I'm fine, no problem",
        "Dont worry, I'm fine",
    ):
        assert classify_response_intent(phrase) == "clear_ok", phrase
