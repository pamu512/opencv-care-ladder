"""Fail-closed spoken-reply intent. Deterministic; optional LLM stays off."""

from __future__ import annotations

import os
import re
from typing import Literal

ResponseIntent = Literal["clear_ok", "needs_human", "unclear"]
SpeakerReplyKind = Literal["ok", "call_caregiver", "silence"]

_HURT = (
    "hurt",
    "pain",
    "help",
    "fallen",
    "fell",
    "emergency",
    "need someone",
    "come here",
    "come now",
    "please come",
)
_SOFT_OK = (
    "don't worry",
    "dont worry",
    "do not worry",
    "i'm fine",
    "im fine",
    "i am fine",
    "i'm okay",
    "im okay",
    "i am okay",
    "i'm ok",
    "im ok",
    "all good",
    "all right",
    "alright",
)
_GROAN = frozenset({"ugh", "nngh", "nnngh", "groan", "mmm", "uhh", "uhhh", "hmm", "mm"})
# Affirmations that contain a negation word. Scrub before the "not before ok" scan.
_OK_WITH_NEGATION = ("don't worry", "dont worry", "do not worry")
_NEG_WORDS = frozenset({"not", "no", "nothing", "never"})
_OK_WORDS = frozenset({"ok", "okay", "fine", "alright"})
_TOKEN = re.compile(r"[a-z0-9']+")


def intent_llm_enabled() -> bool:
    """Optional LLM classifier stays off unless explicitly gated."""
    return os.environ.get("CARE_LADDER_INTENT_LLM", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_groan(text: str) -> bool:
    letters = "".join(c for c in text if c.isalpha())
    return text in _GROAN or letters in _GROAN


def _has_okish(text: str) -> bool:
    if any(token in text for token in _SOFT_OK):
        return True
    padded = f" {text} "
    return (
        " ok " in padded
        or " okay " in padded
        or " fine " in padded
        or text in {"ok", "okay", "fine"}
        or "yes i'm" in text
        or "yes i’m" in text
        or "yes i am" in text
    )


def _negation_precedes_ok(text: str) -> bool:
    """True when not/n't/no/nothing/never sits before an OK token.

    "don't worry" / "do not worry" are scrubbed first so the negation inside
    those affirmations does not poison a later "I'm fine".
    """
    scrubbed = text
    for phrase in _OK_WITH_NEGATION:
        scrubbed = scrubbed.replace(phrase, " ")
    tokens = _TOKEN.findall(scrubbed)
    neg = False
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok in _NEG_WORDS or tok.endswith("n't"):
            neg = True
        if tok in _OK_WORDS:
            return neg
        if tok == "all" and i + 1 < n and tokens[i + 1] in {"good", "right"}:
            return neg
        i += 1
    return False


def classify_response_intent(raw: str) -> ResponseIntent:
    """Map a free-form reply to clear_ok / needs_human / unclear.

    Fail-closed rules (case-insensitive, no LLM):
    - empty / groan / non-lexical -> unclear
    - negation of OK ("I'm not ok", "isn't fine") -> unclear (NOT clear_ok)
    - mixed hurt ("I'm ok but I hurt") -> needs_human (never invent OK)
    - call / help / hurt / come -> needs_human
    - soft OK ("don't worry", "do not worry") and plain OK -> clear_ok
    - else -> unclear (treated as silence)
    """
    text = raw.casefold().strip().replace("’", "'").replace("‘", "'")
    if not text or _is_groan(text):
        return "unclear"

    hurt = any(token in text for token in _HURT)
    call = "call" in text
    okish = _has_okish(text)
    if call or hurt:
        return "needs_human"
    if okish and _negation_precedes_ok(text):
        return "unclear"
    if okish:
        return "clear_ok"
    return "unclear"


def intent_to_reply_kind(intent: ResponseIntent) -> SpeakerReplyKind:
    if intent == "clear_ok":
        return "ok"
    if intent == "needs_human":
        return "call_caregiver"
    return "silence"
