"""License gate for the mandated Kaggle fall + Computer Vision family."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

LicenseRole = Literal["allow_train", "eval_only", "exclude"]

SEARCH_URL = "https://www.kaggle.com/datasets?search=fall&tags=13207-Computer+Vision"
TRAIN_SLUG = "elwalyahmad/fall-detection"
EVAL_SLUG = "uttejkumarkandagatla/fall-detection-dataset"
EXCLUDED_SLUGS = frozenset(
    {
        "simuletic/cctv-incident-dataset-fall-and-lying-down-detection",
        "soumicksarker/multiple-cameras-fall-dataset",
    }
)

# Next permissive hits on the same search page if the primary card fails the gate.
FALLBACK_TRAIN_SLUGS = (
    "payutch/fall-video-dataset",
    "antonygarciag/walker-fall-detection",
)

_KAGGLE_VIEW = "https://www.kaggle.com/api/v1/datasets/view/{slug}"

_NC = re.compile(r"\bnc\b|non-?commercial", re.I)
_SA = re.compile(r"\bsa\b|share-?alike", re.I)
_CC0 = re.compile(r"\bcc0\b|public domain", re.I)
_CC_BY = re.compile(r"cc\s*by", re.I)
_PERMISSIVE = re.compile(r"\b(mit|apache|bsd)\b", re.I)
_ODB = re.compile(r"open database|odbl|contents:\s*©|data files\s*©|original authors", re.I)
_OTHER = re.compile(r"^other\b|unknown|unspecified", re.I)


@dataclass(frozen=True)
class DatasetCard:
    slug: str
    license_name: str
    version: int | None = None
    kaggle_id: int | None = None
    total_bytes: int | None = None
    last_updated: str | None = None
    has_computer_vision_tag: bool = False
    listed_on_search_page: bool = False
    title: str = ""
    role: str = ""


def classify_license(name: str) -> LicenseRole:
    text = (name or "").strip()
    if not text:
        return "exclude"
    if _ODB.search(text) and not _CC0.search(text):
        return "eval_only"
    if _NC.search(text) or _SA.search(text):
        return "exclude"
    if _OTHER.search(text):
        return "exclude"
    if _CC0.search(text) or _PERMISSIVE.search(text):
        return "allow_train"
    if _CC_BY.search(text):
        return "allow_train"
    return "exclude"


def parse_kaggle_view(payload: dict[str, Any]) -> DatasetCard:
    tags = payload.get("tags") or []
    names: list[str] = []
    for tag in tags:
        if isinstance(tag, dict):
            names.append(str(tag.get("name") or tag.get("ref") or "").lower())
        else:
            names.append(str(tag).lower())
    slug = str(payload.get("ref") or payload.get("ownerRef") or "")
    if payload.get("ownerRef") and payload.get("titleNullable") and "/" not in slug:
        slug = f"{payload.get('ownerRef')}/{slug}"
    version = payload.get("currentVersionNumber")
    if version is None:
        version = payload.get("currentVersionNumberNullable")
    return DatasetCard(
        slug=slug,
        license_name=str(payload.get("licenseName") or payload.get("licenseNameNullable") or ""),
        version=int(version) if version is not None else None,
        kaggle_id=payload.get("id"),
        total_bytes=payload.get("totalBytes") or payload.get("totalBytesNullable"),
        last_updated=payload.get("lastUpdated"),
        has_computer_vision_tag=any("computer vision" in n for n in names),
        title=str(payload.get("title") or payload.get("titleNullable") or ""),
    )


def fetch_kaggle_view(
    slug: str,
    *,
    opener: Callable[[str], Any] | None = None,
) -> DatasetCard:
    url = _KAGGLE_VIEW.format(slug=slug)
    if opener is None:
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    else:
        payload = opener(url)
    return parse_kaggle_view(payload)


def assert_train_allowed(card: DatasetCard) -> DatasetCard:
    if card.slug in EXCLUDED_SLUGS:
        raise PermissionError(f"excluded dataset slug: {card.slug}")
    if card.slug == EVAL_SLUG:
        raise PermissionError(f"eval-only until a commercial grant is quoted: {card.slug}")
    role = classify_license(card.license_name)
    if role == "eval_only":
        raise PermissionError(f"eval-only license {card.license_name!r} on {card.slug}")
    if role != "allow_train":
        raise PermissionError(f"non-permissive license {card.license_name!r} on {card.slug}")
    if not card.has_computer_vision_tag and not card.listed_on_search_page:
        raise PermissionError(f"{card.slug} is outside the fall+Computer Vision family")
    return DatasetCard(
        slug=card.slug,
        license_name=card.license_name,
        version=card.version,
        kaggle_id=card.kaggle_id,
        total_bytes=card.total_bytes,
        last_updated=card.last_updated,
        has_computer_vision_tag=card.has_computer_vision_tag,
        listed_on_search_page=card.listed_on_search_page,
        title=card.title,
        role="train",
    )


def next_train_fallback(cards: Iterable[DatasetCard]) -> DatasetCard | None:
    for card in cards:
        try:
            return assert_train_allowed(card)
        except PermissionError:
            continue
    return None


def require_kaggle_config_dir(*, env: dict[str, str] | None = None) -> Path:
    source = os.environ if env is None else env
    raw = (source.get("KAGGLE_CONFIG_DIR") or "").strip()
    if not raw:
        raise RuntimeError(
            "KAGGLE_CONFIG_DIR is unset. Point it at a directory that contains "
            "kaggle.json (never commit that file). See README Fall classifier train."
        )
    path = Path(raw).expanduser()
    creds = path / "kaggle.json"
    if not creds.is_file():
        raise RuntimeError(f"KAGGLE_CONFIG_DIR={path} has no kaggle.json")
    return path
