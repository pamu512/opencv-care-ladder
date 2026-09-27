"""JSON file store for RoutineProfile (process-local; gitignore data/).

Galuxium SaaS Tasks 1-12 stay separate. When Postgres tenancy lands, replace
this with a ``routine_profiles`` table keyed by ``tenant_id`` (+ optional
``monitored_id``). Same RoutineProfile fields; do not change the API.
"""

from __future__ import annotations

import re
from pathlib import Path

from care_ladder.learning.profile import RoutineProfile, new_profile

_SAFE_KEY = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_subject_key(subject_key: str) -> str:
    cleaned = _SAFE_KEY.sub("_", subject_key.strip()) or "unknown"
    return cleaned[:128]


class RoutineProfileStore:
    """One JSON file per subject under ``data/routine_profiles/``."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else Path("data") / "routine_profiles"

    def _path(self, subject_key: str) -> Path:
        return self.root / f"{_safe_subject_key(subject_key)}.json"

    def load(self, subject_key: str) -> RoutineProfile:
        path = self._path(subject_key)
        if not path.is_file():
            return new_profile(subject_key)
        return RoutineProfile.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, profile: RoutineProfile) -> RoutineProfile:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(profile.subject_key)
        path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        return profile
