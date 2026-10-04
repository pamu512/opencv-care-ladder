"""Index, preprocess, dedupe, and seeded grouped split for fall frames."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

_FALL_POS = re.compile(r"(?:^|[_\s/-])fall(?:[_\s/-]|detected|$)", re.I)
_FALL_NEG = re.compile(
    r"no[_\s-]?fall|not[_\s-]?fall|walking|sitting|standing|no[_\s-]?fall[_\s-]?detected",
    re.I,
)
_GROUP_ID = re.compile(r"^(subj(?:ect)?|video|cam|person)[-_]?\w+$", re.I)
_FRAME_TAIL = re.compile(r"[_-]?(?:frame|img|f)?[_-]?\d+$", re.I)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    group: str


@dataclass(frozen=True)
class Split:
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]


def infer_label(path: Path) -> int | None:
    parts = "/".join(path.parts).replace("\\", "/")
    if _FALL_NEG.search(parts):
        return 0
    if _FALL_POS.search(parts) or re.search(r"(^|/)fall(/|$)", parts, re.I):
        return 1
    return None


def infer_group(path: Path) -> str:
    id_parts = [part for part in path.parts[:-1] if _GROUP_ID.match(part)]
    if id_parts:
        return "/".join(id_parts)
    parent = path.parent.name or "root"
    stem = _FRAME_TAIL.sub("", path.stem)
    return f"{parent}/{stem or path.stem}"


def index_image_dir(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = infer_label(path)
        if label is None:
            continue
        samples.append(Sample(path=path, label=int(label), group=infer_group(path)))
    return samples


def preprocess_bgr(image: np.ndarray, size: int = 32) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("empty image")
    import cv2

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.clip(chw, 0.0, 1.0)


def dedupe_samples(samples: Sequence[Sample]) -> list[Sample]:
    seen: set[str] = set()
    out: list[Sample] = []
    for sample in samples:
        digest = hashlib.sha256(sample.path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append(sample)
    return out


def stratified_group_split(
    samples: Sequence[object],
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 47,
) -> Split:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    grouped: dict[str, list[object]] = {}
    for sample in samples:
        grouped.setdefault(getattr(sample, "group"), []).append(sample)

    def _group_label(items: list[object]) -> int:
        labels = [int(getattr(item, "label")) for item in items]
        return 1 if sum(labels) * 2 >= len(labels) else 0

    buckets: dict[int, list[str]] = {0: [], 1: []}
    for group, items in grouped.items():
        buckets[_group_label(items)].append(group)
    rng = np.random.default_rng(seed)
    train: list[object] = []
    val: list[object] = []
    test: list[object] = []
    for label in (0, 1):
        names = buckets[label]
        rng.shuffle(names)
        n = len(names)
        n_train = max(int(round(n * ratios[0])), 0 if n == 0 else 1 if n >= 3 else n)
        n_val = int(round(n * ratios[1])) if n >= 3 else 0
        if n_train + n_val >= n and n >= 3:
            n_train = max(1, n - 2)
            n_val = 1
        train_names = names[:n_train]
        val_names = names[n_train : n_train + n_val]
        test_names = names[n_train + n_val :]
        for name in train_names:
            train.extend(grouped[name])
        for name in val_names:
            val.extend(grouped[name])
        for name in test_names:
            test.extend(grouped[name])
    return Split(train=list(train), val=list(val), test=list(test))  # type: ignore[arg-type]


def write_fixture_tree(root: Path, *, n_per_class: int = 16, seed: int = 47, size: int = 48) -> list[Sample]:
    """Synthetic standing (vertical) vs fallen (horizontal) bars. Apache-2.0 fixtures."""
    import cv2

    rng = np.random.default_rng(seed)
    root = Path(root)
    written: list[Sample] = []
    for label, name in ((1, "fall"), (0, "no_fall")):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = _fixture_frame(label, rng, size=size)
            path = folder / f"{name}_{i:03d}.png"
            if not cv2.imwrite(str(path), img):
                raise RuntimeError(f"failed to write fixture {path}")
            written.append(Sample(path=path, label=label, group=f"fx-{name}-{i // 4}"))
    return written


def _fixture_frame(label: int, rng: np.random.Generator, size: int = 48) -> np.ndarray:
    img = np.full((size, size, 3), 18, dtype=np.uint8)
    img = np.clip(img + rng.integers(0, 18, img.shape, dtype=np.uint8), 0, 255).astype(np.uint8)
    color = (
        int(rng.integers(190, 255)),
        int(rng.integers(20, 90)),
        int(rng.integers(20, 90)),
    )
    if label == 1:
        y = int(rng.integers(size // 3, (2 * size) // 3))
        img[max(0, y - 3) : min(size, y + 4), 4 : size - 4] = color
    else:
        x = int(rng.integers(size // 3, (2 * size) // 3))
        img[4 : size - 4, max(0, x - 3) : min(size, x + 4)] = color
    return img
