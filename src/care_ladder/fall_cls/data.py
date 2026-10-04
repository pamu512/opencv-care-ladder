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


def _name_to_label(name: str) -> int | None:
    text = name.lower()
    if _FALL_NEG.search(text) or any(tok in text for tok in ("walk", "sit", "stand")):
        return 0
    if "fall" in text or "lay" in text:
        return 1
    return None


def _parse_yaml_names(text: str) -> list[str] | None:
    names: list[str] = []
    inline = re.search(r"names\s*:\s*\[([^\]]+)\]", text)
    if inline:
        return [part.strip().strip("'\"") for part in inline.group(1).split(",") if part.strip()]
    in_names = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("names:"):
            in_names = True
            rest = line[6:].strip()
            if rest.startswith("[") and rest.endswith("]"):
                return [part.strip().strip("'\"") for part in rest[1:-1].split(",") if part.strip()]
            continue
        if in_names:
            if not line or line.startswith("#"):
                continue
            if re.match(r"^\d+\s*:", line):
                names.append(line.split(":", 1)[1].strip().strip("'\""))
                continue
            if line.startswith("-"):
                names.append(line[1:].strip().strip("'\""))
                continue
            break
    return names or None


def load_class_names(root: Path) -> list[str] | None:
    candidates = [root / "data.yaml", root / "data.yml", *sorted(root.rglob("data.yaml"))]
    for path in candidates:
        if path.is_file():
            parsed = _parse_yaml_names(path.read_text(errors="ignore"))
            if parsed:
                return parsed
    return None


def yolo_label_path(image: Path) -> Path | None:
    sidecar = image.with_suffix(".txt")
    if sidecar.is_file():
        return sidecar
    parts = list(image.parts)
    if "images" in parts:
        idx = parts.index("images")
        alt = Path(*parts[:idx], "labels", *parts[idx + 1 : -1], image.stem + ".txt")
        if alt.is_file():
            return alt
    nested = image.parent.parent / "labels" / image.parent.name / f"{image.stem}.txt"
    if nested.is_file():
        return nested
    return None


def infer_label_from_yolo(path: Path, class_names: list[str] | None = None) -> int | None:
    txt = yolo_label_path(path)
    if txt is None:
        return None
    ids: list[int] = []
    for line in txt.read_text(errors="ignore").splitlines():
        bits = line.split()
        if bits:
            try:
                ids.append(int(float(bits[0])))
            except ValueError:
                continue
    if not ids:
        return None
    if class_names:
        mapped = [_name_to_label(class_names[i]) for i in ids if 0 <= i < len(class_names)]
        mapped = [m for m in mapped if m is not None]
        if not mapped:
            return None
        return 1 if 1 in mapped else 0
    return 1 if 0 in ids else 0


def infer_group(path: Path) -> str:
    id_parts = [part for part in path.parts[:-1] if _GROUP_ID.match(part)]
    if id_parts:
        return "/".join(id_parts)
    parent = path.parent.name or "root"
    stem = _FRAME_TAIL.sub("", path.stem)
    return f"{parent}/{stem or path.stem}"


def index_image_dir(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    names = load_class_names(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = infer_label(path)
        if label is None:
            label = infer_label_from_yolo(path, names)
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
