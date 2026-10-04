"""Kaggle CLI download with license verify. Credentials never land in datasets/."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

from care_ladder.fall_cls.licenses import (
    DatasetCard,
    assert_train_allowed,
    classify_license,
    fetch_kaggle_view,
    require_kaggle_config_dir,
)


def download_kaggle_dataset(
    slug: str,
    dest: Path,
    *,
    config_dir: Path | None = None,
    runner: Callable[..., object] | None = None,
    view_fetch: Callable[[str], DatasetCard] | None = None,
    role: str = "train",
) -> DatasetCard:
    cfg = Path(config_dir) if config_dir is not None else require_kaggle_config_dir()
    card = (view_fetch or fetch_kaggle_view)(slug)
    if role == "train":
        card = assert_train_allowed(card)
    elif role == "eval":
        if classify_license(card.license_name) == "exclude":
            raise PermissionError(f"cannot eval on excluded license {card.license_name!r}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        slug,
        "-p",
        str(dest),
        "--unzip",
    ]
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(cfg)
    run = runner or subprocess.run
    result = run(cmd, env=env, check=False)
    code = getattr(result, "returncode", 0)
    if code not in (0, None):
        raise RuntimeError(f"kaggle download failed for {slug} (exit {code})")
    leaked = list(dest.rglob("kaggle.json"))
    for item in leaked:
        item.unlink()
    # ponytail: refuse to keep any file that looks like the Kaggle API token
    for path in dest.rglob("*"):
        if path.is_file() and path.suffix == ".json":
            text = path.read_text(errors="ignore")
            if '"key"' in text and "username" in text and "kaggle" in text.lower():
                path.unlink()
    return card
