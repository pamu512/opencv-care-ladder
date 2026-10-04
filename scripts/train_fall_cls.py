#!/usr/bin/env python
"""Train, eval, and export models/fall_cls_v1.onnx from fixtures or Kaggle.

Usage:
    python scripts/train_fall_cls.py --fixture
    python scripts/train_fall_cls.py --kaggle
    python scripts/train_fall_cls.py verify-licenses
    python scripts/train_fall_cls.py download --role train
    python scripts/train_fall_cls.py eval --split test

Requires KAGGLE_CONFIG_DIR only for --kaggle / download. CI uses --fixture.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from care_ladder.fall_cls.data import dedupe_samples, index_image_dir, stratified_group_split  # noqa: E402
from care_ladder.fall_cls.download import download_kaggle_dataset  # noqa: E402
from care_ladder.fall_cls.licenses import (  # noqa: E402
    EVAL_SLUG,
    EXCLUDED_SLUGS,
    FALLBACK_TRAIN_SLUGS,
    SEARCH_URL,
    TRAIN_SLUG,
    assert_train_allowed,
    classify_license,
    fetch_kaggle_view,
    next_train_fallback,
    require_kaggle_config_dir,
)
from care_ladder.fall_cls.model import load_opencv_net  # noqa: E402
from care_ladder.fall_cls.pipeline import (  # noqa: E402
    prepare_kaggle_roots,
    refresh_manifest_file_hashes,
    run_fixture_train,
    run_kaggle_train,
)


def _slug_dest(repo: Path, slug: str) -> Path:
    return repo / "datasets" / slug.replace("/", "-")


def cmd_verify_licenses() -> int:
    print(f"family: {SEARCH_URL}")
    print(f"excluded: {sorted(EXCLUDED_SLUGS)}")
    train_card = fetch_kaggle_view(TRAIN_SLUG)
    print(
        f"train {train_card.slug}: license={train_card.license_name!r} "
        f"v{train_card.version} cv_tag={train_card.has_computer_vision_tag}"
    )
    try:
        assert_train_allowed(train_card)
        print("train gate: allow")
    except PermissionError as exc:
        print(f"train gate: blocked ({exc})")
        fallback_cards = []
        for slug in FALLBACK_TRAIN_SLUGS:
            card = fetch_kaggle_view(slug)
            fallback_cards.append(
                type(card)(
                    **{
                        **card.__dict__,
                        "listed_on_search_page": True,
                    }
                )
            )
            print(f"fallback {card.slug}: license={card.license_name!r}")
        chosen = next_train_fallback(fallback_cards)
        if chosen is None:
            print("no commercial-permissive fallback on the search page")
            return 1
        print(f"next permissive hit: {chosen.slug} ({chosen.license_name})")
    eval_card = fetch_kaggle_view(EVAL_SLUG)
    print(f"eval {eval_card.slug}: license={eval_card.license_name!r} role={classify_license(eval_card.license_name)}")
    for slug in EXCLUDED_SLUGS:
        card = fetch_kaggle_view(slug)
        print(f"exclude {card.slug}: license={card.license_name!r}")
    return 0


def cmd_download(role: str) -> int:
    require_kaggle_config_dir()
    slug = TRAIN_SLUG if role == "train" else EVAL_SLUG
    dest = _slug_dest(ROOT, slug)
    if role == "train":
        try:
            card = download_kaggle_dataset(slug, dest, role="train")
        except PermissionError as exc:
            print(f"primary train slug blocked: {exc}")
            for fb in FALLBACK_TRAIN_SLUGS:
                try:
                    card = fetch_kaggle_view(fb)
                    card = type(card)(**{**card.__dict__, "listed_on_search_page": True})
                    from care_ladder.fall_cls.licenses import assert_train_allowed as _allow

                    _allow(card)
                    dest = _slug_dest(ROOT, fb)
                    card = download_kaggle_dataset(
                        fb,
                        dest,
                        role="train",
                        view_fetch=lambda s, c=card: c,
                    )
                    print(f"downloaded fallback {fb} because primary failed the license gate")
                    break
                except PermissionError:
                    continue
            else:
                raise
    else:
        card = download_kaggle_dataset(slug, dest, role="eval")
    refresh_manifest_file_hashes(ROOT, role, dest)
    print(json.dumps({"slug": card.slug, "license": card.license_name, "dest": str(dest)}, indent=2))
    return 0


def cmd_train(fixture: bool) -> int:
    if fixture:
        result = run_fixture_train(repo=ROOT)
    else:
        require_kaggle_config_dir()
        dest, eval_dest = prepare_kaggle_roots(ROOT, downloader=cmd_download)
        result = run_kaggle_train(dest, repo=ROOT, eval_root=eval_dest if eval_dest.exists() else None)
    print(json.dumps({k: (v.__dict__ if hasattr(v, "__dict__") else v) for k, v in result.items()}, indent=2, default=str))
    return 0


def cmd_eval(split_name: str, role: str) -> int:
    from care_ladder.fall_cls.model import predict_opencv
    import cv2

    if role == "eval":
        root = _slug_dest(ROOT, EVAL_SLUG)
    else:
        root = ROOT / "tests" / "fixtures" / "fall_cls"
        if not root.exists():
            from care_ladder.fall_cls.data import write_fixture_tree

            write_fixture_tree(root)
    samples = dedupe_samples(index_image_dir(root))
    split = stratified_group_split(samples, seed=47)
    chosen = {"train": split.train, "val": split.val, "test": split.test}[split_name]
    net = load_opencv_net(ROOT / "models" / "fall_cls_v1.onnx")
    ys = []
    ps = []
    for sample in chosen:
        img = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        pred, _ = predict_opencv(net, img)
        ys.append(sample.label)
        ps.append(pred)
    import numpy as np

    y = np.asarray(ys)
    p = np.asarray(ps)
    tp = int(((y == 1) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    print(json.dumps({"n": len(chosen), "sensitivity": sens, "confusion": [tn, fp, fn, tp]}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", action="store_true", help="train on tiny repo fixtures (CI / no creds)")
    ap.add_argument("--kaggle", action="store_true", help="download + train the license-gated Kaggle slug")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("verify-licenses")
    p_dl = sub.add_parser("download")
    p_dl.add_argument("--role", choices=("train", "eval"), default="train")
    p_ev = sub.add_parser("eval")
    p_ev.add_argument("--split", choices=("train", "val", "test"), default="test")
    p_ev.add_argument("--role", choices=("fixture", "eval"), default="fixture")
    args = ap.parse_args()
    if args.cmd == "verify-licenses":
        return cmd_verify_licenses()
    if args.cmd == "download":
        return cmd_download(args.role)
    if args.cmd == "eval":
        return cmd_eval(args.split, args.role)
    if args.kaggle:
        return cmd_train(fixture=False)
    return cmd_train(fixture=True)


if __name__ == "__main__":
    raise SystemExit(main())
