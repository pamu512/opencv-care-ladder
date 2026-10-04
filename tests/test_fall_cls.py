"""Fall classifier train/eval/export: no Kaggle credentials required."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from care_ladder.fall_cls.licenses import (
    EVAL_SLUG,
    EXCLUDED_SLUGS,
    TRAIN_SLUG,
    DatasetCard,
    assert_train_allowed,
    classify_license,
    next_train_fallback,
    parse_kaggle_view,
    require_kaggle_config_dir,
)
from care_ladder.fall_cls.data import (
    dedupe_samples,
    infer_group,
    infer_label,
    index_image_dir,
    preprocess_bgr,
    stratified_group_split,
    write_fixture_tree,
)
from care_ladder.fall_cls.manifest import metadata_sha256, pin_record, sha256_tree

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "datasets" / "manifest.json"
LICENSES = REPO / "datasets" / "LICENSES.md"
MODEL = REPO / "models" / "fall_cls_v1.onnx"
CARD = REPO / "models" / "MODEL_CARD.md"
GITIGNORE = REPO / ".gitignore"


def test_cc0_and_cc_by_are_train_ok_nc_sa_is_not():
    assert classify_license("CC0: Public Domain") == "allow_train"
    assert classify_license("CC BY 4.0") == "allow_train"
    assert classify_license("CC BY-NC-SA 4.0") == "exclude"
    assert classify_license("Other (specified in description)") == "exclude"
    assert classify_license("Database: Open Database, Contents: © Original Authors") == "eval_only"


def test_parse_elwaly_card_is_train_family():
    card = parse_kaggle_view(
        {
            "ref": "elwalyahmad/fall-detection",
            "licenseName": "CC0: Public Domain",
            "currentVersionNumber": 1,
            "id": 3755247,
            "totalBytes": 47033585,
            "lastUpdated": "2023-09-19T05:29:47.69Z",
            "tags": [{"name": "computer vision"}],
        }
    )
    assert card.slug == TRAIN_SLUG
    assert card.has_computer_vision_tag is True
    assert_train_allowed(card).role == "train"


def test_uttej_is_eval_only_until_grant():
    card = parse_kaggle_view(
        {
            "ref": EVAL_SLUG,
            "licenseName": "Database: Open Database, Contents: © Original Authors",
            "currentVersionNumber": 1,
            "tags": [{"name": "image"}],
        }
    )
    assert classify_license(card.license_name) == "eval_only"
    with pytest.raises(PermissionError, match="eval-only"):
        assert_train_allowed(card)


def test_excluded_slugs_never_enter_train_mix():
    assert "simuletic/cctv-incident-dataset-fall-and-lying-down-detection" in EXCLUDED_SLUGS
    assert "soumicksarker/multiple-cameras-fall-dataset" in EXCLUDED_SLUGS
    for slug in EXCLUDED_SLUGS:
        card = DatasetCard(slug=slug, license_name="CC0: Public Domain", version=1)
        with pytest.raises(PermissionError, match="excluded"):
            assert_train_allowed(card)


def test_fallback_skips_excluded_and_eval_only():
    cards = [
        DatasetCard(slug=TRAIN_SLUG, license_name="CC BY-NC-SA 4.0", version=1, has_computer_vision_tag=True),
        DatasetCard(slug=EVAL_SLUG, license_name="Database: Open Database, Contents: © Original Authors", version=1),
        DatasetCard(
            slug="simuletic/cctv-incident-dataset-fall-and-lying-down-detection",
            license_name="CC BY-NC-SA 4.0",
            version=1,
            has_computer_vision_tag=True,
        ),
        DatasetCard(
            slug="payutch/fall-video-dataset",
            license_name="CC0: Public Domain",
            version=1,
            has_computer_vision_tag=False,
            listed_on_search_page=True,
        ),
    ]
    chosen = next_train_fallback(cards)
    assert chosen is not None
    assert chosen.slug == "payutch/fall-video-dataset"


def test_download_requires_kaggle_config_dir(monkeypatch):
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    with pytest.raises(RuntimeError, match="KAGGLE_CONFIG_DIR"):
        require_kaggle_config_dir(env={})


def test_download_does_not_copy_credentials_into_datasets(tmp_path, monkeypatch):
    from care_ladder.fall_cls.download import download_kaggle_dataset

    cfg = tmp_path / "kcfg"
    cfg.mkdir()
    (cfg / "kaggle.json").write_text('{"username":"x","key":"secret-test-key"}')
    dest = tmp_path / "datasets" / "elwalyahmad-fall-detection"
    dest.mkdir(parents=True)

    def fake_run(cmd, **kwargs):
        (dest / "Fall Detected").mkdir()
        (dest / "Fall Detected" / "a.jpg").write_bytes(b"jpg")
        class R:
            returncode = 0
        return R()

    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(cfg))
    download_kaggle_dataset(
        TRAIN_SLUG,
        dest,
        config_dir=cfg,
        runner=fake_run,
        view_fetch=lambda slug: parse_kaggle_view(
            {
                "ref": slug,
                "licenseName": "CC0: Public Domain",
                "currentVersionNumber": 1,
                "tags": [{"name": "computer vision"}],
            }
        ),
    )
    leaked = list(dest.rglob("kaggle.json"))
    assert leaked == []
    assert not any("secret-test-key" in p.read_text() for p in dest.rglob("*") if p.is_file() and p.suffix in {".json", ".md", ".txt"})


def test_label_and_group_from_path():
    assert infer_label(Path("train/Fall Detected/clip1_f03.jpg")) == 1
    assert infer_label(Path("train/No Fall Detected/clip1_f03.jpg")) == 0
    assert infer_label(Path("images/walking/a.png")) == 0
    assert infer_group(Path("subj12/video03/frame_001.png")) == "subj12/video03"
    assert infer_group(Path("Fall Detected/img_01.png")).startswith("Fall Detected/")


def test_seeded_stratified_group_split_keeps_video_together(tmp_path):
    samples = []
    for video in ("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"):
        for i, label in enumerate((0, 0, 1, 1)):
            p = tmp_path / video / f"{label}_{i}.png"
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(b"x")
            samples.append(type("S", (), {"path": p, "label": label, "group": video})())
    a = stratified_group_split(samples, ratios=(0.7, 0.15, 0.15), seed=47)
    b = stratified_group_split(samples, ratios=(0.7, 0.15, 0.15), seed=47)
    assert [s.path for s in a.train] == [s.path for s in b.train]
    groups = (
        {s.group for s in a.train},
        {s.group for s in a.val},
        {s.group for s in a.test},
    )
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    n = len(samples)
    assert abs(len(a.train) / n - 0.70) < 0.2


def test_preprocess_resize_normalize_and_dedupe(tmp_path):
    img = np.zeros((80, 40, 3), dtype=np.uint8)
    img[10:70, 15:25] = (20, 200, 40)
    chw = preprocess_bgr(img, size=32)
    assert chw.shape == (3, 32, 32)
    assert 0.0 <= float(chw.min()) and float(chw.max()) <= 1.0
    write_fixture_tree(tmp_path, n_per_class=3, seed=1)
    # clone one file so exact-hash dedupe drops it
    src = next((tmp_path / "fall").glob("*.png"))
    (tmp_path / "fall" / "dup.png").write_bytes(src.read_bytes())
    indexed = index_image_dir(tmp_path)
    deduped = dedupe_samples(indexed)
    assert len(deduped) == len(indexed) - 1


def test_gitignore_keeps_license_pin_and_onnx_exception():
    text = GITIGNORE.read_text()
    assert "datasets/" in text or "datasets/*" in text
    assert "!datasets/LICENSES.md" in text
    assert "!datasets/manifest.json" in text
    assert "models/*.onnx" in text
    assert "!models/fall_cls_v1.onnx" in text
    assert "kaggle.json" in text


def test_committed_license_pin_and_card_exist():
    assert LICENSES.is_file()
    body = LICENSES.read_text()
    assert TRAIN_SLUG in body
    assert EVAL_SLUG in body
    assert "simuletic/" in body
    assert "soumicksarker/" in body
    assert "commercial" in body.lower()
    pin = json.loads(MANIFEST.read_text())
    assert pin["train"]["slug"] == TRAIN_SLUG
    assert pin["train"]["version"] == 1
    assert pin["train"]["metadata_sha256"] == metadata_sha256(pin_record(pin["train"]))
    assert pin["eval"]["slug"] == EVAL_SLUG
    assert pin["eval"]["role"] == "eval-only"
    assert CARD.is_file()
    card = CARD.read_text()
    assert "sensitivity" in card.lower()
    assert "0.85" in card
    assert "elwalyahmad/fall-detection" in card


def test_fixture_train_exports_onnx_that_opencv_loads(tmp_path):
    from care_ladder.fall_cls.model import (
        evaluate,
        export_onnx,
        load_opencv_net,
        predict_opencv,
        train_classifier,
    )

    onnx = pytest.importorskip("onnx")
    _ = onnx
    root = tmp_path / "fx"
    samples = write_fixture_tree(root, n_per_class=16, seed=47)
    model = train_classifier(samples, seed=47, epochs=8, size=32)
    metrics = evaluate(model, samples)
    assert metrics.sensitivity >= 0.85
    out = tmp_path / "fall_cls_v1.onnx"
    export_onnx(model, out)
    assert out.is_file() and out.stat().st_size > 200
    net = load_opencv_net(out)
    fall_img = next(p.path for p in samples if p.label == 1)
    no_img = next(p.path for p in samples if p.label == 0)
    import cv2

    pred_fall, _ = predict_opencv(net, cv2.imread(str(fall_img)), size=32)
    pred_no, _ = predict_opencv(net, cv2.imread(str(no_img)), size=32)
    assert pred_fall == 1
    assert pred_no == 0


def test_shipped_onnx_loads_under_opencv_and_optional_ort():
    pytest.importorskip("cv2")
    assert MODEL.is_file(), "models/fall_cls_v1.onnx must be committed (fixture-trained)"
    from care_ladder.fall_cls.model import load_opencv_net, predict_opencv
    import cv2

    net = load_opencv_net(MODEL)
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    img[20:28, 8:40] = 220  # horizontal bar = fixture fall
    label, probs = predict_opencv(net, img, size=32)
    assert label in (0, 1)
    assert probs.shape == (2,)
    assert abs(float(probs.sum()) - 1.0) < 1e-3

    try:
        import onnxruntime as ort
    except ImportError:
        return
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    x = preprocess_bgr(img, size=32)[None].astype(np.float32)
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: x})[0]
    assert out.shape[-1] == 2


def test_imbalanced_train_is_not_always_one_class(tmp_path):
    """Regression for the Mac Kaggle run: unweighted CE collapsed to always-fall (spec=0)."""
    from care_ladder.fall_cls.model import evaluate, is_collapsed, train_classifier

    all_fx = write_fixture_tree(tmp_path, n_per_class=24, seed=3)
    samples = [s for s in all_fx if s.label == 1] + [s for s in all_fx if s.label == 0][:8]
    assert sum(s.label == 1 for s in samples) == 24
    assert sum(s.label == 0 for s in samples) == 8
    model = train_classifier(samples, seed=47, epochs=12, size=32)
    metrics = evaluate(model, samples)
    assert not is_collapsed(metrics), metrics
    assert metrics.specificity > 0.0
    assert metrics.sensitivity > 0.0
    tn, fp, fn, tp = metrics.confusion
    assert tn + fn > 0  # predicted no_fall at least once
    assert tp + fp > 0  # predicted fall at least once


def test_yolo_txt_labels_when_folders_have_no_class_name(tmp_path):
    import cv2

    img_dir = tmp_path / "images" / "train"
    lab_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    (tmp_path / "data.yaml").write_text("names: [Fall Detected, Walking, Sitting]\n")
    blank = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "a.jpg"), blank)
    cv2.imwrite(str(img_dir / "b.jpg"), blank)
    (lab_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (lab_dir / "b.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    indexed = index_image_dir(tmp_path)
    by_name = {s.path.name: s.label for s in indexed}
    assert by_name["a.jpg"] == 1
    assert by_name["b.jpg"] == 0


def test_kaggle_path_fetches_eval_set_when_config_present(tmp_path, monkeypatch):
    from care_ladder.fall_cls.pipeline import prepare_kaggle_roots

    called: list[str] = []

    def fake_dl(role: str) -> int:
        dest = tmp_path / "datasets" / (
            "elwalyahmad-fall-detection" if role == "train" else "uttejkumarkandagatla-fall-detection-dataset"
        )
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{role}.ok").write_text("1")
        called.append(role)
        return 0

    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "kcfg"))
    (tmp_path / "kcfg").mkdir()
    (tmp_path / "kcfg" / "kaggle.json").write_text("{}")
    train_root, eval_root = prepare_kaggle_roots(tmp_path, downloader=fake_dl)
    assert called == ["train", "eval"]
    assert train_root.exists()
    assert eval_root.exists()


def test_sha256_tree_is_stable(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"abc")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"def")
    first = sha256_tree(tmp_path)
    second = sha256_tree(tmp_path)
    assert first == second
    assert first["a.bin"] == hashlib.sha256(b"abc").hexdigest()
