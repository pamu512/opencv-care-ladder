# Kaggle fall classifier (opencv-care-ladder)

Approved plan: ax decide 20261002-081028 (GROK-SIDE). Anoop: implement in Cursor, skip the ax harness.

## Layout

Existing repo owns scripts/, models/, configs/, tests/, src/care_ladder/, pyproject.
No sibling trainer. No product architecture rewrite.

- `src/care_ladder/fall_cls/` library (license gate, index/split, numpy CNN, ONNX)
- `scripts/train_fall_cls.py` CLI
- `configs/fall_cls_v1.yaml` pins
- `datasets/LICENSES.md` + `datasets/manifest.json` tracked; dataset bytes gitignored
- `models/fall_cls_v1.onnx` + `models/MODEL_CARD.md` shipped

## Data

Family: Kaggle search `fall` + Computer Vision tag.
Train: `elwalyahmad/fall-detection` after live `licenseName` check (API 2026-10-04: CC0).
Eval-only: `uttejkumarkandagatla/fall-detection-dataset`.
Exclude: simuletic (NC-SA), soumicksarker (Other).
Download: Kaggle CLI + `KAGGLE_CONFIG_DIR`. Never commit creds or bytes.

## Train

Seeded stratified 70/15/15, grouped by video/subject id when present.
Tiny depthwise-separable CNN, numpy SGD, ONNX opset 13, OpenCV DNN infer.
No Ultralytics/AGPL by default.

## CI

Tests use fixtures and mocked download. They do not need Kaggle credentials
or a full dataset. Full train: `KAGGLE_CONFIG_DIR` + `python scripts/train_fall_cls.py --kaggle`.
