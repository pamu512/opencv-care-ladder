# MODEL_CARD fall_cls_v1

Non-clinical edge classifier. OpenCV cues still drive the care ladder; this ONNX is a
versioned artifact for later sibling consume PRs. Not a diagnosis.

## Artifact

- File: `models/fall_cls_v1.onnx`
- SHA256: `4d1a31122ff62d1c9214b0b139fb767430ca69d0b7e441c5e620e48e4f26687a`
- Input: `images` float32 NCHW, 3x32x32, BGR resize then /255
- Output: `scores` softmax `[no_fall, fall]`
- Stack: numpy-trained depthwise-separable CNN, ONNX opset 13, Apache-2.0 (OpenCV DNN)
- Default train code does not import Ultralytics or any AGPL package

## Data

- Family search: https://www.kaggle.com/datasets?search=fall&tags=13207-Computer+Vision
- Train slug (commercial-permissive, verified at download): `elwalyahmad/fall-detection`
- Eval-only slug: `uttejkumarkandagatla/fall-detection-dataset`
- Domain of this card: **fixture**
- Image root: `tests/fixtures/fall_cls`
- Split: Seeded stratified 70/15/15 grouped by video/subject id when those tokens exist in the path.
- Seed: 47
- Counts: train=16 val=8 test=8
- Written: 2026-10-04

## In-domain metrics

Target fall sensitivity: 0.85 (report target, not a hard ship gate).

| split | n | sensitivity | specificity | accuracy | confusion |
| --- | --- | --- | --- | --- | --- |
| train | 16 | 1.000 | 0.375 | 0.688 | tn=3 fp=5 fn=0 tp=8 |
| val | 8 | 1.000 | 0.750 | 0.875 | tn=3 fp=1 fn=0 tp=4 |
| test | 8 | 1.000 | 0.500 | 0.750 | tn=2 fp=2 fn=0 tp=4 |

Sensitivity meets the 0.85 report target on fixture bars. This cloud run had no KAGGLE_CONFIG_DIR, so weights are fixture-trained, not in-domain Kaggle. Specificity can stay low on this tiny set. Re-run with credentials for real in-domain numbers. Not a clinical claim.

## Cross-dataset

Cross-dataset eval (`uttejkumarkandagatla/fall-detection-dataset`) was not run. The set stays eval-only until `datasets/LICENSES.md` quotes a commercial-training grant. When the bytes are local: `python scripts/train_fall_cls.py eval --role eval`.

## License

See `datasets/LICENSES.md`. Dataset bytes are gitignored. Weights in this file come only
from the fixture tree and/or a license-gated Kaggle train slug.

## How to retrain on Kaggle

```bash
export KAGGLE_CONFIG_DIR=$HOME/.kaggle   # directory that holds kaggle.json
pip install -e ".[train]"
python scripts/train_fall_cls.py --kaggle
```
