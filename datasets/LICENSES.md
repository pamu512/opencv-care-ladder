# Dataset licenses (Kaggle fall + Computer Vision)

Family search (only slugs on this page, or tagged Computer Vision, may enter the mix):
https://www.kaggle.com/datasets?search=fall&tags=13207-Computer+Vision

Bytes live under `datasets/` and are gitignored. This file and `datasets/manifest.json`
are the only tracked records. Download uses the Kaggle CLI with `KAGGLE_CONFIG_DIR`
pointing at a directory that contains `kaggle.json`. That credential file is never
copied into the repo.

The live gate (`care_ladder.fall_cls.licenses`) reads the Kaggle dataset-view API
`licenseName` at download time. CC0 and CC BY (without NC or ShareAlike) are
commercial-permissive. NC, ShareAlike, unverified Other, and Open Database
"Contents: Original Authors" are not a train grant.

## TRAIN: elwalyahmad/fall-detection

- Card: https://www.kaggle.com/datasets/elwalyahmad/fall-detection
- Kaggle API `licenseName` verified 2026-10-04: **CC0: Public Domain**
- `currentVersionNumber`: 1
- Computer Vision tag: yes
- Autoclaw cross-check on 2026-10-02 noted **CC BY 4.0 on-card, not CC0**.
  The official view API on 2026-10-04 still reports CC0. Both CC0 and CC BY 4.0
  are commercial-permissive (CC BY needs attribution). If a later download sees
  a non-permissive card, the script refuses this slug and takes the next
  permissive hit on the same search page.
- Role: training + in-domain split (70/15/15, grouped when video/subject ids exist)

## EVAL-ONLY: uttejkumarkandagatla/fall-detection-dataset

- Card: https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset
- Kaggle API `licenseName` verified 2026-10-04: **Database: Open Database, Contents: Original Authors**
- `currentVersionNumber`: 1
- Role: **eval-only**. The Open Database "Original Authors" line is not a
  commercial-training grant. Do not mix these images into shipped weights until
  this file quotes an explicit commercial grant.
- Commercial grant quoted: **none**

## EXCLUDE: simuletic/cctv-incident-dataset-fall-and-lying-down-detection

- Card: https://www.kaggle.com/datasets/simuletic/cctv-incident-dataset-fall-and-lying-down-detection
- Kaggle API / card License field: **CC BY-NC-SA 4.0**
- The dataset description text also says "License: CC BY 4.0" in one paragraph.
  The card License field (NC-SA) controls. Non-commercial + ShareAlike withholds
  commercial use. Stay out of the training mix.

## EXCLUDE: soumicksarker/multiple-cameras-fall-dataset

- Card: https://www.kaggle.com/datasets/soumicksarker/multiple-cameras-fall-dataset
- Kaggle API `licenseName`: **Other (specified in description)**
- Description text on 2026-10-04 does not grant commercial training. Stay out
  until that grant is copied here.

## Fallback order if the primary train card fails the live gate

Recorded so a blocked CC0/CC BY check does not invent a slug from another family:

1. `payutch/fall-video-dataset` (CC0: Public Domain, listed on the mandated search page)
2. `antonygarciag/walker-fall-detection` (CC0: Public Domain, listed on the same page)

If a fallback is used, the download log and this file must say why the primary
failed.

## Stack license

Training code is numpy + OpenCV + ONNX export. OpenCV is Apache-2.0. Default
extras do not install Ultralytics (AGPL-3.0).
