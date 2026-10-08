# Care Ladder

[![CI](https://github.com/pamu512/opencv-care-ladder/actions/workflows/ci.yml/badge.svg)](https://github.com/pamu512/opencv-care-ladder/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/demo-live-brightgreen)](https://d2u7pls4da2poz.cloudfront.net/ui/)

Agentic Vision care ladder for the **OpenCV AI Competition 2026** (Agentic Vision track).

OpenCV cues drive a configurable **confirm → smart-speaker check-in → dial escalation** workflow, with privacy blur/silhouette, pre-event clips, and an incident timeline API for judges.

This is a **demo / simulator** stack: telephony is a stub dialer, the smart speaker is a scripted simulator, and there is **no live camera** on the demo path. It does **not** claim clinical diagnosis.

Labeled harness tip: **n=11** cases (`docs/eval-metrics.json`) with cue recall 1.00, no false escalations, mean time-to-confirm **~18 s** (demo-scale, not clinical). `fall_cls` ONNX is demoted: not on the `/ui/` decision path (see section below).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Adaptive schedule learning

The ladder learns the household's usual day from closed incidents
(`RoutineProfile`: rapid -> settled after 10 confirmed-OK days, freezable) and
adapts the stillness timeout with a timeline-visible explain line. This is
schedule-baseline learning only - not machine learning, not clinical, and it
never touches rung order or the fail-closed emergency gate. See
`docs/superpowers/specs/2026-09-27-adaptive-schedule-learning.md`.

## How OpenCV cues change rungs

1. **Vision (`CueDetector`)** watches frames in a plan zone and may emit a structured `CueEvent`:
   - `no_movement` - person-like blob still past `triggers.no_movement.timeout_sec`
   - `no_visibility` - monitored person leaves / cannot be seen in zone
   - `camera_occlusion` - lens covered / unreadable (privacy + camera-health, **not** distress)
   - `distress_heuristic` - simple motion/pose heuristic (not a medical assessment)
2. **Orchestrator (`run_incident`)** starts an incident from that cue and walks `configs/demo_home.yaml` **rungs** in order, appending an audit event per step:
   - `reperceive` → **real re-check**: when the caller passes the `CueDetector` and buffered pre-event frames, a cold copy of the detector re-observes the frames and logs a second `CueEvent` per cue kind that re-fires (`result: reobserved`, `confirmed: true|false` on the audit detail); without them the rung honestly logs `stub_ok` / `no_frames_buffered` (occlusion holds as camera-health either way)
   - `speaker_prompt` → smart-speaker simulator “Are you okay?” (or “clear the lens?” on occlusion)
   - `wait` → listen / settle window (bounded in demo; interruptible by Acknowledge)
   - `notify_caretaker` → family **BotThread** page (Telegram inform card `1|2|3`, or FakeTelegram stub) + console fallback; Acknowledge or Telegram ack stands the ladder down
   - `dial_contact` → stub dial caregiver / secondary
   - `emergency` → **fail-closed** unless `params.enabled: true` (demo YAML keeps it `false`; even if enabled, code audits only and **never** places a real 911 call)
3. **Branching from cue + replies** (what judges see in the timeline):
   - Speaker intent **`clear_ok`** (soft OK like “don’t worry”, or “I’m fine”) → resolve (no dial)
   - Speaker intent **`needs_human`** (help / mixed hurt: never invents OK) → jump to notify / dial
   - Speaker intent **`unclear`** (silence, groan) → continue down the ladder
   - Occlusion + silence → notify on **camera-health** basis, inform caretaker, **no distress claim**, no dial
   - Mid-ladder **Acknowledge** → resolve reason `caregiver_ack`, remaining rungs skipped
   - Dial **`answered`** → resolve
   - Dial **`no_answer`** → escalate to next dial / skip non-dial rungs (logged jumps)

The spoken-reply classifier is **deterministic** (`src/care_ladder/channels/response_intent.py`). An optional LLM path stays **off** unless `CARE_LADDER_INTENT_LLM` is set (and even then the demo uses the rules above).

Demo fixtures:
- `no_movement_ok` - injects a `no_movement` cue with a verbal **"I'm fine"** speaker reply (spec §10 Path A): incident resolves at the check-in rung, no dial, no emergency.
- `no_movement_silence` - injects a `no_movement` cue with an empty speaker script (silence) so the ladder escalates through notify + dial stubs.
- `path_b_inflight` - same cue, left **open** on notify so Acknowledge can stand the ladder down mid-flight (`caregiver_ack`).
- `family_paged_inflight` - silence path left **open** in BotThread `family_paged` so Telegram webhook or console Acknowledge can stop dial.
- `speaker_soft_ok` / `speaker_needs_human` - same stillness cue, different intent (`clear_ok` vs `needs_human`) and therefore different next tools.
- `opencv_occlusion` - **covered-lens frames** through `CueDetector` → `camera_occlusion` → ask to clear the lens → inform caretaker (no distress, no dial). UI label: **No reading · lens covered**.
- `opencv_distress` - **on-floor shape frames** through `CueDetector` → `distress_heuristic` (non-clinical aspect/y heuristic) → silence escalates. This is the fixture pre-seeded on an empty store at cold start.
- `opencv_stillness` - feeds **synthetic numpy frames** through `CueDetector.observe` (OpenCV), then `run_incident`; audit cue is tagged `source: opencv_cue_detector`.
- `opencv_dnn_person` - feeds a **real photo** through the **MediaPipe person-detection ONNX via OpenCV 5 DNN** (`models/`, run `scripts/download_models.sh` once), then the ladder; audit cue is tagged `source: opencv_dnn_person_detector` with detector name. Frames attach as silhouettes.

## Privacy (blur / silhouette)

Before clips or caregiver views leave the device path, use:

- `care_ladder.privacy.blur_faces(frame)` - Gaussian-blur person/face-like ROIs
- `care_ladder.privacy.to_silhouette(frame)` - filled grayscale person mask (not a full-color photo)

**Enforced at attach:** `run_incident(..., pre_event_frames=...)` maps frames through blur (default) or silhouette **before** setting `pre_event_frame_count`. Audit cue `detail` includes `privacy: "blur"` / `"silhouette"`; a non-zero attach count is refused without that flag. Do not ship raw identifiable video in demos.

## Reserved phones & emergency policy

- Demo contacts use **NANP reserved fiction** numbers only: **NPA-555-01XX**  
  (`+12125550101` caregiver Jamie, `+12125550102` secondary Sam in `configs/demo_home.yaml`).
- When `CARE_LADDER_ENV=demo` (default), `load_care_plan` **rejects** non-reserved and emergency-like phones (`911`, `112`, etc.).
- **Never** configure or dial real 911 / real personal numbers in this repo.
- Emergency rung is **disabled by default** (`enabled: false`) - fail-closed in plan and in code.
- **`StubDialer`** returns scripted outcomes only; it does not open a PSTN/VoIP session.

## Quiet hours

`quiet_hours` with `policy: soft_suppress_non_distress` is **enforced** in `run_incident`: non-distress cues (`no_movement`, `no_visibility`) during the window get an audit `suppress` event and status `suppressed` (distress still runs). Demo API fixtures pin a midday clock so judge demos always walk the ladder.

## Judge demo: show the incident timeline

App entrypoint: `care_ladder.api.app:app`

### 1. Start the API

```bash
./scripts/run_demo.sh
# equivalent: uvicorn care_ladder.api.app:app --host 127.0.0.1 --port 8000
```

### 2. Run the demo fixture

```bash
# Injected cue (silence → dial stubs)
curl -s -X POST http://127.0.0.1:8000/demo/run \
  -H 'Content-Type: application/json' \
  -d '{"fixture":"no_movement_silence"}'

# OpenCV path: synthetic frames → CueDetector → ladder
curl -s -X POST http://127.0.0.1:8000/demo/run \
  -H 'Content-Type: application/json' \
  -d '{"fixture":"opencv_stillness"}'
```

Returns `{"incident_id":"<id>"}`.

### 2b. Or use the caregiver console UI

Open **http://127.0.0.1:8000/ui/** - **setup + day archive** console. The **Demo scenarios**
strip opens automatically on a browser's first visit (cold start) and an empty store
pre-seeds one `distress_heuristic` incident so the fall-signature path is visible
immediately; the strip collapses on later visits. A read-only **Family chat (mirror)**
panel shows BotThread state and the last inform card (`GET /family/runtime`, source
`stub|live|demo_fixture`). Acknowledge on an incident card remains a secondary ack
source. Status language is Calm Care-Tech (Care plan active / Checking on Pat /
Camera blocked / Resolved). Uploads state upfront that cloud analysis of a clip takes
minutes.

### Telegram BotThread (optional live)

Family page after speaker silence: `idle → speaker_window → family_paged → pressure → calling_N → closed`.
Without credentials the runtime stays on **FakeTelegram / stub** (CI and CloudFront stay deterministic).

```bash
export TELEGRAM_BOT_TOKEN=...       # opt-in live Bot API
export TELEGRAM_CHAT_ID=...         # family chat
export TELEGRAM_WEBHOOK_SECRET=...  # required once the token is set
```

Webhook: `POST /telegram/webhook` (callback `ack:N` or text `1|2|3`). Unconfigured (no token): `200 {"ok":false,"reason":"telegram_not_configured"}`. The secret header is not required. When `TELEGRAM_BOT_TOKEN` is set, the handler checks `X-Telegram-Bot-Api-Secret-Token` against `TELEGRAM_WEBHOOK_SECRET` before applying an ack or enqueueing the update. A missing secret or a mismatch returns `200 {"ok":false,"applied":false,...}` and does not ack (fail closed, without a non-200 that would make Telegram retry). Vision still picks the next tools; Telegram is confirm-before-escalate, not a replacement for OpenCV cues.

- **Path A: verbal OK** (`no_movement_ok`) - check-in clears, no dial.
- **Path B: silence → escalate** (`no_movement_silence`) - notify then dial stubs, jumps logged.
- **Path B: ack mid-flight** (`path_b_inflight`) - Acknowledge stands the ladder down.
- **Family page · leave open** (`family_paged_inflight`) - BotThread `family_paged`; webhook or Acknowledge stops dial.
- **Camera blocked** (`opencv_occlusion`) - covered lens is camera-health, not distress.
- **OpenCV stillness** (`opencv_stillness`) - synthetic frames through `CueDetector`.

Why this design (cited research): [`docs/research-brief.md`](docs/research-brief.md).
Known failure modes and limitations: [`docs/failure-modes.md`](docs/failure-modes.md).
Competitive landscape: [`docs/competitive-landscape.md`](docs/competitive-landscape.md).

### 3. Fetch the timeline

```bash
# list
curl -s http://127.0.0.1:8000/incidents | python3 -m json.tool

# full ordered audit trail for one incident
curl -s http://127.0.0.1:8000/incidents/<incident_id> | python3 -m json.tool
```

Expect ordered `events` with tools such as `cue` → `reperceive` → `speaker_prompt` → `wait` → `dial_contact` → … → `resolve` (or jumps), **≥3 audit events**. Interactive docs: http://127.0.0.1:8000/docs

## Demo care plan

See `configs/demo_home.yaml`. Emergency rung is **disabled by default** (fail-closed). Phones are reserved fiction (`+1212555010x`). Speaker and dial channels are **simulators/stubs**, not a live smart speaker or carrier dial.

## Tests

```bash
.venv/bin/pytest tests/ -v
```

E2E demo path: `tests/test_e2e_demo.py` (fixture → incident has ≥3 audit events).

## Fall classifier train (Kaggle, this repo only)

`models/fall_cls_v1.onnx` is **not in the `/ui/` decision path**. Nothing in the
serving stack (`CueDetector`, `run_incident`, the API) imports or executes it;
the live fall signature is the MediaPipe pose chain + OpenCV heuristics. The
classifier is a versioned artifact for later sibling consume PRs, and this repo
owns its training. There is no second trainer.

Datasets stay on the Kaggle `fall` + Computer Vision search. Train slug is
`elwalyahmad/fall-detection` after a live license check (CC0 on the 2026-10-04
view API; Autoclaw had noted CC BY 4.0 on-card; both are commercial-permissive).
`uttejkumarkandagatla/fall-detection-dataset` is eval-only until
`datasets/LICENSES.md` quotes a commercial grant. `simuletic/...` (NC-SA) and
`soumicksarker/...` (Other) stay out.

`datasets/` bytes are gitignored. `datasets/LICENSES.md` and
`datasets/manifest.json` (pinned version + metadata sha256) are tracked.

### CI / no credentials (default)

```bash
pip install -e ".[dev,train]"   # train extra is onnx + kaggle CLI; onnx needed to export
python scripts/train_fall_cls.py --fixture
.venv/bin/pytest tests/test_fall_cls.py -v
```

`--fixture` trains the same tiny depthwise CNN on synthetic bars under
`tests/fixtures/fall_cls/`, writes `models/fall_cls_v1.onnx` and
`models/MODEL_CARD.md`. Unit tests mock Kaggle and never download the full set.

### Full Kaggle train (machine with credentials)

```bash
# kaggle.json lives in this directory. Do not copy it into the repo.
export KAGGLE_CONFIG_DIR=$HOME/.kaggle
pip install -e ".[train]"
python scripts/train_fall_cls.py verify-licenses
python scripts/train_fall_cls.py --kaggle
```

`--kaggle` downloads the license-gated train slug (or the next permissive hit
on the same search page) and the eval-only set, does a seeded grouped 70/15/15
split, trains with class weights plus balanced batches and a val threshold,
exports ONNX, and rewrites the model card. 0.85 sensitivity is a report target,
not a ship gate. A run that predicts only one class is refused.

OpenCV DNN loads the ONNX (`cv2.dnn.readNetFromONNX`). Apache-2.0 stack only;
Ultralytics is not a default extra.

## AWS deployment (live)

**Live demo (verified):** https://d2u7pls4da2poz.cloudfront.net/ui/ - CloudFront HTTPS -> ALB -> ECS Fargate (X86_64), DynamoDB incident store, S3 silhouette clips, EventBridge cue bus with a CloudWatch archive rule. CI on every push runs the test suite + real-footage evaluation, builds the amd64 image, pushes to ECR, re-registers the task definition, and rolls the service - a push to `main` is a verified deploy (`scripts/infra.sh` replays the provisioning). Local run via `./scripts/run_demo.sh` needs no AWS credentials.

- [`infra/README.md`](infra/README.md) - architecture and how this meets the "meaningful AWS" bar
- [`infra/ecs-task-outline.md`](infra/ecs-task-outline.md) - task/service outline
- Root [`Dockerfile`](Dockerfile) - builds a runnable API image (`uvicorn care_ladder.api.app:app`)

