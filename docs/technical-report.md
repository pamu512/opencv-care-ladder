# Care Ladder - Technical Report

**OpenCV AI Competition 2026, powered by AWS** · Agentic Vision path · Team: Care Ladder (Anoop Pamu)

## 1. What it is

Why I built this

My mother lives alone in India. I'm in Hong Kong and my sister is in the EU. She refused assisted living and didn't want a full-time caretaker at home, and that's her right.

We put a camera in her home, but it didn't help much. She hated being watched, and neither of us could watch a feed 24/7 from opposite sides of the world. She also often forgets to charge her phone. More than once that ended with panicked calls to neighbors, asking them to check on her because she was out of camera view and not answering.

In the past six months she has fallen four times. Once she was hurt badly enough that she had to drag herself out of the bathroom to reach a phone.

The camera was never the problem. The problem was that nobody could act on what it saw. Care Ladder turns the camera from a feed someone has to watch into something that checks in first. OpenCV notices stillness, a sudden fall, or a covered lens. Before anyone is paged, a voice check-in in the room asks if she's okay, so it works even when her phone is dead. If she says she's fine, it stands down and nobody gets a frightening call. If she's silent, it pages the family chat, and any of us can acknowledge from anywhere. If nobody responds, it moves on to the next person. Every step is logged, so we can see exactly what happened and why. Clips are blurred before they leave the device, because she should never feel watched.

It is not a medical device and it will not call emergency services. It helps a family spread across three time zones know their mum is okay.

Care Ladder is a remote wellness monitor for seniors and recovering people. OpenCV 5
perception emits structured cues (`no_movement`, `no_visibility`, `distress_heuristic`);
an agent walks a **configurable YAML escalation ladder** - re-perceive → smart-speaker
check-in → wait → dial primary → dial secondary → gated emergency - with **every step,
skip, and jump written to an audit trail** surfaced as a caregiver incident timeline.

**Core claim (Agentic Vision bar):** vision output *changes what the system does next*.
The same `no_movement` cue resolves with no dial when the person answers "I'm fine"
(Path A) and escalates through two dial rungs on silence (Path B). A chatbot that
explains a fixed result this is not.

**What it is not:** no clinical or accuracy claims, no real telephony (stub dialer,
reserved NANP fiction numbers), no live camera on the demo path. Failure modes are
documented in [`failure-modes.md`](failure-modes.md).

## 2. OpenCV 5 usage (substantive image/video analysis)

| Component | OpenCV 5 modules | Role |
| --- | --- | --- |
| Person detection | `cv.dnn.readNet` + MediaPipe person-detector ONNX (OpenCV Zoo), `cv.dnn.NMSBoxes`, `cv.resize`, `cv.copyMakeBorder` | Real person localization on frames; three OpenCV 5 portability fixes were required (below) |
| **Pose fall signature** | `cv.dnn.readNet` + MediaPipe BlazePose ONNX (OpenCV Zoo), `cv.dnn.NMSBoxes`, `resize`, `minMaxLoc` | 17 keypoints on the detected person → torso angle + hip height → temporal state machine: **sudden** vertical→horizontal = fall; gradual = not cue-worthy (two-tier, below) |
| Presence / zones | `findContours`, `contourArea`, `moments`, `boundingRect`, `pointPolygonTest`, `THRESH_OTSU` | Otsu adaptive segmentation + contour blobs vs plan-zone polygons |
| Motion | `createBackgroundSubtractorMOG2`, `absdiff` | MOG2 foreground ratio (robust to illumination drift) with frame-diff fallback |
| Morphology | `morphologyEx` (OPEN/CLOSE), `getStructuringElement(MORPH_ELLIPSE)` | Mask cleanup for privacy + blob paths |
| Privacy | `inRange` (YCrCb skin + HSV chroma), `GaussianBlur`, `drawContours(FILLED)` | Face/person blur and silhouette transforms applied **before** any frame is persisted or served |
| Encoding | `imencode` (PNG) | Privacy-transformed pre-event frames to the caregiver console |

**OpenCV 5 portability findings** (vendored wrapper: `src/care_ladder/vision/mppersondet.py`,
adapted from OpenCV Zoo's Apache-2.0 `mp_persondet.py`):
1. `Net.getInputs()` no longer exists in OpenCV 5 - input size must be known (224×224).
2. The new DNN graph engine returns the two output blobs in swapped order vs the 4.x
   wrapper; detected by trailing dimension (scores=1, deltas=12).
3. `NMSBoxes` returns a flattened index array - reshape before fancy-indexing.

## 3. Agentic workflow (perception → decision → action)

```
frames ──▶ CueDetector ──▶ CueEvent ──▶ run_incident (orchestrator)
             │  DNN person box / contour blob         │
             │  BlazePose keypoints → torso/hip        ├─ rung 1: reperceive (re-check)
             │    sudden v→h = FALL (fast path)       │
             │  zone polygon test                      ├─ rung 2: speaker_prompt ── "ok" ─▶ resolve
             │                                         │            └ "call_caregiver" ─▶ jump to dial
             │                                         ├─ rung 3: wait (skipped if listen window consumed - jump logged)
             │                                         ├─ rung 4/5: dial_contact ── answered ─▶ resolve
             │                                         │                └ no_answer ─▶ next dial (jump logged)
             │                                         └─ rung 6: emergency - FAIL-CLOSED (audit-only even if enabled)
             └── privacy transform (blur/silhouette) ──▶ pre-event frames attached to incident
```

Branching is driven by cue kind + channel replies; every jump emits an audit event with
`from_index`/`to_index`/`reason`. Quiet hours (`soft_suppress_non_distress`) suppress
non-distress cues inside the window - suppression itself is audited.

## 4. Evaluation (labeled fixtures, reproducible - honest scope first)

`python scripts/evaluate.py --dnn --json docs/eval-metrics.json`

The harness is **n=10 cases**: synthetic fixtures plus three real-footage cases (one
hard fall, one gradual collapse, one pedestrian negative). That is demo-scale evidence,
not a clinical validation; failure modes are documented in
[`failure-modes.md`](failure-modes.md) and the numbers below should be read with that
sample size in mind.

| Metric | Result |
| --- | --- |
| Cue recall (positive cases) | 1.00 (7/7 on this n=10 set: still person, leaves zone, on-floor, real photo via DNN, **real fall clip**, post-fall stillness, pedestrians-exit) |
| False-escalation rate (negative cases) | 0.00 (3/3: active person, pet motion, illumination ramp; pedestrian bend-overs held silent on real footage) |
| Mean time-to-confirm | 20.5 s across cases (incl. real clips; synthetic-only cases ≈3 s; plan timeouts user-configured) |

A perfect score on ten cases is a smoke test that every path still fires, not a
sensitivity claim. Known failure modes (pets defeating stillness, lighting churn,
blob-not-person) are enumerated in the failure-modes doc and encoded as the negative
cases above.

**Real-footage validation** (clips fetched by `scripts/download_clips.sh`, cases run in CI):

| Clip | Ground truth | System behavior |
| --- | --- | --- |
| KU Leuven `kul_fall_1` (nursing-home fall re-enactment) | hard fall to floor | `distress_heuristic` · `sudden_vertical_to_horizontal` at t=100.8 s |
| KU Leuven `kul_fall_2` | gradual collapse onto bed | pose fast path silent (keypoints unreliable on a motionless subject) → `no_movement` stillness escalation at plan timeout |
| OpenCV `vtest` | pedestrians walking/bending | **no distress cue**; mild `no_visibility` only when people exit frame (correct) |

Two-tier escalation is deliberate design: sudden falls escalate instantly past the
verbal rung; gradual collapses (and deliberate lying down - crouching, bending) are
covered by the stillness ladder rather than the fall signature. Tuning the fall path
on the real clips is what eliminated the pedestrian bend-over false positive.

The harness (`src/care_ladder/vision/eval_cases.py`) synthesizes noisy-room sequences
with textured person stand-ins and sensor-noise + light-drift backgrounds; the negative
cases encode the failure-modes doc as executable tests. Building this harness found and
fixed three real detector bugs (fixed threshold → Otsu; background-slab segmentation →
area rejection; single-frame noise arming `no_visibility` → sustained-presence gate).

## 5. AWS deployment (meaningful component)

| Service | Use | Evidence |
| --- | --- | --- |
| **ECR** | Image registry, scan-on-push | `367597235216.dkr.ecr.us-east-1.amazonaws.com/care-ladder:demo` |
| **ECS Fargate (X86_64)** | Runs OpenCV 5 DNN pair + FastAPI | cluster `care-ladder-demo`, service desired=1, self-healing |
| **DynamoDB** | Durable incident store (audit trail) | table `care-ladder-incidents` |
| **S3** | Pre-event clips (privacy-transformed only) | `care-ladder-demo-367597235216`, public access blocked |
| **EventBridge** | Cue bus (`CareLadderCue`) + archive rule | bus `care-ladder` → `/aws/events/care-ladder-cues` |
| **CloudFront + ALB** | HTTPS front (stable URL) | `https://d2u7pls4da2poz.cloudfront.net` |
| **CloudWatch Logs** | Observability | `/ecs/care-ladder-demo` |

Task definition: `infra/task-definition.json` (0.5 vCPU / 1 GB, awsvpc). CI (GitHub
Actions) runs tests + clip evals, builds the image **natively amd64 with an
architecture gate**, pushes to ECR, re-registers the task def and rolls the service -
a push to `main` is a verified deploy. The full pose fall path runs inside Fargate:
uploading the real fall clip via the public endpoint yields
`distress_heuristic` / `sudden_vertical_to_horizontal` / `source: pose_heuristics`
(~5 min async job at 5 Hz sampling on 0.5 vCPU).

**Fall classifier scope:** `models/fall_cls_v1.onnx` is a Kaggle-trained edge
classifier that is **not wired into the `/ui/` decision path** - nothing in the
serving stack imports it; OpenCV cues and the pose chain drive the ladder. Its
card metrics are honest and modest: in-domain Kaggle test sensitivity 0.664 /
specificity 0.735 (below the 0.85 report target), cross-dataset sensitivity
0.522 (`models/MODEL_CARD.md`). It is versioned for later sibling consume PRs.

## 6. Responsible use

- Emergency rung **fail-closed** by default; even enabled it audits only - code cannot place a real emergency call.
- Demo contacts restricted to NANP reserved fiction (NPA-555-01XX); the plan loader **rejects** other numbers in demo env.
- Privacy transform enforced at attach: a non-zero pre-event frame count without blur/silhouette is refused; the caregiver API serves only transformed copies.
- No identity biometrics; no medical claims; failure modes and demo-vs-real boundaries documented.

## 7. Reproduce

```bash
python3 -m venv .venv && uv pip install -e ".[dev]"   # or pip
./scripts/download_models.sh                            # person-detector + pose ONNX (18 MB)
./scripts/download_clips.sh                             # KU Leuven fall clips + vtest (40 MB)
python -m pytest -q                                     # 70 tests
python scripts/evaluate.py --dnn --json docs/eval-metrics.json   # includes real-footage cases
./scripts/run_demo.sh                                   # local API + /ui/ console
```

Live: AWS Fargate endpoint (URL current as of submission; see README).
