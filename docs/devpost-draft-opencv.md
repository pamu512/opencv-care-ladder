# Devpost field draft - Care Ladder (OpenCV AI Competition 2026)

**Status:** Paste-ready filing note for Anoop. OpenCV track only. Not an Amazon / Alexa+ Fire TV submission.
**Competition:** https://opencv26.devpost.com/
**Draft already started:** https://devpost.com/submit-to/30984-opencv-ai-competition-2026-powered-by-aws/manage/submissions/1199879-care-ladder-vision/edit
**Deadline:** Oct 26, 2026 @ 11:45pm PDT
**Team (proposal):** NOTATEAM · member Anoop Pamu / pamu512
**Repo tip at draft time:** main ~00a5e95 (adaptive schedule learning landed, CI green)
**Do not invent grant award status.** Compute grant selection is separate from this final package.

No ML hype. No clinical claims. No live 911. Demo/simulator stack.

---

## Official final submission requirements (opencv26.devpost.com)

Every entry must:

1. Use **OpenCV 5** for substantive image or video analysis.
2. Run a **meaningful component on AWS**.

Judges also need:

| Deliverable | Requirement |
| --- | --- |
| Technical report | Problem, users, architecture, OpenCV 5, AWS, evaluation, limitations, responsible use |
| Code repository | Public or private judge-accessible; pinned deps; clear build / deploy / test |
| Architecture diagram | OpenCV 5 + AWS (and agent / COOL if relevant) |
| Live demo | Working web endpoint **or** arranged live screen-share |
| Project video | Public or unlisted, **≤ 5 minutes**; team, app working, architecture, principal results |
| Evaluation evidence | Including failure cases / limitations |

**Optional path claimed here:** **Agentic Vision** (not COOL).

Agentic Vision bar (rules): OpenCV 5 output must change a later plan, tool call, action, or human-approval request. A chatbot that only explains a fixed vision result does not qualify.

Additional Agentic Vision evidence:

- Agent workflow diagram (perception → decision/orchestration → action)
- Trace or demo showing OpenCV 5 output changes a later decision / tool / action
- Evaluation of task success, failure handling, observability, human control

**Overall rubric weights:** Technical 30 · Innovation 20 · Impact 20 · UX 10 · Docs 10 · Cloud/responsible ops 10.

---

## Asset inventory vs gaps (as of 2026-09-27 HKT)

| Asset | Status | Pointer |
| --- | --- | --- |
| Technical report | Ready | `docs/technical-report.md` |
| Code repo + pinned deps + CI | Ready | https://github.com/pamu512/opencv-care-ladder · `requirements-lock.txt` · `pyproject.toml` · GitHub Actions CI |
| Live web endpoint | Ready | https://d2u7pls4da2poz.cloudfront.net/ui/ |
| Agent workflow diagram | Ready (HTML) | `docs/agentic-workflow.html` (screen-capture for gallery) |
| AWS architecture prose | Ready | `infra/README.md` · `infra/ecs-task-outline.md` · `infra/task-definition.json` |
| Eval + failure modes | Ready | `docs/eval-metrics.json` · `docs/failure-modes.md` |
| Demo / video scripts | Ready | `docs/demo-script.md` · `docs/demo-video-script.md` |
| Local mp4 in repo | Present, 300.0s (= 5:00 hard cap) | `docs/demo/care-ladder-demo.mp4` (~4.6 MB) |
| YouTube / Vimeo public or unlisted link | **GAP** | Devpost embeds YouTube/Vimeo only; upload still needed |
| Gallery thumbnail (JPG/PNG/GIF, ~3:2) | **GAP** | Capture from `/ui/` or diagram |
| Static PNG architecture for gallery | Soft gap | HTML diagram exists; export PNG/SVG screenshot recommended |
| Agentic-workflow HTML accuracy | Soft gap | Diagram text says "Fargate · Graviton ARM64"; live demo is **ECS Fargate X86_64**. Fix before gallery shot or caption "design preference; live tip is X86_64" |
| This Devpost draft on main | Was only on closed `ide/adaptive-schedule-learning-c860` | Expanded file to land on main via PR / CloudAgent |
| COOL Award extras | N/A | Not pursuing COOL |
| Grant award claim | Do not invent | Proposal submitted under NOTATEAM; do not claim selected unless confirmed |

---

## Paste-ready Devpost fields

### Project name

```
Care Ladder Vision
```

(Already used on the started draft. Alternate if you rename: `Care Ladder`.)

### Tagline (≤140 chars)

```
OpenCV 5 cues drive a confirm-before-escalate care ladder on AWS, with privacy blur and a full audit timeline.
```

(118 characters.)

### Built with (tags; max 25)

```
OpenCV 5
Python
FastAPI
AWS
Amazon ECS
Amazon Fargate
Amazon DynamoDB
Amazon S3
Amazon EventBridge
Amazon CloudFront
Amazon ECR
Amazon ALB
MediaPipe
ONNX
Docker
GitHub Actions
YAML
```

Optional extras if slots remain: `Uvicorn`, `Pydantic`, `pytest`, `CloudWatch`.

Do **not** tag Alexa / Fire TV / Amazon Developer Hackathon tooling here. OpenCV track only.

### Try it out / links

| Label | URL |
| --- | --- |
| Live demo (caregiver console) | https://d2u7pls4da2poz.cloudfront.net/ui/ |
| Source repository | https://github.com/pamu512/opencv-care-ladder |
| Technical report | https://github.com/pamu512/opencv-care-ladder/blob/main/docs/technical-report.md |
| Agent workflow diagram | https://github.com/pamu512/opencv-care-ladder/blob/main/docs/agentic-workflow.html |
| AWS architecture | https://github.com/pamu512/opencv-care-ladder/blob/main/infra/README.md |
| Failure modes | https://github.com/pamu512/opencv-care-ladder/blob/main/docs/failure-modes.md |
| Eval metrics (JSON) | https://github.com/pamu512/opencv-care-ladder/blob/main/docs/eval-metrics.json |
| Competition page | https://opencv26.devpost.com/ |

Video link: **paste YouTube/Vimeo URL after upload** (repo mp4 alone will not embed).

### Project story (Markdown for Devpost description)

```markdown
## Inspiration

One in four adults over 65 falls each year. For people living alone, the dangerous fall is the one nobody sees. Families often face a false choice: do nothing, or leave a camera feed that feels like surveillance.

Care Ladder is a third path: a camera that **checks in before it escalates**, and proves every step on an audit timeline. Built for the OpenCV AI Competition 2026 (Agentic Vision path), team **NOTATEAM** (Anoop Pamu).

## What it does

OpenCV 5 perception emits structured cues (`no_movement`, `no_visibility`, `camera_occlusion`, `distress_heuristic`). An orchestrator walks a configurable YAML escalation ladder:

1. Re-perceive (confirm the cue; a covered lens holds as camera-health, not distress)
2. Smart-speaker check-in ("Are you okay?" or "Could you clear the lens?")
3. Wait / listen (Acknowledge can stand the ladder down mid-flight)
4. Notify caregiver / family BotThread (Telegram inform card with 1|2|3, or FakeTelegram stub; console Acknowledge is a secondary ack; inform-only on occlusion)
5. Dial primary caregiver (stub)
6. Dial secondary (stub)
7. Emergency rung (**fail-closed** by default; even enabled, code audits only and never places a real 911 call)

**Agentic Vision evidence:** the same stillness cue resolves with no dial on a `clear_ok` reply (Path A, including soft OK like "don't worry"), escalates on silence (Path B), and jumps to a human on `needs_human` (mixed hurt never invents OK). A covered camera is a different cue (`camera_occlusion`) and a different tool path: ask to clear the lens, inform the caretaker, never claim distress. Every jump is logged with `from_index` / `to_index` / `reason` on the caregiver timeline.

**Family Telegram (confirm-before-escalate, not the vision claim):** after speaker silence the family gets a BotThread inform card (inline 1|2|3 or reply `1`/`2`/`3`). Any family or console ack stops dial. OpenCV cues still choose the next tools; Telegram is the family page after the speaker, and without `TELEGRAM_BOT_TOKEN` the demo stays on an honest stub / chat mirror.

Privacy is enforced at attach: blur or silhouette runs **before** any pre-event frame is stored or served. Raw identifiable video never leaves the device path.

**Adaptive schedule learning (honest scope):** the ladder learns a household's usual day from closed-incident histograms (`RoutineProfile`: rapid, then settled after about ten confirmed-OK days, freezable) and adapts the stillness timeout with a timeline-visible explain line. This is schedule-baseline learning only. It is not a neural net, not a risk score, not clinical, and it never auto-enables the emergency rung or changes rung order.

## How we built it

**OpenCV 5 (substantive analysis)**

- Person detection: MediaPipe person-detector ONNX via `cv.dnn` (OpenCV Zoo), with OpenCV 5 portability fixes
- Fall signature: BlazePose ONNX keypoints → torso angle + hip height; **sudden** vertical→horizontal = distress; gradual transitions stay on the stillness ladder (two-tier design)
- Presence / zones: Otsu, contours, `pointPolygonTest`
- Motion: MOG2 background subtractor + absdiff fallback
- Privacy: Gaussian blur / silhouette before persist; `imencode` for caregiver clips

**Agent + AWS**

- FastAPI caregiver console and incident API
- ECS Fargate (live tip: **X86_64**) behind CloudFront + ALB
- DynamoDB incident store, S3 privacy-filtered clips, EventBridge cue bus + archive rule, ECR, CloudWatch Logs
- GitHub Actions: tests + real-footage eval → build amd64 image → ECR → roll service (push to `main` is a verified deploy)

**Demo honesty**

- Demo / simulator stack: stub dialer, scripted speaker, reserved NANP fiction numbers (`NPA-555-01XX`)
- No live camera required for judge fixtures; upload and synthetic paths available
- No medical or accuracy claims beyond the labeled harness

## Challenges we ran into

- OpenCV 5 DNN graph differences vs 4.x wrappers (input API, output blob order, NMS index shape)
- Separating sudden falls from pedestrian bend-overs without false distress cues
- Gradual collapses: pose unreliable on a motionless subject, so stillness timeout must carry the second tier
- Keeping emergency fail-closed in plan **and** code while still showing the full ladder to judges
- Adaptive learning: badge number must equal the actual detector arming threshold (spec section 6)

## Accomplishments that we're proud of

- Labeled eval (with DNN + real clips): cue recall **1.00**, false-escalation rate **0.00**, mean time-to-confirm ~20.5 s (`docs/eval-metrics.json`)
- Real KU Leuven fall re-enactment fires `distress_heuristic` / `sudden_vertical_to_horizontal`; gradual bed collapse escalates via stillness; OpenCV vtest pedestrians do not fire distress
- Live HTTPS demo on AWS with privacy-only clips and a full audit timeline
- Confirm-before-escalate UX that families can understand in one click (Path A / Path B fixtures)

## What we learned

- Agentic Vision is not "LLM narrates a frame." It is: perception changes the next tool or human gate, and the change is observable.
- Fail-closed emergency and reserved demo numbers are features judges can trust, not footnotes.
- Schedule learning helps timeouts stay household-realistic only if it stays explainable and never touches safety gates.

## What's next

- Continuous on-device camera path (demo today is upload / fixture oriented)
- Real telephony behind Secrets Manager with explicit live grant (out of demo scope)
- Optional COOL / Graviton Arm path with measured baselines (not claimed in this entry)
- Broader real-footage eval beyond the current demo-scale set

## Links for judges

- Live demo: https://d2u7pls4da2poz.cloudfront.net/ui/
- Repo: https://github.com/pamu512/opencv-care-ladder
- Technical report: `docs/technical-report.md`
- Agent diagram: `docs/agentic-workflow.html`
- AWS sketch: `infra/README.md`
- Failure modes: `docs/failure-modes.md`
```

### Learning honesty blurb (short; also for gallery caption / About)

```
The ladder learns a household's usual day from audit histograms (rapid, then settled after about ten confirmed-OK days) so stillness timeouts stay explainable on the timeline. Not a neural net, not a risk score, not clinical. It never auto-enables emergency or changes rung order.
```

### Agentic Vision checkbox / path note (if the form asks)

```
Agentic Vision path: yes. COOL path: no.
OpenCV 5 CueEvent output selects and steers ladder tools (reperceive, speaker_prompt, dial_contact, resolve/jump). Same cue, different actions by reply. Trace visible on /incidents/{id} and the /ui/ timeline.
```

### Team bio (if Additional Info asks)

```
NOTATEAM - Anoop Pamu (pamu512). Solo builder. Care Ladder: OpenCV 5 perception + confirm-before-escalate orchestrator on AWS for senior / recovery wellness monitoring. Prior Devpost and competition work under the same handle. Focus: responsible, auditable agentic vision rather than clinical claims.
```

---

## Architecture diagram pointer (for gallery + report)

Primary judge-facing diagram:

- Interactive / SVG: `docs/agentic-workflow.html`
- OpenCV + AWS prose diagram: `infra/README.md` (S3 privacy clips → ECS/Fargate OpenCV+FastAPI → EventBridge cue bus → orchestrator audit → DynamoDB / caregiver `/ui/`)

**Caption suggestion (honest about arch):**

```
Care Ladder agentic workflow: OpenCV 5 perception → YAML ladder decision → AWS action path. Live demo runs on ECS Fargate X86_64 behind CloudFront; silhouette-only clips in S3; cues on EventBridge.
```

---

## Video checklist (human)

1. Record or refresh using `docs/demo-video-script.md` (target 4:30–5:00; **hard max 5:00**).
2. Existing repo file `docs/demo/care-ladder-demo.mp4` is exactly 300s. Confirm content matches current UI (adaptive learning badge) before upload; re-record if stale.
3. Upload to **YouTube (unlisted)** or **Vimeo**; paste URL into Devpost Video field.
4. Show: team intro, Path A, Path B, OpenCV/DNN or fall beat, architecture, principal eval numbers, responsible-use lines.
5. Description attribution: KU Leuven fall clips; OpenCV Zoo models; OpenCV vtest.

---

## Next human steps (submit click)

1. Open draft: https://devpost.com/submit-to/30984-opencv-ai-competition-2026-powered-by-aws/manage/submissions/1199879-care-ladder-vision/edit
2. Paste name, tagline, story, Built with, Try-it-out links from this file.
3. Upload thumbnail + gallery shots (UI Path A/B, silhouette clip panel, architecture screenshot).
4. Upload video to YouTube/Vimeo → paste link.
5. Attach or link technical report / architecture (repo links are enough if form allows URL fields; zip/PDF only if Additional Info requires upload).
6. Confirm Agentic Vision path; do **not** claim COOL; do **not** claim AWS compute grant unless officially selected.
7. Agree to terms → **Submit** (editable until deadline; submit early for optional Devpost eligibility review).
8. Keep Amazon / Alexa+ Fire TV Devpost and Galuxium drafts separate. Do not paste this package there.

---

## Reproduce (for judges / README mirror)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
./scripts/download_models.sh
./scripts/download_clips.sh   # optional real-footage cases
pytest -q
python scripts/evaluate.py --dnn --json docs/eval-metrics.json
./scripts/run_demo.sh         # http://127.0.0.1:8000/ui/
```

Live: https://d2u7pls4da2poz.cloudfront.net/ui/
