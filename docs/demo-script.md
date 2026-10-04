# Care Ladder - Demo Script (~3:30 Telegram-proven cut)

Judging weights: Technical 30 · Innovation 20 · Impact 20 · UX 10 · Docs 10 · Cloud/responsible ops 10.

Setup: `./scripts/run_demo.sh`, local `http://localhost:8000/ui/`, **Demo scenarios** open, `configs/demo_home.yaml` and `docs/eval-metrics.json` in tabs, fall clip pre-seeded the same day. Dry-read the quoted lines in `docs/demo-video-script.md` (~3:20). Hard stop 3:30 (4:00 absolute max).

This table matches that cut. It is not a second story. Telegram prove-or-drop: **PROVE**. Family chat (mirror) stays in-frame on Path B.

| Time | Section | On screen | Say / show |
| --- | --- | --- | --- |
| 0:00-0:18 | Cold open | Title ≤12s, then README or `/ui/` hero | "Every year, one in four adults over sixty-five falls. Living alone, the dangerous fall is the one nobody sees. Care Ladder is a third path: a camera that checks in before it escalates, and proves every step. The ladder decides what happens next." |
| 0:18-0:52 | Plan + Path A | `demo_home.yaml` rungs, then **Path A · verbal OK** | YAML: look again, ask, wait, notify Telegram, emergency fail-closed. Cue → "Are you okay?" → "I'm fine" → **Resolved**, no dial. |
| 0:52-1:48 | Path B · Telegram | **Path B · silence escalates**, then **Family page · leave open**, **Family chat (mirror)** in-frame, **Acknowledge** the open card | Silence → notify (`telegram`, `console`) → inform card (`Pat did not answer the spoken check-in`, 1/2/3). Stub-mode, same card the bot would send. Jump logged. Calls stay reserved-fiction backup. Ack → **acked**, verdict **family or caregiver acknowledged - ladder stood down**, `caregiver_ack`. |
| 1:48-2:22 | DNN + privacy | **Person DNN · real photo** | Real photo, annotated detection frame, silhouette clip. Cue recall 1.0 on ten labeled cases, zero false escalations. Harness, not a medical claim. |
| 2:22-3:12 | Pre-seeded fall | Red distress incident from `clips/kul_fall_1.avi` | KU Leuven nursing-home re-enactment. Torso ~70°, sudden vertical to horizontal. Distress jumps past the verbal rung. Research re-enactment, not a clinical study. |
| 3:12-3:30 | Close | Repo + `https://d2u7pls4da2poz.cloudfront.net/ui/` in chrome | "Care Ladder: OpenCV five sees, the ladder decides, and every step is on the record. Confirm before you escalate. The repo and the live CloudFront demo are on screen." |

## Recording notes

- Record the browser at 1440p+. Zoom the timeline on Path A and the Family chat (mirror) on Path B.
- **No wall-clock analysis.** Use the same-day pre-seeded fall tab. Do not watch the upload tick.
- Path B silence finishes through stub dials and leaves the last Telegram line as the pressure tick. Click **Family page · leave open** so the 1/2/3 inform card is readable, then Acknowledge that open card. Do not ack the already-resolved silence card and claim the ladder stood down.
- Do not narrate Path B as dial-primary then dial-secondary. One honest clause is enough: calls stay reserved-fiction backup.
- Cut if over, in order: Path B ack mid-flight, Camera blocked, Learning badge, long AWS/arch VO, title past ~12s.
- Clip attribution on screen: KU Leuven Advise fall-simulation dataset, nursing-home re-enactments, research use (see `clips/README.md`).
- Keep `docs/failure-modes.md` one click away if judges ask about pets or occlusion.
- Live cite: `https://d2u7pls4da2poz.cloudfront.net/ui/`. Landing: `https://pamu512.github.io/opencv-care-ladder/`. Repo: `https://github.com/pamu512/opencv-care-ladder`.
