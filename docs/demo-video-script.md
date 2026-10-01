# Care Ladder - Demo Video Production Script

**Target: 4:00-5:00 judge cut.** Hard stop at 5:00. Judging weights: Technical 30 · Innovation 20 · Impact 20 · UX 10 · Docs 10 · Cloud/responsible ops 10.

MUST voiceover is the quoted lines in the shot table only. Read those verbatim. At ~135 wpm, **~600 words** is a **~4:30** dry read, leaving room for clicks, zooms, and breath. Cut-if-over beats are not in that count.

## Record-day checklist

Same day, on the machine you will record. Every box before you arm the recording.

### Bench

- [ ] `cd ~/Documents/GitHub/opencv-care-ladder && ./scripts/run_demo.sh` - local UI at `http://localhost:8000/ui/`. Use local paths for instant clicks.
- [ ] Browser 1440p+, zoom ~125%, full screen, bookmarks hidden. Open **Demo scenarios** (collapsed) so Path A, Path B, and Person DNN are one click away.
- [ ] Tabs ready: (1) local `/ui/`, (2) pre-seeded fall incident, (3) `configs/demo_home.yaml` (zones → triggers → rungs), (4) `docs/eval-metrics.json` (cue_recall). 
- [ ] Mic check + dry read of **only the quoted lines**. Target: **~4:30**.
- [ ] Clock plan: if still on AWS beat at **4:40**, drop all **Cut if over** items and go to the close. The edit must end inside **4:00-5:00**.

### Fall pre-seed (critical)

- [ ] Upload `clips/kul_fall_1.avi` and wait for finish (~2-4 min on M-series).
- [ ] Re-seed on the recording machine. Do not use stale CloudFront IDs.
- [ ] Local UI: `http://localhost:8000/ui/` → upload `clips/kul_fall_1.avi`.
- [ ] Confirm cue is `distress_heuristic` (`sudden_vertical_to_horizontal`).

### Must console beats (practice once, off record)

- [ ] **Path B · silence escalates** → timeline → `jump` row → **Acknowledge** flips to pill and `caregiver_ack` lands on timeline.
- [ ] DNN incident: annotated detection frame visible in telemetry panel.
- [ ] Ladder rail can land on **Emergency, fail-closed, never reached**.

### Say these exactly

- [ ] **1 in 4** adults over sixty-five (shot 1).
- [ ] Cue recall **one-point-zero / 1.0**, **zero false escalations** (shot 3).
- [ ] Torso **seventy degrees / ~70°**, sudden vertical to horizontal (shot 7).

## Shot list + VO

| # | Time | On screen (action) | VO (read verbatim) |
| --- | --- | --- | --- |
| 1 | 0:00-0:20 | Title card / README top, scroll slightly | "Every year, one in four adults over sixty-five falls. Living alone, the dangerous fall is the one nobody sees. Care Ladder is a third path: a camera that checks in before it escalates, and proves every step. The ladder decides what happens next." |
| 2 | 0:20-0:45 | `configs/demo_home.yaml`, then the ladder rail. End on **Emergency, fail-closed, never reached**. | "Everything is one care plan in a YAML file: a living-room zone, a stillness timeout, and the ordered ladder. Look again, ask out loud, wait, then notify and call caregivers. Emergency is fail-closed, off by default, and cannot place real calls. Contacts are reserved fictional numbers. The behavior lives in config, not code." |
| 3 | 0:45-1:15 | Split: left `/ui/`, right `docs/eval-metrics.json`. Recall line in frame. | "OpenCV 5 does the seeing. A MediaPipe person detector runs in OpenCV's DNN module, real ONNX inference. MOG2 background subtraction and Otsu segmentation drive the cues. On ten labeled cases, cue recall is one-point-zero, with zero false escalations on pets or lighting shifts. This is a harness, not a medical claim. Read that recall line in the file. The cue starts every rung that follows." |
| 4 | 1:15-1:50 | `/ui/` → click **"Path A · verbal OK"** → watch timeline build; zoom on land. | "The cue fires. The person hasn't moved. The ladder starts with look again, then the speaker asks: are you okay? I'm fine, and that's it. Resolved at the check-in. No dial, and no emergency. Every step lands in the audit timeline: the cue, the spoken check-in, and the reply. The verdict line says why. Vision changed what the system did next. This is confirm before escalate." |
| 5 | 1:50-2:30 | Click **"Path B · silence escalates"**. Mouse `jump` row, then **Acknowledge**. Button flips to pill; `caregiver_ack` lands on timeline. | "Same cue, same plan, but this time, silence. The ladder keeps walking. The agent dials the primary caregiver, no answer, logged, then the secondary, who answers. Notice the jump: the wait rung was already consumed, so the agent skipped it, and that skip is on the audit trail. The caregiver sees the reasoning, clicks Acknowledge, and that act is on the timeline. A person can still stop the ladder. Same YAML. Different next tool." |
| 6 | 2:30-3:05 | Click **"Person DNN · real photo"**. Point at annotated detection frame, then open pre-event clip panel. | "This is a real photograph through the ONNX detector. A person localized, and the cue tagged with the detector name. Then the privacy contract: the caregiver's clip panel shows only silhouette frames. Blur happens before storage. Raw pixels never leave the device. The caregiver sees a silhouette, not the person in the room. That rule is enforced when the clip is attached. The detection frame is in the telemetry panel." |
| 7 | 3:05-3:55 | Tab 2: **pre-seeded fall incident** (red distress card). Mouse pose chips, then banner, then timeline jump. | "On real footage from the KU Leuven dataset: a genuine fall, a nursing-home re-enactment, not a synthetic blob. Two models chained: person detection, then BlazePose. Torso angle hits seventy degrees, the hip drops, and the transition is sudden, vertical to horizontal in under a second. This signature fires a distress cue, and the ladder escalates immediately, past the verbal rung. Bending over is gradual. It doesn't fire. A slow collapse waits, then escalates on stillness. Falls escalate instantly. Two tiers, both on the record. This is a research re-enactment, not a clinical study." |
| 8 | 3:55-4:20 | Progress bar ticking, then cut to `GET /incidents/{id}` raw JSON. | "The upload API is async. The clip analyzes in the cloud on AWS, about five hertz, while the console stays live. This JSON is the machine-readable truth: cue, chosen rung, tool result, and jump reason. Vision changed every tool call after it. Judges can score that payload. Every agent decision is structured." |
| 9 | 4:20-4:45 | `docs/agentic-workflow.html` or README architecture. | "The stack runs on AWS Fargate, DynamoDB, and S3 for silhouette clips, behind CloudFront. Cues are deliverable EventBridge events, not log lines. GitHub Actions evaluates real footage, so a push to main is a verified deploy. Quiet hours are in the plan. No medical claims. No real phone calls. Failure modes are documented." |
| 10 | 4:45-5:00 | README / repo root; end card with repo URL + live demo URL | "Care Ladder: OpenCV five sees, the ladder decides, and every step is on the record. Confirm before you escalate. The repo and the live demo are on screen." |

## Console v2

### Must

- **Acknowledge after Path B** (shot 5). Click **Acknowledge**. Button flips to pill, `caregiver_ack` lands on timeline.
- **Detection frame** (shot 6). Point at annotated box in telemetry panel.
- **Emergency, fail-closed, never reached** (shot 2). End mouse on never-reached rungs.

### Cut if over

Drop these in order if crossing 5:00 or hitting 4:40 on AWS beat.

1. **Path B · ack mid-flight.** Ack while status is open.
2. **Camera blocked.** Click `opencv_occlusion`.
3. **Four-up explain rail.** Point at Learning schedule badge.
4. **Live CloudWatch / EventBridge peek** on shot 9.

## Cut-in cheatsheet (edit-time saves)

- **No wall-clock analysis**: Cut from "upload started" straight to the pre-seeded incident.
- **Seed check**: Ensure `clips/kul_fall_1.avi` reports `done` with `distress_heuristic`.
- **Zooms**: 150%+ browser zoom reads better than post-zoom.
- **Exact numbers**: recall 1.0 · zero false escalations · torso ~70° · 1-in-4 fall statistic.

## Attribution

- Fall footage: KU Leuven Advise group, "High-quality fall simulation data".
- Models: OpenCV Zoo (MediaPipe person-detection + BlazePose ONNX, Apache-2.0).
- vtest pedestrian clip: OpenCV samples (Apache-2.0).
