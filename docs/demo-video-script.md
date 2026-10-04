# Care Ladder - Demo Video Production Script

**Target: ~3:30 Telegram-proven judge cut.** Hard stop at 3:30. Absolute max 4:00. Judging weights: Technical 30 · Innovation 20 · Impact 20 · UX 10 · Docs 10 · Cloud/responsible ops 10.

MUST voiceover is the quoted lines in the shot table only. Read those verbatim. At ~135 wpm, **~460 words** is a **~3:25** dry read, leaving room for clicks, zooms, and breath inside 3:30. Cut-if-over beats are not in that count.

Telegram prove-or-drop: **PROVE**. Path B is not in the cut unless **Family chat (mirror)** is in-frame with inform card text.

## Record-day checklist

Same day, on the machine you will record. Every box before you arm the recording.

### Bench

- [ ] `cd ~/Documents/GitHub/opencv-care-ladder && ./scripts/run_demo.sh` - local UI at `http://localhost:8000/ui/`. Use local paths for instant clicks.
- [ ] Browser 1440p+, zoom ~125%, full screen, bookmarks hidden. Open **Demo scenarios** so Path A, Path B, Family page, and Person DNN are one click away.
- [ ] Tabs ready: (1) local `/ui/`, (2) pre-seeded fall incident, (3) `configs/demo_home.yaml` (zones → triggers → rungs, `notify_caretaker` with `channels: [telegram, console]` before dial), (4) `docs/eval-metrics.json` (cue_recall).
- [ ] Mic check + dry read of **only the quoted lines**. Target: **~3:20**.
- [ ] Clock plan: if still on the fall beat at **3:15**, skip every **Cut if over** item and go to the close. The edit must end inside **~3:30** (4:00 absolute max).

### Fall pre-seed (critical)

- [ ] Upload `clips/kul_fall_1.avi` and wait for finish (~2-4 min on M-series).
- [ ] Re-seed on the recording machine. Do not use stale CloudFront IDs.
- [ ] Local UI: `http://localhost:8000/ui/` → upload `clips/kul_fall_1.avi`.
- [ ] Confirm cue is `distress_heuristic` (`sudden_vertical_to_horizontal`).

### Must console beats (practice once, off record)

- [ ] **Path A · verbal OK** → timeline → **Resolved**, no Dial row, resolve reason `speaker_ok`.
- [ ] **Path B · silence escalates** → **Family page · leave open** → **Family chat (mirror)** shows the inform card → **Acknowledge** on the *open* Family page card (not the already-resolved silence card).
- [ ] DNN incident: annotated detection frame in **Vision telemetry**, silhouette in **Pre-event clip**.
- [ ] Pre-seeded fall: red distress card, banner **Fall signature detected**.

### Say these exactly

- [ ] **1 in 4** adults over sixty-five (shot 1).
- [ ] Cue recall **one-point-zero / 1.0**, **ten labeled cases**, **zero false escalations** (shot 4).
- [ ] Torso **seventy degrees / ~70°**, sudden vertical to horizontal (shot 5).
- [ ] **research re-enactment, not a clinical study** (shot 5).

## Shot list + VO

| # | Time | On screen (action) | VO (read verbatim) |
| --- | --- | --- | --- |
| 1 | 0:00-0:18 | Title card ≤12s, then README top or `/ui/` hero. Do not linger on the title. | "Every year, one in four adults over sixty-five falls. Living alone, the dangerous fall is the one nobody sees. Care Ladder is a third path: a camera that checks in before it escalates, and proves every step. The ladder decides what happens next." |
| 2 | 0:18-0:52 | Tab 3: `configs/demo_home.yaml` (zones, then rungs: spoken check-in, wait, `notify_caretaker` channels telegram+console, dials after, emergency fail-closed). Cut to `/ui/` → click **Path A · verbal OK**. Zoom timeline: Spoken check-in “I'm fine” / `clear_ok` → Resolve. No Dial. Status **Resolved**. | "Everything is one care plan in a YAML file: a living-room zone, a stillness timeout, and the ordered ladder. Look again, ask out loud, wait, then notify the family on Telegram. Emergency is fail-closed, off by default. Contacts are reserved fiction. The cue fires. Stillness in the zone. The speaker asks: are you okay? I'm fine, and that's it. Resolved at the check-in. No dial, and no emergency. Every step lands in the audit timeline. The verdict line says why. Vision changed what the system did next." |
| 3 | 0:52-1:48 | **Family chat (mirror) stays in-frame for this whole beat.** Click **Path B · silence escalates**. Timeline: Spoken check-in `unclear` / silence → **Skip (logged)** `listen_window_already_consumed` → **Notify caregiver** (`telegram`, `console`). Do not linger on Dial rows; the fixture still walks stub dials after notify. Immediately click **Family page · leave open** so the mirror holds the inform card: `Care Ladder · stillness cue` / `Pat did not answer the spoken check-in` / `1 · I'm on it` / `2 · Call Pat now` / `3 · Can't talk`. Source pill is `stub` or `demo_fixture`. Click **Acknowledge** on that *open* card (status was **Care plan active**). Button flips to **acked**. Verdict: **family or caregiver acknowledged - ladder stood down**. Timeline: Caregiver note `caregiver_ack`, Family chat close `family_ack` (source console), resolve `caregiver_ack`. | "Same cue, same plan, but this time, silence. The ladder keeps walking. After the spoken check-in, it pages the family. Keep Family chat, the mirror, in frame. That text is the Telegram inform card: Pat did not answer the spoken check-in. Reply one, two, or three. This console is stub-mode. Same card the bot would send. Notify caregiver sits on the plan before any dial. Notice the jump: the wait rung was already consumed, and that skip is on the audit trail. Calls stay reserved-fiction backup. The caregiver clicks Acknowledge. The button becomes acked. The verdict says family or caregiver acknowledged, ladder stood down, and caregiver_ack is on the timeline. A person can still answer the page. Same YAML. Different next tool." |
| 4 | 1:48-2:22 | Click **Person DNN · real photo**. Point at annotated detection frame in Vision telemetry, then Pre-event clip (privacy: silhouette). Optional flash of `docs/eval-metrics.json` recall line. | "This is a real photograph through the ONNX detector. A person localized, and the cue tagged with the detector name. Then the privacy contract: the caregiver's clip panel shows only silhouette frames. Blur happens before storage. Raw pixels never leave the device. On ten labeled cases, cue recall is one-point-zero, with zero false escalations on pets or lighting shifts. This is a harness, not a medical claim. OpenCV 5 does the seeing. The detection frame is in the telemetry panel." |
| 5 | 2:22-3:12 | Tab 2: **pre-seeded fall incident** (red distress card). Mouse pose chips (torso ~70°), banner **Fall signature detected**, then timeline jump past the verbal rung. On-screen attribution: KU Leuven Advise, nursing-home re-enactment, research use, not clinical. | "On real footage from the KU Leuven dataset: a genuine fall, a nursing-home re-enactment, not a synthetic blob. Two models chained: person detection, then BlazePose. Torso angle hits seventy degrees, the hip drops, and the transition is sudden, vertical to horizontal in under a second. The red card says Fall signature detected. This cue fires distress, and the ladder escalates immediately, past the verbal rung. Bending over is gradual. It doesn't fire. A slow collapse waits, then escalates on stillness. Falls escalate instantly. Two tiers, both on the record. This is a research re-enactment, not a clinical study." |
| 6 | 3:12-3:30 | Chrome shows `https://github.com/pamu512/opencv-care-ladder` and live `https://d2u7pls4da2poz.cloudfront.net/ui/`. End card may repeat both URLs. | "Care Ladder: OpenCV five sees, the ladder decides, and every step is on the record. Confirm before you escalate. The repo and the live CloudFront demo are on screen." |

## Console v2

### Must

- **Family chat (mirror) on Path B** (shot 3). Inform card text readable. Telegram prove-or-drop is PROVE.
- **Acknowledge the open Family page card** (shot 3). Button flips to **acked**. Verdict uses **family or caregiver acknowledged - ladder stood down**. Do not Acknowledge the already-resolved Path B silence card and claim the ladder stood down. That silence fixture already resolved `dial_answered`.
- **Path A, no dial** (shot 2). Status **Resolved**, resolve reason `speaker_ok`.
- **Detection frame + silhouette** (shot 4). Point at both.
- **Fall signature** (shot 5). Red distress card, KU Leuven honesty on screen.

### Cut if over

Drop these in order if crossing 3:30 or hitting 3:15 still on the fall beat.

1. **Path B · ack mid-flight.** Extra open-notify click. Shot 3 already uses Family page + Acknowledge.
2. **Camera blocked.** Click `opencv_occlusion`.
3. **Learning schedule badge.**
4. **Long AWS / arch VO.** No Fargate, EventBridge, or CloudWatch narration in this cut.
5. **Title past ~12s.**

Do not spend spoken time on `models/fall_cls_v1.onnx`. OpenCV cues still drive the ladder.

## Cut-in cheatsheet (edit-time saves)

- **No wall-clock analysis**: Cut from "upload started" straight to the pre-seeded incident.
- **Seed check**: Ensure `clips/kul_fall_1.avi` reports `done` with `distress_heuristic`.
- **Path B mirror**: `no_movement_silence` finishes through stub dials and leaves the last Telegram line as the pressure tick. Click **Family page · leave open** so the 1/2/3 inform card is the thing in frame.
- **Zooms**: 150%+ browser zoom reads better than post-zoom.
- **Exact numbers**: recall 1.0 · ten labeled cases · zero false escalations · torso ~70° · 1-in-4 fall statistic.
- **Live cite**: `https://d2u7pls4da2poz.cloudfront.net/ui/` · landing `https://pamu512.github.io/opencv-care-ladder/` · repo `https://github.com/pamu512/opencv-care-ladder`.

## Attribution

- Fall footage: KU Leuven Advise group, "High-quality fall simulation data".
- Models: OpenCV Zoo (MediaPipe person-detection + BlazePose ONNX, Apache-2.0).
- vtest pedestrian clip: OpenCV samples (Apache-2.0).
