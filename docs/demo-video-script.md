# Care Ladder Vision v2 - Demo Video Script (main @ 269a695)

**Target:** ~2:45-3:10 Chatterbox cut. Hard stop 3:30. Absolute max 4:00.
**Pin:** PR #17 / merge `269a695`. Live `/ui/` Demo scenarios, Fall signature button, upload-takes-minutes, reperceive, fall_cls demoted, n=11 / 18.0s, short AWS beat, Why I built this open.
**Rules:** No Alexa. No em dashes. Fall frames stay blurred. Do not claim fall_cls ONNX decides. No YouTube / Devpost / Final Submit from this package.

Ax: grok peer ## CONSENSUS: APPROVE (autoclaw z.ai 401). Spoken word count ~313.

## Shot list + VO (read verbatim)

| # | Clock | On screen | VO |
| --- | --- | --- | --- |
| 1 | 0:00-0:32 | Title ≤8s under opening line, then live CloudFront `/ui/` with Demo scenarios open (Fall signature visible) | "My mother lives alone in India. In six months she fell four times. We had a camera, but nobody could watch it around the clock. Care Ladder turns that camera into something that checks in before it escalates, with blurred clips so she never feels watched. Here is the live console. Demo scenarios opens on first visit, including a Fall signature path." |
| 2 | 0:32-0:56 | Path A · verbal OK → Resolved, no Dial | "One YAML care plan: zone, stillness timeout, ordered ladder. Look again, ask out loud, wait, then page family. Emergency is fail-closed. The cue fires. The speaker asks are you okay. She says I'm fine. Resolved at check-in. No dial. The audit timeline shows why. Vision changed the next tool." |
| 3 | 0:56-1:20 | Family chat mirror in frame; silence → Acknowledge → acked | "Same cue, silence this time. After the spoken check-in it pages the family. That text is the Telegram inform card. Reply one, two, or three. The caregiver clicks Acknowledge. The ladder stands down. Same YAML. Different next tool." |
| 4 | 1:20-1:52 | Person DNN + silhouette; no fall_cls decision claim | "A real photo through the Open C V person detector. Then privacy: only silhouette frames reach the caregiver. Blur before storage. On eleven labeled cases, cue recall is one point zero, false escalations zero, mean time to confirm eighteen seconds. Demo scale, not clinical. A separate fall classifier ONNX exists for research, but it is not on the live decision path. Open C V cues still drive the ladder." |
| 5 | 1:52-2:24 | Fall signature / blurred fall banner; upload-minutes hint if shown | "Fall signature on the guided row. On KU Leuven research footage, person detection then pose: sudden vertical to horizontal. The red card says Fall signature detected. Frames stay blurred. The ladder can re-perceive: a second cue opinion from buffered frames before it escalates. Upload analysis runs in the cloud and can take minutes, so the demo pins a finished fall for judges. Research re-enactment, not a clinical study." |
| 6 | 2:24-2:44 | Close card GitHub + CloudFront | "It runs on AWS: CloudFront to Fargate, with DynamoDB and privacy clips in S three. Care Ladder: Open C V five sees, the ladder decides, every step on the record. Confirm before you escalate. Repo and live demo are on screen." |

## Cut if over
1. Second AWS sentence
2. Path B color ("Reply one, two, or three")
3. Upload-minutes sentence last (keep if possible)

## Pipeline
Chatterbox mlx Amazon-locked settings + Playwright stills + assemble static (same as opencv-telegram-20261004).
