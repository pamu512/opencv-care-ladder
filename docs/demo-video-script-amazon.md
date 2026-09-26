# Care Ladder - Amazon Build, Ship, Shape update (in-window)

Demo video (≤3 min) - English VO, ~135 wpm pacing. Path A (stillness) then
Path B (camera occluded). Before/after framing per PRD §6.

| # | Time | Shot | VO (word-for-word) |
|---|------|------|--------------------|
| 1 | 0:00–0:12 | BEFORE: existing OpenCV Care Ladder console (`/ui/`), fire `no_movement_silence` fixture, rail lights up cue → speaker → dial ×2 | Before this update, Care Ladder already watched the room with OpenCV and climbed a careful ladder: notice, ask, then call. But it asked through a speaker stub, and the caregiver watched from a web console. |
| 2 | 0:12–0:20 | Title card: "Care Ladder for Alexa+ - same project, significant in-window update" | This is the same project, significantly updated for Alexa+. The camera still triggers everything. What changed is what happens next. |
| 3 | 0:20–0:45 | Fire TV dashboard `/firetv/` all-clear state. Arrow keys demonstrate D-pad focus moving between cards | The caregiver surface is now a Fire TV dashboard. Calm by design: silhouette only, never live video, and a remote-friendly interface that works with just the D-pad. |
| 4 | 0:45–1:10 | Demo console → "Path A · stillness cue". Watch status pill flip to Re-checking, then Checking on Meera; transcript panel shows both Alexa+ attempts with real prompt text | When OpenCV sees four minutes of stillness, the ladder starts. Alexa+ checks in by voice - two attempts. The transcript on the TV is the real conversation, quoted from the incident's audit trail. |
| 5 | 1:10–1:25 | Wait: the 45-second window message, then status flips to Calling Anoop; push note appears; audit trail fills | No answer. After a forty-five second wait, the ladder notifies Anoop - push plus this TV - with the whole trail preserved, timestamped. |
| 6 | 1:25–1:40 | Click "Acknowledge" → Resolved · acknowledged state, trail stays | A human closes the loop. The acknowledgement is part of the audit trail, not a silent read receipt. |
| 7 | 1:40–2:05 | Demo console → "Path B · camera occluded". Hero shows camera-blocked state; silhouette shows "No reading · lens covered"; the blanket prompt in transcript; notify card says inform, no distress claim | Cover the camera and Care Ladder does not panic. Occlusion reads as privacy, not distress. It asks Meera to move the blanket, and if there's no answer it informs Anoop on exactly that basis - no distress is ever claimed. |
| 8 | 2:05–2:25 | Terminal: run `alexa_sim.py` against the live server; show the MCP transcript lines (initialize → start → notify → request_call) | Under the hood, the Alexa+ path runs on a self-hosted MCP server - Streamable HTTP, current spec - exposing seven care-flow tools. This simulated Alexa+ agent is an MCP client driving the real ladder over HTTP. |
| 9 | 2:25–2:45 | Footer of the TV: wellness disclaimer, emergency off; emergency gate modal opened, hold attempted, stays locked | Two things this system will not do: claim to be medical, or call emergency services on its own. Emergency is off by default and hard-locked in this build. |
| 10 | 2:45–3:00 | Split screen: OpenCV console (before) + Fire TV (after) | Same project, same fail-closed spine - now with an Alexa+ agent doing the check-in and Fire TV keeping the family in the loop. Care Ladder. |

## Pre-record checklist
- Local: `./scripts/run_demo.sh` (port 8000 occupied on the dev machine - use a free port like 8010), then open `/firetv/`.
- Live AWS: `https://d2u7pls4da2poz.cloudfront.net/firetv/` (after this branch deploys - until then, local only).
- Fire both fixtures once before recording so the audit trail has history.
- Terminal ready with the sim command:
  `.venv/bin/python -m care_ladder.mcp_server.alexa_sim --url http://127.0.0.1:<port>`
- Record browser at 1280×720; VO ~390 words ≈ 2:55 at 135 wpm.

## Notes
- All phone numbers on screen are reserved fictional (555) 010-2276.
- The TV is data-driven: every state change reflects a real incident's audit
  events from the API - it is not a disconnected mock slideshow.
