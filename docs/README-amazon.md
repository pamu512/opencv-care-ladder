# Care Ladder

[![CI](https://github.com/pamu512/care-ladder-amazon/actions/workflows/ci.yml/badge.svg)](https://github.com/pamu512/care-ladder-amazon/actions/workflows/ci.yml)
[![Alexa+](https://img.shields.io/badge/Alexa%2B-primary-orange)](https://github.com/pamu512/care-ladder-amazon)
[![Fire TV](https://img.shields.io/badge/Fire%20TV-supporting-informational)](https://github.com/pamu512/care-ladder-amazon)
[![MCP](https://img.shields.io/badge/MCP-Streamable%20HTTP-blue)](https://github.com/pamu512/care-ladder-amazon)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Camera cue → multi-rung care ladder → **Alexa+** agent (self-hosted MCP) → **Fire TV** caregiver audit (silhouette-only). Wellness ladder, not a medical device.

## Hackathon map

| Surface | Role |
| --- | --- |
| **Alexa+** | **Primary.** Self-hosted MCP care-flow tools + in-repo simulated Alexa+ MCP client. Multi-rung, multi-turn, incident session state — not single-turn Q&A. |
| **Fire TV** | **Supporting.** Calm Care-Tech caregiver surface at `/firetv/` — D-pad focus, Ambient Hearth layout, data-driven from the live ladder API. |
| **MCP** | Spec 2025-11-25+ Streamable HTTP at `/mcp`. Seven tools: `start_or_resume_incident` · `check_in_prompt` · `advance_rung` · `resolve_incident` · `get_incident_status` · `notify_caretaker` · `request_call`. |
| **AWS Builder mini** | **Not filed.** No Bedrock / AgentCore in-tree. |
| **Open source** | MIT. Significant in-window update of the OpenCV Care Ladder spine (vision still triggers; Amazon surfaces are new). |

## Agentic proof (not thin MCP)

Session state is keyed by **household + incident**. One incident id survives the tool sequence:

1. `start_or_resume_incident` — start, or resume the same open incident
2. `check_in_prompt` — fail-closed intent (`clear_ok` / `needs_human` / `unclear`)
3. `advance_rung` / `notify_caretaker` / `resolve_incident` — same session, same memory

Every tool return includes a `session_snapshot` (`household_id`, `incident_id`, `rung`, `status`, `tools` trail) so the agent transcript reads as one memory, not disconnected calls. In-repo MCP **client**: `python -m care_ladder.mcp_server.alexa_sim` (prints `SESSION … rung=… status=…` banners). Soft “don’t worry” can stand the ladder down; “okay but hurt” will not.

## Quickstart (≤60s to something on screen)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

.venv/bin/uvicorn care_ladder.api.app:app --port 8010
# Fire TV:  http://127.0.0.1:8010/firetv/
# Alexa+ sim (MCP client over Streamable HTTP):
.venv/bin/python -m care_ladder.mcp_server.alexa_sim --url http://127.0.0.1:8010
# Soft OK (same incident, then resolve):
.venv/bin/python -m care_ladder.mcp_server.alexa_sim --url http://127.0.0.1:8010 --answer "don't worry"
```

Amazon household plan: `configs/amazon_demo_home.yaml`. Legacy OpenCV plan (before/after story): `configs/demo_home.yaml`.

## Demo paths

| Path | Fixture / sim | What you should see |
| --- | --- | --- |
| Soft OK | `--answer "don't worry"` or `alexa_path_a_soft_ok` | Clear OK · same incident · ladder stands down |
| Needs human | `--answer "I'm okay but I think I'm hurt"` or `alexa_path_a_needs_human` | Needs human · does **not** clear · notify |
| Unclear | `--answer "nngh"` or `alexa_path_a_unclear` | Unclear · re-ask then notify |
| Silence | `--answer ""` or `alexa_path_a` | Two attempts · wait · notify · simulated call |
| Path B occlusion | `alexa_path_b` | Coverage is **privacy**, never distress |

Fire TV demo console (bottom right) drives the same `/demo/run` fixtures. Shared live host from the OpenCV stack (until a dedicated Amazon hostname exists): https://d2u7pls4da2poz.cloudfront.net/firetv/

## Privacy & fail-closed

- Silhouette / processed-on-device frames only — **no live caregiver video**.
- Emergency dial **off** by default; demo build is hard-locked (hold-to-review explains, never calls).
- Reserved fiction only: `(555) 010-2276`. Never PSTN. Never 911 from this build.
- Footer on the TV: wellness ladder, not a medical device.

## Friction log

Honest tool/SDK notes (bonus): [`docs/friction-log.md`](docs/friction-log.md).

Failure modes: [`docs/failure-modes.md`](docs/failure-modes.md). Why this design: [`docs/research-brief.md`](docs/research-brief.md).

## Origins

Significant in-window update of [OpenCV Care Ladder](https://github.com/pamu512/opencv-care-ladder). Vision cues are vendored here as the **trigger** (stillness / no-visibility / distress-family); the Amazon work is MCP + Fire TV + rungs. The OpenCV repo stays separate until after OpenCV judging. Optional vision proofs (`opencv_stillness`, `opencv_dnn_person`) remain fixtures — they are not the lead demo. Models/clips are not committed; `scripts/download_models.sh` and `scripts/download_clips.sh` fetch them if you want those paths.

## Tests

```bash
.venv/bin/pytest tests/ -v
```

Amazon-facing: `test_mcp_server.py`, `test_alexa_sim.py`, `test_firetv_app.py`, `test_amazon_*.py`, `test_response_intent.py`. CI on this repo is **test-only** (no auto-deploy to the shared OpenCV CloudFront).

## License

MIT © 2026 Anoop Pamu. See [LICENSE](LICENSE).
