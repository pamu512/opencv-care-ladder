# Adaptive Schedule Learning — Implementation Plan

> **For Hermes:** TDD per task. Spec: `docs/superpowers/specs/2026-09-27-adaptive-schedule-learning.md`. Anoop (via Devt) asked for this on **all three** Care Ladder tracks: OpenCV main (CareCV), Amazon `amazon/alexa-plus-fire-tv`, Galuxium SaaS path on main.

**Goal:** Ship explainable rapid→settled schedule learning (RoutineProfile + adaptive stillness timeouts + UI badge) across OpenCV, Amazon Alexa+, and Galuxium without clinical/risk/911 claims.

**Order:** (1) shared module + wire on `main` → (2) port to Amazon branch → (3) Galuxium hooks (JSON now; Postgres when tenancy lands). Do **not** merge Amazon → main until ~Oct 26.

## Global constraints

- Spec authoritative: `2026-09-27-adaptive-schedule-learning.md`
- Path A/B outcomes unchanged; pin or disable learning in timeout-exact tests
- `emergency.enabled` fail-closed; learning never enables emergency
- No black-box risk scores; audit must explain timeout
- Amazon: work only on `amazon/alexa-plus-fire-tv`
- Galuxium SaaS Tasks 1–12 remain separate; this plan is additive
- YAGNI: no online ML, no rehab vertical

---

## Task 1: RoutineProfile model + JSON store (main)

**Files:** `src/care_ladder/learning/profile.py`, `src/care_ladder/learning/store.py`, `tests/test_routine_profile.py`

- [ ] Models: `LearningPhase`, `RoutineProfile`, optional `LearningConfig` on care plan (or parallel pydantic)
- [ ] JSON store under `data/routine_profiles/` (gitignore `data/`)
- [ ] `record_incident_outcome(...)`, `effective_no_movement_timeout_sec(plan, profile)`, promote rapid→settled after N OK days
- [ ] freeze / reset / mark_settled helpers
- [ ] Tests for rapid factor, settle promotion, freeze skip, explain string
- [ ] Commit: `feat(learning): RoutineProfile JSON store and phase rules`

## Task 2: Wire orchestrator + care-plan learning block (main)

**Files:** `src/care_ladder/models.py` / `plan_loader.py`, `src/care_ladder/ladder/orchestrator.py` (or cue path), `configs/demo_home.yaml` (additive `learning:`), tests

- [ ] Optional `learning:` on CarePlan with defaults when missing
- [ ] On incident start: resolve effective timeout from profile; stash on cue/incident detail
- [ ] On resolve: update profile + audit `routine_profile_update`
- [ ] Fixtures that assert exact timeouts: `learning.enabled: false` or seeded settled profile
- [ ] All existing Path A/B tests green
- [ ] Commit: `feat(learning): adaptive stillness timeout from RoutineProfile`

## Task 3: UI badge + explain line (main `/ui`)

**Files:** `src/care_ladder/api/static/index.html` (and any incident API that returns profile)

- [ ] Badge: Learning schedule / Schedule settled / Learning frozen
- [ ] Timeline explain line from last profile update or incident detail
- [ ] Optional minimal POST freeze/reset/mark-settled (demo-safe)
- [ ] Commit: `feat(learning): UI schedule learning badge`

## Task 4: Port to Amazon branch

**Branch:** `amazon/alexa-plus-fire-tv` only

- [ ] Cherry-pick or re-apply Tasks 1–3
- [ ] Fire TV Calm Care-Tech short badge labels
- [ ] Amazon fixtures / MCP paths still pass
- [ ] Push branch; **do not merge main**
- [ ] Commit(s) on amazon branch: `feat(learning): adaptive schedule on Alexa+ / Fire TV`

## Task 5: Galuxium hooks

- [ ] Same learning module used when SaaS runs on main
- [ ] If Postgres tenancy exists: add `routine_profiles` table + store impl; else keep JSON and note in Galuxium README
- [ ] Facility UI shows same badge; admin freeze/reset OK as stub
- [ ] Append one shot to Galuxium demo shot list: learning badge rapid→settled
- [ ] Commit: `feat(learning): Galuxium surfaces for RoutineProfile`

## Task 6: Docs / Devpost one-liners (light)

- [ ] Touch `docs/devpost-draft-opencv.md`, `docs/devpost-draft-amazon.md`, `docs/devpost-draft-galuxium.md` with one honest learning sentence each (no ML hype)
- [ ] Commit: `docs: adaptive schedule learning filing note`

---

## Done when

- main: tests green + badge visible + audit explain
- amazon branch: same behavior + Fire TV badge; not merged to main
- Galuxium path uses same API
- Spec success criteria checkboxes can be marked by implementer
