# Adaptive Schedule Learning — Care Ladder (all hackathons)

**Date:** 2026-09-27  
**Owner implementer:** Hermes  
**Applies to:** OpenCV Care Ladder (main / CareCV), Amazon Alexa+ Fire TV (`amazon/alexa-plus-fire-tv`), Galuxium Care Ladder SaaS  
**Related:** Galuxium design `2026-09-17-galuxium-care-ladder-saas-design.md`; Amazon PRD `2026-09-26-amazon-care-ladder-alexa-plus-prd.md`; base design `2026-09-11-agentic-senior-care-ladder-design.md`

## 1. Problem

A fixed YAML timeout that never moves feels stagnant after week one. Families and facilities need the ladder to **learn the person’s usual day fast**, then **settle** so only real deviations open incidents.

## 2. What “learn” means (safe)

| Learns | Does not learn |
| --- | --- |
| Usual still / wake / leave-zone windows from audit | Clinical diagnosis or psych labels |
| Adaptive `no_movement.timeout_sec` and check-in wait windows | Black-box “risk score” |
| `learning_phase: rapid \| settled` | Auto-enabling emergency / 911 |
| Explainable “usual until X; today since Y” copy | Face ID / identity biometrics |

Ladder **shape** (rung order, contacts, notify vs dial) stays human-configured. Timing and sensitivity move.

## 3. Phases

| Phase | When | Behavior |
| --- | --- | --- |
| **rapid** | Default for new household / tenant / plan | Shorter stillness timeouts (more sensitive), more check-ins; staff/family can mark “this is normal” to accelerate learning |
| **settled** | After N confirmed-OK calendar days (default **10**) with no operator freeze, or explicit “mark settled” | Timeouts from `RoutineProfile` percentiles; fewer false starts |

Operator controls: **freeze learning**, **reset learning** (back to rapid + clear histogram), **mark settled**.

## 4. Data model (hackathon-honest stub)

No online ML required by deadline. Ship histograms + rules judges can feel.

```text
RoutineProfile
  subject_key: str          # household_id or tenant_id + monitored id
  learning_phase: rapid | settled
  frozen: bool
  confirmed_ok_days: int
  settled_after_days: int   # default 10
  # hour-of-day (0..23) buckets — counts of cue kinds and OK resolves
  still_hour_hist: list[int]      # length 24
  leave_zone_hour_hist: list[int]
  ok_resolve_hour_hist: list[int]
  # derived / cached
  suggested_no_movement_timeout_sec: int
  usual_still_end_hour: int | null   # for explain copy
  updated_at: datetime
```

Persist:

- **OpenCV main / Amazon (process-local):** JSON file under `data/routine_profiles/{subject_key}.json` (or audit-store sibling) — gitignore `data/`.
- **Galuxium SaaS:** Postgres table `routine_profiles` keyed by `tenant_id` (+ optional `monitored_id`); same fields. If SaaS tenancy not landed yet, use the JSON store and leave a clear TODO to migrate in Galuxium Task 1+.

## 5. Update rules (deterministic)

On each closed incident (or demo fixture resolve):

1. If `frozen`, skip updates (still serve timeouts).
2. Bucket cue start hour into the matching hist; if resolve was OK / Path A style, increment `ok_resolve_hour_hist` and maybe `confirmed_ok_days` (one bump per local calendar day max).
3. Recompute `suggested_no_movement_timeout_sec`:
   - **rapid:** `max(plan_floor, min(plan_timeout, plan_timeout * 0.5))` for demo use a visible delta (e.g. plan 900s → effective ~300–450s) so judges see change; clamp to `[min_sec, plan.triggers.no_movement.timeout_sec]`.
   - **settled:** blend plan timeout with profile (e.g. p80 of observed still durations, or hour-aware multiplier). Prefer simple and tested over clever.
4. Promote to **settled** when `confirmed_ok_days >= settled_after_days` and not frozen.
5. Always write an audit event: `tool: routine_profile_update` with phase, suggested timeout, and one-line explain.

Config floors (care plan optional block, additive):

```yaml
learning:
  enabled: true
  settled_after_days: 10
  min_no_movement_timeout_sec: 120
  rapid_timeout_factor: 0.4   # effective = max(min, plan_timeout * factor) while rapid
```

Missing `learning:` → treat as enabled with defaults (so all three demos show the story).

## 6. Orchestrator / cue wiring

Before starting a `no_movement` wait / when building detector timeout:

1. Load `RoutineProfile` for subject.
2. Effective timeout = profile suggestion if learning enabled and not overridden by explicit demo fixture that pins timeout.
3. Attach to cue/incident detail: `learning_phase`, `effective_timeout_sec`, `explain` string.

Path A (`no_movement_ok`) and Path B (`no_movement_silence`) **behavior outcomes must stay the same** for fixed fixtures; learning may still update profile and show badge. Prefer fixture override `learning.enabled: false` OR seed a profile in settled phase for regression tests that assert exact timeouts.

## 7. UI (all surfaces that show incidents)

Badge near mode / household:

- Rapid: **Learning schedule**
- Settled: **Schedule settled**
- Frozen: **Learning frozen**

On timeline / incident detail: one explain line, e.g.  
`Usual still often ends by 09:00; today still since 07:10 · timeout 6m (learning)`.

Amazon Fire TV: same badge copy in Calm Care-Tech chrome (short label OK).  
Galuxium facility console: same + optional admin “Freeze / Reset / Mark settled” buttons (stub POST is fine).

## 8. Demo / filing story (shared VO beat)

“First days the ladder learns fast — more check-ins while it maps the day. Once the schedule settles, it only escalates on real deviations, and every timeout stays explainable on the timeline.”

Do **not** claim neural nets, clinical accuracy, or HIPAA.

## 9. Branch / hackathon map

| Track | Branch | Persist | Notes |
| --- | --- | --- | --- |
| OpenCV CareCV | `main` | JSON file | Primary shared module + tests land here first |
| Amazon Care Alexa+ | `amazon/alexa-plus-fire-tv` | JSON file | Port/cherry-pick shared module; **do not merge to main** until ~Oct 26 OpenCV judging gate |
| Galuxium SaaS | `main` (SaaS tasks) or follow Galuxium plan branch | JSON now → Postgres with tenancy | Same module; tenant_id when Task 1+ exists |

## 10. Non-goals

- Risk scores, relapse/psychosis detection, rehab care-plan vertical (later)
- Auto emergency enablement
- Online gradient ML / cloud training loop for MVP
- Changing rung order automatically

## 11. Success criteria

- [x] Shared `RoutineProfile` + phase rules + tests on `main`
- [x] Effective stillness timeout uses profile; audit explains it
- [x] UI badge rapid → settled visible in `/ui` (Fire TV short labels on Amazon branch)
- [ ] Amazon branch has equivalent behavior without merging to main
- [x] Galuxium plan references this spec; SaaS path uses same API (Postgres later OK)
- [x] Existing Path A/B tests still pass (fixtures pin or disable learning as needed)
- [x] No clinical / 911 / black-box risk copy in UI
