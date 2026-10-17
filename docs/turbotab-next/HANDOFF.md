# Handoff

The current state only. When a new state replaces this one, the replaced body goes, verbatim and
under a dated heading, to the top of `HANDOFF_HISTORY.md`, which holds every earlier state and the
rulings recorded in them. Rules live in the root `CLAUDE.md` and in `ORCHESTRATOR.md`; rulings live
in the documents they amend.

## State on 2026-10-16 (evening), checked against git

- **turbotab-next** is the R0b merge (feat/r0b-int at 18659faa, landed and pushed on 2026-10-16), on top of R0a and stage 1. It holds:
  - the claims registry with its first 27 claims (`turbotab/core/claims/`; `docs/turbotab-next/reference/CLAIMS_REFERENCE.md` lists them);
  - stage 1's four fix branches, the claims registry spec, the replication data checks, the onboarding guidance;
  - `plans/SWARM_PLAN.md`, the research swarm's draft, for the pause.
- **Environments:**
  - `venv-next/` in the main checkout is TurboTab v2's environment, built with uv from `turbotab/server/constraints.txt` (CI's pins); `scripts/py` runs it from any worktree.
  - `venv/` is Classic's, untouched.
  - The fast tiers pass under `venv-next` on 18659faa: server 189, core 3,561, models 34.
- **CI:** read it on turbotab-next's head through the public check-runs API before you rely on it.
  - On 18659faa (ci/full-r0b-int) every job passed: the four fast shards, the full tier's three shards, the browser mocks, the image and both launchers. The known-red e2e-real had not finished when R0b landed.
  - The full tier runs in three shards, each with its own 100-minute deadline and its slowest files posted as annotations.
  - One server test is intermittent on Linux (INBOX, "W1, intermittent test").
- **No CI job installs R,** so the R reference tests (the trial core against R's gee, and others) run only on a machine with R and its packages (INBOX, R0b).

## Done since 2026-10-11

- **Stratification optimism in Lean 4** (914a054a): the exact decomposition and the stratification sandwich are proved, and the general law is stated (notes/lean-strat/).
- **The claims backtest** (e6e58809): 14.5% (5.7–23.3%) of a seeded sample of 117 decision-setting claims carry an S1 defect (notes/claims-backtest/).
- **Stage 1, the 15 S1 defects,** each fixed test first and re-reviewed until approved. B-281 ended at the honest refusal; partial reporting is filed in INBOX as v2.x.
- **The claims registry, R0** (plans/R0_PLAN.md):
  - **R0a, the scaffold:** 8 runs, 1.74M fresh tokens.
  - **R0b, the first 27 claims and the splits:** 31 runs, 9.76M fresh (about 570M cached), against about 3M to 3.5M planned. Its 14 verifier passes found 10 blockers, 12 majors and 55 minors. R0_PLAN, "R0b landed", records what it settled and what it hands R1.
- **The claims registry spec** (plans/CLAIMS_REGISTRY.md, draft 2), ruled by Nolan on 2026-10-15.
- **Replication data checks** (notes/replication/): Bang 2009 and Menke 2015 can be replicated approximately; Sullivan 2021 replaces Yang 2014.
- **The onboarding guidance,** in three passes (notes/onboarding-audit/).
- **plans/SWARM_PLAN.md,** drafted by the orchestrator, docs only (3c131b0a), for the pause.

## Known red

- **The e2e-real job is still `continue-on-error`.** The same five journeys failed on every CI run of 2026-10-16; check each for a defect before tuning a wait (INBOX, "W1, e2e-real").

## Next, in order

The goal, re-ordered on 2026-10-15 at Nolan's direction so that the product is walkable sooner:
- Nolan walks his own project, NHANES fasting glucose under Predict, end to end in the app (step 3).
- The Estimate lane passes its first real replication (step 4).
- Then W2–W4 finish the lanes toward v2.0.0.

Send Nolan a short status at the end of each step, ending with one line of cost and rework. Tell him before any step expected to exceed about 3M subagent tokens or 25 agent runs. Wait for him only where a step says **[Nolan]**. **This week's pause is after step 3's walkthrough** (Nolan, 2026-10-15).

1. **Check, in a fresh session, that the roles load.** This session's workflow agents get neither
   the root CLAUDE.md nor the roles, since both load at session start, so in the new session:
   - Make one agent call with `agentType: 'builder'`, worktree isolation and no `model`.
   - Check that it quotes builder.md's first step.
   - Check that it sees one copy of CLAUDE.md, not two.

   Until that check, agents are told to read CLAUDE.md and their role file by path
   (ORCHESTRATOR §0 step 4).

   **[Nolan]** The remaining 15 worktrees: Nolan approved their removal on 2026-10-16, branches
   kept, and removes them himself at the shell. No agent removes them: auto mode refused it. The
   two with unlanded code are saved as `wip/2026-10-05-*` commits (INBOX, worktree clean-up).
2. **W1 part 2** (WALKABLE_PLAN §5; the packages are under Leftovers), then your own walkthrough of NHANES glucose under Predict, with a friction log, before Nolan sees it. Every claim that journey puts on screen is audited and registered as R1's first batch. Why: this is the walkable product, and the friction log decides what gets work next. First, check the five e2e-real journeys that fail on CI for defects: the NHANES one is the journey Nolan walks. **[Nolan]** He judges the walkthrough on screen. Then pause.

   Bring `plans/SWARM_PLAN.md` to the pause. Still open in it for Nolan: whether Haiku may go
   below his Sonnet floor for analysts. Until he answers, no Haiku.
3. **The Sullivan 2021 replication** (Am J Prev Med 61:563). Why: the first end-to-end check of the Estimate lane. It reports the decisions the paper stated, the ones it left unstated, and how far each unstated option moves the published number.
4. **W2–W4,** each registering the claims it adds, and building what SWARM_PLAN §8 asks of them (a run record per Fit, results as data, which decisions mattered as data, an analyst mode, data provenance). The remaining S2/S3 fixes and R0b's S1 INBOX lines go in the order the walkthrough and the replication surface them.
5. **Heavy runs, each at a time Nolan sets** **[Nolan]**:
   - the T1 rerun at 600 datasets (the small-sample standard is in code, A-001);
   - the reference-journey recapture, then the materiality calibration on it.
6. **The rest of the research groundwork:**
   - the Lean pilot, on about a dozen invariance and exactness claims;
   - the stage-3 audit, with decoy set v2;
   - the Bang 2009 and Menke 2015 replications;
   - data provenance in ingest (each file's source URL and release date).
7. **The research swarm,** built to plans/SWARM_PLAN.md once its gates (plans/CLAIMS_REGISTRY.md §8) are passed. It writes to a corpus kept apart from the registry.

## Leftovers that still hold (from the state of 2026-10-11)

- **W1 part 2** (WALKABLE_PLAN §5):
  - W1-E: the quest-log shell on live /quest (it owns `turbotab/frontend/CLAUDE.md`, which the onboarding guidance
    rewrote on 2026-10-15; brief its builder on the new file);
  - W1-F: the Predict cards;
  - W1-G: Predict Results v0;
  - e2e quest-nhanes-predict and quest-own-csv.
- **Phase-4 leftovers:**
  - a plan_units field for the leaf cap on yes/no and class outcomes;
  - the forest and boosted trees reading "units" alike;
  - omics' TuningDecl.clause;
  - the preview lineage's coding;
  - the wide elastic net's threads in the record.
- **W1 part 1 leftovers:**
  - tune the e2e-real waits, then drop continue-on-error;
  - mock handlers for the new hooks;
  - the seal card's chain of refusals;
  - the engine serving the held-out no-predictor baseline;
  - the calibration intercept (a methods ruling, yours);
  - Cancel beside a held Fit;
  - recapture dietary-prediction.
- **After W1:** W2a/W2b, Predict built to BEATS_PREDICT, then W3/W4 Estimate built to
  BEATS_MODELS_RESULTS.

## Open for Nolan

- **Haiku for swarm analysts** (SWARM_PLAN §10): whether analysts may go below his Sonnet floor. Until he answers, no Haiku.
- On 2026-10-15 he ruled:
  - **The agent files:** `CLAUDE.md` and `.claude/agents/` are public in the repo and kept out of the release zip.
  - **Proprietary tools:** the rule covers the product only.
  - **The paper's tag** is `v1.0.0` on main; no other tag is wanted.
  - **Disputed rulings:** a ruling that evidence shows wrong is shown to him once, and he re-rules.
  - **Status and spend:** each status carries a line of cost and rework, and he is told before a step expected to exceed about 3M tokens or 25 agent runs.
  - **The pause** comes after his walkthrough.
- On 2026-10-16 he answered SWARM_PLAN's two questions: the swarm's compute runs on his machine under the heavy-run rules, and analysts have no cap like the build agents' (spend is still measured and reported).

## Nolan's context

- **His real project is NHANES fasting glucose under Predict,** used at a fasting-draw visit; on the
  medication answers, the app recommends. The light walk gave R² 0.155 (cycle CV) and 0.18 on the
  sealed 2017–18 cycle; DIQ is not in the fixture. He re-ran it "several times over" and never
  passed about 0.25, and that only after trimming glucose's tails before upload (INBOX, "From
  Nolan's question on predicting glucose").
- **Direction A is accepted provisionally.** He wants to judge it on a real project.
- **Strategy homework he may bring back:** whether researchers' pain is doing or defending; his
  stance on recommendations and the lock; the anchor paper; a classroom pilot.

## Heavy runs to schedule with Nolan

- The T1 rerun at 600 datasets: about 60 minutes at 4 jobs. Approved on 2026-10-11 to run after
  stage 1 lands; it still needs a time from him.
- The reference-journey recapture (the drivers now press Fit), then the materiality calibration on
  it.
