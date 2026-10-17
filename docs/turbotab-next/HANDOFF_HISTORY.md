# Handoff history

Every earlier state of `HANDOFF.md`, verbatim and newest first, with the rulings each one recorded. `HANDOFF.md` holds the current state only; when it is replaced, the replaced block is added here, at the top, under a dated heading. Documents that cite a ruling by the heading it had in HANDOFF ("UI and scope discussion", "The orchestrator's methods floor", "Engine follow-ups found") find it here.

## 2026-10-16 · R0a landed, R0b next

### State on 2026-10-16, checked against git

- **turbotab-next** is the stage-1 merge (feat/s1-int, fast-forwarded and pushed on 2026-10-16). It holds:
  - stage 1's four fix branches;
  - the claims registry spec, ruled;
  - the replication data checks;
  - the onboarding guidance: the root `CLAUDE.md`, `.claude/agents/`, `ORCHESTRATOR.md` and `scripts/py`.
- **Environments:**
  - `venv-next/` in the main checkout is TurboTab v2's environment, built with uv from `turbotab/server/constraints.txt` (CI's pins); `scripts/py` runs it from any worktree.
  - `venv/` is Classic's, untouched.
  - The fast tiers pass under `venv-next`: server 189, core 3,012, models 34, plus the trials and missing-data acceptance files.
- **CI:** read it on turbotab-next's head through the public check-runs API before you rely on it.
  - The full tier runs in three shards. Each stops at its own 100-minute deadline, names a test hung past 20 minutes, and posts its slowest files and tests as annotations (2026-10-16). The cause was four runs that ran out the job's 120 minutes with no report.
  - On c2f07562 every job passed except the known-red e2e-real.
  - One server test is intermittent on Linux (INBOX, "W1, intermittent test").

### Done since 2026-10-11

- **Stratification optimism in Lean 4** (914a054a): the exact decomposition and the stratification sandwich are proved, and the general law is stated (notes/lean-strat/).
- **The claims backtest** (e6e58809):
  - about 970 decision-setting claims inventoried, and a seeded sample of 117 refereed blind and audited against the code;
  - 14.5% (5.7–23.3%) of the sample carry an S1 defect, and 44% none;
  - the plan, the sealed key, every verdict and the fix list are in notes/claims-backtest/.
- **Stage 1, the 15 S1 defects:**
  - Each was fixed test first, against the old code, and re-reviewed by a fresh agent until approved: three rounds for most, seven for B-281.
  - **B-281 ended at the honest refusal.** A trial in which an exact LP finds anyone separated is refused, naming who and on which terms. A fit too ill-conditioned for float64 to resolve is refused as nearly singular. Partial reporting is filed in INBOX as v2.x.
  - The stability-selection bug that was known red is fixed (S0).
  - The reviewers' follow-ups are in INBOX under "From the claims backtest".
- **The claims registry spec** (plans/CLAIMS_REGISTRY.md, draft 2). Nolan ruled its three questions on 2026-10-15: grades in the "why" layer; an unsupported claim never refuses; the swarm's corpus is kept apart from the registry.
- **Replication data checks** (notes/replication/): Bang 2009 and Menke 2015 can be replicated approximately from public data. Yang 2014 cannot, so Nolan chose Sullivan 2021 instead.
- **The onboarding guidance,** in three passes (notes/onboarding-audit/):
  - a Fable xhigh audit and revision;
  - four cold-start probes, then an Opus xhigh revision;
  - a fresh Fable xhigh wisdom review.

### Known red

- **The e2e-real job is still `continue-on-error`** until its waits are tuned to CI's runners.

### Next, in order

The goal, re-ordered on 2026-10-15 at Nolan's direction so that the product is walkable sooner:
- Nolan walks his own project, NHANES fasting glucose under Predict, end to end in the app (step 3).
- The Estimate lane passes its first real replication (step 4).
- Then W2–W4 finish the lanes toward v2.0.0.

Send Nolan a short status at the end of each step, ending with one line of cost and rework. Tell him before any step expected to exceed about 3M subagent tokens or 25 agent runs. Wait for him only where a step says **[Nolan]**. **This week's pause is after step 3's walkthrough** (Nolan, 2026-10-15).

1. **Check, in a fresh session, that the roles load.** Everything else in this step was done on
   2026-10-16:
   - the memory trimmed, from 209 KB to 63 KB, with a backup kept beside it;
   - the pip-install line struck, and WAVE_C6A_PLAN marked landed;
   - the methods documents' test paths fixed;
   - `scripts/state.sh` added;
   - 89 landed or detached worktrees removed.

   A check that same day found that this session's workflow agents get neither the root CLAUDE.md
   nor the roles, since both load at session start. The harness does give each agent its own
   attribution lines. So, in the new session:
   - Make one agent call with `agentType: 'builder'`, worktree isolation and no `model`.
   - Check that it quotes builder.md's first step.
   - Check that it sees one copy of CLAUDE.md, not two.

   Until that check, agents are told to read CLAUDE.md and their role file by path
   (ORCHESTRATOR §0 step 4): Nolan was remote on 2026-10-16, so step 2 went ahead in the old session.

   **[Nolan]** The remaining 15 worktrees: Nolan approved their removal on 2026-10-16, branches
   kept, and removes them himself at the shell. No agent removes them: auto mode refused it. The
   two with unlanded code are saved as `wip/2026-10-05-*` commits (INBOX, worktree clean-up).
2. **The claims registry, R0** (plans/CLAIMS_REGISTRY.md §7, plans/R0_PLAN.md): the registry, its meta-tests and the 15 S1 claims. Why: the claims the walkthrough puts on screen need a home before they are audited. Stop at R0's scope: the S2/S3 wording fixes wait for the walkthrough's pull.
   - **R0a, the scaffold, landed on 2026-10-16:** 8 agent runs and 1.74M fresh tokens. What it hands R0b is in R0_PLAN, "Carried from R0a".
   - **R0b, the 15 claims and the splits, is next:** three packages and an integrator, about 3M to 3.5M. Nolan was told it crosses the 3M line, on 2026-10-16.
3. **W1 part 2** (WALKABLE_PLAN §5; the packages are under Leftovers), then your own walkthrough of NHANES glucose under Predict, with a friction log, before Nolan sees it. Every claim that journey puts on screen is audited and registered as R1's first batch. Why: this is the walkable product, and the friction log decides what gets work next. **[Nolan]** He judges the walkthrough on screen. Then pause.

   While W1 part 2 builds, write `plans/SWARM_PLAN.md` yourself, docs only with no subagents
   (Nolan approved this on 2026-10-16), and bring it to the pause:
   - It opens with a plain-language TL;DR (his ask).
   - It settles the driver, the corpus record, dataset intake and the pilot's acceptance.
   - He answered its two questions on 2026-10-16:
     - The compute runs on his machine, under the heavy-run rules.
     - Analysts are Haiku or Sonnet agents with no cap like the build agents' (spend is still
       measured and reported).
   - Still open: whether Haiku may go below his Sonnet floor for analysts. Until he answers,
     no Haiku.
   - It requires real exploration before the lock: each attempt and its reason are logged, and
     its effect on the result is measured. A one-pass run is flagged. Nothing is searched after
     the seal.
   - It settles the analysts' model and effort by a pilot comparison.

   Why: W2–W4 build the driver and the run records, so the swarm's needs go into those waves
   rather than into a retrofit.
4. **The Sullivan 2021 replication** (Am J Prev Med 61:563). Why: the first end-to-end check of the Estimate lane. It reports the decisions the paper stated, the ones it left unstated, and how far each unstated option moves the published number.
5. **W2–W4,** each registering the claims it adds. The remaining S2/S3 fixes go in the order the walkthrough and the replication surface them.
6. **Heavy runs, each at a time Nolan sets** **[Nolan]**:
   - the T1 rerun at 600 datasets (the small-sample standard is in code, A-001);
   - the reference-journey recapture, then the materiality calibration on it.
7. **The rest of the research groundwork:**
   - the Lean pilot, on about a dozen invariance and exactness claims;
   - the stage-3 audit, with decoy set v2;
   - the Bang 2009 and Menke 2015 replications;
   - data provenance in ingest (each file's source URL and release date).
8. **The research swarm,** built to plans/SWARM_PLAN.md (step 3) once its gates (plans/CLAIMS_REGISTRY.md §8) are passed. It writes to a corpus kept apart from the registry.

### Leftovers that still hold (from the state of 2026-10-11)

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

### Open for Nolan

- Nothing is pending. On 2026-10-15 he ruled:
  - **The agent files:** `CLAUDE.md` and `.claude/agents/` are public in the repo and kept out of the release zip.
  - **Proprietary tools:** the rule covers the product only.
  - **The paper's tag** is `v1.0.0` on main; no other tag is wanted.
  - **Disputed rulings:** a ruling that evidence shows wrong is shown to him once, and he re-rules.
  - **Status and spend:** each status carries a line of cost and rework, and he is told before a step expected to exceed about 3M tokens or 25 agent runs.
  - **The pause** comes after his walkthrough.
- The venv question was the orchestrator's call: `venv-next/` for TurboTab v2, and `venv/` stays Classic's.

### Nolan's context

- **His real project is NHANES fasting glucose under Predict,** used at a fasting-draw visit; on the
  medication answers, the app recommends. The light walk gave R² 0.155 (cycle CV) and 0.18 on the
  sealed 2017–18 cycle; DIQ is not in the fixture.
- **Direction A is accepted provisionally.** He wants to judge it on a real project.
- **Strategy homework he may bring back:** whether researchers' pain is doing or defending; his
  stance on recommendations and the lock; the anchor paper; a classroom pilot.

### Heavy runs to schedule with Nolan

- The T1 rerun at 600 datasets: about 60 minutes at 4 jobs. Approved on 2026-10-11 to run after
  stage 1 lands; it still needs a time from him.
- The reference-journey recapture (the drivers now press Fit), then the materiality calibration on
  it.

## 2026-10-15 · stage 1 waiting to land

### State on 2026-10-15, checked against git

- **turbotab-next** is e6e58809 (the claims backtest), not yet pushed; origin has 914a054a.
- **feat/s1-int** (54fe6a95, worktree `.worktrees/s1-int`) holds three of stage 1's four fix
  branches (fix/s1-prediction, fix/s1-routing, fix/s1-data), the claims registry spec, the
  replication data checks and this order of work. It is not on turbotab-next yet.
- **fix/s1-inference** (6c08230e) holds B-281's fallback round (1234a7c5, eafb47c7, 6c08230e):
  a trial in which some people are separated is refused, and partial reporting is filed as v2.x.
  It is not in feat/s1-int yet, and no review verdict on the fallback is recorded in the repo.
- **docs/agent-onboarding** (stacked on feat/s1-int) holds the onboarding guidance: two
  independent revisions and a wisdom review, `notes/onboarding-audit/AUDIT.md`, `REVISION2.md` and
  `WISDOM_REVIEW.md`. Not merged.
- **The main checkout** has untracked copies of `notes/replication/`. Three match feat/s1-int;
  `estimate_candidates.md` differs. Git refuses the merge of feat/s1-int there until they are
  reconciled.

### Done since 2026-10-11

- **Stratification optimism in Lean 4** (914a054a, pushed): the exact decomposition and the
  stratification sandwich are proved, and the general law is stated (notes/lean-strat/).
- **The claims backtest** (e6e58809): about 970 decision-setting claims inventoried, and a seeded
  sample of 117 refereed blind and audited against the code. 14.5% (5.7–23.3%) carry an S1 defect,
  and 44% none. The plan, the sealed key, every verdict and the fix list are in
  notes/claims-backtest/; the lessons are in its README.
- **Stage 1, the 15 S1 defects,** each fixed test first against the old code and re-reviewed by a
  fresh agent: three rounds of fix-ups, six for B-281. The reviewers' follow-ups are in INBOX under
  "From the claims backtest". The stability-selection bug that was known red is fixed (S0). Waiting
  to land: step 1.
- **Replication data checks** (notes/replication/): Bang 2009 and Menke 2015 can be replicated
  approximately from public data. Yang 2014 cannot (its mortality-file version is gone), so Nolan
  replaced it with Sullivan 2021.
- **The onboarding guidance:** see step 2.

### Known red

- **The e2e-real job is still `continue-on-error`** until its waits are tuned to CI's runners.

### Next, in order

The goal, re-ordered 2026-10-15 at Nolan's direction so that the product is walkable sooner: Nolan
walks his own project, NHANES fasting glucose under Predict, end to end in the app (step 4); the
Estimate lane passes its first real replication (step 5); then W2–W4 finish the lanes toward
v2.0.0. Send Nolan a short status at the end of each step. Wait for him only where a step says
**[Nolan]**.

1. **Land stage 1.** Why: R0 registers the 15 S1 claims and needs their tests on the base.
   - Confirm that a fresh reviewer approved B-281's fallback round; the verdict is in the stage-1
     workflow's journal, not in the repo. Give it one review if none ran. A defect in the refusal
     itself gets fixed; a new edge of partial reporting goes to INBOX as v2.x.
   - The integrator merges fix/s1-inference into feat/s1-int, regenerates, runs the fast tier,
     pushes ci/s1-int, and you read CI.
   - Reconcile the main checkout's untracked notes/replication/, then merge feat/s1-int into
     turbotab-next and push it.
2. **Land the onboarding guidance** (docs/agent-onboarding). Why: every later wave's agents start
   from it.
   - **[Nolan]** Before the merge, he answers whether `CLAUDE.md` and `.claude/agents/` may be
     public (ruled 2026-10-15: yes, and kept out of the release zip).
   - After the merge, start a fresh session if the agent registry does not list the four roles. In
     one agent call with `agentType: 'builder'`, worktree isolation and no `model`, check three
     things: it quotes builder.md's first step; it sees one copy of CLAUDE.md, not two; and its
     system reminder gives it attribution lines. If it gets none, keep PREAMBLE_TEMPLATE's
     ATTRIBUTION field.
   - Apply the memory recommendations (the onboarding bundle in this session's scratchpad; if it is
     lost, AUDIT.md §2 S5). Strike `plans/WAVE_C6A_PLAN.md:375`, which tells builders to pip install.
     `venv-next/` was built on 2026-10-15 with uv from `turbotab/server/constraints.txt` (CI's pins);
     `scripts/py` and the hook hint use it.
3. **The claims registry, R0, in one wave** (plans/CLAIMS_REGISTRY.md §7; Nolan ruled its three
   questions on 2026-10-15): the registry, its meta-tests and the 15 S1 claims. Why: the claims the
   walkthrough puts on screen need a home before they are audited. Stop at R0's scope: the S2/S3
   wording fixes wait for the walkthrough's pull.
4. **W1 part 2** (WALKABLE_PLAN §5; the packages are under Leftovers), then your own walkthrough of
   NHANES glucose under Predict, with a friction log, before Nolan sees it. Every claim that
   journey puts on screen is audited and registered as R1's first batch. Why: this is the walkable
   product, and the friction log decides what gets work next. **[Nolan]** He judges the walkthrough
   on screen.
5. **The Sullivan 2021 replication** (Am J Prev Med 61:563, chosen by Nolan on 2026-10-15). Why: the
   first end-to-end check of the Estimate lane. It reports the decisions the paper stated, the ones
   it left unstated, and how far each unstated option moves the published number.
6. **W2–W4,** each registering the claims it adds. The remaining S2/S3 fixes go in the order the
   walkthrough and the replication surface them.
7. **Heavy runs,** each at a time Nolan sets **[Nolan]**: the T1 rerun at 600 datasets (the
   small-sample standard is in code, A-001); the reference-journey recapture, then the materiality
   calibration on it.
8. **The Lean pilot** on about a dozen invariance and exactness claims; **the stage-3 audit** with
   decoy set v2; the Bang 2009 and Menke 2015 replications; **data provenance in ingest** (each
   file's source URL and release date).
9. **The research swarm,** once its gates (plans/CLAIMS_REGISTRY.md §8) are passed. It writes to a
   corpus kept apart from the registry (ruled 2026-10-15).

### Leftovers that still hold (from the state of 2026-10-11)

- **W1 part 2** (WALKABLE_PLAN §5):
  - W1-E: the quest-log shell on live /quest (it owns `turbotab/frontend/CLAUDE.md`, which step 2
    rewrites; land step 2 first or brief its builder);
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

### Open for Nolan

- From the wisdom review (`notes/onboarding-audit/WISDOM_REVIEW.md` §6): whether a ruling of his
  that meets contrary evidence is shown to him once and re-ruled; whether each step's status should
  carry its cost and rework (agent runs, rounds, defects caught after build); and what a wave may
  spend before he is told in advance.
- On 2026-10-15 he ruled:
  - `CLAUDE.md` and `.claude/agents/` are committed to the public repo and kept out of the release zip.
  - "No proprietary tools" covers the product only.
  - The paper's tag is `v1.0.0` on main (2026-09-19); no other tag is wanted.
  - The venv drift is the orchestrator's call: `venv-next/` (pinned to `turbotab/server/constraints.txt`, as CI is) is TurboTab v2's environment, and `venv/` stays Classic's, untouched.
- Nothing is pending from the quest-log design: FOUNDATION §0 superseded the two rulings listed on 2026-10-10 (WALKABLE_PLAN §6).

### Nolan's context

- **His real project is NHANES fasting glucose under Predict,** used at a fasting-draw visit; on the
  medication answers, the app recommends. The light walk gave R² 0.155 (cycle CV) and 0.18 on the
  sealed 2017–18 cycle; DIQ is not in the fixture.
- **Direction A is accepted provisionally.** He wants to judge it on a real project.
- **Strategy homework he may bring back:** whether researchers' pain is doing or defending; his
  stance on recommendations and the lock; the anchor paper; a classroom pilot.

### Heavy runs to schedule with Nolan

- The T1 rerun at 600 datasets: about 60 minutes at 4 jobs. Approved on 2026-10-11 to run after
  stage 1 lands; it still needs a time from him.
- The reference-journey recapture (the drivers now press Fit), then the materiality calibration on
  it.

## 2026-10-11 · paused to save usage

**PAUSED on 2026-10-11 to save usage** (91% of the week's limit; it resets Thursday 2026-10-15). turbotab-next = c07e0c3e.

**Landed since 2026-10-10:**
- **C6a phases 1–4** (05325dca, ea8c7ee0, 55102da1, 37d70847):
  - the tuning engine, nested in every fit, with an honest fit count;
  - ridge, Huber, random forest and XGBoost;
  - boosted trees tuned; the elastic net on its path search with full coding;
  - the split's seed, design and units reaching every refit;
  - no family switched on by key anywhere (test_mc2's allowlist is empty).
- **W1 part 1** (c07e0c3e):
  - a held Fit can be pressed;
  - the launcher rebuilds a stale UI;
  - the drivers and browser journeys press Fit;
  - AskFor is extracted from the Record;
  - a skip-pattern blank is read as "not asked";
  - the held-out rows wait for the triage.
- **Design and methods docs:**
  - FOUNDATION §0 (Nolan's rulings: the tapestry draws meaning; "because you said");
  - calm/QUEST_LOG_CRITIQUE_2026-10-10.md;
  - calm/BEATS_MODELS_RESULTS.md (Estimate) and calm/BEATS_PREDICT.md (Predict on NHANES glucose, with my rulings in §17);
  - plans/WALKABLE_PLAN.md;
  - T1 redesigned (paired contrasts, 600 datasets);
  - the small-sample standard ruled (RECIPES §4.6);
  - notes/STRATIFICATION_OPTIMISM.md and notes/SMALL_SAMPLE_STANDARD.md;
  - V2X_SEAMS row 50 (a local conformal reading proposer).

**Known red:**
- **Fast core is intermittent** on 3 test_heldout_guarantee cases (elastic_net, ridge and screened_elastic_net with select_stability): the exactly collinear full-coded dummies trip lars_path in variable_selection.lasso_first.
- **The new e2e-real job is `continue-on-error`** until its waits are tuned to CI's runners.

**Next, in order (Thursday):**
1. ~~Fix the stability-selection collinearity bug (S)~~ — fixed on 2026-10-15 (S0, stage 1).
2. **W1 part 2** (WALKABLE_PLAN §5):
   - W1-E: the quest-log shell on live /quest;
   - W1-F: the Predict cards;
   - W1-G: Predict Results v0;
   - e2e quest-nhanes-predict and quest-own-csv;
   - then my own walkthrough of NHANES glucose under Predict, with a friction log, before Nolan sees it.
3. The small-sample standard in code (RECIPES §4.6; done on 2026-10-15, A-001), then the T1 rerun (a heavy run, about 60 minutes at 4 jobs; approved).
4. **Phase-4 leftovers:**
   - a plan_units field for the leaf cap on yes/no and class outcomes;
   - the forest and boosted trees reading "units" alike;
   - omics' TuningDecl.clause;
   - the preview lineage's coding;
   - the wide elastic net's threads in the record.
5. **W1 part 1 leftovers:**
   - tune the e2e-real waits, then drop continue-on-error;
   - mock handlers for the new hooks;
   - the seal card's chain of refusals;
   - the engine serving the held-out no-predictor baseline;
   - a ruling on the calibration intercept;
   - Cancel beside a held Fit;
   - recapture dietary-prediction.
6. **W2a/W2b:** Predict built to BEATS_PREDICT, then W3/W4 Estimate built to BEATS_MODELS_RESULTS.
7. **Heavy:** the reference-journey recapture (the drivers now press Fit), then the materiality calibration on it.

**Nolan's context:**
- **His real project is NHANES fasting glucose under Predict,** used at a fasting-draw visit; on the medication answers, the app recommends. The light walk gave R² 0.155 (cycle CV) and 0.18 on the sealed 2017–18 cycle; DIQ is not in the fixture.
- **Direction A is accepted provisionally.** He wants to judge it on a real project.
- **Strategy homework he may bring back:** whether researchers' pain is doing or defending; his stance on recommendations and the lock; the anchor paper; a classroom pilot.

## 2026-10-10 · sprint active

**State on 2026-10-10 (~06:20 UTC), sprint active.**

**Landed tonight:**
- **E1q (e7bd40ef):** stacked NHANES cycles with design-based trends, trials, case-control samples and matched sets, the specification curve and leave-one-site-out validation. These are engine-only; routing them falls to E1, D1 and C1.
- **E1c (this merge):**
  - the outcome gate on every data route (Predict's seal included);
  - F12;
  - Surfaceable and the path fuzzer;
  - materiality, its ledger and the triage;
  - plain words on the cards.
  - Combined with E1q, the full suite passed 4,431 of 4,432 tests. The one failure was the participant-flow test's known flake, fixed in 6fa7c23c.

**The quest-log design** is on `design/quest-models-stage` @ 4ea11554, NOT merged. It was shown to Nolan as https://claude.ai/artifact/U7FKBERAxsf2NkKGwDiWgB. Two rulings are pending:
- the card's word budget, against a question's own options;
- the Recommended wording for observational designs, where "an association" is recommended.

**The next wave** follows `plans/WAVE_C6A_PLAN.md`, which ends with my rulings in §7.
- **Phase 1:** RT-1a, then MC-2b-1; SRC; Q-ad (triage dispositions and the invariance instrument); Q-b (the substitution sentence and the split sentence); Q-c (progress after Fit and the §7.2 wording).
- **Phases 2–4:** RT-1b, then the families, then RT-5a, RT-5f and MC-2b.

**Open for Nolan, from E1c:** I11's forward steps, which I ruled in plan §7.3.

**Heavy runs to schedule with Nolan:**
- the reference-journey recapture: the materiality calibration has 9 proof cases, and rows × SMD under-warns band 2;
- T1 and T12.

**State on 2026-10-10 (~03:15 UTC), sprint active.** `turbotab-next` = `0b807a5e`, GREEN on both CI tiers (fast and full, 10 jobs). Landed since 2026-10-08:
- **E1a (5bcafd14):**
  - the model-family contract declarations, MC-1/MC-2a, with the syntax-tree no-switch test;
  - the stage registry, P0.4 (`core/quest.py`).
- **The experiment winners (d0ae80ac):**
  - the decision-log format marker and migrations;
  - the causal learner key renames;
  - the DML classifier fix.
- **The ten view kinds (90d7d4fd):** `src/components/views`, `/lab/views`.
- **E1p (683a3224):** Bland–Altman, subgroup clustering and dietary-pattern cores; the citation registry and `refs.bib` with Crossref-verified DOIs.
- **E1b (0b807a5e):**
  - Fit records the lock, and nothing is served before it;
  - the F11 leak is closed, with the held-out pipeline guarantee test;
  - the Confirm sweep and record list (P0.5);
  - ordering fixes, part of P0.6;
  - pinned deps and constraints, with CI split into a fast tier and a full tier (`ci/full-*`).
- **Docs:**
  - `SURFACING_POLICY.md`: materiality M(i,s), the ledger, the tier function and the two-phase triage;
  - `audit/EXTERNAL_AUDIT_2026-10-09.md`;
  - `MODEL_FAMILY_CONTRACT.md`, `V2X_SEAMS.md`, the crosswalk with Nolan's rulings, `RECIPES_AND_TUNING.md` draft 3, and FOUNDATION rewritten for the quest log.

**Running at compaction:**
- **E1c (wf_0ad98ab7-b11):**
  - the outcome-beside-a-column gate on the data routes;
  - F12 (the imbalance wrapper's early stopping);
  - Surfaceable (executable fires, reads and holds) plus the path fuzzer;
  - materiality plus the ledger, with the Confirm sweep moved onto M and the NHANES 3-noticing proof;
  - plain words.
- **E1q (wf_a7a411c8-f80):** stacking cycles plus trends; trials; case-control and matched sets; the specification curve plus leave-one-site-out validation. Its integrator is fixing a test_stage_registry label mismatch for the new contracts.
- **The quest-log design (wf_d50f5b62-3fc):** static Models-stage screens on branch `design/quest-models-stage`, reviewed by Fable. Screenshots go to the session scratchpad `quest-shots/`, and must be shown to Nolan.

**Landing protocol:**
- Read EVERY verifier problem, not just the verdict.
- Confirm CI myself through the public check-runs API.
- Regenerate METHODS_REFERENCE and the packets (with --reuse) when they conflict.

**Subagent policy:**
- Opus for statistics and safeguards; Sonnet only for plumbing, with an Opus verifier; never below Sonnet.
- Never use pkill.
- Repair whenever any non-note problem is found.

**Next:**
- C6a: the new families plus the RT-1 tuning engine with `TuningPlan.strategy`;
- MC-2b;
- the rest of P0.6 (set_missing/set_fill split, substitution before Fit, and more);
- D1: Describe wiring (DESCRIBE_LABELS to move into the contracts);
- E1: design routing;
- C7a: exhibit recording;
- X1/X2: the manuscript model and LaTeX;
- P0.7: the shell, after Nolan reviews the design;
- P0.9: noticings, family by family, on materiality;
- the verifier replay benchmark (approved);
- regenerate the review packets at a clean commit.

**Rulings since 2026-10-08:**
- the triage sweep at the lock (replacing "decide all");
- non-journey noticings (about 138) deferred; Describe, experiments, Word and the contract intelligence stay in 2.0;
- the outcome column may show in Your data;
- RECIPES draft 3's four settlements approved;
- Classic tests may be edited, but not Classic app code;
- the DoD is delegated to the orchestrator and frozen (correctness changes only);
- the planned road is kept, not a slice-first change;
- Nolan is fine with the CI failure emails.

## 2026-10-06 · paused after the platform fixes; the design rulings of 2026-10-06 to 10-08

**State (2026-10-06, paused by Nolan after the platform fixes).** `turbotab-next` is at the merge that adds the design docs (on top of `9c349c4e`). The last CI run (`9c349c4e`) is green on all four jobs: Linux core and server tests, the Docker server image, and the macOS and Windows launchers.

**The engine (done and independently verified).** Waves 1, 2a, 2b and 2c and their repairs cover nearly every method in `V2_DEFINITION_OF_DONE.md`:
- data in: XPT, codebooks, joins;
- survey across families;
- multiple imputation compatible with the analysis model;
- prediction validation;
- omics: QC-RLSC and ComBat;
- scales;
- NCI usual intake;
- estimands and Table 2;
- causal ML and time-varying exposures;
- explainability;
- regression calibration;
- functional form;
- Explore;
- multiclass substitution;
- the leash;
- previews for every number-changing kind;
- the export bundle with replay.

The release track also landed: the legacy app is retired, with its references in `reference/`; the one-command launcher; server mode with sign-in; Docker; the generated methods reference and the five expert review packets.

**The design direction (Nolan's rulings, 2026-10-05).**
- Calm over complete: `calm/FOUNDATION.md` sets the five-second test, two registers, desktop only, the color roles and the canvas grammar.
- Four synchronized structures sit on branch `explore/calm`. On 2026-10-06 Nolan picked the quest log as the best so far, to be designed further (below).
- Exploration is a guide plus a gallery, walked on the first visit.
- Noticings become threads that return at later stages.

**The structure (Nolan, 2026-10-06): the quest log, redesigned.**
- Hover to see the questions you need to answer.
- The sections and questions are built dynamically from what the engine needs from this analysis.
- An overall progress bar sits at the top right, so users are not discouraged.
- First comes a crosswalk of what the engine must surface for the user to decide, and at what moment. More ideas will come from it.
- His framing, verbatim. Before the fit: "see what the engine needs to surface that a user decides, dynamically build the tapestry to show the consequences of their decision-making". After training, it pivots to: "show them the results and help them interpret with the tapestry, let them decide what to include and not include in their manuscript results section."
- **The orchestrator's methods floor on curating results:**
  - under inference, the locked plan's primary analysis is always reported;
  - what is left out of the results section goes to the supplement and stays in the record.

  This keeps the curation from becoming selective reporting.
- `calm/FOUNDATION.md` was rewritten for the quest log on 2026-10-08 (merged with the calm kit at 0fc1d57c). BLUEPRINT §11 and §11.4 still describe the older objective list by guideline section; follow FOUNDATION where they differ.

The magic is designed in `understanding/UNDERSTANDING_LAYER.md`: 367 threads in 17 families, with sentinels and a coverage test. It is **not built yet**; about one thread in five has engine pieces.

**Approved and not yet built** (the DoD amendment, 2026-10-05):
- the families ridge, Huber, random forest and XGBoost;
- Table 1;
- stacking NHANES cycles;
- nested tuning;
- explanations, evaluation and robustness on screen;
- Classic's missing exploration views.

`RECIPES_AND_TUNING.md` specifies the recipes, overrides and tuning. Its draft 3 must fold in Nolan's rulings.

**Ruled by Nolan (2026-10-06):**
1. **`RECIPES_AND_TUNING.md` "Decisions for Nolan":**
   - the trees "Try both" by default, keeping blanks or taking the fill, chosen in each training fold;
   - fits over about 2 minutes wait for Fit;
   - three scope-cut groups stay in v2: faster search for big tables, more preprocessing options and the inference extensions. Only the kept-comparisons group goes to INBOX: standard settings as a kept version, trunk versions and re-tuned bands.
2. **The NHANES export:** commit it **gzipped** as a test fixture. The harness reads the `.gz`, and CI then runs the 42 test files that read it.
3. **The structure:** the quest log, redesigned (above).
4. **`UNDERSTANDING_LAYER.md` §7:**
   - open noticings go through the TRIAGE SWEEP (Nolan, 2026-10-09, replacing the ruling of 2026-10-06): before the lock, or under Predict before the held-out rows open, every open noticing arrives with the engine's recommended disposition pre-filled, shown with its reason: "doesn't change your numbers here" → a quiet supplement line; "could bias the estimate" → a limitation sentence; "act on it" → points to the decision. Blockers must be resolved. The user reviews, changes any line, and presses Confirm all; every disposition is recorded. A limitation sentence appears only where the issue could move the estimate and nothing was done about it (this replaced "decide or dismiss each before the lock");
   - clean checks go in the supplement;
   - "noticings" are named in two places.

**UI and scope discussion with Nolan (2026-10-06/07).** The full notes are in the orchestrator's memory, `turbotab-quest-log-tapestry`.
- **Seven fixed stages:**
  1. Your data (the file, the domain, what each column is)
  2. Your question (the outcome, then the goal, worded around it)
  3. First look (shaped by the goal)
  4. Who's in
  5. Models
  6. Results
  7. Write-up

  The questions inside each stage are dynamic. The progress bar has seven segments: an unreached stage shows empty, and a reopened stage drops back and says why.
- **Labels:** Decide · Confirm · For the record. The engine's terms asked/stated/silent never reach the UI.
  - Each stage ends with one Confirm sweep of its defaults.
  - Hovering a line elaborates slightly. Clicking it expands the card to its options, with a way back.
- **Results** are exhibits: a figure or table with its caption, the finding, and the interpretation. For each:
  - choose among drafted wordings, or write your own;
  - choose a placement: Results, Discussion, Supplement, or leave the exhibit out. Every analysis stays listed.
- **Full-tapestry flowcharts,** each savable as a figure:
  - the participant flow, or samples and features for omics;
  - the analysis flowchart before training, where the Fit button lives.
- **Export:** an Overleaf-ready LaTeX zip and Word from one manuscript model. Every citation is checked against Crossref. Classic's `ml/latex_report.py` and `ml/manuscript_validator.py` are references.
- **Goals:** Describe / Estimate an effect / Predict. A second question picks the shape, filtered by domain. Several goals in one paper run as sequential tracks over the shared stages, with Write-up merging them.
- **Added to v2:**
  - dietary patterns;
  - clustering into subgroups;
  - Bland–Altman, both between methods and between models.
- **Designed experiments added to v2** (2026-10-07) as one named milestone after the core slices:
  - parallel and cluster-randomized trials;
  - design declaration and routing (no confounder hunt; precision adjustment);
  - no exclusions after randomization; intention-to-treat and per-protocol analysis sets;
  - the CONSORT checklist and flow diagram;
  - causal wording;
  - missing-outcome sensitivity analyses.

  Case-control, with restrictions, and matched sets, with conditional logistic regression, ride with the milestone.

  Deferred to v2.x: crossover trials, repeated-measures trial models, and per-protocol and complier effects.
- **Mediation is out of v2.** Mediators are still recognized, so they stay out of the adjustment set.
- **Ruled (2026-10-07):** not every option on a design question must be supported. A rare design may carry "Not available yet", with its reason and an exit that keeps the work.
- **The definition of done was amended on 2026-10-07** with all of the above. See `V2_DEFINITION_OF_DONE.md`.
- **Plain words:** "exposure", "confounder" and "estimand" stay quiet terms only. The card asks "What do you think affects glucose?"
- **Validation of `319a513f`** (2026-10-06):
  - Python 2,848 passed;
  - frontend 201 and the calm kit 242 passed;
  - mock browser tests: 1 failure in the setup (`m2-journeys` crashes on the mock server's health check before it can skip);
  - Classic: 49 failed and 3,245 passed. Most look like repository self-checks that the release track broke. Triage is pending; the run hangs under xdist without `--timeout`.

  CI does not run the browser tests.

**The crosswalk (2026-10-08)** is in `crosswalk/`: 708 items placed across the seven stages, with 21 order conflicts and their fixes (audited against the display-order rule on 2026-10-08, then revised after review), and the gaps.
- **`SIZING.md`:** about 23% done by effort, 734 units remaining, 10 to 16 weeks.
- **Nolan ruled its six questions:**
  1. the outcome beside a column is hidden until the lock under Estimate;
  2. Your data is a ledger plus Confirm;
  3. Describe can have no single outcome;
  4. the noticings that fire on the reference journeys ship in 2.0, plus blockers and family checks; the rest are deferred (re-ruled 2026-10-09);
  5. blanks are split at the stage line;
  6. a shared-step change keeps a read-only earlier row.
- **UI rulings, also 2026-10-08:** the rail and Write-up are one document in two views; a long fit runs as a server job with notification; a change after results is a kept version (Predict) or a secondary (Estimate), and the label "revised after first results" appears only in the comparison and the methods text.

**Engine follow-ups found (not started):**
- the energy preview that quotes a coefficient before the lock;
- the exposure and estimand previews that draw nothing;
- the metabolomics zero-row crash;
- the elastic net's penalty, which is platform-dependent (a methods decision: a tighter tolerance or a one-standard-error rule);
- the speed targets (DoD gate 5), which need a quiet machine;
- the final re-audit.
See INBOX.md.

**Next, when resumed.**
1. Commit the NHANES fixture gzipped, and confirm CI runs the real-data tests.
2. Run the crosswalk: every decision, noticing and result the engine surfaces, when it surfaces, what it depends on, and what the tapestry shows. It is the input to the quest-log redesign.
3. Write `RECIPES_AND_TUNING.md` draft 3 with the rulings.
4. Build in vertical slices: each is an engine thread, its quest-log screen and a drive. Start with the prototype plan's phase 0 (`UNDERSTANDING_LAYER.md` §6), together with the shelf and tuning work.

---

## History: the state on 2026-10-01

**State (2026-10-01): M1 is done and tagged `next-m1`.** The production app runs the full NHANES
journey in a real browser: the Record with its card-based decisions, the pipeline banner, and the
stage. The stage has the transform player (the flip plays each method's real storyboard), previews
of every option on the user's own data, finding evidence, savable figures in journal style with
provenance captions, and the Results (models against a baseline, coefficients, and substitution
curves with a refit uncertainty band). Checks: 444 Python tests, 100 frontend tests, and the
m1-journey Playwright spec against a real server. Run records: `m1/w1-result.json` and
`m1/w2-result.json`. Screens: `m1/screens/`.

**Next: M2** (BLUEPRINT §9). It covers the opening sequence for all five lenses (orientation,
grain and repeats, eligibility, the seal), findings with preview-before-apply and deferral, and the
wide omics data path, benchmarked. Start with the "M2-first" and "[M2]" items in INBOX.md (the energy
card's word budget first). Then triage the rest of that file, merge `main` in, write M2_CONTRACT.md,
and launch.

---

## History: the pause of 2026-09-27 (mid-M1)

Paused at Nolan's request (weekly usage limit). Everything is committed and pushed. Start here.

## Where it stands

- **M0 is done** and tagged `next-m0`.
- **M1 part 1 is done and merged** on `turbotab-next`: validators, the server-side interview
  Router, column roles, the participant flow, the sealed split, the generic consequence diff, and
  previews for every M1 question. Also three model families, in-fold pipelines, fit, substitution
  curves, publishable decision sentences, teaching for all 11 questions, pack proposals, and
  findings that name their lever. Workers now spawn on demand.
- **Checks:** 395 Python tests and 27 frontend tests pass.
- **The live NHANES journey** runs end to end through the API in about 10 s. Fit takes about 2.5 s,
  and previews take 1–30 ms. Details are in `m1/w1-result.json` → `server.measurements`.
- **M1 part 2 has not started.** That's the frontend: the Record and the pipeline panel rebuilt on
  the chosen design, then integration, two reviewers, and fixes.

## The design decision — mostly made (see BLUEPRINT §11.1)

Nolan ruled from the screenshots: `stage` is the base; no per-option preview thumbnails; `stage`'s lineage diagram is canonical; prototype a **pipeline banner** (the whole pipeline as a compact strip at the top, everything below is the working window). **The slider then became a transform player** (BLUEPRINT §11.1): one two-state *now ⇄ with this choice* flip that plays the method's own labeled storyboard as it flips (forward, or in reverse on the way back; brisk; interruptible) and drives every view; step dots pause on any step; switching options morphs directly between results without replaying the storyboard; saves capture any labeled state. Backend work this adds: consequence views gain optional `steps`, and the energy-adjustment builder supplies residual/density/partition storyboards. **The design is fully ruled — build it.**

### Background: the three prototypes

Three prototypes answer the wall-of-text problem (BLUEPRINT §11) on the same real NHANES fixture.
Branches are on origin:

| Branch | Angle | Visible words (S1 preview) |
|---|---|---|
| `explore/stage` | Hovering or focusing an option morphs the pipeline panel into its consequence: a primary before/after view, plus lineage and distribution | 269 |
| `explore/inline` | Every option card carries a sparkline preview; the focused card enlarges in a fixed stage under the strip; a pin compares two options | 181 |
| `explore/scrub` | One before→after transformation per option, scrubbed; table cells roll like an odometer; sidenotes | 243 |

Screenshots, frame strips, word counts and each designer's rationale and weaknesses are in each
branch under `docs/turbotab-next/m1/explore/<angle>/`, and in `m1/w1-result.json` → `explore`.

**Nolan has seen `stage/s1-density-light.png` and said:** *"I really like the screenshot you sent me.
It's a really good and promising design style."* Make `stage` the base. Graft in what the others do
better, if it survives a side-by-side look:

- inline's compare pin and its lower word count;
- scrub's column-identity morph in the table (`fat_total` → `fat_total_adj` → `fat_total_per_kcal`)
  and its sidenotes.

Also fix the weaknesses the designers named:

- Stage's first-preview transition snaps.
- Histogram bins morph across different units. Crossfade when the unit changes, and morph only
  within the same unit.
- Hover-to-preview needs a touch equivalent.

## Found by the live journey — carry these into M1 part 2

1. **Complete cases drop 86% of NHANES.** `meds_hbp` and `meds_chol` are blank for 18,405 rows, and
   a blank there means "not asked" (DRIVE_RUBRIC §4). The missing-values question must say so
   through its preview, and should offer leaving those columns out, before the user records
   "complete cases".
2. **Nested nutrients distort substitution.** The fat_total → carb curve holds `fat_sat`, `fat_mon`
   and `fat_poly` fixed while `fat_total` moves. The design warns about it, but roles should mark
   nested components (sugar ⊂ carb, fat subtypes ⊂ fat_total), and substitution pairs should
   respect them.
3. **Boosted trees score below the mean baseline** (CV R² −0.04) with no concern attached. The shelf
   and results should say when a family underperforms the baseline.
4. **The linear substitution band has zero width.** Resampling rows through one fitted linear model
   cannot vary its average effect. The band needs a refit bootstrap (PRODUCT_VISION §06c, mark 2).

The other 36 notes from this run are in `INBOX.md` under "From M1 workflow 1", untriaged.

## A framing Nolan added at the pause

*Modeling decision provenance* — his name for a reproducibility problem in nutrition research:
readers can't reconstruct which choices turned raw data into a model's inputs. He noted the lineage
diagram illustrates it well. It is now North star item 4 in the BLUEPRINT. It raises the lineage
and the row flow from teaching aids to deliverables: M1 part 2 should make them publication-quality;
M5's export carries them as figures plus a replayable provenance record.

## Next session, in order

1. Look at the three prototypes side by side (their screenshots and frames). Rule on the design and
   record the ruling in BLUEPRINT §11.
2. Merge the chosen prototype's reusable components and `explore/fixtures` into `turbotab-next`.
3. Launch M1 workflow 2 on that design and the four points above: the frontend Record (questions
   from the Router, previews, teaching drawer, findings with levers), the pipeline panel (Rows flow,
   Columns lineage, Results with metrics and substitution curves), integration, two reviewers
   (function, and taste against DRIVE_RUBRIC), and a fixer.
4. Drive it with Playwright on the real NHANES export, triage the inbox, merge `main` in, and tag
   `next-m1`.
