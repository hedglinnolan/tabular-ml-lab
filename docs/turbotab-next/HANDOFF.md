# Handoff

The current state only. When a new state replaces this one, the replaced body goes, verbatim and
under a dated heading, to the top of `HANDOFF_HISTORY.md`, which holds every earlier state and the
rulings recorded in them. Rules live in the root `CLAUDE.md` and in `ORCHESTRATOR.md`; rulings live
in the documents they amend.

## State on 2026-10-15, checked against git

- **turbotab-next** is e6e58809 (the claims backtest), not yet pushed; origin has 914a054a.
- **feat/s1-int** (54fe6a95, worktree `.worktrees/s1-int`) holds three of stage 1's four fix
  branches (fix/s1-prediction, fix/s1-routing, fix/s1-data), the claims registry spec, the
  replication data checks and this order of work. It is not on turbotab-next yet.
- **fix/s1-inference** (6c08230e) holds B-281's fallback round (1234a7c5, eafb47c7, 6c08230e):
  a trial in which some people are separated is refused, and partial reporting is filed as v2.x.
  It is not in feat/s1-int yet, and no review verdict on the fallback is recorded in the repo.
- **docs/agent-onboarding** (stacked on feat/s1-int) holds the onboarding guidance: two
  independent revisions, `notes/onboarding-audit/AUDIT.md` and `REVISION2.md`. Not merged.
- **The main checkout** has untracked copies of `notes/replication/`. Three match feat/s1-int;
  `estimate_candidates.md` differs. Git refuses the merge of feat/s1-int there until they are
  reconciled.

## Done since 2026-10-11

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

## Known red

- **The e2e-real job is still `continue-on-error`** until its waits are tuned to CI's runners.

## Next, in order

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

## Leftovers that still hold (from the state of 2026-10-11)

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

## Open for Nolan

- Nothing pending. On 2026-10-15 he ruled:
  - `CLAUDE.md` and `.claude/agents/` are committed to the public repo and kept out of the release zip.
  - "No proprietary tools" covers the product only.
  - The paper's tag is `v1.0.0` on main (2026-09-19); no other tag is wanted.
  - The venv drift is the orchestrator's call: `venv-next/` (pinned to `turbotab/server/constraints.txt`, as CI is) is TurboTab v2's environment, and `venv/` stays Classic's, untouched.
- Nothing is pending from the quest-log design: FOUNDATION §0 superseded the two rulings listed on 2026-10-10 (WALKABLE_PLAN §6).

## Nolan's context

- **His real project is NHANES fasting glucose under Predict,** used at a fasting-draw visit; on the
  medication answers, the app recommends. The light walk gave R² 0.155 (cycle CV) and 0.18 on the
  sealed 2017–18 cycle; DIQ is not in the fixture.
- **Direction A is accepted provisionally.** He wants to judge it on a real project.
- **Strategy homework he may bring back:** whether researchers' pain is doing or defending; his
  stance on recommendations and the lock; the anchor paper; a classroom pilot.

## Heavy runs to schedule with Nolan

- The T1 rerun at 600 datasets: about 60 minutes at 4 jobs. Approved on 2026-10-11 to run after
  stage 1 lands; it still needs a time from him.
- The reference-journey recapture (the drivers now press Fit), then the materiality calibration on
  it.
