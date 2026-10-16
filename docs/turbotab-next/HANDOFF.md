# Handoff

The current state only. When a new state replaces this one, the replaced body goes, verbatim and
under a dated heading, to the top of `HANDOFF_HISTORY.md`, which holds every earlier state and the
rulings recorded in them. Rules live in the root `CLAUDE.md` and in `ORCHESTRATOR.md`; rulings live
in the documents they amend.

## State on 2026-10-16, checked against git

- **turbotab-next** is the stage-1 merge (feat/s1-int, fast-forwarded and pushed on 2026-10-16). It holds:
  - stage 1's four fix branches;
  - the claims registry spec, ruled;
  - the replication data checks;
  - the onboarding guidance: the root `CLAUDE.md`, `.claude/agents/`, `ORCHESTRATOR.md` and `scripts/py`.
- **Environments:**
  - `venv-next/` in the main checkout is TurboTab v2's environment, built with uv from `turbotab/server/constraints.txt` (CI's pins); `scripts/py` runs it from any worktree.
  - `venv/` is Classic's, untouched.
  - The fast tiers pass under `venv-next`: server 189, core 3,012, models 34, plus the trials and missing-data acceptance files.
- **CI:** read it on turbotab-next's head through the public check-runs API before you rely on it. The last full tier, on ci/full-s1-int before the inference branch merged, failed only test_wp7's boosted-trees pin, which has since been re-pinned and passes under CI's versions.

## Done since 2026-10-11

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

## Known red

- **The e2e-real job is still `continue-on-error`** until its waits are tuned to CI's runners.

## Next, in order

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
2. **The claims registry, R0, in one wave** (plans/CLAIMS_REGISTRY.md §7): the registry, its meta-tests and the 15 S1 claims. Why: the claims the walkthrough puts on screen need a home before they are audited. Stop at R0's scope: the S2/S3 wording fixes wait for the walkthrough's pull.
3. **W1 part 2** (WALKABLE_PLAN §5; the packages are under Leftovers), then your own walkthrough of NHANES glucose under Predict, with a friction log, before Nolan sees it. Every claim that journey puts on screen is audited and registered as R1's first batch. Why: this is the walkable product, and the friction log decides what gets work next. **[Nolan]** He judges the walkthrough on screen. Then pause.

   While W1 part 2 builds, write `plans/SWARM_PLAN.md` yourself, docs only with no subagents
   (Nolan approved this on 2026-10-16), and bring it to the pause:
   - It opens with a plain-language TL;DR (his ask).
   - It settles the driver, the corpus record, dataset intake and the pilot's acceptance.
   - It asks him two things: where the compute runs, and the token budget.

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

- Nothing is pending. On 2026-10-15 he ruled:
  - **The agent files:** `CLAUDE.md` and `.claude/agents/` are public in the repo and kept out of the release zip.
  - **Proprietary tools:** the rule covers the product only.
  - **The paper's tag** is `v1.0.0` on main; no other tag is wanted.
  - **Disputed rulings:** a ruling that evidence shows wrong is shown to him once, and he re-rules.
  - **Status and spend:** each status carries a line of cost and rework, and he is told before a step expected to exceed about 3M tokens or 25 agent runs.
  - **The pause** comes after his walkthrough.
- The venv question was the orchestrator's call: `venv-next/` for TurboTab v2, and `venv/` stays Classic's.

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
