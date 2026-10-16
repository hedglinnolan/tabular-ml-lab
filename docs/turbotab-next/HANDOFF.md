# Handoff

The current state only. When a new state replaces this one, the replaced body is appended, dated and
verbatim, to `HANDOFF_HISTORY.md`, which holds every earlier state and the rulings recorded in them.
Rules live in the root `CLAUDE.md` and in `ORCHESTRATOR.md`; rulings live in the documents they amend.

## State on 2026-10-15

turbotab-next = the stage-1 merge (feat/s1-int).

## Landed since 2026-10-11

- **Stratification optimism in Lean 4** (914a054a): the exact decomposition and the stratification sandwich are proved, and the general law is stated — notes/lean-strat/.
- **The claims backtest** (e6e58809): about 970 decision-setting claims inventoried, a seeded sample of 117 refereed blind and audited against the code. 14.5% (5.7–23.3%) carry an S1 defect, and 44% none. The plan, the sealed key, every verdict and the fix list are in notes/claims-backtest/; the lessons are in its README.
- **Stage 1, the 15 S1 defects** (this merge of fix/s1-prediction, fix/s1-inference, fix/s1-routing and fix/s1-data). Each fix starts with a test that fails on the old code, and each was re-reviewed by a fresh agent until approved: three rounds of fix-ups, four for B-281. The reviewers' follow-ups are in INBOX under "From the claims backtest".
  - The known-red stability-selection bug is fixed (S0): exact copies and full codings are found before the lasso path, a coding's spare level is left off only where the coding ties, and the result no longer depends on level order in the reviewer's known-truth test.
- **Replication data checks** (notes/replication/): Bang 2009 and Menke 2015 can be replicated approximately from public data. Yang 2014 cannot (its mortality-file version is gone), so Nolan chose to replace it with a continuous-NHANES Estimate paper; the candidates are in estimate_candidates.md.
- **The onboarding audit** (branch docs/agent-onboarding): a root `CLAUDE.md` for every agent, four agent definitions in `.claude/agents/`, `ORCHESTRATOR.md`, this file split from `HANDOFF_HISTORY.md`, and the audit in notes/onboarding-audit/.

## Known red

- **The e2e-real job is still `continue-on-error`** until its waits are tuned to CI's runners.

## Next, in order

Re-ordered 2026-10-15 at Nolan's direction: the product walkable sooner.

1. **Land stage 1:** B-281's fallback round, then merge fix/s1-inference and push turbotab-next.
2. **The onboarding audit:** an independent Fable audit and revision of the base instructions agents get when they join the project, as orchestrator or executor. Done on docs/agent-onboarding; merge it, then apply notes/onboarding-audit's memory recommendations.
3. **The claims registry, R0, in one wave** (plans/CLAIMS_REGISTRY.md; Nolan ruled its three questions on 2026-10-15): the registry, its meta-tests and the 15 S1 claims.
4. **W1 part 2** (WALKABLE_PLAN §5; the package list is under Leftovers below), then my walkthrough of NHANES glucose under Predict, with a friction log, before Nolan sees it. Every claim that journey puts on screen is audited and registered as R1's first batch.
5. **The Sullivan 2021 replication** (Am J Prev Med 61:563, chosen by Nolan on 2026-10-15), as the first end-to-end check of the Estimate lane. It reports the decisions the paper stated, the ones it left unstated, and how far each unstated option moves the published number.
6. **W2–W4,** each registering the claims it adds. The remaining S2/S3 fixes go in the order the walkthrough and the replication surface them. At each wave's end, the builder names its three most doubtful claims.
7. **Heavy runs, scheduled with Nolan:** the T1 rerun at 600 datasets (the small-sample standard is now in code, A-001); the reference-journey recapture, then the materiality calibration on it.
8. **The Lean pilot** on about a dozen invariance and exactness claims; **the stage-3 audit** with decoy set v2 (sealed outside the repo, never shown to an agent with its key, and fork agents are never referees); the Bang 2009 and Menke 2015 replications; **data provenance in ingest** (each file's source URL and release date).
9. Then the research swarm, once its gates are passed. It writes to a corpus apart from the registry (ruled 2026-10-15).

## Leftovers that still hold (from the state of 2026-10-11)

- **W1 part 2** (WALKABLE_PLAN §5):
  - W1-E: the quest-log shell on live /quest;
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
  - a ruling on the calibration intercept;
  - Cancel beside a held Fit;
  - recapture dietary-prediction.
- **After W1:** W2a/W2b, Predict built to BEATS_PREDICT, then W3/W4 Estimate built to BEATS_MODELS_RESULTS.

## Open for Nolan

- Nothing is pending from the quest-log design: the two rulings listed on 2026-10-10 (the card's word budget against a question's own options; the Recommended wording for observational designs) were superseded by FOUNDATION §0 on 2026-10-10 (WALKABLE_PLAN §6).
- A ruling on the calibration intercept (W1 part 1 leftovers).
- From the onboarding audit (notes/onboarding-audit/AUDIT.md §6): whether a root `CLAUDE.md` and `.claude/agents/` may ship in the public repository and its release zip, and whether the `v1.0-paper` tag is still wanted.

## Nolan's context

- **His real project is NHANES fasting glucose under Predict,** used at a fasting-draw visit; on the medication answers, the app recommends. The light walk gave R² 0.155 (cycle CV) and 0.18 on the sealed 2017–18 cycle; DIQ is not in the fixture.
- **Direction A is accepted provisionally.** He wants to judge it on a real project.
- **Strategy homework he may bring back:** whether researchers' pain is doing or defending; his stance on recommendations and the lock; the anchor paper; a classroom pilot.

## Heavy runs to schedule with Nolan

- The T1 rerun at 600 datasets (about 60 minutes at 4 jobs; approved on 2026-10-11, to run after the stage-1 merge).
- The reference-journey recapture (the drivers now press Fit), then the materiality calibration on it.
