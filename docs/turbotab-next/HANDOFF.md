# Handoff

**State on 2026-10-15.** turbotab-next = the stage-1 merge (feat/s1-int).

**Landed since 2026-10-11:**
- **Stratification optimism in Lean 4** (914a054a): the exact decomposition and the stratification sandwich are proved, and the general law is stated — notes/lean-strat/.
- **The claims backtest** (e6e58809): about 970 decision-setting claims inventoried, a seeded sample of 117 refereed blind and audited against the code. 14.5% (5.7–23.3%) carry an S1 defect, and 44% none. The plan, the sealed key, every verdict and the fix list are in notes/claims-backtest/; the lessons are in its README.
- **Stage 1, the 15 S1 defects** (this merge of fix/s1-prediction, fix/s1-inference, fix/s1-routing and fix/s1-data). Each fix starts with a test that fails on the old code, and each was re-reviewed by a fresh agent until approved: three rounds of fix-ups, four for B-281. The reviewers' follow-ups are in INBOX under "From the claims backtest".
  - The known-red stability-selection bug is fixed (S0): exact copies and full codings are found before the lasso path, a coding's spare level is left off only where the coding ties, and the result no longer depends on level order in the reviewer's known-truth test.
- **Replication data checks** (notes/replication/): Bang 2009 and Menke 2015 can be replicated approximately from public data. Yang 2014 cannot (its mortality-file version is gone), so Nolan chose to replace it with a continuous-NHANES Estimate paper; the candidates are in estimate_candidates.md.

**Known red:**
- **The e2e-real job is still `continue-on-error`** until its waits are tuned to CI's runners.

**Next, in order:**
1. **The claims registry spec,** for Nolan's review, before W1 part 2. Every methods rule, threshold and teaching sentence becomes a declared object (schema, basis, grade, test) that code reads; its sentences are generated from the parameters actually used. The 39 S2 and 16 S3 fixes wait for it, so their sentences are not hand-patched twice. A ruling stays "proposed" until its code and test exist.
2. **W1 part 2** (WALKABLE_PLAN §5, unchanged from the list below), then my walkthrough of NHANES glucose under Predict, with a friction log, before Nolan sees it.
3. **W2–W4,** each registering the claims it adds and absorbing the S2/S3 fixes in its area. At each wave's end, ask the builder for its three most doubtful claims.
4. **Heavy runs, scheduled with Nolan:** the T1 rerun at 600 datasets (the small-sample standard is now in code, A-001); the reference-journey recapture, then the materiality calibration on it.
5. **The Lean pilot:** about a dozen invariance and exactness claims ("doesn't change your numbers"), each theorem's hypotheses checked at run time as predicates, the app asserting only when every one holds.
6. **The stage-3 audit** of the remaining claims through the registry, with decoy set v2 (16 decoys and 5 controls, approved by Nolan, sealed outside the repo; never shown to an agent with its key, and fork agents are never referees).
7. **The replication pilot:** Bang 2009 (Predict), Menke 2015 (Describe) and an Estimate paper. Each replication reports the decisions the paper stated, the ones it left unstated, and how far each unstated option moves the published number.
8. **Data provenance in ingest:** record each file's source URL and release or revision date, not only its hash.
9. Then the research swarm, once its gates are passed.

The 2026-10-11 lists below still hold for the phase-4 and W1 part-1 leftovers.

**Waiting on Nolan:** thumbs up or down on Sullivan 2021 (Am J Prev Med 61:563) as the Estimate paper.

---

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
