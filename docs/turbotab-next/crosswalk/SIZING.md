# Sizing: the road to v2.0.0

How much is left against the amended definition of done (`V2_DEFINITION_OF_DONE.md`, amended 2026-10-07), as work packages in the road's order. The packages come from the gaps and disagreements in [`CROSSWALK.md`](CROSSWALK.md) and from the specs that already size their own work: `RECIPES_AND_TUNING.md` §9 (RT-1 to RT-14, PR-1 to PR-4) and `understanding/UNDERSTANDING_LAYER.md` §5 (U1 to U14).

## The headline

- **What is done:** about **214 units** of work (listed below): the engine, verified, with the release track.
- **What remains:** about **370 units** (350 if question 5 is answered as recommended). That is **1.7 times what is done**: by effort, v2 is about **37% done**.
- **How far the finish line moved:** 284 of the 370 remaining units (77%) exist because of the amendments of 2026-10-05 (146) and 2026-10-07 (138). Against the definition of done as approved on 2026-10-02/03, 86 units would remain, and v2 would be about **71% done**. The amendments moved the line from roughly 71% done to roughly 37% done.
- **What kind of work remains:** 201 of the 370 units touch the interface (16 interface only, 185 both), 165 are engine only, and 4 are design. The engine's methods are largely done; the screens, the noticings and the new scope are not.
- **Pace, roughly:** the engine waves of 2026-09-27 to 2026-10-06 ran at about 20 units a working day. At that pace the remainder is about 18 working days; but slices end in drives, and the design rulings, drives and expert reviews wait on people, so plan on five to eight weeks of calendar. This is the least certain number here.

**Units.** S = 1 (one module and its reference test), M = 3 (a few modules, or an explanation path or tuning), L = 8 (a new workflow), XL = 20 (a milestone-sized wave, like wave 2a or 2b). The scale follows the sizes in `RECIPES_AND_TUNING.md` §9. Every estimate is relative and uncertain by about a third either way; the ratio of remaining to done is more reliable than any single package.

## Remaining, by road stage

| Road stage | Units | Share |
|---|---|---|
| 0 · Before the slices | 41 | 11% |
| 1 · Core slices | 213 | 58% |
| 2 · Describe and the new methods | 36 | 10% |
| 3 · Designed experiments | 30 | 8% |
| 4 · Export | 20 | 5% |
| 5 · Re-audit and release | 30 | 8% |
| **All** | **370** | |

| Origin | Units |
|---|---|
| v2 as approved (2026-10-02/03) | 86 |
| amendment of 2026-10-05 | 146 |
| amendment of 2026-10-07 | 138 |

## The work packages

Each package: what it is, what it depends on, its size, whether it is engine, interface or both, and which version of the definition of done brought it in. A slice is an engine thread, its quest-log screen and a drive (DoD §6, road item 5).

### 0 · Before the slices

| Package | What it is | Depends on | Size | Engine or interface | Brought in by |
|---|---|---|---|---|---|
| **P0.1** The NHANES fixture in CI | Confirm that the gzipped NHANES export is the one the 42 real-data test files read (a gzipped adult file is at `turbotab/core/tests/acceptance/wp14_data/`), that CI runs them, and that the mock browser tests run, with the `m2-journeys` health-check crash fixed (HANDOFF, validation of `319a513f`). DoD §6 road item 1. | — | S | engine | v2 as approved (2026-10-02/03) |
| **P0.2** `RECIPES_AND_TUNING.md` draft 3 | Fold in the rulings of 2026-10-06 (trees try both; the 2-minute hold; three groups kept; the kept comparisons to v2.x). DoD §6 road item 3. | — | S | design | amendment of 2026-10-05 |
| **P0.3** The quest-log design on the crosswalk | Amend FOUNDATION §3/§6 and BLUEPRINT §11 to the seven stages; add the purpose-registry entries for every new element; settle the five questions in CROSSWALK.md; design the new view kinds (table, forest, curve, calibration, decision curve, specification curve, overlap, embedding, matrix, page). DoD §6 road item 4. | P0.2, the five questions | M | design | amendment of 2026-10-07 |
| **P0.4** The stage registry and reopen reasons | Engine map of every Router key, every non-Router decision kind, every finding and every noticing to one of the seven stages and an objective; per-stage progress; a "reopened because …" record when an earlier answer changes later stages (CROSSWALK, Gaps engine 1; disagreements 11 and 19). | P0.3 | M | engine | amendment of 2026-10-07 |
| **P0.5** The Confirm sweep and the record list | Per stage, the defaults set for the user with a would-change-a-number test, and one endpoint for the For the record lines: ingest warnings, profile basis, not-applicable reasons, checked-clean records (Gaps engine 2). | P0.4 | M | engine | amendment of 2026-10-07 |
| **P0.6** The ordering fixes | Roles recorded as a completion once readings settle; the draw separated from the validation scheme; the split as For the record under Estimate; the survey question under every goal; a design slot; the Router exception for "Decide now"; substitution before Fit; the after-estimates exemptions for explanations, diagnostics responses and updating; the missing-values question skipped when nothing is blank (disagreements 1, 5, 9, 10, 13, 14). | P0.4 | L | engine | amendment of 2026-10-07 |
| **P0.7** The quest-log shell | Seven stages, the seven-segment bar with its drop-back reason, Decide · Confirm · For the record, hover elaboration, card expansion with a way back, the manuscript rail, the canvas grammar, all driven by the stage registry. Replaces the linear Record (Gaps interface 1). | P0.3, P0.4, P0.5 | L | interface | amendment of 2026-10-07 |
| **P0.8** Fit, the hold and a visible lock | A Fit record that releases the estimate stages; the scheduler's 2-minute hold (RT-8); the plan lock shown with its time and SHA-256 (disagreement 12; Gaps engine 7). | P0.4 | M | both | amendment of 2026-10-07 |
| **P0.9** The thread machinery | UNDERSTANDING_LAYER U1, U2, U3, U6, U8 and U13: the registry and contract, the ledger extension, the census, context lines, the open-noticings gate (at the lock and at the opening) and the coverage registry. | P0.4 | L | engine | amendment of 2026-10-05 |
| **P0.10** Engine previews and follow-ups | The energy preview that quotes a coefficient before the lock; the exposure and estimand previews that draw nothing; the event and goal previews that draw only a note; the metabolomics zero-row crash; the elastic net's platform-dependent penalty (a methods decision); the zero-row and chronology refusals with exits (INBOX 255). | — | M | engine | v2 as approved (2026-10-02/03) |

### 1 · Core slices

| Package | What it is | Depends on | Size | Engine or interface | Brought in by |
|---|---|---|---|---|---|
| **C1** Slice: Your data | Data-in screens for joins and the codebook (client functions exist, no component); the per-column ledger endpoint and view; readings asked in Your data (question 3); lens, orientation and the feature table; finding dispositions; a sheet and member picker; the category-spellings repair; the Confirm sweep; a drive. | P0.6, P0.7 | L | both | v2 as approved (2026-10-02/03) |
| **C1b** Stacking cycles and practice datasets | Stack files with a shared schema (NHANES cycles), with the cycle and the four-year weight read; Classic's practice datasets in the demo. | C1 | M | both | amendment of 2026-10-05 |
| **C2** Slice: Your question | The outcome card with every kind (ordinal and time to event too), unit, scale, order, follow-up window and prediction horizon; the outcome-definition readings (diagnosed or diseased, defined from columns, ascertainment); the goal, the shape and "add another goal" screens; intended use and the moment of use; a drive. | P0.6, P0.7 | L | both | v2 as approved (2026-10-02/03) |
| **C3a** Slice: First look, outcome-free | The pre-seal notices stage (U4) with the rank key; the seal-aware index; the six groups with checked-clean lines; "Worth a look"; the walk; "Decide now"; the context view; heavy passes scheduled, never on hover. | P0.9, C1 | L | both | amendment of 2026-10-05 |
| **C3b** First look: the outcome door and Classic's views | The outcome door (after Who's in, question 2) with `view_outcome` recorded and an Explore-to-canvas adapter; Classic's missingness patterns, skew and outlier table and pre-fit VIF table; the embedding and matrix views. | C3a, C4 | M | both | amendment of 2026-10-05 |
| **C4** Slice: Who's in | Cards that read every engine field (grain to the seal); the clusters question split from its model term; the seal with the Confirm sweep before it; the participant flowchart in full (units, ghosted domains, leavers against stayers) and the samples-and-features flow; a drive. | P0.6, P0.7, C2 | L | both | v2 as approved (2026-10-02/03) |
| **C5** Slice: Models under Estimate | Bespoke cards for the estimand, adjustment, time-varying, form, modification and causal questions; gates and screens for Model 1, sensitivity, calibration, scales, batch and multiplicity; overlap and weight views; the substitution pair before Fit; the open-noticings gate; the analysis flowchart with Fit; a drive of the dietary inference journey. | P0.6, P0.8, P0.9, C4 | XL | both | v2 as approved (2026-10-02/03) |
| **C6a** Prediction: tuning and the four families | RT-1 (the search engine), RT-4, RT-5a to RT-5f (boosted trees tuned, ridge, Huber, random forest, XGBoost, the elastic net on the path search), RT-11; nested tuning; each family through the method contract with a reference test and an explanation path. | P0.2 | XL | engine | amendment of 2026-10-05 |
| **C6b** Prediction: recipes, versions and cost | RT-2, RT-3 (the trees' own blanks, tried both ways), RT-6 to RT-10, RT-12 to RT-14: recipes declared and drawn, overrides, versions kept and labeled "revised after first results", cost estimates, export and replay of versions, acceptance tests T1–T17. | C6a | XL | engine | amendment of 2026-10-05 |
| **C6c** Prediction: the groups kept in v2 | Faster search for big tables (successive halving, Hyperband, TPE, BOHB, a Thorough budget, the tuning curve); more preprocessing options (native categories, per-family Pareto and robust scaling, a per-model log1p); the inference extensions (Huber as a weighted M-estimator with a design-based or cluster sandwich; DML and TMLE nuisance learners from the registry, tuned inside cross-fitting). | C6a, C6b | L | engine | amendment of 2026-10-07 |
| **C6d** Slice: Models under Predict | PR-1 to PR-4 (recipe lines, the tuning line, the comparison columns, Results settings); the validation scheme as a Confirm; selection, levers and intended use on their cards; a drive of the metabolomics prediction journey. | C6b, C4, P0.7 | L | interface | amendment of 2026-10-05 |
| **C6e** In-fold PCA for omics | Components summarized inside each training fold when features far outnumber samples. | C6a | M | engine | amendment of 2026-10-05 |
| **C7a** The exhibit model | Decision kinds for wording and placement; drafted wordings with claim strengths (association, effect with assumptions named, causal for trials only, prediction performance, describes the model, inconclusive null); the methods floor enforced; one list of every analysis run; "Which of my decisions mattered?" served by the engine. | P0.8 | L | engine | amendment of 2026-10-07 |
| **C7b** The locked primary kept beside a change | Estimate stages computed on the locked plan's state and on the changed state, the change shown as a labeled secondary (disagreement 15). | C7a | L | engine | amendment of 2026-10-07 |
| **C7c** Slice: Results | Fetch and draw the eleven result stages no screen reads; the exhibit view with wording and placement; the new view kinds; the opening of the held-out rows with the threshold, recalibration and nested-CV offer before it; explanations on screen; late-born noticings as labels. | C7a, C5, C6d, P0.3 | L | both | amendment of 2026-10-05 |
| **C8** Slice: Write-up, the manuscript side | The rail in production and the full-width manuscript; the export screen, the live checklist and the plan download; author-only text; the bundle tabulating every stage (INBOX 250, 257, 259); figure numbers from placement; small-cell suppression. | C7a, P0.7 | L | both | v2 as approved (2026-10-02/03) |
| **T1** Noticings: phases 0 to 2 | The prototype plan of UNDERSTANDING_LAYER §6: diet-day-to-day-variance and clin-predictor-after-prediction-time, then one noticing per lens with the sentinel proof, then a second per lens; each proven end to end (detector, card, context lines, sentence, S1, replay, drive). | P0.9, C3a | L | both | amendment of 2026-10-05 |
| **T2** Noticings: the family rollout | The remaining families in the order the journeys fire them (K1, K2, S4, S5, K3, K4, K6): about 111 missing and 175 partial noticings. Sized for all 367; question 5 recommends shipping those that fire on the twelve reference journeys, which roughly halves it. | T1 | XL | both | amendment of 2026-10-05 |
| **T2+** Noticings: the rest of the rollout | The second half of T2, if question 5 keeps all 367 in v2.0.0. | T2 | XL | both | amendment of 2026-10-05 |
| **T3** Missing consumers | U11: competing risks, re-anchoring time zero, IPCW, leave-one-batch-out and leave-one-site-out validation, the specification curve, parallel analysis, re-scoring at deployment noise; each through a method contract. | T1 | L | engine | amendment of 2026-10-05 |
| **T4** The honest-score ladder, sentinels and the supplement | U9, U10 and U7: process-only baselines and grouped-minus-random deltas; the family sentinels with negative controls; thread sentences, supplement S1, the IDA paragraph and the PROBAST/ROBINS evidence table. | T1 | L | both | amendment of 2026-10-05 |

### 2 · Describe and the new methods

| Package | What it is | Depends on | Size | Engine or interface | Brought in by |
|---|---|---|---|---|---|
| **D1** Describe as a goal | A third goal through every purpose branch, contract label, checklist and sentence; usual intake ungated with its own key; a design-based descriptive estimator for means and prevalence by group; Table 1; the "no single outcome" path (question 4). | P0.6, C2 | L | engine | amendment of 2026-10-07 |
| **D2** Sequential tracks | Per-track outcome, goal, seal and plan; the fixed track order; Write-up's merge. | D1, C8 | L | both | amendment of 2026-10-07 |
| **D3** Dietary patterns | PCA, factor analysis, cluster analysis and reduced-rank regression, each as a method contract with a reference test; derived without the outcome as an exposure, refit inside each fold under Predict. | D1 | L | engine | amendment of 2026-10-07 |
| **D4** Subgroups of similar people | Clustering with k by a declared rule, its contract and reference test. | D1 | M | engine | amendment of 2026-10-07 |
| **D5** Bland–Altman agreement | Between two methods and between two models' predictions. | D1 | M | engine | amendment of 2026-10-07 |
| **D6** Trends across stacked cycles | Design-based trends over pooled survey cycles. | C1b, D1 | M | engine | amendment of 2026-10-07 |
| **D7** Describe screens and journey | The Describe shapes and exhibits on screen; the dietary Describe journey run as a second track beside the dietary inference journey, merged into one paper. | D1–D6, C7c | M | both | amendment of 2026-10-07 |

### 3 · Designed experiments

| Package | What it is | Depends on | Size | Engine or interface | Brought in by |
|---|---|---|---|---|---|
| **E1** The design, declared and routed | The design question with "Not available yet" exits; routing to precision adjustment (no confounder search), no exclusion after randomization, and causal wording. | P0.6, C2, C5 | L | both | amendment of 2026-10-07 |
| **E2** Trial analyses | Intention-to-treat and per-protocol sets; parallel and cluster-randomized trials with intervals for few clusters; missing-outcome sensitivity analyses. | E1 | L | engine | amendment of 2026-10-07 |
| **E3** CONSORT | The CONSORT flow diagram and checklist. | E2, C4 | M | both | amendment of 2026-10-07 |
| **E4** Case-control and matched sets | Odds ratios only and no prevalence; prediction recalibrated to the population; conditional logistic regression; sets kept together in folds. | E1 | L | engine | amendment of 2026-10-07 |
| **E5** The trial journey and its review packet | A trial fixture, its reference journey and drive, and the trial methodologist's packet. | E1–E4 | M | both | amendment of 2026-10-07 |

### 4 · Export

| Package | What it is | Depends on | Size | Engine or interface | Brought in by |
|---|---|---|---|---|---|
| **X1** The manuscript model | One model from the record, the placements, the wordings and the author's text; `\label`/`\ref` for every exhibit; `\todo` where only the author can write. | C7a, C8 | L | engine | amendment of 2026-10-07 |
| **X2** Overleaf-ready LaTeX | The LaTeX project zip (Classic's `ml/latex_report.py` as reference). | X1 | M | engine | amendment of 2026-10-07 |
| **X3** Word | The Word document from the same model. | X1 | M | engine | amendment of 2026-10-07 |
| **X4** References checked | A citation registry, `refs.bib`, and every DOI checked against Crossref. | X1 | M | engine | amendment of 2026-10-07 |
| **X5** The manuscript gate | DoD gate 7: every number in prose, captions and the user's own wordings traces to the record; every `\ref` resolves; every DOI verifies (Classic's `ml/manuscript_validator.py` as reference). | X2, X3, X4 | M | engine | amendment of 2026-10-07 |

### 5 · Re-audit and release

| Package | What it is | Depends on | Size | Engine or interface | Brought in by |
|---|---|---|---|---|---|
| **R1** The final re-audit | The whole app, every layer; the 77 findings verified closed; no critical issue (DoD gate 1). | all above | L | both | v2 as approved (2026-10-02/03) |
| **R2** The speed gate | DoD gate 5 measured on a quiet machine, scheduled with Nolan (the machine is beside the bed). | R1 | M | engine | v2 as approved (2026-10-02/03) |
| **R3** Drives | DRIVE_RUBRIC 18/18 on the twelve reference journeys; Nolan drives at least the dietary inference and the metabolomics prediction journeys. | R1 | L | both | v2 as approved (2026-10-02/03) |
| **R4** Expert review | Six packets regenerated (five lenses and trials); findings addressed or waived. | R1 | L | both | v2 as approved (2026-10-02/03) |
| **R5** Release | User guide, regenerated methods reference, CITATION; launchers and server mode smoke-tested; the Classic triage (49 failures); browser tests in CI; the PR and the `v2.0.0` tag. | R3, R4 | M | both | v2 as approved (2026-10-02/03) |

## The order, and why

1. **Before the slices.** The stage registry (P0.4), the ordering fixes (P0.6) and the shell (P0.7) come first, because every slice draws its objectives from the registry and every fix changes what a slice's screen may ask. The thread machinery (P0.9) comes before First look because First look is mostly noticings.
2. **Core slices, in stage order.** Your data and Your question first (C1, C2), because Who's in reads both; then Who's in (C4), First look (C3a, C3b, which needs the cohort for its outcome door), Models (C5 under Estimate; C6 under Predict, whose engine work C6a and C6b can start at once on P0.2), Results (C7) and Write-up's manuscript side (C8). The noticing phases (T1) ride with the slices whose cards they land on, starting with the two phase-0 noticings (DoD §6: "they start with the understanding layer's phase 0, the shelf and tuning").
3. **Describe and the new methods** after the core, because Describe needs the goal, the outcome card and the survey question in their final form (D1 needs P0.6 and C2), and tracks need Write-up's merge (D2 needs C8).
4. **Designed experiments** as one milestone after the core slices (DoD §6, road item 7). E1 needs the design slot from P0.6.
5. **Export** after the exhibit model and Write-up (X1 needs C7a and C8).
6. **Re-audit and release** last: the re-audit covers everything above.

**What can run in parallel.** The prediction engine (C6a, C6b, C6c, C6e), the new methods' engines (D3 to D6), the trial analyses (E2, E4) and the export writers (X2 to X4) are engine work with reference tests and no screen; they can run beside the slices once their dependencies land.

**Long CPU runs** (the noise ceiling's permutations, leave-one-batch-out with in-fold ComBat, the speed gate, the regenerated journeys and packets) are scheduled with Nolan, never run unannounced.

## What is done, in the same units

| Done | Size |
|---|---|
| M0: the foundations (datastore, graph, Router skeleton) | L |
| M1: the NHANES journey end to end, the Record and the stage | XL |
| M2: the opening sequence for five lenses, findings, the seal, the wide path | XL |
| M3: the methodology audit, the shared core, the top engine item per lens, replay, the thin export | XL |
| The audit's 77 critical and major findings closed; math and methods verified | XL |
| Intelligence: the readings ledger (BLUEPRINT §14.1–14.3) | L |
| Routing | L |
| The completeness pass (waves 1a and 1b, with repairs) | XL |
| The extended methods (wave 2a: causal ML, time-varying, explainability, usual intake, ComBat) | XL |
| The modeling sequence (wave 2b: forms, modifiers, Explore, calibration, multiclass substitution, the leash) | XL |
| Previews for every number-changing kind, and the export bundle with replay (wave 2c) | XL |
| The release track: launcher, server mode, Docker, methods reference, five review packets | L |
| Design: the understanding catalogs (367 noticings), the calm foundation and four structures | L |
| Design: the understanding layer and First look brief | L |
| The Classic parity audit; the recipes-and-tuning spec (draft 2) | M |
| The crosswalk | M |
| **All** | **214 units** |

Sources: `HANDOFF.md` ("The engine (done and independently verified)"), `V2_DEFINITION_OF_DONE.md` §6 ("Done (through 2026-10-06)"), and BLUEPRINT §9 (the milestones).

## What would shrink the road

- **Question 5** (CROSSWALK.md): shipping the noticings that fire on the reference journeys, and stating or deferring the rest, removes T2+ (20 units).
- **Question 3**: asking readings in Your data only where they change a number keeps C1 at L; asking every reading of every column there would grow it.
- **The kept groups** (C6c) and **dietary patterns** (D3) are the largest single additions of 2026-10-07 after the quest log and the export.
- **Nothing here is displaced.** The definition of done says a new idea enters v2 only by displacing something; the 2026-10-07 amendment added without displacing, at Nolan's direction. If the date matters more than the scope, the candidates to move to v2.x are T2+, C6c and D3.
