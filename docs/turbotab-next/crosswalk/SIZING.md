# Sizing: the road to v2.0.0

How much is left against the amended definition of done (`V2_DEFINITION_OF_DONE.md`, amended 2026-10-07), as work packages in the road's order. The packages come from the gaps and disagreements in [`CROSSWALK.md`](CROSSWALK.md) and from the specs that size their own work: `RECIPES_AND_TUNING.md` §9 (RT-1 to RT-14, PR-1 to PR-4) and `understanding/UNDERSTANDING_LAYER.md` §5 (U1 to U14). Patched on 2026-10-08 after the completeness critic's review: the packages that were sized below their own specs now carry those specs' sizes, and the work no package held has one.

## The headline

- **What is done:** about **222 units** of work (listed below): the engine, verified, with the release track and three fixes waiting on their branches.
- **What remains:** about **734 units** (640 if question 4 is answered as recommended). That is **3.3 times what is done**: by effort, v2 is about **23% done** (26% under question 4's recommendation).
- **How far the finish line moved:** 636 of the 734 remaining units (87%) exist because of the amendments of 2026-10-05 (426) and 2026-10-07 (210). Against the definition of done as approved on 2026-10-02/03, 98 units would remain, and v2 would be about **69% done**.
- **What kind of work remains:** 480 of the 734 units touch the interface (26 interface only, 454 both), 230 are engine only, and 24 are design.
- **The noticings are the largest part:** T1, T2 and T2+ come to 236 units. Question 4 decides about 94 of them.
- **Pace, roughly:** the engine waves of 2026-09-27 to 2026-10-06 ran at about 20 units a working day. At that pace the remainder is about 37 working days; but slices end in drives, and the design rulings, drives and expert reviews wait on people, so plan on 10 to 16 weeks of calendar. This is the least certain number here.
- **What the patch changed.** The first sizing said 370 units remain. The completeness critic found packages sized below their own specs and work that no package held. The largest corrections are below.
  - The noticing rollout is now counted per detector rather than as two XL waves.
  - The RT packages now carry RECIPES §9's sizes, and so do the kept groups, the Results slices and the quest-log design.
  - New packages hold U5, U12, U14, plain words, DoD gates 2 and 3, and the last four data-in items.
  - The NHANES fixture in CI, the Classic triage, the previews' leash and the zero-row refusal moved to done.

**Units.** S = 1 (one module and its reference test), S–M = 2, M = 3 (a few modules, or an explanation path or tuning), M–L = 5.5, L = 8 (a new workflow), XL = 20 (a milestone-sized wave, like wave 2a or 2b). A package built from parts shows their sum as a number. The scale follows the sizes in `RECIPES_AND_TUNING.md` §9 and `UNDERSTANDING_LAYER.md` §5, and where a package covers their work packages it carries their sizes. Every estimate is relative and uncertain by about a third either way; the ratio of remaining to done is more reliable than any single package.

## Remaining, by road stage

| Road stage | Units | Share |
|---|---|---|
| 0 · Before the slices | 84 | 11% |
| 1 · Core slices | 507 | 69% |
| 2 · Describe and the new methods | 55 | 7% |
| 3 · Designed experiments | 30 | 4% |
| 4 · Export | 20 | 3% |
| 5 · Re-audit and release | 38 | 5% |
| **All** | **734** | |

| Origin | Units |
|---|---|
| v2 as approved (2026-10-02/03) | 98 |
| amendment of 2026-10-05 | 426 |
| amendment of 2026-10-07 | 210 |

## The work packages

Each package: what it is, what it depends on, its size, whether it is engine, interface or both, and which version of the definition of done brought it in. A slice is an engine thread, its quest-log screen and a drive (DoD §6, road item 5). P0.1 is done on its branch (below).

### 0 · Before the slices

| Package | What it is | Depends on | Size | Engine or interface | Brought in by |
|---|---|---|---|---|---|
| **P0.2** `RECIPES_AND_TUNING.md` draft 3 | Fold in the rulings of 2026-10-06 (trees try both; the 2-minute hold; three groups kept; the kept comparisons to v2.x); the Fit rule settled in CROSSWALK.md (under Estimate and Describe nothing is served before the lock, and pressing Fit stays a job command); the design and references the draft owes for the three restored groups, including how adaptive search keeps replay exact. DoD §6 road item 3. | — | M | design | amendment of 2026-10-05 |
| **P0.3a** The quest-log design | Amend FOUNDATION §3/§6 and BLUEPRINT §11 to the seven stages; design each stage's screens through the calm review loop (every design was judged too busy on 2026-10-05); settle the six questions in CROSSWALK.md. DoD §6 road item 4. | P0.2, the six questions | L | design | amendment of 2026-10-07 |
| **P0.3b** View kinds and purposes | Design the ten new view kinds (table, forest, curve, calibration, decision curve, specification curve, overlap, embedding, matrix, page; S each, 10) and add the purpose-registry entries for every new element (M, 3). | P0.3a | 13 | design | amendment of 2026-10-07 |
| **P0.4** The stage registry and reopen reasons | Engine map of every Router key, every non-Router decision kind, every finding, every noticing and every one of the 30 compute stages (CROSSWALK.md, "Engine stages and quest stages") to one of the seven stages and an objective; per-stage progress; a "reopened because …" record that names the changed answer and the stage that went stale; the readiness list for the Models decisions with no Router key, with scales, batch and the omics normalization ordered ahead of the families (Gaps engine 1; disagreements 11, 19, 20 and 21). | P0.3a | M–L | engine | amendment of 2026-10-07 |
| **P0.5** The Confirm sweep and the record list | Per stage, the defaults set for the user with a would-change-a-number test (six sweeps, `other:confirm-sweep:*`), and one endpoint for the For the record lines: ingest warnings, profile basis, not-applicable reasons, checked-clean records (Gaps engine 2). | P0.4 | M | engine | amendment of 2026-10-07 |
| **P0.6** The ordering fixes | Roles recorded as a completion once readings settle; the draw separated from the validation scheme; the split as For the record under Estimate; the survey question under every goal; a design slot; the Router exception for "Decide now"; substitution before Fit; the after-estimates exemptions for explanations, diagnostics responses and updating; the missing-values question skipped when nothing is blank. Added by the display-order audit (2026-10-08): the split recorded by the server at the end of Who's in under Estimate; a changed validation scheme as its own kind (`set_validation`), so the draw's time never moves; `set_missing` split into who is kept (its Router slot stays before the split) and the fill (`set_fill`, with a Router key `fill` after the forms and the modifiers), with a migration for older logs, which brings forward the fill's move that D2 sizes; the clusters term recorded with the grouping (disagreements 1, 5, 7, 8, 9, 10, 13, 14). | P0.4 | L | engine | amendment of 2026-10-07 |
| **P0.7** The quest-log shell | Seven stages, the seven-segment bar with its drop-back reason, Decide · Confirm · For the record, hover elaboration, card expansion with a way back, "Waiting for" on questions not yet answerable (disagreement 20), the manuscript rail, the canvas grammar, all driven by the stage registry. Replaces the linear Record (Gaps interface 1). | P0.3a, P0.4, P0.5 | L | interface | amendment of 2026-10-07 |
| **P0.8** Fit, the hold and a visible lock | RECIPES RT-8 (M: cost, the scheduler's 2-minute hold, the Fit and Refit job payload), counted here only; the serving gate under Estimate and Describe that withholds every estimate stage until the track's lock, with the lock recorded when Fit is pressed (no new decision kind); the lock shown with its time and SHA-256 (M). Added by the display-order audit (2026-10-08): under Predict, no estimate stage served before Fit is pressed, so no score is marked seen before the Confirm sweep; under Estimate and Describe, the explore stage's relationship points withheld until the track's lock (disagreements 4 and 12). Disagreement 12; Gaps engine 7. | P0.2, P0.4 | 6 | both | amendment of 2026-10-07 |
| **P0.9** The thread machinery | UNDERSTANDING_LAYER §5 at its own sizes: U1 the registry and contract (M), U2 the ledger extension (M), U3 the census (S), U5 card families in the Router with the asked-rows gate per journey (M), U6 context lines (S–M), U8 the open-noticings gate at the lock and at the opening (S), U13 the coverage registry (S). | P0.4 | 14 | engine | amendment of 2026-10-05 |
| **P0.10** Engine previews and follow-ups (what is left) | The energy preview's leash and the previews that cannot draw are done on `fix/previews-leash` (128bf41d), and the zero-row refusal on `fix/zero-rows-refusal` (19109ef4); both await merge. Left: the event and goal previews that draw only a note; the elastic net's platform-dependent penalty (a methods decision); the chronology refusal with exits; a check that the metabolomics zero-row crash is the refused empty frame. | — | S–M | engine | v2 as approved (2026-10-02/03) |
| **P0.11** The noticing interface (U14) | Card-family rows, the context line, the manuscript mark, the open-noticings card and the supplement view, as calm components the slices reuse; First look's guide is in C3a. | P0.7, P0.9 | M–L | interface | amendment of 2026-10-05 |
| **P0.12** Plain words | The engine's card text rewritten so "exposure", "confounder" and "estimand" stay quiet terms (about 396 uses in `teaching/content.py` and `estimand.py`, many in strings a card shows), refusal and preview text included, with a word-list check in CI. DoD 2026-10-07: "Plain words on every card". | P0.3a | L | both | amendment of 2026-10-07 |

### 1 · Core slices

| Package | What it is | Depends on | Size | Engine or interface | Brought in by |
|---|---|---|---|---|---|
| **C1** Slice: Your data | Data-in screens for joins and the codebook (client functions exist, no component); the per-column ledger endpoint and view; readings asked in Your data (question 2); lens, orientation and the feature table; finding dispositions; a sheet and member picker; the category-spellings repair; the Confirm sweep; the data routes gated by the outcome's gates once it is chosen (`table` without the outcome beside other columns until the lock or the draw, `columns` and `histogram` without its distribution until Who's in or the draw; disagreement 2); a drive. | P0.6, P0.7 | L | both | v2 as approved (2026-10-02/03) |
| **C1b** Stacking cycles and practice datasets | Stack files with a shared schema (NHANES cycles), with the cycle and the four-year weight read; Classic's practice datasets in the demo. | C1 | M | both | amendment of 2026-10-05 |
| **C1c** Data in: the remaining readings | Gaps engine 11's four items no package held: the mixed-units conversion (S–M), order for text predictors (S), check-all-that-apply blocks (M), and the instrument-kind and supplement readings (S–M). | C1 | 8 | both | amendment of 2026-10-05 |
| **C2** Slice: Your question | The outcome card with every kind (ordinal and time to event too), unit, scale (the log scale offered for any positive outcome), order, follow-up window and prediction horizon; the outcome-definition readings (diagnosed or diseased, defined from columns, ascertainment); the outcome card's preview limited to what defines the outcome, its distribution moved to First look (disagreement 3); the goal, the shape and "add another goal" screens; intended use and the moment of use, with the threshold range declared from the decision's harms (disagreement 16); a drive. | P0.6, P0.7 | L | both | v2 as approved (2026-10-02/03) |
| **C3a** Slice: First look, outcome-free | The pre-seal notices stage (U4, M–L) with the rank key; the seal-aware index; the six groups with checked-clean lines; "Worth a look"; the walk; "Decide now"; the context view; heavy passes scheduled, never on hover. | P0.9, P0.11, C1 | L | both | amendment of 2026-10-05 |
| **C3b** First look: the outcome and Classic's views | The outcome views by question 1 (the outcome alone after Who's in, on the rows kept so far, returning when Models redraws the counts; the outcome beside a column after the lock under Estimate and Describe, the server withholding its points until then, and after the draw under Predict; the batch-balance verdict the one exception; CROSSWALK, the display gates) with `view_outcome` recorded and an Explore-to-canvas adapter (M); Classic's missingness patterns, skew and outlier table and pre-fit VIF table (S each); the embedding and matrix views (S–M each). | C3a, C4 | L | both | amendment of 2026-10-05 |
| **C4** Slice: Who's in | Cards that read every engine field (grain to the seal); the clusters question split from its model term; the seal with the Confirm sweep before it, the split's seed control among its lines (Classic's, 2026-10-05); the participant flowchart in full (units, ghosted domains, leavers against stayers) and the samples-and-features flow, each placed as an exhibit; a drive. | P0.6, P0.7, C2 | 9 | both | v2 as approved (2026-10-02/03) |
| **C5** Slice: Models under Estimate | Bespoke cards for the effect, adjustment, time-varying, form, modification and causal questions; "only the direct part" as Not available yet; gates and screens for Model 1, sensitivity, calibration, scales, batch and multiplicity; the survey estimators as a Confirm; overlap and weight views; the substitution pair before Fit; the open-noticings gate; the analysis flowchart with Fit; a drive of the dietary inference journey. | P0.6, P0.8, P0.9, C4 | 22 | both | v2 as approved (2026-10-02/03) |
| **C5b** The omics chain in Models | The normalization asked in Models instead of as a repair option in Your data (`decision:omics-normalization`), with the D-ratio, log and autoscaling steps as a Confirm and their in-fold counts on the samples-and-features flow. | C5 | M | both | v2 as approved (2026-10-02/03) |
| **C6a** Prediction: tuning and the four families | RECIPES §9 at its own sizes: RT-1 the search engine (L), RT-4 (S), RT-5a boosted trees tuned (S), RT-5b ridge (S), RT-5c Huber (M), RT-5d random forest (L), RT-5e XGBoost (M–L), RT-5f the elastic net on the path search (M), RT-11 (S); each family through the method contract with a reference test and an explanation path. | P0.2 | 32 | engine | amendment of 2026-10-05 |
| **C6b** Prediction: recipes, versions and cost | RECIPES §9 at its own sizes: RT-2 recipes declared (M–L), RT-3 the trees' own blanks tried both ways (M), RT-6 decisions and stamps (M), RT-7 versions kept and labeled "revised after first results" (L), RT-9 (S–M), RT-10 (S), RT-12 export and replay of versions (M), RT-13 (S, plus a heavy run scheduled with Nolan), RT-14 acceptance tests T1–T17 (M). RT-8 is in P0.8. With P0.8's RT-8, the RT packages come to about 64 units (M–L read as 5.5, S–M as 2). | C6a | 30 | engine | amendment of 2026-10-05 |
| **C6c** Prediction: the groups kept in v2 | Each through a contract and an independent reference test (DoD §2): successive halving, Hyperband, TPE and BOHB with exact replay (M each, 12); the Thorough budget and the tuning curve (S each, 2); native categories (M); per-family Pareto and robust scaling (S each, 2); a per-model log1p (S); Huber under inference as a weighted M-estimator with a design-based or cluster sandwich, checked against R robsurvey (L); DML and TMLE nuisance learners from the registry, tuned inside cross-fitting (M–L). | C6a, C6b | 33 | engine | amendment of 2026-10-07 |
| **C6d** Slice: Models under Predict | PR-1 to PR-4 at RECIPES §9's sizes (M, M, S–M, S: recipe lines, the tuning line, the comparison columns, Results settings); the validation scheme as a Confirm; selection, levers and intended use on their cards; a drive of the metabolomics prediction journey (M). | C6b, C4, P0.7 | 12 | interface | amendment of 2026-10-05 |
| **C6e** In-fold PCA for omics | Components summarized inside each training fold when features far outnumber samples. | C6a | M | engine | amendment of 2026-10-05 |
| **C7a** The exhibit model | Decision kinds for wording and placement; drafted wordings with claim strengths (association, effect with assumptions named, causal for trials only, prediction performance, describes the model, inconclusive null); the methods floor enforced; one list of every analysis run; "Which of my decisions mattered?" served by the engine; the flowcharts placed like any exhibit. | P0.8 | L | engine | amendment of 2026-10-07 |
| **C7b** The locked primary kept beside a change | Estimate stages computed on the locked plan's state and on the changed state, the change shown as a labeled secondary (disagreement 15); the same for a Describe track. | C7a | L | engine | amendment of 2026-10-07 |
| **C7c** Slice: Results under Estimate | Fetch and draw the seven estimate stages no screen reads (effects, sensitivity, secondary, calibration, scales, causal, modification; S–M each, about 10); the exhibit view with wording and placement (M); the table and forest views (S–M each, about 3); residual Q-Q (S); late-born noticings as labels (S). | C7a, C5, P0.3b | 18 | both | amendment of 2026-10-05 |
| **C7d** Slice: Results under Predict | Fetch and draw evaluation and explanations (S–M each); the curve, calibration and decision-curve views (S–M each, about 5); the opening of the held-out rows with the threshold, recalibration and nested-CV offer before it, and the gate (M); explanations on screen (M); the cross-model importance table (S). | C7a, C6d, P0.3b | 16 | both | amendment of 2026-10-05 |
| **C8** Slice: Write-up, the manuscript side | The rail in production and the full-width manuscript; the export screen, the live checklist and the plan download; author-only text; the bundle tabulating every stage (INBOX 250, 257, 259); figure numbers from placement; small-cell suppression. | C7a, P0.7 | L | both | v2 as approved (2026-10-02/03) |
| **T1** Noticings: phases 0 to 2 | The prototype plan of UNDERSTANDING_LAYER §6: diet-day-to-day-variance and clin-predictor-after-prediction-time, then one noticing per lens with the sentinel proof, then a second per lens. Each of the 11 proofs is end to end (detector, card, context lines at two later stages, sentence, S1, replay, drive): about 1 unit of integration each, the 11 carrying items' wiring (0.5 each), the U12 fixtures it needs (the confounded genomics sibling and a derived NHANES fixture among them), each passing the asked-rows gate. | P0.9, P0.11, C3a | XL | both | amendment of 2026-10-05 |
| **T2** Noticings: the family rollout (the journeys' half) | After T1, 111 noticing items are missing and 164 partial. All 17 families roll out in the order the journeys fire them: K1, K2, S4, S5, K3, K4 and K6 (UNDERSTANDING_LAYER §6), then K5, E2, E1, S1, K7, E3, S7, S3, S2 and S6 by thread count. A missing item is a detector with its reference test on a fixture where it fires and one where it stays silent (S, 1); a partial one needs the card row, context lines, sentence, S1 row and the honored-confirmation tests (0.5); one new fixture serves about five detectors (S). That is about 216 units for all of them; T2 is the half that fires on the twelve reference journeys (question 4). | T1 | 108 | both | amendment of 2026-10-05 |
| **T2+** Noticings: the rest of the rollout | The second half, if question 4 keeps all 367 in v2.0.0. Answered as recommended, it becomes T2s. | T2 | 108 | both | amendment of 2026-10-05 |
| **T3** Missing consumers | U11, each through a method contract with a reference test: competing risks (M), re-anchoring time zero (M), IPCW (M), leave-one-batch-out and leave-one-site-out validation (S–M), the specification curve (M), parallel analysis or MAP (S), re-scoring at deployment noise (S–M). | T1 | 17 | engine | amendment of 2026-10-05 |
| **T4** The honest-score ladder, family checks and the supplement | U9 (M), U10 (M: the 17 family checks, `sentinel:S1` to `sentinel:E3`, with negative controls) and U7 (M: thread sentences, supplement S1, the IDA paragraph and the PROBAST/ROBINS evidence table). | T1 | 9 | both | amendment of 2026-10-05 |

### 2 · Describe and the new methods

| Package | What it is | Depends on | Size | Engine or interface | Brought in by |
|---|---|---|---|---|---|
| **D1** Describe as a goal | A third goal through every purpose branch, contract label, checklist and sentence; usual intake ungated with its own key; a design-based descriptive estimator for means and prevalence by group; the "no single outcome" path (question 3); Describe's open-noticings gate and its track lock, with a later change run as a labeled secondary (Settled here). | P0.6, C2 | 11 | engine | amendment of 2026-10-07 |
| **D1b** Table 1 | The participant characteristics table every nutrition paper has, design-based under the survey answer, by group of what you study under Estimate; it uses D1's descriptive estimator. Approved on 2026-10-05. | D1 | M | both | amendment of 2026-10-05 |
| **D2** Sequential tracks | A track id on every record; per-track outcome, goal, seal and plan; the lock, the after-estimates mark and the estimates shown under prediction scoped to the track (not one per project); the fixed track order; the rows rule (a Predict track after a track that read its outcome validates by resampling); the fill of blanks moved to each track's Models (question 5), scoping to the track the `set_fill` kind that P0.6 builds (disagreement 7); Write-up's merge, with one checklist per track's guideline. | D1, C8 | 16 | both | amendment of 2026-10-07 |
| **D3** Dietary patterns | PCA, factor analysis, cluster analysis and reduced-rank regression, each as a method contract with a reference test; derived without the outcome as what you study, refit inside each fold under Predict. | D1 | L | engine | amendment of 2026-10-07 |
| **D4** Subgroups of similar people | Clustering with k by a declared rule, its contract and reference test. | D1 | M | engine | amendment of 2026-10-07 |
| **D5** Bland–Altman agreement | Between two methods and between two models' predictions. | D1 | M | engine | amendment of 2026-10-07 |
| **D6** Trends across stacked cycles | Design-based trends over pooled survey cycles. | C1b, D1 | M | engine | amendment of 2026-10-07 |
| **D7** Describe screens and journey | The Describe shapes and exhibits on screen; Describe's Who's in (domains, who is kept, grouping for the variance) and Write-up (STROBE-nut with its analytic items marked not applicable, its methods order, the bundle); the dietary Describe journey run as a second track beside the dietary inference journey, merged into one paper, with its capture. | D1–D6, C7c | L | both | amendment of 2026-10-07 |

### 3 · Designed experiments

| Package | What it is | Depends on | Size | Engine or interface | Brought in by |
|---|---|---|---|---|---|
| **E1** The design, declared and routed | The design question with "Not available yet" exits (crossover, repeated-measures trial models, complier and per-protocol effects); routing to precision adjustment (no search for other adjustments), no exclusion after randomization, and causal wording. | P0.6, C2, C5 | L | both | amendment of 2026-10-07 |
| **E2** Trial analyses | Intention-to-treat and per-protocol analysis sets (the set, not the per-protocol effect); parallel and cluster-randomized trials with intervals for few clusters; missing-outcome sensitivity analyses. | E1 | L | engine | amendment of 2026-10-07 |
| **E3** CONSORT | The CONSORT flow diagram, placed as an exhibit, and the checklist. | E2, C4 | M | both | amendment of 2026-10-07 |
| **E4** Case-control and matched sets | Odds ratios only and no prevalence; prediction recalibrated to the population; conditional logistic regression; sets kept together in folds. | E1 | L | engine | amendment of 2026-10-07 |
| **E5** The trial journey and its review packet | A trial fixture, its reference journey, capture and drive, and the trial methodologist's packet. | E1–E4 | M | both | amendment of 2026-10-07 |

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
| **G1** The honesty and teaching gates | DoD gate 2: the claims ledger covers every drafted wording and claim strength (C7a) and every method added since 2026-10-03 (the families, the kept groups, Describe, patterns, clustering, Bland–Altman, the trial analyses), each verified or labeled a convention. DoD gate 3: the pedagogy audit on the quest log and the word-budget gate. Each package writes its own ledger rows; this is the closing pass. | all slices | L | both | v2 as approved (2026-10-02/03) |
| **R1** The final re-audit | The whole app, every layer; the 77 findings verified closed; no critical issue (DoD gate 1). | all above | L | both | v2 as approved (2026-10-02/03) |
| **R2** The speed gate | DoD gate 5 measured on a quiet machine, scheduled with Nolan (the machine is beside the bed). | R1 | M | engine | v2 as approved (2026-10-02/03) |
| **R3** Drives | DRIVE_RUBRIC 18/18 on the twelve reference journeys; Nolan drives at least the dietary inference and the metabolomics prediction journeys. | R1 | L | both | v2 as approved (2026-10-02/03) |
| **R4** Expert review | Six packets regenerated (five lenses and trials); findings addressed or waived. | R1 | L | both | v2 as approved (2026-10-02/03) |
| **R5** Release | User guide, regenerated methods reference, CITATION; launchers and server mode smoke-tested; the PR and the `v2.0.0` tag. The Classic triage and the browser tests in CI are done (listed below). | R3, R4 | M | both | v2 as approved (2026-10-02/03) |

**T2s**, if question 4 is answered as recommended: each of the about 138 noticings outside the reference journeys is stated as one methods sentence or listed in INBOX for v2.x, and the coverage test (U13) checks it; about 14 units in place of T2+.

## The order, and why

1. **Before the slices.** The stage registry (P0.4), the ordering fixes (P0.6) and the shell (P0.7) come first, because every slice draws its objectives from the registry and every fix changes what a slice's screen may ask. The thread machinery (P0.9) and the noticing interface (P0.11) come before First look, because First look is mostly noticings. Plain words (P0.12) come early because every slice's cards carry the engine's text. The branches awaiting merge (the fixture in CI, the previews' leash, the zero-row refusal) merge first.
2. **Core slices, in stage order.** Your data and Your question first (C1, C1c, C2), because Who's in reads both. Then:
   - Who's in (C4);
   - First look (C3a, and C3b, which needs the cohort for the outcome views);
   - Models (C5 and C5b under Estimate; C6 under Predict, whose engine work C6a and C6b can start at once on P0.2);
   - Results (C7a to C7d) and Write-up's manuscript side (C8).

   The noticing phases (T1) ride with the slices whose cards they land on, starting with the two phase-0 noticings (DoD §6: "they start with the understanding layer's phase 0, the shelf and tuning"). T2 follows the family order, each family passing the asked-rows gate.
3. **Describe and the new methods** after the core, because Describe needs the goal, the outcome card and the survey question in their final form (D1 needs P0.6 and C2), and tracks need Write-up's merge (D2 needs C8).
4. **Designed experiments** as one milestone after the core slices (DoD §6, road item 7). E1 needs the design slot from P0.6.
5. **Export** after the exhibit model and Write-up (X1 needs C7a and C8).
6. **Gates, re-audit and release** last: G1 closes gates 2 and 3, and the re-audit covers everything above.

**What can run in parallel.** The prediction engine (C6a, C6b, C6c, C6e), the missing consumers (T3), the new methods' engines (D3 to D6), the trial analyses (E2, E4) and the export writers (X2 to X4) are engine work with reference tests and no screen; they can run beside the slices once their dependencies land.

**Long CPU runs** (the noise ceiling's permutations, leave-one-batch-out with in-fold ComBat, RT-13's regenerated journeys, the speed gate, the regenerated packets) are scheduled with Nolan, never run unannounced.

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
| The crosswalk, with the completeness critic's patch | M |
| The Classic triage: Classic's suite passes on `turbotab-next` (45a923a9, merged in 40dfe0f4) | M |
| The NHANES fixture gzipped, read by every NHANES test, with CI running it and the mock browser tests (2957b9a0, on `fix/nhanes-fixture-ci`; merge pending). This was P0.1. | S |
| The previews' leash: no estimate in a preview before the lock, and a preview that cannot draw says what is missing (128bf41d, on `fix/previews-leash`; merge pending). Part of P0.10. | M |
| The zero-row refusal: no stage is handed an empty frame (19109ef4, on `fix/zero-rows-refusal`; merge pending). Part of P0.10. | S |
| **All** | **222 units** |

Sources: `HANDOFF.md` ("The engine (done and independently verified)"), `V2_DEFINITION_OF_DONE.md` §6 ("Done (through 2026-10-06)"), BLUEPRINT §9 (the milestones), and the commits named in the table.

## What would shrink the road

- **Question 4** (CROSSWALK.md): shipping the noticings that fire on the reference journeys, and stating or deferring the rest, replaces T2+ (108 units) with T2s (about 14). It is the largest single lever.
- **Question 2**: asking readings in Your data only where they change a number keeps C1 at L; asking every reading of every column there would grow it.
- **Question 1**: hiding the outcome beside a column until the lock under Estimate leaves C3b one door fewer to design and explain.
- **The kept groups** (C6c, 33 units) and **dietary patterns** (D3) are the largest single additions of 2026-10-07 after the quest log and the export.
- **Nothing here is displaced.** The definition of done says a new idea enters v2 only by displacing something; the 2026-10-07 amendment added without displacing, at Nolan's direction. If the date matters more than the scope, the candidates to move to v2.x are T2+, C6c and D3 (together about 149 units).
