# TurboTab v2 — definition of done

**Status: APPROVED by Nolan, 2026-10-02**, with the scope expanded at his direction: everything first listed as deferred, up to "deep multi-file assembly", is IN v2, along with all four of the orchestrator's recommended calls. Once approved, this is the finish line. A new
idea enters v2 only by displacing something on this list; otherwise it goes to `INBOX.md` for v2.x.
This guards against the pattern Nolan diagnosed in the old project: "close two, open three, the
goalposts move."

**Amended 2026-10-03.** Nolan put **codebook import** into v2 (§1). The modeling-sequence review
(MODELING_SEQUENCE.md §0) refined the rows in §2. Most of its changes correct rows already listed. Two are
additions: multivariate regression calibration, and sensitivity to unmeasured confounding. Each resolves a
conflict inside rows already in v2.

**Amended 2026-10-05 (Nolan, after the Classic parity audit, `docs/turbotab-next/parity/CLASSIC_PARITY.md`).** He approved bringing these into v2:
- **The prediction shelf:** **ridge** and **Huber**, as linear and penalized models; **random forest**; **XGBoost**. Each enters as a plug-in through the method contract, with a reference test and an explanation path.
- **Table 1:** the participant characteristics table every nutrition paper has.
- **Stacking files:** stacking with a shared schema, so NHANES cycles can be pooled. Joins were already in.
- **In-scope fixes:**
  - explanations, evaluation and robustness drawn on screen;
  - each model's preprocessing recipe drawn with its reason;
  - boosted trees' own missing-value handling made reachable;
  - nested tuning for boosted trees;
  - in-fold PCA for omics.
- **Classic's audit and exploration views**, returned as First look views and threads: missingness patterns, the skew and outlier table, a pre-fit VIF table, residual Q-Q, a cross-model importance table, and the split and seed control.
- **Classic's practice datasets,** folded into the demo.

LightGBM, ExtraTrees, kNN, SVM, naive Bayes, LDA and neural networks stay v2.x. Per-model preprocessing overrides and hyperparameter optimization are specified in `RECIPES_AND_TUNING.md`.

**Amended 2026-10-07 (Nolan, after the recipes-and-tuning rulings and a UI and scope discussion; `HANDOFF.md` has the record).** These are additions at his direction; nothing was displaced. He expects no further major changes, "but I cannot make promises since the act of design revealed this conversation in the first place."
- **Recipes and tuning,** as ruled:
  - the trees "try both" for blanks by default, chosen in each training fold;
  - fits expected to take over about 2 minutes wait for the Fit action;
  - three groups the spec had cut stay in v2: faster search for big tables (successive halving, Hyperband, TPE, BOHB, a Thorough budget and the tuning curve), more preprocessing options (native categories, per-family Pareto and robust scaling, a per-model log1p), and the inference extensions (Huber under inference as a weighted M-estimator with a design-based or cluster sandwich; DML and TMLE nuisance learners from the model registry, tuned inside cross-fitting);
  - the kept-comparisons group moves to v2.x (§5).
- **Goals.** Describe, Estimate an effect and Predict, each with question shapes filtered by domain and outcome. Several goals in one paper run as **sequential tracks**: the shared stages run once, each track has its own Models and Results, and Write-up merges them.
- **Describe** becomes a full goal:
  - survey-weighted means and prevalence by group;
  - the usual-intake distribution and the share below a requirement;
  - trends across stacked cycles;
  - Table 1.
- **New methods:**
  - dietary patterns (PCA, factor analysis, cluster analysis, reduced-rank regression);
  - subgroups of similar people (clustering);
  - Bland–Altman agreement, between two methods and between models' predictions.
- **Designed experiments,** one named milestone after the core slices:
  - parallel and cluster-randomized trials;
  - the design declared and routed: precision adjustment for the randomization factors and prespecified baseline covariates, and no confounder selection;
  - no exclusion after randomization in the primary analysis;
  - intention-to-treat and per-protocol analysis sets;
  - the CONSORT checklist and flow diagram;
  - wording that may state a causal effect;
  - missing-outcome sensitivity analyses.

  Case-control sampling (odds ratios only, no prevalence, prediction recalibrated) and individually matched sets (conditional logistic regression) come with it.
- **The presentation:** the quest log (BLUEPRINT §11 and `calm/FOUNDATION.md` to be amended to match):
  - **Seven fixed stages,** with dynamic questions inside: Your data, Your question, First look, Who's in, Models, Results, Write-up.
  - **A seven-segment progress bar.**
  - **Decide · Confirm · For the record,** with one Confirm sweep of defaults at the end of each stage.
  - **Open noticings** go through the TRIAGE SWEEP (Nolan, 2026-10-09, replacing the ruling of 2026-10-06): before the lock, or under Predict before the held-out rows open, every open noticing arrives with the engine's recommended disposition pre-filled, shown with its reason: "doesn't change your numbers here" → a quiet supplement line; "could bias the estimate" → a limitation sentence; "act on it" → points to the decision. Blockers must be resolved. The user reviews, changes any line, and presses Confirm all; every disposition is recorded. A limitation sentence appears only where the issue could move the estimate and nothing was done about it (amended 2026-10-09). Clean checks go in the supplement, and "noticings" are named in two places.
  - **Full-tapestry flowcharts,** each savable as a figure: participant flow (samples and features for omics) and the analysis flowchart before training.
  - **Results as exhibits:** drafted wordings or your own; placement in Results, Discussion or the Supplement; every analysis that was run stays listed.
  - **Plain words on every card.** "Exposure", "confounder" and "estimand" are quiet terms only.
- **Export:** an Overleaf-ready LaTeX project and a Word document, rendered from one manuscript model:
  - `\label`/`\ref` for every exhibit;
  - `refs.bib` with every DOI checked against Crossref;
  - `\todo` where only the author can write.
- **A rare design or analysis outside v2** may appear as a "Not available yet" option, with its reason and an exit that keeps the work. Every option need not be supported.

**Amended 2026-10-08 (Nolan, after the crosswalk and the model-family contract; `crosswalk/CROSSWALK.md` and `MODEL_FAMILY_CONTRACT.md` have the detail).**
- **The model-family contract enters v2** (MC-1 to MC-13, MC-17 to MC-19; about 62 units net). Every family declares, satisfies and passes clauses C1 to C14 before it joins the shelf. That includes the four new families, which are the first built against it.
- **The pre-fit ranking:** families are ranked live at model selection, on the input each family would actually receive (the settled shared steps plus its own recipe).
  - The ranking is **outcome-blind**: no relationship to the outcome and no score, although the outcome's own counts (events, classes) are allowed.
  - The corrected comparison decides the winner after Fit.
- **Post-fit intelligence:**
  - task-model alignment, with noise and permutation floors;
  - the penalty-shrinkage view;
  - the hypothesis noticing ("prediction plus explainability begets further inference"). It is always labeled exploratory and never folded into a locked primary, and in v2.0 it never adds a candidate model (that goes to INBOX);
  - a named-phenomena registry in two registers.
- **Order:** scales, batch correction and omics normalization move ahead of the families question, so the ranking sees each family's real input.
- **Displaced to v2.x by this amendment** (Nolan's displacement rule): the groups kept on 2026-10-07 (package C6c, about 33 units):
  - successive halving, Hyperband, TPE and BOHB, the Thorough budget and the tuning curve;
  - native categories, per-family Pareto and robust scaling, and the per-model log1p;
  - Huber under inference, and nuisance learners from the registry with their tuning.

  The causal lane's learner key is renamed `nuisance_forest` meanwhile.
- **The crosswalk's rulings:**
  - Under Estimate, the outcome beside a column is hidden until the lock.
  - Your data is one ledger plus a Confirm sweep.
  - Describe may start with "No single outcome".
  - ~~All 367 noticings ship in v2.0.0~~. **Re-ruled 2026-10-09:** v2.0.0 ships the thread machinery, the blockers, the 17 family checks and every noticing that fires on the reference journeys, rolled out family by family with the slices. The roughly 138 that no reference journey triggers are deferred to later 2.0.x/v2.x releases, each family once it has realistic test data; until then each is stated in the supplement or listed in INBOX, enforced by the coverage test (U13). In SIZING, T2+ (108 units) becomes T2s (about 14).
  - Blanks are split at the stage line: who is kept in Who's in, how kept blanks are filled in each track's Models.
  - A shared-step change after first results keeps a read-only earlier row.
- **UI rulings:**
  - The manuscript rail and Write-up are one document in two views.
  - A long fit runs as a server job with a notification.
  - A change after results is a kept version (Predict) or a secondary analysis (Estimate). The label "revised after first results" appears only in the comparison and the methods text.
- **The research record (approved):** every fit records its outcome-blind input descriptors, and the comparison results are kept in a versioned format. Pooling across users is opt-in, descriptors only, and needs IRB review.
- **Seam guards (the orchestrator's call, under the freeze's correctness rule).** Eight S-sized hooks keep the v2.x seams open, about 8 units inside packages already planned (`V2X_SEAMS.md`):
  1. a format marker on the decision log, with migrations applied before validation;
  2. both colliding causal learner keys renamed, with parse aliases and retired values never reused;
  3. `TuningPlan.strategy`;
  4. version keys computed over the fields that differ from their defaults;
  5. the read-only "revised after first results" row's trunk key and log sequence persisted;
  6. deferred designs and options kept as named, refused values, never deleted;
  7. estimate stages declared on `Stage`, with `ESTIMATE_STAGES` derived from them;
  8. D2's track machinery written for any order.
- **Testing (the orchestrator's call):** a thin headless driver ("run this plan file on this data") and a path fuzzer check the routing's invariants over thousands of generated journeys. Under the fuzzer, the reference journeys remain the numerical check.

**The one-sentence test.** v2 is done when a nutrition researcher in any of the five lenses can take
their own table from upload to a defensible description, effect estimate or prediction (from
observational data or a designed experiment), with a methods section and a manuscript draft a reviewer
accepts, and every number, label and sentence on the way is verified against an independent reference.

---

## 1 · The journeys that must work end to end (the product)

Ten reference journeys: one **prediction** and one **inference** analysis per lens, each on a
reference fixture. The NHANES export runs the dietary and clinical ones. **Added 2026-10-07:**
- a **Describe** journey (dietary, NHANES: the usual-intake distribution and survey-weighted prevalence by group), run as a second track beside the dietary inference journey, so one paper merges both;
- a **randomized-trial** journey on a trial fixture.

Each journey goes:

upload (CSV, TSV, Parquet, Excel, **SAS XPT**) → opening sequence → seal → modeling sequence →
results → export.

The bar for each journey:
- no dead end;
- every choice previewed on the canvas;
- every decision recorded as a publishable sentence;
- a stated reason for every refusal, with a way forward.

**Codebook import (Nolan, 2026-10-03):** the researcher's data dictionary can be imported in one of three
forms: a variable/label/unit/codes table (CSV or Excel), an NHANES codebook, or the variable labels carried by
an XPT file. Its *structured* fields settle readings as the user's own documentation: units, value-code
tables, and the variable type. Its *free-text* labels only strengthen the guess on the ask card, and the
user confirms them in one tap. The app asks only about what remains (BLUEPRINT §14.2).

**Multi-file assembly, minimal form:** joining two or more files on a shared identifier, with a
preview of row counts (one-to-one and one-to-many). NHANES ships as separate files joined on
`SEQN`, so without this most NHANES users cannot start.

## 2 · Methods in v2

Each method lives behind a method contract (BLUEPRINT §13) and has an acceptance test against an
independent reference.

| Scope | In v2 |
|---|---|
| **Prediction (shared)** | in-fold preprocessing, nested tuning; linear, penalized and boosted-tree families; calibration; bootstrap optimism; DeLong; selection optimism stated; the price of explainability measured |
| **Inference (shared)** | a declared exposure, estimand and adjustment set; OLS, logistic (odds ratios), ordinal, Cox, mixed models and GEE, design-based survey estimation; cluster-robust and HC3 intervals; multiple imputation compatible with the analysis model, pooled by Rubin's rules (D1 for multi-df tests); splines with nonlinearity tests; declared secondary analyses; the analysis-plan lock; the effect measure (conditional or marginal, with g-computation standardization); exposure families with FDR; sensitivity to unmeasured confounding (E-value; robustness value) |
| **Dietary** | energy adjustment (the five models plus all-components, each with its correct estimand); implausible intake by fixed rules and Goldberg, with a sensitivity view; repeated recalls by averaging and by univariate and multivariate regression calibration (a declared secondary analysis); substitution curves with refit bands; NHANES design |
| **Clinical** | plausibility repairs; time points and the temporal seal; Cox; mixed models; calibration; Riley sample size |
| **Metabolomics** | orientation; **QC-drift correction (QC-RLSC)**; LOD-aware handling; PQN, log and scaling in-fold; feature-wise FDR inference |
| **Genomics** | count normalization in-fold; regularized families with screening at p ≫ n; batch as a covariate; FDR |
| **Survey instruments** | sentinel codes; reverse coding; **scale scoring with reliability (ω; α labeled customary)**; ordinal models; the attenuation statement, with disattenuation refused for formative indices |
| **Explainability** | **inductive-bias curves**: each top exposure's effect per family on shared axes (ALE with support masks), gated by a held-out performance floor; the **full SHAP suite** (beeswarm, per-observation attributions, with stability across reseeds); **interaction ranking**; the **architecture lane** beside the data lane on the canvas (linear: the fitted equation; trees: split structure; elastic net: shrinkage) |
| **Causal inference** (shortest leash) | **DoubleML and TMLE** for a declared exposure and estimand, with flexible nuisance models; **time-varying exposures** by g-methods (marginal structural models with inverse-probability weights; the parametric g-formula); declared assumptions (positivity, no unmeasured confounding, time ordering) and their diagnostics (overlap, weight distribution) shown before any estimate |
| **Dietary, extended** | the **NCI usual-intake method** (amount-only, and the two-part model for episodically consumed foods); **multiclass substitution curves** (one per class) |
| **Genomics, extended** | **batch correction (ComBat) fit in-fold**, beside batch as a covariate |

## 3 · Quality gates (all must pass)

1. **Correct.** The audit's 77 critical and major findings are closed and verified. A final re-audit
   of the whole app finds no critical issue in any layer. The acceptance harness (independent
   references) is green.
2. **Honest.** Every claim the app makes is in the claims ledger, verified against its primary source
   or labeled as a convention. Customary and sound are labeled separately (North star 5).
3. **Teaches.** The pedagogy audit finds no orphan or duplicate element on a primary screen. The
   word-budget gate is green.
4. **Feels right.** DRIVE_RUBRIC passes 18/18 on the reference journeys. **Nolan has driven at least
   the dietary inference and the metabolomics prediction journeys himself.**
5. **Fast enough.** Previews take < 1 s at the 95th percentile. The opening sequence takes < 30 s on
   500 × 20,000 and on 1,000,000 × 30. Any fit expected to take > 30 s shows its estimate first.
6. **Reproducible.** The export carries the methods section, the participant-flow and lineage
   figures, a replayable provenance record, and auto-filled **TRIPOD+AI** (prediction),
   **STROBE-nut** (observational inference) or **CONSORT** (trials) checklists that list their unanswered items. Replaying the record
   reproduces the model matrix and the estimates.
7. **The manuscript is checked** (added 2026-10-07). It is checked like replay, as a function of the record:
   - every number in the LaTeX and Word output traces to a value in the record;
   - every `\ref` resolves;
   - every citation's DOI is verified.

   CI runs the real-data tests on the committed NHANES fixture (gzipped) and the browser tests against the mock server.

## 4 · Release requirements

- **Human expert review**, one domain methodologist per lens, from a per-domain review packet: the
  methods offered, how they chain, the defaults, and the exact sentences the app writes. Findings are
  addressed, or Nolan waives a lens explicitly. A trial methodologist reviews the designed-experiments
  milestone from its own packet.
- **Runs where researchers are:** a one-command local launcher on macOS and Windows, and the
  university-server mode (Docker, auth), both smoke-tested.
- **Docs:** a short user guide, a methods reference generated from the contracts, and a CITATION
  file.
- **Release mechanics:** the legacy app is retired from the branch, Classic is untouched on `main`,
  v2 merges by PR with CI green, and the tag is `v2.0.0`.

## 5 · Explicitly NOT in v2 (deferred to v2.x)

- deep multi-file assembly (fuzzy keys, conflict resolution);
- an in-app AI assistant;
- multi-user collaboration;
- the All of Us adapter;
- **added 2026-10-07:**
  - mediation analysis (mediators are still recognized and kept out of the adjustment set);
  - crossover trials, repeated-measures trial models and complier effects;
  - the kept comparisons ("Compare with standard settings" as a kept version, shared-step changes after scores kept as versions, re-tuned substitution bands);
  - running goals as parallel tracks (v2 runs them in sequence);
  - the research forum (publishing analyses as structured objects), an aspiration with no scope yet.
- **added 2026-10-08:**
  - the kept tuning and preprocessing groups (C6c; listed in the amendment above);
  - the hypothesis noticing adding a candidate model;
  - neural and in-context families (MLP, FT-Transformer, TabPFN-style), held to the contract's v2.x packages MC-14 to MC-16b;
  - the headless Python and R packages and an agent skill, beyond the driver v2 needs for testing.

## 6 · The road from here to done

**Done (through 2026-10-06):** methods verification; intelligence; routing; the completeness pass; the
modeling sequence; the extended methods; the release track. CI is green on every platform.

**From here (2026-10-07):**
1. The NHANES fixture is committed, and CI runs it and the mock browser tests.
2. **The crosswalk:** what the engine must surface for the user to decide, and when, before and after training.
3. **`RECIPES_AND_TUNING.md` draft 3.**
4. **The quest-log redesign** on the crosswalk.
5. **Core vertical slices:** each is an engine thread, its screen and a drive. They start with the understanding layer's phase 0, the shelf and tuning.
   - The model-family contract's declarations, the live ranking and the switch retirement land first.
   - The noticings roll out family by family alongside the slices.
6. **Describe and the new methods.**
7. **The designed-experiments milestone.**
8. **The LaTeX and Word export,** with the manuscript gate.
9. **Re-audit** → Nolan's drives → expert review → packaging → `v2.0.0`.
