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

LightGBM, ExtraTrees, kNN, SVM, naive Bayes, LDA and neural networks stay v2.x. Per-model preprocessing overrides and hyperparameter optimization are being specified.

**The one-sentence test.** v2 is done when a nutrition researcher in any of the five lenses can take
their own table from upload to a defensible prediction *or* inference result and a methods section a
reviewer accepts, and every number, label and sentence on the way is verified against an
independent reference.

---

## 1 · The journeys that must work end to end (the product)

Ten reference journeys: one **prediction** and one **inference** analysis per lens, each on a
reference fixture. The NHANES export runs the dietary and clinical ones. Each journey goes:

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
   figures, a replayable provenance record, and auto-filled **TRIPOD+AI** (prediction) or
   **STROBE-nut** (inference) checklists that list their unanswered items. Replaying the record
   reproduces the model matrix and the estimates.

## 4 · Release requirements

- **Human expert review**, one domain methodologist per lens, from a per-domain review packet: the
  methods offered, how they chain, the defaults, and the exact sentences the app writes. Findings are
  addressed, or Nolan waives a lens explicitly.
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
- the All of Us adapter.

## 6 · The road from here to done

methods verification (running) → intelligence (WP13–15) → routing (WP16–18) → completeness pass
(QC drift, scale reliability, batch, XPT, codebook import and minimal joins, and the engine defects MS1–MS8 from
the modeling-sequence review) → the modeling-sequence spec,
reviewed and built → the extended methods (causal ML, time-varying exposures, NCI, ComBat,
multiclass substitution) on the modeling sequence's exposure and estimand machinery → presentation resumed for Explore, the modeling sequence, the inductive-bias
curves and export → re-audit → Nolan's drives → expert review → packaging → `v2.0.0`.
