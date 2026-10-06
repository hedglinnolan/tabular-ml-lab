# Did what made Classic special make it into TurboTab v2?

*Parity report for Nolan · 2026-10-05 · branch `turbotab-next` at `72663217` · read-only review, nothing was run*

**How to read this.** The **engine** is the calculations behind the app. The **screen** is what you can see and click. A feature can be finished in the engine and still be invisible. Paths: `core/` means `turbotab/core/` and `fe/` means `turbotab/frontend/src/`. Classic files keep their own paths (`pages/`, `ml/`, `utils/`). **"Training rows only"** means a step learns from the training part of each cross-validation round and never sees the rows it is scored on.

---

## 1. The short answer

Partly. Most of it was rebuilt in the engine, but much of it is not on the screen yet. The ideas behind the four things you named (per-model preprocessing, data intelligence, explainability and robustness) were rebuilt, and in most places they are sounder than Classic: every learned step is fitted on training rows only (`core/models/pipeline.py:1-8`), the explanations are exact rather than approximate (`core/models/explain.py:15-26`), and the scores are corrected for the optimizm of picking the best model (`core/models/selection.py:12-29`). However, the whole explainability suite and the robustness numbers are calculated but never reach a screen. The explanation step runs only when a request is recorded (`core/stages/__init__.py:756-758`), no screen control sends that request, and the screen never fetches the evaluation, explore, explain or sensitivity results (`fe/components/record/Record.tsx:163-179`, `fe/components/banner/Banner.tsx:32-38`). Per-model preprocessing was deliberately reshaped into one shared recipe, and each model adds only what it declares it needs (`docs/turbotab-next/MODELING_SEQUENCE.md:141`, `core/models/pipeline.py:668-685`). So the per-model tabs and Smart Defaults are gone by ruling, and several options went with no ruling at all: skew transforms, outlier capping, target encoding and MICE for prediction. The model shelf was cut on purpose from Classic's 22 entries to three prediction families (`docs/turbotab-next/V2_DEFINITION_OF_DONE.md:51`). The ones researchers would miss most are random forest, ridge and, because reviewers ask for it by name, XGBoost.

**The tally across the 236 mapped capabilities:** 78 better, 19 the same, 68 partial, 47 missing, 24 dropped on purpose. Of the 97 that are better or the same, 33 are not on any screen, and that includes all 10 explainability ones (my count from the mapping's screen column). One "better" row (LIB-01) covers things Classic never showed its users, so it should not count toward what was lost.

---

## 2. Theme by theme

| Theme | Classic | v2 engine | v2 screen | Status |
|---|---|---|---|---|
| **Data audit and intelligence** | Import Doctor with one-click fixes (`ml/import_doctor.py:315-927`), multi-file roster, joining and stacking files, audit tiles, duplicate rows | Runs Classic's Import Doctor (`turbotab/engine.py:154-155`). Adds reversible repairs, each with a methods sentence (`core/repairs.py:16-38`), and stricter detection of codes like 999 (`core/detectors/codes.py:1-38`). Flags implausible values against NHANES 2017-18 percentiles and CDC growth charts (`core/detectors/plausibility.py:1-34`). Proposes units but never assumes them (`core/units.py:1-23`). Joins files but cannot stack them (`core/assembly.py:1-31`). No duplicate-row check. | Findings and repairs are drawn (`fe/components/record/Findings.tsx`, `Repairs.tsx`). Upload takes one file at a time (`fe/.../StartScreen.tsx:166-171`), and there is no join screen. | **Better** for one file, **narrower** for several |
| **Exploratory analysis** | Galleries, correlation heatmap, outlier heatmap, Feature Explorer, clustering, PCA/UMAP/topology, Table 1 (`ml/clustering.py`, `ml/table_one.py:52`) | Explore findings, each tied to a fix, computed on training rows (`core/stages/explore.py:4-20`). Looking at the outcome is recorded and disclosed (`explore.py:22-31`). No clustering, PCA, UMAP or topology (`core/consequences.py:28` lists embeddings as future work). No Table 1. | Explore is never fetched (absent from `Record.tsx:163-179`) | **Narrower**. What survives is sounder but not visible. |
| **Feature engineering and selection** | Polynomials, general transforms, ratios, binning, PCA/topology features; several selectors with a consensus matrix | Splines, quintiles and cut points learned on training rows (`core/methods/exposure_form.py:1-14, 79-80`). Energy ratios as proper energy-adjustment methods (`core/decisions.py:130-131`). Selection menu ordered by soundness, refused outside cross-validation (`core/models/variable_selection.py:1-42`). Stability selection (`variable_selection.py:14-20`). | Form questions drawn generically. Selection has no control: only its record wording exists (`fe/components/record/sentences.tsx`). | **Narrower menu, sounder methods**. Selection is not on screen. |
| **Per-model preprocessing** | One tab per chosen model (`pages/05_Preprocess.py:673`), each with its own imputation including MICE, Yeo-Johnson/log1p, outlier capping, robust or min-max scaling, target encoding, PCA and k-means (`ml/pipeline.py:347-369, 414-473`). Smart Defaults (`pages/05_Preprocess.py:497-505`). An interpretability toggle (`:554-560`). | One shared recipe. Each family adds only what it declares, and today that is just the elastic net's scaling (`core/models/pipeline.py:668-685`; `core/models/elastic_net.py:40`). Ruled "not a question" (`MODELING_SEQUENCE.md:141`). | A per-model "model matrix" picture and caption are drawn (`core/models/previews.py:495-517`). Each model's step list is sent but never read (`fe/components/stage/LiveScenes.tsx:173-175` reads only the shared lineage). | **Reshaped on purpose**. Several options lost without a ruling (§4c). |
| **Model shelf** | 22 entries, 11 kinds of model (`ml/model_registry.py:324-868`) | 3 prediction families (linear, elastic net, boosted trees), plus inference families Classic never had: ordinal, mixed, GEE, Cox, feature-wise (`core/models/__init__.py:24-31`) | Drawn in rank order with stated concerns (`fe/.../ChoiceQuestions.tsx:532-572`; `Shelf.tsx`) | **Narrower on purpose** (`V2_DEFINITION_OF_DONE.md:51`), broader for inference |
| **Training and comparison** | Splits, cross-validation, Optuna tuning, BCa bootstraps, baselines, calibration, train-vs-test table, residual/ROC/confusion plots, 8 coaching detectors (`ml/model_coach.py:1061-1568`) | Test set sealed by person or by time (`core/seal.py:641-666`). Repeated cross-validation (`core/models/validation.py:10-12`). A no-predictor baseline with a corrected interval (`core/models/baseline.py:1-31`). Model choice corrected for optimizm (`selection.py:12-29`). Calibration slope with intervals (`core/models/performance.py:29-60`). Bootstrap optimizm with at least 500 resamples (`validation.py:14-23`). Only the elastic net is tuned (`elastic_net.py:1-5`). No ROC points, confusion matrix or residual plot under prediction. | The models against the baseline, and the coefficients, are drawn (`fe/.../Comparison.tsx:45-118`; `Coefficients.tsx:76-99`). Calibration curves, paired comparisons and the evaluation stage are not. | **Better engine, screen shows a fraction** |
| **Explainability** | Permutation importance (`pages/07_Explainability.py:776`), SHAP by tree/linear/kernel methods (`:883-910`), dependence plots, Bland–Altman, external validation, subgroups (`:560`) | Exact SHAP for linear models and boosted trees (`core/models/explain.py:15-26`), with stability across refits (`:28-33`). Effect curves masked where data are thin (`:49-55`). Interaction ranking (`:42-47`). Each person's prediction split exactly into contributions (`:633-641`). Numeric and yes/no outcomes only (`core/stages/explain.py:27`). No permutation importance, Bland–Altman or second-file validation. | **None.** The stage needs a recorded request (`core/stages/__init__.py:756-758`), and no control records one; the request appears only as record wording (`fe/components/record/sentences.tsx:152`). | **Better engine, absent on screen** |
| **Robustness and sensitivity** | Seed sensitivity with a robustness verdict, and feature dropout (`pages/08_Sensitivity_Analysis.py:151-155`; `ml/sensitivity.py:57-69`) | Spread across repeated cross-validation (`core/models/metrics.py:40-42`; `core/models/artifacts.py:119-120`). SHAP stability (`explain.py:28-33`). Sensitivity to exclusion rules (`core/stages/sensitivity.py:1-11`). E-value for unmeasured confounding (`core/models/effects.py:50`). No feature dropout and no verdict bands. | None fetched (`Record.tsx:163-179`) | **Partial**. Different, sounder questions, not on screen. |
| **Hypothesis testing** | A test page with assumption checks (`ml/stats_tests.py:12-153`) | No test page. Model-based inference instead (`V2_DEFINITION_OF_DONE.md:52`). False-discovery correction across declared exposure families (`core/estimand.py:934-944`). | None | **Missing** (no ruling), replaced in spirit by the inference lane |
| **Reporting and export** | LaTeX and PDF manuscript (`ml/latex_report.py:24`), Table 1, TRIPOD, markdown report, ZIP, discussion draft | Methods text built from the record (`core/provenance.py:53-65`; `core/export/methods.py:1-25`). TRIPOD+AI and STROBE-nut checklists (`core/export/checklists.py:1-25`). A byte-identical ZIP that can be replayed (`core/export/bundle.py:1-21`; `core/export/replay.py:7-27`). Export refused while anything is unsettled (`core/export/gate.py:1-24`). No LaTeX, PDF, Table 1 or discussion. | Each decision's sentence is drawn (`Record.tsx:761-805`). No export button, only per-figure save (`fe/components/stage/save/SaveMenu.tsx`). | **Better reproducibility, narrower manuscript, not reachable** |
| **Teaching** | 8-chapter theory reference with demos (`pages/11_Theory_Reference.py`) | A "why?" on every question and a sourced concept drawer marked settled, convention or disputed (`core/teaching/__init__.py:1-14`). Simulated demos dropped by ruling (`docs/turbotab-next/BLUEPRINT.md:266-268`). | Drawn (`fe/components/record/Question.tsx:180-203`) | **Reshaped on purpose**, drawn |

---

## 3. What v2 does better, and why

- **Nothing learned from the data sees the rows it is scored on.** Every learned step is refitted inside each training round: imputation, energy adjustment, functional form, encoding, scaling and selection (`core/models/pipeline.py:1-8`; `core/methods/levers.py:1-11`). Two places this changes the outcome compared with Classic: implausible values are now handled before the test set is sealed rather than on training rows after the split (`core/repairs.py:51-55`), and repeated cross-validation replaces Classic's re-split, which counted as opening the test set (`core/models/validation.py:10-12`).
- **Methods are checked against independent references.** Each method must pass an acceptance test against an independent reference (`V2_DEFINITION_OF_DONE.md:46-47, 65-67`). Leverage and Cook's distance match R's `hatvalues` and `cooks.distance` (`core/models/effects.py:39-44`). A validation audit raised 170 findings; all 122 serious ones were re-tested and confirmed (`docs/turbotab-next/audit/AUDIT_REPORT.md:22-30`), and closing them is a release gate (`V2_DEFINITION_OF_DONE.md:65-66`).
- **The explanations are exact.** SHAP is computed in closed form for linear models and with exact tree paths for boosted trees (`core/models/explain.py:15-26`), where Classic fell back to KernelSHAP, an approximation (`pages/07_Explainability.py:910`). Each person's contributions add up exactly to their prediction (`explain.py:633-641`). No curve is drawn for a model that does not beat the no-predictor baseline (`explain.py:53-55`).
- **Honest scores.** Choosing the best model is corrected for its optimizm (`core/models/selection.py:12-29`). Reporting the winner's own score as "the result" is refused (`selection.py:47-55`). Every model is compared with a no-predictor baseline using a corrected interval (`core/models/baseline.py:1-31`). Riley's minimum sample size is calculated before the shelf is ranked (`selection.py:899-967`).
- **The price of explainability is measured, not asserted.** Classic's interpretability toggle (`pages/05_Preprocess.py:554-560`) became a measurement on your own rows: what a spline regression costs or gains against the best flexible model (`selection.py:970-984`).
- **Nutrition knowledge is built in.** Plausibility comes from weighted NHANES 2017-18 percentiles and CDC growth charts (`core/detectors/plausibility.py:1-34`). Units are proposed and confirmed, never assumed (`core/units.py:1-23`). Energy adjustment states its estimand (`V2_DEFINITION_OF_DONE.md:53`). Substitution curves are drawn (`fe/components/stage/results/Curves.tsx:1-6`).
- **A whole inference lane Classic did not have.** Mixed models, GEE, Cox, ordinal and survey-design models (`core/models/__init__.py:24-31`). Multiple imputation compatible with the analysis model and pooled by Rubin's rules (`core/methods/missing.py:6-28`). Influence diagnostics and the E-value (`effects.py:39-50`).
- **Everything is on the record.** Every answer becomes a sentence, and the analysis can be replayed exactly from that record (`core/export/replay.py:7-27`). A refusal always comes with a way forward (`BLUEPRINT.md:122-123`).

Caveat: the explanations, the score corrections, the measured price of explainability and Riley's sentence are not on screen yet (§4b).

---

## 4. What is missing or narrower

### 4a. Dropped on purpose (each has a ruling)

| What | Ruling |
|---|---|
| kNN, SVM, Naive Bayes, LDA, neural network, random forest, ExtraTrees | Prediction shelf is "linear, penalized and boosted-tree families" (`V2_DEFINITION_OF_DONE.md:51`); parity not binding (`BLUEPRINT.md:57`) |
| Per-model preprocessing as a question (per-model tabs) | "Model-specific preprocessing: not a question; stated per family in the lineage" (`MODELING_SEQUENCE.md:141`) |
| Smart Defaults and Quick/Advanced modes | `BLUEPRINT.md:393-395`; detection never pre-selects (`BLUEPRINT.md:72`) |
| The in-app AI "Deep Analysis" | `V2_DEFINITION_OF_DONE.md:96` |
| Re-using one project's settings on the next cohort | Deferred to v2.x (`BLUEPRINT.md:434-435`) |
| Fitted model files in the export | The export carries "never fitted objects" (`BLUEPRINT.md:98`) |
| Trimming the outcome before the split | Refused by design (`core/decisions.py:3548-3556`) |
| Simulated theory demos | "The explanation of an option is its effect on the user's own data" (`BLUEPRINT.md:266-268`) |
| JSON upload | Not in the format list (`V2_DEFINITION_OF_DONE.md:25`) |

### 4b. In v2's own plan, but not finished

These are not losses. The engine or the spec already has them, and the road to done schedules the screens: "presentation resumed for Explore, the modeling sequence, the inductive-bias curves and export" (`V2_DEFINITION_OF_DONE.md:106`).

- **The whole explainability suite is unreachable** (`core/stages/__init__.py:756-758`; no control).
- **Evaluation is not drawn:** the spline benchmark, the interpretable model's cost, subgroup performance, calibration curves, paired comparisons, and the robustness spread across repeats. None of these stages is fetched (`Record.tsx:163-179`).
- **Riley's minimum sample size** is computed but not shown (`selection.py:953-964`).
- **Explore findings and their fixes** are computed but not shown (`core/stages/explore.py:8-20`).
- **Export, methods text and checklists** exist with no button (`core/export/bundle.py:1-21`).
- **Each model's recipe** is sent to the screen and ignored (`core/stages/modeling.py:510` vs `LiveScenes.tsx:173-175`). The spec promises it: "stated per family in the lineage" (`MODELING_SEQUENCE.md:141`).
- **Two gaps against v2's own spec.** First, only the elastic net is tuned, but the definition of done asks for "nested tuning" (`V2_DEFINITION_OF_DONE.md:51`; boosted trees run at fixed settings, `core/models/boosted_trees.py:34-41`). Second, "in-fold PCA for omics" is specified (`MODELING_SEQUENCE.md:138`) but there is no PCA anywhere in `core/`.
- **Boosted trees and missing values.** The spec says the lineage should state whether boosted trees use their own handling of missing values or the shared fill (`MODELING_SEQUENCE.md:141`). In practice they get the shared fill whenever imputation is chosen (`core/models/pipeline.py:541-572, 946-959`).

### 4c. Lost without a ruling (accidental)

**Per-model preprocessing options.**
- No Yeo-Johnson or log1p transform for predictors (no PowerTransformer in `core/`; Classic `ml/pipeline.py:429-431`).
- No outlier capping (Classic `ml/pipeline.py:439-442`).
- No robust or min-max scaling (Classic `:449-451`).
- No target or ordinal encoding (Classic `:473`).
- No MICE under prediction: v2 refuses multiple imputation there, because it cannot be applied to new patients (`core/decisions.py:4460-4471`).

v2's stated reasons cover skew (it "says nothing about which scale is right", `core/structural.py:33-36`) and generic outlier rules (`core/detectors/plausibility.py:623-626`). These are design positions, not rulings.

**Intelligence.**
- Classic scoped each data issue to the chosen models. For example, it said skew matters for your linear and distance models but not your trees (`utils/insight_ledger.py:187-197`; `ml/model_coach.py:854-855`). v2's coach talks only about your rows, never about the models (`core/coach.py:5-8`). For the "intelligence understanding" you named, this is the main loss.
- No leakage scan (Classic `ml/eda_actions.py:847`).
- No duplicate-row check.
- No "start here / try next" picks (Classic `ml/model_coach.py:519`). This one is partly deliberate, because v2 never pre-selects (`ChoiceQuestions.tsx:532`).

**Explainability.**
- No permutation importance (Classic `pages/07_Explainability.py:776`).
- No Bland–Altman agreement between models.
- No validation on a second file (`pages/07_Explainability.py:560`).
- No explanations for multiclass, ordinal or survival outcomes (`core/stages/explain.py:27`).

**Robustness.**
- No feature dropout (Classic `pages/08_Sensitivity_Analysis.py:152`).
- No robust/fragile verdict bands (Classic `ml/sensitivity.py:57`).

**Data and reporting.**
- No stacking of files, for example to pool NHANES cycles (`core/assembly.py:1-31`).
- No Table 1 (Classic `ml/table_one.py:52`).
- No hypothesis-test page (Classic `ml/stats_tests.py:12-153`).
- No LaTeX or PDF output (Classic `ml/latex_report.py:24`).
- No discussion draft.
- No residual, ROC or confusion-matrix views under prediction.
- No clustering, PCA, UMAP or topology views.
- No per-group cohort runs.
- No seed control or hyperparameter controls (`fe/.../SealQuestions.tsx:28-31`).

### 4d. The model shelf: Classic's 22 entries against v2

**What adding a model costs.** A new model is one module that fills in the family contract: tasks, a short description of its built-in assumptions, cautions, whether it needs scaling, whether it handles missing values, how it is built, and how it judges itself on your data (`core/models/base.py:64-89`). It is registered in one call (`base.py:113-127`). The lineage picture, the per-model preview and the run-time estimate then come for free ("a new family needs nothing else", `core/models/pipeline.py:670-675`; `core/models/cost.py:1-15`). Each method also needs a test against an independent reference (`V2_DEFINITION_OF_DONE.md:46-47`). The hidden cost is in explanations: SHAP is computed only for HistGradientBoosting and five named linear model types (`core/models/explain.py:351-362`). Any other model gets effect curves, but no SHAP and no per-person breakdown, until someone writes a SHAP path for it.

Sizes: **S** is one module plus its reference test. **M** also needs a new explanation path or tuning. **L** is a new workflow.

| Classic entry (`ml/model_registry.py`) | In v2? | Why | Would researchers miss it? | Cost to add |
|---|---|---|---|---|
| `elasticnet` (:370) | Yes, with better tuning (`core/models/elastic_net.py:1-5, 44-59`) | — | — | — |
| `lasso` (:347) | Yes, as the elastic net's pure-L1 setting (`elastic_net.py:16`) | Not chosen separately | Little | — |
| `ridge` (:324) | **Partly.** Pure ridge is never tried: the mixes start at 0.1, or 0.2 for logistic (`elastic_net.py:16-19`) | No ruling; ridge is a "penalized" model under DoD:51 | **Yes.** It is the usual first choice for many correlated nutrients or metabolites. | **S**: a ridge module, plus adding it to the linear SHAP list (`explain.py:351-352`) |
| `logreg` (:395) | Yes: unpenalized (`core/models/linear.py:120-140`) and penalized (`elastic_net.py:53-59`); no pure-L2 version | — | Little | Same fix as ridge |
| `glm` (:702) | Yes, with robust intervals and Firth (`linear.py:1-14`) | — | — | — |
| `histgb_reg`, `histgb_clf` (:519, :543) | Yes, as boosted trees, **untuned** (`boosted_trees.py:34-41`) | DoD:51 asks for nested tuning | Some, through weaker trees | **M** for tuning (already in scope) |
| `huber` (:723) | No | **No ruling.** It is a "linear" model under DoD:51. | Some. Intake outcomes have heavy tails. | **S**: it reports coefficients, so it can join the linear SHAP list; numeric outcomes only |
| `xgb_reg`, `xgb_clf` (:774, :804) | No. Installed (`BLUEPRINT.md:90-91`) but never imported. | **No ruling.** It is a "boosted-tree family". | Some. Reviewers ask for it by name, though it is the same kind of method as HistGB. | **S** for the shelf; **S–M** for SHAP (wire XGBoost's own exact values and check them against v2's) |
| `lgbm_reg`, `lgbm_clf` (:836, :868) | No (same as XGBoost) | No ruling | Less than XGBoost | Same as XGBoost |
| `rf` (:747) | No. Used only inside the causal lane (`core/models/causal.py:402-407`). | DoD:51 | **Yes, the most.** In my judgment it is the machine-learning model nutrition reviewers know best. | **M**: S for the shelf (skip bootstrap optimizm, as trees do, `boosted_trees.py:30-32`), plus a SHAP path for scikit-learn forests |
| `extratrees_reg`, `extratrees_clf` (:469, :493) | No | DoD:51 | Rarely | **S** once random forest exists, **M** otherwise |
| `knn_reg`, `knn_clf` (:422, :445) | No | DoD:51 | Rarely | **S** for the shelf; curves only, no SHAP |
| `svr`, `svc` (:569, :593) | No | DoD:51 | Rarely, and slow | **S** for the shelf; curves only |
| `gaussian_nb` (:619) | No | DoD:51 | Rarely | **S**; curves only |
| `lda` (:639) | No | DoD:51 | Occasionally, in metabolomics | **S**; SHAP needs work |
| `nn` (:660) | No. No PyTorch in `core/`. | DoD:51 | A little | **L**: a module, architecture choice, training curves and special cross-validation handling (Classic needed it, `pages/08_Sensitivity_Analysis.py:130`) |

**Summary:** 6 of the 22 entries are fully covered and ridge is partly covered. 5 are absent without a ruling (Huber, XGBoost ×2, LightGBM ×2). 10 are dropped by ruling.

---

## 5. Recommendation

The rule (`V2_DEFINITION_OF_DONE.md:3-4`): a new item enters v2 only by displacing something already on the list, or by your decision. Otherwise it goes to `INBOX.md` for v2.x.

**A. Finish what is already in v2.** None of this needs displacement. In order of value:
1. **Put explanations on screen (M).** A control that records the explanation request, and a panel for the beeswarm, each person's breakdown, the effect curves, interactions and stability (`V2_DEFINITION_OF_DONE.md:58`). This one item is most of the answer to your question.
2. **Put evaluation and robustness on screen (M).** Riley's sentence alone is S. The rest: the interpretable model's measured cost, the spline benchmark, paired comparisons, calibration curves and the spread across repeats (`V2_DEFINITION_OF_DONE.md:51`).
3. **Draw each model's recipe beside the lineage (S).** Include whether boosted trees use their own missing-value handling or the shared fill. This is the ruling's own promise (`MODELING_SEQUENCE.md:141`), and it is the honest form of "per-model preprocessing" in v2.
4. **Explore findings and the export screen (M each).** Already scheduled (`V2_DEFINITION_OF_DONE.md:106`).
5. **Close the two spec gaps (M each).** Nested tuning for boosted trees (DoD:51) and in-fold PCA for omics (`MODELING_SEQUENCE.md:138`).

**B. Your decision. Small, and arguably inside DoD:51's own wording.**
- **Ridge (S)** and **Huber (S)**. Both are "linear, penalized" models. I would add them.

**C. Your decision. Each needs to displace something.**
- **Random forest (M).** The strongest shelf candidate.
- **Table 1 (M).** Every nutrition paper has one.
- **Stacking files, for example NHANES cycles (M).**
- **Model-aware coaching (S–M).** Let each model's self-assessment (`core/models/base.py:87`) say which open findings matter for it. This could fit the intelligence work already on the road (`V2_DEFINITION_OF_DONE.md:102`).

If you want any of these, my suggestion is to look for what to displace among the "extended" rows (`V2_DEFINITION_OF_DONE.md:59-61`). That is my judgment, not a ruling.

**D. INBOX for v2.x.**
- XGBoost and LightGBM (same kind of method as HistGB), ExtraTrees, kNN, SVM, Naive Bayes, LDA, neural network.
- The hypothesis-test page (the inference lane supersedes it).
- General skew transforms and outlier capping. v2's splines, outcome scale and plausibility tiers are the sounder answer. Winsorizing could return later as a declared sensitivity analysis.
- Feature dropout, permutation importance, Bland–Altman, validation on a second file.
- LaTeX/PDF output, clustering/PCA/UMAP views, per-group cohort runs.

---

## 6. Open questions for you

1. **Per-model preprocessing.** The shared-recipe rule (`MODELING_SEQUENCE.md:141`) answers "which preprocessing for which model" by declaration rather than by a choice per model. Is drawing each model's recipe enough? Or do you want a per-model choice back, knowing it would mix preprocessing differences into the model comparison?
2. **The shelf.** Should ridge and Huber go in as "linear, penalized" models? And should random forest be in v2 (and if so, what does it displace) or wait for v2.x?
3. **Table 1 and stacking NHANES cycles.** Are these in v2, each displacing something, or in v2.x?

---

## 7. The skeptic's corrections (applied to how this report should be read)

I checked the cited code at commit 72663217 using a git-archive snapshot. During the review the branch moved to 6f2d12b6: the legacy app was retired and docs moved into reference/ and archive/. The cited core files changed only slightly; seal.py is the largest, at 66 lines.

The report's headline holds up, and in places it is too generous. The core finding is confirmed with stronger evidence: no UI path can trigger or show explanations, evaluation, explore, sensitivity, selection or levers. The frontend sends no set_explain, set_sensitivity, set_selection, set_levers, set_intended_use or view_outcome, and it fetches only 18 named stages.

Corrections on the owner's four areas:

- **Per-model preprocessing.** It is even narrower than reported. The trees' declared native missing-value handling is dead code in the flow: the missing question is always asked, so trees always get the shared fill or complete cases, and the app's own teaching note says 'the trees never see a blank'. The only real per-family difference is the elastic net's StandardScaler.
- **Intelligence.** Each family's cautions and strengths never reach a screen. The coach is about the rows only. Model-aware advice is limited to n/p/events concerns on the shelf and a post-fit collinearity concern for the linear family.
- **Explainability.** The 'exact where Classic was approximate' bullet is wrong for the shared families. Classic already used TreeExplainer and LinearExplainer for them, and KernelSHAP only for the models v2 dropped.
- **Robustness.** Repeated k-fold, bootstrap and internal–external validation cannot be chosen: SealQuestions hard-codes kfold. 'Bootstrap ≥500' is a default, not a floor (the schema allows 20).

Other status corrections:

- **Orientation:** asked only under assay lenses.
- **Cardinality and missing details:** shown only inside the outcome picker.
- **Interpretability preference:** v2 measures a different thing, and only when boosted trees are fit.
- **MICE:** Classic's was an outcome-free IterativeImputer, so v2's refusal rationale misapplies; multivariate imputation under prediction is lost without a ruling.
- **Skew citation:** it concerns the outcome's scale, not predictors.
- **Lasso and ridge:** neither can be chosen.
- **Shelf:** more families than three predict.
- **Tally:** C02-28 should be excluded like LIB-01.
- **'Dropped on purpose':** several items are deferrals or omissions, not rulings.

Also listed: Classic capabilities the report does not name.

| Rows | The claim | What the code shows | Corrected status |
|---|---|---|---|
| REPORT-S3-exact-SHAP (also EX-03) | The explanations are exact rather than approximate: v2 uses closed-form linear SHAP and exact TreeSHAP, where Classic fell back to KernelSHAP (pages/07_Explainability.py:910). | For the families v2 kept, Classic was already exact. It used shap.TreeExplainer for 'tree' models and shap.LinearExplainer for 'linear' ones (pages/07_Explainability.py:882-887). The registry tags ridge, lasso, elasticnet, logreg, glm and huber as 'linear' and HistGB as 'tree' (ml/model_registry.py:338,361,385,411,535,559,714,738). KernelSHAP ran only for kNN, SVM, NB and NN (registry :437,:460,:585,:609,:693), which v2 dropped. v2 genuinely adds reseed stability, ALE with support masks and H-st | equivalent method for the shared families; the additions are engine-only |
| EX-02, EX-03, EX-04, EX-06, EX-07, EX-09, EX-10, EX-11, EX-12, EX-16 | Explainability rows are marked 'improved' or 'equivalent'. | No user can reach any of it. The explain stage requires a recorded set_explain (core/stages/__init__.py:756-758), and nothing on the server records one for the user (set_explain appears only in decisions.py:1529, its validators at explain.py:1708-1710, and its sentence). The frontend never sends set_explain: it appears only as record wording (components/record/sentences.tsx:152-153). 'explain' is not a Router key (core/interview.py:91-98), so no composer exists for it. The frontend never fetches | engine-only (missing for a UI user); count as a loss on screen, not 'better' |
| SE-01 (and report §3 'repeated cross-validation replaces Classic's re-split') | Random-seed robustness is improved by repeated k-fold (repeat_sd) and SHAP reseeds. | The only split control sends validation:'kfold', repeats:10, n_boot:500, seed:0 and folds:5 as constants (components/record/ask/SealQuestions.tsx:27-38, comment: 'the validation choice is not drawn yet'). So a UI project never gets a repeated-k-fold headline, repeat_sd (artifacts.py:119-120), bootstrap optimizm or internal–external validation. The model comparisons do run internally on at least 10×K repeats (core/models/folds.py:23-25, 49; validation.py:369-372), but fit.comparisons is never dra | engine-only; robustness is missing on screen |
| TC-15 (and report §2 'Bootstrap optimizm with at least 500 resamples') | Bootstrap optimizm refits the pipeline on at least 500 resamples; the analytic CIs improve on Classic's BCa bootstrap. | 500 is the default, not a floor. The schema accepts n_boot from 20 (core/decisions.py:393, 410, 733). MODELING_SEQUENCE row 11 asks for 'bootstrap ≥ 500' (docs/turbotab-next/MODELING_SEQUENCE.md:142), so the floor is itself a spec gap. Bootstrap validation also cannot be chosen from the UI (SealQuestions.tsx:33). The intervals are analytic (LeDell, delta method, DeLong), not BCa, and Comparison draws the fold mean ± SD, not the CI (components/stage/results/Comparison.tsx:63-85). Classic drew BCa | partial: intervals computed, not shown; 'at least 500' is false |
| C05-22 / report §4b 'boosted trees get the shared fill whenever imputation is chosen' | HistGB declares handles_missing; native NaN handling applies when no strategy is set; the lineage should say which. | The native path is dead in the flow. The missing question has no gate (core/interview.py:480-501 gates; NEEDS 'missing': () at :138), so it is always asked before the split. MissingAsk offers only leave-out, blanks-as-level, complete cases and impute (components/record/ask/ChoiceQuestions.tsx:209-330). Either way the trees see no blank: complete cases drop the rows (stages/rows.py:1441-1449) and impute fills them (pipeline.py:447, 541-572). The app's own teaching note says so: 'the trees never s | equivalent to Classic in effect (always imputed or complete cases); the 'per-family' missing handling is not real |
| C03-12 (and C05-02 'Rank, fit tag, inductive bias and concerns are drawn') | Family cautions such as 'No coefficients: effects are read from curves' are drawn on the models card (boosted_trees.py:22-27; ChoiceQuestions.tsx:564-582). | The cautions and strengths tuples never reach a screen. The shelf artifact carries only rank, fit, concerns, inductive_bias and estimate (core/stages/modeling.py:188-196). No frontend component reads 'cautions' or 'strengths', and useModels (/api/models) is defined but unused (frontend/src/api/queries.ts:257-258). ModelsAsk's line is the teaching consequence or the inductive bias (ChoiceQuestions.tsx:556-560). Under prediction, cautions such as the elastic net's 'Which of several correlated pred | partial; the model-side 'intelligence' on screen is thinner than stated |
| C05-07 / PP-04 (and report §3 'price of explainability') | Classic's High/Balanced/Performance interpretability toggle is improved by a measured interpretable-vs-flexible cost. | These answer different questions. Classic's toggle set which transforms (PCA, KMeans, log) a pipeline could use (pages/05_Preprocess.py:552-562). v2's interpretable_cost compares the spline benchmark with the best flexible family, and it returns None without one (core/models/selection.py:998-1010; stages/evaluation.py:278, 286-292). Only boosted trees count as flexible (selection.py:914-920; boosted_trees.py:32). Fit only linear and elastic net and nothing is measured. The evaluation stage is ne | replaced by a different measurement, engine-only; not a like-for-like improvement |
| C01-04 | Transpose on import is improved. | Under any lens other than an assay lens, the orientation question is 'not_applicable' with the reason 'No assay lens is on, and other tables are not exported turned around' (core/interview.py:306-311). Classic offered the transpose for any file. The improvement holds only for metabolomics and genomics. | improved for assay lenses; narrower (not offered) for dietary, clinical and survey |
| C01-16, C01-18 | Cardinality and missing-values detail are equivalent; ColumnPicker draws dtype, n_unique and missing counts (ColumnPicker.tsx:154-157). | ColumnPicker is rendered only inside TargetAsk, the outcome picker (components/record/ask/FactQuestions.tsx:17, 108-137). There is no audit table. MissingAsk names blank columns only inside its option lines (ChoiceQuestions.tsx:222-276). Classic had dedicated cardinality, missing-values, numeric-statistics and duplicate-row expanders (pages/01_Upload_and_Audit.py:1015, 1044, 1070, 1089). | partial |
| C02-24 | Influence diagnostics are improved. | They are computed only in the inference effects stage, which the frontend never fetches, and respond_diagnostic is never sent. Classic drew leverage and Cook's distance in EDA for every regression task (pages/02_EDA.py:2919-2938). | partial: inference-only and engine-only |
| TC-11, TC-14, TC-18, TC-20, TC-30, C04-01, C04-05, C04-08, C03-02, RE-05, RE-12, RE-13 (status scheme) | These are counted among the 78 'better'. | Each depends on a decision the UI never sends (set_levers, set_selection, set_intended_use, set_explain, view_outcome) or on a stage or endpoint it never reads (explore, evaluation, export; useMethods, useFiles and useJoinPreview are defined but unused, api/queries.ts:243-285). Classic drew its versions: the class-weight toggle (06:2086-2110), the selection consensus (04:585), the TRIPOD checklist and Table 1 (10:2409, 2483). The report does disclose that 33 of 97 are off screen, but the headlin | engine-only: report them as 'built, not usable yet' rather than 'better' |
| C02-28 (tally) | Counted among the 19 'equivalent'. | The row says Classic never showed these actions ('so this is not a loss'). It is the same kind of row as LIB-01, which the report excludes, and it should be excluded too. | exclude from the parity tally (18 the same) |
| REPORT-§2 model shelf / §4d | 3 prediction families (linear, elastic net, boosted trees), plus inference families: ordinal, mixed, GEE, Cox, feature-wise. | Mixed, GEE, proportional odds and Cox inherit purposes ('prediction','inference') (core/models/base.py:193) and rank on the prediction shelf with a 'fair' prediction concern (core/models/repeated.py:605-606, 679-680). A screened elastic net registers under omics (core/methods/omics.py:1580; teaching/content.py:1129). Feature-wise is the only inference-only family (featurewise.py:311-313). None of these extra families has SHAP (explain.py:351-362). | wording correction (more families predict; same SHAP gap) |
| REPORT-§4c MICE | No MICE under prediction: v2 refuses multiple imputation there because it cannot be applied to new patients. | Classic's 'iterative' option was a single, outcome-free sklearn IterativeImputer fitted inside the pipeline (ml/pipeline.py:404-407), which can be deployed under prediction. v2's refusal concerns MI with the outcome in the imputation model (core/decisions.py:4460-4471). Under prediction v2 imputes only by median, most-frequent or the energy line (core/models/pipeline.py:548-571). So the real loss is multivariate in-fold imputation under prediction, with no ruling, and the stated rationale does n | lost without ruling (rationale corrected) |
| REPORT-§4c skew rationale | v2's stated reasons cover skew: it 'says nothing about which scale is right' (core/structural.py:33-36). | That passage is about the outcome's scale question (core/structural.py:16-22: 'a positive, markedly skewed outcome is asked the scale it is analyzed on'). It is not about transforming predictors. No stated position on Yeo-Johnson or log1p for predictors exists at that citation. | lost without ruling or stated reason |
| M-02, C05-19 | Lasso is improved; the penalized-linear family is equivalent. | A user cannot pick lasso or ridge. The inner CV picks a mix from (0.1…1.0) for regression and only (0.2, 0.6, 1.0) for logistic (core/models/elastic_net.py:16-19, 51-58), so sparsity is not guaranteed and pure ridge or pure-L2 logistic is never fit. Classic's default logreg was L2 with C=1 (ml/model_registry.py:400-405), and Ridge and Lasso were separately choosable (:324-368). | partial |
| TC-12, M-03 | At p>n the fit runs in float32 with threads (wide.py:1-15). | for_wide wraps only the regression ElasticNetCV (core/models/elastic_net.py:49-52). The penalized logistic path (:53-59) and the other families do not use it. | partial (regression elastic net only) |
| C05-16 | The double-transformation guardrail is improved (exposure_form.py:52-59). | v2 has no user log or power transform for predictors, so Classic's hazard (FE log, then preprocess log) cannot arise. The cited code is form-staleness invalidation, a different mechanism. PP-13 already treats the analogous case as 'nothing to guard'. | not applicable |
| REPORT-§4a ruling labels | In-app AI Deep Analysis, JSON upload and simulated theory demos were dropped on purpose, each with a ruling. | DoD:93-96 lists the AI assistant under 'deferred to v2.x', not dropped. JSON's absence from the format list (DoD:25) is an omission, not a ruling. BLUEPRINT.md:266-268 is a design principle ('the explanation of an option is its effect on the user's own data'), not a ruling on the Theory Reference. | deferred / unruled omission / principle-based |
| REPORT-§3 'Methods are checked' | A validation audit raised 170 findings; all 122 serious ones were re-tested and confirmed (AUDIT_REPORT.md:22-30). | The cited lines show 122 defects confirmed as real: 77 distinct critical or major problems (docs/turbotab-next/audit/AUDIT_REPORT.md:25-28). They do not show the defects closed. The DoD's final re-audit is still ahead on its road (V2_DEFINITION_OF_DONE.md:65-66, 106-107). Closure evidence, if any, sits in audit/fix-*-result.json, which the report does not cite. | plausible overstatement: process evidence, not verification evidence |

### Classic capabilities the inventory missed

- Built-in practice datasets (pages/01_Upload_and_Audit.py:746-760). v2's StartScreen offers only an upload or an opened path; no sample-data entry point.
- Numeric column statistics table with skewness and an IQR outlier scan (pages/01_Upload_and_Audit.py:1089-1096). No v2 audit table.
- Co-missingness pattern matrix (pages/02_EDA.py:1150-1165). There is no missingness-pattern view anywhere in turbotab/core.
- Residual normality: Q-Q plot and Shapiro-Wilk on a quick OLS (pages/02_EDA.py:2886-2899). v2's prediction lane has nothing equivalent.
- Pre-fit VIF table (pages/02_EDA.py:2905-2917). v2 has only a post-fit Belsley near-singularity concern for the linear family at condition number ≥1,000 (core/models/linear.py:43-89; core/stages/modeling.py:1837-1842).
- EDA 'Suggested Interactions (auto-detected)' by mutual information (pages/02_EDA.py:1801-1806). v2's H-statistic is post-fit and unreachable (explain.py:42-47).
- Cross-model feature-importance comparison table (pages/07_Explainability.py:1682-1700).
- Three-way train/validation/test split sliders and a user seed (pages/06_Train_and_Compare.py:119, 257-268). The v2 UI hard-codes seed 0 and 5 folds (SealQuestions.tsx:27-38); the report mentions the seed but not the lost validation split.
- Per-model k-means cluster features and constant or mean imputation (ml/pipeline.py:362-369, 410-415). The report's §4c lost list names PCA (as a spec gap) but not k-means features or these fills.
- Per-family missing-value handling: the one declared difference besides scaling (handles_missing) never takes effect in the flow (see the C05-22 correction). §4c should list it as a per-model capability Classic also lacked, so the 'one shared recipe' really is one recipe for every family except the elastic net's scaler.
- Pure-L2 penalized logistic, Classic's default logreg configuration (ml/model_registry.py:400-405). The report treats it as 'Little' miss, but it is the default many Classic users actually ran.
