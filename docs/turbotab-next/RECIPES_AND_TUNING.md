# Recipes and tuning

**Status: DRAFT 3, for Nolan's review.** The orchestrator wrote draft 1 on 2026-10-06, acting as methods expert, to answer Nolan's two questions of 2026-10-05. Draft 2, written the same day, took in every blocker and major from three reviews (methods; product and leash; engine) and the cheap minors. Draft 3, written on 2026-10-08, folds in every ruling made since: Nolan's of 2026-10-06 and 2026-10-07, the crosswalk's and the UI's of 2026-10-08, the definition of done's amendment of 2026-10-08 (the model-family contract in, package C6c out) and its seam guards. It settles the one methods question the rulings left to the orchestrator: how "Try both" runs below an effective size of 300 (§4.6). The same day it took in a review of draft 3 (two majors and eleven minors, each checked against its source first). "What changed in draft 3", "What changed after review of draft 3", "Review notes not taken" and "What changed after review (draft 2)" close the document.

**What it governs.**
- **BLUEPRINT:** North star 5, §4 (the stage graph), §11.3 (the leash), §11.4 (stated, asked and silent tiers) and §13 (the method contract and its relations).
- **MODELING_SEQUENCE:**
  - §0, rulings 3, 4, 6 and 13;
  - §1, rows 9–12 (row 10 is line 141, "Model-specific preprocessing");
  - the run order in §1.1, MS6 and the BBC-CV correction.
- **V2_DEFINITION_OF_DONE,** as amended on 2026-10-05, 2026-10-07 and 2026-10-08. Those amendments:
  - brought in ridge, Huber, random forest and XGBoost;
  - required nested tuning for boosted trees;
  - required boosted trees' native handling of missing values to be reachable;
  - made "Try both" the trees' default for blanks, and made long fits wait for Fit;
  - moved the kept comparisons to v2.x (2026-10-07), then displaced package C6c to v2.x (2026-10-08);
  - brought in the model-family contract with its live, outcome-blind ranking, and eight seam guards.
- **`crosswalk/CROSSWALK.md`:** the Models stage, rulings 5 and 6, and "Settled here" (Fit and the lock). **`crosswalk/SIZING.md`:** packages C6a, C6b, C6d and P0.8, which carry this spec's work packages at their own sizes (§9).
- **`MODEL_FAMILY_CONTRACT.md`:** C3 (inputs and recipe slots), C4 (assess on the actual input) and C6 (tuning) reference §2.4 and §4.1 here and add to them. §2.6 here follows C4.
- **`V2X_SEAMS.md`:** rows 1 to 13, the two missing hooks and the guards that keep the deferred items folding in cheaply (§9, §10).
- The parity report (`parity/CLASSIC_PARITY.md`) and the calm foundation (`calm/FOUNDATION.md`).

**What was read.**
- **For draft 1:**
  - `turbotab/core/models/`: `pipeline.py`, `base.py`, `elastic_net.py`, `boosted_trees.py`, `linear.py`, `inner_cv.py`, `selection.py`, `cost.py`, `validation.py`, `folds.py`, `previews.py`, `explain.py`, `causal.py`;
  - `stages/modeling.py` and `methods/omics.py`;
  - Classic's `pages/05_Preprocess.py`, `pages/06_Train_and_Compare.py` and `ml/model_registry.py`.
- **Added for draft 2:**
  - `models/design_cv.py`;
  - `models/selection.py`: the compared families, the declared result, `mark_final`;
  - `methods/levers.py`: `ImbalanceCorrected` and the form rule;
  - `contracts.py`;
  - `decisions.py`: `OpenSeal`, `view_outcome`, `register_kind`;
  - `reference/journeys.py`;
  - `teaching/content.py`;
  - the abstract of Perez-Lebel et al. 2022.
- **Read by the reviewers as well:** `seal.py`, `stages/evaluation.py`, `stages/__init__.py`, `export/replay.py`, `export/record.py`, `models/explain.py` and `frontend/src/components/record/ask/ChoiceQuestions.tsx`.
- **Added for draft 3:**
  - `HANDOFF.md`: the rulings of 2026-10-06, the UI and scope discussion of 2026-10-06/07, and the crosswalk and UI rulings of 2026-10-08;
  - `V2_DEFINITION_OF_DONE.md`: the amendments of 2026-10-05, 2026-10-07 and 2026-10-08, and the seam guards;
  - `crosswalk/CROSSWALK.md` (the six questions, "Settled here", the Models stage, disagreements 12 and 20) and `crosswalk/SIZING.md`;
  - `MODEL_FAMILY_CONTRACT.md` (§0, C1 to C6, §3, §5, §6 and its review notes);
  - `V2X_SEAMS.md`;
  - `decisions.py`: `SetMissing` and `MissingSpec`, for the split at the stage line.

Nothing was run, because of quiet hours.

---

## 0 · The answers, in plain words

### "Is there a downside to smart defaults, and to letting people break them?"

Yes. Defaults have their downsides and overrides have theirs. Neither is a reason to forbid anything.

**Defaults go wrong in three ways.**
1. **They are hidden.** A reader cannot reconstruct what was done (North star 4). Classic applied its Smart Defaults silently to every model. Its median-fill default also hid that multiple imputation existed.
2. **The data chose them outside the resampling.** Classic's defaults read the data profile before any model was validated: robust scaling when outliers were detected, Yeo–Johnson when columns were skewed. A choice the data make has to be made inside each training fold, or the score does not include it.
3. **They are customary but not sound, or the reverse.** A library default is not a methods argument. For example, scikit-learn's random-forest regressor tries every column at every split. That is bagging, not Breiman's random forest.

**Breaking defaults goes wrong in three other ways.**
1. **Choosing after seeing scores flatters the winner.** The rows that chose a version also grade it (Varma & Simon 2006; Cawley & Talbot 2010). That includes the quiet route: typing in by hand the values a tuning run picked on these rows.
2. **The comparison stops being like for like.** If trees get one fill and the linear model another, a difference in their scores mixes up the model with its inputs.
3. **More knobs mean more ways to pick an unsound one.** For example, rebalancing the classes makes every predicted probability wrong.

**So TurboTab allows overrides per model, with four safeguards.** This spec builds all four.
1. **Every default is stated with its reason**, as a sentence whose phrase you can click. In the quest log it sits in the stage's Confirm sweep, or under For the record when it changes nothing on your table.
2. **Every override is recorded, and the comparison shows it.**
3. **A choice made after you have seen first results is disclosed.** Where the design allows, the earlier version also stays in the comparison, so the final score accounts for choosing among the versions you saw. Two limits are said plainly:
   - nothing corrects for how a later version was designed after the earlier scores were seen;
   - a change to the shared steps (who is kept, the fill, Explore's levers, the selection step, the energy model, the forms) is disclosed, not corrected. Its earlier scores stay in the comparison for reference only, labeled "revised after first results" (§3.2).

   A holdout drawn before any score is seen removes both limits. Before the first fit, the app suggests one when you expect to try several versions.
4. **Unsound overrides get the leash, and leakage stays impossible by construction.** Values copied from a tuning result on these rows count as tuned on these rows, and so does fixing the option the folds already chose (§3.5).

**A third path:** besides "keep the default" and "change it", there is **"Try both; keep what predicts better."** The alternatives join the tuning search inside each training fold, so the reported score already includes the choice. It is the tree models' default for blanks: each training fold chooses between keeping blanks as blanks and filling them (§2.3).

There is no "Advanced mode" (BLUEPRINT §11.4).

### "Does TurboTab support hyperparameter optimization?"

**Today, partly.**
- The elastic net tunes its penalty and its lasso–ridge mix inside every training fold.
- Nothing else is tuned. Boosted trees run at scikit-learn's standard settings.

**After this spec, yes.**
- Every flexible family has a declared search space, with ranges from the literature.
- A seeded random search runs inside each outer training fold. Its inner splits keep each person's rows together, respect time, and keep survey PSUs whole, as the outer splits do.
- **One plan** is fixed before the first fit and used in every fit, so the procedure that is scored is the one that is deployed.
- The budget is known in advance, so the time is shown first. A fit longer than about 2 minutes waits for you to press Fit on the analysis flowchart. It then runs as a server job, and you are told when it is done.
- After the fit you see the tuned values and how much they varied from fold to fold.
- You may set values by hand. They are labeled and disclosed.

**Under inference, tuning means almost nothing.** The reported models have no settings to tune. Their real choices are declared in the analysis plan (§5), and pressing Fit locks that plan.

### What v2 builds, and what waits

**Built in v2** (SIZING C6a, C6b and C6d, with RT-8 in P0.8):
- **Four new families:** ridge, robust linear regression (Huber, under prediction), random forest and XGBoost, each through the model-family contract.
- **Nested tuning** for boosted trees, XGBoost and the random forest. The penalized families' penalty is searched by the same engine.
- **Tree models' own handling of blanks made reachable,** with "Try both" as their default.
- **The overrides:**
  - the four safeguards;
  - "Try both; keep what predicts better";
  - values set by hand;
  - the estimate shown first, and long fits waiting for Fit.
- **Versions after first results:** a changed recipe or tuning keeps its earlier version in the comparison; a changed shared step keeps its earlier scores for reference only. Both are labeled "revised after first results", in the comparison and the methods text only.
- **The pre-fit ranking** reads each family's actual input, through the recipes here (§2.6; the contract's C4).
- **Three seam hooks** for the v2.x items (§9).

**Waits for v2.x** (each is in §10, with its seam):
- **Package C6c, displaced on 2026-10-08:** successive halving, Hyperband, TPE and BOHB; the Thorough budget and the tuning curve; native categories; per-family Pareto and robust scaling; the per-model log1p; Huber under inference; nuisance learners for DML and TMLE built from the family registry, and their tuning.
- **The kept comparisons, moved on 2026-10-07:** "Compare with standard settings" as a kept version; shared-step changes kept as versions that are fitted and compared; re-tuned substitution bands.
- Smaller items from drafts 1 and 2: monotone constraints, LightGBM, survival versions of the new families, flat BBC-CV and a cross-validated penalty for post-double-selection lasso.

---

## 1 · What the code does today

| # | Finding | Where | Consequence |
|---|---|---|---|
| F1 | Boosted trees' native handling of blanks is dead code. The missing-values question is always asked, and every answer that keeps the rows fills each blank first. | `interview.py`; `pipeline.shared_steps`; the comment at `teaching/content.py:1122` | Trees never see a blank. |
| F2 | Boosted trees are not tuned. | `boosted_trees.py` `build` | DoD §2 asks for nested tuning. |
| F3 | Families differ only in the elastic net's standardization, the screened elastic net's screen, and the imbalance wrapper (applied to every family when the lever is on). | `pipeline.family_steps`, `build_pipeline` | "Model-specific preprocessing" is in practice one recipe. |
| F4 | The screened elastic net tunes its penalty on features screened with the outcomes of the inner validation rows. The screen is fitted on the whole outer training fold before `ElasticNetCV` splits it. | `methods/omics.py`, `elastic_net.py` | The outer score stays honest, but the penalty is biased toward too little shrinkage. |
| F5 | The mix grid starts at 0.1 (0.2 for logistic). The logistic `Cs` grid is fixed, not scaled per row. The path estimators average per-fold scores: `RidgeCV` with explicit splits averages R², and none of them is survey-weighted. | `elastic_net.py` | The inner objective is not the pooled primary score (ruling 4), nor weighted under the population answer (ruling 13). |
| F6 | `cost.time_one_fit` times one `pipeline.fit`. | `cost.py` | A search would be timed as one untuned fit. |
| F7 | The shelf counts bootstrap refits for families that are never bootstrapped. | `stages/modeling._estimates` | Trees' estimates run long. |
| F8 | SHAP has code paths only for HistGB and five named linear classes. | `explain.model_kind`, `LINEAR_MODELS` | The new families need paths. |
| F9 | The causal lane's lasso draws its inner folds by row (`KFold(5, shuffle=True)`). | `models/causal.make_learner` | When rows repeat, one person can sit on both sides of the penalty's split. |
| F10 | `inner_cv.fit_pipeline` is the path every stage-level fit takes: all 22 call sites in 9 files. The exceptions are the timing fit and `fit_pipeline`'s own early-stopping branch. | a census of `stages/*.py` | A search placed there is nested everywhere. |
| F11 | The early-stopping branch fits the steps before the model on every row, the stopping rows included, and only then splits off `X_val`. | `inner_cv.fit_pipeline` | When a step reads the outcome (selection, screen, inner-CV forms), stopping is optimistic. |
| F12 | `ImbalanceCorrected` has no `early_stopping` parameter. The wrapped HistGB therefore draws its own stopping split, by position and after resampling, and the wrapper's recalibration `cv` holds position splits for the outer fit's rows. | `methods/levers.py`, `inner_cv._stops_early` | Under oversampling, copies of one row sit on both sides of the stopping split. |
| F13 | Thresholds are read from each fit's own row count: `EARLY_STOPPING_ROWS`, and the elastic net's `inner_folds(n_rows)`. | `inner_cv.py`, `elastic_net.py` | Outer folds and the final refit can run different procedures near a threshold. |
| F14 | Scores seen, compared families, `OpenSeal.family`, `AtOpening.scores` and `mark_final` are keyed by family. `compared_families` returns nothing once any holdout exists, even one drawn after scores were seen. | `selection.py`, `decisions.py`, `seal.py` | Versions within a family cannot be tracked, and a late holdout frees compared families. |
| F15 | `fit_pipeline` takes groups, order and a seed: no strata, PSUs or weights. No stage passes a seed. | `inner_cv.py`, `stages/*` | Inner splits cannot follow a design, and every inner draw uses seed 0. |
| F16 | **New in draft 3.** `scores_seen.json` holds only family keys per outcome, and the stage cache that holds earlier fits "stays disposable". | `selection.py:496`; `graph.py:307` | An earlier fit's scores, once the cache is cleared, cannot be shown again or rebuilt (V2X_SEAMS row 11). |

F10 is why the design below stays small. F11–F16 must be fixed before the search and the versions are built on them.

F11 is fixed in `inner_cv.fit_pipeline` (2026-10-09, `turbotab/core/tests/test_stopping_rows.py`): the stopping units are drawn first, and the steps and their inner splits are fit on the other rows. RT-1's `fit_parts` keeps that order.

F12 is fixed in `methods/levers.ImbalanceCorrected` (2026-10-09, `turbotab/core/tests/test_imbalance_stopping_rows.py`). The wrapper declares `early_stopping` and `validation_fraction`, so `fit_pipeline` hands it the stopping rows as `X_val`. Only the other rows are resampled. The deployed fit and every recalibration fit stop on the stopping rows as they are, and the recalibration splits cover only the rows the fit trains on. The 10,000-row threshold reads the rows the wrapper receives, never the resampled count. `inner_cv` reads the wrapper's effective setting (`stopping_setting`, which defers to the wrapped model), so a wrapper built with its defaults takes the same path. Fit on its own, the wrapper draws whole units by the `groups` or `order` it is handed; given inner splits without either, it refuses rather than draw rows that split a person. This is RT-4's interface; the search itself is still to come.

**How Classic did it, for contrast.**
- One preprocessing tab per model, with a Smart Defaults / Advanced radio. Smart Defaults filled the options in from the data profile.
- Optuna's TPE ran 30 unseeded trials, each scored by accuracy or RMSE on one validation split outside the cross-validation.
- **v2 keeps:** choosing per model, and values set by hand.
- **v2 drops:**
  - the modes;
  - defaults the data chose outside the resampling;
  - tuning on one split;
  - improper objectives;
  - unseeded searches.

---

## 2 · Recipes

### 2.1 The rule

**The analysis has one shared trunk:** the answers to its own questions.
- values below a detection limit, and normalization;
- who is kept (Who's in), and the track's fill (Models; §2.3);
- batch correction and scale scores;
- the energy model and the declared forms;
- blanks as a level of their own, and one-hot encoding;
- Explore's levers, and the selection step.

Scales, batch correction and the omics normalization are answered before the families question, so the ranking sees each family's real input (DoD, 2026-10-08).

**Each family adds or replaces only what it declares.** That is its *recipe*. A recipe has five slots, and a family declares only the slots that can change something for it:

| Slot | What it decides |
|---|---|
| `missing` | whether the family takes the track's fill, the blanks themselves, or tries both |
| `scale` | how its columns are scaled before a penalty |
| `encoding` | how a category's levels become columns |
| `transform` | a per-model reshaping of numeric predictors |
| `outliers` | a per-model cap on extreme predictor values |

**What each option carries:**
- a plain label and a quiet term;
- a consequence of 16 words or fewer;
- a *customary* label with its source;
- a *sound* label for each purpose;
- a leash rung for each purpose.

**Tiers, and where they sit on screen.** The engine's tier names never reach the UI; the quest log's labels do (§6.0).
- **Stated:** the default appears as a sentence with a clickable phrase. On screen it is a line in the Models **Confirm** sweep, because its alternative would change a number.
- **Silent:** its option's footprint on this table is empty, as measured by the consequence planner. On screen it sits under **For the record**, collapsed, and in the export. Examples:
  - a missing slot on a table with no blanks, or when no blank reaches the family;
  - "every level kept" with no categorical predictor;
  - any slot whose options change no number, such as scaling for the unpenalized linear model.
- **Asked:** none. No recipe slot is a **Decide**.

**The amendment to row 10.** "Not a question" becomes: "Stated per family, with its reason, and silent where it changes nothing. Under prediction its phrase can be changed with `set_recipe`, under the four safeguards (§0). Under inference the plan decides (§5)."

**Families outside this spec** take the shared trunk and have nothing to tune: `featurewise`, `proportional_odds`, `mixed`, `gee` and `cox`.

### 2.2 Every family's recipe

**The shared fill** is the Predict track's fill, learned within each training fold without the outcome:
- the median for numbers;
- the line on total energy for energy-bearing nutrients (`EnergyAwareImputer`);
- the most frequent value for categories and two-valued numbers;
- a missing indicator per filled column, when the track's fill asks for one.

**Defaults** (bold marks a departure from the trunk; "option" means available under prediction):

| Slot | Linear | Robust linear (Huber) | Ridge, elastic net | Random forest, boosted trees, XGBoost |
|---|---|---|---|---|
| missing | shared fill | shared fill | shared fill | **Try both** (§2.3): each training fold chooses between keeping blanks as blanks and the shared fill. Options: keep blanks as blanks; the shared fill |
| scale | silent: changes no prediction | silent: its fit rescales itself | **every column on one scale**. Options: Gelman's two-SD scaling; columns in their units (ranked lower) | not declared: no split moves |
| encoding | first level as reference | first level as reference | **every level its own column**. Option: reference (ranked lower) | **every level its own column**, with no option |
| transform | none. Option: Yeo–Johnson (ranked lower); Try both | as linear | as linear | not declared |
| outliers | none. Option: cap at the 1st and 99th percentiles (ranked lower); Try both | none: the model is the answer | as linear | not declared |
| settings | nothing to tune | threshold fixed at 1.345; may be set by hand | penalty, and for the elastic net the mix, searched along a path (§4.1) | searched (§4.1) |

**Reasons and labels** (C means customary, with its source; S means sound for prediction, with its reason):

- **Shared fill.**
  - Reason: "A straight line needs a number in every cell; the fill is learned on training rows only."
  - C: median fill. S: it can be deployed and never sees the outcome (Sisk et al. 2023).
- **Try both (the tree models' default for blanks).**
  - Reason: "Each training fold tries keeping blanks and filling them, and keeps whichever predicts better there; the score includes that choice." (20 words)
  - C: fill first.
  - S: whether a blank carries meaning on this table is unknown before the fit, and the choice is nested, so the reported score includes it (Varma & Simon 2006; Bischl et al. 2023). Each option's own case is below.
- **Keep blanks as blanks.**
  - Reason: "Each split learns which way a blank goes, so a blank can carry meaning."
  - S: Josse et al. 2024 (consistency); Perez-Lebel et al. 2022: "Native support for missing values in supervised machine learning predicts better than state-of-the-art imputation with much less computational cost."
  - **Caveat, stated in the sound label:** it relies on a blank meaning the same thing where the model is used. Sisk et al. 2023 found that missing indicators can harm prediction under outcome-dependent missingness, and routing uses the blank in the same way. "Try both" does not remove this caveat: it tests the blank's meaning on these rows, not where the model will be used.
- **The shared fill, for trees.**
  - S: like for like with the other models. Under "Try both" it carries indicators for the routed columns, so the contest is fair (Perez-Lebel et al. 2022: "When using imputation, it is important to add indicator columns"). Many indicators with few rows can overfit (Van Ness et al. 2023), which the folds' choice can detect.
- **Every column on one scale.**
  - Reason: "The penalty pulls every coefficient by one rule, so the columns must share a scale."
  - C: glmnet's default; ESL §3.4.1.
  - S, with a tension stated: standardizing a yes/no or level column divides it by √(π(1−π)). On its own scale, a rare level is then penalized *less* than a common one, which is backwards for the coefficients with the least information. Gelman's two-SD scaling avoids this: continuous columns are divided by two SDs, and 0/1 columns stay 0/1 (Gelman 2008).
- **Every level its own column (penalized families).**
  - Reason: "With a penalty, a reference level would be shrunk differently from the rest; keeping all treats them alike."
  - C: drop the first level. S: glmnet's `makeX`.
- **Every level its own column (trees).**
  - Reason: "A tree can then split off any level, the first one included."
- **Reference level (linear, robust linear).**
  - Reason: "With an intercept, one level must be the reference."
- **No transform.**
  - Reason: "The shape of an effect is the curve question's job, asked once for every model."
  - C: Yeo–Johnson for skew (Classic's Smart Defaults).
  - S: ranked below the spline rule, because a column's skew says nothing about the shape of its effect.
- **No cap.**
  - Reason: "Capping changes your data; a robust model limits outliers' pull instead."
  - C: percentile capping. S: ranked lower.

**Robust linear regression (`huber`). New; prediction only; numeric outcomes only.**
- **Card label:** "Robust linear regression", with the quiet term "Huber".
- **Inductive bias:** "Straight-line effects fitted so that far-off rows count less."
- **How it is built:** a thin scikit-learn wrapper around statsmodels' `RLM` with `HuberT(t=1.345)`, the scale re-estimated by MAD, and no penalty. The threshold keeps 95% of least squares' efficiency under normal errors (Huber 1964; Holland & Welsch 1977). It may be set by hand. It is not tuned by cross-validation, because squared error, the primary score, would walk it toward no robustness at all.
- **Its caution, worded exactly:** "Its slopes match least squares' when the errors are spread the same way at every value of the predictors. With skewed errors its predictions shift from the mean toward the median, and squared error charges it for that."
- **Not offered under inference** in v2 (§5). It declares `purposes = ("prediction",)`; nothing keeps it out of inference by its key (V2X_SEAMS row 8; the contract's no-switch test).

**Random forest (`random_forest`). New.**
- **Inductive bias:** "Many deep trees, each on resampled rows with random columns, averaged; effects are step functions."
- **Standard settings:**
  - 500 trees;
  - columns tried per split: √p for classes, p/3 for numbers;
  - smallest leaf: 10 rows for probabilities (ranger's probability-forest default) and 5 for numbers;
  - every row drawn, as a bootstrap.
- **Departure stated:** scikit-learn's regressor uses every column at every split (`max_features=1.0`), which is bagging.
- **Sources:** randomForest (Liaw & Wiener 2002); ranger (Wright & Ziegler 2017); Probst, Wright & Boulesteix 2019.
- **Not the causal lane's forest.** That learner (200 trees, at least 5 rows per leaf) is renamed `nuisance_forest`, with a parse alias for the old spelling (the contract's §3.3; seam guard 2).

**Boosted trees (`boosted_trees`, histogram gradient boosting). Exists, amended.**
- It is tuned (§4.1). Its standard settings are always among the candidates, once for each option a "Try both" slot tries (§4.2).
- Its `defaults_version` becomes "2": blanks tried both ways, and tuned. Version "1" is today's fill and standard settings (§3.2).
- `describe()` states the plan and the tuned values.

**XGBoost (`xgboost`). New.**
- **Card consequence:** "The same kind of model as boosted trees, in the XGBoost library reviewers often name." The preview names the minutes it adds.
- **Blanks** follow a default branch learned in training (Chen & Guestrin 2016).
- **Only the tree booster is declared.** The linear booster "treats missing values as zeros", and ridge and the elastic net cover linear models.
- **`scale_pos_weight` is not a declared setting.** Reweighting goes through Explore's imbalance lever, which recalibrates.
- **The wrapper also provides:**
  - the early-stopping interface of HistGB (`early_stopping`, `validation_fraction`, `fit(X, y, X_val, y_val)`);
  - a margin `decision_function`;
  - label encoding to 0…K−1;
  - safe column names: `[`, `]` and `<` are replaced, and the original names are mapped back.

**Ridge (`ridge`). New.**
- **Inductive bias:** "Straight-line effects all shrunk toward zero together; correlated predictors share weight; none is dropped."
- **Card consequence:** "A penalized straight-line model that shrinks every effect and keeps every predictor."

**Elastic net (`elastic_net`, `screened_elastic_net`). Exists, amended.**
- It has the same recipe as ridge.
- Its penalty and mix are searched by the shared path search (§4.3), which refits the screen in every inner split. This fixes F4 and F5.

**Registration order:** `linear`, `elastic_net`, `ridge`, `huber`, `boosted_trees`, `random_forest`, `xgboost`.

**Tasks.**
- Ridge, the random forest and XGBoost take numeric, yes/no and multiclass outcomes. An ordered outcome is fit as unordered classes, with the existing order-blind cost.
- Robust linear regression takes numeric outcomes only.
- Time to event waits for v2.x.

### 2.3 Blanks: who is kept, the track's fill, and each family's recipe

**The split at the stage line** (crosswalk ruling 5, 2026-10-08). The old missing-values answer did two jobs. They now sit in two stages:
- **Who's in, once for every track:** who is kept and what each blank means. That is complete cases on named columns, or keep every row; the columns left out; the blanks that mean "not asked" or "below detection". Complete cases is valid for every goal.
- **Each track's Models:** how the kept blanks are filled.
  - **Predict:** the shared fill, learned in each training fold, with its indicator and category-level choices. It is stated, so it sits in the Models Confirm sweep. Each family's missing slot then decides whether the family takes that fill, the blanks themselves, or tries both.
  - **Estimate:** multiple imputation compatible with the track's own model, or a single fill under block and record. It belongs to the plan (§5).
  - **Describe:** each quantity is estimated on the rows that hold its values, with the share of blanks stated. No family is fitted.

SIZING D2 builds the split in the engine: it moves the fill's fields out of the shared `SetMissing` answer and into each track's Models. For this spec, one thing matters: **the missing slot reads the track's fill, never the shared answer.**

**The rule, under Predict with every row kept.** A tree family takes "Try both" by default (`choose: [native, fill]`, the ruling of 2026-10-06):
- each training fold chooses, on its inner folds, between the blanks themselves in every routed column and the shared fill with indicators for those columns;
- the choice is part of the search (§3.3(c), §4.2), at every size: below an effective size of 300 it is made at the standard settings (§4.6);
- the score includes the choice;
- the researcher may set either option instead, under the four safeguards.

**Routing is computed, not flagged.** Each step declares:
- `reads(spec) -> columns`: the columns whose values its output needs;
- `passes: frozenset`: the kinds of raw value it passes through unchanged in columns it does not read. v2 declares only `"blanks"`. Native categories add `"categories"` in v2.x without a second sweep of every step (V2X_SEAMS row 5).

`family_spec` walks the family's step list in order and computes `native`: the columns no blank-intolerant step reads. The imputer and the indicators then skip those columns.
- **Pass blanks through:** the log step, the levels and one-hot steps, the impute `ColumnTransformer` (for columns it does not fill), the scaler and the variance filter.
- **Blank-intolerant on the columns they read:** the energy adjustment, the omics normalization, batch correction, scale scoring, a declared form and the selection step.

**Three consequences, stated.**
- **Tree families skip Explore's spline rule.** The basis adds only monotone copies of a column, which move no split, and that keeps the trees' columns routable. This is consistent with their transform slot not being declared.
- **The selection step reads every input.** When it is on, no blank reaches the trees, and the recipe line says so: "No blanks reach the trees: the selection step needs every value."
- **When no column is routed, "Try both" has nothing to try.** The two options then build the same input, so the missing slot is silent and the plan adds no candidate for it (§4.2).

| Who's in · the track's fill | Goal | Linear, ridge, elastic net, robust linear | Tree families |
|---|---|---|---|
| Complete cases | every goal | Rows with a blank predictor leave. | The same rows. The slot is silent (no blanks remain). |
| Keep every row · the fill in each training fold | Predict | The shared fill, with indicators if the track's fill asks for them. | **Default: Try both.** Each training fold chooses between blanks kept in the routed columns and the shared fill with indicators for those columns. Columns that a blank-intolerant step reads are filled first in both options. **Options:** keep blanks as blanks (no indicator for a routed column; the sentence says so: "the blank is its own signal"); the shared fill, exactly as the other models get it. |
| Keep every row · one fill | Estimate | The shared fill for scores; the coefficient table follows the plan. | The shared fill. Keeping blanks is not offered. |
| Keep every row · multiple imputation | Estimate | Each completed copy has no blanks. | The same. |
| Keep every row · multiple imputation | Predict | Refused today (`decisions.py:_missing_fits_the_purpose`). | Refused. |
| Blanks as their own level (the track's fill, for categories) | Predict, Estimate | Categorical and two-valued columns get a `Missing` level. | The same. Routing covers numeric columns only. |
| Keep every row | Describe | No family. | No family. |

**What a deployed tree does with a blank.** If a column had no blanks in a training fold, a later blank goes:
- for HistGB and the forest, to the child with more samples (scikit-learn's documented rule);
- for XGBoost, to the library's default branch.

The lineage note says which. When the final refit chose the fill, a later blank is filled as in training, and the note says that instead.

**Explanations.**
- TreeSHAP follows each split's learned direction for blanks.
- Effect curves are drawn over observed values, and the caption counts the rows whose value was blank.
- Substitution curves are unchanged: energy sources are always filled.

**The Who's in card**, under every goal:
- The option labeled "Fill in each fold" today (`teaching/content.py:879`) is **relabeled "Keep every row"**, because who is kept and how blanks are filled are now decided apart. Its partner "Multiple imputation" moves to the Estimate track's Models with D2, so the card offers "Complete cases" and "Keep every row", with the columns left out.
- The consequence of "Keep every row": "Everyone stays in; how each model handles their blanks is set in Models." (13 words)
- **One label, one meaning.** The exclusions card in the same stage already says "Keep every row" for no restriction (`teaching/content.py:786`). That option is relabeled "No restriction", so Who's in never shows one label with two meanings.
- The Estimate track's fill question moves to its Models with D2; its copy is the plan's and is unchanged.
- The comment at `teaching/content.py:1122` is corrected.

### 2.4 How a recipe is declared (`base.py`)

```python
from dataclasses import dataclass, field

RecipeSlotName = Literal["missing", "scale", "encoding", "transform", "outliers"]

@dataclass(frozen=True)
class RecipeOption:
    key: str                      # "native", "fill", "standard", "two_sd", "none", "every_level",
                                  # "reference", "yeo_johnson", "winsorize"
    label: str                    # plain: "Keep blanks as blanks"
    term: str                     # quiet: "missing incorporated in attributes"
    consequence: str              # ≤ 16 words
    customary: str                # with its source
    sound: Mapping[str, str]      # purpose -> why it is (or is not) sound
    rung: Mapping[str, Rung]      # purpose -> contracts.Rung

@dataclass(frozen=True)
class RecipeSlot:
    slot: RecipeSlotName
    default: Mapping[str, str]    # purpose -> option key: the one option used where the folds
                                  # cannot choose (§4.6), and under inference
    reason: str                   # ≤ 22 words
    options: tuple[RecipeOption, ...]
    default_choose: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
                                  # purpose -> the options a "Try both" default tries, in order;
                                  # the first is `default` and wins ties. Trees' missing slot:
                                  # {"prediction": ("native", "fill")}
    searchable: bool = False      # may join "Try both; keep what predicts better"

class FamilyBase:
    ...
    recipe: tuple[RecipeSlot, ...] = ()
    tuning: TuningDecl | None = None   # §4.1
    defaults_version: str = "1"        # bumped whenever a default changes; part of a version's identity
```

**The existing flags become derived:**
- `needs_scaling`: the default scale option is not `none`;
- `handles_missing`: `native` is among the missing options.

**`register_family` checks:**
- every option is labeled for both purposes;
- each default exists and is offered for its purpose;
- a `default_choose` names at least two options of a searchable slot, each offered for its purpose, and starts with that purpose's `default`;
- the word budgets;
- a family may declare `native` only when its estimator accepts NaN (checked by a fit on a three-row frame with one blank).

**The recipes travel inside the spec.** `DesignSpec` gains `recipes: Mapping[family, RecipeSpec]`. `build_pipeline`, `family_steps` and `describe_steps` apply `family_spec(spec, family, purpose)` themselves, so none of the roughly nine call sites can forget it. `family_spec` resolves these fields:

| Field | What it holds |
|---|---|
| `native` | the routed columns; the imputer's and the indicators' lists exclude them |
| `onehot_drop` | `"first"` or `None`, read by both one-hot paths (`MissingLevelEncoder` gains `drop`) |
| `scaler` | `standard`, `two_sd` or `none`: one `PenaltyScaler` exposing `center_` and `scale_` per column (1 and 0 where a column is not scaled). `elastic_net.coefficients`, `explain._undo_scaling` and `linear_equation` undo it by reading `center_` and `scale_`, never the scaler's name, so a later scaler (Pareto, robust) needs no new undo path (V2X_SEAMS row 6) |
| `transform`, `outliers` | an in-fold step, after Explore's steps and before scaling |
| `skip_form_rule` | true for tree families |
| `options` | for a slot under "Try both": the resolved spec of each option, so a search fits one head per option |

`GET /api/models` (`FamilyInfo`) gains `recipe` and `tuning`. The model-family contract adds its own members beside them (its MC-1).

### 2.5 How a recipe is drawn

- **Lineage.** The trunk is drawn once. A family that departs gets a short spur where it departs, grouped by shared departure: "blanks kept or filled, chosen in each fold · boosted trees, random forest, XGBoost". Pointing at a family lights its path in the choice color.
- **Model-matrix preview.** `previews.models_preview` compares each family's `family_spec` with the shared spec, not step names, because routing changes no step name. Its caption names the difference in 20 words or fewer. Under "Try both" it names both inputs: "Boosted trees try 42 columns with 1,204 blanks in 6, or 48 columns filled with 6 markers." `table_focus` shows five rows of the touched columns, with blank cells reading "blank". It shows predictor values of training rows only, never the outcome.
- **The set_recipe preview** also returns the edited family's input profile and assessment (the contract's MC-6), so a held change shows how the shelf would order the families before it is recorded. Like the profile, it stops before the first step that reads the outcome's values (§2.6), so a preview never runs the selection step, the screen or the inner-CV forms.
- **`DesignModel`** gains:
  - `recipe: list[RecipeLine]` (slot, option, label, term, reason, changed, when, silent);
  - `variant` (§3.2);
  - `plan` (§4.2).

### 2.6 The pre-fit ranking: the shelf, on each family's actual input

The shelf orders the families on the families question, a Decide in Models. It follows the model-family contract's C4 (DoD, 2026-10-08), which this spec feeds through `family_spec`. This section replaces draft 2's fixed shelf scores.

**When it ranks.**
- **Live, at model selection.** It waits only for the Decide answers that come before the families question:
  - who is kept, and categories, from the earlier stages;
  - the energy model and the forms, where they apply;
  - scales, batch correction and the omics normalization, which now come before the families question;
  - under Predict, the selection question when it is asked.

  Until they are in, the shelf shows "Waiting for: [question]", with a link.
- **It does not wait for the Confirm sweep.** The track's fill, the levers, the validation scheme, the in-fold omics steps, the recipes, the trees' "Try both" and tuning all sit in the Models Confirm sweep, which comes after the families are chosen. The shelf ranks on the defaults in force and says so once: "Ranked on the defaults now set." When the sweep or a later answer changes one, the shelf re-ranks. That is legal because the shelf reads no outcome relation and no score: re-ranking is a recomputation, not a choice made after seeing results.

**What it reads: the input each family would actually receive.** For every registered family that can model the task, the contract's input profile is computed through `family_spec` (§2.4) on:
- **an unselected family:** its stated default recipe and tuning (§2.2);
- **a selected family:** its current `set_recipe` and `set_tuning` slots, so a recorded recipe change re-ranks it, and a held one is previewed (§2.5);
- **a "Try both" slot, the trees' default:** one profile per option. The card shows both options' measures, and the score is the lower of the two assessments, because the shelf cannot know which option the folds will pick.

The profile's rows are the training rows, so its count matches the line "Ranked for N training rows"; n_plan (§4.2) stays the tuning plan's size.

**Where the profile stops** (the contract's C4 and MC-4). "The input each family would actually receive" never means running a step that reads the outcome to build the profile. Each step declares `reads_outcome`: `"none"`, `"counts"` or `"values"`.
- **Steps that read only the outcome's counts** (the spline rule's k, the imbalance correction's class counts) are fed those counts, and the profile continues past them.
- **The profile stops at the first step that reads the outcome's values:** the selection step, the omics screen, the inner-CV forms, and, under inference, multiple imputation. Its measures are taken on the matrix entering that step, labeled "before selection" (or "before the fill"); widths after it come from the step's declared size rule, and the card says "by rule".
- **Under inference,** the measures that read predictor values use one fill that ignores the outcome, labeled so, and blank counts come from the blank mask, never from a fill.

The `set_recipe` preview stops at the same place (the contract's MC-6).

**What "outcome-blind" means here** (ruled on 2026-10-08):
- **no relation between a predictor and the outcome, and no score.** The profile builder takes no outcome argument;
- **the outcome's own counts are allowed:** events and class sizes feed Riley's minimum, events per variable and the effective size, and the count-only steps (the spline rule's k, the imbalance correction's class counts).

The contract's outcome-permutation test enforces exactly this: permuting the outcome keeps its counts, and the ranking must not move.

**What it never does.** The shelf orders the families; it never picks one. After Fit, the corrected comparison (BBC-CV) decides (§3.4).

**The four new families' assessments.** Each family's `assess` reads its profile fields (the contract's §3.2). The base scores below come from the profile's training rows and columns, as draft 2's shelf scores did. The contract's MC-5 adds each family's other reads, and every change to a score names its profile field and source, or says "convention". No validated rule predicts which family will perform best, so these are conventions.

| Family | Base score | Profile fields read (the contract's §3.2) |
|---|---|---|
| Ridge | 2.0. At p ≥ n, 3.0 (below the elastic net's 4.0). | rows, columns, p/n, concentration, spectrum, condition number, indicators |
| Robust linear | 1.5 (numeric only; after linear on ties). | rows, columns, outlying share |
| Random forest | 2.0 from 2,000 training rows; 1.5 from 500; "fair" at 1.0 below 500. When predictors outnumber rows, "fair" at 1.5. Under inference, as boosted trees. | rows, columns, blanks routed, irregularity |
| XGBoost | Boosted trees' assessment less 1.0 (at least 0.5), read through `same_kind_as`, never by key: the same kind of model, offered by name. | boosted trees' reads |

At 2,000 training rows or more the order is therefore: boosted trees 3.0, elastic net 2.5, then the forest, ridge and XGBoost at 2.0, before MC-5's further reads. The two families the reference journeys pick today (`first_models`) stay first. RT-13 checks every journey's pick before and after (§9).

---

## 3 · Overrides and versions

### 3.1 The decisions

```python
class VariantSpec(_Value):            # a version's identity: every resolved choice
    family: str
    defaults_version: str
    space_version: str | None = None
    recipe: dict[RecipeSlotName, str] = {}            # every declared slot's option
    choose: dict[RecipeSlotName, list[str]] = {}      # every "Try both" slot, defaults included
    tuning: Literal["automatic", "lighter", "standard", "manual"] = "automatic"
    values: dict[str, float | int | str] = {}

class Stamp(_Value):                  # filled by the server's completion, never by a client
    target: str | None
    track: str | None                 # D2 scopes every stamp to its track
    shown: list[VariantSpec]          # every version whose scores were served for this outcome
    tuning_shown: list[str]           # versions whose tuned values or fold shares were served
    holdout: Literal["none", "sealed", "after_scores", "opened"]   # §3.3

class SetRecipe(_DecisionModel):
    kind: Literal["set_recipe"] = "set_recipe"
    family: str
    overrides: dict[RecipeSlotName, str] = {}      # one option for a slot; replaces its default,
                                                   # a default "Try both" included
    choose: dict[RecipeSlotName, list[str]] = {}   # "Try both; keep what predicts better"
                                                   # both empty: back to the stated defaults
    note: str | None = None                        # the researcher's reason, quoted
    stamp: Stamp | None = None

class SetTuning(_DecisionModel):
    kind: Literal["set_tuning"] = "set_tuning"
    family: str
    mode: Literal["automatic", "lighter", "standard", "manual"] = "automatic"
    values: dict[str, float | int | str] = {}  # manual: the changed values, the rest standard;
                                               # automatic: values held, the rest searched
    source: str | None = None
    stamp: Stamp | None = None

class DeclareVersion(_DecisionModel):           # prediction with no holdout (§3.4)
    kind: Literal["declare_version"] = "declare_version"
    variant: str
    stamp: Stamp | None = None
```

**Slots.**
- `set_recipe` and `set_tuning` write slots keyed by family (`register_kind(..., key=lambda d: d.family)`); D2 adds the track to the key.
- `declare_version` writes one slot per track's outcome.
- **"Lighter" is a `set_tuning`.** Choosing it in the Fit action's confirmation (§6.2) records `set_tuning(mode="lighter")` for each listed family before the job starts. The tuning line then reads "Tuning: lighter · changed by you" in the Models Confirm sweep, and the methods text states it.

**Stamps.**
- Completions fill `stamp` on `select_models`, `set_split`, `set_recipe`, `set_tuning`, `declare_version`, `open_seal` and `revert`. Revert gains a completion. A shared-step answer recorded after first results gains one too, so the read-only rows can name the step that changed and the methods text can disclose it (§3.2).
- **Key views** (`graph.register_key_view`) drop `stamp`, `note` and `source` from every stage key, so a stamp never triggers a refit.
- The three new kinds land after the decision log's format marker (seam guard 1), so a later change to them migrates instead of breaking old logs.

**Validators.**
- The family must be selected.
- Each option must exist and be offered for the purpose; otherwise the server answers 409 with the exits in §7.3.
- `choose` names at least two options of a searchable slot. A slot may not appear in both `overrides` and `choose`.
- `values` keys must be among the family's declared settings (§4.1). So a class weight, `scale_pos_weight` or XGBoost's linear booster cannot be expressed.
- Values must pass the estimator's own validation. A value outside the declared range adds a concern, not a refusal.
- Under inference, `set_recipe` is refused (§5). The refusal reads the purpose, never a family key (V2X_SEAMS row 7).

**Previews.** `register_consequence` covers all three kinds: §2.5's views, including the edited family's profile and assessment; §6.2's views; and the estimate (§4.4).

### 3.2 Versions

**A version is a family together with its resolved recipe and its tuning choice.**
- **Its key:** `family~` plus 8 hex digits of the sha256 of the canonical JSON of the `VariantSpec` fields that **differ from their defaults** (seam guard 4; V2X_SEAMS row 3).
  - `family` and `defaults_version` are always included, and `space_version` whenever the family is searched.
  - A recipe entry, a `choose` entry, the tuning mode and the values count as differing only when they differ from the family's stated defaults under the spec's `defaults_version`.
  - So a field or slot added later with a default that reproduces today's behavior (a search strategy, a budget, a constraint, a new slot) leaves every recorded key unchanged, and no recorded version comes back as a phantom earlier version.
- **Display** never shows the key. The current version is "Boosted trees"; its earlier versions sit under the label "revised after first results", in the comparison only (§3.7).
- **`defaults_version`** sits inside the identity, so the same words can never name two procedures.
- **Pre-spec records** (`scores_seen.json` listing bare family keys) map to `LEGACY_DEFAULTS`, the resolved version 1 of each existing family (today's fill and standard settings). A version seen under the old defaults therefore stays in the comparison as an earlier version, refit on the current trunk, since a pre-spec record holds no trunk decision key or scores. It is never silently renamed. When this happens the Models card says so once: "Boosted trees now try both ways with blanks and are tuned; the earlier version stays in the comparison."
- **Every fitted version carries one `role`:** `current`, `earlier`, `final` or `secondary`. A v2.x comparator (V2X_SEAMS row 10) is one more role, not a new field.

**The versions shown are in the log, and a revert cannot unsee them.**
- Every stamp copies the versions shown so far for its track's outcome.
- The fold gains one rule: the slot `versions_shown` collects stamps from **every** record, reverted records included.
- A revert of `set_recipe` is therefore allowed. The changed version, once shown, simply becomes an earlier version.
- Replay rebuilds every earlier version from `decisions.jsonl` alone.

**The versions file.** `scores_seen.json` becomes the versions file, one per track, carrying a format string (`turbotab-versions/1`; V2X_SEAMS rule 5). It holds one entry for each version whose scores were served on each trunk:
- its `VariantSpec` and key;
- the log sequence number at which its scores were first served, and whether its tuning result was served;
- the **trunk decision key** it was scored on: the sha256 of the canonical JSON of the data fingerprint and every decision slot the contract's `trunk` stage (MC-3) reads, directly or through the stages it depends on, each through its key view, so stamps never count. It leaves out the split, which has its own path (the holdout status, §3.3). It holds **no stage version**, and it follows seam guard 4's rule: a slot that is unanswered, or added later with a default that reproduces earlier behavior, is left out;
- the **trunk stage key** (MC-3) and the TurboTab version that computed it, for identity and for a rebuild on the same engine;
- the scores last served for it on that trunk: estimate and interval per primary and secondary score.

The file is a fast mirror; the log stays the source of truth. The scores are kept here, and not only in the stage cache, which stays disposable (F16).

**Why two keys.** A stage key hashes the stage's code version and every upstream key (`graph.py:stage_key`), and stage versions are bumped often (MC-3 itself bumps shelf 15→16 and design 24→25). Read-only status therefore reads the trunk decision key only, which changes when an answer changes and never when the engine does.

**What an engine upgrade does.** The stage keys change, so every version on the current trunk is refit by the new engine as current, and the comparison shows its new scores. The versions file keeps the TurboTab version that computed each entry. No upgrade creates a read-only row, and none is ever labeled "revised after first results". A read-only row computed by an earlier engine keeps the scores it served, and the export names the version that computed them.

**What is fitted.** The design stage builds one pipeline for each of:
- the current version of every selected family;
- with holdout status `none` or `after_scores`: every shown version of a selected family that is not current. These are the *earlier versions*, held in `design.objects["earlier"]`;
- after the seal was opened: the version declared at the opening (role `final`), and the current version when it differs (role `secondary`).

**Where earlier versions are read.**
- Only the fit stage and the evaluation stage read them: the comparison, BBC-CV, and `design_bbc` under the population answer.
- Every other stage (explanations, effects, calibration, substitution, sensitivity) reads one version per family: the current one.
- Rows carry `variant` beside `family`.
- `compared_families`, `vouch`, `scored_in` and `explained_in` map a version to its family. This fixes the refusal bug the engine review found, where a version key in `scores_seen` made every `select_models` refuse.

**Cost of keeping an earlier version.** Its out-of-fold predictions are cached in the stage cache under (design stage key, `VariantSpec`, split key). It is refit only when something it depends on changes, and its refit then counts in the estimate (§4.4).

**The read-only rows** (crosswalk ruling 6; seam hook 2; V2X_SEAMS row 11). Under Predict, before the opening, the comparison keeps what was seen in two cases where nothing is refit:
- **A shared step changed after first results.** A change to a shared step creates no version: who is kept, the track's fill, Explore's levers, the selection step, the energy model, the forms, scales, batch and normalization.
- **A version changed under a sealed holdout** (§3.3): the earlier version is not fitted again.

**Which rows.** Every entry in the versions file whose trunk decision key differs from the current one; and, under a holdout sealed before any score, every entry on the current trunk whose version is no longer its family's current version. Nothing extra is written at the change: every served score already carries its trunk decision key.

**What they show.** The scores as served, gray, collapsed with the label "revised after first results" and what changed (§3.7).

**What they never do.** Both kinds are treated alike. They are never refit, never enter BBC-CV or the selection-corrected estimate, can never be declared final (`open_seal.variant` and `declare_version` refuse them), are never scored on the held-out rows, and are never exported as a model. An earlier trunk's inputs differ from the current ones, and its rows too when who is kept changed, so its rows are shown beside the comparison, not compared.

**Reverting brings a row back.** Revert the shared-step answer, or give the earlier answer again, and the trunk decision key matches again: its entries are on the current trunk and are fitted as usual. Under a sealed holdout, revert the recipe or tuning change and the earlier version is current again.

**How they survive.** The scores live in the versions file, so a cleared stage cache still serves the rows. Folding the log up to the recorded sequence reproduces the trunk decision key on any engine version, and rebuilds the trunk on the current engine. That is all v2.x needs to fit them again as kept versions (§10). The trunk stage key matches as well only on the same engine version, so v2.x never relies on it.

**What is said.** The methods text discloses the change under ruling 3 and points to the earlier scores (§3.6).

### 3.3 The three paths

**(a) A choice declared before any score is seen.**
- Its stamp shows no versions.
- It is the family's only version. Nothing was selected, so there is nothing to correct.
- When it is the only family fitted, its own score may be the result, as `declared_result` allows today.

**(b) A choice made after first results.** Under prediction it is always allowed and always disclosed. What happens next depends on the holdout status the server stamps:

| Holdout status | When | A recipe or tuning change made after first results |
|---|---|---|
| `none` | no rows held out | The earlier version stays, fitted, as a gray row labeled "revised after first results". The result is the selection-corrected estimate over every version shown. |
| `sealed` | a holdout drawn **before** the first score for this outcome was served (the split's record precedes the first served sequence) | No earlier version is fitted: the final model is declared on cross-validation before the held-out rows open, so the held-out score is untouched by the change ("the fundamental 'untouched test set' principle", Bischl et al. 2023). The earlier version's scores stay as a read-only row, as for a shared step, so nothing seen is overwritten silently (UI ruling, 2026-10-08). |
| `after_scores` | a holdout drawn after a score was served | Treated as `none`: the earlier version is kept, and compared families stay. The held-out rows were in the scores that drove the change, so their score is labeled "these rows were in cross-validated scores seen before they were held out" and reported beside the selection-corrected estimate. Drawing such a holdout is block and record; its preview says why. |
| `opened` | the seal was opened | Block and record. The change runs as a secondary analysis, and the version declared at the opening stays `final` (`AtOpening` and `mark_final` match on the version). |

**One exception.** Fixing a slot to the option its served "Try both" search favored is not a kept version but a data-derived choice (§3.5). Fixing any other option follows this table.

**A shared-step change made after first results** creates no version at any status. Before the opening it leaves read-only rows (§3.2); after the opening it runs as a secondary analysis, as any change.

**What path (b) corrects, and what it does not.**
- **Corrected:** choosing among the versions shown. BBC-CV covers the final pick among fixed versions (Tsamardinos et al. 2018).
- **Not corrected, and said:** how a later version was designed after the earlier ones' scores were seen (adaptive data analysis).
- **Disclosed under ruling 3, not corrected:** changes to the shared trunk after first results, which create no version. Their earlier scores stay as read-only rows. They are listed in the TRIPOD+AI model-building item, beside every version changed after first results.
- **Prevention:** before the first fit with no holdout, the Models card states once: "Every version whose scores you see stays in the comparison. Hold out rows now if you expect to try several." (20 words) It appears only when a holdout is offered without concern at this size.

**(c) Try both; keep what predicts better.**
- `SetRecipe.choose`, or a family's `default_choose`, puts the named options into the family's search as a categorical dimension, mapped from one Sobol coordinate. The standard settings are tried once with each option (§4.2).
- Each outer training fold picks its own option, so the outer score belongs to a procedure that chooses.
- **Shown:** how often each option won across outer folds, and the option the final refit chose (the deployed one).
- **Rung under prediction:** the trees' default for blanks, recommended. On any other searchable slot it is available, before or after first results, with its minutes shown. It is not labeled Recommended after first results: with no holdout the earlier version is kept either way, so trying both adds cost and no extra correction.

### 3.4 Which version is reported and used

- **A holdout drawn (any status):** the version named at the opening (`open_seal.variant`).
- **No holdout:** BBC-CV's choice among every version in the comparison. `declared_result` already declares `selection["best"]`. Read-only rows are never among them.
- **When BBC-CV's choice is an earlier version,** the Models card says: "The earlier version scores best, so it is the model reported and used." (13 words) Two exits follow:
  - **"Use the earlier version"**: "Makes it current again, so every view describes the model reported." This records a `set_recipe` or `set_tuning` restoring it.
  - **"Report the current version instead"**: "Its own score is reported, labeled as chosen after the scores were seen." This records `declare_version`. The result is that version's own cross-validated score, labeled "chosen after the scores were seen", with the selection-corrected estimate beside it (an amendment to row 12(b), §10).

  Until one exit is taken, every view other than the comparison describes the current version, and its caption says so.

### 3.5 Values and options from tuning results (the quiet leak)

**When a choice counts as data-derived.** The final refit's search chooses its values, and the option of each "Try both" slot, on all training rows, every outer test fold included. The fold shares come from searches whose training rows hold the other folds' test rows. A choice is therefore *data-derived* when it is recorded after a tuning result for this outcome was served (`stamp.tuning_shown` is not empty), and it is one of:
- a manual value;
- a value held under `automatic`;
- a `set_recipe` override that fixes a slot this family's served "Try both" search covered to an option that search **favored**: the option its final refit chose, or any option that won at least half of its outer folds.

**The trees' default is covered by this rule.** Once their fold shares were shown, fixing their blanks to the option the folds kept is data-derived. Fixing the other option is not: nothing the folds showed flatters it, since they preferred the one not chosen. It is an ordinary change after first results (§3.3(b)): the earlier version, which tried both, stays in the comparison, and BBC-CV corrects the choice among the versions shown. A researcher who wants the favored option already has it honestly, by keeping "Try both".

The server decides this from the served `TuningRecord`; the client cannot claim otherwise.

**What happens, by holdout status:**
- **`none` or `after_scores`: block and record.**
  - The version is fitted and shown, labeled "settings chosen on these rows: not an honest estimate", or, for an option, "option chosen on these rows: not an honest estimate".
  - It is excluded from BBC-CV and from the declared result.
  - The coach line, for values: "These values come from tuning on these rows, so this score would be too kind." (15 words) For an option: "The folds already chose this on these rows, so fixing it here would make its score too kind." (18 words)
  - The exits: "Tune inside each fold" for values; "Keep trying both" for an option, which returns the slot to its search.
- **`sealed`:** available. The holdout absorbs it.
- **`opened`:** a secondary analysis, as any change.

This is the reachable form of "tuning outside the resampling". Bischl et al. 2023 warn that "the more we tune, the smaller our data set, … the more expressed this optimistic bias will be".

### 3.6 Sentences (the methods register, `voice.register_sentence`)

The methods text is one of the two places the label "revised after first results" appears; the comparison is the other (UI ruling, 2026-10-08).

| Path | Sentence |
|---|---|
| Stated default: Try both | "Whether boosted trees received blanks in 6 columns as they were, learning at each split which way they go (missing incorporated in attributes; Twala et al. 2008), or used the fill learned in each training fold with a missing indicator per column, was chosen inside each training fold, so the reported score includes the choice (nested cross-validation); the folds kept blanks in 41 of 50 folds, and the final model keeps them. The other models used the fill learned in each training fold." |
| Blanks kept, set by the researcher | "Boosted trees received blanks in 6 columns as they were and learned at each split which way they go (missing incorporated in attributes; Twala et al. 2008); no missing indicator was added for those columns." |
| Before any score | "Boosted trees were given the fill used by the other models instead of trying both, declared before any score was seen: ⟨note⟩." |
| After first results, no holdout | "Ridge was revised after first results: after the cross-validated scores of the linear model and ridge were seen, ridge was changed to rescale measured numbers only, leaving yes/no and category columns at 0 or 1 (Gelman's two-SD scaling). The earlier version was kept in the comparison, and the result is corrected for choosing among the 3 versions shown (BBC-CV); it is not corrected for how the later version was designed after the earlier scores were seen." |
| After first results, against the folds' choice | "Boosted trees were revised after first results: after the folds had kept blanks in 41 of 50 folds, boosted trees were changed to use the shared fill (⟨note⟩). The earlier version, which tried both, was kept in the comparison, and the result is corrected for choosing among the 3 versions shown (BBC-CV); it is not corrected for how the later version was designed after the earlier scores were seen." |
| Holdout sealed | "…; the held-out rows were sealed before any score was seen, so the held-out score does not depend on this change, and the earlier version's cross-validated scores are reported for reference only." |
| Holdout after scores | "…; the held-out rows were set aside after cross-validated scores on them were seen, so the earlier version was kept and their score is labeled accordingly." |
| Seal opened | "…after the held-out rows were opened; it is a secondary analysis, and the declared result is unchanged." |
| Shared step revised after first results | "The energy model was revised after first results, from ⟨the residual method⟩ to ⟨the density method⟩; the cross-validated scores seen before the change are reported for reference (Table S⟨n⟩), were not used to choose the model or to correct the result, and the change itself is disclosed, not corrected." |
| Try both, chosen by the researcher | "Whether ridge used values as measured or reshaped toward a bell curve was chosen inside each training fold, so the reported score includes the choice (nested cross-validation); the folds reshaped in 12 of 50 folds, and the final model uses values as measured." |
| Automatic | "Boosted trees' settings were tuned inside each training fold by a seeded random search over 18 settings (the standard settings with blanks kept and with blanks filled, and 16 quasi-random ones), scored by squared error on 3 inner folds kept by participant; the plan was set once, for training folds of about 13,600 participants, and used in every fit, so the reported scores include the tuning (nested cross-validation)." |
| Small sample | "With 214 participants in each training fold, too few to rank settings, boosted trees used the standard settings of scikit-learn ⟨version⟩; whether they kept blanks or used the shared fill was still chosen inside each training fold, a choice between two (blanks kept in 27 of 50 folds; the final model keeps them)." |
| Standard | "Boosted trees used the standard settings of scikit-learn ⟨version⟩, untuned." |
| Manual | "Boosted trees used a learning rate of 0.05 and 63 leaves per tree, set by hand before any score was seen (source: Smith et al. 2024); their other settings were standard." |
| Data-derived | "Boosted trees' settings were set by hand to values chosen by tuning on these rows; that version's cross-validated score is not an honest estimate and is excluded from the result." |
| Data-derived option | "Boosted trees were set to keep blanks as blanks after the folds had kept them in 41 of 50 folds; that version's cross-validated score is not an honest estimate and is excluded from the result." |
| Declared version | "The current version of boosted trees was declared after the scores were seen; its own cross-validated score is reported, beside the selection-corrected estimate over the 3 versions shown." |

### 3.7 The comparison table

| Column | Header | What it holds |
|---|---|---|
| Model | Model | the family's name; its technical name as the quiet term |
| Inputs | What it was given | described against the trunk: "shared inputs"; "shared inputs; the folds chose: blanks kept in 41 of 50"; "shared inputs, blanks kept as blanks"; "shared inputs; changed by you: blanks filled"; "shared inputs; set by you to the folds' choice: blanks kept" |
| Settings | Settings | "tuned in each fold"; "standard"; "set by hand"; "set by hand from tuning on these rows"; "nothing to tune" |
| Score | the primary score in words ("Expected squared error on new people") | estimate and interval, as today |

**The scaling and encoding a family's own definition requires are part of the model.** The penalized families' standardization and full coding are stated in their recipe line, not counted as an input difference.

**The line under the table** appears only when a difference changes the *information* a model sees: blanks kept versus filled (including a "Try both" family whose folds chose differently from the others' fill), a transform, a cap, or any change by the researcher. It reads: "These models were given different information, so each score is for the inputs and the model together." (17 words) The paired-difference sentences then compare whole procedures, and say so.

**Revised after first results.** Two kinds of gray line carry the label, collapsed, and open on a click:
- **Earlier versions,** under their family's current row: "Revised after first results · 2 earlier versions, kept so the final score accounts for choosing among them." (17 words)
- **Read-only rows,** under the table, newest first, one line per earlier trunk: "Revised after first results · scores before the energy model changed, shown for reference only." (14 words) Under a sealed holdout, an earlier version is a read-only row under its family's current row: "Revised after first results · earlier version's scores, for reference; the held-out rows are still sealed." (15 words)

The label appears nowhere else on screen: not on a stage, not in the progress bar, not on a card (§6.0).

---

## 4 · Tuning

### 4.1 The search spaces

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Dimension:
    name: str             # the estimator's parameter
    label: str            # plain: "how big each correction step is"
    term: str             # quiet: "learning rate"
    low: float
    high: float
    scale: Literal["log", "linear", "int", "log_int", "choice", "share_of_units"]
    choices: tuple = ()
    source: str = ""

@dataclass(frozen=True)
class TuningDecl:
    kind: Literal["none", "path", "search"]
    dimensions: tuple[Dimension, ...] = ()   # searched
    by_hand: tuple[Dimension, ...] = ()      # settable by hand, never searched (Huber's threshold)
    standard: Mapping[str, Any] = field(default_factory=dict)   # the standard settings
    standard_source: str = ""                # "scikit-learn's defaults", "ranger's defaults"
    fixed: Mapping[str, Any] = field(default_factory=dict)      # stated, never searched
    early_stopping: Mapping[str, Any] | None = None
    out_of_bag: bool = False
    space_version: str = ""                  # "boosted_trees/1"
    reason: str = ""                         # ≤ 22 words
```

The model-family contract adds `structural` (settings that are identity, never searched), tunability, transfer and a cost model (its C6).

| Family | Kind | Searched (range, scale) | Fixed | Standard settings | Sources |
|---|---|---|---|---|---|
| Linear | none | | | | |
| Robust linear | none | (by hand: threshold t in [1, 3]) | MAD scale; no penalty | t = 1.345 | Huber 1964; Holland & Welsch 1977 |
| Ridge | path | Penalty per row λ, on scaled columns: 50 points from 10⁻⁵ to 10², log. Classes: logistic with l2, C = 1/(n·λ) on the same grid. | | | Probst, Boulesteix & Bischl 2019: of the six learners they studied, glmnet gained more from tuning than xgboost or ranger (mean AUC tunability 0.069, 0.043 and 0.010 against package defaults). |
| Elastic net | path | Mix (0.1, 0.5, 0.7, 0.9, 0.95, 1) × 100 ratios r from 10⁻³ to 1 of each split's own λ_max. Logistic: mix (0.2, 0.6, 1) × 8 ratios. | The pure-ridge end belongs to ridge. | | glmnet's λ path; Probst et al. 2019 (JMLR) |
| Random forest | search | Share of columns per split [0.05, 1], linear. Share of rows per tree [0.2, 1.0], drawn with replacement (adapted from tuneRanger's [0.2, 0.9] without replacement, since scikit-learn subsamples only with replacement). Smallest leaf as a share of the fit's units, log scale, from 1 row to a tenth of them. | 500 trees in every candidate | §2.2 | Probst, Wright & Boulesteix 2019; tuneRanger; Probst et al. 2019 (JMLR: n^x, x ∈ [0, 1]); Kruppa et al. 2014 (leaves of about 10% of n for probability machines) |
| Boosted trees | search | Learning rate [0.01, 0.3], log. Leaves per tree [4, 128], log integer. Smallest leaf [2, 200], log integer, capped at a twentieth of the plan's units. L2 pull [10⁻³, 10], log. Share of columns per split [0.3, 1]. Number of trees [25, 500], log integer, only when early stopping is off. | Early stopping by the plan (§4.2): up to 1,000 trees, patience 20 | scikit-learn's defaults: learning rate 0.1, 31 leaves, leaf 20, no L2, all columns, 100 trees; its own early stopping above 10,000 training rows, decided once by the plan's rows | Probst et al. 2019 (JMLR); scikit-learn's HistGB documentation |
| XGBoost | search | Learning rate [0.01, 0.3], log. Depth [2, 10], integer. Least child weight as rows-equivalent [1, 64], log, times the mean hessian at the base score (1 for squared error; p̄(1−p̄) for yes/no; the mean over classes of 2p̄ₖ(1−p̄ₖ) for more than two classes, the factor 2 being xgboost's softmax hessian, confirmed on xgboost 3.3: root cover over n equals 2p(1−p), amended 2026-10-10). Row share [0.5, 1]. Column share [0.3, 1]. L2 pull λ [10⁻³, 100], log. Rounds [25, 1,000], log integer, only when early stopping is off. | Early stopping by the plan: up to 2,000 rounds, patience 50. Tree booster; α = 0; `tree_method="hist"`; threads recorded. | XGBoost's defaults: learning rate 0.3, depth 6, child weight 1, every row and column, λ = 1, 100 rounds | XGBoost's tuning notes; Probst et al. 2019 (JMLR: λ 2^[−10, 10]; child weight 2^[0, 7]) |

**Why the least child weight is scaled.** It is a sum of hessians. At 5% prevalence, an unscaled 64 would demand about 1,350 rows per leaf.

**The standard settings are always candidates:** once, or once for each option a "Try both" slot tries. They come first in the candidate list, so ties go to them.

**The ranges are conventions,** narrowed from the sources above. The methods reference states each range with its source, and the prediction reviewer's packet asks for them to be checked.

### 4.2 The plan, fixed once

**The strategy in v2: a seeded quasi-random search.** It is plain random search over a scrambled Sobol sample, with no successive halving and no TPE. The reasons, from strongest to weakest:
1. **Replay.** The candidate list is fixed before any score is seen, and the result does not depend on the order in which fits finish. Optuna's FAQ, by contrast: "We recommend executing optimization of a study sequentially if you would like to reproduce the result."
2. **The budget is known before the run,** so the estimate is exact in fits (DoD gate 5).
3. **The spaces have low effective dimension.** There, random search is "a surprisingly strong baseline" (Bischl et al. 2023; Bergstra & Bengio 2012).
4. **Halving was dropped.** At most nutrition sizes it would engage for one round or none. Where it does engage, count-type settings mean different things at a ninth of the rows: the better configuration's "superiority … was only observable after full evaluation" (Bischl et al. 2023).

Halving, Hyperband, TPE and BOHB are v2.x (package C6c, displaced on 2026-10-08; §10). They plug into one field, below.

**The plan.** It is a `TuningPlan`, computed once in the design stage, recorded and stated:

| Element | Rule |
|---|---|
| Strategy | **`TuningPlan.strategy`** (seam hook 1; V2X_SEAMS rows 1–2). v2 has one value, `"sobol"`. The strategy owns the candidate generator and the fit count, so the estimate (§4.4) asks the strategy how many fits it will make instead of computing a formula of its own. `TuningRecord` lists the candidates in the order they were evaluated. A v2.x strategy is one more value with its own generator and count. |
| Effective size n_eff | Numeric outcome: units. Yes/no or classes: the units in the rarest class, each unit counted by its most common class, as the stratified splits count it. Riley et al. 2021 state the concern in terms of a "small effective sample size". |
| Plan size n_plan | n_eff of one outer training fold of the headline split: (K−1)/K of the training units, or the median training fold when the folds follow time. **Every fit uses this plan unchanged:** every outer fold of every repeat, bootstrap resamples, the folds of Bates et al.'s nested cross-validation, internal–external refits, evaluation's design-based folds, and the final refit. Only settings defined as shares of a fit's units rescale. This fixes F13 for tuned families. The same rule fixes K, its floor and the no-inner-folds switch for every family, and the early-stopping switch for the standard candidates. |
| Standard candidates s | The standard settings once for each combination of options that the family's "Try both" slots try, counting only slots with a non-empty footprint (§2.3). Tree families under their default: s = 2. Otherwise s = 1. |
| Candidates C | s + S, where S is the Sobol sample. n_plan below 300: S = 0, the standard candidates only (§4.6). 300–999: S = 8. 1,000 and up: S = 16. The forest uses S = 8 from 300: it is the least tunable. "Lighter" halves S. "Standard settings" sets S = 0 at every size. A "Try both" slot also maps one Sobol coordinate to its options, so the Sobol candidates try both too. |
| Inner folds K | Searched families: 3 (a convention; every candidate meets the same splits). Path families: `inner_folds(n_plan)`, which is 5 for 100 to 5,000 and otherwise 3. **One floor, set once in the plan:** every inner fold holds at least 2 units of the rarest class and, under the population answer, at least 2 PSUs. So K = min(K, ⌊m/2⌋), where m is n_plan for a yes/no or class outcome, and under the population answer the fewest PSUs in any outer training fold the stages draw (the splits are fixed before any fit). Below K = 2, that is with m under 4, the plan draws no inner folds (§4.6). A fit that holds fewer than 2 per inner fold at the plan's K (a bootstrap resample, a smaller outer fold) keeps the plan's K and draws its folds as evenly as its units allow; the `TuningRecord` counts such fits, and the methods sentence states the count once. |
| Early stopping | On in every candidate fit when n_plan ≥ 1,500, so that the stopping set inside an inner fit holds about 100 or more units of the rarest class. Below that, the number of trees is searched. |
| Stopping set | A tenth of the fit's units: whole units, stratified by class, the latest ones when the folds follow time. It is drawn before anything else, and no step is fitted on it (§4.3). |
| Score | The comparison's strictly proper primary as a pooled per-row loss over the inner validation rows (`validation.loss_rows`: squared error, log loss or ranked probability score). It is survey-weighted under the population answer, and rounded to 1e-9 relative before the argmin. |
| Choice | The lowest pooled loss. Ties go to the lower index, so the standard candidates first, and among them the first option of each "Try both" slot. |
| Refit | The chosen candidate on all of the fit's rows, through `fit_pipeline`. |
| Forest, out of bag | Used only when every row is its own unit, the folds do not follow time, the sample answer applies, no step before the model reads the outcome, and there is no imbalance correction. Each candidate is then fit once with 500 trees and scored on its out-of-bag predictions (tuneRanger's method). Otherwise inner folds are used. |

**Cost of one outer fit,** in fits on that fit's rows, as the `"sobol"` strategy counts it:
- **Searched:** F = w·[(K − 1)·C + 1].
- **Nothing to choose** (C = 1: standard settings and no "Try both" option with a footprint): F = w, the refit alone.
- **Out of bag:** F = C + 1.
- **Path:** F = w·[(K − 1)·r + 1] path fits, where r is the number of recipe options under "Try both" (1 otherwise).
- **w** is 5 with the imbalance correction (its recalibration refits: 1 + 5 folds × 4/5), and 1 otherwise.

**Worked example.**
- 17,000 training participants, a numeric outcome with blanks in 6 routed columns, comparison folds 10 × 5, boosted trees at their defaults.
- n_plan = 13,600, so s = 2, S = 16, C = 18, K = 3, and early stopping is on.
- F = 2 × 18 + 1 = 37 fits per outer fit.
- With 51 outer fits (50 comparison folds and the final refit), that is 1,887 fits.

### 4.3 Where it runs: nested in every fit

**`fit_pipeline(pipeline, X, y, *, groups, order, design=None, seed)`.**
- `design` is (strata, PSU, weights). It fixes F15.
- A `TunedPipeline` is dispatched first, before the early-stopping branch.
- The seed is the split's seed, carried in the plan.

**`TunedPipeline(Pipeline)`.**
- It has one extra parameter, `search: TuningPlan | None = None`. The default lets scikit-learn's slicing, which rebuilds with `self.__class__(steps, …)`, work.
- Its `fit` always searches. A direct `.fit(X, y)` draws its splits from row keys.
- `at(candidate)` returns a plain `Pipeline`, for timing and pinned refits.
- No path can skip the search, and none can search on rows the fit was not given.

**One per-split helper, `fit_parts`, shared with the plain path of `fit_pipeline`.** For the rows it is given (the rows stay an argument, so a v2.x strategy can hand it a subsample; V2X_SEAMS row 1), it:
1. draws the stopping units first;
2. fits the steps before the model on the remaining rows, and transforms `X_val` with them (fixes F11);
3. draws every nested split those rows need: each step's `cv`, and the imbalance correction's recalibration `cv`;
4. fits the model.

Inside a search, the helper fits one head per inner split per recipe option, and every candidate in that split shares it. Under the trees' default that is two heads per split: the blanks kept, and the shared fill with indicators. Nothing in the inner loop learns from its own validation or stopping rows.

**Inner splits are drawn as the outer ones are:**
- whole units, when rows repeat;
- forward chaining by whole unit, when the folds follow time;
- **under the population answer, whole PSUs:** `GroupKFold` over stratum × PSU labels, not stratified within strata. An NHANES training fold of a design-based fold holds one PSU per stratum, about 15 PSUs, so K stays 3. The plan's K already counts the PSUs (§4.2): each inner fold holds at least 2, and with fewer than 4 PSUs in an outer training fold the plan draws no inner folds and keeps the first standard candidate, stated (§4.6). The sentence says: "inner splits keep whole PSUs; with two PSUs per stratum, strata cannot be kept in every inner fold". This holds in the fit stage, in evaluation and in the final refit alike, so one procedure is scored and deployed. Models are fit unweighted, as the comparison fits them; only the inner loss is weighted;
- otherwise, shuffled by the seed over content keys, and stratified by class where the splitter allows.

**Early stopping always draws whole units,** even under the population answer: it scores no design estimate.

**Path families use one path search, not scikit-learn's CV estimators.** This fixes F4 and F5. For each inner split it:
- refits the head (the screen included);
- runs the path on that split: ridge in closed form by SVD over the λ grid; `enet_path` at fixed ratios of that split's λ_max; logistic, warm-started along its grid;
- scores the pooled (and, under the population answer, weighted) inner loss;
- chooses on the pooled curve;
- refits plain `Ridge`, `ElasticNet` or `LogisticRegression` at the chosen values, with λ = r·λ_max for the elastic net on the fit's own rows.

The final estimator is plain, so `alpha_` and `l1_ratio_` readers move to the tuning record (RT-5f).

**The imbalance correction gets the early-stopping interface** (fixes F12):
- `ImbalanceCorrected` gains `early_stopping`, `validation_fraction` and `fit(X, y, X_val, y_val)`;
- the stopping units are drawn before resampling; only the remaining rows are resampled; `X_val` passes untouched;
- its recalibration splits are drawn for the rows each fit receives.

So a candidate is the wrapped, recalibrated model, and the search scores exactly what will be deployed.

**Cancel** is checked before every candidate fit (about 2 seconds to stop).

**So the search is nested in every fit:**
- every outer fold of the headline and of the comparison substrate;
- every bootstrap resample (only path families are bootstrapped);
- every fold of the nested cross-validation interval;
- every internal–external refit;
- evaluation's design-based folds;
- the final refit, whose chosen values are the deployed model's.

**Pinned refits.**
- Substitution bands and explanation reseeds describe the deployed model, so a searched family's refits hold the final fit's values through `at(chosen)`. A "Try both" slot is held at the final fit's option too.
- Their caption says "conditional on its tuned settings". A re-tuned band is v2.x.
- Path families re-tune, as today.
- Validation never pins.

### 4.4 The estimate, and Fit

**The estimate.**
- `cost.py` times `at(center)` once: every dimension at the midpoint of its scale, with the first option of each "Try both" slot.
- `ShelfFamily` gains `per_fit_seconds` and `outer_fits`.
- `outer_fits` counts every fit the stages will make:
  - the comparison folds and the final refit;
  - bootstrap refits, only for families that are bootstrapped (fixes F7);
  - evaluation's design-based folds, under the population answer;
  - internal–external refits;
  - sensitivity analyses that refit;
  - the nested cross-validation interval, only when offered or asked (its offer label shows its own minutes);
  - pinned refits, which count as plain fits.
- The estimate is `per_fit_seconds × Σ F(plan)`, where each F comes from the plan's strategy. It is said as "about 30 minutes, most of it tuning". F is recomputed wherever the plan is shown, without re-timing.

**DoD gate 5: a fit expected to take over 30 seconds shows its estimate first.** The estimate appears:
- on the shelf;
- in the preview of every decision whose recording would start such fits: `select_models`, `set_recipe`, `set_tuning`, and upstream answers (the consequence planner knows which stages refit);
- on the Fit action of the analysis flowchart.

**Where Fit lives, and what pressing it does** (the ruling of 2026-10-06; CROSSWALK, "Settled here"; UI ruling of 2026-10-08).
- **Fit is on the analysis flowchart,** at the end of each track's Models, after the open-noticings gate (Estimate and Describe) and the Models Confirm sweep. It reads "Fit · about 30 min".
- **A fit expected to take over about 2 minutes (a convention) does not start on its own.** The hold is in the scheduler, not a stage requirement, so stages stay pure functions of the log and replay bypasses it. The fit stage waits as `stale` until Fit is pressed. Shorter fits compute live and may already be done when Fit is pressed.
- **A pressed fit runs as a server job.** The page may be closed; a notification says when it is done, and Cancel stops it within about 2 seconds.
- **The hold re-arms** when a new estimate exceeds 1.5 times the last one confirmed.
- **Under Predict, pressing Fit is a job command,** not a decision: nothing enters the Record, and Results opens. Choosing "Lighter" in its confirmation is the one exception: it records `set_tuning(mode="lighter")` for each listed family before the job starts (§3.1).
- **Under Predict, the server does not withhold scores until Fit.** The screen fetches them only after Fit, but a short fit computes live, so a client could fetch earlier. Every score, tuned value or fold share the server serves is stamped as shown whenever it is fetched (§3.2), so an early fetch counts exactly as one after Fit, and nothing seen escapes the record.
- **Under Estimate and Describe, pressing Fit records the track's lock:** the existing system record `lock_plan`, so no new decision kind. The server serves no estimate stage before the lock, as `estimand.served_gate` already withholds one whose question is open. The lock is shown with its time and SHA-256 (SIZING P0.8).
- **That guarantee needs seam guard 7.** `served_gate` withholds the stages in `ESTIMATE_STAGES`, a list kept by hand, which already leaves out `usual_intake` (INBOX: "The usual-intake stage is not one of the estimate stages"). P0.4 declares estimate stages on `Stage` and derives the list (seam guard 7). Describe's stages (the usual-intake distribution, survey-weighted means and prevalence by group, trends) are declared as estimates there, as the crosswalk's "Describe has a gate and a lock" requires, which settles that INBOX item for Describe tracks. T10 checks that none is served before `lock_plan`.
- This amends BLUEPRINT §4 (§10).

**Choices made in Models** (recipe phrases, tuning, and the per-family exceptions) are held in the stage's local state, as family choices are today (`ChoiceQuestions.tsx:533`). The stage's Confirm records them together, one `set_recipe` or `set_tuning` per family changed, and the flowchart's Fit then carries the estimate. The families question is a Decide, recorded when answered. **No recipe or tuning slot is asked.**

**Illustration** (rough): the worked example, at about 1 second per fit.

| Plan | Estimate |
|---|---|
| Automatic (18 settings) | about 30 minutes |
| Lighter (10 settings) | about 18 minutes |
| Standard settings, both blank options tried (2) | about 4 minutes |
| Standard settings, one blank option set (1) | about 1 minute |

Under the population answer, evaluation's design-based folds (10 × 2) add about 20 × 37 fits (about 12 minutes). The nested cross-validation interval, about 800 outer fits, is offered with its own time: here, many hours.

### 4.5 How the correction covers tuning and the choice of version

1. **Within a version, tuning is nested** (§4.3). Its out-of-fold predictions come from the whole tuned procedure, the "Try both" choice included. Nothing about the tuning is left to correct.
2. **Across families and versions,** BBC-CV corrects choosing among those predictions (§3.3).

**What is not corrected, and is said:**
- how later versions were designed after earlier scores (§3.3);
- shared-step changes after first results (ruling 3), whose earlier scores are shown read-only;
- the pairwise intervals, which are descriptive (`COMPARISONS_NOTE`);
- at p ≫ n, the label "likely too narrow" (`SELECTION_NOT_NESTED`).

**The alternative not taken:** Tsamardinos et al.'s flat BBC-CV over every (family, setting) pair. At our budgets it costs about the same. It hides how settings move from fold to fold, and it would make the deployed settings a pick from pooled predictions. It is v2.x (§10).

### 4.6 Small samples

**Below an effective size of 300, searched families use their standard settings by default.**
- The line says why, in the grain's unit noun, or in events:
  - "Tuning: standard settings (214 people are too few to rank settings)."
  - "Tuning: standard settings (31 events are too few to rank settings)."
- Tuning stays available, nested as always.
- **The reason is the variance of the chosen settings, not optimism.** Nesting already removes optimism. Riley et al. 2021 found tuning parameters "estimated with large uncertainty … when development data sets have a small effective sample size", which "can lead to considerable miscalibration". Van Calster et al. 2020 found shrinkage's calibration slope often more variable between samples than without it. Martin et al. 2021 recommend quantifying exactly this variability.
- 300 is a convention. T1 reports nested tuning against standard settings on fresh data at an effective size of 200. If tuning wins there by more than 2 standard errors, the threshold is revisited.

**"Try both" below 300 (settled in draft 3).**

*The decision.* "Try both" runs at every size. Below an effective size of 300 the plan drops the Sobol candidates and keeps the standard ones (§4.2): the standard settings once with each option. Each training fold therefore still chooses between keeping blanks as blanks and the shared fill, at the standard settings, on the plan's inner folds, and the score includes the choice. Nothing else is chosen. The tuning line reads "standard settings"; the recipe line still reads "try both".

*Why.*
1. **The small-sample rule guards against a different risk.** It exists because a setting chosen from many candidates on few rows varies from sample to sample, and a shrinkage or complexity setting estimated with large uncertainty can miscalibrate the model (Riley et al. 2021; Van Calster et al. 2020). The blanks choice is between two candidates, both at the standard settings. Neither can land on an extreme setting (tiny leaves, a large step, a weak penalty), and neither changes how far the model shrinks. The fill option does add one indicator column per routed column, and many indicators with few rows can overfit (Van Ness et al. 2023), which is exactly this regime. The nested choice is what guards against that: the folds keep the fill only where its indicators predict better there, and the reported score includes the choice.
2. **The risk of overfitting a choice grows with the number of candidates compared and shrinks with the sample** (Cawley & Talbot 2010). Two is the fewest there can be. When the candidates are equally good and their errors independent and normal, the best of two looks better than it is by about 0.56 standard errors on average, and the best of 18 by about 1.82 (the expected maximum of normal errors). Nesting keeps that flattery out of the reported score either way; what remains is the cost of a wrong pick.
3. **A wrong pick between two rarely costs much.** Both options are scored on the same inner folds, so the folds compare them on their paired difference. If that difference is estimated with standard error σ_d and the true gap is Δ, the folds pick the worse option with probability Φ(−Δ/σ_d), so the expected cost is Δ·Φ(−Δ/σ_d). That is at most about 0.17·σ_d, reached at Δ ≈ 0.75·σ_d (a derivation under the normal approximation; T8 checks it numerically). The folds err often only when the two options predict about equally well, which is when an error costs least. "Little" is relative to σ_d, which is itself large on small tables.
   - **Caveat.** The bound assumes the inner estimate of the gap is unbiased for the gap at the outer fold's size. The inner fits see only (K−1)/K of the fold's rows, two thirds at K = 3, so on a learning curve the option that gains more from extra rows is under-rated there. T8(c) measures the net effect on fresh data.
   - The bound and the expected maxima enter the claims ledger (DoD gate 2) as this spec's derivations, labeled so, with T8(d) as their check.
4. **One rule, and the ruling kept at every size.** The small-sample rule becomes "drop the random candidates", which reads the same for every family and every slot, and Nolan's default holds on small tables too.
5. **The instability is shown, not hidden.** The fold shares say how firm the choice was, as Martin et al. 2021 recommend. An even split means the two options predict about equally well here.

*The alternative not taken:* below 300, one stated option with "Try both" available. It would read consistently with "too few to rank settings", but it would settle on no evidence a question the folds can answer at little cost.

*Where the folds cannot choose.* The plan caps K once so that every inner fold holds at least 2 units of the rarest class, or 2 PSUs (§4.2). When even K = 2 cannot meet that, with fewer than 4 in the plan's fit, the plan draws no inner folds, in every fit: the slot takes its first option, keeping blanks as blanks (the trees' `default`, with the sources of §2.2), and the line says so: "Too few events to choose inside each fold: blanks kept as blanks." Searched families keep their standard settings. A path family cannot choose its penalty without inner folds, so at that size it is not offered, and its shelf card gives Riley's concern as the reason. The shelf already carries that concern at such sizes.

*Checked.* T8(c) (slow): at an effective size of 200, "Try both" at the standard settings against each option alone, on fresh data, with 6 and with 30 routed columns. It reports fresh-data loss and the calibration slope's mean and spread, since miscalibration is the harm the small-sample rule names. If "Try both" loses to the better single option by more than 2 standard errors on either measure, this rule is revisited, as the 300 threshold is.

**Halving subsamples are gone.** Early-stopping sets are stratified by class, and need an effective size of at least 1,500 (§4.2).

**Penalized families below Riley's minimum** carry the concern: "A penalty does not make up for too few rows: its strength is estimated with large uncertainty here (Riley et al. 2021)."

**Shown at every size:**
- the chosen penalty or setting in each outer fold;
- for penalized families, the out-of-fold calibration slope per fold;
- for a "Try both" slot, the fold shares.

**A penalty at the edge of its grid** in more than half the folds adds a concern: "the penalty sits at the edge of the range tried".

### 4.7 Reproducibility

**Seeds.**
- Every seed is derived from the sha256 of canonical JSON, never Python's salted `hash()`:
  - candidates from (split seed, family, space version);
  - inner splits from (split seed, fit's content keys);
  - stopping sets likewise.
- Sobol uses scipy's seeded generator (`rng=` from scipy 1.15; `seed=` before).

**Threads.**
- Every fit runs at a recorded `n_jobs` or `nthread`, and provenance records it.
- Forest prediction runs single-threaded, so the trees' outputs are summed in tree order.
- Inner losses are rounded before the argmin, so a near-tie cannot flip on floating point.
- The elastic net's inner curve is exact at every point, for a number, a yes/no and classes alike (`models/exact_path.py`: feature-sign search, and proximal Newton for the log loss, warm-started from the strongest penalty). An iterative solver's tolerance would otherwise sit above the rounding: coordinate descent at 10⁻⁴ was 10⁻⁵ to 10⁻³ off, and `saga` at the logistic branch's 10⁻³ was 2 × 10⁻³ off. A matrix too wide for it (more than 500 columns, or 600 logistic coefficients) keeps scikit-learn's solvers, and its penalty is not promised to be the same on every platform.
- More than two classes under a pure lasso (mix 1.0) have no unique coefficients: one number added to a feature's coefficient in every class changes no probability, and no penalty while it stays between the feature's two middle coefficients. With an even number of classes the solver stopped where the last bit put it (NHANES-like glucose quartiles: the same penalty and probabilities, coefficients 0.265 apart). Each feature is reported at the middle of that interval, its coefficients' median zero, whichever solver ran (`exact_path.middle_of_ties`).

**What is recorded.** A `TuningRecord` (`tuning_` on the fitted pipeline) holds:
- the plan: strategy, kind, n_plan and its unit, s, C, K, early stopping, seed, space version, defaults version, threads;
- the library versions;
- the candidates, in the order they were evaluated;
- each candidate's pooled inner score, in every outer fold, not only the chosen one's (V2X_SEAMS row 4, so the v2.x tuning curve needs no refit);
- the chosen values, and the chosen option of each "Try both" slot;
- the number of fits and the seconds taken.

Each outer fold records its chosen candidate. The numbers join the provenance estimates. Every fit's input profile (the contract's C4) and the versions file (§3.2) are what the research record keeps (DoD, 2026-10-08).

**Two kinds of replay,** both at the recorded thread counts:
- **pinned replay** (a new replay mode) refits at the recorded values and must reproduce the deployed predictions without searching. It is the replay of record for any v2.x strategy whose candidates depend on earlier scores (V2X_SEAMS row 2);
- **re-run replay** searches again and must reproduce the recorded choices.

Replay's 1e-12 tolerance stands for every family. If T13 shows XGBoost cannot meet it even at a pinned thread count, the record carries a per-family tolerance (the contract's C12), and DoD gate 6's text is amended with Nolan's approval.

Each version's model matrix hash is exported.

### 4.8 What is shown after the fit

Results gains a compact **Settings** table per tuned family:
- the deployed values, in plain words with their term ("how big each correction step is · learning rate · 0.05");
- each value's 10th–90th percentile across outer folds ("0.03 to 0.08 in 50 folds").

**Strips.** Only the one or two settings that varied most get a strip: one dot per outer fold, with the deployed value marked. The rest sit behind "More angles".

**A "Try both" slot shows its share of folds,** with the deployed option: "Blanks kept in 41 of 50 folds; the final model keeps them." Its Why? adds one sentence: "An even split means the two predict about equally well here."

The tuning curve and "Compare with standard settings" are v2.x (§10).

### 4.9 Manual values

- **Where:** every searched family, and the robust linear threshold, through `SetTuning(mode="manual", values=…)`.
- **Entry:** one number field per setting, with its range and its term. Fields are pre-filled with the **standard** values, never the tuned ones. Only the values changed are recorded; the rest stay standard.
- **Partial values:** values given under `automatic` hold those settings fixed and search the rest.
- **Labeled** "set by hand" everywhere, with any source given.
- **Disclosed** by when they were set (§3.3). Values recorded after a tuning result was shown fall under §3.5.
- **Limits:** outside the declared range adds a concern; impossible values are refused by the estimator's own validation.

---

## 5 · Purpose: what tuning and recipes mean under inference

**The reported models have no hyperparameters.** Their choices are declared in the plan before any estimate is seen: the energy model, the forms, the adjustment set, the track's fill (multiple imputation compatible with the model), and the secondaries.

**What follows:**
- **No tuning for the declared model.** The tuning line is silent unless a penalized family is chosen for its labeled, shrunk table. That family's penalty is searched by the path search; its table carries no intervals.
- **No estimate before the lock.** Under Estimate and Describe, the server serves no estimate stage until pressing Fit on the analysis flowchart records the track's lock (§4.4). The shrunk table is an estimate and waits with the rest.
- **Recipes are the plan's.** Every recipe line is stated text marked "set by your plan", not an openable phrase. `set_recipe` under inference is refused, with exits:
  - for a transform: "Ask the curve question";
  - for a cap: "Plausibility repairs";
  - for "Try both": "Declare one option".
- **Blanks.** Who is kept is Who's in's answer; how kept blanks are filled is the Estimate track's own Models answer (§2.3). Tree families take that fill; keeping blanks as blanks is not offered, and neither is "Try both".
- **Post-double-selection lasso keeps its plug-in penalty** (Belloni, Chernozhukov & Hansen 2014).
- **DML and TMLE keep the causal lane's stated learners in v2.**
  - One fix lands now: the lasso's inner folds are drawn by unit through `inner_splits` (F9; RT-11).
  - The learner keys that collide with families are renamed: `random_forest` becomes `nuisance_forest`, and the `boosted_trees` learner is renamed with it, because RT-5a tunes the family of that name. Old spellings parse as the new ones, and retired spellings are never reused (seam guard 2; the contract's MC-2b).
  - Learners built from the family registry, and nuisance tuning inside cross-fitting, are v2.x (package C6c, displaced on 2026-10-08; §10). Bach et al. 2024 (full text, per the methods review) found tuning on the full sample and on folds performed similarly. Before they fold in, the bug logged in INBOX must be fixed: a factory learner on a yes/no outcome is fitted as a regressor.
- **Robust linear regression is not offered under inference** in v2. RLM takes no weights and offers only i.i.d. covariances (H1–H3), so it cannot meet ruling 6 or the cluster-robust and HC3 intervals in the DoD. A weighted M-estimator with a design-based or cluster sandwich is v2.x (C6c; §10).

**Refused under inference** (the requests a client can send):

| Request | Why | Exit |
|---|---|---|
| A per-model transform or cap on any model | It changes what the plan set out to estimate, without the plan. | The curve question; plausibility repairs |
| "Try both" on any model | Inference may not choose its model from the data it reports on. | Declare one option |

**Impossible by construction, so no validator and no menu item:**
- choosing a setting by an effect estimate (no tuning under inference reads one);
- conventional intervals on coefficients penalized by cross-validation (the table has none).

**Under inference no cross-validated score is shown** (ruling 13), so there is no after-first-results path for versions. A change after the lock runs as a labeled secondary analysis beside the locked primary (CROSSWALK, "Never overwrite silently"; SIZING C7b).

---

## 6 · The calm UI

Desktop only. Two registers: plain words on the card, and the technical name as a quiet term.

### 6.0 Where this sits in the quest log

**The stage.** Everything here lives in **Models**, the fifth of the seven stages, once per track. Under Predict, the screen shows Models in this order (CROSSWALK §5, with the DoD's reorder of 2026-10-08):
1. **the Decides that shape every family's input,** each where it applies: the energy model; scales, batch correction and the omics normalization; and the selection question, when it is asked;
2. **the families question** (Decide), with the shelf ranked live (§2.6);
3. **the Confirm sweep:** the validation scheme (`default:validation-scheme`), the in-fold levers (`decision:set_levers`), the track's fill (§2.3), the recipe lines (`decision:set_recipe`), the trees' "Try both" (`default:trees-try-both-blanks`) and the tuning line (`decision:set_tuning`);
4. **the analysis flowchart,** with Fit (§4.4).

The engine's order differs: the Router answers the validation scheme and the levers before the families. The display-order rule below allows the difference, and its first condition holds through "Ranked on the defaults now set": the shelf ranks on the scheme's and the levers' defaults until the sweep confirms or changes them, and re-ranks when one changes.

**The three labels.** The engine's tier names (asked, stated, silent) never reach the UI.

| On screen | What of this spec sits there |
|---|---|
| **Decide** | the families question. No recipe or tuning slot is asked. |
| **Confirm** | every stated recipe line whose alternative would change a number here; the trees' "Try both"; the tuning line; the Predict track's fill (§2.3) |
| **For the record** | silent recipe lines; the plan's numbers (n_plan, s, C, K, the seed); the routed columns when no blank reaches a family |

**Plain words.** Card text says "blanks", "fill", "keep what predicts better", "settings". The technical name rides along as a quiet term ("missing incorporated in attributes", "nested cross-validation", "BBC-CV"). "Exposure", "confounder" and "estimand" appear on no card in this spec.

**"Revised after first results"** appears only in the comparison (§3.7) and the methods text (§3.6) (UI ruling, 2026-10-08). It is never a stage badge, never on the progress bar, and never on a card. When a change after first results reopens Models, the stage drops back and says why in plain words, without the label: "Reopened: you changed how boosted trees handle blanks."

**The display-order rule** (orchestrator, 2026-10-08). The screen may show this spec's elements in an order other than the engine's because all three conditions hold:
1. **Nothing shown depends on an unanswered decision without saying so.** The shelf ranks before the Confirm sweep and says "Ranked on the defaults now set"; before the Decide answers it needs, it says "Waiting for: [question]". The minutes on the shelf and on Fit are computed on the defaults in force and recomputed when one changes.
2. **Every answer the engine fills in is recorded and visible.** Every recipe default, the trees' "Try both" and the tuning plan sit in Confirm, or in For the record when they change nothing here. The small-sample switch is stated in the tuning line. All of them reach the methods text and the export, through each version's `VariantSpec` and its `TuningRecord`.
3. **No view touches rows or the outcome before its gate opens.** The model-matrix preview shows predictor values of training rows after the seal, never the outcome. The shelf reads only the outcome's counts, and its profile stops before any step that reads the outcome's values (§2.6). Tuning results, fold shares and scores appear on screen only after Fit; under Predict the server stamps any it serves as shown, whenever they are fetched (§4.4). Under Estimate and Describe the server serves no estimate before the lock, once every estimate stage is declared (seam guard 7).

### 6.1 The recipe lines

**Lines are grouped by shared departure,** under the chosen families:
- "Boosted trees, random forest and XGBoost **try both** with blanks in 6 columns: kept as blanks, or filled."
- "Ridge and elastic net **put every column on one scale**."

**Rules.**
- Changing a grouped phrase applies to every family in it; the stage's Confirm then records one `SetRecipe` per family. A family set differently gets its own line ("Random forest **fills blanks first** · changed by you").
- A line whose footprint is empty is silent, under For the record.
- A line with zero routed columns says why: "No blanks reach the trees: the selection step needs every value."
- The shared fill's line ("The other models use the fill learned in each training fold.") is silent when the table has no blanks.
- **One quiet link, "What each model is given",** sits under the lines. It lists each chosen family's non-silent slots with their stated phrases, each clickable. This makes every override one step away, the linear model's included.

**Pointing at a phrase elaborates it in one line** (the HANDOFF ruling: hovering "elaborates slightly"). **Clicking it opens the phrase, which takes over the card's focal region.**
- The family list collapses to the chosen names, and Escape returns to it.
- Options open with their consequence and at most one quiet label. Arrow keys move through them.
- The canvas layout follows the option's footprint:
  - missing and encoding: **Routing**;
  - scale, transform and cap: **Strip**.
- The flip plays the storyboard. For "Try both", the Routing canvas shows both inputs side by side.

**Copy for every option.** The word-budget gate runs on it before the build.

| Slot · option | Label | Consequence (words) | Quiet term |
|---|---|---|---|
| missing · choose (the trees' default) | Try both; keep what predicts better | "Each training fold keeps blanks or fills them, whichever predicts better there." (12) | chosen by nested cross-validation |
| missing · native | Keep blanks as blanks | "Each split learns which way a blank goes, so a blank can carry meaning." (14) | missing incorporated in attributes |
| missing · fill | Fill them first, like the other models | "Blanks get each fold's fill, marked as filled; every model sees the same inputs." (14). Without indicators: "Blanks get each fold's fill; every model sees the same inputs." (11) | in-fold imputation |
| any other searchable · choose | Try both; keep what predicts better | "Each training fold picks one; the final score includes that choice." (11) | chosen by nested cross-validation |
| scale · standard | Put every column on one scale | "Each column is centered and divided by its spread, so the penalty treats all alike." (15) | standardization (glmnet's default) |
| scale · two_sd | Rescale measured numbers only | "Yes/no and category columns stay 0 or 1; only measured numbers are rescaled." (13) | Gelman's two-SD scaling |
| scale · none | Leave columns in their units | "The penalty then depends on each column's units, such as grams or milligrams." (13) | unstandardized penalty |
| encoding · every_level | Keep every level as its own column | "Each level gets its own column, so the penalty treats every level alike." (13) | full dummy coding |
| encoding · reference | Compare each level with the first | "One level is the baseline; the others are measured against it." (11) | reference coding |
| transform · none | Use values as measured | "Numbers enter as recorded; curved effects are handled by the curve question." (12) | no transform |
| transform · yeo_johnson | Reshape skewed columns | "Each column is reshaped toward a bell curve, learned in each training fold." (13) | Yeo–Johnson transform |
| outliers · none | Keep extreme values | "Extreme values stay as recorded; a robust model limits their pull instead." (12) | no cap |
| outliers · winsorize | Cap extreme values | "Values beyond each training fold's 1st and 99th percentiles are set to those limits." (14) | winsorizing at 1% and 99% |

**Why?** for the trees' missing slot. It names only what applies to this table, so the last sentence appears only when an energy model exists. The full text is 58 words, 67 with the last sentence:

> "Trees can send a blank down its own branch. When a blank carries meaning, such as a test not ordered because the patient looked well, that often predicts better than filling it (Perez-Lebel et al. 2022). So each training fold tries both and keeps the better. Either way, blanks must mean the same where the model is used. Columns your energy model uses are always filled first."

### 6.2 The tuning line

**One line states the shared plan:** "**Tuning: automatic** for boosted trees, random forest and XGBoost · about 30 min".
- Changing it applies to every listed family in one action; the stage's Confirm then records one `SetTuning` per family.
- A family set differently gets its own line ("Random forest: standard settings · changed by you").
- "Set by hand" first asks which model.
- Every count and minute is computed from the plan for this table.

| Option | Consequence (words) | Quiet term |
|---|---|---|
| Automatic (Recommended) | "Tries up to 18 settings inside each training fold and keeps the best." (13). When the listed families try the same number, "up to" is dropped. | nested random search |
| Standard settings | "The standard settings, the same in every fold." (8) | the family's source: "scikit-learn's defaults", "ranger's defaults", "XGBoost's defaults" |
| Set by hand | "You choose the values; the record says when and why." (10) | |

**Each option's minutes sit at its right edge.** "Lighter" is not listed. It appears just in time, in the Fit action's confirmation when the fit is held: "Lighter tuning · tries 10 settings · about 18 min". Choosing it records `set_tuning(mode="lighter")` for the listed families before the job starts, and the tuning line in Confirm then reads "Tuning: lighter · changed by you" (§3.1).

**The canvas uses the Angles layout:**
- **"What will it try?"** shows this table's storyboard: "18 settings" → "each tried on 3 inner folds of about 9,000 people" → "the best refit on the training fold" → "repeated in each of 51 fits".
- **"What will it cost?"** shows minutes by option.

**Why?** (55 words):

> "Tuning tries several settings and keeps the one that scores best. If the rows that pick a setting also grade it, the grade is too kind. So tuning runs inside each training fold, and the rows held back grade the result. Most of the time goes to the 50 folds the models are compared on."

**Variants of the line.**
- **Below an effective size of 300:** "Tuning: standard settings (214 people are too few to rank settings)." Its Why? (56 words):

  > "With 214 people, the best of many settings would change from sample to sample, so each model keeps its standard settings. Whether trees keep or fill blanks is still chosen in each fold: a choice between two rarely costs much when it goes wrong, because it goes wrong mostly when the two predict about equally well."
- **Under inference:** silent unless a penalized family is chosen.

### 6.3 The override flow

1. The default is stated, with its reason under "Why does this matter?".
2. Pointing at a phrase elaborates it in one line, its reason. Clicking it expands the card to its options, with a way back (§6.1); only then does each option preview on the canvas, with the shelf's order beside it. Nothing is recorded.
3. Choosing holds the change in the stage. The phrase shows its new value, with the quiet label "changed by you".
4. **After first results,** the options keep their soundness order, under one line chosen by holdout status:
   - **none:** "You've seen these scores. The earlier version stays in the comparison, so the final score accounts for choosing between them." (20 words)
   - **sealed:** "The held-out rows are still sealed, so this change can't flatter the final result." (14)
   - **after scores:** "These held-out rows were in the scores you saw, so the earlier version stays." (14)
   - **opened:** "The held-out rows are open, so this runs as a secondary analysis." (12)

   **The line depends on the slot as well.** On a slot whose "Try both" result was shown, the option the folds favored (§3.5) carries a different line under none or after scores: "The folds already chose this on these rows, so fixing it here would make its score too kind." (18), with the exit "Keep trying both". The slot's other options take the line for the holdout status.

   For a shared step changed after first results, the line reads: "You've seen these scores. They stay in the comparison for reference; this change is disclosed, not corrected." (17)

   "Try both; keep what predicts better" is labeled available, with its minutes.
5. **The stage's Confirm records everything held.** The analysis flowchart's Fit then carries the minutes and the cost of kept versions: "Fit · about 45 min · 2 earlier versions kept". It passes the five-second test.

### 6.4 The comparison table

- It follows §3.7.
- **Colors:**
  - hues by family, in selection order (sage, plum, ochre, steel, clay);
  - a family's versions share its hue;
  - earlier versions and read-only rows are gray;
  - a sixth or seventh selected family is drawn gray, with a direct label, in charts. The table needs no color.
- Technical names appear only as quiet terms ("nested cross-validation", "BBC-CV").
- **The label "revised after first results"** heads the collapsed gray lines and appears on no other element (§6.0).

### 6.5 Vocabulary in two registers

**Quantities in units take the grain's noun** ("people", "visits"). Quantities in rows say "rows".

| Parameter | On the card | Quiet term |
|---|---|---|
| `learning_rate`, `eta` | how big each correction step is | learning rate |
| `max_leaf_nodes` | how many groups each tree may split the rows into | leaves per tree |
| `min_samples_leaf` | the fewest rows a group may hold | minimum leaf size |
| `min_child_weight` | how much evidence a split needs, in rows | minimum child weight |
| `l2_regularization`, `reg_lambda` | how strongly each group's value is pulled toward zero | L2 regularization |
| `max_iter`, `n_estimators` | how many trees | number of trees; boosting rounds |
| `max_depth` | how many questions deep each tree goes | maximum depth |
| `max_samples`, `subsample` | the share of rows each tree sees | row subsampling |
| `colsample_bytree`; HistGB `max_features` | the share of columns each tree or split sees | column subsampling |
| forest `max_features` | the share of columns each split tries | mtry |
| ridge and elastic net λ | how strongly coefficients are pulled toward zero | penalty (λ) |
| `l1_ratio` | how freely it may drop predictors entirely | lasso–ridge mix |
| robust linear `t` | how far a row may sit before it counts less | Huber threshold |

### 6.6 Purpose registry entries (BLUEPRINT §11.2)

| Element | The question it answers |
|---|---|
| Recipe line and its phrase; "What each model is given" | 1 · What is this choice? |
| Model-matrix preview, with both inputs under "Try both" | 2 · What will it change in my model? |
| Tuning line, and the Fit action with minutes and kept versions | 2 · What will it change, and at what cost? |
| First-fit line about kept versions; after-first-results lines | 3 · Why does that matter for my result? |
| Comparison columns "What it was given" and "Settings"; the "revised after first results" lines, kept versions and read-only rows | 5 · What did I decide, and can a reviewer reproduce it? |
| Settled values; stability strips; "Try both" fold shares | 3 and 5 |

The shelf's own lines ("Ranked on the defaults now set", "Waiting for", the "Looked at" line) are registered by the contract's MC-17.

---

## 7 · Method contracts, relations and leash rows

### 7.1 Contracts

All are registered with `contracts.register_contract`, in new modules `models/recipes.py` and `models/tuning.py`. A contract that reads the outcome declares scope `model`, as `variable_selection` does, which is what `observed_scope` returns.

| Key | Slot · scope | Decision · stage | Place |
|---|---|---|---|
| `family_recipe` | in_fold · training_fold | `set_recipe` · design | row 10 |
| `native_missing` | model · model (split directions are learned with the outcome) | `set_recipe` · design | row 10 |
| `recipe_choice_nested` | model · model | `set_recipe.choose`, `default_choose` · fit | row 11 |
| `hyperparameter_search` | model · model | `set_tuning` · fit | row 11; run order §1.1, step 9 |
| `path_search` | model · model | `set_tuning` · fit | row 11 |
| `manual_settings` | model · model | `set_tuning` · fit | row 11 |
| `shown_versions` | evaluation · model | `set_recipe`, `set_tuning`, `declare_version` · fit | rows 11–12 |
| `read_only_rows` | evaluation · model | any shared-step answer after first results; a recipe or tuning change under a sealed holdout · fit | row 12 |

### 7.2 Relations

| Id | Kind | Condition → what fires | Enforced by |
|---|---|---|---|
| `recipe_stated` | implies | a family selected → its line (silent when its footprint is empty), spur and methods clause | `family_spec`; voice |
| `fill_in_track_models` | implies | rows kept → who is kept is Who's in's answer; the fill is each track's Models answer; the missing slot reads the track's fill | `family_spec`; D2 |
| `trees_try_both_default` | implies | prediction, rows kept, a tree family, at least one routed column → `choose: [native, fill]` unless the researcher set one option; at every size | `family_spec`; plan |
| `trees_route_blanks` | implies | prediction, rows kept, a tree family taking blanks (alone or under "Try both") → the routed columns and their count, in the lineage and the sentence | `family_spec` |
| `blank_intolerant_fill_first` | implies | a step that cannot pass a blank reads a column → that column is filled for every family | `family_spec` |
| `no_column_routed_said` | implies | a tree family with zero routed columns while blanks exist → the line says which step needs every value; "Try both" is silent and adds no candidate | `family_spec`; voice; plan |
| `trees_skip_form_rule` | implies | Explore's spline rule on → tree families do not take it | `family_spec` |
| `routed_no_indicator` | implies | a routed column kept as blanks → no indicator, and the sentence says so; under "Try both" the fill option carries indicators for those columns | `family_spec`; voice |
| `native_not_under_inference` | conflicts (not offered) | inference → the fill | validator |
| `recipes_differ_marked` | implies | compared versions differ in the information they see → the line under the table; pairwise sentences speak of procedures | fit stage |
| `choice_after_scores_disclosed` | implies | a stamp shows versions → the sentence names them, carries "revised after first results", and says what is not corrected | completion; voice |
| `revised_label_placement` | implies | "revised after first results" → the comparison and the methods text only; never a stage badge, the progress bar or a card | voice; stage registry (P0.4) |
| `shown_versions_stay` | implies | holdout status none or after scores → every shown version is fitted, reverts included | fold; design stage |
| `version_key_non_default` | implies | a version → its key is computed over the fields that differ from their defaults; adding an optional field re-keys nothing | `VariantSpec` |
| `read_only_rows_kept` | implies | Predict, before the opening: an entry on an earlier trunk decision key, or, under a holdout sealed before any score, an earlier version on the current trunk → a read-only row, with its trunk decision key, trunk stage key and log sequence in the versions file; never refit, in BBC-CV, final, scored on held-out rows or exported; a revert brings it back | versions file; fit stage; `open_seal` and `declare_version` validators; export |
| `read_only_by_decisions` | implies | read-only status → read from the trunk decision key, never a stage key; a stage version bump with no answer changed creates no read-only row, and every version on the current trunk is refit as current | versions file; fit stage |
| `compared_families_stay` | conflicts (refused, exists) | dropping a family whose score was shown, unless sealed or opened → exit "Keep the compared families" | `selection._compared_families_stay` (by version) |
| `sealed_holdout_absorbs_change` | implies | a holdout sealed before any score → no earlier version fitted; its scores stay read-only; the sentence says why | completion |
| `holdout_after_scores_labeled` | conflicts (block and record) | a holdout drawn after scores → treated as none; its score labeled | `set_split` validator; completion |
| `opened_change_is_secondary` | conflicts (block and record) | seal opened → the change is secondary; the opened version stays final | validator; `mark_final` |
| `deployed_is_best_unless_declared` | implies | no holdout → BBC-CV's choice is reported and used; when it is an earlier version, the line and its exits | fit stage |
| `values_from_tuning_data_derived` | conflicts (block and record) | after a tuning result was shown, no sealed holdout: values set or held, or a slot fixed to an option its served "Try both" search favored (the final refit's, or one that won at least half the outer folds) → labeled; excluded from BBC-CV and the result; exits "Tune inside each fold" and "Keep trying both". Fixing an option the search did not favor is a kept version instead | completion; validator |
| `folds_choose_nested` | implies | `choose` or `default_choose` in force → the options join the search, the standard settings once per option; shares shown | tuning |
| `plan_frozen` | implies | a searched or path family → one plan, set from n_plan, used in every fit | design stage |
| `plan_strategy_counts` | implies | any plan → its strategy generates the candidates and counts the fits; the estimate reads the count | `TuningPlan.strategy`; cost |
| `tuning_nested_everywhere` | implies | a searched family → the search runs in every fit (§4.3) | `fit_pipeline` |
| `tuning_splits_follow_outer` | implies | units repeat, time order, or the population answer → inner splits by unit, forward-chained, or by whole PSU | `inner_splits`, `design` |
| `stopping_rows_first` | implies | early stopping → stopping units drawn first, kept out of every step's fit and of any resampling | `fit_parts`; `ImbalanceCorrected` |
| `path_search_refits_head` | implies | a path family → the head is refit in every inner split | tuning |
| `tuning_on_primary` | implies | any search → pooled strictly proper primary, weighted under the population answer | tuning |
| `tuning_estimate_first` | implies | a fit over 30 s → its estimate in the preview and on Fit; over 2 min → waits for Fit on the analysis flowchart, then runs as a server job | cost; scheduler |
| `fit_records_lock` | implies | Estimate or Describe, Fit pressed → `lock_plan` recorded; no estimate stage served before it, Describe's included | server; `served_gate` (P0.8); estimate stages declared on `Stage` (seam guard 7, P0.4) |
| `predict_scores_stamped` | implies | Predict, any score, tuned value or fold share served → stamped as shown, before or after Fit | completion; versions file |
| `small_n_standard_settings` | implies | effective n_plan below 300 → the standard candidates only, one per "Try both" option, stated with the count | plan |
| `try_both_floor` | implies | the plan's K capped once so each inner fold holds at least 2 units of the rarest class, or 2 PSUs; fewer than 4 in the plan's fit → no inner folds in any fit: each slot's first option and the standard settings, stated | plan |
| `shrinkage_small_n` | implies | a penalized family below Riley's minimum → the concern and the fold spread | `shelf_order`; fit |
| `penalty_at_edge` | implies | the chosen penalty at a grid edge in most folds → the concern | fit |
| `manual_labeled` | implies | manual values → "set by hand" everywhere, with when | voice |
| `pinned_for_description` | implies | a band or reseed of a searched family → tuned values and the chosen option held; caption "conditional on its tuned settings" | `pinned_to_full_fit` |
| `inference_recipe_is_the_plan` | conflicts (refused) | inference, `set_recipe` → refused with the exits in §5 | validator (reads the purpose) |
| `shelf_on_actual_input` | implies | the families question → each family assessed on its profile through `family_spec`, one profile per "Try both" option, outcome-blind | the contract's C4 (MC-4, MC-5) |

**The chain test** (BLUEPRINT §13) gains one reference chain: an NHANES-shaped prediction with blanks, families {linear, boosted trees, ridge}, rows kept, the population answer.
- It asserts `fill_in_track_models`, `trees_try_both_default`, `trees_route_blanks`, `blank_intolerant_fill_first` (on the energy nutrients), `recipes_differ_marked`, `plan_frozen`, `tuning_nested_everywhere` and `tuning_splits_follow_outer` (whole PSUs).
- After a scripted recipe change once first results were seen, it also asserts `choice_after_scores_disclosed`, `shown_versions_stay` and `revised_label_placement`. After a scripted change to the energy model, it asserts `read_only_rows_kept`; after a scripted stage version bump, `read_only_by_decisions`.
- Each must appear in the lineage, in the participant flow where it applies, and in the methods sentence.

### 7.3 Leash rows, added to MODELING_SEQUENCE §4

| Step | Prediction | Inference |
|---|---|---|
| A family's stated recipe | stated (Confirm); silent when it changes nothing here (For the record) | stated, "set by your plan" |
| Trees try both with blanks | recommended (rows kept; the default, at every size) | refused; exit: declare one |
| Trees keep blanks as blanks | available | not offered |
| Trees take the shared fill | available | the only option |
| Penalized: every column on one scale | recommended | the plan's (shrunk table) |
| Penalized: Gelman's two-SD scaling | available | not offered |
| Penalized: columns in their units, or a reference level | ranked lower | not offered |
| Unpenalized: scaling | silent | silent |
| Per-model reshaping (Yeo–Johnson) | ranked lower; the curve question first | refused; exit: the curve question |
| Per-model cap | ranked lower; robust linear regression offered | refused; exit: plausibility repairs |
| Try both on any other slot | available | refused; exit: declare one |
| Automatic tuning, nested | recommended from an effective size of 300 | not applicable (path search only, for a labeled shrunk table) |
| Standard settings | available; recommended below 300 | not applicable |
| Values by hand before any tuning result was shown | available, labeled | not applicable |
| Values by hand or held, or a "Try both" slot fixed to the option its folds favored, after that result was shown | block and record (no holdout, or a holdout after scores); exits "Tune inside each fold" and "Keep trying both"; available under a holdout sealed before any score | not applicable |
| Any other recipe or tuning change after first results | available: disclosed and labeled "revised after first results"; the earlier version kept and the choice among versions corrected, unless a holdout was sealed before any score (then kept read-only) | not applicable (a secondary after the lock) |
| A shared-step change after first results | available: disclosed, not corrected; the earlier scores kept read-only, labeled "revised after first results" | not applicable (a secondary after the lock) |
| Drawing a holdout after scores were seen | block and record: labeled, and treated as none | not applicable |
| A change after the seal was opened | block and record: secondary | not applicable |
| Reporting a version other than the best (no holdout) | available: its own score labeled, the selection-corrected estimate beside it | not applicable |
| Robust linear regression | available (numeric) | not offered in v2 |

**Impossible by construction.** These are stated in the methods reference, with no validator and no menu item:
- tuning on all training rows and then cross-validating (its reachable form is §3.5);
- tuning on AUC or accuracy (the search scores only the primary);
- class reweighting set on a family (only Explore's imbalance lever, which recalibrates);
- XGBoost's linear booster;
- the robust linear threshold tuned by cross-validation;
- a read-only row declared final, scored on the held-out rows or exported as a model.

---

## 8 · Acceptance tests, each with an independent reference

All are Tier A unless marked.

**Slow tests.** T1, T8(c) and T12 carry the `slow` marker, run in the acceptance harness only, and are scheduled with Nolan outside quiet hours.

**T1 · Tuning outside nested cross-validation is optimistic; nesting is not.** (slow)
- **Generators:**
  - (a) null: n = 200; ten N(0, 1) predictors; y ~ Bernoulli(0.3).
  - (b) signal: logit = −0.85 + 0.8x₁ − 0.6x₂ + 0.5x₁x₃.
- **Procedure F (flat: what not to do),** in plain scikit-learn: the plan's candidates evaluated by 5-fold cross-validation on all rows; the best one's log loss reported.
- **Procedure N (ours):** the fit stage's cross-validated log loss for boosted trees, with the automatic plan forced on.
- **Truth:** retrain each procedure on 4/5 of the rows, and score it on 20,000 fresh rows. In (a), the floor is the entropy, 0.6109 nats.
- **Assertions, over 200 datasets:**
  - F's mean is below its truth by more than 3 Monte Carlo standard errors;
  - N's mean is within 2 standard errors of its truth;
  - with ridge added, the selection-corrected estimate is within 2 standard errors or conservative.
- **Also reported, not asserted:** nested tuning against standard settings on fresh-data loss at an effective size of 200 (§4.6).

**T2 · Known answers.**
- (a) **Ridge's path search** on the diabetes data, with explicit inner splits, against numpy's closed form at each λ and the **pooled** squared error's argmin. Index exact; score to 1e-10. The test also asserts that the scorer is the pooled loss.
- (b) **Candidates** equal the standard candidates, one per "Try both" option and in that order, followed by scipy's scrambled Sobol mapped through each documented scale, recomputed in the test, with the seed derived from sha256.
- (c) **Pinned replay** reproduces the deployed predictions exactly at the recorded thread count.
- (d) **The elastic net's path search** equals independent `ElasticNet` and `LogisticRegression` refits over the same ratios and splits, choosing on the pooled loss.
- (e) **Ties:** two candidates with losses equal after rounding resolve to the lower index.

**T3 · Inner splits follow the outer ones.** The search logs every inner (training, validation, stopping) set.
- **3 rows per person:** no person on both sides; stopping rows are whole persons.
- **Time-ordered:** every inner validation unit is later than every inner training unit.
- **NHANES-shaped design** (15 strata, 2 PSUs each, outer design K = 2): every inner split keeps whole PSUs; K equals min(3, ⌊P/2⌋), P the fewest PSUs in any outer training fold, in every fit; the weighted inner loss equals a hand computation.
- **The oversample lever with 3 rows per person:**
  - no person sits on both sides of the stopping split;
  - no stopping row is resampled;
  - the recalibration splits cover only the candidate's own rows.

**T4 · No leakage, by perturbation.**
- `observed_scope` returns `model` for the search contracts.
- Perturbing outer fold k's held-out rows leaves fold k's chosen values, chosen option and fit unchanged bit for bit.
- Perturbing a training row changes them.

**T5 · The native path is reachable.** Blanks in three numeric columns that no energy step reads, and in one energy nutrient; "Keep every row" in Who's in; prediction.
- (i) **With the missing slot set to keep blanks,** the HistGB, forest and XGBoost model steps receive NaN in exactly those three columns, and the nutrient is filled.
- (ii) **Under the default "Try both",** each inner split fits two heads: the native head passes NaN in the three columns, and the fill head passes filled values with an indicator for each of them.
- (iii) The linear model receives no NaN.
- (iv) HistGB fit directly on a hand-built matrix gives identical predictions.
- (v) The lineage and the sentence name the routed columns, and say no indicator was added when blanks are kept.
- (vi) **With Explore's spline rule on,** the trees still receive the three columns blank.
- (vii) **With the selection step on,** none are routed, the line names the selection step, and "Try both" adds no candidate.
- (viii) Under inference, `set_recipe` gets a 409 with its exits.
- (ix) The missing slot reads the track's fill: changing the Predict track's indicator answer changes the fill head, and Who's in's answer is unchanged.

**T6 · The estimate counts the fits.** (Tier B)
- An instrumented counter equals the strategy's count, Σ F(plan) over the outer fits, with and without the imbalance lever, with and without the out-of-bag rule, and with "Try both" on and off.
- The estimate obtains its count from `TuningPlan.strategy`, not from a formula of its own.
- The estimated time is within a factor of 3 of the measured time.

**T7 · The after-first-results paths, scripted.** Fit {linear, ridge}, serve the fit, then record `set_recipe(ridge, overrides={"scale": "two_sd"})`, a slot no served search covered.
- (i) The stamp lists both versions even when the client sends none.
- (ii) The sentence says "revised after first results" and "after the cross-validated scores of … were seen", and states what is not corrected.
- (iii) The next fit has three rows, the earlier one collapsed under the label "revised after first results".
- (iv) BBC-CV's set has three versions and matches an independent BBC-CV written from Tsamardinos et al.'s algorithm, on the same draws.
- (v) **A revert of the change** keeps the changed version, once shown, as an earlier version, and a replay from `decisions.jsonl` alone rebuilds both.
- (vi) **A holdout sealed before any score:** no earlier version fitted; its scores shown as a read-only row; the sealed sentence; `open_seal.variant` and `declare_version` refuse that row, and a revert of the change makes the earlier version current and fitted again.
- (vii) **A holdout drawn after scores:** the earlier version is fitted and kept; no sealed sentence; dropping a compared family is still refused; the held-out score is labeled.
- (viii) **The reported version** is BBC-CV's choice. When that is the earlier version, the line and both exits appear. `declare_version` reports the current version's own score, labeled, with the corrected estimate beside it.
- (ix) **After the opening,** a change runs as secondary, and the opened version stays final.
- (x) A versions file holding version keys does not make `select_models` refuse.
- (xi) A pre-spec record listing `boosted_trees` maps to its legacy version, and keeps it as an earlier version.
- (xii) **A shared-step change** (the energy model) after first results: the earlier trunk's rows stay read-only under the label; they are not in BBC-CV; `open_seal` and `declare_version` refuse them; they are never scored on the held-out rows nor exported. With the stage cache cleared, the rows are still served from the versions file, and folding the log to the recorded sequence reproduces the recorded trunk decision key, and, on the same engine version, the recorded trunk stage key. Reverting the energy-model answer puts those entries back on the current trunk, fitted as usual.
- (xiii) **Version keys:** adding an optional field with its default to `VariantSpec`, or a recipe slot whose default reproduces today's behavior, leaves every recorded key unchanged; a version at every default keys on its family, defaults version and space version alone.
- (xiv) **The trees' blanks fixed to the folds' choice.** On T8(b)'s generator (blanks in x₁ carry the outcome), fit {linear, boosted trees} and serve the fit with its fold shares; the test first asserts that the final refit kept blanks and that keeping blanks won more than half the outer folds, so only that option is favored. Then `set_recipe(boosted_trees, overrides={"missing": "native"})` is data-derived: fitted, labeled "option chosen on these rows: not an honest estimate", excluded from BBC-CV and the result, with the coach line, the exit "Keep trying both" and the data-derived option sentence. Under a holdout sealed before any score the same change is allowed.
- (xv) **The trees' blanks fixed against the folds' choice.** On the same fit, `set_recipe(boosted_trees, overrides={"missing": "fill"})` is not data-derived: the earlier version, which tried both, stays fitted; BBC-CV's set has three versions; the sentence is "against the folds' choice".
- (xvi) **An engine upgrade.** Bumping the stage version of `trunk`, `design` or `shelf` with no answer changed creates no read-only row; every version on the current trunk is refit as current; no "revised after first results" line or sentence appears.

**T8 · Try both.**
- (a) The shares across folds equal an independent count from the per-fold records, and each fold's record holds every candidate's inner loss.
- (b) On a generator where the outcome depends on whether x₁ is blank, keeping blanks wins most folds. This is a sanity check, not a performance claim.
- (c) (slow) **Small samples:** at an effective size of 200, on (b)'s generator and on one where blanks are uninformative, each with 6 and with 30 routed columns, "Try both" at the standard settings against each option alone, on 20,000 fresh rows, over 200 datasets. It reports fresh-data loss and the calibration slope's mean and spread across datasets. If "Try both" is worse than the better single option by more than 2 Monte Carlo standard errors on either measure, §4.6's rule is revisited.
- (d) **The regret bound:** a numerical check that Δ·Φ(−Δ/σ_d) peaks at about 0.17·σ_d near Δ = 0.75·σ_d, against an independent root-finder.

**T9 · Small samples.**
- 5,000 units at 3% prevalence: the plan uses the standard candidates, two for the trees under "Try both", and the line counts events.
- At an effective size of 214: the plan holds exactly the two standard candidates for boosted trees, the folds choose between them, the tuning line reads "standard settings" and the recipe line reads "try both".
- With 5 units of the rarest class in the plan's fit, every fit uses K = 2. With 3, no fit draws inner folds: the slot takes "keep blanks as blanks" in every fit, and the line says so. A fit holding fewer than 4 units of the rarest class keeps K = 2, and the record counts it.
- n = 150: ridge carries the Riley concern, and its 10th–90th percentile range of λ equals an independent computation.

**T10 · Inference refusals and the lock.** Each refusal in §5 returns its exits. The causal lasso's inner folds keep units whole. A recorded `random_forest` or `boosted_trees` causal learner parses as its renamed key. On a Describe track, no estimate stage (the usual-intake distribution, the survey-weighted means and prevalence) is served before `lock_plan`, and `ESTIMATE_STAGES` equals the stages declared as estimates on `Stage`.

**T11 · Each family against an independent reference.**
- Ridge: the closed form, to 1e-8.
- Robust linear:
  - an independent IRLS of Huber's M-estimator with MAD scale, written from Holland & Welsch (1977), to 1e-6;
  - R `MASS::rlm`'s published stackloss output, with the constants copied in with their source.
- Forest and XGBoost: direct library fits, exactly at the same thread count; XGBoost also against native `xgb.train`.
- SHAP:
  - XGBoost's `pred_contribs` against v2's TreeSHAP on small fixtures (the `<` split rule handled);
  - the forest's compiled TreeSHAP against v2's numpy TreeSHAP to 1e-6, blanks included;
  - ridge's and robust linear's SHAP against the closed form.

**T12 · F4 is fixed.** (slow) On a p ≫ n null fixture (n = 120, p = 2,000):
- paired over 50 datasets, the screened elastic net's penalty is larger with the screen refit per inner split (by 3 standard errors);
- both outer scores are within 2 standard errors of the fresh-data truth.

**T13 · Reproducibility.**
- The same fixture and seed at the same recorded thread count give identical choices and predictions, to replay's 1e-12, for every family.
- A different seed gives a different candidate list.
- Re-run replay reproduces the recorded choices.
- Pinned replay reproduces the deployed predictions without searching, and the `TuningRecord` lists its candidates in the order evaluated.

**T14 · The chain test** of §7.2.

**T15 · Budgets, purposes and API.** (Tier B)
- Every new string is within its budget.
- Every new view has a registry entry.
- `GET /api/models` lists recipes, `default_choose` and tuning.
- A coefficient is undone exactly through a scaler with arbitrary centers and scales (V2X_SEAMS row 6).
- The label "revised after first results" appears in no stage state, reopen reason or card string.

**T16 · Values from tuning results.**
- On T1's null generator: copy the final refit's values into manual. The version is block-and-record, labeled, and excluded from BBC-CV and the result. The test reports the size of its optimism.
- After "Try both" fold shares were shown, fixing the trees' blanks to the option the search favored is data-derived; fixing the other option is not, and its earlier version stays.
- With a holdout sealed before any score, the same values are allowed.

**T17 · One plan.**
- At 340 effective units with 5 folds: every outer fit, bootstrap resample and the final refit carry the same `TuningPlan`.
- At 1,600: early stopping is decided once, by the plan.
- At 214: every fit carries the same two standard candidates.
- At 5 units of the rarest class: every fit carries K = 2; at 3, every fit carries no inner folds, whatever its own count.

---

## 9 · Engine work packages

**Sizes** follow SIZING: S = 1 (one module and its reference test), S–M = 2, M = 3 (a few modules, an explanation path or tuning), M–L = 5.5, L = 8 (a new workflow). Each package below carries SIZING's size, and each table names the SIZING package that holds it.

**Order** follows BLUEPRINT §12 and SIZING's road: C6a first (the math, then the families), then C6b (recipes, versions and their tests), with RT-8 in P0.8; C6d, the screens, once C6b, SIZING C4 (Who's in) and the quest-log shell (P0.7) are in.

**Package C6c is not here.** Its items left v2 on 2026-10-08 and are INBOX (§10). Its seam is kept: `TuningPlan.strategy`, below.

**Scope added in draft 3, and its size.** Draft 3 added work that SIZING's packages did not count: stamps on every shared-step answer after first results; a versions file holding served scores and two trunk keys; read-only rows of two kinds, with their refusals and their revert; and the long fit as a server job with a notification. RT-6 absorbs the stamps at its size, since they are one more completion on existing kinds. The rest is not absorbed: RT-7 grows by S–M (2) for the read-only rows and the served scores, and RT-8 by S (1) for the notification. C6b becomes 32 and P0.8 becomes 7, about 3 units in all, recorded as an amendment to SIZING (§10).

### C6a · Prediction: tuning and the four families (SIZING: 32)

| WP | Size | What | Where | Tests |
|---|---|---|---|---|
| RT-1 · The search engine | L | `TuningDecl`, `Dimension`, `TuningPlan` (frozen, effective n, and **`strategy`**, one value `"sobol"` in v2, which owns the candidate generator and the fit count); the standard candidates, one per "Try both" option, then Sobol candidates with sha256 seeds; `fit_parts` (rows an argument, stopping units first, one head per split and per option, nested splits redrawn); pooled weighted loss with rounding; K capped once in the plan at 2 units of the rarest class, or 2 PSUs, per inner fold, with no inner folds (each slot's first option) when even K = 2 cannot meet it; out-of-bag scoring and its conditions; the path search; `TunedPipeline` (`search=None`, `at`); `TuningRecord` (candidates in evaluation order, every candidate's inner loss); cancel. `fit_pipeline` gains `design=` and dispatch; seeds threaded from the split. Fixes F11, F13 and F15. | new `models/tuning.py`; `models/inner_cv.py`; stage call sites | T2(a–e), T3, T4, T9, T13, T17 |
| RT-4 · Imbalance correction | S | The early-stopping interface; stopping units before resampling; recalibration splits per fit. Fixes F12. | `methods/levers.py` | T3 |
| RT-5a · Boosted trees tuned | S | Its `tuning` declaration; `defaults_version` "2", with "Try both" as its blanks default; `describe`. | `models/boosted_trees.py` | T1, T2(c) |
| RT-5b · Ridge | S | The family, path declaration, full coding; `Ridge` in `explain.LINEAR_MODELS` until MC-2a replaces that list. | new `models/ridge.py`; `models/explain.py` | T2(a), T11 |
| RT-5c · Robust linear | M | The `RLM` wrapper; `purposes = ("prediction",)`, with no key check anywhere; numeric only; linear SHAP. | new `models/huber.py`; `models/explain.py` | T11 |
| RT-5d · Random forest | L | The family and its defaults, "Try both" included; out-of-bag tuning; a compiled TreeSHAP (the `shap` package's TreeExplainer, added to the server requirements once T11 confirms it matches v2's TreeSHAP with blanks; otherwise explanations cover the rows v2's TreeSHAP can explain within the stage's budget, and the caption counts them); the forest's explanation scale (probability for SHAP; log-odds of the clipped probability for curves, labeled). | new `models/forest.py`; `models/explain.py` | T2(c), T11 |
| RT-5e · XGBoost | M–L | The wrapper (early-stopping interface, margin `decision_function`, label encoding, safe names, pinned threads); "Try both" as its blanks default; `same_kind_as` boosted trees; SHAP from `pred_contribs`. | new `models/xgboost_family.py`; `models/explain.py` | T11, T13 |
| RT-5f · Elastic net on the path search | M | Full coding; per-row logistic grid; `alpha_` and `l1_ratio_` readers moved to the record; the screen refit per split. Fixes F4 and F5. | `models/elastic_net.py`; `methods/omics.py` | T2(d), T12 |
| RT-11 · Causal lasso folds by unit | S | `inner_splits` for the lasso's folds. Fixes F9. Lands with the contract's MC-2b renaming of the learner keys that collide with families (`random_forest`, `boosted_trees`), with parse aliases and tombstones (seam guard 2). | `models/causal.make_learner` | T10 |

### C6b · Prediction: recipes, versions and cost (SIZING: 30; 32 with this spec's amendment)

| WP | Size | What | Where | Tests |
|---|---|---|---|---|
| RT-2 · Recipe declaration | M–L | `RecipeSlot` (with `default_choose`), `RecipeOption`, `defaults_version`; recipes in `DesignSpec`; `family_spec` inside `build_pipeline`, `family_steps` and `describe_steps`, resolving each "Try both" option's spec; `PenaltyScaler` (`center_`, `scale_`; two-SD), undone by reading `center_` and `scale_`; `MissingLevelEncoder(drop=)`; transform and cap steps; the trees' form-rule skip; lineage spur; `register_family` checks; `FamilyInfo`; teaching options for the new families. | `models/base.py`, `models/pipeline.py`, `models/explain.py` (`_undo_scaling`, `linear_equation`), `models/elastic_net.coefficients`, `models/artifacts.py`, `models/lineage.py`, `stages/__init__.py` (reads, version bumps), `teaching/content.py` | T15, T5(v) |
| RT-3 · The trees' own blanks, tried both ways | M | `reads(spec)` and `passes` (a set; only `"blanks"` in v2) on every step; `native` computed by walking the steps; the imputer and indicators skip routed columns; the trees' `default_choose`, with indicators on the fill option; the missing slot reading the track's fill (D2 moves the fill into each track's Models); the preview of both inputs, its caption and `table_focus`; the "Keep every row" copy in Who's in, and "No restriction" on the exclusions card; the teaching note. | `models/pipeline.py`; `methods/*` steps; `models/previews.py`; `teaching/content.py`; `ChoiceQuestions.tsx` (copy only) | T5 |
| RT-6 · Decisions and stamps | M | `SetRecipe` (`overrides`, `choose`), `SetTuning`, `DeclareVersion`, `Stamp` (with its track); completions on seven kinds (revert gains one) and on shared-step answers after first results; holdout status; the data-derived rule, the trees' default included; validators and exits, the inference refusal reading the purpose; key views; sentences, with "revised after first results"; previews with estimates, and with the edited family's profile (the contract's MC-6). The new kinds land after the decision log's format marker (seam guard 1). | `decisions.py`, `voice.py`, `models/previews.py`, `graph.py` | T7(i–ii), T10, T16 |
| RT-7 · Versions kept and labeled "revised after first results" | L, plus S–M (2) | `VariantSpec` keys over the fields that differ from their defaults (seam guard 4); `versions_shown` folded across reverts; `LEGACY_DEFAULTS`; the versions file (`turbotab-versions/1`, per track) with specs, keys, served sequences, trunk decision keys, trunk stage keys with the TurboTab version, and served scores; read-only status from the trunk decision key, never a stage key; the read-only rows of both kinds (an earlier trunk; an earlier version under a sealed holdout), their keys and log sequence persisted there (seam guard 5), refused by `open_seal`, `declare_version` and the export, and brought back by a revert; one `role` on every fitted version; `compared_families`, `vouch`, `scored_in` and `explained_in` by version; `design.objects["earlier"]`; fit and evaluation (`design_bbc`) include earlier versions; `variant` on rows; the out-of-fold cache; `declared_result` and `declare_version`; the seal by version (`open_seal.variant`, `AtOpening`, `mark_final`). | `models/selection.py`, `seal.py`, `stages/modeling.py`, `stages/evaluation.py`, the frontend's results types | T7 |
| RT-9 · Fit artifact and results data | S–M | `FittedModel.variant`, `role`, `inputs`, `settings`, `tuning`; per-fold records through a `cross_validate` hook, holding the chosen option and every candidate's inner loss; `pinned_to_full_fit` holds the tuned values and the chosen option, with its caption. | `models/artifacts.py`, `models/metrics.cross_validate`, `stages/modeling.py`, `stages/explain.py` | T8, T9 |
| RT-10 · Contracts and relations | S | §7's contracts (scope `model` where the outcome is read), relations, the chain, methods clauses. | `models/tuning.py`, `models/recipes.py` | T4, T14 |
| RT-12 · Export and replay of versions | M | Threads pinned and recorded; versions and the versions file in provenance; `TuningRecord` numbers among provenance estimates; pinned-replay mode, the replay of record for any strategy whose candidates depend on earlier scores; per-version matrix hashes; read-only rows rebuilt from the log. | `export/record.py`, `export/replay.py` | T13, T7(xii) |
| RT-13 · Catalog, shelf, journeys | S, plus a heavy run | Contract catalog entries (review lenses through the contract's `review_lenses`); the four new families' base assessments (§2.6), which MC-5 extends; `first_models` checked per journey; the journeys and review packets regenerated as a run scheduled with Nolan. | `reference/catalog.py`, `reference/journeys.py` | the journeys |
| RT-14 · Acceptance tests | M | T1 to T17. | `core/tests/acceptance/` | |

### P0.8 · Fit, the hold and a visible lock (SIZING: 6; 7 with this spec's amendment)

| WP | Size | What | Where | Tests |
|---|---|---|---|---|
| RT-8 · Cost, the hold and Fit | M, plus S (1) | Timing at `at(center)`; the fit count from the plan's strategy, with its multipliers; `outer_fits` counted across stages; the F7 fix; preview estimates; the scheduler's 2-minute hold (bypassed in replay); the Fit and Refit job payload; the server job, its notification and cancel. Counted in P0.8 only. | `models/cost.py`, `stages/modeling._estimates`, the job scheduler | T6 |

P0.8's other M is not this spec's: under Estimate and Describe, the serving gate that withholds every estimate stage until the track's lock, the lock recorded when Fit is pressed, and the lock shown with its time and SHA-256. That gate is only as complete as its list of estimate stages, so it waits on seam guard 7 in P0.4 (estimate stages declared on `Stage`, Describe's included; §4.4).

### C6d · Slice: Models under Predict (SIZING: 12)

**Presentation**, once the engine passes verification:

| WP | Size | What |
|---|---|---|
| PR-1 | M | Grouped recipe lines in the Models Confirm sweep, silent ones under For the record; "What each model is given"; phrases taking the focal region; Routing (both inputs under "Try both") and Strip canvases; option copy |
| PR-2 | M | The shared tuning line and its small-sample variant; per-family exceptions; Fit on the analysis flowchart with its held state, Lighter in its confirmation, and the long fit's notification; the Angles canvas |
| PR-3 | S–M | Comparison columns; the "revised after first results" lines (earlier versions and read-only rows), collapsed; the information line; the first-fit and after-first-results lines; the earlier-version-best line and its exits |
| PR-4 | S | Results Settings: the compact table, one or two strips, the "Try both" fold shares, More angles |

C6d's other parts are not this spec's: the validation scheme as a Confirm; selection, levers and intended use on their cards; and a drive of the metabolomics prediction journey (M). The contract's MC-17, the shelf card's two registers, lands with C6d.

### The seam hooks

Each is S, sits inside a package above, and is counted among the definition of done's eight seam guards (about 8 units in all, DoD 2026-10-08), not in SIZING's C6 totals.

| Hook | Lands in | Opens (V2X_SEAMS) | Test |
|---|---|---|---|
| `TuningPlan.strategy`, owning the candidate generator and the fit count; `TuningRecord` lists candidates in evaluation order (seam guard 3; missing hook 1) | RT-1, RT-8 | successive halving, Hyperband, TPE and BOHB (rows 1–2) | T6, T13 |
| Version keys over the fields that differ from their defaults (seam guard 4) | RT-7 | the Thorough budget, strategies, monotone constraints, native categories and the neural families' slots, with no recorded version re-keyed (rows 3, 5, 20) | T7(xiii) |
| The read-only row's trunk decision key, trunk stage key and log sequence, persisted in the versions file (seam guard 5; missing hook 2) | RT-7, with C7d | shared-step changes kept as versions (row 11); v2's own read-only rows survive a cleared cache | T7(xii) |

**Guards inside the packages, at no extra size:**
- RT-1: `fit_parts` keeps its rows an argument (row 1).
- RT-1 and RT-12: candidates recorded in evaluation order; pinned replay is the replay of record for any strategy whose candidates depend on earlier scores (row 2).
- RT-9: every candidate's inner loss per outer fold (row 4).
- RT-3: steps declare what they pass as a set, blanks first (row 5).
- RT-2: undo paths read `center_` and `scale_`, never a scaler's name (row 6; T15).
- RT-6: the refusal under inference reads the purpose (row 7).
- RT-5c: Huber's purposes are declared, never checked by key (row 8).
- RT-11, with MC-2b: the colliding causal learner keys renamed, with aliases and tombstones (seam guard 2; row 9).
- RT-7 and RT-9: one `role` on every fitted version (row 10).
- RT-6: the new kinds land after the decision log's format marker (seam guard 1).
- RT-7: the versions file carries a format string (rule 5).

### With the model-family contract

- **MC-1 and MC-2a land in C6a before RT-5b to RT-5e,** so the four families enter through the contract; MC-12 lands early with its expected failures. RT-2's recipe members join MC-1's declarations later; the overlap is netted in the contract.
- **MC-4,** the input profile, follows RT-2, RT-3 and MC-3. **MC-5** rewrites `assess` for every family, the four new ones included (§2.6).
- **MC-6** follows RT-6: the `set_recipe` preview returns the profile and the assessment.
- **MC-11** shares replay with RT-12, and **MC-12** shares the harness with RT-14.
- **MC-17** lands with C6d.

**Heavy runs** are scheduled with Nolan, never run unannounced: RT-13's regenerated journeys, T1, T8(c), T12 and the contract's MC-19 soundness runs. The dev machine is beside the bed.

---

## 10 · Amendments, INBOX and sources

### Amendments (on Nolan's approval)

- **MODELING_SEQUENCE §1 row 10:** "stated per family with its reason, silent where it changes nothing; under prediction its phrase can be changed (`set_recipe`) under the four safeguards of RECIPES_AND_TUNING §0; tree families try both ways with blanks by default; under inference the plan decides."
- **Row 11 gains:** "by a seeded random search over a plan fixed once and used in every fit, inside every outer training fold, on the comparison's pooled primary, with inner splits drawn as the outer ones are (whole PSUs under the population answer); a 'Try both' slot joins the search, its standard settings tried once per option at every size."
- **Row 12(b) gains:** "With no holdout the reported and deployed version is BBC-CV's choice; a different version may be declared after scores, reporting its own score labeled 'chosen after the scores were seen' beside the selection-corrected estimate. A holdout drawn after scores is labeled so, and does not count as sealed. A change after first results is labeled 'revised after first results' in the comparison and the methods text; a shared-step change keeps the earlier scores as read-only rows, never refit, corrected, declared final or exported."
- **MODELING_SEQUENCE §4** gains the rows of §7.3.
- **BLUEPRINT §4:** a heavy stage whose estimate exceeds 2 minutes waits for the user's Fit action on the analysis flowchart instead of starting on its own, and runs as a server job with a notification; replay is exempt. Under Estimate and Describe, pressing Fit records the track's lock (`lock_plan`), and no estimate stage is served before it.
- **MODEL_FAMILY_CONTRACT C4,** to match the rulings of 2026-10-08: the readiness list names who is kept (Who's in) rather than "the missing-values answer"; the Predict track's fill is a Models Confirm, ranked on its default; scales, batch and the omics normalization come before the families question (question 4, ruled); "outcome-blind" reads as no outcome relation and no score, with the outcome's counts allowed (question 1, ruled).
- **MODEL_FAMILY_CONTRACT C3:** steps declare `passes`, a set of the raw value kinds they pass through unchanged (only `"blanks"` in v2), instead of a `passes_blanks` flag, so native categories add `"categories"` without a second sweep of every step (V2X_SEAMS row 5; §2.3).
- **MODEL_FAMILY_CONTRACT C6:** the standard settings are not one "candidate 0" but one standard candidate per combination of "Try both" options with a footprint, listed first, with ties going to the first (§4.1, §4.2).
- **CROSSWALK:** `q:missing` in Who's in asks who is kept, with the option relabeled "Keep every row"; the fill becomes an item of each track's Models (ruling 5; SIZING D2).
- **CROSSWALK, Models' Decide list:** scales, batch and the omics normalization (Decide 10–12) move ahead of the families question (Decide 6), as the DoD ruled on 2026-10-08, recorded as disagreement 21 (the contract's question 4).
- **SIZING:** C6b becomes 32 and P0.8 becomes 7, for the scope draft 3 added to RT-7 and RT-8 (§9).
- **V2X_SEAMS row 11 and seam guard 5:** the read-only row persists its trunk decision key as well as the trunk stage key and the log sequence. The decision key decides read-only status and survives an engine upgrade; the stage key is kept for identity and for a rebuild on the same engine (§3.2).
- **Already made:** the 2026-10-05 amendment's "are being specified" now reads "are specified in `RECIPES_AND_TUNING.md`".

### INBOX (v2.x)

None of these is a recorded value in v2, so none needs a refused placeholder (V2X_SEAMS rule 2). Each folds in as new declarations under the version-key guard, through the seam named.

**Package C6c, displaced on 2026-10-08** (V2X_SEAMS §1):
- **Successive halving and Hyperband,** for large tables (Jamieson & Talwalkar 2016; Li et al. 2018). Seam: `TuningPlan.strategy`, whose own generator and fit count they supply; `fit_parts` takes their subsamples (row 1).
- **TPE and BOHB** (Bergstra et al. 2011; Falkner et al. 2018). Seam: the strategy; candidates recorded in evaluation order; pinned replay as their replay of record (row 2).
- **The Thorough budget:** one more `SetTuning.mode` value, keyed without re-keying (row 3).
- **The tuning curve** (Dodge et al. 2019), from full-fidelity scores: drawn from the per-fold inner losses RT-9 keeps (row 4).
- **Native categories** (HistGB, XGBoost), with the 255-level limit and a sparse-level concern: `"categories"` in the steps' `passes` (row 5).
- **Per-family Pareto and robust scaling:** new `scale` options, undone through `center_` and `scale_` (row 6).
- **The per-model log1p:** a new `transform` option, refused under inference by purpose (row 7).
- **Robust linear regression under inference:** a weighted M-estimator with a design-based or cluster sandwich, checked against R `robsurvey`, declared through the contract's `InferenceDecl` (row 8).
- **Nuisance learners from the family registry,** tuned inside cross-fitting (Bach et al. 2024): the estimators already take a factory; the colliding keys are renamed in v2; the factory-classifier bug in INBOX is fixed first (row 9).

**The kept comparisons, moved on 2026-10-07** (V2X_SEAMS §2):
- **"Compare with standard settings" as a kept version:** one more `role`, a comparator (row 10).
- **Shared-step changes kept as versions,** fitted and compared: rebuilt by folding the log to the read-only rows' recorded sequence, which reproduces their trunk decision key on any engine version (row 11; about an L on top of RT-7).
- **Re-tuned substitution bands:** one optional field on `set_substitution` (row 12).

**Other:**
- Monotone constraints, declared from domain knowledge (row 20).
- LightGBM (row 18); survival versions of the new families (row 21).
- Flat BBC-CV over every configuration (row 49).
- A cross-validated penalty for post-double-selection lasso (row 43).
- The hypothesis noticing adding a candidate model, through a `terms` recipe slot (row 13; the contract's question 2).

In-fold PCA for omics stays its own v2 item (SIZING C6e).

### Sources

**Tuning**
- Bergstra J, Bengio Y. Random search for hyper-parameter optimization. *JMLR* 2012;13:281–305.
- Bischl B, Binder M, Lang M, et al. Hyperparameter optimization: foundations, algorithms, best practices, and open challenges. *WIREs Data Min Knowl Discov* 2023;13:e1484.
- Cawley GC, Talbot NLC. On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR* 2010;11:2079–2107.
- Optuna FAQ, "How can I obtain reproducible optimization results?"
- Probst P, Boulesteix A-L, Bischl B. Tunability: importance of hyperparameters of machine learning algorithms. *JMLR* 2019;20(53):1–32.
- Probst P, Wright MN, Boulesteix A-L. Hyperparameters and tuning strategies for random forest. *WIREs Data Min Knowl Discov* 2019;9:e1301. The tuneRanger documentation (`replace = FALSE`; `num.trees = 1000`; out-of-bag evaluation).
- Kruppa J, Liu Y, Biau G, et al. Probability estimation with machine learning methods for dichotomous and multicategory outcome: theory. *Biom J* 2014 (and *BioData Mining* 2014;7:2 on terminal node size).

**Tuning, for the v2.x items** (cited to place them; not read for this draft, and read when each is designed)
- Bergstra J, Bardenet R, Bengio Y, Kégl B. Algorithms for hyper-parameter optimization. *NeurIPS* 2011.
- Dodge J, Gururangan S, Card D, Schwartz R, Smith NA. Show your work: improved reporting of experimental results. *EMNLP* 2019.
- Falkner S, Klein A, Hutter F. BOHB: robust and efficient hyperparameter optimization at scale. *ICML* 2018.
- Jamieson K, Talwalkar A. Non-stochastic best arm identification and hyperparameter optimization. *AISTATS* 2016.
- Li L, Jamieson K, DeSalvo G, Rostamizadeh A, Talwalkar A. Hyperband: a novel bandit-based approach to hyperparameter optimization. *JMLR* 2018;18(185):1–52.

**Selection and validation**
- Ambroise C, McLachlan GJ. Selection bias in gene extraction. *PNAS* 2002.
- Bates S, Hastie T, Tibshirani R. Cross-validation: what does it estimate and how well does it do it? *JASA* 2023.
- Tsamardinos I, Greasidou E, Borboudakis G. Bootstrapping the out-of-sample predictions for efficient and accurate cross-validation. *Mach Learn* 2018;107:1895.
- Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics* 2006;7:91.
- Wieczorek J, Guerin C, McMahon T. K-fold cross-validation for complex sample surveys. *Stat* 2022;11:e454.

**Small samples**
- Martin GP, Riley RD, Collins GS, Sperrin M. *Stat Methods Med Res* 2021;30:2545–2561.
- Riley RD, Snell KIE, Martin GP, et al. *J Clin Epidemiol* 2021;132:88–96.
- Van Calster B, van Smeden M, De Cock B, Steyerberg EW. *Stat Methods Med Res* 2020;29:3166–3178.

**Missing values in trees**
- Chen T, Guestrin C. XGBoost. *KDD* 2016. The XGBoost FAQ and "Notes on Parameter Tuning".
- Josse J, Chen JM, Prost N, Scornet E, Varoquaux G. On the consistency of supervised learning with missing values. *Statistical Papers* 2024.
- Perez-Lebel A, Varoquaux G, Le Morvan M, Josse J, Poline J-B. *GigaScience* 2022;11:giac013. Quoted from the abstract's conclusion.
- Sisk R, Sperrin M, Peek N, van Smeden M, Martin GP. *Stat Methods Med Res* 2023;32:1461–1477.
- Twala BETH, Jones MC, Hand DJ. *Pattern Recognit Lett* 2008;29:950–956.
- Van Ness M, Bosschieter TM, Halpin-Gregorio R, Udell M. The missing indicator method: from low to high dimensions. *KDD* 2023:5004–5015. (As cited in the model-family contract's sources.)
- scikit-learn documentation for `HistGradientBoosting*`, `RandomForest*` and `RidgeCV`.

**Families**
- Breiman L. Random forests. *Mach Learn* 2001;45:5–32.
- Liaw A, Wiener M. *R News* 2002;2(3):18–22.
- Wright MN, Ziegler A. ranger. *J Stat Softw* 2017;77(1).
- Gelman A. Scaling regression inputs by dividing by two standard deviations. *Stat Med* 2008;27:2865–2873.
- Hastie T, Tibshirani R, Friedman J. *ESL*, 2nd ed., §3.4.1. glmnet documentation (`standardize`, `makeX`).
- Huber PJ. *Ann Math Stat* 1964;35:73–101.
- Holland PW, Welsch RE. *Commun Stat Theory Methods* 1977;6:813–827.
- statsmodels `RLM` documentation (no weights; covariances H1–H3).

**Inference**
- Bach P, Schacht O, Chernozhukov V, Klaassen S, Spindler M. arXiv:2402.04674 (2024).
- Belloni A, Chernozhukov V, Hansen C. *Rev Econ Stud* 2014;81:608–650.
- Gelman A, Loken E. The garden of forking paths. 2013.

**Caveats.**
- Probst et al.'s spaces and tunability values were read from the arXiv HTML versions. The expert review packet (DoD §4) checks them against the published text.
- §4.6 paraphrases Cawley & Talbot 2010 and does not quote it; the packet checks the paraphrase.
- The regret bound and the expected maxima in §4.6 are this spec's derivations under a normal approximation, not citations. T8(d) checks them numerically.

---

## Rulings folded in, and what is left

**Every ruling since draft 2, and where it lands.**

| # | Ruling | Date and source | Where |
|---|---|---|---|
| 1 | The tree models try both with blanks by default (`choose: [native, fill]`), chosen in each training fold; "Fill in each fold" becomes "Keep every row". | Nolan, 2026-10-06 (HANDOFF; DoD 2026-10-07) | §2.2, §2.3, §2.4, §3.3(c), §4.2, §6.1 |
| 2 | A fit expected to take over about 2 minutes waits for Fit, pressed on the analysis flowchart. Under Estimate and Describe, pressing Fit records the lock, and the server withholds estimates until then. | Nolan, 2026-10-06; CROSSWALK "Settled here", 2026-10-08 | §4.4, §5, §10 |
| 3 | The three groups kept on 2026-10-07 are displaced to v2.x as package C6c; their seam is `TuningPlan.strategy`. | DoD, 2026-10-08 | §0, §4.2, §9, §10 |
| 4 | The kept comparisons stay v2.x, but a shared-step change after first results keeps a read-only earlier row, its trunk key and log sequence persisted; versions are keyed over the fields that differ from their defaults. | DoD, 2026-10-07; crosswalk ruling 6 and seam guards 4 and 5, 2026-10-08 | §3.2, §3.3, §3.7, §9 |
| 5 | The pre-fit ranking is live at model selection, on each family's actual input, outcome-blind (no outcome relation, no score; the outcome's counts allowed), and does not wait for the Confirm sweep; scales, batch and normalization come before the families question. | DoD and the model-family contract, 2026-10-08 | §2.5, §2.6 |
| 6 | Blanks are split at the stage line: who is kept in Who's in, how kept blanks are filled in each track's Models. | Crosswalk ruling 5, 2026-10-08 | §2.1, §2.3, §5 |
| 7 | "Revised after first results" appears only in the comparison and the methods text. A change after results is a kept version under Predict, a secondary under Estimate. | UI ruling, 2026-10-08 | §3.6, §3.7, §6.0, §6.3, §6.4 |
| 8 | A long fit runs as a server job with a notification. | UI ruling, 2026-10-08 | §4.4 |
| 9 | Seven stages; Decide · Confirm · For the record; plain words on every card. | Nolan, 2026-10-06/07; DoD 2026-10-07 | §2.1, §6.0 |
| 10 | Eight seam guards, three of them in this spec's packages. | DoD, 2026-10-08 (the orchestrator's call under the freeze's correctness rule) | §9 |

**Settled by the orchestrator in draft 3** (methods calls, each with its reason in place):
- **"Try both" below an effective size of 300** runs at the standard settings: the plan keeps the standard candidates, one per option, and drops the random ones (§4.6).
- **The standard settings are candidates once per "Try both" option at every size,** so the small-sample rule is one rule for every slot (§4.2).
- **One floor for the inner folds, set once in the plan:** at least 2 units of the rarest class, or 2 PSUs, per inner fold. Where even K = 2 cannot meet it, no fit draws inner folds, and the slot takes its first option, keeping blanks as blanks, stated (§4.2, §4.6).
- **Under a sealed holdout, a version changed after first results** is not fitted again, and its scores stay as a read-only row, so nothing seen is overwritten silently (§3.3).
- **Read-only rows** are every served score whose trunk decision key is not the current one, and, under a sealed holdout, every earlier version; they are never refit, corrected, declared final, scored on the held-out rows or exported, and a revert brings them back (§3.2).
- **"Keep every row" keeps one meaning in Who's in:** the exclusions card's option of the same name becomes "No restriction" (§2.3).
- **Fixing a "Try both" slot after its result was shown** is data-derived only when it fixes the option the search favored (the final refit's, or one that won at least half the outer folds). Fixing another option is a kept version, corrected by BBC-CV (§3.5).
- **Read-only status reads the trunk's answers, not its code.** An engine upgrade refits every version as current and never creates a "revised after first results" row (§3.2).
- **Describe's estimates wait for the lock** through seam guard 7, which settles INBOX's usual-intake item for Describe tracks (§4.4).

**For Nolan.** No new question. Four settlements change what a user sees, so they are flagged for his review:
- "Try both" on small tables (§4.6);
- read-only rows under a sealed holdout, which go beyond the UI ruling's "a kept version (Predict)"; reverting the change brings the earlier version back (§3.2, §3.3);
- "No restriction" on the exclusions card (§2.3);
- **the trees' blanks after first results.** With "Try both" as the default, fixing the trees' blanks to the option the folds kept is block and record: its score is labeled not honest and left out of the result, with "Keep trying both" as the exit, since that already makes the same choice honestly. Fixing the other option (for example the fill, because blanks will not mean the same where the model is used) keeps the earlier version and is corrected. Under a holdout sealed before any score, either fix is simply allowed (§3.5).

---

## What changed in draft 3

**Rulings.**
- **"Try both" is the tree models' blanks default,** declared through `default_choose`, with indicators on the fill option; "Keep every row" labels Who's in's option (§2.2–§2.4).
- **"Try both" on small tables is settled:** below 300 it runs at the standard settings, with five reasons, four sources, a derivation and a test that can overturn it (§4.6, T8(c–d)).
- **The standard settings are candidates once per option,** so the worked example tries 18 settings and "Standard settings" costs about 4 minutes when blanks are tried both ways (§4.2, §4.4).
- **Fit lives on the analysis flowchart.** Long fits wait for it and run as a server job with a notification; under Estimate and Describe, pressing it records the lock (§4.4, §5).
- **Package C6c is INBOX,** each item with its seam and its source (§10). `TuningPlan.strategy` owns the candidates and the fit count (§4.2).
- **Read-only rows** keep a shared-step change's earlier scores, from a versions file that carries trunk keys and served scores (§3.2). Version keys hash only the fields that differ from their defaults.
- **The shelf** is rewritten to the contract's C4: live, on each family's actual input, outcome-blind, not waiting for the Confirm sweep, with "Try both" profiled per option (§2.6).
- **Blanks split at the stage line:** the missing slot reads the track's fill (§2.3).
- **"Revised after first results"** labels the comparison's gray lines and the methods sentences, and nothing else (§3.6, §3.7, §6.0).
- **The quest log:** §6.0 places every element under Decide, Confirm or For the record, and checks the display-order rule.

**Also.**
- **Work packages** carry SIZING's C6a, C6b, P0.8 and C6d, with the three seam hooks and eleven guards inside them, and the contract's packages they meet (§9).
- **Tests:** T2, T4, T5, T6, T7, T8, T9, T10, T13, T15, T16 and T17 gain the rulings' cases; T8(c) joins the slow tests.
- **F16** records that earlier scores do not survive a cleared cache today.
- **The causal learner keys** that collide with families are renamed with aliases (§5).
- **Spelling repaired:** draft 2 printed "optimiztic", "optimizm" and "sensitivity analyzes"; they read "optimistic", "optimism" and "sensitivity analyses".
- **One label, one meaning:** the exclusions card's "Keep every row" becomes "No restriction", since the missing-values card now uses that label (§2.3).
- **"Decisions for Nolan"** became "Rulings folded in, and what is left".

---

## What changed after review of draft 3

**Majors.**
- **One rule for fixing a "Try both" slot after its result was shown.** Draft 3 called any such override data-derived, yet scripted the trees' switch to the fill as a kept, corrected version. Now only fixing the option the search favored (its final refit's, or one that won at least half the outer folds) is data-derived, with its coach line, label, sentence and the exit "Keep trying both"; fixing another option is a kept version. The worked after-first-results example and T7 move to ridge's scale, which no search covers, and T7(xiv–xv) script both trees' cases. The §6.3 line now depends on the slot (§3.3, §3.5, §3.6, §6.3, §7.2, §7.3, T16).
- **Read-only status reads the trunk's answers, not its code.** A stage key hashes stage versions, so an engine upgrade would have turned every earlier score into a false "revised after first results" row. The versions file now carries a trunk decision key (the answers the trunk reads, with no stage version and no split) beside the trunk stage key and the TurboTab version. An upgrade refits every version as current and creates no read-only row; T7(xvi) checks it, and T7(xii) checks the decision key on any engine version (§3.2, §7.2, §9, §10).

**Minors.**
- **One floor for inner folds,** set once in the plan: at least 2 units of the rarest class, or 2 PSUs, per inner fold; below K = 2, no fit draws inner folds. Fits that fall short keep the plan's K and are counted (§4.2, §4.3, §4.6, §7.2, T3, T9, T17).
- **§4.6's reasons are exact:** the fill option adds indicator columns, and the nested choice guards them; the inner-versus-outer bias is a stated caveat; the bound enters the claims ledger; T8(c) adds the calibration slope and 6 against 30 routed columns; the card says "rarely costs much".
- **The shelf's profile stops at the first step that reads the outcome's values,** with "before selection", "by rule" and the outcome-free fill under inference, and the `set_recipe` preview stops there too (§2.5, §2.6).
- **§6.0 gives the screen order,** and says where the engine's differs.
- **Predict's scores are stamped whenever served,** before or after Fit, and "Lighter" records a `set_tuning` (§3.1, §4.4, §6.2).
- **Plain words:** the refusal no longer says "estimand". Pointing elaborates in one line, and the preview waits for the click (§5, §6.1, §6.3).
- **§10 lists the contract's C3 and C6, the crosswalk's reorder, SIZING and V2X_SEAMS row 11 as amendments.**
- **Describe's no-estimate-before-lock guarantee names seam guard 7,** and T10 checks it (§4.4, §9).
- **Read-only rows of both kinds** (an earlier trunk; an earlier version under a sealed holdout) share one definition and one set of refusals, and a revert brings them back (§3.2).
- **The spec's dataclasses would run:** defaults after non-defaults, and `field(default_factory=dict)` for mappings (§2.4, §4.1).
- **Sizes:** RT-7 grows by 2 and RT-8 by 1 for draft 3's added scope, and "SIZING C4 (Who's in)" is named in full (§9).

## Review notes not taken

- **The stricter reading of §3.5,** in which every override of a slot a served search covered is data-derived, is not taken. Fixing the option the folds did not favor is not flattered by them, so labeling its score "not an honest estimate" would be a false label. The review's three fixes are applied on the narrower rule, and its flag for Nolan is reworded to match.
- **Read-only status from log order** (a stamped shared-step answer after the served sequence), the review's first route for the second major, is not taken. Its alternative, a trunk decision key, is: with it a reverted or re-answered shared step matches again, instead of leaving a duplicate read-only row beside the same trunk.
- **Withholding Predict's scores on the server until Fit,** the review's first route for that minor, is not taken. Fit under Predict stays a job command that records nothing, so the server would need a new marker; stamping every served score as shown keeps the record complete without one.
- **Refusing a fit that cannot meet the plan's K** is not taken. Such a fit keeps the plan's K and is counted, so no stage fails partway.

---

## What changed after review (draft 2)

**Blockers.**
- **Hand-copied tuned values were a leak.** Values set or held after a tuning result was shown now count as data-derived: block and record, excluded from BBC-CV and the result, unless a holdout was sealed before any score (§3.5, T16).
- **The sealed-holdout loophole.** The holdout status is stamped by the server. A holdout drawn after scores no longer counts as sealed, nor frees compared families (§3.3, T7(vii)).
- **Earlier versions now live in the log.** Stamps survive reverts, replay rebuilds every version, and the compared-families refusal bug is fixed (§3.2, T7(v, x)).
- **The seal is keyed by version.** The opened version stays final, and a later change runs as secondary (§3.3, T7(ix)).

**Statistics.**
- **One plan,** fixed from the outer training fold's effective size, now runs in every fit (§4.2, T17).
- **Thresholds count effective size** (events for classes), and stopping sets are stratified (§4.6).
- **NHANES inner splits** keep whole PSUs without strata, with the weighted loss, in both stages (§4.3, T3).
- **The correction claim is worded exactly.** The deployed version is stated, with a way to report another (§3.3–3.4).
- **Early stopping** draws its units first and fits nothing on them. The imbalance wrapper gets the same interface (F11, F12).
- **Path families** use one pooled, weighted path search; this also fixes F4 (§4.3).
- **Out-of-bag tuning** has strict conditions and 500 trees.
- **The forest's ranges are re-sourced**, and XGBoost's child weight is scaled by task.
- **Successive halving is dropped** (§4.2).
- **Huber is prediction-only**, with its estimand reworded.

**Engine.**
- **`fit_pipeline`** dispatches the search first, redraws nested splits, threads seeds and takes a design (§4.3).
- **The cost** counts every multiplier and stage (§4.4).
- **Long fits wait** in the scheduler; there is no stopwatch slot in the Record.
- **Version identity** carries `defaults_version`, with legacy mapping.
- **Replay** pins threads and keeps 1e-12.
- **Explanations** have defined paths for the forest and XGBoost.
- **Routing** is computed from each step's `reads`.
- **Contract scopes** are `model` where the outcome is read.
- **Work packages** are resized, with new ones for the imbalance wrapper, export and replay, and the catalog and journeys.

**Product and leash.**
- **Fits start only from the one primary action** on the Models card.
- **The missing card is relabeled.**
- **Native blanks** survive Explore's spline rule, and the line says when none reach the trees.
- **Every override** is one step away through "What each model is given", with copy written for every option.
- **The comparison** describes inputs against the trunk, flags only information differences, and collapses earlier versions.
- **"Try both" is no longer pushed** after scores.
- **Lines are grouped.**
- **One tuning line** covers every tuned family, with computed counts.
- **Plain words throughout:** "keep blanks as blanks"; "accounts for".
- **Silent tiers** follow the data.
- **Lighter appears just in time**, and Thorough is cut.
- **Unreachable leash rows** are relabeled "impossible by construction".
- **Manual fields** are pre-filled with the standard settings.
- **XGBoost and robust linear regression** are explained in plain words.
- **Colors** are assigned by family.

**Scope.** The approved core is marked, and ten items move to INBOX (§0, §10).

**Minors not taken.** The re-tuned band (INBOX), and a per-family replay tolerance (kept only as a contingency, §4.7).
