# Recipes and tuning

**Status: DRAFT 2, for Nolan's review.** The orchestrator wrote draft 1 on 2026-10-06, acting as methods expert, to answer Nolan's two questions of 2026-10-05. Draft 2, written the same day, takes in every blocker and major from three reviews (methods; product and leash; engine) and the cheap minors. The last section, "What changed after review", lists them.

**What it governs.**
- **BLUEPRINT:** North star 5, §4 (the stage graph), §11.3 (the leash), §11.4 (stated, asked and silent tiers) and §13 (the method contract and its relations).
- **MODELING_SEQUENCE:**
  - §0, rulings 3, 4, 6 and 13;
  - §1, rows 9–12 (row 10 is line 141, "Model-specific preprocessing");
  - the run order in §1.1, MS6 and the BBC-CV correction.
- **V2_DEFINITION_OF_DONE,** as amended on 2026-10-05 (branch `design/understanding`). That amendment:
  - brought in ridge, Huber, random forest and XGBoost;
  - required nested tuning for boosted trees;
  - required boosted trees' native handling of missing values to be reachable;
  - said overrides and tuning "are being specified". This is that specification.
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
1. **Every default is stated with its reason**, as a sentence whose phrase you can click. A default that changes nothing on your table is silent.
2. **Every override is recorded, and the comparison shows it.**
3. **A choice made after you have seen scores is disclosed.** Where the design allows, the earlier version also stays in the comparison, so the final score accounts for choosing among the versions you saw. Two limits are said plainly:
   - nothing corrects for how a later version was designed after the earlier scores were seen;
   - a change to the shared steps (the missing-values answer, Explore's levers, the selection step) is disclosed, not corrected.

   A holdout drawn before any score is seen removes both limits. Before the first fit, the app suggests one when you expect to try several versions.
4. **Unsound overrides get the leash, and leakage stays impossible by construction.** Values copied from a tuning result on these rows count as tuned on these rows (§3.5).

**A third path:** besides "keep the default" and "change it", there is **"Try both; keep what predicts better."** The alternatives join the tuning search inside each training fold, so the reported score already includes the choice.

There is no "Advanced mode" (BLUEPRINT §11.4).

### "Does TurboTab support hyperparameter optimization?"

**Today, partly.**
- The elastic net tunes its penalty and its lasso–ridge mix inside every training fold.
- Nothing else is tuned. Boosted trees run at scikit-learn's standard settings.

**After this spec, yes.**
- Every flexible family has a declared search space, with ranges from the literature.
- A seeded random search runs inside each outer training fold. Its inner splits keep each person's rows together, respect time, and keep survey PSUs whole, as the outer splits do.
- **One plan** is fixed before the first fit and used in every fit, so the procedure that is scored is the one that is deployed.
- The budget is known in advance, so the time is shown first. A fit longer than about 2 minutes waits for you to press Fit.
- After the fit you see the tuned values and how much they varied from fold to fold.
- You may set values by hand. They are labeled and disclosed.

**Under inference, tuning means almost nothing.** The reported models have no settings to tune. Their real choices are declared in the analysis plan (§5).

### What v2 builds now, and what waits

**The approved core, built here:**
- **Four new families:** ridge, Huber (as "robust linear regression", under prediction), random forest and XGBoost.
- **Nested tuning** for boosted trees, XGBoost and the random forest. The penalized families' penalty is searched by the same engine.
- **Boosted trees' native handling of blanks made reachable**, together with the other tree families'.
- **The overrides:**
  - the four safeguards;
  - "Try both; keep what predicts better";
  - values set by hand;
  - the estimate shown first.

**Moved to INBOX by this draft** (each is in §10):
- successive halving and other multi-fidelity search;
- the "Thorough" budget, and the tuning curve;
- "Compare with standard settings" as a kept version;
- native categories;
- per-family Pareto and robust scaling (the omics normalization keeps them);
- the per-model log1p transform;
- Huber under inference;
- nuisance learners for DML and TMLE built from the family registry, and their tuning;
- re-tuned substitution bands;
- changes to the shared steps after scores kept as versions.

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
| F11 | **New.** The early-stopping branch fits the steps before the model on every row, the stopping rows included, and only then splits off `X_val`. | `inner_cv.fit_pipeline` | When a step reads the outcome (selection, screen, inner-CV forms), stopping is optimiztic. |
| F12 | **New.** `ImbalanceCorrected` has no `early_stopping` parameter. The wrapped HistGB therefore draws its own stopping split, by position and after resampling, and the wrapper's recalibration `cv` holds position splits for the outer fit's rows. | `methods/levers.py`, `inner_cv._stops_early` | Under oversampling, copies of one row sit on both sides of the stopping split. |
| F13 | **New.** Thresholds are read from each fit's own row count: `EARLY_STOPPING_ROWS`, and the elastic net's `inner_folds(n_rows)`. | `inner_cv.py`, `elastic_net.py` | Outer folds and the final refit can run different procedures near a threshold. |
| F14 | **New.** Scores seen, compared families, `OpenSeal.family`, `AtOpening.scores` and `mark_final` are keyed by family. `compared_families` returns nothing once any holdout exists, even one drawn after scores were seen. | `selection.py`, `decisions.py`, `seal.py` | Versions within a family cannot be tracked, and a late holdout frees compared families. |
| F15 | **New.** `fit_pipeline` takes groups, order and a seed: no strata, PSUs or weights. No stage passes a seed. | `inner_cv.py`, `stages/*` | Inner splits cannot follow a design, and every inner draw uses seed 0. |

F10 is why the design below stays small. F11–F15 must be fixed before the search is built on them.

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
- the missing-values answer;
- batch correction and scale scores;
- the energy model and the declared forms;
- blanks as a level of their own, and one-hot encoding;
- Explore's levers, and the selection step.

**Each family adds or replaces only what it declares.** That is its *recipe*. A recipe has five slots, and a family declares only the slots that can change something for it:

| Slot | What it decides |
|---|---|
| `missing` | whether the family takes the trunk's fill, or the blanks themselves |
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

**Tiers.**
- **Stated:** the default appears as a sentence with a clickable phrase.
- **Silent:** shown in the export only. A slot is silent when its option's footprint on this table is empty, as measured by the consequence planner. Examples:
  - a missing slot on a table with no blanks;
  - "every level kept" with no categorical predictor;
  - any slot whose options change no number, such as scaling for the unpenalized linear model.
- **Asked:** none.

**The amendment to row 10.** "Not a question" becomes: "Stated per family, with its reason, and silent where it changes nothing. Under prediction its phrase can be changed with `set_recipe`, under the four safeguards (§0). Under inference the plan decides (§5)."

**Families outside this spec** take the shared trunk and have nothing to tune: `featurewise`, `proportional_odds`, `mixed`, `gee` and `cox`.

### 2.2 Every family's recipe

**Shared fill** is the trunk's in-fold fill, learned within each training fold without the outcome:
- the median for numbers;
- the line on total energy for energy-bearing nutrients (`EnergyAwareImputer`);
- the most frequent value for categories and two-valued numbers;
- a missing indicator per filled column, when the missing-values answer asked for one.

**Defaults** (bold marks a departure from the trunk; "option" means available under prediction):

| Slot | Linear | Robust linear (Huber) | Ridge, elastic net | Random forest, boosted trees, XGBoost |
|---|---|---|---|---|
| missing | shared fill | shared fill | shared fill | **keep blanks as blanks** (§2.3). Options: the shared fill; Try both |
| scale | silent: changes no prediction | silent: its fit rescales itself | **every column on one scale**. Options: Gelman's two-SD scaling; columns in their units (ranked lower) | not declared: no split moves |
| encoding | first level as reference | first level as reference | **every level its own column**. Option: reference (ranked lower) | **every level its own column**, with no option |
| transform | none. Option: Yeo–Johnson (ranked lower); Try both | as linear | as linear | not declared |
| outliers | none. Option: cap at the 1st and 99th percentiles (ranked lower); Try both | none: the model is the answer | as linear | not declared |
| settings | nothing to tune | threshold fixed at 1.345; may be set by hand | penalty, and for the elastic net the mix, searched along a path (§4.1) | searched (§4.1) |

**Reasons and labels** (C means customary, with its source; S means sound for prediction, with its reason):

- **Shared fill.**
  - Reason: "A straight line needs a number in every cell; the fill is learned on training rows only."
  - C: median fill. S: it can be deployed and never sees the outcome (Sisk et al. 2023).
- **Keep blanks as blanks.**
  - Reason: "Each split learns which way a blank goes, so a blank can carry meaning."
  - C: fill first.
  - S: Josse et al. 2024 (consistency); Perez-Lebel et al. 2022: "Native support for missing values in supervised machine learning predicts better than state-of-the-art imputation with much less computational cost."
  - **Caveat, stated in the sound label:** it relies on a blank meaning the same thing where the model is used. Sisk et al. 2023 found that missing indicators can harm prediction under outcome-dependent missingness, and routing uses the blank in the same way.
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
- **Not offered under inference** in v2 (§5).

**Random forest (`random_forest`). New.**
- **Inductive bias:** "Many deep trees, each on resampled rows with random columns, averaged; effects are step functions."
- **Standard settings:**
  - 500 trees;
  - columns tried per split: √p for classes, p/3 for numbers;
  - smallest leaf: 10 rows for probabilities (ranger's probability-forest default) and 5 for numbers;
  - every row drawn, as a bootstrap.
- **Departure stated:** scikit-learn's regressor uses every column at every split (`max_features=1.0`), which is bagging.
- **Sources:** randomForest (Liaw & Wiener 2002); ranger (Wright & Ziegler 2017); Probst, Wright & Boulesteix 2019.

**Boosted trees (`boosted_trees`, histogram gradient boosting). Exists, amended.**
- It is tuned (§4.1). Candidate 0 is scikit-learn's standard settings.
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

**The shelf** (prediction scores; registration order breaks ties):

| Family | Score |
|---|---|
| Ridge | 2.0. At p ≥ n, 3.0 (below the elastic net's 4.0). |
| Robust linear | 1.5 (numeric only; after linear on ties). |
| Random forest | 2.0 from 2,000 rows; 1.5 from 500; "fair" at 1.0 below 500. When predictors outnumber rows, "fair" at 1.5. Under inference, as boosted trees. |
| XGBoost | Boosted trees' score less 1.0 (at least 0.5): the same kind of model, offered by name. |

At 2,000 rows or more the order is therefore: boosted trees 3.0, elastic net 2.5, then the forest, ridge and XGBoost at 2.0. The two families the reference journeys pick today (`first_models`) stay first. RT-13 checks every journey's pick before and after (§9).

**Registration order:** `linear`, `elastic_net`, `ridge`, `huber`, `boosted_trees`, `random_forest`, `xgboost`.

**Tasks.**
- Ridge, the random forest and XGBoost take numeric, yes/no and multiclass outcomes. An ordered outcome is fit as unordered classes, with the existing order-blind cost.
- Robust linear regression takes numeric outcomes only.
- Time to event waits for v2.x.

### 2.3 Missing values: the answer and the recipe, by purpose (the dead path, fixed)

**The rule.**
- **The missing-values answer** decides which rows enter, and which fill the trunk learns.
- **The recipe** decides which families take that fill.
- **Under prediction, with the rows kept, a tree family takes the blanks themselves** in every numeric predictor that no blank-intolerant step reads.

**Routing is computed, not flagged.** Each step declares:
- `reads(spec) -> columns`: the columns whose values its output needs;
- `passes_blanks`: whether a blank in a column it does not read passes through unchanged.

`family_spec` walks the family's step list in order and computes `native`: the columns no blank-intolerant step reads. The imputer and the indicators then skip those columns.
- **Pass blanks through:** the log step, the levels and one-hot steps, the impute `ColumnTransformer` (for columns it does not fill), the scaler and the variance filter.
- **Blank-intolerant on the columns they read:** the energy adjustment, the omics normalization, batch correction, scale scoring, a declared form and the selection step.

**Two consequences, stated.**
- **Tree families skip Explore's spline rule.** The basis adds only monotone copies of a column, which move no split, and that keeps the trees' columns routable. This is consistent with their transform slot not being declared.
- **The selection step reads every input.** When it is on, no blank reaches the trees, and the recipe line says so: "No blanks reach the trees: the selection step needs every value."

| Missing answer | Purpose | Linear, ridge, elastic net, robust linear | Tree families |
|---|---|---|---|
| Complete cases | both | Rows with a blank predictor leave. | The same rows. The slot is silent (no blanks remain). |
| Keep every row (`impute`) | prediction | The shared fill, with indicators if asked for. | **Default:** blanks kept in the routed columns; columns that a blank-intolerant step reads are filled first. No missing indicator is added for a routed column, and the sentence says so ("the blank is its own signal"). **Option:** the shared fill, exactly as the other models get it. Under "Try both", the fill option carries indicators for those columns, so the contest is fair (Perez-Lebel et al.: "When using imputation, it is important to add indicator columns"). |
| Keep every row, single fill | inference | The shared fill for scores; the coefficient table follows the plan. | The shared fill. Keeping blanks is not offered. |
| Multiple imputation | inference | Each completed copy has no blanks. | The same. |
| Multiple imputation | prediction | Refused today (`decisions.py:4460`). | Refused. |
| Blanks as their own level | both | Categorical and two-valued columns get a `Missing` level. | The same. Routing covers numeric columns only. |

**What a deployed tree does with a blank.** If a column had no blanks in a training fold, a later blank goes:
- for HistGB and the forest, to the child with more samples (scikit-learn's documented rule);
- for XGBoost, to the library's default branch.

The lineage note says which.

**Explanations.**
- TreeSHAP follows each split's learned direction for blanks.
- Effect curves are drawn over observed values, and the caption counts the rows whose value was blank.
- Substitution curves are unchanged: energy sources are always filled.

**The missing-values card**, under prediction:
- The `impute` option is **relabeled "Keep every row"**, because tree models no longer fill.
- Its consequence becomes "Blanks are filled in each training fold; tree models may keep them as blanks." (14 words)
- The inference copy is unchanged.
- The comment at `teaching/content.py:1122` is corrected.

### 2.4 How a recipe is declared (`base.py`)

```python
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
    default: Mapping[str, str]    # purpose -> option key
    reason: str                   # ≤ 22 words
    options: tuple[RecipeOption, ...]
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
- the word budgets;
- a family may declare `native` only when its estimator accepts NaN (checked by a fit on a three-row frame with one blank).

**The recipes travel inside the spec.** `DesignSpec` gains `recipes: Mapping[family, RecipeSpec]`. `build_pipeline`, `family_steps` and `describe_steps` apply `family_spec(spec, family, purpose)` themselves, so none of the roughly nine call sites can forget it. `family_spec` resolves these fields:

| Field | What it holds |
|---|---|
| `native` | the routed columns; the imputer's and the indicators' lists exclude them |
| `onehot_drop` | `"first"` or `None`, read by both one-hot paths (`MissingLevelEncoder` gains `drop`) |
| `scaler` | `standard`, `two_sd` or `none`: one `PenaltyScaler` exposing `center_` and `scale_` per column (1 and 0 where a column is not scaled), so `elastic_net.coefficients`, `explain._undo_scaling` and `linear_equation` undo it as they do today |
| `transform`, `outliers` | an in-fold step, after Explore's steps and before scaling |
| `skip_form_rule` | true for tree families |

`GET /api/models` (`FamilyInfo`) gains `recipe` and `tuning`.

### 2.5 How a recipe is drawn

- **Lineage.** The trunk is drawn once. A family that departs gets a short spur where it departs, grouped by shared departure: "blanks kept · boosted trees, random forest, XGBoost". Pointing at a family lights its path in the choice color.
- **Model-matrix preview.** `previews.models_preview` compares each family's `family_spec` with the shared spec, not step names, because routing changes no step name. Its caption names the difference in 20 words or fewer: "Boosted trees: 42 columns; 1,204 blanks in 6 of them reach the trees." `table_focus` shows five rows of the touched columns, with blank cells reading "blank".
- **`DesignModel`** gains:
  - `recipe: list[RecipeLine]` (slot, option, label, term, reason, changed, when, silent);
  - `variant` (§3.2);
  - `plan` (§4.2).

---

## 3 · Overrides and versions

### 3.1 The decisions

```python
class VariantSpec(_Value):            # a version's identity: every resolved choice
    family: str
    recipe: dict[RecipeSlotName, str] # every declared slot's option, defaults included
    choose: dict[RecipeSlotName, list[str]] = {}
    tuning: Literal["automatic", "lighter", "standard", "manual"]
    values: dict[str, float | int | str] = {}
    defaults_version: str
    space_version: str | None

class Stamp(_Value):                  # filled by the server's completion, never by a client
    target: str | None
    shown: list[VariantSpec]          # every version whose scores were served for this outcome
    tuning_shown: list[str]           # versions whose tuned values or fold shares were served
    holdout: Literal["none", "sealed", "after_scores", "opened"]   # §3.3

class SetRecipe(_DecisionModel):
    kind: Literal["set_recipe"] = "set_recipe"
    family: str
    overrides: dict[RecipeSlotName, str] = {}      # {} = back to the stated defaults
    choose: dict[RecipeSlotName, list[str]] = {}   # "Try both; keep what predicts better"
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
- `set_recipe` and `set_tuning` write slots keyed by family (`register_kind(..., key=lambda d: d.family)`).
- `declare_version` writes one slot per outcome.

**Stamps.**
- Completions fill `stamp` on `select_models`, `set_split`, `set_recipe`, `set_tuning`, `declare_version`, `open_seal` and `revert`. Revert gains a completion.
- **Key views** (`graph.register_key_view`) drop `stamp`, `note` and `source` from every stage key, so a stamp never triggers a refit.

**Validators.**
- The family must be selected.
- Each option must exist and be offered for the purpose; otherwise the server answers 409 with the exits in §7.3.
- `choose` names at least two options of a searchable slot.
- `values` keys must be among the family's declared settings (§4.1). So a class weight, `scale_pos_weight` or XGBoost's linear booster cannot be expressed.
- Values must pass the estimator's own validation. A value outside the declared range adds a concern, not a refusal.
- Under inference, `set_recipe` is refused (§5).

**Previews.** `register_consequence` covers all three kinds: §2.5's views, §6.2's views, and the estimate (§4.4).

### 3.2 Versions

**A version is a family together with its resolved recipe and its tuning choice.**
- **Its key:** `family~` plus 8 hex digits of the sha256 of the canonical JSON of its `VariantSpec`.
- **Display** never shows the key: "Boosted trees", "Boosted trees, earlier version".
- **`defaults_version`** sits inside the identity, so the same words can never name two procedures.
- **Pre-spec records** (`scores_seen.json` listing bare family keys) map to `LEGACY_DEFAULTS`, the resolved version 1 of each existing family (today's fill and standard settings). A version seen under the old defaults therefore stays in the comparison as an earlier version. It is never silently renamed. When this happens the Models card says so once: "Boosted trees now keep blanks as blanks and are tuned; the earlier version stays in the comparison."

**The versions shown are in the log, and a revert cannot unsee them.**
- `scores_seen.json` gains, per version: its `VariantSpec`, the log sequence number at which its scores were first served, and whether its tuning result was served. The file stays as a fast mirror. The log is the source of truth.
- Every stamp copies the versions shown so far for its outcome.
- The fold gains one rule: the slot `versions_shown` collects stamps from **every** record, reverted records included.
- A revert of `set_recipe` is therefore allowed. The changed version, once shown, simply becomes an earlier version.
- Replay rebuilds every earlier version from `decisions.jsonl` alone.

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

### 3.3 The three paths

**(a) A choice declared before any score is seen.**
- Its stamp shows no versions.
- It is the family's only version. Nothing was selected, so there is nothing to correct.
- When it is the only family fitted, its own score may be the result, as `declared_result` allows today.

**(b) A choice made after scores are seen.** Under prediction it is always allowed and always disclosed. What happens next depends on the holdout status the server stamps:

| Holdout status | When | A change made after scores |
|---|---|---|
| `none` | no rows held out | The earlier version stays as a gray row. The result is the selection-corrected estimate over every version shown. |
| `sealed` | a holdout drawn **before** the first score for this outcome was served (the split's record precedes the first served sequence) | No earlier version is kept: the final model is declared on cross-validation before the held-out rows open, so the held-out score is untouched by the change ("the fundamental 'untouched test set' principle", Bischl et al. 2023). |
| `after_scores` | a holdout drawn after a score was served | Treated as `none`: the earlier version is kept, and compared families stay. The held-out rows were in the scores that drove the change, so their score is labeled "these rows were in cross-validated scores seen before they were held out" and reported beside the selection-corrected estimate. Drawing such a holdout is block and record; its preview says why. |
| `opened` | the seal was opened | Block and record. The change runs as a secondary analysis, and the version declared at the opening stays `final` (`AtOpening` and `mark_final` match on the version). |

**What path (b) corrects, and what it does not.**
- **Corrected:** choosing among the versions shown. BBC-CV covers the final pick among fixed versions (Tsamardinos et al. 2018).
- **Not corrected, and said:** how a later version was designed after the earlier ones' scores were seen (adaptive data analysis).
- **Disclosed under ruling 3:** changes to the shared trunk after scores, which create no version (the missing-values answer, Explore's levers, the selection step, the energy model, the forms). These are listed in the TRIPOD+AI model-building item, beside every version changed after scores.
- **Prevention:** before the first fit with no holdout, the Models card states once: "Every version whose scores you see stays in the comparison. Hold out rows now if you expect to try several." (20 words) It appears only when a holdout is offered without concern at this size.

**(c) Try both; keep what predicts better.**
- `SetRecipe.choose` puts the named options into the family's search as a categorical dimension, mapped from one Sobol coordinate.
- Each outer training fold picks its own option, so the outer score belongs to a procedure that chooses.
- **Shown:** how often each option won across outer folds, and the option the final refit chose (the deployed one).
- **Rung under prediction:** available, before or after scores, with its minutes shown. It is not labeled Recommended after scores: with no holdout the earlier version is kept either way, so trying both adds cost and no extra correction.

### 3.4 Which version is reported and used

- **A holdout drawn (any status):** the version named at the opening (`open_seal.variant`).
- **No holdout:** BBC-CV's choice among every version in the comparison. `declared_result` already declares `selection["best"]`.
- **When BBC-CV's choice is an earlier version,** the Models card says: "The earlier version scores best, so it is the model reported and used." (13 words) Two exits follow:
  - **"Use the earlier version"**: "Makes it current again, so every view describes the model reported." This records a `set_recipe` or `set_tuning` restoring it.
  - **"Report the current version instead"**: "Its own score is reported, labeled as chosen after the scores were seen." This records `declare_version`. The result is that version's own cross-validated score, labeled "chosen after the scores were seen", with the selection-corrected estimate beside it (an amendment to row 12(b), §10).

  Until one exit is taken, every view other than the comparison describes the current version, and its caption says so.

### 3.5 Values from tuning results (the quiet leak)

**When a value counts as data-derived.** The final refit's search chooses its values on all training rows, every outer test fold included. A value is therefore *data-derived* when it is recorded after any tuning result for this outcome was served (`stamp.tuning_shown` is not empty), and it is one of:
- a manual value;
- a value held under `automatic`;
- a `set_recipe` override of a slot that a served "Try both" search covered.

The server decides this; the client cannot claim otherwise.

**What happens, by holdout status:**
- **`none` or `after_scores`: block and record.**
  - The version is fitted and shown, labeled "settings chosen on these rows: not an honest estimate".
  - It is excluded from BBC-CV and from the declared result.
  - The coach line: "These values come from tuning on these rows, so this score would be too kind." (15 words)
  - The exits: "Tune inside each fold", and, for a recipe slot, "Try both; keep what predicts better".
- **`sealed`:** available. The holdout absorbs it.
- **`opened`:** a secondary analysis, as any change.

This is the reachable form of "tuning outside the resampling". Bischl et al. 2023 warn that "the more we tune, the smaller our data set, … the more expressed this optimiztic bias will be".

### 3.6 Sentences (the methods register, `voice.register_sentence`)

| Path | Sentence |
|---|---|
| Stated default | "Boosted trees received blanks in 6 columns as they were and learned at each split which way they go (missing incorporated in attributes; Twala et al. 2008); no missing indicator was added for those columns. The other models used the fill learned in each training fold." |
| Before any score | "Boosted trees were given the fill used by the other models instead of keeping blanks, declared before any score was seen: ⟨note⟩." |
| After scores, no holdout | "After the cross-validated scores of the linear model and boosted trees were seen, boosted trees were changed to use the shared fill. The earlier version was kept in the comparison, and the result is corrected for choosing among the 3 versions shown (BBC-CV); it is not corrected for how the later version was designed after the earlier scores were seen." |
| Holdout sealed | "…; the held-out rows were sealed before any score was seen, so the held-out score does not depend on this change." |
| Holdout after scores | "…; the held-out rows were set aside after cross-validated scores on them were seen, so the earlier version was kept and their score is labeled accordingly." |
| Seal opened | "…after the held-out rows were opened; it is a secondary analysis, and the declared result is unchanged." |
| Try both | "Whether boosted trees kept blanks or used the shared fill was chosen inside each training fold, so the reported score includes the choice (nested cross-validation); the folds kept blanks in 41 of 50 folds, and the final model keeps them." |
| Automatic | "Boosted trees' settings were tuned inside each training fold by a seeded random search over 17 settings (the standard settings and 16 quasi-random ones), scored by squared error on 3 inner folds kept by participant; the plan was set once, for training folds of about 13,600 participants, and used in every fit, so the reported scores include the tuning (nested cross-validation)." |
| Standard | "Boosted trees used the standard settings of scikit-learn ⟨version⟩, untuned." |
| Manual | "Boosted trees used a learning rate of 0.05 and 63 leaves per tree, set by hand before any score was seen (source: Smith et al. 2024); their other settings were standard." |
| Data-derived | "Boosted trees' settings were set by hand to values chosen by tuning on these rows; that version's cross-validated score is not an honest estimate and is excluded from the result." |
| Declared version | "The current version of boosted trees was declared after the scores were seen; its own cross-validated score is reported, beside the selection-corrected estimate over the 3 versions shown." |

### 3.7 The comparison table

| Column | Header | What it holds |
|---|---|---|
| Model | Model | the family's name; its technical name as the quiet term |
| Inputs | What it was given | described against the trunk: "shared inputs"; "shared inputs, blanks kept as blanks"; "shared inputs; changed by you: blanks filled"; "shared inputs; the folds chose: blanks kept in 41 of 50" |
| Settings | Settings | "tuned in each fold"; "standard"; "set by hand"; "set by hand from tuning on these rows"; "nothing to tune" |
| Score | the primary score in words ("Expected squared error on new people") | estimate and interval, as today |

**The scaling and encoding a family's own definition requires are part of the model.** The penalized families' standardization and full coding are stated in their recipe line, not counted as an input difference.

**The line under the table** appears only when a difference changes the *information* a model sees: blanks kept versus filled, a transform, a cap, or any change by the researcher. It reads: "These models were given different information, so each score is for the inputs and the model together." (17 words) The paired-difference sentences then compare whole procedures, and say so.

**Earlier versions** collapse under their family's current row as one gray line: "2 earlier versions, kept so the final score accounts for choosing among them." That line opens on a click.

---

## 4 · Tuning

### 4.1 The search spaces

```python
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
    standard: Mapping[str, Any] = {}         # candidate 0
    standard_source: str = ""                # "scikit-learn's defaults", "ranger's defaults"
    fixed: Mapping[str, Any] = {}            # stated, never searched
    early_stopping: Mapping[str, Any] | None = None
    out_of_bag: bool = False
    space_version: str = ""                  # "boosted_trees/1"
    reason: str = ""                         # ≤ 22 words
```

| Family | Kind | Searched (range, scale) | Fixed | Standard settings (candidate 0) | Sources |
|---|---|---|---|---|---|
| Linear | none | | | | |
| Robust linear | none | (by hand: threshold t in [1, 3]) | MAD scale; no penalty | t = 1.345 | Huber 1964; Holland & Welsch 1977 |
| Ridge | path | Penalty per row λ, on scaled columns: 50 points from 10⁻⁵ to 10², log. Classes: logistic with l2, C = 1/(n·λ) on the same grid. | | | Probst, Boulesteix & Bischl 2019: of the six learners they studied, glmnet gained more from tuning than xgboost or ranger (mean AUC tunability 0.069, 0.043 and 0.010 against package defaults). |
| Elastic net | path | Mix (0.1, 0.5, 0.7, 0.9, 0.95, 1) × 100 ratios r from 10⁻³ to 1 of each split's own λ_max. Logistic: mix (0.2, 0.6, 1) × 8 ratios. | The pure-ridge end belongs to ridge. | | glmnet's λ path; Probst et al. 2019 (JMLR) |
| Random forest | search | Share of columns per split [0.05, 1], linear. Share of rows per tree [0.2, 1.0], drawn with replacement (adapted from tuneRanger's [0.2, 0.9] without replacement, since scikit-learn subsamples only with replacement). Smallest leaf as a share of the fit's units, log scale, from 1 row to a tenth of them. | 500 trees in every candidate | §2.2 | Probst, Wright & Boulesteix 2019; tuneRanger; Probst et al. 2019 (JMLR: n^x, x ∈ [0, 1]); Kruppa et al. 2014 (leaves of about 10% of n for probability machines) |
| Boosted trees | search | Learning rate [0.01, 0.3], log. Leaves per tree [4, 128], log integer. Smallest leaf [2, 200], log integer, capped at a twentieth of the plan's units. L2 pull [10⁻³, 10], log. Share of columns per split [0.3, 1]. Number of trees [25, 500], log integer, only when early stopping is off. | Early stopping by the plan (§4.2): up to 1,000 trees, patience 20 | scikit-learn's defaults: learning rate 0.1, 31 leaves, leaf 20, no L2, all columns, 100 trees; its own early stopping above 10,000 training rows, decided once by the plan's rows | Probst et al. 2019 (JMLR); scikit-learn's HistGB documentation |
| XGBoost | search | Learning rate [0.01, 0.3], log. Depth [2, 10], integer. Least child weight as rows-equivalent [1, 64], log, times the mean hessian at the base score (1 for squared error; p̄(1−p̄) for yes/no; the mean of p̄ₖ(1−p̄ₖ) for classes). Row share [0.5, 1]. Column share [0.3, 1]. L2 pull λ [10⁻³, 100], log. Rounds [25, 1,000], log integer, only when early stopping is off. | Early stopping by the plan: up to 2,000 rounds, patience 50. Tree booster; α = 0; `tree_method="hist"`; threads recorded. | XGBoost's defaults: learning rate 0.3, depth 6, child weight 1, every row and column, λ = 1, 100 rounds | XGBoost's tuning notes; Probst et al. 2019 (JMLR: λ 2^[−10, 10]; child weight 2^[0, 7]) |

**Why the least child weight is scaled.** It is a sum of hessians. At 5% prevalence, an unscaled 64 would demand about 1,350 rows per leaf.

**Candidate 0 is always the standard settings.**

**The ranges are conventions,** narrowed from the sources above. The methods reference states each range with its source, and the prediction reviewer's packet asks for them to be checked.

### 4.2 The plan, fixed once

**The algorithm: a seeded quasi-random search.** It is plain random search over a scrambled Sobol sample, with no successive halving and no TPE. The reasons, from strongest to weakest:
1. **Replay.** The candidate list is fixed before any score is seen, and the result does not depend on the order in which fits finish. Optuna's FAQ, by contrast: "We recommend executing optimization of a study sequentially if you would like to reproduce the result."
2. **The budget is known before the run,** so the estimate is exact in fits (DoD gate 5).
3. **The spaces have low effective dimension.** There, random search is "a surprisingly strong baseline" (Bischl et al. 2023; Bergstra & Bengio 2012).
4. **Halving was dropped.** At most nutrition sizes it would engage for one round or none. Where it does engage, count-type settings mean different things at a ninth of the rows: the better configuration's "superiority … was only observable after full evaluation" (Bischl et al. 2023). Halving, Hyperband, TPE and BOHB go to INBOX.

**The plan.** It is a `TuningPlan`, computed once in the design stage, recorded and stated:

| Element | Rule |
|---|---|
| Effective size n_eff | Numeric outcome: units. Yes/no or classes: the units in the rarest class, each unit counted by its most common class, as the stratified splits count it. Riley et al. 2021 state the concern in terms of a "small effective sample size". |
| Plan size n_plan | n_eff of one outer training fold of the headline split: (K−1)/K of the training units, or the median training fold when the folds follow time. **Every fit uses this plan unchanged:** every outer fold of every repeat, bootstrap resamples, the folds of Bates et al.'s nested cross-validation, internal–external refits, evaluation's design-based folds, and the final refit. Only settings defined as shares of a fit's units rescale. This fixes F13 for tuned families. The same rule fixes K for path families and the early-stopping switch for candidate 0. |
| Candidates C | n_plan below 300: standard settings only (§4.6). 300–999: 1 + 8. 1,000 and up: 1 + 16. The forest uses 1 + 8 from 300: it is the least tunable. "Lighter" halves the Sobol sample. |
| Inner folds K | Searched families: 3 (a convention; every candidate meets the same splits). Path families: `inner_folds(n_plan)`, which is 5 for 100 to 5,000 and otherwise 3. |
| Early stopping | On in every candidate fit when n_plan ≥ 1,500, so that the stopping set inside an inner fit holds about 100 or more units of the rarest class. Below that, the number of trees is searched. |
| Stopping set | A tenth of the fit's units: whole units, stratified by class, the latest ones when the folds follow time. It is drawn before anything else, and no step is fitted on it (§4.3). |
| Score | The comparison's strictly proper primary as a pooled per-row loss over the inner validation rows (`validation.loss_rows`: squared error, log loss or ranked probability score). It is survey-weighted under the population answer, and rounded to 1e-9 relative before the argmin. |
| Choice | The lowest pooled loss. Ties go to the lower index, so candidate 0 first. |
| Refit | The chosen candidate on all of the fit's rows, through `fit_pipeline`. |
| Forest, out of bag | Used only when every row is its own unit, the folds do not follow time, the sample answer applies, no step before the model reads the outcome, and there is no imbalance correction. Each candidate is then fit once with 500 trees and scored on its out-of-bag predictions (tuneRanger's method). Otherwise inner folds are used. |

**Cost of one outer fit,** in fits on that fit's rows:
- **Searched:** F = w·[(K − 1)·C + 1].
- **Out of bag:** F = C + 1.
- **Path:** F = w·[(K − 1)·r + 1] path fits, where r is the number of recipe options under "Try both" (1 otherwise).
- **w** is 5 with the imbalance correction (its recalibration refits: 1 + 5 folds × 4/5), and 1 otherwise.

**Worked example.**
- 17,000 training participants, a numeric outcome, comparison folds 10 × 5.
- n_plan = 13,600, so C = 17, K = 3, and early stopping is on.
- F = 2 × 17 + 1 = 35 fits per outer fit.
- With 51 outer fits (50 comparison folds and the final refit), that is 1,785 fits.

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

**One per-split helper, `fit_parts`, shared with the plain path of `fit_pipeline`.** For the rows it is given, it:
1. draws the stopping units first;
2. fits the steps before the model on the remaining rows, and transforms `X_val` with them (fixes F11);
3. draws every nested split those rows need: each step's `cv`, and the imbalance correction's recalibration `cv`;
4. fits the model.

Inside a search, the helper fits one head per inner split per recipe option, and every candidate in that split shares it. Nothing in the inner loop learns from its own validation or stopping rows.

**Inner splits are drawn as the outer ones are:**
- whole units, when rows repeat;
- forward chaining by whole unit, when the folds follow time;
- **under the population answer, whole PSUs:** `GroupKFold` over stratum × PSU labels, not stratified within strata. An NHANES training fold of a design-based fold holds one PSU per stratum, about 15 PSUs. Where a fit holds fewer than K PSUs, K drops to the PSU count (at least 2); below that the plan uses standard settings, stated. The sentence says: "inner splits keep whole PSUs; with two PSUs per stratum, strata cannot be kept in every inner fold". This holds in the fit stage, in evaluation and in the final refit alike, so one procedure is scored and deployed. Models are fit unweighted, as the comparison fits them; only the inner loss is weighted;
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
- Substitution bands and explanation reseeds describe the deployed model, so a searched family's refits hold the final fit's values through `at(chosen)`.
- Their caption says "conditional on its tuned settings". A re-tuned band goes to INBOX.
- Path families re-tune, as today.
- Validation never pins.

### 4.4 The estimate, and when a fit starts

**The estimate.**
- `cost.py` times `at(center)` once: every dimension at the midpoint of its scale.
- `ShelfFamily` gains `per_fit_seconds` and `outer_fits`.
- `outer_fits` counts every fit the stages will make:
  - the comparison folds and the final refit;
  - bootstrap refits, only for families that are bootstrapped (fixes F7);
  - evaluation's design-based folds, under the population answer;
  - internal–external refits;
  - sensitivity analyzes that refit;
  - the nested cross-validation interval, only when offered or asked (its offer label shows its own minutes);
  - pinned refits, which count as plain fits.
- The estimate is `per_fit_seconds × Σ F(plan)`, said as "about 30 minutes, most of it tuning". F is recomputed wherever the plan is shown, without re-timing.

**DoD gate 5: a fit expected to take over 30 seconds shows its estimate first.** The estimate appears:
- on the shelf;
- in the preview of every decision whose recording would start such fits: `select_models`, `set_recipe`, `set_tuning`, and upstream answers (the consequence planner knows which stages refit);
- on the primary action.

**A fit over 2 minutes (a convention) does not start on its own.**
- The hold is in the scheduler, not a stage requirement, so stages stay pure functions of the log and replay bypasses it.
- The fit stage waits as `stale`, with "Fit · about 30 min" on the card and in the banner.
- Pressing the button is a job command, not a decision, so nothing enters the Record.
- The hold re-arms when a new estimate exceeds 1.5 times the last one confirmed.
- This amends BLUEPRINT §4 (§10).

**Choices made on the Models card** (families, recipe phrases, tuning) are held in the card's local state, as family choices are today (`ChoiceQuestions.tsx:533`). The one primary action records them together. **No asked slot exists.**

**Illustration** (rough): the worked example, at about 1 second per fit.

| Plan | Estimate |
|---|---|
| Automatic (17 settings) | about 30 minutes |
| Lighter (9 settings) | about 16 minutes |
| Standard settings | about 1 minute |

Under the population answer, evaluation's design-based folds (10 × 2) add about 20 × 35 fits (about 12 minutes). The nested cross-validation interval, about 800 outer fits, is offered with its own time: here, many hours.

### 4.5 How the correction covers tuning and the choice of version

1. **Within a version, tuning is nested** (§4.3). Its out-of-fold predictions come from the whole tuned procedure. Nothing about the tuning is left to correct.
2. **Across families and versions,** BBC-CV corrects choosing among those predictions (§3.3).

**What is not corrected, and is said:**
- how later versions were designed after earlier scores (§3.3);
- trunk changes after scores (ruling 3);
- the pairwise intervals, which are descriptive (`COMPARISONS_NOTE`);
- at p ≫ n, the label "likely too narrow" (`SELECTION_NOT_NESTED`).

**The alternative not taken:** Tsamardinos et al.'s flat BBC-CV over every (family, setting) pair. At our budgets it costs about the same. It hides how settings move from fold to fold, and it would make the deployed settings a pick from pooled predictions. It goes to INBOX.

### 4.6 Small samples

**Below an effective size of 300, searched families use their standard settings by default.**
- The line says why, in the grain's unit noun, or in events:
  - "Tuning: standard settings (214 people are too few to rank settings)."
  - "Tuning: standard settings (31 events are too few to rank settings)."
- Tuning stays available, nested as always.
- **The reason is the variance of the chosen settings, not optimizm.** Nesting already removes optimizm. Riley et al. 2021 found tuning parameters "estimated with large uncertainty … when development data sets have a small effective sample size", which "can lead to considerable miscalibration". Van Calster et al. 2020 found shrinkage's calibration slope often more variable between samples than without it. Martin et al. 2021 recommend quantifying exactly this variability.
- 300 is a convention. T1 reports nested tuning against standard settings on fresh data at an effective size of 200. If tuning wins there by more than 2 standard errors, the threshold is revisited.

**Halving subsamples are gone.** Early-stopping sets are stratified by class, and need an effective size of at least 1,500 (§4.2).

**Penalized families below Riley's minimum** carry the concern: "A penalty does not make up for too few rows: its strength is estimated with large uncertainty here (Riley et al. 2021)."

**Shown at every size:**
- the chosen penalty or setting in each outer fold;
- for penalized families, the out-of-fold calibration slope per fold.

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

**What is recorded.** A `TuningRecord` (`tuning_` on the fitted pipeline) holds:
- the plan: kind, n_plan and its unit, C, K, early stopping, seed, space version, defaults version, threads;
- the library versions;
- the candidates;
- each candidate's pooled inner score;
- the chosen values;
- the number of fits and the seconds taken.

Each outer fold records its chosen candidate. The numbers join the provenance estimates.

**Two kinds of replay,** both at the recorded thread counts:
- **pinned replay** (a new replay mode) refits at the recorded values and must reproduce the deployed predictions;
- **re-run replay** searches again and must reproduce the recorded choices.

Replay's 1e-12 tolerance stands for every family. If T13 shows XGBoost cannot meet it even at a pinned thread count, the record carries a per-family tolerance, and DoD gate 6's text is amended with Nolan's approval.

Each version's model matrix hash is exported.

### 4.8 What is shown after the fit

Results gains a compact **Settings** table per tuned family:
- the deployed values, in plain words with their term ("how big each correction step is · learning rate · 0.05");
- each value's 10th–90th percentile across outer folds ("0.03 to 0.08 in 50 folds").

**Strips.** Only the one or two settings that varied most get a strip: one dot per outer fold, with the deployed value marked. The rest sit behind "More angles".

A recipe option the folds chose shows its share of folds.

The tuning curve and "Compare with standard settings" go to INBOX.

### 4.9 Manual values

- **Where:** every searched family, and the robust linear threshold, through `SetTuning(mode="manual", values=…)`.
- **Entry:** one number field per setting, with its range and its term. Fields are pre-filled with the **standard** values, never the tuned ones. Only the values changed are recorded; the rest stay standard.
- **Partial values:** values given under `automatic` hold those settings fixed and search the rest.
- **Labeled** "set by hand" everywhere, with any source given.
- **Disclosed** by when they were set (§3.3). Values recorded after a tuning result was shown fall under §3.5.
- **Limits:** outside the declared range adds a concern; impossible values are refused by the estimator's own validation.

---

## 5 · Purpose: what tuning and recipes mean under inference

**The reported models have no hyperparameters.** Their choices are declared in the plan before any estimate is seen: the energy model, the forms, the adjustment set, the missing-data method and the secondaries.

**What follows:**
- **No tuning for the declared model.** The tuning line is silent unless a penalized family is chosen for its labeled, shrunk table. That family's penalty is searched by the path search; its table carries no intervals.
- **Recipes are the plan's.** Every recipe line is stated text marked "set by your plan", not an openable phrase. `set_recipe` under inference is refused, with exits:
  - for a transform: "Ask the curve question";
  - for a cap: "Plausibility repairs";
  - for "Try both": "Declare one option".
- **Post-double-selection lasso keeps its plug-in penalty** (Belloni, Chernozhukov & Hansen 2014).
- **DML and TMLE keep the causal lane's stated learners in v2.** One fix lands now: the lasso's inner folds are drawn by unit through `inner_splits` (F9; RT-11). Learners built from the family registry, and nuisance tuning, go to INBOX. Bach et al. 2024 (full text, per the methods review) found tuning on the full sample and on folds performed similarly.
- **Robust linear regression is not offered under inference** in v2. RLM takes no weights and offers only i.i.d. covariances (H1–H3), so it cannot meet ruling 6 or the cluster-robust and HC3 intervals in the DoD. A weighted M-estimator with a design-based or cluster sandwich goes to INBOX.
- **Keeping blanks as blanks is not offered.**

**Refused under inference** (the requests a client can send):

| Request | Why | Exit |
|---|---|---|
| A per-model transform or cap on any model | It changes the estimand behind the plan's back. | The curve question; plausibility repairs |
| "Try both" on any model | Inference may not choose its model from the data it reports on. | Declare one option |

**Impossible by construction, so no validator and no menu item:**
- choosing a setting by an effect estimate (no tuning under inference reads one);
- conventional intervals on coefficients penalized by cross-validation (the table has none).

**Under inference no cross-validated score is shown** (ruling 13), so there is no after-scores path. A change after estimates is recorded by the plan lock.

---

## 6 · The calm UI

Desktop only. Two registers: plain words on the card, and the technical name as a quiet term.

### 6.1 The recipe lines

**Lines are grouped by shared departure,** under the chosen families:
- "Boosted trees, random forest and XGBoost **keep blanks as blanks** in 6 columns."
- "Ridge and elastic net **put every column on one scale**."

**Rules.**
- Changing a grouped phrase applies to every family in it, recording one `SetRecipe` per family. A family set differently gets its own line ("Random forest **fills blanks first** · changed by you").
- A line whose footprint is empty is silent.
- A line with zero routed columns says why: "No blanks reach the trees: the selection step needs every value."
- The shared fill's line ("The other models use the fill learned in each training fold.") is silent when the table has no blanks.
- **One quiet link, "What each model is given",** sits under the lines. It lists each chosen family's non-silent slots with their stated phrases, each clickable. This makes every override one step away, the linear model's included.

**Opening a phrase takes over the card's focal region.**
- The family list collapses to the chosen names, and Escape returns to it.
- Options open with their consequence and at most one quiet label. Arrow keys move through them.
- The canvas layout follows the option's footprint:
  - missing and encoding: **Routing**;
  - scale, transform and cap: **Strip**.
- The flip plays the storyboard.

**Copy for every option.** The word-budget gate runs on it before the build.

| Slot · option | Label | Consequence (words) | Quiet term |
|---|---|---|---|
| missing · native | Keep blanks as blanks | "Each split learns which way a blank goes, so a blank can carry meaning." (14) | missing incorporated in attributes |
| missing · fill | Fill them first, like the other models | "Blanks get each fold's fill, marked as filled; every model sees the same inputs." (14). Without indicators: "Blanks get each fold's fill; every model sees the same inputs." (11) | in-fold imputation |
| any searchable · choose | Try both; keep what predicts better | "Each training fold picks one; the final score includes that choice." (11) | chosen by nested cross-validation |
| scale · standard | Put every column on one scale | "Each column is centered and divided by its spread, so the penalty treats all alike." (15) | standardization (glmnet's default) |
| scale · two_sd | Rescale measured numbers only | "Yes/no and category columns stay 0 or 1; only measured numbers are rescaled." (13) | Gelman's two-SD scaling |
| scale · none | Leave columns in their units | "The penalty then depends on each column's units, such as grams or milligrams." (13) | unstandardized penalty |
| encoding · every_level | Keep every level as its own column | "Each level gets its own column, so the penalty treats every level alike." (13) | full dummy coding |
| encoding · reference | Compare each level with the first | "One level is the baseline; the others are measured against it." (11) | reference coding |
| transform · none | Use values as measured | "Numbers enter as recorded; curved effects are handled by the curve question." (12) | no transform |
| transform · yeo_johnson | Reshape skewed columns | "Each column is reshaped toward a bell curve, learned in each training fold." (13) | Yeo–Johnson transform |
| outliers · none | Keep extreme values | "Extreme values stay as recorded; a robust model limits their pull instead." (12) | no cap |
| outliers · winsorize | Cap extreme values | "Values beyond each training fold's 1st and 99th percentiles are set to those limits." (14) | winsorizing at 1% and 99% |

**Why?** for the missing slot. It names only what applies to this table, so the last sentence appears only when an energy model exists. The full text is 57 words:

> "Trees can send a blank down its own branch. When a blank means something, such as a test not ordered because the patient looked well, that usually predicts better than filling it (Perez-Lebel et al. 2022). It relies on blanks meaning the same where the model is used. Columns your energy model uses are always filled first."

### 6.2 The tuning line

**One line states the shared plan:** "**Tuning: automatic** for boosted trees, random forest and XGBoost · about 30 min".
- Changing it applies to every listed family in one action, recording one `SetTuning` per family.
- A family set differently gets its own line ("Random forest: standard settings · changed by you").
- "Set by hand" first asks which model.
- Every count and minute is computed from the plan for this table.

| Option | Consequence (words) | Quiet term |
|---|---|---|
| Automatic (Recommended) | "Tries 17 settings inside each training fold and keeps the best." (11) | nested random search |
| Standard settings | "The standard settings, the same in every fold." (8) | the family's source: "scikit-learn's defaults", "ranger's defaults", "XGBoost's defaults" |
| Set by hand | "You choose the values; the record says when and why." (10) | |

**Each option's minutes sit at its right edge.** "Lighter" is not listed. It appears just in time, in the Fit button's confirmation when the fit is held: "Lighter tuning · tries 9 settings · about 16 min".

**The canvas uses the Angles layout:**
- **"What will it try?"** shows this table's storyboard: "17 settings" → "each tried on 3 inner folds of about 9,000 people" → "the best refit on the training fold" → "repeated in each of 51 fits".
- **"What will it cost?"** shows minutes by option.

**Why?** (55 words):

> "Tuning tries several settings and keeps the one that scores best. If the rows that pick a setting also grade it, the grade is too kind. So tuning runs inside each training fold, and the rows held back grade the result. Most of the time goes to the 50 folds the models are compared on."

**Variants of the line.**
- Below an effective size of 300: "Tuning: standard settings (214 people are too few to rank settings)."
- Under inference: silent unless a penalized family is chosen.

### 6.3 The override flow

1. The default is stated, with its reason under "Why does this matter?".
2. Pointing at a phrase previews its options on the canvas. Nothing is recorded.
3. Choosing holds the change in the card. The phrase shows its new value, with the quiet label "changed by you".
4. **After scores were seen,** the options keep their soundness order, under one line chosen by holdout status:
   - **none:** "You've seen these scores. The earlier version stays in the comparison, so the final score accounts for choosing between them." (20 words)
   - **sealed:** "The held-out rows are still sealed, so this change can't flatter the final result." (14)
   - **after scores:** "These held-out rows were in the scores you saw, so the earlier version stays." (14)
   - **opened:** "The held-out rows are open, so this runs as a secondary analysis." (12)

   "Try both; keep what predicts better" is labeled available, with its minutes.
5. **One primary action** at the bottom of the card records everything held, and carries the cost of kept versions: "Refit · about 45 min, 2 earlier versions kept". It passes the five-second test.

### 6.4 The comparison table

- It follows §3.7.
- **Colors:**
  - hues by family, in selection order (sage, plum, ochre, steel, clay);
  - a family's versions share its hue;
  - earlier versions are gray;
  - a sixth or seventh selected family is drawn gray, with a direct label, in charts. The table needs no color.
- Technical names appear only as quiet terms ("nested cross-validation", "BBC-CV").

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
| Model-matrix preview | 2 · What will it change in my model? |
| Tuning line and the Fit button, with minutes and kept versions | 2 · What will it change, and at what cost? |
| First-fit line about kept versions; after-scores lines | 3 · Why does that matter for my result? |
| Comparison columns "What it was given" and "Settings"; collapsed earlier versions | 5 · What did I decide, and can a reviewer reproduce it? |
| Settled values; stability strips | 3 and 5 |

---

## 7 · Method contracts, relations and leash rows

### 7.1 Contracts

All are registered with `contracts.register_contract`, in new modules `models/recipes.py` and `models/tuning.py`. A contract that reads the outcome declares scope `model`, as `variable_selection` does, which is what `observed_scope` returns.

| Key | Slot · scope | Decision · stage | Place |
|---|---|---|---|
| `family_recipe` | in_fold · training_fold | `set_recipe` · design | row 10 |
| `native_missing` | model · model (split directions are learned with the outcome) | `set_recipe` · design | row 10 |
| `recipe_choice_nested` | model · model | `set_recipe.choose` · fit | row 11 |
| `hyperparameter_search` | model · model | `set_tuning` · fit | row 11; run order §1.1, step 9 |
| `path_search` | model · model | `set_tuning` · fit | row 11 |
| `manual_settings` | model · model | `set_tuning` · fit | row 11 |
| `shown_versions` | evaluation · model | `set_recipe`, `set_tuning`, `declare_version` · fit | rows 11–12 |

### 7.2 Relations

| Id | Kind | Condition → what fires | Enforced by |
|---|---|---|---|
| `recipe_stated` | implies | a family selected → its line (silent when its footprint is empty), spur and methods clause | `family_spec`; voice |
| `trees_route_blanks` | implies | prediction, rows kept, a tree family → the routed columns and their count, in the lineage and the sentence | `family_spec` |
| `blank_intolerant_fill_first` | implies | a step that cannot pass a blank reads a column → that column is filled for every family | `family_spec` |
| `no_column_routed_said` | implies | a tree family with zero routed columns while blanks exist → the line says which step needs every value | `family_spec`; voice |
| `trees_skip_form_rule` | implies | Explore's spline rule on → tree families do not take it | `family_spec` |
| `routed_no_indicator` | implies | a routed column → no indicator, and the sentence says so | `family_spec`; voice |
| `native_not_under_inference` | conflicts (not offered) | inference → the fill | validator |
| `recipes_differ_marked` | implies | compared versions differ in the information they see → the line under the table; pairwise sentences speak of procedures | fit stage |
| `choice_after_scores_disclosed` | implies | a stamp shows versions → the sentence names them, and what is not corrected | completion; voice |
| `shown_versions_stay` | implies | holdout status none or after scores → every shown version is fitted, reverts included | fold; design stage |
| `compared_families_stay` | conflicts (refused, exists) | dropping a family whose score was shown, unless sealed or opened → exit "Keep the compared families" | `selection._compared_families_stay` (by version) |
| `sealed_holdout_absorbs_change` | implies | a holdout sealed before any score → no earlier version kept; the sentence says why | completion |
| `holdout_after_scores_labeled` | conflicts (block and record) | a holdout drawn after scores → treated as none; its score labeled | `set_split` validator; completion |
| `opened_change_is_secondary` | conflicts (block and record) | seal opened → the change is secondary; the opened version stays final | validator; `mark_final` |
| `deployed_is_best_unless_declared` | implies | no holdout → BBC-CV's choice is reported and used; when it is an earlier version, the line and its exits | fit stage |
| `values_from_tuning_data_derived` | conflicts (block and record) | values or held options recorded after a tuning result was shown, no sealed holdout → labeled; excluded from BBC-CV and the result | completion; validator |
| `folds_choose_nested` | implies | `choose` given → the options join the search; shares shown | tuning |
| `plan_frozen` | implies | a searched or path family → one plan, set from n_plan, used in every fit | design stage |
| `tuning_nested_everywhere` | implies | a searched family → the search runs in every fit (§4.3) | `fit_pipeline` |
| `tuning_splits_follow_outer` | implies | units repeat, time order, or the population answer → inner splits by unit, forward-chained, or by whole PSU | `inner_splits`, `design` |
| `stopping_rows_first` | implies | early stopping → stopping units drawn first, kept out of every step's fit and of any resampling | `fit_parts`; `ImbalanceCorrected` |
| `path_search_refits_head` | implies | a path family → the head is refit in every inner split | tuning |
| `tuning_on_primary` | implies | any search → pooled strictly proper primary, weighted under the population answer | tuning |
| `tuning_estimate_first` | implies | a fit over 30 s → its estimate in the preview and on the action; over 2 min → waits for Fit | cost; scheduler |
| `small_n_standard_settings` | implies | effective n_plan below 300 → standard settings, stated with the count | plan |
| `shrinkage_small_n` | implies | a penalized family below Riley's minimum → the concern and the fold spread | `shelf_order`; fit |
| `penalty_at_edge` | implies | the chosen penalty at a grid edge in most folds → the concern | fit |
| `manual_labeled` | implies | manual values → "set by hand" everywhere, with when | voice |
| `pinned_for_description` | implies | a band or reseed of a searched family → tuned values held; caption "conditional on its tuned settings" | `pinned_to_full_fit` |
| `inference_recipe_is_the_plan` | conflicts (refused) | inference, `set_recipe` → refused with the exits in §5 | validator |

**The chain test** (BLUEPRINT §13) gains one reference chain: an NHANES-shaped prediction with blanks, families {linear, boosted trees, ridge}, rows kept, the population answer.
- It asserts `trees_route_blanks`, `blank_intolerant_fill_first` (on the energy nutrients), `recipes_differ_marked`, `plan_frozen`, `tuning_nested_everywhere` and `tuning_splits_follow_outer` (whole PSUs).
- After a scripted change once scores were seen, it also asserts `choice_after_scores_disclosed` and `shown_versions_stay`.
- Each must appear in the lineage, in the participant flow where it applies, and in the methods sentence.

### 7.3 Leash rows, added to MODELING_SEQUENCE §4

| Step | Prediction | Inference |
|---|---|---|
| A family's stated recipe | stated; silent when it changes nothing here | stated, "set by your plan" |
| Trees keep blanks as blanks | recommended (rows kept) | not offered |
| Trees take the shared fill | available | the only option |
| Penalized: every column on one scale | recommended | the plan's (shrunk table) |
| Penalized: Gelman's two-SD scaling | available | not offered |
| Penalized: columns in their units, or a reference level | ranked lower | not offered |
| Unpenalized: scaling | silent | silent |
| Per-model reshaping (Yeo–Johnson) | ranked lower; the curve question first | refused; exit: the curve question |
| Per-model cap | ranked lower; robust linear regression offered | refused; exit: plausibility repairs |
| Try both; keep what predicts better | available | refused; exit: declare one |
| Automatic tuning, nested | recommended from an effective size of 300 | not applicable (path search only, for a labeled shrunk table) |
| Standard settings | available; recommended below 300 | not applicable |
| Values by hand before any tuning result was shown | available, labeled | not applicable |
| Values by hand or held after a tuning result was shown | block and record (no holdout, or a holdout after scores); available under a holdout sealed before any score | not applicable |
| A change after scores were seen | available: disclosed; the earlier version kept and the choice among versions corrected, unless a holdout was sealed before any score | not applicable (the plan lock) |
| Drawing a holdout after scores were seen | block and record: labeled, and treated as none | not applicable |
| A change after the seal was opened | block and record: secondary | not applicable |
| Reporting a version other than the best (no holdout) | available: its own score labeled, the selection-corrected estimate beside it | not applicable |
| Robust linear regression | available (numeric) | not offered in v2 |

**Impossible by construction.** These are stated in the methods reference, with no validator and no menu item:
- tuning on all training rows and then cross-validating (its reachable form is §3.5);
- tuning on AUC or accuracy (the search scores only the primary);
- class reweighting set on a family (only Explore's imbalance lever, which recalibrates);
- XGBoost's linear booster;
- the robust linear threshold tuned by cross-validation.

---

## 8 · Acceptance tests, each with an independent reference

All are Tier A unless marked.

**Slow tests.** T1 and T12 carry the `slow` marker, run in the acceptance harness only, and are scheduled with Nolan outside quiet hours.

**T1 · Tuning outside nested cross-validation is optimiztic; nesting is not.** (slow)
- **Generators:**
  - (a) null: n = 200; ten N(0, 1) predictors; y ~ Bernoulli(0.3).
  - (b) signal: logit = −0.85 + 0.8x₁ − 0.6x₂ + 0.5x₁x₃.
- **Procedure F (flat: what not to do),** in plain scikit-learn: the 17 candidates evaluated by 5-fold cross-validation on all rows; the best one's log loss reported.
- **Procedure N (ours):** the fit stage's cross-validated log loss for boosted trees, with the automatic plan forced on.
- **Truth:** retrain each procedure on 4/5 of the rows, and score it on 20,000 fresh rows. In (a), the floor is the entropy, 0.6109 nats.
- **Assertions, over 200 datasets:**
  - F's mean is below its truth by more than 3 Monte Carlo standard errors;
  - N's mean is within 2 standard errors of its truth;
  - with ridge added, the selection-corrected estimate is within 2 standard errors or conservative.
- **Also reported, not asserted:** nested tuning against standard settings on fresh-data loss at an effective size of 200 (§4.6).

**T2 · Known answers.**
- (a) **Ridge's path search** on the diabetes data, with explicit inner splits, against numpy's closed form at each λ and the **pooled** squared error's argmin. Index exact; score to 1e-10. The test also asserts that the scorer is the pooled loss.
- (b) **Candidates** equal scipy's scrambled Sobol mapped through each documented scale, recomputed in the test, with the seed derived from sha256.
- (c) **Pinned replay** reproduces the deployed predictions exactly at the recorded thread count.
- (d) **The elastic net's path search** equals independent `ElasticNet` and `LogisticRegression` refits over the same ratios and splits, choosing on the pooled loss.
- (e) **Ties:** two candidates with losses equal after rounding resolve to the lower index.

**T3 · Inner splits follow the outer ones.** The search logs every inner (training, validation, stopping) set.
- **3 rows per person:** no person on both sides; stopping rows are whole persons.
- **Time-ordered:** every inner validation unit is later than every inner training unit.
- **NHANES-shaped design** (15 strata, 2 PSUs each, outer design K = 2): every inner split keeps whole PSUs; K equals min(3, PSUs in the fit); the weighted inner loss equals a hand computation.
- **The oversample lever with 3 rows per person:**
  - no person sits on both sides of the stopping split;
  - no stopping row is resampled;
  - the recalibration splits cover only the candidate's own rows.

**T4 · No leakage, by perturbation.**
- `observed_scope` returns `model` for the search contracts.
- Perturbing outer fold k's held-out rows leaves fold k's chosen values and fit unchanged bit for bit.
- Perturbing a training row changes them.

**T5 · The native path is reachable.** Blanks in three numeric columns that no energy step reads, and in one energy nutrient; "Keep every row"; prediction.
- (i) The HistGB, forest and XGBoost model steps receive NaN in exactly those three columns, and the nutrient is filled.
- (ii) The linear model receives no NaN.
- (iii) HistGB fit directly on a hand-built matrix gives identical predictions.
- (iv) The lineage and the sentence name the routed columns, and say no indicator was added.
- (v) **With Explore's spline rule on,** the trees still receive the three columns blank.
- (vi) **With the selection step on,** none are routed, and the line names the selection step.
- (vii) Under inference, `set_recipe` gets a 409 with its exits.

**T6 · The estimate counts the fits.** (Tier B)
- An instrumented counter equals Σ F(plan) over the outer fits, with and without the imbalance lever, and with and without the out-of-bag rule.
- The estimated time is within a factor of 3 of the measured time.

**T7 · The after-scores paths, scripted.** Fit {linear, boosted trees}, serve the fit, then record `set_recipe(boosted_trees, missing="fill")`.
- (i) The stamp lists both versions even when the client sends none.
- (ii) The sentence says "after the cross-validated scores of … were seen" and states what is not corrected.
- (iii) The next fit has three rows, the earlier one collapsed and labeled.
- (iv) BBC-CV's set has three versions and matches an independent BBC-CV written from Tsamardinos et al.'s algorithm, on the same draws.
- (v) **A revert of the change** keeps the changed version, once shown, as an earlier version, and a replay from `decisions.jsonl` alone rebuilds both.
- (vi) **A holdout sealed before any score:** no earlier row; the sealed sentence.
- (vii) **A holdout drawn after scores:** the earlier row is kept; no sealed sentence; dropping a compared family is still refused; the held-out score is labeled.
- (viii) **The reported version** is BBC-CV's choice. When that is the earlier version, the line and both exits appear. `declare_version` reports the current version's own score, labeled, with the corrected estimate beside it.
- (ix) **After the opening,** a change runs as secondary, and the opened version stays final.
- (x) A `scores_seen.json` holding version keys does not make `select_models` refuse.
- (xi) A pre-spec record listing `boosted_trees` maps to its legacy version, and keeps it as an earlier version.

**T8 · Try both.**
- The shares across folds equal an independent count from the per-fold records.
- On a generator where the outcome depends on whether x₁ is blank, keeping blanks wins most folds. This is a sanity check, not a performance claim.

**T9 · Small samples.**
- 5,000 units at 3% prevalence: the plan uses standard settings, and the line counts events.
- n = 150: ridge carries the Riley concern, and its 10th–90th percentile range of λ equals an independent computation.

**T10 · Inference refusals.** Each refusal in §5 returns its exits. The causal lasso's inner folds keep units whole.

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

**T14 · The chain test** of §7.2.

**T15 · Budgets, purposes and API.** (Tier B)
- Every new string is within its budget.
- Every new view has a registry entry.
- `GET /api/models` lists recipes and tuning.

**T16 · Values from tuning results.**
- On T1's null generator: copy the final refit's values into manual. The version is block-and-record, labeled, and excluded from BBC-CV and the result. The test reports the size of its optimizm.
- With a holdout sealed before any score, the same values are allowed.

**T17 · One plan.**
- At 340 effective units with 5 folds: every outer fit, bootstrap resample and the final refit carry the same `TuningPlan`.
- At 1,600: early stopping is decided once, by the plan.

---

## 9 · Engine work packages

**Sizes** follow the parity report: S is one module and its reference test; M adds an explanation path or tuning; L is a new workflow.

**Order** follows BLUEPRINT §12: the math first (RT-1 to RT-5), then methods and routing (RT-6 to RT-11), then export and the catalog.

| WP | Size | What | Where | Tests |
|---|---|---|---|---|
| RT-1 · The search engine | L | `TuningDecl`, `Dimension`, `TuningPlan` (frozen, effective n); Sobol candidates with sha256 seeds; `fit_parts` (stopping units first, head per split and per option, nested splits redrawn); pooled weighted loss with rounding; out-of-bag scoring and its conditions; the path search; `TunedPipeline` (`search=None`, `at`); `TuningRecord`; cancel. `fit_pipeline` gains `design=` and dispatch; seeds threaded from the split. Fixes F11, F13 and F15. | new `models/tuning.py`; `models/inner_cv.py`; stage call sites | T2(a–e), T3, T4, T13, T17 |
| RT-2 · Recipe declaration | M–L | `RecipeSlot`, `RecipeOption`, `defaults_version`; recipes in `DesignSpec`; `family_spec` inside `build_pipeline`, `family_steps` and `describe_steps`; `PenaltyScaler` (`center_`, `scale_`; two-SD); `MissingLevelEncoder(drop=)`; transform and cap steps; the trees' form-rule skip; lineage spur; `register_family` checks; `FamilyInfo`; teaching options for the new families. | `models/base.py`, `models/pipeline.py`, `models/explain.py` (`_undo_scaling`, `linear_equation`), `models/elastic_net.coefficients`, `models/artifacts.py`, `models/lineage.py`, `stages/__init__.py` (reads, version bumps), `teaching/content.py` | T15, T5(iv) |
| RT-3 · The native path | M | `reads(spec)` and `passes_blanks` on every step; `native` computed by walking the steps; the imputer and indicators skip routed columns; the preview compares specs; caption and `table_focus`; the "Keep every row" card copy; the teaching note. | `models/pipeline.py`; `methods/*` steps; `models/previews.py`; `teaching/content.py`; `ChoiceQuestions.tsx` (copy only) | T5 |
| RT-4 · Imbalance correction | S | The early-stopping interface; stopping units before resampling; recalibration splits per fit. Fixes F12. | `methods/levers.py` | T3 |
| RT-5a · Boosted trees tuned | S | Its `tuning` declaration; `describe`. | `models/boosted_trees.py` | T1, T2(c) |
| RT-5b · Ridge | S | The family, path declaration, full coding; `Ridge` in `explain.LINEAR_MODELS`. | new `models/ridge.py`; `models/explain.py` | T2(a), T11 |
| RT-5c · Robust linear | M | The `RLM` wrapper; prediction only; numeric only; linear SHAP. | new `models/huber.py`; `models/explain.py` | T11 |
| RT-5d · Random forest | L | The family and its defaults; out-of-bag tuning; a compiled TreeSHAP (the `shap` package's TreeExplainer, added to the server requirements once T11 confirms it matches v2's TreeSHAP with blanks; otherwise explanations cover the rows v2's TreeSHAP can explain within the stage's budget, and the caption counts them); the forest's explanation scale (probability for SHAP; log-odds of the clipped probability for curves, labeled). | new `models/forest.py`; `models/explain.py` | T2(c), T11 |
| RT-5e · XGBoost | M–L | The wrapper (early-stopping interface, margin `decision_function`, label encoding, safe names, pinned threads); SHAP from `pred_contribs`. | new `models/xgboost_family.py`; `models/explain.py` | T11, T13 |
| RT-5f · Elastic net on the path search | M | Full coding; per-row logistic grid; `alpha_` and `l1_ratio_` readers moved to the record; the screen refit per split. Fixes F4 and F5. | `models/elastic_net.py`; `methods/omics.py` | T2(d), T12 |
| RT-6 · Decisions and stamps | M | `SetRecipe`, `SetTuning`, `DeclareVersion`, `Stamp`; completions on seven kinds (revert gains one); holdout status; the data-derived rule; validators and exits; key views; sentences; previews with estimates. | `decisions.py`, `voice.py`, `models/previews.py`, `graph.py` | T7(i–ii), T10, T16 |
| RT-7 · Versions | L | `VariantSpec` keys; `versions_shown` folded across reverts; `LEGACY_DEFAULTS`; `scores_seen.json` with specs and sequences; `compared_families`, `vouch`, `scored_in` and `explained_in` by version; `design.objects["earlier"]`; fit and evaluation (`design_bbc`) include earlier versions; `variant` on rows; the out-of-fold cache; `declared_result` and `declare_version`; the seal by version (`open_seal.variant`, `AtOpening`, `mark_final`). | `models/selection.py`, `seal.py`, `stages/modeling.py`, `stages/evaluation.py`, the frontend's results types | T7 |
| RT-8 · Cost and when fits start | M | Timing at `at(center)`; F(plan) with its multipliers; `outer_fits` counted across stages; the F7 fix; preview estimates; the scheduler's 2-minute hold (bypassed in replay); the Fit and Refit action payload. | `models/cost.py`, `stages/modeling._estimates`, the job scheduler | T6 |
| RT-9 · Fit artifact and results data | S–M | `FittedModel.variant`, `earlier`, `inputs`, `settings`, `tuning`; per-fold records through a `cross_validate` hook; `pinned_to_full_fit` holds tuned values, with its caption. | `models/artifacts.py`, `models/metrics.cross_validate`, `stages/modeling.py`, `stages/explain.py` | T8, T9 |
| RT-10 · Contracts and relations | S | §7's contracts (scope `model` where the outcome is read), relations, the chain, methods clauses. | `models/tuning.py`, `models/recipes.py` | T4, T14 |
| RT-11 · Causal lasso folds by unit | S | `inner_splits` for the lasso's folds. Fixes F9. | `models/causal.make_learner` | T10 |
| RT-12 · Export and replay | M | Threads pinned and recorded; versions in provenance; `TuningRecord` numbers among provenance estimates; pinned-replay mode; per-version matrix hashes. | `export/record.py`, `export/replay.py` | T13 |
| RT-13 · Catalog, shelf, journeys | S, plus a heavy run | `FAMILY_LENSES` and contract catalog entries; shelf scores; `first_models` checked per journey; the journeys and review packets regenerated as a run scheduled with Nolan. | `reference/catalog.py`, `reference/journeys.py` | the journeys |
| RT-14 · Acceptance tests | M | T1 to T17. | `core/tests/acceptance/` | |

**Presentation**, once the engine passes verification:

| WP | Size | What |
|---|---|---|
| PR-1 | M | Grouped recipe lines; "What each model is given"; phrases taking the focal region; Routing and Strip canvases; option copy |
| PR-2 | M | The shared tuning line; per-family exceptions; the Fit button's held state, with Lighter in its confirmation; the Angles canvas |
| PR-3 | S–M | Comparison columns; collapsed earlier versions; the information line; the first-fit and after-scores lines; the earlier-version-best line and its exits |
| PR-4 | S | Results Settings: the compact table, one or two strips, More angles |

---

## 10 · Amendments, INBOX and sources

### Amendments (on Nolan's approval)

- **MODELING_SEQUENCE §1 row 10:** "stated per family with its reason, silent where it changes nothing; under prediction its phrase can be changed (`set_recipe`) under the four safeguards of RECIPES_AND_TUNING §0; under inference the plan decides."
- **Row 11 gains:** "by a seeded random search over a plan fixed once and used in every fit, inside every outer training fold, on the comparison's pooled primary, with inner splits drawn as the outer ones are (whole PSUs under the population answer)."
- **Row 12(b) gains:** "With no holdout the reported and deployed version is BBC-CV's choice; a different version may be declared after scores, reporting its own score labeled 'chosen after the scores were seen' beside the selection-corrected estimate. A holdout drawn after scores is labeled so, and does not count as sealed."
- **MODELING_SEQUENCE §4** gains the rows of §7.3.
- **BLUEPRINT §4:** a heavy stage whose estimate exceeds 2 minutes waits for the user's Fit action instead of starting on its own; replay is exempt.
- **V2_DEFINITION_OF_DONE,** amendment of 2026-10-05: "… are being specified" becomes "… are specified in `RECIPES_AND_TUNING.md`". Huber enters as robust linear regression on the prediction shelf.

### INBOX (v2.x)

- Successive halving, Hyperband, TPE and BOHB, for large tables.
- "Thorough" budgets, and the tuning curve (Dodge et al. 2019) from full-fidelity scores.
- "Compare with standard settings" as a kept version.
- Native categories (HistGB, XGBoost), with the 255-level limit and a sparse-level concern.
- Per-family Pareto and robust scaling; the log1p per-model transform.
- Robust linear regression under inference: a weighted M-estimator with a design-based or cluster sandwich, checked against R `robsurvey`.
- Nuisance learners from the family registry, and nuisance tuning inside cross-fitting (Bach et al. 2024).
- Re-tuned substitution bands.
- Trunk answers changed after scores kept as versions.
- Monotone constraints, declared from domain knowledge.
- LightGBM; survival versions of the new families.
- Flat BBC-CV over every configuration.
- A cross-validated penalty for post-double-selection lasso.

In-fold PCA for omics stays its own DoD item.

### Sources

**Tuning**
- Bergstra J, Bengio Y. Random search for hyper-parameter optimization. *JMLR* 2012;13:281–305.
- Bischl B, Binder M, Lang M, et al. Hyperparameter optimization: foundations, algorithms, best practices, and open challenges. *WIREs Data Min Knowl Discov* 2023;13:e1484.
- Cawley GC, Talbot NLC. On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR* 2010;11:2079–2107.
- Optuna FAQ, "How can I obtain reproducible optimization results?"
- Probst P, Boulesteix A-L, Bischl B. Tunability: importance of hyperparameters of machine learning algorithms. *JMLR* 2019;20(53):1–32.
- Probst P, Wright MN, Boulesteix A-L. Hyperparameters and tuning strategies for random forest. *WIREs Data Min Knowl Discov* 2019;9:e1301. The tuneRanger documentation (`replace = FALSE`; `num.trees = 1000`; out-of-bag evaluation).
- Kruppa J, Liu Y, Biau G, et al. Probability estimation with machine learning methods for dichotomous and multicategory outcome: theory. *Biom J* 2014 (and *BioData Mining* 2014;7:2 on terminal node size).

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

**Caveat.** Probst et al.'s spaces and tunability values were read from the arXiv HTML versions. The expert review packet (DoD §4) checks them against the published text.

---

## Decisions for Nolan

The methods above are the orchestrator's call. Three choices were product-level. **Nolan ruled on them on 2026-10-06, and draft 3 must fold the rulings in before anything is built from this spec.**
1. **Missing values: "Try both" is the default.** Under prediction, with the rows kept, the tree families' missing slot defaults to `choose: [native, fill]`. Each training fold picks whether the trees keep blanks as blanks or take the shared fill, and the score includes the choice (§3.3(c)). This replaces "keep blanks as blanks" as the default in §2.2, §2.3 and §6.1. "Fill in each fold" still becomes "Keep every row". (Proposed: keep blanks as blanks by default.)
2. **A fit expected to take over about 2 minutes waits for the Fit action**, as proposed (§4.4). BLUEPRINT §4's "live" rule changes for long fits only.
3. **Scope cuts.** Three groups move to INBOX as proposed:
   - faster search for big tables: halving, Hyperband, TPE and BOHB; the Thorough budget and the tuning curve;
   - more preprocessing options: native categories, per-family Pareto and robust scaling, and the per-model log1p;
   - the inference extensions: Huber under inference; nuisance learners from the registry, and their tuning.

   **One group stays in v2:** "Compare with standard settings" as a kept version, shared-step changes after scores kept as versions, and re-tuned substitution bands.

**Left to the orchestrator in draft 3:**
- how "Try both" runs below an effective size of 300, where searched families otherwise use their standard settings (§4.6);
- the design and cost of trunk versions and re-tuned bands.

---

## What changed after review

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
