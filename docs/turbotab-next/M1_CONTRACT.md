# M1 contract — the energy-adjustment slice, with consequence previews

M1 is done when a nutrition researcher can take the real NHANES export from upload to fitted models
and substitution curves in a browser, see every choice's consequence on their own data **before**
recording it, and watch the pipeline re-flow when they change an earlier answer. Read BLUEPRINT §0,
the North star, §4 and §11 first. Contract-as-code already on `turbotab-next`:
`turbotab/core/decisions.py` (the M1 kinds and slots), `turbotab/core/stages/__init__.py` (the M1
graph, with stubs), `turbotab/core/consequences.py` (preview views and the planner),
`turbotab/core/graph.py` (`Bundle`). Build on those; don't redefine them.

## 1 · The interview — the Router, server-side

`ProjectView` gains `interview: InterviewStep[]`, in asking order:

`lens · target · task · purpose · roles · exclusions · missing · split · energy_adjustment · models · substitution`

```
InterviewStep { key: QuestionKey, status: "answered"|"open"|"waiting"|"skipped"|"not_applicable",
                decision_id: string|null, reason: string|null, waiting_on: string[] }
```

- At most one step is `open`: the first applicable unanswered one. Later ones are `waiting`, and
  `waiting_on` names what they wait for (a question key or a stage). While a stage the first one
  needs is still computing, it is `waiting` on that stage and none is open; a stage that failed
  or was stopped holds nothing back.
- `task` is `skipped` when detection is high-confidence; `reason` is the detection in the app's
  voice. The client's "Ask me anyway" reopens it (as in M0).
- `energy_adjustment` is `not_applicable` unless the lens includes `dietary`, a column has the role
  `energy`, and at least one `exposure` is an energy-bearing nutrient. `reason` says which is missing.
- `substitution` waits on a fresh `fit`, and is `not_applicable` with fewer than two energy-bearing
  nutrient predictors.
- Changing an earlier answer never reopens later ones: they stay answered and their stages go stale.
- Module: `turbotab/core/interview.py`. The frontend renders what the Router says; it does not
  decide the order.

## 2 · Decisions

The kinds and slots are defined in `decisions.py`. Validators (409 Refusal with `exits`, each exit a
decision the user can take instead):

| Kind | Refuse when |
|---|---|
| `set_task` | `binary` with ≠ 2 distinct values; `multiclass` with > 20 (`task_mismatch`) |
| `set_roles` | a key is not a column; the target is given a role (`target_has_role`); no column is exposure, covariate or energy (`no_predictors`) |
| `set_exclusions` | a column is missing or not numeric; `low ≥ high` |
| `set_energy_adjustment` | method ≠ none and: energy column lacks the `energy` role; nutrients empty or not all `exposure`; the method is not applicable (`method_not_applicable`, exits = the applicable methods); strata not categorical with ≤ 10 levels |
| `select_models` | an unknown family; a family that cannot model this task |
| `set_substitution` | donor = recipient; either is not an energy-bearing nutrient predictor |

`set_split` is never refused for small n — the consequence is shown instead (shelf never shortened).

**Every record carries `sentence`**, authored when it is recorded by
`turbotab.core.voice.sentence_for(decision, state_before, ctx) -> str`: one publishable sentence,
backticks around data values, no machinery. Examples: "Energy was adjusted by the residual method:
each nutrient is replaced by its residual on `kcal`, fit on training rows." · "`177` rows outside
`500`–`5000` kcal were excluded as implausible intakes."

## 3 · Stages and artifacts

`Bundle` stages serve only `data` to clients (`StageResult.artifact`); frames and objects stay in
the cache.

- **roles** → `{ columns: [{ column, proposed: Role, confidence: "high"|"medium"|"low", reason (≤ 16 words), linked_to: string|null, unit: string|null }], repeats: { column, n_units, max_rows_per_unit }|null }`
  Recognizers: identifier (id-like name or ≥ 95% unique; `SEQN`), energy, exposures (nutrients;
  metabolites/genes under omics lenses), design (`WT*`, `SDMV*`, weight/strata/PSU), flag
  (`imputed_*`, `*_flag` → `linked_to` its base column), time (dates, cycle, year, visit),
  covariates (age, sex, gender, BMI, race, education, income…), excluded (free text, constants).
- **proposals** → `{ exclusions: [{ rule: ExclusionRule, label, affected: int, evidence: {status, source} }], energy: { energy_column|null, nutrients: string[], strata_candidates: string[], applicability: {method: {ok, reason}}, usual: string|null, usual_evidence: {status, source}|null }|null }`
  Pack-sourced (NUTRITION_PACK §02 for implausible intake, §04 for energy). Offered, never pre-selected.
- **cohort** → `Bundle(data={ steps: RowStep[], n_final, predictors: string[] }, frames={"rows": row_id, "measured": row_id})`
  Steps: `loaded` → `outcome_measured` → one step per exclusion rule → `complete_cases` (only when
  `missing == "complete_case"`). Predictors = columns with role exposure, covariate or energy.
  `measured` is every row whose outcome is recorded.
- **split** → `Bundle(data={ n_train, n_holdout, holdout, seed, folds, grouped_by|null, n_groups|null, stratified, note }, frames={"assignment": row_id, partition, fold, "sealed": row_id})`
  Grouped by the identifier when it repeats; stratified for classification; seeded; CV folds
  assigned on training rows, also grouped. `holdout == 0` → cross-validation only. The held-out
  rows are drawn over `measured`, not the cohort, so an exclusion or missing-values answer never
  moves a row across the seal; `sealed` lists them all, in the cohort or not, and no preview reads them.
- **shelf** → `{ families: [{ key, label, rank, fit: "good"|"fair"|"poor", concerns: string[], inductive_bias }] }`
  Every family that can model the task, ordered; concerns stated (n, p, events), never hidden.
- **design** → `Bundle(data={ lineage: Lineage, matrix: {n_rows, n_cols}, models: [{ family, label, steps: [{key, label, detail}] }], estimand: string|null, substitution_pairs: [{donor, recipient}], warnings: string[] }, objects={"pipelines": ...})`
  Step order: impute (if `impute`) → EnergyAdjuster (if method ≠ none) → one-hot → scale (families
  that need it) → model. Lineage lanes raw → adjusted → matrix; wide tables collapse role groups
  into count nodes (`collapsed: true`).
- **fit** → `Bundle(data={ task, primary_metric, metric_labels, n_train, n_holdout, models: [{ family, label, cv: {metric: {mean, sd, folds}}, holdout: {metric: value}|null, coefficients: [{feature, estimate, ci_low, ci_high, p}]|null, fit_seconds, concerns }] }, objects={"fitted": ...})`
  CV on training rows with the split's folds; holdout scored once. Regression: R², RMSE, MAE.
  Binary: AUC, Brier, log loss. Multiclass: accuracy, macro-F1, log loss. Coefficients for linear
  families (CIs via statsmodels when `purpose == "inference"`). Progress per model and fold; honors
  `cancelled()`.
- **substitution** → `{ donor, recipient, step_kcal, ks, total_kind, estimand|null, note, models: [{ family, label, delta, ci_low|null, ci_high|null, on_support_fraction, stopped_at|null, effect_label }] }`
  `substitution_curve` on ≤ 5,000 training rows, through each fitted pipeline on raw inputs.

## 4 · Consequence previews

`POST /api/projects/{pid}/preview` with the `Decision` an option would record → `PreviewResult`
(`consequences.py`). Nothing is recorded. p95 under 500 ms on the NHANES export (measure it).

| Kind | Views (primary first) | Built by |
|---|---|---|
| `set_exclusions` | row_flow · distribution of the excluded column with the cut marked | rows agent |
| `set_missing` | row_flow · table_focus of the rows/cells affected | rows agent |
| `set_split` | row_flow (the train/held-out fork) | rows agent |
| `set_roles` | lineage (which columns enter the model) | rows agent |
| `set_energy_adjustment` | relationship (the nutrient most correlated with energy, against energy, before/after) · lineage · distribution | modeling agent |
| `select_models` | lineage of the model matrix per family (e.g. trees skip scaling) | modeling agent |

Everything else falls through to `diff_views` via `register_transform`. A preview of a decision
that would be refused answers with the refusal (409). `DistributionView.cuts` marks values on the
axis (a rule's bounds).

## 5 · Teaching

`GET /api/teaching` → `TeachingEntry[]`, one per question key:

```
TeachingEntry { key, title, question, one_liner, why, consumer,
                options: [{ value, label, consequence }],
                terms: [{ term, definition }],
                drawer: { sections: [{ heading, body }] } | null,
                evidence: { status, source } | null }
```

Word budgets (a test enforces them): title ≤ 8 · question ≤ 14 · one_liner ≤ 22 · why ≤ 60 ·
consumer ≤ 16 · option label ≤ 4 · option consequence ≤ 16 · term definition ≤ 25. Drawer content
is sourced from `docs/turbotab/research/*` with section references, and may be longer.

## 6 · Findings

The `Finding` shape adds `summary` (≤ 20 words), `routes_to` (a question key or null), `lever_label`
(≤ 5 words, e.g. "Adjust for energy"), and `group` (a pager key for same-kind findings, or null).
Text is normalized (no raw markdown, one terminal period, the app's voice). Routes in M1: energy
adjustment → `energy_adjustment`; implausible intake → `exclusions`; identifiers, survey design and
imputation flags → `roles`. A finding with no lever in this version says so in its summary instead of
pretending.

## 7 · Model families

`turbotab/core/models/`: a `ModelFamily` protocol — `key, label, tasks, inductive_bias (≤ 20 words),
strengths, cautions, needs_scaling, handles_missing, build(task, purpose, n_rows, n_features),
coefficients(...)`. M1 families: `linear` (OLS / logistic; statsmodels for inference),
`elastic_net` (penalized, inner CV on training folds only), `boosted_trees` (HistGradientBoosting,
native missing values). `GET /api/models` → the registry. Later milestones add families by
registering, never by editing a switch.

## 8 · Ownership

| Agent | Owns |
|---|---|
| **rows** | validators in `decisions.py`; `interview.py`; `stages/rows.py`; `consequences.diff_views` and the preview builders/transforms for exclusions, missing, split, roles; `routes/preview.py`; `ProjectView.interview`; wiring `DecisionRecord.sentence` (calling `voice.sentence_for`); lazy worker spawn and idle retire in `jobs.py` |
| **modeling** | `turbotab/core/models/`; `stages/modeling.py`; preview builders for energy adjustment and models; `routes/models.py` |
| **voice** | `turbotab/core/voice.py`; `turbotab/core/teaching/` + `routes/teaching.py`; `stages/proposals.py`; the findings additions in `stages/findings.py`; the task reason in `stages/target.py` restated; the word-budget test |
| **frontend** (next workflow) | the Record, findings, teaching drawer, the pipeline panel, consequence previews |

## 9 · Acceptance

On the real NHANES export, in a browser: lens dietary → outcome `glucose` → prediction → roles
confirmed → exclusions (pack rule) → complete cases → 20% holdout → residual energy adjustment →
all three families → fit → a fat → carbohydrate substitution curve. Then change the energy method
to density and watch lineage, metrics and curves re-flow. Every option previews its consequence
within 500 ms. Every finding with a lever routes to its question. The DRIVE_RUBRIC checklist passes.
