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
  families. Under `purpose == "inference"` their intervals follow how the rows were sampled
  (`turbotab/core/models/inference.py`, audit WP2): HC3 on independent rows; CR2 with
  Bell–McCaffrey degrees of freedom whenever an identifier repeats, however the seal was drawn;
  none below the unit floor (the refusal and its exits are recorded); Firth's penalized likelihood
  when a column separates a binary outcome. Each coefficient adds `se` and `df`, and each model an
  `inference` record (estimator, covariance, caption, grouped_by, n_clusters, separated, refused,
  exits). Progress per model and fold; honors `cancelled()`.
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

---

# M1 part 2 — the frontend on the ruled design, plus four fixes from the live journey

Part 1 (§1–§9) is built and merged. Part 2 builds the production frontend on the design Nolan ruled
(BLUEPRINT §11.1), and fixes what the live NHANES journey found (HANDOFF.md). The `stage` prototype
lives on the branch at `turbotab/frontend/src/explore/stage/` (route `/lab/explore/stage`): **lift its
views into production components.** Don't rebuild them from scratch. The lineage diagram is the one
Nolan called the best he has seen.

## 10 · The layout

```
┌ header ─────────────────────────────────────────────────────────────────────┐
├ PIPELINE BANNER: Rows 21,849 → 21,348 → 2,943 ▸ train 2,352 | held out 591 · Columns 11 → 10 (residual)
│                 · Models 3 · Result R² 0.077 (CV)        ← each segment is a button
├──────────────── working window ──────────────────────────────────────────────┤
│ THE RECORD (left)                 │ THE STAGE (right, sticky)                │
│ the Router's questions; answered  │ PREVIEW · <option>   [now ⇄ with this]   │
│ ones settled into sentences;      │   ● ● ●  step dots      pinned r 0.86→0.14│
│ findings with levers              │ primary view (full width)                │
│                                   │ secondary views                          │
└───────────────────────────────────┴──────────────────────────────────────────┘
```

- **The banner** is the "once over the world" view (Nolan's idea): a compact, always-visible strip of
  what the pipeline is right now. It shows the row counts through the flow and the column path to
  the model matrix (with the energy method). It shows the models, and the primary result once fitted.
  It is also the map of where you are (DRIVE_RUBRIC §2.4). When a decision changes, its segments veil
  and propagate in order. Pressing a segment focuses the stage on that segment's full view. Built from
  the cohort / split / design / fit artifacts.
- **The stage** shows exactly one thing, chosen by a focus shared by the Record, the banner and the
  stage:
  ```ts
  type StageFocus =
    | { kind: "option"; decision: Decision; label: string }   // previewing an option (hover/focus/arrow keys)
    | { kind: "finding"; findingId: string }                   // a finding's evidence
    | { kind: "banner"; segment: "rows" | "columns" | "models" | "result" }
    | { kind: "live" };                                        // nothing focused
  ```
  `live` shows the Results once `fit` is fresh. Before that, it shows "your data now": the row flow
  and lineage of the current state. Focus lives in one small context created by `ProjectScreen`
  (`src/state/focus.tsx`, `useStageFocus()`). It is not a global singleton.
- The M0 `PipelinePanel` (the Rows / Columns / Results card stack) is retired. Its jobs move to the
  banner and the stage.

## 11 · The transform player (BLUEPRINT §11.1 — read it)

- **One control.** The *Your data now ⇄ With this choice (preview)* flip plays the storyboard forward;
  flipping back plays it in reverse. It is brisk (≤ 900 ms total) and interruptible, and it drives
  every view on the stage at once. Space flips it. Under `prefers-reduced-motion` it is instant.
- **The storyboard** is the method's own labeled steps, from `view.story` (§12). Step dots sit beside
  the flip; clicking one pauses on that step. With an empty story, the flip is a direct before ⇄ after
  morph.
- **Arrow keys move between options.** Their position on the flip holds, and the stage morphs directly
  between results without replaying the story.
- **Pinned readout.** The headline numbers stay visible in both states (r 0.86 → 0.14; n 21,348 →
  2,943).
- **Save.** Every view has a save menu: *now* · *the current step* · *with this choice* · *before/after
  pair*, as SVG and PNG (2×). Saves use the journal style (DESIGN_LANGUAGE §07: serif, greyscale-safe,
  dash patterns) and carry a caption naming the choice and the basis ("Preview, not recorded: …" for
  an unrecorded option). An interpolated mid-animation frame can never be saved.
- **Morph within a unit; crossfade across units.** When the y-axis unit changes (g → g per kcal),
  crossfade the bars or points instead of morphing them through meaningless values. This was the
  `stage` designer's own flagged weakness.
- Hover-to-preview needs a touch equivalent: tapping an option previews it, and a second tap or the
  record button records it.

## 12 · Backend additions

1. **Storyboards.** Every consequence view gains `story: list[<Kind>Frame]` (default empty), the
   method's intermediate states between before and after, each with a `label` (≤ 8 words):
   `RelationshipFrame {label, points, r, fit_line: {slope, intercept}|null}` ·
   `DistributionFrame {label, hist}` · `LineageFrame {label, lineage}` ·
   `TableFrame {label, columns, rows}` · `RowFlowFrame {label, steps}`.
   Energy adjustment supplies real steps:
   - residual: fit each nutrient on energy (the before points with the fitted line) → residuals
     (centered at 0) → (after: the average added back);
   - partition: energy split into its parts (lineage frames);
   - density and standard: no intermediate steps.
2. **Labeled marks.** Distribution builders emit `marks` (labeled, with the level a by-group cut
   applies to); remove the deprecated `cuts`.
3. **Finding evidence.** `GET /api/projects/{pid}/findings/{fid}/evidence` → `PreviewResult`, the
   views that show why the finding was raised. Energy adjustment: the most energy-correlated nutrient
   against energy. Implausible intake: the energy distribution with the rule's marks. Flags and
   identifiers: a table_focus. Fallback: table_focus plus the distribution of the first numeric
   affected column.
4. **Live-journey fix 1 — blanks that mean "not asked".** `proposals` gains
   `missing: { columns: [{ column, n_missing, share, likely_not_asked: bool, reason }] }`.
   `likely_not_asked` is set for a mostly-blank (≥ 50%) yes/no or medication-like column; the reason
   says why. `SetMissing` gains `drop_columns: list[str] = []`, which removes those columns from the
   predictors before the strategy applies; cohort and design read it. The missing-values question
   offers "Leave out `meds_hbp` and `meds_chol` (blank on 84%), then complete cases" as an option
   beside the plain strategies, and its preview shows the rows that choice saves. On NHANES, complete
   cases dropped 86% of rows because of exactly these two columns.
5. **Live-journey fix 2 — nested nutrients.** The roles artifact gains `nested_in: str|null` per
   column. It holds when the name pattern matches (sugar ⊂ carbohydrate; fat_sat / fat_mon / fat_poly
   ⊂ fat_total) and the component ≤ its parent on ≥ 99% of rows. Substitution moves a parent's
   children proportionally (their shares within the parent are held fixed) and says so in the curve's
   note. Substitution pairs exclude a parent paired with its own child. Tier A test: moving fat_total
   moves fat_sat / fat_mon / fat_poly proportionally and keeps every row's parts summing to its total.
6. **Live-journey fix 3 — say when a family loses to the baseline.** Each fitted model gains
   `baseline: {metric, value}` (the outcome's mean for regression; the class prior for
   classification) and, when it does worse, a concern in plain words ("Predicts worse than the
   outcome's average: CV R² −0.04"). The Results and the shelf show it.
7. **Live-journey fix 4 — an honest uncertainty band.** Remove the row-resampling band: through a
   single fitted model it has zero width for linear models, and a zero-width band reads as
   certainty. `SetSubstitution` gains `n_boot: int = 0`. When it is > 0, the substitution stage refits
   each family on bootstrap resamples of ≤ 2,000 training rows, reports progress, and honors cancel.
   *Superseded by the audit's WP5 (2026-10-02, MA-12):* a 2,000-row refit made the band about twice
   too wide at 10,000 rows. Each refit now resamples every training row (whole units when rows
   repeat) up to 10,000 rows; past that it draws 10,000 and the band is rescaled by √(m/n). The
   default band is the curve ± 1.96 standard errors of 200 refits; 1,000 or more give a
   percentile band. A family's band needs 90% of its refits to succeed, and the caption states
   the rows, the refits and how many succeeded.
   The Results offer "Add an uncertainty band (about N s)", with N measured.

## 13 · Results (the stage's `live` view once fitted)

- **Model comparison.** Per family: CV mean ± sd as a dot and interval on one shared axis, the holdout
  value as a hollow dot, and the baseline as a reference line. Concerns sit beside the family name.
  The order follows the shelf.
- **Coefficients** for linear families: a forest plot of the exposures, with CIs under inference.
  Under prediction, a note says the coefficients are not interpreted.
- **Substitution curves** (PRODUCT_VISION §06c's marks, minus the ilr null overlay, which is
  inboxed): one line per family with dash patterns, the region between curves shaded as disagreement,
  the curve stopping at the support limit with an end marker and its reason, a density strip of
  on-support fraction, effect labels "per 100 kcal at k = 100", and the band when computed. A small
  donor × recipient matrix navigates the pairs (this is the substitution question's control).
- **Every result traces to a decision.** A metric's tooltip names the split and the folds; a curve's
  caption names the energy method and its estimand.

## 14 · Ownership for part 2

| Agent | Owns |
|---|---|
| **backend** | §12 (all seven), with Tier A tests for 4–7; openapi regenerated |
| **record** | `ProjectScreen` layout; `src/state/focus.tsx`; the banner; the Record rebuilt on the Router (`view.interview`), server sentences, the teaching layers (one-liner, *why?* in place, the drawer, term cards); findings with levers and paging (focusing one sets `{kind:"finding"}`); options that set `{kind:"option"}` on hover, focus and arrow keys; mocks for interview, teaching and findings |
| **stage** | `src/components/stage/**`: `<Stage pid view focus onFocus/>`, the transform player, every view (lifted from `explore/stage`), save/export, the Results (§13); the preview and evidence query hooks; mocks for preview, evidence, fit and substitution |

The record and stage agents meet only at `<Stage pid view focus onFocus/>` and the focus context. The
record agent renders a placeholder Stage until the merge.

## 15 · Acceptance

On the real NHANES export: the full journey of §9, with every option previewed on the stage and its
storyboard playing on the flip. The missing-values question offers to leave out the mostly-blank
columns. A fat_total → carb curve moves the fat subtypes proportionally. Boosted trees say they lose
to the baseline. The uncertainty band, when requested, is a real refit band. Every finding focuses
its evidence. The banner tells you where you are. A saved plot is a real state with a provenance
caption. Then change the energy method after fitting and watch the banner, the stage and the Results
re-flow. DRIVE_RUBRIC passes.
