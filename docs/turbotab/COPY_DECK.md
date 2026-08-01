# Copy deck — the Guided door

**Generated in part.** `python docs/turbotab/tools/copydeck.py regen`.
Do not hand-edit the generated sections; edit the source and regenerate.

Every user-facing string in the Guided door, by step and by state, with the condition that triggers it. It exists so copy can be reviewed **without running the app** — a reviewer should not have to drive a server to read a sentence.

## How much of this is generated

The catalogues are **generated**: `features.CATALOGUE`, `selection.METHODS`, `router.plan()`'s questions, and `grain`'s answers, exits and disclosures are all data, so this tool prints what is actually there and they cannot drift.

The refusals, receipts and transcript lines are **hand-assembled**, because they are f-strings raised at roughly 105 call sites across `api.py`, `project.py`, `features.py` and `selection.py`, plus 51 string literals inside `web/index.html`. Extracting those would need either an AST walk that cannot resolve the interpolations or a runtime harness driving every error path — both of which produce a *worse* artifact than transcribing them, because a half-resolved f-string is not reviewable copy.

**That difficulty is a finding, not a workaround.** `GUIDED-013` records it: copy that lives at its raise site cannot be reviewed, translated or kept consistent, and this deck is the symptom rather than the cure.

Against drift, each hand entry carries a probe — a distinctive fragment of the real string — and `copydeck.py check` asserts that fragment is still in the file it came from. **This is weaker than generation**: it catches a string that changed, not one that was added. Said plainly rather than sold as equivalent.

---

### Data & Target · the grain question

*Trigger: the step is reached and a target has been chosen. Never skipped — no confidence makes it moot (constitution §02).*

**Question.** Can one person appear in more than one row?

**Why we ask.** This decides how your held-out rows are chosen. If the same person lands on both sides, your held-out numbers will look better than the model is.

**Who consumes the answer.** The lockbox reads this to decide whether the held-out set is drawn by person or by row, and records it as the seal's stated basis. Multi-file assembly reads the same answer rather than asking again. Answering wrongly does not raise an error — it produces held-out numbers that are optimistic by an amount nothing on screen can show you.

**Options.**

- No, one row per person
- Yes, people repeat
- I'm not sure
- My design isn't described here

The second option opens a follow-up: *which column identifies the person?* — populated from `grain.suggestion`, which offers the name heuristic's candidates first and shape-only candidates after.

### Data & Target · what the user reads after answering

| Answer | Trigger | Copy |
|---|---|---|
| No, one row per person | `set_grain` records `one_row_per_person` | Recorded: one row per person. The held-out rows will be drawn at random, which is the right choice when every row is a different participant. |
| Yes, people repeat | `set_grain` records `people_repeat` | Recorded: people repeat, identified by `<column>`. Whole people will be held out rather than individual rows, so nobody appears on both sides of the split. |
| I'm not sure | `set_grain` records `not_sure` | Recorded: unknown. That is a legitimate answer and the analysis continues — but because the shape is unknown, the held-out rows are drawn by row, and if your rows do repeat people the same person will sit on both sides. Held-out performance would then read better than the model is. Your numbers will be labeled exploratory until this is settled, and you can settle it at any point before training. |

### Data & Target · the contradiction interruption

*A CONSEQUENCE (`DESIGN_LANGUAGE.md` §09): always pushed, and it resolves or is attested — never a dead end.*

| Trigger | Copy |
|---|---|
| The user answers **one row per person** and a column repeats regularly | ``{col}`` has {n} distinct values across {rows} rows, about {each} each. That is the shape of repeated measures, and you answered one row per person. One of those two readings is wrong, and which one changes how the held-out rows are chosen. |
| The user answers **people repeat**, naming a column that is unique per row | You said people repeat, but `{col}` has a different value on every one of its {n} rows. Grouping by it would hold out one row per group, which is the row-level split you were trying to avoid. |
| The user names a column that is not in the table | ``{col}`` is not a column in this table, so the held-out rows cannot be grouped by it. |

**The two exits.** Both travel with the refusal, so an interface cannot render the interruption without its way out.

| Exit | Label | Detail |
|---|---|---|
| `resolve` | Change my answer | Go back to the question and answer it differently. |
| `attest` | My answer is right — the data really is like this | Continue with one row per person. The repetition is recorded as a noted disagreement, and it travels into the methods section as a stated limitation rather than disappearing. |

*The absent-column case carries only the `resolve` exit, and that is correct rather than a dead end: a column that does not exist cannot be attested to.*

### Data & Target · the eligibility question

*Trigger: the grain question has been answered and the seal has not been drawn. Clause §01 fixes that position; the seal is refused until this is settled, and "everyone" is a recorded answer rather than a skip.*

**Question.** Is your study restricted to part of this data?

**Why we ask.** If your research question only applies to certain participants — an age range, a range of outcome values, one site — say so now. It becomes an exclusion criterion in your methods, and those rows are removed before any held-out set is drawn.

**What we are NOT showing you, and why.** We are not showing you the outcome's distribution here. An exclusion that comes from your research question is reportable; one that comes from looking at the data is a different thing, and it belongs later.

**Who consumes the answer.** The exclusion runs before the lockbox draws anything, so the held-out set is drawn from the population you actually studied rather than from a wider one. The count it removes is recorded as a participant-flow line with its reason, which is what a CONSORT or TRIPOD flow diagram needs. Answering after the seal is drawn is not possible — it would mean the held-out rows were chosen from people the study is not about.

**Options.**

- No, the study is about everyone here
- Yes → which column, and what range?

**The evidence beside it, and its caption.** Bounded by §04: this answers *is this data corrupted?* and cannot answer *where should I cut?* — observed min/max, missing count, impossible-value flags and, for a categorical column, the distinct values. No median, no quantiles, no per-value counts.

> Observed range and impossible values only — enough to tell you whether this column is corrupted, not enough to pick a cut-point from.

### Data & Target · what the user reads after answering eligibility

| Answer | Trigger | Copy |
|---|---|---|
| No, the study is about everyone here | `set_eligibility` records `everyone` | No eligibility restriction: all {N} rows are in the study population, and the held-out set is drawn from all of them. |
| Yes, restricted | `set_eligibility` records `restricted` with a column, a range and a reason | {k} of {N} rows were excluded before the held-out set was drawn: {criterion}. {reason} Those rows are gone before anything is held out, so the held-out set describes the population you studied rather than a wider one. |

### Data & Target · what the user reads once the seal is drawn

*Keyed on the recorded basis, so the states constitution §03 insists on stay different sentences — an undetermined seal and a verified cross-sectional one cannot render alike, because they are not the same string.*

| Basis | Exploratory? | Copy |
|---|---|---|
| `cross_sectional` | no | 27 rows (15%) are held out and will not be looked at again until the models are scored. |
| `grouped` | no | 27 rows (15%) from 9 subjects are held out, chosen by subject rather than by row, so no subject appears in both halves. |
| `undetermined` | **yes** | 27 rows (15%) are held out, drawn BY ROW because the data's shape is unknown. This is not a verified clean split: if rows repeat people, the same person is on both sides and held-out performance will read better than the model is. Treat these numbers as exploratory, and answer the grain question when you can. |
| `repetition_found_grouping_abandoned` | **yes** | 27 rows (15%) are held out, drawn BY ROW. Rows do repeat per subject, but there are too few subjects to hold any out whole — so the same subject can appear on both sides and held-out performance will read better than the model is. Treat these numbers as exploratory. |

**After an attested contradiction**, the seal sentence gains: *Note: this split rests on your answer, which disagreed with the shape of the data. That disagreement is on the record and belongs in the methods section.* — and the seal is marked exploratory.

### Features · the two questions

**`choose_features`** — Are there quantities your question is really about?

*Trigger: the Features step is reached with a target chosen.*

**Why we ask.** A ratio or an interaction you already reason about clinically is usually a better feature than the columns it came from. Anything built here from one row at a time is applied now and shown to you; anything that learns from the column's distribution is recorded and fitted inside the training folds instead.

**Who consumes the answer.** Row-local columns are added to the working table immediately and every later step sees them, so `ml.dataset_profile` profiles them and the models receive them as ordinary columns. Distribution-dependent ones are stored as a spec that the per-model preprocessing pipeline fits inside each training fold — nothing is computed over the held-out rows. Adding or removing a column marks every downstream result stale.

**Options.** build a feature · skip this step

**`choose_selection`** — Should the models be given every column, or a chosen subset?

*Trigger: the Features step is reached with a target chosen.*

**Why we ask.** Narrowing to the strongest features can help a small study. The catch is that choosing them using all your data lets the held-out rows influence which columns exist — so the choice is recorded now and made again inside each training fold.

**Who consumes the answer.** The answer becomes a selection spec on the project, which the per-model pipeline reads and refits per training fold. It also becomes a sentence in the methods section naming the method and the timing. Nothing is selected at the moment you answer — a set chosen now would have been chosen with the held-out rows in view.

**Options.** every column · a chosen subset

### Preprocess · the mechanism question, asked per column

*Trigger: the Preprocess step is reached and a column has missing values. Asked BEFORE the strategy, because the answer decides which strategies are legitimate. Never skipped — the app cannot know.*

**Question.** Could a blank in `{column}` mean something?

**Why we ask.** In records collected during care, a missing value is often a decision rather than an accident — a test not ordered because the patient looked well is different from a test that was ordered and lost. If a blank carries information, filling it in throws that information away.

**Who consumes the answer.** The answer decides how this column is handled and what the methods section has to say. 'Yes' routes to an explicit Missing category or an indicator, which keep the signal, and makes plain imputation a blocked choice that needs a typed acknowledgment. It also records the stability assumption — that a blank will still mean the same thing wherever the model is deployed — as a stated limitation, because it may not hold across sites.

**Options.**

- Yes — a blank here means something
- No — these are accidents of collection
- I'm not sure

### Preprocess · the strategies, and why each is where it is

| Branch | Strategy | Label | Executes | Because |
|---|---|---|---|---|
| numeric | `indicator` | Add a was-it-missing column and leave the value blank | now (row-local) | Row-local: the new column is 1 where this row's value is blank and 0 where it is not. Nothing about any other row is consulted. |
| numeric | `impute_median` | Fill with the median | in training folds | Stateful: the median is a fact about the whole column. Fitted inside each training fold, never over the sealed rows. |
| numeric | `impute_mean` | Fill with the mean | in training folds | Stateful: the mean is a fact about the whole column, and a more fragile one than the median — one extreme value moves it. |
| numeric | `impute_mice` | Fill by modeling it from the other columns (MICE) | in training folds | Stateful, and the most so: MICE fits a model per column against the others, so it learns the joint distribution of the training rows. |
| numeric | `leave` | Leave it alone for now | now (row-local) | Nothing is computed and nothing is deferred. Recorded so that 'decided to leave it' and 'never looked at it' are different states. |
| categorical | `explicit_category` | Keep blanks as an explicit `Missing` category | now (row-local) | Row-local: a blank becomes a literal `Missing` token using nothing but that row's own cell, so it can execute now. |
| categorical | `indicator` | Add a was-it-missing column and leave the value blank | now (row-local) | Row-local: the new column is 1 where this row's value is blank and 0 where it is not. Nothing about any other row is consulted. |
| categorical | `impute_mode` | Fill with the most common value | in training folds | Stateful: the most common value is a fact about the whole column, so computing it over the full table would compute it over the held-out rows too. |
| categorical | `leave` | Leave it alone for now | now (row-local) | Nothing is computed and nothing is deferred. Recorded so that 'decided to leave it' and 'never looked at it' are different states. |

### Preprocess · the informative-missingness blocker

*A CONSEQUENCE. Fires when the user has stated the missingness is informative AND chosen a strategy that fills the blanks. `I'm not sure` deliberately does NOT fire it.*

> You said a blank in `{column}` means something, and {strategy} would replace every one of those {n_missing:,} blanks with {filler}. The fact that the value was missing would no longer be in the data at all, and no model can recover it afterward.  If that is what you want, say so and it is recorded as a stated limitation. If it is not, an explicit `Missing` category keeps the blank as its own answer and costs nothing.

**The two exits.** Acknowledgment is TYPED, not a click.

| Exit | Label | Detail |
|---|---|---|
| `explicit_category` | Keep the blanks as their own category | A blank becomes a literal `Missing` value, so the model can use it the way it uses any other level. |
| `attest` | Fill them anyway — I know what these blanks are | Recorded as a stated limitation: the missingness signal is removed deliberately, and the methods section says so. |

### Preprocess · the two refusals that have no exit

| Trigger | Copy |
|---|---|
| The outcome is named inside a MICE imputation scope | The outcome `{target}` cannot be one of the columns the imputation model reads. An imputer fitted with the outcome in scope writes the outcome's own information into the feature columns, so every number scored afterwards is scored against features that already encode the answer. You recorded that this model is for PREDICTING an outcome for a new person, and at deployment that leak has nowhere to come from — the features would carry information the app will not have. So it is not offered as a choice here. (If you were estimating how strongly something is associated with the outcome, the answer would be the opposite one: the outcome belongs in the imputation model, and leaving it out biases the association toward the null. `research/CLINICAL_SURVEY_PACK.md` §A2.) |
| A mechanism is stated informative | *(recorded, not refused)* This analysis assumes that a blank in `{column}` will mean the same thing wherever the model is used as it means here. That assumption is not checkable from this dataset — missingness patterns are a property of how a site collects data, and a model that reads a blank as a signal will read a differently-collected blank as the same signal. |

### Features · the transform catalogue

*Every entry states its own clause-§06 classification and why. Row-local entries execute immediately and post a receipt; deferred entries are recorded and fitted inside each training fold.*

#### Row-local — executes immediately

| Label | Explainability | Why this scope | Receipt / methods sentence |
|---|---|---|---|
| log(x) | low | Row-local: the value computed for a row uses only that row's own cells, so it cannot carry information from any other row — including the held-out ones. | `log({a})` was computed from `{a}` directly; values at or below zero are undefined and become missing. |
| log(x + 1) | low | Row-local: the value computed for a row uses only that row's own cells, so it cannot carry information from any other row — including the held-out ones. | `log1p({a})` was computed from `{a}` directly, which is defined at zero. |
| sqrt(x) | low | Row-local: the value computed for a row uses only that row's own cells, so it cannot carry information from any other row — including the held-out ones. | `sqrt({a})` was computed from `{a}` directly; negative values are undefined and become missing. |
| x squared | medium | Row-local: the value computed for a row uses only that row's own cells, so it cannot carry information from any other row — including the held-out ones. | `{a}` squared was computed from `{a}` directly. |
| x cubed | medium | Row-local: the value computed for a row uses only that row's own cells, so it cannot carry information from any other row — including the held-out ones. | `{a}` cubed was computed from `{a}` directly. |
| 1 / x | medium | Row-local: the value computed for a row uses only that row's own cells, so it cannot carry information from any other row — including the held-out ones. | `1/{a}` was computed from `{a}` directly; zeros are undefined and become missing. |
| A / B | low | Row-local: the value computed for a row uses only that row's own cells, so it cannot carry information from any other row — including the held-out ones. | The ratio `{a} / {b}` was computed row by row; rows where `{b}` is zero are undefined and become missing. |
| A x B (interaction) | high | Row-local: the value computed for a row uses only that row's own cells, so it cannot carry information from any other row — including the held-out ones. | The interaction `{a} x {b}` was computed row by row. |
| A - B | low | Row-local: the value computed for a row uses only that row's own cells, so it cannot carry information from any other row — including the held-out ones. | The difference `{a} - {b}` was computed row by row. |
| Is this value missing? | low | Row-local: the value computed for a row uses only that row's own cells, so it cannot carry information from any other row — including the held-out ones. Whether a cell is blank is a fact about that cell, not about the column's distribution. | A binary indicator was added recording whether `{a}` was missing, so a model can use the fact of the blank as signal. |
| Bin by cut-points I supply | low | Row-local: the value computed for a row uses only that row's own cells, so it cannot carry information from any other row — including the held-out ones. The edges come from the user, not from the data, so no other row is consulted. | `{a}` was grouped into bins at cut-points {edges}, which were specified rather than derived from the data. |
| Encode categories in an order I state | low | Row-local: the value computed for a row uses only that row's own cells, so it cannot carry information from any other row — including the held-out ones. The order comes from the user's knowledge of the variable, not from the data's shape. | `{a}` was encoded in the order {order}, which was stated rather than inferred. |

#### Deferred — fitted inside the training folds

| Label | Explainability | Why this scope | Receipt / methods sentence |
|---|---|---|---|
| Bin into equal-sized groups (quantiles) | medium | The bin edges are quantiles of the column, so every row's bin depends on where the other rows fall. Computing it over the whole table would fit it on the held-out rows too, which is the canonical preprocessing leak — so it is recorded now and fitted inside each training fold. | `{a}` will be grouped into {n_bins} equal-sized bins, with the cut-points computed within each training fold. |
| Bin into equal-width groups | medium | The edges are spaced between the column's minimum and maximum, and both come from the data — so an extreme value in any row moves every other row's bin. Computing it over the whole table would fit it on the held-out rows too, which is the canonical preprocessing leak — so it is recorded now and fitted inside each training fold. | `{a}` will be grouped into {n_bins} equal-width bins, with the range computed within each training fold. |
| Bin by clustering the values | high | The cluster centres are fitted to the column's whole distribution. Computing it over the whole table would fit it on the held-out rows too, which is the canonical preprocessing leak — so it is recorded now and fitted inside each training fold. | `{a}` will be grouped into {n_bins} clustered bins, fitted within each training fold. |
| Encode categories by how common they are | medium | The order is derived from counts across the whole column, so one row's code depends on every other row. Computing it over the whole table would fit it on the held-out rows too, which is the canonical preprocessing leak — so it is recorded now and fitted inside each training fold. | `{a}` will be encoded by category frequency, computed within each training fold. |
| Center and scale | low | The mean and standard deviation are properties of the column. Computing it over the whole table would fit it on the held-out rows too, which is the canonical preprocessing leak — so it is recorded now and fitted inside each training fold. | `{a}` will be centered and scaled using the mean and standard deviation of each training fold. |
| Principal components | high | Components are fitted to the covariance of the whole table, so every component encodes every row. Computing it over the whole table would fit it on the held-out rows too, which is the canonical preprocessing leak — so it is recorded now and fitted inside each training fold. | {n_components} principal components will be computed, fitted within each training fold. |

### Features · selection methods

| Method | Explainability | Methods sentence (the timing IS the copy) |
|---|---|---|
| Mutual information with the outcome | low | The top {n} features by mutual information with `{target}` will be selected within each training fold. |
| LASSO (L1 penalty) | low | Features surviving a LASSO penalty will be selected within each training fold, with the penalty tuned on that fold. |
| Recursive feature elimination | medium | Features will be eliminated recursively down to {n}, refitting within each training fold. |
| Univariate association with the outcome | low | The top {n} features by univariate association with `{target}` will be selected within each training fold. |
| Stability selection over resamples | high | Features selected in at least half of resamples will be kept, with the resampling done inside each training fold. |

*Choosing `scope='train_rows'` (Classic's behavior) rewrites "within each training fold" as "once over the training rows (held-out rows excluded)", so a project can state which happened rather than imply the stronger claim.*

---

### Data & Target · refusals, receipts and transcript lines

| State | Trigger | Copy | Source |
|---|---|---|---|
| upload · unreadable file | `engine.read_table` cannot parse the upload | '{filename}' parsed to {n} rows and {c} columns. There is nothing to diagnose. | `turbotab/engine.py` |
| seal · attempted before the grain question | `POST /decision {kind: seal}` while `project.grain is None` | The grain question comes before the seal: whether one person can appear in more than one row decides how the held-out rows are chosen. | `turbotab/api.py` |
| seal · attempted with no target | `POST /decision {kind: seal}` while `project.target is None` | The held-out set is drawn against the outcome, so the target comes first. | `turbotab/api.py` |
| seal · project-level refusal | `AnalysisProject.seal_lockbox` with no recorded grain | The test set cannot be sealed before the grain question is answered: whether one person can appear in more than one row decides how the held-out rows are chosen. Constitution §01 fixes that order, and §02 is why. | `turbotab/project.py` |
| seal · a second seal attempted | `seal_lockbox` when `barrier_raised` is already true | This project already has a sealed test set. Redrawing it would re-partition the study: rows sealed since upload would become trainable and earlier results would no longer be comparable. | `turbotab/project.py` |
| seal · attempted before eligibility | `POST /decision {kind: seal}` while `project.eligibility is None` | The eligibility question comes before the seal: whether your study is restricted to part of this data decides which rows the held-out set is drawn from. Answering 'the study is about everyone here' settles it. | `turbotab/api.py` |
| eligibility · project-level refusal at the seal | `AnalysisProject.seal_lockbox` with no recorded eligibility answer | The test set cannot be sealed before the eligibility question is answered: whether your study is restricted to part of this data decides which rows the held-out set is drawn from. Constitution §01 puts eligibility between the grain and the seal, and §04 is why — an exclusion applied afterwards would mean the held-out rows came from people the study is not about. Answering 'the study is about everyone here' is a recorded answer and settles this. | `turbotab/project.py` |
| eligibility · asked before the grain | `set_eligibility` while `project.grain is None` | The grain question comes before eligibility: constitution §01 fixes the order as grain, then eligibility, then the seal. | `turbotab/project.py` |
| eligibility · restricted after the seal | `set_eligibility` when `barrier_raised` is true — §04's *permanently off the menu*, routed rather than refused flat | The test set is already sealed, so an eligibility criterion cannot be applied now: the held-out rows were drawn from a population that included the rows you are excluding. Constitution §04 routes this back to the pre-seal question, which needs a re-seal — and a re-seal re-partitions the study. | `turbotab/project.py` |
| eligibility · a restriction with no reason | `set_eligibility(restricted, ...)` with an empty `reason` | An exclusion criterion needs its reason. Participant flow reports how many rows were excluded AND why; a criterion with no reason cannot become a methods sentence, and one that cannot be written down should not be applied. | `turbotab/eligibility.py` |
| eligibility · a restriction with no range | `set_eligibility(restricted, ...)` with no minimum, maximum or values to keep | A restriction needs a range or a set of values to keep. Without one, the honest answer is that the study is about everyone here, which is its own recorded answer. | `turbotab/eligibility.py` |
| eligibility · the criterion empties the study | the criterion keeps zero rows | That criterion removes every row ({n} of {n}). Either the range is wrong or the column is not what it looks like — nothing downstream can run on an empty study. | `turbotab/eligibility.py` |
| grain · restated after the seal | `set_grain` when `barrier_raised` is true | The test set is already sealed, and it was drawn against the grain answer recorded at the time. Changing that answer now would describe a split that was not drawn this way. | `turbotab/project.py` |
| seal · too few rows with an outcome | fewer than 10 rows have a non-null target | Only {n} rows have a value for '{target}', which is too few to hold any out and still have a study left. | `turbotab/engine.py` |
| target · the event level is not defaulted | applying `set_positive_class` with no chosen level | Setting the event needs the level being predicted. There is no default: whether the event is (say) death or survival is the research question, not something the file can say. | `turbotab/api.py` |
| repair · the finding has no automatic fix | `POST /decision {kind: apply}` on a finding whose preview is not applicable | That finding has no automatic repair — it needs a human decision. | `turbotab/api.py` |

### Preprocess · refusals, receipts and transcript lines

| State | Trigger | Copy | Source |
|---|---|---|---|
| missingness · a column with no blanks | `route_missingness` on a column that is complete | `{column}` has no missing values, so there is no missingness to route. Asking about a column that is complete would be the interview inventing work. | `turbotab/project.py` |
| missingness · an unknown mechanism | `declare` with a mechanism outside the three answers | '{mechanism}' is not one of ['informative', 'not_informative', 'not_sure']. The mechanism is asked, never inferred — `not_sure` is a real answer. | `turbotab/missingness.py` |
| missingness · the indicator column name is taken | `route_missingness` with `indicator` when `{column}_was_missing` exists | '{name}' already exists in this table. Remove it first, or the indicator would silently replace it. | `turbotab/project.py` |
| settled · the step was worked | `settle_preprocess()` after at least one column was routed | Missingness settled: {k} column(s) changed now, {n} recorded to be fitted inside the training folds, {m} deliberately left alone. | `turbotab/missingness.py` |
| settled · the step was skipped | `settle_preprocess(skipped=True)` | Preprocessing was skipped; no missingness routing was recorded and every column goes forward as it is. | `turbotab/project.py` |
| settled · why nothing visibly changed | the step is settled and at least one strategy deferred — the honest report of a step whose output is decisions rather than a changed table | Your table looks the same because it is the same. Filling a blank with a median means computing that median, and computing it over every row would compute it over the held-out rows too — so the decision is recorded now and the arithmetic happens inside each training fold, where it can only see training data. What you just did is the part that cannot be automated; what is left is bookkeeping the pipeline does on its own. | `turbotab/missingness.py` |
| settled · nothing was deferred | the step is settled and every strategy was row-local or leave | Nothing was deferred, so nothing is waiting: every answer here either changed the table or deliberately left it alone. | `turbotab/missingness.py` |
| settled · columns still unanswered | the step is settled while a column with blanks was never routed | {n} column(s) with missing values have not been answered yet. | `turbotab/missingness.py` |
| models · the disclosure above the shelf | rendered with the model list, always; not conditional on a poor verdict existing | Every model is available. This order is about your data, not about which models are any good — a model low on this list is one whose concern applies to a table this shape, and you may have a reason it does not apply to yours. Select whatever you intend to train. | `turbotab/models.py` |
| models · the third group's label | the group header, shown even when the group is empty | Not recommended for this data | `turbotab/models.py` |
| models · a low-ranked model was selected | `select_models` with at least one `not_recommended` key; the sentence the methods section carries, not an on-screen warning | {n} of the selected model(s) carry a stated concern for a table this shape: {name} — {the coach's own clause}. Selected deliberately; the concern is recorded so it can be reported rather than discovered. | `turbotab/models.py` |
| models · nothing selected | `select_models([])` | Choose at least one model. Preprocessing is configured per model, so there is nothing to configure until you say what you intend to train. | `turbotab/models.py` |
| models · chosen before the seal | `select_models` while `barrier_raised` is false | Models are chosen after the seal: the shelf is ordered by the shape of your data, and the shape it reads must be the shape the models will actually be fitted on. | `turbotab/project.py` |
| recipe · the rendered skip for scaling | a model whose `requires_scaled_numeric` capability is true; shown where the question would have been | This model measures distances or penalizes coefficients, so a column measured in thousands would dominate one measured in units purely because of its scale. The registry records this as a property of the model, not of your data. | `turbotab/recipes.py` |
| recipe · the rendered skip for not scaling | every other model | Tree-based and rule-based models split on order rather than on distance, so rescaling a column changes nothing they can see. Scaling them is harmless and pointless. | `turbotab/recipes.py` |
| recipe · the variant question was suppressed | `worth_asking` measured the two scalings and found them immaterial; shown as the reason no question appeared | σ/IQR varies by {pct} across {n} numeric columns — close to the constant 1.35 a Gaussian column gives, so the two scalings differ by roughly one global factor and no scale-equivariant model can tell them apart. | `turbotab/recipes.py` |
| recipe · the variant question was raised | `worth_asking` measured the two scalings and found them material on this data | σ/IQR varies by {pct} across {n} numeric columns — heavy tails in some columns and not others, so standard and robust scaling would weight the features differently against one another and the choice changes the fit. | `turbotab/recipes.py` |
| recipe · a shared setting borrowed from another model | `resolved_recipes` under the uniform answer, on every model other than the one the settings came from | Applied to every model because you chose one shared preparation; this is {model}'s setting. | `turbotab/project.py` |
| preparation mode · the question | asked once, after the models are chosen | Should each model get the preparation it needs, or should they all get the same preparation so the comparison is about the models? | `ml/router.py` |
| preparation mode · why we recommend per-model | shown with the question; states the recommendation AND what it costs, because a recommendation with no cost attached is advice the reader cannot weigh | Per-model is the usual choice and what we recommend: a model handicapped by preparation it does not suit is not informative either. The cost is that a difference between two models then reflects the model and its preparation together — so if you pick it, that caveat is written into your methods section automatically. | `ml/router.py` |
| preparation mode · per-model chosen | the methods sentence recorded on the decision | Each model receives the preparation it needs: scaling where the model measures distances or penalizes coefficients, none where it splits on order. | `turbotab/project.py` |
| preparation mode · uniform chosen | the methods sentence recorded on the decision; a recorded answer, because choosing to hold preparation constant is itself a methods sentence | Every model receives the same preparation, so differences between them are differences between the models rather than between their pipelines. | `turbotab/project.py` |
| preparation mode · the caveat, into Limitations | automatically, on choosing per-model; never on uniform | Models were compared under per-model preprocessing: each was given the preparation appropriate to it rather than a single shared pipeline. A difference in performance between two models therefore reflects the model and its preparation together, and the two cannot be separated from these results alone. This is the usual choice — a model handicapped by preparation it does not suit is not informative either — and it is stated so the comparison is read for what it is. | `turbotab/project.py` |

### Explore · refusals, receipts and transcript lines

| State | Trigger | Copy | Source |
|---|---|---|---|
| trim · the label saying what it is NOT | every successful `trim_training_rows`; §04's two objects look identical in a spreadsheet, so the trim says which one it is | This narrows the TRAINING rows only. It does not change who your study is about: the held-out rows are untouched, N is unchanged, and nothing here belongs in participant flow. If you meant to restrict the population the model is for, that is the eligibility question, it is asked before the seal, and it does change N. | `turbotab/obligations.py` |
| trim · attempted before the seal | `trim_training_rows` while `barrier_raised` is false | A robustness trim is post-seal by definition: it narrows the training partition, and there is no training partition until the test set is sealed. Before the seal, narrowing the study is an eligibility criterion — a different object (§04), asked as a different question, and it changes N. | `turbotab/project.py` |
| trim · with no stated reason | `trim_training_rows` with an empty `reason` | A trim's reason is what the report has to print beside the breakdown. Without it the disclosure would say that some rows were outside a range nobody can explain. | `turbotab/obligations.py` |
| trim · with no bounds | `trim_training_rows` with neither a minimum nor a maximum | A trim with no bounds narrows nothing, so there is no extrapolation to disclose. | `turbotab/obligations.py` |
| trim · the receipt, which is also an obligation | a train-only trim succeeds; the sentence goes in the transcript AND becomes what the report must discharge | The model was fitted on training rows with {range} ({reason}). {k} of {n} held-out rows fall outside that range, so performance must be reported separately for in-range and out-of-range rows rather than as one number. | `turbotab/obligations.py` |

### Features · refusals, receipts and transcript lines

| State | Trigger | Copy | Source |
|---|---|---|---|
| transform · a stateful one was applied | `features.apply` on any entry whose scope is `stateful` | '{label}' learns from the column's distribution, so applying it to the working table now would fit it on the held-out rows too. It is recorded as a decision and fitted inside each training fold instead. {because} | `turbotab/features.py` |
| transform · a row-local one was declared | `features.declare` on an entry whose scope is `row_local` | '{label}' is row-local, so it executes immediately rather than being declared. Use apply(). | `turbotab/features.py` |
| transform · a capability this door declines to build | `features.get` on `polynomial` or one of its four aliases (`poly`, `polynomial_features`, `polynomialfeatures`, `interactions`, `all_interactions`) | Generating a whole polynomial basis is not offered here, and the reason is a routing answer rather than a missing feature.  Two arguments, and they are different. First: degree 2 over ten numeric columns produces 55 new terms — 10 squares and 45 pairwise products — that nobody chose one at a time, each carrying explainability cost. Mass generation is the opposite of this interview's premise. Second: on a 140-row study those 55 terms are p/n ≈ 0.39, which is the overfitting regime; the expansion is most attractive on exactly the small studies where it does the most harm.  If your question really is about interactions, the route is a model that captures them rather than columns that manufacture them. Trees and gradient boosting get interactions for free, so this is a model choice at the modeling step, not a feature choice here.  If you want ONE interaction because you already reason about it clinically, that is what `product`, `ratio` and `difference` are — named, chosen, and each posting its own receipt. | `turbotab/features.py` |
| transform · the new column name is taken | `features.apply` when the generated name already exists | '{name}' already exists in this table. Remove it first, or the new column would silently replace it. | `turbotab/features.py` |
| binning · no cut-points supplied | `bin_fixed` applied without `edges` | Binning by supplied cut-points needs at least two edges. Without them the edges would have to come from the data, which is a different transform and defers. | `turbotab/features.py` |
| encoding · no order supplied | `ordinal_declared` applied without `order` | Encoding in a stated order needs the order. Deriving it from the data is a different transform and defers. | `turbotab/features.py` |
| remove · a source column was named | `remove_feature` on a column this step did not create | '{column}' was not created here, so removing it is not this step's to do. Only engineered columns can be removed. | `turbotab/project.py` |
| deferred preview · why there are no values | `features.preview` on any stateful entry | Not computed here. This transform learns from the column's distribution, so it is fitted inside each training fold at modeling time — there is no single set of values to show before then. | `turbotab/features.py` |
| selection · the outcome offered as a candidate | `selection.declare` with the target among the candidates | '{target}' is the outcome and cannot also be a candidate feature: selecting the target predicts it perfectly. | `turbotab/selection.py` |
| selection · a scope outside the two permitted | `selection.declare` with any scope but `train_rows` / `train_folds` | scope must be 'train_rows' or 'train_folds'; got '{scope}'. There is no third option, and in particular there is no option that fits on the whole table. | `turbotab/selection.py` |
| selection · a spec arriving with a chosen set | `set_selection` with `spec['selected']` populated | This selection spec carries an already-chosen feature set. Selection is performed inside the training folds at modeling time; a set chosen now would have been chosen using the held-out rows. | `turbotab/project.py` |
| selection evidence · nothing was withheld from the ranking | `selection.evidence` where the mask excludes no row — before the seal, or with no mask at all | Nothing was withheld from this ranking, so it saw every row in the table. Treat it as exploratory. | `turbotab/selection.py` |
| selection evidence · the normal case | `selection.evidence` where the seal withheld rows from it | Ranked on training rows only, and not applied. What is actually selected is refitted inside each training fold, so this ordering is indicative rather than the answer. | `turbotab/selection.py` |
| selection · ranking before a target is chosen | `GET /selection/evidence` with no target | Ranking features needs the outcome first. | `turbotab/api.py` |
| receipt · a column was removed | `remove_feature` succeeds | The engineered column `{column}` was removed. | `turbotab/project.py` |
| settled · the step was worked | `settle_features(skipped=False)` | Feature work settled: {n} column(s) added now, {d} transform(s) recorded for fitting inside the training folds[, and a selection spec recorded]. | `turbotab/project.py` |
| settled · the step was skipped | `settle_features(skipped=True)` | Feature work was skipped; the original columns go forward unchanged. | `turbotab/project.py` |
| selection · every column, recorded | `set_selection(None)` | No feature selection: every candidate column is offered to the models. | `turbotab/project.py` |

### Cross-step · refusals, receipts and transcript lines

| State | Trigger | Copy | Source |
|---|---|---|---|
| grain · the answer is recorded | `set_grain` succeeds (transcript line, distinct from the disclosure) | Asked whether one person can appear in more than one row; the answer recorded was: {said}. | `turbotab/project.py` |
| seal · the transcript line | `seal_lockbox` succeeds | A test set of {n} rows was sealed before exploration and held by row label, on the basis '{basis}' ({source}). | `turbotab/project.py` |
| identity · a sealed label went missing | `assert_identity_intact` after a repair renumbered rows | {n} sealed row label(s) are no longer in the table (e.g. {labels}). Something renumbered the rows after the test set was sealed, so the quarantine no longer refers to the rows it was drawn from. | `turbotab/project.py` |
| upload · repeated row labels | `from_dataframe` on a frame whose index has duplicates | '{name}' has repeated row labels ({n} of {total}). Row identity in this project is the index label, so repeated labels leave no way to say which row a decision refers to. | `turbotab/project.py` |

### Empty and terminal states

*Assembled by hand — these live in `web/index.html` as markup, which is the least reviewable place copy can live (`GUIDED-013`).*

| State | Trigger | Copy |
|---|---|---|
| No project yet | first load | Drop a CSV to begin. |
| A clean file | `diagnose` returns no findings | This file reads as a clean table. |
| No features engineered | the Features step, before any transform | Nothing added yet. The original columns go forward unless you build something. |
| Selection not set | the Features step, before a selection answer | Every column will be offered to the models. |
| Findings stale | any answer changed underneath computed findings | These were computed under an earlier answer. |
| Downstream stale | a feature was added or removed | Results computed before this change no longer describe the current feature set: {why}. |

