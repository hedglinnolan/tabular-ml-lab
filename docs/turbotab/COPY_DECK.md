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
| grain · restated after the seal | `set_grain` when `barrier_raised` is true | The test set is already sealed, and it was drawn against the grain answer recorded at the time. Changing that answer now would describe a split that was not drawn this way. | `turbotab/project.py` |
| seal · too few rows with an outcome | fewer than 10 rows have a non-null target | Only {n} rows have a value for '{target}', which is too few to hold any out and still have a study left. | `turbotab/engine.py` |
| target · the event level is not defaulted | applying `set_positive_class` with no chosen level | Setting the event needs the level being predicted. There is no default: whether the event is (say) death or survival is the research question, not something the file can say. | `turbotab/api.py` |
| repair · the finding has no automatic fix | `POST /decision {kind: apply}` on a finding whose preview is not applicable | That finding has no automatic repair — it needs a human decision. | `turbotab/api.py` |

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
| selection evidence · no training mask supplied | `selection.evidence` called without `train_mask` | No training mask was supplied, so this ranking saw every row. Treat it as exploratory. | `turbotab/selection.py` |
| selection evidence · the normal case | `selection.evidence` with a training mask | Ranked on training rows only, and not applied. What is actually selected is refitted inside each training fold, so this ordering is indicative rather than the answer. | `turbotab/selection.py` |
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

