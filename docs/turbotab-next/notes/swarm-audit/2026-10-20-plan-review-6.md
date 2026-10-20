# Sixth review of the swarm plan, two notes after Nolan's first walk (2026-10-20)

A fresh Fable 5.1 reader, read-only, asked to try to refute two notes the orchestrator proposed after
Nolan's first walk on Windows hit two engine failures on a non-default selection answer (c0b6e126,
f3b62418): note A, a crash sweep before the engine is frozen; note B, refusals as a survivorship
filter on claims. Nolan asked for an independent verifier before the change. Its report, verbatim:

## Verdicts

### Note A (crash sweep before the freeze): AMEND
### Note B (refusals as a survivorship filter): AMEND
### The open choice: build the intercept-only fit after the pilot, with the scoping rule; not before, and not never.

Both notes rest on a premise that is true but incomplete, and one fact neither note saw changes both: **the driver cannot answer `set_selection`, or any other declaration.** Details and evidence below.

---

## 0. The premise ("no automated check could have found these"): true as stated, but for a different reason than given, and it hides a larger gap

Verified:
- The path fuzzer offers one answer for selection, `{"kind": "set_selection", "method": "none"}` (`turbotab/core/tests/path_fuzzer.py:193`), and it fits nothing: stage artifacts are mocked (`artifacts_for`, :209-230; the module docstring, :1-3). So even a widened option list would not reach a fit-time or preview-time crash.
- The dry run takes "the app's own first-ranked option as the driver offers it" (`research/swarm/dry_run.py:12-13`) and answers no declaration ("a declaration the driver lists no option for ... is not made", :16-18; its `also_open` handling reads only readings, roles and findings, :262-305).
- CI's browser journeys pick `option-method:none` for the selection line (`turbotab/frontend/e2e/quest-nhanes-predict.spec.ts:138-139`; `quest-own-csv.spec.ts:157-158`).

Not as stated:
- Tests do exercise non-none selections through the server: stepwise refused and its screening exit fitted (`turbotab/core/tests/acceptance/test_explore_repair.py:809-840`), `elastic_net` posted (`turbotab/server/tests/test_outcome_gate_api.py:166`), screening and stepwise in state (`test_explore_3_selection.py:240, 297`; `test_explore_chain.py:165, 270`), the selection question's own preview (`test_previews_wave2b.py:209`). What none did was (a) preview `select_models` with a non-none selection in state, or (b) fit a selection on data where it keeps nothing. The gap is in combinations and data, not in "every walk takes the default". Bug 1 existed since bc9dba8e (2026-10-05) put the select step into the pipeline (`git log -S'selection_step' -- turbotab/core/models/pipeline.py`).

The larger gap, decisive for both notes:
- `the_question` (`research/swarm/driver.py:598-670`) composes options only for the Router's open question (`opts.compose`, `research/swarm/options.py:627-642`, whose `COMPOSERS` are keyed on `interview.QUESTION_KEYS`, `turbotab/core/interview.py:106-114`). Declaration lines appear under `also_open` with their reading `items` only (`line_items`, driver.py:549-556; a quest line's `items` are `ReadItem`s, `turbotab/core/quest.py:1016-1035`). `answer`/`preview` resolve ids against `_composed` and `_ways` only (`chosen`, driver.py:883-891). `set_selection` is a `Declaration` with no Router key (quest.py:559-560), as are `set_intended_use`, `set_scales`, `set_batch`, `set_model_sequence`, `set_sensitivity`, `set_measurement_error`, `set_usual_intake`, `set_multiplicity`, `set_levers`, `set_updating`, `set_validation` (quest.py:548-590). `grep -n declar research/swarm/*.py docs/turbotab-next/plans/DRIVER_PLAN.md` finds no handling; SWARM_PLAN.md never says "declaration". The shell's `option-method:none` ids are built in the frontend (`turbotab/frontend/src/quest/cards/predict/payloads.fixture.json`), not served.
- So neither an analyst nor the §8 sweep ("through the driver") can take or vary a selection. Note A's "each option the driver offers" would have missed both of today's bugs; note B's selection scoping is vacuous for the pilot as planned; and the measurement fixture's graded "batch handling" (§5 item 6, SWARM_PLAN.md:253) is a declaration (`set_batch`) the swarm cannot reach.

---

## 1. Note A: AMEND

Findings:
1. **Not a duplicate, a reordering.** §6 already sweeps "around the default plan" (SWARM_PLAN.md:288-304) at step 3, after the waves (:317), because the conventional-persona plans exist only then. The default-plan half has no such dependency. If the crash sweep's clean run on the freeze commit is the default-plan sweep's measured run, no fit is paid twice; the only extra compute is one rerun per defect fixed.
2. **Where it sits.** It needs packages 2 (replay truncation), 3 (the runner), 4 (the cut-point seal the glucose card fixes, SWARM_PLAN.md:512-514) and 5 (the intake). That is exactly §6 Order step 1, "a scripted fit on each dataset" (:310-311): the crash sweep replaces the scripted fit and yields the same sizing. Not a new package; a deliverable of package 3 plus a line in §6.
3. **It would not have caught bug 1 as written.** A one-at-a-time change around the default previews each option in the default's context. The Models preview failed only after a non-default selection was in state (`turbotab/core/models/previews.py:532-545`, the c0b6e126 diff). §6's refit "revisits the one decision, fits" (:294-296); nothing re-previews later decisions. Add: after each change, preview every later decision's default again. Preview cost only.
4. **"Any uncaught exception is a defect" is not mechanical today.** HTTP 500 is clean (`turbotab/server/errors.py:43-55` handles only `ApiError`, `Refusal`, `ProjectNotFound`, `UnknownColumn`). But a fit-time refusal in words and a scikit-learn crash are both a stage `error` string `"ValueError: ..."` (`turbotab/core/graph.py:980`); `seal.py:1684-1713` classes only IndexError/KeyError/TypeError/... as internal, so "at least one array or dtype is required" reads as a reason. The runner must list every failed fit's message and a reader (or a match against the sentences the engine writes: the contracts' `rung="refused"` relations, `nothing_kept`, `stages/modeling.py:608-693`) must class it. Dozens at most.
5. **Cheaper routes do not reach it.** Widening the fuzzer's options changes nothing (it fits nothing, finding 0). A narrower crash sweep (one dataset) misses data-dependent refusals, which is the point of finding 6 below.
6. **Without declarations in the driver it protects nothing of today's class.** Pair it with package 1 adding declarations to `ask`/`preview`/`answer`.

Replacement text, §6 Order step 1:

> 1. **The crash sweep, which is the sizing:** on each dataset, the sweep around the default plan: every option the driver offers at every decision of the default plan, the declarations included, each previewed, then answered and fitted once, and after each change every later decision's default previewed again. An HTTP 500, or a failed fit whose message the engine did not write, is an engine defect, fixed before the freeze; the sweep runs again on the fixed commit. A refusal is recorded with the engine's reason and the option refused. The commit the crash sweep passes on is the frozen one, and its fits are the default-plan half of the sweep (step 3), so no fit is paid twice. Its run time sets each dataset's fit budget and the sweep's compute estimate.

§10 package 3, add to Contents: "each refit classed as scored, refused (the engine's reason) or failed (an HTTP 500, or a fit error the engine did not write), and the crash sweep's report". §10 package 1, add: "the open declaration lines in `ask`, with their contracts' options, answered and previewed like a question, so the analysts and the sweep reach selection, batch handling, intended use, scales, levers, shrinkage, validation and the sensitivity analyses" (size: a composer per declaration in `options.py` plus tests; I guess 1-2M more on package 1).

---

## 2. Note B: AMEND

Findings:
1. **The survivorship argument is sound, and general.** `Selector._kept` (`turbotab/core/models/variable_selection.py:558-610`): univariable keeps terms with p < 0.25, the elastic net its CV support, stability frequencies at or above tau, backward AIC can drop every term; each is empty exactly when no candidate relates to the outcome on the fold. Screening keeps at least one (`screening_keep` returns max(1, ...), :149-151). VIP keeps at least one unless every VIP equals 1 exactly, so a one-candidate plan under VIP keeps nothing (`> 1.0`, :608): a constructed case the crash sweep may hit. The refusal fires in `transform` (:619-624) in any one training fold, and no fold-level handling exists (the only `except` in tuning.py, :1819, is about a count), so one fold kills the refit.
2. **The sweep's refusal share is not the analyst's.** §6 rescores every refit with 5 x 5 folds (:297-299); the chance that some fold keeps nothing is higher than in the analyst's 5-fold fit. A plan the analyst fitted can be refused in the sweep. Report the sweep's share as the sweep's, apart from §7's refusal rate.
3. **Other refusals with the same effect** (all data-keyed, all censor by regime): stepwise at answer time (`_stepwise_has_rows`, :1006-1030) and its in-fit backstop (:596-605); no training rows (`stages/modeling.py:632`); a one-valued outcome (:651); a batch confounded with the outcome or small groups' intercepts (:684); raw omics values into a linear family (:693); families blocked under a surveyed population (`previews.py:566-569`). So the reporting rule must be general (note B's first sentence is). The intercept-only case is the only one where the refusal is an engine gap rather than the method's own limit; the scoping rule belongs to it alone, and a method's-limit refusal is part of the claim, not a censor.
4. **Scoping suffices for the pilot's own decisions.** The model and the personas are decided on the Estimate fixture (§7:357-364, 396-398), where selection is `not_offered` for the reported model (`variable_selection.py:808-835`); the Predict fixture plants signal, so selection keeps something; a refused plan "counts as neither" (:372-375). And selection is unreachable through the driver anyway (section 0).
5. **The size guess is light.** The fit touches `pipeline.py` (1142 lines: select to scale to model with a zero-column matrix, or a fallback estimator), `tuning.py` (1999, the tuned head), `explain.py` (1896, the equation and tables from `intercept_`, :454, :956, :990), `stages/modeling.py` (3756, coefficient tables), the methods sentence (`decision_sentence`, :1044), the export and the shell's tables; inclusion frequencies already count folds that kept nothing (:653-684). R1-G at 1.46M (HANDOFF.md:89) was a registry change with no fit-path surface; §10's engine packages run 4-12M and package 4's note says guard waves cost about three times a wording wave. A fit-path change on the frozen engine, plausibly 2-4M, would move the freeze.

Replacement text, §7 after "Material, in the loss's units":

> **Refused refits.** A refit the engine refuses has no score. The sweep records it with the engine's reason and the option refused. A decision's materiality on a dataset is reported beside the share of its refits refused and their reasons. The sweep's share is its own (its 5 x 5 folds refuse where the analyst's folds did not) and is kept apart from the analysts' refusal rate.

§9 step 1, add:

> A proposal about a decision states the share of that decision's refits refused in its region and whether the refusals sit in one marker region. A refusal that is the method's own limit (backward elimination where the columns outnumber the fold's rows) is part of the claim. A refusal that is the engine's gap is not: until the engine fits the intercept alone where a selection keeps no predictor, a claim about variable selection is scoped to the datasets where every swept selection kept a predictor, and says so.

§8 Needed, add (after the pilot): "The intercept-only model where a selection keeps no predictor in a fold: the mean or the base rate scored in that fold, the final model and the held-out score included, the folds that kept nothing counted beside the inclusion frequencies."

The open choice: **after the pilot.** It is not needed for any pilot decision (finding 4), it would move the freeze (finding 5), and the claim cycle is a plumbing test. Not never: the honest model is well defined and the refusal is an engine gap, so §9's growth rule for new options does not govern it. Revisit only if the crash sweep, with declarations in the driver, shows the empty selection firing on more than one pilot dataset.

---

## 3. What else these failures show is weak (design-changing)

1. **Declarations are outside the swarm** (section 0). Package 1 must add them or the pilot measures none of the twelve declaration decisions, selection and batch handling among them; §3's reach and §6's "each offered decision" silently exclude them.
2. **Fit-time refusals are untyped and carry no exits.** An answer-time `Refusal` has a code and `exits` (`variable_selection.py:1020-1030`; the driver keeps them, driver.py:681-691). A fit-time refusal is a `ValueError` whose prose is the only signal (graph.py:980) and whose "ways forward" are sentences (`nothing_kept`, :97-104). The record's `result.refused`, §7's stuck-run classing and note A's classing all rest on reading prose, against §7's "applied by a script, not by the orchestrator" (:335-336). A typed fit refusal with exits (a small engine change, package 1 or 4) makes all three mechanical, and lets an analyst act on it instead of being stuck.

---

## 4. Three most doubtful claims, and what I did not check

1. That no driver path answers a declaration. I read `the_question`, `chosen`, `line_items` and the quest line's `items` type, not `opts.resolve` or `wait_open` in full, and did not run the driver; a declaration with `stated` (levers, validation) might surface a press I did not trace.
2. The 2-4M size for the intercept-only fit: from the files and surfaces named, not a design; a fallback estimator at the model step alone could be smaller if the scaler tolerates a zero-column frame.
3. That the crash sweep as written misses bug 1: it depends on how package 3's runner is built; I argued from §6's wording.

Not checked: how the preview route turns a ValueError into a 500 (inferred from errors.py's handler list); which selection method Nolan chose and on how many folds it kept nothing; a Bang refit's wall time; `.github/workflows/v2.yml` beyond its acceptance comment; the five review notes beyond a grep for declarations and refusals (none mention either).

---

**The orchestrator's check of doubt 1** (2026-10-20): `COMPOSERS` (`research/swarm/options.py:595-620`) holds the 24 Router keys from `lens` to `open_seal` and no declaration; `chosen` (`driver.py:883-891`) resolves only against them. The finding holds. All amendments are adopted in SWARM_PLAN, as are the intercept-only fit after the pilot and the typed fit refusal (in package 1).
