# The crosswalk

What TurboTab v2's engine needs you to decide, when it asks, and what the canvas shows; then, after the fit, what each result is and where it can go. This is the input to the quest-log redesign (`HANDOFF.md`, "The structure (Nolan, 2026-10-06)").

Written 2026-10-08 against `turbotab-next` at `2627066d`, and patched the same day after the completeness critic's review (the critic's notes not taken are listed at the end). Every item is in [`crosswalk.json`](crosswalk.json), with `engine_source` naming the file and symbol behind each claim. [`SIZING.md`](SIZING.md) turns the gaps into work packages.

## How to read this

- **Stages and labels.** Each of the seven stages lists its objectives under the ruled labels:
  - **Decide:** a question you answer, or an open noticing you decide or dismiss;
  - **Confirm:** a default set for you, swept once at the end of the stage, listed only when another choice would change a number;
  - **For the record:** collapsed, and not counted toward progress;
  - **Shown:** previews, views and refusals that draw on the canvas but are not objectives.
- **Status, in plain words.** Every line carries one status:
  - *on screen:* the engine has it and today's app draws it;
  - *engine only:* the engine has it and no screen reads it;
  - *partial:* some of it exists;
  - *missing:* none of it exists;
  - *new scope:* added to v2 on 2026-10-05 or 2026-10-07 and not built.

  When a line applies only to some goals or lenses, they follow the status.
- **Goals.** Goals use your words: Describe, Estimate (an effect; the engine's `inference`) and Predict (`prediction`).
- **Plain words.** Card titles avoid "exposure", "confounder" and "estimand" (DoD, 2026-10-07): *what you study* stands for the exposure, *what you adjust for* for the confounders. Square brackets mark words filled from your data, as in "[sugar]" (`crosswalk.json` writes them in angle brackets).
- **Ids.** Every line ends with its id in `crosswalk.json`. The few claims made here outside an item cite `file:symbol` inline.

## At a glance

- **Merged.** The seven stage maps listed 891 entries. They fold into **708 items**, and 177 of those absorbed an entry from another stage. Each item lives in one stage: the stage where you decide it.
- **Every noticing has a home.** All 367 noticings in the understanding catalogs are placed:
  - 266 are items of their own;
  - the other 101 are carried by the engine question or noticing that already does their job. For example, `shared-unit-repeats` is the grain question.

  Each one carries its family (S1 to E3), and the 17 family checks that run under every lens are items too (`sentinel:S1` to `sentinel:E3`).
- **The engine is far ahead of the screens.** Of the 462 Decide and Confirm objectives:
  - 40 can be answered on today's screens (9%);
  - 79 exist only in the engine;
  - 188 are partly built;
  - 127 are missing;
  - 28 are new scope.
- **The gap differs by half:**
  - before the fit, it is mostly wiring and noticings;
  - after the fit, the engine already computes nearly every result, and almost none is drawn. Of the 147 items in Results and Write-up, 70 are engine only and 4 are on screen.
- **First look decides almost nothing itself.** It shows 230 noticings and decides none of them:
  - Who's in decides 81;
  - Models decides 134;
  - Your data and Your question decide 15.

  First look's own objectives are opening the highlights, and the few blockers that stop everything.
- **Models is the heaviest stage:** 231 items, with 132 Decide and 63 Confirm. Most are noticings that land on 14 cards. A journey sees far fewer: the reference journeys answered 2 to 8 engine cards in Models (below, "The reference journeys against the load caps").
- **Order conflicts.** In 20 places, the engine's order and the ruled stage order disagree (below). Each has a proposed fix. Several of the questions and rulings below come from them.

## Six questions for you

**Ruled by Nolan, 2026-10-08:**
1. Under Estimate, the outcome beside a column is **hidden until the lock**. Recommended.
2. Your data shows **one ledger with a Confirm sweep**, and Decide asks only the readings that change a number. Recommended.
3. Describe **can start with no single outcome**, through a new answer to the outcome question. Recommended.
4. **All 367 noticings ship in v2.0.0.** This was NOT the recommendation: it keeps the full rollout of about 236 units, so the sizing stays at about 734 units.
5. Blanks are **split at the stage line**: who is kept is decided in Who's in, and how kept blanks are filled in each track's Models. Recommended.
6. A shared-step change after first results under Predict keeps the earlier scores as a **read-only row**, labeled "revised after first results". Recommended.

The questions as put follow.

Only questions the crosswalk cannot settle. Each has a recommendation.

**1. Under Estimate, may First look show the outcome beside another column before the plan locks?**

`FIRST_LOOK_BRIEF.md` §9 Q1 left this open, and `HANDOFF.md` records no ruling. The brief offered two answers:
- **A counted door** (the brief's recommendation): every view of the outcome beside a column is recorded (`decision:view_outcome`) and disclosed as a forking path, and is never pointed at or ranked.
- **Hidden until the lock** (the brief's strict alternative, STROBE's initial-data-analysis line read literally, Heinze et al. 2024): before the lock, only the outcome alone opens (its distribution, blanks and event count; O1 in the brief's §6.1). The outcome beside a column (O3) opens after the lock, labeled exploratory.

**Why it matters.** First look comes before Models, where the form of what you study and the adjustment set are chosen. A view of the outcome beside a predictor informs "nothing by eye" there (`WHAT_EXPLORATION_MAY_DECIDE.md` §2, the predictor–outcome row). Yet the engine lists the door's curves in plan order, what you study first (`noticing:explore::relationship`). The engine also computes the outcome views only after Who's in: the explore stage requires `split` and reads the cohort (`stages/__init__.py`, `Stage("explore", …)`).

*Recommendation:* hidden until the lock, under Estimate.
- The outcome alone opens after Who's in, on the rows analyzed, recorded: it informs data quality and the effective sample size (`WHAT_EXPLORATION_MAY_DECIDE.md`, the outcome-alone row).
- The outcome beside a column waits for the lock.
- Under Predict, both open after the seal on the training rows, recorded, as today.
- Several goals follow the strictest rule, so a paper with an Estimate track hides the outcome beside a column until that track's lock.

It removes a door the calm design would have to explain, and the IDA paragraph can say "no association with the outcome was examined before the plan was fixed" (`FIRST_LOOK_BRIEF.md` §6.5). If you prefer the door, its list follows the table's column order and never singles out what you study.

**2. How much of "what each column is" does Your data ask?**

Two rulings conflict:
- The ruled scope puts units, codes and kinds in Your data.
- The engine asks each reading where it is first used (`ask.py:CONSUMERS`). That follows BLUEPRINT §14.2 ("ask only where it matters") and §14.3 ("the consumer sets the scope").
- The engine also refuses the roles answer until Who's in is answered (`sequence.py:_answers_in_order`).

*Recommendation:* Your data shows one ledger of every column with its proposed reading.
- **Decide:** it asks only the readings that change a number on some path under this lens. These are the energy unit and days, body-measure units, sex coding, identifiers, design columns and value meanings.
- **Confirm:** roles, and codes or amounts, sit in Your data's Confirm sweep, settled in one tap.
- **Completion:** the engine records the roles answer itself once the readings are settled, so the roles question never shows.
- **Later columns:** the ask card returns at a later card only for a column that enters the model later.

**3. Can Describe start without an outcome?**

Dietary patterns, clustering, Bland–Altman between two methods and usual intake have no single outcome. The ruled order is "the outcome, then the goal", and the engine requires an outcome: `cohort`, `seal_plan` and `explore` declare `requires=("target", …)` (`stages/__init__.py`).

*Recommendation:* give the outcome question an answer, "No single outcome: I want to describe these data". It would:
- set the goal to Describe;
- skip the event, kind and follow-up;
- drop the participant flow's "outcome recorded" step.

**4. How many of the 367 noticings must ship in v2.0.0?**

The definition of done requires that open noticings be decided before the lock or the seal, and that the road start with the understanding layer's phase 0. It does not say that all 367 ship. Today:
- 111 noticing items are missing;
- 175 are partial;
- none is wired end to end.

The full rollout (T2 and T2+ in `SIZING.md`) is about 216 units, the largest part of the road.

*Recommendation:* v2.0.0 ships:
- the thread machinery;
- the T1 blockers and the 17 family checks (`sentinel:*`, placed in the table "Thread families and where they land");
- every noticing that fires on the twelve reference journeys.

Each remaining noticing is either stated as one methods sentence, or listed in INBOX for v2.x, and the coverage test (U13) enforces that. This removes T2+ (about 108 units) for about 14 units of sentences and INBOX lines.

**5. When a paper has several goals, where are blanks filled?**

The shared stages run once (HANDOFF, 2026-10-07). But the goals' rules on blanks differ, so one shared Who's in answer cannot fit them all:
- multiple imputation is refused under Predict, and a single fill is blocked under Estimate (`decisions.py:_missing_fits_the_purpose`);
- multiple imputation compatible with the analysis model reads Models answers, the forms and the energy model (disagreement 7).

*Recommendation:* split the question at the stage line.
- **Who's in, once for every track:** who is kept and what each blank means. That is complete cases on named columns, or keep every row; the columns left out; the blanks that mean "not asked" or "below detection". Complete cases is valid for every goal.
- **Each track's Models:** how the kept blanks are filled. Estimate uses multiple imputation compatible with its own model. Predict uses a fill in each training fold (its recipe, `RECIPES_AND_TUNING.md` §2.3). Describe estimates each quantity on the rows that hold its values, with the share of blanks stated.

This also resolves disagreement 7 for the fill: it is chosen where the model it must match is known. *The alternative* asks the fill once per track inside Who's in, which breaks "the shared stages run once". SIZING D2 is sized for the recommendation.

**6. Under Predict, what does "keeps the earlier version" mean for a change to a shared step (the trunk)?**

Two rulings meet here:
- **2026-10-07:** after results, prediction keeps the earlier version, labeled "revised after first results" in the comparison only.
- **2026-10-06 (DoD §5):** "shared-step changes after scores kept as versions" moved to v2.x. The shared steps are the missing-values answer, the levers, the energy model and the forms (`crosswalk.json`, `stages[results].boundary_issues`).

So in v2 a recipe or tuning change keeps a version (RECIPES RT-7), but a shared-step change can only be disclosed (`decisions.py:disclose`).

*Recommendation:* for a shared-step change after first results, the comparison keeps the earlier fit's scores as a read-only row labeled "revised after first results". The row is read from the earlier fit artifact, which the engine never deletes (BLUEPRINT §4: a stale artifact is "served with `fresh: false`, never deleted"). It cannot be declared final, scored on the held-out rows or exported as a model. That meets "never overwrite silently" without bringing the v2.x versions back. *The alternatives:* disclosure only, so the earlier scores leave the table; or shared-step versions in v2 (about an L on top of RT-7).

## Settled here, not asked

Rulings already made, and the methods calls the crosswalk owns. Each cites its source.

- **Under Predict, the open-noticings gate is at the opening of the held-out rows.** This was question 1 of the first draft, and it was already ruled. UNDERSTANDING_LAYER §7 ruling 1 (2026-10-06): every open thread that feeds the plan "must be decided or dismissed before estimates appear, and under prediction before the seal opens". §2.6 places the card "before the held-out rows are scored (`open_seal`)".
  - So the Predict gate is `gate:open-noticings-before-seal`, just before `q:open_seal`.
  - The engine's guard on the columns the draw reads stays at drawing (`seal.py:_the_draw_reads_settled_values`).
  - `gate:open-noticings-before-lock` serves Estimate and Describe only; under Predict it was a duplicate.
- **Fit, and when estimates appear** (disagreement 12).
  - **Under Estimate and Describe,** the server serves no estimate stage (`estimand.ESTIMATE_STAGES`) before the track's plan is locked. You lock it by pressing Fit on the analysis flowchart, after the open-noticings gate and the Confirm sweep. The lock is the existing system record (`lock_plan`, recorded today by `server/service.py:_lock_when_shown`), so no new decision kind is needed.
  - **Under Predict,** nothing locks. Results opens when you press Fit.
  - **The fits themselves** compute as today. A short fit may already be done when you press Fit; a fit expected to take over about 2 minutes waits for the press, in the scheduler.
  - **Why this keeps both rulings.** `RECIPES_AND_TUNING.md` §4.4 says pressing the button "is a job command, not a decision" and "the hold is in the scheduler, not a stage requirement". Nolan's ruling 2 says "BLUEPRINT §4's 'live' rule changes for long fits only". Computing stays live. What changes is when estimates are served, which `estimand.served_gate` already does while a question an estimate rests on is open.
  - This replaces the first draft's fix ("a Fit record, a system kind beside `lock_plan`, holding every estimate stage"), which contradicted §4.4. `RECIPES_AND_TUNING.md` draft 3 states it (SIZING P0.2).
- **Describe has a gate and a lock.** Describe reports estimates too: weighted means and prevalence by group, the usual-intake distribution, trends. A domain, a weight or a screen chosen after seeing them is the same forking path (Gelman & Loken 2013). So a Describe track follows Estimate's rule:
  - its open noticings are decided or dismissed before its first estimate (`gate:open-noticings-before-lock` now lists Describe);
  - its first estimate locks the track's plan;
  - a later change runs as a labeled secondary beside the first result (`decision:post_lock_secondary`).

  SIZING D1 builds it.
- **Several goals: the order, the locks and the rows.**
  - **Order:** Estimate, then Describe, then Predict (`default:track_order`). An estimate shown under Predict locks a later Estimate plan at once (`server/service.py:_lock_after_prediction`). Describe's estimates by group can pair the outcome with what you study, which would be an outcome view before an Estimate lock (`FIRST_LOOK_BRIEF.md` §6.1, O3).
  - **Locks are per track, in v2.** The first draft said per-track locks wait for v2.x, which cannot hold. With one project-wide lock (`decisions.py:ProjectState.plan_locked`), every Describe or Predict decision after the Estimate lock would be marked "after the estimates were seen" (`decisions.disclose`). So the lock, the after-estimates mark and the record of estimates shown under prediction (`SHOWN_UNDER_PREDICTION`) are scoped to the track. Running tracks in parallel stays v2.x (DoD §5).
  - **Rows.** Estimate and Describe read every analyzed row, because the seal is purpose-scoped (BLUEPRINT §12 ruling 3). A later Predict track on the same outcome therefore cannot claim that its held-out rows were never read (`FIRST_LOOK_BRIEF.md` §6.3). So when an earlier track has read the Predict track's outcome:
    - the Predict track validates by resampling the whole procedure (repeated cross-validation with its correction, or the bootstrap; `result:bbc_cv`) and draws no held-out rows;
    - its methods state that its choices followed the earlier track's results, as hand-set levers are stated: outside the corrected score (`result:hand_levers`).

    A Predict track whose outcome no earlier track reads draws its held-out rows in the shared Who's in, as usual.
  - SIZING D2 builds the track id, the scoped lock and mark, and this row rule.
- **A direct effect is "Not available yet".** Mediation is out of v2 (HANDOFF; DoD §5). A direct effect is a mediation estimand: it needs the mediator–outcome confounders and the interaction (`estimand.direct_questions`). So the effect card offers the whole effect, and "only the direct part" reads "Not available yet", with its reason and an exit to the whole effect (`q:estimand`). Mediators are still recognized and kept out of the adjustment set.
- **Methods text changes only through its decisions** (methods editing). Each methods sentence is the record's own, and every number in it must trace to the record (DoD gate 7).
  - A sentence changes when its decision changes: its "change" link opens the card that asks it (disagreement 19).
  - The author adds text in author-only paragraphs. They export as `\todo` until written, and pass the manuscript gate once written.
  - Results wordings stay as ruled: a draft, or your own.
- **The rail is a view, not a stage.** The rail shows the manuscript in every stage, but only Write-up's own objectives fill Write-up's segment: what the export still waits for, the merge of tracks, the placement review, the Discussion drafts and the export. Author-only text never counts and never blocks.
- **Multivariate regression calibration is stated, not asked.** The contract implies it whenever the model holds two or more error-prone intakes, and refuses calibrating one at a time (`methods/calibration.py`, relation `multivariate`; DoD amendment of 2026-10-03). The measurement-error card says which form runs (`decision:set_measurement_error`).

## The stages

### 1 · Your data

The file, its joins and stacking, the codebook, the lens, which way round the table is, and what each column is.

The order follows the engine's own dependencies:
1. **The file and its joins.** A join rebuilds the table, every stage reads it, and a join is refused once the held-out rows are drawn (`seal.py:DECISION_A`).
2. **The lens.** It is first in `interview.py:QUESTION_KEYS`, and no noticing exists before it: the findings stage requires it.
3. **Orientation.** It must come before the outcome (`sequence.py:_orientation_turns_before_the_target`).
4. **The codebook.** It comes before any reading, because it settles many readings at once.
5. **The value repairs,** in the order their SQL composes (`repairs.py`, family priority).
6. **The column readings.** The roles question among them is disagreement 1 below.

**Decide, in order** (46).

1. Which table do you want to analyze? `q:upload` (on screen)
2. Does another file belong with this one? Join it on an ID both files share `decision:join_files` (engine only)
3. The join left out a group of people, for example a whole survey cycle `noticing:linkage-quality` (partial)
4. Stack files with the same columns (several NHANES cycles) into one table `decision:stack-files` (missing; dietary, clinical, survey)
5. What kind of measurements are in this table? `q:lens` (on screen)
6. The lens you chose and the table disagree `noticing:lens-contradiction` (on screen)
7. Which way round is this table: is each row a sample, or is each row a gene or metabolite? `q:orientation` (partial; metabolomics, genomics)
8. Which column names the features, and which columns describe them (m/z, retention time, gene IDs)? `decision:set_feature_table` (engine only; metabolomics, genomics)
9. Do you have a data dictionary for this table? `decision:import_codebook` (engine only)
10. Your codebook says one thing and the values say another (for example, height in meters with a median of 166) `noticing:codebook-contradicted` (engine only)
11. For each problem found in a column: repair it, keep the values as they are, or hold it for the question it belongs to `decision:finding-disposition` (on screen)
12. `crp` writes '<0.20' for values below the detection limit; they are your lowest values, not blanks `noticing:below-detection` (on screen; metabolomics, clinical)
13. `bmi` arrived as text but is mostly numbers ('5,4' may be a decimal comma) `noticing:text-numbers` (on screen)
14. Zeros arrived as 5.4e-79 from the SAS transport file `noticing:sas-transport-zeros` (on screen)
15. 999, 7777 or 7/9 hide in a numeric column as if they were values `noticing:missing-value-codes` (on screen)
16. Total energy is in kilojoules, not kilocalories `noticing:energy-in-kj` (on screen; dietary)
17. `smoker` stores yes and no as text; which level counts as 1? `noticing:two-level-text` (on screen)
18. Some cells hold infinity `noticing:infinite-values` (on screen)
19. Dates read both month-first and day-first; which is it? `noticing:ambiguous-dates` (on screen)
20. `sex` arrives as Male, male, M, Female, female and F: six spellings, two sexes `noticing:category-spellings` (partial)
21. Some rows are not data (a repeated header, a 'Total' row), use placeholder dates (1900-01-01), or stand for many people `noticing:file-integrity` (partial)
22. `sample_type` has a level, pooled_qc, whose 8 rows are one pooled sample injected again and again, not participants `noticing:reference-rows` (partial; metabolomics)
23. Pooled QCs and an injection order are here: correct the drift from them? `decision:qc-rlsc` (partial; metabolomics)
24. These values back-transform to one million per sample: they are log2 CPM, so logging them again would compress every difference `noticing:omics-value-scale` (partial; genomics, metabolomics)
25. Spreadsheet software turned 14 gene symbols into dates (2-Sep was SEPT2), and some IDs carry versions or repeat `noticing:gene-identifiers` (partial; genomics)
26. About your assay: platform (LC or GC-MS, NMR, a targeted panel), sample matrix, internal standards, and any processing done before export `noticing:assay-card` (partial; metabolomics)
27. These are qPCR Ct values, methylation betas, genotypes (0/1/2), microbiome abundances or Olink NPX: each reads differently `noticing:assay-value-types` (missing; genomics, metabolomics)
28. Tell me about these columns: here is my best guess for each, confirm in one tap `decision:readings-ask-card` (on screen)
29. What is each column: something you measured about people, an ID, a survey-design column, a flag, a time, or left out? `q:roles` (partial)
30. Is `bp_sys` a characteristic the models can adjust for? (one column whose role was only guessed) `decision:confirm_role` (partial)
31. Are `smoking`'s numbers 1, 2 and 3 codes for groups, or amounts? `reading:code-or-amount` (partial)
32. Is `kcal` one day's intake in kilocalories, a total over several days, or kilojoules? `reading:energy-unit-and-days` (partial; dietary)
33. Is weight in kg or lb, height in cm, m or inches, and age in years or months? `reading:body-measure-units` (partial; dietary, clinical)
34. In `RIAGENDR`, is 1 female or male? `reading:sex-coding` (partial; dietary, clinical)
35. Is alcohol in grams or in standard drinks? (It sets the calories in each unit) `reading:energy-source-factor` (partial; dietary)
36. Read in years, `age` flags most rows: record its unit before anything is judged by age `noticing:age-unit-in-question` (partial; clinical)
37. Survey weights, strata and PSUs are here (or missing); which weight fits the variables you use; the table pools several cycles `noticing:survey-design-columns` (partial; dietary, clinical, survey)
38. Protein, fat, carbohydrate and alcohol shares add to 100%, so one must be left out `noticing:compositional-parts` (partial; dietary)
39. `bp_1`, `bp_2` and `bp_3` look like repeated measures of one quantity `noticing:wide-repeated-measures` (partial)
40. `batch`, `injection_order` and `plate` describe how samples were run, and `subject_id` repeats across samples `noticing:acquisition-and-subject-columns` (on screen; metabolomics)
41. A column is free text, an ordered category the alphabet would scramble, or two units in one column `noticing:column-meaning-gaps` (partial)
42. Which instrument measured the diet (24-hour recall, FFQ, record), what the nutrient columns are, and whether energy is intake `noticing:dietary-column-meaning` (partial; dietary)

Noticings decided here (4), each on the card or question it changes. Each is an open noticing until it is decided or dismissed:

<details><summary>What each column is · 2</summary>

- Matrix and sample sheet do not line up `thread:genomics-sample-sheet-alignment` (missing; genomics)
- What a 'don't know' means `thread:survey-dont-know-meaning` (partial; survey)

</details>

<details><summary>Family checks (every lens) · 2</summary>

- Done before upload check, under every lens: what was done to these values before you got them? `sentinel:S6` (missing)
- What a value means check, under every lens: what does this value say, in what unit, about whom? `sentinel:K1` (partial)

</details>


**Confirm sweep, last** (9): defaults set for you, each with an alternative that would change a number.

- Read from your data: these columns were settled by their values, each with its evidence and a way to change it `default:read-from-data` (partial)
- Saturated fat is counted inside total fat `reading:nested-parts` (partial; dietary)
- `SEQN` names each row; `household_id` groups rows `noticing:identifier-columns` (on screen)
- `imputed_bmi` marks values filled in earlier, not a fact about the person `noticing:flag-columns` (on screen)
- Two columns have no name (a saved row index); three columns hold the same value on every row `noticing:unnamed-or-constant-columns` (on screen)
- Confirm what was set for you in Your data (only defaults whose alternative would change a number) `other:confirm-sweep:data` (missing)
<details><summary>Noticings stated with a default (3)</summary>

- The matrix was quantile-normalized, RMA-processed or z-scored across all samples before upload, test samples included `thread:genomics-normalized-across-all-samples` (missing; Estimate, Predict; genomics) · on What each column is
- A value at the top means 'this or more': public-use top codes, instrument ceilings, '>x' upper limits `thread:shared-top-coded` (partial; Estimate, Predict) · on What each column is
- A 5555 means 'more than 21', and an age of 80 means '80 and over': censoring codes, not amounts `thread:survey-top-codes` (partial; Estimate, Predict; survey) · on What each column is

</details>

**For the record**, collapsed and not counted toward progress (11):

<details><summary>11 lines</summary>

- What was read: rows, columns, each column's type and the file's fingerprint `record:ingest-facts` (partial)
- Only the first sheet (or the first SAS data set) was read; the text was read as Latin-1; some rows were short; column types were guessed from the first rows `noticing:ingest-warnings` (engine only)
- Every column summarized; the lens hints read every row, or a fixed sample of 5,000 rows when the table is too big `record:profile` (partial)
- What your codebook settled, the units it documents, and the answers of yours that stand `record:codebook-settled` (engine only)
- Each row is a person: no assay lens is on, and other tables do not arrive turned around `record:orientation-not-asked` (on screen; dietary, clinical, survey, other)
- The data section of the methods: files joined, the codebook used, the lens, the orientation, each repair, and the readings settled from the values `export:data-methods` (engine only)
- Joining paired each person with several rows, so a row is no longer a person `noticing:join-changes-the-row` (partial)
- `arm` looks like what a study compares: was it assigned at random? `noticing:design-hint-arm` (new scope; Estimate, Predict)
- `set_id` holds one case and up to four controls: these rows are matched sets `noticing:design-hint-matched-sets` (new scope; Estimate, Predict)
- `status` reads Case, Control and QC: the cases were sampled separately, so the share of cases was set by the design `noticing:design-hint-case-control` (partial; Predict, Estimate; metabolomics, clinical, genomics)
- The arm never varies within a school: twelve schools were randomized, not 2,400 children `noticing:design-hint-cluster-randomized` (new scope; Estimate)

</details>

**Shown on the canvas or as a refusal**, not an objective (11):

<details><summary>11 previews, views and refusals</summary>

- This join cannot be made: the ID repeats in both files, the files share no value, one file writes the ID as numbers and the other as text, or the held-out rows are already drawn `refusal:join` (engine only)
- This looks like dietary intake: total energy follows protein, carbohydrate and fat `preview:lens-hints` (on screen)
- 'Something else, or not sure' cannot sit beside a named lens `refusal:lens-other-alone` (on screen)
- The table cannot be turned now: the outcome is already chosen, the shape does not allow it, or the held-out rows are drawn `refusal:orientation` (partial; metabolomics, genomics)
- A repair previewed: the changed cells and the column's distribution before and after `preview:apply_repair` (on screen)
- A reading confirmed, previewed (codes or amounts for a group column, a unit, a sex coding) `preview:confirm_reading` (engine only)
- QC drift correction 'Not available yet': a batch has fewer than five pooled QCs, or injections fall after its last QC; the other options stay, each previewed `refusal:qc-rlsc-not-available` (partial)
- That confirmation cannot be recorded: no such column, the outcome has its own questions, the column holds labels not numbers, or the value is not one this reading takes `refusal:readings` (on screen)
- The column roles come after the outcome, the goal and who is in; answer those first `refusal:roles-not-yet` (on screen)
- Undoing that confirmation would leave a recorded answer resting on a reading nobody settled; change that answer first `refusal:revert-unsettles` (on screen)
- That repair cannot be applied now: it is not offered, the reference rows must leave before the seal, the QC answer does not fit the run, or the logged options wait for the zeros `refusal:repairs` (on screen)

</details>

### 2 · Your question

The outcome, its card, the design, the goal and its shape, and whether the paper has a second goal.

**Order:**
1. **The outcome.**
2. **The outcome card's rows.** The event, kind and follow-up may be answered in any order among themselves (`sequence.py:OUTCOME_QUESTIONS`).
3. **The design.** It is new. It comes before the goal because it changes which goals and wordings are valid: a case-control sample allows no prevalence, and a trial allows causal wording.
4. **The goal.** The engine already refuses it while the event, kind or follow-up is open (`sequence.py:_answers_in_order`).
5. **The shape.** It is filtered by lens and outcome. Describe's shapes are declared here and built in Models.
6. **"Add another goal".**

**The Confirm sweep can add a Decide.** Answering "follow-up varies" turns a yes/no outcome into a time to event, which then needs its time column. So the sweep reopens the list.

**Decide, in order** (33).

1. What is your outcome: the column you want to describe, explain or predict? `q:target` (on screen)
2. Which value of the outcome is the event you are counting? `q:event` (on screen)
3. What kind of outcome is it: a number, yes/no, ordered levels, separate classes, or a time until an event? `q:task` (partial)
4. Analyze the outcome as it is (a difference in means) or on the log scale (a ratio of geometric means)? `q:outcome_scale` (partial)
5. Are these levels ordered, and which is lowest? `q:outcome_order` (partial)
6. What unit is the outcome in? `q:outcome_unit` (engine only)
7. Was everyone followed for the same time, or could some leave before the event? (For a time to event: which column holds each row's follow-up?) `q:follow_up` (partial)
8. Count follow-up from a later start, stop it at a horizon, or name when each row entered? `decision:follow_up_window` (engine only; Estimate, Predict)
9. A column says follow-up varies: analyze as a time to event, or record that everyone was followed equally long `refusal:set_censoring` (on screen)
10. The outcome column has impossible values, numbers stored as text, or values below a detection limit `noticing:outcome-findings-resurface` (on screen)
11. How were people assigned or sampled: observed as they were, randomized, sampled by outcome, or matched? `q:study-design` (partial)
12. What do you want to do with the outcome: describe it, estimate what affects it, or predict it? `q:goal` (partial)
13. What shape does that take? (filtered by your field and your outcome) `q:shape` (new scope)
14. Will a decision rest on a threshold of this risk, or is it a risk estimate only? `q:intended_use` (engine only; Predict)
15. When, where and on whom will the model be used? `q:moment_of_use` (missing; Predict)
16. Add another goal to this paper? `decision:add_goal` (new scope)
17. A design or analysis v2 does not run (a crossover trial, a repeated-measures trial model, a complier or per-protocol effect, a direct effect) reads "Not available yet", with its reason and an exit that keeps the work `refusal:not-available-yet` (new scope)

Noticings decided here (16), each on the card or question it changes. Each is an open noticing until it is decided or dismissed:

<details><summary>The outcome card · 15</summary>

- Do you mean having the disease, or being diagnosed with it? Undiagnosed cases sit among your controls. `thread:outcome-diagnosed-or-disease` (missing; clinical, survey)
- The outcome is a cut-point or composite of columns in your table: which ones, and at which cut? `thread:outcome-defined-from-columns` (partial)
- Could the outcome only be found in people who were tested? Does 'not tested' mean 'negative'? `thread:shared-outcome-ascertainment` (missing)
- Some columns are consequences or copies of the outcome (told by a doctor, on medication for it) `thread:outcome-consequences-among-columns` (partial)
- Follow-up runs from months to years, so many 'no' rows are not-yet-knowns `thread:shared-follow-up-varies` (partial)
- Some people died before the event could happen `thread:shared-competing-death` (missing; Estimate, Predict)
- The outcome goes back from 1 to 0 for some people: is it a state, a one-time event, or a recurring one? `thread:clin-state-or-event` (partial; clinical)
- The outcome was also measured at baseline: model the follow-up value adjusted for baseline, or the change? `thread:shared-baseline-outcome-present` (partial; Estimate, Predict)
- Many people have exactly zero: are they non-users, or just not on the recall day? `thread:shared-zero-mass-outcome` (missing)
- A count over observation times that differ: should it be a rate? `thread:shared-count-outcome-exposure-time` (missing)
- The outcome piles up at a limit (0 or 100): is the limit real, or the instrument's ceiling? `thread:shared-bounded-outcome` (missing)
- Are the metabolites what you study, the result of the diet, or the path between them? `thread:metab-metabolome-role` (partial; Estimate, Predict; metabolomics)
- Who are the results, or the model, meant for: these participants, or a population they differ from? `thread:shared-target-population-shift` (missing)
- In a diet or intervention study the genes are the outcomes and the diet is what you study, not the other way round `thread:genomics-genes-are-the-outcomes` (partial; Estimate, Predict; genomics)
- The uploaded genes were already chosen using these samples' outcome `thread:genomics-features-preselected-on-outcome` (partial; Predict, Estimate; genomics)

</details>

<details><summary>Family checks (every lens) · 1</summary>

- Outcome and clock check, under every lens: what is the outcome, and when does its clock start and stop? `sentinel:K6` (missing)

</details>


**Confirm sweep, last** (5): defaults set for you, each with an alternative that would change a number.

- These data are read as an observational study `default:design_observational` (new scope)
- The yes/no outcome is counted over one period for everyone, since no column reads as a follow-up time `default:follow_up_one_period` (on screen; metabolomics, genomics, survey)
- Predicted risks are scored at the median follow-up time `default:prediction_horizon` (engine only; Predict)
- The outcome is analyzed on its own scale (a difference in means) `default:outcome_scale_original` (partial)
- Confirm what was set for you in Your question (only defaults whose alternative would change a number) `other:confirm-sweep:question` (missing)

**For the record**, collapsed and not counted toward progress (9):

<details><summary>9 lines</summary>

- The outcome has two values, so it is yes/no (or its decimals fill a grid, so it is a number) `default:task_settled` (on screen)
- No event level or follow-up to set: a number has no event level, and only a yes/no outcome or a time to event has a follow-up `default:outcome_steps_not_applicable` (on screen)
- Numeric levels are ordered by their values `default:order_by_value` (engine only)
- With no follow-up in the data, the model detects the outcome now rather than forecasting it `default:diagnostic_or_prognostic` (missing; Predict, Estimate)
- Tracks run in order: estimating an effect first, then describing, then predicting `default:track_order` (new scope)
- The methods sentences for the outcome, its event, kind, scale, order, unit, follow-up and goal `export:methods_sentences:question` (partial)
- Reporting checklist items answered here `export:checklist_anchors` (engine only; Estimate, Predict)
- The outcome and goal in the analysis plan and its hash `export:plan_slots` (engine only; Estimate)
- Onset is known only to lie between two visits `thread:clin-event-found-at-visits` (missing; Estimate, Predict; clinical)

</details>

**Shown on the canvas or as a refusal**, not an objective (19):

<details><summary>19 previews, views and refusals</summary>

- Rows with the outcome recorded, and its distribution `preview:set_target` (on screen)
- The outcome read as a quantity, or as classes `preview:set_task` (on screen)
- Which rows become 1 and which become 0 `preview:set_event` (partial)
- The outcome on the scale analyzed `preview:set_outcome_scale` (on screen)
- The outcome is rare (4% of rows) or common (40%) `noticing:design-hint-outcome-prevalence` (engine only; Predict, Estimate)
- The levels in the declared order `preview:set_outcome_order` (on screen)
- Rows at risk at the landmark, and follow-up as the model reads it `preview:set_follow_up` (partial; Estimate, Predict)
- What the goal changes downstream `preview:set_purpose` (partial)
- 'Same for everyone' and the outcome's unit change no number, so the card quotes its sentence `preview:unpreviewed_outcome_kinds` (partial)
- Where the decision thresholds fall on the model's risks `preview:set_intended_use` (engine only; Predict)
- That column can't be the outcome (the row number, a missing column, a column an eligibility rule reads, or one that would redraw the held-out rows) `refusal:set_target` (on screen)
- That kind does not fit these values `refusal:set_task` (on screen)
- The event must be one of the outcome's two levels `refusal:set_event` (on screen)
- That follow-up can't be used as named `refusal:set_follow_up` (on screen; Estimate, Predict)
- The log scale isn't possible here, or the held-out rows are already drawn `refusal:set_outcome_scale` (on screen)
- The order must place every level once; the unit must describe the outcome `refusal:set_outcome_order_unit` (engine only)
- Answer how the rows repeat first: imputed copies treated as repeats can't support inference `refusal:set_purpose` (on screen; Estimate; dietary, clinical)
- Later refusals whose exit is 'change the goal' `refusal:goal-routed-elsewhere` (on screen)
- The outcome is one day's recalled intake, and much of its spread is day-to-day noise `thread:diet-intake-as-outcome` (partial; dietary)

</details>

### 3 · First look

Exploration, shaped by the goal.

**The walk:**
- On arrival nothing is asked.
- "Worth a look" shows at most three highlights.
- A blocker ("Decide this first") replaces the primary action. An example is a batch perfectly confounded with the outcome.
- The six groups follow the fixed STRATOS order, and the outcome door comes last, behind the goal's rule.

**Shown here, decided later.** Every noticing First look shows is decided at the stage that changes because of it, and returns there attributed ("You looked at this"). Its progress counts in that stage, not here. Today the engine computes First look's outcome-free checks only after the seal (`stages/explore.py`), which is disagreement 4.

| First look group | Shown here | Decided at Your data | Your question | Who's in | Models | Results |
|---|---|---|---|---|---|---|
| Who is in the data | 60 | 5 | 2 | 40 | 13 | 0 |
| What is missing | 23 | 0 | 0 | 15 | 8 | 0 |
| Each variable | 70 | 2 | 0 | 16 | 52 | 0 |
| Variables together | 26 | 2 | 0 | 0 | 24 | 0 |
| Over time and by batch | 38 | 1 | 0 | 10 | 27 | 0 |
| The outcome | 13 | 0 | 3 | 0 | 10 | 0 |
| **Total** | **230** | 10 | 5 | 81 | 134 | 0 |

**Decide, in order** (4).

1. Worth a look: at most three highlights (two per group at most), each a claim ≤ 20 words, 'Affects …', its measure beside its reference and a 'Why is this here?' line `other:first-look:worth-a-look` (partial)
2. A batch perfectly confounded with the outcome: no model is fitted until it is answered `refusal:batch-perfectly-confounded` (partial; Predict, Estimate)
3. What the lenses noticed: the structural diagnosis, the lens packs and the app's own findings, each a one-line claim with the question it routes to `noticing:findings-stage` (partial)
4. Decide now: a look's question opens here as its own card ('Who's in · asked from First look'); each option previews on the same canvas; Esc returns; the item reads 'Decided: …' `decision:first-look:decide-now` (partial)

**Confirm sweep, last** (0): defaults set for you, each with an alternative that would change a number.

- None. This stage owns no number-changing default.

**For the record**, collapsed and not counted toward progress (5):

<details><summary>5 lines</summary>

- Opening an outcome view is recorded as looked at (forking paths): which view, on which rows, and each lever's answer at that first look `decision:view_outcome` (engine only; Estimate, Predict)
- Which rows a look reads: outcome-free looks every row before the seal; outcome views the training rows (prediction) or every analyzed row (inference); O2 verdicts every row `default:first-look:rows-read` (engine only)
- The references beside each measure: chance shares and named conventions, never pass or fail `default:first-look:references` (partial)
- Explore's methods sentence: which rows it read (the held-out rows never read) and that every lever was offered first as an in-fold rule `result:explore_sentence` (engine only; Predict, Estimate)
- What you have looked at: opening an outcome-free look is noted (seen marks, the supplement's 'highlighted and opened' list); it is not a decision and never enters the methods `other:looked-at-ledger` (missing)

</details>

**Shown on the canvas or as a refusal**, not an objective (26):

<details><summary>26 previews, views and refusals</summary>

- First look · before you plan: the card is a guide (Worth a look, six groups with counts, the outcome row, 'Why does this matter?', one primary action) and asks nothing until a decision is opened `other:first-look:guide` (missing)
- Your data now: every column in column order with its kind, shape, blank share and look number; the outcome row reads 'The outcome · opens in its group' `preview:first-look:index` (partial)
- Who is in the data: who and what are the rows? `other:first-look:group:who` (partial)
- What is missing: where are the blanks, and why? `other:first-look:group:missing` (partial)
- Each variable: is each column what it says? `other:first-look:group:each` (partial)
- Variables together: what moves together or duplicates? `other:first-look:group:together` (partial)
- Over time and by batch: did how or when it was measured change it? `other:first-look:group:over_time` (partial)
- The outcome: a door, not a group to browse. Its own distribution (O1) and 'With one column…' (O3); every view is recorded; under inference never pointed at or ranked `other:first-look:outcome-door` (partial; Estimate, Predict, Describe)
- The outcome's distribution: a histogram, or the share of each class (the rarer class's share) `noticing:explore::outcome_distribution` (engine only; Estimate, Predict)
- How the outcome moves with one continuous predictor (binned means or event share over its tenths) `noticing:explore::relationship` (engine only; Estimate, Predict)
- A finding's evidence drawn on the canvas: why it was raised, on the user's own rows `preview:finding-evidence` (on screen)
- Changed since you looked: a look whose numbers moved after a decision returns at the top of its group with its before → after `preview:changed-since-you-looked` (engine only)
- Classic's exploration views returned as First look views and threads: missingness patterns, the skew and outlier table, the pre-fit VIF table `exhibit:classic-exploration-views` (missing)
- The walk on the first visit: 'Look at the first' → 'Next · 2 of 3' → 'Continue to Who's in'; a T1 item reads 'Decide this first'; on a revisit 'Continue' `other:first-look:walk` (missing)
- First look's segment of the progress bar `other:first-look:progress` (missing)
- The five view kinds a look draws: row_flow, lineage, table_focus, distribution, relationship (at most three per item, one primary) `other:views:closed-vocabulary` (on screen)
- The sample map: PCA scores (QC vs participants, batches), PC1 against run order, the PC × covariate R² grid `other:views:embedding` (missing)
- The clustered matrix: correlations with values printed (≤ 30 columns), named clusters without values (to ~500), sample × sample at width `other:views:matrix` (missing)
- Layout by footprint: Focus (one column), Strip (several), Flow (which rows), Routing (roles or structure), Angles (competing conventions) `other:views:footprint-layouts` (partial)
- The context view: the same measure split by the lens's structural variable where it differs most, or one line ('Similar for women and men') `other:views:context-view` (missing)
- Wide data: a distribution of a per-feature statistic plus the top-12 Strip; hexbins above ~5,000 rows; cached per data state, nothing recomputed on hover `other:views:wide-aggregates` (partial)
- The outcome group: 'Not available yet · opens once the split is recorded' `refusal:outcome-door-not-yet` (partial; Predict, Estimate)
- An outcome view refused: no outcome chosen, no predictor named, or an unknown column `refusal:view_outcome` (engine only; Predict, Estimate)
- A look whose readings are unsettled is not shown; its reading is asked at Your data `refusal:look-needs-a-settled-reading` (partial)
- The findings stage fails rather than sample: findings count rows, so a sample would state wrong numbers `refusal:findings-memory-budget` (on screen)
- First look shaped by the goal: the outcome door's rule and the rows it reads follow the goal; several goals share one First look under the strictest goal's rule `other:first-look:goal-shaping` (new scope)

</details>

### 4 · Who's in

What one row is, groups above the person, the survey design, who is eligible, missing values, and the seal.

**Order.** The Router asks:
1. grain;
2. repeats or time points;
3. unit;
4. combining;
5. predicting forward;
6. *(roles)*;
7. clusters;
8. survey;
9. exclusions;
10. missing values;
11. the split.

The source is `interview.py:QUESTION_KEYS`, and any answer the Router has not reached is refused (`sequence.py:_answers_in_order`).

**Under Predict, the Confirm sweep runs just before the seal.** The draw reads the stated grain, repeat kind and time column, and changing them after the draw is refused (`seal.py:DECISION_A`, `DRAW_READS`).

**The participant flowchart closes the stage.** For omics it is the samples-and-features flow. It is provisional until the analysis flowchart, because Models answers still change the counts (disagreement 7).

**Decide, in order** (76).

1. Can one person appear in more than one row? `q:grain` (on screen)
2. Are these repeats of one measurement, or different time points? `q:repeat_kind` (partial)
3. When you analyze this, what is one row? `q:unit` (on screen)
4. How should each person's rows be combined? `q:aggregation` (partial)
5. Are you predicting something later from measurements taken earlier? `q:temporal` (on screen; Predict)
6. Are participants grouped in sites, centers, households or batches? `q:clusters` (partial; Estimate, Predict, Describe)
7. Should the estimates (or the scores) describe the surveyed population, or these participants? `q:survey` (partial; Estimate, Describe, Predict)
8. Who's in for a description: survey-weighted groups, exclusions as domains, who is kept when values are blank, no seal `new:describe_whos_in` (new scope; Describe)
9. Which analysis set: everyone as randomized (intention to treat), with per protocol as a secondary? `decision:analysis-sets` (new scope; Estimate)
10. Rows are matched sets: keep each set together `new:matched_sets_grain` (new scope; Estimate, Predict)
11. Is your study restricted to part of this data? `q:exclusions` (partial)
12. A check found something that belongs to this question `noticing:findings_routed_here` (on screen)
13. How should rows with missing predictor values be handled? `q:missing` (partial; Estimate, Predict, Describe)
14. Who is missing: complete cases drop rows, and the people dropped differ from those kept `noticing:complete_case_loss` (engine only; Estimate, Predict)
15. This missing-values answer does not fit the goal `refusal:missing_by_purpose` (on screen; Estimate, Predict)
16. The imputation does not match the analysis model or the clustering `refusal:missing_compatibility` (partial; Estimate)
17. Predictors summarized from records later than the outcome `refusal:predictors_after_outcome` (on screen; Estimate)
18. These rows are imputed copies, not repeats `refusal:imputed_copies` (on screen; Estimate; dietary, clinical)
19. No rows (or too few) remain for the analysis `refusal:zero_rows` (missing)
20. Repair or keep the values the held-out rows are drawn by `decision:settle_draw_findings` (partial; Predict)
21. Repetition was found but the seal could not group it, so held-out scores are exploratory `noticing:exploratory_seal` (on screen; Predict)
22. A holdout this size would hold too few events for an honest score `noticing:holdout_below_floor` (partial; Predict)
23. How many rows should be sealed for one final, untouched score? `q:split` (partial; Predict)
24. Withdraw the held-out rows, change this, then draw them again `decision:withdraw_seal` (on screen; Predict)
25. Rows held out after scores were seen: labeled, the earlier version kept `new:split_after_scores` (missing; Predict)

Noticings decided here (51), each on the card or question it changes. Each is an open noticing until it is decided or dismissed:

<details><summary>Who is eligible · 25</summary>

- Some energy reports are implausible for the person `thread:diet-implausible-reporters` (partial; dietary)
- An implausible intake is one bad day, not one bad person `thread:diet-implausible-day-vs-person` (partial; dietary)
- Who leaves at each step differs from who stays `thread:shared-selection-flow` (partial)
- Rows were removed before upload by an unstated rule `thread:shared-rows-filtered-before-upload` (missing)
- Some rows had the outcome when the clock started `thread:clin-prevalent-cases-at-baseline` (partial; clinical)
- The reference test was done only for some people `thread:clin-partial-verification` (missing; Predict; clinical)
- Everyone was selected for something that what you study, or the outcome, causes `thread:shared-selection-on-a-consequence` (missing; Estimate)
- Some changed their diet because of a diagnosis `thread:diet-changed-because-of-disease` (partial; Estimate; dietary)
- People told they have a disease changed what they eat `thread:clin-diagnosis-changes-exposure` (missing; Estimate; clinical)
- Participants already knew they had the outcome `thread:shared-prevalent-disease-reverse-causation` (missing; Estimate)
- Participants span several life-stage groups, pregnancy included `thread:diet-dri-life-stage` (partial; dietary)
- Pregnancy or kidney disease changes what a biomarker means `thread:clin-physiological-state` (partial; clinical)
- Some patients could never have received the treatment `thread:clin-contraindication-positivity` (partial; Estimate; clinical)
- Cases' samples were taken after diagnosis `thread:shared-sampled-after-diagnosis` (missing; metabolomics, genomics)
- Collection or storage differs between groups `thread:metab-preanalytical-aligned` (missing; metabolomics)
- Tissues differ by group `thread:genomics-mixed-tissue-types` (missing; genomics)
- A technical covariate drives a leading component `thread:genomics-sample-quality-drives-components` (missing; genomics)
- Recorded sex disagrees with expression `thread:genomics-sex-from-expression` (missing; genomics)
- Some respondents answered without reading `thread:survey-careless-responding` (missing; survey)
- Answers stop partway and never resume `thread:survey-breakoff` (missing; survey)
- A trajectory no body follows `thread:clin-implausible-trajectory` (partial; clinical)
- Eligibility required a high reading `thread:shared-entry-on-a-high-reading` (missing; Estimate)
- What you study is defined after follow-up starts, so early follow-up cannot hold an event (immortal time) `thread:clin-time-zero` (partial; Estimate)
- A zero is a day without the food, not a never-eater `thread:diet-zero-is-a-day` (partial; Estimate, Describe; dietary)
- Fasting state, time of draw, storage time, freeze-thaw, hemolysis or RNA integrity shift measurements, and differ by group `thread:shared-preanalytical-handling` (partial)

</details>

<details><summary>What one row is · 12</summary>

- The same record or person appears more than once `thread:shared-duplicate-records` (partial)
- These columns hold one value per person `thread:shared-one-value-per-unit` (partial; Estimate)
- An ID is unique only within a site or cycle `thread:shared-id-unique-only-within-a-site` (missing)
- Rows are encounters; patients repeat `thread:clin-encounters-not-patients` (partial; clinical)
- Rows are foods within a recall day `thread:diet-food-rows-sum-to-days` (missing; dietary)
- Rows are households' food acquisition `thread:diet-household-level-intake` (missing; dietary)
- Rows are meals nested in people `thread:diet-meal-level-rows` (partial; Predict; dietary)
- Several samples come from one person `thread:genomics-paired-or-repeated-samples` (partial; genomics)
- Two samples are near-identical `thread:genomics-near-identical-samples` (missing; genomics)
- The rows are cells; the n is donors `thread:genomics-cells-are-not-replicates` (partial; genomics)
- Each participant received every diet in turn `thread:clin-crossover-design` (new scope; Estimate; clinical)
- In a long table, a per-person column summarizes all of the person's visits, future ones included `thread:shared-unit-summary-spans-the-future` (partial)

</details>

<details><summary>Missing values · 6</summary>

- A blank after a 'No' means zero or not asked `thread:shared-skip-pattern-structural-zero` (partial)
- A blank means the test was not ordered `thread:shared-informative-missingness` (partial)
- Values below detection are censored `thread:shared-left-censoring` (partial; metabolomics, clinical)
- Non-detects cluster by batch `thread:metab-missing-by-batch` (missing; metabolomics)
- A column was not collected at some sites or cycles `thread:shared-missing-by-design-block` (partial)
- A blank FFQ line usually means never `thread:diet-ffq-blank-means-never` (missing; dietary)

</details>

<details><summary>Family checks (every lens) · 3</summary>

- Not independent check, under every lens: what is one unit, and which rows belong together? `sentinel:S4` (partial)
- Who is in check, under every lens: who is in these rows, and how did they get here? `sentinel:S5` (missing)
- What a zero or blank means check, under every lens: is this blank or zero a fact, a skip, a non-detection or a gap? `sentinel:K2` (missing)

</details>

<details><summary>Groups above the person · 2</summary>

- What you study varies mostly between groups, not within them `thread:shared-exposure-varies-between-groups` (missing; Estimate)
- Neighboring places resemble each other `thread:shared-spatial-dependence` (missing)

</details>

<details><summary>Samples and features · 1</summary>

- One sample is unlike the others `thread:genomics-outlier-sample` (partial; genomics, metabolomics)

</details>

<details><summary>The seal · 1</summary>

- Case mix shifts over calendar time `thread:shared-case-mix-shift` (partial; Predict)

</details>

<details><summary>The survey design · 1</summary>

- An opt-in sample has no design to weight by `thread:survey-nonprobability-sample` (missing; Estimate, Describe; survey)

</details>


**Confirm sweep, last** (18): defaults set for you, each with an alternative that would change a number.

- These rows are read as [repeats/time points] from [evidence] `default:repeat_kind_stated` (on screen)
- A row whose screening value is not recorded cannot be confirmed eligible, so it leaves `default:unrecorded_rows_excluded` (partial)
- Goldberg screen inputs: the BMR equation, PAL, recall days, and Black 2000's variation `default:goldberg_inputs` (partial; Estimate, Predict; dietary)
- Number of imputations: at least 20, or more when more rows are incomplete `default:missing_m` (partial; Estimate)
- Imputation matches the analysis model (SMC-FCS) `default:imputation_model_compatible` (engine only; Estimate)
- Repeated rows are imputed by person (values that never change, once per person) `default:imputation_levels_clustered` (engine only; Estimate)
- Values below a detection limit are filled as censored, never by the median `default:below_detection_fill` (engine only; Estimate, Predict; metabolomics, clinical)
- The weight comes from the smallest subsample your variables were measured on `default:survey_weight_choice` (partial; Estimate, Describe; dietary, clinical, survey)
- The held-out rows and the folds are drawn with seed 0; another seed draws other rows `default:split-seed` (partial; Predict)
- Each row is a different person (a unique identifier) `default:grain_stated` (on screen)
- The latest rows by [time column] are held out `default:temporal_time_column` (on screen; Predict)
- How the held-out rows were drawn: grouped by [id], or one row per person `default:seal_basis` (on screen; Predict)
- The data's own imputed copies are analyzed one by one and pooled by Rubin's rules `default:imputed_copies_pooled` (partial; Estimate, Predict; dietary, clinical)
- Confirm what was set for you in Who's in (only defaults whose alternative would change a number) `other:confirm-sweep:whos_in` (missing)
<details><summary>Noticings stated with a default (4)</summary>

- Features present in the process blanks at levels near the samples' (contamination, not biology) `thread:metab-blank-contamination` (partial; metabolomics) · on Samples and features
- Children's BMI and height mean different things at different ages and need age- and sex-specific z-scores `thread:clin-pediatric-growth-scale` (partial; clinical) · on Who is eligible
- Some values are impossible (decimal slips, wrong unit), while others are extreme but real `thread:shared-impossible-vs-extreme` (partial) · on Who is eligible
- What you study changed because the disease had already begun: weight or cholesterol falls before death, cancer or dementia `thread:clin-preclinical-disease-before-event` (partial; clinical) · on Who is eligible

</details>

**For the record**, collapsed and not counted toward progress (32):

<details><summary>32 lines</summary>

- Rows with no value for the outcome leave `default:outcome_measured` (on screen)
- The every-row analysis is reported beside the screen `default:every_row_analysis` (engine only; Estimate)
- No rows are held out: every row estimates the coefficients `default:split_under_inference` (partial; Estimate)
- Rows whose follow-up ended before the landmark leave `default:landmark_rows` (on screen)
- Complete cases are judged on settled predictors only `default:complete_case_predictors` (engine only; Estimate, Predict)
- Participant flow, savable as a figure `exhibit:participant_flowchart` (partial)
- CONSORT flow: enrolled, allocated, followed up, analyzed, by arm `flowchart:consort` (new scope; Estimate; clinical, dietary)
- Samples and features: how many samples and features remain at each step `exhibit:samples_and_features_flow` (partial; metabolomics, genomics)
- Data quality differs by sex, age or income `noticing:explore::quality_by_group` (partial)
- Under the survey design, excluded rows stay in the design as a domain `default:survey_domain` (partial; Estimate, Describe; dietary, clinical, survey)
- After exclusions a stratum keeps one PSU `noticing:lonely_psu` (partial; Estimate, Describe; dietary, survey, clinical)
- Participants are related `thread:genomics-relatedness` (missing; genomics)
- Answers cluster by interviewer `thread:survey-interviewer-clustering` (partial; survey)
- Blanks fall in a column that will be logged or splined `thread:shared-missing-in-a-derived-term` (partial; Estimate)
- What a zero means here `thread:metab-zeros-meaning` (partial; metabolomics)
- Unticked boxes are No, not blank `thread:survey-check-all-that-apply` (missing; survey)
- A few skipped items should not cost the respondent `thread:survey-item-nonresponse` (partial; survey)
- The imputation model knows the survey design `thread:survey-imputation-carries-the-design` (partial; Estimate; survey, dietary)
- A median fill would mirror energy in the residual `thread:diet-median-fill-breaks-the-residual` (partial; Estimate; dietary)
- Failed libraries are uneven across groups `thread:genomics-excluded-libraries-by-group` (missing; genomics)
- An injection failed `thread:metab-failed-injection` (partial; metabolomics)
- Some samples or variants fail genotyping QC `thread:genomics-genotype-qc` (missing; genomics)
- The same sample was injected twice `thread:metab-technical-replicates` (partial; metabolomics)
- Samples from one person are more alike than any effect `thread:metab-person-fingerprint` (partial; Predict; metabolomics)
- Some recalls were unreliable or have no second day `thread:diet-recall-completeness` (missing; dietary)
- Everyone already has a disease (index-event bias) `thread:clin-index-event-selection` (partial; Estimate; clinical)
- Some rows are children whom adult limits misjudge `thread:shared-children-among-adults` (partial)
- Inflammation distorts micronutrient biomarkers `thread:clin-inflammation-adjustment` (missing; clinical)
- Fasting and non-fasting draws are mixed `thread:clin-fasting-status` (partial; clinical)
- A hemolyzed specimen still reports a number `thread:clin-specimen-quality-flags` (missing; clinical)
- Manual readings pile up on terminal digits 0 and 5, and just below diagnostic thresholds `thread:clin-digit-preference` (missing; clinical)
- Impossible values are errors; extreme ones are the sickest `thread:clin-impossible-vs-extreme` (partial; clinical)

</details>

**Shown on the canvas or as a refusal**, not an objective (26):

<details><summary>26 previews, views and refusals</summary>

- Rows per person, or each row its own unit `preview:grain_views` (on screen)
- Repeats against time points `preview:repeat_kind_views` (on screen)
- One row per person against records staying as rows `preview:unit_views` (on screen)
- Units' records, combined `preview:aggregation_views` (on screen)
- The latest rows held out, or drawn at random `preview:temporal_views` (on screen; Predict)
- Rows per group, the fixed effects added, or the validation across sites `preview:clusters_views` (on screen; Estimate, Predict)
- These participants against the population `preview:survey_views` (on screen; Estimate)
- Who these rules would exclude, and where the cut falls `preview:exclusions_views` (on screen)
- Rows kept when values are missing; blanks as a level; what leaving columns out saves `preview:missing_views` (on screen; Estimate, Predict)
- Where the held-out rows come from `preview:split_views` (on screen; Predict)
- An earlier question comes first: later questions depend on earlier answers `refusal:not-yet` (on screen)
- The grain answer disagrees with the table `refusal:grain` (on screen)
- Combining or ordering cannot proceed as asked `refusal:row_block` (on screen)
- The grouping answer does not fit the column or the goal `refusal:clusters` (on screen; Estimate, Predict)
- The survey design is incomplete or unsettled `refusal:survey` (on screen; Estimate)
- A rule on the outcome selects on the outcome `refusal:rule_on_outcome` (on screen)
- This screen cannot run as given `refusal:screens` (partial)
- Values below detection need a censoring-aware fill `refusal:below_detection` (engine only; Estimate, Predict; metabolomics, clinical)
- The held-out rows cannot be drawn yet `refusal:split` (on screen; Predict)
- The held-out rows are drawn: withdraw them before changing this `refusal:sealed` (on screen; Predict)
- Reference rows cannot leave once rows are held out `refusal:rows_after_the_seal` (on screen; Predict; metabolomics)
- The time column cannot order the rows, so the latest cannot be held out `refusal:chronology_cannot_order` (partial; Predict)
- Who is in the analysis: rows at each step `result:cohort` (on screen)
- What a held-out set of each size can measure, and how the seal would be drawn `result:seal_plan` (partial; Predict)
- The methods sentences for who is in `export:methods_sentences:whos_in` (partial)
- No one leaves the primary analysis after randomization `new:no_exclusion_after_randomization` (new scope; Estimate)

</details>

### 5 · Models

**Estimate, in the Router's order:**
1. what you study and its effect;
2. the adjustment set;
3. the time-varying lane;
4. the energy model;
5. the forms;
6. modifiers and the causal lane (both stated by default, so they sit in Confirm);
7. the model families.

Then the declarations with no Router key: Model 1, sensitivity analyses, regression calibration, scales, batch, the omics normalization and multiplicity, and the substitution pair (disagreement 13).

**Under an assay lens, either goal:** how raw counts or intensities are normalized (`decision:omics-normalization`; it moved here from Your data's repair, which keeps only the reading), then the in-fold steps in the Confirm sweep.

**Predict:**
1. the validation scheme;
2. selection and the in-fold levers;
3. the model families;
4. recipes and tuning.

**Describe:** usual intake, weighted means and prevalence, patterns, clustering, agreement.

**End of every track:**
1. the open-noticings gate (Estimate and Describe; under Predict it waits for the opening, in Results);
2. the Confirm sweep;
3. the analysis flowchart with Fit.

Under Estimate and Describe, nothing is served before you press Fit, and pressing it locks the track's plan and shows the first estimates (Settled here). Under Predict, pressing Fit opens Results.

**Decide, in order** (132).

1. What do you think affects [glucose]? Its whole effect, per what unit, as a difference or a ratio ("only the direct part" reads Not available yet) `q:estimand` (partial; Estimate)
2. How do [age, sex and the others] relate to [sugar] and [glucose]? (one tap per block that shares a guess) `q:adjustment` (partial; Estimate)
3. Does [sugar] change over time for each person, and how should its history be followed? `q:time_varying` (partial; Estimate)
4. How should the analysis account for how much people eat overall? `q:energy_adjustment` (partial; Estimate, Predict; dietary)
5. Should [sugar] and each number you adjust for enter as a straight line or a curve? `q:form` (partial; Estimate)
6. Which models should be fit? (inference: which model fits this question) `q:models` (partial; Estimate, Predict)
7. Which columns should Model 1 adjust for? (this field's Model 1: age, sex and total energy) `decision:set_model_sequence` (engine only; Estimate)
8. Which removal rules should be reported beside the main analysis, as checks? `decision:set_sensitivity` (engine only; Estimate, Predict)
9. Correct the intakes in the model for day-to-day swings in the recalls, all of them together, as a secondary analysis? `decision:set_measurement_error` (engine only; Estimate; dietary)
10. Which questionnaire items make up each scale, and should its score be corrected for unreliability? `decision:set_scales` (engine only; Estimate, Predict; survey)
11. How should the batch column be handled? `decision:set_batch` (engine only; Estimate, Predict; metabolomics, genomics)
12. These columns are raw counts or intensities: how are they normalized? `decision:omics-normalization` (partial; Estimate, Predict; metabolomics, genomics)
13. Which calories should replace which (for example, 100 kcal of sugar swapped for protein)? `decision:substitution-pair` (on screen; Estimate, Predict; dietary)
14. How sensitive is the result to the outcomes that are missing? `decision:trial-missing-outcome-sensitivity` (new scope; Estimate)
15. Cases were sampled by outcome: report odds ratios only; recalibrate predictions to the population's prevalence `decision:case-control` (new scope; Estimate, Predict)
16. A finding held for this question: raw intensities need normalizing; batch is uneven over the outcome; more predictors than rows `noticing:findings-routed-to-models` (partial)
17. More candidate predictors than rows: selection or a filter comes first `noticing:explore::wide` (engine only; Estimate, Predict)
18. Pairs of predictors move almost together `noticing:explore::collinear` (engine only; Estimate, Predict)
19. Should predictors be selected inside each fold? Were any chosen beforehand using these people's outcomes? `decision:set_selection` (engine only; Predict)
20. More models on the shelf: ridge, robust linear (Huber), random forest, XGBoost `decision:new-families` (missing; Predict, Estimate)
21. Describe: the usual-intake distribution of a nutrient, and the share below its requirement `decision:set_usual_intake` (engine only; Describe; dietary)
22. Describe: survey-weighted means and prevalence by group, and trends across stacked survey cycles `decision:describe-estimates` (new scope; Describe; dietary, clinical, survey)
23. Build dietary patterns (principal components, factors, clusters, reduced-rank regression) as what you study `decision:dietary-patterns` (new scope; Estimate, Describe, Predict; dietary)
24. Find subgroups of similar people `decision:clustering-subgroups` (new scope; Describe)
25. How well do two measurements (or two models' predictions) agree? `decision:bland-altman` (new scope; Describe, Predict)
26. Before you see estimates: these noticings still shape the plan. Decide or dismiss each `gate:open-noticings-before-lock` (missing; Estimate, Describe)
27. Fit · about N min `other:fit` (partial)

Noticings decided here (105), each on the card or question it changes. Each is an open noticing until it is decided or dismissed:

<details><summary>The adjustment set · 26</summary>

- Rows come from several hospitals or clinics whose patients, practices and outcome rates differ `thread:clin-site-differences` (partial; Estimate, Predict; clinical)
- The variables that drove sampling (oversampled groups, exam season, cycle) also drive the data, so weighted and unweighted answers can disagree `thread:survey-informative-design` (missing; Estimate, Predict; survey)
- Treatment lowers the very measurement it was given for: antihypertensives on BP, statins on LDL, glucose-lowering drugs on glucose and HbA1c `thread:clin-treated-measurements` (partial; clinical)
- Under-reporting rises with body size, so reported energy falls where physiology says it should rise `thread:diet-misreporting-tracks-body-size` (partial; Estimate, Predict; dietary)
- Sodium or potassium 'intake' is estimated from spot urine by an equation built on age, weight and creatinine `thread:diet-spot-urine-estimated-excretion` (missing; Estimate, Predict; dietary)
- The expression profile records habits the sample sheet does not (smoking), or contradicts the self-report `thread:genomics-exposure-written-in-the-profile` (missing; Estimate, Predict; genomics)
- Some participants' values are lowered by treatment (antihypertensives on blood pressure, statins on LDL) `thread:shared-treated-values` (partial)
- Some respondents agree with everything, opposite statements included: a response style masquerading as the trait `thread:survey-acquiescence` (missing; survey)
- Answers given by phone, online, in another language or by a proxy are not the same measurement `thread:survey-mode-proxy-nonequivalence` (missing; Estimate, Predict; survey)
- Fat-soluble vitamins and carotenoids travel on lipoproteins, so their concentration tracks cholesterol and triglycerides `thread:clin-lipid-carried-micronutrients` (missing; Estimate, Predict; clinical)
- Nutrients are computed from the foods in the model, so adjusting a food for its own nutrients removes its effect `thread:diet-food-and-its-nutrients` (missing; Estimate, Predict; dietary)
- Genetic ancestry structures the samples and lines up with case status or with the diet you study `thread:genomics-ancestry-structure` (missing; Estimate, Predict; genomics)
- Differences in the mix of cell types, not regulation within cells, drive the signal `thread:genomics-cell-mix-drives-signal` (missing; Estimate, Predict; genomics)
- When and how the sample was drawn (time of day, fasting or fed, season) shapes expression and may differ between groups `thread:genomics-draw-conditions` (missing; genomics)
- Samples split on a leading component by something no column names (a lane, an extraction day, a tissue) `thread:genomics-unrecorded-structure` (missing; genomics)
- An assay, instrument, food-composition database or questionnaire version changed at a date or between sites `thread:shared-method-change` (missing)
- Intake or a biomarker varies by season, and season is unevenly spread across groups `thread:shared-season` (partial)
- Pooled cycles or waves asked a question differently, under the same name or a renamed one `thread:survey-instrument-changed-across-cycles` (missing; survey)
- The treatment you study is given to the sicker people, so it marks how ill they were; in a prediction model it marks the indication `thread:clin-confounding-by-indication` (partial; Estimate, Predict; clinical)
- A clinical measurement taken alongside the diet lies on the path from diet to disease (LDL, BP, HbA1c, BMI) `thread:clin-biomarker-on-the-path` (partial; Estimate, Predict; clinical)
- A column is a score, component or cluster label fitted on these same rows, possibly using the outcome `thread:shared-score-fitted-on-these-rows` (missing; Estimate, Predict)
- Something you adjust for is measured so poorly that adjusting for it removes little of the bias it causes `thread:shared-mismeasured-confounder` (partial; Estimate)
- Race/ethnicity, sex/gender or SES columns: something to adjust for, a group to check fairness in, or a predictor? `thread:shared-sensitive-attribute-role` (partial; Estimate, Predict)
- The normalizer itself moves with the outcome or with what you study: creatinine in kidney disease, a global shift that breaks PQN, TIC closure `thread:metab-normalizer-carries-biology` (partial; Estimate, Predict; metabolomics)
- A column measured at the same visit as what you study may cause it, or may be a step on its path to the outcome `thread:shared-mediator-or-confounder` (engine only; Estimate)
- Treatment started during follow-up because of high risk, so high-risk features look protective `thread:shared-treatment-paradox` (missing; Estimate, Predict)

</details>

<details><summary>The exposure and its effect · 11</summary>

- In a trial, rescue medication, stopping the diet or dropping out after randomization changes what the treatment effect means `thread:clin-intercurrent-events` (missing; Estimate, Describe; clinical)
- What you study and the outcome were reported at the same sitting, and people change what they eat after a diagnosis `thread:survey-same-sitting-reverse-causation` (missing; survey)
- The zero group mixes never-consumers with people who stopped, often because they got sick `thread:diet-former-consumers-in-reference` (partial; dietary)
- Nutrient totals include or omit dietary supplements, and supplement users form a separate population `thread:diet-supplements-in-totals` (partial; dietary)
- The 'non-drinkers' include people who quit because they got sick `thread:survey-former-users-in-the-reference` (missing; Estimate, Predict; survey)
- A column is a binned copy of another (bmi_cat beside bmi, age_group beside age), or the outcome is a cut of a recorded measure `thread:shared-coarsened-copy` (missing; Estimate, Predict)
- Several columns are parts of a whole: time-use summing to 24 hours, macronutrients summing to 100% of energy, cell or microbial proportions `thread:shared-compositional-parts` (missing; Estimate, Predict)
- Some samples were drawn after diagnosis or treatment began: the metabolome may be the disease, not its risk `thread:metab-sample-timing-vs-diagnosis` (missing; metabolomics)
- The table holds what you study several ways, inviting a choice after the results are seen `thread:shared-many-versions-of-the-exposure` (partial; Estimate, Predict)
- A polygenic score column whose weights may have been learned on these participants, or in another ancestry `thread:genomics-polygenic-score-provenance` (missing; Estimate, Predict; genomics)
- A recognized instrument brings its published scoring, missing-item rule, cut-points and clinically important difference, but only if it was used unmodified `thread:survey-instrument-published-scoring` (partial; Estimate, Predict; survey)

</details>

<details><summary>The omics chain · 11</summary>

- One sample drives correlations or sits far from the rest `thread:shared-omics-outlier-sample` (partial; Estimate, Predict)
- Missingness tracks abundance: blanks are below a detection limit, not lost at random `thread:metab-left-censored-nondetects` (engine only; metabolomics)
- A lab's assay or calibration changed across cycles, sites or calendar time `thread:clin-assay-change` (partial; clinical)
- Intake shifts by survey cycle, site, interviewer or food-composition release, and only some of those shifts are artifacts `thread:diet-assessment-batch` (partial; dietary)
- Samples split by processing batch on the leading components, and batch lines up with case status `thread:genomics-batch-aligned-with-case` (engine only; genomics)
- Cases were sequenced deeper than controls, so depth alone can predict case status `thread:genomics-depth-tracks-case` (engine only; genomics)
- The batches were already corrected before export, possibly with the outcome in the model `thread:metab-pre-corrected-batches` (partial; metabolomics)
- Cases and controls (or the groups you compare) were run in different parts of the sequence, batches or plates, so drift can impersonate the outcome `thread:metab-run-order-aligned-with-outcome` (partial; metabolomics)
- Library size (or urine dilution) differs between cases and controls `thread:shared-omics-sample-total-tracks-outcome` (engine only; Estimate, Predict)
- A processing variable (batch, plate, run, sequencing date, assay lot, interviewer) shifts the measurements and may line up with the outcome `thread:shared-process-aligned-with-outcome` (engine only; Estimate, Predict)
- External effect sizes or polygenic weights must use the same effect allele and strand as these genotypes `thread:genomics-allele-harmonization` (missing; Estimate, Predict; genomics)

</details>

<details><summary>Measurement error · 10</summary>

- Only a handful of people have a second recall, so λ itself is too uncertain to correct with `thread:diet-few-replicate-persons` (partial; dietary)
- Day 1 against day 2: much of each nutrient's spread is day-to-day noise, and the share differs by nutrient `thread:diet-day-to-day-variance` (engine only; dietary)
- An FFQ intake for everyone, and 24-hour recalls or a second instrument for the same nutrient in a subsample `thread:diet-ffq-recall-substudy` (partial; dietary)
- A recovery biomarker (doubly labeled water, urinary N, K, Na) can check self-report `thread:diet-recovery-biomarker` (missing; dietary)
- A nutrient whose two recalls barely rank people, so no correction can rescue it `thread:diet-too-noisy-to-correct` (partial; dietary)
- What you study is assigned from a group or predicted by a model (Berkson-type error), not measured with noise `thread:shared-exposure-assigned-or-predicted` (missing)
- Some answers came from a proxy (a parent, a spouse), not the participant `thread:shared-proxy-respondent` (missing)
- Replicate readings (bp_1, bp_2, bp_3; duplicate assays) reveal regression dilution `thread:shared-replicate-reliability` (partial)
- A better measure of what you study exists for some people (a biomarker, a weighed record, doubly labeled water) `thread:shared-validation-subsample` (partial)
- The first 24-hour recall covers the day before the blood or urine draw, so the biomarker reads yesterday's meal `thread:diet-recall-day-before-the-draw` (missing; dietary)

</details>

<details><summary>Family checks (every lens) · 9</summary>

- Shortcut check, under every lens: does a process (batch, site, date, plate, run order) line up with the outcome? `sentinel:S1` (partial)
- Leak in time check, under every lens: was this known at the moment of use, or only after the outcome? `sentinel:S2` (missing)
- Leak in meaning check, under every lens: is the outcome written inside a predictor? `sentinel:S3` (missing)
- Drift and transport check, under every lens: will the place, time or instrument of use look like this? `sentinel:S7` (missing; Predict)
- How well it measures check, under every lens: how much of this value is the person, and how much the instrument? `sentinel:K3` (missing)
- Causal place check, under every lens: where does this column sit between what you study and the outcome? `sentinel:K4` (missing; Estimate, Describe)
- Structure among variables check, under every lens: which columns are built from others? `sentinel:K5` (partial)
- Reference and context check, under every lens: compared with whom, on what scale, under what physiology? `sentinel:K7` (missing)
- Support check, under every lens: can these data support this claim where it is made? `sentinel:E1` (partial)

</details>

<details><summary>How numbers enter the model · 8</summary>

- Blanks follow a gate question by design: 'No' to ever drinking skips 'drinks per day' `thread:survey-skip-pattern-blanks` (partial; survey)
- Values pile up on a form's default (120/80 mmHg, 98.6 °F): entries, not measurements `thread:clin-default-value-entries` (partial; clinical)
- A deficiency or anemia flag uses one cut-off for everyone, when the reference cut depends on sex, age, pregnancy, altitude or smoking `thread:clin-population-specific-cutoffs` (missing; clinical)
- Many people report none: whether they form a non-consumer group depends on whether the zeros are people or days `thread:diet-mass-at-zero` (engine only; Estimate, Predict; dietary)
- Values heap at round numbers: terminal-digit preference in blood pressure, whole-kilogram self-reported weight, ages ending in 0 or 5 `thread:shared-heaping-digit-preference` (missing)
- Many respondents sit at the scale's minimum or maximum: the instrument runs out of room there `thread:survey-floor-ceiling` (partial; survey)
- A predictor coded 1–5 (education, a single Likert item) is a set of ordered categories, not equally spaced amounts `thread:survey-ordinal-predictor-spacing` (partial; Estimate, Predict; survey)
- Some predictors are exact functions of others (BMI from weight and height, eGFR, HOMA-IR, energy from macronutrients, nutrient densities) `thread:shared-derived-predictors` (missing; Estimate, Predict)

</details>

<details><summary>What a prediction model may use · 7</summary>

- Hundreds of diagnosis and medication codes: no code does not mean no disease, and longer records hold more codes `thread:clin-coded-history-absence` (missing; clinical)
- A candidate predictor was recorded after the moment the model will be used, or after the outcome itself `thread:clin-predictor-after-prediction-time` (partial; Estimate, Predict; clinical)
- A predictor was recorded after the outcome, or after the moment the model would be used `thread:shared-predictor-after-the-index` (partial)
- The outcome is rare, so accuracy and AUC hide what matters: how many flagged patients have it, at the threshold that would change care `thread:clin-rare-outcome-threshold` (partial; Estimate, Predict; clinical)
- The model was trained on a two-day mean but will see one recall (or an FFQ) in use `thread:diet-deployment-measurement-heterogeneity` (missing; Predict; dietary)
- The model will meet samples from another lab, platform or study, perhaps one at a time, and the training data already hold more than one `thread:genomics-new-site-new-platform` (partial; Estimate, Predict; genomics)
- The next batch, lab or platform will not look like these: the honest score is across batches `thread:metab-cross-batch-transport` (partial; Estimate, Predict; metabolomics)

</details>

<details><summary>Questionnaire scales · 6</summary>

- An item works differently by sex, language, mode or occasion at the same trait level (differential item functioning) `thread:survey-dif` (missing; survey)
- The construct was measured twice, or against a reference: a retest, a substudy, or self-report beside a measurement `thread:survey-repeat-or-reference-measurement` (engine only; survey)
- A summed scale is one number only if its items measure one thing `thread:survey-dimensionality` (partial; Estimate, Predict; survey)
- An item runs against its scale. The instrument's key decides, and the key is checked again after it is declared `thread:survey-reverse-keying` (partial; Estimate, Predict; survey)
- A second factor made only of the reverse-worded items is a wording effect, not a second construct `thread:survey-wording-method-factor` (partial; survey)
- A score's reliability in this sample sets how much its coefficient is diluted and how much any model can explain. The correction depends on purpose `thread:survey-reliability-attenuation` (engine only; Estimate, Predict; survey)

</details>

<details><summary>Which models fit · 6</summary>

- Rows are follow-up intervals (start, stop, event) with time-varying covariates, not people or visits `thread:shared-rows-are-follow-up-intervals` (missing)
- A small clinical block sits beside thousands of omics features, and one penalty treats them alike `thread:shared-unequal-blocks` (partial; Predict, Describe)
- Many short questionnaires per person (diaries, EMA): a scale has a within-person meaning and a between-person meaning `thread:shared-ema-within-between` (missing)
- Participants lost to follow-up differ at baseline from those who stayed `thread:shared-informative-dropout` (partial)
- Respondents leave a panel between waves, and who leaves depends on what they said before `thread:shared-panel-attrition` (missing)
- The table already holds the established risk factors or an existing score, so the question is what the new markers add `thread:clin-incremental-value` (missing; Predict; clinical)

</details>

<details><summary>Many tests · 4</summary>

- Sixty nutrient columns are a scan, not a hypothesis, and they behave like far fewer independent tests `thread:diet-nutrient-wide-scan` (partial; Estimate, Predict; dietary)
- Hundreds of features are a few dozen compounds (adducts, isotopes, in-source fragments, both ion modes) `thread:metab-one-compound-many-features` (partial; Estimate, Predict; metabolomics)
- Hundreds of questionnaire items screened one by one: many tests, related items, and replication across cycles `thread:survey-question-wide-scan` (missing; Estimate, Predict; survey)
- With 392 features and 72 people, noise alone reaches a high score: the bar any finding must clear `thread:metab-wide-noise-ceiling` (partial; Estimate, Predict; metabolomics)

</details>

<details><summary>The causal and time-varying lanes · 3</summary>

- Repeated FFQs over follow-up: average up to each event, and stop updating at an intermediate diagnosis `thread:diet-repeated-ffq-cumulative` (partial; dietary)
- Treatment started during follow-up changes who has the outcome, so the prognosis being predicted is 'under current care' `thread:clin-treatment-during-follow-up` (partial; Predict, Estimate; clinical)
- What you study is defined by something that happens after follow-up starts (immortal time) `thread:shared-time-zero` (partial; Estimate, Predict)

</details>

<details><summary>The energy model · 2</summary>

- A collaborator's file already holds energy-adjusted or density nutrients, which fail the nutrient test and must not be adjusted twice `thread:diet-delivered-already-adjusted` (missing; Estimate, Predict; dietary)
- The energy budget does not close: a source (usually alcohol) sits inside total energy but in no column `thread:diet-energy-budget-gap` (engine only; Estimate, Predict; dietary)

</details>

<details><summary>Modifiers · 1</summary>

- A genotype predicts what people eat (lactase persistence and milk, ALDH2 and alcohol), so gene–diet questions carry gene–diet correlation and sparse cells `thread:genomics-genotype-shapes-the-diet` (partial; Estimate, Predict; genomics)

</details>

<details><summary>The design · 1</summary>

- In an unblinded trial, the outcome is self-reported or judged by someone who knows the arm `thread:shared-outcome-assessed-knowing-the-arm` (new scope; Estimate, Describe)

</details>


**Confirm sweep, last** (63): defaults set for you, each with an alternative that would change a number.

- Could [sugar]'s effect differ by another characteristic, or combine with a second factor? `q:modification` (partial; Estimate)
- Also estimate the effect with a method that learns the adjustment flexibly (double ML or TMLE)? `q:causal` (partial; Estimate)
- You are testing many related factors at once (every nutrient, every metabolite): how should the many tests be accounted for? `decision:set_multiplicity` (partial; Estimate; metabolomics, genomics, dietary, survey)
- Models are compared by repeated cross-validation, corrected for picking the best `default:validation-scheme` (partial; Predict)
- Rules repeated inside each fold: curves for every number, a variance filter, a class-imbalance correction `decision:set_levers` (engine only; Predict, Estimate)
- What each model is given: keep blanks as blanks or fill them, put columns on one scale, reshape or cap `decision:set_recipe` (missing; Predict)
- Tree models try both: keep blanks as blanks or take the shared fill, chosen inside each fold `default:trees-try-both-blanks` (missing; Predict)
- Tuning: automatic for [boosted trees, forest, XGBoost] · about N min (or standard settings, or set by hand) `decision:set_tuning` (missing; Predict)
- Each predictor enters as each model takes it; a spline benchmark is on the shelf `default:form-under-prediction` (on screen; Predict)
- Causal lane settings: learner chosen for the table's size, 5 folds repeated 5 times `default:causal-settings` (engine only; Estimate)
- Summarize thousands of features into components inside each fold `decision:in-fold-pca-omics` (missing; Predict; metabolomics, genomics)
- Log2 after the normalization, the D-ratio filter (with pooled QCs) and autoscaling, each fitted in each training fold `default:omics-in-fold-steps` (engine only; Estimate, Predict; metabolomics, genomics)
- Adjust for the factors the randomization balanced and the baseline measures you named beforehand (no search for other adjustments) `decision:trial-precision-adjustment` (new scope; Estimate)
- For the surveyed population, each model uses its survey-weighted form (linear, logistic, multinomial, ordinal, Cox); unweighted is blocked and recorded `default:survey-estimators` (engine only; Estimate, Describe; dietary, clinical, survey)
- Groups were randomized: analyze at the group level with intervals for few clusters `decision:cluster-randomized` (new scope; Estimate)
- People were matched: compare within each matched set (conditional logistic regression) `decision:matched-sets` (new scope; Estimate)
- Steps of 100 kcal; a band from refits when you ask (Taylor band under the survey design) `default:substitution_step_and_band` (partial; Estimate, Predict; dietary)
- Each lever is offered first as an in-fold rule; a lever pulled by hand after an outcome view is disclosed as outside the corrected score (prediction) or as made after the view (inference) `default:explore:levers-in-fold-first` (engine only; Predict, Estimate)
- Some predictors barely vary `noticing:explore::low_variance` (engine only; Estimate, Predict)
- Intervals: robust (HC3), clustered by person (CR2), or by the survey design; Firth's fit if a column separates the outcome `default:interval-method` (engine only; Estimate)
- Confirm what was set for you in Models (only defaults whose alternative would change a number) `other:confirm-sweep` (missing)
<details><summary>Noticings stated with a default (42)</summary>

- Whether a test was ordered carries information: a clinician suspected something `thread:clin-informative-test-ordering` (partial; clinical) · on How numbers enter the model
- FFQ frequency categories are consumption frequencies, not Likert responses and not amounts `thread:diet-ffq-frequency-not-a-scale` (missing; dietary) · on How numbers enter the model
- A category has too few rows to estimate, or is missing from some folds `thread:shared-sparse-levels` (partial; Estimate, Predict) · on How numbers enter the model
- A predictor's tails are thin, so the far ends of effect curves rest on a few people `thread:shared-tails-and-support` (partial; Estimate, Predict) · on How numbers enter the model
- A predictor is zero for many people and continuous above zero (pack-years in never smokers, supplement dose in non-users) `thread:shared-zero-mass-predictor` (missing; Estimate, Predict) · on How numbers enter the model
- Reported amounts pile onto round numbers (10 and 20 cigarettes, 7 and 8 hours, 30 minutes): rounding, not real mass `thread:survey-heaped-answers` (missing; survey) · on How numbers enter the model
- A column is a clinical formula or a cut-point of other columns (eGFR, Friedewald LDL, BMI, MAP, 'hypertension yes/no', a lab's H/L flag) `thread:clin-derived-clinical-variables` (missing; Estimate, Predict; clinical) · on How numbers enter the model
- Few events spread over intake categories leave some categories with almost none expected `thread:diet-rare-events-sparse-categories` (partial; Estimate, Predict; dietary) · on How numbers enter the model
- Quintiles of absolute intake sort people by sex and body size before they sort them by diet `thread:diet-quantiles-sort-by-sex-and-size` (partial; Estimate, Predict; dietary) · on How numbers enter the model
- Several microarray probes map to one gene, and some probes map to several `thread:genomics-many-probes-one-gene` (partial; genomics) · on Many tests
- Self-reported height, weight or BP sits beside or among measured values, and the two differ systematically `thread:clin-self-report-vs-measured` (missing; clinical) · on Measurement error
- One reading or one blood draw is a noisy measure of a person's usual level (regression dilution) `thread:clin-single-reading-dilution` (partial; clinical) · on Measurement error
- Day-to-day error grows with intake, so correction must work on the scale where the error is additive, which is the scale the model uses `thread:diet-error-grows-with-intake` (partial; dietary) · on Measurement error
- One blood draw captures little of a person's usual level of a metabolite `thread:metab-low-biological-icc` (partial; metabolomics) · on Measurement error
- Day 2 can read lower than day 1 and weekends can differ: these are the instrument's effects, not diet change `thread:diet-recall-nuisance-effects` (partial; dietary) · on Measurement error
- A null or a low importance may be an intake the instrument cannot see `thread:diet-invisible-exposure` (missing; Estimate, Predict; dietary) · on Measurement error
- A planned modifier's groups do not share the same range of what you study `thread:shared-interaction-support` (partial; Estimate, Predict) · on Modifiers
- The score is an index defined by its components (a diet-quality score), not a reflective scale `thread:survey-formative-or-reflective` (engine only; survey) · on Questionnaire scales
- Whether a higher score means worse or better decides what every coefficient's sign says `thread:survey-score-direction` (missing; survey) · on Questionnaire scales
- Forty collinear items are one construct: the score for inference, score against items compared by resampling for prediction, and explained as a group `thread:survey-items-or-score` (partial; Estimate, Predict; survey) · on Questionnaire scales
- The scale you study and the outcome share items or content `thread:survey-item-overlap-with-outcome` (partial; Estimate, Predict; survey) · on Questionnaire scales
- A change in coding system (ICD-9-CM to ICD-10-CM on 1 Oct 2015) shows up as a jump in recorded prevalence `thread:clin-coding-system-transition` (missing; clinical) · on The adjustment set
- A biomarker swings with the season of collection, and in NHANES season is tied to latitude `thread:clin-season-of-draw` (missing; clinical) · on The adjustment set
- Recalls were collected across (or within) particular seasons, and seasonal foods and nutrients move with the calendar `thread:diet-season-of-assessment` (missing; dietary) · on The adjustment set
- Something usually adjusted for in this question is not in the file `thread:shared-missing-confounder` (partial; Estimate, Predict) · on The adjustment set
- A diet-quality index is already an energy density and a formative score, and its population mean needs a ratio `thread:diet-quality-index-is-a-density` (partial; dietary) · on The energy model
- A swap of k kcal moves some people off any diet that was observed `thread:diet-substitution-support` (engine only; Estimate, Predict; dietary) · on The energy model
- An effect per one SD of an energy residual means nothing to a reader; per a serving or 5% of energy does `thread:diet-meaningful-increment` (partial; Estimate, Predict; dietary) · on The energy model
- Items in one analysis ask about different windows: past 30 days, past 12 months, 'usually', 'ever' `thread:survey-recall-period-mismatch` (missing; survey) · on The exposure and its effect
- A few transcripts (globin, rRNA, mitochondrial) take most of the reads, or most genes shift one way, so depth-only scaling misleads `thread:genomics-few-transcripts-take-the-reads` (partial; genomics) · on The omics chain
- Which features are identified, and how confidently (MSI level), sets what a name in a result may claim `thread:metab-annotation-confidence` (partial; metabolomics) · on The omics chain
- Samples differ in overall concentration (dilution, total signal), and the totals may differ with the outcome `thread:metab-dilution` (engine only; metabolomics) · on The omics chain
- The pooled QC does not contain what the study samples contain `thread:metab-qc-representativeness` (missing; metabolomics) · on The omics chain
- Genes move in large correlated modules, so thousands of columns carry far fewer independent signals and importance splits across a module `thread:genomics-coexpression-modules` (missing; Estimate, Predict; genomics) · on The omics chain
- Two features are a substrate and its product: their ratio indexes a pathway step, and it cancels dilution `thread:metab-ratio-features` (missing; Estimate, Predict; metabolomics) · on The omics chain
- Interview-process columns (foods named, respondent, language) predict health through frailty, not diet `thread:diet-recall-process-shortcuts` (partial; Estimate, Predict; dietary) · on What a prediction model may use
- Outcome rates and case mix drift across calendar years, so yesterday's calibration is not tomorrow's `thread:clin-calendar-drift` (partial; Estimate, Predict; clinical) · on What a prediction model may use
- A derived diet feature divides by body weight, and the outcome is made of the same weight measurement `thread:diet-ratio-features-share-the-outcome` (partial; Estimate, Predict; dietary) · on What a prediction model may use
- People were enrolled because a lab was high, so it falls on re-measurement whatever the treatment `thread:clin-regression-to-the-mean` (missing; clinical) · on Which models fit
- Who leaves the study early depends on who they are `thread:clin-loss-to-follow-up` (partial; clinical) · on Which models fit
- For a survey's mortality follow-up, age can be the clock instead of time on study `thread:clin-age-time-scale` (partial; clinical) · on Which models fit
- In routine-care follow-up, sicker patients come back sooner and more often, so the visits themselves carry information `thread:clin-informative-visit-process` (missing; clinical) · on Which models fit

</details>

**For the record**, collapsed and not counted toward progress (20):

<details><summary>20 lines</summary>

- A second model further adjusted for [BMI] runs beside the main one, because its timing is unknown `default:secondary-further-adjusted` (engine only; Estimate)
- A quintile table is produced beside the curve, with the trend test labeled customary `default:quintiles-beside-spline` (engine only; Estimate)
- Scores are design-based: whole PSUs within strata, every loss weighted `default:design_based_cv` (engine only; Predict; dietary, clinical, survey)
- The plan is fixed the first time estimates appear; later changes are marked `other:plan-lock` (partial; Estimate)
- A category has almost no rows or no events, so its estimate runs to infinity `thread:clin-sparse-levels-separation` (engine only; Estimate, Predict)
- Each feature's technical noise in the QCs, relative to its spread across people, is a reliability, not just a filter `thread:metab-feature-technical-reliability` (engine only; metabolomics)
- One baseline diet assessment stands for decades of follow-up `thread:diet-single-baseline-long-follow-up` (missing; Estimate, Predict; dietary)
- In a small sample, the scale's structure and reliability cannot be re-estimated, so the published ones are the evidence `thread:survey-small-sample-psychometrics` (missing; Estimate, Predict; survey)
- High-fiber eaters also smoke less and exercise more; read the estimate with its E-value `thread:diet-healthy-lifestyle-cluster` (partial; Estimate, Predict; dietary)
- On arrays, many probes sit at background and a few at the scanner ceiling, so the floor and the ceiling are not values like the rest `thread:genomics-array-detection-floor` (missing; genomics)
- Most genes barely register, and an outcome-blind filter decides how many are really tested `thread:genomics-low-expression-filter` (partial; Estimate, Predict; genomics)
- A feature's variance depends on its mean; on counts the settled scale already handles it, and a guard checks that filtering did `thread:genomics-mean-variance-scaling` (missing; Estimate, Predict; genomics)
- A dilution series shows which features respond to concentration at all, and which saturate `thread:metab-dilution-linearity` (partial; metabolomics)
- Variance rises with abundance and explodes near the detection floor after the log `thread:metab-variance-structure` (partial; Estimate, Predict; metabolomics)
- QC RSD fell after correction, but the correction was fitted to make that number small `thread:metab-correction-honest-check` (partial; Estimate, Predict; metabolomics)
- 5,000 rows but only 61 events: the events, not the rows, set how much the data can say `thread:shared-events-not-rows` (engine only; Estimate, Predict)
- The variance has only as many degrees of freedom as PSUs minus strata, which caps how much the model can ask `thread:survey-design-df-budget` (partial; survey)
- Unequal weights shrink the sample: the model's complexity budget is the effective n (and the effective events), not the row count `thread:survey-weight-effective-n` (partial; survey)
- With far more features than samples, the data cannot pin down a unique model `thread:genomics-more-genes-than-samples` (engine only; genomics)
- What fold change this study could have seen, given its own technical and biological variance `thread:metab-detectable-effect` (missing; Predict, Estimate; metabolomics)

</details>

**Shown on the canvas or as a refusal**, not an objective (16):

<details><summary>16 previews, views and refusals</summary>

- Explore's levers previewed: which predictors bend or leave in each training fold, the terms kept by selection, the risks against the threshold, shrinkage `preview:explore-levers` (engine only; Predict)
- Concerns stated beside the choices: below the minimum sample size; totals track the outcome; energy explains most of a nutrient `noticing:shelf-and-design-concerns` (partial)
- Blocked until recorded or refused, each with a way forward `refusal:leash-in-models` (partial)
- This swap can't be drawn as asked (energy sources left out, nested parts, unsettled units) `refusal:substitution_blocked` (partial; Estimate, Predict; dietary)
- This model has no estimator for the surveyed population `refusal:population_blocked` (engine only; Estimate; dietary, clinical, survey)
- Rows in each sensitivity analysis `preview:sensitivity_views` (engine only; Estimate, Predict)
- Each person's donor and recipient, before and after one step `preview:substitution_views` (engine only; Estimate, Predict; dietary)
- The analysis flowchart: what will be fit, on which rows, in what order (savable as a figure) `flowchart:analysis` (missing)
- The plan is already fixed and the lock can't be undone `refusal:plan_already_locked` (engine only; Estimate)
- The outcome bends against a continuous predictor, and choosing a form by eye moves optimism outside the score `thread:shared-curvature-seen-in-explore` (engine only; Estimate, Predict)
- The relationship with what you study looks different in one subgroup `thread:shared-subgroup-seen-in-explore` (partial; Estimate, Predict)
- Your outcome (weight, BMI, diabetes) is energy-related, so energy may lie on the path `thread:diet-energy-related-outcome` (engine only; Estimate, Predict; dietary)
- Age, sex, BMI, kidney function and fasting move much of the metabolome, and may differ by outcome `thread:metab-clinical-factors` (partial; Estimate, Predict; metabolomics)
- A predictor nearly is the outcome: another measure of it, its definition, a function of it, or its consequence `thread:shared-outcome-proxy` (partial; Estimate, Predict)
- The event is common, so an odds ratio no longer reads as a risk ratio `thread:shared-common-outcome-measure` (engine only; Estimate)
- A feature present in one group and absent in the other is information, not a missing value `thread:metab-group-specific-detection` (partial; Estimate, Predict; metabolomics)

</details>

### 6 · Results

Each result is an exhibit, and each appears only after Fit (`other:fit`). The section "After training" below lists each one with its wordings, placement and pre-included flag. Under Predict, the final model, the threshold and the recalibration are fixed here; the open-noticings gate is cleared; and then the held-out rows open once.

**Decide, in order** (59).

1. How should this finding read? Pick a drafted wording or write your own `decision:exhibit_wording` (new scope)
2. Where does it go: Results, Discussion, Supplement, or left out? `decision:exhibit_placement` (new scope)
3. The effect of [what you study] on [the outcome] in each model you declared: unadjusted, Model 1, the primary, Model 3 `exhibit:table2` (engine only; Estimate)
4. The risk difference and risk ratio, averaged over your participants `exhibit:marginal_contrasts` (engine only; Estimate)
5. The effect of always versus never [what you study] over follow-up `exhibit:time_varying_estimate` (engine only; Estimate)
6. The effect estimated by [double ML / TMLE], with its assumptions and overlap `exhibit:causal_estimate` (engine only; Estimate)
7. A check failed: show the effect before and after the midpoint, refit without the influential rows, or keep the estimate labeled? `decision:respond_diagnostic` (engine only; Estimate)
8. The primary beside the model further adjusted for [BMI] `exhibit:secondary_further_adjusted` (engine only; Estimate)
9. Your estimate corrected for day-to-day variation in the recalls, beside the uncorrected one `exhibit:regression_calibration` (engine only; Estimate; dietary)
10. How reliably each scale measures, and its coefficient corrected for that `exhibit:scales_reliability` (engine only; Estimate, Predict; survey)
11. Does the effect differ across [modifier]? On both scales, against one reference `exhibit:effect_modification` (engine only; Estimate)
12. Every factor in the family you tested (every nutrient, every metabolite), each with its estimate and q-value `exhibit:exposure_family` (partial; Estimate; metabolomics, genomics, dietary, survey)
13. The same model on other rows: every row, and each screen you declared `exhibit:declared_sensitivity` (engine only; Estimate, Predict)
14. How strong would something unmeasured, affecting both, have to be to explain this away? `exhibit:unmeasured_confounding` (engine only; Estimate)
15. Which of your decisions mattered? The estimate across every alternative you declared `exhibit:which_decisions_mattered` (partial; Estimate)
16. Backward elimination as a labeled sensitivity analysis `result:selection_sensitivity_inference` (engine only; Estimate)
17. The other coefficients: adjustment terms, not effect estimates `exhibit:table2_appendix` (engine only; Estimate)
18. You changed the plan after seeing estimates: this runs as a secondary analysis beside the primary `decision:post_lock_secondary` (partial; Estimate, Predict, Describe)
19. How well each model predicts new people, and the one result you report `exhibit:performance_table` (partial; Predict)
20. How far apart the models are, pair by pair `exhibit:family_comparisons` (engine only; Predict)
21. Shrink the coefficients by the calibration slope before the model is used? `decision:set_updating` (engine only; Predict)
22. Predictors outnumber rows: run the nested cross-validation interval? (about N minutes) `decision:nested_cv_offer` (partial; Predict)
23. Before you open the held-out rows: these noticings still change the honest score. Decide or dismiss each `gate:open-noticings-before-seal` (missing; Predict)
24. Choose your final model on cross-validation, then open the held-out rows once `q:open_seal` (on screen; Predict)
25. The held-out score of the model you declared final `exhibit:held_out_result` (on screen; Predict)
26. Are the predicted risks right? Observed against predicted `exhibit:calibration_curves` (engine only; Predict)
27. Would using the model help decisions across reasonable risk thresholds? `exhibit:decision_curve` (engine only; Predict)
28. How well it predicts within each group `exhibit:subgroup_performance` (engine only; Predict)
29. Scored on each site by models fit on the others `exhibit:internal_external_cv` (engine only; Predict)
30. How well it predicts in the surveyed population `exhibit:design_based_cv` (engine only; Predict; dietary, clinical, survey)
31. What the interpretable model costs or gains against the best flexible one `exhibit:interpretable_cost` (engine only; Predict)
32. How often each predictor was selected across folds `exhibit:inclusion_frequencies` (engine only; Predict)
33. Earlier versions you changed after seeing scores, kept in the comparison `exhibit:versions_comparison` (missing; Predict)
34. The settings each tuned model used, and how much they varied across folds `exhibit:settings_table` (missing; Predict)
35. The fitted linear model's coefficients `exhibit:fit_coefficients` (on screen; Predict, Estimate)
36. Draw the held-out rows again, recorded as after their scores were seen `decision:reseal` (engine only; Predict)
37. Who your participants are (Table 1) `exhibit:table1` (missing)
38. Usual intake of [nutrient] in the population, and the share below the requirement `exhibit:usual_intake_distribution` (engine only; Describe; dietary)
39. Survey-weighted means and prevalence by group `exhibit:weighted_prevalence_by_group` (new scope; Describe; dietary, clinical, survey)
40. How it changed across survey cycles `exhibit:trends_across_cycles` (new scope; Describe; dietary, survey)
41. The dietary patterns found, described by their loadings `exhibit:dietary_patterns` (new scope; Describe, Estimate, Predict)
42. Subgroups of similar people `exhibit:clusters` (new scope; Describe)
43. How well two methods, or two models, agree `exhibit:bland_altman` (new scope; Describe, Predict)
44. The effect of the assigned treatment (intention to treat), with per-protocol beside it `exhibit:trial_results` (new scope; Estimate)
45. Describe how each fitted model uses its inputs? `decision:set_explain` (engine only; Predict, Estimate)
46. Which inputs each model leans on, and how stable that is across refits `exhibit:shap_importance` (engine only; Predict, Estimate)
47. Which inputs each model leaned on, side by side (one table across models) `exhibit:cross-model-importance` (partial; Predict, Estimate)
48. Each top input's curve in every model, on shared axes `exhibit:inductive_bias_curves` (engine only; Predict, Estimate)
49. Each person's prediction split into its inputs `exhibit:shap_beeswarm_observations` (engine only; Predict, Estimate)
50. Pairs of inputs the model combines `exhibit:interactions_h` (engine only; Predict, Estimate)
51. What each model is: its equation, its trees, its shrinkage path `exhibit:architecture_lane` (engine only; Predict, Estimate)

Noticings decided here (8), each on the card or question it changes. Each is an open noticing until it is decided or dismissed:

<details><summary>After the fit · 6</summary>

- This null can't say there is no effect: its interval still includes the smallest effect that matters `thread:shared-null-is-inconclusive` (missing; Estimate)
- Your model says HDL raises risk; that reversed once TC/HDL entered `thread:clin-expected-directions` (partial; Predict, Estimate; clinical)
- Your p-values pile toward 1, so the model is wrong somewhere `thread:genomics-pvalue-distribution` (missing; Estimate; genomics, metabolomics)
- Your top feature is metformin: the model found the prescription `thread:metab-treatment-marker` (missing; Predict, Estimate; metabolomics)
- There is no second measure here: a null can't rule out a slope several times larger `thread:diet-borrowed-validity-coefficients` (partial; Estimate; dietary)
- Only a fraction of features have names, so any pathway reading runs over what the assay could name `thread:metab-enrichment-background` (missing; Estimate, Predict; metabolomics)

</details>

<details><summary>Family checks (every lens) · 2</summary>

- Noise and multiplicity check, under every lens: does the signal beat noise and the number of looks taken? `sentinel:E2` (partial; Estimate, Predict, Describe)
- Reading the result check, under every lens: what can this explanation or null claim, and for whom? `sentinel:E3` (missing; Estimate, Predict)

</details>


**Confirm sweep, last** (5): defaults set for you, each with an alternative that would change a number.

- Threshold range 5% to 50%, and a threshold chosen inside each fold `default:decision_threshold` (engine only; Predict)
- Curves by accumulated local effects (partial dependence ranked lower) `default:explain_curve_method` (engine only; Predict, Estimate)
- Calibration read at the median follow-up time (no horizon was declared) `default:horizon_calibration` (engine only; Predict)
- Robustness value first for a linear outcome; the E-value on the spread of the population your result describes `default:unmeasured_order_and_sd` (engine only; Estimate)
- Confirm what was set for you in Results (only defaults whose alternative would change a number) `other:confirm-sweep:results` (missing)

**For the record**, collapsed and not counted toward progress (19):

<details><summary>19 lines</summary>

- Did the primary model's assumptions hold? `exhibit:diagnostics` (engine only; Estimate)
- Do the residuals of the primary model follow the shape it assumes? (residual Q-Q) `exhibit:residual-qq` (missing; Estimate, Predict)
- A no-predictor baseline and a regression with curves are always fitted beside the chosen models `exhibit:spline_benchmark` (engine only; Predict)
- Every analysis that was run, listed `other:results_inventory` (missing)
- Unadjusted always shown; only the rows of what you study read as effects; Model 3 labeled `default:model_sequence_display` (engine only; Estimate)
- No cross-validated R² under inference `default:fit_statistics_withheld` (engine only; Estimate)
- Compared on a strictly proper score over at least 10 x K folds; AUC reported beside it `default:proper_primary_and_substrate` (partial; Predict)
- Optimism-corrected by refitting the whole pipeline on 500 or more resamples `default:bootstrap_optimism` (engine only; Predict)
- Corrected for picking the best of several models `result:bbc_cv` (partial; Predict)
- Choices made by hand after looking at the outcome, which the score does not cover `result:hand_levers` (engine only; Predict, Estimate)
- What you study and the outcome come from the same questionnaire at one sitting `thread:survey-common-method` (missing; Estimate; survey)
- [family] predicts no better than the mean, so there is nothing to explain `thread:shared-model-no-better-than-baseline` (engine only; Predict)
- Two models score the same but rank the predictors differently `thread:shared-rashomon-disagreement` (partial; Predict)
- The average curve is flat, but it rises steeply in one group `thread:shared-learned-interaction` (partial; Predict)
- Only 3 of your 12 panel features appear in more than half the refits `thread:panel-instability` (partial; Predict; metabolomics, genomics)
- The score sits within what shuffled labels reach `thread:noise-ceiling` (missing; Predict)
- These importances describe the oversampled sample, not the country `thread:survey-explanations-describe-the-sample` (missing; Predict, Estimate; survey, dietary, clinical)
- This cell rests on too few effective participants to publish `thread:survey-estimate-reliability` (partial; Describe, Estimate; survey, dietary)
- A top-feature list read by pathway needs, as its background, the features that could have been selected `thread:genomics-enrichment-background` (missing; Estimate, Predict; genomics)

</details>

**Shown on the canvas or as a refusal**, not an objective (10):

<details><summary>10 previews, views and refusals</summary>

- The best model's own score can't be the result after you compared several on the same rows `refusal:winner_own_score` (engine only; Predict)
- An explanation describes the model's predictions, not what changing an intake would do `refusal:explain_as_effect` (engine only; Predict, Estimate)
- Estimates wait until the questions they rest on are answered `refusal:served-gate` (engine only; Estimate, Predict)
- The held-out rows can't be opened yet (no final model, stale fit, already opened) `refusal:open_seal_refusals` (partial; Predict)
- The primary result cannot be dropped from Results, and factors in a tested family cannot be trimmed by their p-values `refusal:report_by_significance` (partial; Estimate)
- What opening does: the held-out rows are scored once, then stand `preview:open_seal_views` (engine only; Predict)
- What the recorded response would show beside the estimate `preview:diagnostic_views` (engine only; Estimate)
- Where each curve method evaluates the model `preview:explain_views` (engine only; Predict, Estimate)
- Shrinkage's effect on calibration; the decision curve over a threshold range `preview:updating_and_intended_use` (engine only; Predict)
- What the estimate is: which effect, on what scale, conditional on what `result:estimand_caption` (engine only; Estimate)

</details>

### 7 · Write-up

One manuscript in two views: the rail, present in every stage, and full width here.

**Order:**
1. **What the export still waits for.** Each item's exit opens the stage that owns it, and that stage drops back with the reason (`export/gate.py:missing`).
2. **The merge of tracks.**
3. **The placements, as reviewed.**
4. **The Discussion drafts.**
5. **Author-only text.** It never blocks the export; it becomes `\todo`.
6. **The export.**

Write-up's only number-changing default is the small-cell threshold, so its Confirm sweep is usually absent. The rail is a view of the manuscript, not a stage: only the objectives above fill Write-up's segment (Settled here).

**Decide, in order** (10).

1. What the export still waits for, each with a way forward `refusal:export-gate` (engine only)
2. Your file changed since TurboTab read it: put it back, or start a new project `refusal:input-changed` (engine only)
3. One paper from several goals: shared methods once, then each goal's models and results `decision:merge-tracks` (missing)
4. Limitations drafted from what was noticed: keep each one in the Discussion, edit it, or drop it `noticing:limitations-draft` (missing; shared, dietary, clinical, metabolomics, survey)
5. Things found only after the fit: no better than the baseline, unstable, reversed direction, p-value pile-up `noticing:late-born-threads` (missing; Estimate, Predict)
6. The export would show identifiers or cells small enough to identify someone `noticing:identifiable-values` (missing)
7. What only you can write: title, objectives, setting, dates, ethics, funding, sources `decision:author-owed-items` (partial)
8. Export the analysis for the manuscript `export:bundle` (engine only; Estimate, Predict, Describe)
9. Overleaf-ready LaTeX project (zip) `export:latex-overleaf` (new scope)
10. Word document `export:word` (new scope)

**Confirm sweep, last** (2): defaults set for you, each with an alternative that would change a number.

- Cells under 11 participants are suppressed (change the threshold to your data-use agreement's) `default:small-cell-threshold` (missing; clinical, survey, dietary)
- Confirm what was set for you in Write-up (only defaults whose alternative would change a number) `other:confirm-sweep:writeup` (missing)

**For the record**, collapsed and not counted toward progress (32):

<details><summary>32 lines</summary>

- Sections follow STROBE-nut (estimating an effect, or describing) or TRIPOD+AI (prediction) `default:methods-guideline-order` (engine only; Estimate, Predict, Describe)
- Answers you changed before seeing any result are left out; changes after are kept and marked `default:superseded-folded-out` (engine only; Estimate, Predict)
- Decisions made after the estimates were seen (or the held-out rows opened) `result:after-estimates-section` (partial; Estimate, Predict)
- Counts in the methods match the participant flow as it stands now `default:restated-counts` (engine only)
- Survey sentences are restated on your current survey answer `default:survey-sentences-restated` (engine only; Estimate; survey, dietary, clinical)
- Paragraphs the analysis wrote: model sequence, imputation, cross-validation, calibration `result:analysis-paragraphs` (engine only; Estimate, Predict)
- One sentence says where the provenance record is and what a replay checks `default:reproducibility-sentence` (engine only)
- The plan is described as declared in TurboTab before any estimate was shown, never as 'preregistered' `default:plan-wording` (engine only; Estimate)
- Each methods sentence a check wrote, with its evidence inside `noticing:thread-sentences` (missing; shared, dietary, clinical, metabolomics, genomics, survey)
- Supplement table S1: what the data showed, what you said it meant, what it changed `exhibit:supplement-s1` (missing)
- Checked, nothing beyond its reference (in the supplement) `exhibit:clean-checks` (missing)
- What was screened before the plan, and whether the outcome's associations were looked at `result:ida-paragraph` (missing)
- Checklist items answered by what was noticed `noticing:checklist-thread-anchors` (partial; Estimate, Predict)
- For each bias domain, what was found that bears on it (never a rating) `exhibit:bias-evidence-table` (missing; Estimate, Predict)
- How many factors were tested, and how that was accounted for `noticing:number-of-tests` (engine only; Estimate)
- Who these results apply to `noticing:applicability` (partial)
- A supplementary table of what was read from your data, cited by one sentence `noticing:settled-readings-table` (partial)
- Figures and tables are numbered by where you place them `default:figure-and-table-numbering` (partial)
- Figures are drawn as published: serif, grayscale, dash patterns, numbered caption `default:journal-figure-style` (engine only)
- Scores on survey data are labeled unweighted, describing these participants, not the population `default:unweighted-caption` (engine only; Predict; survey, dietary)
- Defaults that changed nothing on this table, listed only in the export `default:silent-defaults-in-export` (missing)
- STROBE-nut checklist: where each item is answered, and what only you can supply `exhibit:strobe-nut-checklist` (engine only; Estimate, Describe)
- TRIPOD+AI checklist: where each item is answered, and what only you can supply `exhibit:tripod-ai-checklist` (engine only; Predict)
- The checklist as it stands now, with what the export still waits for `exhibit:live-checklist` (engine only; Estimate, Predict, Describe)
- CONSORT checklist (trials) `exhibit:consort-checklist` (new scope; Estimate; clinical, dietary)
- Other reporting items the checks answer: RECORD, STROBE-ME, MIAME/MINSEQE, STARD, RoB 2 `exhibit:other-guidelines` (missing; clinical, metabolomics, genomics, shared)
- The analysis plan as declared, with its time and SHA-256, for registration `export:analysis-plan` (engine only; Estimate)
- The provenance record: decisions, input hashes, engine and package versions, matrix hashes, every reported number `export:provenance-record` (engine only)
- Anyone can replay this analysis and check every number `export:replay` (engine only)
- References with every DOI checked against Crossref `export:refs-bib` (new scope)
- The supplement: S1, clean checks, the screens, saved figures, analyses left out, settled readings `export:supplement-document` (missing)
- The results tables in the bundle (CSV and Markdown) `export:results_tables` (partial; Estimate, Predict)

</details>

**Shown on the canvas or as a refusal**, not an objective (10):

<details><summary>10 previews, views and refusals</summary>

- Your methods section, written from your answers `result:methods-text` (engine only)
- Manuscript rail: a slim left rail with the sentence count, opened over the card column `other:manuscript-rail` (partial)
- The whole manuscript, full width: Methods, Results, Discussion, Supplement `other:manuscript-full-width` (partial)
- A sentence shaped by a noticing ends with a quiet 'noticed' link back to First look `other:noticed-link` (missing)
- Analyses that ran but the export does not report yet `result:unexported-analyses` (partial; Estimate, Predict)
- Figure: what each column became on its way into the model `flowchart:lineage` (partial; Estimate, Predict)
- Save any view as a journal figure, captioned with the choice and rows that made it, or add it to the supplement `export:figure-save` (on screen)
- Exporting counts every model's score as seen `preview:export-counts-scores-seen` (engine only; Predict)
- The manuscript is checked: every number traces to the record, every reference resolves, every DOI verifies `export:manuscript-gate` (new scope)
- The seventh segment: what 'done' means for Write-up `other:write-up-progress` (missing)

</details>

## How it varies by goal and by domain

**By goal.** Counts of objectives (Decide, Confirm, For the record) per stage that apply to every goal, or to one goal (an item can serve two goals):

| Stage | Every goal | Estimate only or also | Predict only or also | Describe | Lens-specific |
|---|---|---|---|---|---|
| 1 · Your data | 59 | 7 | 6 | 0 | 27 |
| 2 · Your question | 34 | 10 | 12 | 0 | 7 |
| 3 · First look | 6 | 3 | 3 | 0 | 0 |
| 4 · Who's in | 72 | 39 | 24 | 9 | 62 |
| 5 · Models | 84 | 115 | 103 | 11 | 134 |
| 6 · Results | 5 | 45 | 51 | 9 | 17 |
| 7 · Write-up | 26 | 16 | 12 | 4 | 7 |

**Describe** today:
- **Who's in:** the survey population question, which the engine asks only under inference (`interview.py:_survey_gate`); exclusions become domains; who is kept when values are blank; grouping, for the variance (`new:describe_whos_in`, `q:missing`, `q:clusters`).
- **Models:** usual intake, weighted means and prevalence by group, trends across stacked cycles, patterns, clustering and agreement, then the open-noticings gate and Fit.
- **Results:** Table 1 and those exhibits.
- **Write-up:** the STROBE-nut checklist with its analytic items marked not applicable, STROBE-nut's order for the methods, and the bundle (`exhibit:strobe-nut-checklist`, `default:methods-guideline-order`, `export:bundle`).

There is no seal. A Describe track locks at its first estimate, behind its own open-noticings gate, and a later change runs as a labeled secondary (Settled here). Every Describe item is new scope or gated to inference, because `decisions.py:Purpose` has only two values. About 159 two-valued purpose branches would route a third value as prediction (`contracts.py:ContractOption.for_purpose`, `custom_sound.py`, `methods/missing.py` fall back to it).

**Estimate** carries the deepest Models stage:
- the cards for what you study, the adjustment set, the time-varying lane, energy, forms, modifiers and the causal lane;
- the open-noticings gate and Fit, which locks the plan;
- Table 2 and its companions.

Under Estimate the split question changes no reported number, so it becomes For the record (`seal.py:INFERENCE_SPLIT_REASON`).

**Predict** adds:
- **Your question:** intended use and the moment of use;
- **Who's in:** the seal;
- **Models:** validation, selection, the in-fold levers, recipes and tuning;
- **Results:** the comparison, one opening of the held-out rows, calibration, the decision curve and explanations.

**Several goals** run as tracks:
- **Shared stages:** Your data, Your question, First look and Who's in run once, and the shared First look follows the strictest goal's rule.
- **Own stages:** each track has its own Models and Results.
- **Order, locks and rows:** Estimate, then Describe, then Predict, each with its own lock; a Predict track after a track that read its outcome validates by resampling (Settled here).

A shared Who's in cannot hold one fill for every goal: multiple imputation is refused under prediction, and a single fill is blocked under inference (`decisions.py:_missing_fits_the_purpose`). Question 5 recommends deciding who is kept once, in Who's in, and the fill in each track's Models.

**By domain.** Counts of lens-specific objectives per stage:

| Stage | dietary | clinical | metabolomics | genomics | survey | something else |
|---|---|---|---|---|---|---|
| 1 · Your data | 11 | 8 | 10 | 8 | 5 | 1 |
| 2 · Your question | 0 | 3 | 2 | 3 | 2 | 0 |
| 3 · First look | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 · Who's in | 19 | 24 | 12 | 12 | 10 | 0 |
| 5 · Models | 42 | 28 | 23 | 23 | 31 | 0 |
| 6 · Results | 9 | 4 | 5 | 4 | 8 | 0 |
| 7 · Write-up | 6 | 6 | 3 | 2 | 5 | 0 |

- **Dietary:**
  - Your data: the energy unit and days (Atwater), body-measure units and sex coding, NHANES joins, stacking cycles, and the survey design columns.
  - Who's in: the energy screens (Willett, Goldberg).
  - Models: the energy model, substitution, measurement error (regression calibration), usual intake and patterns.
  - Most cited noticings: day-to-day variance, implausible reporters, energy carrying the nutrient.
- **Clinical:**
  - plausibility repairs, mixed units and censored lab values;
  - time to event, competing death and landmarks;
  - Cox, mixed models and GEE;
  - Riley's sample size;
  - noticings on treated values, prevalent cases and time zero.
- **Metabolomics:**
  - orientation and the feature table;
  - reference rows and QC drift correction, before the seal;
  - limit-of-detection handling;
  - the normalization asked in Models, then log and scaling inside the folds;
  - batch and feature-wise FDR;
  - the samples-and-features flow.
- **Genomics:**
  - orientation and damaged gene identifiers;
  - the scale of the counts;
  - screening when there are more features than samples;
  - batch as a covariate, or ComBat inside the folds;
  - FDR.
- **Something else, or not sure:** no pack runs. The shared items and the 17 family checks (`sentinel:S1` to `sentinel:E3`) are what fire; `record:orientation-not-asked` is the one item that names this lens.
- **Survey instruments:**
  - sentinel codes and skip patterns;
  - reverse keying;
  - scales and their reliability (ω, with α labeled customary);
  - ordinal outcomes;
  - the survey design and its weights.

## What the tapestry shows

How many items per stage draw each canvas mode of `calm/FOUNDATION.md` §5. An item can draw two. "No view" counts records, refusals and sentences with nothing to draw.

| Stage | Focus | Strip | Flow | Routing | Angles | No view |
|---|---|---|---|---|---|---|
| 1 · Your data | 20 | 17 | 13 | 18 | 4 | 15 |
| 2 · Your question | 32 | 5 | 10 | 9 | 10 | 18 |
| 3 · First look | 8 | 8 | 1 | 0 | 2 | 21 |
| 4 · Who's in | 38 | 19 | 79 | 12 | 22 | 20 |
| 5 · Models | 70 | 36 | 26 | 43 | 74 | 12 |
| 6 · Results | 13 | 17 | 6 | 2 | 21 | 38 |
| 7 · Write-up | 2 | 14 | 2 | 4 | 2 | 34 |

- **Your data.** At rest, the canvas is a Strip of every column in gray (FOUNDATION §5, rule 8).
  - A join or stacking draws a Flow of rows.
  - The lens and the orientation draw a Routing of columns.
  - The readings draw a Strip, ordered by consequence.
  - A repair draws a Focus on the column, before and after.
- **Your question.** A Focus on the outcome: its distribution or its levels, the event lit, and the follow-up against the horizon. The goal should draw a Routing of what each goal brings; today it draws only a note (`fact_previews.py:purpose_views`).
- **First look.**
  - The overview is an index (a Strip).
  - An item is a focal view, a context view and "More angles".
  - The outcome door is a Focus, and a relationship view per column.
- **Who's in.** The Flow dominates: the participant flow at rest, with the changed step lit. Angles compare row-wise and grouped scores. The participant flowchart closes the stage.
- **Models.** Angles and Focus dominate: trade-offs between methods, and relationships and distributions on your data. The adjustment set and each model's lineage draw a Routing. The analysis flowchart joins the Flow of rows to the Routing of columns, and Fit sits on it.
- **Results.** Exhibits are tables, forests and curves, and most of them fall outside the closed view vocabulary of `consequences.py` (row flow, lineage, table focus, distribution, relationship). That is why "No view" is high. New view kinds are a recorded design decision (Gaps).
- **Write-up.** The rail and the full-width document. There is no view kind for a page.

## Status counts

| Stage | Items | On screen | Engine only | Partial | Missing | New scope | Decide | Confirm | For the record | Shown |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 · Your data | 77 | 25 | 9 | 34 | 6 | 3 | 46 | 9 | 11 | 11 |
| 2 · Your question | 66 | 18 | 10 | 21 | 12 | 5 | 33 | 5 | 9 | 19 |
| 3 · First look | 35 | 3 | 7 | 16 | 8 | 1 | 4 | 0 | 5 | 26 |
| 4 · Who's in | 152 | 35 | 7 | 69 | 35 | 6 | 76 | 18 | 32 | 26 |
| 5 · Models | 231 | 2 | 45 | 100 | 74 | 10 | 132 | 63 | 20 | 16 |
| 6 · Results | 93 | 3 | 49 | 17 | 16 | 8 | 59 | 5 | 19 | 10 |
| 7 · Write-up | 54 | 1 | 21 | 11 | 16 | 5 | 10 | 2 | 32 | 10 |
| **All** | **708** | **87** | **148** | **268** | **167** | **38** | **360** | **102** | **128** | **118** |

Over all items: 12% on screen, 21% engine only, 38% partial, 24% missing, 5% new scope.

**Noticings and engine items differ:**
- **Noticings** (334 items carry a catalog thread): 10 on screen, 32 engine only, 175 partial, 111 missing.
- **Engine questions, decisions, defaults and exhibits** (374 items): 77 on screen, 116 engine only, 93 partial, 56 missing, 32 new scope.

## After training: every result is an exhibit

**The pivot.** Before Fit, the canvas shows the consequence of a choice. After Fit, it shows the evidence behind a result.

**Under Estimate and Describe,** nothing is served before you press Fit; pressing it locks the track's plan (the system's `lock_plan` record; `server/service.py:_lock_when_shown` does this today on the first estimate served) and shows the first estimates. Results opens with one line: the lock time and the plan's SHA-256 (`plan_lock.py:digest`).

**Under Predict,** Results opens on the cross-validated comparison with its declared basis (`models/selection.py:declared_result`). The held-out rows open once, after the final model, the threshold and the recalibration are fixed (MODELING_SEQUENCE §1 row 12a; `seal.py:_open_seal_once_on_a_fresh_fit`).

**An exhibit** is a figure or table with its caption, the finding, and its interpretation drawn on the tapestry. For each one you:
- choose a drafted wording, or write your own;
- place it in Results, the Discussion or the Supplement, or leave it out. Left out, it is still listed and stays in the record.

**The floor** (HANDOFF, "The orchestrator's methods floor"):
- **Primary results.** Under Estimate, the locked primary always stays in Results. Under Predict, so does the held-out score, or the declared cross-validated result.
- **Exposure families.** A declared family is never trimmed by p-value (`refusal:report_by_significance`).
- **Explanations** are never worded as effects (`models/explain.py:DESCRIBES`; `refusal:explain_as_effect`).
- **Causal wording** is offered only for trials (`V2_DEFINITION_OF_DONE.md`, 2026-10-07).

**Claim strengths the drafts use:**
- descriptive;
- association;
- estimated effect, with its assumptions named. This is offered only under Estimate, with the estimand declared and the unmeasured-confounding exhibit beside it;
- causal, for trials only;
- prediction performance;
- "describes the model", for explanations;
- labeled secondary, or "suggested by data inspection";
- inconclusive null (`shared-null-is-inconclusive`).

The engine writes none of these drafts today. `estimand.py:caption` says "The total effect of …" for every inference estimate, whatever the design. That is a claim-strength rule to build (Gaps).

**Pre-included:** 23 of 48 exhibits.

**Never overwrite silently:**
- **Under Estimate,** a change after the lock runs as a secondary analysis beside the locked primary. Today the engine instead marks the change and recomputes every estimate on the new answers (`decisions.py:disclose`). Only post-hoc modifiers (`methods/interaction.py:POST_HOC`) and regression calibration already behave as ruled.
- **Under Describe,** the same: the first result stays, and a change runs as a labeled secondary (Settled here).
- **Under Predict,** the earlier version is kept and labeled "revised after first results", in the comparison only. That is RECIPES RT-6 and RT-7, not built. For a change to a shared step, see question 6.
- **After the held-out rows open,** drawing them again is a reseal, and the first opening stays the reported result (`seal.py:reported_result`).

### Decisions after the fit

- A check failed: show the effect before and after the midpoint, refit without the influential rows, or keep the estimate labeled? `decision:respond_diagnostic` (Decide; engine only)
- You changed the plan after seeing estimates: this runs as a secondary analysis beside the primary `decision:post_lock_secondary` (Decide; partial)
- Shrink the coefficients by the calibration slope before the model is used? `decision:set_updating` (Decide; engine only)
- Predictors outnumber rows: run the nested cross-validation interval? (about N minutes) `decision:nested_cv_offer` (Decide; partial)
- Before you open the held-out rows: these noticings still change the honest score. Decide or dismiss each `gate:open-noticings-before-seal` (Decide; missing)
- Choose your final model on cross-validation, then open the held-out rows once `q:open_seal` (Decide; on screen)
- Draw the held-out rows again, recorded as after their scores were seen `decision:reseal` (Decide; engine only)
- Describe how each fitted model uses its inputs? `decision:set_explain` (Decide; engine only)
- How should this finding read? Pick a drafted wording or write your own `decision:exhibit_wording` (Decide; new scope)
- Where does it go: Results, Discussion, Supplement, or left out? `decision:exhibit_placement` (Decide; new scope)
- Threshold range 5% to 50%, and a threshold chosen inside each fold `default:decision_threshold` (Confirm; engine only)
- Curves by accumulated local effects (partial dependence ranked lower) `default:explain_curve_method` (Confirm; engine only)
- Calibration read at the median follow-up time (no horizon was declared) `default:horizon_calibration` (Confirm; engine only)
- Robustness value first for a linear outcome; the E-value on the spread of the population your result describes `default:unmeasured_order_and_sd` (Confirm; engine only)
- Confirm what was set for you in Results (only defaults whose alternative would change a number) `other:confirm-sweep:results` (Confirm; missing)

### The exhibits

#### Estimate an effect

| Exhibit | Pre-included | Default place | May move to | Status | Why |
|---|---|---|---|---|---|
| The effect of [what you study] on [the outcome] in each model you declared: unadjusted, Model 1, the primary, Model 3 `exhibit:table2` | yes | Results | fixed | engine only | The locked primary under inference (methods floor; STROBE 16a: crude and adjusted). |
| The risk difference and risk ratio, averaged over your participants `exhibit:marginal_contrasts` | yes | Results | Supplement | engine only | Pre-included when the declared effect measure is marginal (ranked first for a common binary outcome). |
| The effect of always versus never [what you study] over follow-up `exhibit:time_varying_estimate` | yes | Results | fixed | engine only | The primary when the exposure changes over time. |
| The effect estimated by [double ML / TMLE], with its assumptions and overlap `exhibit:causal_estimate` | yes | Results | Supplement | engine only | Declared in the plan with the model families. |
| The primary beside the model further adjusted for [BMI] `exhibit:secondary_further_adjusted` | yes | Results | Supplement | engine only | Declared by the adjustment answers (unknown timing). |
| Your estimate corrected for day-to-day variation in the recalls, beside the uncorrected one `exhibit:regression_calibration` | yes | Results | Supplement | engine only | A declared secondary analysis. |
| How reliably each scale measures, and its coefficient corrected for that `exhibit:scales_reliability` | no | Supplement | Results, Left out (still listed, kept in the record) | engine only | Reliability belongs in Methods or Results; the correction is a declared secondary. |
| Does the effect differ across [modifier]? On both scales, against one reference `exhibit:effect_modification` | yes | Results | Supplement | engine only | Pre-included when declared before the lock; declared after, it is labeled and goes to the Supplement by default. |
| Every factor in the family you tested (every nutrient, every metabolite), each with its estimate and q-value `exhibit:exposure_family` | yes | Results | Supplement | partial | Every member stays in the record: a summary in Results, the full table in the Supplement. |
| The same model on other rows: every row, and each screen you declared `exhibit:declared_sensitivity` | yes | Supplement | Results | engine only | Declared in the plan: a Supplement table plus one Results sentence (STROBE 12e, 17). |
| How strong would something unmeasured, affecting both, have to be to explain this away? `exhibit:unmeasured_confounding` | no | Supplement | Results, Discussion, Left out (still listed, kept in the record) | engine only | Offered for every inference estimate; one sentence in Results or Discussion, the benchmark table in the Supplement. Never a pass or fail. |
| Which of your decisions mattered? The estimate across every alternative you declared `exhibit:which_decisions_mattered` | no | Supplement | Discussion, Left out (still listed, kept in the record) | partial | A sensitivity view only, never a way to choose. |
| Backward elimination as a labeled sensitivity analysis `result:selection_sensitivity_inference` | no | Supplement | Left out (still listed, kept in the record) | engine only | A labeled sensitivity analysis. |
| The other coefficients: adjustment terms, not effect estimates `exhibit:table2_appendix` | no | Supplement | Left out (still listed, kept in the record) | engine only | Adjustment terms are never shown as effects (Westreich and Greenland 2013). |
| Did the primary model's assumptions hold? `exhibit:diagnostics` | no | Supplement | Left out (still listed, kept in the record) | engine only | Checks that pass are clean checks (Supplement, ruling of 2026-10-06). |
| Do the residuals of the primary model follow the shape it assumes? (residual Q-Q) `exhibit:residual-qq` | no | Supplement | Left out (still listed, kept in the record) | missing | A model check; checks that pass are clean checks (Supplement, ruling of 2026-10-06). |

Drafted wordings (bracketed words are filled from the record; "your own" is always offered):

- **The effect of [what you study] on [the outcome] in each model you declared: unadjusted, Model 1, the primary, Model 3**
  - *association*: "Each [increment] higher [exposure] was associated with a [estimate] [unit] difference in [outcome] (95% CI [lower] to [upper]), adjusted for [Model 2 covariates]."
  - *estimated effect, assumptions named*: "If there is no unmeasured confounding and the declared model form holds, [increment] more [exposure] would change [outcome] by [estimate] (95% CI [lower] to [upper])."
  - *inconclusive null*: "The data did not provide evidence of an association between [exposure] and [outcome] ([estimate]; 95% CI [lower] to [upper]); effects larger than [bound] are unlikely."
- **The risk difference and risk ratio, averaged over your participants**
  - *association*: "Standardized to these participants, the risk of [outcome] was [r1] at [exposure level] and [r0] at [reference] (risk difference [RD], 95% CI; risk ratio [RR])."
  - *estimated effect, assumptions named*: "Under the stated assumptions, setting everyone to [exposure level] rather than [reference] would change the risk of [outcome] by [RD] (95% CI)."
- **The effect of always versus never [what you study] over follow-up**
  - *estimated effect, assumptions named*: "Had everyone [always] rather than [never] [exposure], the [t]-year risk of [outcome] would have been [r1] versus [r0] (difference [d], 95% CI), under the stated assumptions (g-formula)."
  - *association*: "Sustained [exposure] was associated with a [d] difference in the [t]-year risk of [outcome] (marginal structural model, weights truncated at [q])."
- **The effect estimated by [double ML / TMLE], with its assumptions and overlap**
  - *estimated effect, assumptions named*: "With the adjustment learned flexibly ([method], [learner]), the estimated effect was [estimate] (95% CI), assuming no unmeasured confounding and adequate overlap (Figure S[n])."
  - *comparison*: "The flexible estimate ([a]) was close to (differed from) the primary estimate ([b])."
- **The primary beside the model further adjusted for [BMI]**
  - *labeled secondary*: "Further adjusted for [column], whose timing relative to [exposure] is unknown, the estimate was [estimate] (95% CI); this is not a total effect."
- **Your estimate corrected for day-to-day variation in the recalls, beside the uncorrected one**
  - *labeled secondary*: "Corrected for day-to-day variation in the recalls (regression calibration, attenuation factor [λ]), the estimate was [b] (95% CI), against [a] uncorrected. The test of association is the uncorrected one."
- **How reliably each scale measures, and its coefficient corrected for that**
  - *descriptive*: "[Scale] had McDonald's omega of [ω] (Cronbach's alpha [α], customary)."
  - *labeled secondary*: "Corrected for the scale's unreliability, the coefficient was [b] (uncorrected [a])."
- **Does the effect differ across [modifier]? On both scales, against one reference**
  - *declared before the lock*: "The effect of [exposure] was [e1] among [level 1] and [e2] among [level 2] (ratio of ratios [r]; relative excess risk due to interaction [x])."
  - *suggested by data inspection*: "In an analysis suggested by data inspection, the effect of [exposure] appeared to differ by [modifier] ([e1] versus [e2]); this was not planned."
- **Every factor in the family you tested (every nutrient, every metabolite), each with its estimate and q-value**
  - *association*: "Of [k] [exposures] tested, [m] were associated with [outcome] at a false discovery rate of 5% (Table S[n])."
  - *inconclusive null*: "None of the [k] [exposures] was associated with [outcome] at a false discovery rate of 5%."
- **The same model on other rows: every row, and each screen you declared**
  - *robust*: "Estimates were similar under [screen] and with every row kept ([lowest] to [highest]; Table S[n])."
  - *sensitive*: "Applying [screen] moved the estimate from [a] to [b]; the primary rule is [rule]."
- **How strong would something unmeasured, affecting both, have to be to explain this away?**
  - *robustness value*: "An unmeasured confounder would have to explain [RV]% of the remaining variance of both [exposure] and [outcome] to move the estimate to zero; [benchmark], the strongest measured covariate, explains [x]%."
  - *E-value*: "An unmeasured confounder associated with both [exposure] and [outcome] by a risk ratio of [E] each could explain away the estimate; weaker confounding could not."
- **Which of your decisions mattered? The estimate across every alternative you declared**
  - *specification range*: "Across [n] declared alternatives, the estimate ranged from [a] to [b]; the choice of [decision] moved it most."
- **Backward elimination as a labeled sensitivity analysis**
  - *labeled sensitivity*: "In a sensitivity analysis, backward elimination kept [k] of [p] covariates, and the estimate for [exposure] was [estimate]."
- **The other coefficients: adjustment terms, not effect estimates**
  - *fixed*: "Coefficients for the adjustment terms are listed in Table S[n]; they are not effect estimates."
- **Did the primary model's assumptions hold?**
  - *fixed*: "Model checks (proportional hazards, influence) are shown in Figure S[n]."
- **Do the residuals of the primary model follow the shape it assumes? (residual Q-Q)**
  - *fixed*: "Residuals of the primary model are shown against a normal distribution in Figure S[n]."

#### Predict

| Exhibit | Pre-included | Default place | May move to | Status | Why |
|---|---|---|---|---|---|
| How well each model predicts new people, and the one result you report `exhibit:performance_table` | yes | Results | fixed | partial | The declared result row is always reported (TRIPOD+AI 23). |
| A no-predictor baseline and a regression with curves are always fitted beside the chosen models `exhibit:spline_benchmark` | yes | Results | fixed | engine only | A row in the performance table. |
| The held-out score of the model you declared final `exhibit:held_out_result` | yes | Results | fixed | on screen | The held-out score at the first opening is always the reported result. |
| Are the predicted risks right? Observed against predicted `exhibit:calibration_curves` | yes | Results | Supplement | engine only | TRIPOD+AI 23a asks for calibration beside discrimination. |
| Would using the model help decisions across reasonable risk thresholds? `exhibit:decision_curve` | yes | Results | Supplement | engine only | Pre-included when decision support is the declared intended use. |
| How well it predicts within each group `exhibit:subgroup_performance` | no | Results | Supplement, Left out (still listed, kept in the record) | engine only | Named subgroups from the intended use (TRIPOD+AI 14, 23a). |
| How well it predicts in the surveyed population `exhibit:design_based_cv` | yes | Results | Supplement | engine only | Pre-included when the scores describe the surveyed population. |
| Scored on each site by models fit on the others `exhibit:internal_external_cv` | no | Supplement | Results, Left out (still listed, kept in the record) | engine only | TRIPOD+AI 12d and 23b. |
| How far apart the models are, pair by pair `exhibit:family_comparisons` | no | Supplement | Left out (still listed, kept in the record) | engine only | Pair-by-pair differences support, but do not replace, the declared result. |
| What the interpretable model costs or gains against the best flexible one `exhibit:interpretable_cost` | no | Discussion | Supplement, Left out (still listed, kept in the record) | engine only | The price of explainability, measured. |
| How often each predictor was selected across folds `exhibit:inclusion_frequencies` | no | Supplement | Left out (still listed, kept in the record) | engine only | Selected variables are never called "the predictors". |
| Earlier versions you changed after seeing scores, kept in the comparison `exhibit:versions_comparison` | no | Supplement | Left out (still listed, kept in the record) | missing | Shown in the comparison only, labeled "revised after first results". |
| The settings each tuned model used, and how much they varied across folds `exhibit:settings_table` | no | Supplement | Left out (still listed, kept in the record) | missing | Tuned values and how much they varied across folds. |
| The fitted linear model's coefficients `exhibit:fit_coefficients` | no | Supplement | Left out (still listed, kept in the record) | on screen | Under prediction, the fitted equation belongs to the architecture lane. |

Drafted wordings (bracketed words are filled from the record; "your own" is always offered):

- **How well each model predicts new people, and the one result you report**
  - *prediction performance*: "The [family] model, chosen by cross-validation among [k] models, had a [proper score] of [x] (corrected for the choice; AUC [y], customary headline)."
  - *below baseline*: "No model predicted [outcome] better than the no-predictor baseline ([score] against [baseline])."
- **A no-predictor baseline and a regression with curves are always fitted beside the chosen models**
  - *prediction performance*: "A regression with curves, fitted as a benchmark, scored [x]."
- **The held-out score of the model you declared final**
  - *prediction performance*: "Scored once on [n] held-out participants, the [family] model had a [metric] of [x] (95% CI [l] to [u])."
- **Are the predicted risks right? Observed against predicted**
  - *prediction performance*: "Predicted and observed risks agreed (calibration slope [s], intercept [i])."
  - *miscalibrated*: "Predicted risks were too extreme (calibration slope [s]); shrinking the coefficients by [s] corrected this."
- **Would using the model help decisions across reasonable risk thresholds?**
  - *prediction performance*: "Across risk thresholds from [a]% to [b]%, using the model gave a higher net benefit than treating everyone or no one."
  - *limited benefit*: "The model added net benefit only between thresholds of [a]% and [b]%."
- **How well it predicts within each group**
  - *prediction performance*: "Discrimination was [x] among [group 1] and [y] among [group 2]; calibration slopes were [s1] and [s2]."
- **How well it predicts in the surveyed population**
  - *prediction performance*: "Weighted to the surveyed population, the model's [metric] was [x] (design-based cross-validation)."
- **Scored on each site by models fit on the others**
  - *prediction performance*: "Fit on the other sites and scored on each, performance ranged from [a] to [b] (pooled [c])."
- **How far apart the models are, pair by pair**
  - *prediction performance*: "Paired differences between models, with corrected-t intervals, are in Table S[n]."
- **What the interpretable model costs or gains against the best flexible one**
  - *comparison*: "The interpretable [family] scored [d] below (above) the best flexible model, on a corrected comparison."
- **How often each predictor was selected across folds**
  - *stability*: "[k] of the [m] selected predictors were chosen in fewer than 40% of folds (Table S[n])."
- **Earlier versions you changed after seeing scores, kept in the comparison**
  - *labeled*: "The settings of [family] were revised after first results; the earlier version is kept in the comparison."
- **The settings each tuned model used, and how much they varied across folds**
  - *fixed*: "The settings each tuned model used, and their variation across folds, are in Table S[n]."
- **The fitted linear model's coefficients**
  - *fixed*: "The fitted equation is given in Table S[n]."

#### Describe, the new methods and trials

| Exhibit | Pre-included | Default place | May move to | Status | Why |
|---|---|---|---|---|---|
| Who your participants are (Table 1) `exhibit:table1` | yes | Results | fixed | missing | Every nutrition paper has one; Describe lists it as a deliverable. |
| Usual intake of [nutrient] in the population, and the share below the requirement `exhibit:usual_intake_distribution` | yes | Results | fixed | engine only | The Describe track's primary. |
| Survey-weighted means and prevalence by group `exhibit:weighted_prevalence_by_group` | yes | Results | Supplement | new scope | A Describe track's result. |
| How it changed across survey cycles `exhibit:trends_across_cycles` | no | Results | Supplement, Left out (still listed, kept in the record) | new scope | Needs stacked cycles. |
| The dietary patterns found, described by their loadings `exhibit:dietary_patterns` | no | Results | Supplement, Left out (still listed, kept in the record) | new scope | A declared pattern method. |
| Subgroups of similar people `exhibit:clusters` | no | Results | Supplement, Left out (still listed, kept in the record) | new scope | A declared clustering method; k by a declared rule. |
| How well two methods, or two models, agree `exhibit:bland_altman` | no | Results | Supplement, Left out (still listed, kept in the record) | new scope | Agreement between two methods or two models. |
| The effect of the assigned treatment (intention to treat), with per-protocol beside it `exhibit:trial_results` | yes | Results | fixed | new scope | The only design whose wording may state a causal effect. |

Drafted wordings (bracketed words are filled from the record; "your own" is always offered):

- **Who your participants are (Table 1)**
  - *descriptive*: "Table 1 describes the [n] participants analyzed[, by level of exposure]."
- **Usual intake of [nutrient] in the population, and the share below the requirement**
  - *descriptive*: "The median usual intake of [nutrient] was [m] [unit] (5th to 95th percentile [a] to [b])."
  - *prevalence (only under the EAR cut-point conditions)*: "An estimated [s]% (95% CI) of [population] had usual intakes below the EAR."
- **Survey-weighted means and prevalence by group**
  - *descriptive*: "An estimated [p]% (95% CI) of [population] [had the characteristic]: [p1]% among [group 1] and [p2]% among [group 2]. Estimates resting on too few effective participants are grayed."
- **How it changed across survey cycles**
  - *descriptive*: "Mean [measure] changed from [a] in [first cycle] to [b] in [last cycle]."
- **The dietary patterns found, described by their loadings**
  - *descriptive*: "[k] patterns, chosen by [rule], explained [v]% of the variance in food intake; the first loaded mainly on [foods]."
- **Subgroups of similar people**
  - *descriptive*: "[k] subgroups were found ([rule]); the largest ([n]%) was characterized by [features]."
- **How well two methods, or two models, agree**
  - *descriptive*: "On average [method A] read [bias] [unit] higher than [method B]; 95% of differences fell between [l] and [u]."
- **The effect of the assigned treatment (intention to treat), with per-protocol beside it**
  - *causal (trials only)*: "Assignment to [arm] changed [outcome] by [d] (95% CI) compared with [control] (intention to treat); per protocol, [d2]."

#### Explanations (any goal; they describe the model, never an effect)

| Exhibit | Pre-included | Default place | May move to | Status | Why |
|---|---|---|---|---|---|
| Which inputs each model leans on, and how stable that is across refits `exhibit:shap_importance` | no | Results | Supplement, Left out (still listed, kept in the record) | engine only | Results under prediction; Supplement under inference, covariates labeled as adjustment terms. |
| Which inputs each model leaned on, side by side (one table across models) `exhibit:cross-model-importance` | no | Supplement | Results, Left out (still listed, kept in the record) | partial | An explanation describes each model (models/explain.py:DESCRIBES); never worded as an effect. |
| Each top input's curve in every model, on shared axes `exhibit:inductive_bias_curves` | no | Results | Supplement, Left out (still listed, kept in the record) | engine only | Results under prediction, Supplement under inference. |
| Each person's prediction split into its inputs `exhibit:shap_beeswarm_observations` | no | Supplement | Left out (still listed, kept in the record) | engine only | Per-person attributions. |
| Pairs of inputs the model combines `exhibit:interactions_h` | no | Supplement | Left out (still listed, kept in the record) | engine only | Pairs the model combines. |
| What each model is: its equation, its trees, its shrinkage path `exhibit:architecture_lane` | no | Supplement | Left out (still listed, kept in the record) | engine only | What each model is. |

Drafted wordings (bracketed words are filled from the record; "your own" is always offered):

- **Which inputs each model leans on, and how stable that is across refits**
  - *describes the model*: "The model relied most on [x], [y] and [z]. These describe its predictions, not what changing them would do."
- **Which inputs each model leaned on, side by side (one table across models)**
  - *describes the model*: "Across the [k] models, [input] ranked highest in mean absolute SHAP value in [m] of them (Table S[n]); these rankings describe the models, not effects."
- **Each top input's curve in every model, on shared axes**
  - *describes the model*: "Each model's learned curve for [x] is drawn on shared axes (Figure [n]); the models agree where the data are dense."
- **Each person's prediction split into its inputs**
  - *describes the model*: "Per-person contributions to each prediction are shown in Figure S[n]."
- **Pairs of inputs the model combines**
  - *describes the model*: "The model combined [x] and [y] most strongly (H² [h])."
- **What each model is: its equation, its trees, its shrinkage path**
  - *fixed*: "The fitted equation, tree structure or shrinkage path of each model is in Supplement S[n]."

#### Made before training, placed the same way

| Exhibit | Pre-included | Default place | May move to | Status | Why |
|---|---|---|---|---|---|
| Participant flow, savable as a figure `exhibit:participant_flowchart` | yes | Results | Supplement | partial | STROBE item 13c recommends a flow diagram; under Estimate its counts are final at the analysis flowchart (disagreement 7). |
| Samples and features: how many samples and features remain at each step `exhibit:samples_and_features_flow` | yes | Results | Supplement | partial | Omics readers expect the samples and the features kept at each step; the in-fold feature steps are stated as varying by fold. |
| CONSORT flow: enrolled, allocated, followed up, analyzed, by arm `flowchart:consort` | yes | Results | fixed | new scope | CONSORT requires the flow diagram for a trial (DoD 2026-10-07); placement fixed. |
| The analysis flowchart: what will be fit, on which rows, in what order (savable as a figure) `flowchart:analysis` | yes | Supplement | Results, Left out (still listed, kept in the record) | missing | It documents what was fitted, on which rows and in what order; the plan export carries the same content. |

Drafted wordings (bracketed words are filled from the record; "your own" is always offered):

- **Participant flow, savable as a figure**
  - *descriptive*: "Of [N0] participants in the file, [N1] were excluded ([reasons with counts]) and [N] were analyzed (Figure [n])."
- **Samples and features: how many samples and features remain at each step**
  - *descriptive*: "[S] of [S0] samples and [F] of [F0] features passed quality control (Figure [n]); feature filters fitted in each training fold are described in the Methods."
- **CONSORT flow: enrolled, allocated, followed up, analyzed, by arm**
  - *descriptive*: "[N] participants were randomized ([n1] to [arm 1], [n2] to [arm 2]); [m1] and [m2] were analyzed as randomized (Figure [n])."
- **The analysis flowchart: what will be fit, on which rows, in what order (savable as a figure)**
  - *descriptive*: "The analysis plan, fixed before any estimate was seen, is shown in Figure S[n]."
  - *descriptive (Predict)*: "The analysis, as declared before the held-out rows were opened, is shown in Figure S[n]."

Figure and table numbers follow placement (`default:figure-and-table-numbering`), so the flowcharts are placed like any exhibit (the last table above).

### Noticings born after the fit

These need a fit, so First look never shows them. They label an exhibit, or add a disclosure or a sensitivity analysis. A new analysis they ask for after the lock is a secondary.

- A check failed: show the effect before and after the midpoint, refit without the influential rows, or keep the estimate labeled? `decision:respond_diagnostic` (engine only; Estimate)
- Shrink the coefficients by the calibration slope before the model is used? `decision:set_updating` (engine only; Predict)
- How often each predictor was selected across folds `exhibit:inclusion_frequencies` (engine only; Predict)
- This null can't say there is no effect: its interval still includes the smallest effect that matters `thread:shared-null-is-inconclusive` (missing; Estimate)
- Your model says HDL raises risk; that reversed once TC/HDL entered `thread:clin-expected-directions` (partial; Predict, Estimate; clinical)
- Your p-values pile toward 1, so the model is wrong somewhere `thread:genomics-pvalue-distribution` (missing; Estimate; genomics, metabolomics)
- Your top feature is metformin: the model found the prescription `thread:metab-treatment-marker` (missing; Predict, Estimate; metabolomics)
- There is no second measure here: a null can't rule out a slope several times larger `thread:diet-borrowed-validity-coefficients` (partial; Estimate; dietary)
- Only a fraction of features have names, so any pathway reading runs over what the assay could name `thread:metab-enrichment-background` (missing; Estimate, Predict; metabolomics)
- Corrected for picking the best of several models `result:bbc_cv` (partial; Predict)
- Choices made by hand after looking at the outcome, which the score does not cover `result:hand_levers` (engine only; Predict, Estimate)
- What you study and the outcome come from the same questionnaire at one sitting `thread:survey-common-method` (missing; Estimate; survey)
- [family] predicts no better than the mean, so there is nothing to explain `thread:shared-model-no-better-than-baseline` (engine only; Predict)
- Two models score the same but rank the predictors differently `thread:shared-rashomon-disagreement` (partial; Predict)
- The average curve is flat, but it rises steeply in one group `thread:shared-learned-interaction` (partial; Predict)
- Only 3 of your 12 panel features appear in more than half the refits `thread:panel-instability` (partial; Predict; metabolomics, genomics)
- The score sits within what shuffled labels reach `thread:noise-ceiling` (missing; Predict)
- These importances describe the oversampled sample, not the country `thread:survey-explanations-describe-the-sample` (missing; Predict, Estimate; survey, dietary, clinical)
- This cell rests on too few effective participants to publish `thread:survey-estimate-reliability` (partial; Describe, Estimate; survey, dietary)
- A top-feature list read by pathway needs, as its background, the features that could have been selected `thread:genomics-enrichment-background` (missing; Estimate, Predict; genomics)

## Where the engine order and the stage order disagree

Each item gives the disagreement, its evidence, and the fix. The fixes are folded into `SIZING.md` (package P0.6 and the slices).

1. **The roles answer is refused in Your data.**
   - *Evidence:* `roles` is the thirteenth key of `interview.py:QUESTION_KEYS`, after the outcome, the goal and the row block, and `sequence.py:_answers_in_order` refuses it with `not_yet`. The roles stage also reads the target, purpose, grain, follow-up, outcome scale and task (`stages/__init__.py`, `Stage("roles", …)`).
   - *Fix (engine):* record `set_roles` as a completion once every predictor's role is settled through `confirm_role` and `confirm_readings`. Neither is a Router slot, so neither is refused for order. A proposal that changes after Your question or Who's in returns as "changed since you confirmed", with the reason.
   - *Fix (interface):* Your data shows the column ledger, and no roles question (question 2).
2. **Readings are asked where they are used, not at upload.**
   - *Evidence:* `ask.py:CONSUMERS` puts the ask card on combining, survey, exclusions, estimand, adjustment, energy and models.
   - *Fix:* as in question 2. The engine needs a per-column ledger endpoint: every reading kind, its state, and the consumer that needs it (Gaps).
3. **The outcome's reading depends on Who's in.**
   - *Evidence:* `target_info` reads the working table that grain, unit and combining reshape (`stages/__init__.py`), and the combining question needs the outcome (`sequence.py:_aggregation_knows_the_outcome`). A combining answer can change the outcome's kind.
   - *Fix:* when the structure stage reads repeated rows, ask "which value of the outcome counts" on the outcome card. A Who's in answer that changes the outcome's kind reopens Your question with the reason.
4. **First look comes before the seal, but the engine explores after it.**
   - *Evidence:* the explore stage requires `target` and `split`, and depends on `cohort` (`stages/__init__.py`, `Stage("explore", …)`). MODELING_SEQUENCE §1 row 1 places Explore after the seal.
   - *Fix:* the pre-seal notices stage (UNDERSTANDING_LAYER U4) computes First look's outcome-free looks from the oriented table. Explore's outcome-free findings move into it: low variance, more predictors than rows, collinear pairs, and quality by group. The outcome's views open after Who's in, as question 1 decides.
5. **The split is two decisions in one kind, placed twice.**
   - *Evidence:* `decisions.py:SetSplit` and `SplitSpec` hold both the holdout and the validation scheme. The Router asks the split after missing values, before the estimand.
   - *Fix (engine):* separate the draw (Who's in) from the validation scheme (a Models Confirm), as two kinds, or by re-recording the split with the same draw. Under Estimate the split question becomes For the record (`seal.py:INFERENCE_SPLIT_REASON`).
6. **"Confirm last" collides with the seal under Predict.**
   - *Evidence:* the draw reads the stated grain, the repeat kind and the time column, and changes after the draw are refused (`seal.py:DECISION_A`, `DRAW_READS`).
   - *Fix:* under Predict, Who's in runs its Confirm sweep just before the seal, and the seal is the stage's last Decide.
7. **Who's in depends on Models answers.**
   - *Evidence:* imputation compatible with the analysis model reads the declared forms and the energy model (`decisions.py:_imputation_fits_the_analysis`; `methods/missing.py:missing_block`). Complete cases are counted on the covariates the adjustment set keeps (`decisions.py:left_out`). The cohort reads `form_domains` and the landmark (`stages/rows.py:domain_of`, `landmark_of`).
   - *Fix:* label the participant flowchart provisional at the end of Who's in, and draw its final counts on the analysis flowchart. A Models answer that invalidates the imputation reopens Who's in with the reason.
8. **The clusters question decides a model term in Who's in.**
   - *Evidence:* under inference its options are fixed effects plus CR2, cluster only, or none (`estimand.py:grouping_card`).
   - *Fix:* Who's in asks only whether people are grouped, and by what. How the model handles the grouping moves to the Models Confirm sweep.
9. **Survey under Predict has no question.**
   - *Evidence:* `interview.py:_survey_gate` and `survey.py:not_applicable_reason` make it not applicable under prediction. Yet `set_survey` drives design-based cross-validation (`survey.py:_asked_under_inference`, ruling 13), and only Explore offers it.
   - *Fix:* ask the survey question under every goal. Under Predict it asks whose performance the scores estimate.
10. **The design has no slot.**
    - *Evidence:* there is no design kind, `decisions.py:Purpose` has two values, and `GrainAnswer` lost Classic's `DESIGN_NOT_DESCRIBED` escape hatch (`turbotab/grain.py`).
    - *Fix:* add `q:study-design` in Your question, before the goal, with observational stated by default. It routes Who's in (no exclusion after randomization; analysis sets) and Models (precision adjustment).
11. **Ten Models decisions have no Router key.**
    - *Evidence:* `set_model_sequence`, `set_sensitivity`, `set_measurement_error`, `set_scales`, `set_batch`, `set_usual_intake`, `set_intended_use`, `set_selection`, `set_levers` and `set_explain` are absent from `interview.py:QUESTION_KEYS`.
    - *Fix:* the stage registry gates each by its applicability, as the Router does, and lists it among its stage's objectives. The Router itself needs no change.
12. **Estimates are served, and the plan locks, before Fit is pressed.**
    - *Evidence:* `server/service.py:_lock_when_shown` locks on the first served artifact of `estimand.py:ESTIMATE_STAGES`, and the fit stage requires only `models`.
    - *Fix (engine):* under Estimate and Describe, `stage_result` withholds every estimate stage until the track's plan is locked, as `estimand.served_gate` withholds one whose question is open. Pressing Fit records the lock, the existing system `lock_plan`, after the open-noticings gate. Pressing Fit stays a job command and computing stays live for short fits (`RECIPES_AND_TUNING.md` §4.4; ruling 2). Under Predict, pressing Fit opens Results and locks nothing.
    - *Corrected:* the first draft held every estimate stage until a new "Fit" decision kind. That contradicted §4.4 ("nothing enters the Record"; "the hold is in the scheduler, not a stage requirement") and is withdrawn (Settled here).
13. **Substitution is always asked after the lock.**
    - *Evidence:* `interview.py:MUST_BE_FRESH["substitution"]` is `"fit"`, and its slots are in `plan_lock.py:plan_slots`.
    - *Fix:* declare the pair in Models, before Fit, with its preview (`data_previews.py:substitution_views`). Results draws the curve. The pair's preview, its refusal and its step and band now live in Models (`preview:substitution_views`, `refusal:substitution_blocked`, `default:substitution_step_and_band`).
14. **Sanctioned after-fit displays count as plan changes.**
    - *Evidence:* `explain` is in `ESTIMATE_STAGES`, and the diagnostic responses and model updating are plan slots.
    - *Fix:* exempt `set_explain`, `respond_diagnostic` and `set_updating` from the after-estimates mark, and label them as companion displays.
15. **A change after the lock overwrites the primary.**
    - *Evidence:* `decisions.py:disclose` marks the record, and every estimate stage recomputes on the new state.
    - *Fix (engine):* compute the estimate stages on the locked plan's state, and again on the changed state. The change is the labeled secondary.
16. **Intended use straddles three stages.**
    - *Evidence:*
      - MODELING_SEQUENCE §1 row 2 places it under the estimand;
      - it is offered as an Explore lever (`stages/explore.py:_use_lever`);
      - its preview needs a fit (`explore_previews.py:intended_use_views`);
      - the threshold must be fixed before the opening (row 12a).
    - *Fix:* ask intended use in Your question, as Predict's shape. Fix the threshold range and the recalibration in Results, just before the opening.
17. **Reference rows and QC drift correction sit in two stages.**
    - *Evidence:* they are repairs before the seal, in Your data (`reference_rows.py:_before_the_seal`), but their leaving is a participant-flow step (`stages/rows.py:cohort_flow`, `reference:i`).
    - *Fix:* decide them in Your data, because they must leave before the outcome's kind is read, and draw them as the first step of the participant flow.
18. **The Router's order inside Models differs from the ruled list.**
    - *Evidence:* the time-varying question comes before the energy model and the forms (`interview.py:QUESTION_KEYS`).
    - *Fix:* adopt the Router's order in the quest log. It costs nothing.
19. **The methods section is ordered by guideline, not by stage.**
    - *Evidence:* `export/methods.py:SECTION_OF`.
    - *Fix:* the stage registry maps every decision kind to its stage, so a sentence's "change" link opens the right card.
20. **The Router answers in order; the quest log lets you look ahead.**
    - *Evidence:* `sequence.py:_answers_in_order` refuses an answer to a question still waiting behind an earlier one (`refusal:not-yet`).
    - *Fix (interface):* every stage can be opened and read. A question whose earlier answers are missing shows "Waiting for: [the question]", with a link to it, and is not answerable. The one exception is "Decide now" from First look (disagreement 9's Router exception, P0.6).

### Engine stages and quest stages

The engine computes 30 stages (`stages/__init__.py:build_graph`). A stage goes stale when an answer it reads changes, and the quest stage that shows it drops back with that answer as its reason (`interview.py` module docstring: later answers "stay answered and its stages go stale"). Every stage except `target_info`, `proposals` and `seal_plan` is heavy (it runs as a job).

| Engine stage | Reads (examples) | Answered in | Shown in |
|---|---|---|---|
| `ingest` | joins | Your data | Your data (`record:ingest-facts`) |
| `oriented` | orientation, feature table | Your data | Your data |
| `profile` | the oriented table | — | Your data (For the record); First look's index |
| `findings` | lens, outcome, units, sex codings, categories, combining | Your data; goes stale on Your question and Who's in | Your data's noticings; First look's groups |
| `structure` | the oriented table, the date reading | Who's in (grain, repeats) | Who's in |
| `working` | repairs, grain, unit, combining | Your data, Who's in | every later stage reads it |
| `target_info` | outcome, kind, unit, scale | Your question | Your question |
| `roles` | roles and readings; also the outcome, goal, grain, follow-up | Your data (disagreement 1) | Your data |
| `proposals` | roles, outcome, goal, units, repeats, Model 1 | Your data, Models | the adjustment card's guesses |
| `cohort` | outcome, roles, exclusions, missing values, the adjustment set, forms, landmark | Who's in; goes stale on Models (disagreement 7) | the participant flow |
| `seal_plan` | roles, kind, event, goal, clusters | Who's in | the split question's options |
| `split` | the split | Who's in (the draw), Models (the validation scheme) | the seal |
| `explore` | outcome, split | First look | the outcome door (computed after the seal; disagreement 4) |
| `shelf` | roles, cohort, split | Models | which models fit |
| `forms` | goal, cohort | Models | the form card |
| `design` | models, roles | Models | the analysis flowchart |
| `causal_design` | what you study and its effect | Models | overlap, before any estimate |
| `usual_intake` | lens, goal | Models (Describe) | Results |
| `fit`, `substitution`, `sensitivity`, `calibration`, `secondary`, `scales`, `effects`, `causal`, `time_varying`, `modification`, `explain`, `evaluation` | the plan's slots | Models (declared) | Results, after Fit (`estimand.ESTIMATE_STAGES`) |

P0.4's stage registry maps these as well as the questions, so a drop-back can name both the answer that changed and the result that went stale.

## Gaps

### Engine

1. **The stage registry.** The engine knows only `QUESTION_KEYS`. It has no notion of the seven stages, of which objectives count toward a stage, or of why a stage reopened: later answers stay answered and go stale (`interview.py`, module docstring).
   - *Needed:* a registry that maps every Router key, every non-Router decision kind, every finding and every noticing to a stage and an objective.
   - *Also needed:* a "reopened because …" record, for example "the join added 12 columns, so the roles are read again".
2. **The Confirm sweep and the record list.**
   - *Confirm sweep:* nothing lists, per stage, the defaults set for you and whether another choice would change a number. The Router reports only its skipped steps. For Your data, `readings.read_from_data` and `GET /projects/{pid}/readings` are the starting point.
   - *Record list:* no endpoint collects the For the record lines. These are the ingest warnings (`datastore.py:DatasetInfo.warnings`), the profile's basis, the not-applicable reasons and the checked-clean records.
3. **The understanding layer.** No noticing is wired end to end (111 noticing items are missing and 175 partial). The following work packages from UNDERSTANDING_LAYER §5 are missing:
   - U1, the thread registry and contract;
   - U2, the ledger extension;
   - U3, the census;
   - U4, the pre-seal notices stage;
   - U5, card families in the Router, with the gate that reports asked rows per journey;
   - U6, context lines;
   - U7, thread sentences, supplement S1 and the IDA paragraph;
   - U8, the open-noticings gate at the lock and at the opening;
   - U9, the honest-score ladder;
   - U10, the 17 family checks (now items, `sentinel:S1` to `sentinel:E3`);
   - U12, the fixtures: one where each noticing fires and one where it stays silent;
   - U13, the coverage registry;
   - U14, the calm interface for noticings: card-family rows, the context line, the manuscript mark, the open-noticings card, the supplement view.

   Some consumers the catalogs need have no code (U11):
   - competing risks;
   - re-anchoring time zero;
   - IPCW;
   - leave-one-batch-out and leave-one-site-out validation;
   - the specification curve;
   - parallel analysis;
   - re-scoring at deployment noise.
4. **Describe, tracks and designs.** `decisions.py:Purpose` has two values, and `ProjectState` holds one outcome, goal, seal and plan. Needed:
   - a Describe goal, routed through every purpose branch, contract label, checklist and sentence (`voice.py:_PURPOSE_CLAUSE` has two purposes), with its own open-noticings gate and lock;
   - per-track state and a merge: a track id on every record; the lock, the after-estimates mark (`decisions.disclose`) and the estimates shown under prediction (`SHOWN_UNDER_PREDICTION`) scoped to the track; the rows rule for a Predict track that follows a track that read its outcome; the fill of blanks in each track's Models (question 5);
   - a design slot, with "Not available yet" exits;
   - the trial analyses;
   - the case-control restrictions and conditional logistic regression.
5. **New methods:**
   - survey-weighted means and prevalence by group (no design-based descriptive estimator exists outside the model families and usual intake);
   - trends across stacked cycles (stacking is not built);
   - Table 1;
   - dietary patterns;
   - clustering;
   - Bland–Altman.
6. **Recipes and tuning.**
   - `SetRecipe`, `SetTuning`, `DeclareVersion` and the stamps are absent from `decisions.py`.
   - The trees' native handling of blanks is dead code (RECIPES F1), and only the elastic net tunes (F2).
   - Ridge, Huber, random forest and XGBoost are absent (RECIPES §9, RT-1 to RT-14).
   - The groups kept in v2 are not built: faster search, more preprocessing options and the inference extensions. Neither is in-fold PCA for omics.
7. **Fit and the lock.** There is no Fit command. The fit stage requires only `models`, and the server locks the plan on the first estimate it serves (`server/service.py:_lock_when_shown`). Needed, as settled above: under Estimate and Describe, a serving gate that withholds every estimate stage until the track's lock, and a Fit job command that records the existing system `lock_plan` after the open-noticings gate. The 2-minute hold (RECIPES §4.4, RT-8) is not built. No new decision kind is needed.
8. **Exhibits.**
   - No decision kind records a wording or a placement, so the manuscript is not a function of the record.
   - There are no drafted wordings and no claim-strength rule.
   - Nothing enforces the methods floor or lists every analysis run.
   - The locked primary is not kept beside a post-lock change (`decisions.py:disclose`).
   - "Which of my decisions mattered?" is not served; it exists only in the prototypes.
9. **Export.**
   - There is no manuscript model and no LaTeX or Word writer. Classic's `ml/latex_report.py` and `ml/manuscript_validator.py` are references.
   - There is no citation registry, no `refs.bib`, no Crossref check, and no manuscript gate (DoD gate 7).
   - The bundle tabulates only Table 2 (with its appendix and calibration) or the performance table. Causal, time-varying, substitution, sensitivity, secondary, scales, usual-intake, explanation and evaluation results are not tabulated (INBOX 250, 257, 259).
   - Figure numbers are hard-coded (`export/figures.py`).
   - There is no CONSORT checklist, and "if applicable" items cannot be marked not applicable (INBOX 251).
   - There is no small-cell suppression.
10. **The ordering fixes** from the disagreements above:
    - the roles completion;
    - the draw separated from the validation scheme;
    - the survey question under Predict;
    - a Router exception that lets "Decide now" open a later question whose needs are met;
    - substitution before Fit;
    - the after-estimates exemptions.
11. **Data in:**
    - stacking files with a shared schema;
    - a sheet and member picker (only the first Excel sheet and the first XPT member are read);
    - a per-column ledger endpoint;
    - a level-map repair for category spellings;
    - the mixed-units conversion;
    - order for text predictors;
    - check-all-that-apply blocks;
    - the instrument-kind and supplement readings.
12. **Refusals with exits:**
    - Zero rows left after exclusions or complete cases crash the design stage with scikit-learn's error (INBOX 255).
    - An unorderable time column raises `ValueError` in `stages/rows.py:split_stage` instead of refusing with exits.
13. **Previews and follow-ups** (HANDOFF, "Engine follow-ups found"):
    - the energy preview quotes a coefficient before the lock;
    - the exposure and estimand previews draw nothing;
    - the event and goal previews draw only a note;
    - the metabolomics zero-row crash;
    - the elastic net's platform-dependent penalty.
14. **View kinds.** The closed vocabulary in `consequences.py` (row flow, lineage, table focus, distribution, relationship) has no table, forest, curve, calibration, decision curve, specification curve, overlap, embedding, matrix or page. Each is a recorded design decision. The footprint layouts (Focus, Strip, Flow, Routing, Angles) have no engine field (`PreviewResult`; `contracts.py:MethodContract`).
15. **Plain words.** The engine writes the card text: `teaching/content.py` and `estimand.py` use "exposure", "confounder" or "estimand" about 396 times, many in strings a card shows. The DoD (2026-10-07) asks for plain words on every card, so the engine's card text is rewritten, with a word-list check.
16. **The omics chain in Models.** The normalization is chosen today as a repair option of the `omics_scale` finding in Your data (`methods/omics.py:normalization_of`). MODELING_SEQUENCE §1 step 4 places the method in Models. Your data keeps only the reading.
17. **Smaller engine items the patch found:**
    - the log scale is valid for any positive outcome (`_outcome_scale_fits`) but offered only when the scale question fires (`default:outcome_scale_original`);
    - a direct effect becomes "Not available yet" with an exit to the whole effect (`q:estimand`);
    - the residual Q-Q plot has no engine code (`exhibit:residual-qq`);
    - the cross-model importance table has no engine code, though each family's importance exists (`exhibit:cross-model-importance`);
    - the split's seed has no control (`default:split-seed`).

### Interface

1. **The quest-log shell.** Production has the linear Record (`components/record/Record.tsx`). The following exist only as the calm-kit prototype, on fixtures:
   - the seven stages;
   - the seven-segment bar with its drop-back reason;
   - Decide · Confirm · For the record;
   - hover elaboration and card expansion;
   - the manuscript rail (`calm-kit/parts.tsx:ManuscriptRail`).

   The prototype (`.worktrees/calm/…/explore/calm-quest`) is also built on STROBE's methods headings, not on the seven stages.
2. **Data in.** `api/client.ts` has functions for files, joins, join previews and codebooks, and no component calls them. There is no stacking screen, and no practice dataset on the Start screen.
3. **Cards that read every engine field.** Six Router questions render through `GenericAsk`: estimand, adjustment, time-varying, form, modification and causal. Measure labels, derived roles, knots, contrast levels, trim and folds go unread. The bespoke cards omit options the engine offers:
   - TaskAsk: ordinal and time to event;
   - SealAsk: the validation scheme and the floor;
   - MissingAsk: ranked methods, below-detection fills, the imputation model and m;
   - ExclusionsAsk: several rules, and "keep unrecorded rows";
   - ClustersAsk: structural groupings;
   - SurveyAsk: the design reading;
   - RepeatKindAsk: imputed copies;
   - AggregationAsk: per-column rules.
4. **Decision kinds no control emits** (`frontend_inventory.json`, "Decision kinds the frontend never emits"): `set_feature_table`, `set_categorical`, `set_outcome_unit`, `set_intended_use`, the follow-up window, `set_model_sequence`, `set_sensitivity`, `set_measurement_error`, `set_scales`, `set_batch`, `set_usual_intake`, `set_selection`, `set_levers`, `set_explain`, `respond_diagnostic`, `set_updating`, `reseal` and `view_outcome`.
5. **First look.** None of it is drawn:
   - the index;
   - the six groups;
   - "Worth a look";
   - the walk;
   - the outcome door;
   - the context view;
   - an adapter from Explore's findings to the canvas's views.

   The profile and the column summaries read every row, the outcome included (`/columns`), so the index must not draw the outcome.
6. **Flowcharts.**
   - The participant flow counts rows only (`export/figures.py:participant_flow`). It has no units under a repeated grain, no ghosted survey domains, and no comparison of leavers with stayers.
   - There is no samples-and-features flow, no CONSORT flow, and no analysis flowchart with Fit on it.
7. **Results.**
   - Eleven result stages are never fetched: effects, sensitivity, secondary, calibration, scales, usual intake, causal, explain, modification, explore and evaluation.
   - The plan locks silently, and the after-estimates marks are not shown.
   - Under Estimate, `Coefficients.tsx` shows energy terms beside the exposure as if they were effects.
   - There is no exhibit view with wording and placement.
8. **Write-up.**
   - There is no full-width manuscript and no export screen. `GET /export` and `/checklist` have no client function; `/methods` and `/plan` have client functions that no component calls.
   - There is no live checklist, no place to write author-only text, and no merge of tracks.
9. **Purposes and pedagogy.** Every new element needs a purpose entry (BLUEPRINT §11.2) and a pass of the pedagogy reviewer. None exists yet for Write-up, the exhibits or First look's looks. `purposes.py:VIEW_PURPOSES` answers "change" and "matters", not "is the data what it says".
10. **Classic's six views.** All six approved on 2026-10-05 now have an item: missingness patterns, the skew and outlier table and the pre-fit VIF table in First look (`exhibit:classic-exploration-views`), residual Q-Q and the cross-model importance table in Results, and the split and seed control in Who's in.

## Thread families and where they land

Each catalog noticing belongs to one of the 17 families of `UNDERSTANDING_LAYER.md` §3.1 (Appendix A lists 333). The 34 the completeness review added after Appendix A carry the family their own entry names (`catalogs/completeness_review.json`, "failure_prevented"); one, the identifiable values in the export, is outside the families by design. A thread lands where the item that carries it lives. Each family's check that runs under every lens (`sentinel:*`, U10) lands where most of its threads do.

| Family | Threads | Your data | Your question | Who's in | Models | Results | Write-up | Its sentinel lands in |
|---|---|---|---|---|---|---|---|---|
| S1 Shortcut | 16 | 0 | 0 | 4 | 12 | 0 | 0 | Models |
| S2 Leak in time | 8 (1 from the review) | 0 | 1 | 3 | 3 | 1 | 0 | Models |
| S3 Leak in meaning | 9 (1 from the review) | 0 | 4 | 0 | 5 | 0 | 0 | Models |
| S4 Not independent | 25 (3 from the review) | 4 | 0 | 20 | 1 | 0 | 0 | Who's in |
| S5 Who is in | 25 (1 from the review) | 5 | 0 | 16 | 4 | 0 | 0 | Who's in |
| S6 Done before upload | 6 | 4 | 0 | 0 | 2 | 0 | 0 | Your data |
| S7 Drift and transport | 10 (1 from the review) | 0 | 1 | 1 | 8 | 0 | 0 | Models |
| K1 What a value means | 55 (6 from the review) | 39 | 0 | 5 | 11 | 0 | 0 | Your data |
| K2 What a zero or blank means | 23 (1 from the review) | 0 | 0 | 14 | 9 | 0 | 0 | Who's in |
| K3 How well it measures | 46 (5 from the review) | 4 | 1 | 8 | 32 | 1 | 0 | Models |
| K4 Causal place | 26 (1 from the review) | 0 | 2 | 3 | 21 | 0 | 0 | Models |
| K5 Structure among variables | 26 (3 from the review) | 2 | 0 | 2 | 22 | 0 | 0 | Models |
| K6 Outcome and clock | 29 (2 from the review) | 2 | 18 | 3 | 6 | 0 | 0 | Your question |
| K7 Reference and context | 12 | 0 | 0 | 5 | 7 | 0 | 0 | Models |
| E1 Support | 19 (1 from the review) | 0 | 0 | 3 | 15 | 1 | 0 | Models |
| E2 Noise and multiplicity | 20 (4 from the review) | 0 | 0 | 0 | 9 | 11 | 0 | Results |
| E3 Reading the result | 11 (3 from the review) | 0 | 0 | 1 | 3 | 7 | 0 | Results |
| None (an obligation, not a validity family) | 1 | 0 | 0 | 0 | 0 | 0 | 1 | — |
| **All** | **367** | **60** | **27** | **88** | **170** | **21** | **1** | |

## The reference journeys against the load caps

The caps (`UNDERSTANDING_LAYER.md` §2.8): at most 3 highlights in First look, at most 1 context line per card, at most 3 surfaced noticings per stage, and each reference journey within its catalog's target of asked rows. The gate that reports asked rows per journey (U5) does not exist yet, so two measures stand in.

**What each journey answers today.** Engine cards answered in the review-packet captures (`review-packets/captures/*.json`, commit `2ba3c5727a`), mapped to the quest stage that asks them. Repeated answers on one card count once; each repaired finding counts once. Roles are left out (a completion in the quest log, disagreement 1), and so is the split under Estimate (For the record). The last column is what the catalogs allow on top, in noticing rows.

| Journey | Your data | Your question | Who's in | Models | Results | All | Noticing rows the catalogs target |
|---|---|---|---|---|---|---|---|
| dietary-inference | 4 | 2 | 3 | 8 | 0 | 17 | about 5 new cards; 4 surfaced (§2.8) |
| dietary-prediction | 3 | 2 | 3 | 4 | 1 | 13 | none given (shared: 8–12 asks with batching) |
| clinical-inference | 3 | 2 | 2 | 6 | 0 | 13 | 4–5 rows on 3 cards; 7–9 stated (clinical.json, NHANES) |
| clinical-prediction | 5 | 4 | 7 | 2 | 0 | 18 | 8–9 rows on 5 cards; 4–6 surfaced (§2.8) |
| metabolomics-inference | 8 | 3 | 3 | 6 | 0 | 20 | 2–4 cards; 5–7 surfaced (§2.8) |
| metabolomics-prediction | 8 | 3 | 4 | 3 | 0 | 18 | 3–5 cards; 5–7 surfaced (§2.8) |
| genomics-inference | 4 | 3 | 3 | 6 | 0 | 16 | 2–3 asks; 0–1 surfaced (§2.8, shipped fixture) |
| genomics-prediction | 4 | 3 | 4 | 3 | 0 | 14 | 2–3 asks; 0–1 surfaced (§2.8, shipped fixture) |
| survey-inference | 2 | 3 | 2 | 6 | 0 | 13 | about 4 asks; 0 surfaced (survey.json, single instrument) |
| survey-prediction | 2 | 3 | 3 | 3 | 0 | 11 | about 4 asks; 0 surfaced (survey.json, single instrument) |
| dietary-describe | — | — | — | — | — | not captured | none yet (new journey, 2026-10-07) |
| randomized-trial | — | — | — | — | — | not captured | none yet (new journey, 2026-10-07) |

**What the crosswalk could put before each journey.** Decide / Confirm items whose goal and lens fit the journey, before any condition in `fires_when` is read. This is an upper bound: Models lists 132 Decide items, but a journey answers at most 8 engine cards there today.

| Journey | Your data | Your question | First look | Who's in | Models | Results | Write-up |
|---|---|---|---|---|---|---|---|
| dietary-inference | 33 / 7 | 26 / 3 | 4 / 0 | 44 / 11 | 71 / 31 | 32 / 3 | 10 / 2 |
| dietary-prediction | 33 / 7 | 28 / 4 | 4 / 0 | 41 / 10 | 64 / 27 | 34 / 4 | 10 / 2 |
| clinical-inference | 30 / 6 | 28 / 3 | 4 / 0 | 44 / 13 | 63 / 28 | 30 / 3 | 10 / 2 |
| clinical-prediction | 30 / 6 | 30 / 4 | 4 / 0 | 40 / 12 | 56 / 25 | 35 / 4 | 10 / 2 |
| metabolomics-inference | 33 / 6 | 27 / 4 | 4 / 0 | 40 / 10 | 60 / 23 | 33 / 3 | 10 / 1 |
| metabolomics-prediction | 33 / 6 | 29 / 5 | 4 / 0 | 39 / 10 | 53 / 21 | 35 / 4 | 10 / 1 |
| genomics-inference | 30 / 7 | 28 / 4 | 4 / 0 | 43 / 8 | 63 / 21 | 31 / 3 | 9 / 1 |
| genomics-prediction | 30 / 7 | 30 / 5 | 4 / 0 | 42 / 8 | 56 / 19 | 33 / 4 | 9 / 1 |
| survey-inference | 27 / 7 | 27 / 4 | 4 / 0 | 38 / 9 | 68 / 24 | 31 / 3 | 10 / 2 |
| survey-prediction | 27 / 7 | 29 / 5 | 4 / 0 | 36 / 8 | 61 / 20 | 35 / 4 | 10 / 2 |
| dietary-describe | 33 / 6 | 23 / 3 | 3 / 0 | 31 / 6 | 41 / 7 | 11 / 1 | 9 / 2 |
| randomized-trial | 30 / 6 | 28 / 3 | 4 / 0 | 44 / 13 | 63 / 28 | 30 / 3 | 10 / 2 |

**Reading the two tables.**
- Today's journeys are within reach of calm: 11 to 20 engine cards each.
- The noticings are what would break the caps: the catalogs allow 2 to 9 asked rows per journey on top, and Models alone has 14 cards that noticings land on.
- So the U5 gate is a release requirement, not a report: a noticing that would push a journey past its target joins an existing row or is stated (§2.8). P0.9 builds the gate, and T1 and T2 must pass it.
- The Describe and trial journeys have no capture yet; they get one with D7 and E5.

## How this was built

1. **The stage maps.** Seven stage maps were made read-only over the repo at `2627066d`, one per ruled stage. Each listed every question, decision, default, noticing, preview, refusal and exhibit its stage touches, with file and symbol.
2. **Folding.** Entries with the same id, or the same engine decision under another id, were folded into one item. For example, Write-up's Table 2 is Results' Table 2, and the seal appeared in both Who's in and Models. Every folded entry is in the item's `merged_from` and `cross_refs`.
3. **Home stage.** An item lives where it is decided. A noticing shown earlier carries `noticed_at`; later appearances are `returns_at`. Each catalog noticing is placed by these rules, in order:
   1. born after the fit: Results;
   2. settled before First look: Your data or Your question;
   3. decided on the outcome card: Your question;
   4. decided at a Who's in question: Who's in;
   5. decided or returning on a Models card: Models;
   6. otherwise, the first stage it returns to.

   Each catalog noticing records its reason in `placement_reason`.
4. **Engine items carry noticings.** A noticing the engine already handles under another name is carried by that item (`threads`), not duplicated. Containers are listed in `groups`, not as items: Models' noticing cards, Results' context-line groups and First look's "Settled" lines.
5. **Ordering.** Within a stage, items follow the recommended order of the Decide objectives, then Confirm, For the record and Shown. Noticings are grouped by the card they are decided on.
6. **What is carried over, and what is new.** Status, goals, lenses and `engine_source` come from the stage maps and the catalogs. The exhibit wordings, placements and pre-included flags are new in this crosswalk, and are proposals for review.
7. **The patch.** The completeness critic found 29 gaps. Each was checked against its source before it was applied. The patch:
   - moved results behind Fit and split six Models noticings at the lock;
   - placed the substitution, omics and survey-estimator items in Models, and added a Confirm sweep to each stage;
   - added thread families and the 17 family checks, and Classic's last three views;
   - placed the flowcharts as exhibits, and covered Describe in Who's in and Write-up;
   - fixed 57 references to ids that did not exist;
   - rewrote 52 titles and 21 group titles in plain words.

## Critic notes not taken

The critic found 29 gaps. Each was checked against its source. All were applied except the parts below.

- **`default:complete_case_predictors` stays For the record.** The critic named eight defaults to relabel Confirm; seven are now Confirm.
  - This one counts complete cases on the predictors the model keeps (`stages/rows.py:cohort_inputs`; `decisions.py:left_out`).
  - No alternative is offered or sound: counting blanks in columns the model never reads would drop rows for nothing.
  - Its count changes only when the adjustment set changes, which is decided on that card.
- **Univariate or multivariate regression calibration is not added as a choice.** The contract implies the form:
  - multivariate whenever the model holds two or more error-prone intakes, univariate with one;
  - calibrating one intake at a time is refused (`methods/calibration.py`: options `multivariate`, `univariate`, `one_at_a_time`; relation `multivariate`).

  So the form is stated, not asked. The card's title now says "all of them together", and its canvas says which form runs (Settled here).
- **`metab-treatment-marker` keeps its outcome-blind part before the lock.**
  - Its "top feature" form needs a fit, so its `noticed_at` is cleared, as the critic asked.
  - But the catalog's first route is "outcome-blind, always allowed": a drug-class annotation, a medication column, or presence–absence bimodality (`catalogs/metabolomics.json`). Under inference, medication is "decided by meaning before the lock".
  - That part stays on the adjustment card, recorded in the item's notes.
- **The Fit rule is settled, not put to Nolan as a question.** The critic was right that the first fix contradicted RECIPES §4.4 and ruling 2. The corrected fix keeps both rulings, so no product choice is left (Settled here). It is folded into P0.2 and P0.8, as the critic asked.
- **The family landing counts are not the critic's.**
  - The critic's examples were S4: Who's in 34, and K6: Your question 26.
  - Counting each catalog thread once, at the stage of the item that carries it, gives S4: Who's in 20, and K6: Your question 18. The table above states that basis.
  - The rest of that gap was applied: a family on each thread, the table, the 17 family checks, and the full rollout order in SIZING T2.
- **The track order stays a methods ruling.** The critic allowed a ruling or a question, so the order stays in Settled here. The parts it found wrong are corrected there:
  - per-track locks are now in v2;
  - the rows the Estimate track reads now set how a later Predict track validates.

  The blanks across tracks, which the critic also raised, became question 5.
- **RECIPES §9 adds up to about 64 units, not about 61,** with M–L read as 5.5 and S–M as 2. C6a and C6b carry those sizes; RT-8 is counted once, in P0.8.
