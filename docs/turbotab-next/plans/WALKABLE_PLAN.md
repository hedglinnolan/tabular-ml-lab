Planned 2026-10-10 on turbotab-next @ bf289d1c, with C6a phase 3 integrated on feat/wave-c6a3-int and not merged.

# Plan: from the mockup to a walkable TurboTab v2, NHANES glucose under Predict first

I edited, committed and pushed nothing, and ran no fits and no test suites. Three Explore agents surveyed the live frontend, BE1–BE24 and the Predict path. I spot-checked the load-bearing claims myself; anything only inferred from code is marked "inferred".

## 0. Summary

1. **Today's live app already walks NHANES glucose under Predict, through the old screen.**
   - The old screen is the linear Record on the left, the Stage on the right and a 4-segment Banner on top.
   - Since P0.8 (Fit, commit 35fec532), three things block it:
     - The built UI is stale: `turbotab/frontend/dist` dates from Oct 2 and has no Fit button. I verified this. The launcher serves it as is.
     - A fit expected to take over 120 s is held, and then no Fit button ever appears (inferred from code). This is certain for any tuned NHANES shelf.
     - Every real-server e2e spec and journey driver predates the Fit press, so they stall (inferred) and the captures are stale (2026-10-05).
2. **The engine already serves a complete quest log** through `GET /api/projects/{pid}/quest`, along with `/triage`, `/record`, `/fit`, `/fit/cancel`, `/methods`, `/checklist` and `/export`. No frontend code calls any of them except `/fit`. The shell (P0.7) is therefore mostly a frontend job.
3. **The Predict post-Fit engine is rich.** It serves:
   - out-of-fold R², MSE, RMSE and MAE against a no-predictor baseline;
   - BBC-CV selection;
   - calibration slope, intercept and curve;
   - the sealed held-out score;
   - SHAP, ALE inductive-bias curves with support masks and a performance floor;
   - the architecture lane and the interpretable cost.

   Live, the user sees only the comparison table and the opening of the held-out rows.
4. **None of BE1–BE24 is needed to walk NHANES Predict.** They are the Estimate (sugar) journey's requirements: about 78 units, mostly colliding with C6a phase 3's `decisions.py`, `quest.py` and `voice.py`. They belong in waves 3–4.
5. **Proposed waves:**
   - **W1 "Walk it":** a quest-log shell on live data, reusing today's cards. NHANES Predict and an own CSV run from upload to the bundle. Predict beats are written in parallel.
   - **W2 "Predict, drawn":** Predict's Models and Results built to the new beats.
   - **W3 "Estimate Models, drawn"** and **W4 "Estimate Results and the paper":** Direction A on live data.

## 1. Today's live app

**Routes** (`turbotab/frontend/src/router.tsx:48-70`, `App.tsx`):
- `/` is StartScreen: upload, file browser, typed path, project list.
- `/p/:pid` is ProjectScreen (`screens/ProjectScreen.tsx`): Record, Banner, Stage and JobChips, with SSE.
- Every `/lab/*` route exists only with `VITE_MOCK=1`: the calm kit, `/lab/views` (the ten view kinds), the m3 replays and the methods and calm prototypes. The static `calm.html`, `protos.html` and `views.html` builds use fixtures only.

**The client** (`src/api/client.ts`, the only module that calls fetch):
- Wired: health, projects, upload, project view, decisions, stages and run, columns, jobs, teaching, preview, finding evidence, `pressFit`, SSE.
- In `client.ts` but used by no component: table, histogram, readings, methods, plan, models, files, joins, codebooks.
- No client function at all: `/quest`, `/record`, `/triage`, `/materiality`, `/export`, `/checklist`, `/fit/cancel`, and the `decide_now` parameter. The confirm sweep (`{"kind":"confirm_sweep",…}` posted to `/decisions`) has no control.

**Stage by stage** (stage keys from `core/quest.py:106-114`):

| Stage | Live today | Missing |
|---|---|---|
| **Your data** | Upload of csv, tsv, gz, parquet, xlsx and xpt (`POST /projects/upload`, no size cap in local mode); lens, orientation and roles cards; the readings ask card (`record/ask/AskCard.tsx`); findings with Repairs | Join and codebook screens (hooks exist, unused); the Confirm sweep; For the record (`/record`) |
| **Your question** | Target, task, event, purpose (prediction or inference only, `FactQuestions.tsx:237`); follow-up and design through `GenericAsk` | A card for `set_intended_use`; **moment of use (missing in the engine too: no `q:moment_of_use`, no `sentinel:S2`)**; the sweep |
| **First look** | Only the findings list | Nothing reads the explore stage; `view_outcome` is unreachable |
| **Who's in** | Grain, repeats, unit, aggregation, temporal, clusters, survey, exclusions and missing-values cards; SealAsk (holdout share and its basis) | **SealAsk hard-codes `validation:"kfold", folds:5, repeats:10, n_boot:500, nested_cv:false`** (`SealQuestions.tsx:26-39`, "WP9"); no sweep |
| **Models** | ModelsAsk: a multi-select shelf by rank with its cost estimate (`ChoiceQuestions.tsx:524-600`); energy-adjustment and substitution cards; the estimand, adjustment, form, modification and causal questions through GenericAsk | No control for `set_selection`, `set_levers`, `set_validation`, `set_model_sequence`, `set_scales`, `set_sensitivity` or `set_explain`; no sweep; no flowchart with Fit on it |
| **Results** | **FitPress** (`stage/results/Results.tsx:82-128`, the only caller of `/fit`): the CV comparison, opening the held-out rows, held-out scores, linear coefficients, substitution curves, SVG and PNG save | Evaluation, explain, calibration, effects, sensitivity and secondary are never read. The ten views are mounted only in `/lab/views`. Cancel is not wired |
| **Write-up** | Nothing | Methods, plan, checklist and the export zip |

**Blockers today:**
- **B1. Stale build.** `dist` is from Oct 2 with no `press-fit` (verified). `turbotab/deploy/launch.py` builds only when `dist` is missing, so the launcher and a bare `python -m turbotab.server` show a UI with no Fit, and Results are unreachable.
- **B2. A held fit dead-ends** (inferred, `server/service.py:1718-1731`).
  - When the estimate exceeds `HOLD_SECONDS=120` (`core/fit_press.py:51`), the fit produces no artifact.
  - The Stage shows Results only when `fit.artifact` exists (`Stage.tsx:117,134`), so FitPress never renders, and the frontend never reads `StageStatus.held`.
  - Any tuned shelf on 21,849 rows is held.
- **B3. Real-server e2e specs never press Fit.** This covers `m1-journey`, `m2-journeys` and `m3-no-dead-end` (its `nhanes-prediction` journey included), so they likely stall (inferred). CI runs only the mock suite, where the real-server specs skip, so nothing has noticed.
- **B4. The drivers never press Fit while following the Router:**
  - `turbotab/core/reference/journeys.py:544-575` `follow()` (the `dietary-prediction` glucose journey);
  - `docs/turbotab-next/m3/capture_fixtures.py`.

  The fix is small: `server_drive.Drive.artifact` already presses and releases (`server_drive.py:453-466`).
- **B5. Methods hazard on NHANES.**
  - `meds_hbp` (15,552 blank) and `meds_chol` (17,204 blank) are the only columns with blanks (verified by count). Both are skip patterns: a blank means "not asked".
  - The default `MissingSpec.categorical="impute"` (`decisions.py:480-503`) fills a category by its most frequent value (`models/pipeline.py:659`). The agent reports this turns the blanks into "on medication".
  - Complete cases keep 2,996 rows, a biased group (mean glucose 126 against 104.6).
  - `missing_category` exists. A "blank = not asked → No" recode does not (INBOX.md:86).
  - This is a false-claim blocker for the walk.
- **B6. No moment-of-use question.** Triglycerides and HDL come from the same fasting draw as glucose, and nothing asks whether the model may read them at prediction time.

**How to launch today:** `turbotab/deploy/Start TurboTab.command` or `python3 turbotab/deploy/launch.py` (port 8787, local mode), or `venv/bin/python -m turbotab.server --port 8787` plus `npm run dev`. The stale `dist` must be rebuilt first.

## 2. The walkable v0 (NHANES glucose under Predict first, then an own CSV)

**Definition.** Every stage is reached through the quest line. Every open line can be answered through a card, or confirmed in its stage's sweep. Fit is reachable with its estimate and a Cancel. The held-out rows open once, after the triage. Results show honest held-out numbers with calibration and what drives the model. Write-up downloads the bundle. No number is shown that the engine did not serve. Nothing is in Direction A's style yet: the card idiom inside the new shell is today's.

| Stage | v0 minimum | Reuses |
|---|---|---|
| **Your data** | Upload; lens, readings, findings and roles as quest lines; Data sweep; For the record from `/record`. On NHANES, `imputed_*` read as flags (TRUST names them in the complete-case sentence) | Record ask cards (after extraction, §4); Stage previews or kit `CanvasFrame` |
| **Your question** | Outcome; goal Predict; task For the record; **an intended-use card** (engine exists: `decisions.py:1761-1778`; decision support refused for a numeric outcome, `decision_curve.py:473-488`). Moment of use waits for W2 (the beats design it): v0 is honest because the methods list the predictors | Record cards plus one new card |
| **First look** | Reached; lists its noticings by where each is decided; "Nothing to decide here" | Findings list |
| **Who's in** | Exclusions (implausible intake: 194 rows below 500 kcal, 307 above 5,000); missing values with **B5 fixed**; **a validation-scheme control replacing SealAsk's hard-code** (k-fold, repeated, bootstrap optimism, internal–external by cluster such as `cycle_begin_year`; `core/validation.py`); the held-out draw last | SealAsk plus the validation card |
| **Models** | Selection card (`set_selection`, which the families line waits for); collinearity Decide (BMI from weight and height); families with cost; Models sweep (validation, in-fold rules `set_levers`); triage of noticings; **Fit with "about N min", the held state and Cancel**; the flowchart as today's lineage view | ModelsAsk, Lineage, new small cards |
| **Results** (Predict order) | Results sweep (`set_updating` for linear); triage (`GET /triage` plus `confirm_sweep(stage="results", sweep="noticings")`); name the final model (`selection.py:657-722`, quoting BBC-CV); open the held-out rows; held-out RMSE, MAE and R² against the baseline; calibration curve; the explain offer (`set_explain`), then SHAP and ALE curves | Results.tsx comparison and opening; `components/views` CalibrationView and CurveView |
| **Write-up** | Methods at full width in TRIPOD+AI order; the checklist (17 of 52 items have rules; the capture scored 4 answered, 11 partly answered, 37 unanswered); download the bundle (refused until the held-out rows are opened, `export/gate.py:142-155`) | `GET /methods`, `/checklist`, `/export` |

**Own CSV:**
- The same shell. Router keys with no bespoke card render through GenericAsk.
- v0 must guarantee that every open line on any path has a renderer (acceptance in §5).
- Joins, codebooks and stacking stay out of v0 (C1 and C1b).

**Direction A now or later.**
- No stage needs Direction A to be *walkable*.
- To be *judged*, Predict's Models and Results need their own beats, then a build (W2).
- Stages 1–4 keep today's cards in the shell until after Nolan judges W2. The return grammar (BE1) and plain names (BE13) reach them later.
- Estimate's Models and Results get Direction A in W3 and W4.

## 3. Engine requirements BE1–BE24

Sizes: S=1, S–M=2, M=3, M–L=5.5, L=8. "Phase 3" is `feat/wave-c6a3-int` @346b6acb: integrated, pushed to `ci/`, not merged. It changes `decisions.py` (VAL), `quest.py` (TRUST, `QUEST_VERSION` 5), `voice.py` (TRUST), `server/service.py`, `readings.py`, the stage files, the model files and `generated.ts`. **`estimand.py` is not touched by phase 3 as built**, but MC2-CLEAN (no branch yet) may touch it. Phase 4 (MC-2b-3) will hold `stages/modeling.py`, `effects.py` and `evaluation.py`.

| # | Status | Serves it today (core/…) | Size | Needed for | Conflict with phase 3/4 |
|---|---|---|---|---|---|
| BE1 return grammar | NEW (parts exist) | `materiality.invariance` :219; `estimand.SUBSTITUTION_METHODS` :2224; `quest.ChangedSince` :785 | 5.5 | W3 | quest, decisions, voice |
| BE2 answer text on records | PARTIAL | `DecisionRecord.sentence` decisions.py:2130 | 2 | W3 (v0 uses the record's sentence through `/methods` `record_id`) | decisions, service |
| BE3 comparison view data | PARTIAL | `energy.energy_factor` :413, `omitted_energy`; `materiality.smd` :255 (unsigned) | 3 | W3 | none if a new module |
| BE4 human unit (step) | PARTIAL | `step_kcal` on SubstitutionSpec only; `e_values(delta=)`; TRUST adds `voice._swap_step` | 3 | W3 | decisions, voice, effects |
| BE5 expected direction | NEW | none | 2 | W3 | decisions, quest, voice |
| BE6 Model 1 in the sweep, ladder before Fit | PARTIAL | `estimand.model_sequence_card` :1383 | 3 | W3 | quest, voice |
| BE7 second comparison (exact contrast) | PARTIAL | `substitution_stage` bootstrap band (modeling.py:2604); TRUST labels it an exhibit | 3 | W3 | decisions, voice, quest, stages |
| BE8 family settled, with its basis | NEW | none | 2 | W3 | decisions (VAL), quest, modeling |
| BE9 triage acts | PARTIAL | `sweep.triage` :644, `recommend` :519 | 3 | W4 | decisions |
| BE10 which choices mattered | PARTIAL | `materiality.ledger` :1180, `verify`; `InferenceTable.cov` not served | 5.5 | W4 | modeling, stages/__init__, quest |
| BE11 interpretation checks | NEW | inputs: robustness value and benchmarks (effects.py), `exposure_tests` | 3 | W4 | quest, effects |
| BE12 wording | NEW | `substitution.effect_label` only | 5.5 | W4 | voice, quest, decisions |
| BE13 names | PARTIAL | `ImportCodebook.labels`, `codebook.Entry.label` | 5.5 | W3 (W2 needs plain names for Predict, see P7) | voice, quest |
| BE14 paper sentences | PARTIAL | voice receipts, `provenance.MethodsText`, `export/methods.py` | 5.5 | W3 | voice (heavy overlap with TRUST), decisions |
| BE15 flowchart | PARTIAL | `design_stage` modeling.py:567 | 3 | W3 | modeling (phase 4), quest |
| BE16 sources | PARTIAL | 2 of 10 in `citations.json` | 2 | W4 (can run any time) | none |
| BE17 partner asked | PARTIAL | partner inferred: `substitution_words` :596, `_in_place_of` :1534 | 3 | W3 | decisions, voice |
| BE18 partner mix | NEW | none | 2 | W3 | none |
| BE19 diagnosis marker | PARTIAL | not-asked blanks coach.py:316,573; SetModification | 5.5 | W4 | decisions, quest, voice, interaction |
| BE20 reverse causation | NEW | none for same-visit designs | 2 | W4 | quest |
| BE21 survey "unweighted" | PARTIAL | `survey_design_absent` | 1 | W3 | voice |
| BE22 sealed checks | PARTIAL | `plan_lock.plan_of` :79 | 3 | W3 | decisions, voice, stages/__init__ |
| BE23 tail question | NEW | no clinical cut-off fact | 3 | W4 | quest, stages/__init__ |
| BE24 per-column repairs | PARTIAL | `repairs.SAS_ZERO` :107 | 2 | W4 (on NHANES Predict, 119 `bp_di` zeros are a predictor; v0 survives on today's plausibility offer) | none |

- **Totals:** about 78. W3 (Models) is BE1–BE8, BE13–BE15, BE17, BE18, BE21 and BE22, about 46.5. W4 (Results) is BE9–BE12, BE16, BE19, BE20, BE23 and BE24, about 31.5.
- **Clear of phase 3:** BE3 (if a new module), BE16, BE18, BE24.
- **Every BE-item that touches `decisions.py`, `quest.py` or `voice.py` waits for phase 3 to merge.** Those that touch `stages/modeling.py`, `effects.py` or `evaluation.py` (BE4, BE7, BE10, BE11, BE15) wait for phase 4's MC-2b-3, or coordinate with it.

**Predict engine items** (proposed P-list, to be confirmed by the Predict beats):

| P | What | Status | Size | Wave | Files and conflict |
|---|---|---|---|---|---|
| P1 | B5: a skip-pattern blank gets an honest default: `missing_category` recommended, plus a "not asked → No" recode | NEW (small) | 2 | **W1** | `core/methods/missing.py`, `core/coach.py`, `core/repairs.py`. Not phase-3 files; avoid `decisions.py` |
| P2 | The open-noticings gate blocks the opening of the held-out rows (U8; today the triage does not block it) | NEW | 1 | **W1** | `core/seal.py` validators (not phase 3) |
| P3 | Moment of use: "what the model may read at prediction time", compiled to predictor roles plus a methods sentence (`q:moment_of_use`, `sentinel:S2`) | NEW | 3 | W2 | decisions, quest, voice (after phase 3) |
| P4 | Where it fails: error by outcome range (the ≥126 mg/dL tail) and calibration at the high end; subgroups through intended use (exists, `evaluation.py:323-341`) | NEW | 2 | W2 | a new stage file plus `stages/__init__` (after phase 3; avoid `evaluation.py` until MC-2b-3) |
| P5 | How it is judged: the declared yardstick for a skewed outcome (skew 4.58, log skew 2.38, so `structural.scale_question` stays silent, verified; MSE is driven by the tail) | NEW (methods ruling) | 2 | W2 | metrics declaration |
| P6 | Cross-model importance table | PARTIAL | 2 | W2 | `models/explain.py` (after phase 3) |
| P7 | Plain names for quest lines and NHANES columns (P0.12 subset plus an NHANES label file) | PARTIAL | 3 | W2 | quest, voice |
| P8 | Sectioned manuscript endpoint (`export/methods.py` sections) for the rail | NEW | 1 | W1 (after phase 3: `service.py`) | routes, service |

Prediction intervals (conformal) are not in the DoD; send them to INBOX.

**C6a dependencies of this journey:**
- **Phase 3 is needed:**
  - RT-5a: boosted trees tuned, rank 3.0, the shelf's first choice for glucose.
  - RT-5f: the elastic net on its path search, rank 2.5.
  - F15-rest: explain's and calibration's refits get the seed, design and cancel.
  - VAL: family refusals and tuning ranges.
  - TRUST: quest v5. A refusal the fit waits on becomes a Decide ahead of the families question, which both NHANES captures hit. Results counts only Predict's exhibits.
  - **The shell (W1-E) must build on quest v5.** Merge phase 3 before W1-E starts.
- **Phase 4 is not needed.** MC-2b-3 leaves glucose unchanged; MC-2b-4 is omics only. It only constrains files.
- **C6b is not needed for v0.** "Lighter" (`set_tuning`, RT-6) would shorten fits. Recipes and versions come later; C6d's PR-2 (the held-state tuning line) is partly W1-B.

**Fit time (extrapolated from plans, not measured):**
- XGBoost and tuned boosted trees run 35 fits per outer fit, about 1,785 per family.
- A full 7-family tuned shelf on 21,849 rows: roughly 1.5–3 h. Linear, ridge, elastic net and Huber take minutes.
- Every tuned shelf exceeds 120 s, so it is held.

## 4. Frontend work (P0.7 and after)

**What is reusable, and how:**
- **The live data is there.**
  - `QuestLog {stages[{key, name, reached, progress{answered, required, complete}, sweep{lines, answered, heading, action, confirmed_by, changed}, lines[QuestLine], reopened[{sentence,…}]}], kinds, fit: FitLock{locked, sha256, pressed, opened, held, estimate_seconds, reason}}`.
  - `QuestLine {id, key, source, label: Decide|Confirm|For the record, name, status: answered|open|waiting|set_for_you, waiting_for, reopened_by, would_change, changes_nothing, items}` (`core/quest.py:799-906`).
  - **The answer text is not on a QuestLine.** v0 looks up `decision_id` in `/methods` lines (`record_id` and sentence); BE2 replaces this later.
- **Record cards.** The `ask(key)` dispatcher lives inside the 1,240-line `components/record/Record.tsx` (around 485-660). Extract it into `<AskFor pid view stepKey/>`; then both Record and the shell's card column render today's cards.
- **The kit on turbotab-next** (`src/explore/calm-kit`):
  - Reusable: `tokens.css`, `base.css`, `kit.module.css`, the fonts, `Why`, `POINT_GRACE_MS`.
  - The canvas (`CanvasFrame`, `Views`, `Hist`, `Lineage`, `FlowBars`, `Cells`) reads engine `ConsequenceView` and `LineageView` types, so a small adapter from the live `PreviewResult` lets it draw live previews in the calm style.
  - Must be rewritten: `Shell`, `Card`, `OptionList`, `Continue` and `ManuscriptRail`, `Overlay` and `Column` are bound to the fixture `WalkApi`. They become props-driven, keeping the CSS.
- **The Direction A mockup** (branch `design/quest-meaning-first` @ba2d1fde, worktree `.claude/worktrees/wf_97e7ce9b-235-1`):
  - Can move to the kit almost as is: `quest-log/meaning/Choices.tsx` (Choices, Checks, Pointable, Why), `Foot` (in `cards.tsx`), the `QuestTop` and `Manuscript` overlay layouts in `Meaning.tsx`, and `meaning.module.css`.
  - Cannot move as is:
    - `cards.tsx` and `tapestry.tsx` hard-code the sugar journey (NUTRIENTS, PARTNERS, GROUP_LINES, REVIEWER).
    - `Comparison.tsx` imports the fixture's `MX`, `name` and `gr`. Its views (Days, Balance, Spread, Emblem, ComparisonFrame) must be refactored to data props before they join `components/views` as the 11th kind. That happens in W3, fed by BE3 and BE18.
  - Merge the docs (`BEATS_MODELS_RESULTS.md`) to turbotab-next now; keep the explore code on its branch as reference.
- **Rule conflict.** `turbotab/frontend/CLAUDE.md` still prescribes three voices (serif, sans, mono chips), while FOUNDATION §2 rules one family (Source Sans 3). Update CLAUDE.md in W1.

**Shell components** (all new under `src/quest/`; old screen kept at `/p/:pid/classic` until W2 closes):

| Part | v0 (W1) | Later |
|---|---|---|
| Seven-stage line | Names and the open stage from `/quest`; reopen sentence beside the bar and atop the card | Plain names (P7, BE13) |
| Progress bar | Seven segments from `progress`; empty when not reached; drop-back from `reopened` | — |
| Hover panel | `lines` by label, with answers through `decision_id`; "Waiting for" links; "May change as you answer" | BE2 short answers |
| Card and tapestry | Grid of 380–440 px card, tapestry at least 60%; open line expanded through AskFor; answered lines collapsed; lines to come quiet | Direction A cards (W2, W3) |
| One primary action | Foot: Continue, "Confirm all N", Fit (at the flowchart's end), "Open the held-out rows", "Download" | — |
| Confirm sweep | `sweep.heading` and `sweep.action`; Confirm lines with reason and `would_change`; change opens a card; POST `confirm_sweep` | — |
| For the record | `GET /record`, one collapsed disclosure | — |
| Manuscript rail | P8's sectioned methods (TRIPOD order); count; newest marked; arrival line at the card's foot | BE14 paper/record split; Results paragraphs (BE12) |
| Tapestry | Kit CanvasFrame for option previews; Stage's Results scene, then views | Predict drawings (W2), comparison kind (W3) |
| Fit | FitLock: estimate, held, Cancel (`/fit/cancel`), lock time and SHA under Estimate | PR-2 "Lighter" (C6b) |

## 5. Waves

Each package owns its files; no two packages in a wave share a file. Generated files (`openapi.json`, `generated.ts`, METHODS_REFERENCE, the packets) are regenerated by the integrator only. Tiers: Opus builds anything touching statistics, guards or the leash; Sonnet only builds plumbing, always with an Opus verifier; nothing below Sonnet. The repair round triggers whenever a verifier lists any problem.

### W1 "Walk it", about 31.5 units

W1-A to W1-D and W1-H can start now on turbotab-next. W1-E, W1-F, W1-G and W1-A's P8 follow the phase-3 merge.

| Pkg | What | Owns | Tier | Size |
|---|---|---|---|---|
| W1-A | API surface: client and hooks for quest, record, triage, materiality, checklist, export, `cancelFit`, `confirm_sweep`; SSE invalidates `/quest`; P8 route | `src/api/{client,queries,schema,events}.ts`; `server/routes/projects.py` and its service method (after phase 3) | Sonnet + Opus verifier | 2 + 1 |
| W1-B | Fit reachable: read `held` and `estimate_seconds`, show Fit with "about N min" and Cancel when held; launcher rebuilds a stale dist (source newer than dist, or a version stamp) | `src/components/stage/{Stage.tsx,results/Results.tsx}`, `turbotab/deploy/launch.py` | Sonnet + Opus verifier | 2 |
| W1-C | Drivers and real-server journeys press Fit; new `e2e/quest-nhanes-predict.spec.ts` and `e2e/quest-own-csv.spec.ts`; old specs fixed; fresh light captures | `turbotab/core/reference/journeys.py` (`follow`), `docs/turbotab-next/m3/capture_fixtures.py`, `turbotab/frontend/e2e/*` | Sonnet harness; **Opus** writes the reference tests | 3 |
| W1-D | Extract `AskFor` from Record.tsx | `src/components/record/{Record.tsx,AskFor.tsx}` | Sonnet + Opus verifier | 2 |
| W1-E | The shell v0 (§4), route switch, kit move (tokens, base, kit CSS, Choices, Why, Foot) | `src/quest/**`, `src/kit/**`, `src/router.tsx`, `src/App.tsx`, `frontend/CLAUDE.md` | **Opus** (leash display, design rulings) + Opus verifier + calm/pedagogy reviewer | 8 |
| W1-F | Predict cards with no live control: intended use, selection, validation scheme (also inside SealAsk), in-fold rules, explain offer, shrinkage | `src/quest/cards/predict/**` | Sonnet + Opus verifier | 3 |
| W1-G | Predict Results v0: comparison, triage, final model, opening, held-out scores against baseline, CalibrationView and CurveView adapters from fit and explain artifacts, evaluation's interpretable cost | `src/quest/results/**` | **Opus** (number display) | 3 |
| W1-H | P1 skip-pattern default and recode; P2 opening gate | `core/methods/missing.py`, `core/coach.py`, `core/repairs.py`, `core/seal.py` plus tests | **Opus** | 3 |
| W1-I | **The Predict beats:** see the next list | `docs/turbotab-next/calm/BEATS_PREDICT.md` | **Opus** design owner, then one independent Opus critic | 5.5 |

**W1-I in detail:**
- **What it covers.** The beats for Models and Results under Predict on NHANES glucose, in Direction A's form (understands, feels, one decision, final card copy, the tapestry, the manuscript sentence, the engine served and still needed), with the P-list confirmed and sized. It must cover:
  - what the model may read at prediction time;
  - whom it is for and how it will be used;
  - how it is judged and sealed;
  - how good it is, where it fails and what drives it.
- **The engine pieces the beats draw on:**
  - intended use and subgroups (`decisions.py:1761-1778`; `evaluation.py:323-341`);
  - the validation schemes (`core/validation.py`: k-fold, repeated folds, bootstrap optimism :232, internal–external by cluster :457, Bates nested CV :722);
  - the seal and the sealed held-out score (`seal.py:33-36, 956, 1121`);
  - BBC-CV (`models/selection.py:13, 77`; the final-model refusal :657-722);
  - out-of-fold metrics with intervals and the no-predictor baseline with its Nadeau–Bengio verdict (`metrics.py:81, 89`);
  - calibration slope, intercept and curve (`models/artifacts.py:341-377`);
  - the spline benchmark and the interpretable cost (`selection.py:1013`);
  - SHAP and TreeSHAP with reseed stability, the H-statistic, and ALE with support masks and the performance floor (`models/explain.py:62-105, 1193`);
  - the architecture lane: equation :947, trees :1009, shrinkage path :1060;
  - the shelf ranking and its cost (`fit_press.py`, `models/cost.py`);
  - the TRIPOD+AI checklist (`export/checklists.py`).
- **Inputs:**
  - `core/tests/path_fuzzer.py` `answered_through(NHANES, "prediction")` for the line structure: instant, no fits, fake numbers;
  - one light capture from W1-C's fixed `dietary-prediction` driver (linear, ridge and elastic net; meds as `missing_category`), scheduled with Nolan;
  - NHANES risks: `meds_*` mark a diagnosis; triglycerides and HDL come from the same draw; BMI is a function of weight and height; 9 cycles span 16 years, so validation by cycle; the fasting subsample; unweighted.
- **No static mockup:** W2 builds the beats straight onto the live shell.
- **Methods rulings the orchestrator owns** (the beats propose them, an adversarial check follows):
  - whether energy adjustment and the substitution pair, both still asked under Predict on the dietary lens, stay;
  - the yardstick for a skewed outcome;
  - internal–external validation by cycle.

**W1 acceptance** (real-server e2e runs on a fresh `TURBOTAB_HOME`; prefer adding a light job to CI's full tier so the runs happen off Nolan's machine):
1. **`quest-nhanes-predict.spec.ts`:**
   - Uploads the decompressed `nhanes.csv.gz` through the UploadZone.
   - Walks all seven stages by the quest line, answers every open line through its card and confirms each sweep with its one button.
   - Families linear and ridge only, to keep the fit light. Fit, triage, final model, open the held-out rows, download the bundle (methods in TRIPOD sections).
   - Asserts at every step that each bar segment and hover-panel status equals `/quest`.
2. **Independent references** (pytest, seconds):
   - Rebuild the split from the seal record's held-out row ids. The held-out RMSE, MAE and R² of the linear family against `sklearn.LinearRegression` refit on the recorded predictors, to 1e-8.
   - The baseline RMSE against numpy (the held-out y around the training mean).
   - The held-out calibration slope and intercept against `statsmodels` OLS of y on ŷ, to 1e-8.
   - The e2e checks that the screen prints the artifact's numbers.
3. **Held fit:** a vitest on a held `StageStatus` and `FitLock` shows the estimate, Fit and Cancel. A real-server check under a tuned family waits for Nolan's scheduling.
4. **Own CSV:**
   - `quest-own-csv.spec.ts` on `turbotab/sample_data/clinical_longitudinal.csv` (binary Predict with a temporal seal; its capture ran in 15 s) reaches the opened held-out rows.
   - A contract test: a Python test runs `path_fuzzer` over the sample datasets × goals, writes every `(source, key)` that `quest_log` emits, and a vitest asserts each has a card renderer or a sweep home.
5. **W1-H:**
   - On NHANES, the recommended fill for `meds_*` is never the mode. After a "not asked → No" recode, the count of "Yes" equals the answered "Yes" count (an independent pandas count).
   - `open_seal` is refused while the triage holds open noticings (a hand-built state).
6. **Old specs:** `m1-journey`, `m2-journeys` and `m3-no-dead-end` press Fit and pass. The heavy journeys (metabolomics about 27 min, survey about 31 min) are scheduled with Nolan or run in CI.

**What Nolan can do at the end of W1:**
- Double-click Start TurboTab and upload NHANES or one of his own CSVs.
- Walk the seven stages with the bar filling and the hover panel working, answering each question with today's cards inside the new shell, and confirming the sweeps.
- Press Fit with its estimate and Cancel, triage the noticings, pick the final model and open the held-out rows.
- See held-out error against the baseline, calibration and what drives the model; watch the methods rail grow; download the bundle.
- Read the Predict beats and react to them.

### W2 "Predict, drawn", about 35 units

Final sizes come from the beats. Starts once Nolan has read the beats and phase 3 has merged.
- **Engine (Opus):** P3 moment of use (3), P4 where it fails (2), P5 the yardstick (2), P6 cross-model importance (2), P7 plain names (3), TRIPOD rules for this path (2).
  - Ownership: P3 and P7 share `quest.py` and `voice.py`, so serialize them or give them to one owner.
  - P4 uses a new stage file, not `evaluation.py`, until MC-2b-3 lands.
- **Frontend:**
  - Predict Models to the beats (Opus, 8), including the "what the model reads at the moment of use" drawing.
  - Predict Results to the beats (Opus, 8): how good in mg/dL, where it fails, what drives it across families on shared axes, "Put this in my paper" with TRIPOD.
  - Kit consolidation (Sonnet + Opus verifier, 3).
- **Acceptance:**
  - The NHANES Predict e2e from upload to the bundle in the new design.
  - References: the W1 references, plus P4's error by range against numpy on the held-out rows, the linear SHAP closed form, and the ALE values on the shared grid against an independent hand computation for the linear family.
- **What Nolan can do:** walk his glucose model in the meaning-first design and judge it.

### W3 "Estimate Models, drawn", about 57.5 units

After phase 3, and after phase 4 for the stage files.
- BE-items for Models, about 46.5 (Opus).
- The comparison view kind, refactored from `Comparison.tsx` to props (Opus, 3).
- Estimate Models M1–M7 on live data (Opus, 8).
- **Acceptance:**
  - The sugar journey on NHANES from upload to Fit (lock and SHA).
  - BE3's average day against a pandas mean per column.
  - BE18's projection against a numpy least-squares fit.
  - BE1's settled energy model against `materiality.invariance` and an independent statsmodels identity (equal estimates to 1e-10).
- **What Nolan can do:** the second journey's Models, live.

### W4 "Estimate Results and the paper", about 39.5 units

- BE-items for Results, about 31.5 (Opus).
- R1–R3 on live data (Opus, 8).
- **Acceptance:**
  - Tail split at 126 mg/dL against statsmodels on indicator-split outcomes.
  - Choices refit against fresh fits.
  - Interpretation verdicts by §6's rules on hand-built cases.
  - Paragraphs within 90 words, every number traced to the record.
- **What Nolan can do:** "Put this in my paper" on the sugar journey.

## 6. Decisions for Nolan

1. **What the glucose model is for, at the moment it is used.**
   - Two candidate uses: before any blood test (diet, body measures, age), or at a visit with a fasting draw (HDL, triglycerides).
   - Whether the medication answers, which mark a diagnosis, may be predictors.
   - This shapes the first Predict beat and the predictor set.
2. **Heavy runs, since the machine is beside his bed.**
   - A full tuned 7-family shelf is extrapolated at 1.5–3 h (not measured). Recommendation: first walk on linear, ridge, elastic net and Huber (minutes); add boosted trees, XGBoost and random forest in a run scheduled while he is away.
   - The heavy real-server e2e journeys and recaptures: recommendation is CI's full tier, not his machine.
3. **Deployment.** Recommendation: the local launcher on his Mac for W1 and W2 (local mode allows the typed path and file browser). Server mode later, if coauthors should click it.
4. **His own CSV.** Which real file to walk at the end of W1. It stays local.

**Risks for the orchestrator, not Nolan:**
- Phase 3 must merge before W1-E: quest v5 and the regenerated `generated.ts`.
- MC2-CLEAN may touch `estimand.py`; check before W3.
- Quest line names are engine wording until P7.
- Hover answers use receipt sentences until BE2, which may read long.
- The mock suite has no `/quest` or `/fit`. Shell vitests use real `/quest` JSON captured by W1-C.
- Name clash: the beats' BE1–BE24, SIZING's E1–E5 (designed experiments) and the sprint waves E1a–E1q. Call the beats' items BE1–BE24 in plans.
- HANDOFF's two pending rulings from `design/quest-models-stage` are superseded by FOUNDATION §0. Say so in HANDOFF.
