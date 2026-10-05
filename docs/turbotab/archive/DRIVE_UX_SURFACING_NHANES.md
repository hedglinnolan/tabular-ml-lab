# TurboTab — UX / Feature-Surfacing Investigation

**Build under test:** `Tabular ML Lab — TurboTab walking skeleton` at `http://127.0.0.1:8777/` (badge reads *"walking skeleton · your data"*).
**Branch:** `TurboTab` (confirmed via `git branch --show-current`).
**Dataset:** `nhanes_1999_2018_yayhoo_fasting_diet_imputed.csv` — 21,849 rows × 29 columns (NHANES fasting/dietary, 9 pooled cycles, 6 `imputed_*` boolean flags).
**Path driven:** the Guided door, dietary/clinical (NHANES) flow, target = `glucose` (regression), purpose = prediction.
**Instrument:** a human-equivalent driver in the Claude-in-Chrome extension. This report is the Condition-3 instrument the project lacks — it records what was actually *visible on screen*, in what order, and whether shipped capability reached the user.

---

## 1 · Executive summary

The front of the Guided funnel is strong and, in places, exemplary. Upload, the lens question, target selection with task-type detection, the prediction-vs-inference purpose question, and the entire Explore findings surface all render cleanly, with accurate domain content, evidence badges, confidence markers, per-column chips, distribution histograms, and honestly-disclosed "not built" routes. The domain packs (dietary + clinical) fire correctly and the copy is close to the COPY_DECK.

The investigation surfaced **one structural blocker and one ordering defect that dominate everything else**:

1. **The seal is never drawn, and that stalls the whole back half of the app.** The Guided interview asks the lens, target, task-type, and purpose — then jumps straight to Explore. **The grain question (Q3), the eligibility question (Q8), and the seal itself are never presented as answerable steps.** Because the model shelf is explicitly gated on the seal ("models are chosen after the held-out set is sealed"), the **Train step renders only a heading with no model selection and no fit control**, so **no model can be fitted through this happy path**. Everything downstream — permutation importance, ROC/figures, the checklist, the manuscript's performance rows — therefore stays in an empty/gated state. This matches the register's own admission (`cross-profile-row-scope`, `target-lockbox-settings`) that "the Guided interview does not ask for a seal at all yet," but the driven consequence — the app cannot complete a modeling run — is larger than the register's line conveys.

2. **The opening sequence is out of order versus `OPENING_SEQUENCE.md`.** The lens (Q1) is specified to come **first, before the structural diagnosis** (because diagnosis is field-sensitive). In the running app the diagnosis runs on upload with no lens set, and the lens question is placed **below the target picker inside the Target step**. Both the lens card and the target card are labeled **"01"** (duplicate step number).

3. **A visible dead-end control.** The Explore impossible-value cards offer "Exclude those rows from the study" — with a reason text-box and an active-looking button — but it is gated on two preconditions the app itself prints ("Needs the grain question answered", "Needs the held-out set not yet sealed"), neither of which is answerable anywhere in the interview. The control looks operable but cannot complete.

Net: **Correct** is largely satisfied through Explore. **Surfaced** is satisfied for the reachable steps and is genuinely beautiful in the Explore layer. But the happy path **cannot reach the seal, a fitted model, Explain numbers, figures, or a filled report** in this build.

---

## 2 · Method & what this was anchored to

I confirmed the working tree (`TurboTab`) and built the expectation set from the anchor docs **before** driving, in this order:

- `docs/turbotab/FEATURE_REGISTER.md` — the 182-row register; `state` field (guided-only 63, both 49, guided-native 17, core 7, classic-only 46). classic-only items were treated as deliberately-out, never as bugs.
- `docs/turbotab/COPY_DECK.md` — every Guided string by step/state; the primary anchor for "what string should appear and when."
- `docs/turbotab/OPENING_SEQUENCE.md` — the nine-question opening up to the seal, with firing conditions and fixtures.
- `docs/turbotab/PRODUCT_VISION.md` §06b (the "correct · surfaced · beautiful" ruling and the explicit statement that the page harness "cannot prove visibility").
- `docs/turbotab/DRIVE_PREREG_NHANES.md` — the adjudicator's four withheld predictions for this exact file (used to check whether the four gaps reproduce).

I then drove the app end-to-end in Chrome, taking screenshots at each meaningful step and transcribing on-screen copy. Each observation below is classified as **surfaced-correct / partially-wired / not-visible / deliberately-out (with ID) / unreadable / wrong-order**.

*(One process note for the PM: the file could not be uploaded programmatically — the extension's upload allowlist accepts only the original session attachment, which carried multiple hard links that its safety check refuses; copies in other folders were filtered out. The user performed the single upload by hand; every step after that was driven automatically.)*

---

## 3 · Happy-path walkthrough (step by step)

Screenshots were captured at each step during the session. On-screen copy is quoted where it matters.

### 3.1 Landing (Data, empty)
Clean landing. Left nav shows the full journey (Data · Target · Explore · Features · Preprocess · Train · Explain · Report) plus **Coach Ledger** and **Read as Draft** side panels. Card: *"Bring your table. One CSV or TSV. It is read into memory, diagnosed, and never written to disk…"* — **surfaced-correct**. (Minor: the register/COPY_DECK "No project yet" empty string is *"Drop a CSV to begin."*; the card uses richer copy. Cosmetic.)

### 3.2 Data (post-upload)
Receipt: *"nhanes_1999_2018_yayhoo_fasting_diet_imputed.csv was loaded: 21849 rows, 29 columns."* + `change`. Two "noticed" findings render immediately:
- **"9 features need the same repair: read as binary"** — imputed_bmi, imputed_bp_di, imputed_bp_sys, imputed_height, imputed_waist, imputed_weight, … `[Show me what changes] [Decide later]` (bulk-repair / `audit-binary-text`). *Note:* the six `imputed_*` columns are `bool` dtype, not text, yet the message says "read as binary … written as text" — a wording mismatch (see prereg #4).
- **"9 things stand out in the shape of this file."** *"Ranked by how much they affect what you do next. Nothing has been applied — the import doctor reports and proposes…"* with a `First rows` affordance.

**Wrong-order finding (see §6).** This diagnosis ran on upload with **no lens set**, which contradicts `OPENING_SEQUENCE.md`'s load-bearing rule that the lens is set first.

### 3.3 Target step
Rendered order within the step:
1. **"01 What are you predicting?"** — all 29 columns as chips with dtype. `SEQN` appears as a selectable target chip, **unmarked** (prereg #1). I selected `glucose`.
2. Task-type receipt: *"glucose is the target. The engine reads it as regression at high confidence. Target is continuous numeric (1154 unique values) - regression"* + `change` — **surfaced-correct** (`target-task-detection`).
3. **"01 What kind of measurements are in this table?"** — the **lens** (multi-select). Rich per-option "Why we ask" tooltips. I picked **Dietary intake + Clinical measurements and labs** (the NHANES intersection). Commit button counts up: *"Pick at least one" → "Record this one" → "Record these 2."* Receipt (`MEASUREMENTS`): *"The measurements were described as dietary intake and clinical measurements and labs. Domain conventions for these fields informed the defaults offered below; each is stated with its reasoning and was open to being overridden."* — **surfaced-correct**.
4. Rendered skip: *"NOT ASKED — SETTLED FROM THE FILE: Is glucose a regression problem? … Stated in the transcript rather than asked — change it if it is wrong."* `[Ask me anyway]` — **surfaced-correct** (task-type override as a skip, DESIGN_LANGUAGE §09).
5. **"2.5 What is this model for?"** — prediction vs inference. Options: *"Predicting an outcome for a new person"* / *"Estimating how strongly something is associated with the outcome."* I picked prediction. Receipt (`BUILT FOR`): *"The model was built to predict the outcome for a new individual; handling was chosen to maximize predictive accuracy at deployment rather than to keep any single coefficient unbiased."* — **surfaced-correct** (`model-purpose`).

**Numbering defect:** the lens card and the target card are **both labeled "01"**; purpose is "2.5". There is no Q3/Q8 in between.

### 3.4 The gap where grain, eligibility, and the seal should be
After purpose (2.5) the interview goes **straight to Explore**. There is **no grain question** (Q3 "Can one person appear in more than one row?"), **no eligibility question** (Q8), and **no seal** anywhere between Target and Explore. Selecting the target also lit **Explore in the nav before purpose was even answered** — Explore is not gated on the opening sequence completing. (See §4/§6 for classification and consequence.)

### 3.5 Explore — the strong layer
- Profile card **"01 What the profile says about your data."** *"Large sample (n=21,849). All model types are viable. Low dimensionality (p/n=0.00). Feature space is manageable."* Table: rows 21849, **features 28** (so `SEQN` is counted as a predictor — prereg #1), numeric/categorical 25/3, missing overall 5.4%, p/n 0.001, data sufficiency abundant, target glucose · regression, mean(SD) 107.56 (35.58). **surfaced-correct.**
- **Physiologic-flags** card ("2 physiologic flags", caution) — bp_di, kcal, bp_sys, weight, height (5 of 9 columns), with `[Show me what this means] [Decide at Preprocess] [Dismiss] [⚑ Mark for manuscript]`.
- **IMPOSSIBLE — BP_DI**: 125 entries outside 15–220 mmHg, table of first 12/125 rows, `[Set these entries to missing] [Keep as is] [Exclude those rows from the study]`. **IMPOSSIBLE — KCAL**: 9 entries outside 100–30,000 kcal.
- **Clinical-lens implausible-values** card distinguishing impossible-vs-abnormal-but-real: *"…different from the 1,369 values outside 800–4500 kcal, which are abnormal but real and must be kept: excluding abnormal-but-possible values would remove the sickest patients and bias the model toward the healthy."* (clinical lens, warning, confidence: high).
- **"Nutrient associations need energy adjustment"** with a **SETTLED** evidence badge (dietary lens, confidence: high) — *"`kcal` is total energy, and 19 other numeric columns are candidate nutrients."* **surfaced-correct** (dietary pack + `evidence-badge`).
- **"15 features with outliers"** — accurate long-form prose on the 1.5×IQR fence vs plausibility bounds, plus **per-feature distribution histograms** (bp_sys skew=1.12 median 121.776 mean 124.019; bp_di skew=−0.79; …). **surfaced-correct.**
- **IMPROBABLE VALUES** — 9 features, paginated `‹ 1/9 ›`, bp_sys 174 entries outside 90–200, advisory only.

This layer is where the product's thesis is visibly working: ranked, domain-aware, honest, and readable.

### 3.6 Features
- Deferred/stateful transform catalogue: *"Encode categories by how common they are"* (MEDIUM EXPLAINABILITY COST), *"Center and scale"* (LOW), *"Principal components"* (HIGH) — each with a `column…` dropdown, bin/component inputs, and correct training-fold-leak rationale. **surfaced-correct.**
- Selection question **"02 Should the models be given every column, or a chosen subset?"** with `use every column` dropdown, `Rank them for me`, `Use every column`. I chose every column and settled. Receipt: *"Feature work settled: 0 column(s) added now, 0 transform(s) recorded for fitting inside the training folds"* + *"This step is settled. Adding a feature now would make everything downstream stale."* On settle, **Explore's nav dot changed to an amber ring** (deferred-findings indicator) — a nice status affordance.

### 3.7 Preprocess
- Intro: *"PREPROCESSING, PER MODEL: The table below resolves once models are chosen, and models are chosen after the held-out set is sealed. Nothing is missing — the step that fills it has not happened."* (This is the app naming the seal as the gate.)
- **"01 What do the blanks in your table mean?"** asks only the two high-null **categorical** columns: `meds_chol` (78.7% missing) and `meds_hbp` (71.2%), each Yes/No/"I'm not sure". The same question also appears as a **dimmed, inactive Explore-stack duplicate** ("Is the missingness in `meds_chol` informative?"). I settled preprocessing; receipts: *"This step is settled. Everything recorded here is fitted inside the training folds when the models are trained."* + *"Missingness settled: 0 recorded… 2 column(s) with missing values have not been answered yet…"* **surfaced-correct.**

### 3.8 Train — blocked
The Train region contains **only**: the label "Train", state "open", a veil *"stale — an earlier answer changed"*, and the heading **"Which models should be fitted?"** — **no model shelf, no checkboxes, no fit button** (confirmed via DOM read). The shelf is gated on the seal, which is never drawn, so **no model can be fitted**. This is the terminal functional state of the happy path.

### 3.9 Explain — gated, SHAP deliberately out
- **"01 Which columns is the model actually using?"** (permutation importance) — gated: *"No model has been fitted yet. Permutation importance is a drop in a metric when a column is shuffled, so there has to be a metric first — choose models in Train."*
- **"SHAP is not offered here"** card with full rationale: *"…`shap` is a production dependency and is deliberately absent from the development environment because it pulls numba and llvmlite… the four research packs contain no explainability content at all… (`GUIDED-101`). Classic offers SHAP and states its source…"* — **deliberately-out, surfaced well** (`GUIDED-232`/`GUIDED-101`). Credit: the absence is explained, not silent.

### 3.10 Report ≈ Read-as-draft
The "Report" nav anchors to the **read-as-draft** methods manuscript (TRIPOD/PROBAST scaffold): rows for Data preparation, Model-building procedure, *"Performance reported as discrimination AND calibration AND clinical utility"* [APP], Model presentation [APP], Fairness and subgroup evaluation, Open-science items [USER], Limitations/intended-use [USER+APP TEMPLATE], with a PROBAST (Wolff et al., 2019) reference. Almost every app-filled row reads *"not filled by the app yet … filled when a model has been fitted / when a run has scored the held-out rows."* **surfaced** (`cross-read-as-draft`) but **empty**, because no model was fitted.

---

## 4 · Findings table

| # | Step | Observation | On-screen copy (excerpt) | Classification |
|---|---|---|---|---|
| F1 | Target/opening | Grain (Q3), eligibility (Q8), and the **seal** are never presented as answerable steps; interview jumps purpose → Explore | *(absent)* | **not-visible** (register-known: `cross-profile-row-scope`, `target-lockbox-settings`) |
| F2 | Train | Model shelf never renders (gated on the unmade seal); Train shows only a heading + "stale" veil ⇒ **no model can be fitted** | "Which models should be fitted?" (no controls) | **partially-wired / blocked** |
| F3 | Data/Target | Lens (Q1) rendered **after** the target picker and **after** diagnosis already ran; both cards labeled "01" | duplicate "01" | **wrong-order** |
| F4 | Explore | "Exclude those rows from the study" renders a reason box + button but is gated on grain+seal, which are unanswerable ⇒ dead-end | "Needs the grain question answered… Needs the held-out set not yet sealed" | **partially-wired (dead-end)** |
| F5 | Explain | Permutation importance present but gated (no model upstream) | "No model has been fitted yet…" | **partially-wired (blocked by F2)** |
| F6 | Report | Read-as-draft scaffold renders but is unfilled (no model/run) | "not filled by the app yet" | **partially-wired (blocked by F2)** |
| F7 | Data | Binary-repair message says "written as text" for 6 columns that are `bool` dtype | "read the same way… read as binary" | **copy defect (minor)** |
| S1 | Target | Lens multi-select, per-option tooltips, counting commit button, methods receipt | "Record these 2" / MEASUREMENTS receipt | **surfaced-correct** |
| S2 | Target | Task-type detection + rendered "settled from the file" skip | "regression at high confidence" | **surfaced-correct** |
| S3 | Target | Purpose 2.5 (prediction vs inference) + BUILT FOR receipt | "built to predict the outcome for a new individual" | **surfaced-correct** |
| S4 | Explore | Profile card, physiologic flags, impossible/improbable cards with row tables, distributions | (see §3.5) | **surfaced-correct** |
| S5 | Explore | Dietary pack: energy-adjustment (SETTLED badge) + implausible-intake; clinical pack: impossible-vs-abnormal split | "Nutrient associations need energy adjustment [SETTLED]" | **surfaced-correct** |
| S6 | Features/Preprocess | Explainability-cost tags, training-fold leak rationale, settle receipts, amber "stale" ring on Explore | "0 transform(s) recorded for fitting inside the training folds" | **surfaced-correct** |
| S7 | Explain | "SHAP is not offered here" with cited rationale | "…(`GUIDED-101`)…" | **deliberately-out (surfaced well)** |

---

## 5 · Known / expected states — confirmed, not missed

These were accounted for against the register/prereg and are **not** filed as defects:

- **Seal not asked in Guided yet** — `cross-profile-row-scope` ("the Guided interview does not ask for a seal at all yet") and `target-lockbox-settings` (classic-only). Confirmed. Its downstream consequence (F2) is the headline.
- **SHAP/sensitivity absent from Explain** — deliberate (`GUIDED-232`, `GUIDED-101`); surfaced as an explicit "SHAP is not offered here" card. Confirmed.
- **"Mark the whole column as not trustworthy — not built"** (`GUIDED-096`) — named on the shelf rather than hidden, on both impossible-value cards. Confirmed.
- **Explore stack bound of five / no "N more" affordance on this file** — `GUIDED-149`; prereg says this file pushes 5, collapses 0, so no collapse affordance renders. Confirmed (no "N more" affordance appeared).
- **Q1.5 orientation correctly does not fire** — sample-major table, not a transposed assay. Confirmed.
- **Copy lives at raise sites** — `GUIDED-013`; not separately re-verified, but consistent with what was seen.
- **Prereg's four NHANES gaps** — all reproduced: (1) `SEQN` not recognized as an identifier, offered as a target chip and counted among the 28 predictors; (2) survey-design absence (no `WTMEC2YR`/`SDMVSTRA`/`SDMVPSU`) produces **no finding**; (3) nine pooled cycles (`cycle_begin_year`) produce **no finding** and it is offered as an ordinary predictor; (4) the six `imputed_*` flags are read as ordinary binaries and not connected to their base columns.

Items in the brief's "known deliberate states" list that involve a **fitted model** — FIG_DRAW (only `roc`/`item_correlations` draw), ROC single-model overlay (`GUIDED-236`), inference asked-but-unwired (`GUIDED-231`), checklist reads a constant (`GUIDED-238`), fourth-series hue (`DRIVE-016`), nn/torch cannot be fitted (`TEST-038`) — **could not be reached or observed on this build**, because the seal→model-fit gate (F2) never opens. They are neither confirmed nor refuted here (see Open questions).

---

## 6 · Condition-3 findings (unreadable / unnoticed / wrong-order) — called out

The instrument the project explicitly lacks. On this drive:

**Wrong-order (the significant one).**
- The **lens (Q1) is surfaced after diagnosis and after the target picker**, inside the Target step, contradicting `OPENING_SEQUENCE.md` §01 ("the lens is first, before diagnosis, because diagnosis is field-sensitive"). The structural diagnosis (9 binary-repair + 9 shape findings) is computed on upload with no lens in effect. On this clinical/dietary table the mis-order is benign, but it defeats the very rationale the doc gives for the ordering.
- **Duplicate "01" numbering:** the lens card and the target card both display "01". Purpose is "2.5". A reader cannot use the numbers to orient.

**Not-visible / unnoticed.**
- **Grain (Q3), eligibility (Q8), and the seal** are entirely absent from the interview. A user is never asked whether a person can appear in more than one row, never asked about study eligibility, and never draws a held-out set. The app *references* all three (in gating copy) but provides no card to answer them.
- **A whole capability behind the fold:** the later steps (Features, Preprocess, Train, Explain, Report) live far down a single long scroll; the nav dots for un-settled steps do not obviously advertise that real, interactive content exists below. During the drive it initially read as if the app ended at Explore + the draft. The content is there, but the affordance to reach it is weak.

**Dead-end wiring (looks operable, isn't).**
- The Explore "Exclude those rows from the study" control renders a reason text-box and an enabled-looking button, but cannot complete because its two printed preconditions (grain answered, seal drawn) are unreachable. A user can type a reason and press it to no effect.

**Unreadable.** Nothing was clipped, contrast-broken, or truncated on the reachable steps at 1440-wide. The Explore cards are long but legible; tooltips are dense but readable. No unreadable defects observed.

**Credit where due.** The Explore layer, the lens/purpose receipts, the evidence badges, the explainability-cost tags, the honest "not built"/"not offered here" disclosures, and the amber "stale" ring are all surfaced cleanly and are genuinely well-designed. The *"correct + surfaced + beautiful"* bar is met for the reachable interview.

---

## 7 · Open questions for the team

1. **Is the missing seal expected in this exact build, or a regression?** The register says the seal isn't in the Guided interview "yet," but the brief asked me to walk "to the seal." Confirm whether a seal step is intended to be live at `:8777` today. As it stands, the Guided happy path **cannot produce a fitted model or a completed report**.
2. **How is the seal meant to be drawn?** The copy implies it precedes model selection ("models are chosen after the held-out set is sealed"), but no UI draws it. What action is supposed to raise the barrier — a dedicated card after eligibility, or something folded into Train?
3. **Grain/eligibility surfacing.** `target-grain-question` is marked "Built at L13" in the register, yet no grain card renders. Is it wired but hidden, or not mounted in this page?
4. **The reachability of Train/Explain/figures.** Because F2 blocks model fitting, none of the model-dependent known-states (ROC/FIG_DRAW/checklist/nn-torch/DRIVE-016) could be exercised. If those are expected to be demoable, the seal→fit gate needs to open first.
5. **Duplicate "01" numbering and lens placement** — intended, or an artifact of the lens being retrofitted into the Target step?
6. **The dead-end "Exclude those rows" control** — should it be hidden/disabled until grain+seal exist, rather than rendering an operable-looking button and reason box?
7. **Binary-repair wording** — "written as text" is inaccurate for the six `bool` `imputed_*` columns; worth a copy fix and, per prereg #4, worth considering whether `imputed_*` should be linked to their base columns.

---

*Prepared from a live drive of the TurboTab walking skeleton on the NHANES fasting/diet dataset. Screenshots were captured at each step during the session. On-screen copy quoted above was transcribed directly from the running app.*
