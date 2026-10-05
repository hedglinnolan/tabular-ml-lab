# TurboTab UX / Feature-Surfacing Investigation — Run 3

**Date:** 2026-08-12
**App under test:** TurboTab "walking skeleton" (Guided door) at `http://127.0.0.1:8777/`
**Dataset (user-dropped, per-session):** `nhanes_1999_2018_yayhoo_fasting_diet_imputed.csv` — 21,849 rows × 29 columns
**Method:** Chrome extension driving a controlled tab; three distinct legitimate paths via "Start over" + re-upload between each. DOM/console/network inspected, not just screenshots.

---

## 0. Executive summary

The user's premise — "many tables render with errors, and I think L59 should have fixed them" — is **largely a misdiagnosis, in two ways**:

1. **L59 was never a "fix broken tables" loop.** Its five accepted parts were narrow: (A) make the *empty Train step say why it is empty* + a retry; (B) fix the *receipt/eligibility N sentence*; (C) the positive-class question (measured and **handed back — NOT built**); (D1) partial-undo; (D2) the duplicated "01" numbering. None of L59 touches the bulk of the app's tables.

2. **On the happy paths I walked, the tables do NOT render with errors.** The Data diagnosis, target picker, opening-sequence cards, seal, the entire Explore stack (profile, missingness cards, impossibility tables, histograms, pull palette, prevalence widget), Features catalogue, Preprocess receipts, and the Report manuscript **all render correctly**. I found exactly **one** literal broken-render (`[object Object]` chip, path 2), plus a set of *logical* inconsistencies (N counts) and *stuck/gated states* — none of which are "tables rendering with errors" in the malformed/NaN/stack-trace sense.

**The real, dominant defect is not a table at all: `GET /project/{id}/models` returns HTTP 500, dataset-wide.** On this specific NHANES file the model shelf cannot load for *either* a regression target *or* a classification target (two different sessions, two different project ids, both 500). Because no model can be fitted, every downstream *results* table (Explain permutation-importance, figures, checklist, Report performance rows) stays in its empty/templated state. **This is what the user is experiencing as "empty/incomplete tables" — a server 500 cascading downstream, not a rendering bug.**

**What L59 demonstrably fixed and is working:** L59-A. When the shelf 500s, the Train step now **names its own failure** ("this step is not empty, it is broken"), prints `HTTP 500 Internal Server Error`, and offers a **retry that genuinely re-fetches** (verified in the network log). In run 2 this was a silent bare heading with zero controls. That regression-of-experience is fixed.

**What is still broken / still open (highlights):**
- **`/models` 500 persists** (the trigger, DRIVE-035, was deliberately not chased in L59/L60). The server returns a bare 21-char `Internal Server Error` with no detail — the client cannot self-diagnose further.
- **L59-B (N consistency) is only half-landed.** On a 0-missing target (path 1) all N sentences agree. On a **heavily-target-missing** target (path 2, `meds_chol`), the eligibility sentence, the seal count, the positive-class finding, and the manuscript disagree — **three different bases (21,849 / 10,645 / 4,645)** and a self-contradictory manuscript sentence. The DRIVE-031 family is still live.
- **`[object Object]` render bug** in the positive-class finding chip (path 2) — a genuine JS stringification defect.
- **Repeat/aggregation dead-end** (path 3): choosing grain=repeat → one-row-per-person → aggregate leads to a **permanently disabled seal** (`class="answer notbuilt"`) with **no control to name the identifier column** the flow says you can name.
- **DRIVE-034** (SETTLED EAR badge on `kcal`) reproduced verbatim — still present (deferred to the not-yet-run L60 Part D).
- Deliberate states re-confirmed (not bugs): SHAP not offered, reverse-coding NOT BUILT, eligibility "Yes"→NOT OFFERED, lens ordering below target (numbering fixed, ordering deferred), Q4–Q7 suppression when grain=one-row.

---

## 1. Tested branch / HEAD / L59 presence

| Item | Value |
|---|---|
| Branch | `TurboTab` |
| HEAD tested | `0dca497` — *"docs(turbotab): the L60 prompt — the event nobody chose, and DRIVE-032 corrected"* |
| **L59 present?** | **YES.** 6 commits in the tree: `3f2f016` (L59 prompt) → `e18ee3f` (A) → `60dfcf5` + `b0764e2` (B) → `6a74d83` (C+D) → `d4a5943` (*"L59 adjudicated: accepted, five of five"*). |

So the L59 fixes **are** in the running build. HEAD is the L60-prompt doc, which sits *after* L59 adjudication; L60's own work (positive-class binding, preparation-mode surface, EAR/DRIVE-034) is **not yet built** at this HEAD.

**Important scope correction for the user:** L59's commit subjects touch `web/index.html`, `api.py`, `project.py`, `eligibility.py`, `engine.py`, `grain.py`, and tests — the Train empty-state, the receipt base, the positive class, and the undo. **No L59 commit is a general table-rendering fix.** If the expectation was "L59 repaired broken tables across the app," that expectation doesn't match what L59 shipped.

---

## 2. Paths walked (all three legitimate for this NHANES dietary+clinical file)

| Path | Target | Task type | Grain | Why chosen |
|---|---|---|---|---|
| **1** | `kcal` (energy) | regression (continuous, 4,276 unique, **0 missing**) | one row per person | Continuous/dietary target; maximises dietary-pack tables incl. the prevalence/EAR widget. Distinct from run 1 (glucose) and run 2 (`meds_hbp`). |
| **2** | `meds_chol` | binary classification (object/text, **17,204 missing**) | one row per person | A *different* clinical target than run 2's `meds_hbp`; heavy target-missingness deliberately stress-tests L59-B; text-binary tests the positive-class question (L60-A / DRIVE-032). |
| **3** | `bp_sys` (systolic BP) | regression (1,957 unique, **0 missing**) | **people repeat** | Fires the **Q4–Q7 repeat/aggregation chain** — tables untouched by paths 1–2. Defensible for pooled NHANES cycles. |

---

## 3. Table-by-table rendering audit

Legend for **L59-should-cover?** — whether L59's five parts were scoped to this surface.

### 3.1 Path 1 — `kcal` regression (sealed; 0 missing)

| # | Table / widget | Rendered? | Notes / evidence | L59 cover? | Status |
|---|---|---|---|---|---|
| 1 | Data receipt ("…loaded: 21849 rows, 29 columns") | ✅ correct | — | no | fine |
| 2 | "9 features need the same repair: read as binary" | ✅ correct | binary_text group (gender, meds_hbp, meds_chol, 6× imputed_*) | no | fine |
| 3 | "9 things stand out in the shape of this file" | ✅ correct | ranked diagnosis, "First rows" affordance | no | fine |
| 4 | Target picker (29 columns w/ dtypes) | ✅ correct | **SEQN listed & selectable as target/predictor, unflagged** | no | fine (but see 5.4) |
| 5 | Opening-sequence cards (lens 01, target 02, purpose 2.5, grain 03, elig 08, seal 09) | ✅ correct | numbering correct; lens renders *below* target (ordering deferred) | D2 (numbering) | **D2 fixed; ordering still deferred** |
| 6 | Explore profile table | ✅ correct | rows 21849, features 28, 25/3 num/cat, missing 5.4%, p/n 0.001, sufficiency "abundant", target "kcal · regression", mean(SD) 2,120.08 (1,003.07), range 0–15,594, skew 1.66 | no | fine |
| 7 | Finding: "2 features with high missingness" (meds_hbp, meds_chol) | ✅ correct | critical | no | fine |
| 8 | Finding: "`kcal` holds impossible values and abnormal ones" [SETTLED] | ✅ correct | impossible-vs-extreme coaching | no | fine |
| 9 | Finding: "Nutrient associations need energy adjustment" [SETTLED] | ✅ correct | dietary lens | no | fine |
| 10 | Finding: "15 features with outliers" + embedded histograms | ✅ correct | BP_SYS/BP_DI/WEIGHT histograms render; "See every distribution" | no | fine |
| 11 | Impossibility tables (IMPOSSIBLE—BP_DI 125 rows; IMPOSSIBLE—KCAL 9 rows; IMPROBABLE 174, paged 1/9) | ✅ correct | red-highlighted cells, "first 12 of N", paging works | no | fine |
| 12 | Per-column missingness decision cards (meds_chol 78.7%, meds_hbp 71.2%) | ✅ correct | informative-missingness Q fires per column | no | fine |
| 13 | Finding: "501 records report an implausible daily intake" [CONVENTION] | ✅ correct | "All 6 shown" (bounded stack, no remainder here) | no | fine |
| 14 | Pull palette (Physiologic plausibility / Missingness by feature / Correlation matrix / Distribution of each feature / **Reverse-coding audit NOT BUILT**) | ✅ correct | one affordance disabled w/ reason (deliberate) | no | fine |
| 15 | **Prevalence-of-inadequacy widget** | ✅ renders, ⚠️ wrong content | Asked kcal → **"Prevalence of inadequacy for kcal is computed by the EAR cut-point method. [SETTLED]"** — DRIVE-034 reproduced | no | **DRIVE-034 still present (L60 Part D, not yet run)** |
| 16 | Features catalogue (log(x), log(x+1), … w/ explainability-cost labels + column dropdowns) | ✅ correct | — | no | fine |
| 17 | Preprocess receipt ("Missingness settled: 0 recorded… 2 columns not answered") | ✅ correct | — | no | fine |
| 18 | **Train shelf** | ❌ **cannot load** | `GET /project/d192e4143e49/models` → **HTTP 500**; card shows **L59-A message** + retry (retry re-fetched → 500 again) | **A** | **L59-A working; underlying 500 unfixed (DRIVE-035, not chased)** |
| 19 | Explain ("Which columns is the model actually using?") | ✅ correct (templated) | "No model has been fitted yet… choose models in Train" | no | correct cascade |
| 20 | "SHAP is not offered here" | ✅ correct | GUIDED-101/232 deliberate | n/a | deliberate |
| 21 | Report read-as-draft manuscript | ✅ correct | Names modeled base consistently (see 4.1); [AUTHOR REQUIRED] gaps render | B | **L59-B holds on 0-missing** |

**Path 1 verdict: zero broken renders.** The only failure is the `/models` 500 (server), surfaced cleanly by L59-A. All tables render.

### 3.2 Path 2 — `meds_chol` binary classification (sealed; 17,204 missing)

| # | Table / widget | Rendered? | Notes / evidence | L59 cover? | Status |
|---|---|---|---|---|---|
| 1 | Data + diagnosis cards | ✅ correct | (now "8 features need the same repair" — meds_chol is the target) | no | fine |
| 2 | Target receipt | ✅ correct | "classification at high confidence… object type (categorical/binary)" | no | fine |
| 3 | **Positive-class finding: "Which of these is the event you are predicting?"** | ✅ **fires**, ❌ **one broken chip** | "'meds_chol'… 'False' (1,001 rows) and 'True' (9,644 rows). 17,204 rows have no outcome recorded." **First chip renders literal `[object Object]`** (`span.chip.arrived`), persists after selecting the Clinical lens | **C (question exists but not wired per L60)** | **RENDER BUG (new/independent of L59)** |
| 4 | Explore profile (classification) | ✅ correct | "1,001 minority-class events for 28 candidate parameters (EPV = 35.8)"; rows 21849, 26/2, missing 2.5%, classes 2 | no | fine |
| 5 | Missingness / impossibility / histograms | ✅ correct | same family as path 1 | no | fine |
| 6 | Eligibility + seal receipts | ✅ render, ⚠️ **inconsistent N** | eligibility "drawn from all 21849"; seal "697 rows (15%)" — see 4.2 | B | **L59-B inconsistency still live** |
| 7 | Report manuscript N sentence | ✅ renders, ⚠️ **self-contradictory** | "Of **4,645** rows with a value for the outcome, 697 sealed… 3,948 for fitting, **with 150 of the held-out rows carrying the outcome**" | B | **inconsistent (see 4.2)** |
| 8 | Train shelf | ❌ cannot load | `GET /project/23bc08387137/models` → **HTTP 500**; L59-A message + retry render | A | **L59-A working; 500 unfixed** |
| 9 | Explain / SHAP | ✅ correct (templated / deliberate) | same as path 1 | no | correct cascade |

**Path 2 verdict: one literal render bug (`[object Object]`)** + logical N-inconsistencies. Everything else renders.

### 3.3 Path 3 — `bp_sys` regression, grain = people repeat (NOT sealed — blocked)

| # | Table / widget | Rendered? | Notes / evidence | L59 cover? | Status |
|---|---|---|---|---|---|
| 1 | Repeat chain Q4 "Are these repeats or different time points?" (04) | ✅ correct | fires on grain=repeat | no | fine |
| 2 | Q5 "When you analyze this, what is one row?" (05) | ✅ correct | One row per person / per record / not described | no | fine |
| 3 | Q6 "How should each person's rows be combined?" (06) — aggregation menu | ✅ renders, ⚠️ **dead-ends** | "Their mean / first / last / change from first"; after choosing → **"There is no identifier column recorded, so there is nothing to combine rows by."** | no | **partial-wired dead-end** |
| 4 | "WHAT ONE ROW IS" receipt | ✅ renders, ⚠️ **contradicts seal card** | "people repeat, and no column identifying the person has been named. Held-out rows are drawn BY ROW until one is… numbers are labeled exploratory until a person column is named, and you can name it at any point before the seal." | no | **no control exists to name it** |
| 5 | **Seal card 09** | ❌ **permanently blocked** | "the rows have not been combined yet…"; **"Draw it now" is `disabled`, `class="answer notbuilt"`, tagged "NOT YET"** | no | **dead-end / notbuilt** |
| 6 | Explore profile (pre-seal) | ✅ correct | renders even unsealed | no | fine |
| 7 | Train card | ⚠️ hidden/gated | bare heading tagged **"stale — an earlier answer changed"**, **zero controls, no L59-A message** (unsealed → `/models` never fetched) | A | **silent-empty variant persists when gated by seal, not by 500** |

**Path 3 verdict: the whole grain=repeat + unit=person + aggregate route is a dead-end** — the seal can never be drawn, so nothing downstream is reachable. This is path-dependent and does not occur on paths 1–2. No `[object Object]` here (`objectObjectCount = 0`).

---

## 4. The N-consistency story (DRIVE-031 / L59-B), path by path

### 4.1 Path 1 (`kcal`, 0 missing) — CONSISTENT ✅
- Eligibility: "all 21849 rows… drawn from all of them."
- Seal: "**3,277** rows (15%) are held out." (3,277 / 21,849 = 15.0%)
- Manuscript: "Of **21,849** rows with a value for the outcome, 3,277 were sealed… and **18,572** were available for fitting." (3,277 + 18,572 = 21,849 ✓)

All three surfaces agree because every row has an outcome. **L59-B's requirement holds here.**

### 4.2 Path 2 (`meds_chol`, 17,204 missing) — INCONSISTENT ❌ (three bases)
- Positive-class finding: **10,645** labeled (1,001 False + 9,644 True), 17,204 missing.
- Eligibility receipt: "all **21,849** rows… **drawn from all of them**."
- Seal: "**697** rows (15%)."  → 697 / 0.15 ≈ **4,647** (matches neither 21,849 nor 10,645).
- Manuscript: "Of **4,645** rows with a value for the outcome, **697** were sealed… **3,948** were available for fitting, **with 150 of the held-out rows carrying the outcome**."

Problems, all simultaneously on screen / in one export:
1. **10,645 ≠ 4,645** — the app reports two different "rows with a value for the outcome" counts on two surfaces.
2. Eligibility "drawn from all 21,849" cannot be reconciled with a 697-row (15%) seal (15% of 21,849 would be 3,277). Same shape as run 2's DRIVE-031.
3. The manuscript sentence is **internally self-contradictory**: it frames 697 as sealed "of 4,645 rows with a value for the outcome," yet says only **150** of those 697 carry the outcome. (150/697 ≈ 4,645/21,849 — i.e. the seal actually drew by-row from the whole population, so the "of 4,645 with-outcome" framing is wrong.)

**Interpretation:** L59-B's *manuscript-names-a-base* half is present (the manuscript does name 4,645), but cross-surface reconciliation is not achieved on a heavily-missing target — the eligibility sentence and the positive-class finding are not brought into agreement with it. **DRIVE-031 is still live in this class of target.** I did not chase the server mechanism; likely tied to `object`-dtype/`binary_text`-vs-numeric target parsing that L60 §00 discusses.

---

## 5. Other findings (classified)

| Finding | Classification | Path | L59? | Detail |
|---|---|---|---|---|
| `/models` → HTTP 500, **dataset-wide** | **broken (server)** | 1 & 2 | trigger not chased | Both a regression and a classification project 500. Body is a bare 21-char `Internal Server Error` (text/plain), no traceback/detail exposed. |
| Train self-diagnosing broken state + working retry | **surfaced-correct (L59-A win)** | 1 & 2 | **A — works** | "…it is broken. HTTP 500 Internal Server Error" + "Try loading the models again" re-fetches. |
| `[object Object]` chip in positive-class finding | **broken (render)** | 2 | no | `span.chip.arrived` stringifies an object; persists regardless of lens; 1 occurrence. |
| N inconsistency (three bases) | **broken (logic)** | 2 | B — partial | See §4.2. |
| Repeat/aggregation dead-end; seal `notbuilt`/disabled | **broken (workflow)** + **not-visible control** | 3 | no | Q6 asks how to combine but no identifier can be named; promised "name a person column" control absent (IMPORT-257 territory). |
| Silent empty Train ("stale", zero controls, no message) when gated by unsealed seal | **broken (variant)** | 3 | A — not covered | L59-A only fires on a fetched-and-500 `/models`; the seal-gated case still shows the run-2-style bare heading. |
| DRIVE-034: SETTLED EAR badge on `kcal` | **broken (content) — deferred** | 1 | no (L60 Part D) | Reproduced verbatim; energy has no EAR in the cut-point sense. Not yet worked. |
| Post-seal "Exclude those rows" buttons stay enabled | **broken (affordance)** | 1 | no | Project sealed, yet `data-plaus-route="exclude_the_rows"` buttons `disabled:false` + editable reason inputs, while the card's own prose says §04 refuses exclusion after the seal. (Not clicked — would attempt an illegal mutation.) |
| SEQN selectable as target/predictor, unflagged | **not-visible guard** | all | no (GUIDED-231 port) | Participant sequence id would be fed to models as a predictor; nothing flags it. Manuscript counts it among "28 candidate predictor parameters." |
| survey-design / weights / pooled-cycle unsurfaced | **not-visible guard** | all | no (GUIDED-231 port) | `cycle_begin_year` present; pooled 1999–2018 cycles never surfaced; no survey-weight prompt. |
| Lens (01) renders **below** target (02) | **wrong-order — deliberate** | all | D2 (numbering only) | Numbering now correct (was both "01" in run 2); ordering half of DRIVE-020 deliberately deferred. |

---

## 6. Deliberate / expected states — re-confirmed (NOT filed as bugs)

- **SHAP "not offered here"** — GUIDED-101 / GUIDED-232. Renders with its two-reason explanation. ✅ present, correct.
- **Reverse-coding audit — NOT BUILT** — pull-palette chip disabled with reason. ✅
- **Eligibility "Yes → which column, and what range?" — NOT OFFERED** — the restrict branch is deliberately not offered; only "No, the study is about everyone here." ✅
- **Q1.5 (table orientation) / Q4–Q7 suppression** — Q4–Q7 correctly suppressed on paths 1–2 (grain=one row per person); correctly fired on path 3 (grain=repeat). ✅
- **Counted remainder / bounded stack (GUIDED-149)** — Explore stack bounded; on this data "All N shown" (no remainder to hide). ✅
- **"Mark the whole column as not trustworthy — not built"** (GUIDED-096) — named on the impossibility card shelf rather than omitted. ✅
- **D2 numbering** — lens 01 / target 02 / purpose 2.5 / grain 03 / … all correct (run-2's dual-"01" is fixed). ✅

---

## 7. Status of run-2 findings (continuity)

| Run-2 finding | Run-3 status |
|---|---|
| **(a)** Train renders as bare heading tagged "stale", zero controls → no model fitted | **Partly changed.** When the project is **sealed** (paths 1–2), L59-A replaces the silent empty state with a self-diagnosing "it is broken / HTTP 500 / retry" — **fixed presentation**. But the underlying cause (shelf can't load) persists because **`/models` still 500s**. And when Train is gated by an **unsealed** seal (path 3), the **original silent bare-heading-tagged-"stale" state still appears** (no `/models` fetch, so L59-A's message never triggers). |
| **(b)** Undo of a bulk group-repair reverts only 1 of N | **Not exercised this run** (I did not apply/undo the binary-repair group, to keep paths clean and avoid the DRIVE-033 confound). No new evidence; not reproduced, not refuted. |
| **(c)** N inconsistency ("all 21,849 rows" vs 15% of labeled subset; target-missing rows dropped) | **Still live on missing-target paths.** Path 1 (0 missing) is consistent; path 2 (`meds_chol`) shows **three disagreeing bases** and a self-contradictory manuscript sentence (§4.2). |
| **(d)** Prereg gaps: SEQN-as-predictor unflagged; survey-design/weights absent; pooled cycles unsurfaced; positive-class "which level is the event" never firing | **SEQN / survey-design / pooled-cycle: still unflagged** (all paths). **Positive-class question: now observed to FIRE for a text binary** (`meds_chol`, path 2) — consistent with L60 §00 (text binary raises it; numeric 0/1 does not; and even when raised it is not binding — I left it unanswered and the seal still proceeded). L60 Part A (make it fire on numeric 0/1 + bind it) is **not built** at this HEAD. |
| **(e)** TRIPOD/PROBAST checklist half-constant | **Not reached** — the checklist lives downstream of a fitted model, which is unreachable (shelf 500 / blocked seal) on all three paths. No new evidence. |

---

## 8. Condition-3 (surfaced / readable / correctly ordered) observations

**Surfaces cleanly (credit):**
- The Explore evidence-on-the-card pattern (histograms embedded under the outlier claim) is genuinely readable; impossibility tables highlight offending cells in red with clear bounds.
- The L59-A Train failure card is exemplary: it distinguishes "broken" from "empty," shows the exact HTTP status, and gives a retry — a model for how a failed step should read.
- Path 3's grain=repeat receipts are honest about the leak risk ("the same person can sit on both sides… numbers are labeled exploratory until a person column is named").

**Unreadable / wrong-order / mislabeled:**
- **`[object Object]`** chip (path 2) — the one literal rendering error; reads as a developer artifact leaking to the user.
- **Contradictory receipts** (path 3): "WHAT ONE ROW IS" says you may seal by-row now (exploratory) while the seal card says you cannot until rows are combined — two on-screen statements that disagree, ending in a disabled button.
- **Lens ordering** (all paths): the lens question (01) still renders *below* the target picker (02). Numbering is fixed; visual order is not (deliberately deferred).
- **DRIVE-034**: a **SETTLED** badge on a method applied outside its domain (EAR for energy) is worse than an unbadged wrong answer — it lends false authority.

---

## 9. Open questions for the team

1. **What makes `/models` 500 on this exact file?** It is dataset-wide (regression *and* classification) but L59 measured `/models` healthy for both shapes on the fixtures. The real user CSV (`nhanes_1999_2018_yayhoo_fasting_diet_imputed.csv`) triggers something the fixtures don't. The server returns a bare `Internal Server Error` with no detail — **needs server logs** (client is blind, by design). This is DRIVE-035, deliberately not chased — but it is the single thing blocking every results table.
2. **Why does `meds_chol` have two "rows-with-outcome" counts (10,645 vs 4,645)?** Almost certainly the `object`-dtype / `binary_text`-vs-numeric target parsing L60 §00 flags. Reconciling these is the substance of a complete L59-B/DRIVE-031 fix.
3. **Should the positive-class finding's lens/source chip be a string?** The `[object Object]` suggests a `Claim`/lens object reaches the chip renderer un-serialized. One-line fix, but user-facing.
4. **Repeat path with no identifier:** should the flow (a) offer a "name the person column" control (the receipts promise one), (b) refuse the aggregation menu until an id is named, or (c) allow the by-row exploratory seal the receipt describes (currently `notbuilt`)? Right now it dead-ends with a disabled seal.
5. **Post-seal exclusion controls** remain enabled after sealing despite the §04 refusal text — is the refusal enforced server-side on click, or only stated in prose?

---

### Appendix — key evidence captured
- `git log` confirms L59 (6 commits) present; HEAD `0dca497`.
- Network: `GET /project/d192e4143e49/models` → 500 (path 1); `GET /project/23bc08387137/models` → 500 (path 2). 500 body = 21 chars, `text/plain`, no traceback.
- DOM: positive-class chip = `<span class="chip arrived">[object Object]</span>` (path 2, 1 occurrence; 0 on paths 1 & 3).
- DOM: path-3 seal button `disabled: true`, `class="answer notbuilt"`; no identifier-naming control present.
- Manuscript strings (path 2) extracted verbatim via page text; N figures in §4.2.
