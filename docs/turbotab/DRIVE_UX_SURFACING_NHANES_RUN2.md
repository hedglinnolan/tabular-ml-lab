# TurboTab — UX / Feature-Surfacing Report (Run 2)

**Date:** 2026-08-12
**Driver:** Automated UX walkthrough via Claude-in-Chrome, against the live app at `http://127.0.0.1:8777/`
**Dataset:** `nhanes_1999_2018_yayhoo_fasting_diet_imputed.csv` — 21,849 rows × 29 columns (NHANES 1999–2018 fasting/dietary), Guided door.
**Target chosen for the drive:** `meds_hbp` (binary clinical outcome → classification).

---

## 1. Executive summary

The pull moved the needle on the two biggest run‑1 blockers at the front of the flow, but the **single most important defect survives**: **you still cannot fit a model through the Guided door.**

- **Fixed (front half):** The **seal now draws** — "Draw it now" is a real, working control, and it produced `THE HELD-OUT SET · SEALED — 945 rows (15%) are held out…`. The **grain question (Q3, "Can one person appear in more than one row?") now fires**, as do purpose (Q2.5) and eligibility (Q8) in the right order. This is genuine progress and directly reflects the L58 seal‑path and DRIVE‑017 grain‑card commits.
- **Still broken (back half):** Drawing the seal did **not** unlock a working Train step. The `Train` section renders as a bare heading — **"Which models should be fitted?"** — permanently tagged **"stale — an earlier answer changed,"** with **zero interactive controls** (verified in the DOM: `controlCount: 0`). No model shelf, no fit button. Consequently Explain says *"No model has been fitted yet,"* the figure layer never draws model outputs, and the TRIPOD checklist / methods draft stay templated. The entire back half remains unreachable — the same outcome as run 1, now for a different proximate reason (run 1: the seal never drew; run 2: the seal draws but the Train card is empty and stuck stale).
- **Newly observed bugs:** (a) **Undo scope** — undoing a 9‑feature bulk "read as binary" apply reverted only `gender` (1 of 9); (b) **N inconsistency** — the seal/study‑population receipts claim "all 21,849 rows," but the held‑out set is actually 15% of the **6,297** rows that have a non‑missing target (15,552 target‑missing rows silently dropped; only the READ AS DRAFT paragraph discloses this); (c) an **aggressive staleness cascade** with no recompute affordance.
- **Condition 3 (surfaced/beautiful):** The flow is visually clean and readable. The headline surfacing defects are **wrong‑order** (the lens Q1 renders *below* the target picker and *after* the diagnosis, both cards mislabeled "01"), a **misleading study‑population receipt** (N), and a **dead‑end stale Train card**.

**Bottom line:** A user driving this build gets a beautiful, honest front half and then hits a wall at Train. Fixing the empty/stale Train card is the unlock for everything the brief calls "the back half."

---

## 2. Tested commit / branch state

- **Branch:** `TurboTab`
- **HEAD:** `b4b8246` — *"TEST-078: both documented fast-tier durations were wrong, in opposite directions"*
- Recent history (relevant): `ff340e0` *"L58-C: the seal gets a control, and the gate is the server's"*, `72e1b34` *"L58-A+B: the door opens, and the guard that would have caught it"*, `d66ce3b`/`bbba335` *DRIVE-017 — "a human cannot fit a model through the Guided door, and the grain card is built and did not fire."*
- Working tree clean (`git status` empty).

The history confirms the pull specifically touched the seal control, the door/gate, and the grain card — exactly the areas run 1 flagged. My live findings are consistent with those commits landing at the front and **not** closing the Train gap DRIVE‑017 names.

**Register counts (current `FEATURE_REGISTER.md` header):** `both` 49 · `core` 7 · `classic-only` 46 · `guided-only` 64. No naive‑reading expectation shift from these vs. the brief; the state semantics (guided‑only/both/guided‑native/core reachable in Guided; classic‑only deliberately absent) are unchanged.

---

## 3. Method & anchors

Expectations were re‑derived fresh from the pulled docs before driving:

- **`OPENING_SEQUENCE.md`** — for a cross‑sectional NHANES table the expected asked sequence is **Q1 lens → (diagnosis) → Q2 target → Q2.5 purpose → Q3 grain → Q8 eligibility → SEAL**, with **Q1.5 (orientation) and Q4–Q7 (repeats chain) suppressed**. §02 pegs a cross‑sectional CSV at ~4 asked.
- **`COPY_DECK.md`** — seal basis copy, grain/eligibility receipts, contradiction exits.
- **`FEATURE_REGISTER.md`** — `data-lens-question`, `bulk-repair`, `target-grain-question`, `target-positive-class`, `model-purpose`, `nutrition-pack-content`, `evidence-badge`, `table-orientation`.
- Framing from `PRODUCT_VISION.md` §06b (correct / surfaced / beautiful) and `LOOP.md` §05.

Driving was done end‑to‑end (upload state confirmed → opening sequence → every Guided step → seal → Explore/Features/Preprocess → Train → Explain → Report/checklist), clicking through branches and using DOM inspection to verify rendered vs. present. Screenshots saved to disk (paths in §9).

**Note on the app:** the page **virtualizes by viewport**, so `read_page`/`get_page_text` return only what is near the scroll position. Several conclusions below (especially the empty Train card) were confirmed by direct DOM queries rather than screenshots alone.

---

## 4. What changed vs. Run 1

### 4a. Fixed / improved

| # | Run‑1 finding | Run‑2 status | Evidence |
|---|---|---|---|
| 1 | **Seal was never drawn** | **FIXED** | "Draw it now" is active once target+grain+eligibility are answered; clicking it produced `THE HELD-OUT SET · SEALED — 945 rows (15%) are held out and will not be looked at again until the models are scored.` (matches COPY_DECK `cross_sectional` basis). |
| 2 | **Grain (Q3) skipped** | **FIXED** | `03 · Can one person appear in more than one row?` now fires with all four options + "Show me what each answer does." Answering "one row per person" wrote the COPY_DECK receipt verbatim. |
| 3 | Purpose / eligibility not reached | **FIXED** | `2.5 · What is this model for?` (prediction/inference) and `08 · Is your study restricted to part of this data?` both fire; each writes a methods receipt ("BUILT FOR…", "STUDY POPULATION…"). |
| 4 | Q4–Q7 handling | **Correct** | Repeats chain correctly suppressed for a one‑row‑per‑person table. |

### 4b. Still broken / unchanged

| # | Finding | Run‑2 status | Evidence |
|---|---|---|---|
| 1 | **No model can be fitted (Train)** | **STILL BROKEN** | `Train` section = `open · stale — an earlier answer changed · 01 Which models should be fitted?` with **0 controls** (DOM `controlCount:0`; body is a lone `card-train` div containing only a code comment). Explain: *"No model has been fitted yet … choose models in Train."* Persisted after settling Features **and** Preprocess. |
| 2 | **Lens (Q1) below target, both "01"; diagnosis before lens** | **STILL PRESENT** | The lens card `What kind of measurements are in this table?` renders *after/below* the `What are you predicting?` picker, both labeled `01`. The binary‑repair + shape diagnosis ran on upload, before any lens. |
| 3 | **SEQN usable as predictor** | **STILL PRESENT** | `SEQN` sits in the target/feature pickers with no identifier/leakage flag anywhere (DOM: `seqn_flagged:false`). |
| 4 | **Survey‑design absence unflagged** | **STILL PRESENT** | No survey‑weight / design warning anywhere (DOM: no `WTDR*`/`SDMV*`/"survey weight"/"survey design" text), despite a dietary‑NHANES lens and a file with no weight columns. |
| 5 | **Pooled cycles unnoticed** | **STILL PRESENT (nuanced)** | `cycle_begin_year` is never flagged as a pooling dimension; grain "one row per person" was accepted with **no contradiction fired** (arguably correct if SEQN is unique per row, but the pooled‑cycle structure is never surfaced). |
| 6 | **`imputed_*` read as plain binaries** | **PARTIALLY ADDRESSED** | Now surfaced as a bulk "read as binary" repair proposal (good). But they remain ordinary predictors — nothing flags that these are missingness indicators that could leak; and the app converts them `bool→0/1` without commentary on their role. |
| 7 | **Half the checklist items read a constant (GUIDED‑238)** | **STILL PARTIALLY PRESENT** | TRIPOD checklist: 12 items, **6 distinct auto‑filled values**; the rest repeat placeholders ("not filled by the app yet" / "No … decision recorded yet"). Confounded by the unfitted‑model state, but the constant‑reading pattern is visible. |

### 4c. Newly broken / newly observed

1. **Undo reverts only one column of a bulk apply.** After "Apply to the 9 selected" on the "read as binary" group, **"Undo the last change" reverted only `gender`**; `meds_hbp`, `meds_chol`, and all six `imputed_*` stayed `Int64` (verified in the target picker dtypes). The apply tooltip promised *"records one decision covering the whole group. It can be undone"* — undo does not honor that scope.
2. **Study‑population / seal N is internally inconsistent.** `STUDY POPULATION — No eligibility restriction: all 21849 rows are in the study population, and the held-out set is drawn from all of them.` But the seal drew **945 rows (15%)** — 15% of the **6,297** rows with a non‑missing `meds_hbp` (15,552 are missing). Only the READ AS DRAFT methods paragraph reconciles it: *"Of 6,297 rows with a value for the outcome, 945 were sealed … and 5,352 were available for fitting."* The plain receipts a user actually reads (`STUDY POPULATION`, the seal line) never disclose the 15,552 dropped rows and imply 945 is 15% of 21,849.
3. **Aggressive staleness cascade with no recovery.** Settling Preprocess and Features marked **both Explore and Train** `stale — an earlier answer changed`. The stale Train card offers **no recompute/refresh/fit control** — it is a dead end. (This is the proximate mechanism of the Train blocker.)

---

## 5. Full happy‑path walkthrough

**Landing / upload (confirmed).** After the user loaded the CSV into the automation tab, the app showed the post‑upload state: `DATA · recorded — nhanes_1999_2018_yayhoo_fasting_diet_imputed.csv was loaded: 21849 rows, 29 columns` plus two `noticed` cards and the Target step open. *(screenshot: ss_5783 / post‑upload)*

**Diagnosis — bulk "read as binary" (surfaced‑correct).** Card: *"9 features need the same repair: read as binary."* Clicking **"Show me what changes"** rendered a clean **BEFORE/AFTER diff** on `imputed_bmi` (False→0, True→1) plus a **"WHICH FEATURES? 9 of 9 selected"** multi‑select (the 6 `imputed_*`, `meds_chol`, `meds_hbp`, `gender`), a plain‑language receipt, and **"Apply to the 9 selected" / "Leave all 9 as they are."** Applying flipped the columns to `Int64` and produced an **"Undo the last change"** link. *This is `bulk-repair` + `audit-preview` + `audit-undo`, and the preview/apply half is excellent.* **Bug:** undo then reverted only `gender` (see §4c‑1).

**Ranked shape findings.** A second `noticed` card, *"9 things stand out in the shape of this file,"* is the same 9 binary items surfaced as ranked findings; after the bulk apply it collapsed to *"Nothing structural stood out."* The individual findings render with full detail, tags (`warning`, `confidence: …`, columns), and **"Show me what this means / Decide at … / Dismiss / ⚑ Mark for manuscript"** actions.

**Lens (Q1) — recorded, but out of place.** Selecting **Dietary intake + Clinical measurements and labs** enabled **"Record these 2"** and wrote a `MEASUREMENTS` methods sentence. But this card sits **below** the target picker and both are labeled **"01,"** and it runs **after** the on‑upload diagnosis — so the lens cannot influence the diagnosis (only later Explore findings). *(Wrong‑order — see §7.)*

**Target (Q2).** Selecting `meds_hbp` produced *"the engine reads it as **classification** at **high** confidence. Target is binary (0/1)."* Tooltip: `Int64 · 2 unique · 15552 missing`. **The positive‑class "which level is the event" question never appeared** (DOM confirmed) — see §6.

**Purpose (Q2.5).** *"What is this model for?"* → **"Predicting an outcome for a new person"** wrote `BUILT FOR — built to predict the outcome for a new individual; handling was chosen to maximize predictive accuracy…` — and downstream the missing‑indicator route treats indicators as legitimate (correct under a prediction objective).

**Grain (Q3).** *"Can one person appear in more than one row?"* → **"No, one row per person"** → receipt matches COPY_DECK. No contradiction fired.

**Eligibility (Q8).** *"Is your study restricted to part of this data?"* — **"No, the study is about everyone here."** The **"Yes → which column, and what range?"** branch is a dimmed **"NOT OFFERED"** with an honest tooltip: naming a column+range "is not built in this build … a button that silently did nothing would be worse than one that says so." *(Honest gap, not a dead button.)*

**Seal (fixed).** With target+grain+eligibility answered, **"Draw it now"** was active. Clicking it drew `THE HELD-OUT SET · SEALED — 945 rows (15%)…`. **Front half complete.**

**Explore (rich, lens‑aware).** Profile table (`target meds_hbp · classification`, `classes 2`, `class sizes 0=770 · 1=5527`, `imbalance moderate`). Lens‑gated findings appeared: *"1 feature with high missingness"* (`meds_hbp`/`meds_chol` critical), *"`kcal` holds impossible values … [SETTLED]"*, *"Nutrient associations need energy adjustment [SETTLED]" (dietary lens)*, *"`imputed_bp_di` piles up on 0 [CONVENTION]"*, *"Moderate imbalance"* (7.2:1, use F1/PR‑AUC), *"16 features with outliers"* with distribution sparklines and **"See every distribution."** The bounded stack shows a **counted, typed remainder — "2 more — 1 caution, 1 info · 1 about this table · 1 from the dietary lens"** (GUIDED‑149, expected). Pull‑based explorers: *Physiologic plausibility · Missingness by feature · Correlation matrix · Distribution of each feature · Reverse‑coding audit [NOT BUILT]*. **PREVALENCE OF INADEQUACY** widget is wired — asking `kcal / modeled usual intake / against the EAR` returned *"Prevalence of inadequacy for kcal is computed by the EAR cut‑point method [SETTLED]"* (nutrition‑pack content; note it computed for energy rather than refusing — a domain nuance worth a second look).

**Features (surfaced‑correct).** Full transform catalogue — `log(x+1)`, `sqrt(x)`, powers, `1/x`, `A/B`, `A×B`, "Is this value missing?", quantile/cut‑point/cluster binning, ordinal encoding, center‑and‑scale, principal components — each labeled with **explainability cost** and **row‑local vs. fold‑fitted** leak‑safety language. Card 02 `choose_selection` ("every column or a subset?") with "Rank them for me / Use every column." Settled cleanly.

**Preprocess (surfaced‑correct).** `What do the blanks in your table mean?` for `meds_chol` (17,204 / 21,849 missing) with Yes/No/Not‑sure and **"Settle preprocessing."** After settling: *"This step is settled. Everything recorded here is fitted inside the training folds when the models are trained."*

**Train (STILL BROKEN — the wall).** `Train` becomes the active step, but its card renders only the heading **"Which models should be fitted?"** and a **"stale — an earlier answer changed"** tag, with **zero controls**. No model list, no fit button, no recompute. Settling Features/Preprocess did not populate it. *(DOM‑verified.)*

**Explain (empty, honest).** `Which columns is the model actually using?` → *"No model has been fitted yet … choose models in Train."* Plus **"SHAP is not offered here"** with the GUIDED‑101 rationale (deliberate). "RUN THE OTHER WAY — nothing to run both ways."

**Report / READ AS DRAFT / TRIPOD.** The methods draft builds live and is genuinely good — it even exposes the true row accounting (6,297 / 5,352 / 945) that the receipts hide — and ends *"11 sentences · 1 gap only you can fill"* with an `[AUTHOR_REQUIRED]` prompt. The TRIPOD+AI checklist (12 of 27 items) renders ITEM / WHERE ADDRESSED / AUTO‑FILLED TEXT / NEEDS YOUR INPUT, but half the auto‑filled cells are repeated placeholders (§4b‑7), and everything model‑dependent is empty because nothing is fitted.

---

## 6. Findings table (by category)

| # | Observation | Category | Register / ID |
|---|---|---|---|
| F1 | Train step renders heading only, "stale," 0 controls — no model shelf / fit | **broken (bug)** | DRIVE‑017 (still open) |
| F2 | Lens (Q1) below target picker; both cards "01"; diagnosis precedes lens | **wrong‑order** | `data-lens-question`, OPENING_SEQUENCE §01 |
| F3 | Undo of bulk apply reverts only 1 of 9 columns | **broken (bug)** | `audit-undo` |
| F4 | STUDY POPULATION/seal say "all 21,849"; true modeling N is 6,297 (15,552 dropped) | **broken / not‑visible** | COPY_DECK seal basis |
| F5 | Staleness cascade marks Explore+Train stale; no recompute affordance | **broken (bug)** | — |
| F6 | Positive‑class "which level is the event" never asked for 0/1 target | **not‑visible / possible gap** | `target-positive-class` |
| F7 | SEQN usable as predictor, unflagged | **not‑surfaced (gap)** | prereg |
| F8 | Survey‑design / weights absence unflagged | **not‑surfaced (gap)** | prereg, `nutrition-pack-content` |
| F9 | Pooled cycles (`cycle_begin_year`) not surfaced | **not‑surfaced (gap)** | prereg |
| F10 | TRIPOD checklist: 6/12 auto‑fills are repeated constants | **partially‑wired** | GUIDED‑238 |
| F11 | Bulk "read as binary": preview diff + per‑feature select + apply + receipt | **surfaced‑correct** | `bulk-repair`, `audit-preview` |
| F12 | Evidence badges SETTLED/CONVENTION on advisories | **surfaced‑correct** | `evidence-badge` |
| F13 | Purpose (prediction/inference) records "BUILT FOR" + routes missing‑indicator | **surfaced‑correct** | `model-purpose` |
| F14 | Dietary/clinical lens surfaces energy‑adjustment, impossible‑kcal, prevalence widget | **surfaced‑correct** | `nutrition-pack-content` |
| F15 | Seal draws with correct `cross_sectional` basis copy | **surfaced‑correct** | COPY_DECK |
| F16 | Feature/transform catalogue with cost labels + leak‑safe fold disclosures | **surfaced‑correct** | `choose_features` |
| F17 | Prevalence‑of‑inadequacy computed for `kcal` (energy) rather than refusing | **partially‑wired (domain)** | `nutrition-pack-content` |
| F18 | Tooltips occasionally overlap the control they describe | **unreadable (minor)** | — |

---

## 7. Condition 3 — surfaced / beautiful (called out)

Condition 1 (does the controller render it) is largely satisfied for the front half. Condition 3 (is shipped capability actually visible, unclipped, in the right order) is where the run’s real defects live:

- **Wrong order — the lens.** `OPENING_SEQUENCE.md` places the lens **first, before the diagnosis**, precisely because the diagnosis is field‑sensitive. On screen the lens renders **below the target picker** and **after** the on‑upload diagnosis. Both the lens and target cards are mislabeled **"01."** Effect: the lens can’t reframe the diagnosis (it only colors later Explore findings), which is the exact failure mode the doc warns about.
- **Not‑visible / misleading — the modeling N.** The receipts a user reads (`STUDY POPULATION`, the seal line) assert "all 21,849 rows." The real fitted/held‑out base is **6,297** (15,552 target‑missing rows dropped). Only the collapsible READ AS DRAFT paragraph tells the truth. A reader who trusts the on‑screen receipts will mis‑state their sample and mis‑read "945 (15%)."
- **Dead‑end — the stale Train card.** The single most consequential Condition‑3 failure: the Train capability exists behind the scenes (the draft even projects "28 candidate predictor parameters … a performance metric estimated on 945 rows"), but on screen there is **nothing to click** — a heading and a "stale" tag. The capability is invisible and unreachable.
- **Minor unreadable.** A couple of hover tooltips overlap the button they describe (the clinical‑lens hover covered the "Record" button; the seal hover covered the "Draw it" label). No clipping/contrast/truncation issues otherwise; the app is genuinely handsome and the transcript‑as‑methods pattern reads well.

---

## 8. Known / expected states confirmed (do not file as bugs)

- **SHAP absent from Explain** — surfaced as *"SHAP is not offered here"* with the two‑part rationale (**GUIDED‑101**; classic‑only `explain-shap`, `sens-*`). Deliberate. ✔
- **Reverse‑coding audit "NOT BUILT"** in Explore — honest disabled state (no survey lens). ✔
- **Eligibility manual column+range "NOT OFFERED"** — honestly explained, not a dead button. ✔
- **Explore stack bounded with a counted, typed remainder** ("2 more — 1 caution, 1 info…") — **GUIDED‑149**, expected, not truncation. ✔
- **Q1.5 table‑orientation did not fire** — correct (`table-orientation` is guided‑native and gated on a feature‑major assay; NHANES is neither). ✔
- **Q4–Q7 repeats/temporal chain did not fire** — correct for a one‑row‑per‑person table. ✔
- **nn/torch not fittable (TEST‑038)** — could not be exercised because the Train shelf never rendered; torch is deliberately absent regardless. ✔ (untestable this run)
- **Classic‑only rows** (cardinality table, duplicate‑row detection, Excel/JSON ingest, SHAP/sensitivity) — not expected in Guided; not filed.

---

## 9. Open questions / recommendations

1. **Why is the Train card empty and permanently stale?** The DOM shows `#card-train` containing only a GUIDED‑153 comment and no model‑shelf markup, under a persistent `#stale-train` tag. Is the model shelf gated on a stale flag that never clears once any upstream step is (re)settled? This is the one fix that unlocks the entire back half (Train → Explain → figures → checklist → report). Recommend: verify the controller emits the model list + fit control when the seal is drawn, and that "stale" offers a recompute path rather than blanking the card.
2. **Did my early undo cause the staleness?** The undo of `gender` (§4c‑1) and later step settles both preceded the stale tags. It’s worth checking whether a clean run *without* an undo populates Train — I could not test this because the CSV can’t be re‑uploaded from the automation tab (Start over requires re‑upload). If staleness is undo‑triggered, that’s still a bug (stale ≠ blank), but it narrows the repro.
3. **Positive‑class question for numeric 0/1 targets.** `target-positive-class` says the event level is *always asked, never pre‑selected at any confidence.* For `meds_hbp` (0/1) it was neither asked nor stated on screen. Is the question only wired for text/categorical two‑level targets? If so, a `1`‑is‑the‑event default is being applied silently for numeric binaries — the exact thing the register says must never happen.
4. **Receipt N.** Recommend the seal and STUDY POPULATION receipts disclose the target‑missingness drop (e.g., "6,297 of 21,849 rows have the outcome; 945 (15% of 6,297) held out"), matching the draft.
5. **SEQN / survey design.** An NHANES‑aware pack that reads a `SEQN` column and no survey weights should (a) flag `SEQN` as an identifier, not a predictor, and (b) note that design‑based estimation isn’t possible without weights. Both are currently silent.
6. **Prevalence‑of‑inadequacy for energy.** The widget computed an EAR cut‑point result for `kcal`; energy inadequacy isn’t an EAR/cut‑point concept — worth confirming this shouldn’t be a "refusal is an answer" case.

---

### Screenshot references (saved to disk)

Key frames captured during the drive (outputs folder, `screenshot-*.jpg` / in‑session IDs):
post‑upload state (`ss_5783`), bulk‑repair diff (`ss_3082`), apply+undo (`ss_9682`/`ss_9451`), lens recorded (`ss_9534`), target recorded (`ss_8381`), purpose+grain+eligibility (`ss_3036`/`ss_8498`/`ss_1248`), **seal drawn** (`ss_4301`), Explore findings (`ss_1449`/`ss_6903`/`ss_9914`), Features catalogue (`ss_5264`/`ss_0053`), Preprocess + Explain "no model fitted" (`ss_5667`/`ss_9562`), methods draft + TRIPOD (`ss_2100`). The decisive Train‑is‑empty result was confirmed by DOM inspection rather than a screenshot (the card is a bare heading).
