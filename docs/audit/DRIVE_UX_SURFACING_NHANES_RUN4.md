# TurboTab UX / Feature-Surfacing Report — Run 4

**App:** Tabular ML Lab — TurboTab walking skeleton, `http://127.0.0.1:8777/`
**Dataset:** `nhanes_1999_2018_yayhoo_fasting_diet_imputed.csv` — 21,849 rows × 29 columns
**Date:** 2026-08-12
**Method:** Fresh drive on the current build; two legitimate paths via *Start over*. On-screen behaviour observed through a controlled Chrome tab; where the UI was blocked, behaviour was reproduced in-process against `turbotab.api` with a FastAPI `TestClient`.

---

## Executive summary

- **Build is fresh and consistent.** `/dev/status` reports `page_newer_than_engine: **false**`, `rev 1e350ca`, matching git `HEAD` (`1e350ca`, "TEST-086"). Re-checked mid-drive — unchanged. Everything on screen is trustworthy.
- **The `/models` 500 is NOT fixed — it still fires**, and it blocks the entire Train → Fit → Explain → Figures half of the app. The Train step self-diagnoses cleanly (L59-A good variant). Reproduced root cause: an **unhandled `ModuleNotFoundError` for `xgboost` / `lightgbm`** imported at *module scope* in `ml/model_registry.py:18-19`, reached via `api.py:2186 → project.py:1924 → models.py:370 shelf()`. Both packages are in `requirements.txt`; the host's running venv is evidently missing them (they became module-scope imports on this path in recent L59/L60 work). With them installed, `/models` returns **200 (n_available=12) for every target tested** — there is no value-dependent failure. Fix: reinstall requirements + restart the host. (See caveat on Python version below.)
- **Headline (meds_hbp event-level gate) — VERIFIED, but only reachable in-process** because the `/models` 500 blocks fitting on screen. The event gate is a **real gate**: training a two-level target without answering "which level is the event" returns **400** with a clear refusal; after the level is answered, it fits.
- **DRIVE-040 (`event: 1.0`) — CONFIRMED at the payload/source level.** After choosing the event level `True`, the fitted result carries `positive_label: "1.0"` and `figure_bundle.py:400` emits `payload["event"] = str(event)` → `"1.0"`. The **named** level survives only in the prose transcript ("`'meds_hbp'` was encoded with True as the event (1) and False as the comparison (0)"). **Readability verdict: it reads badly** — a bare `1.0` the user never typed and that appears nowhere in their data. Full verdict below.
- **grain = "people repeat" — STILL DEAD-ENDS (DRIVE-036), with changed presentation.** The Q4→Q5→Q6 chain now *fires* and renders a populated aggregation menu (progress vs run 3's empty menu), but selecting any combine method is refused — "There is no identifier column recorded, so there is nothing to combine rows by" — the person-identifier follow-up defined in the API contract is **never surfaced in the UI**, rows never combine, and the seal stays disabled ("Draw it now / NOT YET"). The seal receipt even *claims* "you can name it at any point before the seal," but no such control exists — a copy-vs-capability contradiction.
- **N-consistency (L59-B) is now clean**; the post-seal Exclude control refuses correctly; eligibility "Yes" is deliberately NOT OFFERED; lens-below-target ordering persists (treated as expected).

---

## Tested HEAD + `/dev/status`

| Field | Value |
|---|---|
| git branch | `TurboTab` |
| git HEAD | `1e350ca1208c6b980034cece2ee2cc3a0e1473b8` ("TEST-086: the denominator moved again…") |
| `/dev/status` rev | `1e350ca` |
| `engine_loaded_at` | 1786588397.34 |
| `page_mtime` | 1786588103.34 |
| **`page_newer_than_engine`** | **`false`** (engine newer than page — consistent) |

Re-checked `/dev/status` mid-drive after an inconsistency appeared in my sandbox (see the Python-version caveat) — values were byte-identical, confirming the host build never shifted. **Not stale.**

---

## Path 1 — `meds_hbp` (the event-level headline)

`meds_hbp` is a two-level boolean-text column: `True` = 5,527, `False` = 770, blank = 15,552 (read as `object`). Only 6,297 of 21,849 rows carry an outcome — the heavily-missing-target scenario relevant to the N-consistency carry-over.

### On-screen walkthrough (up to the seal)

The opening sequence fired in this order and read cleanly:

1. **Target picker** ("02 What are you predicting?") → selecting `meds_hbp` produced the receipt *"meds_hbp is the target. The engine reads it as classification at high confidence. Target is object type (categorical/binary)."* **No event-level question appears here** — consistent with the design (the event gates fit, not target-select).
2. **Lens** ("01 What kind of measurements…") — rendered *below* the target card (both step-numbered; lens "01" under target "02"). Selected *Clinical measurements and labs* → *Record this one*.
3. **Q2.5 "What is this model for?"** → *Predicting an outcome for a new person*.
4. **Q3 grain "Can one person appear in more than one row?"** → *No, one row per person*.
5. **Q8 eligibility "Is your study restricted to part of this data?"** → *No, the study is about everyone here*. ("Yes → which column, and what range?" is greyed **NOT OFFERED** — deliberate.)
6. **Seal "09 Draw the held-out set"** → enabled with no event question outstanding, and drew:

> **THE HELD-OUT SET · SEALED** — "945 rows (15% of the 6,297 with a value for the outcome; the other 15,552 of 21,849 rows have none) are held out and will not be looked at again until the models are scored."

This seal disclosure is a **Condition-3 credit**: it reconciles all three bases (945 test / 6,297 valid / 21,849 total) in one sentence, and 945 = 15% × 6,297 is correct. This is exactly the L59-B N-consistency concern from run 3, and here the numbers agree and are explained.

### The `/models` 500 blocks the rest of the UI

Navigating to Train:

> **01 Which models should be fitted?** — "The model shelf could not be loaded, so there is nothing here to choose from — this step is not empty, it is broken. Nothing about your data or your answers is lost, and the held-out set is still sealed." `HTTP 500 Internal Server Error` [Try loading the models again]

Network log confirmed: `GET http://127.0.0.1:8777/project/6494f1c6f345/models → 500`. This is the **L59-A "good" self-diagnosing variant** — it names the failure, shows the status code, preserves state, and offers a retry, rather than the silent bare-heading "stale" variant. **Because the shelf cannot load, no model can be selected, nothing can be fitted, and the event gate + DRIVE-040 figure are unreachable on screen.** The Explain step degrades correctly too: *"No model has been fitted yet… choose models in Train"* and *"SHAP is not offered here"* (GUIDED-101/232, deliberate).

### Event-level gate — VERIFIED in-process (real gate)

Reproduced with the real file against `turbotab.api`:

- **Refusal without an answer** — `POST /project/{id}/train` on `meds_hbp` with no event chosen returns **400**:
  > "Which level of `'meds_hbp'` is the event has not been recorded, and it decides what every score means — sensitivity and specificity are of the event, and the curves are drawn against it. There is no default: whether the event is (say) death or survival is the research question, not something the file can say. Answer "Which of these is the event you are predicting?" on the outcome, then fit."
- **The event card names the real levels** — the `set_positive_class` finding (`id=positive_class__meds_hbp`, title *"Which of these is the event you are predicting?"*): *"`'meds_hbp'` is the outcome and holds two values — `'False'` (770 rows) and `'True'` (5,527 rows). 15,552 rows have no outcome recorded."* (`levels: ["false","true"]`, counts, `suggested: "true"` with reason "conventionally the event").
- **After answering, it fits** — applying `set_positive_class` with `choice="true"`, then training, succeeds: job → *done*, "Fitting Logistic Regression on 5,352 rows."

The gate is genuine: it is enforced at fit/scoring (correctly *not* at the seal, because `draw_holdout` does not stratify by class — the split is byte-identical whichever level is the event). This matches the L60/DRIVE-032 design.

### DRIVE-040 (`event: 1.0`) — CONFIRMED + readability verdict

After choosing the event = `True`:

- `GET /project/{id}/training` carries `"positive_label":"1.0"`.
- `turbotab/figure_bundle.py:400` sets `payload["event"] = str(event)` → the figure payload serves `"event": "1.0"`.
- The **named** level survives only in the prose transcript / draft: *"`'meds_hbp'` was encoded with True as the event (1) and False as the comparison (0)."*

**Condition-3 readability verdict:** The value `1.0` reads **badly**. The user picked the level **"True"** from a card that clearly named "True" and "False" with counts; the machine-facing surfaces (figure `event` field, `positive_label`) then present `1.0` — a value the user never typed and that appears nowhere in their column (whose two spellings are `True`/`False`). To decode it, the reader must recall the one encoding sentence buried in the transcript. It is not *wrong* (the encoding is faithful), but it is an **implementation artifact leaking to the surface**, and it is strictly less legible than the named level would be. Because `figure_bundle.py:400` emits `str(event)` with no name look-up, the on-screen figure caption would render `1.0` **just as badly as the raw payload** — so my answer to the orchestrator's question ("does it read as badly on screen as in the payload?") is **yes, it would read equally badly on screen** — *if* it could be reached, which on this build it cannot, because the `/models` 500 blocks every fitted figure. DRIVE-040 remains **OPEN and user-visible in principle**; the fix (carry the chosen level name alongside the encoded value) is still owed.

---

## `/models` 500 — status and captured diagnosis

**Status: STILL FIRES on the host.** Confirmed on screen (`GET /project/6494f1c6f345/models → 500`).

**Captured reproduction** (sandbox venv, real file, via `docs/turbotab/tools/why_models_500.py` and direct `TestClient`). The repo's own `venv/bin/python` is a macOS venv and won't run in the Linux sandbox, so a fresh venv was built and the diagnostic run in-process. The traceback:

```
File ".../turbotab/api.py", line 2186, in get_models
    entries, ranked_on = project.model_shelf_ranked()
File ".../turbotab/project.py", line 1924, in model_shelf_ranked
    return (_models.shelf(prof, self.task_type or "regression", ...
File ".../turbotab/models.py", line 370, in shelf
    from ml.model_registry import get_registry
File ".../ml/model_registry.py", line 18, in <module>
    from xgboost import XGBRegressor, XGBClassifier
ModuleNotFoundError: No module named 'xgboost'
```

`ml/model_registry.py:18-19` imports `xgboost` and `lightgbm` at **module scope**; the import is triggered lazily inside `shelf()`, so it only detonates when the model shelf is first requested — i.e. exactly at the Train step. The exception is a raw `ModuleNotFoundError` (not an `HTTPException`), so it reaches Starlette unhandled and is flattened to the opaque 21-character `Internal Server Error` the browser sees.

**This is an environment failure, not a data or logic bug:**

- Installing `xgboost` + `lightgbm`, `GET /models` returns **200, n_available=12**, and a sweep of six targets (`meds_hbp`, `bp_sys`, `bp_di`, `weight`, `height`, …) **all return 200** — no value-dependent 500 anywhere.
- Both packages are declared in `requirements.txt` (`xgboost>=2.0.0`, `lightgbm`). They became module-scope imports on the `/models` path in recent L59/L60 work; the host's running venv was evidently not reinstalled afterward, so the running engine imports a package it doesn't have.

**Recommended fix:** `pip install -r requirements.txt` in the host venv, then restart the app.

**Honest caveat (please pass to whoever closes this):** my sandbox runs **Python 3.10**, whereas the host runs **Python 3.13** (repo `venv/bin/python → python3.13`). The `/models` path does not import `figure_bundle`, so the version gap does not affect this diagnosis — but for a *definitive* host traceback, run `venv/bin/python docs/turbotab/tools/why_models_500.py <csv>` **on the host**. If the host venv unexpectedly *does* have xgboost/lightgbm, the host traceback will point elsewhere and should be captured directly, since my environment cannot faithfully mirror 3.13.

---

## grain = "Yes, people repeat" re-test (DRIVE-036)

Path 2 target = `bp_sys` (regression, high confidence, 1,957 unique). Lens = clinical. Q2.5 = prediction. Then grain = **"Yes, people repeat."**

**The chain now fires further than run 3, then dead-ends at the same wall:**

- Grain receipt: *"…people repeat, and no column identifying the person was named — so the held-out rows cannot be drawn by person."*
- **Q4 "Are these repeats or different time points?"** fired → *Repeated measurements of the same quantity* ("asked, because the evidence was thin").
- **Q5 "When you analyze this, what is one row?"** fired → *One row per person* ("each person's records are combined into one before anything is held out").
- **Q6 "How should each person's rows be combined?"** fired with a **populated menu** — *Their mean · The first · The last · The change from the first* (run 3 showed an empty menu here). But **selecting "Their mean" is refused**:
  > "There is no identifier column recorded, so there is nothing to combine rows by."
- **Seal "09 Draw the held-out set"** is **disabled — "Draw it now / NOT YET"** — because rows were never combined.

**Root cause (confirmed against the API contract).** `GET /project/{id}/grain` declares the "Yes, people repeat" option with `follow_up: "which column identifies the person?"`. That follow-up is **never rendered in the UI** — after choosing "people repeat" the app goes straight to the receipt "no column identifying the person was named" and then Q4. With no identifier captured, `keep_identifier`/aggregation has nothing to group by (`project.py:1226` refusal string), rows never combine, and the seal is permanently blocked.

**Aggravating copy-vs-capability contradiction.** The seal receipt states: *"Your numbers are labeled exploratory until a person column is named, **and you can name it at any point before the seal.**"* Yet no naming control is surfaced anywhere on the page (natural-language element search returned only the refusal message). The prose promises an action the interface does not provide.

**Verdict: DRIVE-036 is STILL BROKEN (changed presentation).** Fixed vs run 3: the Q4→Q5→Q6 chain now fires and the aggregation menu is populated. Still broken: the person-identifier follow-up is unsurfaced, aggregation is refused, and the seal dead-ends — now with an added false promise in the receipt. NHANES `SEQN` is present in the file and is the obvious identifier, but the app offers no way to designate it.

---

## Table / rendering audit

- **File-loaded receipt** and **"9 features need the same repair: read as binary"** (the `imputed_*` bool columns) render cleanly; the shape-findings card *"9 things stand out in the shape of this file"* is present with a *First rows* affordance.
- **Impossible-value tables** render well: `IMPOSSIBLE — BP_DI` (125 entries, "below 15 mmHg"), `IMPOSSIBLE — KCAL` (9 entries, "below 100 kcal"), each with ROW / value / BOUND columns, an "All N affected rows" expander, honest truncation ("Showing the first 12 of 125"), and pagination ("1 / 9"). The counts in the affordances match the counts behind them (GUIDED-149 bound behaviour).
- **Improbable-values** section correctly labelled *Advisory / reported, not proposed* (no PASS/FAIL stamp).
- **Profile table** (rows 21,849; features 28; p/n=0.00 "Low dimensionality") renders under Explore in the sealed path.
- **Target dtype chips** correct (`meds_hbp` object, `bp_sys` float64 with hover "1957 unique · 0 missing").
- No `[object Object]`, no NaN, no raw dict rendering seen on any card that loaded. (The one card known historically to risk `[object Object]` — the positive-class finding — has clean server-side params (`levels`/`spellings`/`counts`); its on-screen chip lives near the fit step and was unreachable behind the `/models` 500, so its rendering is **unverified this run**.)

---

## Carry-over status table

| Item (source) | Run-3 state | Run-4 status |
|---|---|---|
| `/models` HTTP 500 (DRIVE-035) | 500 | **STILL BROKEN.** Root cause captured: module-scope `xgboost`/`lightgbm` import missing in host venv (`model_registry.py:18-19`). Not value-dependent. |
| L59-A Train self-diagnosis | good vs silent variants | **GOOD variant confirmed** — "it is broken", shows `HTTP 500`, preserves state, retry offered. No silent bare-heading seen. |
| L59-B N-consistency (21,849/10,645/4,645 disagreement) | disagreeing bases | **FIXED / clean** — seal disclosure reconciles 945 / 6,297 / 21,849 in one sentence; arithmetic correct. |
| Event-level gate for numeric/two-level target (L60, DRIVE-032) | new | **VERIFIED (in-process)** — real 400 refusal until answered; fits after. |
| DRIVE-040 `event: 1.0` | OPEN | **CONFIRMED OPEN** — `positive_label:"1.0"` + `figure_bundle.py:400 str(event)`; name only in transcript. Reads badly (verdict above). On-screen unreachable due to `/models` 500. |
| grain=repeat aggregation dead-end (DRIVE-036) | dead-end, empty menu, seal `answer notbuilt` | **STILL BROKEN, changed** — chain fires, menu populated, but selection refused ("no identifier column"), identifier follow-up unsurfaced, seal "NOT YET". New copy-vs-capability contradiction in seal receipt. |
| `[object Object]` chip in positive-class finding | present | **Data clean server-side** (no `[object Object]` in params); on-screen chip **unverified** (behind `/models` block). |
| Prereg gap — `SEQN` usable as predictor | gap | **STILL PRESENT** — `SEQN` appears in the `/features` payload as an available predictor, not flagged/excluded. |
| Prereg gap — survey design / weights | gap | **N/A this file** — no survey-weight columns present in this CSV; app still does not ask about design/weights. |
| Prereg gap — pooled cycles unsurfaced (`cycle_begin_year`) | gap | **STILL PRESENT** — pooled 1999–2018 cycles not surfaced as a design concern. |
| DRIVE-034 SETTLED-EAR badge (energy has no EAR) | reproduced/"not yet built" at run-3 HEAD | **NOT re-verified this run** — nutrition/prevalence sits behind `figure_bundle`, which won't parse under the sandbox's Python 3.10 (PEP-701 f-string, `figure_bundle.py:677`); valid on host 3.13. findings.json marks DRIVE-034 FIXED at this HEAD. |
| Post-seal "Exclude rows" enabled-despite-refusal | enabled despite refusal | **CHANGED / safe** — button stays enabled (minor affordance nit) but clicking it **refuses correctly** ("The test set is already sealed… Constitution §04 routes this back to the pre-seal question, which needs a re-seal") and performs **no mutation**. |

---

## Known / expected states (not bugs)

- **Lens rendered below the target picker** (lens "01" under target "02"); the diagnosis runs pre-lens. Treated as expected per this run's brief.
- **Eligibility "Yes → which column, and what range?" NOT OFFERED** — deliberate (`elig-question` guided-only; classic-only pieces withheld).
- **SHAP not offered in Explain** (GUIDED-101 / GUIDED-232) — deliberate; the four research packs contain zero explainability content, so method choice is unsourced. Surfaced as "SHAP is not offered here," not silently absent.
- **Reverse-coding never proposes a reversal** (`survey-reverse-coding-audit`, section B1.2 SETTLED) — deliberate.
- **"Mark the whole column as not trustworthy" → `not built`** (GUIDED-096) — deliberately named on the shelf rather than omitted.
- **Q1.5 (orientation) suppressed** — correct; the lens is clinical, not an assay pack, and the table is sample-major.
- **Aggregation menu `mean / first / last / change`** — v1 menu; slope/AUC/usual-intake "filed, not built."
- **Counted-remainder bound of five** (GUIDED-149) — a measured parameter, not the prototype's two.

---

## Condition-3 findings (surfaced / beautiful)

**Reads badly / wrong:**

- **`event: 1.0` (DRIVE-040)** — the single clearest legibility regression: after the user names "True" as the event, machine surfaces show `1.0`. Would read equally badly on screen; unreachable here only because of the `/models` block.
- **grain=repeat seal receipt promises a control that doesn't exist** — *"you can name it at any point before the seal"* with no naming affordance anywhere. A user who reads the promise and hunts for the control finds only a refusal.
- **`/models` opaque 500 on the host** — the *only* thing standing between this build and a complete happy path; and the failure text a raw browser sees is 21 characters. (Mitigated well by the Train step's own self-diagnosis, which is a Condition-3 *credit*.)

**Surfaces cleanly (credit):**

- **Seal N-disclosure** naming all three row bases in one sentence — model of legible honesty on a heavily-missing target.
- **Train step self-diagnosis** — turns an infra 500 into a readable, state-preserving, recoverable message ("this step is not empty, it is broken").
- **Post-seal Exclude refusal** — clear §04 reasoning, no silent mutation.
- **Impossible-value tables** — honest truncation, matching counts, advisory-vs-actionable framing kept distinct.
- **Event-level card** — names both levels with counts and a stated (overridable) suggestion; no pre-selection.

---

## Open questions for the orchestrator

1. **Host `/models` traceback.** My reproduction points squarely at a missing `xgboost`/`lightgbm` in the host venv, and the fix is a reinstall + restart. Please confirm on the host with `venv/bin/python docs/turbotab/tools/why_models_500.py <csv>` (Python 3.13) — if the host venv *does* have those packages, the real cause is elsewhere and the host traceback should be captured directly. The whole Train/Fit/Explain/Figures half of the app is dark until this clears.
2. **Should the meds_hbp headline be re-driven on screen after the venv is fixed?** The event gate and DRIVE-040 were only reachable in-process this run; a real UI drive would let us judge the *on-screen* rendering of `event: 1.0` and the positive-class chip (the `[object Object]` question), both currently unverifiable.
3. **DRIVE-034 (nutrition EAR refusal)** could not be re-verified in-sandbox (Python 3.10 can't parse `figure_bundle.py`'s 3.12+ f-string). It needs either an on-screen drive (dietary lens) or an in-process check under Python 3.13.
4. **DRIVE-036 person-identifier follow-up.** The API contract defines `follow_up: "which column identifies the person?"` for the repeat grain, but the UI never renders it. Is the missing follow-up card the intended fix, and should the seal receipt's "you can name it at any point" copy be pulled until it exists?
