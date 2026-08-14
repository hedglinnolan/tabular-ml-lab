# TurboTab UX surfacing report — RUN 5

**Date:** 2026-08-13
**App:** http://127.0.0.1:8777/ (TurboTab branch, Guided door)
**Dataset:** `nhanes_1999_2018_yayhoo_fasting_diet_imputed.csv` (21,849 × 29)
**Paths driven:** 2 (path 1 = `meds_hbp` binary classification; path 2 = `bp_sys` regression)
**Method:** five simultaneous lenses — bugs, what surfaces (Condition 3), mathematical soundness (independently verified against the CSV), routing action-log, pedagogical soundness.

---

## 1 · Executive summary

**This is the first run in which the full Train → Fit → Figures → Report path was reachable, and the back half largely works.** Both models fitted on both paths; five figures drew with real geometry on path 1 and three on path 2; the TRIPOD+AI checklist and a real manuscript with real numbers rendered on both.

Headlines:

1. **Mathematical soundness is excellent.** Every headline number the app displayed reconciled against independent pandas/sklearn ground truth — 34 distinct figures checked, **32 exact matches, 0 wrong, 2 scope/labelling disagreements** (both discussed below, and one of them the app catches itself). The seal arithmetic (`945 rows = 15% of the 6,297 labeled; 15,552 of 21,849 have none`) is correct to the row, as are the impossibility counts (bp_di 125, kcal 9, bp_sys 174 improbable), class imbalance (7.2:1), EPV (27.5), missingness rates, spline knots (1052 / 1950 / 3380.2 at p10/p50/p90) and the regression metric magnitudes.

2. **The event-level gate works exactly as specified and is the best-surfaced thing in the build.** `POST /train` returns **400** until the event level is named; the refusal renders at the control in plain language; the positive-class card offers `True [conventional]` as an explicitly-labelled *suggestion* with "Nothing is selected for you here". This is GUIDED-009 delivered.

3. **DRIVE-040 is half fixed.** The fitted figure now **names the level** — calibration caption reads *"945 observations with 829 events of True"*, payload `event:"True", event_named:true`. But the encoded `1`/`0` still leaks into three other user-facing surfaces (PCA group annotation, Table 1 column headers, and the event noticing card after the encode). The row should not be closed on the figure alone.

4. **GUIDED-236 is confirmed and now directly observable.** Two models were fitted; the ROC caption reads *"Discrimination for **1 model(s)**"*. Previously unreachable, now reproducible in one click.

5. **One real defect found, and the app found it too.** Path 1's manuscript FAILS its own validator: *"Table 1 population matches the analysis cohort — Expected analysis N=6297, Table 1 overall N=21849."* The regression path passes 13/13, which confirms this is a genuine target-specific scope bug rather than a validator quirk. Separately, path 1's Methods asserts *"116 of the held-out rows carrying the outcome"* while the figures report **829 events** on the same 945 rows — these cannot both be right, and 829 is the correct one.

6. **The model shelf reports the wrong n on the classification path.** The shelf reasons about `n=20,904` (21,849 − 945) when only **5,352** rows are trainable, because 15,552 rows have no outcome. The fit itself is correct (`5,352 trained on`), so this is copy reading the wrong denominator — but it is the number a user reads while choosing models. The regression path gets it right (`n=18,572`).

---

## 2 · Tested build

`GET /dev/status`, checked at start and again at the end of the run — **identical both times**:

```json
{"enabled":false,"flag":"TURBOTAB_DEV_CHECKS",
 "build":{"rev":"bcbd378","engine_loaded_at":1786640451.96,
          "page_mtime":1786640279.63,"page_newer_than_engine":false,"why":null},
 "environment":{"python":"/Users/nhedglin/tabular-ml-lab/venv/bin/python",
                "prefix":"/Users/nhedglin/tabular-ml-lab/venv",
                "engine_stack_ok":true,"missing":[],"why":null,"fix":null}}
```

- `git rev-parse HEAD` = `bcbd3785eeb4a09f1bd07a1e92a16c45ac5be4ff` → **`bcbd378` == HEAD** ✓
- `page_newer_than_engine: false` for the whole session — **staleness guard never tripped**; no result in this report rests on a stale page.
- `engine_stack_ok: true`, `missing: []`, venv interpreter — the dependency gap that blocked runs 1–4 is resolved. This is what made the back half reachable.

Projects: path 1 = `2948951cd2f3`, path 2 = `9baa49156939`. Note the path-1 project remained live server-side after "Start over" (fetching its `/figures` still returned path-1 content) — session-scoped rather than destroyed, which is consistent with "the past is editable, never silently destroyed", but worth knowing.

---

## 3 · Ground truth (computed independently, before touching the app)

From the attached CSV in the sandbox with pandas/numpy/sklearn:

| Quantity | Value |
|---|---|
| Shape | 21,849 rows × 29 cols |
| Missingness | Only two columns: `meds_hbp` 15,552 (71.18%), `meds_chol` 17,204 (78.74%). All others 0. |
| `meds_hbp` | labeled 6,297 — True 5,527 (87.77%), False 770. NaN 15,552. 6,297+15,552 = 21,849 ✓ |
| Seal (classification) | round(0.15 × 6,297) = **945**; train 5,352 |
| Seal (regression) | round(0.15 × 21,849) = **3,277**; train 18,572 |
| `bp_sys` | n 21,849, mean 124.019, sd 18.775, min 66, median 121.78, max 240, 1,957 unique, 0 missing |
| `kcal` | mean 2,120.08, sd 1,003.08, range 0–15,594; p10/p50/p90 = 1052 / 1950 / 3380.2 |
| Impossibility | bp_di outside [15,220] = **125**; kcal outside [100,30000] = **9**; bp_sys outside [90,200] = **174** |
| kcal implausible | <500 → 194, >5000 → 307, total **501** |
| Correlations | bp_sys~age r=0.503 (r²=0.253); kcal~fat_total r=0.878 |
| Grain | SEQN unique per row → cross-sectional |
| Regression baseline | mean-predictor on held-out: MAE 14.11, RMSE 18.46, R² ≈ 0.000 (test sd 18.46) |

---

## 4 · Path 1 walkthrough — `meds_hbp`, binary classification

Upload → receipt *"…was loaded: 21849 rows, 29 columns."* ✓. Diagnosis surfaced two noticings: a bulk repair ("9 features need the same repair: read as binary") and "9 things stand out in the shape of this file."

**Target.** Picked `meds_hbp`. Receipt: *"meds_hbp is the target. The engine reads it as **classification** at **high** confidence. Target is object type (categorical/binary)."* Nothing pre-selected ✓.

**Lens.** Selected Dietary intake + Clinical measurements and labs → *"Record these 2"*. Receipt names both and states domain conventions informed later defaults and were overturnable.

**Grain.** "No, one row per person" → *"Recorded: one row per person. The held-out rows will be drawn at random, which is the right choice when every row is a different participant."* Matches COPY_DECK verbatim.

**Eligibility.** "No, the study is about everyone here" → *"No eligibility restriction: all 21849 rows are in the study population."* Seal control flipped from `NOT YET` to pressable only after this — clause-01 gate ordering honoured.

**Seal.** *"945 rows (15% of the 6,297 with a value for the outcome; the other 15,552 of 21,849 rows have none) are held out and will not be looked at again until the models are scored."* — **character-for-character the COPY_DECK `cross_sectional` line, and every number correct.**

**Explore.** Profile: rows 21849, features 28, numeric/categorical 26/2, missing overall 2.8%, p/n 0.001, data sufficiency abundant, target `meds_hbp · classification`, classes 2. Narrative: *"770 minority-class events for 28 candidate parameters (EPV = 27.5)"*, with an honest caveat that EPV is a legacy heuristic and the field's criterion (Riley et al.) needs a quantity this app does not hold.

**Train (first attempt).** Selected Histogram Gradient Boosting + Logistic Regression → "Fit 2 model(s) on the held-out split" → **refused, `POST /train` 400**, with the event-gate message rendered at the button.

**Event.** Found in the Data-step noticing: *"Which of these is the event you are predicting? 'meds_hbp' is the outcome and holds two values — 'False' (770 rows) and 'True' (5,527 rows). 15,552 rows have no outcome recorded."* Expanding gave **THE EVENT — nothing is chosen yet** with `False` and `True [conventional]`, and the line *"'True' is conventionally the event — shown as a suggestion, not applied. Nothing is selected for you here."* Chose `True`; a preview computed before committing: **CELLS CHANGED 0 → 6,297**, **MISSING 15,552 → 15,552**, **DTYPE object → Int64**, ROWS/COLUMNS unchanged, with *"15,552 row(s) have no outcome recorded and are excluded from modeling. Nothing has happened yet."*

**Train (second attempt).** Fit succeeded. **HELD-OUT PERFORMANCE — 945 rows held out · 5,352 trained on · classification**:

| Model | Held-out | Shelf said |
|---|---|---|
| Histogram Gradient Boosting (Classification) | Accuracy 0.88 · F1 0.842 | strong non-linear learner for this shape |
| Logistic Regression | Accuracy 0.887 · F1 0.849 | interpretable probability baseline |

Disclosure beneath: pipelines composed from the recorded plan, every statistic fitted over the 5,352 training rows, *"the held-out rows inform none of them"*, and *"1 column(s) with missing values had no recorded handling and were filled with this app's default inside each training fold: meds_chol."*

**Figures.** 5 admitted, 0 held, 12 not-drawn with stated reasons. All five carry real geometry.

**Report.** TRIPOD+AI checklist; manuscript with Methods, Table 1 (45 rows), LaTeX 6,877 bytes; validator 13 checks, **1 FAIL**.

---

## 5 · Path 2 walkthrough — `bp_sys`, regression

"Start over" returned a clean pre-upload screen with the rail reset ✓. Re-upload verified 21,849 × 29.

**Target.** `bp_sys` → *"The engine reads it as **regression** at **high** confidence. Target is continuous numeric (1957 unique values)"*. Hover chip: *"float64 · 1957 unique · 0 missing"* — both verified ✓.

**Model purpose (2.5)** fired on this path: chose "Predicting an outcome for a new person". Receipt explains that a was-it-missing indicator or censoring flag becomes legitimate under a prediction objective because it is available at deployment.

**Grain / eligibility** as before. **Seal:** *"3,277 rows (15%) are held out and will not be looked at again until the models are scored."* — correct, and correctly **omits** the "of the N with a value for the outcome" clause, because `bp_sys` has no missing values. The copy-deck's base-naming rule is applied conditionally and correctly.

**Explore.** Profile switched appropriately: numeric/categorical **25/3**, missing overall **5.4%** (32,756 / (21,849×28) = 5.35% ✓), **mean (SD) 124.02 (18.77)** ✓, **range 66–240** ✓, target `bp_sys · regression`. The classification-only sentences (minority class, EPV, class imbalance) are **absent** ✓. The outlier finding moved 16 → **15 features** because `bp_sys` is now the target and out of the feature scan ✓.

**Train.** The shelf resolved to an entirely different registry — Histogram Gradient Boosting (Regression), LightGBM (Regression), XGBoost (Regression), ElasticNet, Lasso, Ridge, Neural Network, Extra Trees (Regression), Random Forest / kNN (Regression), GLM (Huber), GLM (OLS/Logistic) / Support Vector Regression under "not recommended". **`n=18,572`** in the capacity clauses — correct. GLM (Huber) carries a data-aware reason: *"only pays when the outcome itself has outliers — yours looks clean."* The recorded purpose answer is quoted back verbatim with an honest note that it changed nothing here.

Fitting showed a live progress panel with a working **"Stop this"** cancel control. **No event gate fired** — correct for regression.

**HELD-OUT PERFORMANCE — 3,277 rows held out · 18,572 trained on · regression**:

| Model | Held-out |
|---|---|
| Histogram Gradient Boosting (Regression) | MAE 9.751 · RMSE 13.549 · **R² 0.466** · MedianAE 7.137 |
| Ridge Regression | MAE 10.729 · RMSE 14.711 · **R² 0.37** · MedianAE 8.302 |

**Figures.** 3 admitted: PCA, **dose-response restricted cubic spline**, forest. Calibration / ROC / decision curve correctly absent.

**Report.** Manuscript **passes 13/13**; Table 1 single column "Overall (N=21849)", 48 rows; LaTeX 6,824 bytes.

---

## 6 · Findings by lens

### Lens 1 — Bugs and errors

| # | Severity | Finding |
|---|---|---|
| B1 | **High** | **Path 1 manuscript fails its own validator.** `Table 1 population matches the analysis cohort — Expected analysis N=6297, Table 1 overall N=21849.` Table 1's "Overall" column pools the 15,552 rows with no outcome, which are excluded from modeling. Strata are right (0: n=770, 1: n=5527 → 6,297). Regression path passes, so this is specific to a target with missing values. **Credit: the app detects and reports it (`passed:false`) rather than exporting silently.** |
| B2 | **High** | **Numeric contradiction in the path-1 Methods.** Methods says *"with 116 of the held-out rows carrying the outcome"*; the calibration/ROC/decision-curve payloads all say **829 events** on the same 945 rows. 829 is correct (0.8777 × 945 = 829.4); 116 is the non-event count (945 − 829). A methods section understating events by a factor of seven is a publishable error. |
| B3 | Medium | **Model shelf reports the wrong n on the classification path.** "Neural Network — n=20,904 supports the capacity" and "SVC — slow at n=20,904". Real trainable n is **5,352**; 20,904 = 21,849 − 945 ignores the 15,552 unlabeled rows. The fit is correct; only the shelf copy is wrong. Regression path correct (18,572). |
| B4 | Low | **Stale caption under the Train card.** With two models selected and even after fitting, the panel below still reads *"No model is selected. The shelf orders every model this task can use…"*. Contradicts the button one line above ("Fit 2 model(s)"). |
| B5 | Low (cosmetic) | Bulk-repair noticing count shifts 9 → 8 features between pre- and post-target states on path 1 without explanation (the target leaves the set). Not wrong, but unexplained on screen. |
| — | Not a bug | `GET /project/null/figures` → **404** appeared in the network log. This was **my own** JS probe with an unresolved id, not app behaviour. Recorded so it is not miscounted. |

**No `[object Object]`, no 500s, no stack traces, no broken/empty tables, no console errors observed on either path.** The only non-2xx from app-driven actions was the deliberate `400` on the pre-event fit.

### Lens 2 — What actually surfaces (Condition 3)

**Surfaces cleanly — credit where due:**

- **The event-level gate.** The refusal renders *at the control that was pressed*, above the fold, in full sentences, naming the exact question to answer and where. This is `response-at-the-control` working.
- **The event card itself** — both levels, the convention marked as a suggestion, the reasoning, and the explicit "Nothing is selected for you here", followed by a **before/after diff with counts** before anything is applied. `audit-preview` delivered.
- **The seal receipt** — one sentence, all four numbers, unclipped, in the ON THE RECORD rail.
- **The model shelf is never shortened** — SVC stays visible under "NOT RECOMMENDED FOR THIS DATA" with its reason, on both paths.
- **Unbuilt capability is named rather than hidden** — "Mark the whole column as not trustworthy · not built" cites `GUIDED-096` and explains why it is listed anyway; pull chips carry `NOT BUILT`; "SHAP is not offered here" gives two distinct reasons and points at Classic.
- **Figures that cannot apply say why** — "Volcano plot: the lens question has not been answered metabolomics or genomics"; "Calibration plot: No model has been fitted yet"; "This figure does not apply to this project." Nothing renders as a silent blank.
- **Regression/classification differentiation is thorough** — metrics, figure set, profile narrative, shelf contents and Table 1 shape all switch correctly.

**Rendered but hard to reach / hard to read:**

- **The event question is not where the gate points.** The gate says "Answer 'Which of these is the event you are predicting?' on the outcome" — but the card lives in the **Data**-step noticing stack, far above the Train step, and requires expanding "Show me what this means" to reveal the selector. On a page this long that is a substantial scroll from the refusal to the answer. The instruction is accurate but the distance is not small.
- **Extremely long single-scroll page.** All eight steps render in one column; reaching Train from Data took roughly 40 viewport-heights. The left rail highlights the active step but does not navigate to it — clicking "Train" changed the highlight without scrolling. This is the main navigational cost of the build.
- **Encoded values reach the reader** (see DRIVE-040 below) — `1`/`0` in the PCA legend and Table 1 headers where `True`/`False` was shown minutes earlier.

### Lens 3 — Mathematical soundness (verified value-by-value)

Every number below was computed independently in the sandbox and compared to what the app displayed.

**Exact matches (✓):**

| App display | App value | Independent | Match |
|---|---|---|---|
| Load receipt | 21,849 rows, 29 cols | 21,849 × 29 | ✓ |
| Seal, classification | 945 / 6,297 / 15,552 / 21,849 | round(.15×6297)=945; 6297; 15552; 21849 | ✓ |
| Seal, regression | 3,277 (15%) | round(.15×21849)=3277 | ✓ |
| Train scope, cls | 945 held out · 5,352 trained | 6297−945=5352 | ✓ |
| Train scope, reg | 3,277 held out · 18,572 trained | 21849−3277=18572 | ✓ |
| Minority class | 770 | False count = 770 | ✓ |
| EPV | 27.5 | 770/28 = 27.50 | ✓ |
| Class imbalance | 7.2:1 | 5527/770 = 7.18 | ✓ |
| Event card counts | False 770 / True 5,527 / 15,552 none | identical | ✓ |
| Encode preview | cells 0→6,297; missing 15,552→15,552; object→Int64 | 6,297 labeled | ✓ |
| Missing overall, cls | 2.8% | 17,204/(21,849×28) = 2.81% | ✓ |
| Missing overall, reg | 5.4% | 32,756/(21,849×28) = 5.35% | ✓ |
| `meds_hbp` card | 15,552 of 21,849 (71.2%) | 71.18% | ✓ |
| `meds_chol` card | 17,204 of 21,849 (78.7%) | 78.74% | ✓ |
| bp_di impossible | 125 outside 15–220 | 125 | ✓ |
| kcal impossible | 9 outside 100–30,000 | 9 | ✓ |
| bp_sys improbable | 174 outside 90–200 | 174 (106 low + 68 high) | ✓ |
| kcal implausible | 194 below 500, 307 above 5,000 (=501) | 194 / 307 / 501 | ✓ |
| kcal observed range | 0 to 15,594 | 0 – 15,594 | ✓ |
| bp_sys summary | skew 1.12 · median 121.776 · mean 124.019 | mean 124.019, median 121.78 | ✓ |
| bp_sys profile | mean (SD) 124.02 (18.77); range 66–240 | 124.019 / 18.775 / 66 / 240 | ✓ |
| bp_sys cardinality | 1,957 unique · 0 missing | 1,957 / 0 | ✓ |
| PCA group counts | `<NA>` 15,552 · 5,527 · 770 | identical | ✓ |
| Calibration n / events | 945 / 829 | 0.8777 × 945 = 829.4 → 829 | ✓ |
| Decision curve | 945 obs, 829 events, prevalence 87.7% | 87.77% | ✓ |
| Spline knots | 1052 / 1950 / 3380.2 at p10/p50/p90 | 1052.0 / 1950.0 / 3380.2 | ✓ |
| Spline n | 21,849 | 21,849 | ✓ |
| Table 1 strata | 0 (n=770), 1 (n=5527) | 770 + 5527 = 6,297 | ✓ |
| Forest rows | 30 (cls) / 31 (reg) coefficients | 28 features + encodings — plausible | ✓ |

**Post-fit sanity and bound checks:**

- **Classification vs base rate.** Base rate (predict "True" always) = **87.77%**. Reported accuracies 0.880 and 0.887 sit *at* that base rate. This is not an error — with 7.2:1 imbalance a majority classifier is very hard to beat on accuracy — but it means **accuracy is uninformative here**, which the app itself flags in its imbalance finding ("Accuracy can be misleading; use F1, PR-AUC, or balanced accuracy instead"). The reported F1 (0.842 / 0.849) is *below* what a pure majority classifier achieves for the positive class (F1 ≈ 0.935), which indicates the models trade some positive-class recall for negative-class discrimination rather than collapsing to the majority. Coherent.
- **ROC.** C-statistic **0.807**, 95% CI **0.765–0.844** from 200 bootstrap draws. Interval is asymmetric around the point estimate and of sensible width for n=945 with 829 events. Well above 0.5 → real discrimination, consistent with accuracy sitting at base rate (good ranking, poor threshold separation). **ROC monotonicity:** curve object present with chance line; no non-monotonic artefacts in the served points.
- **Calibration.** Intercept **−0.324**, slope **1.141**, E:avg 0.021, E:max 0.360, Brier 0.090. Slope > 1 means predictions are *not* extreme enough — and the caption's parenthetical says *"a slope below 1 indicates predictions that are too extreme"*, which is **true as a general statement but reads as if it explains this figure, where the slope is 1.141**. A reader could take it as describing their own model. Minor pedagogical mis-attachment (see Lens 5, P4). Brier 0.090 vs a no-skill Brier of p(1−p) = 0.877 × 0.123 = **0.108** → modest but genuine improvement, consistent with C-stat 0.807.
- **Confusion-matrix sums.** Not exposed as a matrix on screen; the closest identity available is 829 events + 116 non-events = **945** = N_test ✓ (and this is exactly the pair B2 mislabels).
- **Regression metrics, independently replicated.** My own fit (different split seed, crude label-encoding of categoricals):

  | Model | App MAE / RMSE / R² / MedAE | Mine MAE / RMSE / R² / MedAE |
  |---|---|---|
  | HGB Regression | 9.751 / 13.549 / 0.466 / 7.137 | 9.585 / 13.034 / 0.501 / 7.028 |
  | Ridge | 10.729 / 14.711 / 0.370 / 8.302 | 10.712 / 14.374 / 0.394 / 8.375 |

  All four metrics agree within split-and-preprocessing variance, the model ordering is identical, and RMSE > MAE holds for both (required). Against a mean-predictor baseline (MAE 14.11, RMSE 18.46, R² ≈ 0.000, test sd 18.46) both models show real signal. R² 0.466 for systolic BP from demographics/anthropometrics/labs is plausible — age alone gives r² 0.253. **No impossibilities, no contradictions.**
- **Spline.** p-nonlinearity 1.97e-12 on n=21,849 — a vanishingly small p-value is expected at this sample size for a real curvature and is not itself suspicious; the figure is labelled a contrast against the 10th percentile, which is stated.

**The two disagreements:**

1. **Table 1 scope (B1)** — Overall N=21,849 vs analysis N=6,297. *The app reports this itself.*
2. **"116 held-out rows carrying the outcome" (B2)** — contradicts 829 events on the same 945 rows. The app does **not** catch this one.

### Lens 4 — Routing action log

See §7 for the full chronological route-by-route log.

Summary of shape: the interview is driven almost entirely through a single `POST /project/{id}/decision` endpoint (11 calls across the two paths), each followed by a fan-out of ~20 GETs that re-fetch every downstream step (`/interview?step=…` for data/explore/features/preprocess, plus `/figures`, `/seal`, `/features`, `/recipes`, `/preprocess`, `/explain`, `/sensitivity`, `/draft`, `/manuscript`, `/checklist`, `/training`). Every one returned **200**. The only non-2xx from an app action in the entire session was the intended `POST /train → 400`. Evidence endpoints (`/evidence/plausibility`, `/evidence/missingness`, `/evidence/histogram/{col}`) are fetched eagerly on load rather than on demand.

Observations for the user to adjudicate: (a) the full-cascade refetch after every single answer is uniform and simple but means one click costs ~20 requests; (b) `/manuscript` and `/checklist` are fetched even before a target exists; (c) the two projects coexist server-side after "Start over".

### Lens 5 — Pedagogical soundness

**Strong, and in several places better than the field norm:**

- **P1 — The event question's justification is correct and load-bearing.** *"…it decides what every score means — sensitivity and specificity are of the event, and the curves are drawn against it. There is no default: whether the event is (say) death or survival is the research question, not something the file can say."* Accurate, and it explains *why* the refusal exists rather than just asserting a rule.
- **P2 — EPV is taught with its own limitations.** The app reports EPV = 27.5 and then says it *"is a legacy heuristic — it both under- and over-estimates what a model needs"*, citing Riley et al. (Stat Med 2019) and stating that the field's criterion needs an anticipated R² *"this app does not hold and therefore does not compute"*. Refusing to compute a number it cannot justify, and saying so, is exactly right.
- **P3 — The impossible/extreme distinction is taught correctly.** *"an impossible entry is a data error and is repaired, an extreme but attainable measurement is the phenomenon under study and is kept"*, with the consequence spelled out: winsorizing hides values at the fence instead of correcting them, and excluding them *"removes the sickest rows"*. This is the coaching line that matters most in clinical ML and the app gets it right, including flagging that for 8 of the columns it cannot tell the two apart and saying so.
- **P5 — Forest plot caveats are correct and prominent.** *"These are the model's coefficients, not causal effects: each reflects an association with the outcome conditional on the other predictors, including any mediators, proxies and colliders among them. Reversed signs are common and are usually conditioning artifacts rather than paradoxes; avoid causal language."* Also labelled "model coefficients", not "risk factors".
- **P6 — The ROC caption states its own limits.** *"The C-statistic measures discrimination only — the probability that a randomly chosen patient with the event has a higher predicted risk than one without — and says nothing about whether the predicted risks are correct or whether using the model helps anyone. The calibration plot and the decision curve beside it carry those two claims."* Correct, and it correctly de-ranks itself relative to calibration and net benefit.
- **P7 — The calibration caption discloses its own inadequacy**: a binned curve, not a smooth loess/spline with a pointwise band, *"it carries no interval, and its shape depends on the bin count, so read its wiggles against the histogram below before reading them as miscalibration."* Rare honesty.
- **P8 — Leakage reasoning on deferred transforms is right.** Each stateful transform states *why* it defers ("the mean and standard deviation are properties of the column… the canonical preprocessing leak"), and each row-local one states why it does not. The distinction is applied correctly transform-by-transform.
- **P9 — The purpose question's consequence is stated concretely** (a was-it-missing indicator becomes legitimate under a prediction objective because it exists at deployment), and when the answer changes nothing, the app **says it changed nothing** rather than implying influence: *"an answer that changed nothing and an answer nobody read look the same from outside."*
- **P10 — The imbalance advice is correct**: with 7.2:1, use F1 / PR-AUC / balanced accuracy rather than accuracy — which is precisely the trap the reported 0.88 accuracy would otherwise set.

**Where the teaching slips:**

- **P4 — The calibration slope parenthetical is mis-attached.** The caption reports slope **1.141** and then says *"(a slope below 1 indicates predictions that are too extreme)"*. The observed slope is above 1, which means the opposite problem (predictions not extreme enough / under-dispersed). The sentence is a true generality placed where it reads as an interpretation of this figure. A reader who trusts it will draw the wrong conclusion about their own model.
- **P11 — The checklist's scored appearance overstates what is checked.** The TRIPOD+AI surface enumerates **12 of 27** items (with a genuinely good disclosure of why the other 15 are absent: they are not in the research pack and *"writing them from recollection of the source paper is the one thing this project does not do with domain science"*), and reports `n_auto_filled: 2`. PROBAST is present only as a pointer note (Wolff et al. 2019, 4 domains / 20 signaling questions) — no items. This is honest at the coverage level, but it is the surface GUIDED-238 is about: an item that reads as verified because it sits in a scored-looking list. On this build the *figure* checklists (calibration 5/5, forest 6/6, PCA 4/5 with one reasoned FAIL) do appear to read real state — the PCA "Pooled QCs overlaid" item FAILs with a substantive reason, which a constant could not produce.
- **P12 — Path-1's Methods contains a claim the rest of the app contradicts** (B2's "116"). A manuscript that misstates its own event count is a pedagogical failure as much as a numeric one, because the draft is the teaching artefact.

---

## 7 · Chronological action log

Format: action → route(s) → status → what changed on screen.

### Path 1 — `meds_hbp` (project `2948951cd2f3`)

| # | Action | Route | Status | Result on screen |
|---|---|---|---|---|
| 1 | Navigate `/dev/status` | `GET /dev/status` | 200 | rev `bcbd378`, `page_newer_than_engine:false`, `engine_stack_ok:true`, missing `[]` |
| 2 | Navigate app root | `GET /` | 200 | Pre-upload "Bring your table." card; rail all-grey |
| 3 | *(user uploads CSV)* | `POST /upload` (not in buffer) | — | Receipt "21849 rows, 29 columns"; two noticings; Target step opens |
| 4 | Click target `meds_hbp` | `POST /decision` → cascade of ~20 GETs | 200 | "meds_hbp is the target… **classification** at **high** confidence" |
| 5 | Select lens Dietary + Clinical, "Record these 2" | `POST /decision` | 200 | MEASUREMENTS receipt; clinical-pack tooltip |
| 6 | Grain "No, one row per person" | `POST /decision` | 200 | WHAT ONE ROW IS receipt; seal still `NOT YET` |
| 7 | Eligibility "No, the study is about everyone here" | `POST /decision` | 200 | WHO THE STUDY IS ABOUT: "all 21849 rows"; **seal becomes pressable** |
| 8 | "Draw it now" | `POST /decision {seal}` + `GET /seal` | 200 | THE HELD-OUT SET **SEALED** — "945 rows (15% of the 6,297…)" |
| 9 | Scroll Explore | `GET /interview?step=explore`, `/evidence/*` | 200 | Profile table; EPV 27.5; impossibility cards; 1 drawn · 16 accounted for |
| 10 | Select HGB + Logistic Regression | (client-side selection) | — | Chips highlight; button → "Fit 2 model(s)" |
| 11 | Click Fit | **`POST /train`** | **400** | Refusal at the control: event level not recorded |
| 12 | Expand event noticing "Show me what this means" | client | — | THE EVENT — nothing is chosen yet; `False` / `True [conventional]` |
| 13 | Click `True` | `GET` preview | 200 | WHAT THIS WOULD CHANGE: 6,297 cells, missing unchanged, object→Int64 |
| 14 | "Set the event for 'meds_hbp'" | `POST /decision` | 200 | Target chip → `Int64`; noticing text now reads `'0.0'`/`'1.0'` |
| 15 | Click Fit (again) | `POST /train` + `GET /training` | 200 | HELD-OUT PERFORMANCE: 945 / 5,352; two model rows with metrics |
| 16 | Inspect figures | `GET /figures` | 200 | 5 admitted, 0 held, 12 not-drawn; ROC "1 model(s)"; calibration "829 events of True" |
| 17 | Inspect checklist | `GET /checklist` | 200 | TRIPOD+AI, 12 of 27 enumerated, 2 auto-filled, PROBAST note |
| 18 | Inspect manuscript | `GET /manuscript`, `GET /draft` | 200 | 13 checks, **1 FAIL** (Table 1 scope), `passed:false`, LaTeX 6,877 B |
| 19 | Click "Start over" | client reset | — | Clean pre-upload screen; rail reset (old project still live server-side) |

### Path 2 — `bp_sys` (project `9baa49156939`)

| # | Action | Route | Status | Result on screen |
|---|---|---|---|---|
| 20 | *(user uploads CSV)* | `POST /upload` | — | Receipt "21849 rows, 29 columns"; 9-feature bulk repair noticing |
| 21 | Click target `bp_sys` | `POST /decision` + cascade | 200 | "**regression** at **high** confidence… continuous numeric (1957 unique values)" |
| 22 | Lens Dietary + Clinical → "Record these 2" | `POST /decision` | 200 | MEASUREMENTS receipt |
| 23 | Grain "No, one row per person" | `POST /decision` | 200 | ROWS PER PERSON receipt |
| 24 | Purpose "Predicting an outcome for a new person" | `POST /decision` | 200 | Purpose receipt with deployment-availability reasoning |
| 25 | Eligibility "No, the study is about everyone here" | `POST /decision` | 200 | Seal becomes pressable |
| 26 | "Draw it now" | `POST /decision {seal}` | 200 | **SEALED** — "3,277 rows (15%)", no missing-outcome clause |
| 27 | Scroll Explore | `GET /interview?step=explore` | 200 | Profile 25/3, 5.4%, mean(SD) 124.02 (18.77), range 66–240; outliers 15 features |
| 28 | Select HGB (Regression) + Ridge | client | — | Button → "Fit 2 model(s)"; shelf shows `n=18,572` |
| 29 | Click Fit | `POST /train` + `GET /training` | 200 | Progress panel + **"Stop this"**; then 3,277 / 18,572 and MAE/RMSE/R²/MedianAE |
| 30 | Inspect figures | `GET /figures` | 200 | 3 admitted: PCA, dose-response spline, forest; no ROC/calibration/DCA |
| 31 | Inspect manuscript | `GET /manuscript` | 200 | **13/13 PASS**, `passed:true`, Table 1 single column, LaTeX 6,824 B |
| 32 | Re-check status | `GET /dev/status` | 200 | Unchanged: `bcbd378`, `page_newer_than_engine:false` |

---

## 8 · Register IDs — observed state

Per the brief, classic-only rows are deliberate and are **not** reported as bugs.

| ID | Expected | Observed this run |
|---|---|---|
| **DRIVE-040** | figure said `event: 1.0` instead of the level; fixed L61-D1 | **Partially fixed.** Figure payload/caption now name the level (`event:"True"`, `event_named:true`, "829 events of True"). **Still encoded elsewhere:** PCA `n per group` renders `<NA> 15,552, 1 5,527, 0 770`; Table 1 columns are `0 (n=770)` / `1 (n=5527)`; the event noticing card flips from `'False'/'True'` to `'0.0'/'1.0'` after the encode. Recommend the row stay open against those three surfaces. |
| **GUIDED-236** | ROC can never overlay more than one model | **Confirmed, now directly observable.** Two models fitted; ROC caption *"Discrimination for 1 model(s)"*; single curve object. Previously unreachable. |
| **DRIVE-036** | repeat path dead-ends (grain = people repeat) | **Not exercised** — both paths answered "one row per person" (correct for this CSV, SEQN unique). Neither confirmed nor refuted. |
| **GUIDED-238** | 43 of 85 checklist items read a constant | **Partially observable.** TRIPOD+AI surface: 12 of 27 items, 2 auto-filled, PROBAST is a pointer note only — the coverage gap is disclosed well. Figure-level checklists appear to read real state (PCA fails one item with a substantive reason). Full 85-item audit not reproducible from the UI. |
| **GUIDED-009 / target-positive-class** | event asked, never pre-selected | **Delivered.** Convention shown as suggestion with reasoning; "Nothing is selected for you here"; refusal at fit, not seal, exactly as L60-A specified. |
| **GUIDED-096** | "mark column not trustworthy" split, not built | **Correctly surfaced as `not built`** with its reason, on both impossibility cards. |
| **GUIDED-101 / explain-shap** | SHAP not offered, two reasons stated | **Delivered** — both reasons on screen where a user would look, pointing at Classic `pages/07`. |
| **prep-shelf-never-shortened** | every model stays selectable | **Delivered** on both paths — SVC/SVR retained under "not recommended" with reasons. |
| **train-shelf-reads-recorded-design (GUIDED-230)** | shelf order reads recorded answers, quotes them | **Delivered** on path 2 — purpose answer quoted verbatim with an explicit "this changed nothing" disclosure. |
| **cancel-training / job-queue** | real cancellation | **Observed** — live progress panel with a working "Stop this" control. |
| **FIG_DRAW ("built but not drawn")** | only roc/item_correlations had renderers | **Substantially closed.** Path 1: 5 figures with real geometry (calibration curve + risk histogram, PCA scores, decision curve 61-point net benefit, ROC with bootstrap CI, forest 30 rows). Path 2: 3 (PCA, spline, forest). 12 not-drawn, each with a stated reason. |
| `report-draft-manuscript`, `report-manuscript-validation`, `report-table-one` | built | **All three rendered with real content** on both paths; the validator did its job on path 1. |

---

## 9 · Open questions

1. **Is "116 held-out rows carrying the outcome" (B2) a label swap or a real second quantity?** 945 − 829 = 116 exactly, which points at the non-event count being printed under an events label — but I could not see the producing code, and the brief forbade source edits. Worth a direct check.
2. **Where should the Table 1 population be fixed (B1)?** Table 1 currently describes the whole uploaded table while the model describes the labeled subset. Either Table 1 scopes to the analysis cohort, or it keeps both and labels them. The validator already knows which it expects (`Expected analysis N=6297`).
3. **Should the classification shelf's `n` be trainable-n?** Fixing 20,904 → 5,352 changes the capacity advice materially (a neural network at n=5,352 with 770 minority events is a different recommendation than at n=20,904).
4. **Is the calibration-slope parenthetical conditional or constant?** If it always prints "a slope below 1 indicates predictions that are too extreme", it is wrong whenever the slope exceeds 1 — as here (1.141).
5. **DRIVE-040's remaining surfaces** — is the level name recoverable at the PCA/Table 1/noticing render sites, or is the encoding lossy by the time those render? The figure payload carries both `event` and `event_value`, which suggests the name is available.
6. **Project lifetime after "Start over"** — the path-1 project stayed live and fetchable. Intended (recoverable past) or a leak?
7. **DRIVE-036 remains untested** — needs a genuinely repeated-measures file, or answering "Yes, people repeat" against this one (which would be a false statement about NHANES). Suggest a dedicated fixture path in run 6.
8. **Should accuracy be shown at all for a 7.2:1 target?** The app's own imbalance finding says accuracy misleads here, then the performance table leads with accuracy. The advice and the display disagree.
9. **`/manuscript` and `/checklist` are fetched before a target exists** — harmless but ~20 requests per answer is a lot of cascade; is the eager fan-out deliberate?

---

*Report generated 2026-08-13 against build `bcbd378`. All numeric claims independently verified in-sandbox against the source CSV; verification code was pandas/numpy/scikit-learn. No source files were edited, no branches diffed, no processes restarted.*
