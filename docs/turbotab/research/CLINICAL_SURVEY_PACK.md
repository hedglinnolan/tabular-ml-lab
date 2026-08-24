# Clinical measurements & labs, and survey instruments — research specification

Two packs in one file because they share a spine and because the same dataset (NHANES, any cohort with
a PROM) usually carries both. Same structure as the other packs: by modeling step, and within each by
**diagnostic**, **coaching**, **presentation**, with **SETTLED** · **CONVENTION** · **DISPUTED** on
every recommendation.

**Research caveat.** Publisher domains (Wiley, OUP, PMC, arXiv, BMJ, COSMIN) were blocked by the
session egress policy, so specifics come from search-surfaced abstracts, author-hosted preprints, and
guideline text. Every load-bearing claim is attributed to a named source so it can be verified before
shipping.

---
---

# Part A · Clinical measurements and labs

## A1 · Import and structure

### A1.1 Unit harmonization

**Diagnostic.** Detect **multimodal distributions within a single named analyte column** — the
signature of mixed units. Concretely: fit a 2-component Gaussian mixture on log values; flag when
component means differ by a ratio near a known conversion factor. Test against: glucose ×18.0
(mmol/L→mg/dL), creatinine ×88.4, total cholesterol ×38.67, triglycerides ×88.57, bilirubin ×17.1,
calcium ×4.0 (approx), hemoglobin ×10 (g/dL→g/L). Height in inches vs cm (mode near 66 vs 168); weight
lb vs kg (165 vs 75); temperature °F vs °C (98.6 vs 37).

Parse unit strings from adjacent `*_unit` columns and normalize against **UCUM**, the standard LOINC
uses for unit coding. Flag free-text variants (`mg/dl`, `MG/DL`, `mg%`). Cross-tabulate implied units
by site if a site/lab ID exists. **Detect analyte columns with no unit information at all — the common
and dangerous case.**

**Coaching.**

> *"Column `glucose` has two clusters whose centers differ by a factor of ~18 — the exact
> mmol/L-to-mg/dL conversion. This almost always means two sites or two eras reported different units
> into the same field. Harmonize before modeling: a mixed-unit predictor is not a noisy predictor, it
> is a variable whose meaning changes between rows, and no amount of regularization repairs that."*

> *"No unit column was found for 7 lab analytes. Units are the single most common silent corruption in
> pooled clinical data. Please confirm units per analyte against the source data dictionary — TurboTab
> will not guess."*

**★ Design rule: never auto-convert. Detect, propose, require explicit confirmation.** **[SETTLED that
heterogeneity must be resolved; auto-conversion is an embarrassment risk]** — auto-conversion fails
catastrophically for analytes whose conversion factor depends on molecular weight (many drug levels,
hormones), where LOINC/UCUM services themselves return errors for a missing molecular weight.

**Presentation.** **Unit-audit table**: analyte | n | unit string(s) observed | implied unit(s) from
distribution | median per implied unit | action taken. Exactly what a reviewer asking "how did you
harmonize multi-site labs?" needs. · **Pre/post harmonization density overlay** per affected analyte,
log-x, with a marker at the conversion-implied second mode.

### A1.2 ★ Reference ranges vs physiological plausibility — the distinction that matters

Maintain **two separate, differently-purposed bound sets**:

1. **Reference interval (RI)** — the central 95% of a healthy reference population, i.e. the
   2.5th–97.5th percentiles, per CLSI EP28-A3c (which also sets a minimum reference sample of n=120).
   *By construction ~5% of healthy people fall outside.* **Use only for annotation. Never for
   exclusion.**
2. **Physiological plausibility bounds** — values incompatible with a living patient. Use for flagging
   as suspected data error.

Suggested plausibility bounds (widely used in EHR cleaning; still **[CONVENTION]** — Kahn et al. note
that plausibility limits are institution- and observation-specific):

| Measure | Plausible | Typical adult RI |
|---|---|---|
| SBP | 30–300 mmHg | 90–120 |
| DBP | 10–200 mmHg | 60–80 |
| Heart rate | 20–300 bpm | 60–100 |
| Temperature | 25–45 °C | 36.5–37.5 |
| SpO₂ | 40–100 % | 95–100 |
| Adult weight | 20–400 kg | — |
| Adult height | 100–250 cm | — |
| BMI | 10–100 kg/m² | 18.5–24.9 |
| Serum Na | 90–200 mmol/L | 135–145 |
| Serum K | 1.0–10.0 mmol/L | 3.5–5.0 |
| Glucose | 10–2000 mg/dL | 70–99 fasting |
| Creatinine | 0.1–30 mg/dL | 0.6–1.2 |
| Hemoglobin | 2–25 g/dL | 12–17 |
| WBC | 0.0–500 ×10⁹/L | 4–11 |
| Platelets | 0–3000 ×10⁹/L | 150–400 |

Additional structural checks from **Kahn et al.'s harmonized data-quality framework** (*eGEMs* 2016):
**conformance** (type/format/vocabulary), **completeness**, **plausibility** — atemporal (is the value
believable) and **temporal (is the *trajectory* believable)**, and **uniqueness**.

**Temporal plausibility:** adult height changing >5 cm between visits; weight changing >30% in <30
days; sex recorded inconsistently across encounters; labs timestamped before birth or after death.

**Repeated-digit / default-value detection:** excess mass at 120/80, 98.6, 0, and at round numbers — a
documented EHR artifact reflecting value preference and manual entry rather than measurement.

**⚠ Pediatric and growth data:** never apply adult bounds. Use age-and-sex-specific modified z-score
flags (the CDC growth-chart biologically-implausible-value approach), not fixed cutoffs.

**Coaching.**

> *"4 systolic BP values are below 30 mmHg. These are physiologically impossible in a living outpatient
> and are almost certainly entry errors — TurboTab recommends setting them to missing and reporting the
> count. **This is different from the 812 values above 140 mmHg, which are abnormal but real and must
> be kept:** excluding abnormal-but-possible values would remove the sickest patients and bias the
> model toward the healthy."* **[SETTLED]**

> *"Reference intervals are not disease thresholds. The interval is the central 95% of a healthy
> reference population, so about 1 in 20 healthy people fall outside it by construction, and intervals
> differ by laboratory, instrument, assay, age, sex and over time. TurboTab will draw reference bands
> on plots for orientation, but recommends against converting labs to normal/abnormal flags for
> modeling."*

> *"Dichotomizing a continuous lab at its reference limit discards information. Royston, Altman &
> Sauerbrei (*Stat Med* 2006) showed dichotomizing a normally distributed continuous predictor costs
> about as much power as throwing away a third of your sample, and choosing a data-derived 'optimal'
> cutpoint introduces serious bias. Model the lab continuously, with a restricted cubic spline if
> nonlinearity is plausible."* **[SETTLED]**

**⚠ Embarrassment risk:** do not present the EHR's own `abnormal_flag` column as portable. Those flags
are lab- and instrument-specific and change when a lab re-validates an assay.

**Presentation.** **Distribution-with-reference-band figure** — density per analyte with the RI as a
shaded band, plausibility bounds as dashed rules, and a rug showing individual extremes. Shows cohort
case-mix *and* data quality simultaneously; facet by outcome for extra value. · **Data-cleaning
provenance table**: variable | n implausible | rule applied | n set to missing | n remaining. Reviewers
and PROBAST assessors look for this specifically.

### A1.3 Lab value formats and censored values

**Diagnostic.** Detect columns typed as text that are >80% numeric-parseable — a near-certain sign of
embedded qualifiers. Detect censoring tokens: `<`, `>`, `≤`, `≥`, `<LOD`, `<LLOQ`, `>ULOQ`, `TNTC`,
`undetectable`, `not detected`, `trace`, `QNS`, `hemolyzed`, `see comment`, `pending`, `cancelled`.
Report the **left-censoring fraction** and the **detection limit** per analyte (usually inferable as
the modal `<X` value). Detect thousands separators, European decimal commas, scientific notation.
Detect result columns mixing quantitative and qualitative results (a troponin column with both `0.04`
and `negative`).

**Coaching.**

> *"18% of `hs-CRP` values are recorded as `<0.3` — left-censored at the assay's limit of detection.
> TurboTab has not substituted a number. Simple substitution (LOD, LOD/2, LOD/√2) biases estimates and
> the bias grows with the censored fraction. With 18% censored, TurboTab recommends either (a) maximum
> likelihood / censored regression assuming a lognormal distribution, or (b) distribution-based
> multiple imputation of the sub-LOD values."*

> **[DISPUTED at low censoring fractions]** *"Below roughly 5–10% censoring, LOD/√2 substitution is
> widely used and rarely changes conclusions materially; above ~20% it is not defensible. TurboTab uses
> 10% as its warning threshold, which is a convention, not a proven cutoff."*

> **The prediction/inference asymmetry:** *"If your goal is prediction rather than an unbiased
> exposure–outcome estimate, the censoring point is itself informative and stable — 'below detection'
> is a real, reproducible piece of clinical information available at deployment. A censoring indicator
> plus a substituted value is often defensible here even though it would not be for inference."*
> **[CONVENTION — well-argued, not formally settled]**

> *"`TNTC` and `QNS` are not censoring at a detection limit — they are measurement failures. Treat them
> as missing, not as extreme values."*

**Presentation.** **Censoring summary table**: analyte | n | % below LOD | LOD value(s) | % above ULOQ |
handling chosen. · **Censored-aware distribution plot** — histogram of detected values with a
distinctly-colored bar at the LOD whose height is the censored count, labeled `<0.3 (n=214)`. **Never
silently plot substituted values as if measured — that is the visual equivalent of fabricating data.**

---

## A2 · ★ Missing data — where TurboTab differentiates itself

Generic ML tooling gets this wrong, and the correct answer is field-specific.

### Diagnostic

- Per variable: missingness rate and, crucially, **missingness–outcome association** — the crude event
  rate among those with the value present vs absent, plus a univariable OR/HR for the missingness
  indicator itself.
- **Missingness–missingness structure.** Co-missingness clustering recovers panel structure — the
  entire liver panel missing together means the panel was not ordered.
- **Missingness–covariate association.** Is missingness predicted by age, site, admission type,
  calendar time?
- **Calendar-time discontinuities** in missingness (an assay introduced in year 3; a site joining in
  year 5) — these break the "mechanism is stable" assumption.
- **Measurement frequency** as a candidate variable: number of times ordered, time since last order.
- **Classify each variable into a missingness archetype:** *not ordered* (informative — clinician
  judged it unnecessary, or the patient was too well or too sick) · *ordered but not resulted* (QNS,
  cancelled — closer to MCAR) · *not applicable* (pregnancy tests in men; structural zeros — must not
  be imputed) · *not captured* (present in care, absent from the extract) · *skip-logic / conditional*.

### Coaching

> *"`ferritin` is missing for 62% of patients, and patients missing it have a 3.1% event rate versus
> 11.4% among those with a value. In EHR data, missingness usually means the test was not ordered, and
> that decision is made by a clinician who has already seen the patient. **Missingness here is a fact
> about the patient's clinical trajectory, not a defect in the measurement.** This is MNAR, and
> standard multiple imputation — which assumes MAR — does not fix it."*
> **[SETTLED: Agniel, Kohane & Weber, *BMJ* 2018;361:k1479 found the mere presence of a lab order was
> significantly associated with survival for 233 of 272 (86%) tests, and the *timing* of ordering was
> more predictive than the result itself for 118 of 174 (68%) tests.]**

**★ The prediction / inference split — the single most useful thing this pack says:**

> *"What you should do depends on what you are doing.*
> *For **prediction** — a model that works at the bedside — the missing-indicator method is legitimate
> and often improves performance. The indicator carries the clinician's implicit judgment, and at
> deployment the same indicator will be observable. **This is the opposite of the advice you'll find in
> an inference textbook.***
> *For **inference** — an unbiased estimate of the association between ferritin and the outcome — the
> missing-indicator method is known to give biased estimates and should not be used; multiple
> imputation or a principled MNAR model is required."*
> **[SETTLED for the inference half; well-supported for the prediction half — Sisk, Sperrin, Peek, van
> Smeden & Martin, *Stat Methods Med Res* 2023.]**

**The mandatory caveat that separates a credible tool from a naive one:**

> *"Missing indicators buy predictive performance at the cost of an extra assumption: that the
> missingness mechanism is stable across development, validation and deployment. Sperrin and colleagues
> warn this assumption is generally dubious and especially so in routinely collected health data,
> because the propensity to order a test varies by provider, by setting, and over time as guidelines
> change. If your model is deployed at a hospital that orders ferritin routinely, an indicator trained
> where it was ordered selectively will silently mislead. Recommendation: use indicators if you want
> them, but (a) declare the assumption in the manuscript, (b) validate externally at a site with
> different ordering practice, and (c) check whether missingness rates drift over calendar time in your
> own data."*

> *"The *Annals of Internal Medicine* review 'Addressing Missingness in Predictive Models That Use
> Electronic Health Record Data' (2025;178(10)) organizes MNAR methods into **selection models** (model
> the mechanism jointly with the outcome) and **pattern-mixture models** (stratify by pattern and
> combine). Both require untestable assumptions; the recommended practice is **sensitivity analysis
> across plausible assumptions**, not selecting one 'correct' method."*

**Anti-patterns to name explicitly:**

1. **Complete-case by default.** With clinically-driven missingness, complete cases are a systematically
   sicker (or better-worked-up) subpopulation. Flag when listwise deletion drops >10% of rows.
2. **Mean/median imputation.** Understates variance, destroys the distribution, indefensible in a
   manuscript. **[SETTLED as bad]**
3. **Imputing with the outcome excluded from the imputation model.** Biases associations toward the
   null. The outcome *must* be in the imputation model. **[SETTLED]**
4. **Imputation fit before splitting.** Parameters learned on the full dataset = leakage. Imputation
   must be fit inside the resampling loop.
5. **Imputing structurally-missing values** (pregnancy test in men, PSA in women).
6. **Reporting "no missing data"** after having dropped rows upstream. Track and surface rows lost at
   each step.
7. **Multiple imputation with no plan for deployment.** MI is hard to apply to a single incoming
   patient. A regression-imputation or explicit deployment-time rule is the pragmatic alternative — and
   must be pre-specified.

> *On the number of imputations:* *"The classic m=5 assumed a low fraction of missing information and
> point estimation only. For confidence intervals, m≥20 is now typical (White, Royston & Wood 2011), and
> von Hippel's two-stage rule (pilot run → estimate FMI → set m so Monte Carlo error is small relative
> to the SE) is more defensible. TurboTab defaults to m = max(20, ceil(100 × FMI))."* **[CONVENTION]**

### Presentation

- **Missingness matrix / UpSet plot** — rows = patients sorted by pattern, columns = variables. Reveals
  panel co-missingness at a glance. Publication-grade when variables are ordered by missingness rate
  and the top 10–15 patterns are labeled with counts.
- **★ Missingness-vs-outcome forest plot** — one row per variable, the odds ratio for the outcome
  associated with *being missing*, 95% CI, reference line at 1. **This figure is the visual argument for
  informative missingness and is unusual enough in clinical papers to be a genuine contribution.**
- **★ Missingness-over-calendar-time line plot** — % missing per variable by month/quarter. Exposes
  assay introductions, site onboarding, COVID-era discontinuities. **Essential evidence for or against
  the missing-indicator stability assumption.**
- **Table**: variable | n missing (%) | archetype | event rate if present | event rate if missing |
  handling method.

---

## A3 · Table One / cohort description

**Diagnostic.** Auto-classify each variable (continuous-normal, continuous-skewed via |γ₁|>1 or
Shapiro–Wilk with a sample-size caveat, binary, nominal, ordinal, date, identifier, free text) · detect
the natural stratifier (the outcome for prediction papers, the exposure for comparative papers, the
development/validation split for TRIPOD papers, or site) · compute per stratum n (%), mean (SD) for
symmetric, median [IQR] for skewed, **and n missing per variable per stratum** · compute the
**standardized mean difference (SMD)** for every variable: continuous `(x̄₁−x̄₂)/√((s₁²+s₂²)/2)`; binary
`(p̂₁−p̂₂)/√((p̂₁(1−p̂₁)+p̂₂(1−p̂₂))/2)`; multi-category via the Yang & Dalton multinomial extension ·
**detect the Table 1 fallacy setup** — a randomized design where a p-value column would be actively
wrong.

**Coaching.**

> *"TurboTab has built Table 1 with standardized mean differences rather than p-values. SMDs describe
> the magnitude of imbalance and are independent of sample size; p-values in a baseline table answer a
> question nobody asked — whether the observed groups differ from a hypothetical random draw — and they
> systematically mislead by declaring trivial differences significant in large cohorts and important
> differences non-significant in small ones. The STROBE explanation-and-elaboration document states
> that significance tests should be avoided in descriptive tables."* **[SETTLED among methodologists;
> still routinely violated]**

> *"An SMD below 0.10 is the conventional threshold for negligible imbalance."* **[CONVENTION — widely
> used since Austin's propensity-score work, but explicitly a rule of thumb; recent work (Hripcsak et
> al., *Stat Med* 2025) shows the 0.1 rule behaves poorly at small n, where chance imbalance routinely
> exceeds 0.1 without meaningful bias. **Show the SMD value and let the reader judge — never stamp
> PASS/FAIL.**]**

> *For randomized data:* *"This looks like a randomized comparison. Baseline significance testing is a
> documented error — the 'Table 1 fallacy.' A study of 765 RCTs found a median of 10 baseline tests per
> trial, with 3% interpreted as evidence of imbalance; by construction those are false positives,
> because randomization guarantees the null is true. TurboTab has suppressed the p-value column."*
> **[SETTLED]**

> *"Report medians with IQRs, not means with SDs, for right-skewed labs (CRP, ferritin, troponin, LOS,
> D-dimer). A mean CRP of 42 mg/L with SD 90 tells the reader nothing about a typical patient."*
> **[CONVENTION, near-universal]**

> *"Put the missing count in Table 1 itself, per variable per column. Reviewers ask for it every time,
> and burying it in the text is the most common revision request on a cohort paper."* **[CONVENTION]**

**What belongs in Table 1.** Demographics — age (continuous *and* clinically meaningful bands if used
in the model), sex, race/ethnicity **with an explicit note on how it was ascertained** and, for the
U.S., that race is a social not biological variable. **⚠ If a race-based clinical equation appears** (old
MDRD/CKD-EPI with a race coefficient, race-adjusted spirometry), flag it: the **2021 race-free CKD-EPI
creatinine equation** is the current NKF–ASN Task Force recommendation. · **All candidate predictors in
the model** — non-negotiable. · The outcome, and for time-to-event, **median follow-up by reverse
Kaplan–Meier**, events, censoring counts. · Setting/site/era if multi-site or multi-year. · Comorbidity
summary (Charlson/Elixhauser) if administrative. · Key exclusions' effect belongs in the flow diagram,
not the table.

**Presentation.** Variable rows with indented category levels; columns = Overall | Stratum A | Stratum B
| SMD | Missing n(%). Units in the row label. **Consistent decimal places, and never more precision than
the measurement supports** — reporting mean age as 62.4173 is an instant credibility loss. Footnote
every abbreviation; footnote which summary statistic is used and why; **explicitly state the denominator
when missingness varies by row** (% of non-missing vs % of total — pick one, state it, be consistent). ·
Also offer a **"Table 1 as a figure"** — a small-multiple panel of distributions by stratum, increasingly
accepted and far more informative, **[CONVENTION: offer it as a supplement, not a replacement.]**

---

## A4 · EDA and presentation

What clinical prediction papers are expected to contain, in the order a reader encounters them.

### A4.1 · Participant flow diagram

Track row counts through every filter: source extract → each eligibility criterion **separately** →
missing-data exclusions → analysis set → development/validation split, recording the *reason* and the
*count* at each node.

> *"Every observational prediction paper needs a flow diagram; STROBE expects it and reviewers will
> request it if absent. **The critical detail is that each exclusion gets its own box with its own n** —
> a single 'excluded n=4,312' box tells the reader nothing about who is missing from your model's
> population."* **[CONVENTION, effectively mandatory]**

Publication-grade = every arithmetic path adds up exactly, and the final n matches Table 1's n.
**TurboTab should *verify* the arithmetic and refuse to render an inconsistent diagram.**

### A4.2 · Distribution plots against reference ranges

Covered in A1.2. Publication-grade = reference band shaded and **labeled with its source** (which lab,
which population), plausibility bounds visually distinct from the reference band, faceted by outcome,
log scale where the analyte spans orders of magnitude, n per panel.

### A4.3 · ★ Calibration plot — the single most important figure in a clinical prediction paper

**Diagnostic.** Compute Van Calster et al.'s **calibration hierarchy** (*J Clin Epidemiol* 2016):

- **Mean calibration / calibration-in-the-large** — observed event rate vs mean predicted risk; O:E = 1.
- **Weak calibration** — calibration **intercept** (target 0) and **slope** (target 1) from a logistic
  model of the outcome on the logit of predicted risk. **Slope <1 ⇒ overfitting** (predictions too
  extreme); slope >1 ⇒ underfitting.
- **Moderate calibration** — the flexible calibration curve: among patients predicted R%, is the
  observed rate R%? Estimated by loess or restricted cubic splines with a pointwise 95% CI.
- **Strong calibration** — correct within every covariate pattern. Acknowledged as utopian; **do not
  report it as achievable.**

Also compute **E:avg**, **E:max**, **E:90**, the **estimated calibration index**, the **Brier score**
and **scaled Brier / index of prediction accuracy**.

**Coaching.**

> *"Calibration is what clinical prediction cares about most. A model with excellent discrimination but
> poor calibration will systematically over- or under-treat: if it tells a clinician a patient's risk is
> 30% when it is really 8%, the AUC is unaffected but the treatment decision is wrong. Van Calster and
> colleagues call calibration 'the Achilles heel of predictive analytics' precisely because it is the
> property most often omitted from papers."* **[SETTLED]** (*BMC Medicine* 2019)

> *"Report calibration intercept and slope with CIs **and** the flexible curve. The two numbers alone
> can hide a curve that is badly wrong in the clinically relevant risk range while averaging out to a
> slope of 1."*

**Anti-patterns.** The **Hosmer–Lemeshow test** — deprecated: it depends arbitrarily on the number of
groups, has poor power in small samples, rejects everything in large samples, and gives a p-value where
a magnitude is needed. **[SETTLED as deprecated among prediction methodologists; still commonly seen, so
TurboTab should explain rather than merely refuse.]** Also **10-decile binned calibration plots** —
better than nothing, worse than a smooth curve, sensitive to bin choice.

**★ What makes a calibration plot publication-grade:**

1. Predicted risk on x, observed proportion on y, same scale, **square aspect ratio**.
2. **45° reference line**, dashed, labeled "ideal".
3. **Flexible (loess/spline) curve with a shaded 95% pointwise band.**
4. **A distribution of predicted risks along the bottom** — a spike histogram or rug, ideally split by
   outcome (events above the axis, non-events below). **Without it the reader cannot tell whether the
   curve's wild behavior at 0.8 is based on 3 patients or 300. This is the detail most often missing
   and most often requested by reviewers.**
5. **Annotation box:** calibration intercept (95% CI), slope (95% CI), C-statistic (95% CI), E:avg,
   E:max, n, events.
6. **Do not truncate the axis to hide the sparse tail** — show it and let the confidence band widen.
7. External validation: one panel per validation cohort, shared axes.

### A4.4 · ROC curve

C-statistic with 95% CI (DeLong or bootstrap). For survival models, Harrell's C or Uno's C **with the
truncation time stated.**

> *"Report the C-statistic as a measure of **discrimination only** — the probability that a randomly
> chosen patient with the event has a higher predicted risk than one without. It says nothing about
> whether the predicted risks are correct, and nothing about whether using the model helps anyone."*
> **[SETTLED]**

> *"An ROC curve is a conventional figure but a low-information one. If space is tight, a calibration
> plot and a decision curve are worth more. TurboTab will render all three."* **[CONVENTION with
> methodological backing]**

**Anti-patterns.** Reporting **accuracy**, **F1**, or a single confusion matrix at threshold 0.5 for a
clinical risk model. The 0.5 threshold is arbitrary and almost never the clinically relevant one;
accuracy is dominated by prevalence. **[SETTLED as inappropriate for risk models; ubiquitous in
ML-flavored clinical papers and a common reviewer target.]** Also "Youden index optimal cutpoint"
presented as if the harms of false positives and false negatives were equal — they essentially never
are.

**Presentation.** Square, diagonal chance line, AUC with CI annotated lower-right, multiple models
overlaid with a legend (not separate panels). Label axes **"Sensitivity"** and **"1 − Specificity"**
(clinical convention) rather than TPR/FPR.

### A4.5 · ★ Decision curve analysis — the clinical-utility figure

Net benefit at threshold `p_t`: **NB = (TP/n) − (FP/n) × (p_t/(1−p_t))** (Vickers & Elkin, *Med Decis
Making* 2006), computed across a threshold range plus the "treat all" and "treat none" strategies.

> *"A decision curve answers the question the AUC cannot: would a clinician using this model be better
> off than treating everyone or no one? **Choose the threshold range from the clinical decision, not
> from the data** — if the intervention is a cheap statin, clinicians might act at 5% risk; if it is a
> biopsy, at 20%. Please confirm the default range brackets where reasonable clinicians would
> disagree."* **[SETTLED as the recommended clinical-utility method]**

> *"A model that fails to beat 'treat all' across the entire plausible threshold range has no
> demonstrated clinical value regardless of its AUC. This is the most common and most useful negative
> finding a decision curve produces."*

**Anti-patterns.** Plotting net benefit across 0–100% threshold, where the curves are dominated by
ranges no clinician would use, and reading off a maximum. Also comparing curves without accounting for
the fact that **a miscalibrated model produces a misleading decision curve — DCA presupposes
calibration.**

**Presentation.** x = threshold probability (secondary axis for cost:benefit ratio if space permits),
y = net benefit; **y-axis lower bound around −0.05 rather than 0 so the reader can see curves going
negative**; thick line = treat none (horizontal at 0), thin line = treat all, distinct line per model;
legend inside; **shaded region marking the clinically plausible threshold range**; a rug or secondary
panel showing the distribution of predicted risks.

### A4.6 · Kaplan–Meier and time-to-event

**Diagnostic.** Detect time + event-indicator pairs. Compute **median follow-up by reverse KM**. Detect
administrative censoring dates. **Detect competing risks** — if the event indicator has >2 levels, or a
death/other-cause column exists.

> *"If competing risks are present, **1 − Kaplan–Meier overestimates cumulative incidence**, because KM
> treats the competing event as censoring — as if those patients could still develop the outcome later,
> which they cannot. Report the **Aalen–Johansen cumulative incidence function** instead. Use
> cause-specific hazard models for etiology and Fine–Gray subdistribution models for prediction."*
> **[SETTLED; ⚠ a very common error — a tool that produced KM curves for cardiovascular events in an
> elderly cohort without flagging competing mortality would be badly wrong.]**

> *"Report median follow-up using the reverse Kaplan–Meier method, not the median of observed follow-up
> times, which is biased by early events."* **[SETTLED]**

> *"For risk *prediction*, report predicted risk at specified clinically meaningful horizons (1, 5, 10
> years) — not just a hazard ratio. Calibration must then be assessed at each horizon."*

**Presentation.** **Numbers-at-risk table beneath the plot is mandatory** — and if you include one, drop
the censoring tick marks, which clutter without adding information the risk table doesn't already give.
**Choose cumulative incidence (rising from 0) rather than survival (falling from 1) when events are
rare** — a curve living between 0.95 and 1.00 wastes the panel; if using survival, do not truncate the
y-axis without a clear break marker. Include Greenwood 95% bands. **Truncate the x-axis where the
at-risk set becomes too small** (a common convention: stop where <10–20% of the original cohort remains)
**and say so.** Log-rank p only if a comparison is the point; for a prediction paper, curves by
predicted-risk quartile are more informative than by a single covariate.

### A4.7 · Forest plot

**⚠ The critical warning most tools omit:**

> *"These are the model's coefficients, not causal effects. A predictor's coefficient in a prediction
> model reflects its association with the outcome **conditional on the other predictors** — including
> mediators, proxies, and colliders. Reversed signs are common and are usually not paradoxes but
> conditioning artifacts. TurboTab labels this figure 'model coefficients,' not 'risk factors,' and
> recommends you avoid causal language in the accompanying text."* **[SETTLED among methodologists; the
> conflation is endemic in the applied literature.]**

> *"Put all continuous predictors on a comparable scale (per SD, or per clinically meaningful unit like
> per 10 mmHg) before plotting them together, and state which. A forest plot mixing 'per year of age'
> with 'per mmol/L of creatinine' invites false visual comparison."* **[CONVENTION]**

**Presentation.** **Log-scale x-axis for ratio measures — non-negotiable**; a linear axis makes OR 0.5
and OR 2.0 look asymmetric when they are equal and opposite. Vertical reference at 1; point size
proportional to precision; reference categories shown as rows with no estimate; the numeric estimate and
CI printed in a right-hand column; predictors grouped by domain and **ordered meaningfully, not by
significance.**

### A4.8 · ★ Prediction stability plots — the modern addition

Refit the **entire** modeling pipeline (including any variable selection) in B bootstrap resamples
(Riley & Collins recommend on the order of 1000), apply each bootstrap model to the original data, and
produce: **prediction instability plot** (bootstrap predictions vs original-model predictions, one point
per patient per bootstrap); **MAPE**; **calibration instability plot** (all bootstrap calibration curves
overlaid); classification and decision-curve instability plots. (Riley & Collins, *Biometrical Journal*
2023.)

> *"A single point estimate of the C-statistic hides how much your model depends on the particular
> patients you happened to sample. A prediction instability plot shows, per individual, how much the
> predicted risk would have moved had you drawn a different sample. Wide vertical spread means an
> individual patient's predicted risk is not trustworthy even if average performance looks fine."*
> **[Emerging — [CONVENTION] rather than expected, but it is what good reviewers now ask for.]**

**Presentation.** Scatter with the 45° line, semi-transparent points (α ≈ 0.02), MAPE annotated;
calibration instability as many thin grey curves plus the original in bold.

### A4.9 · Fairness and subgroup performance

Recompute C-statistic, calibration intercept/slope, O:E and net benefit within **pre-specified**
subgroups (sex, race/ethnicity, age band, site, insurance/deprivation index), reporting subgroup n and
events.

> *"TRIPOD+AI explicitly asks for evaluation in relevant subgroups and for a discussion of fairness.
> **Report per-subgroup calibration, not just per-subgroup AUC** — a model can discriminate equally well
> in two groups while being systematically miscalibrated in one, which is the mechanism by which
> prediction models perpetuate disparities."* **[SETTLED as a reporting expectation post-TRIPOD+AI]**

**Presentation.** Subgroup table (n / events / C / calibration slope / intercept per row) plus a paired
forest-style plot of calibration slope with CI by subgroup, reference line at 1. **Flag subgroups where
events <30 as uninterpretable rather than reporting a point estimate.**

---

## A5 · Modeling

### A5.1 Calibration first

Enforce that calibration is computed for **every** model, including tree ensembles and neural nets.

> *"Rank your models on calibration and clinical utility, not on AUC alone. Two models with identical
> AUC can differ enormously in whether their probabilities are usable."* **[SETTLED]**

> *"Random forests and boosted trees frequently produce miscalibrated probabilities out of the box. If
> you use them, either report the calibration curve honestly or apply post-hoc recalibration (Platt
> scaling / isotonic regression) fit on **held-out** data — never on the training data, and never on the
> test data you then report performance from."*

### A5.2 ★ Class imbalance — the flagship warning

> *"TurboTab has **not** applied class-imbalance correction, and recommends you don't. Van den
> Goorbergh, van Smeden, Timmerman & Van Calster (*JAMIA* 2022;29(9):1525–1534) showed that random
> undersampling, random oversampling and SMOTE all led to **poor calibration — strong overestimation of
> the probability of belonging to the minority class — without improving discrimination.** Any apparent
> gain in sensitivity/specificity was reproducible simply by shifting the classification threshold, with
> no distortion of the probabilities. Their conclusion: outcome imbalance is not a problem in itself;
> correcting it may worsen model performance, and inaccurate probabilities reduce clinical utility
> because treatment decisions become ill-informed."*
> **[SETTLED for risk prediction models. Replicated for ML methods by Carriero et al., *Stat Med* 2025.]**

> *"If what you actually want is to classify at a particular operating point, set the threshold to
> reflect the clinical trade-off — that is what decision curve analysis is for — and leave the
> probabilities alone."*

> **The nuance TurboTab should get right:** *"Rare outcomes do create a real problem — but it is
> small-sample overfitting, not imbalance per se, and the remedy is penalization (ridge, LASSO, Firth's
> correction for separation) and adequate sample size, not resampling."* **[SETTLED]**

**⚠ Embarrassment risk, inverted:** a tool that ships with SMOTE in the default pipeline — as many
AutoML tools do — is broadcasting that it does not know the clinical prediction literature. **This is a
differentiator.**

### A5.3 Discrimination vs calibration

> *"Discrimination is a **ranking** property and is invariant to any monotone transformation of the
> predictions. Calibration is a **magnitude** property. You need both, and they are not substitutes.
> Report: C-statistic with CI, calibration intercept and slope with CIs, the flexible calibration curve,
> O:E ratio, Brier score (and scaled Brier), and net benefit over a clinically-motivated threshold
> range."* **[SETTLED — essentially TRIPOD+AI's performance expectation]**

### A5.4 Sample size

Compute Riley et al.'s minimum sample size for model development (`pmsampsize` / `pmvalsampsize`).
Inputs: number of candidate predictor **parameters** — *count parameters, not variables*; a 4-knot
spline is 3 parameters, a 5-level factor is 4 — anticipated prevalence, and anticipated model R².

- **Binary outcomes** (Riley et al., *Stat Med* 2019, Part II) — three criteria: global shrinkage ≥0.9
  (≤10% overfitting); absolute difference ≤0.05 between apparent and adjusted Nagelkerke R²; precise
  estimation of overall outcome risk (margin of error ≤0.05).
- **Continuous outcomes** (Part I) — four criteria, adding precise estimation of the residual SD and of
  the mean predicted value.
- **External validation** (Riley et al., *Stat Med* 2021) — targets precision of the calibration slope,
  C-statistic and O:E ratio; Collins et al.'s rule of thumb is a minimum of ~100 events **and** ~100
  non-events, with ~200 events preferred.

> *"The events-per-variable rule of 10 is a legacy heuristic that both under- and over-estimates
> requirements depending on prevalence and expected model strength; use the criteria-based
> calculation."* **[SETTLED that EPV≥10 is superseded; the newer thresholds 0.9 and 0.05 are themselves
> CONVENTION — chosen, not derived.]**

> ⚠ *"**Candidate predictors count toward sample size even if they are later dropped.** If you screen 40
> variables and keep 8, you must size for 40 — data-driven selection consumes degrees of freedom whether
> or not it appears in the final model. This is the sample-size mistake PROBAST most often catches."*

### A5.5 Modeling practice

> *"Avoid univariable pre-screening of predictors by p-value. It is one of PROBAST's explicit
> high-risk-of-bias signals: it discards variables that matter only in combination, and it invalidates
> the p-values in the final model."* **[SETTLED]**

> *"Avoid stepwise selection. It produces unstable variable sets, biased coefficients, and confidence
> intervals with wrong coverage. Prefer pre-specification on clinical grounds, or penalized regression
> which shrinks rather than selects abruptly."* **[SETTLED; note LASSO's own instability in small
> samples — see the stability plots in A4.8.]**

> *"Model plausible nonlinearity with restricted cubic splines; 3–5 knots at fixed quantiles is the
> standard default (Harrell, *Regression Modeling Strategies*)."* **[CONVENTION with strong backing]**

> *"Internal validation must resample the **entire** modeling pipeline — imputation, transformation,
> selection, tuning. Bootstrap optimism correction is the recommended default (it uses all the data and
> has smaller variance than a single split); repeated k-fold CV is acceptable. **A single train/test
> split is the weakest option and is discouraged at typical clinical sample sizes.**"* **[SETTLED that
> the full pipeline must be inside the loop; the bootstrap-vs-CV preference is CONVENTION.]**

**Anti-patterns.** Tuning on the test set · reporting apparent performance without optimism correction ·
leakage via post-baseline predictors (a lab drawn after the outcome occurred) · **immortal time bias**
from an index date defined by a later event · and — specific to EHR — using a variable recorded *because*
the outcome happened (a "palliative care consult" predicting death).

---

## A6 · Reporting

**TRIPOD+AI** (Collins et al., *BMJ* 2024;385:e078378) is a **27-item checklist** superseding TRIPOD
2015 and harmonizing reporting across regression and machine learning, with a companion **TRIPOD+AI for
Abstracts**. Items TurboTab can substantially auto-populate: title/abstract identification as a
development and/or validation study · source of data, setting, eligibility, and **participant flow** ·
outcome definition and blinding of outcome assessment to predictors · **predictor definitions and
measurement, including timing relative to the index** · **handling of missing data with the mechanism
discussed, not just the method named** · **sample size justification** · model-building procedure
including selection, and **all hyperparameters and the tuning procedure** · **performance: discrimination
AND calibration AND clinical utility** · **model presentation sufficient for others to compute
predictions** (full coefficients and intercept, or the model object) · **fairness/subgroup evaluation** ·
open-science items (data, code, protocol, funding, conflicts) · limitations and intended use, users and
setting.

Also surface **PROBAST** (Wolff et al., *Ann Intern Med* 2019) — 4 domains (participants, predictors,
outcome, analysis) and **20 signaling questions**; the Analysis domain covers study size, handling of
continuous predictors, missing data, selection of predictors, overfitting/optimism, and performance
measures. *"If your paper is later included in a systematic review, this is the instrument that will be
applied to it. Reading it before you write is cheap insurance."*

**Presentation.** **★ Auto-generated TRIPOD+AI checklist table** — item | where addressed | auto-filled
text | ⚠ needs your input. Journals increasingly require the completed checklist as a submission file;
producing it automatically is a genuine time saver and a strong feature. · Auto-generated Methods
paragraph in the field's register. · **Model card**: intended use, population, predictors and their
required units, performance with CIs, subgroup performance, known limitations, and **deployment-time
missing-data policy.**

---
---

# Part B · Survey and questionnaire instruments

## B1 · Import and structure

### B1.1 Detecting Likert blocks

**Diagnostic.** Candidate items: integer-valued columns with few distinct values (typically 2–11),
identical or near-identical value sets across columns, low cardinality relative to n. **Block detection**
combines three signals: **name pattern** (a shared prefix with a numeric suffix — `phq_1..phq_9`,
`Q12a..Q12h`, `SF36_03`, `bdi.item.14` — parsed with a regex family tolerating zero-padding);
**identical response-value support**; and **column adjacency plus elevated inter-item correlation**.

**Response-scale detection:** infer min, max, step and number of points from the observed support
**union across the block**, not per item — a rarely-endorsed extreme category may be absent from a
single item. Detect anchoring: 1–5 vs 0–4 vs −2–+2 vs 1–7 vs 0–10 (NRS) vs 0–100 (VAS).

**★ Sentinel-code detection — the highest-yield check in this pack.** Flag values that break the observed
contiguous run: `7`, `8`, `9`, `77`, `88`, `99`, `-1`, `-8`, `-9`, `98`, `999`, SAS special missing
(`.a/.b/.c`), `NA` strings. **In a 1–5 item, a `9` is not "extremely agree," it is "don't know" or
"refused."** A `0` in a 1–5 item may be "not applicable."

**Known-instrument fingerprinting.** Match block length + response range + name prefix against a library:
PHQ-9 (9 items, 0–3), GAD-7 (7, 0–3), PHQ-2, EQ-5D-5L (5, 1–5) + EQ-VAS (0–100), SF-36/SF-12, PROMIS
short forms, HADS (14, 0–3), CES-D (20, 0–3), Rosenberg (10, 1–4), Big Five Inventory, MoCA/MMSE,
AUDIT-C, ISI, PSQI, EORTC QLQ-C30, WHOQOL-BREF, Barthel, mRS, PSS (10 or 14, 0–4). **A hit should trigger
the instrument's published scoring algorithm and missing-data rule, not a generic one.**

**Careless/insufficient-effort responding (C/IER) screening:** **longstring** (max run of identical
consecutive responses), **individual response variability** (per-person SD across items; ~0 =
straightlining), **even–odd consistency**, **psychometric antonyms/synonyms** correlation, **Mahalanobis
distance**, and response time if available.

**Coaching.**

> *"TurboTab detected a 9-item block (`phq_1`–`phq_9`) on a 0–3 scale matching the PHQ-9. It will use
> the PHQ-9's published scoring (sum, range 0–27, cut-points 5/10/15/20) rather than a generic scale
> score. **Confirm this is the PHQ-9 and not a modified version** — modified instruments must not be
> scored with the original's cut-points."* **[⚠ Fingerprint matching that is confidently wrong produces
> a wrong published cut-point in a manuscript. Always present the match as a hypothesis.]**

> *"Item `q14` contains the value 9 while all other items in this block range 1–5. This is almost
> certainly a 'don't know'/'refused' sentinel code, not a response. TurboTab has **not** recoded it. If
> it is a sentinel, recoding it to missing is essential — a 9 treated as a response would shift this
> item's mean by [X] and propagate into every scale score and every model."* **[SETTLED that sentinels
> must be recoded; ⚠ never auto-recode — some legitimate scales do run 0–9.]**

> *"'Don't know' is not automatically the same as 'missing.' On attitude items it is often a substantive
> response, and dropping it can bias the sample toward people with formed opinions."* **[DISPUTED — a
> genuine, unresolved survey-methodology question.]**

> *"[N] respondents ([X]%) gave the same answer to ≥[k] consecutive items, and [M] have zero variance
> across the entire block — consistent with careless responding. The literature (Ward & Meade, *Annu Rev
> Psychol* 2023) recommends **pre-specifying screening rules** rather than choosing them after seeing
> results — post-hoc exclusion is a researcher degree of freedom. Recommendation: flag, report the
> count, and run the analysis both with and without."* **[CONVENTION; thresholds are arbitrary — no
> consensus longstring cutoff exists, so never present one as authoritative.]**

**Presentation.** **Block-detection summary table**: block | items | n | response range | scale points |
instrument match (confidence) | sentinel codes found | items with out-of-range values. · **★
Response-value audit heatmap** — items on rows, observed distinct values on columns, cell = count, with
out-of-support values highlighted. **Instantly surfaces sentinel codes and mis-keyed items.**

### B1.2 ★ Reverse-coded items — the hard constraint

**Diagnostic.** Corrected **item–rest correlation** per item; flag negatives · polychoric inter-item
matrix, flagging items whose correlations with the block are predominantly negative · compare item
**text** (if labels exist) against a lexicon of negation and valence-reversal markers (`not`, `never`,
`rarely`, `difficult`, `unable`, `failed`, `dis-`, `un-`, `-less`) · check whether a codebook declares
reverse-coded items · **after any user-declared reversal, re-run the check** and warn if an item is
still negatively correlated (double-reversal, or a genuinely misfitting item).

**Coaching — the central sentence of this section:**

> *"TurboTab will not infer reverse-coding from correlations, and neither should you. A negative
> item–rest correlation has at least four incompatible explanations: (1) the item is reverse-worded and
> needs recoding; (2) it was **already** reverse-scored upstream, and reversing again would corrupt it;
> (3) it is negatively worded and loads on a **method factor** rather than the construct — the
> well-documented finding that scales mixing positive and negative wording become two-dimensional, where
> the second dimension is wording, not content; (4) it genuinely does not belong to the scale.
> **Correlations cannot distinguish these.** Reverse-coding must come from the instrument's published
> scoring key or your codebook."* **[SETTLED — a real hard limit. A tool that auto-reverses is a tool
> that will silently invert a published scale.]**

> *"After reversal, item `x` still correlates −0.12 with the rest of its scale. Either it was already
> reversed in the source data, or it does not belong to this scale."*

> *"Even correctly reverse-coded items can produce an artifactual 'negative wording' factor. If your
> factor analysis finds a second factor consisting exclusively of the reverse-worded items, that is a
> method effect, not a substantive dimension — say so rather than naming it."* **[SETTLED as a
> phenomenon; how to model it (correlated residuals, method factor, bifactor) is DISPUTED.]**

**Presentation.** **Reverse-coding audit table**: item | text (truncated) | item–rest r (raw) | reversal
declared? | item–rest r (after reversal) | status — **re-rendered after every declared change.** ·
**Item–rest correlation dot plot**, items sorted, reference lines at 0 and at 0.30 (Nunnally &
Bernstein's conventional minimum). **[0.30 is CONVENTION, not a law.]**

---

## B2 · Scale construction

**Diagnostic.** Detect whether the instrument has a **published scoring algorithm** and whether it uses
a sum, a mean, a weighted score, or a **norm-based T-score** (PROMIS scores are T-scores with mean 50,
SD 10 in a reference population — computed via IRT, not by summing) · per-respondent item completion
rate within each scale · scale distribution: min, max, mean, SD, skew, **% at floor**, **% at ceiling** ·
unidimensionality (parallel analysis on the polychoric matrix, first-to-second eigenvalue ratio,
single-factor CFA fit).

**Coaching.**

> **Sum vs mean [CONVENTION with a clear decision rule]:** *"Use whichever the instrument's published
> scoring specifies; **if it has established clinical cut-points (PHQ-9 ≥10, GAD-7 ≥10, HADS ≥8), you
> must use the published sum**, because rescaling silently invalidates every cut-point in the
> literature. For an ad-hoc or newly-derived scale, the **mean** is generally preferable: it stays on the
> item metric (a score of 3.2 is interpretable as 'a bit above the midpoint'), and it is comparable
> across respondents who answered different numbers of items."*

> ⚠ *"Do not invent a rescaling of an established instrument. Converting a PHQ-9 sum to a 0–100 scale
> makes it uncomparable to every published paper and every clinical guideline that uses it."*

> **Prorating:** *"Prorating (person-mean substitution) is the near-universal convention and is what most
> manuals specify. But it is only defensible when the scale is unidimensional and the items are roughly
> interchangeable, because it implicitly assumes the missing item would have scored like the
> respondent's average. It also **understates the uncertainty** in the resulting score."*

> **Threshold:** *"The dominant convention is a 'half rule' — prorate if the respondent answered at least
> half the items, otherwise set the scale score to missing. Individual instruments override this: EORTC
> QLQ-C30 requires more than half of a multi-item scale's items; PHQ-9 conventions typically allow up to
> 2 of 9 missing; SF-36 uses a half-scale rule. TurboTab applies the instrument-specific rule when it
> recognizes the instrument, and 50% otherwise."* **[CONVENTION — the 50% threshold is not empirically
> derived and TurboTab should say so.]**

> **The better method, stated honestly:** *"With substantial item-level missingness, the methodologically
> preferred approach is **multiple imputation at the item level**, not prorating or imputing the total.
> Eekhout et al. (*J Clin Epidemiol* 2014) found item-level MI performed best across missingness
> patterns; imputing composite scores overestimated standard errors when >50% of participants had
> missing data, though at n≤500 and ≤10% missingness the two performed similarly. With many scales the
> item count can exceed what the imputation model supports, in which case **passive imputation or parcel
> summaries** are valid alternatives (Eekhout et al. 2018)."* **[SETTLED that item-level MI is preferred
> where feasible; the practical cutover to parcels is CONVENTION.]**

> **When a scale is not unidimensional:** *"Parallel analysis on the polychoric correlations of
> `wellbeing_1..12` suggests 2 factors, and a single-factor CFA fits poorly. Summing all 12 items
> produces a score that mixes two constructs — a given total can be reached by very different response
> patterns, so **the score is not interpretable.** Options in order of preference: (a) score the
> published subscales separately; (b) report a bifactor model and use omega-hierarchical to quantify how
> much of the total-score variance is attributable to a general factor; (c) if the general factor is
> weak, report the subscales and abandon the total."* **[SETTLED that a multidimensional total is
> uninterpretable; which remedy to pick is DISPUTED.]**

**Anti-patterns.** **Deleting items to maximize Cronbach's alpha** ("alpha if item deleted" mining) —
capitalizes on chance, inflates the reported alpha, and typically *narrows* content validity. **[SETTLED
as bad practice]** · standardizing items before summing without saying so · computing a total when the
manual specifies subscales only.

**Presentation.** **Scale construction table**: scale | k items | scoring (sum/mean/T) | theoretical
range | observed range | n scored | n set missing by the rule | % prorated | mean (SD) | median [IQR] |
% floor | % ceiling | ω / α with CI. · **Score distribution histogram** per scale **with theoretical
min/max as axis limits** — not data-driven limits; **this is what makes floor/ceiling visible** —
vertical lines at published cut-points, % at floor and ceiling annotated on the panel.

---

## B3 · Reliability and validity

**Diagnostic.** **Cronbach's α** with a bootstrap or analytic 95% CI · **McDonald's ω** — and be specific
about *which*: ω_total (variance attributable to all common factors) vs ω_hierarchical (the general
factor only, in a bifactor model). Compute **categorical/ordinal ω** from a polychoric-based,
WLSMV-estimated CFA when items have ≤5 categories · **AVE**, **coefficient H**, **glb** as supplementary
· corrected item–rest correlations (flag <0.30) · **mean inter-item correlation** (target 0.15–0.50 per
Clark & Watson 1995) · **α-if-item-deleted** — compute, but present with a warning, not as a to-do list ·
test–retest: **ICC(2,1) or ICC(A,1)** with CI, plus **SEM** and **smallest detectable change** ·
structural validity: single-factor and published-structure CFA with **CFI, TLI, RMSEA (with CI), SRMR**.

**Coaching.**

> **The critique, stated accurately:** *"Cronbach's α is the most over-reported statistic in the
> questionnaire literature. Sijtsma (*Psychometrika* 2009) showed that α is a **lower bound** to
> reliability that can be arbitrarily far below the true value, and — crucially — that **α is not a
> measure of internal consistency or unidimensionality**, despite being universally described that way.
> A high α is compatible with a strongly multidimensional scale, and simply increasing the number of
> items raises α regardless of structure. α equals reliability only under **tau-equivalence** — equal
> true-score loadings across items — which real scales essentially never satisfy."* **[SETTLED]**

> *"McDonald's ω is the mainstream recommended alternative: model-based, no tau-equivalence requirement,
> so it behaves correctly for the congeneric items that are the norm. McNeish (*Psychological Methods*
> 2018) argues for abandoning α in favor of ω; Flora (*AMPPS* 2020, 'Your Coefficient Alpha Is Probably
> Wrong, but Which Coefficient Omega Is Right?') gives the practical workflow and — importantly —
> explains that 'omega' names a **family**, so you must state which one and which underlying factor
> model you fitted."* **[SETTLED that ω is generally preferable]**

> **The honest dissent — a good tool includes it:** *"This is not unanimous. Raykov & Marcoulides
> ('Neither Cronbach's Alpha nor McDonald's Omega,' *Psychometrika* 2021) and Savalei & Reise ('Don't
> Forget the Model in Your Model-Based Reliability Coefficients,' *Collabra* 2019) point out that ω
> inherits every assumption of the factor model used to compute it — a misspecified model yields a wrong
> ω with unearned authority — and that when the model fits, α and ω often differ trivially. Practical
> recommendation: **report both, with confidence intervals, along with the factor model ω is based on
> and its fit.**"* **[DISPUTED — present both positions, do not pick a winner.]**

> **Thresholds — present as conventions, never as pass/fail:** *"α or ω ≥0.70 is the usual minimum for
> group-level research; ≥0.90 (some say ≥0.95) is expected before scores are used for **individual**
> clinical decisions, because at α=0.70 an individual's confidence interval is very wide. Terwee et al.'s
> criteria (*J Clin Epidemiol* 2007) call for α between **0.70 and 0.95** — **the upper bound matters:
> α above 0.95 usually signals redundant, near-duplicate items and narrowed content validity, not an
> excellent scale.**"* **[CONVENTION — the 0.70 threshold traces to a casual remark in Nunnally and has
> no derivation. TurboTab must not display a green check for 0.71 and a red X for 0.69.]**

> **★ The claim TurboTab should make loudly:** *"Reliability is a property of **scores in a sample**, not
> of an instrument. 'The scale has been validated' is not a statement that can be true. Report α/ω
> computed in **your** data, with a CI, even for a well-established instrument."* **[SETTLED and
> routinely violated — a genuinely useful coaching line.]**

> *"'Ordinal alpha' (α on polychoric correlations) is sometimes recommended for Likert items, but
> Chalmers (*EPM* 2018) argues it estimates the reliability of a latent continuum that was never
> observed, not of the score you actually computed. TurboTab reports categorical ω from an ordinal CFA
> instead, and notes the choice."* **[DISPUTED, leaning against ordinal alpha.]**

> **Validity:** *"Reliability is necessary but not sufficient. A perfectly reliable score can measure the
> wrong thing. If you are presenting a new or adapted scale you also need **structural validity**,
> **construct validity** (does it correlate with what theory says it should — state hypotheses *before*
> looking), and, for a PROM, **content validity**, which COSMIN rates as the most important property."*
> **[SETTLED]**

**Anti-patterns.** Reporting α to three decimals as if precise (report a CI) · one α for a
multidimensional instrument · citing the developers' α instead of your own · deleting items until α
clears 0.70 · interpreting a CFA that fits badly.

**Presentation.** **Reliability table**: scale | k | α [CI] | ω_total [CI] | ω_h | mean inter-item r |
range of item–rest r | ICC [CI] | SEM | SDC. · **Item statistics table**: item | text | n | % missing |
mean (SD) | % floor | % ceiling | item–rest r | standardized loading (SE) | α-if-deleted. · CIs
everywhere; **the estimator named** (polychoric/WLSMV vs Pearson/ML); the factor model behind ω specified
with its fit indices; the sample stated.

---

## B4 · ★ Ordinal vs interval — the long-running dispute

**The most genuinely disputed thing in either pack, and the app's credibility depends on not pretending
otherwise.**

**Diagnostic.** Number of response categories per item (the key moderator) · skewness and floor/ceiling
per item and per scale · whether compared groups have **different response distribution shapes** (this is
when metric models fail hardest) · whether the target is an **item** or a **multi-item scale score** ·
whether the variable is a **predictor** or an **outcome** · variance heterogeneity across groups.

**The case for ordinal treatment.** Liddell & Kruschke (*JESP* 2018;79:328–348, "Analyzing ordinal data
with metric models: What could possibly go wrong?") surveyed every article mentioning "Likert" in JPSP,
Psychological Science, and JEP:General and found **100% of those analyzing ordinal data used a metric
model.** Their simulations show metric treatment can produce inflated false-alarm rates, reduced
detection of real effects, distorted effect sizes, and — most seriously — **systematic inversions**,
where the metric analysis reports the opposite ordering of means from the truth. They show the same for
interactions and trends, and explicitly demonstrate that **averaging multiple items into a scale does not
fix it.** They recommend ordered-probit (cumulative link) models. **The inversion result is the strongest
argument in the literature.**

**The case for metric treatment.** Norman (*Adv Health Sci Educ* 2010;15:625–632, "Likert scales, levels
of measurement and the 'laws' of statistics") reviews evidence back to the 1930s that parametric methods
are robust to violations of normality and interval-level measurement, and argues that for typical Likert
applications parametric analysis will not "get the wrong answer." Carifio & Perla make the parallel
argument that the ordinal objection applies to individual *items*, whereas summated *scales* behave close
enough to interval for practical purposes.

**Where the disagreement actually is.** The two camps are largely arguing about different regimes.
Liddell & Kruschke's failures are most severe with **few categories, strong skew/floor/ceiling, and
unequal distribution shapes across groups.** Norman's robustness holds best with **many categories,
roughly symmetric distributions, similar shapes across groups, and multi-item scale scores.**

**TurboTab's decision rule — presented as the app's opinion, clearly labeled:**

| Situation | Recommendation | Status |
|---|---|---|
| Single Likert **item** as an **outcome** | Ordinal (cumulative link / ordered probit) | Strongly recommended |
| Item with ≤4 categories, any role | Ordinal | Strongly recommended |
| Item/scale with pronounced floor or ceiling (>15–20% at an endpoint) | Ordinal | Strongly recommended |
| Groups with visibly different distribution **shapes**, not just locations | Ordinal — **exactly where inversions arise** | Strongly recommended |
| Multi-item **scale score** from ≥5 items, ≥5 categories each, roughly symmetric, no floor/ceiling | Metric defensible; report ordinal as sensitivity | Defensible either way |
| Likert item as a **predictor** among many | Metric usually low-risk; consider splines or dummies | Low stakes |
| **Descriptives** of individual items | Frequencies/percentages, **never a mean** | **[SETTLED — a mean of 3.4 on a 1–5 agreement item is not interpretable and hides bimodality]** |

> *"Whichever you choose, run the other as a sensitivity analysis and say so. **If the substantive
> conclusion is the same under both, the dispute is moot for your paper and you can say that in one
> sentence — which is a much stronger position than picking a side.**"*

**Anti-patterns.** Reporting the mean and SD of a single Likert item as the primary result · Pearson
correlation between two 5-point items (use polychoric) · a t-test on an item with 60% at the ceiling ·
treating a 0–10 NRS as continuous without checking the enormous digit-preference spikes at 0, 5 and 10.

**Presentation.** **Sensitivity-analysis comparison table**: effect | metric estimate [CI] | ordinal
estimate (OR) [CI] | conclusion unchanged? — a compact, honest, reviewer-disarming exhibit. · For ordinal
outcomes, plot **predicted category probabilities** across the predictor range, far more interpretable
than a proportional-odds coefficient.

---

## B5 · EDA and presentation

### B5.1 ★ Diverging stacked bar chart — the field-standard Likert figure

> *"The diverging stacked bar chart is the standard graphic for Likert data (Heiberger & Robbins, *J Stat
> Softw* 2014;57(5); implemented as `likert()` in the R **HH** package). Percentages agreeing extend
> right of a zero line, percentages disagreeing extend left, and the neutral category is **split down the
> middle and shown in a neutral color** so it straddles zero. This is what lets a reader compare items at
> a glance, which separate histograms or a table of means cannot do."* **[CONVENTION, near-universal;
> genuinely the field standard.]**

> **The one genuinely contested design choice:** *"How to treat the neutral midpoint is disputed.
> Splitting it across zero preserves total bar length and makes the agree/disagree split visually honest,
> but slightly distorts the apparent size of both wings. Placing the whole neutral category on one side,
> or excluding it and reporting it separately, are both defensible. TurboTab defaults to splitting and
> states the choice in the caption."* **[DISPUTED design question, low stakes, but the caption must say
> which.]**

> *"For an even-numbered (forced-choice) scale with no midpoint, diverging bars work cleanly. **For scales
> with no natural agree/disagree polarity — frequency scales, item-specific response options — a diverging
> chart imposes a direction that may not exist;** a 100% stacked bar or a heatmap is more honest."*
> **[CONVENTION]**

**What makes it publication-grade:**

1. **Sorting** — order items by net agreement or top-2-box percentage, not alphabetically or by item
   number, unless the item order is itself the message. **State the sort in the caption.**
2. **n per item** printed at the right edge — especially where item-level missingness varies.
3. **Percentage labels** on segments above a legibility threshold (≥5%); suppress the rest rather than
   overprinting.
4. **A single, ordered, colorblind-safe diverging palette** with the ordinal sequence encoded by
   **lightness**, so categories read as ordered even in greyscale. **Never a categorical palette for
   ordered categories.**
5. **Full item text** on the y-axis, wrapped, not `Q7a`. If text is unavailable, say so.
6. Zero line drawn as a solid rule.
7. **The response-scale anchors in the legend verbatim** ("Strongly disagree … Strongly agree"), not
   "1 … 5."
8. Faceting by group supported, with item order identical across facets.
9. State whether percentages are of respondents answering the item or of all respondents.

### B5.2 · Item-response distribution panel

Small multiples, one bar chart per item, shared y-axis (percentage). **Complements the diverging chart by
showing shape — bimodality is visible here and invisible in a diverging bar.** Publication-grade = shared
axes, n per panel, floor/ceiling categories highlighted.

### B5.3 · Floor and ceiling effects

> *"Floor or ceiling effects are conventionally flagged when more than **15%** of respondents achieve the
> lowest or highest possible score (Terwee et al. 2007). The consequence is substantive, not cosmetic:
> patients at the ceiling cannot be distinguished from one another, so reliability is reduced in that
> range, responsiveness to improvement is lost, and it usually indicates the instrument lacks items at
> that end of the construct — a content-validity problem. [X]% of your sample is at the ceiling of
> `[scale]`; this scale will not detect improvement in those patients, and any model predicting change in
> it will be attenuated."* **[The 15% figure is a widely adopted CONVENTION from Terwee, not an
> empirically derived constant — say so.]**

**Presentation.** Total-score histogram with **theoretical** axis limits, floor and ceiling bars in a
warning color with percentages annotated; plus a per-item floor/ceiling bar chart when the effect is
item-driven.

### B5.4 · Inter-item correlation matrix

> *"Use **polychoric** rather than Pearson correlations for Likert items; Pearson attenuates the
> associations, which cascades into underestimated loadings and understated reliability. Check the matrix
> is positive definite — polychoric matrices estimated pairwise sometimes are not, in which case apply
> smoothing (minimum-trace factor analysis smoothing is the recommended algorithm) **and report that you
> did.**"* **[SETTLED that polychoric is appropriate; smoothing choice is CONVENTION.]**

**Presentation.** Lower-triangle heatmap, diverging palette centered at 0 with a **fixed −1 to +1
domain — never auto-scaled**, because auto-scaling makes weak matrices look strong. Items ordered by
hierarchical clustering with a dendrogram; values printed if k ≤ ~15; blank/greyed diagonal. **Caption
must state polychoric vs Pearson and the n used.**

### B5.5 · Scree plot, parallel analysis, and factor loadings

> *"Determine the number of factors with **parallel analysis on the polychoric correlations**, not by
> eyeballing the scree plot and not by the eigenvalue>1 (Kaiser) rule, which over-extracts. For estimation
> with ≤5 categories, use **WLSMV** with polychoric correlations rather than ML."* **[SETTLED that
> Kaiser's rule is inferior; parallel-analysis-on-polychoric is the current CONVENTION best practice.]**

> *"Report CFA fit with multiple indices — CFI, TLI, RMSEA with its CI, and SRMR. The familiar cutoffs
> (CFI/TLI ≥0.95, RMSEA ≤0.06, SRMR ≤0.08) come from Hu & Bentler (1999) and have been criticized ever
> since (Marsh, Hau & Wen 2004) for being treated as golden rules when they were derived under specific
> conditions and behave differently with categorical data and different model sizes. **Report the values;
> do not report PASS/FAIL.**"* **[DISPUTED — a tool stamping "good fit" on CFI=0.951 and "poor fit" on
> 0.949 would be embarrassing.]**

> *"Do not run an EFA and a CFA on the same data and present the CFA as confirmation. That is circular.
> Split the sample, or label the CFA as exploratory."* **[SETTLED]**

**Presentation.** **Scree plot with the parallel-analysis reference line overlaid** and the retained
number marked — both curves on one panel with a legend; **the bare scree plot alone is no longer
sufficient.** · **Loading plot/table** with standardized loadings and SEs. **If loadings below a threshold
are suppressed for readability, state the threshold in the caption and provide the unsuppressed matrix in
the supplement** — silently hiding cross-loadings is a documented way factor structures look cleaner than
they are. **[CONVENTION to suppress at 0.30–0.40; the transparency requirement is the important part.]** ·
A **heatmap of the loading matrix** is often more readable than a table for k > 20.

---

## B6 · Modeling

**Diagnostic.** Detect ordinal outcomes (scale scores with few distinct values, single items, global
ratings, staged outcomes like mRS) · test the **proportional odds assumption** (score/Brant test, or
comparison to a partial-proportional-odds model) · detect the item-level vs scale-level predictor decision
· compute the **reliability of each predictor scale** — needed to reason about attenuation.

**Coaching.**

> *"For an ordinal outcome, use a cumulative link (proportional odds) model rather than a linear model on
> the score or a dichotomization into 'responder/non-responder.' The PO model generalizes the Wilcoxon and
> Kruskal–Wallis tests while allowing covariate adjustment, handles arbitrarily many ties, and uses the
> full ordering."* **[SETTLED]**

> *"Check the proportional odds assumption, but do not panic if it is violated. As Harrell argues,
> violation is generally not fatal — the PO estimate remains an interpretable summary of the overall
> tendency, much as a hazard ratio does under non-proportional hazards. If violation is severe and
> substantively important, use a partial proportional odds model, a continuation-ratio model, or report
> category-specific effects."* **[Assumption checking SETTLED; how much violation matters is DISPUTED,
> with Harrell's tolerant position now widely but not universally shared.]**

> **Item-level vs scale-level predictors [DISPUTED]:** *"Entering the k individual items uses more
> information and can find that a subset carries the signal. But items are highly collinear, coefficients
> become uninterpretable and unstable, and you spend k degrees of freedom instead of 1 — with a real
> overfitting cost that shows up in optimism-corrected validation. Entering the scale score is
> parsimonious and interpretable but assumes all items contribute as the key says, and — importantly —
> **the score's measurement error attenuates its coefficient toward zero**, so you will understate the
> construct's association by roughly a factor of its reliability. Practical guidance: for **prediction**,
> item-level with penalization is reasonable and should be compared against the scale score in
> optimism-corrected internal validation — **let the validation decide.** For **inference about the
> construct**, use the scale score and either correct for attenuation or use a latent-variable predictor."*
> **[DISPUTED — no consensus; the honest move is to frame the trade-off and empirically compare.]**

> *"An unreliable predictor scale attenuates its estimated effect by approximately its reliability. With
> ω=0.70, a true standardized coefficient of 0.30 is expected to show up as roughly 0.25 — and your study
> is correspondingly underpowered. Report reliability alongside the model so readers can interpret the
> coefficient."* **[SETTLED for classical error in a single predictor; with multiple mismeasured
> predictors the direction of bias is not guaranteed — do not over-claim.]**

**Anti-patterns.** Median-splitting a scale score · using a clinical cut-point as a modeling threshold when
the continuous score is available · treating a T-scored PROMIS measure as a raw sum · entering both a total
score and its constituent subscales (exact collinearity). **Sample size:** the same logic as Part A —
item-level entry of a 30-item instrument is 30 parameters.

**Presentation.** **Predicted-probability-by-category plot** for ordinal outcomes (stacked area across the
predictor range) — far more communicative than an OR table · **model comparison table**: specification
(scale score / item-level / latent) | df | optimism-corrected C or R² | calibration slope | conclusion ·
coefficient forest plot with the same "these are not causal effects" caveat as Part A.

---

## B7 · Reporting — COSMIN

> *"**COSMIN** (COnsensus-based Standards for the selection of health Measurement INstruments) is the
> reference framework for measurement properties of patient-reported outcome measures. It supplies a
> **taxonomy** — reliability (internal consistency, reliability, measurement error), validity (content
> validity; construct validity comprising structural validity, hypotheses testing, and cross-cultural
> validity/measurement invariance; criterion validity), and responsiveness — plus a **Risk of Bias
> checklist** (*Qual Life Res* 2018) and a **Study Design checklist**. The guideline for systematic reviews
> of PROMs is at version 2.0 (2024). If you are developing or validating an instrument, COSMIN is the
> framework reviewers will apply; if you are merely *using* a validated instrument, you do not need to
> satisfy all of it, but you do need to report which version, in which language, with what recall period,
> and with what scoring and missing-item rules."* **[SETTLED as the field's reference framework]**

**★ What a reviewer expects when a questionnaire is used as a variable — TurboTab's checklist:**

1. **Instrument name, version, language, and citation** to the development paper — and to a validation
   paper *in a population like yours*.
2. **Number of items, response scale with anchors, recall period, and direction of scoring** (higher =
   worse or better — **state it; this is the most common source of sign confusion in the literature**).
3. **Scoring algorithm used**, and whether it is the published one.
4. **Which items were reverse-coded**, per the scoring key.
5. **Missing-item rule** and how many respondents it affected.
6. **Reliability in this sample** — α and/or ω with CIs, not the developers' figures.
7. **Floor and ceiling percentages.**
8. **Any modification** to the instrument, however small — and an acknowledgment that modification
   invalidates published norms and cut-points.
9. **Permission/licensing status** for proprietary instruments — failing to obtain one is a real problem
   at publication.
10. **Careless-responding screening**, if performed, with the rule pre-specified.
11. If a scale is newly derived: the full item list, the factor model, loadings, the derivation sample, and
    a clear statement that it is not yet independently validated.

> *"Terwee et al.'s criteria also set practical sample-size expectations for a psychometric study: a
> minimum of about 50 respondents, and roughly 7 per item (100+ commonly cited for factor analysis). If
> your instrument-development analysis is below this, say so as a limitation."* **[CONVENTION — n:item
> ratio rules have been repeatedly shown to be poor guides compared to communality and overdetermination,
> so present this as a floor, not a target.]**

**Presentation.** **Instrument description table** covering items 1–7, one row per instrument — a compact
exhibit reviewers appreciate and almost no paper includes · auto-generated **Measures** subsection with
placeholders explicitly marked where the app cannot know the answer (version, language, license) ·
**COSMIN-mapped evidence table** when the study *is* a validation study.

---
---

# Cross-cutting · Where confident automation would embarrass the tool

Ranked by damage.

1. **Auto-reverse-coding survey items from correlations (B1.2).** Silently inverts a published scale. No
   correlational signature distinguishes "needs reversing" from "already reversed" from "method factor."
   **Detect and ask. Never act.**
2. **Auto-converting lab units (A1.1).** A factor-of-18 error in glucose. Molecular-weight-dependent
   conversions fail even in official LOINC/UCUM services. **Detect and ask.**
3. **Auto-recoding sentinel values (B1.1).** Recoding a legitimate 9 on a 0–9 scale to missing, or failing
   to recode a 99 that means "refused." **Detect and ask.**
4. **Kaplan–Meier under competing risks (A4.6).** Systematically overestimates cumulative incidence.
5. **Shipping SMOTE / class-weighting in a default pipeline (A5.2).** Directly contradicts the primary
   literature and destroys the property clinical prediction cares about most.
6. **Applying a fingerprinted instrument's published clinical cut-points (B1.1).** A modified PHQ-9 scored
   with PHQ-9 cut-points produces a wrong clinical claim. **Present matches as hypotheses.**
7. **Excluding abnormal-but-possible clinical values as "outliers" (A1.2).** Removes the sickest patients.
   Physiologically impossible ≠ abnormal, and generic outlier rules (±3 SD, IQR fences) are wrong here.
8. **Applying adult plausibility bounds to pediatric or pregnancy data (A1.2).**
9. **★ Stamping PASS/FAIL on threshold-based criteria** — α≥0.70, SMD<0.10, CFI≥0.95, 15% floor/ceiling,
   RMSEA≤0.06. Every one is a convention, several are actively contested, and a binary verdict on a
   continuous quantity is both statistically wrong and professionally embarrassing. **Show the value, show
   the convention, let the user judge.**
10. **Picking a side in the ordinal/interval dispute (B4) or the alpha/omega dispute (B3).** Present both,
    offer a sensitivity analysis, label the app's default as a default.
11. **Mean-imputing anything, ever, without a loud warning (A2).**
12. **Labeling model coefficients as "risk factors" or "effects" (A4.7).**
13. **Fitting imputation, scaling, or selection outside the resampling loop** and reporting the resulting
    performance as validated (A5.5).

---

# Suggested pack manifest structure

Both packs share a shape the app can implement uniformly:

```
pack:
  detectors:      [ {id, trigger, computation, confidence, requires_confirmation: bool} ]
  reference_data: [ unit_conversion_table, plausibility_bounds, reference_intervals,
                    instrument_fingerprints, sentinel_code_lexicon ]
  advisories:     [ {id, trigger_expr, text, evidence_status: SETTLED|CONVENTION|DISPUTED,
                     citations[], both_sides_text?} ]
  figures:        [ {id, when_applicable, spec, publication_grade_checklist[]} ]
  tables:         [ {id, columns, footnote_requirements[]} ]
  reporting:      [ {checklist: TRIPOD+AI | COSMIN | STROBE-nut | MSI,
                     item_map, autofillable[], needs_user[]} ]
  hard_stops:     [ never_auto_convert_units, never_auto_reverse_code,
                    never_auto_recode_sentinels, never_stamp_pass_fail ]
```

**★ The `evidence_status` field should be surfaced in the interface next to every advisory — a small
SETTLED / CONVENTION / DISPUTED badge. That single design decision is what would make TurboTab
trustworthy to a methodologist, because it makes the tool's epistemic position legible rather than
uniformly confident.**
