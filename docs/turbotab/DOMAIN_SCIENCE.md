# Domain science — what the four research threads mean for the product

`DOMAIN_PACKS.md` says *how* a pack plugs in without becoming a different app. This document says
*what the science actually turned out to be*, and — the part that matters — **which of it is a new
product primitive and which is just content in a table**.

The four threads are persisted in full at `research/METABOLOMICS_PACK.md`, `research/NUTRITION_PACK.md`,
`research/GENOMICS_PACK.md`, `research/CLINICAL_SURVEY_PACK.md`. They were commissioned as
*step × (diagnostic · coaching · presentation)* and every recommendation carries **SETTLED ·
CONVENTION · DISPUTED**.

**The headline.** Four independent threads, four literatures, and they converged on the same seven
structural facts. Those seven are the product. The per-domain thresholds are reference data — real
work, but shippable as a table. **What follows is the part that changes the app.**

---

## 01 · The seven convergences

### 1 · The evidence badge is the single highest-leverage primitive

All four threads, unprompted, arrived at the same recommendation: **surface the epistemic status of
every claim the app makes.** The clinical/survey thread states it outright —

> *"That single design decision is what would make TurboTab trustworthy to a methodologist, because it
> makes the tool's epistemic position legible rather than uniformly confident."*

This is not a new card type, so it passes guard #1 of `DOMAIN_PACKS.md`. It is a **token in the design
language** — a small badge rendered beside any advisory, finding, or default:

| Badge | Meaning | Rendering obligation |
|---|---|---|
| **SETTLED** | Methodological consensus. A tool asserting the opposite would be wrong. | May be a pre-selected default with its reason shown. |
| **CONVENTION** | No strong evidence base, but field expectation. Deviating invites reviewer friction. | May be pre-selected, but **stated as convention**, never as fact. |
| **DISPUTED** | Live disagreement among competent methodologists. | **Never defaulted silently.** Both sides stated. A sensitivity analysis offered. |

This subsumes and sharpens the existing `derived` / `convention` / `offered` markers in
`DOMAIN_PACKS.md` §07. `derived` maps to SETTLED, `convention` to CONVENTION, and `offered` splits —
some offered items are DISPUTED (present both sides) and some are merely expensive (present the cost).
**The existing three markers describe the app's confidence; the new three describe the field's. The
second is the honest one, and it is the one a reviewer can check.**

Direct consequence for the governing rule. *"The app may be silent, and it may refuse, but it must
never assert something false"* has a fourth mode: **the app may state that the field disagrees.** That
is not hedging. It is the only true sentence available, and the packs make it a sentence the app can
actually write, with citations.

### 2 · There is a class of thing the app must detect and must not act on

Every pack produced a `hard_stops` list, and the entries are structurally identical:

| Hard stop | Why detection is easy and action is forbidden |
|---|---|
| **Never auto-convert lab units** | A bimodal glucose column at a ratio of 18 is unambiguous evidence of mixed units. But conversion depends on molecular weight for many analytes, and official LOINC/UCUM services themselves error on it. |
| **Never auto-reverse-code survey items** | A negative item–rest correlation has **four incompatible causes** — needs reversing / already reversed / method factor / doesn't belong. No correlational signature separates them. Acting silently inverts a published scale. |
| **Never auto-recode sentinel values** | A `9` in a 1–5 block is almost certainly "refused." But some legitimate scales run 0–9. |
| **Never apply a fingerprinted instrument's published cut-points** | The block matches PHQ-9's shape. A *modified* PHQ-9 scored with PHQ-9 cut-points produces a wrong clinical claim. |
| **Never stamp PASS/FAIL on a threshold** | α≥0.70, SMD<0.10, CFI≥0.95, Q²>0.5, RMSEA≤0.06, 15% ceiling — every one is a convention and several are actively contested. |

The common shape: **high-confidence detection, irreversible-if-wrong action, and no signal in the data
that resolves the ambiguity.** This is exactly clause 06 of the lockbox constitution — *declaration
separated from execution* — generalized past the leakage question that produced it. The litmus there
was *does this row's output depend on any other row?* The litmus here is: **can the data distinguish
the causes of what I just detected? If not, the app declares and the user executes.**

This is not the block-and-record rung of the three-rung ladder. Block-and-record is for things with a
*rare legitimate use*. These are things where the app **genuinely cannot know**, and the honest
rendering is a finding with a proposed action and no pre-selection — which the Router already supports.
**No new component. A new reason for an existing one.**

### 3 · The prediction/inference fork changes the correct answer, not just the emphasis

This is the deepest finding and the one with real product consequences.

| Domain | The question | Correct for **prediction** | Correct for **inference** |
|---|---|---|---|
| Clinical EHR | missing labs | **Missing-indicator method** — the indicator carries the clinician's judgment and is observable at deployment | **Never** — known to bias estimates; MI or a principled MNAR model |
| Clinical labs | values below LOD | censoring indicator + substituted value is defensible | ML / censored regression; substitution biases |
| Nutrition | 2 recalls per person | the **mean is fine** for ranking people | **not adequate** for any prevalence or percentile claim; NCI method required |
| Survey | a 30-item instrument | **item-level with penalization**, validated empirically | **scale score**, with attenuation correction or a latent variable |
| Metabolomics | features vs compounds | feature-level is fine for a classifier | compound-level count is what you may claim |

**These are not stylistic preferences. The advice inverts.** A tool that gives the inference answer to
someone building a bedside model is wrong, and vice versa.

TurboTab currently assumes prediction throughout — reasonably, since it is a predictive-modeling app.
But the opening sequence asks *what are you predicting*, and never asks *why*. The finding is that the
same dataset, the same target, and the same lens can require opposite handling.

**The proposal, and it is a small one:** the target question (`OPENING_SEQUENCE.md` §03.2) gains a
second part.

> **What is this model for?**
> — **Predicting an outcome for a new person** → we optimize for the number being right at the bedside
> — **Estimating how strongly something is associated with the outcome** → we optimize for the
>   coefficient being unbiased

It is a CHOICE question by the routing constitution — always asked, with a preview. It is not
skippable at any confidence, because nothing in the data reveals it. It fires **once**, and it changes
the default on roughly a dozen downstream decisions per pack. That is the best ratio in the entire
opening sequence: **one question, and the app stops being wrong half the time for half its users.**

This is also the honest answer to the pre-seal length worry. This question does not lengthen the
interview meaningfully, and it is the clearest case yet of the product owner's own standard: *if the
answer were wrong, would a downstream number be wrong or misleading?* Emphatically yes.

### 4 · Orientation is confirmed, twice, independently

Genomics: **genes-in-rows is the near-universal convention**, with a four-step detection cascade.
Metabolomics: **vendor exports are overwhelmingly features-in-rows**, with a five-step cascade — and an
explicit warning that **shape alone is not sufficient**, because a targeted panel can be 40 metabolites
× 400 samples, i.e. the reverse of the p ≫ n prior.

The user's instinct — that the orientation question belongs before the import doctor — is confirmed by
both threads, and the second one supplies the reason the naive version fails. **The detection cascade
must be ID-grammar-first and shape-last.** A heuristic that reads "1,847 columns × 80 rows, therefore
transposed" gets the targeted panel exactly backwards.

Both threads say the same thing about rendering: **present the inference with its evidence and require
confirmation.** *"I think features are in rows because column 1 matches an m/z pattern and 4,812 rows
carry `M###T###` labels."* That is a rendered skip with a stated basis, which is what the design
language already prescribes.

### 5 · Leakage is one rule with four faces

Every pack independently named its own leakage mode, and they are the same defect:

- **Metabolomics:** feature selection outside the CV loop (Ambroise & McLachlan 2002 — near-zero error
  from pure noise); imputation across the whole dataset before splitting.
- **Nutrition:** the energy-adjustment residual regression `N ~ E` fit on train+test; variance
  components and λ estimated on the full data; **row-level splitting when a person contributes multiple
  recalls.**
- **Clinical:** imputation parameters learned before splitting; tuning on the test set; **immortal time
  bias** from an index date defined by a later event; using a variable recorded *because* the outcome
  happened.
- **Survey:** scale-score standardization and factor loadings fit outside the fold.

The unifying statement, which the app should carry once rather than four times: **every parameter
estimated from data — including ones that do not look like model parameters — must be estimated inside
the resampling loop, and rows that share an identity must not be split across folds.**

The second clause is what the grain question already protects. The first clause is broader than the
app currently enforces, because *energy-adjustment residuals* and *factor loadings* and *Box–Cox λ* do
not look like model fitting to a user. **This is a leakage-detector family, not four detectors.**

### 6 · The circular-figure family

Four packs, four instances of the same defect: **a result presented as evidence for itself.**

- Heatmap of the top-50 features **selected by the test being illustrated** — the block structure is
  guaranteed by construction.
- **QC RSD improved after QC-based correction** — the correction is fit to minimize exactly that number.
- **EFA and CFA on the same data**, with the CFA presented as confirmation.
- **PLS-DA scores plot with no permutation test** — the method separates random labels at p ≫ n.
- Post-hoc power analysis.
- Filtering features by a group difference before testing them.

Every one is legitimate as a *display* of an established result and illegitimate as *evidence* for it.
The metabolomics pack supplies the resolution, and it generalizes: **label every figure EXPLORATORY or
CONFIRMATORY, and refuse to let a confirmatory figure into the results bundle without its validation
companion.** The generated caption states the circularity where it exists —

> *"Features pre-selected by the differential test shown in Fig. 3; the block structure is a consequence
> of that selection, not independent support for it."*

That caption is the whole feature. It costs a sentence and it is the difference between a figure a
reviewer stops on and a figure they accept.

### 7 · The reporting checklist is the manuscript ledger's actual target

Four packs, four checklists, one architecture:

| Domain | Standard | Size |
|---|---|---|
| Clinical prediction | **TRIPOD+AI** (*BMJ* 2024) + PROBAST for risk-of-bias | 27 items + 20 signaling questions |
| Nutrition | **STROBE-nut** (*PLoS Med* 2016) | 24 nutrition-specific additions to STROBE |
| Survey / PROM | **COSMIN** (taxonomy + risk-of-bias + study-design checklists) | v2.0, 2024 |
| Metabolomics | **MSI/CAWG** + **mQACC** + **QComics**; lipidomics and proteomics equivalents | 4-level identification scheme |

Each thread independently identified the same feature as the highest-value deliverable: **a rendered
checklist, auto-populated from what the session actually did, with the items only the user can supply
explicitly marked.** Three of four called it the thing most likely to make researchers recommend the
tool.

This is what the manuscript ledger is *for*, and it reframes it. The ledger is not a log of what
happened. It is a **checklist-shaped artifact with two column types: what the app knows, and what it
must ask.** The lens selects which checklist. The session fills it. The unfilled cells are the app's
to-do list for the user, and journals increasingly require the completed file as a submission
attachment.

---

## 02 · What this means for the figure layer

The figure layer is the stated priority, and the research changes what it is.

**Every pack specified its signature figures as a *checklist*, and the checklist items are
overwhelmingly about annotation rather than geometry.**

| Figure | The plot | The publication-grade delta |
|---|---|---|
| **Calibration plot** (clinical) | predicted vs observed, loess curve | **a spike histogram of predicted risks along the bottom, split by outcome** — without it the reader can't tell whether the wild behavior at 0.8 is 3 patients or 300; 45° line labeled "ideal"; annotation box with intercept, slope, C, E:avg, n, events; **do not truncate the sparse tail** |
| **PCA scores** (metabolomics) | scatter of PC1 vs PC2 | **% variance in the axis labels**; **QCs overlaid**; **aspect ratio proportional to variance explained**; T² ellipse distinguished from group ellipses; a second panel colored by injection order |
| **Kaplan–Meier** (clinical) | step curves | **numbers-at-risk table beneath — mandatory**; cumulative incidence rather than survival when events are rare; x-axis truncated where <10–20% remain at risk, **and say so** |
| **Restricted cubic spline** (nutrition) | fitted curve + CI band | **a rug or histogram of the exposure underneath**, so the reader sees the dramatic upturn is driven by 11 people; truncate at the 1st–99th percentile; p for non-linearity |
| **Diverging stacked bar** (survey) | Likert percentages across a zero line | **sort by net agreement and say so**; n per item at the right edge; **anchors verbatim in the legend, not "1…5"**; ordinal palette encoded by *lightness* so it survives greyscale |
| **Shrinkage plot** (nutrition) | three overlaid densities | annotate the 5th/95th of each — **the visible narrowing is the entire argument for usual-intake modeling, in one image** |
| **Volcano** (metabolomics) | log2FC vs significance | **q on the y-axis, or the cut line drawn at the p corresponding to q=0.05, stated in the caption**; and **FC computed from normalized-but-not-scaled data** — after autoscaling a "fold change" is in z-units and is meaningless |
| **Forest plot** (clinical) | estimates + CIs | **log-scale x-axis for ratio measures — non-negotiable**; labeled "model coefficients," not "risk factors" |

**The conclusion for the build.** The figure layer is a **caption-and-annotation engine wrapped around
a plotting library**, not a plotting library with captions bolted on. The differentiator is not that
TurboTab can draw a calibration curve — every library can. It is that TurboTab draws the risk
distribution under it, annotates the six numbers a reviewer wants, refuses to truncate the tail, and
writes the caption naming the test, the correction, and the n.

Which means the figure spec has a shape, and it is the same shape as a pack advisory:

```
figure:
  id, when_applicable
  layers:        [ the geometry ]
  annotations:   [ required numeric annotations, with their sources ]
  checklist:     [ publication-grade items, each pass/fail against this render ]
  caption:       [ generated text; names test, correction, n, transformations ]
  tier:          EXPLORATORY | CONFIRMATORY
  companions:    [ figures that must accompany this one to be admissible ]
```

`companions` is the piece that has no analogue in the current app and is load-bearing: a PLS-DA scores
plot's companion is its permutation plot, and a confirmatory figure with a missing companion is not
rendered into the results bundle.

**And two cross-cutting rules the packs agree on, which belong in the design language rather than in
any pack:** consistent group colors and shapes across *every* figure in a session, colorblind-safe; and
n stated in every legend.

---

## 03 · What is a primitive, and what is a table

The discipline that keeps this tractable. Everything the four threads produced sorts into one of three
bins, and only the first costs design.

**Primitives — new capability, needs design and a decision.**

1. The **evidence badge** (SETTLED/CONVENTION/DISPUTED) and its rendering obligations.
2. The **purpose question** (prediction vs inference) and the fork it drives.
3. The **hard-stop class** — detect, declare, never execute.
4. The **sensitivity fork** — run it both ways, report whether the conclusion changes. Named as the
   highest-value cheap addition by three of four threads, and currently absent from the app entirely.
5. **Figure tiering and companions** — EXPLORATORY vs CONFIRMATORY, and the admissibility rule.
6. The **checklist engine** — one artifact, four checklist definitions, two column types.
7. The **generalized leakage detector** — any parameter estimated from data, inside the loop.

**Reference data — real work, no design.** Unit conversion tables (with per-compound qualifiers), lab
plausibility bounds vs reference intervals, DRI tables by age/sex/pregnancy/lactation, instrument
fingerprints, sentinel-code lexicons, BMR equations, QC threshold conventions, feature-ID grammars.
**This is the bulk of the volume and none of it touches the interface.**

**Prose — coaching text with citations.** The largest category by word count and the cheapest to ship,
because the packs wrote it already. It flows into the existing finding and skip components unchanged.

The ratio is the reassuring part: **seven primitives, and everything else is content.** Breadth does
not eat the design language, which was the whole bet of `DOMAIN_PACKS.md` §02.

### The correction that has to be attached to this section

**The three bins above sort by design cost. They do not sort by value, and reading them as though they
did inverts the truth.** The product owner caught this and the correction belongs here rather than in a
chat log:

> *"That research also revealed real content we can introduce into the app with the correct routing. It
> is more than just a series of documents that dictate the meta structure of the app, it is also a well
> of knowledge from real research we can leverage in all facets of the app."*

The seven primitives are scaffolding. **A researcher does not pay for scaffolding.** They pay for the
app knowing that a bimodal glucose column at a ratio of exactly 18.0 is a unit collision, that BMI must
equal FMI + FFMI, that their two recalls cannot yield a prevalence of inadequacy, and that the 812
blood pressures above 140 mmHg are the sickest patients rather than outliers. **That is the content
bin, and it is the product.** The primitives exist so the content can be delivered without lying.

So the bins are a *sequencing* aid and nothing more. Nothing in §05 licenses shipping the scaffolding
and deferring the science. **A pack with badges and no content is a worse product than the app we have
today**, because it has added ceremony without adding knowledge.

---

## 03b · Where the content lands — the routing inventory

Written as an inventory because content that lives only in a research file is content that will be
forgotten, and this repository has already learned that lesson once: *a record that points at
ephemeral storage will eventually lie, and it lies toward "the work is gone."* A research file that no
builder prompt ever cites is the same failure with a different address.

**This is a standing instruction.** Every builder prompt that touches a pack surface must name the
specific findings it is drawing from these four files, and no loop should consist only of primitives.

### Import doctor — detectors that do not exist today

| Detector | Source | Why it earns its place |
|---|---|---|
| **Atwater reconstruction** — `4P + 4C + 9F + 7A` vs declared energy, with the ratio table (4.18 → kJ; drift with total energy → mixed units across rows) | nutrition | Nothing else in the app can infer an energy unit, and a unit error there propagates into every downstream number invisibly |
| **BMI = FMI + FFMI** identity; recompute BMI from wt/ht² and compare to declared (±0.2) | nutrition | A free data-quality check on a derived column |
| **Bimodal analyte at a known conversion ratio** — glucose ×18.0, creatinine ×88.4, cholesterol ×38.67, height in vs cm, temp °F vs °C | clinical | A mixed-unit predictor is not a noisy predictor; it is a variable whose meaning changes between rows |
| **Sentinel codes in a bounded block** — a `9` in a 1–5 item, `77/88/99`, `-8/-9`, SAS special missing | survey | The highest-yield check in the survey pack. Detect and ask; never recode |
| **Feature-ID grammars** — `M123T456`, `784.5876@8.21`, `ENSG…`, `PC 34:1`, UniProt | metabolomics, genomics | Drives orientation ID-grammar-first, which is the version that does not get a targeted panel backwards |
| **NHANES DXA 5-implicate structure** | nutrition | Unhandled, the naive N is inflated 5× and standard errors are far too small |
| **Repeated-digit / default-value mass** at 120/80, 98.6, 0 | clinical | A documented EHR artifact — value preference and manual entry, not measurement |
| **Temporal implausibility** — adult height moving >5 cm between visits; weight ±30% in <30 days; labs timestamped after death | clinical | Kahn et al.'s framework distinguishes atemporal from temporal plausibility; we only do the first |
| **Already-transformed data** — negatives present, or max <40 with low dynamic range | metabolomics | A second log transform is a silent catastrophe |
| **Zero vs NA semantics differ by vendor** — XCMS fills, MZmine writes 0, MaxQuant writes 0 for "not quantified" | metabolomics | Defaulting wrong corrupts every subsequent step |

### Coaching — sentences that correct what the app currently implies

The line that matters most, because it answers the product owner's own drive observation that a
diastolic pressure near zero is an entry error rather than an outlier:

> *"4 systolic values are below 30 mmHg — physiologically impossible in a living outpatient, almost
> certainly entry errors. **This is different from the 812 values above 140 mmHg, which are abnormal but
> real and must be kept:** excluding abnormal-but-possible values would remove the sickest patients and
> bias the model toward the healthy."*

**Physiologically impossible and statistically extreme are different categories, and generic outlier
rules (±3 SD, IQR fences) cannot tell them apart.** That is a correction to behavior we ship.

Others in the same class: *reference intervals are not disease thresholds* (~5% of healthy people fall
outside by construction) · *reliability is a property of scores in a sample, not of an instrument —
"the scale has been validated" is not a statement that can be true* · *nutrients cluster by food source,
not by biology, so an association with any nutrient in a cluster is an association with the food
source* · *these are model coefficients, not risk factors* · *in EHR data, missingness is a fact about
the patient's clinical trajectory, not a defect in the measurement.*

### EDA and figures — the signature set

The shrinkage plot · missingness-vs-outcome forest · missingness-over-calendar-time · calibration with
its risk-distribution rug · decision curve · diverging stacked bar · FMI/FFMI with BMI iso-lines ·
intake-vs-DRI with the EAR region shaded — **and the refusal to compute a prevalence of inadequacy from
an AI, or from the RDA, or from anything that is not a usual-intake distribution.** A refusal with a
stated reason is content too, and it is the kind a reviewer notices.

### Preprocessing and modeling — content that changes defaults

Energy adjustment's five models with their estimands named · the modified 80% rule and why the plain one
deletes the finding you are looking for · QRILC vs half-minimum routed by the missingness-vs-intensity
plot · PQN over TIC for biofluids, with TIC's named failure mode · **calibration ranked above AUC** ·
splines over dichotomization · Riley sample size over EPV≥10, counting candidate parameters including
the ones later dropped.

### The anti-pattern registries are an audit of the engine we already have

Across the four files there are on the order of **150 named anti-patterns**, each written as a specific,
checkable behavior. They were commissioned as content for the packs. They are also, unmodified, a
conformance suite against the current codebase — and the first pass already found a real defect.

**Worked example, found on the first look.** `ml/dataset_profile.py:429` advises *"Use class weights in
training"* and *"Consider SMOTE or other resampling."* `ml/eda_recommender.py:419` repeats it.
`ml/narrative_engine.py:1065` writes it **into the generated manuscript**: *"To address class imbalance,
class_weight='balanced' was applied…"*

The clinical pack's flagship finding is that this is wrong for risk prediction. Van den Goorbergh, van
Smeden, Timmerman & Van Calster (*JAMIA* 2022;29:1525) showed random undersampling, random oversampling
and SMOTE all produce **strong overestimation of minority-class probability without improving
discrimination**, and that any apparent sensitivity gain is reproducible by simply shifting the
threshold. Replicated for machine-learning methods by Carriero et al. (*Stat Med* 2025). Rare outcomes
are a real problem, but the problem is **small-sample overfitting, not imbalance**, and the remedy is
penalization and adequate sample size.

So the app currently recommends a step that damages the property clinical prediction cares about most,
and then asserts it in the artifact that is the product. **That is not a future feature request; it is a
defect the research found in shipped code.** Under the governing rule it is the serious kind — the app
asserting something false — and it should be filed as such rather than as a pack enhancement.

The generalizable move: **run each pack's anti-pattern registry against the engine before building the
pack.** The registries are cheap to check, they are already written, and they measure the app against
the literature rather than against its own description of itself — which is precisely the record-vs-draw
distinction the audit work has been circling.

---

## 04 · The honest risks

**The verification debt is real and is recorded.** All four threads hit an egress proxy that blocked
publisher domains. Every specific numeric threshold in the four files is search-surfaced rather than
read from primary text. The genomics and metabolomics threads flag items `[verify-at-build]`
explicitly; the nutrition thread lists four documents that must be read before shipping — including
**the DRI tables, which the pack must ship as data, not as prose**. Shipping a wrong number is the
single worst failure mode a pack has, and the packs say so themselves.

**Two thresholds contradict each other across threads and must not be silently reconciled.** The
metabolomics benchmark says QRILC for MNAR and random forest for MAR; a major proteomics benchmark says
random forest is robust *even* under MNAR. Both are cited. **Present both. The disagreement is the
finding.**

**The packs are wide and no domain reviewer has seen them.** `DOMAIN_PACKS.md` §06 already names this,
and the reviewable form is unchanged: *here are the default choices and the methods sentence each
produces — which would you object to?* What the research adds is that this document is now long enough
to be worth an expert's hour, and specific enough that their objections would be concrete.

**Scope discipline still governs.** The filter from `DOMAIN_PACKS.md` §04 — *build the science that
changes a sentence a reviewer would challenge* — survives contact with the research and is doing more
work than before, because the research volume is now far larger than the buildable set. Applied to what
came back: the calibration hierarchy is in; the class-imbalance warning is in; the EAR cut-point rules
are in; energy adjustment's substitution estimand is in. Fifteen scaling variants are still out. The
D-ratio-vs-RSD debate is in only as far as offering both. The distinction holds.

---

## 05 · Sequencing

Ordered by ratio of what it changes to what it costs, not by domain.

1. **The evidence badge.** A token, a field on advisories, and a rendering rule. It is the cheapest of
   the seven primitives and the one every subsequent piece of content depends on, because a pack
   advisory without a badge is exactly the confidently-uniform voice the badge exists to prevent.
2. **The purpose question.** One CHOICE card in the opening sequence, and the fork it sets. Highest
   ratio of correctness-gained to interview-length-added in the whole document.
3. **Orientation detection**, ID-grammar-first, rendered with its evidence. Already specified; now
   confirmed twice and with the failure mode of the naive version named.
4. **The figure spec** — tier, annotations, checklist, caption, companions — and then the signature
   figures per pack, starting with the ones that are cross-domain: distribution-against-reference,
   missingness structure, and the correlation heatmap that needs a domain-aware treatment at p ≫ n.
5. **Hard stops**, as findings with no pre-selection. Detectors are mostly already specified in the
   pack files.
6. **The checklist engine**, seeded with one checklist rather than four.
7. **Reference data**, verified against primary sources — the long tail, parallelizable, and the place
   where being wrong is most expensive.

The sensitivity fork and the generalized leakage detector sit outside this order deliberately: both are
engine-level, both are independent of any pack, and both can land at any point.

**And one constraint on the whole sequence, from §03b.** No loop may consist only of primitives. Each
one carries content from the research files alongside its scaffolding, and each builder prompt names
the specific findings it is drawing on. The anti-pattern audit runs ahead of the pack it belongs to,
because a defect the research already found in shipped code outranks a pack feature that has not been
built yet.
