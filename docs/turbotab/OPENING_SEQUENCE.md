# The opening sequence — everything before the seal

**What this covers.** The first step of the app, from upload to a drawn lockbox. It is specified
in one place because the *order* is load-bearing and expensive to change later, and because the
question "is this still an interview or has it become a form?" can only be answered by seeing the
whole thing at once.

**Why it is allowed to be long.** In the product owner's words:

> These are non-negotiable points of agreement that the app and the modeler must come to before
> they can begin the work of predictive modeling.

That is the distinction from Classic's ~32 questions. Those were *configuration* — options offered
because the pipeline had a stage for them. These are **agreements about what the data is and what
is being asked of it**, and no modeling result means anything until they exist. A question here
earns its place by this test: *if the answer were wrong, would a downstream number be wrong or
misleading?* If not, it belongs later or nowhere.

**And the app is advisory throughout.** It states what it believes and why, and the user overturns
it. Every inference below renders as a skip the user can open (`DESIGN_LANGUAGE.md` §09), never as
a decision taken on their behalf.

---

## 01 · The sequence

| # | Question | Fires | Kind |
|---|---|---|---|
| 1 | **What kind of measurements are in this table?** | always | asked · multi-select |
| — | *structural diagnosis, repairs, impossibility pass* | always | findings |
| 2 | **What are you predicting?** (+ which level is the event) | always | asked |
| 3 | **Can one person appear in more than one row?** | always | asked |
| 4 | **Are these repeats or different time points?** | only if 3 = repeat | usually **stated**, overturnable |
| 5 | **When you analyze this, what is one row?** | only if 3 = repeat | asked |
| 6 | **How should each person's rows be combined?** | only if 5 = person | asked |
| 7 | **Are you predicting something later from measurements earlier?** | only if 4 = time points **and** 5 = row | asked |
| 8 | **Is your study restricted to part of this data?** | always | asked |
| — | **SEAL** | | |

Nothing may be resequenced. Three placements are load-bearing and easy to get wrong:

- **The lens is first, before diagnosis**, because diagnosis is field-sensitive. 1,847 columns
  across 80 rows reads as malformed to a general-purpose import doctor and is the expected shape
  for an assay panel. Setting the lens first turns a false alarm into a correct reading.
- **The impossibility pass is pre-seal** — not for leakage reasons (setting a physiologically
  impossible value to missing is row-local and leaks nothing) but because a stratified or grouped
  split computed over corrupted values is a worse split, and impossible entries are normally an
  exclusion that changes N.
- **Aggregation is pre-seal and cannot move.** Decision A's identity barrier already forces it:
  combining three visits into one person-row changes *what a row is*, and a seal drawn beforehand
  names rows that no longer exist. This is not a preference; it is the same rule that governs
  `melt_repeated`.

**Target precedes aggregation** for a reason that is easy to miss: if the outcome is measured at
every visit, combining rows requires deciding *which outcome*. Question 6 cannot be asked coherently
without question 2 answered.

---

## 02 · How many actually fire

The sequence is nine rows and most datasets see four to six. Worked from the fixtures:

| Dataset | Asked | Stated | Skipped |
|---|---:|---:|---:|
| Cross-sectional clinical CSV | 4 | 0 | 4 |
| NHANES-style: 2 dietary recalls per person | 6 | 1 | 2 |
| Longitudinal clinical, visit-level modeling | 7 | 1 | 1 |
| Untargeted metabolomics, one sample per subject | 4 | 0 | 4 |

Worst case is seven questions before modeling begins, against Classic's ~32 asked regardless of
the data. The count *tracks the shape of the study*, which is the differentiator's whole claim
applied to the opening.

---

## 03 · The questions

### 1 · The lens

> **What kind of measurements are in this table?**
> Pick all that apply. This changes what we look for and what we suggest — it never limits what
> you can do.
> ☐ Metabolomics or proteomics · ☐ Genomics or transcriptomics · ☐ Dietary intake
> ☐ Clinical measurements and labs · ☐ Survey or questionnaire instruments
> ☐ Something else, or not sure

Multi-select because the first audience is an intersection — dietary **and** clinical is NHANES
exactly, dietary **and** metabolomics is a nutrition-metabolomics study. "Something else, or not
sure" is first-class: the app is fully functional with no lens, and uncertainty is never more
expensive than a confident wrong answer.

Records a decision and a methods sentence, because a lens the manuscript cannot see is a lens the
reader cannot check. Detection runs as a *suggestion* and as a *contradiction detector*, never as
the answer. See `DOMAIN_PACKS.md`.

### 2 · The target

Unchanged from the built step, with the positive-class question already specified: for a two-level
outcome the interesting question is **which level is the event**, never "is this binary." Never
pre-selected at any confidence — "alive/dead" has no correct default, because whether the event is
death or survival is the research question.

### 3 · Grain

Built. *"Can one person appear in more than one row?"* The heuristic is a suggestion and a
contradiction detector; name lists and ratio bounds cannot close this and must not be tuned as
though they could.

### 4 · Repeats or time points — usually stated, not asked

Neither grain nor unit-of-analysis asks the thing that determines whether averaging is correct:
**what varies between one person's rows?** It is largely inferable, and a strong domain prior makes
it a rendered skip:

> *Not asked: these look like repeated measurements of the same intake rather than different time
> points — the two recall dates are 4 and 11 days apart with no visit structure.*
> — **Ask me anyway**

Evidence that resolves it: a date column with meaningful spacing, or a visit label, means time
points. Identical dates, or a replicate index, means repeats. Where the evidence is thin, it is
asked rather than guessed.

### 5 · Unit of analysis

> **When you analyze this, what is one row?**
> You told us people appear more than once. That leaves two honest options, and they lead to
> different analyses.
> — **One row per person** → *we combine each person's records into one. How?*
> — **One row per record** → *records stay as they are, and held-out people never appear in training*

**No default.** Guessing at grain is what produced the leak this whole constitution exists to
prevent, and the same reasoning binds one level down.

### 6 · Aggregation — the menu is domain-shaped

The same structural fact means three different things, and only the lens can tell them apart:

| Repeats are… | Correct treatment | Recommended default |
|---|---|---|
| replicate measurements of one quantity (dietary recalls, technical replicates) | **average** — this reduces measurement error rather than losing information | mean, with the reason stated |
| different time points (clinical visits, time course) | **averaging destroys the signal** | none — baseline / last / change / slope, asked |

For dietary specifically the app has something real to say rather than a menu:

> *You have 2 recalls per person. Using their mean rather than a single day reduces the
> within-person measurement error that attenuates diet–outcome associations.*

v1 menu: **mean · first · last · change from baseline**. Slope, area under the curve and
usual-intake modeling are real practice and materially more work — filed, not built.

### 7 · Temporal prediction

Fires only when time points survive as rows. A random split — even grouped by person — is
optimistic when the task is predicting a later outcome from earlier measurements, and TRIPOD
treats temporal validation as a distinct thing from internal validation.

> **Are you predicting something that happens later from measurements taken earlier?**

Yes → chronological split, grouped as well where people repeat. No → grouped is sufficient.
`ml/splits.py` already carries both strategies with sixteen equivalence tests; what has been
missing is the routing that decides when each applies.

### 8 · Eligibility

Built. Asked in scientific terms with the outcome's distribution **withheld** — an exclusion that
comes from the research question is reportable; one that comes from looking at the data is a
different thing. Observed min/max and impossible-value flags are permitted because they answer
*"is this data corrupted?"*, not *"where should I cut?"*

Domain packs supply *candidate* criteria (implausible energy intake for dietary) as suggestions,
never as defaults, because an exclusion changes N.

---

## 04 · Fixtures — the sequence is not right until it is tested

Each domain needs a fixture carrying its characteristic shape, and the assertions are as much
about **not firing** as about firing.

| Fixture | Shape | Must produce |
|---|---|---|
| `metabolomics_untargeted` | 80 × 1,847, log-normal, missingness concentrated in low-abundance features, run-order column, pooled QC rows | wide shape read as expected, not malformed · LOD-shaped missingness finding · run-order finding · QC rows flagged as not-participants |
| `dietary_recalls` | 500 people × 2 recalls, macronutrient columns summing to ~100%, energy column, some implausible intakes | grain = repeat · repeats-not-timepoints **stated** · mean recommended with the measurement-error reason · compositional finding · implausible-intake criterion offered |
| `clinical_longitudinal` | 200 people × 3 visits, vitals, some physiologically impossible values | grain = repeat · time points stated · averaging *not* recommended · temporal question fires · impossibility pass pre-seal |
| `survey_instrument` | 300 × 40 Likert items, some reverse-coded | ordinal recognized as declared, not frequency-derived · scale scoring offered |
| `clinic_visits` *(existing)* | the generic messy fixture | **zero new questions** when every pack is installed |

That last row is guard #2 from `DOMAIN_PACKS.md` made executable: a pack that fires on
non-matching data has failed, and the value check's `irrelevant_questions` metric already measures
it.

**Pooled QC samples deserve their own line.** They look exactly like participants and must never
enter a model, while being needed for quality assessment. A generic tool models them silently.
This is a class of error only the lens can see, and it is the cheapest demonstration that the
opening question earns its place.

---

## 05 · Open

- **Whether this is still an interview.** Nine rows, four to six firing. The count is defensible
  on the fixtures; whether it *feels* like an interview is a drive question, not an analysis one.
- **Slope, AUC and usual-intake modeling** are deferred from the aggregation menu. Usual-intake
  modeling in particular is close enough to nutrition practice that omitting it may make the app
  feel like a toy to that audience.
- **The defaults themselves** are in `DOMAIN_PACKS.md` §07, with the reasoning and a confidence
  marker on each, because they were set by judgment and the math rather than by a domain reviewer.
  They are the first thing to show a colleague in the field.
