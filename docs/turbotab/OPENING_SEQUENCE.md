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
| 1.5 | **Which way round is this table?** | only if 1 includes an assay pack **and** the shape reads feature-major | asked |
| — | *structural diagnosis, repairs, impossibility pass* | always | findings |
| 2 | **What are you predicting?** (+ which level is the event) | always | asked |
| 2.5 | **What is this model for?** | always | asked |
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

  **What "before diagnosis" means in the code, stated because the first reading of this line was
  wrong** (L20). The detectors in `ml/import_doctor.py` take a frame and nothing else. They are
  field-blind by construction, they are frozen, and no lens will ever reach them. So the lens is a
  parameter of `engine.rank_findings` — *the one function that produces the finding list the app
  presents* — and not of the detector pass underneath it.

  That is not a weaker claim, and the reason it must be this way is worth the sentence: **reframing
  annotates and never deletes.** A user who reads *"these are different analytes, not one analyte
  measured twice"* and still wants to reshape the table can, because `apply` re-runs the raw
  diagnosis and gets the real repair. A lens that erased the reading at generation would take that
  route away and turn the annotation into a deletion by another name.

  The governing rule is about what the app **asserts**, not about what it computes. Nothing reaches
  a user except through `rank_findings`, and
  `test_the_lens_reaches_every_finding_list_the_app_presents` is what makes that a check rather
  than a habit.
- **Question 1.5 is the one that genuinely acts before the diagnosis**, and it is what gives this
  ordering its teeth (L24). Everything above says the lens is a parameter of `rank_findings` —
  presentation — and defends that at length, correctly. But *presentation* is the whole of what a
  lens can fix, and there is a failure it cannot touch: **an assay table exported features-in-rows
  and samples-in-columns is transposed, and every finding computed on it is garbage.** The
  "columns" are participants, so column dtypes are meaningless, missingness per column is
  missingness per participant, the impossibility pass compares one subject's entire panel against a
  reference range for a single analyte, and the target list on offer is a list of sample
  identifiers. **Annotation cannot fix a frame.**

  So this question does not annotate the diagnosis, it precedes it: answering *"each row is a
  feature"* transposes the working table, records the decision, and states it in the methods
  sentence — *"the table was supplied with features in rows and samples in columns, and was
  transposed to one row per sample before any diagnosis was run."*

  **It fires narrowly**, because a question asked of a table it does not describe is guard #2
  broken: only when the lens includes an assay pack (a clinical export or a survey is not shipped
  transposed) **and** the shape reads feature-major. The reading is the ratio of the spread of row
  means to the spread of column means, on a log scale — in a sample-major assay table the columns
  are analytes and differ by orders of magnitude while the rows are comparable samples, and
  feature-major is that fact with the axes exchanged. Measured rather than chosen: every fixture in
  this tree reads between 0.05 and 1.51, a transposed copy of `metabolomics_untargeted.csv` reads
  23, and the threshold is 4.

  **Two placements follow from it.** The target question is *withheld* while 1.5 is open — on a
  feature-major table the column list is a list of samples — and `set_orientation` refuses once a
  target exists, because after the turn that column is a row. And it is refused after the seal, by
  Decision A rather than by preference: transposing changes what a row *is*, which is the same
  class as `melt_repeated`.

  **One detector had to be taught to stay quiet** (`GUIDED-042`). The lens contradiction check
  reads per column, so on a feature-major table it measured missingness-against-abundance across
  the wrong axis and told the user, in the app's most interruptive voice, that their blanks *"do
  not look like non-detections"* — when read the right way round they do. Two readings competed,
  *the lens is wrong* and *the table is turned around*, and the second explains the first. It now
  defers to 1.5, which is where that is settled.
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

The sequence is ten rows and most datasets see four to six. Worked from the fixtures:

| Dataset | Asked | Stated | Skipped |
|---|---:|---:|---:|
| Cross-sectional clinical CSV | 4 | 0 | 5 |
| NHANES-style: 2 dietary recalls per person | 6 | 1 | 3 |
| Longitudinal clinical, visit-level modeling | 7 | 1 | 2 |
| Untargeted metabolomics, one sample per subject | 4 | 0 | 5 |
| The same metabolomics table exported features-in-rows | 5 | 0 | 4 |

Worst case is seven questions before modeling begins, against Classic's ~32 asked regardless of
the data. The count *tracks the shape of the study*, which is the differentiator's whole claim
applied to the opening.

**Question 1.5 costs nothing on nine of ten tables and is the difference between an analysis and a
wasted afternoon on the tenth.** That asymmetry is the argument for it: it is a question with a
narrow, measurable firing condition and an unbounded consequence when it does not get asked.

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

### 1.5 · Which way round is the table

> **Which way round is this table?**
> Across 396 rows and 80 numeric columns, the rows differ from each other by orders of magnitude
> and the columns barely differ at all. In an assay table that is what features in rows looks like:
> different analytes have very different abundances, and samples of the same kind do not.
> — **Each row is a sample or participant**
> — **Each row is a feature, and the columns are samples**

A FACT, not a CHOICE — there is one true answer and the rewrite follows from it rather than
expressing a preference — and **never skippable at any confidence**, by two independent guards:
`_skip_is_permitted` admits only `task_type` and `missingness`, and the shape reading never returns
`high`. A reading that could auto-advance would be the app transposing a table on its own
authority, which is the single most destructive silent act available to it.

Both answers are recorded, because both are claims. *"The table was already one row per sample"* is
a sentence a methods section can carry; without it, a table that was checked and a table nobody
looked at read identically (§09's recorded-absence rule).

Two refusals, and each has a reason worth reading rather than a validation message: **duplicate
feature names** are refused, because two rows with one name become two columns with one name and
every consumer downstream silently sees whichever one pandas hands it; and a **name collision with
`sample_id`** is refused for the same reason applied to the identifiers.

### 2 · The target

Unchanged from the built step, with the positive-class question already specified: for a two-level
outcome the interesting question is **which level is the event**, never "is this binary." Never
pre-selected at any confidence — "alive/dead" has no correct default, because whether the event is
death or survival is the research question.

### 2.5 · What the model is for — prediction or inference

> **What is this model for?**
> Both are legitimate and they are optimized for different things, so several
> later answers change depending on which you say. Nothing in the file reveals
> it — only you know what the paper claims.
> — **Predicting an outcome for a new person**
> — **Estimating how strongly something is associated with the outcome**

**The deepest of the seven research convergences** (`DOMAIN_SCIENCE.md` §01.3).
The same dataset, the same target and the same lens require **opposite** handling
in at least five places across four domains — missing labs, values below the
detection limit, repeated recalls, a 30-item instrument, features versus
compounds. *The advice inverts.* A tool that gives the inference answer to
somebody building a bedside model is wrong, and so is the reverse.

A **CHOICE** by the routing constitution: always asked, never skippable at any
confidence, and never defaulted at any confidence — nothing in the data reveals
it, and a pre-selected purpose would be the app deciding what the user's paper
is about.

**It earns its place by §00's own test** — *if the answer were wrong, would a
downstream number be wrong or misleading?* It fires once and changes the default
on roughly a dozen decisions per pack, which is the best ratio in the sequence.

The first consumer is the missing-data route, and it is the clearest case: a
was-it-missing indicator carries the clinician's decision to order a test, so it
is observable at deployment and legitimate for prediction, and a known source of
bias in an association estimate. Under an inference objective it is blocked with
both exits — resolve by imputing inside the folds, or attest and carry it as a
stated limitation. The second is the class-imbalance advice, where the app was
recommending a step that damages calibration and writing it into the manuscript
(`GUIDED-049`).

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
| *a transposed copy of* `metabolomics_untargeted` | 396 × 81, feature ids in the first column, sample ids for headers | question 1.5 fires · the target question is **withheld** until it is answered · answering *features in rows* produces an 80 × 397 table and a methods sentence · the lens contradiction detector stays **silent** |

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
