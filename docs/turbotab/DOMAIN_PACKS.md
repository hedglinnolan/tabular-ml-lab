# Domain packs — how the app becomes field-aware without becoming a different app

**The problem.** TurboTab should be genuinely good at metabolomics, at dietary intake data, at
body composition, and eventually at fields nobody has named yet — while the interview stays one
card at a time. The naive reading of "support more fields" is "ask more questions," which is the
failure the whole product exists to escape: Classic asks ~32 questions regardless of the dataset,
and *that constancy is the indictment.*

**The resolution.** Domain knowledge should make the interview **shorter**, not longer. It mostly
changes the *answers*, not the *questions*.

---

## 01 · The opening question

> **What kind of measurements are in this table?**

Asked once, immediately after upload and **before the structural diagnosis**, because the
diagnosis is itself field-sensitive. Multi-select, because real studies are intersections — a
nutrition study with metabolomic outcomes is both, and that intersection is the first target
audience.

The product owner's framing, which is the reason this question exists at all:

> **It's a simple proxy for "what exactly is your research goal here?" — which we can't ask
> outright because it's a fuzzy question with a million possible answers. Tell us what type of
> data it is, and your research question will manifest to the app in the choices you make along
> the way.**

Two deliberate choices in the wording. It asks about **the data**, not about the person — a
statistician analyzing someone else's samples can answer "what is this" more readily than "what
field are you in," and the data framing is what the rest of the interview actually consumes. And
it says *measurements*, because that is the level at which a pack's detectors operate.

**"Not listed" is a first-class answer, not a dead end.** The app is fully functional with no
lens; a pack is an accelerator. Any design in which an unlisted field degrades the experience has
built a tool for four disciplines rather than a tool that is unusually good at four.

### It follows the grain question's architecture exactly

Because it is the same kind of question — one the user knows and the engine can only infer.

- **The user's answer is the answer.** Detection never overrides it.
- **The heuristic is a suggestion**, offered beside the options.
- **The heuristic is a contradiction detector.** If the answer is "clinical chemistry panel" and
  the table holds 1,847 log-normal features with a run-order column, that disagreement is
  evidence worth raising — by the escalation rule that governs everything else: *escalate on
  evidence that a reading is wrong, never on the size of the consequence.*
- **The answer is a recorded decision, not hidden state.** It produces a methods sentence, and
  that sentence is what licenses every domain default downstream. A lens the manuscript cannot
  see is a lens the reader cannot check.

### The first payoff arrives immediately

Wide assay data — 1,847 features, 80 samples — looks malformed to a general-purpose import
doctor, which is why the lens is set *before* diagnosis rather than after. A pack turns a false
alarm into a correct reading, which is the cheapest possible demonstration that the question was
worth asking.

---

## 02 · What a pack changes, and what it must never change

A pack supplies **detectors, reference data, conventions, concerns, and prose.** It converts
questions into stated facts.

The same dataset, without and with a metabolomics lens:

| | Questions asked |
|---|---|
| **No lens** | target · grain · eligibility · *"47 features are skewed — transform?"* · *"which scaling?"* · *"how should missing values be filled?"* · models · per-model config |
| **Metabolomics lens** | target · grain · eligibility · models |

Everything else becomes either a **rendered skip carrying its reason** —

> *Not asked: log-transform applied — concentration data is log-normal by construction, and this
> is near-universal in metabolomics.* — **Ask me anyway**

— or a **finding a generic tool would never raise**:

> **Your missing values cluster in the lowest-abundance features.** In metabolomics that usually
> means below the detection limit — left-censored rather than missing at random — and filling
> with a median would place non-detections in the middle of the distribution.

> **There is a run-order column.** Instrument drift is often the largest single variance component
> in a metabolomics run, larger than the biology.

Fewer questions, and every one that remains is about *this* data. That is the routing thesis one
level deeper than the Router applies it today.

### The unit of domain knowledge is a finding

This is the architectural claim that keeps breadth tractable. The app has exactly one way of
saying *the engine noticed something about your data*: a finding, with evidence, consequence,
severity, and a proposed action, routed by the existing Router through the existing tiers.

*"Your macronutrient columns sum to ~100% — these are compositional, and ordinary correlation
between them is not interpretable"* is a finding. It needs no new card type, no new severity, no
new component.

**Adding a domain means adding detectors and reference data. It never means adding interface.**

---

## 03 · The three guards

1. **A pack may not add interview components.** It supplies findings and defaults; it cannot
   invent a card type. This is what stops breadth from eating the design language.
2. **A pack must not fire on non-matching data.** Tested directly: run the generic fixtures
   through every pack and assert zero new questions. The value check's `irrelevant_questions`
   metric already measures this failure.
3. **Every default a pack pre-selects states its reason and is overturnable.** A pack raises
   confidence; it never removes the user's answer. Rendered skips, as everywhere else.

A fourth, on voice: **conventions are stated as conventions.** *"The field convention here is
Pareto scaling"* is honest. *"You should use Pareto scaling"* is not — the app never speaks in
the user's name (`DESIGN_LANGUAGE.md` §06), and a pack is the place where that rule is easiest to
break.

---

## 04 · What science earns its place

Scope is unbounded unless something decides. The rule comes from the product's own thesis rather
than from methodology:

> **Build the science that changes a sentence a reviewer would challenge. Skip the science that
> only changes what a practitioner would prefer.**

The product is the methods section. So:

- **Energy adjustment — in.** A nutrition reviewer rejects an unadjusted nutrient association.
- **Compositional handling — in.** Ordinary correlation on parts of a whole is not interpretable,
  and this bites both metabolomics and macronutrient percentages.
- **Detection-limit missingness — in.** *"Imputed with the column median"* is a sentence a
  metabolomics reviewer stops on.
- **Batch effects — in.** Uncorrected run-order drift invalidates the comparison.
- **Fifteen scaling variants — out.** The methods sentence is identical; only taste differs.

The filter also says when a pack is *finished*, which is what makes "supports metabolomics" a
claim rather than an aspiration.

---

## 05 · The risk that would embarrass us

**A pack that fires on the wrong data asserts something false in the one place the app has
promised it never will** — and it does so authoritatively, which makes it harder for the user to
catch than an ordinary bug. This is why the lens is *asked* rather than inferred, and why the
detector is demoted to suggestion and contradiction detector.

The discipline: if a detector cannot state its evidence in a sentence, it is not confident enough
to change a default.

---

## 06 · Open — the part that needs a domain expert, not an engineer

The structure above is buildable and testable without field expertise. The **content** is not.
Whether Goldberg cutoffs are the exclusion a user expects, whether residual energy adjustment or
nutrient density is the convention in a given subfield, whether QRILC or half-minimum is what a
reviewer wants to read — being confidently wrong on these poisons exactly the trust the app is
built to earn.

The reviewable form of that ask is narrow, and the copy deck is already its format: *here are
eleven default choices and the methods sentence each one produces — which would you object to?*
That is a document to mark up, not a consulting engagement.

Also open: whether the pack list is coarse (metabolomics) or fine (untargeted LC-MS metabolomics).
Coarse risks being wrong about a subfield; fine risks a list nobody finds themselves on. The
current lean is **coarse**, and it survives scrutiny for a specific reason: *a pack does not need
the label, it needs properties.* Detection-limit imputation is triggered by "missingness
concentrates in low-abundance features," not by "this is LC-MS." Batch correction is triggered by
"a run-order column exists," not by "untargeted." **The label sets priors; the data resolves them
into findings.** Nothing is asserted from the label alone, which is what makes a coarse list safe.

---

## 07 · The defaults, and the reasoning behind each

Set by judgment and by the mathematics rather than by a domain reviewer, because one was not
available. Each carries a **confidence marker**, and the marker governs the treatment: `derived`
defaults are pre-selected with their reason shown; `convention` defaults are pre-selected but
stated *as* convention; `offered` items are never defaulted at all.

**The principle that decided the hard cases:** where a statistically optimal method and a
transparently explainable one disagree, **prefer the explainable one.** The product is the methods
section, and *"values below the detection limit were imputed as half the minimum observed"* is a
sentence a reader can evaluate. A more sophisticated imputation that a reader cannot check buys
accuracy the app cannot spend.

### Metabolomics / proteomics

| Default | Confidence | Reasoning |
|---|---|---|
| log-transform | **derived** | Concentrations are bounded below by zero and combine multiplicatively; the resulting distribution is log-normal by construction, not by convention. |
| Pareto scaling | **convention** | Auto-scaling gives every feature equal weight including noise-dominated low-abundance ones; Pareto (divide by √SD) retains some magnitude information. A defensible compromise, not a fact — auto-scaling is offered beside it. |
| detection-limit imputation, half-minimum | **convention** | The *detection* is derived: missingness correlating with abundance rank is left-censoring, not randomness. The *method* is a choice, and half-minimum wins on explainability over QRILC per the principle above. |
| batch correction | **offered** | Fires only when a run-order column exists **and** intensity correlates with it. Detection is derived; correction is never automatic because it alters every value. |
| QC-RSD feature filter | **offered** | Requires pooled QC rows. Drops features, so it changes the analysis and must be chosen. 30% is convention, stated as such. |
| pooled QC rows excluded from modeling | **derived** | They are not participants. Modeling them is an error with no legitimate reading. |

### Dietary intake

| Default | Confidence | Reasoning |
|---|---|---|
| average repeated recalls | **derived** | A single 24-hour recall is a noisy estimate of usual intake, and that noise attenuates diet–outcome associations toward null. Averaging reduces it. This is measurement-error reduction, not information loss. |
| energy adjustment required | **derived** | Every nutrient association is confounded by total intake. That the adjustment is needed is not in dispute. |
| residual method as the default *form* | **convention** | Decorrelates the nutrient from energy explicitly, which makes the resulting coefficient interpretable. Nutrient density is offered beside it. |
| macronutrient compositionality flagged | **derived** | Columns summing to a constant are compositional; correlation between parts of a whole is negatively biased by construction. This gates the collinearity figure rather than adding a step. |
| implausible-intake exclusion | **offered** | Changes N, so it is an eligibility criterion the user states — never a silent filter. |

### Clinical measurements and labs

Mostly built. Physiologic plausibility bounds and unit harmonization exist and are now exact-match
rather than substring. The pack adds one prior: **missingness here often means *not ordered*,
which is informative in the opposite direction from metabolomics** — a test not run because the
clinician saw no reason to run it. The mechanism question already asks this; the pack supplies the
prior, not the answer.

### Genomics / transcriptomics — deliberately thin in v1

The pack recognizes the shape (extreme p, count data rather than concentrations, severe
multiple-testing burden) and sets p ≫ n priors on model ranking. **It asserts no normalization
default.** CPM, TPM and VST are not interchangeable and the choice depends on the assay and the
question; a thin pack that declines is honest, and a thick pack that guesses would be the
confidently-wrong failure this document exists to prevent.

### Survey instruments

Ordinal encoding is **declared, never frequency-derived** — the order comes from the instrument,
which makes it row-local rather than deferred. Reverse-coding requires a codebook the app does not
have, so it is **asked**, never inferred from item correlations.

---

## 08 · Packs change what is drawn, not only what is computed

A correlation heatmap of 1,847 features is a grey square. High-dimensional data needs a clustered
summary or a distribution of correlations, not the full matrix, and that is a *presentation*
decision the lens is better placed to make than the user.

So guard #1 — *a pack may not add interview components* — is scoped deliberately: it forbids new
**card types**, not new **figure choices**. A pack may change which visualization answers a
question; it may not invent a new kind of question to ask.
