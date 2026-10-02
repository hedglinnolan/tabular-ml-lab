## 13 · The method contract — how domain methods enter the pipeline (2026-10-02)

Nolan asked how domain methods, such as QC-based drift correction, weave into the general modeling
process. Every domain method declares a **method contract** in a registry, just as model families
do, and the engine enforces it:

- **Slot.** Where it runs: ingest · repairs · reshape · eligibility · *(seal)* · in-fold steps ·
  model · evaluation.
- **Data scope.** What it may learn from:
  - **row-local**: needs no other row; a structural repair, before the seal;
  - **reference rows**: learns only from technical replicates such as pooled QCs, blanks or
    standards — never from participants or the outcome; a measurement correction, before the
    seal, applied per batch;
  - **training fold**: learns from study rows, so it is fitted in-fold and never sees held-out
    rows;
  - **descriptive**: may read every row to answer "is this data corrupted?" but informs no
    modeling choice.

  Lockbox constitution §06's test decides the scope: does row *i*'s output depend on other rows, or
  on the outcome?
- **Needs.** The roles and columns it requires (for example run order and QC labels). Intelligence
  supplies these as evidence that the method is needed.
- **Routing.** Its question, where it sits in the sequence, its options labeled customary versus
  sound for each purpose (North star 5), and its leash rung for each purpose (§11.3).
- **Storyboard.** Its real, labeled steps for the transform player.
- **Sentence.** The methods sentence, including any departure from convention with its
  justification.

**The worked example.** QC-RLSC drift correction (Dunn et al. 2011) has scope *reference rows* and
runs before the seal. The order is: correct drift from the QC rows over run order → the QC rows
leave the cohort (an eligibility step) → the seal → PQN, log and scaling in-fold. The contrast
matters: PQN learns its reference spectrum from study samples, so its scope is *training fold*.

The per-domain completeness pass (after routing, before presentation resumes) adds methods only
through this contract. That pass covers QC drift correction, scale scoring with reliability and
attenuation, batch handling, and usual-intake modeling.

**Relations: how one action leads to another** (Nolan: *"we may need to be creative/prudent as well
in terms of tying it all together how one action leads to another"*). The connective tissue between
methods is the product, not a by-product. Each contract also declares its relations to other
decisions:
- **implies**: follow-ons the app *states* rather than asks (drift correction → the QC rows leave the
  cohort);
- **enables / disables**: downstream options that open or close (a log transform changes what a
  substitution curve means; complete cases under inference arms the row-loss concern);
- **invalidates**: answers that must be *re-asked*, never silently kept, when this one changes;
- **conflicts**: pairs that cannot coexist, refused with an exit that names the way forward.

The Router derives order from these relations wherever it can, instead of from a hand-written list.
The record and the canvas show the chain ("because you chose X, Y now…"). A **chain test** asserts
that every implied consequence appears in the participant flow, the lineage and the methods
sentence.
