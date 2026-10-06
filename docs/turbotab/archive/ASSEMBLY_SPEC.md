# Multi-file assembly — the specification

**Why this file exists.** The research behind it lived in an agent's scratchpad, which is
ephemeral, while its conclusions survived only as seven ledger rows filed by a builder. That is
precisely the failure `FEATURE_PARITY.md` names as the ephemeral-pointer rule — *a record that
points at volatile storage will eventually lie, and it lies toward "the work is gone."* Written
here so the reasoning outlives the session that produced it.

The product decision is settled (`PRODUCT_VISION.md` §04b, "Guided is never the less capable
door"): **Guided ships multi-file assembly.** Combining files is the interaction that most needs a
dynamic surface, because a researcher decides correctly only when they can watch the working table
morph under each join, merge or split. Routing them to Classic for that would route them to the
one place the morph cannot be shown.

---

## 01 · Scope and gating

**Release 1 — append (concat) and join/merge**, built on `ml/join_doctor.py`, wrapped in the
grain-first interview and the row-accounting receipt below. The danger and the value both
concentrate here.

**Release 2 — longitudinal reshape (wide↔long) and fuzzy / tolerance matching.** Reshape is *not*
an advanced extra — cross-sectional versus longitudinal is part of what the user is deciding about
(`PRODUCT_VISION.md` §04b). It is staged only because thresholds and time-grain add a second axis
of cognitive load, not because it is peripheral.

**Gate.** Construction may not begin until `TRANSITION_PLAN.md` §05's three freeze gates are met.
The engine substrate is capable but its defect state was unknown for most of this project's life;
a static receipt asserts once, and a live morphing preview asserts at every frame.

`join_doctor.py` already exposes `KeyCandidate` (with `.confidence`, `.distinctness`,
`.repeats_on_both_sides`, `.headline()`), `JoinDiagnosis.can_proceed`, `find_key_candidates`,
`detect_nested_ids`, `execute_join`, `plain_summary` and `suggest_best`. R1 is predominantly
interaction design, not new engine capability.

---

## 02 · What the research established

### Named patterns

| Pattern | Source | What it does |
|---|---|---|
| **Append vs Merge as two verbs** | Power Query; Tableau Prep; Alteryx | Stacking rows and adding columns are separate commands with separate dialogs. The commonest novice error is reaching for the wrong operation entirely. |
| **Venn picker with live matched/unmatched counts** | Tableau Prep | Join type chosen by clicking a Venn region; a bar chart reports rows from each input, matched and unmatched, unmatched in red, clickable through to the rows. |
| **Three physical outputs (L / J / R)** | Alteryx Join | Unmatched left, matched, unmatched right are all *materialized*. Nothing is silently dropped; "which participants fell out?" is a first-class object. |
| **Declared cardinality that fails loudly** | `pandas.merge(validate=)`; `dplyr` `relationship=` | The safe path is the default and the dangerous path requires an explicit statement of intent. dplyr warns on many-to-many by default because it "often means you have a mistake." |
| **Inferred but editable keys** | Dataprep by Trifacta; Power Query | The machine pre-fills keys and join type and surfaces them for confirmation. Dataprep's own docs warn the inference can be wrong. |
| **Fuzzy match as opt-in with a visible threshold** | Power Query fuzzy merge (default 0.80) | Approximate matching is never the default and its looseness is always on screen. |
| **Restructure wizards that ask for roles** | SPSS Restructure Wizard; `tidyr` pivot pedagogy | Never asks a clinician to know the word "pivot" — asks what an ID means and what a time-point means. |
| **Long format with explicit repeat markers** | REDCap (`redcap_repeat_instance`, `redcap_event_name`) | The structure carries its own metadata about what a row means. |

### Principles

1. **Grain before keys.** *What does one row represent in each file?* is load-bearing; key
   selection is downstream of it. A 1:1 versus 1:many merge is only interpretable once both grains
   are known.
2. **The machine proposes, the human disposes.** Every inference is a confirmable proposal, never a
   silent commitment.
3. **Account for every row.** Matched *and* unmatched rows are materialized, counted, inspectable.
4. **The safe operation is the default; danger requires a deliberate act.**
5. **Preview the consequence in rows and participants**, not in SQL semantics. *"240 patients
   become 1,180 rows"* beats *"inner join, m:m cardinality."*
6. **Name operations by intent.** *"Add more patients"* / *"Add more measurements"* beats
   *"Union"* / *"Join"* for a clinical audience.
7. **Reshape is a role-assignment interview, not a formula.**

### Documented failures

- **Silent row multiplication.** Duplicate keys on both sides produce a Cartesian product; 3 × 2 =
  6 rows per key. In clinical data this fabricates person-time and inflates n, biasing every
  downstream statistic. This is not hypothetical here: `docs/FINDINGS_LEDGER.md` records
  `age ↔ age` across two survey cycles rated **high** confidence delivering 144 Cartesian rows,
  with the existing guard passing because promised equalled delivered.
- **Silent participant drop-out.** An inner join quietly discards anyone missing from either file;
  without row accounting the cohort shrinks invisibly and selection bias enters with no trace.
- **Key type and format mismatch.** `"007"` against `7`, or trailing whitespace, matches zero rows
  and reads as "no overlap."
- **Wrong-key inference rubber-stamped.** Confirmation must be a real decision, not a pre-checked
  box.
- **Fuzzy matching as a silent default.** Loose thresholds fabricate matches between distinct
  entities.
- **Aggregation-during-merge ambiguity.** When a per-patient value meets per-visit rows, whether it
  broadcasts or aggregates must be named, not buried.
- **The empirical base rate.** Roughly 30% of papers with Excel gene lists contain conversion
  errors, and the rate did not improve across 2014–2020 despite publicity. Awareness does not fix
  this class; the guardrail has to be structural.

---

## 03 · The interaction

### Step 0 · Operation intent, in clinical language

> You're adding a second file. Which is it?
> - **More participants** — same measurements, new people
> - **More measurements** — same people, new columns

Routes to concat versus join and prevents the wrong-operation error before anything else happens.

### Step 1 · Grain — the keystone

> **What does one row represent?**

Asked once per file, pre-filled from uniqueness statistics and **confirmed by the human**. This
determines the expected relationship (1:1, 1:many, many:many) before any key is chosen.

**This is the same question the lockbox needs** — see §05.

### Step 2 · Relationship inference with editable keys and live preview

The proposed key from `suggest_best` with a plain-language headline: *"These files share
`patient_id`; it is unique in File A and repeats in File B, so each patient's demographics will
attach to all of their visits."* The Venn and bar accounting update live as the key changes.
Low-confidence candidates are withheld from auto-selection but revealable.

### Step 3 · The row-accounting receipt, before commit

Screenshottable, and written to be pasted into a methods section:

> File A: 240 patients. File B: 1,180 visits (240 patients).
> **Matched:** 235 patients / 1,150 visits.
> **Dropped from A:** 5 patients had no visits — *excluded by inner join.*
> **Dropped from B:** 30 visits belong to patients not in A.
> **Result: 1,150 rows, one per patient-visit.** Your row count went from 240 to 1,150 because
> demographics were copied onto each visit; this is expected for a one-to-many match.

Every number is clickable through to the actual rows.

---

## 04 · Question grammar (`DESIGN_LANGUAGE.md` §09)

| Tier | Applied to assembly |
|---|---|
| **FACT** — inferable, skippable with a rendered skip | Key detection and normalization (`"007"` matching `7`); per-file grain inference; join-type default; column alignment for append. Auto-resolved chip, expand to change, nothing blocks. |
| **CHOICE** — always asked, preview before apply | Operation intent; grain confirmation; join type and which participants to keep; whether a per-patient value broadcasts onto visits. Live Venn plus row-count delta. |
| **CONSEQUENCE** — blocker, resolve or typed attestation | Row multiplication (duplicate keys on **both** sides); a key whose evidence contradicts the match; disjoint key ranges suggesting different populations; sub-threshold fuzzy matching (R2). |

**The escalation rule, and it is not a percentage.** Escalate on *evidence that the join is
wrong*, never on the magnitude of its consequence:

- **Row multiplication** → blocker, unconditionally. Fabricating rows is almost always an error.
- **Weak key evidence with a low match rate** → blocker. The key is probably wrong.
- **Disjoint key ranges** (File A holds 1–240, File B holds 500–740) → blocker. Different
  populations, not a cohort choice.
- **Any participant drop** → a prominent CHOICE stating the exact count. Losing rows is common and
  often correct; a fixed threshold would falsely imply that losing 19% needs no thought.

A percentage cutoff manufactures a line the data cannot support. The failure mode is not losing
participants — it is losing them *silently*.

---

## 05 · The grain question is shared with the lockbox

The keystone of assembly turns out to be the same question the seal needs (`ROADMAP.md`, lockbox
constitution §02): **"Can one person appear in more than one row?"**

- Asked **once**, pre-seal, and recorded once.
- A project arriving through assembly has already answered it; **the seal inherits the answer**
  (`basis_source: inherited_from_assembly`).
- A single-file project is asked directly (`basis_source: user_stated`).
- The heuristics (`detect_repeated_subjects`, `rank_grouping_candidates`) are demoted from source
  of truth to a *suggestion* and a *contradiction detector*. A user who states one row per person
  while a column repeats three times per value is evidence somebody is wrong — that earns an
  interruption, by the same escalation rule as §04.

`IMPORT-020` and `IMPORT-022` exist because the app guessed at this instead of asking, and a
failed guess rendered as a clean lock over a real leak. Name lists and ratio bounds cannot close
it and must not be tuned as though they could.

---

## 06 · What the dynamic surface changes

The static tools interrogate the user about grain because a form cannot show it. **A surface that
animates the working table can let the user watch one-row-per-patient become one-row-per-visit.**
That is the capability the web format buys, and it answers the research's own strongest open
question — whether clinicians can correctly answer an abstract grain question about their own
files.

**Ask only what showing cannot answer.** Where the engine genuinely cannot tell (`IMPORT-015`),
showing both candidate shapes and letting the user pick is exactly what a live preview makes
possible, and guessing is exactly what it makes unnecessary.

---

## 07 · Acceptance criteria — from the audit, not invented

These are the seven requirements the recovered audit produced. They are the build's gate, and
freeze gate 3 requires each to have a named test.

| Row | Requirement |
|---|---|
| `IMPORT-001` | The row-multiplication blocker. Refuse **before** animating; a legitimate one-to-many must still pass. |
| `IMPORT-005` | The grain question, and why it must not be guessed from column names. |
| `IMPORT-015` | Grain where the engine cannot tell — show both shapes and let the user pick. |
| `IMPORT-007` | The change map must match the executed frame. This is the precondition for a preview existing at all: "show rather than ask" is only true if what it shows is what will happen. |
| `IMPORT-006` | Blanks the merge *created* must look different from blanks it *inherited*. Two cells that look identical and mean opposite things. |
| `IMPORT-011` | The consequence prose the preview narrates. Watching a table change only teaches if the app can say what the change means. |
| `IMPORT-014` / `IMPORT-017` | The consequences invisible in the shape — a stack that looks perfect while duplicating people, and a dtype coercion the planner knows about and the map omits. This is what the narration is for. |

---

## 08 · Open questions

1. **Does the grain question survive contact with real clinicians?** Untested. §06 argues the live
   preview reduces the exposure, but a wrong grain answer still poisons everything downstream.
2. **Typed-attestation friction versus abandonment** for non-programmers. The reflexive-comply
   failure mode is real and unmeasured in this audience.
3. **Broadcast versus aggregate** when a per-patient value meets per-visit rows — researchers are
   documented as genuinely confused, and no source establishes a safe default.
4. **Fuzzy date-window matching** (labs within N days of a visit) has no consensus window — R2.
5. **Three-plus file assembly** compounds grain and multiplication risk across steps; the receipt's
   clarity at step four of a chain is untested.
6. **Whether the receipt improves published methods** is plausible and unproven. The 30% gene-error
   base rate motivates the intervention; it does not validate this specific one.
