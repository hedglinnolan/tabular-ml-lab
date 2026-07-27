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
