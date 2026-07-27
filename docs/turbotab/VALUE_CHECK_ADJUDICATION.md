# Routing value check — adjudication note

**The pre-registration is not edited.** This note is how a pre-registered study handles a
protocol ambiguity discovered during analysis: the deviation is published beside the protocol,
with the ruling and its grounds, and the original stays frozen. `VALUE_CHECK_PREREG.md` reads
today exactly as it read at `e14af90`, before any Router code existed.

## The ambiguity

`deferral_closes` is `None` on both clean datasets, because they contain nothing deferrable —
one required decision each, zero repairable findings. The prereg says two things about this
metric that conflict at the edge case:

- the headline: *"exactly 1.0"* — under which `None` fails on two of three datasets;
- the definition one sentence later: *"a single deferred item that fails to resurface at a step
  that can act on it is a bug that fails the check outright"* — under which an empty set of
  deferrals is vacuously satisfied.

The ambiguity is a drafting defect in the pre-registration, and the pre-registration was written
by the adjudicator. The builder did not create this problem; the protocol did.

## The ruling

**Reading (B) — vacuous satisfaction — is the binding interpretation. The check passes.**

Grounds, in order of weight:

1. **The prereg's own definition of failure is an event, not a quota.** The failure condition is
   *a deferred item that fails to resurface*. No deferrals, no such item. The headline number was
   shorthand for that definition, not a separate requirement.
2. **Reading (A) contradicts the prereg's own clean-dataset guard.** Requiring a deferral to
   occur on data with nothing to defer would manufacture exactly the ceremony that the guard two
   paragraphs earlier exists to forbid. A protocol must not be read so that one clause can only
   be satisfied by violating another.
3. **The prereg itself treats not-applicable as legitimate** — it calls Classic's `NaN` on this
   metric *"correctly"* recorded. The same logic extends to Guided on a dataset with nothing to
   defer.

## Why the process holds

The builder made the (B) determination after seeing the failure — the exact hazard
pre-registration exists to prevent — and then did the three things that keep the discipline
intact: did not edit the prereg, computed **both** verdicts into
`data/routing-value-check.json` (`passes_under_literal_reading: false` is permanent record),
and referred the ruling upward instead of resolving it silently.

That is what makes this note an adjudication rather than a rationalization: the adverse reading
is preserved in the data, the protocol is unmodified, and the ruling is published by the party
that wrote the ambiguous clause, against its own drafting.

## Precedent set for future preregs

- Metrics that can be structurally inapplicable must state their empty-set behavior explicitly.
  The correct phrasing, for the record: *"1.0 over the deferrals that occur; vacuously satisfied
  when none occur."*
- A harness facing an ambiguous threshold computes **every** reading and records all of them.
  The builder never picks silently — the divergence goes in the data and the ruling goes in a
  note like this one.

---

# The denominator moved

**Second adjudication, L9.** Same procedure, same discipline: the frozen artifact is not edited,
both readings are computed and preserved in data, and the ruling is published here.

## What happened

`ml/binary_text.py`, built at L9 for `GUIDED-001`, detects a column holding two distinct
non-null values and proposes reading it as binary rather than coercing it to numbers. On two of
the three pre-registered datasets that detector fires on the **outcome column itself**:

| dataset | outcome column | levels |
|---|---|---|
| messy-clinic | `outcome` | `responder` / `non-responder` |
| longitudinal | `outcome` | `improved` / `stable` |
| wide-assay | `responder` | already `int64` 0/1 — no finding |

So `repair::binary_text__outcome` entered the ground-truth inventory that
`turbotab.measure.required_decisions` derives from the engine's findings, and the denominator of
every coverage metric grew.

**Neither door's behavior changed.** Classic renders `import_doctor.diagnose` directly and is
frozen as engine-move-only (`TRANSITION_PLAN.md` §05), so it cannot learn the detector and its
numerator stayed at 1. Guided raises the new question because the Router plans from the merged
finding stream. What moved is the measuring stick, not the thing measured.

## Both readings

Guided measured once, scored against both denominators. Thresholds are the pre-registration's,
unmodified.

### messy-clinic — the contested claim

| | frozen denominator (n=9) | adjudicated denominator (n=10) | threshold |
|---|---:|---:|---|
| Classic covered | 1 | 1 | — |
| **Classic coverage** | **1/9 = 0.111** | **1/10 = 0.100** | — |
| Guided surfaced coverage | 9/9 = 1.000 | 10/10 = 1.000 | ≥ 1.000 · **pass** |
| Guided asked coverage | 9/9 = 1.000 | 10/10 = 1.000 | ≥ 8/9 = 0.889 · **pass** |
| Guided questions asked | 10 | 10 | ≤ 17 · **pass** |
| Guided irrelevant | 1 | 0 | ≤ 4 · **pass** |
| Guided findings-driven | 0.90 | 0.90 | ≥ 0.50 · **pass** |

### wide-assay — regression guard

Unmoved: the outcome is already numeric, so no finding, no denominator change. Classic
1/1 = 1.000; Guided 1/1 = 1.000, 1 question asked, 0 irrelevant. **Pass under both.**

### longitudinal — regression guard, and the one the ruling did not anticipate

| | frozen denominator (n=1) | adjudicated denominator (n=2) | threshold |
|---|---:|---:|---|
| Classic covered | 1 | 1 | — |
| **Classic coverage** | **1/1 = 1.000** | **1/2 = 0.500** | — |
| Guided coverage | 1/1 = 1.000 | 2/2 = 1.000 | ≥ 1.000 · **pass** |
| Guided questions asked | 2 | 2 | ≤ 10 · **pass** |
| Guided irrelevant | 1 | 0 | ≤ 3 · **pass** |

**The check passes under both denominators, on all three datasets.** Guided asks the new
question, so its coverage is 10/10 rather than 9/10; had it not, the honest result would have
been a failure and `BLOCKED.md`.

## The ruling

**The adjudicated denominator is the binding one, and the frozen baseline stays frozen.**

Grounds:

1. **The inventory is ground truth, not a door's output.** `required_decisions` derives from the
   engine's findings, and the engine now finds a decision it previously missed. A denominator
   that omits a real required decision was measuring the wrong thing, generously, for both doors.
2. **The movement is attributable to one named cause and is fully enumerated.** Six numbers moved
   across two datasets, all consequences of `repair::binary_text__outcome`.
   `test_the_adjudicated_reference_differs_from_the_frozen_one_only_as_ruled` asserts exactly
   those six and the one added key, so a *second* drift cannot hide inside this one.
3. **It cuts against the builder, not for.** Classic's coverage falls under the new denominator
   on both affected datasets, and Guided's requirement rises from nine questions to ten. A
   denominator change that made the result more flattering would deserve more suspicion than
   this one does.

`data/routing-baseline.json` is unmodified — measurements byte-identical to `6bfe598`, verified
in `test_the_frozen_baseline_is_the_one_the_prereg_names`. The new measurement sits beside it as
`data/routing-baseline-l9.json`.

## What this does not settle

- **Whether the binary reading of an outcome column is the right question to ask.** It is a real
  decision — which level is positive determines the sign of every effect estimate, and the
  detector correctly declines to guess for `responder`/`non-responder`, marking it medium
  confidence so nothing pre-selects it. But `ml/binary_text.py` does not distinguish a feature
  from the target, and for a target the interesting question is the positive class rather than
  the storage type. If the answer is that the target deserves a differently-worded question, the
  denominator moves again and this note gets a successor.
- **Longitudinal's guard value.** Classic scoring 0.500 on a two-decision dataset is a weaker
  regression guard than 1.000 on a one-decision dataset was. The prereg's own reading — that the
  clean datasets are guards and not evidence — is unaffected, but the guard is now looser.

---

# Coverage carries its denominator

**A standing ruling, not a one-off.** It comes out of the adjudication above, and it survives it.

## The thing neither of us designed

Classic's coverage **numerator is structurally frozen.** `pages/01` renders
`import_doctor.diagnose`, and `ml/import_doctor.py` is frozen as engine-move-only
(`TRANSITION_PLAN.md` §05), so Classic cannot learn any detector the engine gains — not the
binary-text one, not the next one.

The **denominator is not frozen.** `required_decisions` is derived from the engine's findings on
purpose, so that neither door's UI biases the measuring stick. Every time the engine learns to see
something new, the denominator rises.

Put those together and the gap between the doors widens **on its own**, with no change to either
door's routing. Classic's coverage on messy-clinic went 1/9 → 1/10 at L9 because the engine got
better, not because Classic got worse. Left alone, the headline claim inflates itself loop after
loop while measuring nothing new about the thing it exists to measure.

## The rules

1. **Every coverage figure is reported as `k/n @ <commit>`.** A bare ratio is not a result. Two
   ratios with different denominators are not comparable, and nothing in a decimal says so.
2. **The pre-registered claim stays pinned at `n=9` in perpetuity.** That is the number the
   thresholds were banked against and it does not move. `routing-value-check.json` reports Guided
   under the pinned denominator alongside the current one, on every run.
3. **Re-measurements are new rows, never restatements.** A moved metric gets a new file beside the
   old one and a ruling here — `routing-baseline-l9.json` is the first.

## The caveat, in plain words

> Classic's numerator cannot grow, because the import path it renders is frozen. A widening gap
> between the doors is therefore not by itself evidence of better routing: part of it is the
> engine improving underneath both doors while only one of them can act on the improvement.

We would rather publish the caveat than enjoy the number. Anyone quoting a coverage figure from
this project quotes it with its denominator and its commit, or is quoting something that has been
drifting upward without being measured.

**When the freeze lifts, this caveat gets revisited, not deleted** — Classic will be able to learn
new detectors, and the two doors will start moving for comparable reasons again. Until then the
gap is partly an artifact and is labeled as one.

---

## Precedent added

- **A protection that depends on "X does not exist yet" expires the moment X exists, and nothing
  will tell you.** The baseline harness carried that guarantee in its docstring and kept writing
  the file for three loops after the Router landed. Recorded in `FEATURE_PARITY.md` beside the
  principle-locality corollary, and filed as `T0-PREREG-002`.
- **The measurements are frozen; the envelope may gain labels, never lose or alter one.** The
  first phrasing — *never edited, ever* — was too blunt to state what it protected, and would have
  forced provenance into a second file nobody checks. Safe because nothing depends on trusting the
  envelope: a self-declared stamp is swappable, so the load-bearing assertion is the git-read
  values check. Recorded in `FEATURE_PARITY.md` as the frozen-measurement rule.
- **Measurement and comparison must not share a code path.** A suite that re-measures its own
  reference has no reference. `tests/integration/test_routing_baseline.py` compares;
  `scripts/remeasure_routing_baseline.py` measures, refuses to overwrite a frozen baseline, and
  prints what moved.
- **When ground truth moves, report both denominators.** Not the more favorable one, and not the
  older one out of caution — both, in the same table, with the thresholds applied to each.
