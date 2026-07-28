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

# The denominator moved again — this time without moving a number

**Third adjudication, L9c.** Same discipline. Recorded before the change landed, not after.

## What happened

The product owner ruled that **the target's question is different in kind from a feature's**. For
a feature, binary-versus-numeric is a *reading* — the values mean the same either way and the
question is how to store them. For the outcome the reading is nearly forced (two-level text is
binary classification) and the decision that matters is **which level is the event being
predicted**, because that sets the sign of every effect estimate, what sensitivity and specificity
are the sensitivity and specificity *of*, and what the model is trained to detect.

So `ml/binary_text.py` now routes the target column to `positive_class_finding` — *"Which of these
is the event you are predicting?"* — instead of `binary_text_finding`. The required-decision
inventory changed accordingly:

| dataset | removed | added |
|---|---|---|
| messy-clinic | `repair::binary_text__outcome` | `repair::positive_class__outcome` |
| longitudinal | `repair::binary_text__outcome` | `repair::positive_class__outcome` |
| wide-assay | — | — (its outcome is already `int64`) |

## Both readings

**Every number is unchanged.** The denominator's *size* did not move: messy-clinic stays at 10,
longitudinal at 2, wide-assay at 1. Its *composition* did — one required decision was replaced by
another. Guided asks the new question, so:

| dataset | Classic | Guided | Guided @ pinned n |
|---|---|---|---|
| messy-clinic | 1/10 @ `23a9123` | 10/10 @ `23a9123` | 9/9 @ `6bfe598` |
| wide-assay | 1/1 @ `23a9123` | 1/1 @ `23a9123` | 1/1 @ `6bfe598` |
| longitudinal | 1/2 @ `23a9123` | 2/2 @ `23a9123` | 1/1 @ `6bfe598` |

Every threshold passes, under both denominators, as before.

## The ruling

**The new composition is binding. The measurement is recorded as a new row** —
`routing-baseline-l9c.json`, superseding `routing-baseline-l9.json`, which stays on disk so the
chain is readable.

## What this exposed, which is the part worth keeping

**A composition change moves no metric, so the drift detector could not see it.** Every guard we
had compared numbers; this change swapped one required decision for another and every number
stayed identical. It would have passed silently.

Two things now close that:

- `test_the_adjudicated_reference_differs_from_the_frozen_one_only_as_ruled` compares the
  **inventory keys**, not just the metrics, against a table of what the adjudication accounts for.
- `test_both_doors_are_scored_against_one_inventory` asserts the harness's core promise directly.
  It had quietly stopped holding: `_run_guided` diagnosed *without* the target while
  `api.py::_recompute` diagnoses *with* it, so the harness was measuring a door that no longer
  existed — and once the engine's answer depended on whether a target was known, that gap would
  have scored the two doors against different denominators while still producing numbers. Filed as
  `T0-BUILD-005`.

**The general form, worth stating:** *a guard that compares values cannot see a change of
identity.* The ledger has the same shape of hole wherever a test asserts counts and not names.

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

---

# The grain question scores as noise

**Measured at `0c16c81`. The movement entered at `4152020` — L13 task 1, "clause 02: the grain
question ships in the Guided door" — and rode through L13 and L14 before the full suite was next
run end to end. I wrote that commit, and I did not notice. That is the first fact in this entry
because it is the one that generalizes: a check that is only consulted when someone remembers to
run everything is a check with a latency, and this one's latency was two loops.**

## What happened

Guided now asks one more question on every dataset, and the metric counts all three as irrelevant.

| dataset | asked | irrelevant | findings-driven | coverage |
|---|---:|---:|---:|---:|
| messy-clinic | 10 → **11** | 0 → **1** | 0.90 → **0.8182** | 1.0 (unchanged) |
| wide-assay | 1 → **2** | 0 → **1** | 0.00 (unchanged) | 1.0 (unchanged) |
| longitudinal | 2 → **3** | 0 → **1** | 0.50 → **0.3333** | 1.0 (unchanged) |

The question is `state_grain`. Coverage did not move on any dataset, `required_decisions` did not
move on any dataset, and Classic's column did not move at all. One question in, one question
counted as noise, three times over.

The arithmetic is not in dispute. `irrelevant_questions = max(0, questions_asked −
required_decisions)`, and `required_decisions` is built from the engine's findings — `choose_target`
and the `repair::*` family. There is no grain key in that inventory, so a question that covers
nothing in it is, by construction, surplus.

## Both readings

**The metric is right.** Guided asks a question the dataset did not raise. That is exactly what
`irrelevant_questions` was defined to count, and the definition was pre-registered before any
Router code existed. A door that asks an unprompted question on a clean file has asked an
unprompted question, whatever its reason.

**The inventory is stale.** `required_decisions` derives from findings, and clause 02 is the one
requirement in this project that is deliberately *not* finding-driven: the grain question is asked
because the answer cannot be inferred, and asking it only when a detector fires is precisely the
failure `IMPORT-020` and `IMPORT-022` are. Under this reading the inventory has no key for a
decision the constitution makes mandatory, so the door is charged for obeying it.

The sharpest form of the second reading is `longitudinal`. That dataset **is** repeated measures —
it is the one file in the fixture set where the grain answer changes how the holdout is drawn — and
on it the grain question is the single most consequential thing the door can ask. It scores as the
one irrelevant question on that row.

## The ruling

**The worse numbers are recorded. The denominator does not move.**

Adding a `grain::` key to `required_decisions` would take `irrelevant` back to 0 on all three
datasets and lift messy-clinic's coverage from 9/10 to 10/11. It is also the reading that flatters
the door I built, argued by the agent that built it, in a document whose standing rule is that a
moved denominator is an adjudicated act and not a convenience. The second reading may well be
correct. It is not mine to bank.

So: `routing-value-check-l15.json` is written beside the old result with the unflattering numbers
in it, this entry states what moved and why, and `routing-value-check.json` is replaced carrying
its reason inside the file. Whether the inventory should learn about clause 02 is filed as
`GUIDED-018`, open, for the product owner.

**The verdict did not move.** `passes: true`, `passes_under_literal_reading: false` — both
identical to the recorded result. Every threshold still holds with room: messy-clinic's irrelevant
count is 1 against a pre-registered ceiling of 4, the two guard datasets are 1 against 3, and
messy-clinic's findings-driven floor is 0.5 against an achieved 0.8182. Nothing here is close to a
`BLOCKED.md`. The drift check fired on a metric moving, not on a threshold breaking, which is what
it is for.

## What this does not settle

- **Whether the constitution and the metric can both be satisfied.** If the answer is that clause
  02's question belongs in the inventory, then every future constitutional question — eligibility,
  missingness routing, the assembly grain — arrives with the same problem, and the fix should be
  general rather than a key per clause.
- **The two-loop latency.** The drift check works; nothing ran it. That is not fixed by this
  entry, and the value-check suite is too slow for the pre-commit hook that now guards the other
  four gates. Filed as `GUIDED-019`.
