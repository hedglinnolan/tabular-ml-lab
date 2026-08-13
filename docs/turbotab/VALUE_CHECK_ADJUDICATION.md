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

---

# The grain question scores as noise — ruled

**The product owner ruled on `GUIDED-018` at L16. The recorded numbers are correct and they
stand permanently.**

> The prereg defines *irrelevant* as "absent from the decision inventory and cites no finding" —
> both conjuncts hold for grain, so your recorded numbers are correct and they stand permanently.
> Do not add a `grain::` key; that smuggles a new category into an old bucket. Instead name the
> category: report `constitutional` and `irrelevant_net = irrelevant − constitutional` alongside,
> never instead. The threshold keeps binding on literal `irrelevant`.

## The cause, which is the part worth writing down

The harness assumed **every legitimate question originates from a finding.** That assumption is
invisible because it is not written anywhere; it is embedded in an arithmetic identity —
`irrelevant = asked − required_decisions` — and `required_decisions` is built from what the engine
found. Under that assumption the identity is exactly right, and the metric has been correct for
every question the app asked until L13.

Clause §02 introduced a **fourth origin: asked because the app cannot know.** The grain answer is
not derivable from the table — that is the clause's entire premise, and asking it only when a
detector fires is precisely `IMPORT-020` and `IMPORT-022`. So the question cites no finding and
has no inventory key **by design**, and the identity charges the door for obeying the constitution.

**Clause §04's eligibility question is the second one**, landing in this same loop. Missingness
routing (§07) will likely be the third. That is why the fix is a category and not a key: a
denominator that gains an entry per constitutional clause moves every loop, and this document's
standing rule is that a moved denominator is an adjudicated act rather than a convenience.

## What changed, and what deliberately did not

| | |
|---|---|
| `irrelevant_questions` | **unchanged**, 1 on every dataset. The literal count is what the prereg defines. |
| thresholds | **unchanged**, and still applied to the literal count — 4 on messy-clinic, 3 on the guards. |
| `VALUE_CHECK_PREREG.md` | **unedited**. The pre-registration is not amended by a later finding about it. |
| `constitutional` | **new**, reported. 1 on every dataset — the grain question. |
| `irrelevant_net` | **new**, reported. 0 on every dataset. Nothing binds on it. |

`Question.clause` on the Router carries the label, and the harness reads it rather than deciding
for itself which questions are constitutional — a harness that classified its own subject would be
marking its own homework.

**Three conjuncts, all necessary**, or the field is a laundering mechanism: a question counts as
constitutional only if it names a clause **and** cites no finding **and** settles no inventory key.
A clause-bearing question that also covers a required decision is already counted where it belongs.

And the label has to resolve: `test_every_clause_a_router_question_claims_is_a_real_clause` reads
the constitution documents and fails on a question naming `lockbox-99` or a renumbered clause.
Without it, any question could name anything and vanish from the net reading.

## What this does not settle

- **Whether `irrelevant_net` is the more honest headline.** It is published, not promoted. If a
  later loop wants to quote it, that is a new adjudication and the prereg's ceilings do not
  follow it across.
- **Whether the Classic column has constitutional questions too.** It is measured from a frozen
  baseline and cannot gain the label, so its `constitutional` is structurally 0 rather than
  measured 0. A comparison of the two doors' net readings would therefore flatter Guided, and
  nothing in this project should quote it without that sentence attached.

---

# L19 · the lens is the third constitutional question, and the net reading did not move

**What fired.** `test_routing_value_check` failed on drift, not on a threshold, and it named eight
numbers across three datasets. The cause is one question: clause §01 places the lens first in the
pre-seal sequence, `STATE-106` recorded that neither door asked it, and this loop built it.

**What moved, measured rather than argued:**

| Dataset | `questions_asked` | `irrelevant_questions` | `constitutional` | `irrelevant_net` | `coverage` |
|---|---|---|---|---|---|
| messy-clinic | 11 → **12** | 1 → **2** | 1 → **2** | 0 → **0** | 1.0 → 1.0 |
| wide-assay | 2 → **3** | 1 → **2** | 1 → **2** | 0 → **0** | 1.0 → 1.0 |
| longitudinal | 3 → **4** | 1 → **2** | 1 → **2** | 0 → **0** | 1.0 → 1.0 |

`findings_driven` falls on two of the three — 0.8182 → 0.75 and 0.3333 → 0.25 — and it is the same
arithmetic seen from the numerator's side: the denominator gained a question that cites no finding.

**The ruling: this is the category L16 named, arriving for the third time, and it needed no new
machinery.** The lens is asked because *the app cannot know* — the fourth origin. It carries
`clause="lockbox-01"`, cites no finding, and settles no inventory key, so all three conjuncts hold
and `constitutional` picks it up automatically. That the category absorbed a question invented two
loops after it was defined is the strongest evidence available that it was defined at the right
level: the grain question (§02) was the first, eligibility (§04) the second, and this is the third,
with missingness routing (§07) still expected.

**Banking the new numbers, and why that is not the drift this check exists to stop.**

The distinction the pre-push message draws is between *re-recording a number because it moved* and
*adjudicating why it moved and then recording it*. This is the second: the cause is a named
question implementing a named clause against a tracked finding, the movement is exactly +1 per
dataset with no dataset moving differently, and the metric that carries the thesis is unchanged.

- **`irrelevant_questions` still binds, and still passes.** The ceilings are 4 on messy-clinic and
  3 on the guards; the literal count is 2. No threshold was moved and the prereg is unedited.
- **`irrelevant_net` is 0 on all three, unchanged.** Nothing binds on it and nothing is being
  claimed from it — it is reported because L16 published it, and it is quoted here only to say that
  the second reading did not move either.
- **`coverage` is unchanged at 1.0**, which is the metric the roadmap's value check makes stopping
  conditional on. The Guided door still reaches every required decision on every dataset.

**What this does not settle, and is worth watching.** `findings_driven` has now fallen twice for
this reason, and it will fall again with §07. It is a ratio whose denominator legitimately grows
with the constitution, so a threshold on it would bind against implementing clauses — the same
defect the `irrelevant_questions` identity had before L16. Nothing binds on it today. **If a later
loop wants to put a threshold there, it needs a constitutional-adjusted denominator first**, and
that is a new adjudication rather than a tuning.

---

# L21 · the pack's benefit fell because the comparison became fair

**Nothing here binds a threshold.** The routing value check's three datasets are
unaffected — none is wide and none sets a lens. This entry exists because the
*discrimination matrix* moved, and it moved for a reason that has to be recorded
beside the number or the number lies about which direction the product went.

**What happened.** `GUIDED-029`: every per-column question scaled linearly with
the column count. `metabolomics_untargeted.csv` has 308 columns with blanks, so
the interview asked **312 questions before any lens** — roughly ten times the
~32 this project calls Classic's indictment. The metabolomics pack rescued it to
6, which the L20 report published as **−307**.

That number was measured against a broken baseline.

| | L20 base | L21 base | metabolomics delta, L20 → L21 |
|---|---:|---:|---|
| `metabolomics_untargeted` | 312 | **7** | −307 → **−1** |
| `survey_instrument` | 6 | **4** | · → · |
| `dietary_recalls` | 3 | 3 | · → · |
| `clinical_longitudinal` | 3 | 3 | · → · |
| `genomics_expression` | 14 | 14 | −10 → −10 |
| `clinic_visits` *(control)* | 13 | 13 | · → · |

**The pack did not get worse. The baseline got honest.** Per-column questions are
now asked over sets defined by a rule — *"306 numeric columns have blanks;
answer once for all of them"* — so the interview no longer scales with p, and
there are no longer 300 questions lying around for a lens to remove.

**This is the coverage-denominator ruling again, from the other side.** L16
refused to add a `grain::` key to `required_decisions` because a denominator
that grows with the constitution stops being comparable across loops. Here the
*numerator* was the problem: a benefit measured against a baseline nobody would
defend is a number flattering itself, and quoting −307 after this loop would be
quoting a rescue from a fire the app was setting.

**What to quote, and what not to.** The honest claim for the metabolomics pack
is now three findings a generic tool would never raise — left-censored
missingness, instrument drift along run order, pooled QC rows that are not
participants — plus a **−1** on the question count and one grouped stated fact
covering 306 columns. That is a smaller number and a truer one.

**Both bases are recorded** so a later loop can tell a real regression from this
correction. A matrix cell that reads `−1` where the last report said `−307` is
not a regression, and the only thing that makes that legible is this row.

---

# L24 · `DRIVE-002` — the metric penalized the improvement, twice

## What happened

Bulk repairs landed: N findings that are the same repair became one question
with a selectable set. On `messy-clinic` that turns ten repair questions into
two groups plus three ungrouped ones, and the value check went red with

    verdict.passes:      True  → False
    questions_asked:     12    → 8
    irrelevant_questions: 2    → 0
    findings_driven:     0.75  → 0.625
    coverage:            1.0   → 0.4

Two of those five are the result improving. Two were the metric failing to see
it. One is real and is recorded as a cost.

## Coverage 1.0 → 0.4 — a metric regression produced entirely by the metric

`_record` mapped a question to the requirement it settles by **exact key
match**: `covers=q.key if q.key in keys else None`. That was true of every
question that existed when it was written, and false from the moment one
question could settle several. A bulk question keyed `repair_bulk::read_as_binary`
matched no requirement, so the four `repair::<id>` decisions it settles all read
as uncovered — and grouping questions read as the door going silent about them,
which is the exact failure `coverage` exists to catch.

**This is the expiring-guarantee shape on a measurement rather than on a
baseline.** True when written, false when a component landed, announced by
nothing. `FEATURE_PARITY.md`'s third defense is *audit every "before X exists"
claim when X ships*; the claim here was never written down as one, which is why
nothing triggered.

**Repaired in the matcher, not in the record.** The Router already publishes
`Question.covers` — the list the interface reads to stop rendering a grouped
finding twice — so the harness was failing to read a fact the app was already
stating. `QuestionRecord.also_covers` carries it and `covered_keys` is what
every coverage computation now reads. Coverage returns to **10/10**.

Not circular: `covers` is checked against `required_decisions`, which is built
from the engine's findings and from nothing the Router said.

## findings_driven 0.75 → 0.625 — half artifact, half real, and it is recorded

The first half was a defect in the feature and is fixed: the bulk question set
no `triggering_finding`, so a question that exists *because of findings* scored
as not findings-driven. It now cites its first member — the one whose worked
example the card shows, so the citation and the evidence on screen are the same
object. That recovered 0.375 → 0.625.

**The remaining 0.75 → 0.625 is real, and it is arithmetic rather than
regression.** The metric is *findings-driven questions ÷ all questions*.
Grouping compresses the numerator and leaves the denominator's constitutional
questions — lens, target, task type, grain — untouched, so a door that asks
strictly less about findings and exactly the same about the constitution scores
lower. Nine of twelve became five of eight.

**No threshold was moved and the prereg is unedited.** The floor is 0.5 and
0.625 clears it. That is the honest outcome: the number fell, it still passes,
and the reason it fell is written down rather than legislated away.

**What it says about the metric, for a later loop.** `findings_driven` measures
*what fraction of the interview is findings-driven*, and it will fall on any
change that makes findings cheaper to answer without making the constitution
cheaper. That is not a defect to fix now — inventing a per-finding
denominator after seeing this result would be fitting the metric to the outcome,
which is what `test_the_prereg_predates_the_router` exists to prevent. It is
recorded as a known property, and if a later loop wants a different
denominator it must pre-register it.

## questions_asked 12 → 8 and irrelevant 2 → 0

Both improvements, both within the pre-registered ceilings, no threshold
touched. `constitutional` stays at 2 and `irrelevant_net` at 0.

## The ruling

**PASSES.** `routing-value-check.json` is re-recorded with `replaced_because`
naming this section, per the precedent set at L19. `routing-baseline.json` is
untouched — it is the frozen Classic measurement and nothing here concerns it.

## Precedent added

**A metric that maps one question to one requirement expires the day a question
can settle several.** The general form: *any measurement that assumes a
cardinality is a temporal guarantee, and it will fail silently in the direction
that looks like a regression.* When a change makes one control do the work of
N, check the harness before believing the number — and prefer a matcher that
reads what the component declares over one that infers from a key.

---

# L25 · `GUIDED-048` — the purpose question, and a ceiling now touched

## What happened

Question 2.5 landed — *what is this model for?* — and the value check moved on
all three datasets:

    messy-clinic  questions_asked   8 → 9    findings_driven 0.625 → 0.5556
    wide-assay    questions_asked   3 → 4    irrelevant       2 → 3
    longitudinal  questions_asked   4 → 5    irrelevant       2 → 3

**PASSES.** Every pre-registered threshold still holds: 9 ≤ 17 and 4/5 ≤ 10 on
questions, 0/3/3 ≤ 4/3/3 on irrelevant, coverage 10/10, and `findings_driven`
0.5556 clears the 0.5 floor. `constitutional` reads 3 on every dataset and
`irrelevant_net` reads 0 on every dataset, which is the L16 ruling working as
designed: a constitutional question is **reported** in the literal count and
**subtracted** in the net one, and neither reading is allowed to become the only
one quoted.

## Two things that are worth writing down rather than noting as green

**`irrelevant_questions` is now exactly at its ceiling on two datasets.** 3 of a
permitted 3 on both `wide-assay` and `longitudinal`. The next constitutional
question breaches the literal reading, and the loop that adds one will meet a
red gate that is **not** measuring a regression — it will be measuring the
denominator's growth, which L16 already ruled on when it refused to add a
`grain::` key to `required_decisions`.

Recorded now, before it happens, because the expensive version of this is
discovering it mid-loop and being tempted to move the ceiling. **The ceiling
does not move.** When it is breached, the honest options are: the new question
is genuinely not constitutional and should not be asked; or the prereg's
`irrelevant_questions` metric has stopped measuring what it was written to
measure on a door with four-plus constitutional questions, and that is a
pre-registration to redo in the open rather than a threshold to nudge.

**`findings_driven`'s margin is thin.** 0.5556 against a 0.5 floor, down from
0.75 two loops ago. The cause is the same arithmetic recorded at L24 and it has
now happened twice: the metric is *findings-driven ÷ all*, grouping compresses
the numerator, and every constitutional question added enlarges the denominator
alone. Two loops, two falls, same mechanism.

That is now enough occurrences to name the property rather than re-explain it
each time: **`findings_driven` falls on any change that makes findings cheaper
to answer or makes the constitution more explicit, and both of those are
improvements.** It is still not a metric to redefine after seeing a result —
inventing a per-finding denominator now would be fitting it to the outcome,
which `test_the_prereg_predates_the_router` exists to prevent. It is a metric
whose known failure direction is written down, so the loop that finally crosses
0.5 can tell which of the two it is.

## The ruling

**PASSES.** `routing-value-check.json` re-recorded with `replaced_because`
naming this section. `routing-baseline.json` untouched. No threshold moved, and
the prereg is unedited.

---

# L60 — the value check found the capability before the sweep did

## What happened

`L60-A` made the positive-class question fire on a **numeric** two-level target,
not only a textual one. `wide_assay.csv`'s target `responder` is `int64` with
values `{0, 1}` — so a fixture no part of the loop touched started raising
`positive_class__responder`, and the frozen result drifted:

| wide-assay · guided | recorded | now |
|---|---|---|
| `questions_asked` | 4 | **5** |
| `findings_driven` | 0.0 | **0.2** |

**`verdict.passes` did not move**, and neither did any threshold. The pre-push
gate caught it before the branch left the machine, which is what that gate is
for — the loop's own sweep was still running and had not reached this file.

## Both movements are the improvement, and they move in opposite directions

**`questions_asked` 4 → 5 is the point of the change, not a cost of it.** The
fifth question is *which level of `responder` is the event*, on a target where
the app previously chose `classes_[1]` without asking. `max_questions` is a
guard against nagging; a question the constitution requires is not nagging, and
the clean-data ceiling is unbreached.

**`findings_driven` 0.0 → 0.2 is the metric rising for once**, and it is worth
noting because the recorded property of this metric is the opposite. The section
above records that `findings_driven` *falls* on any change that makes findings
cheaper to answer or the constitution more explicit — twice, same mechanism.
Here a new finding arrives **with a decision attached**, so it lands in the
numerator rather than only the denominator. **The metric's known failure
direction has a matching success direction, and this is the first observation of
it.**

## Why this is recorded rather than re-recorded quietly

`wide-assay` is a genomics-shaped fixture with 60 rows and 47 columns, chosen for
dimensionality. **Nobody was thinking about it while building L60-A**, and no
part of the loop names it. The value check registered a behavior change in a
corner of the fixture set the work never looked at — which is the whole argument
for a frozen result rather than a recomputed one.

**It also confirms the fix generalizes.** L60-A's own tests drive `case`/`control`
and a numeric 0/1 built for the purpose. This is an independent target, a
different shape of table, and the question fires there too.

## The ruling

**PASSES.** `routing-value-check.json` re-recorded with `replaced_because` naming
this section. `routing-baseline.json` untouched. No threshold moved and the
prereg is unedited. **The drift is the capability arriving, measured by an
instrument that was not aimed at it.**

## The denominator moved again — L60, two more datasets, and this one is NOT ruled yet

**Diagnosed by the adjudicator, deliberately left unruled.** After the value
check above was re-recorded, the pre-push gate surfaced a second and deeper
drift, in the **Classic** baselines:

| dataset | `required_decisions` | `irrelevant_questions` | `coverage` |
|---|---|---|---|
| `wide-assay` | 1 → **2** | 30 → **29** | 1.0 → **0.5** |
| `leaky-sepsis` | 1 → **2** | 30 → **29** | 1.0 → **0.5** |

**Classic's behavior did not change. The denominator did.** `wide_assay.csv`'s
`responder` and `leaky_sepsis.csv`'s `sepsis` are both `int64` with levels
`{0, 1}` *(verified at `1d2206d`)*, so L60-A's dtype-agnostic trigger makes
choosing the event a **required decision** on both. Classic does not ask it —
`target-positive-class` is `guided-only` and the register records that Classic
"encodes a two-level target by whatever sklearn's `LabelEncoder` does with it."
So Classic covers one of two requirements instead of one of one.

**This is the movement §"The denominator moved" already ruled, reaching two more
datasets by the same mechanism.** `ADJUDICATED_DELTAS` in
`test_routing_baseline.py` permits precisely these numbers for `messy-clinic`
and `longitudinal` — `(1, 2)`, `(31, 30)`, `(1.0, 0.5)` — and
`ADJUDICATED_KEY_DELTAS` names the added key as
`repair::positive_class__outcome`. **Same finding, same deltas, two datasets
further.**

### Why it is written here and not resolved here

**The resolution is not a table edit.** `wide-assay` runs through
`ADJUDICATED_DELTAS`; `leaky-sepsis` compares against its own
`routing-baseline-leaky.json` with no deltas table at all. Doing this properly
means a re-measurement recorded **beside** the old one with the chain kept
readable — the `routing-baseline-l9` → `l9c` pattern — and that is a build, not
a five-minute unblock.

**And extending an enumerated allowance in the same loop as the change that
pressured it is exactly what `LOOP.md` §06.2 is about.** The exception may well
apply — the entries would encode *the same purpose* rather than a nudged value,
and the enumeration exists so "a second drift cannot hide inside the first," a
property that survives extension only if every new entry is enumerated exactly.
**But invoking that exception is a deliberate act and it should be taken with a
clear head, in a loop that owns it, not appended to an adjudication at the tail
of a long session to make a push succeed.**

**The branch stays unpushed until it is ruled.** That is the honest state: the
gate is red for a real reason, the reason is understood, and committing over it
is the act that caused the hook to exist.

---

# The denominator moved a third time — and this one is ruled

**Fifth adjudication, `L61`, and it is the resolution the section above deliberately
declined to write.** `TEST-086` was diagnosed at the tail of `L60` and left OPEN with its analysis
complete, on the grounds that *"invoking that exception is a deliberate act and it should be taken
with a clear head, in a loop that owns it."* This loop owns it.

## §06.2 is invoked, in those words, and here is the reasoning

**`LOOP.md` §06 item 2**: *never accept a moved threshold in the same loop as the change that
pressured it. If a gate is measuring the wrong thing, correct which quantity is gated, on a passing
run, with the reasoning recorded before it is load-bearing. After a breach the same correction is
indistinguishable from relaxing a gate under pressure.*

**The exception applies and it is claimed explicitly.** `L53-A2` set the precedent and the test is
whether the entry changes **purpose or value**:

- **No threshold moved.** `VALUE_CHECK_PREREG.md` is unedited. Every pre-registered bound on
  Guided's coverage, question count, irrelevant count and findings-driven ratio is the number it
  was. Nothing was relaxed to let anything pass.
- **No assertion was weakened.** The two comparison tests assert the same equality against the same
  metrics. What changed is *which reading they compare against*, which is the arrangement L9
  established and this extends.
- **The entries encode the same purpose, not a nudged value.** `ADJUDICATED_DELTAS` is not a
  tolerance — it is an **enumeration of ruled movements**. Adding `wide-assay`'s three is recording
  a fourth instance of a cause already ruled twice, not widening a band.
- **And the enumeration's own property is preserved exactly.** *A second drift cannot hide inside
  the first* survives extension only if every new entry is written out. Every one is, below, and
  the inventory keys with them.

**What would have made this a violation**, stated so the line is visible: changing
`_PREREG_METRICS`, editing a threshold in the prereg, comparing against a tolerance instead of an
equality, or deriving the permitted keys from the engine rather than writing them down. None of
those happened.

## What moved, and why only this one dataset

`L60-A` (`DRIVE-032`) made the target's event question **dtype-agnostic**. Before it,
`positive_class_finding` planned through `read_as_binary_plan`, which opens
`if not _is_texty(s): return None` — right for a feature, wrong for the outcome, where *which level
is the event* is exactly as open on a `0`/`1` column as on `case`/`control`.

So the two datasets whose outcome is **text** were already carrying
`repair::positive_class__outcome` from `L9c`. The two whose outcome is **`int64` `{0,1}`** were
not, and now are:

| dataset | outcome | dtype | moved at |
|---|---|---|---|
| messy-clinic | `outcome` | `responder` / `non-responder` | L9, recomposed at L9c |
| longitudinal | `outcome` | `improved` / `stable` | L9, recomposed at L9c |
| **wide-assay** | **`responder`** | **`int64` {0,1}** | **L61** |
| **leaky-sepsis** | **`sepsis`** | **`int64` {0,1}** | **L61** |

**Classic's behavior did not change.** `target-positive-class` is `guided-only` in the register,
with the reason recorded there: Classic *"encodes a two-level target by whatever sklearn's
LabelEncoder does with it"*, which orders classes alphabetically. It does not ask, it never asked,
and it cannot learn to — `pages/01` renders `import_doctor.diagnose` and `ml/import_doctor.py` is
frozen engine-move-only. **What moved is the measuring stick.**

## Every new entry, enumerated

`ADJUDICATED_DELTAS` — three added, all `wide-assay`:

| entry | frozen | adjudicated |
|---|---:|---:|
| `("wide-assay", "required_decisions")` | 1 | **2** |
| `("wide-assay", "irrelevant_questions")` | 30 | **29** |
| `("wide-assay", "coverage")` | 1.0 | **0.5** |

`ADJUDICATED_KEY_DELTAS` — one added key:

| dataset | added | removed |
|---|---|---|
| `wide-assay` | `repair::positive_class__responder` | — |

`LEAKY_DELTAS` and `LEAKY_KEY_DELTAS` — **new tables, and the reason they are new is below**:

| entry | frozen | adjudicated |
|---|---:|---:|
| `("leaky-sepsis", "required_decisions")` | 1 | **2** |
| `("leaky-sepsis", "irrelevant_questions")` | 30 | **29** |
| `("leaky-sepsis", "coverage")` | 1.0 | **0.5** |

| dataset | added | removed |
|---|---|---|
| `leaky-sepsis` | `repair::positive_class__sepsis` | — |

**The key is the TARGET COLUMN's name and it is not `__outcome` on either.** `TEST-086`'s own note
quotes the precedent's key, `repair::positive_class__outcome`, and that is right for messy-clinic
and longitudinal — both of whose target columns happen to be called `outcome`. wide-assay's is
`responder` and leaky-sepsis's is `sepsis`. Written out per dataset rather than derived from the
target, because a table that computed the key would agree with the engine by construction and could
not notice the subject changing — which is the exact hole `ADJUDICATED_KEY_DELTAS` was created to
close.

## The two mechanisms, and the one that had no guard at all

**This is why `TEST-086` said the fix is not a table edit, and it was right.**

`wide-assay` runs through the three-dataset machinery: a frozen file, an adjudicated reference
beside it, and an enumerated allowance between them. Extending that is table work.

`leaky-sepsis` had **none of it**. `T0-ROUTE-001` gave it its own baseline file on purpose — the
three originals are frozen and every pre-registered threshold is banked against them, so injecting
a leak into one would have invalidated the lot — and it was then compared against that frozen file
**directly**. No adjudicated reference, no deltas table, and therefore no way to absorb a ruled
movement except by editing the frozen artifact or hand-writing a replacement. Both are what
`test_routing_baseline.py` exists to prevent.

Three things close that, and they are the part of this loop worth keeping:

- **`routing-baseline-leaky-l61.json`**, written beside the frozen one, which the comparison now
  reads — the same arrangement the pre-registered three have had since L9.
- **`test_the_leaky_reference_differs_from_the_frozen_one_only_as_ruled`**, which is the
  three-dataset guard's twin: metrics against `LEAKY_DELTAS`, **and inventory keys** against
  `LEAKY_KEY_DELTAS`, because a composition change moves no metric and a size-only check cannot see
  it.
- **`scripts/remeasure_routing_baseline.py --leaky`**, because until `L61` the leaky dataset could
  not be re-measured by the one script allowed to write these files at all. A protected artifact
  with no sanctioned way to supersede it is an artifact that gets edited.

## The chain, and a guard for it

`routing-baseline` → `l9` → `l9c` → **`l61`**, and `routing-baseline-leaky` → **`leaky-l61`**.
Every superseded reading stays on disk with its `measured_at` stamp.
`test_the_chain_of_re_measurements_is_readable_rather_than_implied` asserts each file exists,
carries measurements, names the commit it was taken at, and that no two readings **within a chain**
claim the same commit — a re-measurement reusing its predecessor's stamp is indistinguishable from
an edit of the predecessor.

**That test's first draft was wrong and it caught itself.** It checked stamp uniqueness across both
chains at once and failed on `cd1311e` appearing twice — which is not a collision: the two chains
are separate artifacts and `L61` re-measured both at the same commit. Corrected to per-chain, with
the reason in the code.

## Both readings

Guided is unaffected on either dataset — it raises the question, so its numerator rises with its
denominator. Classic's numerator is structurally frozen (§"Coverage carries its denominator"), so
its coverage falls.

| dataset | | frozen denominator | adjudicated denominator |
|---|---|---:|---:|
| wide-assay | Classic covered | 1 | 1 |
| wide-assay | **Classic coverage** | **1/1 = 1.000** | **1/2 = 0.500** |
| wide-assay | Guided coverage | 1/1 = 1.000 | 2/2 = 1.000 |
| leaky-sepsis | Classic covered | 1 | 1 |
| leaky-sepsis | **Classic coverage** | **1/1 = 1.000** | **1/2 = 0.500** |

**Measured at `cd1311e`**, `venv/bin/python`, 2026-08-13. `verdict.passes` is unchanged and the
pre-registration is unedited.

## The ruling

**The new readings are binding.** Both frozen artifacts are untouched. `TEST-086` closes; the
branch is unblocked.
