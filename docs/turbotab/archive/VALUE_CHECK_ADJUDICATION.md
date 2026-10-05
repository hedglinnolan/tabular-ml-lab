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

---

# L66 · Classic moved, and this time it was not the measuring stick

**Sixth adjudication.** Same procedure: the frozen artifacts are not edited, both readings are
preserved in data, the ruling is published here. What is different about this one is worth
saying in the first sentence, because every previous movement on this file was a *denominator*
movement — the engine learned to find a decision it had been missing, and the thing measured
stood still. **This time the thing measured moved.** Classic's UI changed and its cost metrics
fell.

## What happened

`5acd7cd` merged `main` into `TurboTab` — sixteen commits of Classic fixes. Three of them are
visible to this baseline, because the measured window is Classic's exploration path, pages 01
and 02, and all three change what widgets those pages render:

| commit | what it did | effect on the count |
|---|---|---|
| `7480564` | *"Keep the five diagnostics nothing else computes, and drop the eleven that repeat"* | −9 buttons, 2 re-keyed |
| `f6ce4ae` / `e6187f5` | added the cluster-structure explorer to EDA | +2 (`eda_km_feats`, `eda_km_run`) |
| `6d9e49e` | *"Say what the working table is"* — a confirmation checkbox on page 01 | +1 (`wt_confirm_box`) |

`7480564`'s reasoning is on the record and is not in dispute: the Deep Dive tab strip was
written against an earlier EDA page, and sections 1–5 had since absorbed the target histogram,
the class bar, the missingness bar, the correlation matrix, the outlier heatmap, the
feature-versus-target gallery and the interaction detector. Eleven of fifteen buttons were
re-rendering what the page had already shown. What survives is the classical layer nothing else
implements — NHANES reference-range checking, Q-Q plus Shapiro-Wilk on residuals, variance
inflation, leverage and Cook's distance.

## Every difference, enumerated

Not "six fewer questions". The net is −6 on three datasets and **−5 on longitudinal**, and it is
a net: keys leave, keys arrive, and two keys only change name. Written out per dataset so a
second drift cannot hide inside this one.

**Gone on all four datasets** (8 dropped deep-dive buttons):
`run_advanced_interaction_analysis`, `run_advanced_outlier_influence`,
`run_advanced_quick_probe_baselines`, `run_advanced_target_profile`,
`run_quality_data_sufficiency_check`, `run_quality_feature_scaling_check`,
`run_quality_leakage_scan`, `run_readiness_linearity_scatter`.

**Re-keyed, not removed** — the tab strip's key prefix went with the tab strip, so these two
appear once in each column and net to zero: `run_quality_plausibility_check` →
`run_plausibility_check` (messy-clinic and longitudinal only, the two datasets with a
biomedical column match), `run_readiness_multicollinearity_vif` → `run_multicollinearity_vif`
(all four).

**Gone on three of four** — the recommendation panel's own duplicate of a deep-dive button,
whose label was the bare word `"Run"`: `rec_run_data_sufficiency_check` on messy-clinic and
wide-assay, `rec_run_leakage_scan` on leaky-sepsis. Longitudinal raised no recommendation card,
which is the whole −5-versus-−6 difference.

**Arrived on all four:** `wt_confirm_box`, `eda_km_feats`, `eda_km_run`.

| dataset | keys gone | keys added | of which re-keys | net |
|---|---:|---:|---:|---:|
| messy-clinic | 11 | 5 | 2 | **−6** |
| wide-assay | 10 | 4 | 1 | **−6** |
| longitudinal | 10 | 5 | 2 | **−5** |
| leaky-sepsis | 10 | 4 | 1 | **−6** |

## Both readings

| dataset | metric | `l61` reference | `l66` reference |
|---|---|---:|---:|
| messy-clinic | questions_asked | 34 | **28** |
| messy-clinic | irrelevant_questions | 24 | **18** |
| wide-assay | questions_asked | 31 | **25** |
| wide-assay | irrelevant_questions | 29 | **23** |
| longitudinal | questions_asked | 32 | **27** |
| longitudinal | irrelevant_questions | 30 | **25** |
| leaky-sepsis | questions_asked | 31 | **25** |
| leaky-sepsis | irrelevant_questions | 29 | **23** |

**Nothing else moved, on any dataset.** `required_decisions`, `covered`, `coverage`,
`coverage_ratio`, `findings_driven`, `constitutional`, `irrelevant_net` and `pull_affordances`
are identical between `l61` and `l66`, and the required-decision *inventories* are identical key
for key. The engine was not touched by the merge; only Classic's page 01 and 02 chrome was. That
is the cleanest attribution any movement on this file has had.

**Measured at `f507ce2`**, `venv/bin/python`, 2026-08-23, via
`scripts/remeasure_routing_baseline.py` (with and without `--leaky`). Written to
`routing-baseline-l66.json` and `routing-baseline-leaky-l66.json`; `l61`, `l9c`, `l9` and both
frozen files are unedited and still on disk.

## What this costs the builder, stated plainly

This movement makes Classic **look better**, and it is the first one on this file that does.
Every previous adjudication could point at a denominator change that cut against the builder;
this one cannot, and pretending otherwise would be the exact failure the note exists to prevent.

- The prereg's descriptive headline — *"Classic asks a near-constant ~32 questions regardless of
  the dataset"* — now reads **~26**. The constancy claim survives (28/25/27/25); the number does
  not.
- `VALUE_CHECK_PREREG.md` justifies Guided's messy-clinic ceiling as *"≤ 17 — at most half of
  Classic"*. Half of 34 is 17. Half of 28 is 14. **The ceiling no longer enforces the sentence
  that justified it** — 17 is 0.61× Classic now.
- The claim itself is nonetheless still true in fact, which is the only reason this is a note and
  not a `BLOCKED.md`: Guided's measured `questions_asked` is **9** on messy-clinic (≤ 14) and
  **5** on each guard (≤ 12), so Guided still asks under half of what Classic asks under the new
  reading. The margin narrowed; the direction did not reverse.
- **No pre-registered threshold is touched, and none is missed.** Every threshold binds on
  *Guided*, and Guided did not move. `docs/turbotab/data/routing-value-check.json` is keyed to
  `routing-baseline-l9c.json` and reads neither `l61` nor `l66`, so `verdict.passes` is unchanged
  — and the Classic figures quoted inside it are now stale relative to `HEAD`, which is recorded
  here rather than quietly refreshed.

## The premise that expired, which is the part worth keeping

`TRANSITION_PLAN.md` §05 freezes Classic as engine-move-only, and every baseline on this file has
been read as *Classic does not change, so a moved number means the measuring stick moved*. **That
premise is now false.** Classic is `main`'s product and `main` shipped UI changes to it; the
freeze was a TurboTab-branch convention that a merge does not honor and does not announce.

This is the same shape as the expiry this file's own test docstring already warns about — *a
protection that depends on "X does not exist yet" expires the moment X exists, and nothing will
tell you*. The variant here: **a baseline that depends on "the other branch will not touch this
code" expires at the first merge.** What made it survivable is that the drift detector is a
comparison and not a re-measurement, so the merge produced a failing test rather than a silently
rewritten reference.

## The ruling

**The `l66` readings are binding.** The frozen artifacts and `l61` are untouched; the
pre-registration is unedited; `main`'s dedup is **accepted**, not reverted — it is a UI decision
about buttons that duplicated what the page already renders, made on the mainline product, and
the baseline's job is to record Classic as it is rather than to hold it still.

Two things are ruled *not* settled by this entry, and both belong to a later loop rather than to
this file:

1. **The prereg's "at most half of Classic" wording.** The ceiling stays at 17. Re-deriving it
   from the new Classic would be fitting a threshold to a measurement, which is the one thing
   `VALUE_CHECK_PREREG.md` forbids in its first paragraph. It is recorded as loose, not fixed.
2. **Whether Classic's freeze is still a premise the value check may rest on.** It is not, at
   `HEAD`, and every future merge from `main` will move this baseline again.

## Precedent added

- A baseline that assumes another branch will not touch the measured code must say so, and the
  assumption expires at the first merge. Record the expiry where the assumption is stated, not
  where it breaks.
- A movement that flatters the builder gets **more** enumeration than one that does not, not
  less. This entry lists every key that left, arrived, or was merely renamed, because the
  temptation to summarize is strongest when the summary is favorable.

---

# L67 · Classic asks the grain question, and the harness has nowhere to put it

**Seventh adjudication.** The movement is the smallest and cleanest this file has recorded — one
key, the same key, on all four datasets, nothing removed — and it lands on the one question this
document has already ruled on twice, from the other door. `GUIDED-018` asked whether a
constitutional question may be counted as noise. It was ruled at `L16`, for Guided. **Classic has
now asked one**, and the answer L16 gave depended on a fact about Classic that is no longer true.

## What happened

A fix wave added a subject-declaration control to `pages/01_Upload_and_Audit.py:1310-1327` — an
expander, *"👤 Which column identifies a subject/participant?"*, wrapping a selectbox keyed
`subject_id_declaration`. It closes `IMPORT-022` / `IMPORT-257`: measured repetition under a column
name the app does not recognize now seals as `undetermined` rather than as one-row-per-person, and
the user can declare the subject column instead of the app guessing wrong in silence.

That is constitution **§02, the grain question** — the same clause, asked for the same reason, as
Guided's `state_grain` (`ml/router.py:648`, `clause="lockbox-02"`).

## Every difference, enumerated

**Added on all four datasets:** `subject_id_declaration` — *"Which column identifies a
subject/participant?"*

**Removed: nothing. Re-keyed: nothing.** Every other key in `l66` reproduced exactly.

| dataset | `questions_asked` | `irrelevant_questions` | `constitutional` | `irrelevant_net` |
|---|---|---|---:|---|
| messy-clinic | 28 → **29** | 18 → **19** | 0 | 18 → **19** |
| wide-assay | 25 → **26** | 23 → **24** | 0 | 23 → **24** |
| longitudinal | 27 → **28** | 25 → **26** | 0 | 25 → **26** |
| leaky-sepsis | 25 → **26** | 23 → **24** | 0 | 23 → **24** |

`required_decisions`, `covered`, `coverage`, `coverage_ratio`, `findings_driven` and
`pull_affordances` are unchanged on every dataset, and the required-decision inventories are
identical key for key. **+1 asked, +1 irrelevant, exactly, four times.** A question in, a question
counted as noise — which is `L16`'s opening sentence with the door swapped.

## The accounting question, ruled

The `+1` on `irrelevant_questions` is arithmetic, not judgment: `irrelevant = asked −
required_decisions`, there is no grain key in the inventory, so a question covering nothing in it
is surplus by construction. `L16` settled that the literal count is correct and stands. What is
open is whether Classic's grain question should also be counted in **`constitutional`**, so that
`irrelevant_net` reads 18/23/25/23 rather than 19/24/26/24.

**The harness can technically do it.** `QuestionRecord.clause` is a real field
(`turbotab/measure.py:129`) and `Measurement.constitutional` (`:229-260`) counts any record that
names a clause, cites no finding and settles no inventory key. All three conjuncts hold for
`subject_id_declaration`. The Classic harness simply never sets `clause`, so every Classic widget
is `clause=None` and Classic's `constitutional` is 0 by omission.

**Ruled: not attributed in this loop. Classic's `constitutional` stays 0, and the omission is
recorded as an accounting limitation rather than corrected quietly.** Four grounds, in order of
weight:

1. **The harness would be classifying its own subject.** `L16`'s mechanism rule is explicit:
   *"`Question.clause` on the Router carries the label, and the harness reads it rather than
   deciding for itself which questions are constitutional — a harness that classified its own
   homework."* Guided's question declares `clause="lockbox-02"` in `ml/router.py` and the harness
   reads it. Classic's selectbox declares nothing; attributing a clause to it means a hand-written
   `widget key → clause` constant living in the test file, written by the party the number
   describes. The existing `_map_to_requirement` is not a precedent for this — it maps a Classic
   button to an inventory key by matching the **engine-written `fix_label`**, so ground truth
   external to the harness does the deciding.
2. **It changes nothing that is failing.** `constitutional` and `irrelevant_net` are not in
   `_PREREG_METRICS`, so attribution would not move a single guarded number and would not make the
   drift test pass. The `+1/+1` has to be adjudicated and banked either way. Bundling an unforced
   accounting change into the loop pressured by an unrelated drift is `LOOP.md` §06.2, and unlike
   `L61` there is no same-purpose argument to invoke it on.
3. **`L16` left this question open by name and it is not this loop's to close.** *"Whether the
   Classic column has constitutional questions too"* is listed verbatim under that entry's "What
   this does not settle". It is filed, not forgotten.
4. **The direction is the unusual one and deserves the same suspicion, not less.** Attribution
   makes **Classic** look better — `irrelevant_net` 19 → 18 on messy-clinic. The standing rule here
   is that the reading which flatters a door is not banked by the agent arguing for it, and that
   rule does not acquire an exception when the flattered door is the comparator rather than the
   builder's own.

## The `L16` caveat is now wrong, and this is the correction

`L16` wrote:

> **Whether the Classic column has constitutional questions too.** It is measured from a frozen
> baseline and cannot gain the label, so its `constitutional` is structurally 0 rather than
> measured 0.

**Both halves of that sentence have expired.** `L66` recorded that Classic is no longer frozen —
`main` ships UI changes to it and a merge does not announce them. And Classic has now gained a
constitutional question in fact. So Classic's `constitutional = 0` is no longer *structurally*
zero and no longer *conservatively* zero: it is **measured wrong, by exactly one question per
dataset, in a known direction.**

The caveat that must accompany any cross-door net-reading comparison is therefore strengthened, not
repeated:

> Guided's `constitutional` is read from self-declared clause labels; Classic's is 0 because the
> harness never asks. As of `L67` that 0 is known to be short by one — Classic asks the §02 grain
> question at `pages/01_Upload_and_Audit.py:1310`. Any comparison of the two doors'
> `irrelevant_net` flatters Guided by one question per dataset, and quoting one without this
> sentence is quoting a number the project knows to be wrong.

## Both readings

Neither reading moves a verdict, and it is worth being precise about why: **every pre-registered
threshold binds on Guided, and Guided did not move.** Classic is the comparator; nothing is banked
against its cost metrics.

| | `l66` reference | `l67` reference (binding) | `l67` under attribution (not banked) |
|---|---|---|---|
| Classic `irrelevant_questions`, messy-clinic | 18 | **19** | 19 |
| Classic `constitutional`, messy-clinic | 0 | **0** | 1 |
| Classic `irrelevant_net`, messy-clinic | 18 | **19** | 18 |
| Guided, all metrics, all datasets | — | **unchanged** | unchanged |
| `verdict.passes` | — | **unchanged** | unchanged |

The third column is written down so that the reading this entry declines to bank is on the record
rather than merely declined — the same discipline `routing-value-check.json`'s
`passes_under_literal_reading` keeps.

## Provenance: the stamp says `+wt` and here is why

**The grain control is not committed.** `git show 30f48e4:pages/01_Upload_and_Audit.py` contains no
`subject_id_declaration`; the control exists only in the working tree, alongside other uncommitted
paper-risk fixes from parallel agents. Stamping this measurement `30f48e4` would claim that
re-measuring at that commit reproduces these numbers, and it does not — it reproduces `l66`.

So both files are stamped **`30f48e4+wt`**, and the suffix means *this commit plus an uncommitted
working tree*. It is deliberately not a valid revision: a stamp that cannot be resolved by `git
show` is a stamp that announces its own limitation instead of quietly failing to honor it.

**One correction owed to `L66`, which is not made by editing it.** `routing-baseline-l66.json` is
stamped `f507ce2` with no suffix, and that tree was also dirty. The measurement rule — *the
envelope may gain labels, never lose or alter one* — forbids amending it, so the correction is
recorded here. What makes it a small error rather than a live one: this re-measurement reproduced
every `l66` key on every dataset exactly, adding one and removing none, so whatever else was
uncommitted at `f507ce2` did not move Classic's count.

**Measured** 2026-08-23, `venv/bin/python`, via `scripts/remeasure_routing_baseline.py` with and
without `--leaky`. `l66`, `l61`, `l9c`, `l9` and both frozen files are unedited and on disk.

## The ruling

**The `l67` readings are binding.** The fix is **accepted** — a control that lets a user declare
the subject column, on a page that previously guessed from column names and sealed a wrong guess
silently, is `IMPORT-022` closing, and it is worth one point of measured noise on a metric that
binds nothing. Frozen artifacts untouched, pre-registration unedited, no threshold approached.

Not settled by this entry:

1. **Whether Classic's harness should read clause labels at all**, and if so from what — a
   declaration on the widget, a registry, or something the constitution documents own. This is
   `GUIDED-018`'s Classic half and it is now live rather than hypothetical. It needs a mechanism
   that is not the harness marking its own homework, and that is a design question, not a loop's
   incidental fix.
2. **Whether §02 arriving in Classic changes what the two doors are being compared on.** Classic
   now asks a constitutional question that Guided also asks. That is the first question in the
   measured window the two doors have in common by clause rather than by coincidence, and this
   file has no rule for what that means.

## Precedent added

- **A stamp is a claim about reproducibility.** If the tree the measurement was taken from is not
  the tree a reader can check out, the stamp says so in the stamp — not in a comment beside it.
- **An accounting change that fixes no failing assertion is not part of the loop that revealed the
  need for it.** Bank the drift; file the accounting; do not let the second ride in on the first.

---

# L68 · the repair branch that never fired

**Eighth adjudication, and the most expensive one to the builder's headline that this file has
recorded.** The drift that fired the gate is ordinary — Classic renders more repair buttons, so it
asks more questions. What the investigation found underneath it is not: **the half of the harness
that scores Classic's coverage has never matched a single widget, on any dataset, in any of the six
published readings, the frozen pre-registration included.** Classic's coverage has always been "the
target selectbox, and nothing else." Fixed, it goes from 1/10 to **5/10** on the contested dataset.

## What happened

`DRIVE-067`: the Import Doctor's structural review now also runs on the working table in Step 3 of
`pages/01_Upload_and_Audit.py`. Previously only fresh uploads were reviewed — registry, restored and
combined datasets got none. Where the fixtures have findings, repair buttons now render.

**Added, all under `worktable_*` keys, nothing removed and nothing re-keyed:**

| dataset | widget key | label | covers |
|---|---|---|---|
| messy-clinic | `worktable_1_fix_3_category_variants__sex` | Merge those variants in 'sex' | `repair::category_variants__sex` |
| messy-clinic | `worktable_1_fix_4_numeric_as_text__income` | Convert 'income' to numbers | `repair::numeric_as_text__income` |
| messy-clinic | `worktable_1_fix_5_numeric_as_text__weight` | Convert 'weight' to numbers | `repair::numeric_as_text__weight` |
| messy-clinic | `worktable_1_fix_7_unnamed_columns` | Drop 1 unnamed column(s) | `repair::unnamed_columns` |
| messy-clinic | `worktable_1_fix_8_constant_columns` | Drop 1 constant column(s) | **nothing** |
| messy-clinic | `worktable_1_fix_9_wide_repeated_measures` | Reshape to one row per measurement (long format) | **nothing** |
| wide-assay | `worktable_1_fix_0_wide_repeated_measures` | Reshape to one row per measurement (long format) | **nothing** |

longitudinal and leaky-sepsis have no findings on this surface and did not move at all — the first
time in five adjudications that two datasets are byte-identical to their predecessor.

The last three rows matter as much as the first four: `constant_columns` and `wide_repeated_measures`
are buttons Classic renders for findings the engine does **not** put in `required_decisions`. They
score as covering nothing, which is correct, and they are the evidence that the mapping fix below
is not a blanket "count the new buttons".

## The finding underneath the drift

`_map_to_requirement` has two branches. One maps `target_selectbox → choose_target`. The other maps
a button labeled `"Apply: <fix>"` to the repair whose engine-written `fix_label` it shows.

**The second branch has never returned a value.** Every `covers` recorded for Classic in
`routing-baseline.json`, `l9`, `l9c`, `l61`, `l66` and `l67` is `target_selectbox`. Six readings,
four datasets, one match each.

**That was not a bug and not a lie when it was written.** The "Apply:" surface is Suggested Actions
on a *fresh upload*; the harness seeds a registry dataset via `_seed_dataset_roster`, and before
`DRIVE-067` a registry dataset got no structural review — so there were no repair buttons on the
measured path to match. The branch was **unexercised, not wrong**, and the pre-registered `0.111`
was an accurate measurement of a door that genuinely surfaced one required decision.

`DRIVE-067` ends that. The buttons now render, carrying the bare `fix_label` with no `"Apply:"`
prefix — so the branch still does not fire, and the artifact would publish **`coverage: 1/10` for a
door that now puts five of the ten required decisions to the user.** That is not a conservative
number or a stale one. It is false, and it is false in the direction that flatters the door this
project built.

## The ruling on the mapper: fixed, and the corrected reading is banked

**Ruled: `_map_to_requirement` learns the second surface. Classic's messy-clinic coverage is
`5/10 @ 2da5528`.**

This is the opposite of the `L67` ruling three entries up, and the difference is the whole point,
so it is spelled out rather than asserted:

| | `L67` — the clause question | `L68` — the fix_label question |
|---|---|---|
| who would decide what the widget means | the harness, via a hand-written `key → clause` constant | the **engine**, which wrote both the `fix_label` and the finding key |
| is there ground truth to read | no — Classic declares no clause anywhere | yes — two independent engine-written strings |
| does it move a `_PREREG_METRICS` number | no (`constitutional` is not one) | **yes — `coverage`, the headline** |
| direction | flatters Classic | flatters Classic |
| ruled | declined, recorded as a limitation | **fixed** |

Grounds:

1. **The harness is not classifying its own subject; it is checking that two of the engine's own
   strings agree.** `L16`'s boundary is that the harness must read a label rather than invent one.
   The label here is `r.fix_label`, written by the engine, and the widget key independently embeds
   the engine's finding key (`worktable_1_fix_4_**numeric_as_text__income**`). This is the
   `_map_to_requirement` precedent operating exactly as designed, on a surface that finally exists.
2. **The rule adopted is stricter than the one it extends, not looser.** Surface 1 accepts a
   substring match on one channel. Surface 2 requires the exact `fix_label` **and** the key suffix
   to point at the same requirement, and returns nothing on disagreement or ambiguity. A harness
   marking its own homework would have loosened the test, not tightened it.
3. **Leaving it publishes a false headline.** The prereg's contested claim is a coverage
   comparison. Publishing 1/10 while the door surfaces 5/10 is the single most consequential false
   statement available to this project, and "it fixes no failing assertion" is not a reason to
   leave a known-false headline standing — it is a reason to record both readings and rule, which
   is what this entry does.
4. **The direction is decisive, not incidental.** This correction *narrows the gap the project
   exists to demonstrate.* A discipline whose standing rule is "the reading that flatters the door
   I built is not mine to bank" cannot invoke process tidiness to keep a flattering number it has
   discovered to be wrong. Declining here would be the self-serving option, which is precisely why
   `L67` — where declining was the unflattering option — went the other way.

**Blast radius, bounded deliberately.** Surface 1's branch is byte-identical and returns before
surface 2 is reached, and surface 2 only fires on `worktable_*` keys, which did not exist before
this fix. No number in any superseded reading could have moved. longitudinal and leaky-sepsis
prove it: unchanged, key for key.

## Both readings

| dataset | metric | `l67` | `l68` under the OLD mapper | `l68` binding |
|---|---|---:|---:|---:|
| messy-clinic | questions_asked | 29 | 35 | **35** |
| messy-clinic | irrelevant_questions | 19 | 25 | **25** |
| messy-clinic | covered | 1 | 1 | **5** |
| messy-clinic | **coverage** | 1/10 | 1/10 | **5/10** |
| wide-assay | questions_asked | 26 | 27 | **27** |
| wide-assay | irrelevant_questions | 24 | 25 | **25** |
| wide-assay | coverage | 1/2 | 1/2 | **1/2** |
| longitudinal | all | — | unchanged | **unchanged** |
| leaky-sepsis | all | — | unchanged | **unchanged** |

The middle column is the reading this entry declines — the drift banked without the mapper fix —
written down so that what was rejected is on the record rather than merely rejected.

**Two coincidences worth naming before someone reads them as signal:**

- **messy-clinic's `irrelevant_questions` is back at 25, exactly the frozen pre-registered value**,
  after going 25 → 24 → 18 → 19 → 25 through four unrelated causes. A reader comparing only the
  frozen file to `l68` would conclude the metric never moved. It moved four times.
- **messy-clinic and wide-assay both read `coverage: 0.5`, and they are not the same number** —
  5/10 and 1/2. This is the "Coverage carries its denominator" rule earning its keep: the decimal
  is identical, the `coverage_ratio` field is `5/10 @ 2da5528` and `1/2 @ 2da5528`, and only the
  second form is quotable.

## What this costs the headline

**No pre-registered threshold moves, and none is missed.** Every threshold binds on Guided; Guided
did not move; `verdict.passes` is unchanged. The prereg's Classic column stays pinned at `0.111`
in perpetuity under rule 2 of "Coverage carries its denominator", and the pre-registration is
unedited.

What changes is what may honestly be *said*:

- The narration "Classic surfaces one of ten required decisions; Guided surfaces ten of ten" is
  **retired**. At `HEAD` it is five of ten against ten of ten.
- The `L16` caveat — *"Classic's coverage numerator is structurally frozen … a widening gap is
  therefore not by itself evidence of better routing"* — is now **doubly** expired. `L66` recorded
  that Classic is no longer frozen. This entry records the second half: part of the gap was never
  Classic's numerator being frozen at all, it was **the harness never reading it.** The caveat is
  revisited as `L16` said it would be, not deleted:

> Classic's coverage was measured at 1/n for six readings because the harness's repair branch had
> no surface to match, not because Classic surfaced one decision in principle. From `L68` the
> numerator is read from engine-written fix_labels and reads 5/10 on messy-clinic. Any coverage gap
> quoted from a reading before `l68` is quoting a harness limitation as if it were a door's
> behavior.

## The limitation this entry does NOT fix

`findings_driven` stays `0.0` for Classic, and it is now wrong for the same six widgets. The
harness sets `triggering_finding=None` on every Classic question with a stated reason:
*"the questions around it exist whether or not anything was found."* That reason is **false** for
the working-table repair buttons — `worktable_1_fix_3_category_variants__sex` does not render
unless the `category_variants` finding fired on `sex`.

Not fixed here, and the distinction from the mapper is the ground for it: the mapper had an
**existing branch encoding the intended semantics**, and this loop restored it to a surface it was
written for. `triggering_finding=None` is a **stated editorial judgment about a whole class of
widgets**, and overturning one is a new ruling, not a repair. `findings_driven` binds nothing on
Classic (the ≥ 0.50 floor is Guided's), so the cost of carrying it one more loop is a descriptive
understatement, recorded here. Filed for the next loop.

**Measured at `2da5528`**, tree verified clean with `git status --porcelain`, `venv/bin/python`,
2026-08-24, via `scripts/remeasure_routing_baseline.py` with and without `--leaky`. The stamp is
resolvable — `L67`'s `+wt` precedent applies and is not needed. `l67`, `l66`, `l61`, `l9c`, `l9` and
both frozen files are unedited and on disk.

## The ruling

**The `l68` readings are binding, including the corrected coverage.** `DRIVE-067` is **accepted** —
a structural review that ran only on fresh uploads and silently skipped every registry, restored
and combined dataset was the defect; running it on the working table is the fix, and six more
questions on the messiest fixture is what surfacing real findings costs.

Not settled:

1. **`triggering_finding` for finding-conditional Classic widgets**, above.
2. **Whether any narrative artifact still quotes the 1/10 comparison.** This entry retires the
   claim; it does not go and find every place the claim was written down. `RETROSPECTIVE.md`,
   `ROADMAP.md` and the paper drafts are unaudited for it.

## Precedent added

- **An unexercised branch is not a passing branch.** Six readings agreed on Classic's coverage and
  all six were produced by a mapper half that never once matched. Agreement across re-measurements
  is not evidence that the measurement is reading anything — a metric with a constant value should
  be asked whether it CAN vary.
- **Who wrote the string decides whether reading it is classification.** The harness may match on
  anything the engine wrote, and may not invent a mapping of its own. `L67` and `L68` sit on
  opposite sides of that line and are the worked examples.
- **A correction that narrows the gap you are trying to demonstrate gets banked on discovery.**
  Not deferred to a tidier loop. The tidier loop is the one where the flattering number survives
  another release.
