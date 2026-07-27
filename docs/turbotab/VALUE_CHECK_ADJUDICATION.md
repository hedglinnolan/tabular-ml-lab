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
