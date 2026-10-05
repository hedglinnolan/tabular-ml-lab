# Retrospective pack — the measured state of the sprint, 2026-08-22

**This file exists because the adjudicator who held these facts is about to be cleared, and this
project has paid five separate times for a ruling that lived only in a conversation.** It is
deliberately **evidence and open questions, not conclusions.** The retrospective is where the
conclusions get made, and the product owner makes them.

Every number here carries how it was derived. Where a number is *from a row* rather than
*re-derived*, it says so — and per this project's standing rule, **doubt the second kind first.**

---

## 01 · What the sprint was

**L58 through L65, plus human drives 5, 6 and 7.** Eight loops, roughly 2026-08-09 to 2026-08-22.
All eight accepted. `LOOP.md` §03 carries a row for each and that log is the narrative record; this
file does not repeat it.

**Three things changed about the *method* during this sprint, and none of them was planned:**

1. **The fan-out became the way loops are adjudicated** (L62 onward), then the way prompts are
   *built* (L63 onward). Nobody decided this; it worked once and kept being used.
2. **The parallel test sweep was licensed** (L63), taking a full run from `2:01:30` to about `41:49`.
3. **The execution agent and the adjudicator both began being cleared between loops**, so both
   onboarding documents became load-bearing in a way they were not designed for.

---

## 02 · The ledger is diverging, and this is the sprint's central fact

Re-derived at `3f14f64` by walking `findings.json` at fourteen commits:

| date | total | closed | open + partial |
|---|---:|---:|---:|
| 2026-08-09 | 883 | 366 | **517** |
| 2026-08-12 | 928 | 387 | 541 |
| 2026-08-16 | 968 | 415 | 553 |
| 2026-08-21 | 992 | 434 | 558 |
| 2026-08-22 | **1,004** | **440** | **564** |

**121 findings filed, 74 closed, net +47 still open.** The work is real and the closures are real —
every `FIXED` row carries a named test and the §08.1 probe discipline is enforced. **But the open
count has risen every single week of the sprint.**

`ROADMAP.md`'s definition of done, condition 5, is *"The ledger is closed. Every finding is `FIXED`,
`NOT-A-DEFECT`, or `WONTFIX`. Zero `UNVERIFIED`, zero `OPEN`."*

**At the sprint's observed rate that condition is not reachable.** That is not an argument that
anything is being done wrong — a discovery phase *should* file faster than it closes. It is an
argument that **a stated condition of doneness has quietly stopped describing the plan**, and only
the product owner can say which of the two should move.

**The open question, stated plainly and not answered here:** is the rising count evidence that the
instruments are getting better at finding real defects, or evidence that the app's defect surface is
growing faster than the loops? **Both are consistent with this table.** Nothing measured this sprint
distinguishes them.

---

## 03 · What the method costs, measured

**L65 alone consumed roughly 2.63 million subagent tokens across 30 agents** — a nine-agent
reconnaissance (892,764), a three-agent refutation (326,417), and a fourteen-agent adjudication
(1,415,527). That is one loop of four parts.

**What it bought, and this is the honest case for it:**

- The reconnaissance **destroyed two of the adjudicator's own premises before the prompt shipped**,
  including a prescribed mechanism that does not transfer (`MISC-029`, correction 3).
- The refutation **caught two errors inside the adjudication itself** — a count measured one write
  late, and a probe that zeroed the population it was quantifying over instead of falsifying a value
  (`PM_TRANSITION.md` §06.6). Both would otherwise have entered the record as rulings against a
  report that was right.
- **Six fan-outs, and no driver has ever come back clean.** Two refuters have returned `SOUND` in
  that entire history.

**What it has not been asked:** whether a cheaper configuration buys most of the same value. Nobody
has run a loop *without* a fan-out since L61 and compared. **There is no control arm.**

---

## 04 · The three error shapes that keep recurring after being written down

This is the part a retrospective is actually for. Each of these is recorded, each has a rule
attached, and each recurred anyway.

**A · A correct measurement's authority carrying into a wrong conclusion drawn from it.
Three occurrences, all the adjudicator's.**

- L64: a measurement of *why two candidate fixes fail* became a prescription for a third mechanism
  without anyone asking whether the failure was intrinsic. It was. The agent built it, verified it,
  deleted it, and was right.
- L65 (prompt): a reconnaissance reported *"`turbotab/.venv` has pandas, pytest AND fastapi (sklearn
  absent)"*. The adjudicator read the fastapi half, concluded the hole was unreachable, and
  instructed the agent to **state that as fact**. Both halves were wrong and the agent said so.
- L65 (adjudication): an agent charged the report with miscounting citations, having measured **after
  the loop's own repair rewrote the citation it was counting** — the same one-write-late error it had
  correctly charged against the report one paragraph earlier.

**The rule already exists** (`PM_TRANSITION.md` §06.1) and did not prevent occurrences 2 and 3.
**Open question: why does a written rule fail to fire here, and what would fire instead?**

**B · A guard that measures a suspected *mechanism* rather than the *consequence*.**

Drive 7's `DRIVE-054` is the sharp instance. The app's scroll guards are **green and correct** —
`scrollIntoView` appears exactly once and only in the rail's navigation branch, and
`scrollTo`/`scrollTop`/`scrollBy` are zero. Meanwhile a human lost roughly ten clicks to the page
moving under the cursor, twice changing the **target**, because the cause is layout reflow from 127
wholesale `innerHTML` repaints. *(Re-derived at `0856c1d`.)*

**Nobody has asked how many other guards in this repository have that shape.** It is a countable
question and it has never been counted.

**C · Trap #1 committed by the *verification* rather than by the build.**

`DRIVE-056`. Two independent agents drove the L65 manuscript header to *"11 checks, 0 unmet · 2
declared"* through the page harness, and the adjudicator accepted it as the number the part owed.
**A human then fit six models across three datasets and the panel never moved.** The harness had been
handed a fitted project directly; the interface never delivers that state.

The number was real. The path was not. **The existing rule says "drive it"; this sprint established
that is insufficient — it has to be driven along the path a user walks.** That rule is not yet
written down anywhere except in the row.

---

## 05 · What the seventh drive changed

**Drive 7 (2026-08-22) produced nine findings, one `critical`, in a single evening** —
`DRIVE-054`…`062`, verbatim log committed at `docs/audit/DRIVE7_OBSERVATIONS.md`.

**It is the best-instrumented drive this project has had**, and the reasons are worth keeping because
they are reproducible: the tester reconciled counts against ground truth he checked in a shell before
trusting the screen, quoted verbatim rather than paraphrasing, and **separated "this is wrong" from
"this felt bad"** — a distinction the project had never asked for and which sorted the findings into
two genuinely different repair queues.

**It also corrected the adjudicator three times**, including the `DRIVE-056` acceptance above and a
brief that sent the tester after a sentence that does not exist in the build.

**The standing policy is `PM_TRANSITION.md` §02: drive when a loop has shipped something a person can
see, and accept that reachable-but-unreadable defects accumulate between drives.** Drive 7 is the
first measurement of what that trade actually costs: **nine findings, one critical, none of which any
of 2,774 tests could see.** The policy has never been re-examined against a number.

---

## 06 · Condition three has no instrument, and the sprint produced its first real evidence

`PRODUCT_VISION.md` §06b requires **correct, surfaced, and beautiful.** `ROADMAP.md` condition 7
carries it. `pageharness.py` states in its own docstring that it proves what the controller renders
and **cannot prove visibility.**

**Drive 7 is the first time condition three was measured at length**, and the results are not about
taste:

- **Red rendered zero times in six runs**, while the two states whose own text earns it — an
  uninterpretable holdout and a split the app itself labels *"NOT A VERIFIED CLEAN SPLIT"* — render
  **green** and **amber**. The app defines its palette as *claims*; this is the governing rule failing
  through color rather than through a sentence (`DRIVE-058`).
- The tester's summary of the whole app is worth putting in front of a product owner verbatim:
  *"The skeleton of something researchers would trust is present. What breaks the promise is that the
  page moves underneath you, the summaries go stale the moment you act, the app repeatedly diagnoses
  problems that it gives you no control to fix, and the loudest dangers wear the quietest colors."*

**Open question: does condition three get an instrument, or does it stay a human-only check
indefinitely?** Nothing in eight loops has moved it.

---

## 07 · The documents are now load-bearing in a way they were not designed for

Both the execution agent and the adjudicator are cleared between loops. That makes
`AGENT_ONBOARD.md` and `PM_TRANSITION.md` the entire inheritance.

**Audited at `2761ab8`, the onboard was in better shape than the decay pattern predicts** — its three
pytest invocations parse, every `--ignore` path is real, and two timing claims reproduced exactly.
**But every staleness found pointed the same direction: work described as outstanding that was
already done**, including a paragraph asserting in the present tense that one of the six pre-commit
gates is silently false-green when that had been fixed. **That direction is the dangerous one for a
fresh agent** — it invites re-fixing finished work, or distrusting an honest gate.

`AGENT_ONBOARD.md` is 653 lines. `PM_TRANSITION.md` is ~340. `LOOP.md` is ~570 and its §03 log rows
have grown from two lines to paragraph-length essays.

**Open question: at what length does an onboarding document stop being read in full, and has that
already happened?** Nobody has tested whether a fresh agent actually reads them.

---

## 08 · Questions this pack deliberately does not answer

These are the product owner's, and they are the reason a retrospective is a conversation rather than
a report:

1. **Is the rising open count acceptable, or does the definition of done move?**
2. **Does the fan-out keep running at ~2.6M tokens a loop, and is there a cheaper configuration
   nobody has tried because it worked the first time?**
3. **Drive cadence.** Drive 7 found nine defects no test could see. Does that change *drive when a
   loop ships something visible* into something more frequent?
4. **Does condition three get an instrument, or stay human-only?**
5. **`DRIVE-054` is `critical` and its repair is the identity-preserving DOM write specified at L54
   and never built.** That is a large piece of work. Does it go next, ahead of the domain content?
6. **The three recurring error shapes in §04 each already have a written rule.** If writing the rule
   down does not stop the recurrence, what does?

---

*Compiled at `3f14f64` by the adjudicator, before being cleared. Numbers in §02 re-derived by walking
`findings.json` at fourteen commits; §03 token counts from the workflow runs' own reported totals;
§04 B re-derived at `0856c1d`; §05 and §06 from `docs/audit/DRIVE7_OBSERVATIONS.md`.*
