# Retrospective — L58–L65 and drives 5, 6, 7 · held 2026-08-22

**Decisions from the sprint retrospective, written in the turn they were made, per `LOOP.md` §05
("a ruling is not a ruling until it is in a commit or a ledger note").** Evidence lives in
`RETROSPECTIVE_PACK.md`, which this conversation was held against and which stays unedited. The
product owner's words are quoted where they were his.

Status of each item is marked: **DECIDED** (his ruling, recorded), or **OPEN** (raised, not yet
ruled). The handoff section at the end is written last.

---

## 01 · The backlog is frozen, and that is ruled a defect in the method — DECIDED

**The measurement that reframed pack §02** *(re-derived this session at `6f3efad` vs HEAD
`cd89962`, by ID-diffing `findings.json` between the two commits)*:

- Of ~520 findings open at sprint start, the sprint closed **2**.
- Of ~110 findings filed during the sprint, the sprint closed **65** (59%).

So the sprint's closure rate on fresh findings is healthy; what never happens is a loop pointed at
anything older than itself. The pack's "either the rate changes or the condition does" missed the
third option: **the allocation changes.** No loop prompt in the sprint ever targeted the backlog.

**The product owner's ruling, verbatim:** *"Not a policy I would defend at all."*

Consequence: future loops must carry backlog closure as deliberate work, not as a side effect of
what they happen to find. The mechanism (how much per loop, dedicated loops vs. a slice of each) is
still under discussion — see §02.

**Context he added, recorded because it explains the goalpost feeling:** he has changed the
definition of done once in a major way — *"add domain-informed content starting at step 10 of the
app that cascades down into the routing and what options are presented to the user"* — and that
inherited change, not drift, is why the goalposts feel like they keep moving. The target itself he
regards as stable.

**His stated objective for the method, verbatim:** *"I want to strike the balance between
components are correct and components are built."*

## 01b · Sequencing and the shape of the three excellences — DECIDED

The product owner clarified the one big definition-of-done change: domain-informed content **at
step 0** of the app (not step 10), cascading down into the routing and what options are presented.
He affirmed excellence in all three of capabilities, rules, and design — which is §06b restated,
not a new demand — and tentatively ordered them capabilities → rules → design, then asked for a
recommendation. **He accepted the following, verbatim ruling: "I do accept your recommendations."**

1. **Substrate first.** The identity-preserving DOM write specified at L54 (the `DRIVE-054`
   repair) and the stale-summary-panel class go ahead of the step-0 domain content, because those
   defects tax every subsequent capability, rule, and drive — Drive 7 lost ten state changes to
   the page moving, including the target twice, each a permanent false line in the transcript.
   This settles pack §08.5.
2. **Rules and capabilities fuse: a diagnosis never ships without its lever.** Each piece of
   domain content arrives with the control that acts on it, as one unit of work. Evidence: Run 5a
   warned four times and empowered zero; Run 6 named `patient_id` itself and then recorded that no
   column was named.
3. **Design splits into two kinds.** Design-as-claims (color grammar, stale text — the palette is
   defined as claims) moves into the correctness queue and gets an instrument: the states are
   enumerable and the required tint is derivable from the app's own rules. This is condition
   three's first partial instrument and settles the direction of pack §08.4. Design-as-taste
   (card sameness, tooltip containers, typography) trails and stays human-checked by drives, on
   purpose.
4. **One typed memory queue.** The long-term memory of what the app still owes holds all three
   types in one place with the ledger's discipline — each entry typed (capability / rule /
   design-intent), filed once, closed with evidence. Whether it is a new ledger area or a promoted
   `FEATURE_REGISTER.md` is mechanics, settled in the handoff.

**Condition 5 of the definition of done stands unchanged** (the ledger closes; zero OPEN). What
moves is allocation (§01), not the condition. *(Recorded as the facilitator's reading of his
ruling; he has not contradicted it.)*

## 02 · How the work scales — DECIDED

He raised increasing the number of parallel execution agents (*"get more developers on this
team"*). The facilitator's recommendation, which he accepted (*"I say yes to both"*): **new
parallel capacity points at backlog closure across separated ledger areas first** (GUIDED 93,
STATE 76, TEST 65, CONTRACT 63, IMPORT 56 open at HEAD — largely independent surfaces, each with
written evidence and a named-test closure discipline), **not at parallel feature-building on one
DOM.** The coordination failure mode is on the record: TEST-112, one agent's stray `conftest.py`
silently poisoning another agent's measurement — a cost that scales with builders sharing a
surface, not with closers working separate areas. **Feature work stays serial through the loop
structure until the substrate repair lands; revisit after.**

## 03 · "A long term memory of what needs to go inside it" — DECIDED

His words: *"Ultimately, I want us to have a long term memory when we build this app of what needs
to go inside it."* Clarified in conversation: domain-informed content **at step 0**, cascading
into routing and options; and the memory must hold all three excellences — capabilities, rules,
design intents — in one queue (§01b.4).

**Mechanics, settled here as delegated: the container is `FEATURE_REGISTER.md`, promoted.** It is
already schema-gated at pre-commit, maintained through `tools/register.py`, generated from
`data/register.json`, and its own prose states the thesis — *"the count is what makes 'we owe N
things' a number rather than a feeling."* The extension owed (build work for a future loop, not
this retro): a typed owed-entry — `capability | rule | design-intent` — with states
`owed → built (with evidence) → verified`, so a diagnosis-without-lever like Run 5a's person
column is filed once as an owed capability and cannot be lost to a cleared session. **The ledger
stays defects-only**; the register becomes the memory of what the app still owes.

## 04 · Instruments-vs-surface: the blame-date audit — MEASURED

Run this session at HEAD `cd89962` (script preserved in session scratchpad; method: for each
sprint-filed finding, `git blame` every `file:line` its `ev` cites; bucket by oldest blamed date).

- 110 sprint-filed rows. **42 carry line-level citations blameable at HEAD.**
- Of those 42: **32 cite code last touched before 2026-08-09** (defect discovered, not created),
  **9 cite only sprint-era lines**, 1 unresolvable.
- The 9 sprint-era rows (TEST-077, DRIVE-037, TEST-092, DRIVE-050, DRIVE-051, MISC-025, MISC-032,
  TEST-111, MISC-033) are dominated by the instruments catching the sprint's own fresh work —
  the healthy case.
- Caveats, stated: the direction of error is conservative (sprint churn pushes rows *out* of the
  predates bucket, so 32/42 is a lower bound); the 68 rows without line citations (mostly DRIVE
  rows citing drive logs, and doc-cited rows) are uncovered by this method.

**Reading: on the measurable sample, roughly three-quarters of what the sprint filed was
pre-existing defect surface newly reached by better instruments — the light got better; the
surface was already there.** Not yet ruled on by the product owner; recorded as the measurement
pack §02 said nobody had.

## 05 · Drive cadence — DECIDED

**His ruling, verbatim: "I'd like to stick to an evening per 5 loops unless you think we would be
extended too far past our supply lines with that design."**

The facilitator's supply-lines assessment, given and recorded: **an evening per five loops is
workable, with the two riders that were part of the accepted sequencing** — (a) the substrate
repair's acceptance requires a drive regardless of where it falls in the count, because reflow is
exactly what no harness can feel; (b) a loop that ships a *new interaction pattern* pulls the drive
forward rather than stacking un-driven. Two structural reasons the floor is safe now when it was
not before: backlog-closure loops (§01) mostly re-verify known defects with named tests and are
drive-cheap; and the design-as-claims instrument (§01b.3) converts part of the drive-only defect
class into gated checks. Drive 7 priced the accepted trade: nine findings, one critical, none
visible to 2,774 tests.

**Drives follow the Drive 7 protocol**, now written into `PM_TRANSITION.md` §02: ground truth
reconciled in a shell before trusting the screen, verbatim quotes, and "this is wrong" separated
from "this felt bad."

## 06 · The recurring error shapes get mechanisms, not paragraphs — DECIDED

**His ruling, verbatim: "I am ratifying your second principle."** The principle: **a rule that
lives in prose is a comment; a rule that lives in a gate is a mechanism** — the method held to the
same law the project wrote about its code at `DRIVE-022`. Every future retro item that says "X
recurred" must answer *what gate now fires*, not *what paragraph now exists*.

The three consequences, executed this session:

1. Error shape A (a measurement's authority carrying into an unlicensed conclusion): every
   prescription in a loop prompt carries its own falsification, and pre-ship refuters diff prompt
   assertions against the reconnaissance's reported facts. Written into `LOOP.md` §06.
2. Error shape B (guards that measure a suspected mechanism rather than the consequence): the
   repository-wide count is owed and now has a door — **`TEST-113`, filed OPEN through
   `ledger.py`** — on the backlog queue §01 unfroze.
3. Error shape C (verification along a path no user walks): now an acceptance check in `LOOP.md`
   §06 — *ask of every acceptance probe: how did this state arise?* — rather than a sentence
   living only inside `DRIVE-056`'s row.

## 07 · The fan-out — DECIDED

**His ruling: "I say yes to both"** (this and §02). The refutation layers are kept
unconditionally — they are the demonstrably load-bearing part: six fan-outs, no driver has ever
come back clean, two SOUND refuters in that entire history, and in L65 the refuters caught two
errors inside the adjudication itself that would otherwise have entered the record as rulings
against a report that was right *(pack §03, from the workflow runs' own totals)*. What was never
justified is the **size** (nine-agent reconnaissance, fourteen-agent adjudication, ~2.63M tokens
at L65). **The control arm: on the first backlog-closure loop — the cheapest, best-understood
kind of loop the method now has — run a half-size configuration and diff what the full size would
have caught.** If the half-size arm misses nothing that matters, the savings fund the added
parallel closure capacity; if it misses real things, the 2.6M is finally a priced purchase instead
of an assumed one. This settles pack §08.2.

## 08 · Project management debt — ACKNOWLEDGED, itemized

**His words: "I realize we have been incurring project management debt from this 60+ loop
project."** The debt, named so it can be worked rather than felt (each item already measured
somewhere in the record):

1. **Onboarding staleness points the dangerous direction.** Pack §07, audited at `2761ab8`: every
   staleness found described work as outstanding that was already done — the direction that
   invites a fresh agent to re-fix finished work or distrust an honest gate.
2. **Document growth with untested inheritance.** `AGENT_ONBOARD.md` 653 lines, `PM_TRANSITION.md`
   ~340, `LOOP.md` ~570 with §03 rows grown from two lines to essays. Nobody has tested whether a
   fresh agent reads them in full. The mechanism principle (§06) applies to these documents too:
   rules they carry should migrate toward gates and point-of-use checklists, and prose that is
   only narrative history can be long without being load-bearing.
3. **Prose asserting the opposite of what shipped** — the `MISC-033` class (seven sites, four
   files, one falsified by its own commit), plus the ledger `ev`-rot class (`TEST-109`/`TEST-110`)
   now partially repaired by `set --ev`.
4. **Guards that measure mechanisms rather than consequences** — the countable, never-counted
   class, now doored as `TEST-113`.
5. **No owed-content memory** — settled by §03's register promotion; the debt until it is built is
   that owed capabilities live in drive logs and cleared sessions.

## 09 · What was examined and ruled fine — the number is three

A retrospective that invents changes for everything it touches is worse than one that says what
works. Three things were examined against the sprint's evidence and deliberately **not** changed:

1. **Condition 5 of the definition of done stands** (the ledger closes; zero OPEN). The blame
   audit (§04) showed the surface is not outgrowing the loops; the divergence was allocation, and
   allocation is what moved.
2. **The loop structure itself stands** — prompt, four-to-six parts, adjudication, divergence
   section, log row. Eight loops, eight acceptances, and the divergence section corrected the
   adjudicator all eight times; that is the system working, not failing.
3. **Design-as-taste stays a human-only check, on purpose** — the drive is the instrument, now on
   a ruled cadence with a written protocol. Condition three's *claims* half gets a machine
   instrument; its *taste* half keeps the human one.

---

## 10 · Handoff — what the next session needs before it writes L66, in order

*This section is the reason the retrospective was held. It records decisions; it does not author
the loop. The product owner authors the next loops in a separate session.*

**First, verify state off the machine, not off this page:** `git log --oneline -1` (this file's
last commit should be at or after `5ddd92f`), `python docs/turbotab/tools/ledger.py stats`
(1,005 total / 440 closed / 565 open+partial as of this writing — `TEST-113` is the +1), and the
six pre-commit gates run on any commit you make.

**1. The ruled priority order for the work itself.** Substrate first: the identity-preserving DOM
write specified at L54 (the `DRIVE-054` repair) plus the stale-summary-panel class — these tax
every subsequent capability, rule, and drive, and the product owner ruled them ahead of the step-0
domain content. **A drive gates the substrate loop's acceptance** (reflow is what no harness can
feel; L65 proved the harness can be handed a state no user can reach). After the substrate: the
step-0 domain content, shipped as **rule+lever pairs — a diagnosis never ships without its
lever.** Run 5a (warned four times, empowered zero) and Run 6 (the app named `patient_id` itself,
then recorded that no column was named) are the canonical instances the fusion exists to prevent.

**2. Every loop now carries backlog allocation.** The sprint closed 2 of ~520 pre-existing open
findings *(re-derived at `6f3efad` vs `cd89962`)*; the product owner ruled that indefensible.
Loops target backlog closure as deliberate work. New parallel capacity goes to area-separated
closure agents first (§02); feature work stays serial until the substrate lands.

**3. The fan-out control arm has a designated site.** First backlog-closure loop, half-size
configuration, diff against what full size catches (§07). Keep the refuters regardless.

**4. Drive cadence is ruled and written into `PM_TRANSITION.md` §02.** An evening per five loops;
the substrate loop and any new-interaction-pattern loop pull it forward; the Drive 7 protocol
(ground truth in a shell first, verbatim quotes, wrong-vs-felt-bad separation) is the required
method.

**5. Three new point-of-use checks live in `LOOP.md` §06** — prescriptions carry their
falsification and refuters diff prompt claims against recon facts; acceptance probes must answer
*how did this state arise*; guards measure consequences (`TEST-113` is the audit owed). The
ratified principle behind them: **a recurring error shape gets a gate, not a paragraph.**

**6. The owed-content memory is specified and unbuilt.** Promote `FEATURE_REGISTER.md`: typed
entries (`capability | rule | design-intent`), states `owed → built → verified`, schema-gated as
it already is (§03). Until built, owed capabilities noticed in drives should still be filed
somewhere durable rather than left in observation logs.

**7. The design-as-claims instrument is decided-direction, unspecified.** Color grammar is
enforceable: the states are enumerable, the required tint derivable from the app's own palette
rules (`DESIGN_LANGUAGE.md`). Drive 7's two canonical failures: "NOT A VERIFIED CLEAN SPLIT"
rendered amber, and the interval-wider-than-informative-range seal rendered green, while red
appeared zero times in six runs. Condition three's taste half stays human.

**8. Measurements you do not need to re-derive** (both stamped in this file): the frozen-backlog
split (§01) and the blame-date audit (§04 — 32 of 42 line-citable sprint findings point at
pre-sprint code; the script is preserved in the session scratchpad and is cheap to re-run at any
HEAD).

**9. The PM debt list is §08.** Item 1 (onboard staleness, dangerous direction) is the one that
bites a fresh agent first.

**10. What did not change is §09.** Do not reopen condition 5, the loop structure, or the
human-only taste check without new evidence.

*Retrospective held 2026-08-22, facilitated against `RETROSPECTIVE_PACK.md` at `cd89962`, which
remains unedited. All rulings quoted are the product owner's; all measurements carry their
derivation.*
