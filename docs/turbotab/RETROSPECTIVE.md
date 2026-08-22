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

## 02 · How the work scales — OPEN, direction stated

He raised increasing the number of parallel execution agents (*"get more developers on this
team"*) as the way to buy both correctness and build rate. Raised, not yet ruled; the sprint's own
evidence on coordination costs (TEST-112's cross-agent false green; eight consecutive loops of
adjudicator corrections) is being weighed against it in this conversation.

## 03 · "A long term memory of what needs to go inside it" — OPEN, needs definition

His words: *"Ultimately, I want us to have a long term memory when we build this app of what needs
to go inside it."* The ledger is the long-term memory of **defects**; what he is pointing at is a
durable record of **content and capability still owed** — the domain-informed material of the one
big definition-of-done change. What that instrument is (and whether `ROADMAP.md` /
`DOMAIN_PACKS.md` / `FEATURE_REGISTER.md` already are it, badly) is not yet settled.

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

---

*Sections below this line are appended as the conversation continues.*
