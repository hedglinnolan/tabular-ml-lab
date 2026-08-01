# Running this as an unsupervised loop

The operator's manual: how a loop is shaped, how to run one safely, how to judge the report that
comes back, and what has already run.

**If you are taking this project over, read in this order:** `README.md`, `PRODUCT_VISION.md`,
`ROADMAP.md`, then this file's §02 (the shape), §06 (adjudication) and §03 (the log). The rest is
reference.

`data/findings.json` is the source of truth. `FINDINGS_LEDGER.md` is generated. The agent edits the
JSON through `tools/ledger.py` and regenerates the markdown — never the reverse.

```bash
python docs/turbotab/tools/ledger.py stats            # progress
python docs/turbotab/tools/ledger.py next --n 15      # next batch, as JSON
python docs/turbotab/tools/ledger.py set ID --status OPEN --note "..." --evidence "file:line"
python docs/turbotab/tools/ledger.py regen            # rewrite the markdown
python docs/turbotab/tools/ledger.py check            # schema guard; non-zero on violation
```

`check` enforces what makes the ledger trustworthy: no duplicate ids, no invalid status, **no
`FIXED` without a named regression test**, and no `PARTIAL` / `NOT-A-DEFECT` / `WONTFIX` without a
written reason. It runs in the pre-commit hook, not on discipline — see §05.

---

## 01 · Two numbering systems, and why they are not renamed

`L1`–`L12` in `ROADMAP.md` name **phases**. `L1`–`L29` elsewhere name **loops**.

They coincided through `L8`, when one loop completed one phase. They diverged at `L9` — the
interaction layer — which is one phase and many loops. **Every loop from `L13` onward sits inside
roadmap phase `L9`**, alongside the domain track that `ROADMAP.md` §"The map" now carries.

The collision is documented rather than renamed because the references number ~100 each way and
reach generated files and source. A rename is churn with breakage risk; this paragraph is the fix.

**Reading a loop number in the wild.** An `L` tag on a finding marks **where the defect was
observed**, not where it landed — `GUIDED-046` carries both `L18` and `L24` because that is one
cause with two occurrences. To reconstruct a loop's contents:

```bash
python3 -c "import json,re,collections; d=json.load(open('docs/turbotab/data/findings.json')); \
r=d if isinstance(d,list) else d['findings']; \
[print(f['id'],f.get('item','')[:70]) for f in r if 'L24' in str(f)]"
```

`VALUE_CHECK_ADJUDICATION.md` carries a section per loop that moved a routing metric, and it is the
most detailed narrative record that exists. It is **not** a complete loop log — it only sees loops
that moved the value check — which is why §03 exists.

---

## 02 · The shape of a loop

A loop is **one prompt, run unattended, reporting once.** The prompt is four parts, and the shape
emerged from practice rather than design — it is written down here so it stops being rediscovered.

| Part | Role | Why it is there |
|---|---|---|
| **A** | close the previous loop's gap | The adjudicator names one thing the last report left. Doing it first means an accepted loop never quietly carries debt into the next. |
| **B** | the substantial build | The loop's reason for existing. One thing, deep. |
| **C** | a second build, deliberately different | Two builds of different shape expose an abstraction's seams. One build never does. |
| **D** | a probe, an audit, or a refusal | Something that tests whether what exists holds, rather than adding to it. |

Each part is independently gated: it lands with its own tests, its own ledger rows, its own commit.
**A loop that completes three of four parts is a loop that produced three parts**, not a failure —
which is why the parts are ordered by what would hurt most to lose.

**Rules that make the shape work, all learned by breaking them:**

- **Name the source, not the goal.** *"Build the nutrition detectors"* hands the agent the job of
  deciding what the science is. *"Implement the Atwater reconstruction, source of truth
  `research/NUTRITION_PACK.md` §01, read that section first"* does not. See §04.
- **Two, never three, when testing an abstraction.** The figure spec shipped with exactly two
  figures on instruction, and the second exposed two seams the first did not. A third before the
  spec survived both would have hardened the wrong shape.
- **And stop metering once it has stopped bending.** The rule above is about *discovery*, and read
  as a general cap it is expensive. The packs specify **eight** signature figures (`DOMAIN_SCIENCE.md`
  §02), **ten** import detectors (§03b) and ~150 anti-patterns, all written down to the annotation
  and the threshold. Releasing those two per loop is a schedule invented by the prompt rather than
  by the work. **The test for which phase you are in is not a count — it is whether the last example
  bent the abstraction.** L27's third figure did (`annotations` bent, `tier` did not fit), so figures
  were still in discovery at three; the next one of a *shape already seen* is fill-out, and a
  fill-out part should carry as many instances as the agent can hold. Product owner's framing:
  *"why stop at 2 plots when you know they could make all n plots that work well for a given card."*
- **Order a batch hardest-first.** The corollary that keeps a widened part honest. Five instances
  built easiest-first are five castings of a shape nobody stress-tested; built hardest-first, the
  seam shows while four are still unwritten. Judge "hardest" by what is most likely to break the
  abstraction, not by effort.
- **Say what not to build.** Every loop since `L25` has carried an explicit *deliberately unbuilt*
  clause. Agents finish things; the instruction to stop is load-bearing.
- **Scope note at the end.** When one part is allowed to run long, say which and what may be traded
  against it. *"If C runs long, ship the figure with fewer annotations rather than dropping the
  refusal."*

---

## 03 · The loop log

Terse by design. The narrative lives in commit messages, `VALUE_CHECK_ADJUDICATION.md`, and the
findings themselves; this is the index that makes them findable.

| # | What it did | Why those parts together |
|---|---|---|
| L1 | Verified 370 findings against current code | A backlog nobody has re-read is research, not a plan |
| L2 | Three live Streamlit bugs | Folded into L7 |
| L3 | Walking skeleton — upload → diagnosis → findings → decisions | Prove the architecture before building on it |
| L4–L7 | Characterization tests, AnalysisProject + DAG, split block, detaint + job queue | The engine had to be safe to call before a new door could call it |
| L8 | The Router, EDA only · **value check passed** | Routing is the differentiator; measure it before building on it |
| L13–L17 | `GUIDED-011`–`021` — early Guided hardening | Reconstruct via the snippet in §01 |
| L18 | Split "whether to scale" from "which scaling" (`GUIDED-022`) | Determinacy is a property of a **question**, not an operation. Added the `constitutional` question category rather than moving the coverage denominator |
| L19 | The lens becomes the third constitutional question | |
| L20 | The lens reaches `rank_findings` (`GUIDED-024`–`029`) | Reframing annotates and never deletes — a lens at generation would make presented and executed diverge |
| L21 | Pack benefit fell because the comparison became fair | A metric moving the wrong way for the right reason is recorded, not corrected |
| L22 | `GUIDED-031`–`033`, found by driving | |
| L23 | The drive bugs — `DRIVE-001`/`003`/`004`/`011` | The product owner drove the app; the lens was unreachable |
| L24 | Bulk repairs (`DRIVE-002`) — nine cards, one decision | Value check penalized the improvement twice; recorded, no threshold moved |
| L25 | Purpose question (`GUIDED-048`), evidence badge (`047`), sixth axis (`045`), SMOTE defect (`049`) | The spine primitives everything downstream needs, plus the first anti-pattern audit hit |
| L26 | Prereg Amendment 1 (`050`), figure spec + two figures (`051`), promotability (`052`), study-scoped finding (`053`) | The figure spine, tested by two deliberately different figures |
| L27 | Nutrition pack — Atwater, NHANES design, shrinkage plot, the EAR/AI **refusal** | The reference implementation. A pack that can only add findings has not been tested. **Accepted; all four parts correct and tested, and none of it is reachable from an upload** (`GUIDED-058`). The calibration fix is the part that ships — `weak_calibration` is live in Classic via `pages/06` |
| L28 | Reachability — the badge reaches the claim, the figure layer gets its first consumer, the detectors reach an upload, the refusal is probed from outside | L27 built content nothing could reach. **Accepted; `GUIDED-057`/`058`/`059`/`060` closed**, and an upload now draws two figures and triggers four refusals, verified through the API by the adjudicator rather than through the loop's own tests. The loop's sharpest finding came from the wire and no unit test could have: the dietary pack read a 107 kg body-weight column as a survey weight and asserted a partially-specified design on a clinical table. **Two seams found by widening, not by building** — `evidence.py` grew from one file to an AST walk over every emitter, and the `Callable[[df], Optional[Dict]]` detector contract was **split rather than widened**, on the argument that widening rests on one example |
| L29 | Three more figures, the matcher that read the wrong columns, and the first anti-pattern audit | **The first widened loop** — three figures at once rather than two, ordered hardest-first, on the rule added to §02 this cycle. It worked: the volcano bent the spec exactly where predicted (a precondition on *upstream data state*, which `when_applicable` cannot express) and the spline bent nothing, which is the other half of the result. Five of six figures now reach a user. **The audit is the headline**: 85 anti-patterns checked, 4 hit, and `AUDIT-001` is the SMOTE defect's exact shape — the generated manuscript reports how many tests were significant at raw *p* < 0.05 and names no correction |
| L30 | The audit turned on the app — the manuscript's uncorrected count, the leaking baseline, the exits that named no key | **The loop where the research stopped being content and became a bug report.** `AUDIT-001` removed a false statistical claim from the *generated manuscript* and `AUDIT-002` closed a real leak in the quick baseline, both in code Streamlit users run today. `GUIDED-064`/`072` fixed a class I had half-named: the machine-readable form was lossier than the sentence, in a badge and in a 409's exit. **The most valuable line was in prose and is now `AUDIT-008`** — four of the last four hits share a shape the registries were never written to find: the core already holds the capability and the path that needs it does not read it. Also the loop where **I** broke §05 by running `git add -A` mid-loop; see the rule added there |
| L31 | The recipe lattice prototyped, the purpose-blind refusal, the capability sweep, and the frontend harness | **The loop that measured a design question instead of arguing it.** The lattice prototype changes 6 of 16 cells when the lens moves — reproduced by the adjudicator — and establishes that whatever renders it must mutate cells in place, because `transitioncancel` never fires on a removed element, so a wholesale repaint cannot report what it interrupted. That narrows `GUIDED-073` without closing it. **And the harness earned itself on its first run**: the page never fetches `/figures` (`GUIDED-075`), which two loops of API verification — mine included — structurally could not see |
| L32 | The figure surface, the server's exits, two Classic leaks, and the copy deck walked | **The money maker became visible.** `GUIDED-075` closed and the adjudicator drove it — the page's own controller, live API responses, a real dietary project — and the figure surface renders three admitted figures with annotation boxes, scored checklists, captions and badges. `GUIDED-076` found something worse than discarding: `new Error(object)` stringifies to `[object Object]`, so a 409's exits were destroyed at the throw. **And the loop named the door's dominant defect class from three instances** — the server composes a user-facing string and the interface never renders it. Measured at `GUIDED-080`: six surfaces, and the sharpest is `nutrition/prevalence`, the refusal apparatus the whole domain-track ordering was justified by |
| L33 | The class closed — the prevalence refusal, the seal's basis, the Features step, the lattice, and a standing check | **The dominant defect class stopped being a finding and became a gate.** Five surfaces the server composes and the page never read now reach a person: `nutrition/prevalence` (the four refusals `LOOP.md` §04 says the whole domain-track ordering was justified by, plus the shrinkage plot the single-day refusal offers), the `disclosures` field, `/features` and `/selection/evidence`, and `/recipes`. **The seal was the closest to a governing-rule failure**: it tells a user whether their holdout was drawn by person or by row, and §03's `undetermined` had no renderer at all. Two findings the loop made rather than fixed: `GUIDED-081`, the harness silently dropped every `className` write, so any assertion about styling came back **passing** — the §03 claim would have passed against a page that got §03 exactly backwards; and `GUIDED-084`, `/capabilities` is served precisely so the page cannot claim an affordance the server lacks, and the page computes its own. **The measurement was wrong and the check corrected it**: `evidence/correlations` and `evidence/histograms` were counted unread because `runPull` composes their paths from a variable and no literal grep can see them — seven routes unread, not nine, five of them read by another door |

**Adding a row is part of adjudicating a loop.** Two lines, written when the report is accepted.
This log decayed once because it lived only in chat; that is the failure this project has already
paid for twice, in two different places.

---

## 04 · Loops that build a domain pack

The four research threads in `docs/turbotab/research/` are **3,602 lines and are the authoritative
source** for every pack detector, coaching sentence, threshold and figure specification. They are
not background reading. A loop that builds pack content without citing them has invented its
content, which is the failure this whole apparatus exists to prevent.

Three problems, solved differently.

**Volume.** Nobody holds 3,602 lines. **The task block names the file and the section**, and the
agent reads that slice. A task block that says *"build the nutrition detectors"* without section
pointers is malformed.

**Provenance.** Every pack advisory, detector and figure spec carries a **`source`** naming file and
section, and an **`evidence_status`** of `SETTLED` / `CONVENTION` / `DISPUTED`. A checker verifies
the named section resolves, and runs in `.githooks/pre-commit` beside the other gates. Its limit,
stated wherever it is stated at all: it verifies a source is *named and resolvable*, never that the
claim is faithful — the same posture and the same honest limit as `ledger.py check` enforcing that a
test is named.

**Where the research file and the model's recollection disagree, the file wins.** The files were
built under a blocked egress proxy and say so; a threshold in the file is a recorded, checkable
claim, and one from memory is neither.

**`[verify-at-build]` is a hard stop.** Such a number ships as an `offered` item with its
uncertainty stated, or not at all — never as a hard-coded constant. Where the fact is unavailable,
that is a `BLOCKED.md` entry, not a guess.

**Sequencing.** Four parallel domain verticals is the wrong shape: the packs share the figure spec,
annotation engine, badge rendering and checklist engine, so built four times they are built four
ways. **One pack end-to-end first as the reference implementation** — discovering the abstractions
is the deliverable alongside the pack. Nutrition went first because the product owner can adjudicate
its content, the NHANES fixtures are real, and it is the one pack that forces a **refusal**.

---

## 05 · Guardrails

Append this to any unsupervised prompt.

> **Hard rules.** Stay on branch `TurboTab`. Never push to `main`, never force-push, never open a
> pull request. `ml/import_doctor.py`, `ml/join_doctor.py`, `utils/combine*.py` and
> `pages/01_Upload_and_Audit.py` are **frozen — see `TRANSITION_PLAN.md` §05 for the one statement
> of what that permits and the gates that lift it.** Never edit `FINDINGS_LEDGER.md` by hand; it is
> generated. Never mark a finding `FIXED` without a regression test **verified to fail when the fix
> is reverted** — see `FEATURE_PARITY.md`, "the revert probe". **First command in a fresh clone:**
> `git config core.hooksPath .githooks`. Commit after every batch so nothing is lost. **Domain
> science comes from `docs/turbotab/research/`, never from recollection** — where a research file
> and your memory disagree, the file wins, and a number marked `[verify-at-build]` may not ship as a
> hard-coded constant. If you are blocked or something looks structurally wrong, stop and write what
> you found to `docs/turbotab/BLOCKED.md` rather than guessing.

**One writer at a time, scoped to the shared data files** (`findings.json`, `register.json`, and
their generated markdown). A docs-only commit from another session may land mid-loop **only if** it
touches none of those files and its commit message says so; the loop agent rebases over it. Anything
touching the data files waits. That is the artifact this project has already lost once.

**The adjudicator never runs `git add -A`, and the rule is written here because it was broken here.**
Two commits during `L30` — `9ebf95d` and `7dd6aa6` — carry `docs(turbotab):` subjects about the
frontend stack and contain `turbotab/exits.py`, `turbotab/packs.py` and five other source files that
were the loop agent's uncommitted Part A at that moment. Nothing was lost and every gate stayed
green, but the subject line of a commit is a claim about its contents, and those two assert something
false about themselves — the governing rule failing in the record layer rather than in the app.
**Stage explicit paths, and run `git status` first, every time.** The failure was not carelessness
about the rule; it was not noticing that a conversation about architecture is a moment when a loop
may be running. The docs-only exemption above is what makes an adjudicator's commit safe mid-loop,
and `git add -A` silently converts a docs-only commit into a mixed one.

**Verification loops run in a fresh session, not the builder's.** A builder verifying their own work
reads intent where the job is to read code — the same reason a review pattern never lets the finder
be the judge. And both write the same data files.

**The three gates are a hook, not an instruction.** `.githooks/pre-commit` runs `ledger.py check`,
`register.py check` and `tests/test_american_spelling.py`, and refuses the commit on any failure.
This replaced a line in the guardrails that enforced nothing: commit `8127101` went out with the
spelling test red because the gates were chained with a newline instead of `&&`, so a non-zero exit
did not stop the sequence. **An instruction a tired agent can skip by punctuation is not a gate.**
`core.hooksPath` is local config, so the one command above is the only part still carried by
discipline. Bypass with `--no-verify`, and say why.

**Run the documented setup path, or find out that nobody has.** The `Makefile` names
`./venv/bin/python` and nothing had created it in long enough that the spelling gate's skip list had
`.venv` and not `venv` — so the gate died on a compiled dependency the first time anyone followed
the instructions. A setup path is a claim like any other and decays the same way: silently, while
the people with working environments keep working.

**It decayed again, in the same file, and this paragraph did not catch it.** Found while adjudicating
L27 (`TEST-039`): `PYTEST_OPTS` passes `--timeout=60`, `pytest-timeout` is absent from `venv/`, and so
`make test` — and `make verify`, which line 13 of the `Makefile` calls the CI target — exits 4 having
run **zero** tests. Every count this project has reported came from a hand-rolled `pytest` invocation
instead: the numbers are real, and they are evidence about a command written down nowhere. The rule
above failed because it is prose. **Adjudicating a loop now includes running `make test` once** — the
only form of this rule that can actually fire.

**The freeze** and the three gates that lift it are stated **once**, in `TRANSITION_PLAN.md` §05. Do
not restate them; this file once said "never modify" while §05 said "engine-move-only", and a reader
following the stricter one could not do the work §05 permits.

---

## 06 · Adjudicating the report

The half of this job that was never written down. A report is a **claim**, and the whole project
runs on the rule that a claim needs a record.

**Verify before accepting, and verify the load-bearing claim specifically.** Not everything — the
one thing the rest depends on. Pull the branch first; the agent's commits will not be in your tree.

```bash
git fetch -q origin TurboTab && git rebase origin/TurboTab
python docs/turbotab/tools/ledger.py stats          # do the counts match the report?
grep -c '"GUIDED-0NN"' docs/turbotab/data/findings.json   # was it actually filed?
make test                                           # does the documented gate still run? (§05)
```

**And one grep that is not about the report at all:** does anything outside a test file import what
the loop just built? A module reachable only from its own tests is a specification, and the report
will not say so, because from inside the loop it looks finished. This is how `GUIDED-058` was found.

**Then stop grepping and run it.** The adjudicator has now been wrong three times in the same way,
each time by searching for the shape expected rather than the shape written: `packs.findings` was
called from `engine.py` while the grep covered `api.py` and `ml/`; `AUDIT-001`'s sentence spanned two
f-string lines and a one-line pattern returned nothing; and `runPull` composes `"/evidence/" +
endpoint`, so a literal path search reported two surfaces unread that are fetched on every drive. A
grep answers *does this text appear*, and the question is almost always *does this run*.

The instrument already exists. `turbotab/pageharness.py` runs the page's real controller in node
against captured API responses, and `__harness.calls()` returns exactly which routes it fetched —
no pattern, no ambiguity. Feeding it live responses from a `TestClient` drive takes about thirty
lines and answers reachability questions that no search can. **Where a claim is about behavior,
drive it; reserve grep for claims that are genuinely about the file.**

**What to look for, in order of how often it has mattered:**

1. **Was a named defect *class* filed, or only its instance?** The highest-value finding of `L26`
   was a class the agent named in a docstring and did not file. A class that lives only in prose
   will be forgotten. **This is the single most common gap in an otherwise good report.**
2. **Did a threshold move?** Never accept a moved threshold in the same loop as the change that
   pressured it. If a gate is measuring the wrong thing, correct **which quantity is gated**, on a
   *passing* run, with the reasoning recorded before it is load-bearing. After a breach the same
   correction is indistinguishable from relaxing a gate under pressure.
3. **Does new numerical code have its own tests?** `weak_calibration` was hand-validated and
   exercised only through a figure test. A hand-check that is not a test is a claim without a
   record — the project's own rule, one level in.
4. **Does the code return a value where it should return nothing?** The strongest habit this
   project has: `(None, None)` rather than `(0.0, 1.0)`, because those are the values of *perfect*
   calibration and returning them from ignorance asserts perfection.
5. **Did a sweep terminate where the sweeper's attention ended?** Sweeps find the class they were
   pointed at. Ask what the same lens would find one surface over.
6. **Is a capability being deleted where it should be routed?** The SMOTE defect was fixed by
   routing behind purpose and keeping an offered path, not by removal. The shelf is never shortened.

**Then, before writing the next prompt:** add the §03 log row, and name in Part A the one gap you
found. If you found none, say so — an empty Part A is a real outcome and should be visible.

---

## 07 · Checking in

```bash
git -C . log --oneline TurboTab | head -30
python docs/turbotab/tools/ledger.py stats
python docs/turbotab/tools/ledger.py check
git diff main...TurboTab --stat
cat docs/turbotab/BLOCKED.md 2>/dev/null
```

Three questions worth asking of any result:

1. **Does the `FIXED` count have tests behind it?** `check` enforces that a test is *named*; it
   cannot verify the test is any good. Spot-check two or three.
2. **How many went `NOT-A-DEFECT`?** A high rate means either the agents over-reported or the
   verifier is credulous. Read those notes specifically — that is where a loop quietly goes wrong.
3. **Did anything land where it should not have?** `git diff main...TurboTab --stat` answers it in
   one line.

---

## 08 · What not to hand an unsupervised loop

- **Row identity, and choices like it.** Design decisions with consequences across the whole project
  model. An agent can gather the evidence; it should not make the call alone.
- **Large extractions of untested, safety-critical logic.** Supervised, with characterization tests
  already in place.
- **New construction under a governing rule about what may be asserted.** Design work, not loop work.

Loops are for verification, for well-specified builds with clear gates, and for writing tests
against behavior that already exists. They are not for decisions you would want to argue about.

But the reframe in `ROADMAP.md` holds: those items are not permanently off-limits. They are blocked
on **one decision each**. Make the decision and the execution behind it becomes ordinary loop work.

---

## 09 · Standing dispositions

Rules extracted from loop prompts that have since retired, kept because they still bind:

- **`FIXED` requires a named regression test, verified to fail on revert.** No test, no `FIXED` —
  the finding stays `OPEN` and the test is written later. Roughly one in five first-attempt reverts
  is wrong and produces a plausible false failure, which is why the probe must fail *for the stated
  reason*.
- **"Guided avoids it" is never closure.** Streamlit never retires, so a defect still present in
  Classic stays `OPEN` even where the core or the Guided door has structurally resolved it. Note it
  `resolved-in-core; closes at L11 convergence of <page>` — verbatim, because that phrase is the
  queue for the convergence loop.
- **Tag, don't fix, siblings of a known pattern.** Add `sibling-of: <ID>` and move on. They get one
  batched build, not twenty inline ones.
- **Ambiguity is `OPEN`, never `FIXED`.** A wrongly-closed finding is worse than an open one.
- **A record that points at ephemeral storage will eventually lie, and it lies toward "the work is
  gone."** The original 48 import findings were declared unrecoverable while sitting committed in
  `docs/audit/`. Cite paths that are in the repository.
