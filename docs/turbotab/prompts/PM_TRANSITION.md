# You are taking over as project manager and adjudicator for TurboTab

The previous PM was cleared. Everything durable is committed; this is the contextual onboard for what
is not. **It replaces the L62-era version** — a stale transition beside a current one is the decay this
project has already paid for seven times.

**One thing is unfinished and it is the first thing you do. See §04b.**

---

## 00 · Three operational facts, and read them off the machine rather than off this page

```bash
ps aux | grep "[p]ytest"; git status --short          # 1 · is a loop live? If yes, do NOT write findings.json / register.json
git log --oneline origin/TurboTab..TurboTab | wc -l   # 2 · how far ahead — an unpushed loop is the risk
lsof -nP -iTCP:8777 -sTCP:LISTEN                      # 3 · what is serving
lsof -p <pid> | grep site-packages                    # ...and on WHICH interpreter. ps CANNOT answer this
curl -s localhost:8777/dev/status                     # ...and which build it says it is
```

The app launches with **`make turbotab`** — it prints its interpreter, environment and rev before
binding, and **refuses with exit 2** on a stack it cannot import. `make turbotab-check` binds nothing.
**`make serve` is not TurboTab**; it is the old Streamlit app on 8501.

**`ps` cannot tell you the interpreter** — `venv/bin/python` is a symlink, so `ps` prints the resolved
Homebrew path and a complete venv is indistinguishable from bare system Python. Use `lsof` on the
serving process, or `/dev/status`. That confusion cost four drives.

---

## 01 · What this work is

TurboTab is **research software**. Your job is statistical methodology and software engineering. There
is no patient anywhere in this system; the "biology" is reference data and methodological literature.

**Precision is the safety property and hedging is the defect.** The governing rule:

> The app may be **silent**, and it may **refuse**, but it must never **assert something false.**

There is a calibrated apparatus for uncertainty — SETTLED / CONVENTION / DISPUTED, a resolving `source`
on every claim, six pre-commit gates. **A second, uncalibrated layer of caution does real damage.**

The role requires **ruling against reports** — and against yourself in public. §06 is a list of my own
errors and it is the most useful section here.

---

## 02 · The working relationship

Nolan is the product owner — *"the product design guy."* He does not read the code; you do. He runs an
execution agent on his laptop, pastes its reports to you, and **you rule and write the next prompt.**
**He expects you to be better than him at orchestration detail, so make calls, do not survey options.**
When he reaffirms something, that is a decision.

Each loop prompt goes to `docs/turbotab/prompts/L<n>.md` and is published as an Artifact with a copy
button. **The builder is disposable and lives in the scratchpad — regenerate it.** It reads the repo
file and embeds it **byte-for-byte as a JS string literal** (not escaped element text — HTML entities
are not decoded inside `<script>`), escapes `</` as `<\/`, and **verifies the literal decodes back from
the written file.** Style comes from `DESIGN_LANGUAGE.md`: the five-hue palette, the three-voice type
rule, and **`--stop` is reserved to the blocker band and is not severity styling**, so it must not
appear on a prompt page at all.

### His standing rulings

- **`ROADMAP.md` condition 7** — *"In addition to being correct, the engine must surface and it must be
  beautiful."*
- **Time is the constraint.** He runs loops back to back. His words at L64: *"let's just keep devving."*
- **He is a completist** — *"start fixing more items per loop."*
- **A scoped sweep is acceptable evidence, quoted as scoped.**
- **Ultracode is on.** Orchestrate substantive adjudication with the Workflow tool. §07 is what the
  fan-out has actually found, and it is the argument for keeping it.
- **We do not drive every loop.** *"We need to continue building the product and we cannot drive and
  test every single loop we run."* **The cadence was ruled at the 2026-08-22 retrospective, in his
  words: an evening of driving per five loops**, replacing the trigger-only form, with two riders
  that pull a drive forward regardless of the count: **the substrate repair (`DRIVE-054`) cannot be
  accepted without a drive**, because reflow is precisely what no harness can feel; and **a loop that
  ships a new interaction pattern** (not merely new copy on an existing pattern) **does not stack
  un-driven on top of others that did.** The drive, when it comes, still covers every surface built
  since the last one — say in the request which screens are new. **Drives follow the Drive 7
  protocol** (`docs/audit/DRIVE7_OBSERVATIONS.md` is the exemplar): ground truth reconciled in a
  shell before trusting the screen, verbatim quotes rather than paraphrase, and *"this is wrong"*
  separated from *"this felt bad"* — the separation sorts findings into two different repair queues.
  Reachable-but-unreadable defects still accumulate between drives; Drive 7 priced that trade at
  nine findings, one critical, none visible to 2,774 tests.

**Take his framings seriously; they have been load-bearing more often than mine.**

---

## 03 · Read, in this order

`README.md`, `PRODUCT_VISION.md`, `ROADMAP.md`, then `LOOP.md` §02, §05, §06, §03 (the last six rows are
the live context). Then `DOMAIN_SCIENCE.md`.

**Do not skip these; each is a ruling recorded because it decides work:**

- `PRODUCT_VISION.md` **§06b** (correct, surfaced, beautiful) · **§06c** · "The shelf is never
  shortened" · **§09**, which carries four positions on the scroll rule.
- `ROADMAP.md` **condition 7**; "Why the front of the journey is where the depth belongs."
- `DESIGN_LANGUAGE.md` **§05**, **§05.2**, **§07**, **§10**.
- **`AGENT_ONBOARD.md`** — the execution agent's onboard; everything in it binds you too. **§03 is
  parsed by two guards** — one asserts it still offers a parallel invocation carrying `--dist
  loadfile`, the other reads its `--ignore` list. Editing that block without reading them costs a loop.
- **`VALUE_CHECK_ADJUDICATION.md`** — the authority when a frozen baseline moves.

**Five research files in `research/`, authoritative. Read by section, when cited, never wholesale.
Where a file and your recollection disagree, the file wins.**

---

## 04 · State right now

Branch `TurboTab`, tree clean. Ledger **989 findings, 434 closed** (`OPEN` 471 · `PARTIAL` 84 ·
`FIXED` 424 · `NOT-A-DEFECT` 10) *(from the tool at `2761ab8`)*. Six pre-commit gates green.
**L58–L64 accepted with their `LOOP.md` §03 rows written.**

**Read HEAD off the machine, never off this line** — an earlier version of this paragraph named a
commit and a count that were both one loop stale by the time it was read, which is the decay the whole
document is written against.

**The parallel sweep is licensed** as of L63 and `AGENT_ONBOARD.md` §03 carries the table:
`-n 8 --dist loadfile` at **41:36** against serial **2:01:30**, both 0 failed. L64 re-took it at its own
commit: **0 failed · 2,709 passed · 17 skipped · 9 xfailed in 41:49.** The four tests that wrote inside
the repo now write to `tmp_path` and a guard holds them there.

## 04b · L64 is adjudicated. **Accepted four and a half of five.**

The verification landed and the loop is closed: `LOOP.md` §03 carries its row, the ledger carries the
corrections, everything is pushed. **You are not inheriting an unfinished job.**

**What it settled, so you do not re-open it.** Two dispositions were downgraded to `PARTIAL`
(`TEST-108`, whose named test skips in the very environment its row is about; `GUIDED-238`, whose
named node stays green over a full source revert and was repointed). The agent's **refusal of the
prompt's partition was upheld** and the epistemology endorsed — *can this item ever fail in service*
quantifies over states a producer **could** reach, which is a judgment, not an observation. And the
agent's **refusal of my prescribed sentinel property was upheld**; see §06.1.

**The result to carry forward is `MISC-029`, and it is the best thing in two loops.** Of the manuscript
validator's thirteen checks — the only pass/fail set in the Guided door feeding a count a user sees —
**eight are decided before the manuscript is read**, each frozen one shown to survive a state that
should make it FAIL. The page renders *"13 checks, 0 unmet"* where the honest denominator is **5**, and
asserts beside it *"Every consistency check the validator makes is met by this draft."* That is the
governing rule's *assert-something-false* branch on the artifact that leaves the building. **Four
corrections to that row are mine and are recorded in it.**

### The live queue after L64, in value order

- **`MISC-029`** high, `PARTIAL` — the validator audit above.
- **`MISC-030`** — `_why_not`'s fallback still wrong for nine figure ids.
- **`GUIDED-238`** — 77 checklist items unconverted, and the partition question open.
- **`MISC-028`** — the manuscript reconciliation that cannot fail. Measured, not built. **The prompt's
  own total was off by one and the agent said so: 80, not 79, against 72 outcome rows.**
- **`GUIDED-231`** critical — the inference half was never ported. Large, and blocked on a product
  decision rather than on execution.
- **`GUIDED-232`** / **`GUIDED-233`** — Explain cannot answer its own question; no explainability pack.
  `GUIDED-233`'s own sequencing says the faithfulness harness comes first.
- **`GUIDED-244`**, **`GUIDED-248`**, **`GUIDED-249`**, **`GUIDED-250`**, **`TEST-105`**, **`MISC-023`**.
- **`DRIVE-036`** — needs a genuinely repeated-measures fixture; six drives have never exercised it.

**A drive is due.** L64 shipped two things a person can see — the clinical figures returning for
single-model projects, and the Train refusal carrying its control. **Before it, restart the server so
`/dev/status`'s `rev` equals HEAD**; the one on `:8777` has been stale for a week. A rev mismatch is
what made drive 3 unresolvable and it costs ten seconds to remove.

---

## 05 · The six human drives, and how to intake the seventh

**A human at the screen is the only instrument this project has for `PRODUCT_VISION.md` §06b's third
condition.** `pageharness.py` says so in its own docstring: it proves what the controller renders and
**cannot prove visibility**. Their report is **primary evidence, not a claim awaiting a probe.**

**Sort every observation into: absent · unreachable · reachable-but-unreadable · working-as-designed.**
The third class is the valuable one and has no other home. Check claimed absences against
`register.json` first — **46 rows are `classic-only`**, deliberately not in Guided, each with a reason.

**A tester's wrong diagnosis with a real symptom is still a real finding. The symptom is the evidence;
the cause is yours to establish.** Run 2's cause was a swallowed fetch, not a stale flag. Run 3's was a
server 28 hours stale. Run 4's was the launch, not the venv. **Run 5 is the model**: it reconciled 34
displayed quantities against independent ground truth, flagged its own uncertainty rather than guessing,
and found a `critical` that 2,607 green tests could not — because it chose a target where the event is
the **majority**, and the defective code was accidentally right whenever the event is the minority.
**Your fixtures are mostly the accidentally-right case.**

> **A report from a running app is evidence about the running app, not about the tree.** Before telling
> a human their observation does not reproduce, establish what they were running: the process's start
> time, and **its interpreter** — `lsof`, never `ps`, never `pip list` in your own shell.

---

## 06 · Calibration — my errors this session

**Every loop's divergence section has corrected the adjudicator, eight loops running.** Read it first.
When the agent says it is unsure, it has usually already checked.

1. **I prescribed a mechanism more complex than the problem needed, from a measurement that was
   correct.** The L64 fan-out measured *why* two candidate `classes_` fixes break the nn wrapper. I
   turned that into a prescription for a sentinel-deferring property. The agent built it, verified it,
   removed it as redundant, and was right: a property is a **data descriptor** and intercepts a lookup
   that should have found the instance attribute; `__getattr__` does not intercept it, so the sentinel
   has nothing to do. **The right question was *why does the property break*, not *how do I patch the
   property*.** A measurement of two failures is not a search over mechanisms. Driven and recorded on
   `GUIDED-245`. **This is the most dangerous kind of error I made, because the measurement's authority
   carried into the conclusion drawn from it.**
2. **I asserted a second scored registry that does not exist.** I found 98 `ChecklistItem`s across two
   files and told the fan-out to check whether the second was inflating a compliance count. Its
   `ChecklistItem` is a **different class with no predicate field at all.** The fan-out caught it. The
   instinct — *check the surface nobody counted* — was right and produced `MISC-029`; the assertion was
   not.
3. **I shipped a prompt whose own header count was stale.** L64's header said 983 findings; I filed
   four rows between writing it and committing. The agent noticed. **Write the header last.**
4. **My prose trips the spelling gate almost every time** — five different British spellings across
   three commits this session. The gate catches every one, which is the system working, but it costs a
   commit cycle each time. Write American on the first pass.

   *(And note where this bullet ends up: the first draft of it **listed the five spellings**, and the
   gate refused the commit — the paragraph about tripping the gate tripped the gate. That is the L64
   agent's own find arriving one document over: **writing about a matcher is enough to trip it.** Name
   the class, do not quote the tokens.)*

5. **I read half a measurement and drew the conclusion from that half — and it is error 1 again, a third
   time.** L65's §00.A3 told the agent the module the hook's probe misses is `fastapi`, and that the
   defect is *"latent rather than live — say that rather than dressing it up."* **Both halves false.** The
   set is `{fastapi, sklearn}`; `turbotab/.venv` carries fastapi and lacks sklearn, and `resolve_python`
   selects it whenever `venv/` is absent. **My own reconnaissance had written *"(sklearn absent)"* in the
   same sentence that reported fastapi present.** I read the half that answered the question I had asked
   and never asked what else the gate imports. **The instruction to call it latent is the worst part** —
   I told the agent to state a falsehood plainly, and being wrong in the direction of *understating* the
   defect made it sound like calibration.
6. **My own adjudication agents produced two errors of exactly the kinds this document warns about, and
   only the refuters caught them.** One charged the execution report with miscounting citations, having
   measured **at HEAD, after the loop's own repair rewrote the citation it was counting** — the same
   one-write-late error it had correctly charged against the report in the paragraph above. The other
   declared a test dead because it stayed green under a neuter that **zeroed the population instead of
   falsifying the value**, while that file's own non-emptiness guard was failing in the same run to
   announce the configuration was invalid. **A fan-out does not make the adjudicator right; it makes him
   checkable. Refute your own agents, not only the report's claims.**

**The generalization, which every PM before me also recorded:**

> **Every number you state carries how you got it.** Mark it *(re-derived at `<sha>`)* or *(from the
> row)*, and **doubt the second kind first.** A file's existence is not its contents, a write-up's
> number is not a measurement, a 200 is not a correct consequence, a PID is not a running process, and
> the source tree is not the running app.

**Design is the only work here with no verification loop.** A design proposal ships with a prior-art
check the way a closure ships with a probe.

---

## 07 · The fan-out, which is now the method, and what it actually finds

Four runs: L62's adjudication (11 agents), L63's reconnaissance (12) and adjudication (14), L64's
reconnaissance (8) and adjudication (12). **Every driver in every run has come back `HOLDS_WITH_CORRECTIONS` or worse — no verification
has ever been clean, including mine.** At L64 **two refuters returned `SOUND`** for the first time,
which is the useful shape: the second reader finding nothing is evidence, and it took five fan-outs
to see one.

**What it reliably finds, and a single reader reliably does not:**

- **A fourth surface thirty lines from a fix.** L62's sweep claimed to cover *"every surface that
  reports an n"*; two independent drivers found one more, in the same file.
- **An `act` field that would send the builder to the wrong function.** `GUIDED-236`'s said to split
  `predictions_for`, retracted at L57 and never corrected in the row. `GUIDED-238`'s described a
  compliance count that does not exist.
- **The defect the fix creates.** L63's ROC widening would have published bootstrap intervals that
  depend on tick order; the refuter measured it before it shipped.
- **A verifier's own coverage claim.** At L63 a verifier said a file was "structurally out of reach" of
  the revert detector; its refuter drove it and it was not.

**Two structural facts about running them:**

- **Agent worktrees have been handed out at an unrelated commit with no `turbotab/` directory, three
  recorded times.** Tell every agent to check `git log --oneline -1` first and to recover with
  `git checkout --detach <sha>` or `git archive`. They lose ten minutes each when you forget.
- **Prompts go in as `.join('\n')` arrays, never template literals** — backticks in the prose break the
  script parse, and this prose is full of them.

**Bound the load.** The machine sits beside where Nolan sleeps. Targeted test files only, nothing over
~3 minutes, no full sweeps inside a fan-out, and say the time of day in the brief.

---

## 08 · The agent, and what to protect

**It is exceptional and better than me at several of these.** In eight consecutive loops it has
corrected the adjudicator's premises, and it has **refused a part with a measurement rather than
shipping a weak version** five times. At L64 it did the sharpest version of that yet: it built exactly
what I prescribed, verified it, and then **deleted it and told me why** — and separately **refused to
report a partition at all** rather than add a fourth unreproducible number to the ledger.

**Rule against it when it is wrong and say plainly when it is right. A refusal is a result.**

**What it does not yet catch:** its own coverage claims, and matchers firing on its own prose. At L64 it
tripped two of its own guards by *writing about* the strings they search for — *"writing about a matcher
is enough to trip it"*, which is a variant trap 5b does not warn about. It caught both itself.

---

## 09 · Habits that are load-bearing

- **Verify in an isolated worktree, never the live tree.**
- **Never write the data files while a loop runs.** `ps` and `git status` first.
- **The ledger has exactly one writer and it is `ledger.py`.** `set` **replaces** the note — read the
  existing one and append. **`set` refuses a `FIXED` status without `--test`**, so pass the existing
  value back. **`set` writes `verified_ev` only; `ev` is add-time-only and cannot be corrected through
  the tool** — the L64 agent found that, and it means a stale `ev` on a closed row stays stale.
- File notes **through a Python file, never a shell heredoc**; zsh eats backticks, and it eats an
  unquoted `--include=*.py`.
- **`docs/audit/` is the exempt path** for verbatim drive reports.
- **Never `git add -A`.** Stage explicit paths. Check the diffstat — `ledger.py` serializes at
  `indent=1` and a script that dumps at `indent=2` reformats nine thousand lines.
- **Add the `LOOP.md` §03 row when you accept a loop**, and name in Part A the one gap you found.
- **The check that fires most often:** was a named defect *class* filed, or only its instance?
- **Never accept a moved threshold in the same loop as the change that pressured it** unless you invoke
  §06.2 **in those words** — and verify the guard got **stronger**, not smaller.
- **`Ambiguity is OPEN, never FIXED`** — and per row: *does the fix reach the thing the row's own
  evidence describes?*
- **Push before you might be cleared.** L64's five commits sat unpushed while I ran a 25-minute
  verification. Nothing was lost, and nothing needed to be at risk.

### Standing rules the agents earned — do not re-derive

- **§08.1, two tiers.** A `FIXED` arrives with its **probe output** or it is `PARTIAL`. The revert must
  be **total**. A red quoting a `TypeError`, `ImportError` or signature mismatch is red for the **wrong**
  reason. **And the cheap detector is one command: run the loop's new tests with the loop's own source
  diff reverted.** At L63 that gave 27 failed / 51 passed and settled three dispositions.
- **A part may be REFUSED when carrying it out would violate the criterion the row itself states.**
- **A pin that cannot flip is a comment.** `mark.xfail(strict=True)`, not `pytest.xfail()`.
- **A grep answers "does this text appear", not "does this run".**
- **A matcher that fires on prose has silence that means nothing.**
- **A suite is quotable only if nothing else is writing the tree *or competing for the machine*.**
- **A class goes in the ledger the moment you name it. Say the number, including when it is zero.**
- **No subagent runs a tree-wide git operation.** A subagent gets its own worktree or no write tool.

---

## 10 · What happens next

**L65 is written, run and accepted five of five.** Its §03 row carries the detail; the queue below is
what is left.

1. **Write L66, and Part A is `MISC-033`** — prose asserting the opposite of what L65 shipped, at seven
   sites across four files, **two in production source and one falsified by its own commit**. The
   sharpest instance is a tripwire: `turbotab/test_the_checklist_count_says_what_it_counted.py:49-56`
   writes down the condition under which its own abstraction would be wrong, `:113` meets that condition
   two commits later, and **a green test at `:292` asserts the negation of four of those sentences.**
   Beside it, **`TEST-110`'s class**: its `ev` went stale *inside the loop that repaired it to
   demonstrate the repair*, because the loop's own first commit moved the line — and the sweep is
   structurally blind to it, since a bare `:90` carries no filename token. **A citation with no filename
   is invisible to the instrument built to check citations.** Repaired, but the class is open.
2. **The live queue after L65, in value order.** `GUIDED-238`'s remaining **78** items — but §00.7 of
   L65 is the reason to fix *reach* first: the nine already converted are still unreachable through
   `figure_bundle.SOURCES`. Then `MISC-032` (the Classic download gate blocks a click rather than a
   file; 5 of 13 buttons ungated), `TEST-111` (the spelling gate reads a clipped view — **driven green
   at HEAD while 24 British-stem occurrences sit in `findings.json`**), `TEST-112` (the scratchpad
   conftest hazard, filed against my own method), `MISC-030`, `GUIDED-244/248/249/250`, `MISC-023`.
   **`GUIDED-231`** and **`GUIDED-232`/`233`** remain blocked on a product decision, not on execution.
   **`DRIVE-036`** still needs a genuinely repeated-measures fixture; seven drives have never had one.
3. **THE DRIVE IS OWED AND IT IS NOW TWO LOOPS DEEP.** The product owner chose to run L65 first and
   drive both together. The server on `:8777` was restarted and `/dev/status` reported `rev` equal to
   HEAD at `a759b8b` — **it is stale again the moment anything lands, so re-check before asking.**
   Say which screens are new: **L64's** clinical figures on single-model projects and the Train
   refusal's inline control, and **L65's** manuscript panel header, which now reads *"9 checks, 3 unmet ·
   4 declared"* on an unsealed project and *"11 checks, 0 unmet · 2 declared"* on a fitted one.
4. **Keep adjudicating with a fan-out — and refute your own agents, not only the report.** §07 is the
   argument. At L65 the fan-out caught two adjudication errors that would otherwise have gone into the
   record as rulings against a report that was right; see §06.6. **A fan-out does not make you right, it
   makes you checkable, and only if you point it at yourself too.**
