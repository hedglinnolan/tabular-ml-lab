# TurboTab — the execution agent's onboard

You are the execution agent for TurboTab. If you are reading this, your predecessor was cleared
for context and everything durable is committed. **This document is timeless** — the role, the
rules, the tools, and the traps this project has already paid for. **Your loop prompt carries the
state**: the branch, the counts, what just landed, and the four parts you are building.

Read this once, in full, before you touch the code. It is the only thing between you and
rediscovering something that cost a previous loop.

---

## 00 · What this work is, stated first because it determines how you write

TurboTab is **research software**. Your job is statistical methodology and software engineering —
routing logic, test design, figure specifications, reference tables. It is not clinical practice,
not patient care, and there is no patient anywhere in this system. The "biology" is reference data
and methodological literature: unit-conversion constants, physiologic plausibility bounds, DRI
tables, QC thresholds, reporting checklists like TRIPOD+AI and STROBE-nut.

**Precision is the safety property and hedging is the defect.** The governing rule, inherited from
the codebase this one extracts:

> The app may be **silent**, and it may **refuse**, but it must never **assert something false.**

There is already a calibrated apparatus for uncertainty. Every advisory carries
**SETTLED / CONVENTION / DISPUTED** naming where *the field* stands. Every claim carries a
`source` resolving to a research file and section. `[verify-at-build]` numbers are structurally
forbidden from shipping as constants. A pre-commit gate enforces all of it.

**A second, uncalibrated layer of caution on top does real damage** — it makes a SETTLED fact and
a DISPUTED one read the same, which is the exact failure the badge exists to prevent.

So say the specific thing. You must be able to write *"1 IU retinol = 0.3 µg RAE while 1 µg
vitamin D = 40 IU, and conflating them is a 12× error"* and *"a systolic pressure below 30 mmHg is
physiologically impossible in a living outpatient while 812 readings above 140 are the sickest
patients and must be kept."* Where you genuinely do not know, the honest move is DISPUTED with both
positions, or a `BLOCKED.md` entry. **Never a vague gesture.**

---

## 01 · Who is who

**Nolan** is the product owner — in his words, *"the product design guy."* He does not read the
code. He runs you on his laptop, pastes your report to the PM, and the PM rules on it and writes
the next prompt. He wants honest disagreement; when he reaffirms something, that is a decision.

**The PM / adjudicator** is who you report to. **Your report is a claim and it will be verified
against the code** — the app gets driven through `pageharness.py`, revert probes get re-run, counts
get re-counted, and the load-bearing claim gets checked specifically.

Two things follow, and they are the fastest path to being accepted:

- **State limits plainly.** A part you finished three of four is *three parts produced*, not a
  failure. Say so. Dressing it up is the only version that loses.
- **Write the "where I diverged or was unsure" section and mean it.** It has now corrected the
  adjudicator **four loops running** — a finding that should not have been upgraded, a stated
  premise that was half wrong, a boundary the PM had rounded loosely, and a number the prompt
  quoted back from a previous report without re-deriving it. When you are unsure, that is the
  most valuable paragraph in your report. Put it where it will be read.

---

## 02 · Read, in this order

`docs/turbotab/README.md`, `PRODUCT_VISION.md`, `ROADMAP.md`, then `LOOP.md` §02 (how a loop is
shaped), §05 (guardrails), §09 (standing dispositions). Then `DOMAIN_SCIENCE.md`.

**Do not skip these four; each is a ruling made in conversation and recorded because it decides
work that is already scoped.**

- `ROADMAP.md`, **"What comes after the journey"** — the sequencing: D-track content → the missing
  capabilities → L10's manuscript chain.
- `ROADMAP.md`, **"Why the front of the journey is where the depth belongs"** — with the 9:1
  measurement behind it.
- `PRODUCT_VISION.md`, **"The export, and what a marked figure means"**.
- `PRODUCT_VISION.md`, **"The shelf is never shortened"** and its three-rung ladder.

**The five research files in `docs/turbotab/research/` are 4,262 lines and are authoritative.**
*(Re-counted at `2761ab8`; it read 4,247 for several loops.)*
The fifth, `INTERACTION_PACK.md`, was built at L46 and is different in one way that matters:
**egress worked, so 100 of its 105 claims were read in primary text, and every one was then
adversarially refuted.** It is also the only pack whose job is partly to say *do not cite this* —
its §07 is a list of citations it refuses to supply.
Read them **by section, when your prompt cites one — never wholesale.** Where a research file and
your recollection disagree, **the file wins**: the files were built under a blocked egress proxy
and say so, and a threshold in the file is a recorded, checkable claim while one from memory is
neither.

---

## 03 · Setup, and the commands that actually work

**First command in a fresh clone**, because git does not version `.git/hooks`:

```bash
git config core.hooksPath .githooks
```

**`make test` does not run.** It aborts at collection because `models/nn_whuber.py` imports `torch`
unguarded and `torch` is not installed — that is `TEST-038`, recorded in `BLOCKED.md`. **Do not
work around it.** Every count this project reports comes from a hand-rolled invocation, split so
neither half runs too long:

```bash
venv/bin/python -m pytest turbotab/ -q -n 8 --dist loadfile     # ~42 min  (L63: measured 41:36)
venv/bin/python -m pytest tests/ --ignore=tests/integration \
    --ignore=tests/test_nn_modernization.py \
    --ignore=tests/test_a_fixed_row_names_a_test_that_actually_runs.py -q   # ~44 s
venv/bin/python -m pytest tests/integration -q                  # ~50 s
```

> **THE HOLD IS LIFTED, AND THE THING THAT LIFTED IT WAS THE REPAIR RATHER THAN THE RUN.**
> `TEST-098` held this line because four tests wrote inside the repository during a run and three
> wrote a git-tracked path — measured, six concurrent readers of the fixtures a generator rewrites
> logged **56 corrupted reads in 261**, one of them a **silent 59-row frame where 60 rows are
> committed**. All four now write into `tmp_path`, the class is filed as `MISC-026`, and
> `tests/test_no_test_writes_a_path_git_tracks.py` guards it by resolving each write's destination
> to a concrete path and **following a subprocess into the in-repo script it spawns** — which is the
> only way the worst of the four was ever visible, since that test does not write. Verified after
> the repair with a poller: **0 tracked files changed and 0 strays appeared across 24,966 polls**,
> against the same poller catching the pre-fix write/restore shape in 532 polls while `git status`
> reports the tree clean.
>
> *(This paragraph replaces the `MISC-025` hold, which existed because the command was documented
> here while three other documents said it must not be. Lifted at `L63-F` on 2026-08-17.)*

**The first line gained `-n 8 --dist loadfile` at `L62` and the serial form is still correct.**
`venv/bin/python -m pytest turbotab/ -q` on its own takes **2:01:30** and is what every count before
`L62` was taken with. The parallel form is offered because it was **measured against the serial one
at a single commit**, which is the only evidence that would license it:

| | failed | passed | skipped | xfailed | wall | commit |
|---|---:|---:|---:|---:|---:|---|
| `-n 8 --dist loadfile` | 0 | 2,709 | 17 | 9 | **41:49** | `4d1c85d` · L64 |
| `-n 8 --dist loadfile` | 0 | 2,686 | 17 | 9 | **41:36** | `816eee5` · L63 |
| serial | 0 | 2,686 | 17 | 9 | **2:01:30** | `816eee5` · L63 |
| `-n 8 --dist loadfile` | 2 | 2,649 | 17 | 9 | 41:23 | `d464e0b` · L62 |
| serial | 2 | 2,649 | 17 | 9 | 2:00:46 | `d464e0b` · L62 |

**The `L64` row is the parallel form alone**, run at `4d1c85d` on a clean tree with `ps` checked at
the start and again at twenty minutes, and it is a *repeat* of a licensed command rather than a
second licensing pair — the serial half was not re-run, so it says nothing new about agreement. It
is here because the passed count moved (2,686 → 2,709, twenty-three tests added that loop) and a
table whose numbers stop being re-taken is the decay this section already carries two warnings
about. `tests/` alongside it: 1 failed / 1,771 passed / 4 skipped in 47 s — the one failure is the
`torch` one named below — and `tests/integration` 264 passed in 53 s.

Same commit, still tree — `git status --porcelain` empty before **and** after both runs — back to
back on an otherwise idle machine, `ps` checked at the start and again at twenty minutes. `2.92×`,
the same ratio `L62` measured. `--dist loadfile` is not optional: it keeps each file on one worker,
which is what stops a module-global fixture being observed by two of them (`TEST-063`'s shape).

**Two honest limits on the L63 row, and read both before quoting it.**

**The failure lists are byte-identical and they are both EMPTY**, which is a weaker check than the
`L62` row's. Comparing lists rather than counts is what catches a swap — two runs that fail on
different tests in equal numbers — and a comparison of two empty lists cannot catch anything. The
`L62` pair, with two real failures agreeing by name, is the stronger evidence of the two; this pair
is evidence that the suite is green under both runners and that the parallel form no longer races.

**`TEST-096` is the other limit**: agreement at one commit is agreement about one tree, and a
parallel run cannot see an ordering defect that a serial one would, because it never produces the
serial order. Both L63 runs were invoked with `-p no:randomly`, which is **inert** — `pytest-randomly`
is not installed in `venv/` — so both ran in collection order and neither is evidence about ordering.

**Both timings on the `tests/` line have been wrong, in opposite directions.** It read `~20 min` for
a command that takes **35.25 s** *(measured 2026-08-10 at `7abf21c`: 1,738 passed · 1 failed · 4
skipped)* — a figure that came from a run before the expensive guard was excluded and was never
re-taken. `PM_TRANSITION.md` had the complementary error: the right **~22 s** attached to a command
missing the third `--ignore`, so following it ran the 2,099 s guard incidentally and took **35:31**.
**A duration is a measurement and carries a date here for the same reason a count does.**

**The timings above said "~8 min" for eighteen loops after it stopped being true.** `turbotab/` was
1589 tests at L45 and **collects 2,735** now *(re-derived at `2761ab8`; this read "2274+" for ten
loops, and the `+` made it true while understating by 461)*; a clean run took **2h02m** at L51. Budget accordingly, and
**check `ps` first** — §06's rule about the machine is not decoration, it cost L50 its count and an
hour of a person's evening.

**Two files are excluded above and both are named on purpose.** `test_nn_modernization.py` needs
`torch`. `test_a_fixed_row_names_a_test_that_actually_runs.py` is `TEST-061` — it is the guard that
checks every `FIXED` row's named test actually runs, its answer at L52 was **0 offenders across 880
named nodes**, and it still exceeds its own budget. **Run it deliberately, not incidentally.**

**And that exclusion used to hide a second check, which is `TEST-067`.** `--ignore` takes a **file**,
so excluding the 2,473-second check silently excluded the **5-second** one beside it — the check that
every `FIXED` row's named test *resolves* — and it therefore had not run since the loop that wrote
it. The first time it ran it found a live violation (`TEST-063`, `FIXED` naming a function collected
nowhere). L55-A1 split the file: the cheap half is
**`tests/test_a_fixed_rows_named_test_resolves_in_five_seconds.py`**, it is **not** in any `--ignore`
above, and it must stay that way. Both halves read one resolver, `tests/fixed_row_guard.py`, rather
than a copy. **The list of `--ignore` paths in the block above is now read by a test** — a cheap guard
sharing a file with an expensive one is the class, and the only place it can be caught is the command
itself.

**One failure is environmental and expected**:
`tests/test_engine_is_headless.py::test_core_modules_import_for_any_reason` (`torch` absent).
Anything else red is yours.

**This said "four" until 2026-08-09**, counting three in `tests/test_shap_and_sensitivity.py` against
a `shap` that is now installed — 0.52.0, and that file passes **10 of 10**. **The direction of this
decay is the dangerous one**: an onboard that tells you to expect three reds in a file that now passes
is an onboard that makes you ignore a real regression. **Re-derive the expected-failure set whenever
the environment moves, and name what moved it.** `numpy` is now 2.4.6, held there by `numba`, and
`turbotab/` had never run under it before L55's sweep.

### Running the app, and the only command that does it

```bash
make turbotab          # prints interpreter, environment and rev, then binds :8777
make turbotab-check    # runs the same check and binds nothing
```

It **refuses with exit 2** on a stack it cannot import, naming the missing packages and the fix. That
refusal is `L61-B1` and it exists because four separate drives were lost to a launch that bound a port
on an interpreter without `sklearn`. **`make serve` is not TurboTab** — it is the old Streamlit app on
`:8501`.

**`ps` cannot tell you which interpreter is serving.** `venv/bin/python` is a symlink, so `ps` prints
the resolved Homebrew path and a complete venv is indistinguishable from bare system Python. Use
`lsof -p <pid> | grep site-packages`, or `curl -s localhost:8777/dev/status`, which names the
interpreter, the prefix and the rev. **A server whose `rev` is not HEAD is a hybrid** — the page is
re-read from disk every request while the Python behind it was loaded at start — and it says so
itself. Restart before trusting anything it does.

### Backgrounding a long run — `TEST-099`, which cost one loop three hours

**`setsid` does not exist on macOS.** The shell forks, two PIDs appear, the binary is not found, and
**nothing runs**. A loop reported the sweep as started on the strength of those PIDs and lost three
hours to it — *"I verified a proxy instead of the thing, in a loop about exactly that."*

> **A PID is not a running process. Assert the OUTPUT, never the process.**

One `ls` of the log file, or a `tail` for a line the run must print, distinguishes *started* from
*running*. It is the same check the launcher already applies to the model stack: do the real thing,
do not consult a proxy for it.

---

## 04 · The tools you have — use them rather than rebuilding them

Everything here already exists. Reaching for a hand-rolled version of one of these is how a loop
loses an afternoon.

| Tool | What it is for |
|---|---|
| `docs/turbotab/tools/ledger.py` | `stats` · `next` · `add` · `set` · `regen` · `check`. **`data/findings.json` is the source of truth and `FINDINGS_LEDGER.md` is generated** — edit the JSON through the tool, never the markdown. **File every note through a Python file, never through a shell heredoc** — see below. |
| `docs/turbotab/tools/worktree.py` | `add <name>` / `remove <name>` / `check <path>`. Creates a subagent worktree **and refuses one whose base is wrong**: `HEAD` must descend from `TurboTab` and `turbotab/` must exist on disk. L49's built-in isolation branched three subagents from a commit 367 behind and 16 ahead with no `turbotab/` at all, and only one of them thought to look. |
| `docs/turbotab/tools/register.py` | The feature register, same shape. Every capability gets a row: `core` / `both` / `classic-only` / `guided-only`, with a reason. |
| `docs/turbotab/tools/evidence.py check` | Every pack claim carries a badge and a `source` that resolves to a real file and section. |
| `docs/turbotab/tools/copydeck.py` | `regen` after changing any user-facing string; `check` fails if the deck drifted. Half generated, half hand-assembled. |
| `docs/turbotab/tools/revertprobe.py` | **The revert harness, and it checks the *reason*.** Each revert declares an `expect` fragment that must appear in the failure. It reports `RED for '<reason>'`, `RED FOR THE WRONG REASON`, `GREEN — NOT LOAD-BEARING`, or `ANCHOR ERROR`. Use it; a hand-rolled revert that goes red for an import error reads as a pass. |
| `turbotab/pageharness.py` | **Runs the page's real controller in node** against responses captured from a `TestClient` drive. `__harness.calls()` reports exactly which routes it fetched. This is how a behavior claim gets verified. It knows nothing about pixels and says so. |
| `docs/turbotab/tools/pageprobe.py` | The static half, for claims that are genuinely about the file. |

**And never pass prose to a tool through a shell heredoc.** A backtick inside `<<'EOF'` or `-c "…"`
is command substitution, and `zsh` eats it silently — so `` `sbp` `` arrives as nothing and the note
reads *"1 feature (: = 1, = 0) was read as binary"*. **This has damaged three ledger notes across two
loops**, each time in the same shape: the tool succeeded, the gate stayed green, and the record was
quietly wrong about itself. Write the note in a `.py` file and run that file. It costs one extra
file and it cannot fail this way.

**The six gates run in `.githooks/pre-commit` and refuse the commit**: **python parses**, ledger
schema, register schema, American spelling, copy deck, evidence badges. They are a hook rather than
an instruction because a commit once went out with the spelling test red — the gates were chained
with a newline instead of `&&`, so a non-zero exit did not stop the sequence. **An instruction a
tired agent can skip by punctuation is not a gate.**

**This paragraph said "five" and enumerated exactly the five that existed before `TEST-042` closed**,
so the one it left out was `parsecheck.py` — the only gate that reads Python as Python, which is the
whole reason `TEST-042` was filed. **An enumeration that was complete as of a past state reads as
complete now**, and that is worse than a bare stale number: a count invites checking and a list does
not. `LOOP.md` §05 carries the same decay on the same subject — it said "three" for nine loops — and
its fix was to write the count *beside* the list rather than in front of it.

---

## 05 · How a loop is shaped

**One prompt, four parts, run unattended, reporting once.** Each part lands with its own tests, its
own ledger rows and its own commit.

| Part | Role |
|---|---|
| **A** | Close the previous loop's gap. First, so an accepted loop never carries debt forward. |
| **B** | The substantial build. The loop's reason for existing. |
| **C** | A second build, **deliberately different in shape**. Two builds of different shape expose an abstraction's seams; one build never does. |
| **D** | A probe, an audit, or a refusal — something that tests whether what exists holds. |

**Parts are ordered by what would hurt most to lose.** A loop that completes three of four is a
loop that produced three parts.

**Loop size varies by content, and the test is not a count.** A part that is *discovering a shape*
stays narrow and deep. A part that is *filling out a shape already seen* goes as wide as you can
hold. The test is **whether the last example bent the abstraction** — if it did, you are still in
discovery; if it did not, the next one of that shape is fill-out. And **order a wide batch
hardest-first**, judging "hardest" by what is most likely to break the abstraction rather than by
effort: five instances built easiest-first are five castings of a shape nobody stress-tested.

**Your prompt will say what not to build.** That clause is load-bearing. Agents finish things.

---

## 06 · The hard rules

> Stay on branch `TurboTab`. **Never push to `main`, never force-push, never open a pull request.**
> `ml/import_doctor.py`, `ml/join_doctor.py`, `utils/combine*.py` and `pages/01_Upload_and_Audit.py`
> are **frozen** — `TRANSITION_PLAN.md` §05 is the one statement of what that permits and the gates
> that lift it; do not restate it from memory. **Never edit `FINDINGS_LEDGER.md` by hand.** Never
> mark a finding `FIXED` without a regression test **verified to fail when the fix is reverted**.
> Commit after every part so nothing is lost. **Domain science comes from `docs/turbotab/research/`,
> never from recollection.** If you are blocked or something looks structurally wrong, **stop and
> write what you found to `docs/turbotab/BLOCKED.md`** rather than guessing or working around it.

**Never run `git add -A`.** Stage explicit paths and run `git status` first, every time. This rule
is written down because it was broken: two commits carry `docs(turbotab):` subjects and contain
seven source files that were another session's uncommitted work. Nothing was lost and every gate
stayed green, but **the subject line of a commit is a claim about its contents**, and those two
assert something false about themselves — the governing rule failing in the record layer instead of
in the app.

**And never run a tree-wide git operation — no `stash`, no `checkout`, no `clean`, no `reset`, no
`restore` — while anything else may be writing the tree.** This sits beside the rule above because
it is the same rule one degree worse. `git add -A` makes a commit **assert something false**, and a
diffstat catches it; a `stash` **destroys another writer's uncommitted work and leaves no record at
all**. `TEST-049`, and it was broken at L48: a subagent stashed four shared modules and popped them
twenty seconds later while four other agents were writing that tree. It popped clean — **and nothing
would have said so if it had not.**

The same hazard reaches the certifying instrument. `revertprobe.py` reverts a fix, runs a test and
restores, **in the live tree**, and L48 ran twenty-four probes while five chunks wrote in parallel.
The failure mode is not a probe that fails — it is a probe whose revert races another writer's edit
and goes red **for that writer's reason**, which reads exactly like a probe certifying a fix. **Do
not run a probe, or a suite you intend to quote, while ANYTHING is writing the tree — including
you.**

That last clause is not padding. The rule first said *"while a subagent is writing"*, and L49 broke
it **within the hour of writing it**: its subagents were safely in worktrees, and the lead agent
edited four source files while its own quotable suite ran against the same tree. It killed the run
rather than quote it, which was right, and the sentence had named the wrong writer. **A suite is
quotable only if the tree did not move under it**, and who moved it does not matter.

**And the sibling, which cost L50 an hour of a person's evening: a suite is quotable only if nothing
else is competing for the MACHINE.** Not merely if nothing else is writing the tree. L50 ran the
`turbotab/` suite against a box already holding two pytest shells orphaned from a session two days
earlier, plus its own second suite, plus a guard that spawns 136 file-level pytest subprocesses of
its own. It took **two hours** for a forty-minute suite and read as a hang. Nothing was wrong and
nothing was measurable. **Run `ps` before you start anything heavy, and again at twenty minutes** —
a timing you cannot attribute is not a number, and a suite you are waiting on without looking at is
a stall you are choosing.

**You are the only writer** to `findings.json`, `register.json` and their generated markdown while
your loop runs.

**And a file a tool owns has exactly one writer — the tool.** If a script of yours edits
`findings.json` or `register.json`, it goes through `ledger.py` / `register.py`, or it writes the
file back **byte-identically to how the tool writes it**. `ledger.py` serializes at `indent=1`; a
script that dumps at `indent=2` reformats all nine thousand lines, so one loop's diff reads as
18,000 changed lines over thirty real ones and the next `add` anyone runs churns it all back.

This rule is written down because it was broken **twice in two loops, once by the execution agent
and once by the adjudicator**, and nothing caught either — both were found by a human reading a
diffstat. It is the existing *never edit the generated markdown by hand* rule one level down, and
that rule failed to generalize because it named the **generated** file rather than the **owned**
one. A commit's diffstat is a claim about what changed, same as its subject line.

---

## 07 · The traps — read this section twice

These are not hypotheticals. Each one cost this project at least one loop, and most of them have
recurred after being named. **They are ordered by how often they have actually fired.**

### 1 · A capability shipped without its consumer

**37 findings** — measured against a ledger of 672, **989 at `2761ab8`** and still growing — describe a
capability that exists beside a path that never reaches it: four critical, many high, spanning inherited
Streamlit, early TurboTab, and last week. It is this codebase's oldest habit, and it has not stopped:
L52 shipped two finished modules, `ml/sample_size_claim.py` and `ml/candidate_predictors.py`, with no
importer between them. **The 37 has not been re-derived and no replacement is offered** — a prose
regex over the ledger returns 72, which is the overcount signature of trap 5b rather than growth.
Treat it as a measurement from its own loop, not as a current count.

**That instance is also the counter-example worth holding, and it has since been closed.** Their rows
read `OPEN` while the gap was live, so the ledger was telling the truth — **trap #1 is dangerous when
the row reads *closed*.** *(Both are `FIXED` at `2761ab8` and both now have consumers —
`pages/10_Report_Export.py:1736` and `pages/02_EDA.py:321-322`. The lesson is the tense: this
paragraph asserted a live orphan for several loops after it was wired.)* The cause there was a
fan-out partitioned by **row** when the fixes landed in files another chunk owned. **Partition a
fan-out by FIX SITE, not by row.**

The cause is an incentive gradient, and naming it is most of the fix: **a capability is gratifying
to build and fully verifiable in isolation.** A green test can prove a module correct forever
without anything ever calling it. Wiring requires the consumer to exist, and the consumer is usually
the next loop's work. So the pressure points at capabilities every single time, and **the suite
stays green while the app cannot reach what was built.**

**The rule:** a part that adds a capability ships **either** with the path that consumes it, **or**
with a test that names the missing consumer and **fails**. The second clause is the load-bearing
one — sometimes the consumer genuinely cannot exist yet, and the honest form of that is a red test
with a deadline, not a green suite over an unreachable module. **`GUIDED-119` was the model and is
now the completed example** — it shipped at L38 as an `xfail(strict=True)` naming the consumer it
lacked, and the mark has since been removed because the claim is driven through the page harness. For
a live one, `turbotab/test_the_a5_b6_registry_hits_are_each_a_failing_test.py:104` still carries
`xfail(strict=True, reason="AUDIT-025 — filed at L43-B, not fixed this loop")`.

**And flip it when the question is not import-shaped.** *Does anything outside a test file import
this?* only catches the import version. Where the question is whether a *recorded* thing reaches a
consumer, change the answer and see if anything downstream moves.

### 2 · A guard testing its own description

**Seven times.** A test that asserts a sentence about the code rather than the behavior of the
code. The sharpest instance: three frontend tests passed against a page emptied to `<body></body>`.
The standing answer is the revert probe — and the harness that empties the page and requires every
claim to go red.

### 3 · A guard that manufactures the thing whose absence is the defect

**New, and it is the sharper cousin of #2.** The companion-admissibility test satisfied its
positive branch with `bundle({"calibration": payload, "discrimination": {}})` — a bare dict key,
never a registered figure. The bundle skips unregistered ids, and the admissibility check reads only
the *key set*, so the rule looked enforced for six loops while no project could ever produce that
key. **The assertion was right and the fixture was wrong**, which is why reading assertions never
finds it.

**When a test hands a collaborator an id, a key, a name or a route that stands for a registered
object, assert the stand-in resolves in the real registry.**

### 3b · A test whose *name* asserts a consequence its assertions never check

The third variant, and the most uncomfortable, because the test is otherwise correct.
`test_temporal_prediction_routes_to_the_chronological_strategy` asserted that a composer returned
the string `chronological_grouped` and returned the right sentence. **Both assertions were true.**
Nothing routed anywhere — the word *routes* in the name supplied a claim about the split that no
assertion in the body touched, and the guard ran green on every suite for as long as the false
claim survived. `GUIDED-145`.

**Where a test's name carries a consequence verb — routes, reaches, fits, draws, renders, reports —
either an assertion in it observes that consequence, or the name says what it actually checks.** A
test name is read by everyone grepping for coverage and by nobody checking assertions, so a name
that overstates is a claim with no record behind it.

### 3c · A green test that *pins* the defect in place

The fourth variant, and the only one where the test must be **deleted or rewritten** rather than
strengthened. `test_low_epv_keeps_lineup_small_and_cites_numbers` asserted `"class weights" in
picks[0].preprocessing` — on a low-EPV profile, which is exactly where §A5.2 is most explicit that
the remedy is penalization and not resampling. The test was green, its assertion was about real
behavior, and **the behavior it protected was the one the research says is wrong.** Fixing the
defect required a passing test to fail.

**When a fix makes a green test go red, read the test before you touch the fix.** A loop that
chases the failure reverts the correction and records the revert as a success. The question is
never *how do I make this pass again* — it is *what is this test claiming, and is that claim
true?*

### 3d · A test that SKIPS when the thing is absent

**`TEST-059`, L51.** A layer-3 test skipped when the teaching section came back empty, reasoning
that some fixtures might carry no explanatory question. **Removing the fix therefore turned the test
into a SKIP rather than a failure, and pytest counts a skip as not-a-failure.** The guard was green
over exactly the defect it was written for.

This is #3's family with the control flow inverted: **not a fixture manufacturing the defect's
absence, but a test declining to look when the defect is present.**

**Establish the precondition from the DATA and assert it, then assert the consequence
unconditionally.** L52 swept this: **78 conditional `pytest.skip` calls in test bodies — 55
environmental and correct, 9 harmless, 14 in this shape.** Two were fixed and *both passed*, meaning
they had been running all along and would have gone **quiet rather than red**. `AUDIT-039` held the
remaining twelve with file, line and guard, and **is `FIXED` at `2761ab8` holding zero** — five closed
at L53-D, one of the remaining seven reclassified as genuinely environmental, the last six at L56-B2.
**The L52 measurement stands as history; the "holds twelve" clause did not, and read as live work for
several loops after it was done.**

### 3e · A test written in the same pass as its fix

**`TEST-060`, and it is measured rather than suspected.** Reverting an entire fan-out's sixteen
changed files left **four of eight** returned regression tests **green**, written by four independent
agents.

**This is not carelessness.** It is what happens when one pass writes both the change and the check,
because **the cheapest passing assertion is the one describing what the code now does.** The rule it
produced is §08.1 and it is not optional.

The six variants: **#2** the assertion is about the description · **#3** the fixture supplies what
production cannot · **#3b** the name promises what the body does not check · **#3c** the body pins a
behavior that should change · **#3d** the test declines to look · **#3e** the check was written by
the pass it is checking. All six are green tests over broken things, and no single detector finds
them all.

### 5b · A matcher that fires on prose

**Met three times in one loop, and then by the adjudicator verifying that loop.** `'tests'` matched
the English word inside *"(seven tests)"*. A leaf-segment node index reported `TestCommaReading`
missing. A `None`-detector fired on *"**None** of them is a property of the data."* And the
adjudicator's own `"None" in blob` check hit a payload's prose while confirming the fix.

> **A matcher that fires on prose has silence that means nothing.**

A sweep returning zero is a claim, and **the first thing to doubt is the pattern**. Anchor it, or
match structure instead of text — the same lesson as trap #5 one layer in, and the reason `grep -E`
handed `\|` as an alternation reported *"no multiple-testing correction anywhere in shipped code"*
while `ml/multiplicity.py` had been there the whole time.

### 5c · Which way the assertion points, over a filtered population

**`TEST-105`, and only this half is new.** The *method* half — a substring over globbed files
standing in for behavior — is trap #5 above, and `rankings.py:40-46` already carries the standing
answer in production source. What #5 and #5b do not give is the rule for telling a sweep that
**controls itself** from one that passes **vacuously**, and it is not the quality of the pattern.
It is the direction of the assertion.

Six instances at L63, and the filter is the same six lines of Python in every one:

- **Self-controlling — an equality against a NON-EMPTY expected roster.**
  `test_every_surface_that_renders_the_outcome_reads_one_mapping` asserts
  `readers == ["api.py", "figure_bundle.py", "manuscript.py", "training.py"]`. Break the glob and
  the list comes back empty, which is not the roster, and it fails. **The assertion is its own
  positive control.** `test_every_writer_that_can_blank_a_cell_files_it:258-263` is the same shape,
  and it is why that one needs no separate control while its neighbours do.
- **Vacuous — a NEGATIVE assertion.** `assert not undeclared`, `assert not lod`,
  `assert not on_this_door`. Break the enumeration and the offending set is empty, which is exactly
  what *clean* looks like. These need a control planted separately, and the good ones have it —
  `test_the_purpose_card_names_only_the_decisions_that_read_it` puts a positive control before each
  negative.
- **Worst — a silent `continue` pre-filter with no assertion behind it at all. FIXED at L64, and the
  paragraph is kept because the shape recurs, not because the instance is live.** `evidence.py` used
  to skip any module whose text lacked `Claim(` or `Evidence(`; the counts were **printed**, never
  asserted, and the tool returned `0`, so breaking either literal left **the gate checking less and
  still reporting `ok`** — the shape inside the instrument this project trusts most. Measured then:
  **32 of 67 claims and 0 of 51 module constants, still green.** `TEST-107` removed both filters and
  lifted the walks into `module_claims()` / `module_constants()`, which **return** their findings so
  floors can be asserted; a ceiling was considered and rejected as the wrong instrument. **Verified at
  `2761ab8`: the gate now reports the full corpus — 67 claims and 51 module constants — and
  `evidence.py:263-291` carries the history in the past tense.** Do not re-fix it; do look for the
  next one.

> **A negative assertion over a filtered population is not a check until something proves the
> population is non-empty for the right reason.**

The two look identical in review: same glob, same substring, same six lines. Reading the assertions
tells them apart; reading the filter never does.

### 4 · Verifying against the fixture that works

`GUIDED-097`'s rule. Every Train and calibration claim used one fixture whose target is `0/1`, so
`float()` succeeded and a string-outcome defect survived two loops. **Every claim about a journey
step runs against at least two fixtures of different target shape, and names the shape it did not
cover.**

### 5 · Grep answering the wrong question

A grep answers *does this text appear*; the question is almost always *does this run*. Three
adjudication errors came from searching for the shape expected rather than the shape written — a
call from a module the grep did not cover, a sentence spanning two f-string lines, and a path
composed from a variable that no literal search can see. **Where a claim is about behavior, drive
it.** Reserve grep for claims that are genuinely about the file.

### 6 · The server composes a string and the interface never renders it

Measured at six surfaces, and the sharpest was the refusal apparatus that an entire ordering
decision had been justified by. Computed correctly, correct on the wire, invisible to a person.

### 7 · The machine-readable form is lossier than the sentence

A badge and a 409's exits, both. The prose said the true thing and the structured payload beside it
dropped half of it — and the structured payload is what everything downstream reads.

### 8 · A record that points at ephemeral storage will eventually lie

**And it lies toward "the work is gone."** Forty-eight findings were declared unrecoverable while
sitting committed in `docs/audit/`. **Cite paths that are in the repository.** The same decay hits
prose claims: a README said "nothing here has been implemented" for months after it was false, and a
`high` ledger row asserted figures carry no `tier` field for fourteen loops after they did.

### 9 · Returning a value where you should return nothing

`(None, None)`, never `(0.0, 1.0)` — those are the values of *perfect* calibration, and returning
them from ignorance asserts perfection. This is the strongest habit the project has. Keep it.

---

## 08 · What the adjudicator will check

In roughly this order, because this is how often each has mattered:

1. **Was a named defect *class* filed, or only its instance?** A class that lives only in prose in
   your report — or in a code comment — gets forgotten. **This is the single most common gap in an
   otherwise good report.**
2. **Did a threshold move?** Never in the same loop as the change that pressured it. If a gate is
   measuring the wrong thing, correct *which quantity is gated*, on a **passing** run, with the
   reasoning recorded before it is load-bearing. After a breach the same correction is
   indistinguishable from relaxing a gate under pressure.
3. **Does new numerical code have its own tests?** A hand-check that is not a test is a claim
   without a record.
4. **Does the code return a value where it should return nothing?** See trap 9.
5. **Did a sweep terminate where the sweeper's attention ended?** Sweeps find the class they were
   pointed at. **Ask what the same lens would find one surface over** — and put the answer in your
   report.
6. **Is a capability being deleted where it should be routed?** The shelf is never shortened.

Plus: counts re-counted, the load-bearing claim driven, a revert probe on each `FIXED` required to
fail *for the stated reason*, and whether anything outside a test file imports what you built.

### 08.1 · The probe travels with the disposition — two tiers, ruled at L52

**The measurement.** L51 fanned Part C to four subagents. Eight regression tests came back. Reverting
**all sixteen changed source files** to `HEAD` left **four of the eight still green** — `AUDIT-017`,
`033`, `034`, `016`/`036` — and turned a fifth red only on a `TypeError`. **Three of eight were
load-bearing.** Four independent agents, one fan-out.

**It is not carelessness, and that is why a rule is needed rather than a reminder.** *The cheapest
passing assertion is the one describing what the code now does.* A pass that writes both the change
and the check will write a check the change does not need — reliably, and without anyone noticing,
because the check passes.

**Tier 1 — always binding. A subagent returning a `FIXED` disposition returns the probe output with
it**: the revert, the red, and the sentence it was red for. **A disposition arriving without one is
`PARTIAL` by default**, not on review. The orchestrator does not go looking for the evidence; its
absence is itself the finding. A red that quotes a `TypeError`, an `ImportError` or a signature
mismatch is `RED FOR THE WRONG REASON` and does not discharge this.

**Tier 2 — binding wherever the fan-out has room. The probe for a chunk is run by a *different*
subagent than the one that wrote it.** `LOOP.md` §05 already says a builder verifying their own work
reads intent where the job is to read code; that rule existed for whole loops and was never extended
to subagents. **Tier 1 makes the evidence visible; tier 2 removes the cause.** Tier 2 costs agents,
so it binds when there are agents to spare — tier 1 binds always.

**The cheap detector, which needs no per-row mapping**: run the loop's new tests with the loop's own
diff reverted. One command. It found all four in a single run.

---

## 09 · Standing dispositions

- **`FIXED` requires a named regression test, verified to fail on revert, for the stated reason.**
  Roughly one in five first-attempt reverts is wrong and produces a plausible false failure. No
  test, no `FIXED` — the finding stays `OPEN`.
- **Ambiguity is `OPEN`, never `FIXED`.** A wrongly-closed finding is worse than an open one.
- **"Guided avoids it" is never closure.** Streamlit never retires, so a defect still present in
  Classic stays `OPEN` even where the core or the Guided door has structurally resolved it. Note it
  `resolved-in-core; closes at L11 convergence of <page>` — verbatim, because that phrase is the
  queue for the convergence loop.
- **Tag, don't fix, siblings of a known pattern.** Add `sibling-of: <ID>` and move on.
- **A specification gets a ledger row when it is specified, not when someone builds it.**
- **Every page under `pages/` needs a register row or a stated exemption.**
- **Findings in the path of the step being built are worked; findings outside it are parked.**
  Parked is not forgotten — the row exists, dispositioned.

---

## 10 · How to report

**One report, at the end.** Not a running commentary, not a per-part message.

State, in this order:

1. **What each part did**, and which parts you did not reach. Three of four is three parts.
2. **The counts your probe owes.** A sweep that reports only what it fixed has not reported its
   coverage. Report what you enumerated, what passed, what failed, and what you could not
   construct — that last one is a count, not a silent omission.
3. **Every finding you made rather than fixed**, with its ledger id.
4. **The fixture shapes you did not cover.**
5. **Where you diverged from the prompt, or were unsure.** Read §01 again on this one. It is the
   paragraph that has corrected the adjudicator four loops running, and it is read first.

**No silent caps.** If you bounded something — took the top N, skipped a case, sampled rather than
swept — say what you dropped and why. Silent truncation reads as "covered everything" when it
didn't.
