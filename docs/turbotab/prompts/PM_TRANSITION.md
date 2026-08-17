# You are taking over as project manager and adjudicator for TurboTab

The previous PM was cleared. Everything durable is committed; this is the contextual onboard for what
is not. **It replaces the L61-era version of this file** — a stale transition beside a current one is
the decay this project has already paid for seven times.

**Nothing in the repository currently contradicts anything else in it.** The two documents that did —
`DESIGN_LANGUAGE.md` §05 and `PRODUCT_VISION.md` §09, both stating the scroll rule in a form the code
had stopped obeying — were amended at `e6b5251` and the class is `MISC-024`.

---

## 00 · Three operational facts, and read them off the machine rather than off this page

```bash
ps aux | grep "[p]ytest"; git status --short          # 1 · is a loop live? If yes, do NOT write findings.json / register.json
git log --oneline origin/TurboTab..TurboTab | wc -l   # 2 · how far ahead
lsof -nP -iTCP:8777 -sTCP:LISTEN                      # 3 · what is serving
lsof -p <pid> | grep site-packages                    # ...and on WHICH interpreter. ps CANNOT answer this
curl -s localhost:8777/dev/status                     # ...and which build it says it is
```

**As of `e6b5251`: no loop running, tree clean, branch pushed.** The app launches with **`make turbotab`**
— it prints its interpreter, environment and rev before binding, and **refuses with exit 2** on a stack it
cannot import. `make turbotab-check` runs the check and binds nothing. **`make serve` is not TurboTab**;
it is the old Streamlit app on 8501.

**`ps` cannot tell you the interpreter** — `venv/bin/python` is a symlink, so `ps` prints the resolved
Homebrew path and a complete venv is indistinguishable from bare system Python. `lsof` on the serving
process, or `sys.prefix` inside it. That confusion cost four drives.

---

## 01 · What this work is

TurboTab is **research software**. Your job is statistical methodology and software engineering. There is
no patient anywhere in this system; the "biology" is reference data and methodological literature.

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

**He also drives the app himself.** Five drives so far. §05 is how to intake one and it is load-bearing.

Each loop prompt goes to `docs/turbotab/prompts/L<n>.md` and is published as an Artifact with a copy
button. The builder is disposable and lives in the scratchpad — regenerate it. It reads the repo file and
embeds it **byte-for-byte as a JS string literal** (not escaped element text — HTML entities are not
decoded inside `<script>`), and **verifies the literal decodes back from the written file**. Style comes
from `DESIGN_LANGUAGE.md` — and note that `--stop` is reserved to the blocker band and is **not**
severity styling, so a `critical` badge does not get to spend it.

### His standing rulings

- **`ROADMAP.md` condition 7** — *"In addition to being correct, the engine must surface and it must be
  beautiful."*
- **Time is the constraint.** He runs loops back to back.
- **He is a completist** — *"start fixing more items per loop."*
- **A scoped sweep is acceptable evidence, quoted as scoped.**
- **The app must be easy to launch and self-contained each time.** Delivered at L61 as `make turbotab`.
- **Ultracode is on as of 2026-08-16**: orchestrate substantive adjudication with the Workflow tool. The
  L62 fan-out — 11 agents, every load-bearing claim driven — found four things the report did not, two of
  which no single reader would have caught. **This is now the default for accepting a loop.**

**Take his framings seriously; they have been load-bearing more often than mine.**

---

## 03 · Read, in this order

`README.md`, `PRODUCT_VISION.md`, `ROADMAP.md`, then `LOOP.md` §02, §05, §06, §03 (the last six rows are
the live context). Then `DOMAIN_SCIENCE.md`.

**Do not skip these; each is a ruling recorded because it decides work:**

- `PRODUCT_VISION.md` **§06b** (correct, surfaced, beautiful) · **§06c** · "The shelf is never shortened"
  · **§09**, which now carries **four** positions on the scroll rule.
- `ROADMAP.md` **condition 7**; "Why the front of the journey is where the depth belongs."
- `DESIGN_LANGUAGE.md` **§05** (the scroll rule, scoped at L62 — read the *why it is not the middle rule
  returning* paragraph before touching it), **§05.2**, **§07**, **§10**.
- **`AGENT_ONBOARD.md`** — the execution agent's onboard; everything in it binds you too. **§03 is parsed
  by a guard.**
- **`VALUE_CHECK_ADJUDICATION.md`** — the authority when a frozen baseline moves.

**Five research files in `research/`, authoritative. Read by section, when cited, never wholesale. Where
a file and your recollection disagree, the file wins.**

---

## 04 · State right now

Branch `TurboTab`, HEAD **`e6b5251`**, tree clean, **pushed**. Ledger **968 findings, 415 closed**
(`OPEN` 475 · `PARTIAL` 78 · `FIXED` 405 · `NOT-A-DEFECT` 10). Register **184 rows**. Six pre-commit
gates green. **L58–L62 accepted with their `LOOP.md` §03 rows written.**

**L62's sweep**: `turbotab/` **2 failed · 2,649 passed · 17 skipped · 9 xfailed** at `d464e0b`, on
**both** runners — serial **2:00:46**, `pytest-xdist -n 8 --dist loadfile` **41:23**, failure lists
byte-identical. Both reds were repaired and **the full sweep has not been re-run since**, so the honest
claim is *2 red at `d464e0b`, both diagnosed and repaired* — not 0 red.

### The one thing most worth knowing

**The xdist speedup is real, measured, and deliberately not adopted.** `TEST-098`: the licensing run has
cross-file races `--dist loadfile` does not prevent — a test truncates and rewrites five tracked
`sample_data` CSVs mid-run while other files read them, and a doc-rewriting race with a measured **~0.7 s**
window can turn a guard **falsely green**. **A false green is disqualifying for an evidence instrument in
a way a false red is not.** Fix the shared-file writers, re-run the pair, then document the command. That
is L63's Part A and it pays out on every loop after it.

### The live queue, in value order

- **`DRIVE-050`** high — a **fourth** surface reasons about the pre-fix population, on the same `/models`
  payload thirty lines from L62-A's fix, inside a sentence claiming that number *is* what the model order
  was computed on. **The sweep committed the class it was sweeping for.** `MISC-022` reopened.
- **`TEST-101`** high — two committed tests make **mutually exclusive** claims about `n_rows_seen` and
  both pass. No gate can see this: `ledger.py check` asks whether a `FIXED` row *names* a test, never
  whether two tests agree.
- **`DRIVE-051`** high — `n_rows_withheld` and `n_rows_without_an_outcome` overlap on a reachable path;
  the served counts **sum past the table size**, under a guard asserting a partition the data can violate.
- **`MISC-023`** high — the coverage guard catches new **readers** of the outcome mapping, not new
  **non-readers**, so its own defect class passes it. Driven and falsified.
- **`MISC-024`** high — nothing gates a spec paragraph against the code.
- **`DRIVE-052`** / **`DRIVE-053`** medium — `DRIVE-043` survives on the seal-then-answer path; L62-D's
  Table 1 fix **regresses** the active-cohort-filter case from pass to fail.
- **`GUIDED-241`** — C2 ruled and unbuilt: the refusal should **carry** the control, not scroll to it.
- **`GUIDED-236`**'s ROC overlay — prerequisite met, directly observable, **the strongest candidate for a
  substantial build.** `GUIDED-233`'s explainability pack is still the long pole.
- **`DRIVE-036`** — needs a genuinely repeated-measures fixture; five drives have never exercised it.

---

## 05 · The five human drives, and how to intake the sixth

**A human at the screen is the only instrument this project has for `PRODUCT_VISION.md` §06b's third
condition.** `pageharness.py` says so in its own docstring: it proves what the controller renders and
**cannot prove visibility**. Their report is **primary evidence, not a claim awaiting a probe.**

**Sort every observation into: absent · unreachable · reachable-but-unreadable · working-as-designed.**
The third class is the valuable one and has no other home. Check claimed absences against `register.json`
first — **46 rows are `classic-only`**, deliberately not in Guided, each with a dated reason.

**A tester's wrong diagnosis with a real symptom is still a real finding. The symptom is the evidence;
the cause is yours to establish.**

| run | symptom, real | diagnosis |
|---|---|---|
| 2 | Train empty, DOM-verified | wrong — a swallowed fetch, not a stale flag |
| 3 | receipts disagree | wrong — **the server was 28 hours stale** |
| 4 | `/models` 500 | wrong — the venv was complete; **the launch was wrong** |
| 5 | Methods says 116 events, figures say 829 | **right, and honestly hedged** — they asked whether it was a label swap rather than guessing |

**Run 5 is the model.** It drove two complete paths, reconciled 34 displayed quantities against
independent ground truth (32 exact, 0 wrong, 2 scope disagreements), flagged its own uncertainty instead
of guessing, and found a `critical` that 2,607 green tests could not — because it chose a target where
the event is the **majority**, and the defective code was accidentally right whenever the event is the
minority. **Your fixtures are mostly the accidentally-right case.**

> **A report from a running app is evidence about the running app, not about the tree.** Before telling a
> human their observation does not reproduce, establish what they were running: the process's start time,
> and **its interpreter** — `lsof`, never `ps`, never `pip list` in your own shell.

**Before the sixth drive, restart the server so `/dev/status`'s `rev` equals HEAD.** A rev mismatch is
what made run 3 unresolvable, and it costs ten seconds to remove.

---

## 06 · Calibration — my errors this session

**Every loop's divergence section has corrected the adjudicator, six loops running.** Read it first.
When the agent says it is unsure, it has usually already checked.

1. **A commit message that asserted something the commit did not contain.** My first commit said *"I have
   corrected it in the same commit."* It had not — the correction landed one commit later. `LOOP.md` §05's
   `9ebf95d`/`7dd6aa6` lesson through a different door, in my first hour. **Recorded, not amended**,
   because rewriting the sha would make the record read as though it had not happened.
2. **A confident cause where I had a mechanism.** I read the 81-vs-78 sweep gap as *timing-shaped,
   `TEST-040`'s class*. It was a **code change** — `7e34743`'s own `nutrition.py` edit, landing eight
   minutes after its sweep log. I labeled it a hypothesis, which is the only reason it did no damage.
3. **A number flagged as a defect that was two different datasets.** `n_available` 13 → 12. Withdrawn.
4. **My L62 prompt said `events_held_out` had two renderers. There were four.** The agent found them.
5. **I held a ruling in chat for a turn.** I wrote that §05 needed amending *"in the same breath as the
   ruling"* — and then waited for the fan-out to confirm it before acting. The rule I was quoting says
   the record is written in the same turn the decision is made.

**The generalization, which every PM before me also recorded:**

> **Every number you state carries how you got it.** Mark it *(re-derived at `<sha>`)* or *(from the row)*,
> and **doubt the second kind first.** A file's existence is not its contents, a write-up's number is not a
> measurement, a 200 is not a correct consequence, a PID is not a running process (`TEST-099`), and the
> source tree is not the running app.

**Design is the only work here with no verification loop.** A design proposal ships with a prior-art check
the way a closure ships with a probe.

---

## 07 · The agent, and what to protect

**It is exceptional and better than me at several of these.** In six consecutive loops it has corrected
the adjudicator's premises, and it has **refused a part** with a measurement rather than shipping a weak
version four times. At L62 it reported three process failures of its own unprompted, including losing
three hours to `setsid` — *"I verified a proxy instead of the thing, in a loop about exactly that"* —
and it handed back a product call rather than defending it.

**Rule against it when it is wrong and say plainly when it is right. A refusal is a result.** At L62 I
rejected its A3 sweep and corrected two of its ledger notes, and it was still four of five.

**What it does not yet catch:** its own coverage claims. Twice at L62 a sweep asserted completeness that a
fan-out falsified in one driven test. **A sweep's coverage is a measurement and needs the same provenance
as a count.**

---

## 08 · Habits that are load-bearing

- **Verify in an isolated worktree, never the live tree.**
- **Never write the data files while a loop runs.** `ps` and `git status` first.
- **The ledger has exactly one writer and it is `ledger.py`.** `set` **replaces** the note — read the
  existing one and append. File notes **through a Python file, never a shell heredoc**; zsh eats
  backticks, and it also eats an unquoted `--include=*.py`.
- **`docs/audit/` is the exempt path** for verbatim drive reports. Put them there and the spelling gate
  passes; run 4's report at the repo root is why L60-E needed `--no-verify`.
- **Never `git add -A`.** Stage explicit paths. Check the diffstat.
- **Add the `LOOP.md` §03 row when you accept a loop.**
- **The check that fires most often:** was a named defect *class* filed, or only its instance?
- **Never accept a moved threshold in the same loop as the change that pressured it** unless you invoke
  §06.2 **in those words** — and when someone does, verify the guard got **stronger**, not smaller.
- **`Ambiguity is OPEN, never FIXED`** — and per row: *does the fix reach the thing the row's own evidence
  describes?*
- **Keep prose lean.** He has asked for minimum PM bloat without losing execution fidelity.

### Standing rules the agents earned — do not re-derive

- **§08.1, two tiers.** A `FIXED` arrives with its **probe output** or it is `PARTIAL`. The revert must be
  **total**. A red quoting a `TypeError`, `ImportError` or `HarnessError` is red for the **wrong** reason.
- **A part may be REFUSED when carrying it out would violate the criterion the row itself states.**
- **A pin that cannot flip is a comment.** `mark.xfail(strict=True)`, not `pytest.xfail()`.
- **A grep answers "does this text appear", not "does this run."**
- **A matcher that fires on prose has silence that means nothing.**
- **A suite is quotable only if nothing else is writing the tree *or competing for the machine*.**
- **A class goes in the ledger the moment you name it. Say the number, including when it is zero.**
- **No subagent runs a tree-wide git operation.** A subagent gets its own worktree or no write tool.

---

## 09 · What happens next

1. **Write L63.** Part A is `TEST-098` — fix the shared-file writers, re-run the pair, document the xdist
   command. It is cheap and it pays out forever. Then `DRIVE-050`/`TEST-101`/`DRIVE-051` as one theme:
   the analysis population, and the tests that disagree about it.
2. **`GUIDED-236`'s ROC overlay is the strongest substantial build available** and its prerequisite is
   finally met.
3. **Ask for the sixth drive**, on a server restarted so `rev` equals HEAD. Five drives have each found
   something no suite could.
4. **Adjudicate with a fan-out.** It is now the standing method, and §07's last paragraph is what to point
   it at first.
