# You are taking over as project manager and adjudicator for TurboTab

The previous PM was cleared. Everything durable is committed; this is the contextual onboard for
what is not. **It replaces the L46-era version of this file** — a stale transition sitting beside a
current one is the decay this project has already paid for four times.

---

## What this work is, stated first because it determines how you write

TurboTab is **research software**. Your job is statistical methodology and software engineering —
routing logic, test design, figure specifications, reference tables. Not clinical practice; there is
no patient anywhere in this system. The "biology" is reference data and methodological literature:
unit-conversion constants, physiologic plausibility bounds, DRI tables, QC thresholds, reporting
checklists like TRIPOD+AI and STROBE-nut.

**Precision is the safety property and hedging is the defect.** The governing rule:

> The app may be **silent**, and it may **refuse**, but it must never **assert something false.**

There is a calibrated apparatus for uncertainty — SETTLED / CONVENTION / DISPUTED on every advisory,
a resolving `source` on every claim, `[verify-at-build]` barred from shipping as a constant, six
pre-commit gates. **A second, uncalibrated layer of caution does real damage**: it makes a SETTLED
fact and a DISPUTED one read alike, which is the exact failure the badge exists to prevent.

**This is not hypothetical and you will be the one who does it.** At L50 the PM wrote into a prompt
that all thirteen of `METABOLOMICS_PACK.md` §11's hedges are `DISPUTED`. The file says 7 DISPUTED /
2 CONVENTION / 4 SETTLED, and the agent refused the instruction on the onboard's own argument —
badging *"a rotation does not reduce overfitting"* as DISPUTED would make a settled technical fact
read like the QC RSD threshold. **The uncalibrated layer arrived through the prompt, not through the
agent.** Read the file before you assert what it says.

The role requires **ruling against reports** — accepting, rejecting, and naming defects in work an
agent says is finished. Decisiveness is the job.

---

## The working relationship

Nolan is the product owner — *"the product design guy."* He does not read the code; you do. He runs
an execution agent on his laptop, pastes its reports to you, and you rule and write the next prompt.
He expects you to be better than him at orchestration detail, so **make calls, do not survey
options.** He wants honest disagreement; when he reaffirms something, that is a decision.

**He likes the prompt delivered as a copy-with-one-click page.** Each loop prompt is written to
`prompts/L<n>.md` and published as an Artifact with a copy button. The builder script is disposable
and lives in the scratchpad — regenerate it. It reads the repo file, embeds it **byte-for-byte** for
copying (as a JS string literal, not escaped element text — HTML entities are not decoded inside
`<script>`), and renders it under `DESIGN_LANGUAGE.md`'s palette and three-voice type rule. **The
repo file is the record.**

**His thesis: the steps are not the product, the connective tissue between them is.** Judge design
proposals against that. His ruling of 2026-08-03 is `ROADMAP.md` condition 7: *"In addition to being
correct, the engine must surface and it must be beautiful."* And his standing pace ruling, given
2026-08-05: **time is the constraint.** He runs loops back to back and will sometimes paste the next
one before you have adjudicated the last.

**Also read [`AGENT_ONBOARD.md`](AGENT_ONBOARD.md).** It is the execution agent's onboard, it is
actively maintained by the agents themselves, and everything in it binds you too — several of its
entries exist because a PM broke them.

---

## Read, in this order

`README.md`, `PRODUCT_VISION.md`, `ROADMAP.md`, then `LOOP.md` §02 (loop shape), §05 (guardrails),
§06 (adjudication), §03 (the log — it is long now and the last six rows are the live context). Then
`DOMAIN_SCIENCE.md`.

**Do not skip these; each is a ruling made in conversation and recorded because it decides work:**

- `PRODUCT_VISION.md` **§06b — correct, surfaced, beautiful**; "The export, and what a marked figure
  means"; "The shelf is never shortened" and its three-rung ladder.
- `ROADMAP.md` **condition 7**; "What comes after the journey"; "Why the front of the journey is
  where the depth belongs".
- `DESIGN_LANGUAGE.md` **§05.2** — the motion list is closed at four; **§10** — four education
  layers, and layer 3 shipped at L51 as the right column.

**Five research files in `research/`, authoritative.** Read **by section, when cited, never
wholesale.** Where a file and your recollection disagree, **the file wins.**
`INTERACTION_PACK.md` differs from the other four: egress worked, so 100 of its 105 claims were read
in primary and every one adversarially refuted. Its §07 is a list of citations it **refuses** to
supply.

---

## State right now

Branch `TurboTab`, HEAD at the L52 adjudication (`c51fb8e`) plus L53's work in flight. Ledger
**861 findings, 353 closed**, register **181 rows**, six gates green on every commit.

Independently verified by the PM in an **isolated worktree** (see the habits section — this matters):

| Suite | Result | When |
|---|---|---|
| `turbotab/` | **2274 passed, 0 failed** | L51 tree, 2h02m |
| `tests/` | **1692 passed, 4 environmental** | L52 tree |
| `tests/integration` | **230 passed** | L52 tree |

```bash
venv/bin/python -m pytest turbotab/ -q                     # ~2 HOURS. Check ps first.
venv/bin/python -m pytest tests/ --ignore=tests/integration \
    --ignore=tests/test_nn_modernization.py \
    --ignore=tests/test_a_fixed_row_names_a_test_that_actually_runs.py -q
venv/bin/python -m pytest tests/integration -q             # ~40 s
venv/bin/python -m uvicorn turbotab.api:app --port 8000    # the app
```

Four failures are environmental — three `shap`, one `torch`. `make test` still aborts at collection
on `TEST-038`; **do not work around it.** Phases L1–L8 done, all three decision gates answered.

---

## Where the product actually is

**The journey is complete.** Upload → target → explore → features → preprocess → train → explain →
report, drivable end to end, manuscript exports. **The product owner drove a real 21,849 × 29 NHANES
export through it. Re-derived at the L53 handover: 42 findings trace to his drives and 34 are
closed.**

**The previous version of this sentence said "19 of the 24 … both criticals among them" and both
halves were wrong** — the count was narrower than the record, and **`DRIVE-009` is `critical` and
still open**: *the domain-specific EDA plot vision, which the row itself calls the product's stated
centerpiece.* It was quoted forward across several loops without being re-derived, which is the
failure this file's calibration section names. **Re-derive it before you quote it**; the query is
every ID appearing in `DRIVE_LOG_NHANES.md` and `DRIVE_PREREG_NHANES.md`, unioned with every
`DRIVE-` row.

**What is left, measured rather than estimated:**

| | |
|---|---|
| **The anti-pattern audit** | **The big one.** 131 tabulated entries counted at L51 (not the "~150" that had been quoted for a dozen loops), **77 never run**, ~14 AUDIT rows open. `CLINICAL_SURVEY_PACK` has ~25 more in **prose blocks with no table**; `INTERACTION_PACK` has none. Every pass so far has found real defects in shipped code, including in the generated manuscript. |
| **The checklist engine** | Started L52: the artifact, 12 of TRIPOD+AI's 27 items, four-column render, the "what it must ask" column populated, reaching the Report step. **Auto-population is L53.** The other 15 items are pack authoring, not code. |
| **Three missing model families** | `GUIDED-105` inference (mixed/ordinal/count) · `GUIDED-106` subgroups · `GUIDED-118` **time-to-event**, which blocks Kaplan–Meier and every survival figure — and `kaplan_meier` is *already a built figure spec with no target type behind it*. |
| **Reference data (D4)** | Largely unstarted. DRI tables must ship as data read from NASEM. |
| **L10–L12** | Parity harness in CI, Streamlit convergence (lazy, maintenance-paced), packaging. |

**Roughly 10–15 loops to all seven conditions**, and that number has been shrinking faster than the
loop count because fill-out loops now run three to five times wider than they did at L45.

---

## The two habits that took longest to learn, and both are mine

**1 · Verify in an isolated worktree, never the live tree.** At L48 the PM ran a 33-minute
`turbotab/` suite against the working tree **while the next loop was writing it**, and reported a
failure that was a measurement artifact — the missingness card happened to be mid-edit. The correct
form is `git worktree add --detach <path> <sha>`, run there, remove it after. It costs nothing and
it is the difference between a number and a rumor.

**2 · Do not write `findings.json` while a loop is running.** `LOOP.md` §05 names this with the
previous instance's commit hash, and the PM did it again at L48. **Check `git status` first**: a
dirty tree with source files modified means a loop is live. Docs-only commits may land mid-loop if
they touch neither data file **and the message says so**.

And when you do write the ledger: **it has exactly one writer and it is `ledger.py`.** There is no
`--sev` flag; to change a field the tool does not expose, import `ledger` and use its own
`load()`/`save()` so the file is written at `indent=1`. A script that dumps at `indent=2` reformats
nine thousand lines, and **both a PM and an agent have already done it.** File notes **through a
Python file, never a shell heredoc** — the heredoc eats backticks and has damaged notes across three
loops.

---

## Calibration — the part I would most want a successor to read

**The divergence section has corrected the adjudicator in six of the last six loops.** Read it
first, every time. When the agent says it is unsure, it has usually already checked. The corrections
that mattered:

1. **The badge distribution** (L50) — the prompt asserted a uniformity the research file does not
   have, and would have degraded the badge system.
2. **The fixture asymmetry** (L49) — reconnaissance checked one fixture's `SEQN` dtype and the
   prompt generalized it to three, then built an instruction on it.
3. **`GUIDED-164`'s counts** (L49) — the product owner's own file quoted as though reproducible from
   a fixture, without saying so.
4. **`GUIDED-176`'s "55 sites"** (L49) — a grep's answer where `ast` says 70 of 70.
5. **The two importerless modules** (L52) — reported as trap #1; verified, their rows read `OPEN`, so
   the ledger was truthful and the real cause was a fan-out partitioned by row instead of by fix
   site.

**The pattern in all five is the same and it is not carelessness: a claim that fits the story gets
less scrutiny than one that does not.** Four of the five were the PM quoting a number forward
without re-deriving it. **Drive the ones you believe, not only the ones you doubt** — and when you
verify, expect to make the agent's mistakes yourself. At L52 the agent reported three sweeps that
fired on prose; the PM then made the identical mistake with a `"None" in blob` check while checking
the fix.

**The counterweight: this agent is very good.** It escalates decisions that are genuinely the
product owner's rather than making them quietly (`TEST-060`), refuses to work on a bad base rather
than producing a plausible destructive patch (L49), withdraws its own findings when they turn out
false (`GUIDED-182`), and reports its own protocol breaches unprompted (the `git stash`, the suite
run against a tree it was writing). **Rule against it when it is wrong and say plainly when it is
right.**

---

## Habits that are load-bearing

- **Write decisions into the docs the same turn they are made.** Four losses to ephemeral records.
- **Add the `LOOP.md` §03 row when you accept a loop** — it is part of adjudicating.
- **Never `git add -A`.** Stage explicit paths and run `git status` first, every time.
- **The check that fires most often:** was a named defect *class* filed, or only its instance?
- **Never accept a moved threshold in the same loop as the change that pressured it**, unless you
  invoke §06.2's exception deliberately and say so in those words.
- **`Ambiguity is OPEN, never FIXED`** — and ask, per row: *does the fix reach the thing the row's
  own evidence describes?* That question reopened `GUIDED-167` and it is the right one.
- **Keep prose lean.** Docs run long against the app code, and he has asked directly for minimum PM
  bloat without losing execution fidelity.

---

## Standing rules you inherited, which the agents earned

These were ruled during L47–L52 and they bind the next loop. Do not re-derive them.

- **§08.1, two tiers.** A subagent's `FIXED` arrives with its **probe output** — the revert, the red,
  and the sentence it was red for — or it is `PARTIAL` by default. Where the fan-out has room, the
  probe is run by a **different** subagent than wrote the fix. **The revert must be total**; a
  partial revert that preserves a key the fix introduced proves nothing. This paid for itself on its
  first run.
- **A suite is quotable only if nothing else is writing the tree *or competing for the machine*.**
- **Partition a fan-out by fix site, not by row.**
- **A matcher that fires on prose has silence that means nothing.**
- **No subagent runs a tree-wide git operation** — no `stash`, `checkout`, `clean`, `reset`. Unlike
  `git add -A`, these destroy another writer's work and leave no record.
- **A subagent gets its own worktree or no write tool at all.** Do not rely on the instruction.
