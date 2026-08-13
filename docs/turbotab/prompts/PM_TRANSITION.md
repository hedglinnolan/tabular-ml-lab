# You are taking over as project manager and adjudicator for TurboTab

The previous PM was cleared. Everything durable is committed; this is the contextual onboard for what
is not. **It replaces the L57-era version of this file** — a stale transition sitting beside a current
one is the decay this project has already paid for six times.

**Nothing in the repository currently contradicts anything else in it**, with one exception that is
named in §04 and is deliberate: `DRIVE-032`'s `item` text is stale because `ledger.py set` has no
`--item` (`TEST-082`), and the correction lives in its note.

---

## 00 · Three operational facts, before anything else

**1 · A LOOP IS LIVE.** L61 is running as this is written — 2 `pytest` processes. **Do not write
`docs/turbotab/data/findings.json` or `register.json` until it lands.** Check `git status` and `ps`
first, every time. *I broke this rule this session* — I filed rows while L60's sweep was running.
Nothing was lost, I checked every row, but it was luck, and the recovery was to reset to the agent's
HEAD and reapply. Docs-only commits may land mid-loop **if** they touch neither data file and the
message says so.

**2 · THE BRANCH IS 9 COMMITS AHEAD AND CANNOT BE PUSHED.** `TEST-086` fails the pre-push gate — a
frozen-baseline drift that is **diagnosed and deliberately unruled**. It is L61's Part C. Do not
`--no-verify` past it; §04 says why not, and the analysis you need is already in the row and in
`VALUE_CHECK_ADJUDICATION.md`.

**3 · THE SERVER ON `:8777` IS RUNNING THE WRONG INTERPRETER.** PID 48564 loads
`turbotab/.venv`, which has `fastapi` and `pandas` and **no `sklearn`, `xgboost`, `lightgbm` or
`shap`**. Every `/models` request it answers is a 500. **The correct launch is:**

```bash
venv/bin/python -m uvicorn turbotab.api:app --port 8777
```

Proven end to end at `cf7129c`: same code, same page, `GET /models` → **200, `n_available` 13**.
`make serve` runs the **old Streamlit app** on 8501 — it is not TurboTab and must not be repointed
without a decision. **L61 Part B builds the real launch command**; until it lands, that line is the
whole of it.

---

## 01 · What this work is

TurboTab is **research software**. Your job is statistical methodology and software engineering —
routing logic, test design, figure specifications, reference tables. There is no patient anywhere in
this system; the "biology" is reference data and methodological literature.

**Precision is the safety property and hedging is the defect.** The governing rule:

> The app may be **silent**, and it may **refuse**, but it must never **assert something false.**

There is a calibrated apparatus for uncertainty — SETTLED / CONVENTION / DISPUTED, a resolving
`source` on every claim, six pre-commit gates. **A second, uncalibrated layer of caution does real
damage**: it makes a SETTLED fact and a DISPUTED one read alike.

The role requires **ruling against reports** — accepting, rejecting, and naming defects in work an
agent says is finished. **Decisiveness is the job.** So is ruling against yourself in public; §06 is
the most useful section in this document and it is a list of my own errors.

---

## 02 · The working relationship

Nolan is the product owner — *"the product design guy."* He does not read the code; you do. He runs
an execution agent on his laptop, pastes its reports to you, and you rule and write the next prompt.
**He expects you to be better than him at orchestration detail, so make calls, do not survey
options.** When he reaffirms something, that is a decision.

**He also drives the app himself** and sends the driver's report. Four drives so far. §05 is how to
intake them and it is load-bearing.

**Each loop prompt is written to `docs/turbotab/prompts/L<n>.md` and published as an Artifact with a
copy button.** The builder script is disposable and lives in the scratchpad — regenerate it. It reads
the repo file, embeds it **byte-for-byte** as a JS string literal (not escaped element text — HTML
entities are not decoded inside `<script>`), and renders under `DESIGN_LANGUAGE.md`'s palette and
three-voice type rule. **Verify the embedded literal decodes back to the file byte-for-byte.**

### His standing rulings

- **`ROADMAP.md` condition 7** — *"In addition to being correct, the engine must surface and it must
  be beautiful."*
- **Time is the constraint.** He runs loops back to back. Full `turbotab/` sweeps *"run over two hours
  and occasionally time out."*
- **He is a completist** — *"if we have items on the backlog, it's time to run full test suites less
  frequently and start fixing more items per loop."*
- **A scoped sweep is acceptable evidence, quoted as scoped** — approved 2026-08-10.
- **NEW, and it is now a standing requirement: the app must be *easy to launch and self-contained each
  time*.** Reiterated 2026-08-13 after the three-venv finding. It is L61 Part B.

**Take his framings seriously; they have been load-bearing more often than mine.** This session he
asked *"is it possible the agent was not running on the right version of the app?"* — and that
question cracked open three drives' worth of findings that had been filed against code the server was
not running.

---

## 03 · Read, in this order

`README.md`, `PRODUCT_VISION.md`, `ROADMAP.md`, then `LOOP.md` §02 (loop shape), §05 (guardrails),
§06 (adjudication), §03 (the log — the last six rows are the live context). Then `DOMAIN_SCIENCE.md`.

**Do not skip these; each is a ruling made in conversation and recorded because it decides work:**

- `PRODUCT_VISION.md` **§06b — correct, surfaced, beautiful**; **§06c — explainability under the
  lens**; "The shelf is never shortened" and its three-rung ladder.
- `ROADMAP.md` **condition 7**; "Why the front of the journey is where the depth belongs."
- `DESIGN_LANGUAGE.md` **§05.2** (motion preserves identity), **§07** (in-app/journal duality),
  **§10** (four education layers).
- **`AGENT_ONBOARD.md`** — the execution agent's onboard, actively maintained by the agents, and
  everything in it binds you too. **§03 is parsed by a guard**, so the documented `--ignore` list and
  the check cannot drift. **The fast tier is `~35 s` with three `--ignore` paths, not `~20 min`**
  (`TEST-078` — both figures were wrong, in opposite directions).
- **`VALUE_CHECK_ADJUDICATION.md`** — the authority when a frozen baseline moves. Read
  §"The denominator moved" before ruling `TEST-086`.

**Five research files in `research/`, authoritative.** Read **by section, when cited, never
wholesale.** Where a file and your recollection disagree, **the file wins.**

---

## 04 · State right now

Branch `TurboTab`, HEAD **`976c881`**, tree clean, **9 commits ahead of `origin/TurboTab` and blocked
from pushing.** Ledger **934 findings, 394 closed** (`OPEN` 466 · `PARTIAL` 74 · `FIXED` 387 ·
`NOT-A-DEFECT` 7). Register **183 rows**. Six pre-commit gates green.

**L58, L59 and L60 are accepted with their `LOOP.md` §03 rows written. L61 is running and has no row
yet.** Write it when the loop closes, not before.

### The last sweep, and why it is not green

`turbotab/` at the L60 head: **81 failed · 2,486 passed · 17 skipped · 9 xfailed in 2:00:18**
*(from L60-E)*. **73 of the 81 quote L60-A's new refusal** — fixtures that fit a classification with
the event unchosen, which is exactly what `DRIVE-032` removed. **They were passing by virtue of the
defect.** L61 Part A updates them; I ruled that rather than leaving it open, and refused the
alternative on the agent's own measurement.

**The timing is not comparable.** The product owner drove the app while that sweep ran. The agent said
so before I did. **Results stayed valid; duration did not.** Last clean timing is L56's **1:59:54**.

### The two things holding the branch

- **`TEST-086`** — Classic's recorded coverage falls 1.0 → 0.5 on `wide-assay` and `leaky-sepsis`
  because L60-A made choosing the event a required decision and Classic does not ask it. **Classic did
  not change; the denominator did.** The identical movement is already ruled for two other datasets.
  **Two mechanisms**: `wide-assay` runs through `ADJUDICATED_DELTAS`, `leaky-sepsis` has no deltas
  table and compares against its own baseline. **I did not rule it because extending an enumerated
  allowance in the same loop as the change that pressured it is `LOOP.md` §06.2, and invoking that
  exception at the tail of a session to make a push succeed is how a frozen baseline stops being
  frozen.** Say the words when you invoke it.
- **`DRIVE-041`** — the 73. Ruled (a): update the fixtures. L61 Part A.

### One known internal inconsistency, deliberate

`DRIVE-032`'s `item` still reads as the narrower finding ("the question is never asked"). The true
shape — **it fires on the detector rather than the target, and when raised it was not binding** — is
in its note. `ledger.py set` has no `--item` (`TEST-082`), so this is the only way to record it.

---

## 05 · The four human drives, and how to intake the fifth

**A human at the screen is the only instrument this project has for `PRODUCT_VISION.md` §06b's third
condition.** `pageharness.py` says so in its own docstring: it proves what the controller renders and
**cannot prove visibility**. Their report is **primary evidence, not a claim awaiting a probe**, and
§08.1's probe rule does not apply to it.

**Sort every observation into: absent · unreachable · reachable-but-unreadable · working-as-designed.**
The third class is the valuable one and has no other home. Check claimed absences against
`register.json` first — **46 rows are `classic-only`**, deliberately not in Guided, each with a dated
reason.

**A tester's wrong diagnosis with a real symptom is still a real finding. The symptom is the evidence;
the cause is yours to establish.** Every drive so far has proved this:

| run | symptom, real | diagnosis, wrong |
|---|---|---|
| 2 | Train empty, DOM-verified | "gated on a stale flag" — it was a swallowed fetch |
| 3 | receipts disagree | quoted a clause I could not find — **the server was 28 hours stale** |
| 4 | `/models` 500 | "reinstall requirements" — **the venv was complete; the launch was wrong** |

**And the rule those three earned, which is new and is mine to hand you:**

> **A report from a running app is evidence about the running app, not about the tree.** Before you
> tell a human their observation does not reproduce, establish what they were running: the process's
> start time against source mtimes, **and its interpreter** — `lsof -p <pid> | grep site-packages`,
> not `pip list` in your own shell.

I got this wrong twice in one session, in both directions. `/dev/status` now reports the build
(`TEST-084`); **`TEST-087` adds the interpreter and is unbuilt.**

---

## 06 · Calibration — read this before you assert anything

**Every loop's divergence section has corrected the adjudicator.** Read it first, every time. When the
agent says it is unsure, it has usually already checked.

**My errors this session, and they share one shape: I accepted a proxy for the thing.**

1. **A status code for a consequence.** `DRIVE-024`. I wrote that a grain answer's group column was
   *"an enhancement, not a blocker"* because `set_grain people_repeat` returns **200**. It does — and
   the app then promised *"whole people will be held out"* while the seal drew **by row** and said so
   on the same page. **I told them to ship a door that lied.**
2. **A resting state for a transition.** I ruled that L58's *"13 model controls after the seal"* was
   measured into an already-sealed project and therefore said nothing. Right about the method, wrong
   about the conclusion — the agent drove the press and the shelf appears in the same session. **My
   correction needed correcting.**
3. **The source tree for the running app.** Run 3's tester quoted a sentence I could not find in the
   code, so I called it a misremembering. The server was running 28-hour-old Python behind a
   current page. **They read it exactly as reported.**
4. **A silent no-op for a failure.** `__harness.dispatch(type, el)` takes the event **first**. I called
   it backwards, it did nothing and reported nothing, and I read "no POST" as the page failing and
   pivoted an analysis onto it. `TEST-083`.
5. **Two of my own probes lied before they worked.** One collected an **empty blob** — the shim
   returns nothing for `querySelectorAll("[id]")` — and reported every question as unrendered. The
   other put the original's title **inside** the twin's title, so the original matched inside the
   twin's own card. **Assert your reader read something before believing any absence.**
6. **I wrote `findings.json` during a live loop**, and filed `MISC-022` duplicating the agent's
   `TEST-087` within the hour. Nothing was lost; both were luck.

**The generalization, which the previous PM also recorded and I repeated in new costumes:**

> **Every number you state carries how you got it.** Mark your figures *(re-derived at `<sha>`)* or
> *(from the row)*, and **doubt the second kind first.** A file's existence is not its contents, a
> write-up's number is not a measurement, `ahead 31` is not a stale checkout, a 200 is not a correct
> consequence, and the source tree is not the running app.

**Design is the only work here with no verification loop.** A design proposal should ship with a
prior-art check the way a closure ships with a probe.

---

## 07 · The agent, and what to protect

**It is exceptional and it is better than me at several of these.** In five consecutive loops it has
corrected the adjudicator's premises, and three times it has **refused a part** with a measurement
rather than shipping a weak version:

- **L56-C2**: refused a palette the prompt supplied, because it failed the criterion the row itself
  stated — measured, simulator validated on controls first — **and declined to substitute its own**,
  on the argument that choosing one is a product decision.
- **L57**: **handed back** rather than starting a part at the tail of a long session, citing the scope
  note's own prohibition on half-built parts.
- **L59-C2 / L60**: handed back the positive-class build because deleting the default alone would make
  every figure refuse — the weaker version the rules forbid.
- **L60-E**: reported a blast radius it had **measured wrong**, in the same breath as the number that
  exposed it, and **declined to fix 81 failures by relaxing the thing under test.**

It also reported a trap **inside the guard it wrote to close that trap**, killed a competing suite
mid-loop and said so, and used `--no-verify` once with the reason stated in the commit and the other
five gates named as green.

**Rule against it when it is wrong and say plainly when it is right. A refusal is a result — treating
it as a failure to deliver is how you destroy the behavior.**

---

## 08 · Habits that are load-bearing

- **Verify in an isolated worktree, never the live tree.** `git worktree add --detach <path> <sha>`.
- **Never write the data files while a loop runs.** Check `ps` and `git status` first.
- **Hold pending docs edits outside the worktree** until the gates are green (`AUDIT-046`).
- **The ledger has exactly one writer and it is `ledger.py`.** `set` **replaces** the note — read the
  existing one and append. File notes **through a Python file, never a shell heredoc**; zsh eats
  backticks. Check the diffstat after: a handful of lines, not thousands.
- **`docs/audit/` is the exempt path for verbatim records** — the spelling gate skips it. Drive
  reports quote what the app printed and belong there; earlier ones are in `docs/turbotab/` only
  because they happened to contain no British spellings.
- **`zsh` does not word-split unquoted variables.**
- **Never `git add -A`.** Stage explicit paths.
- **Add the `LOOP.md` §03 row when you accept a loop.**
- **The check that fires most often:** was a named defect *class* filed, or only its instance?
- **Never accept a moved threshold in the same loop as the change that pressured it**, unless you
  invoke §06.2 deliberately and say so **in those words**.
- **`Ambiguity is OPEN, never FIXED`** — and ask, per row: *does the fix reach the thing the row's own
  evidence describes?*
- **Keep prose lean.** He has asked directly for minimum PM bloat without losing execution fidelity.

### Standing rules the agents earned — do not re-derive

- **§08.1, two tiers.** A `FIXED` arrives with its **probe output** or it is `PARTIAL`. The revert must
  be **total**. A red quoting a `TypeError`, `ImportError` or `HarnessError` is red for the **wrong**
  reason; an `AttributeError` that **is** the production defect is the right one.
- **A part may be REFUSED when carrying it out would violate the criterion the row itself states**,
  and the refusal is a **result**. Do not quietly build a weaker version, or your own.
- **A pin that cannot flip is a comment.** `pytest.xfail()` in a body can never XPASS; use
  `mark.xfail(strict=True)` on the case (`TEST-077`).
- **A grep answers "does this text appear", not "does this run."** It is what made L60's 81 a surprise.
- **A matcher that fires on prose has silence that means nothing.**
- **A suite is quotable only if nothing else is writing the tree *or competing for the machine*.**
- **A class goes in the ledger the moment you name it.** **Say the number, including when it is zero.**
- **No subagent runs a tree-wide git operation. A subagent gets its own worktree or no write tool.**

---

## 09 · What happens next

**1 · Adjudicate L61 when it lands.** Its five parts and the checklist are in
`docs/turbotab/prompts/L61.md`. In order of what has mattered: whether the 73 were **re-derived by
running rather than grepping**; whether any fixture sets the event by reaching into the project
instead of recording a decision; whether the launch command **refuses to start** on a stack it cannot
import, driven by breaking the import rather than reading the script; whether Part C invoked **§06.2
in those words** and enumerated every new entry.

**2 · The push unblocks with `TEST-086`.** Offer it once it is ruled; do not push unasked.

**3 · Then the back half, which is finally reachable.** `DRIVE-032` closing means `positive_label` is
recorded rather than guessed — **that was L57's deferred ROC overlay's real prerequisite**
(`GUIDED-236`, and `DRIVE-016` travels with it). `GUIDED-233`'s explainability pack is still the long
pole and still gates both deferred builds.

**4 · Still open from the drives, in rough value order:** `DRIVE-036` (the repeat path dead-ends and
the seal receipt promises a control that does not exist — L61-D2 pulls the sentence, the control is
unbuilt) · `DRIVE-038` (`[object Object]`, narrowed and not reproduced) · `DRIVE-039` (post-seal
exclusion controls stay enabled; the server refuses correctly, so it is an affordance defect) ·
`DRIVE-020`'s ordering half · `GUIDED-238` · the `SEQN` / survey-design / pooled-cycle gaps, which are
`GUIDED-231`'s port rather than defects.

**5 · Ask for the fifth drive once L61 lands**, and ask for it on a correctly launched server. Four
drives have each found something no suite could, and the first three were partly spent on an
environment nobody had checked.
