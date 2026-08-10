# You are taking over as project manager and adjudicator for TurboTab

The previous PM was cleared. Everything durable is committed; this is the contextual onboard for what
is not. **It replaces the L52-era version of this file** — a stale transition sitting beside a
current one is the decay this project has already paid for five times.

**Nothing in the repository currently contradicts anything else in it.** The previous handover carried
a §06 of work that existed only in a cleared conversation, including refutations of claims that were
still committed; **all of it landed on 2026-08-09** and §06 is now the index of where. You can read
`PRODUCT_VISION.md` cold. **Read §07 before you write that something does not exist** — that is the one
section whose warning is still live rather than historical.

---

## 01 · What this work is, stated first because it determines how you write

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

The role requires **ruling against reports** — accepting, rejecting, and naming defects in work an
agent says is finished. Decisiveness is the job.

---

## 02 · The working relationship

Nolan is the product owner — *"the product design guy."* He does not read the code; you do. He runs
an execution agent on his laptop, pastes its reports to you, and you rule and write the next prompt.
He expects you to be better than him at orchestration detail, so **make calls, do not survey
options.** He wants honest disagreement; when he reaffirms something, that is a decision.

**He likes the prompt delivered as a copy-with-one-click page.** Each loop prompt is written to
`prompts/L<n>.md` and published as an Artifact with a copy button. The builder script is disposable
and lives in the scratchpad — regenerate it. It reads the repo file, embeds it **byte-for-byte** as a
JS string literal (not escaped element text — HTML entities are not decoded inside `<script>`), and
renders under `DESIGN_LANGUAGE.md`'s palette and three-voice type rule. **The repo file is the
record.** Verify the embedded literal decodes back to the file byte-for-byte; the builder does this.

**His thesis: the steps are not the product, the connective tissue between them is.** His standing
rulings: `ROADMAP.md` condition 7 — *"In addition to being correct, the engine must surface and it
must be beautiful"*; **time is the constraint**, he runs loops back to back; and **he is a
completist** — *"if we have items on the backlog, it's time to run full test suites less frequently
and start fixing more items per loop."*

**Four things he did that you should expect and use.** He corrected a premise I asserted about his
own app and was right (§05). He reframed an adversarial review I had misread as a verdict when it was
a specification. He supplied an outside domain-agnostic playbook and named its role exactly — *"a
useful mould for what we actually want to form into our domain-aware playbook."* And he apologised
for impatience that had twice produced the best measurement of the session. **Take his framings
seriously; they have been load-bearing more often than mine.**

**Also read [`AGENT_ONBOARD.md`](AGENT_ONBOARD.md).** It is the execution agent's onboard, it is
actively maintained by the agents themselves, and everything in it binds you too. **§03 is now parsed
by a guard** — `tests/test_a_fixed_rows_named_test_resolves_in_five_seconds.py` reads the `--ignore`
list out of it, so the document and the check cannot drift. Ruled 2026-08-09: keep it that way.

---

## 03 · Read, in this order

`README.md`, `PRODUCT_VISION.md`, `ROADMAP.md`, then `LOOP.md` §02 (loop shape), §05 (guardrails),
§06 (adjudication), §03 (the log — the last six rows are the live context). Then `DOMAIN_SCIENCE.md`.

**Do not skip these; each is a ruling made in conversation and recorded because it decides work:**

- `PRODUCT_VISION.md` **§06b — correct, surfaced, beautiful**; **§06c — explainability under the
  lens**, corrected against the adversarial review at `bea4878` and safe to read cold now; "The shelf
  is never shortened" and its three-rung ladder.
- `ROADMAP.md` **condition 7**; "What comes after the journey"; "Why the front of the journey is
  where the depth belongs" **and the standing risk it names** — all three of its unfalsifiable
  front-half designs are now discharged, which is worth knowing before you re-read it as a live worry.
- `DESIGN_LANGUAGE.md` **§05.2** — motion preserves identity, and the L47 measurement table; **§07** —
  the in-app/journal duality; **§10** — four education layers.

**Five research files in `research/`, authoritative.** Read **by section, when cited, never
wholesale.** Where a file and your recollection disagree, **the file wins** — and where a file and a
primary source disagree, see §06's note on `NUTRITION_PACK.md` §04.

---

## 04 · State right now

> **⚠ L57 IS RUNNING RIGHT NOW. Read §04b before you touch anything.** If you are reading this
> because the PM was cleared mid-loop, that has happened before — it is what §06 of the previous
> version of this file existed to repair — and the rule that matters is the one in §08: **the loop
> owns `findings.json` and `register.json` until it reports.** Docs-only commits may land meanwhile
> if they touch neither and say so.

Branch `TurboTab`, HEAD `e69af3d`. Ledger **891 findings, 373 closed** (`FIXED` **367**), register
**182 rows**, six gates green. **L55 and L56 are both accepted and adjudicated**, with their §03 rows
written.

| Suite | Result | Taken at |
|---|---|---|
| `tests/` fast tier | 1,738 passed, 1 failed — the failure is `TEST-038`'s `torch` | L56 |
| `tests/integration` | 262 passed, ~50s | L56 |
| `turbotab/` | **2,464 passed · 17 skipped · 9 xfailed · 0 failed**, 7194.87s (**1:59:54**) | `07c25c6` — the current baseline |

`partition_is_exhaustive()` returns **358 / 0 / 9 = 367**, matching `status==FIXED` exactly.
**`not_pytest` sits at 9 against a cap of 10** — the next tool-invocation row breaches it, and it is
raised on a *passing* run with the reason recorded, never in the loop that trips it.

### 04b · What L57 is doing, because it is in flight

**Five parts** (`prompts/L57.md`, published with a copy button). **A** the categorical-ramp validator
then the palette · **B** the ROC overlay at `_risks_or_refuse` · **C** `GUIDED-233`'s explainability
pack · **D** the anti-pattern audit widened · **E** suite cost round two, xdist priced not adopted.

**As of this writing the tree is dirty with Part A**: `prototypes/interview-feed.html` and
`turbotab/web/index.html` both modified, plus a new
`turbotab/test_the_categorical_ramp_is_separable_and_legible.py` — which is the prototype-first carry
the part specifies. **That is the loop working, not a problem.** Do not clean, stash, checkout or
reset anything.

**The two rulings L57 carries that a successor must not re-derive** are in §00 of that prompt: the
palette is ruled as a **method with two gates and a validator that ships first**, not as a set of
hexes; and C1's site is **`_risks_or_refuse` at `figure_bundle.py:443`**, not `predictions_for` —
`scored` already exists at `:426` and `best = scored[0]` at `:430` is the single line that discards it.
**`positive_label` is per result and must be asserted to agree**, or the overlay draws different
outcomes on one axis.

**The `turbotab/` number is quotable and the reason is checked rather than assumed.** A docs-only
commit (`57d542f`) landed **22 minutes into** that two-hour run, so the tree *did* move under it —
§08's rule says who moved it does not matter. It stays quotable because nothing in `turbotab/` reads
the file it touched: the only test that reads anything under `prompts/` is the five-second `FIXED`-row
guard, and that lives in `tests/`. **If you land a docs-only commit during a sweep, check that
specifically and write down what you checked.**

```bash
venv/bin/python -m pytest tests/ --ignore=tests/integration \
    --ignore=tests/test_nn_modernization.py -q          # ~22s, the fast tier
venv/bin/python -m pytest tests/integration -q          # ~51s
venv/bin/python -m pytest turbotab/ -q                  # ~2 HOURS. Check ps first.
```

**`shap` is now installed** (0.52.0), which it was not for many loops despite being declared in
`requirements.txt:21`. It downgraded `numpy` 2.5.1 → 2.4.6 for `numba`; the fast tier was re-run and
is unaffected, and **L55's sweep is the first full `turbotab/` run under the new numpy — 0 failures.**
`tests/test_shap_and_sensitivity.py` now passes **10 of 10**, so the expected-failure set is **one**
item and not four; `AGENT_ONBOARD.md` §03 said four until `d10346f` corrected it.

**`make test` still aborts at collection on `TEST-038`'s unguarded `torch` import. Do not work around
it.** `torch` is deliberately not installed: it is 1.1 GB and mandatory for a user who never trains a
neural net.

---

## 05 · Where the product actually is

**The journey is complete** and drivable end to end. **42 findings trace to the product owner's
NHANES drives and 34 are closed.** Re-derive that before quoting it: the query is every ID appearing
in `DRIVE_LOG_NHANES.md` and `DRIVE_PREREG_NHANES.md`, unioned with every `DRIVE-` row.

**The largest open thing is `GUIDED-231`, and its framing is the finding.**

> **The inference half of the product was never ported.** `GUIDED-105` (no inference-family models),
> `GUIDED-106` (subgroups), `GUIDED-118` (time-to-event) and `GUIDED-230` are its **symptoms, not its
> peers**, and sizing them separately made a port look like a backlog. The register marks
> `target-goal-selection` and five statistical tests `classic-only`, and
> `pages/09_Hypothesis_Testing.py` is 1,060 lines.

**And the correction the product owner made to it is the more useful half.** The original row said
Guided *cannot declare* a hypothesis-testing goal. **That is false** — `turbotab/purpose.py` asks
*"What is this model for?"* with PREDICTION and INFERENCE, among the first questions the app asks. His
words: *"we straightforwardly do ask prediction v inference, I'm just not sure we really wire it to
any concrete changes in the engine yet."* **Inference is not unasked, it is unwired.**

**That is one shape, and it recurs everywhere.** `purpose` had two consumers of the four its own card
names (three now, after L55-B). The lens reached thirteen modules and not the shelf. `repeat_kind` and
`unit_of_analysis` were recorded and read by nothing that fits a model. `promotable` waited eight
loops. `stale_downstream` was written and never read. The journal CSS was styled with **no producer at
all**. `GUIDED-099` was `FIXED` in prose with an empty `test` field for twenty loops.

**The measurement says this is architectural, not a discipline failure.** Every consumer is
hand-wired, so every one can be forgotten independently, and `purpose.py`'s `CONSUMER` string is a
routing table implemented as hand-maintained prose. **A place where a recorded answer declares what
reads it — so an unrouted answer is a failure rather than an omission — is the strongest architectural
move available.** That is an argument, not a ruling; the product owner has not decided it.

**What else is left, measured rather than estimated:**

| | |
|---|---|
| **The anti-pattern audit** | 131 tabulated entries, **77 never run** (`AUDIT-038`, filed so silence is not read as coverage); `CLINICAL_SURVEY_PACK` has ~25 more in nine prose blocks with no table; `INTERACTION_PACK` has none. |
| **Explainability** | `GUIDED-232` (Guided has permutation importance only; SHAP is `classic-only`) and `GUIDED-233` (**no explainability pack exists** — all five packs return zero for shapley, attribution, feature importance and partial dependence). §06c holds the design. **Read §06 of this file before building any of it.** |
| **Reference data (D4)** | Largely unstarted. DRI tables must ship as data read from NASEM. |
| **L10–L12** | Parity harness in CI, Streamlit convergence, packaging. |

---

## 06 · WHAT WAS OWED, AND WHERE IT LANDED

**This section existed because the work was not in the repository. It is now.** Discharged at the L56
handover on 2026-08-09; kept as an index rather than a second copy, because a restated claim drifts
from the canonical one — the mistake `LOOP.md` §05 records about the freeze.

| Was owed | Landed as | Canonical location |
|---|---|---|
| The adversarial review of §06c | **`GUIDED-237`** (`FIXED`) | `PRODUCT_VISION.md` §06c, rewritten in place at `bea4878` |
| `NUTRITION_PACK.md` §04's two-word error | **`AUDIT-045`** (`FIXED`) | `research/NUTRITION_PACK.md:408`, corrected with the superseded wording quoted and dated |
| The categorical chart ramp | **`DRIVE-015`** (`OPEN`) | the row carries the validator output and the corrected blast radius |
| `TEST-063`'s wrong note | corrected, stays **`PARTIAL`** | the note retracts the premise and dates both fixture halves |
| The L55 §03 row | written | `LOOP.md` §03 |
| The claims-of-absence check | written | `LOOP.md` §06, as its own subsection ahead of the numbered list |

**Three things about the discharge are worth carrying forward, and they are not in the rows.**

**The review is a specification, not a verdict, and that is the product owner's ruling** — *"Just
because a CRAN package exists in the world with a specific plot doesn't mean that would not still be a
useful feature to ship in our app."* It sits at the top of the rewritten §06c because a successor
reading a page of refutations will otherwise conclude the feature was killed. It was not. Nearly every
objection became a build requirement with a citation attached, and the three reference implementations
mean every mark of the figure now has a resolving `source` — which is what the evidence gate wanted.

**The one new thing the review produced rather than refuted is a gate**, and it is the most valuable
line in it: ruling 7 gated on calibration and gated **nothing on generalization**. Held-out
performance now gates the entire explainability surface. Its **value** is `GUIDED-233`'s to source,
not the adjudicator's to pick, so it is *specified and unbuilt* — which is not the same as absent.

**The ramp's blast radius was overstated and correcting it changed the sequencing, not the severity.**
The write-up said 21 built specs used the failing ramp. It is 17 built plus 4 declared-pending, and
after L55-C only `roc` and `item_correlations` have a renderer at all — `WEBC` is consumed at two
lines inside the multi-series path, and per `GUIDED-236` the ROC can never draw more than one curve.
**So the ramp's distinguishing function is reached by nothing in production today.** That makes it the
same class as the journal CSS rather than merely comparable to it, and it is the argument for fixing it
**now**, while it costs one figure instead of twenty-one. It is also the fifth entry in §07's list, and
the only one that is an asserted **presence** rather than an asserted absence.

---

## 07 · Calibration — read this before you assert anything

**The divergence section has corrected the adjudicator in every loop it has been written for.** Read
it first, every time. When the agent says it is unsure, it has usually already checked.

**My failure mode was one thing, four times: I asserted an absence.**

1. *"Guided cannot declare a hypothesis-testing goal"* — without reading `turbotab/purpose.py`. The
   product owner caught it.
2. *"The field has no way to state an estimand for a tree ensemble"* — without reading
   `NUTRITION_PACK.md` §05, which names CoDA. An adversarial reviewer caught it.
3. *"§04 has drifted from Tomova"* — on one reviewer's quote, without opening the paper. A second
   reviewer caught it, and the truth was narrower and sharper.
4. *"The journal CSS is a rule missing its handler"* — written into L55's Part C. The agent measured
   it: **no figure was drawn at all**, two `<svg>` occurrences in 8,479 lines. The correction was
   bigger than the row.
5. **And a fifth, added by my successor, because it is the same failure inverted.** *"21 built figure
   specs use the failing ramp"* — an asserted **presence**, counted from a write-up rather than the
   registry. It is 17 built plus 4 declared-pending, and only **two** figure kinds have a renderer at
   all, so the ramp reaches **one**. The rule generalizes past absences: **a count is a claim, and the
   first thing to doubt is the thing you counted.**

**The generalization is now in `LOOP.md` §06 as its own subsection: a claim that something does not
exist requires reading, not searching.** An absence cannot be established by the evidence that suggests
it. My greps came back empty and I read empty as proof, four times, in a project whose §02 already says
*"a matcher that fires on prose has silence that means nothing"* — that is the tool-side version of the
same rule and I never applied it to myself.

**Two more of mine, both the habit this file warns about:**

- **I quoted stale numbers forward into L55's §01** (873/361 when the ledger read 876/361), two days
  after editing this file to warn against exactly that. The agent caught it.
- **I committed to `findings.json` while a loop was live** — §05's mid-loop exemption is docs-only and
  *excludes* the data files. Nothing was lost, and it is the rule I had been enforcing on the agents
  all session.

**Design is the only work here with no verification loop.** Code gets tests, a `FIXED` row gets a
revert probe, a pack claim gets an evidence badge and a resolving source. A **design proposal** got
nothing until the product owner asked for a reviewer — and it came back with a correct name, three
existing implementations, and a measured refutation of a claim already committed to
`PRODUCT_VISION.md`. **A design proposal should ship with a prior-art check the way a closure ships
with a probe.** One search, before it becomes a ruling.

**The counterweight: this agent is exceptional, and better than me at several of these.** It refused
to replace one unverified diagnosis with another (`TEST-068` — *"the same defect with a newer date"*).
It reported trap 5b **inside the guard it wrote to close trap 5b**. It reported writing a
freshly-filed defect class into its own test on the same day it filed the class. It killed a suite
mid-loop and said so. It declined to add `parentNode` because doing so would change what untested
branches observe, and filed a test asserting the absence so a later loop cannot add it by accident.
**Rule against it when it is wrong and say plainly when it is right.**

---

## 08 · Habits that are load-bearing

- **Verify in an isolated worktree, never the live tree.** `git worktree add --detach <path> <sha>`.
- **Do not write `findings.json` or `register.json` while a loop is running.** Check `git status`
  first: a dirty tree with source files modified means a loop is live. Docs-only commits may land
  mid-loop **only if** they touch neither data file **and the message says so**.
- **The ledger has exactly one writer and it is `ledger.py`.** `set` **replaces** the note — read the
  existing one and append. File notes **through a Python file, never a shell heredoc**; zsh eats
  backticks. Check the diffstat after: a handful of lines, not thousands.
- **`zsh` does not word-split unquoted variables.** I lost a two-hour measurement to `$IG` expanding
  as one argument, which is the L51 lesson made twice.
- **Write decisions into the docs the same turn they are made.**
- **Add the `LOOP.md` §03 row when you accept a loop** — it is part of adjudicating.
- **Never `git add -A`.** Stage explicit paths and run `git status` first, every time.
- **The check that fires most often:** was a named defect *class* filed, or only its instance?
- **Never accept a moved threshold in the same loop as the change that pressured it**, unless you
  invoke §06.2's exception deliberately and say so in those words. L53-A2 is the model: the cap
  changed **purpose, not value**, with the measurement quoted in source.
- **`Ambiguity is OPEN, never FIXED`** — and ask, per row: *does the fix reach the thing the row's own
  evidence describes?*
- **Keep prose lean.** He has asked directly for minimum PM bloat without losing execution fidelity.

---

## 09 · Standing rules you inherited, which the agents earned

Ruled during L47–L55. Do not re-derive them.

- **§08.1, two tiers.** A subagent's `FIXED` arrives with its **probe output** — the revert, the red,
  and the sentence it was red for — or it is `PARTIAL` by default. Where the fan-out has room, the
  probe is run by a **different** subagent than wrote the fix. **The revert must be total.**
- **A red that quotes a `TypeError`, `ImportError`, `HarnessError` or signature mismatch is RED FOR
  THE WRONG REASON** and does not discharge the probe.
- **A suite is quotable only if nothing else is writing the tree *or competing for the machine*.**
- **Partition a fan-out by fix site, not by row.**
- **A matcher that fires on prose has silence that means nothing.** Anchor the pattern or match
  structure. Compare leaf segments, not substrings.
- **A mechanism the codebase has zero instances of is unpriced until the INSTRUMENT is checked.**
  L54-B0 cost rounds because nobody asked whether the page harness implemented the DOM APIs the design
  needed. It did not — and its `appendChild` **copied rather than moved**, which would have silently
  invalidated the proof.
- **Do not sequence certain plumbing behind unpriced discovery.** L54's overrun killed a part that had
  no unknowns at all.
- **A class goes in the ledger the moment you name it** — not in the chunk prompt, not in a docstring,
  not in the report.
- **No subagent runs a tree-wide git operation** — no `stash`, `checkout`, `clean`, `reset`.
- **A subagent gets its own worktree or no write tool at all.** Do not rely on the instruction.

---

## 10 · The next loop

**The product owner ruled the sequencing on 2026-08-09, and he ruled against both build candidates:
neither the explainability audit layer nor the substitution figure goes first.** L56 does the
prerequisites the two of them share. His words for the option chosen: *"Ramp + §06c rewrite as its own
loop first."*

**Half of that ruling was PM work and is already done** — the §06c rewrite (`bea4878`), the three rows,
and `TEST-063`'s note all landed the same day it was made, so L56 has room. What remains for the loop:

- **The categorical ramp** (`DRIVE-015`), while it costs one figure instead of twenty-one.
- **`GUIDED-233`'s explainability pack**, which gates the thresholds for *both* deferred builds and is
  pack authoring rather than code.
- **Suite cost, and it is now a stated constraint rather than an annoyance.** The product owner:
  *"These full suite tests are simply taking too long for the workflow we are currently in. They run
  over two hours and occasionally time out."* **The measured answer already exists in L55's own
  report** — Part D declined a 55-file sweep and ran *"the 11 files that actually reach an `appendChild`
  call site — 217 passed in 6m08s."* That is the pattern: a loop's regression evidence is the
  `turbotab/` files reaching the changed code, **quoted honestly as scoped**, with the full sweep moved
  out of the loop's critical path. `pytest-xdist` 3.8.0 is installed and L53 ruled it kept, but it is
  **not** the first move: `AUDIT-040`'s floor says no width beats ~1,175s, `GUIDED-099`'s registry is
  process-global, `TEST-063`'s guard **requires two nodes in one interpreter** and `--dist load` would
  let it go green while checking nothing, and `TEST-030` says outright that xdist-for-speed breaks
  `tests/workflow/*` silently. **Price it by comparing verdicts, not durations, and pin what must share
  an interpreter with `xdist_group` before adopting any of it.**

**The two deferred builds, unchanged and still ordered behind the pack.** The **audit layer** needs no
new estimator, addresses a documented failure at scale (~1,000 papers compute SHAP on dietary data,
~400 on NHANES, 190 in 2024–25, with published cases showing an r ≥ 0.7 correlation map and then
ranking those same nutrients by split-frequency importance), and the packs already hold the directions
to audit against. The **substitution figure** now has a complete visual specification — its five marks
are enumerated in `PRODUCT_VISION.md` §06c — and **no pack section behind its thresholds**, which
ruling 6 forbids shipping unsourced. Buildable, not yet shippable.

**Owed to L55 and L56: nothing.** Both are adjudicated, both §03 rows are written, and every ruling is
in a commit or a ledger note rather than in a conversation.

### What adjudicating L57 will require

**Read `prompts/L57.md` §00 first** — it holds the two rulings above and the three answers the L56
report asked for. Then, in the order these have mattered:

- **Part A's validator must be able to fail.** Check its controls (red/green must collapse under
  deuteranopia, blue/orange must survive) and check that it **rejects the ramp shipped today**. A CVD
  check that cannot fail is the thing under test.
- **Whether the palette landed in BOTH files via the prototype.** The two `<style>` blocks are 30,968
  characters and byte-identical, and `test_skeleton.py:698` reads the prototype. The carry is what stops
  the surfaces diverging.
- **Whether `positive_label` agreement is asserted** in Part B, and whether **calibration's payload is
  byte-identical before and after, by test** rather than by review.
- **Whether L55-C's one-curve assertion was replaced rather than deleted** — it was written to go red
  the day `GUIDED-236` is fixed.
- **Whether `TEST-063`'s guard was pinned with `xdist_group` before any `-n` ran.** If it was not, the
  reconnaissance destroyed the instrument it was measuring and the result means nothing.
- **Whether E2 re-measured `AUDIT-040`'s floor** rather than quoting the stale 1,175s, which L56's own
  fix invalidated.

**The habit that earned its place twice in two loops**: the agent's divergence section corrected the
adjudicator on three premises at L56 and then **refused a part** on a measurement that overturned a
number the adjudicator had put in the row. `LOOP.md` §02 now carries both the refusal permission and
the rule that **every number a prompt states carries how it was derived.** Honor both — mark your own
figures *(re-derived at `<sha>`)* or *(from the row)*, and doubt the second kind first.
