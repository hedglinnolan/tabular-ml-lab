# You are taking over as project manager and adjudicator for TurboTab

The previous PM was cleared. Everything durable is committed; this is the contextual onboard for what
is not. **It replaces the L52-era version of this file** — a stale transition sitting beside a
current one is the decay this project has already paid for five times.

**Read §06 of this file first.** It lists work that exists only in a cleared conversation, and some of
it contradicts what is currently committed.

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
  lens** (but see §06 of this file: parts of §06c are refuted and not yet corrected); "The shelf is
  never shortened" and its three-rung ladder.
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

Branch `TurboTab`, HEAD `52a8ff0` (L55-C). Ledger **883 findings, 366 closed**, register **182 rows**,
six gates green. **L55 is accepted, four of four parts — but its `turbotab/` sweep was still running
when the PM was cleared, so the loop is verified except for that number.** The agent said it would
update its report page in place when it lands.

Independently verified by the PM in an **isolated worktree**:

| Suite | Result |
|---|---|
| `tests/` fast tier | 1,730 passed, 1 failed — the failure is `TEST-038`'s `torch` |
| `tests/integration` | 262 passed, ~51s |
| `turbotab/` | **outstanding** — L55's own sweep, unfinished at handover |

```bash
venv/bin/python -m pytest tests/ --ignore=tests/integration \
    --ignore=tests/test_nn_modernization.py -q          # ~22s, the fast tier
venv/bin/python -m pytest tests/integration -q          # ~51s
venv/bin/python -m pytest turbotab/ -q                  # ~2 HOURS. Check ps first.
```

**`shap` is now installed** (0.52.0), which it was not for many loops despite being declared in
`requirements.txt:21`. It downgraded `numpy` 2.5.1 → 2.4.6 for `numba`; the fast tier was re-run and
is unaffected. **`turbotab/` has not been run under the new numpy** — that is the first thing L55's
sweep tells you.

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

## 06 · WHAT IS OWED AND IS NOT IN THE REPOSITORY

**This is the section I would most want a successor to read.** The following exists only in a cleared
conversation. Where it contradicts a committed file, **the committed file is wrong.**

### 06.1 · An adversarial review refuted much of `PRODUCT_VISION.md` §06c

Two adversarial reviewers with live literature search were run on §06c's explainability design on
2026-08-09. Both returned **SERIOUS PROBLEMS**. §06c has **not** been corrected. What they found:

- **The "substitution curve" is not novel and is misnamed.** It is a **multivariate forward marginal
  effect along d = e_B − e_A, aggregated as an Average Marginal Effect** (Scholbeck et al. 2024,
  *DMKD* 38:2997–3042), shipping as R package `fmeffects`. §06c calls it "a 1-D ALE along a
  constrained direction," which describes a different object — ALE accumulates local differences over
  the **conditional** distribution; shifting every row by *k* is marginal averaging.
- **§06c's claim that it "stays inside the data's support" is false, and measured false.** On a
  synthetic conserved-energy composition a 300 kcal shift puts **22%** of rows off-support; 500 kcal
  puts **64%** off with 1% negative intakes. **A support mask is mandatory.**
- **"The slope at k=0" does not exist for a tree ensemble** — piecewise constant, derivative zero
  almost everywhere. Report a finite difference at a **stated** *k*, with *k* in the label.
- **§06c's claim that the field has no way to state an estimand for a black-box learner is false**,
  and `NUTRITION_PACK.md` **§05 already contradicts it** by naming CoDA/ilr as a route. ilr is a basis
  change; any learner fits on it. Also available: the leave-one-out parametrisation, and
  g-computation with TMLE/DML.
- **Prior art §06c must position against**: Dumuid et al. 2019 (*Stat Methods Med Res* 28(3):846–857)
  — the compositional isotemporal substitution model; CRAN `codaredistlm` (pairwise one-v-one
  reallocations with CIs, 2022) and `multilevelcoda`; Ho et al. 2021 (*Lancet*) non-linear isocaloric
  substitution; Lundborg & Pfister 2025 (arXiv:2311.18501, **preprint only**), who define the estimand
  and **explicitly exclude random forests and boosted trees**; Mekary et al. 2009 (*AJE*
  170(4):519–527); Fisher, Rudin & Dominici 2019 (*JMLR* 20:177) — **Model Class Reliance**, which is
  the correct framing for what §06c calls "inductive bias."
- **Ruling 3 is unsafe as written.** kcal, total ion current and library size are all **variable**
  totals; 24-hour time is a **fixed** total, and they behave differently (Tomova et al. 2025, *BMC
  Med Res Methodol* 25:100). The case the method was built for is the one case §06c's list omits.
- **Ruling 7's three checks are broken as specified, each demonstrated by simulation.** The
  label-permutation null is vacuous for normalized measures (permuted-label impurity importances still
  sum to 1). Fold stability passes a stable-but-wrong explanation (y independent of X, held-out
  **R² = −0.136**, Kendall τ **0.867**). The deletion curve nearly fails a *correct* explanation
  (deleting the only causal driver cost 0.027 R² because a 0.995-correlated copy substitutes). Replace
  with Altmann et al. 2010 (PIMP) and ROAD (Rong et al. 2022, ICML).
- **The one genuinely new thing, and it is a gate:** ruling 7 gates on calibration and gates nothing
  on generalization. **Held-out performance should gate the entire explainability surface.**
- **`NUTRITION_PACK.md` §04 has a two-word error that is load-bearing.** It says the standard/residual
  models are biased even absent confounding *"because the substituted mixture is the population-average
  mixture."* The phrase "biased even absent confounding" is near-verbatim from Tomova et al. 2022
  (*AJCN* 115(1):189–198, PMC8755101) and is correct. **The word "because" is wrong** — the paper's
  mechanism is **composite variable bias**, information lost when ≥2 distinct-effect components are
  collapsed into a total; the population-average mixture is the paper's *definition of the estimand*,
  in an adjacent sentence. **This matters because it made the substitution curve read as a remedy for a
  bias it does not remedy** — the total is still in the model either way.

**And the product owner's reframe, which outranks all of the above:** *"Just because a CRAN package
exists in the world with a specific plot doesn't mean that would not still be a useful feature to ship
in our app."* **Read the review as a specification, not a verdict.** Nearly every objection converts
to a build requirement with a citation attached, and the existence of `codaredistlm`, `fmeffects` and
Dumuid 2019 gives every element of the figure a resolving source — which is what the evidence gate
requires anyway.

### 06.2 · The categorical chart ramp fails validation, and is not filed

Run on 2026-08-09 with a six-check validator, on both surfaces:

- **`--c1` is byte-identical to `--accent` (`#0E7368`)**, while `DESIGN_LANGUAGE.md` line 55 asserts
  *"charts use a separate categorical ramp so semantic color stays semantic."* The separation is
  asserted and not implemented.
- The shipped ramp **FAILS** chroma floor (`c1`, `c4` read gray) and **FAILS** the normal-vision floor
  (`c1`↔`c2` ΔE 13.8, below 15 — hard to tell apart with full color vision; tritan ΔE 3.3). Dark mode
  adds a contrast WARN at 2.63:1.
- A **three**-series ramp passes all six checks in both modes: light `#3B5BA9` / `#B15A33` / `#6E3FA3`,
  dark `#5B7BC9` / `#D0743F` / `#9A6BC4`, worst CVD ΔE 20.0. **Four cannot pass** — teal, green, gold
  and red are semantically reserved and the remaining hue space collapses under deuteranopia. The
  fourth series folds into "Other," a small multiple, or **dash pattern — which §07 already requires
  for journal view.** Promote that rule in-app.
- **21 built figure specs use the failing ramp.** File it; it is the same class as the journal CSS.

### 06.3 · Ledger and doc writes owed

- **`TEST-063`'s note is wrong and it is mine.** I set it `PARTIAL` saying *"the underlying repair
  looks real and only its record was wrong."* L55 measured both halves false — a total revert of the
  fix returns `GREEN — NOT LOAD-BEARING`, and the `table` fixture has restored `_OPERATIONS` since
  **L18**, so the recorded cause cannot have produced the recorded symptom. **Correct the note.**
  `TEST-068` holds the unexplained symptom as a question, correctly.
- **The L55 `§03` row is unwritten.** Adding it is part of adjudicating.
- **§06.1 and §06.2 above need to become rows and a §06c rewrite.**
- **`LOOP.md` §06 needs the claims-of-absence check** — see §07 below.

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

**The generalization, and it belongs in `LOOP.md` §06: a claim that something does not exist requires
reading, not searching.** An absence cannot be established by the evidence that suggests it. My greps
came back empty and I read empty as proof, four times, in a project whose §02 already says *"a matcher
that fires on prose has silence that means nothing"* — that is the tool-side version of the same rule
and I never applied it to myself.

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

**L56 is unwritten.** The two candidates, and the sequencing argument between them:

- **The explainability audit layer.** ~1,000 papers compute SHAP on dietary data (~400 on NHANES,
  190 in 2024–25), with documented cases publishing an r ≥ 0.7 correlation map and then ranking those
  same nutrients by split-frequency importance. **The audit layer needs no new estimator**, addresses a
  documented failure at scale, and the packs already hold the domain directions to audit against.
- **The substitution figure.** Needs the support mask, the uncertainty band, the linear-ilr null
  overlay, a stated *k*, and a pack section before it can ship honestly — see §06.1.

**I would sequence the audit layer first and it is a product call, not mine.** Either way
`GUIDED-233`'s pack is the prerequisite for the thresholds, and **§06c must be rewritten before either
is built** — it currently contains claims that are refuted and committed.

**Owed to L55 regardless**: its `turbotab/` number, the `§03` row, and `TEST-063`'s corrected note.
