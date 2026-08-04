# You are taking over as project manager and adjudicator for TurboTab

The previous PM was cleared. Everything durable is committed; this is the contextual onboard for
what is not. It replaces the L40-era version of this file — a stale transition sitting beside a
current one is the decay this project has already paid for three times.

---

## What this work is, stated first because it determines how you write

TurboTab is **research software**. Your job is statistical methodology and software engineering —
routing logic, test design, figure specifications, reference tables. Not clinical practice; there is
no patient anywhere in this system. The "biology" is reference data and methodological literature:
unit-conversion constants, physiologic plausibility bounds, DRI tables, QC thresholds, reporting
checklists like TRIPOD+AI and STROBE-nut.

**Precision is the safety property and hedging is the defect.** The governing rule:

> The app may be **silent**, and it may **refuse**, but it must never **assert something false.**

There is already a calibrated apparatus for uncertainty — SETTLED / CONVENTION / DISPUTED on every
advisory, a resolving `source` on every claim, `[verify-at-build]` structurally barred from shipping
as a constant, and six pre-commit gates enforcing it. **A second, uncalibrated layer of caution does
real damage**: it makes a SETTLED fact and a DISPUTED one read alike, which is the exact failure the
badge exists to prevent.

So say the specific thing. You must be able to write *"1 IU retinol = 0.3 µg RAE while 1 µg vitamin
D = 40 IU, and conflating them is a 12× error,"* and *"a systolic pressure below 30 mmHg is
physiologically impossible in a living outpatient while 812 readings above 140 are the sickest
patients and must be kept."* Where you genuinely do not know: DISPUTED with both positions, or a
`BLOCKED.md` entry. **Never a vague gesture.**

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
and lives in the scratchpad — regenerate it. It reads the repo file, embeds it byte-for-byte for
copying, and renders it under `DESIGN_LANGUAGE.md`'s palette and three-voice type rule. **The repo
file is the record**; prompts live in the repo because one nearly existed only in a chat log.

His thesis, restated this session: **the steps are not the product, the connective tissue between
them is.** Judge design proposals against that. And his ruling of 2026-08-03, now `ROADMAP.md`
condition 7: *"In addition to being correct, the engine must surface and it must be beautiful."*

**Also read [`AGENT_ONBOARD.md`](AGENT_ONBOARD.md).** It is the execution agent's onboard and it is
timeless — the tools, the guardrails, and §07's traps. Everything in it binds you too, and two of its
entries exist because a PM broke them.

---

## Read, in this order

`README.md`, `PRODUCT_VISION.md`, `ROADMAP.md`, then `LOOP.md` §02 (loop shape), §05 (guardrails),
§06 (adjudication), §03 (the log). Then `DOMAIN_SCIENCE.md`.

**Do not skip these; each is a ruling made in conversation and recorded because it decides scoped
work:**

- `PRODUCT_VISION.md` **§06b — correct, surfaced, beautiful**, and the live consequence recorded in
  it. Also "The export, and what a marked figure means"; "The shelf is never shortened" and its
  three-rung ladder.
- `ROADMAP.md` **condition 7** on the definition of done; "What comes after the journey";
  "`research/INTERACTION_PACK.md` is scheduled"; "Why the front of the journey is where the depth
  belongs".
- `DESIGN_LANGUAGE.md` **§05.2** — the motion list is closed at four, and the ruling inside it is the
  previous PM's: the collapsed-remainder expand is instant. **§10** — four education layers, two
  shipped.

**Five research files in `research/`, 4,247 lines, authoritative.** Read **by section, when cited,
never wholesale.** Where a file and your recollection disagree, the file wins.
`INTERACTION_PACK.md` is new at L46 and differs in one way that matters: egress worked, so 100 of its
105 claims were read in primary and every one was adversarially refuted. Its §07 is a list of
citations it **refuses** to supply.

---

## State right now

Branch `TurboTab`, HEAD `bc16e6c`, tree clean. Ledger **791 findings, 292 closed**, register **180
rows**, six gates green on every commit.

Last independently verified suites (L45 tree): `turbotab/` **1589 / 0 failed** · `tests/` **1655 / 4
environmental** · integration **211**. L46 reported 1608 / 1655 / 211 and **that is unverified**.

```bash
venv/bin/python -m pytest turbotab/ -q                          # ~15-30 min under load
venv/bin/python -m pytest tests/ --ignore=tests/integration \
    --ignore=tests/test_nn_modernization.py -q                  # ~9 min
venv/bin/python -m pytest tests/integration -q                  # ~40 s
venv/bin/python -m uvicorn turbotab.api:app --port 8000         # the app
```

Four failures are environmental and expected — three `shap`, one `torch`. `make test` still aborts at
collection on `TEST-038`; **do not work around it.** Phases L1–L8 done, all three decision gates
answered, the Guided journey runs upload → report and the manuscript exports.

---

## The two things you are owed, in priority order

### 1 · L46 was reported and never adjudicated, and its report existed only in the cleared log

**Do this first.** Commits `89b029a`, `bf57811`, `ddaab25`. Reconstruct from those and **re-measure**;
what follows is what the report claimed, recorded here because it is otherwise gone.

**Claimed:** `turbotab/` 1608 / 0 failed / 18 xfailed · `tests/` 1655 / 4 · integration 211 · six
gates · ledger 767/292 · register 180.

**Its divergence section, which is where the value is:**

- **Part B came back against a ruling, and that is the pack working.** L45 ruled a promoted card is
  **marked, not moved** — no fifth motion slot. The interaction pack's first run says that is *right
  as a constraint and unsupported as a preference*. Its first draft argued the opposite from Simons,
  Franconeri & Reimer's 97% detection of instant changes; **the adversarial pass reversed it** —
  Simons' gradual condition is a twelve-second transient-free dissolve engineered to suppress the
  motion signal, so it does not transfer to a sub-second UI transition. And **Huhtala et al.,
  INTERACT 2009**, n=40, icons appearing and disappearing in a twelve-icon grid — the nearest
  published analogue to a card list reflowing after a dismissal — found animated transitions gave
  *significantly better user performance* and change blindness in the instant condition. The agent
  left the reversal visible inside the pack rather than absorbing it. **Verify it, and decide what it
  does to `GUIDED-073`.**
- **A2 moved the number A1 was accepted on.** Bound 5 was justified as *collapses on three of sixteen
  fixtures*; `MIN_COLLAPSE = 2` made it **one of sixteen**. The direction strengthens the
  fires-on-the-tail argument and weakens the median as evidence; the agent corrected `BOUND_BECAUSE`
  and re-derived rather than restating. **Ask whether a median over sixteen synthetic fixtures is
  still doing work at 1/16.**
- **It refused Part D's ledger as written, correctly.** `rendered + collapsed + dismissed = served`
  double-counts — a dismissed card is still rendered, as a `.gone` card with an undo note, which is
  the shelf not being shortened. The disjoint form it asserted is `live + cleared + collapsed ==
  served`.
- **A2 and A3 interact in a way neither ruling anticipated:** clearing one card can promote **two**,
  because a remainder of one is shown rather than collapsed. The probe asserts the promoted set is a
  *prefix* of what was behind the affordance.
- **Part C was three layers and `GUIDED-153`'s row named one.** `ml.router.plan` re-presents a
  deferred finding as a repair question and `_is_repairable` admits only findings carrying a
  `fix_kind` — which a pack finding never does, by design. So `PRODUCT_VISION.md` §04's deferral loop
  was open for the **entire pack stream**, not just mislabeled.
- **`GUIDED-155` filed, not fixed, and you owe the ruling.** `DESIGN_LANGUAGE.md` §05's 250 ms /
  300 ms / ≤150 ms have **no source**, may not be attributed to the anchor, and Propagate is a
  stagger with no demonstrated tracking benefit. Correctly not fixed — §06.2 forbids moving a
  threshold in the loop that pressured it.
- **`TEST-045`**: a parametrized test id with a non-ASCII character cannot be addressed by
  `revertprobe.py`, which reports `UNREVERTED: FAIL` — indistinguishable from a broken test. Fixed in
  the instance, filed as the class.
- One domain call it wants checked: three of `PACK_DEFER`'s 21 destinations are judgment —
  `dietary::compositional` → preprocess, `genomics::counts_p_over_n` → preprocess, the three
  survey-design detectors → train. **No drive reaches Train**, so those three are a table, unverified.

### 2 · The NHANES drive produced 24 findings and none is scheduled

The product owner drove a **real** NHANES export — `nhanes_1999_2018_yayhoo_fasting_diet_imputed.csv`,
21,849 × 29, nine pooled cycles — through the Guided door for about two hours. It is all in
**`DRIVE_LOG_NHANES.md`** and the sealed **`DRIVE_PREREG_NHANES.md`**. Read both. They are the most
concentrated evidence this project has about what the app is like to use, and every fixture in the
repo is synthetic and under 600 rows.

**`GUIDED-156` … `GUIDED-179`.** Two critical:

- **`GUIDED-170`** — the prevalence surface accepted `SEQN`, a row identifier, as a nutrient and
  answered *"Prevalence of inadequacy for SEQN is computed by the EAR cut-point method"* with a
  **SETTLED** badge, `may_preselect: true`, and a resolving citation — on the one surface built to
  demonstrate refusal. Its four refusals check the *basis* and the *reference kind* and never check
  whether the subject is a nutrient.
- **`GUIDED-165`** — the impossibility decision changes nothing, and the generated manuscript carries
  **both** *"were set to missing"* and *"were kept as recorded"* about the same column, consecutively.

**The shape is the finding.** Not one of the 24 is a missing capability. Every one is a **seam
between two correct things**, and four granularities of automated sweep — route, field, name,
stand-in, built across L42–L44 — found none of them. Two sub-classes worth carrying forward:

- **Five controls produced nothing visible, from five different causes** — unwired, absent, inert,
  unacknowledged, undrawn — and **two of them were the app responding correctly, out of sight**
  (`GUIDED-167`, `GUIDED-173`). §09's nudge rule covers content arriving *below* the viewport and
  nothing covers a refusal *above* it.
- **Placeholders and `None` reach the user** in the decision sentence and the reviewer checklist
  (`GUIDED-175`, `GUIDED-179`) — a fourth branch of the governing rule nobody authorized: not false,
  not silent, not a refusal, just noise where a sentence was promised.

---

## What I would do next, yours to override

**Adjudicate L46, then make Part A the drive's cheapest criticals.** `GUIDED-170`'s missing refusal
axis and `GUIDED-165`'s inert decision are both small and both are governing-rule failures in the
artifact that leaves the building.

Then, in rough order of what the drive says costs most: the **viewport class** (`GUIDED-167` and
`GUIDED-173` together — one fix, five symptoms), the **placeholder class** (`GUIDED-175`,
`GUIDED-179`), and then `GUIDED-160`'s missing education layer 3 — which is also `GUIDED-174`'s and
`GUIDED-178`'s right panel. **Those three are one design**, and `INTERACTION_PACK.md` now exists to
inform it.

**A standing product question I did not resolve.** He called Preprocess *"one of our moats"* and said
the two layers — per-feature-for-all-models and per-model — are not both present. The register mostly
answers it: **24 `prep-*` rows — 9 `both`, 6 `guided-only`, 3 `guided-native`, 6 `classic-only`** —
each with a dated reason, one of which is a deliberate refusal. So the capability is largely there and
the **arrangement** is what he is reacting to. Start from the register rather than from scratch.

---

## Calibration, and this is the part I would most want a successor to read

**I was corrected four times in one session, and every correction was the same error.**

1. **`AUDIT-030`'s premise** — I quoted an overclaim the row's own note had already struck. It was in
   the row when I wrote the ruling *and* when I wrote the prompt. I read `item` and `act` and skipped
   the note that demolished my argument.
2. **`/recipes reaches nobody`** — I repeated the agent's finding to the product owner **as fact
   without driving it.** False: the panel renders an honest placeholder before models are chosen and a
   full lattice after.
3. **`GUIDED-165`** — I filed a **critical** claiming two buttons record indistinguishably, on a
   reproduction that omitted a field the page actually sends. I approximated the wire instead of
   sending it.
4. **`GUIDED-161` and `GUIDED-173`** — I asserted "nothing reads this" twice; both times a consumer
   existed that I had not looked for.

**The pattern is not carelessness. It is that a finding which fits the story gets less scrutiny than
one that does not** — and all four fit. `LOOP.md` §06 already says *stop grepping and run it* and
*where a claim is about behavior, drive it*. I quoted that at the agent five times while breaking it.
**Drive the ones you believe, not only the ones you doubt.**

The counterweight: **this agent is good, and its divergence section has corrected the adjudicator six
loops running.** Read it first, every time. When it says it is unsure, it has usually already checked.

---

## Habits that are load-bearing

- **Write decisions into the docs the same turn they are made.** Three losses to ephemeral records so
  far, two of them product-owner rulings.
- **Add the `LOOP.md` §03 row when you accept a loop** — it is part of adjudicating. **L45 and L46
  have no row yet**; L45's is owed from my session, L46's follows its adjudication.
- **Never `git add -A`.** Stage explicit paths and run `git status` first, every time.
- **A file a tool owns has exactly one writer.** `ledger.py` serializes at `indent=1`; a script that
  dumps at `indent=2` silently reformats nine thousand lines. Both a PM and the agent did this.
- **Do not write to `findings.json` while a loop is running.** I did, at `42e4faa`, and named the
  violation in that commit's own subject rather than letting the row be swept into the executor's.
- **The check that fires most often:** was a named defect *class* filed, or only its instance?
- **Never accept a moved threshold in the same loop as the change that pressured it**, unless you are
  invoking §06.2's exception deliberately and saying so in those words.
- **Keep prose lean.** Docs run ~32k lines against ~37k of app code, and he has asked directly for
  minimum PM bloat without losing execution fidelity.
