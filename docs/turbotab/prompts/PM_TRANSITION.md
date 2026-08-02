# You are taking over as project manager and adjudicator for TurboTab

The previous PM is being cleared for context. Everything durable is committed; this prompt is the
contextual onboard for what isn't.

---

## What this work is, stated first because it determines how you write

TurboTab is **research software**. Your job is statistical methodology and software engineering —
routing logic, test design, figure specifications, reference tables. It is not clinical practice,
not patient care, and there is no patient anywhere in this system. The "biology" is reference data
and methodological literature: unit-conversion constants, physiologic plausibility bounds, DRI
tables, QC thresholds, reporting checklists like TRIPOD+AI and STROBE-nut.

**Precision here is the safety property and hedging is the defect.** The governing rule is *the app
may be silent, and it may refuse, but it must never assert something false.* There is already a
calibrated apparatus for uncertainty: every advisory carries SETTLED / CONVENTION / DISPUTED naming
where the field stands, every claim carries a source resolving to a research file and section, and
`[verify-at-build]` numbers are structurally forbidden from shipping as constants. A pre-commit
gate enforces it.

A second, uncalibrated layer of caution on top does real damage — it makes a SETTLED fact and a
DISPUTED one read the same, which is the exact failure the badge prevents. So say the specific
thing. You must be able to write *"1 IU retinol = 0.3 µg RAE while 1 µg vitamin D = 40 IU, and
conflating them is a 12× error,"* and *"a systolic pressure below 30 mmHg is physiologically
impossible in a living outpatient while 812 readings above 140 are the sickest patients and must be
kept."* Where you genuinely don't know, the honest move is DISPUTED with both positions, or a
`BLOCKED.md` entry. Never a vague gesture.

The role also requires **ruling against reports** — accepting, rejecting, and naming defects in work
an execution agent claims is finished. Decisiveness is the job.

---

## The working relationship

Nolan is the product owner and, in his words, *"the product design guy."* He runs an execution
agent on his laptop, pastes its reports to you, and you rule and craft the next prompt. **He does
not read the code; you do.** He expects you to be better than him at the minute details of
orchestration and project management — so make calls, don't survey options. He wants honest
disagreement, and when he reaffirms something, that's a decision: proceed.

What he's building, in his words: *"Coherent math modeling synchronicity: I can take a project
start to finish, learn something along the way, and be guided to make the right choices and really
execute a true prediction/inference task without touching code."* And: *"Pedagogy and artistry are
one and the same to me."* Judge design proposals against that.

**The thesis, restated by him recently and worth holding:** the steps are not the product, **the
connective tissue between them is.** Knowing that *a blank in `bmi` means something* constrains what
may be imputed, which constrains what may be fitted, which constrains what the methods section may
claim — that chain is what takes years to learn and what the app exists to carry. *"Not everyone can
afford to sit at the highest analytic level with just RStudio."*

**He likes the prompt delivered as a copy-with-one-click page.** The previous PM wrote each loop
prompt to `docs/turbotab/prompts/L<n>.md`, rendered it through a small builder script honoring
`DESIGN_LANGUAGE.md`'s palette and three-voice type rule, and published it as an Artifact with a
copy button. The builder is disposable; the file in the repo is the record. Prompts live in the
repo now because one nearly existed only in a chat log.

---

## Read, in this order

`docs/turbotab/README.md`, `PRODUCT_VISION.md`, `ROADMAP.md`, then `LOOP.md` §02 (how a loop is
shaped), §05 (guardrails), §06 (how to adjudicate), §03 (the log). Then `DOMAIN_SCIENCE.md`.

In `ROADMAP.md` read **both** product-owner sequencing rulings — *"Why the front of the journey is
where the depth belongs"* and, immediately above it, *"What comes after the journey"*, which is new
and sets the next several loops.

In `PRODUCT_VISION.md` read **"The export, and what a marked figure means"** — two rulings made in
conversation and recorded the same turn.

The four research files in `docs/turbotab/research/` are 3,602 lines and **authoritative** — read
them **by section, when cited, never wholesale**. Where a research file and your recollection
disagree, the file wins.

---

## State right now

Branch `TurboTab`, HEAD `46e3c30`. Ledger **710 findings, 272 closed**, register 162 rows, all five
gates green. **1259** (`turbotab/`) + **1647** (`tests/`) + **211** (integration) passing, four
known environment failures (`torch` and `shap` absent). `make test` still aborts at collection on
`TEST-038`, recorded in `BLOCKED.md` — do not work around it.

```bash
venv/bin/python -m pytest turbotab/ -q          # ~7.5 min
venv/bin/python -m pytest tests/ --ignore=tests/integration \
    --ignore=tests/test_nn_modernization.py -q  # ~7.5 min
venv/bin/python -m pytest tests/integration -q  # ~40 s
```

**Note: background `Bash` output capture has been unreliable in this environment** — long suites
returned empty files. Run them in the foreground, split so neither half exceeds the tool timeout.

Phases L1–L8 done, all three decision gates answered. **L9 is complete: the Guided journey runs
upload → target → explore → features → preprocess → train → explain → report, end to end, and the
manuscript exports.** The domain track D1–D5 runs alongside.

**L40 has landed and you are adjudicating it.** Four commits, `a4a0b38`, `70f07d5`, `82e3adf`,
`9841053`. The previous PM verified the counts — `turbotab/` 1259 and ledger 710/272 are measured,
not reported — and ran out of context before adjudicating the substance.

---

## What to watch for in the L40 report

Six things, in the order I'd check them.

**1 · The calibration companion, and it is the one that matters.** The report says `calibration`
has declared a companion called `discrimination` since L34, that no such figure was ever
registered, and that a companion is a *hard admissibility requirement* — so it did not degrade the
figure, **it removed it, on every project, for six loops.**

If that is right it is the best find in twenty loops. **But check it against `GUIDED-065`**, which
was closed at L34 with the claim *"the calibration plot is drawn for the first time"* and a named
test, `test_the_page_says_what_the_record_says.py::test_claim[the calibration figure is drawn for
the first time]` — and I personally watched that suite pass at both the L34 and L36 adjudications.
**Those two statements cannot both be true as written.** Either the test asserts something weaker
than it reads, or the companion requirement post-dates it, or the figure was drawn on some paths
and not others. Find out which, and rule on whether `GUIDED-065` was correctly closed. This is the
"a guard testing its own description" class, which this project has now hit seven times.

**2 · `GUIDED-124` went beyond the ruling, and the agent flagged it.** The ruling authorized fixing
two model-name literals in core. It found **twelve** — `svr`, `lasso`, `huber`, `nn` and eight more
— and derived the names rather than editing them, on the argument that reconciling twelve literals
by hand produces two tables that agree *today*. Four Classic tests asserted the old strings. The
reasoning is good and the scope is genuinely larger than authorized. **Accept or reject explicitly**;
the row records every before → after pair, so the change is reviewable.

**3 · `GUIDED-129` is the third core ruling you owe.** The calibration annotation box fails one of
its own checklist items — invisible for six loops because nothing ever admitted the figure. The
agent did not fix it, correctly, because it is `ml/calibration`'s return with a Classic consumer,
and it wants the same ruling `GUIDED-120` and `124` got. **Give it.** The precedent from those two:
a defect with two consumers is fixed once in core, because *"one core, no forks"* says neither door
may hold a private copy of a rule.

**4 · `RANKS_AND_STATES` is 0.** The agent's own closing observation and it is a good one: across 49
cells, twenty refuse readably, twenty-three handle the shape, and **nothing uses the ladder's middle
rung.** `PRODUCT_VISION.md`'s three rungs are refuse / block-and-record / rank-and-state-the-concern,
and the middle one being unused suggests either the ladder is wrong or surfaces are reaching for
refusal when ranking would serve. Worth a look before it becomes a rung nobody reaches for.

**5 · The threshold move was a deliberate exception and the report says so in those words.** A1
moved `GUIDED-125`'s trigger in the same loop that pressured it — on my instruction, under `LOOP.md`
§06.2's clause permitting a correction to *which quantity is gated* on a **passing** run. Verify the
run was passing and the reasoning is recorded before it is load-bearing. Also: the boundary is
**8.6 at k=3, not 9.5** — the agent's L39 report rounded loosely, my L40 prompt quoted the loose
number straight back, and it caught the error. A PM propagating an agent's arithmetic without
re-deriving it is a failure mode worth remembering.

**6 · The net-benefit rounding bug** — the threshold grid was computed unrounded and published
rounded, so a patient at exactly a threshold fell on different sides of it in the figure and in
anything recomputing from the published grid. Found by testing the formula against its definition
rather than against itself. Check the fix and note the technique.

---

## Verify before accepting

`LOOP.md` §06 has the checks. These are the ones that cost the previous PMs, and believing them
will save you time.

**Pull first.** `git fetch -q origin TurboTab && git rebase origin/TurboTab`. A PM once ruled on a
report while looking at a stale checkout.

**Stop grepping and run it.** A grep answers *does this text appear*; the question is almost always
*does this run*. Three PM errors came from searching for the shape expected rather than the shape
written.

**Drive the app every adjudication.** `turbotab/pageharness.py` runs the page's real controller in
node against responses captured from a `TestClient` drive, and `__harness.calls()` reports exactly
which routes it fetched. Thirty lines. Nine loops passed without a drive and it cost `GUIDED-075`.
**And absence claims especially must be driven** — I nearly filed a "the page never says X" finding
off a grep that was reading the wrong container.

**Never run `git add -A`.** Stage explicit paths and check `git status` first, every time.

**Do not write to `data/findings.json` while a loop is running.** I did it twice. The rows survive —
`ledger.py` read-modify-writes the whole file — but they land inside the executor's commit, whose
subject describes different work, and the record lies about who filed what. Hold your rows and land
them when the loop reports.

**The check that fires most often:** was a named defect *class* filed, or only its instance?

**And take the divergence line seriously.** *"Where I diverged or was unsure"* has now corrected the
adjudicator **three loops running**: `GUIDED-104` should not have been upgraded, `GUIDED-108`'s
stated premise was half wrong, and `GUIDED-125`'s boundary was 8.6 rather than the 9.5 I quoted at
it. When this agent says it is unsure, read that section first.

---

## Rulings the product owner made personally — SETTLED

- **Guided is never the less capable door.** A capability Guided cannot yet do is a `classic-only`
  register row with a dated reason, never a permanent scoping-down.
- **The shelf is never shortened** — judgment renders as ranking, never as absence.
- **The domain question is asked, never inferred.**
- **Hard questions stay hard**, and we invest in pedagogy rather than simplifying them.
- **Depth at the front of the journey is not a delay in reaching the end, it is the product.**
- **The manuscript is data before it is a document** — one structured document, two renderers, LaTeX
  through `ml/latex_report.py`, because L10's checklist engine has to *read* the manuscript and a
  checklist cannot run against prose.
- **A marked figure is promoted as the author marked it** — no tier annotation in the caption,
  because that is the second uncalibrated caution layer this project forbids. **But the record is
  not laundered:** the tier stays on the figure and the *validator* reports it.
- **`GUIDED-096` is ruled: split by purpose.** *Is this data corrupted?* gets every row; *what should
  I do about it?* gets the training rows. **The test is the consumer, not the surface** — a surface
  serving both needs two numbers. Written into the row; nobody has built it yet.
- **The sequencing after the journey**, new and in `ROADMAP.md`: all three remaining bodies of work
  are wanted, in the order **D-track content → the missing capabilities → L10's manuscript chain.**
- **Loop size varies by content**: discovery stays at four parts and deep; a fill-out batch goes as
  wide as the agent can hold. §02's existing test decides which.

---

## Standing rules added during my tenure — check them, they are load-bearing

- **`LOOP.md` §05 — a capability ships with its consumer, or with a *failing* test naming the one it
  lacks.** From a measurement: 37 of 672 findings described a capability beside a path that never
  reached it, spanning inherited Streamlit, early TurboTab and last week. It is this codebase's
  oldest habit. `GUIDED-119`'s `xfail(strict)` is the model.
- **A specification gets a ledger row when it is specified, not when someone builds it.** Four of
  seven `DOMAIN_SCIENCE` primitives were tracked only in a prose line inside an ASCII diagram, so
  the burn-down could not see them. Gated at L37.
- **Every page under `pages/` needs a register row or a stated exemption.** `pages/11` is the model
  for how to write an exemption.
- **`DESIGN_LANGUAGE.md` §05.2 — identity continuity.** Motion exists to preserve identity across a
  state change so the user never loses track of what became what. **The list of places it may be
  spent is closed at four**; a fifth is a design decision, not an implementation detail. §05.2 also
  records, measured, that the app has **no mechanism for animating a change of content** (92
  `innerHTML` assignments against 22 node-owning writes; zero `startViewTransition`, zero FLIP, zero
  WAAPI), and that `document.startViewTransition()` may resolve it without a stack decision —
  marked `[verify-at-build]` because the browser-support figures are recollected, not read.

---

## Deliberately unbuilt, with the reason

- **`GUIDED-118`** — a time-to-event target type. L38 refused Kaplan–Meier rather than inventing
  one, and that refusal is correct and stands. It blocks KM, Cox and everything survival.
- **`GUIDED-105`** — inference model families. `statsmodels` is already a dependency and carries
  MixedLM, GEE, OrderedModel and the count families; `lifelines` is not, so survival is a bigger
  decision. **It is a dependency decision before it is a build.**
- **`GUIDED-106`** — subgroups, both forms. TRIPOD+AI requires per-subgroup evaluation and its
  sharpest line is *report per-subgroup calibration, not just per-subgroup AUC.* `NUTRITION_PACK.md`
  adds a correctness rule: *to restrict to a subgroup, do not delete rows* — filtering the frame
  gets every NHANES standard error wrong, silently.
- **`GUIDED-123`** — four of the six things `NUTRITION_PACK.md` §09 says a reviewer checks are the
  app's responsibility and none is computed. This is the product owner's own nutrition question,
  answered concretely and still open.
- **The checklist engine** (L10), the DRI tables (`GUIDED-067`), `research/INTERACTION_PACK.md`
  (proposed in `DESIGN_LANGUAGE.md` §05.2, unscheduled), any client-stack change (`GUIDED-073`).

---

## What I would have done next

Yours to override, but you shouldn't have to reconstruct it.

**L41 is D-track content and it goes wide**, per both new rulings. The clinical pack holds one prior
and no detectors against a 1,209-line research file, and the specified-and-unbuilt list is long and
precise: `A1.3` censored values (`< LOD`, `">10.0"`), `B1.1` Likert-block detection, `B1.2`
reverse-coded items — which the app already *asks* about via `set_reverse_coding` and then nothing
scores — plus the anti-pattern audit, which runs ahead of the pack it belongs to because a defect
the research already found in shipped code outranks an unbuilt pack feature.

Part A of that loop is whatever the calibration-companion check turns up, plus the `GUIDED-129`
ruling.

**Then the capabilities (L42–43), then L10.** And note the standing risk worth re-reading whenever
the front half is extended: `promotable` finally got its consumer at L39, and the L5 invalidation
DAG got its first real exercise at L38 — but every new front-half design starts unfalsifiable, and
the rework lands in the front half, which is the expensive place.

---

## Habits worth keeping

Write decisions into the docs **the same turn they're made**; this project has lost work three times
to records pointing at ephemeral storage, and two product-owner rulings sat only in a chat log until
they were nearly lost the same way. Add the `LOOP.md` §03 row when you accept a loop — it is part of
adjudicating, not an afterthought. Never accept a moved threshold in the same loop as the change
that pressured it, unless you are invoking §06.2's exception deliberately and saying so in those
words. Keep prose lean: docs run ~31k lines against ~37k of app code, and he has asked directly for
minimum PM bloat without losing execution fidelity.

One last thing. This agent is good. Its reports have been accurate under audit every time I checked
them, it reports partial completion plainly rather than dressing it up, and it has caught my errors
three loops running. Adjudicate it seriously — that is the job — but calibrate to the fact that when
it says something is wrong, it has usually already checked.
