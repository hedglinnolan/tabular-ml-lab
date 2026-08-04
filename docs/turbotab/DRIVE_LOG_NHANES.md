# Drive log — the real NHANES export, 2026-08-04

The running companion to [`DRIVE_PREREG_NHANES.md`](DRIVE_PREREG_NHANES.md), which is **sealed and
not edited**. Same split the routing value check used: the prereg is frozen before the measurement,
this file is the narrative afterwards, and anything the prereg got wrong is corrected here rather
than there.

**The drive.** The product owner driving
`nhanes_1999_2018_yayhoo_fasting_diet_imputed.csv` — **21,849 × 29**, real, pooled across nine NHANES
cycles — through the Guided door at `127.0.0.1:8000`, reporting by screenshot, with the adjudicator
reproducing each observation before filing it.

**Why this file exists.** Every fixture in this repository is synthetic and under 600 rows.
`DESIGN_LANGUAGE.md` §05 is explicit that a lesson learned on synthetic data is a hypothesis until a
real dataset has seen it, and this is the first time one has.

---

## What was verified before he started, and told to him

- **It loads.** 4.4 MB, well inside the 64 MB ceiling; upload to profiled findings in ~1.4 s.
- **No performance wall at 36× the largest fixture.** Upload → sealed lockbox in under 3 s, holdout
  3,277 rows; two models trained on 21,849 rows in **1.2 s**; Explain ranks `triglycerides`, `age`,
  `waist`, `bp_sys`, `sugar` — clinically sensible for fasting glucose.
- **The L45/L46 attention work does not render on this file.** 14 findings, 5 reach the Explore
  stack, bound is 5, so nothing collapses. Told to him because withholding it would have cost a
  wasted drive; it is operational rather than a finding.

Everything else the pre-flight turned up was withheld and is in the prereg.

---

## The findings, in the order he met them

### `GUIDED-156` · `high` · "Ask me anyway" reopens a question that renders nowhere

**His observation.** *"I wanted to see what happened if I clicked ask me anyway… when I went to
click the very bottom card, it just dismissed itself."*

**Reproduced through the page's own controller.** The server is correct: `unskip` returns 200 and
`confirm_task_type` comes back `status: skipped → asked`, `skip_reason` cleared, both options
present. The page is not: `askedQuestions` is **byte-identical at 12,516 characters** before and
after, and `skipNote` drops from 816 characters to **zero**.

**Two individually correct rules composing into a hole** — the shape this repository finds most
often, and **the first time a user found it rather than a sweep.** The skip renderer draws
`status === "skipped"`. `renderAsked` draws `status === "asked" && !handledElsewhere(key)`.
`confirm_task_type` is on `HANDLED_QUESTION_KEYS` as *"the task-type row inside the target card"* —
so once reopened it is asked **and** handled elsewhere, and neither renderer takes it. The row that
supposedly handles it renders the decision *sentence*, which carries no answer control.

**It is `GUIDED-041`'s defect returning through a different door.** That fix made the button send
`unskip` instead of answering with the engine's own guess, and left this comment in the page:
*"A rendered skip whose reopen affordance discards is worse than no affordance at all — it teaches
that opening a skip loses your place."* The wire was corrected; the user still loses their place.

**The coverage gap is the useful part.** `test_ask_me_anyway_reopens_the_question.py` holds six
tests, including one that dispatches the real click and asserts the request body. All six pass and
every assertion in them is true. **They watch the wire and the record; none renders the page after
the reopen and asks what a person now sees.**

### `GUIDED-157` · `high` · the binary repair records which columns, never which value became 1

Found one click later, driving what he had just applied. `apply_bulk` over the nine-member group
records exactly:

> `9 features (imputed_bmi, imputed_bp_di, imputed_bp_sys, imputed_height, imputed_waist,
> imputed_weight, meds_chol, meds_hbp, gender) were read as binary.`

`params` is `None` and no mapping appears anywhere in the decision. The source values are `gender ∈
{female, male}`, so **"is the coefficient on male or on female" has no answer in the record.**

**This is the product thesis failing at its own load-bearing claim.** The transcript and the
manuscript are meant to be one object at two levels of formality; a methods sentence a reader cannot
reproduce from is not a methods sentence. It produces no wrong number and it makes a reported number
**uninterpretable**, which is worse in the artifact that leaves the building — and the shelf ranks
`ridge` and `glm` for this target. Permutation importance is direction-free, so it survives to
Explain unnoticed.

### `GUIDED-158` · `medium` · nine assertions, eight of them false about the data

The card titles every hit *"is a binary variable **written as text**"*. Measured on his file: six of
the nine are dtype **`bool`**, two are `object` holding Python `True`/`False` (object only because
nulls forbid a bool column), and **only `gender` holds strings.**

Nothing downstream is wrong — the repair is correct for all nine. **The cost is trust, which is what
the whole apparatus of badges and sources is spent buying.** It is the first card a user meets on
this file, it makes nine claims, and eight are wrong about something the reader can check in one
glance at their own CSV. A tool wrong about what you can verify instantly is a tool you stop
believing about what you cannot.

---

### `GUIDED-159` · `high` · the map cannot say "you are here" for five of eight steps

*"Every step is lit up except train. And there is no option to train."* `setMap("train", …)` is the
only such call in the file and is guarded on a run existing, so Train goes from classless straight to
`done`. Only `target` and `eda` can ever be `now`. And `DESIGN_LANGUAGE.md` §04 specifies four states
where the CSS defines three — there is no `waiting`, so a step never reached and a step that is next
render identically. He read the absence as a verdict about availability, which is what it looks like.

### `GUIDED-160` · `high` · two of four education layers shipped

*"Even when hovering over the blurbs… I am not sure I would understand the difference."* Measured:
card rationale 34 words (layer 1, right); the "Why we ask" disclosure **135 words in five sentences
mentioning none of his columns**, where §10 caps layer 2 at two or three; and **zero** hits for any
side panel, `theory_anchors`, or "Save to my review" — layers 3 and 4 and both binding mechanics
absent, with layer 2 carrying layer 3's job.

**§10 named the failure before it happened**: layer 1 cites *expertise reversal*, layer 2 cites
*split attention*. A long abstract disclosure breaks both at once. What it contains is an inventory
of the app's own implementation sites — right for an auditor, wrong for someone deciding.

### `GUIDED-161` · `high` · earmark records and nothing reads it — while defer beside it works

`defer` lands in `deferred_noticings` and the dock renders it. `earmark`, **in the same card**,
changes it not at all. Two controls that look identical have different fates, and both promise to
come back at a step. Worse than a uniformly dead dock: the working one teaches that the mechanism
exists.

### `GUIDED-162` · `medium` · the preview names an apply that does not exist

*"this is what pressing apply would do"* has zero hits in the page — it is the server's text — and
there is no apply handler anywhere. `GUIDED-080`'s class inverted: the interface renders the server's
string, and the string names a control the interface never built.

### `GUIDED-163` · `high` · median fill offered on a column the constitution blocks

`meds_hbp` observed `{True: 5527, False: 770}` with 15,552 missing; the median is 1, so the operation
assigns **every person of unknown medication status to being on blood-pressure medication**, taking
the column to 96.5% ones with *"not asked"* and *"yes"* encoding identically. Driven:
`blocks('informative','impute_median')` → `True`. It sits second in a list of two under a heading
reading **"What the app can do."** He called it *"possibly a bad idea"* from his own knowledge.

### `GUIDED-164` · `high` · 3 plots for a 15-column finding, and the pager he asked for is built

`slice(0, 3)` for plots and `slice(0, 5)` for chips, neither labeled — the card announces fifteen and
shows three. Meanwhile `/evidence/histograms` returns `n_features: 26, per_page: 6, n_pages: 5` and
the click handler already has `data-hist-page`. **Not unreachable — reachable from everywhere except
the card that motivates it.**

And the half he felt without naming: the spec says a finding carries **its** plot, singular, the
image that makes *that* case. Three histograms of the first three columns are evidence *adjacent to*
the claim rather than *for* it, which is why they read as decoration.

### `GUIDED-165` · `critical` · the record asserts an operation that did not happen

*"Clicking 'set these entries to missing' does nothing."* The plausibility endpoint reports **125
flagged `bp_di` entries before the decision and 125 after** — while the transcript carries, verbatim,
*"Entries of `bp_di` outside the impossibility band were set to missing."* `AUDIT-001`'s shape at the
decision layer. The tell is the asymmetry: *"Keep as is"* records *"were kept as recorded,"* which is
**true**, because nothing happened.

**Corrected in place, and the correction is to the adjudicator's own reproduction.** This was first
filed claiming the two buttons record indistinguishably. False — `decide()` copies the sentence to a
top-level `text` field and the test sent only the nested one, so the server used its slug fallback
and that was read as the app's behavior. **The wire was approximated instead of sent**, which is the
exact failure this project's execution agent has been warned against five times, committed at
critical severity.

### `GUIDED-166` · `high` · the impossibility pass manufactures blanks nothing can distinguish

*"Didn't we just settle how to handle missingness as informative or not?"* Explore asks the mechanism
question; the impossibility card two cards later proposes to **create** blanks. Nothing marks a
blank's provenance, so a declaration about *"never asked"* governs *"the app deleted a corrupt
reading"* — opposite correct handling.

**His three instincts map onto three real routes and the app offers one.** Set to missing is clause
06's row-local repair. Exclude the rows is clause 04's eligibility criterion — and `api.py:654` says
so itself: *"Offered, never applied. An exclusion changes N and is reported in participant flow."*
Mark the column corrupted is `GUIDED-096`'s split, and at 0.57% it is the wrong call, which the app
could say.

### `GUIDED-167` · `high` · the app's best behavior is its most invisible

*"It appears the app does nothing when I click those buttons."* Driven: `impute_median` after
declaring informative returns **409 with a typed blocker** naming the column and counting the 17,204
blanks; `drop_rows` returns **400** with the complete-case explanation. **Both exactly right. Neither
reached him.** The page has a refusal renderer, so something between the throw and the eye drops it —
either the detail is not attached to a 409, or the banner renders above a viewport he is two thousand
pixels below. The harness proves what the controller renders and states that it cannot prove
visibility; this one needs the driver.

**And it explains two earlier reports.** A user cannot distinguish *"the control is not wired"* from
*"the app refused and did not tell me"* — and this drive produced both.

### `GUIDED-168` · `high` · a borrowed label promising a different analysis

*"Missingness pattern analysis to me would mean show me the co-missingness pattern visually."* The
title is the **core's**: `ml/eda_recommender.py:292` defines it with a stated deliverable including
*patterns suggesting MCAR vs MAR vs MNAR*. The Guided endpoint returns two per-column cards asking
*"is the missingness informative?"* — and the server's own capability table calls that endpoint
**"Missingness by feature,"** which is exactly what it returns. The page took the label for analysis
A and pointed it at analysis B.

### `GUIDED-169` · `medium` · six of ten palette entries are NOT BUILT, and the organizing idea

Built: plausibility, missingness, collinearity, distributions. Not built: target distribution,
dose-response, stratified trends, outlier influence, quick baselines, reverse-coding audit — and
those titles resolve to the core recommender, with `pages/02_EDA.py:1782` running dose-response
**today**. `MISC-014` applies before the mistake this time: **unrouted is not absent.**

**His organizing idea is the part that matters**: bin the palette by the *question* rather than by
the geometry. The current one mixes both. §01 names combinatorial pathway explosion as the root cause
this product exists to fix and says the answer is *more opinionation* — a flat palette grown to
sixteen entries is the eleven-page maze reappearing as a chip row.

### `GUIDED-170` · `critical` · a SETTLED nutritional claim about a row identifier

He selected **`SEQN`** in the nutrient dropdown and pressed Ask. The app answered *"Prevalence of
inadequacy for `SEQN` is computed by the EAR cut-point method"* with a **SETTLED** badge. Driven:
`refused: false`, `evidence_status: "SETTLED"`, `may_preselect: true`, `source:
research/NUTRITION_PACK.md#07`.

**On the surface built to demonstrate refusal.** `LOOP.md` §04 records that nutrition went first
*"because it is the one pack that forces a refusal"*; `GUIDED-080` called this endpoint *"the refusal
apparatus the whole domain-track ordering was justified by."* Its four refusals check the **basis**
and the **reference kind** and never check whether the subject is a nutrient — so they are complete
along the axes they know, and there is no axis for the thing the user actually got wrong.

**SETTLED is what makes it critical.** `DOMAIN_SCIENCE.md` §01: *"Methodological consensus. A tool
asserting the opposite would be wrong."* It is the one status that may pre-select, and
`may_preselect` came back true. The evidence gate passed it because the gate verifies a source is
named and resolvable and is honest that it can never check the claim is faithful. **This is that
stated limit meeting a real dataset.**

---

## Prereg predictions, as they resolve

| Prediction | Status |
|---|---|
| `SEQN` unrecognized as an identifier, offered as a target and a predictor | **Confirmed, and worse than predicted** — offered as the first target chip unmarked, and then offered as a *nutrient*, producing `GUIDED-170`'s SETTLED claim. The prereg expected it to reach the model; it reached the refusal apparatus. |
| No survey design present and nothing says so | **Confirmed** — never mentioned at any step he drove |
| Nine pooled cycles, nothing notices | **Confirmed** — `cycle_begin_year` was offered as an ordinary predictor and never remarked on |
| The six `imputed_*` flags read as ordinary binaries | **Confirmed, and it went further than predicted** — see `GUIDED-158`, which the prereg did not anticipate |

---

## What this drive has already established

**Three findings in the first two steps**, none of which any sweep, probe or suite in this repository
had produced — and the app has been swept at three granularities (route, field, name) in the last
four loops.

The common shape is not a missing capability. **Every one is a seam between two correct things** — two
renderers that each decline a question, a decision that records the columns but not the mapping, a
detector whose repair is right and whose sentence is wrong, a blocker that fires perfectly into
silence, a label borrowed from the analysis next door.

**Fifteen findings, two critical, in roughly two hours.** Four granularities of automated sweep —
route, field, name, stand-in — had run over this door in the four preceding loops and found none of
them, because not one is a thing that is *absent*. That is the connective tissue the product owner
named as the actual product, measured for the first time by walking it.

**And the adjudicator was corrected twice by the drive**: once by the agent's report (`AUDIT-030`'s
struck premise) and once by its own bad reproduction (`GUIDED-165`). Both are recorded where they
happened rather than quietly fixed.
