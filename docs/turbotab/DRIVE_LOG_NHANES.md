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

## Prereg predictions, as they resolve

| Prediction | Status |
|---|---|
| `SEQN` unrecognized as an identifier, offered as a target and a predictor | **Confirmed in his screenshot** — `SEQN float64` is the first target chip, unmarked. Not yet remarked on by him. |
| No survey design present and nothing says so | Pending — he has not reached a surface that would say it |
| Nine pooled cycles, nothing notices | Pending |
| The six `imputed_*` flags read as ordinary binaries | **Confirmed, and it went further than predicted** — see `GUIDED-158`, which the prereg did not anticipate |

---

## What this drive has already established

**Three findings in the first two steps**, none of which any sweep, probe or suite in this repository
had produced — and the app has been swept at three granularities (route, field, name) in the last
four loops.

The common shape is not a missing capability. Every one is a **seam between two correct things**: two
renderers that each decline a question, a decision that records the columns but not the mapping, a
detector whose repair is right and whose sentence is wrong. That is the connective tissue the product
owner named as the actual product, failing in three different places, found by pressing buttons in an
order nothing tests.
