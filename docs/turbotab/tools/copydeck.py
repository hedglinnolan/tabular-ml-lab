"""Assemble every user-facing string in the Guided door into a reviewable deck.

    python docs/turbotab/tools/copydeck.py regen    # rewrite COPY_DECK.md
    python docs/turbotab/tools/copydeck.py check    # non-zero if it has drifted

It exists so copy can be reviewed **asynchronously, without running the app** —
the product owner should not have to drive a FastAPI server to read a sentence.

## Why this is half generated and half hand-assembled

Generation was attempted first and is only partly possible. Measured across the
Guided door:

| source | walkable by a tool | inline at a call site |
|---|---|---|
| `grain.py` | 16 | 0 |
| `features.py` | 20 | 9 |
| `selection.py` | 6 | 7 |
| `router.py` | 8 | 9 |
| `api.py` | 8 | 48 |
| `project.py` | 1 | 32 |
| `web/index.html` | 0 | 51 (markup + JS literals) |

The catalogues — `features.CATALOGUE`, `selection.METHODS`, `router.plan()`'s
questions, `grain`'s answers, exits and disclosures — are data, so they are
**generated**: this file imports them and prints what is actually there, and
they cannot drift because there is nothing to drift from.

The rest are f-strings raised at ~105 call sites and string literals inside
markup. Extracting those reliably would mean either an AST walk that cannot
resolve the interpolations, or a runtime harness that drives every error path —
both of which produce a *worse* artifact than transcribing them, because a
half-resolved f-string is not reviewable copy.

**That difficulty is a finding, not a workaround** (`GUIDED-013`). Copy that
lives at its raise site cannot be reviewed, translated, or kept consistent, and
the deck is the symptom rather than the cure.

## What stops the hand-written half drifting

A hand-maintained deck drifts. So each hand entry carries the file it came from
and a `probe` — a distinctive fragment of the real string — and `check` asserts
that fragment is still in that file. Change the copy without updating the deck
and `check` goes red.

That is weaker than generation: it catches a string that *changed* and not one
that was *added*. It is stated as weaker rather than sold as equivalent, and
`GUIDED-013` is the row for closing the gap properly.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "docs", "turbotab", "COPY_DECK.md")


# ─────────────────────────────────────────────────────────────────────────────
# The hand-assembled half. Each entry names where the string lives and a probe
# `check` can look for. Adding one is a decision; forgetting to is a red test.
# ─────────────────────────────────────────────────────────────────────────────

HAND: List[Dict[str, Any]] = [
    # ── Data & Target · upload ───────────────────────────────────────────────
    dict(step="Data & Target", state="upload · unreadable file",
         trigger="`engine.read_table` cannot parse the upload",
         copy="'{filename}' parsed to {n} rows and {c} columns. There is nothing to diagnose.",
         source="turbotab/engine.py", probe="There is nothing to diagnose"),

    # ── Data & Target · the seal ─────────────────────────────────────────────
    dict(step="Data & Target", state="seal · attempted before the grain question",
         trigger="`POST /decision {kind: seal}` while `project.grain is None`",
         copy="The grain question comes before the seal: whether one person can "
              "appear in more than one row decides how the held-out rows are chosen.",
         source="turbotab/api.py", probe="The grain question comes before the seal"),
    dict(step="Data & Target", state="seal · attempted with no target",
         trigger="`POST /decision {kind: seal}` while `project.target is None`",
         copy="The held-out set is drawn against the outcome, so the target comes first.",
         source="turbotab/api.py", probe="the target comes first"),
    dict(step="Data & Target", state="seal · project-level refusal",
         trigger="`AnalysisProject.seal_lockbox` with no recorded grain",
         copy="The test set cannot be sealed before the grain question is answered: "
              "whether one person can appear in more than one row decides how the "
              "held-out rows are chosen. Constitution §01 fixes that order, and §02 "
              "is why.",
         source="turbotab/project.py", probe="cannot be sealed before the grain question"),
    dict(step="Data & Target", state="seal · a second seal attempted",
         trigger="`seal_lockbox` when `barrier_raised` is already true",
         copy="This project already has a sealed test set. Redrawing it would "
              "re-partition the study: rows sealed since upload would become "
              "trainable and earlier results would no longer be comparable.",
         source="turbotab/project.py", probe="already has a sealed test set"),
    dict(step="Data & Target", state="seal · attempted before eligibility",
         trigger="`POST /decision {kind: seal}` while `project.eligibility is None`",
         copy="The eligibility question comes before the seal: whether your study "
              "is restricted to part of this data decides which rows the held-out "
              "set is drawn from. Answering 'the study is about everyone here' "
              "settles it.",
         source="turbotab/api.py",
         probe="whether your study is restricted to part of this data"),
    dict(step="Data & Target", state="eligibility · project-level refusal at the seal",
         trigger="`AnalysisProject.seal_lockbox` with no recorded eligibility answer",
         copy="The test set cannot be sealed before the eligibility question is "
              "answered: whether your study is restricted to part of this data "
              "decides which rows the held-out set is drawn from. Constitution §01 "
              "puts eligibility between the grain and the seal, and §04 is why — an "
              "exclusion applied afterwards would mean the held-out rows came from "
              "people the study is not about. Answering 'the study is about everyone "
              "here' is a recorded answer and settles this.",
         source="turbotab/project.py",
         probe="puts eligibility between the grain and the seal"),
    dict(step="Data & Target", state="eligibility · asked before the grain",
         trigger="`set_eligibility` while `project.grain is None`",
         copy="The grain question comes before eligibility: constitution §01 fixes "
              "the order as grain, then eligibility, then the seal.",
         source="turbotab/project.py",
         probe="fixes the order as grain, then eligibility, then the seal"),
    dict(step="Data & Target", state="eligibility · restricted after the seal",
         trigger="`set_eligibility` when `barrier_raised` is true — §04's "
                 "*permanently off the menu*, routed rather than refused flat",
         copy="The test set is already sealed, so an eligibility criterion cannot "
              "be applied now: the held-out rows were drawn from a population that "
              "included the rows you are excluding. Constitution §04 routes this "
              "back to the pre-seal question, which needs a re-seal — and a re-seal "
              "re-partitions the study.",
         source="turbotab/project.py",
         probe="routes this back to the pre-seal question"),
    dict(step="Data & Target", state="eligibility · a restriction with no reason",
         trigger="`set_eligibility(restricted, ...)` with an empty `reason`",
         copy="An exclusion criterion needs its reason. Participant flow reports "
              "how many rows were excluded AND why; a criterion with no reason "
              "cannot become a methods sentence, and one that cannot be written "
              "down should not be applied.",
         source="turbotab/eligibility.py",
         probe="cannot become a methods sentence"),
    dict(step="Data & Target", state="eligibility · a restriction with no range",
         trigger="`set_eligibility(restricted, ...)` with no minimum, maximum or "
                 "values to keep",
         copy="A restriction needs a range or a set of values to keep. Without one, "
              "the honest answer is that the study is about everyone here, which is "
              "its own recorded answer.",
         source="turbotab/eligibility.py",
         probe="which is its own recorded answer"),
    dict(step="Data & Target", state="eligibility · the criterion empties the study",
         trigger="the criterion keeps zero rows",
         copy="That criterion removes every row ({n} of {n}). Either the range is "
              "wrong or the column is not what it looks like — nothing downstream "
              "can run on an empty study.",
         source="turbotab/eligibility.py",
         probe="nothing downstream can run on an empty study"),

    # ── Explore · the robustness trim (clause §05, arming half) ─────────────
    # ── Preprocess · refusals at the project boundary ────────────────────────
    dict(step="Preprocess", state="missingness · a column with no blanks",
         trigger="`route_missingness` on a column that is complete",
         copy="`{column}` has no missing values, so there is no missingness to "
              "route. Asking about a column that is complete would be the "
              "interview inventing work.",
         source="turbotab/project.py",
         probe="interview inventing work"),
    dict(step="Preprocess", state="missingness · an unknown mechanism",
         trigger="`declare` with a mechanism outside the three answers",
         copy="'{mechanism}' is not one of ['informative', 'not_informative', "
              "'not_sure']. The mechanism is asked, never inferred — `not_sure` "
              "is a real answer.",
         source="turbotab/missingness.py",
         probe="asked, never inferred"),
    dict(step="Preprocess", state="missingness · the indicator column name is taken",
         trigger="`route_missingness` with `indicator` when `{column}_was_missing` exists",
         copy="'{name}' already exists in this table. Remove it first, or the "
              "indicator would silently replace it.",
         source="turbotab/project.py",
         probe="or the indicator would silently replace it"),
    dict(step="Preprocess", state="settled · the step was worked",
         trigger="`settle_preprocess()` after at least one column was routed",
         copy="Missingness settled: {k} column(s) changed now, {n} recorded to "
              "be fitted inside the training folds, {m} deliberately left alone.",
         source="turbotab/missingness.py", probe="Missingness settled: "),
    dict(step="Preprocess", state="settled · the step was skipped",
         trigger="`settle_preprocess(skipped=True)`",
         copy="Preprocessing was skipped; no missingness routing was recorded "
              "and every column goes forward as it is.",
         source="turbotab/project.py",
         probe="every column goes forward as it is"),
    dict(step="Preprocess", state="settled · why nothing visibly changed",
         trigger="the step is settled and at least one strategy deferred — the "
                 "honest report of a step whose output is decisions rather than "
                 "a changed table",
         copy="Your table looks the same because it is the same. Filling a blank "
              "with a median means computing that median, and computing it over "
              "every row would compute it over the held-out rows too — so the "
              "decision is recorded now and the arithmetic happens inside each "
              "training fold, where it can only see training data. What you just "
              "did is the part that cannot be automated; what is left is "
              "bookkeeping the pipeline does on its own.",
         source="turbotab/missingness.py",
         probe="the part that cannot be automated"),
    dict(step="Preprocess", state="settled · nothing was deferred",
         trigger="the step is settled and every strategy was row-local or leave",
         copy="Nothing was deferred, so nothing is waiting: every answer here "
              "either changed the table or deliberately left it alone.",
         source="turbotab/missingness.py",
         probe="either changed the table or deliberately left it alone"),
    dict(step="Preprocess", state="settled · columns still unanswered",
         trigger="the step is settled while a column with blanks was never routed",
         copy="{n} column(s) with missing values have not been answered yet.",
         source="turbotab/missingness.py",
         probe="with missing values have not been"),
    # ── Explore · the bounded findings stack (`GUIDED-149`) ─────────────────
    # Four states, and the fourth is the one that is easy to leave out. The
    # sentences are composed by the SERVER and quoted by the page (§05.1 rule
    # 3), so a count on screen cannot disagree with what is behind it.
    dict(step="Explore", state="findings stack · more than the bound was found",
         trigger="`attention.stack` collapses anything — the affordance's own "
                 "sentence, counted and typed by severity",
         copy="{n} more — 3 warnings, 4 cautions",
         source="turbotab/attention.py", probe='headline = f"{n} more — "'),
    dict(step="Explore", state="findings stack · the remainder, typed by stream",
         trigger="the line beneath the affordance; the profile speaks about the "
                 "table and a pack speaks about the field",
         copy="2 from the clinical lens · 5 about this table",
         source="turbotab/attention.py", probe='return f"from the {label}"'),
    dict(step="Explore", state="findings stack · the affordance's stated effect",
         trigger="hover and screen reader, per §05.1 — the control states what "
                 "it will do, and never as a bare verb",
         copy="Adds {n} more findings to this list. They ranked below the top "
              "{bound}; nothing is out of the record either way.",
         source="turbotab/attention.py",
         probe="nothing is out of the record either way"),
    dict(step="Explore", state="findings stack · nothing was collapsed",
         trigger="`attention.stack` collapses nothing. §09's recorded-absence "
                 "rule: without this a reader cannot tell *this is everything* "
                 "from *this is the top few*, which is two claims rendering as "
                 "one",
         copy="All {n} shown.  ·  Nothing stood out in the profile or under the lens.",
         source="turbotab/attention.py",
         probe="Nothing stood out in the profile or under the lens"),

    # ── Explore · the card that arrived because a slot opened (`GUIDED-154`) ──
    dict(step="Explore", state="findings stack · a card was promoted into a "
                               "cleared slot",
         trigger="a dismissal or deferral frees budget and the next collapsed "
                 "finding is pushed. §09's recorded-absence rule from the other "
                 "side: an object appearing without explanation is as "
                 "unexplained as one vanishing without it",
         copy="Moved up when you dismissed a card above.  ·  …deferred…  ·  "
              "…cleared… (when both kinds occurred)",
         source="turbotab/attention.py",
         probe='promoted_because = f"Moved up when you {verb} a card above."'),

    # ── Cross-step · what came back here (`GUIDED-153`) ─────────────────────
    dict(step="Explore", state="deferral · the attribution it comes back with",
         trigger="a deferred noticing renders at the step `ml.router.PACK_DEFER` "
                 "sent it to. `PRODUCT_VISION.md` §04: *pre-checked and "
                 "attributed*, and the attribution is this sentence",
         copy="You set this aside at Explore. {Step} is the step that can act on it.",
         source="turbotab/attention.py",
         probe="is the step that can "),
    dict(step="Explore", state="deferral · deferred with no destination",
         trigger="a finding deferred while carrying no `defer_target`. Reported "
                 "rather than filed under a step nobody chose — `GUIDED-153` "
                 "was exactly that silence",
         copy="This was set aside and nothing recorded where it comes back, so "
              "it cannot be shown at a step. That is a defect rather than a state.",
         source="turbotab/attention.py",
         probe="That is a defect rather than a state"),

    # ── Explore · the impossibility repair, which now happens (`GUIDED-165`) ──
    dict(step="Explore", state="impossible entries · set to missing",
         trigger="the repair executes row-locally on the working table. The "
                 "sentence is the SERVER's now — it used to be composed in the "
                 "page and asserted a repair that never ran",
         copy="{n} entries of `{column}` outside the impossibility band of "
              "{low}–{high} {unit} were set to missing.",
         source="turbotab/project.py",
         probe="outside the impossibility band "),
    dict(step="Explore", state="impossible entries · kept as recorded",
         trigger="its own decision kind, because both buttons used to post "
                 "`note` with the same subject and only the prose differed",
         copy="{n} entries of `{column}` outside the impossibility band were "
              "kept as recorded.",
         source="turbotab/project.py", probe="were kept as recorded"),

    # ── Explore · the prevalence refusal for a non-nutrient (`GUIDED-170`) ────
    dict(step="Explore", state="prevalence · the subject is not a nutrient",
         trigger="`prevalence_of_inadequacy` is asked about a column the "
                 "dietary pack does not recognize. Runs BEFORE the reference "
                 "and basis branches: *this is not a nutrient* dominates *the "
                 "RDA is the wrong reference for it*",
         copy="`{name}` is not a nutrient this pack recognizes — it is a "
              "respondent identifier. … this app holds none for `{name}`. "
              "Answering anyway would put a SETTLED nutritional claim on "
              "whatever column was selected.",
         source="turbotab/nutrition.py",
         probe="is not a nutrient this pack recognizes"),

    # ── Cross-step · a response at the control (`GUIDED-167`) ────────────────
    dict(step="Explore", state="any control · the server refused the press",
         trigger="`setErr` or `showRefusal` with a control that carries a "
                 "`data-ac` slot. The canonical `#upErr` still carries it — it "
                 "is the visible sink DURING upload, and `#sub-upload` is "
                 "`display:none` from the first render after",
         copy="(the server's own reason, quoted — never composed here)",
         source="turbotab/web/index.html",
         probe="if (msg) atControl(AT_CONTROL, esc(msg), \"warn\");"),

    dict(step="Explore", state="trim · the label saying what it is NOT",
         trigger="every successful `trim_training_rows`; §04's two objects look "
                 "identical in a spreadsheet, so the trim says which one it is",
         copy="This narrows the TRAINING rows only. It does not change who your "
              "study is about: the held-out rows are untouched, N is unchanged, "
              "and nothing here belongs in participant flow. If you meant to "
              "restrict the population the model is for, that is the eligibility "
              "question, it is asked before the seal, and it does change N.",
         source="turbotab/obligations.py",
         probe="It does not change who your study is about"),

    dict(step="Explore", state="trim · attempted before the seal",
         trigger="`trim_training_rows` while `barrier_raised` is false",
         copy="A robustness trim is post-seal by definition: it narrows the "
              "training partition, and there is no training partition until the "
              "test set is sealed. Before the seal, narrowing the study is an "
              "eligibility criterion — a different object (§04), asked as a "
              "different question, and it changes N.",
         source="turbotab/project.py",
         probe="A robustness trim is post-seal by definition"),

    # ── Preprocess · the shelf and per-model preparation (L18) ───────────────
    dict(step="Preprocess", state="models · the disclosure above the shelf",
         trigger="rendered with the model list, always; not conditional on a "
                 "poor verdict existing",
         copy="Every model is available. This order is about your data, not "
              "about which models are any good — a model low on this list is "
              "one whose concern applies to a table this shape, and you may "
              "have a reason it does not apply to yours. Select whatever you "
              "intend to train.",
         source="turbotab/models.py", probe="Every model is available"),
    dict(step="Preprocess", state="models · the third group's label",
         trigger="the group header, shown even when the group is empty",
         copy="Not recommended for this data",
         source="turbotab/models.py", probe="Not recommended for this data"),
    dict(step="Preprocess", state="models · a low-ranked model was selected",
         trigger="`select_models` with at least one `not_recommended` key; the "
                 "sentence the methods section carries, not an on-screen warning",
         copy="{n} of the selected model(s) carry a stated concern for a table "
              "this shape: {name} — {the coach's own clause}. Selected "
              "deliberately; the concern is recorded so it can be reported "
              "rather than discovered.",
         source="turbotab/models.py",
         probe="recorded so it can be reported rather than discovered"),
    dict(step="Preprocess", state="models · nothing selected",
         trigger="`select_models([])`",
         copy="Choose at least one model. Preprocessing is configured per "
              "model, so there is nothing to configure until you say what you "
              "intend to train.",
         source="turbotab/models.py",
         probe="nothing to configure until you say what you intend to train"),
    dict(step="Preprocess", state="models · chosen before the seal",
         trigger="`select_models` while `barrier_raised` is false",
         copy="Models are chosen after the seal: the shelf is ordered by the "
              "shape of your data, and the shape it reads must be the shape "
              "the models will actually be fitted on.",
         source="turbotab/project.py", probe="Models are chosen after the seal"),

    dict(step="Preprocess", state="recipe · the rendered skip for scaling",
         trigger="a model whose `requires_scaled_numeric` capability is true; "
                 "shown where the question would have been",
         copy="This model measures distances or penalizes coefficients, so a "
              "column measured in thousands would dominate one measured in "
              "units purely because of its scale. The registry records this as "
              "a property of the model, not of your data.",
         source="turbotab/recipes.py",
         probe="a property of the model, not of your data"),
    dict(step="Preprocess", state="recipe · the rendered skip for not scaling",
         trigger="every other model",
         copy="Tree-based and rule-based models split on order rather than on "
              "distance, so rescaling a column changes nothing they can see. "
              "Scaling them is harmless and pointless.",
         source="turbotab/recipes.py", probe="harmless and pointless"),
    dict(step="Preprocess", state="recipe · the variant question was suppressed",
         trigger="`worth_asking` measured the two scalings and found them "
                 "immaterial; shown as the reason no question appeared",
         copy="σ/IQR varies by {pct} across {n} numeric columns — close to the "
              "constant 1.35 a Gaussian column gives, so the two scalings "
              "differ by roughly one global factor and no scale-equivariant "
              "model can tell them apart.",
         source="turbotab/recipes.py",
         probe="no scale-equivariant model can tell them apart"),
    dict(step="Preprocess", state="recipe · the variant question was raised",
         trigger="`worth_asking` measured the two scalings and found them "
                 "material on this data",
         copy="σ/IQR varies by {pct} across {n} numeric columns — heavy tails "
              "in some columns and not others, so standard and robust scaling "
              "would weight the features differently against one another and "
              "the choice changes the fit.",
         source="turbotab/recipes.py", probe="the choice changes the fit"),
    dict(step="Preprocess", state="recipe · a shared setting borrowed from another model",
         trigger="`resolved_recipes` under the uniform answer, on every model "
                 "other than the one the settings came from",
         copy="Applied to every model because you chose one shared "
              "preparation; this is {model}'s setting.",
         source="turbotab/project.py",
         probe="because you chose one shared"),

    dict(step="Preprocess", state="preparation mode · the question",
         trigger="asked once, after the models are chosen",
         copy="Should each model get the preparation it needs, or should they "
              "all get the same preparation so the comparison is about the "
              "models?",
         source="ml/router.py",
         probe="so the comparison is about the models"),
    dict(step="Preprocess", state="preparation mode · why we recommend per-model",
         trigger="shown with the question; states the recommendation AND what "
                 "it costs, because a recommendation with no cost attached is "
                 "advice the reader cannot weigh",
         copy="Per-model is the usual choice and what we recommend: a model "
              "handicapped by preparation it does not suit is not informative "
              "either. The cost is that a difference between two models then "
              "reflects the model and its preparation together — so if you "
              "pick it, that caveat is written into your methods section "
              "automatically.",
         source="ml/router.py",
         probe="written into your methods section automatically"),
    dict(step="Preprocess", state="preparation mode · per-model chosen",
         trigger="the methods sentence recorded on the decision",
         copy="Each model receives the preparation it needs: scaling where the "
              "model measures distances or penalizes coefficients, none where "
              "it splits on order.",
         source="turbotab/project.py",
         probe="none where it splits on order"),
    dict(step="Preprocess", state="preparation mode · uniform chosen",
         trigger="the methods sentence recorded on the decision; a recorded "
                 "answer, because choosing to hold preparation constant is "
                 "itself a methods sentence",
         copy="Every model receives the same preparation, so differences "
              "between them are differences between the models rather than "
              "between their pipelines.",
         source="turbotab/project.py",
         probe="rather than between their pipelines"),
    dict(step="Preprocess", state="preparation mode · the caveat, into Limitations",
         trigger="automatically, on choosing per-model; never on uniform",
         copy="Models were compared under per-model preprocessing: each was "
              "given the preparation appropriate to it rather than a single "
              "shared pipeline. A difference in performance between two models "
              "therefore reflects the model and its preparation together, and "
              "the two cannot be separated from these results alone. This is "
              "the usual choice — a model handicapped by preparation it does "
              "not suit is not informative either — and it is stated so the "
              "comparison is read for what it is.",
         source="turbotab/project.py",
         probe="cannot be separated from these results alone"),
    dict(step="Explore", state="trim · with no stated reason",
         trigger="`trim_training_rows` with an empty `reason`",
         copy="A trim's reason is what the report has to print beside the "
              "breakdown. Without it the disclosure would say that some rows "
              "were outside a range nobody can explain.",
         source="turbotab/obligations.py",
         probe="a range nobody can explain"),
    dict(step="Explore", state="trim · with no bounds",
         trigger="`trim_training_rows` with neither a minimum nor a maximum",
         copy="A trim with no bounds narrows nothing, so there is no "
              "extrapolation to disclose.",
         source="turbotab/obligations.py",
         probe="narrows nothing, so there is no"),
    dict(step="Explore", state="trim · the receipt, which is also an obligation",
         trigger="a train-only trim succeeds; the sentence goes in the "
                 "transcript AND becomes what the report must discharge",
         copy="The model was fitted on training rows with {range} ({reason}). "
              "{k} of {n} held-out rows fall outside that range, so performance "
              "must be reported separately for in-range and out-of-range rows "
              "rather than as one number.",
         source="turbotab/obligations.py",
         probe="separately for in-range and out-of-range rows"),

    dict(step="Data & Target", state="grain · restated after the seal",
         trigger="`set_grain` when `barrier_raised` is true",
         copy="The test set is already sealed, and it was drawn against the grain "
              "answer recorded at the time. Changing that answer now would describe "
              "a split that was not drawn this way.",
         source="turbotab/project.py", probe="describe a split that was not drawn this way"),
    dict(step="Data & Target", state="seal · too few rows with an outcome",
         trigger="fewer than 10 rows have a non-null target",
         copy="Only {n} rows have a value for '{target}', which is too few to hold "
              "any out and still have a study left.",
         source="turbotab/engine.py", probe="few to hold any out and still have a study left"),

    # ── Data & Target · target ───────────────────────────────────────────────
    dict(step="Data & Target", state="target · the event level is not defaulted",
         trigger="applying `set_positive_class` with no chosen level",
         copy="Setting the event needs the level being predicted. There is no "
              "default: whether the event is (say) death or survival is the research "
              "question, not something the file can say.",
         # `DRIVE-041`. Moved from `api.py` to `engine.record_fix`, which the
         # route now calls — the sentence is unchanged and the file it lives in
         # is not. The deck noticed, which is the probe doing its job.
         source="turbotab/engine.py", probe="the research question, not something"),
    dict(step="Data & Target", state="repair · the finding has no automatic fix",
         trigger="`POST /decision {kind: apply}` on a finding whose preview is not applicable",
         copy="That finding has no automatic repair — it needs a human decision.",
         source="turbotab/engine.py", probe="it needs a human decision"),

    # ── Features ─────────────────────────────────────────────────────────────
    dict(step="Features", state="transform · a stateful one was applied",
         trigger="`features.apply` on any entry whose scope is `stateful`",
         copy="'{label}' learns from the column's distribution, so applying it to "
              "the working table now would fit it on the held-out rows too. It is "
              "recorded as a decision and fitted inside each training fold instead. "
              "{because}",
         source="turbotab/features.py", probe="recorded as a decision and fitted inside each training"),
    dict(step="Features", state="transform · a row-local one was declared",
         trigger="`features.declare` on an entry whose scope is `row_local`",
         copy="'{label}' is row-local, so it executes immediately rather than being "
              "declared. Use apply().",
         source="turbotab/features.py", probe="row-local, so it executes immediately rather than"),
    dict(step="Features", state="transform · a capability this door declines to build",
         trigger="`features.get` on `polynomial` or one of its four aliases "
                 "(`poly`, `polynomial_features`, `polynomialfeatures`, "
                 "`interactions`, `all_interactions`)",
         copy="Generating a whole polynomial basis is not offered here, and the "
              "reason is a routing answer rather than a missing feature.\n\n"
              "Two arguments, and they are different. First: degree 2 over ten "
              "numeric columns produces 55 new terms — 10 squares and 45 pairwise "
              "products — that nobody chose one at a time, each carrying "
              "explainability cost. Mass generation is the opposite of this "
              "interview's premise. Second: on a 140-row study those 55 terms are "
              "p/n ≈ 0.39, which is the overfitting regime; the expansion is most "
              "attractive on exactly the small studies where it does the most "
              "harm.\n\nIf your question really is about interactions, the route is "
              "a model that captures them rather than columns that manufacture "
              "them. Trees and gradient boosting get interactions for free, so "
              "this is a model choice at the modeling step, not a feature choice "
              "here.\n\nIf you want ONE interaction because you already reason "
              "about it clinically, that is what `product`, `ratio` and "
              "`difference` are — named, chosen, and each posting its own receipt.",
         source="turbotab/features.py",
         probe="a model choice at the modeling step, not a feature choice here"),
    dict(step="Features", state="transform · the new column name is taken",
         trigger="`features.apply` when the generated name already exists",
         copy="'{name}' already exists in this table. Remove it first, or the new "
              "column would silently replace it.",
         source="turbotab/features.py", probe="would silently replace it"),
    dict(step="Features", state="binning · no cut-points supplied",
         trigger="`bin_fixed` applied without `edges`",
         copy="Binning by supplied cut-points needs at least two edges. Without them "
              "the edges would have to come from the data, which is a different "
              "transform and defers.",
         source="turbotab/features.py", probe="which is a different transform and defers"),
    dict(step="Features", state="encoding · no order supplied",
         trigger="`ordinal_declared` applied without `order`",
         copy="Encoding in a stated order needs the order. Deriving it from the data "
              "is a different transform and defers.",
         source="turbotab/features.py", probe="Deriving it from the data"),
    dict(step="Features", state="remove · a source column was named",
         trigger="`remove_feature` on a column this step did not create",
         copy="'{column}' was not created here, so removing it is not this step's to "
              "do. Only engineered columns can be removed.",
         source="turbotab/project.py", probe="was not created here, so removing it is not this"),
    dict(step="Features", state="deferred preview · why there are no values",
         trigger="`features.preview` on any stateful entry",
         copy="Not computed here. This transform learns from the column's "
              "distribution, so it is fitted inside each training fold at modeling "
              "time — there is no single set of values to show before then.",
         source="turbotab/features.py", probe="no single set of values to show before then"),

    # ── Features · selection ─────────────────────────────────────────────────
    dict(step="Features", state="selection · the outcome offered as a candidate",
         trigger="`selection.declare` with the target among the candidates",
         copy="'{target}' is the outcome and cannot also be a candidate feature: "
              "selecting the target predicts it perfectly.",
         source="turbotab/selection.py", probe="selecting the target predicts it perfectly"),
    dict(step="Features", state="selection · a scope outside the two permitted",
         trigger="`selection.declare` with any scope but `train_rows` / `train_folds`",
         copy="scope must be 'train_rows' or 'train_folds'; got '{scope}'. There is "
              "no third option, and in particular there is no option that fits on "
              "the whole table.",
         source="turbotab/selection.py", probe="There is no third option, and in particular there is no option"),
    dict(step="Features", state="selection · a spec arriving with a chosen set",
         trigger="`set_selection` with `spec['selected']` populated",
         copy="This selection spec carries an already-chosen feature set. Selection "
              "is performed inside the training folds at modeling time; a set chosen "
              "now would have been chosen using the held-out rows.",
         source="turbotab/project.py", probe="already-chosen feature set"),
    dict(step="Features",
         state="selection evidence · nothing was withheld from the ranking",
         trigger="`selection.evidence` where the mask excludes no row — before the "
                 "seal, or with no mask at all",
         copy="Nothing was withheld from this ranking, so it saw every row in the "
              "table. Treat it as exploratory.",
         source="turbotab/selection.py", probe="Nothing was withheld from this ranking"),
    dict(step="Features", state="selection evidence · the normal case",
         trigger="`selection.evidence` where the seal withheld rows from it",
         copy="Ranked on training rows only, and not applied. What is actually "
              "selected is refitted inside each training fold, so this ordering is "
              "indicative rather than the answer.",
         source="turbotab/selection.py", probe="indicative rather than the answer"),
    dict(step="Features", state="selection · ranking before a target is chosen",
         trigger="`GET /selection/evidence` with no target",
         copy="Ranking features needs the outcome first.",
         source="turbotab/api.py", probe="Ranking features needs the outcome first"),

    # ── Features · receipts and transcript lines ─────────────────────────────
    dict(step="Features", state="receipt · a column was removed",
         trigger="`remove_feature` succeeds",
         copy="The engineered column `{column}` was removed.",
         source="turbotab/project.py", probe="was removed."),
    dict(step="Features", state="settled · the step was worked",
         trigger="`settle_features(skipped=False)`",
         copy="Feature work settled: {n} column(s) added now, {d} transform(s) "
              "recorded for fitting inside the training folds[, and a selection spec "
              "recorded].",
         source="turbotab/project.py", probe="recorded for fitting inside the training"),
    dict(step="Features", state="settled · the step was skipped",
         trigger="`settle_features(skipped=True)`",
         copy="Feature work was skipped; the original columns go forward unchanged.",
         source="turbotab/project.py", probe="the original columns go forward unchanged"),
    dict(step="Features", state="selection · every column, recorded",
         trigger="`set_selection(None)`",
         copy="No feature selection: every candidate column is offered to the models.",
         source="turbotab/project.py", probe="every candidate column is offered to the models"),

    # ── cross-step ───────────────────────────────────────────────────────────
    dict(step="Cross-step", state="grain · the answer is recorded",
         trigger="`set_grain` succeeds (transcript line, distinct from the disclosure)",
         copy="Asked whether one person can appear in more than one row; the answer "
              "recorded was: {said}.",
         source="turbotab/project.py", probe="the answer recorded was"),
    dict(step="Cross-step", state="seal · the transcript line",
         trigger="`seal_lockbox` succeeds",
         copy="A test set of {n} rows was sealed before exploration and held by row "
              "label, on the basis '{basis}' ({source}).",
         source="turbotab/project.py", probe="rows was sealed before exploration"),
    dict(step="Cross-step", state="identity · a sealed label went missing",
         trigger="`assert_identity_intact` after a repair renumbered rows",
         copy="{n} sealed row label(s) are no longer in the table (e.g. {labels}). "
              "Something renumbered the rows after the test set was sealed, so the "
              "quarantine no longer refers to the rows it was drawn from.",
         source="turbotab/project.py", probe="no longer refers to the rows it was drawn from"),
    dict(step="Cross-step", state="upload · repeated row labels",
         trigger="`from_dataframe` on a frame whose index has duplicates",
         copy="'{name}' has repeated row labels ({n} of {total}). Row identity in "
              "this project is the index label, so repeated labels leave no way to "
              "say which row a decision refers to.",
         source="turbotab/project.py", probe="no way to say which row a decision refers to"),
]


# ─────────────────────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _generated_sections() -> List[str]:
    from ml import router
    from turbotab import (eligibility as EL, features as F, grain as G,
                          missingness as MI, selection as S)

    out: List[str] = []

    # ── the grain question ───────────────────────────────────────────────────
    qs = {q.key: q for q in router.plan([], target="<outcome>", step="data")}
    q = qs.get("state_grain")
    out.append("### Data & Target · the grain question\n")
    out.append("*Trigger: the step is reached and a target has been chosen. "
               "Never skipped — no confidence makes it moot (constitution §02).*\n")
    if q:
        out.append(f"**Question.** {q.title}\n")
        out.append(f"**Why we ask.** {q.why}\n")
        out.append(f"**Who consumes the answer.** {q.consumer}\n")
        out.append("**Options.**\n")
        for o in q.options:
            out.append(f"- {o}")
        out.append("")
    out.append("The second option opens a follow-up: "
               "*which column identifies the person?* — populated from "
               "`grain.suggestion`, which offers the name heuristic's candidates "
               "first and shape-only candidates after.\n")

    out.append("### Data & Target · what the user reads after answering\n")
    out.append("| Answer | Trigger | Copy |")
    out.append("|---|---|---|")
    for ans, label in ((G.ONE_ROW_PER_PERSON, "No, one row per person"),
                       (G.PEOPLE_REPEAT, "Yes, people repeat"),
                       (G.NOT_SURE, "I'm not sure")):
        out.append(f"| {label} | `set_grain` records `{ans}` | "
                   f"{_esc(G.answer_disclosure(ans, '<column>'))} |")
    out.append("")

    # ── the contradiction interruption ───────────────────────────────────────
    out.append("### Data & Target · the contradiction interruption\n")
    out.append("*A CONSEQUENCE (`DESIGN_LANGUAGE.md` §09): always pushed, and it "
               "resolves or is attested — never a dead end.*\n")
    out.append("| Trigger | Copy |")
    out.append("|---|---|")
    out.append("| The user answers **one row per person** and a column repeats "
               "regularly | ``{col}`` has {n} distinct values across {rows} rows, "
               "about {each} each. That is the shape of repeated measures, and you "
               "answered one row per person. One of those two readings is wrong, and "
               "which one changes how the held-out rows are chosen. |")
    out.append("| The user answers **people repeat**, naming a column that is "
               "unique per row | You said people repeat, but `{col}` has a different "
               "value on every one of its {n} rows. Grouping by it would hold out one "
               "row per group, which is the row-level split you were trying to avoid. |")
    out.append("| The user names a column that is not in the table | ``{col}`` is not "
               "a column in this table, so the held-out rows cannot be grouped by it. |")
    out.append("")
    out.append("**The two exits.** Both travel with the refusal, so an interface "
               "cannot render the interruption without its way out.\n")
    out.append("| Exit | Label | Detail |")
    out.append("|---|---|---|")
    for e in G._EXITS_STATED_UNIQUE:
        out.append(f"| `{e['kind']}` | {_esc(e['label'])} | {_esc(e['detail'])} |")
    out.append("")
    out.append("*The absent-column case carries only the `resolve` exit, and that is "
               "correct rather than a dead end: a column that does not exist cannot be "
               "attested to.*\n")

    # ── the seal ─────────────────────────────────────────────────────────────
    # ── eligibility, clause §04 ──────────────────────────────────────────────
    out.append("### Data & Target · the eligibility question\n")
    out.append("*Trigger: the grain question has been answered and the seal has "
               "not been drawn. Clause §01 fixes that position; the seal is "
               "refused until this is settled, and \"everyone\" is a recorded "
               "answer rather than a skip.*\n")
    out.append(f"**Question.** {EL.QUESTION}\n")
    out.append(f"**Why we ask.** {EL.WHY}\n")
    out.append(f"**What we are NOT showing you, and why.** {EL.WITHHELD_DISCLOSURE}\n")
    out.append(f"**Who consumes the answer.** {EL.CONSUMER}\n")
    out.append("**Options.**\n")
    for o in EL.OPTIONS:
        out.append(f"- {o}")
    out.append("")
    out.append("**The evidence beside it, and its caption.** Bounded by §04: this "
               "answers *is this data corrupted?* and cannot answer *where should "
               "I cut?* — observed min/max, missing count, impossible-value flags "
               "and, for a categorical column, the distinct values. No median, no "
               "quantiles, no per-value counts.\n")
    out.append(f"> {EL.EVIDENCE_CAPTION}\n")

    out.append("### Data & Target · what the user reads after answering eligibility\n")
    out.append("| Answer | Trigger | Copy |")
    out.append("|---|---|---|")
    # READ FROM THE MODULE, NOT RESTATED. `DRIVE-031`: this line held its own
    # copy of the sentence, so removing a false clause from `eligibility.py`
    # left the deck asserting it — and `check` stayed green, because `check`
    # probes that a hand entry's FRAGMENT is still in its source file and this
    # row is generated with no probe at all. The seal table two blocks down has
    # always called `G.seal_disclosure`; this one did not.
    out.append("| No, the study is about everyone here | `set_eligibility` records "
               "`everyone` | " + _esc(EL.EVERYONE_SENTENCE.format(n="{N}")) + " |")
    out.append("| Yes, restricted | `set_eligibility` records `restricted` with a "
               "column, a range and a reason | " + _esc(
                   "{k} of {N} rows were excluded before the held-out set was "
                   "drawn: {criterion}. {reason} Those rows are gone before "
                   "anything is held out, so the held-out set describes the "
                   "population you studied rather than a wider one.") + " |")
    out.append("")

    out.append("### Data & Target · what the user reads once the seal is drawn\n")
    out.append("*Keyed on the recorded basis, so the states constitution §03 insists "
               "on stay different sentences — an undetermined seal and a verified "
               "cross-sectional one cannot render alike, because they are not the "
               "same string.*\n")
    out.append("| Basis | Exploratory? | Copy |")
    out.append("|---|---|---|")
    for basis in ("cross_sectional", "grouped", "undetermined",
                  "repetition_found_grouping_abandoned"):
        lb = {"seal_basis": basis, "n_test": 27, "fraction": 0.15,
              "n_test_groups": 9, "group_noun": "subjects"}
        flag = "**yes**" if G.is_exploratory_basis(basis) else "no"
        out.append(f"| `{basis}` | {flag} | {_esc(G.seal_disclosure(lb))} |")
    out.append("")
    # `DRIVE-031`. THE SAME FOUR, ON A TABLE WHERE THE OUTCOME IS MISSING SOMEWHERE.
    # The base a seal drew from is a different sentence from the base it did not
    # need to mention, and the deck records both rather than only the tidy one.
    out.append("*Where rows are dropped for a missing outcome, every basis names "
               "the base it drew from — the percentage was previously stated with "
               "no base at all, beside an eligibility receipt that named a "
               "different one.*\n")
    out.append("| Basis | Copy, on a table with a missing outcome |")
    out.append("|---|---|")
    for basis in ("cross_sectional", "grouped", "undetermined",
                  "repetition_found_grouping_abandoned"):
        lb = {"seal_basis": basis, "n_test": 945, "fraction": 0.15,
              "n_test_groups": 9, "group_noun": "subjects",
              "n_total": 6297, "n_rows_before_outcome_drop": 21849}
        out.append(f"| `{basis}` | {_esc(G.seal_disclosure(lb))} |")
    out.append("")
    out.append("**After an attested contradiction**, the seal sentence gains: "
               "*Note: this split rests on your answer, which disagreed with the "
               "shape of the data. That disagreement is on the record and belongs in "
               "the methods section.* — and the seal is marked exploratory.\n")

    # ── the Features step ────────────────────────────────────────────────────
    fq = {q.key: q for q in router.plan([], target="<outcome>", step="features")}
    out.append("### Features · the two questions\n")
    for key in ("choose_features", "choose_selection"):
        q = fq.get(key)
        if not q:
            continue
        out.append(f"**`{key}`** — {q.title}\n")
        out.append(f"*Trigger: the Features step is reached with a target chosen.*\n")
        out.append(f"**Why we ask.** {q.why}\n")
        out.append(f"**Who consumes the answer.** {q.consumer}\n")
        out.append(f"**Options.** {' · '.join(q.options)}\n")

    # ── Preprocess, clause §07 ───────────────────────────────────────────────
    out.append("### Preprocess · the mechanism question, asked per column\n")
    out.append("*Trigger: the Preprocess step is reached and a column has "
               "missing values. Asked BEFORE the strategy, because the answer "
               "decides which strategies are legitimate. Never skipped — the "
               "app cannot know.*\n")
    out.append(f"**Question.** {MI.MECHANISM_QUESTION}\n")
    out.append(f"**Why we ask.** {MI.MECHANISM_WHY}\n")
    out.append(f"**Who consumes the answer.** {MI.MECHANISM_CONSUMER}\n")
    out.append("**Options.**\n")
    for o in MI.MECHANISM_OPTIONS:
        out.append(f"- {o}")
    out.append("")

    out.append("### Preprocess · the strategies, and why each is where it is\n")
    out.append("| Branch | Strategy | Label | Executes | Because |")
    out.append("|---|---|---|---|---|")
    for branch, keys in (("numeric", MI.NUMERIC_STRATEGIES),
                         ("categorical", MI.CATEGORICAL_STRATEGIES)):
        for k in keys:
            sp = MI.strategy(k)
            when = "in training folds" if sp["defers"] else "now (row-local)"
            out.append(f"| {branch} | `{k}` | {_esc(sp['label'])} | {when} | "
                       f"{_esc(sp['because'])} |")
    out.append("")

    out.append("### Preprocess · the informative-missingness blocker\n")
    out.append("*A CONSEQUENCE. Fires when the user has stated the missingness "
               "is informative AND chosen a strategy that fills the blanks. "
               "`I'm not sure` deliberately does NOT fire it.*\n")
    out.append("> " + _esc(MI.INFORMATIVE_IMPUTATION_BLOCKER).replace(
        "\n\n", " ") + "\n")
    out.append("**The two exits.** Acknowledgment is TYPED, not a click.\n")
    out.append("| Exit | Label | Detail |")
    out.append("|---|---|---|")
    for e in MI.BLOCKER_EXITS:
        out.append(f"| `{e['id']}` | {_esc(e['label'])} | {_esc(e['detail'])} |")
    out.append("")

    out.append("### Preprocess · the two refusals that have no exit\n")
    out.append("| Trigger | Copy |")
    out.append("|---|---|")
    out.append("| The outcome is named inside a MICE imputation scope | "
               + _esc(MI.OUTCOME_IN_IMPUTATION_REFUSAL) + " |")
    out.append("| A mechanism is stated informative | *(recorded, not refused)* "
               + _esc(MI.STABILITY_ASSUMPTION) + " |")
    out.append("")

    out.append("### Features · the transform catalogue\n")
    out.append("*Every entry states its own clause-§06 classification and why. "
               "Row-local entries execute immediately and post a receipt; deferred "
               "entries are recorded and fitted inside each training fold.*\n")
    for scope, keys in (("Row-local — executes immediately", F.row_local_keys()),
                        ("Deferred — fitted inside the training folds",
                         F.deferred_keys())):
        out.append(f"#### {scope}\n")
        out.append("| Label | Explainability | Why this scope | Receipt / methods sentence |")
        out.append("|---|---|---|---|")
        for k in keys:
            t = F.get(k)
            out.append(f"| {_esc(t.label)} | {t.explainability_cost} | "
                       f"{_esc(t.because)} | {_esc(t.sentence)} |")
        out.append("")

    out.append("### Features · selection methods\n")
    out.append("| Method | Explainability | Methods sentence (the timing IS the copy) |")
    out.append("|---|---|---|")
    for k, m in S.METHODS.items():
        out.append(f"| {_esc(m.label)} | {m.explainability_cost} | "
                   f"{_esc(m.sentence)} |")
    out.append("")
    out.append("*Choosing `scope='train_rows'` (Classic's behavior) rewrites "
               "\"within each training fold\" as \"once over the training rows "
               "(held-out rows excluded)\", so a project can state which happened "
               "rather than imply the stronger claim.*\n")
    return out


def _hand_sections() -> List[str]:
    out: List[str] = []
    by_step: Dict[str, List[Dict[str, Any]]] = {}
    for e in HAND:
        by_step.setdefault(e["step"], []).append(e)
    for step, entries in by_step.items():
        out.append(f"### {step} · refusals, receipts and transcript lines\n")
        out.append("| State | Trigger | Copy | Source |")
        out.append("|---|---|---|---|")
        for e in entries:
            out.append(f"| {_esc(e['state'])} | {_esc(e['trigger'])} | "
                       f"{_esc(e['copy'])} | `{e['source']}` |")
        out.append("")
    return out


def _empty_states() -> List[str]:
    return [
        "### Empty and terminal states\n",
        "*Assembled by hand — these live in `web/index.html` as markup, which is "
        "the least reviewable place copy can live (`GUIDED-013`).*\n",
        "| State | Trigger | Copy |",
        "|---|---|---|",
        "| No project yet | first load | Drop a CSV to begin. |",
        "| A clean file | `diagnose` returns no findings | This file reads as a "
        "clean table. |",
        "| No features engineered | the Features step, before any transform | "
        "Nothing added yet. The original columns go forward unless you build "
        "something. |",
        "| Selection not set | the Features step, before a selection answer | "
        "Every column will be offered to the models. |",
        "| Findings stale | any answer changed underneath computed findings | "
        "These were computed under an earlier answer. |",
        "| Downstream stale | a feature was added or removed | Results computed "
        "before this change no longer describe the current feature set: {why}. |",
        "",
    ]


def build() -> str:
    lines: List[str] = [
        "# Copy deck — the Guided door",
        "",
        "**Generated in part.** `python docs/turbotab/tools/copydeck.py regen`.",
        "Do not hand-edit the generated sections; edit the source and regenerate.",
        "",
        "Every user-facing string in the Guided door, by step and by state, with "
        "the condition that triggers it. It exists so copy can be reviewed "
        "**without running the app** — a reviewer should not have to drive a "
        "server to read a sentence.",
        "",
        "## How much of this is generated",
        "",
        "The catalogues are **generated**: `features.CATALOGUE`, "
        "`selection.METHODS`, `router.plan()`'s questions, and `grain`'s answers, "
        "exits and disclosures are all data, so this tool prints what is actually "
        "there and they cannot drift.",
        "",
        "The refusals, receipts and transcript lines are **hand-assembled**, "
        "because they are f-strings raised at roughly 105 call sites across "
        "`api.py`, `project.py`, `features.py` and `selection.py`, plus 51 string "
        "literals inside `web/index.html`. Extracting those would need either an "
        "AST walk that cannot resolve the interpolations or a runtime harness "
        "driving every error path — both of which produce a *worse* artifact than "
        "transcribing them, because a half-resolved f-string is not reviewable "
        "copy.",
        "",
        "**That difficulty is a finding, not a workaround.** `GUIDED-013` records "
        "it: copy that lives at its raise site cannot be reviewed, translated or "
        "kept consistent, and this deck is the symptom rather than the cure.",
        "",
        "Against drift, each hand entry carries a probe — a distinctive fragment "
        "of the real string — and `copydeck.py check` asserts that fragment is "
        "still in the file it came from. **This is weaker than generation**: it "
        "catches a string that changed, not one that was added. Said plainly "
        "rather than sold as equivalent.",
        "",
        "---",
        "",
    ]
    lines += _generated_sections()
    lines.append("---\n")
    lines += _hand_sections()
    lines += _empty_states()
    return "\n".join(lines) + "\n"


def _squash(text: str) -> str:
    """Collapse whitespace and quote characters.

    Source strings wrap across implicit concatenation — `"not this "` on one
    line and `"step's to do."` on the next — so a probe that reads naturally
    is never a contiguous substring of the file. Squashing both sides makes the
    match survive rewrapping, which is the commonest edit and the one that
    should NOT count as drift.

    Lossy on purpose: it cannot tell `a b` from `ab`. For detecting that a
    sentence changed, that is enough.
    """
    return "".join(c for c in text if not c.isspace() and c not in "\"'")


def check() -> int:
    """Every hand entry's probe must still be in the file it names."""
    bad: List[str] = []
    for e in HAND:
        path = os.path.join(ROOT, e["source"])
        if not os.path.exists(path):
            bad.append(f"{e['source']} does not exist (for: {e['state']})")
            continue
        text = _squash(open(path, encoding="utf-8").read())
        if _squash(e["probe"]) not in text:
            bad.append(f"{e['source']}: probe {e['probe']!r} is gone "
                       f"(for: {e['state']}) — the copy changed and the deck did not")
    if os.path.exists(OUT):
        if open(OUT, encoding="utf-8").read() != build():
            bad.append("COPY_DECK.md is stale — run `copydeck.py regen`")
    else:
        bad.append("COPY_DECK.md does not exist — run `copydeck.py regen`")
    for b in bad:
        print(f"FAIL {b}", file=sys.stderr)
    if bad:
        print(f"\n{len(bad)} violation(s)", file=sys.stderr)
        return 1
    print(f"ok — {len(HAND)} hand entries probed, deck current")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["regen", "check"])
    args = ap.parse_args()
    if args.command == "regen":
        text = build()
        with open(OUT, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {os.path.relpath(OUT, ROOT)} — {len(text):,} bytes, "
              f"{len(HAND)} hand entries")
        return 0
    return check()


if __name__ == "__main__":
    sys.exit(main())
