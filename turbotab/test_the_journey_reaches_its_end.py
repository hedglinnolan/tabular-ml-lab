"""Explain, Report, and a fitted run that goes stale — the last three surfaces.

Three parts of one loop, in one file because they are one drive: a user who
trains a model then reads why it predicts what it does, reads what they have
written, and changes an earlier answer.

## Explain — `GUIDED-101`, scoped rather than invented

The four research packs contain **zero** explainability content, so what the
field holds about SHAP versus permutation importance resolves to nothing this
repository can cite. That is design work rather than loop work (`LOOP.md` §08),
so this asserts the narrow defensible slice:

* permutation importance on the **held-out rows** — the choice with a leakage
  consequence, exactly as a calibration curve on training predictions is;
* identity from the lockbox **labels** (`MINE-014` is Classic's version of
  getting this wrong, and it is `critical`);
* prose read out of `ml/plot_narrative.py`, not composed here;
* and the promise `FEATURE_REGISTER.md` has been making since `L14` — that an
  explanation is harder to read because of a decision made at Features — at
  last delivered.

## Report — the thesis, tested

*The transcript the user scrolls and the manuscript they export are the same
object at two levels of formality.* `draft.py` already folded decisions into
prose; what changed this loop is that the **preprocessing plan may be
exported**, and it is only safe because `L35-B` made the recorded sentence and
the fitted pipeline one object. Before that the sentence disagreed with the
fit, and exporting it would have put the disagreement in the manuscript.

## Staleness — `GUIDED-094`

`stale_downstream` has been written since `L5` and read by nothing. There are
**three** downstream artifacts now — the fitted run, the selection spec and the
explanation — and each is invalidated by different things, so a veil that knew
about one would be this loop's own defect class.

## Fixture shapes · `GUIDED-097`

`SHAPES` runs the load-bearing claims against a continuous and a binary-string
target; `SHAPES_NOT_COVERED` names the rest.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, draft as _draft, eventfixture           # noqa: E402
from turbotab import explain as _explain                           # noqa: E402
from turbotab import pageharness as H, training as _training      # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

SHAPES = {
    "continuous": ("clinic_visits.csv", "hba1c", ("ridge", "histgb_reg")),
    "binary_string": ("clinic_visits.csv", "outcome", ("logreg", "histgb_clf")),
}

SHAPES_NOT_COVERED = {
    "binary_numeric": (
        "`leaky_sepsis.csv` has a 0/1 target and no blanks, so its pipeline "
        "has no imputer and its Explain and Report claims would exercise a "
        "strictly simpler record than the two covered here."),
    "multiclass": (
        "No fixture has a three-or-more-level outcome. `permutation_importance` "
        "takes one and the scorer would change; neither is driven here."),
    "grouped": (
        "A repeated-measures project seals BY PERSON, and the held-out frame "
        "Explain permutes is drawn the same way — so the grouped seal is "
        "exercised by `test_the_seal_holds.py` and not by this file."),
}


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _journey(client, shape, *, train=True, models=None):
    fixture, target, default_models = SHAPES[shape]
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]

    def decide(kind, **payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (kind, r.text[:250])
        return r

    decide("set_target", column=target)
    decide("set_purpose", answer="prediction")
    decide("set_grain", answer="one_row_per_person")
    decide("set_eligibility", answer="everyone")
    decide("seal", fraction=0.25)
    # `DRIVE-041`. Over the route, and only where the engine asks.
    eventfixture.choose_event_over_http(client, pid, target)
    project = api.STORE.get(pid)
    if train:
        project.training_run = _training.train(
            project, list(models or default_models))
        api._RUNS[pid] = {"run": project.training_run}
    return pid, project, decide


# ── Explain ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_importance_is_computed_on_the_held_out_rows(client, shape):
    """**The choice with a leakage consequence.**

    Importance computed on the rows the model was fitted on is the model
    grading its own homework, and it looks better the more the model overfits —
    the same argument the calibration figure makes about training predictions.

    Asserted on the COUNT the payload reports and on the frame the module
    hands sklearn, because a payload that merely says `held-out rows only`
    would be a label rather than a fact.
    """
    pid, project, _ = _journey(client, shape)
    body = client.get(f"/project/{pid}/explain").json()
    assert body["blocked_by"] is None, body["blocked_by"]
    run = body["run"]
    assert run["scored_on"] == "held-out rows only"

    sealed = set(project.lockbox["labels"])
    table = project.working_table
    expected = int((table[str(project.target)].notna()
                    & pd.Series([i in sealed for i in table.index],
                                index=table.index)).sum())
    assert run["n_rows"] == expected, (
        f"the ranking saw {run['n_rows']} rows and {expected} are sealed")
    assert run["n_rows"] < len(table), "it saw every row"       # control
    assert len(_explain.held_out_frame(project)) == expected
    # THE LABELS, not only the count. A count is re-derivable beside the
    # computation and a set of labels is not: these are the rows that were
    # permuted, and they must be the sealed ones.
    assert set(run["row_labels"]) == {
        str(i) for i in _explain.held_out_frame(project).index}, (
        "the ranking was computed over rows that are not the sealed ones")


def test_the_held_out_rows_are_fetched_by_label(client):
    """`MINE-014` is `critical` and open in Classic: `pages/06` stores the test
    set as POSITIONS, `pages/07` reads them back with `.iloc` into a frame
    `get_data()` may have filtered differently, and the `.iloc` lands on
    different people with no error.

    Decision A's identity barrier exists so this door does not have to make
    that mistake. Asserted the way it fails over there — take rows out from
    under the accessor and watch the frame follow the LABELS rather than the
    positions.
    """
    pid, project, _ = _journey(client, "continuous", train=False)
    before = list(_explain.held_out_frame(project).index)
    assert before, "nothing was sealed"                          # control

    # A cohort filter is exactly what `get_data()` applies between the two page
    # visits in Classic, and it is what makes a stored position mean a
    # different row.
    keep = [l for l in project.df.index if l not in set(before[:3])]
    project.cohort = {"label": "a narrower cohort", "labels": keep}
    after = list(_explain.held_out_frame(project).index)

    assert set(after) <= set(before), (
        "the held-out frame gained rows that were never sealed")
    assert set(before[:3]).isdisjoint(after), (
        "a row removed from the working table is still being explained, which "
        "is what a positional index does and a label cannot")
    assert all(isinstance(a, type(before[0])) for a in after)


@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_prose_is_cores_and_not_this_modules(client, shape):
    """`AUDIT-008` arriving in the last step of the journey is the thing to
    avoid: `ml/plot_narrative.py` already holds sourced interpretation prose
    for this method, and a Guided step that wrote its own beside it would be
    the app owning the right thing and not reaching for it."""
    from ml.plot_narrative import interpretation_permutation_importance

    pid, project, _ = _journey(client, shape)
    run = client.get(f"/project/{pid}/explain").json()["run"]
    assert run["interpretation"] == interpretation_permutation_importance(), (
        "the Explain step composed its own interpretation sentence beside "
        "core's")
    assert "Sklearn" in run["interpretation"], (
        "the sourced attribution did not survive the trip")
    assert run["narrative"], "core's narrative reached nothing"
    assert run["model_name"] in run["narrative"]


def test_shap_says_why_it_is_absent_rather_than_being_silent(client):
    """A missing explanation and an explanation the app declined to give are
    different things, and only one of them is honest as a silence."""
    pid, project, _ = _journey(client, "continuous", train=False)
    body = client.get(f"/project/{pid}/explain").json()
    assert body["shap"]["available"] is False
    why = body["shap"]["why"]
    # BOTH reasons, because the second one would hold even if the dependency
    # were installed and a reader has to be able to tell them apart.
    assert "numba" in why and "GUIDED-101" in why
    assert body["shap"]["where"].startswith("Classic")


def test_before_a_fit_the_step_says_which_state_applies(client):
    """A step that has not happened and a step that produced nothing are
    different sentences — the same rule the training and prevalence surfaces
    follow."""
    pid, project, _ = _journey(client, "continuous", train=False)
    body = client.get(f"/project/{pid}/explain").json()
    assert body["run"] is None
    assert "No model has been fitted yet" in body["blocked_by"]
    assert "shuffled" in body["blocked_by"] or "drop in a metric" in body["blocked_by"]


def test_a_costly_feature_decision_is_named_in_the_explanation(client):
    """**The promise the register has been making since `L14`.**

    Every transform carries `explainability_cost`, and `FEATURE_REGISTER.md`'s
    `prep-pca` row states the consequence in words — *a warning that SHAP will
    refer to PC1/PC2.* Nothing delivered it. A ranking over principal
    components that does not say so is a table of numbers about columns the
    user never measured.
    """
    pid, project, decide = _journey(client, "continuous", train=False)
    numeric = client.get(f"/project/{pid}/features").json()["numeric_columns"]
    usable = [c for c in numeric if project.df[c].notna().any()][:3]
    decide("defer_feature", transform="pca", columns=usable,
           params={"n_components": 2})

    named = _explain.costly_decisions(project)
    assert named, "a high-cost decision was recorded and nothing named it"
    entry = named[0]
    assert entry["key"] == "pca"
    assert entry["explainability_cost"] == "high"
    assert "principal components" in entry["consequence"]
    assert entry["sentence"], "the decision is named without its own sentence"

    body = client.get(f"/project/{pid}/explain").json()
    assert body["costly_decisions"], (
        "the warning is computed and never served, which is the class this "
        "step was supposed to close rather than join")


def test_a_low_cost_decision_raises_no_warning(client):
    """The other half, or the warning is wallpaper. `log` is `low`."""
    pid, project, decide = _journey(client, "continuous", train=False)
    decide("add_feature", transform="log", columns=["glucose"])
    assert _explain.costly_decisions(project) == []


def test_every_high_cost_catalogue_entry_has_a_consequence_sentence():
    """The completeness half. A transform marked `high` with no sentence would
    render a warning with nothing in it, which is worse than no warning."""
    from turbotab import features as _feat, selection as _sel

    costly = [k for k, t in _feat.CATALOGUE.items()
              if t.explainability_cost == _explain.COSTLY]
    assert costly, "nothing in the catalogue is marked high"     # control
    missing = [k for k in costly if k not in _explain.costly_keys()]
    assert not missing, (
        f"these transforms are recorded high-cost and this module has no "
        f"sentence for what they do to a ranking: {missing}")
    methods = [k for k, m in _sel.METHODS.items()
               if m.explainability_cost == _explain.COSTLY]
    assert not [k for k in methods if k not in _explain.costly_keys()], methods


# ── Report ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", sorted(SHAPES), ids=sorted(SHAPES))
def test_the_preprocessing_plan_reaches_the_methods_section(client, shape):
    """**The specific thing `L35-B` made possible, and this part's deadline.**

    `GUIDED-089`'s note says `draft.py` had no reference to missingness or
    preprocess, so the recorded methods sentence was never exported — and until
    `L35-B` that was *fortunate*, because the sentence disagreed with the fit:
    the record said the blank was left and the pipeline filled it with the
    median. It agrees now, so the sentence may be exported, and the string in
    the manuscript is the string on the record and the string the pipeline
    performs.
    """
    pid, project, decide = _journey(client, shape, train=False)
    decide("route_missingness", column="notes", mechanism="informative",
           strategy="explicit_category")
    recorded = [d for d in project.missingness if d["column"] == "notes"][0]

    body = client.get(f"/project/{pid}/draft").json()
    section = [s for s in body["sections"] if s["key"] == "preprocess"][0]
    assert section["sentences"], (
        "the preprocessing plan is recorded and the manuscript does not carry "
        "it, which is what GUIDED-089 said and what L35 made safe to change")
    exported = section["sentences"][0]["text"]
    assert recorded["sentence"] in exported, (
        f"the manuscript rewrote the recorded sentence: {exported!r}")

    # AND THE ASSUMPTION TRAVELS. §07 records the stability assumption as a
    # methods assumption rather than a warning precisely so a manuscript can
    # carry it — it may not hold across sites and a reader has to see it.
    assert recorded["assumption"][:40] in exported, (
        "the stability assumption was recorded and not exported")


def test_the_draft_never_writes_a_claim_the_record_does_not_hold(client):
    """`AUDIT-001` lives next door: the generated manuscript once reported a
    raw *p* < 0.05 count and named no correction. Every sentence here must
    quote a recorded decision or be a visible gap."""
    pid, project, decide = _journey(client, "continuous", train=False)
    decide("route_missingness", column="notes", mechanism="not_informative",
           strategy="impute_mode")
    body = client.get(f"/project/{pid}/draft").json()

    recorded = {(d.get("text") or "").strip()
                for d in client.get(f"/project/{pid}").json()["decisions"]}
    for section in body["sections"]:
        for item in section["sentences"]:
            if item["has_gap"] or item["kind"] == "derived":
                continue
            assert any(r and r in item["text"] for r in recorded), (
                f"the draft states something no decision says: {item['text']!r}")
    for banned in ("significant", "p <", "p<", "correlat", "caused"):
        assert not any(banned in i["text"].lower()
                       for s in body["sections"] for i in s["sentences"]), banned


def test_where_the_app_cannot_know_it_leaves_the_gap_visible(client):
    """`AUTHOR_GAP` is the answer where the app cannot know, never a plausible
    sentence. An exported methods section that reads smoothly and states
    something nobody decided is the governing rule's worst failure, because it
    is the artifact that leaves the building."""
    pid, project, _ = _journey(client, "continuous", train=False)
    body = client.get(f"/project/{pid}/draft").json()
    assert body["n_gaps"] > 0, (
        "a draft with an outcome and no author gap has written the research "
        "question in the user's name")
    gaps = [i for s in body["sections"] for i in s["sentences"] if i["has_gap"]]
    for item in gaps:
        assert _draft.AUTHOR_GAP in item["text"]
        assert "—" in item["text"], (
            "a gap that does not say what is missing is a blank with a label")


def test_an_empty_section_says_what_it_is_waiting_for(client):
    """No internal placeholder ever renders (`GUIDED-007`). A section with
    nothing in it says so in the app's own voice."""
    pid, project, _ = _journey(client, "continuous", train=False)
    body = client.get(f"/project/{pid}/draft").json()
    waiting = [s for s in body["sections"] if s["waiting_for"]]
    assert waiting, "every section is full, so this asserts nothing"
    for section in waiting:
        assert not section["sentences"]
        assert len(section["waiting_for"]) > 20
        for banned in ("TODO", "not built", "placeholder", "coming soon"):
            assert banned.lower() not in section["waiting_for"].lower()


# ── Staleness ────────────────────────────────────────────────────────────────

def test_a_fitted_run_goes_stale_and_says_why(client):
    """`GUIDED-094`. Measured while adjudicating `L34`: the project recorded
    `stale_downstream` correctly and the page's `trainRun` box was
    byte-identical before and after, 923 characters both times.

    The fix is a renderer and a veil — never a recompute and never a clear.
    Principle 4 is *visible, veiled, recoverable*.
    """
    pid, project, decide = _journey(client, "continuous", models=["ridge"])
    fresh = client.get(f"/project/{pid}/training").json()
    assert fresh["run"], "nothing was fitted"
    assert fresh["stale"] == [], "a run went stale before anything changed"

    decide("route_missingness", column="notes", mechanism="informative",
           strategy="explicit_category")
    after = client.get(f"/project/{pid}/training").json()
    assert after["stale"], (
        "the working table changed under a fitted run and the run does not "
        "say so — a held-out number standing over a table that moved")
    assert "notes" in after["stale"][0]["why"], after["stale"]

    # RECOVERABLE, NOT DELETED. The numbers are still there.
    assert after["run"]["results"], "the run was cleared rather than veiled"
    assert (after["run"]["results"][0]["metrics"]
            == fresh["run"]["results"][0]["metrics"])


def test_the_selection_and_the_run_go_stale_for_different_reasons(client):
    """**The scope that makes this Part D rather than a footnote.** There are
    three downstream artifacts now, and a veil that knew about one would be
    this loop's own defect class.

    The selection spec names candidate COLUMNS, so it is invalidated by a
    column arriving — and the fitted run is invalidated by that too, but they
    are stamped separately, so a run fitted AFTER the column was added is not
    stale while the older selection still is.
    """
    pid, project, decide = _journey(client, "continuous", train=False)
    candidates = client.get(f"/project/{pid}/features").json()["numeric_columns"]
    decide("set_selection", method="mutual_info", n_features=2,
           candidates=candidates)
    assert client.get(f"/project/{pid}/features").json()["selection_stale"] == []

    decide("add_feature", transform="log", columns=["glucose"])
    stale = client.get(f"/project/{pid}/features").json()["selection_stale"]
    assert stale, (
        "a column arrived after the candidate pool was recorded and the "
        "selection does not say its pool describes a table that no longer "
        "exists")
    assert "log_glucose" in stale[0]["why"]

    # A run fitted NOW is not stale, while the older selection still is —
    # which is the whole reason each artifact carries its own watermark.
    project.training_run = _training.train(project, ["ridge"])
    api._RUNS[pid] = {"run": project.training_run}
    assert client.get(f"/project/{pid}/training").json()["stale"] == []
    assert client.get(f"/project/{pid}/features").json()["selection_stale"]


def test_the_explanation_says_it_was_recomputed_beside_a_stale_score(client):
    """The third artifact, and it fails differently from the other two.

    `/explain` REFITS from the current record on every request, so the ranking
    is never stale — and the held-out scores it sits beside are the stored
    run's, which can be. Presenting a fresh ranking next to a stale accuracy
    without saying so is two numbers about two different analyses.
    """
    pid, project, decide = _journey(client, "continuous", models=["ridge"])
    assert client.get(f"/project/{pid}/explain").json()["recomputed_note"] is None

    decide("route_missingness", column="notes", mechanism="informative",
           strategy="explicit_category")
    body = client.get(f"/project/{pid}/explain").json()
    assert body["stale"], "the run moved and the explanation does not say so"
    assert body["recomputed_note"], (
        "a freshly computed ranking is rendered beside a stale score with "
        "nothing distinguishing them")
    assert body["run"], "the explanation was withheld rather than qualified"


def test_a_run_with_no_watermark_is_not_reported_as_fresh(client):
    """`None` is a third state — *this run predates the watermark* — and a
    caller that could not tell it from `0` would report an unknown as fresh."""
    pid, project, decide = _journey(client, "continuous", models=["ridge"])
    decide("route_missingness", column="notes", mechanism="informative",
           strategy="explicit_category")
    assert project.stale_since(0), "the log is empty, so this proves nothing"
    assert project.stale_since(None) == [], (
        "an unstamped artifact reported a staleness it cannot know")


# ── the page ─────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not H.available(), reason="no JS engine on this machine")
def test_the_page_renders_explain_report_and_the_veil(client):
    """**The load-bearing claim, driven.** Three surfaces the server composes;
    `GUIDED-080`'s class is the server composing and the interface never
    rendering, and it has been this project's dominant defect."""
    pid, project, decide = _journey(client, "continuous", models=["ridge"])
    decide("route_missingness", column="notes", mechanism="informative",
           strategy="explicit_category")
    seen = client.get(f"/project/{pid}").json()
    routes = {
        f"/project/{pid}": seen,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": []},
        f"/project/{pid}/interview?step=features":
            client.get(f"/project/{pid}/interview?step=features").json(),
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/evidence/plausibility": {"columns": []},
        f"/project/{pid}/features": client.get(f"/project/{pid}/features").json(),
        f"/project/{pid}/figures": client.get(f"/project/{pid}/figures").json(),
        f"/project/{pid}/preprocess":
            client.get(f"/project/{pid}/preprocess").json(),
        f"/project/{pid}/recipes": client.get(f"/project/{pid}/recipes").json(),
        f"/project/{pid}/models": client.get(f"/project/{pid}/models").json(),
        f"/project/{pid}/training": client.get(f"/project/{pid}/training").json(),
        f"/project/{pid}/explain": client.get(f"/project/{pid}/explain").json(),
        f"/project/{pid}/draft": client.get(f"/project/{pid}/draft").json(),
        f"/project/{pid}/gaps": {"gaps": []},
    }
    out = H.run(
        """
        function settle(n){
          var p = Promise.resolve();
          for (var i = 0; i < n; i++) { p = p.then(function(){}); }
          return p;
        }
        settle(28).then(function(){
          __harness.drainRaf();
          __emit({explain: __harness.html('explainBox'),
                  explainWhy: __harness.html('explainWhy'),
                  report: __harness.html('reportBox'),
                  trainClass: __harness.el('sec-train').className,
                  staleTag: __harness.el('stale-train').textContent,
                  calls: __harness.calls().map(function(c){ return c.path; })});
        });
        """, routes=routes, search=f"?project={pid}")

    assert f"/project/{pid}/explain" in out["calls"], (
        "the page never fetched the Explain surface")
    assert f"/project/{pid}/draft" in out["calls"]

    served = routes[f"/project/{pid}/explain"]["run"]
    assert served["ranked"][0]["feature"] in out["explain"], (
        "the ranking was served and rendered nowhere")
    assert served["interpretation"][:40] in out["explainWhy"], (
        "core's interpretation prose reached the payload and not the page")
    assert "SHAP is not offered here" in out["explain"], (
        "the app declines a method and says nothing about it on screen")

    draft = routes[f"/project/{pid}/draft"]
    preprocess = [s for s in draft["sections"] if s["key"] == "preprocess"][0]
    # The longest backtick-free run, because the page renders a `column` as a
    # mono span — so a raw prefix of the sentence would fail for a reason that
    # is about markup rather than about whether the plan reached the reader.
    fragment = max(preprocess["sentences"][0]["text"].split("`"), key=len)
    assert len(fragment) > 20, fragment
    assert fragment in out["report"], (
        "the preprocessing plan reaches the draft and not the page")
    assert "[AUTHOR REQUIRED]" in out["report"], (
        "the gaps are marked in the payload and invisible on the page")

    # THE VEIL, on the same node, naming its reason.
    assert "stale" in out["trainClass"], (
        "the run is stale and the section does not wear it")
    assert "notes" in out["staleTag"], (
        f"the veil says something generic rather than what changed: "
        f"{out['staleTag']!r}")


def test_the_shapes_this_file_does_not_cover_are_named():
    """`GUIDED-097`'s honesty clause. A sweep that reports only what it covered
    has not reported its coverage."""
    assert len(SHAPES) >= 2 and SHAPES_NOT_COVERED
    for shape, reason in SHAPES_NOT_COVERED.items():
        assert shape not in SHAPES, f"{shape} is declared uncovered and covered"
        assert len(reason) > 80, f"{shape}: the reason is a shrug"
