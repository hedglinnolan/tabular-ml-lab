"""`MISC-014` — the sensitivity fork, routed into Guided without its landmines.

`DOMAIN_SCIENCE.md` §03 called primitive 4 *"currently absent from the app
entirely."* It was not: `ml/sensitivity.py`, `pages/08_Sensitivity_Analysis.py`,
`FEATURE_PARITY.md` and `ml/publication.py` all carry it. What was absent is a
Guided route, and Classic's two landmines are the reason porting it needed
thought rather than a call site.

The tests are grouped by what they are protecting:

1. **`STATE-013` cannot be inherited** — both arms score the same rows.
2. **`STATE-034` cannot be inherited** — one verdict, and it is a fact.
3. **The arms actually differ** — the guard that came out of a real defect.
4. **The record is not written** — the fork cannot become the analysis.

`GUIDED-097` — THE FIXTURE RULE. Two target shapes, and the shapes not covered
are named below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab import missingness as M
from turbotab import sensitivity as S
from turbotab.project import AnalysisProject

#: `GUIDED-097`. Both fixtures have real numeric missingness, which is what the
#: axis is about — a fixture with none would exercise the `None` path twice and
#: prove nothing about the fork.
TARGET_SHAPES = {
    "binary classification": ("metabolomics_untargeted.csv", "responder",
                              "classification", ["logreg", "gaussian_nb"]),
    "continuous regression": ("metabolomics_untargeted.csv", "run_order",
                              "regression", ["ridge", "knn_reg"]),
}

#: NOT COVERED, said out loud.
#:
#: CATEGORICAL-ONLY projects get no fork at all, and that is a real limit
#: rather than an oversight: `IMPUTE_MODE`'s alternatives on that branch are
#: `EXPLICIT_CATEGORY` and the indicator pair, and every one of those is
#: row-local in whole or in part, so a record-only swap cannot produce them.
#: `test_a_categorical_only_project_gets_silence` drives it.
#:
#: MULTICLASS is not covered for the same reason it is not covered in
#: `resolution.py` — no fixture has one. The fork's arithmetic does not depend
#: on the number of classes (it compares one metric under two arms), so the
#: expected behavior is that it works; it is simply not driven.
#:
#: SURVIVAL has no task type in this app.
SHAPES_NOT_COVERED = [
    "categorical-only missingness — no wholly-stateful counterpart exists, so "
    "the fork is silent by construction and this is asserted rather than "
    "assumed",
    "multiclass classification — no fixture; the comparison is metric-wise and "
    "should be unaffected, but it is not driven",
    "survival / time-to-event — no task type exists",
]


def _project(name, target, task, *, strategy=M.IMPUTE_MEDIAN, n_columns=5,
             fraction=0.20, branch="numeric"):
    df = pd.read_csv(f"turbotab/sample_data/{name}")
    df = df[df[target].notna()].copy()
    p = AnalysisProject.from_dataframe(df, name)
    p.target, p.task_type = target, task
    p.set_grain("not_sure")
    p.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(p.df.index)
    rng.shuffle(idx)
    labels = idx[:int(round(len(idx) * fraction))]
    p.seal_lockbox(labels, fraction=len(labels) / len(p.df))
    columns = [c["column"] for c in M.survey(p.df, p.target)
               if c["branch"] == branch][:n_columns]
    for column in columns:
        p.route_missingness(column, M.NOT_SURE, strategy)
    return p, columns


# ═══════════ 1 · `STATE-013` CANNOT BE INHERITED ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_both_arms_score_the_same_rows(shape):
    """Classic pools train+val+test and re-splits per variation, dissolving the
    lockbox for numbers printed beside lockbox-derived metrics. Here there is
    no splitter to get wrong: the counterfactual is a shallow copy that shares
    the lockbox, so the same rows are held out in both arms by construction —
    and this asserts the construction rather than trusting it."""
    name, target, task, models = TARGET_SHAPES[shape]
    p, _ = _project(name, target, task)
    other = S._counterfactual(p, S.fork(p)["swaps"])

    assert other.lockbox is p.lockbox, (
        "the alternative arm holds its own lockbox, so a future change could "
        "re-draw it")
    assert list(other.lockbox["labels"]) == list(p.lockbox["labels"])
    assert other.df is p.df, "the alternative arm holds its own frame"

    result = S.run(p, models)
    assert not result.get("unavailable"), result.get("unavailable")
    assert result["n_test"] == len(p.lockbox["labels"])
    assert result["n_train"] + result["n_test"] <= len(p.df)


def test_the_module_contains_no_splitter_at_all():
    """The structural version of the claim above.

    `STATE-013` is a landmine because nothing in Classic checks it. The check
    that would have prevented it is this one, and it is cheap: a module that
    never names a splitter cannot re-split.
    """
    import ast

    source = open("turbotab/sensitivity.py").read()
    tree = ast.parse(source)

    # PROSE IS NOT CODE. This module's docstring quotes Classic's defect by
    # name, so a substring search over the file would flag the description of
    # the thing being avoided. Docstrings are stripped through the AST rather
    # than by splitting on quotes — an earlier version of this test split on
    # `\"\"\"` and took the tail, which read almost none of the file and would
    # have passed against a module that re-split on every line.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    code = ast.unparse(tree)

    # The positive control: the test can see what IS in the module.
    assert "_counterfactual" in code and "training_rows" in code, (
        "the stripped source no longer contains the module's own names, so "
        "the absence assertions below would pass against nothing")

    for forbidden in ("train_test_split", "StratifiedShuffleSplit",
                      "ShuffleSplit", "KFold", "test_size", "_seed_tts"):
        assert forbidden not in code, (
            f"turbotab/sensitivity.py names {forbidden!r} in CODE; STATE-013 "
            f"is the landmine that begins exactly there")


# ═══════════ 2 · `STATE-034` CANNOT BE INHERITED ═══════════

#: Classic's two ladders, and every adjective that would rebuild one. A verdict
#: system is not a function — it is any prose that grades the study — so this
#: reads the rendered strings rather than counting code paths.
FORBIDDEN = [
    "highly robust", "moderately robust", "some instability", "unstable",
    "publication-ready", "without caveat", "robust", "fragile",
    "acceptable", "unacceptable", "good", "poor", "excellent",
    "✅", "⚠️", "❌",
]


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_there_is_one_verdict_and_it_is_a_fact(shape):
    """One statement, and it is *did the leader change* rather than a grade.

    Classic renders coefficient-of-variation bands beside absolute-range bands
    and they can land on different rungs. The defense is not "be careful with
    the second ladder"; it is having no ladder.
    """
    name, target, task, models = TARGET_SHAPES[shape]
    p, _ = _project(name, target, task)
    result = S.run(p, models)
    assert not result.get("unavailable"), result.get("unavailable")

    conclusion = result["conclusion"]
    assert conclusion["changed"] in (True, False)
    prose = " ".join([conclusion["sentence"], result["because"],
                      S.methods_sentence(result) or ""]).lower()
    for word in FORBIDDEN:
        assert word not in prose, (
            f"{shape}: the fork says '{word}'. STATE-034 is two invented "
            f"ladders that can contradict each other; this module has none. "
            f"Said: {prose!r}")

    # The number is reported for the reader to judge, not banded.
    assert isinstance(conclusion["largest_difference"], float)
    assert result["metric"] in S._HEADLINE[task]


def test_the_metric_names_are_the_ones_the_evaluator_produces():
    """A guessed metric name is not a crash — it is a silent *nothing to
    compare* on every project, which is the worst failure this module has.

    So the names are asserted against a real run rather than against a spelling
    that looked right. This is the test that would have caught `roc_auc` where
    `ml.eval` writes `ROC-AUC`.
    """
    from turbotab import training as _training

    for shape in sorted(TARGET_SHAPES):
        name, target, task, models = TARGET_SHAPES[shape]
        p, _ = _project(name, target, task)
        run = _training.train(p, models)
        produced = set()
        for r in run.results:
            produced |= set((r.metrics or {}).keys())
        assert produced, f"{shape}: the run produced no metric at all"
        assert produced & set(S._HEADLINE[task]), (
            f"{shape}: none of {S._HEADLINE[task]} is a name this evaluator "
            f"produces. It produces {sorted(produced)}.")


# ═══════════ 3 · THE ARMS ACTUALLY DIFFER ═══════════

def test_a_fork_whose_other_arm_never_reaches_the_fit_reports_nothing():
    """THE DEFECT THIS GUARD CAME FROM, restored and caught.

    The first `COUNTERPART` paired every fill against `INDICATOR_AND_IMPUTE`.
    Driven, both arms produced a 458-column matrix and every metric differed by
    exactly 0.0000 — because that strategy's indicator half is ROW-LOCAL and is
    written into `project.df` at Preprocess, so a record-only swap leaves it
    out. The module would have reported *this choice changes nothing* about a
    choice it had not made.
    """
    name, target, task, models = TARGET_SHAPES["binary classification"]
    p, _ = _project(name, target, task)

    original = S.COUNTERPART.get(M.IMPUTE_MEDIAN)
    try:
        S.COUNTERPART[M.IMPUTE_MEDIAN] = M.INDICATOR_AND_IMPUTE
        result = S.run(p, models)
    finally:
        S.COUNTERPART[M.IMPUTE_MEDIAN] = original

    assert result.get("unavailable"), (
        "the arms produced the same feature matrix and the fork reported a "
        "comparison anyway")
    assert "never actually varied" in result["unavailable"]
    assert "conclusion" not in result, (
        "a verdict was reported for a fork that did not fork")


def test_every_pairing_forks_between_two_wholly_deferred_strategies():
    """The structural rule behind that guard, checked over the whole table.

    A strategy with a row-local half cannot be swapped by editing the record,
    so pairing one is the defect above waiting to happen. `missingness` owns
    the answer; this reads it rather than keeping a second list.
    """
    assert S.COUNTERPART, "the table is empty; nothing can fork"
    for recorded, alternative in S.COUNTERPART.items():
        for key in (recorded, alternative):
            assert M.strategy(key)["defers"] is True, (
                f"{key} has a row-local half, so swapping the record does not "
                f"swap what the pipeline does")
            assert key not in M.ROW_LOCAL_STRATEGIES
            assert key not in M.MIXED_STRATEGIES, (
                f"{key} is compound: half of it lands at Preprocess and a "
                f"record swap cannot produce it")


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_two_arms_see_different_matrices(shape):
    """Driven directly, because it is the premise everything else rests on."""
    name, target, task, models = TARGET_SHAPES[shape]
    p, _ = _project(name, target, task)
    other = S._counterfactual(p, S.fork(p)["swaps"])
    differ, why = S._arms_differ(p, other, models[0])
    assert differ, why


# ═══════════ 4 · SILENCE, AND AN UNWRITTEN RECORD ═══════════

def test_nothing_recorded_means_no_fork():
    """Silence rather than a fork over a default nobody chose."""
    name, target, task, models = TARGET_SHAPES["binary classification"]
    p, _ = _project(name, target, task, n_columns=0)
    assert p.missingness == []
    assert S.fork(p) is None
    assert S.run(p, models) is None
    assert S.methods_sentence(None) is None


def test_a_categorical_only_project_gets_silence():
    """The named gap in `SHAPES_NOT_COVERED`, asserted rather than assumed.

    `IMPUTE_MODE` has no wholly-stateful counterpart, so a project whose only
    missingness is categorical gets no fork. The app may be silent; it may not
    report a comparison it did not make.
    """
    df = pd.DataFrame({
        "y": [0, 1] * 40,
        "num": range(80),
        "cat": (["a", "b", "c", None] * 20),
    })
    p = AnalysisProject.from_dataframe(df, "categorical.csv")
    p.target, p.task_type = "y", "classification"
    p.set_grain("not_sure")
    p.set_eligibility("everyone")
    p.seal_lockbox(list(df.index)[:16], fraction=0.2)
    p.route_missingness("cat", M.NOT_SURE, M.IMPUTE_MODE)

    assert p.missingness, "nothing was recorded, so this proves nothing"
    assert S.fork(p) is None, (
        "a categorical-only project was offered a fork; every alternative on "
        "that branch is row-local in whole or in part")


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_fork_never_writes_to_the_project(shape):
    """The alternative arm must not become the analysis by accident."""
    name, target, task, models = TARGET_SHAPES[shape]
    p, columns = _project(name, target, task)
    before = [dict(m) for m in p.missingness]
    n_decisions = len(p.decisions)

    S.run(p, models)

    assert [dict(m) for m in p.missingness] == before, (
        "the recorded plan changed; the fork overwrote the user's answer")
    assert len(p.decisions) == n_decisions, (
        "the fork appended a decision; it is a comparison, not a choice")


def test_the_counterfactual_record_is_internally_consistent():
    """A swapped record must not say one thing and fit another.

    Replacing `strategy` alone would leave `label`, `sentence`, `fit_on` and
    `defers` describing the strategy the user chose — which is `GUIDED-089`
    reintroduced inside the module built to compare two honest plans.
    """
    name, target, task, _ = TARGET_SHAPES["binary classification"]
    p, columns = _project(name, target, task)
    other = S._counterfactual(p, S.fork(p)["swaps"])

    swapped = {c: r for c, r in
               ((str(r["column"]), r) for r in other.missingness)}
    for column in columns:
        record = swapped[column]
        spec = M.strategy(record["strategy"])
        assert record["strategy"] == M.IMPUTE_MICE
        assert record["label"] == spec["label"]
        assert record["because"] == spec["because"]
        assert record["defers"] is spec["defers"]
        assert record["sentence"] == M.sentence_for(
            column, record["branch"], record["strategy"])
        assert M.strategy(M.IMPUTE_MEDIAN)["label"] not in record["sentence"]


# ═══════════ THE EVIDENCE, AND THE PROSE ═══════════

def test_every_source_is_a_real_heading_in_a_real_pack():
    """Domain science comes from the research files, never from recollection.

    The evidence gate checks this repo-wide; asserted here too so the failure
    names this module when a pack is reorganized.
    """
    import re
    from pathlib import Path

    heading = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.M)
    for key, badge in S.SOURCES.items():
        filename, _, section = badge["source"].partition("#")
        path = Path("docs/turbotab") / filename
        assert path.exists(), f"{key}: {filename} does not exist"
        sections = {m.group(1).strip()
                    for m in heading.finditer(path.read_text())}
        assert section in sections, f"{key}: no section {section!r} in {filename}"
        assert badge["evidence_status"] in ("SETTLED", "CONVENTION", "DISPUTED")


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_methods_sentence_says_what_was_varied_and_what_happened(shape):
    name, target, task, models = TARGET_SHAPES[shape]
    p, columns = _project(name, target, task)
    result = S.run(p, models)
    line = S.methods_sentence(result)

    assert "same training rows" in line and "same held-out rows" in line, (
        "the manuscript line does not say the split was held fixed, which is "
        "the one thing a reader needs to know to trust the comparison")
    assert columns[0] in line
    assert result["conclusion"]["sentence"] in line
    for word in FORBIDDEN:
        assert word not in line.lower()


# ═══════════ THE ROUTE ═══════════
#
# `LOOP.md` §05: a capability ships with its consumer. The consumer is
# `/project/{id}/sensitivity`, driven here against the real API rather than
# asserted from the source — `GUIDED-080`'s class is a server that composes a
# string nothing fetches, and an endpoint nobody calls is the same shape.

_needs_js_for_the_fork = pytest.mark.skipif(
    not __import__('turbotab.pageharness', fromlist=['x']).available(),
    reason='no JS engine on this machine')


def _client_and_project(shape):
    from fastapi.testclient import TestClient

    from turbotab import api
    name, target, task, models = TARGET_SHAPES[shape]
    p, columns = _project(name, target, task)
    api.STORE.add(p)
    p.selected_models = models
    return TestClient(api.app), p, columns


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_route_serves_the_comparison(shape):
    client, p, columns = _client_and_project(shape)
    body = client.get(f"/project/{p.id}/sensitivity").json()

    assert body["available"] is True, body.get("because")
    result = body["result"]
    assert result["axis"] == "missingness"
    assert result["conclusion"]["changed"] in (True, False)
    assert result["n_test"] == len(p.lockbox["labels"])
    assert body["methods_sentence"]
    assert columns[0] in body["methods_sentence"]
    for word in FORBIDDEN:
        assert word not in body["methods_sentence"].lower()


def test_the_three_empty_answers_are_three_different_sentences():
    """A client that could not tell them apart would render one as the others.

    Nothing recorded, no models chosen, and arms that did not differ are three
    facts, and the route says which.
    """
    from fastapi.testclient import TestClient

    from turbotab import api

    name, target, task, models = TARGET_SHAPES["binary classification"]
    said = {}

    nothing, _ = _project(name, target, task, n_columns=0)
    api.STORE.add(nothing)
    nothing.selected_models = models
    said["nothing recorded"] = TestClient(api.app).get(
        f"/project/{nothing.id}/sensitivity").json()

    no_models, _ = _project(name, target, task)
    api.STORE.add(no_models)
    no_models.selected_models = []
    said["no models"] = TestClient(api.app).get(
        f"/project/{no_models.id}/sensitivity").json()

    same, _ = _project(name, target, task)
    api.STORE.add(same)
    same.selected_models = models
    original = S.COUNTERPART.get(M.IMPUTE_MEDIAN)
    try:
        S.COUNTERPART[M.IMPUTE_MEDIAN] = M.INDICATOR_AND_IMPUTE
        said["arms identical"] = TestClient(api.app).get(
            f"/project/{same.id}/sensitivity").json()
    finally:
        S.COUNTERPART[M.IMPUTE_MEDIAN] = original

    for case, body in said.items():
        assert body["available"] is False, case
        assert body["result"] is None, case
        assert body["because"], f"{case}: an empty answer with no sentence"

    reasons = [b["because"] for b in said.values()]
    assert len(set(reasons)) == 3, (
        f"three different facts produced {len(set(reasons))} distinct "
        f"sentence(s): {reasons}")

    assert "nothing to run both ways" in said["nothing recorded"]["because"]
    assert "No models have been chosen" in said["no models"]["because"]
    assert "never actually varied" in said["arms identical"]["because"]
    # The two that CAN name the axis do, and the one that cannot does not
    # invent one.
    assert said["nothing recorded"].get("fork") is None
    assert said["no models"]["fork"]["axis"] == "missingness"


@_needs_js_for_the_fork
def test_the_fork_reaches_the_reader():
    """`GUIDED-080`'s class again, and this one was caught by a guard rather
    than by foresight.

    `test_every_server_surface_names_its_reader` failed on
    `/project/{id}/sensitivity` the first time the full suite ran: the route
    was composed, tested and served, and the Guided door fetched nothing. That
    is exactly the shape `MISC-014` is about one layer out — a capability that
    exists and is unrouted — arriving inside the fix for it.
    """
    from turbotab import pageharness as PH

    name, target, task, models = TARGET_SHAPES["binary classification"]
    p, columns = _project(name, target, task)
    from turbotab import api
    api.STORE.add(p)
    p.selected_models = models

    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    project = client.get(f"/project/{p.id}").json()
    served = client.get(f"/project/{p.id}/sensitivity").json()
    assert served["available"] is True, served.get("because")

    routes = {
        f"/project/{p.id}": project,
        f"/project/{p.id}/interview?step=data":
            client.get(f"/project/{p.id}/interview?step=data").json(),
        f"/project/{p.id}/interview?step=explore": {"questions": [], "steps": []},
        f"/project/{p.id}/evidence/missingness": {"cards": []},
        f"/project/{p.id}/evidence/plausibility": {"columns": []},
        f"/project/{p.id}/draft": {"paragraphs": []},
        f"/project/{p.id}/gaps": {"gaps": []},
        f"/project/{p.id}/explain": {"run": None, "blocked_by": None,
                                     "costly_decisions": [], "stale": []},
        f"/project/{p.id}/sensitivity": served,
    }
    out = PH.run("__emit(__harness.html('explainBox'));", routes=routes,
                 search=f"?project={p.id}")

    assert out, "the Explain surface rendered nothing at all"
    assert "Run the other way" in out, (
        "the server served a sensitivity comparison and the page rendered "
        "none of it")
    assert served["result"]["conclusion"]["sentence"] in out
    for word in FORBIDDEN:
        assert word not in out.lower(), f"the rendered fork says {word!r}"
