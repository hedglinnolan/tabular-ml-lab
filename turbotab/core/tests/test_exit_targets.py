"""G12 (the owner's Q3, 2026-11-04: "Hard agree on dead buttons issue"): a way forward that names
no decision names the card it opens, its quest stage and line (``turbotab.core.exit_target``), so
pressing it opens that card instead of only dismissing the refusal (FOUNDATION §9 item 1: each
item's exit opens the stage that owns it).

The truth is the served quest log (``quest.quest_log``, built from the state by hand as
``test_confirm_sweep`` builds it): a target names a stage that serves a line under that key, or a
stage alone. The refusals' normalizers carry the target as given and serve none where none is
named; one that names nothing the quest log serves is refused, as an unreadable decision is.

Every way forward in the engine that names no decision is read from the source (an exit literal,
``{"label": ..., "decision": None}``): each names its target, or is listed below with why it opens
no card. A new label-only way forward is classified when it is written.
"""
from __future__ import annotations

import ast
import pickle
from pathlib import Path

import pytest

import turbotab
from turbotab.core import decisions, seal
from turbotab.core.decisions import Refusal
from turbotab.core.exit_target import exit_target, toward
from turbotab.core.graph import StageRefusal
from turbotab.core.tests.test_confirm_sweep import log_of, planned
from turbotab.core.tests.test_opening_gate import BLOCKER, OPEN_NOTICING, opening_ctx, triaged

ENGINE = Path(turbotab.__file__).resolve().parent


def served_keys(log) -> dict[str, set[str]]:
    """Each stage's line keys, as the quest log serves them."""
    return {s.key: {l.key for l in s.lines} for s in log.stages}


LOGS = {
    "inference": log_of(planned()),
    "prediction": log_of(planned(purpose="prediction")),
}


# ── the target names a card the quest log serves ─────────────────────────────


@pytest.mark.parametrize("goal", sorted(LOGS))
def test_a_question_or_declaration_names_the_stage_that_serves_its_line(goal):
    log = LOGS[goal]
    keys = served_keys(log)
    named = [(s.key, l.key) for s in log.stages for l in s.lines
             if l.source in ("question", "declaration") and not l.key.startswith("ask:")]
    assert len(named) >= 10  # the log serves the questions and declarations of a planned project
    for stage, key in named:
        target = toward(key)
        assert target["key"] == key, (stage, key, target)
        assert key in keys[target["stage"]], (stage, key, target)


def test_a_decision_kind_names_the_line_it_answers_where_the_quest_log_serves_one():
    log = LOGS["prediction"]
    keys = served_keys(log)
    everywhere = {k for ks in keys.values() for k in ks}
    # Each from the log's own lines: the outcome, the purpose, the missing-values question and the
    # predictor-selection declaration, by the kinds that answer them.
    for kind, line in (("set_target", "target"), ("set_purpose", "purpose"),
                       ("set_missing", "missing"), ("set_selection", "set_selection"),
                       ("select_models", "models"), ("set_roles", "roles")):
        assert line in everywhere, line
        target = toward(kind)
        assert target["key"] == line and line in keys[target["stage"]], (kind, target)


def test_a_stage_alone_opens_on_its_open_line_and_a_line_by_its_served_key():
    log = LOGS["prediction"]
    assert toward(stage="results") == {"stage": "results", "key": None}
    # The models question, not the Models stage: a name is a question first.
    assert "models" in served_keys(log)["models"]
    assert toward("models") == {"stage": "models", "key": "models"}
    assert toward(stage="models") == {"stage": "models", "key": None}
    assert toward(stage="first_look", key="heaping__age") == {"stage": "first_look",
                                                               "key": "heaping__age"}


@pytest.mark.parametrize("bad", [
    lambda: toward("no_such_question"),
    lambda: toward(stage="nowhere"),
    lambda: toward("missing", stage="whos_in"),
    lambda: toward("confirm_sweep"),  # it sits in the stage it confirms: no card of its own
    lambda: exit_target({"stage": "results", "line": "x"}),
    lambda: exit_target({"stage": "results", "key": 3}),
    lambda: exit_target(7),
])
def test_a_target_that_names_nothing_the_quest_log_serves_is_refused(bad):
    with pytest.raises(ValueError):
        bad()


# ── the refusals carry it ────────────────────────────────────────────────────


AT_MISSING = {"stage": "whos_in", "key": "missing"}  # where the log serves the question
WAYS = [{"label": "Answer the missing-values question first", "decision": None,
         "target": "missing"},
        {"label": "Fill them", "decision": None, "target": AT_MISSING},
        {"label": "Keep the answers as they are", "decision": None}]


def test_the_log_serves_the_missing_values_question_where_the_fixture_says():
    assert "missing" in served_keys(LOGS["inference"])[AT_MISSING["stage"]]


def test_an_answer_time_refusal_serves_each_target_and_none_where_none_is_named():
    r = Refusal("x", "Refused.", WAYS)
    assert [e.get("target") for e in r.exits] == [AT_MISSING, AT_MISSING, None]
    assert "target" not in r.exits[2]  # served as before: label and decision only
    assert r.exits[2] == {"label": "Keep the answers as they are", "decision": None}
    assert r.to_dict()["error"]["exits"][0]["target"] == AT_MISSING


def test_a_stage_refusal_serves_each_target_through_its_pipe_and_its_answers():
    r = StageRefusal("x", "Refused.", WAYS)
    assert [e.get("target") for e in r.exits] == [AT_MISSING, AT_MISSING, None]
    again = pickle.loads(pickle.dumps(r))  # the job pipe carries it to the server
    assert again.as_refusal()["exits"] == r.as_refusal()["exits"]
    assert "target" not in again.as_refusal()["exits"][2]
    changed = StageRefusal("x", "Refused.", WAYS, changes=True).on_answers(planned())
    assert [e.get("target") for e in changed.exits] == [AT_MISSING, AT_MISSING, None]


def test_a_refusal_with_a_target_that_names_nothing_is_refused_when_it_is_raised():
    with pytest.raises(ValueError):
        Refusal("x", "Refused.", [{"label": "Somewhere", "decision": None, "target": "nowhere"}])
    with pytest.raises(ValueError):
        StageRefusal("x", "Refused.", [{"label": "Somewhere", "decision": None,
                                        "target": {"stage": "nowhere"}}])


# ── the high-traffic refusals: the opening's triage ──────────────────────────


def _opening_refusal(findings):
    state, log, t = triaged(findings)
    with pytest.raises(Refusal) as refused:
        seal._the_opening_waits_for_the_triage(decisions.OpenSeal(family="linear"),
                                               opening_ctx(state, t))
    return refused.value, log, t


def test_the_triage_refusal_opens_the_triage_s_stage():
    r, _log, t = _opening_refusal(OPEN_NOTICING)
    assert r.code == "triage_open"
    to_triage = next(e for e in r.exits if e["decision"] is None)
    assert to_triage["label"] == "Triage the open noticings in Results"
    assert to_triage["target"] == {"stage": t.stage, "key": None}
    assert t.stage == "results"  # under Predict the gate is the opening (sweep.gate_stage)
    keep = next(e for e in r.exits if e["decision"] is not None)
    assert "target" not in keep  # a decision is recorded by its press, as before


def test_a_blocker_s_way_forward_opens_the_noticing_s_own_line():
    r, log, t = _opening_refusal(BLOCKER)
    assert r.code == "triage_open"
    (decide,) = r.exits
    blocker = next(i for i in t.items if i.blocker)
    assert decide["label"] == f"Decide: {blocker.summary}"
    # The noticing's line, where the quest log serves it.
    where = [s for s, keys in served_keys(log).items() if blocker.id in keys]
    assert where, "the log serves the blocker's noticing as a line"
    assert decide["target"] == {"stage": where[0], "key": blocker.id}


# ── every label-only way forward in the engine is classified ─────────────────

KEEPS = "keeps what is recorded: it opens no card, and pressing it dismisses the refusal"
WAITS = "waits for work under way to finish: no card holds it"
NOT_YET = "names an analysis TurboTab does not have yet: no card offers it"
SAME_PRESS = "asks for the same press again with a shorter note: no other card"
OUTSIDE = "said before a project is open, or outside it (sign-in, opening a file): no quest card"
NO_KIND = "answers ``set_tuning``, which the log does not record yet (RT-6): no card"
NOT_AN_EXIT = "a labeled option with no decision (a card's group, a screen's route), not a refusal's"
FILE = "a change to a file outside TurboTab: no card holds it"
RERUN = "computes a stage again, which its retry beside its failure does: no card"
ROWS = ("asks for more rows, which the code names no one answer for (``exposure_form.more_rows``): "
        "no card")
ENGINE_GUARD = ("a guard on the engine's own plan (a step scoped after the seal asked to run "
                "before it): no answer of the researcher's changes it")
CONTINUES = "continues past an empty triage, at the stage's own next act: no other card"
# The fit's tables serve their ways forward as ``InferenceExit`` (core/models/artifacts.py), which
# forbids any other field, and these reach it (a table's refusal, a survey or missing-values block,
# a grouping's floor): a target there needs the artifact models to carry one (an open issue).
ARTIFACT = "reaches a fit table's ``InferenceExit``, which carries no target"
HELD = "in core/fit_press.py, held by pp3-act-on-hold: targeted there"

# (path under turbotab/, the label's source) -> why it names no card.
UNTARGETED: dict[tuple[str, str], str] = {
    ("core/decisions.py", "'Keep the data as they are for a later version'"): KEEPS,
    ("core/decisions.py", "'Keep the answers as they are'"): KEEPS,
    ("core/decisions.py", "f'Keep `{decision.column}` as it is, without a follow-up'"): KEEPS,
    ("core/decisions.py", "first"): NO_KIND,
    ("core/decisions.py", "'Keep the standard setting'"): NO_KIND,
    ("core/decisions.py", "'Shorten the note'"): SAME_PRESS,
    ("core/designs.py", "'Keep the data as they are for a later version'"): KEEPS,
    ("core/estimand.py", "'No guess'"): NOT_AN_EXIT,
    ("core/estimand.py", "'Keep the outcome as it is'"): KEEPS,
    ("core/estimand.py", "'Respond once the estimates are shown'"): WAITS,
    ("core/estimand.py", "'Keep the analysis as it is'"): KEEPS,
    ("core/estimand.py", "'Wait for the effects'"): WAITS,
    ("core/export/gate.py", "f'Put `{f.name}` back where it was read'"): FILE,
    ("core/export/gate.py",
     "f'Restore `{f.name}` as it was read, or open the new file as a new project'"): FILE,
    ("core/export/gate.py", "f'Compute {label[:1].lower()}{label[1:]} again'"): RERUN,
    ("core/export/gate.py", "'Export again once it is done'"): WAITS,
    ("core/fit_press.py", "PRESS_FIT"): HELD,
    ("core/fit_press.py", "'Choose the goal'"): HELD,
    ("core/fit_press.py", "f'Triage the open noticings in {where}'"): HELD,
    ("core/fit_press.py", "f'Decide: {i.summary}'"): HELD,
    ("core/methods/exposure_form.py", "last"): ROWS,
    ("core/methods/qc_drift.py", "'Run it in each training fold'"): ENGINE_GUARD,
    ("core/methods/survey.py",
     "'Answer the survey question: the surveyed population, or these participants'"): ARTIFACT,
    ("core/methods/survey.py", "NAME_THE_FOUR_YEAR"): ARTIFACT,
    ("core/methods/survey.py", '"Check the survey question\'s strata and PSU"'): ARTIFACT,
    ("core/models/featurewise.py",
     "'Give the columns you study the role of what you study (the roles question)'"): ARTIFACT,
    ("core/models/featurewise.py", "'Choose another model family'"): ARTIFACT,
    ("core/models/featurewise.py",
     "'Revisit what each test adjusts for (the adjustment question)'"): ARTIFACT,
    ("core/models/featurewise.py", 'f"Combine each `{clusters.column}`\'s rows into one '
                                   '(the unit question, before the seal)"'): ARTIFACT,
    ("core/models/inference.py",
     'f"Combine each `{column}`\'s rows into one (the unit question, before the seal)"'): ARTIFACT,
    ("core/models/inference.py", "'Leave one of the dependent columns out of the model'"): ARTIFACT,
    ("core/models/inference.py", "f'Leave {named} out, or merge its rare levels'"): ARTIFACT,
    ("core/models/inference.py",
     "'Leave columns out of the model, so it has fewer coefficients than rows'"): ARTIFACT,
    ("core/models/inference.py", "f'Fit the model on more than {n_coefficients:,} rows'"): ARTIFACT,
    ("core/models/inference.py",
     "f'Read {_and(columns)} another way, or combine {whose} sparse levels'"): ARTIFACT,
    ("core/models/inference.py", "'A shared-frailty Cox model (not yet in TurboTab)'"): NOT_YET,
    ("core/models/inference.py", "'A random-intercept logistic model (not yet in TurboTab)'"):
        NOT_YET,
    ("core/models/inference.py", "f'Leave {_and(cols)} out, or merge its rare levels'"): ARTIFACT,
    ("core/models/inference.py", 'f"Combine each `{clusters.column}`\'s rows into one '
                                 '(the unit question, before the seal)"'): ARTIFACT,
    ("core/models/inference.py", "f'Leave {_and(separated)} out, or merge its rare levels'"):
        ARTIFACT,
    ("core/models/inference.py", "'Name the column that identifies the unit (the grain question)'"):
        ARTIFACT,
    ("core/models/repeated.py",
     'f"Combine each `{column}`\'s rows into one (the unit question, before the seal)"'): ARTIFACT,
    ("core/models/repeated.py", "'Leave one of the dependent columns out of the model'"): ARTIFACT,
    ("core/models/repeated.py", "f'Leave {_and(separated)} out, or merge its rare levels'"):
        ARTIFACT,
    ("core/models/survey.py", "f'Leave {_and(separated)} out, or merge its rare levels'"): ARTIFACT,
    ("core/models/survey.py", '"Estimate for these participants instead (the survey question), '
                              'where Firth\'s fit is available"'): ARTIFACT,
    ("core/models/survival.py", "f'Leave {_and(cols)} out, or merge its rare levels'"): ARTIFACT,
    ("core/readings.py", "'Name the column that identifies the unit (the grain question)'"):
        ARTIFACT,
    ("core/reference_rows.py", "'Keep the rows as they are'"): KEEPS,
    ("core/row_floor.py", "'Keep the answers as they are'"): KEEPS,
    ("core/scales.py", "'Keep the answers as they are'"): KEEPS,
    ("core/seal.py", "'Keep the held-out rows as they were drawn'"): KEEPS,
    ("core/seal.py", "'Keep the held-out rows as they are'"): KEEPS,
    ("core/seal.py", "'Keep the scheme as it was'"): KEEPS,
    ("core/seal.py", "'Keep the scheme as it is'"): KEEPS,
    ("core/seal.py", "'Wait for the checks'"): WAITS,
    ("core/seal.py", "'Wait for the fit to finish'"): WAITS,
    ("core/seal.py", "'Keep the re-seal'"): KEEPS,
    ("core/sequence.py", "'Decide it once its card is shown'"): WAITS,
    ("core/sequence.py", "'Keep it as it is'"): KEEPS,
    ("core/stages/causal.py", "'Keep the primary model only'"): KEEPS,
    ("core/stages/modeling.py", "f'Say how the zeros of `{zeros[0]}` are read: values below a "
                                "detection limit, or true zeros'"): ARTIFACT,
    ("core/structural.py", '"Keep the outcome\'s scale"'): KEEPS,
    ("core/sweep.py", "'Continue'"): CONTINUES,
    ("eligibility.py", "'Mark the whole column as not trustworthy'"): NOT_AN_EXIT,
    ("server/auth.py", "'Sign in'"): OUTSIDE,
    ("server/routes/projects.py", "'Upload the file instead'"): OUTSIDE,
}


def _label_only_exits():
    """Every exit literal in the engine whose decision is None: (path, label source, its target
    source or None)."""
    out = []
    for path in sorted(ENGINE.rglob("*.py")):
        rel = path.relative_to(ENGINE).as_posix()
        if "/tests/" in f"/{rel}" or path.name.startswith("test_") or rel.startswith("frontend/"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Dict):
                continue
            keys = [k.value if isinstance(k, ast.Constant) else None for k in node.keys]
            if "label" not in keys or "decision" not in keys:
                continue
            decision = node.values[keys.index("decision")]
            if not (isinstance(decision, ast.Constant) and decision.value is None):
                continue
            label = ast.unparse(node.values[keys.index("label")])
            target = (node.values[keys.index("target")] if "target" in keys else None)
            out.append((rel, label, target, node.lineno))
    return out


EXITS = _label_only_exits()


def test_the_engine_s_label_only_ways_forward_are_read():
    assert len(EXITS) > 250  # the scan reads them (288 on the base commit)


def test_every_label_only_way_forward_names_its_card_or_is_listed_with_why_not():
    unclassified = [f"{rel}:{n} {label}" for rel, label, target, n in EXITS
                    if target is None and (rel, label) not in UNTARGETED]
    assert not unclassified, "\n".join(unclassified)


def test_no_listed_way_forward_is_stale():
    """Each listed entry still names a way forward without a target (the same label may be
    targeted elsewhere in its file: the four-year weight's, raised at the fit, opens the survey
    question, while the fit table's copy cannot carry one)."""
    untargeted = {(rel, label) for rel, label, target, _ in EXITS if target is None}
    assert set(UNTARGETED) <= untargeted, sorted(set(UNTARGETED) - untargeted)


def test_every_target_written_as_a_name_names_a_card_the_quest_log_serves():
    """A target given by constants (``toward("missing")``, ``toward(stage="results")``) resolves;
    one given by a value the refusal computes is read by the tests of its own refusal."""
    read = 0
    for rel, _label, target, n in EXITS:
        if not (isinstance(target, ast.Call) and ast.unparse(target.func) == "toward"):
            continue
        args = [a.value for a in target.args if isinstance(a, ast.Constant)]
        kw = {k.arg: k.value.value for k in target.keywords if isinstance(k.value, ast.Constant)}
        if len(args) != len(target.args) or len(kw) != len(target.keywords):
            continue
        toward(*args, **kw)  # raises for a name the log does not serve
        read += 1
    assert read >= 100, read
