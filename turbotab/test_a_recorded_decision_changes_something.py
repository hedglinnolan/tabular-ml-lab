"""The probe that would have caught `GUIDED-095` on the day it appeared.

## Why a probe rather than another finding

`GUIDED-095` was found by a census — somebody read `training.py`, counted the
project attributes it touched, and compared that to the 36 kinds of decision the
app records. That works once. It does not run every loop, and the next severed
seam will be somewhere nobody thinks to count.

So: **for each recorded decision kind, build two projects identical except for
that answer, and assert that something downstream differs.** A decision whose
flip changes nothing anywhere is severed connective tissue, whatever the
receipt says.

## What "downstream" means here, and what it deliberately excludes

The record of the decision itself is NOT downstream. `set_selection` writes
`selection_spec`, and reading `selection_spec` back to prove `set_selection`
worked is the tautology this probe exists to break — it is precisely how
`GUIDED-095` stayed invisible for eight loops, because every surface that
mattered read the record and reported it faithfully.

So `_fingerprint` strips the fields that ARE the record — `engineered`,
`deferred_transforms`, `selection`, `declared`, `selected`, the decision log —
and keeps what is computed FROM them: the working table's columns, the ranked
findings, the figures, the offers, the composed prose, the interview's next
question, and **the fold-fitted pipeline plan**, which is the surface
`GUIDED-095` was about.

## The honest output is the three counts, not the passing test

`test_the_probe_reports_its_own_coverage` prints and asserts them: kinds
probed, kinds that propagated, kinds allow-listed. A fourth count exists
because it has to — kinds for which no flip could be constructed, each with the
reason. A sweep that reports only what it fixed has not reported its coverage.

**The allow-list is in this file, not in a comment**, and each entry states why
that decision legitimately reaches nothing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api                                              # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


# ─────────────────────────────────────────────────────────────────────────────
# The kinds
# ─────────────────────────────────────────────────────────────────────────────

def recorded_kinds():
    """Every decision kind the API accepts, read off the API.

    A list written here would go stale the first time somebody added a kind,
    and a probe that silently stops covering a kind is worse than no probe.

    **THREE FORMS, and the third was missing until L41-D** (`GUIDED-141`). The
    dispatcher tests `decision.kind` three ways and this read two of them:

    * `decision.kind == "x"` — the chain, 38 of them;
    * `decision.kind not in {…}` — the record-only fallthrough, 6;
    * `decision.kind in ("a", "b", …)` — a **group** that shares a body, 4.

    Three of that last group are also in the `==` chain, so they arrived
    anyway. `set_temporal_prediction` is not, and it is only reachable through
    the group's own `else:` — so the denominator every count in this file is
    computed against was one short, and that kind had never been probed.

    This is the same defect this enumerator was written to fix, in a form
    nobody had added yet. `LOOP.md` §03 records the first version: *41 decision
    kinds enumerated — the adjudicator's grep had said 36*. The lesson is not
    that the count was wrong; it is that **a registry recovered by pattern-
    matching a dispatcher is only as complete as the patterns**, and the honest
    form of that is to say how many each pattern contributed.
    """
    import re

    source = (Path(__file__).resolve().parent / "api.py").read_text(
        encoding="utf-8")
    handled = set(re.findall(r'decision\.kind == "([a-z_]+)"', source))
    fallthrough = re.search(r'decision\.kind not in \{([^}]+)\}', source)
    if fallthrough:
        handled |= set(re.findall(r'"([a-z_]+)"', fallthrough.group(1)))
    for group in re.finditer(r'decision\.kind in \(([^)]+)\)', source):
        handled |= set(re.findall(r'"([a-z_]+)"', group.group(1)))
    return sorted(handled)


#: Decisions that legitimately reach nothing downstream, with the reason.
#: **In the file, not in a comment**, because an allow-list that lives in prose
#: is an allow-list nobody re-reads.
ALLOWED_NO_EFFECT = {
    "eligibility_candidates": (
        "A GET-shaped READ served through the decision endpoint so it stays "
        "beside the question it belongs to. It records nothing and changes "
        "nothing by construction — it returns the candidate criteria a pack "
        "has offered, resolved against the frame."),
    "eligibility_evidence": (
        "The same shape, and bounded by clause §04: it answers *is this data "
        "corrupted?* and cannot answer *where should I cut?*. A read, not a "
        "decision."),
    "note": (
        "A free-text line in the transcript, and that is the whole of it. It "
        "reaches the manuscript as prose and is deliberately not wired to any "
        "computation — a note that changed a number would be a decision "
        "wearing a comment's clothes."),
    "flag": (
        "Marks a finding for the exhibit dock. It changes what the EXPORT "
        "carries rather than what any number is, and the export is L10."),
    "unflag": (
        "The inverse of `flag`. It removes a finding from the exhibit dock, "
        "which changes what the export carries and no computed number, for "
        "exactly the reason `flag` does."),
    "dismiss": (
        "Records that a noticing was considered and let go. §09's "
        "recorded-absence rule is the point: dismissed and never-reached are "
        "different states, and neither changes a computation."),
    "undismiss": (
        "The inverse of `dismiss`. It returns a noticing to the ranked list "
        "the user reads; the list itself is recomputed from the data either "
        "way, so nothing computed moves."),
    "defer": (
        "Holds a noticing until the step it targets. The resurfacing is a "
        "render of the record at that step rather than a change to anything "
        "computed, which is what makes deferral reversible."),
    "earmark": (
        "Records a decision that lives somewhere else — another step, or a "
        "person. Its whole content is where it resurfaces; an earmark that "
        "changed a number would be the app acting on work it just said it "
        "cannot do."),
    "acknowledge_blocker": (
        "The typed acknowledgment that lets a `blocker` be passed unresolved. "
        "It changes what the manuscript must carry as a limitation, not what "
        "is computed — the whole design is that the app does not refuse the "
        "user's judgment, it refuses silence."),
    "unskip": (
        "Reopens a question the Router skipped. It records NOTHING about the "
        "answer, deliberately — `GUIDED-041`'s entire defect was that "
        "reopening wrote one. The effect is on the next interview plan, which "
        "this probe reads, but a project with nothing skipped has nothing to "
        "reopen and the fixtures here are in that state."),
}


#: Kinds this probe could not construct a flip for, each with the reason.
#: **A fourth count, not a silent omission.**
NOT_CONSTRUCTED = {
    # `GUIDED-143` L43-C.
    "set_time_column": (
        "The flip needs two columns that both parse as dates, and the app "
        "refuses any column that does not — `set_time_column` validates by "
        "PARSING rather than by name, on purpose, because `visit_date`, "
        "`date_of_visit`, `dov` and `RecallDate` are four spellings of one "
        "thing. `clinical_longitudinal.csv` carries exactly one date column, "
        "so the second arm would have to be a refusal rather than an answer, "
        "and a refusal is not the same project with one answer changed. "
        "WHAT IS PROBED INSTEAD, in "
        "`test_the_chronological_split_is_drawn.py`: recording the column "
        "flips the seal's basis from `chronological_requested_not_drawn` to "
        "`chronological_grouped`, changes which rows are held out, and is "
        "refused after the barrier — three downstream effects, driven end to "
        "end through the real routes. The effect is measured; it is the "
        "TWO-ARM SHAPE this probe requires that the fixture cannot supply."),
    "set_orientation": (
        "Fires only where an assay lens meets a feature-major table, and the "
        "answer that flips it — *the table was already one row per sample* — "
        "is refused once the diagnosis has turned the frame around. Reaching "
        "both arms needs two uploads of two differently-shaped files, which is "
        "a fixture this repository does not have."),
    "set_aggregation": (
        "Requires a repeated-measures table whose grain answer, repeat kind "
        "and unit of analysis are all set to the combination that makes "
        "aggregation legal. `clinical_longitudinal.csv` reaches the question "
        "and both of its answers refuse on this fixture's row counts."),
    "apply_bulk": (
        "Needs a repair GROUP — several columns sharing one fix kind — that "
        "survives to the point of being applicable. The fixtures produce "
        "groups whose members the diagnosis re-derives differently after the "
        "first apply, so the two arms are not the same project with one "
        "answer changed."),
    "decline_bulk": (
        "The other arm of `apply_bulk` — *leave all N as they are* — and "
        "unreachable for the same construction reason: there is no stable "
        "group in these fixtures to decline."),
    "except_from_bulk": (
        "Edits the membership rule of a bulk group, so it inherits "
        "`apply_bulk`'s construction problem."),
    "resolve_blocker": (
        "Drops the column a leakage blocker names. Constructing the flip needs "
        "a project where a blocker is live AND the column is droppable without "
        "changing which questions the Router then asks, and the leaky fixture "
        "does not separate those."),
    "revert": (
        "Undoes the last applied fix, so its flip is *apply then revert* "
        "against *never apply* — which is the same project by construction "
        "when revert works, and that is what `apply`'s own probe already "
        "asserts from the other side."),
}


# ─────────────────────────────────────────────────────────────────────────────
# The flips
# ─────────────────────────────────────────────────────────────────────────────
#
# Each entry: the fixture, the decisions that get the project to the point where
# the kind is answerable, and TWO answers. Everything before the flip is
# identical, so anything that differs afterwards differs because of the answer.

_CLINIC = "clinic_visits.csv"
_METAB = "metabolomics_untargeted.csv"
_SEPSIS = "leaky_sepsis.csv"
_SURVEY = "survey_instrument.csv"
_LONG = "clinical_longitudinal.csv"

_TO_TARGET = [("set_target", {"column": "hba1c"})]
_TO_SEAL = _TO_TARGET + [
    ("set_purpose", {"answer": "prediction"}),
    ("set_grain", {"answer": "one_row_per_person"}),
    ("set_eligibility", {"answer": "everyone"}),
    ("seal", {"fraction": 0.25}),
]
_TO_MODELS = _TO_SEAL + [("select_models", {"models": ["ridge", "histgb_reg"]})]

FLIPS = {
    "set_lens": (_METAB, [], ("set_lens", {"lens": ["metabolomics"]}),
                 ("set_lens", {"lens": ["other"]})),
    # `GUIDED-108`. Putting an identifier-like column back into the models,
    # and taking it out again. It reaches the FIT, which is the point: the
    # column was excluded from what the model is fed.
    "keep_identifier": (
        _SURVEY, [],
        ("__identifier_sealed__", {"column": "respondent_id", "kept": True}),
        ("__identifier_sealed__", {"column": "respondent_id", "kept": False})),
    # `GUIDED-107`. Placing a figure in the results, and taking it back out.
    # The flip is the smallest one in this table and it caught a real gap: the
    # first version recorded the decision, set `promoted_figures` on the
    # project, and did not serialize it — so a decision the transcript showed
    # changed nothing any caller could see, which is a decision in name only.
    "promote_figure": (
        _METAB, [],
        ("promote_figure", {"figure_id": "pca_scores", "promoted": True}),
        ("promote_figure", {"figure_id": "pca_scores", "promoted": False})),
    # The same flip, driven to a SEAL, so the fold-fitted pipeline exists to
    # compare. Unsealed it could only ever move the display surfaces — which is
    # how a lens that reached the lattice and not the fit stayed invisible.
    "set_lens@sealed": (
        _METAB, [], ("__lens_sealed__", {"lens": ["metabolomics"]}),
        ("__lens_sealed__", {"lens": ["other"]})),
    "set_target": (_CLINIC, [], ("set_target", {"column": "hba1c"}),
                   ("set_target", {"column": "glucose"})),
    "set_task_type": (_LONG, [("set_target", {"column": "visit"})],
                      ("set_task_type", {"task_type": "classification"}),
                      ("set_task_type", {"task_type": "regression"})),
    "set_purpose": (_CLINIC, _TO_TARGET,
                    ("set_purpose", {"answer": "prediction"}),
                    ("set_purpose", {"answer": "inference"})),
    # BOTH ARMS ARE ANSWERS THE APP ACCEPTS WITHOUT AN EXTRA DECISION.
    # `one_row_per_person` on this table raises the contradiction check — which
    # is the app working — and acknowledging it would make the two arms differ
    # by two answers rather than one. `not_sure` is first-class here (§03's
    # `undetermined` seal) and is the honest second arm.
    "set_grain": (_LONG, [("set_target", {"column": "sbp"}),
                          ("set_purpose", {"answer": "prediction"})],
                  ("set_grain", {"answer": "not_sure"}),
                  ("set_grain", {"answer": "people_repeat",
                                 "group_col": "subject_id"})),
    "set_repeat_kind": (_LONG, [("set_target", {"column": "sbp"}),
                                ("set_purpose", {"answer": "prediction"}),
                                ("set_grain", {"answer": "people_repeat",
                                               "group_col": "subject_id"})],
                        ("set_repeat_kind", {"kind": "repeats"}),
                        ("set_repeat_kind", {"kind": "time_points"})),
    "set_unit_of_analysis": (
        _LONG, [("set_target", {"column": "sbp"}),
                ("set_purpose", {"answer": "prediction"}),
                ("set_grain", {"answer": "people_repeat",
                               "group_col": "subject_id"}),
                ("set_repeat_kind", {"kind": "repeats"})],
        ("set_unit_of_analysis", {"unit": "record"}),
        ("set_unit_of_analysis", {"unit": "person"})),
    # **ADDED AT L41-D, AND IT HAD NEVER BEEN PROBED** (`GUIDED-141`).
    # `recorded_kinds()` read two of the dispatcher's three forms, and this
    # kind is reachable only through the third — a `decision.kind in (…)` group
    # whose body falls through to an `else:`. So it was outside the denominator
    # every count in this file is computed against, and no assertion here has
    # ever said anything about it.
    #
    # Its preconditions are the narrowest in the app and they are why it sat at
    # the end of the chain: time points rather than repeats, AND the records
    # surviving as rows. `clinical_longitudinal.csv` is a visit schedule, which
    # is exactly that.
    "set_temporal_prediction": (
        _LONG, [("set_target", {"column": "sbp"}),
                ("set_purpose", {"answer": "prediction"}),
                ("set_grain", {"answer": "people_repeat",
                               "group_col": "subject_id"}),
                ("set_repeat_kind", {"kind": "time_points"}),
                ("set_unit_of_analysis", {"unit": "record"})],
        ("set_temporal_prediction", {"temporal": True}),
        ("set_temporal_prediction", {"temporal": False})),
    "set_eligibility": (_CLINIC, _TO_TARGET + [
        ("set_purpose", {"answer": "prediction"}),
        ("set_grain", {"answer": "one_row_per_person"})],
        ("set_eligibility", {"answer": "everyone"}),
        ("set_eligibility", {"answer": "restricted", "column": "age",
                             "minimum": 40,
                             "reason": "adults only, per the protocol"})),
    "seal": (_CLINIC, _TO_TARGET + [
        ("set_purpose", {"answer": "prediction"}),
        ("set_grain", {"answer": "one_row_per_person"}),
        ("set_eligibility", {"answer": "everyone"})],
        ("seal", {"fraction": 0.15}), ("seal", {"fraction": 0.4})),
    "select_models": (_CLINIC, _TO_SEAL,
                      ("select_models", {"models": ["ridge"]}),
                      ("select_models", {"models": ["histgb_reg"]})),
    "set_preparation_mode": (_CLINIC, _TO_MODELS,
                             ("set_preparation_mode", {"mode": "per_model"}),
                             ("set_preparation_mode", {"mode": "uniform"})),
    "set_model_recipe": (_CLINIC, _TO_MODELS,
                         ("set_model_recipe", {"model": "ridge",
                                               "operation": "scale",
                                               "variant": "standard"}),
                         ("set_model_recipe", {"model": "ridge",
                                               "operation": "scale",
                                               "variant": "robust"})),
    "route_missingness": (_CLINIC, _TO_SEAL,
                          ("route_missingness", {"column": "notes",
                                                 "mechanism": "not_informative",
                                                 "strategy": "impute_mode"}),
                          ("route_missingness", {"column": "notes",
                                                 "mechanism": "informative",
                                                 "strategy": "explicit_category"})),
    "route_missingness_bulk": (
        _METAB, [("set_target", {"column": "responder"}),
                 ("set_purpose", {"answer": "prediction"}),
                 ("set_grain", {"answer": "one_row_per_person"}),
                 ("set_eligibility", {"answer": "everyone"}),
                 ("seal", {"fraction": 0.25})],
        ("route_missingness_bulk", {"branch": "numeric",
                                    "mechanism": "not_informative",
                                    "strategy": "impute_median",
                                    "columns": ["mz_0003", "mz_0005"]}),
        ("route_missingness_bulk", {"branch": "numeric",
                                    "mechanism": "not_informative",
                                    "strategy": "impute_mean",
                                    "columns": ["mz_0003", "mz_0005"]})),
    "settle_preprocess": (_CLINIC, _TO_SEAL,
                          ("settle_preprocess", {}),
                          ("settle_preprocess", {"skipped": True})),
    "add_feature": (_CLINIC, _TO_TARGET,
                    ("add_feature", {"transform": "log", "columns": ["glucose"]}),
                    ("add_feature", {"transform": "sqrt", "columns": ["glucose"]})),
    "remove_feature": (_CLINIC, _TO_TARGET + [
        ("add_feature", {"transform": "log", "columns": ["glucose"]}),
        ("add_feature", {"transform": "sqrt", "columns": ["glucose"]})],
        ("remove_feature", {"column": "log_glucose"}),
        ("remove_feature", {"column": "sqrt_glucose"})),
    "defer_feature": (_CLINIC, _TO_SEAL,
                      ("defer_feature", {"transform": "bin_quantile",
                                         "columns": ["age"],
                                         "params": {"n_bins": 3}}),
                      ("defer_feature", {"transform": "bin_uniform",
                                         "columns": ["age"],
                                         "params": {"n_bins": 5}})),
    "set_selection": (_CLINIC, _TO_SEAL,
                      ("set_selection", {"method": "mutual_info",
                                         "n_features": 3,
                                         "candidates": ["age", "glucose",
                                                        "bp_1", "bp_2"]}),
                      ("set_selection", {"method": "lasso", "n_features": 4,
                                         "candidates": ["age", "glucose",
                                                        "bp_1", "bp_2"]})),
    "settle_features": (_CLINIC, _TO_TARGET,
                        ("settle_features", {}),
                        ("settle_features", {"skipped": True})),
    "trim_training_rows": (_CLINIC, _TO_SEAL,
                           ("trim_training_rows", {"column": "age",
                                                   "minimum": 30,
                                                   "reason": "stability"}),
                           ("trim_training_rows", {"column": "age",
                                                   "minimum": 45,
                                                   "reason": "stability"})),
    "set_reverse_coding": (_SURVEY, [],
                           ("set_reverse_coding", {"columns": []}),
                           ("set_reverse_coding", {"columns": ["item_03"]})),
    "apply": (_CLINIC, _TO_TARGET, ("apply", {}), ("__skip__", {})),
}


# ─────────────────────────────────────────────────────────────────────────────
# The fingerprint
# ─────────────────────────────────────────────────────────────────────────────

#: Fields that ARE the record of a decision rather than something computed from
#: it. Reading these back would make every probe pass by tautology, which is
#: exactly how `GUIDED-095` stayed invisible: every surface that mattered read
#: the record and reported it faithfully.
_RECORD_FIELDS = ("engineered", "deferred_transforms", "selection", "declared",
                  "selected", "decisions", "settled", "receipt", "mode",
                  "obligations", "eligibility", "grain", "lens", "purpose",
                  "repeat_kind", "unit_of_analysis", "missingness",
                  # NOT the record — IDENTITY. Two projects built from the same
                  # answers differ in their project id and in when they were
                  # made, and a fingerprint that carried those would report
                  # every flip as propagating. The positive control found it.
                  #
                  # A FINDING'S OWN `id` IS CONTENT AND STAYS. Stripping it
                  # would make two different findings compare equal, which is
                  # the probe going blind in the direction it must not.
                  "project_id", "at", "created_at")


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items()
                if k not in _RECORD_FIELDS}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _get(client, path):
    try:
        r = client.get(path)
    except Exception as exc:                                # pragma: no cover
        return f"raised:{type(exc).__name__}"
    if r.status_code != 200:
        return f"status:{r.status_code}"
    return _strip(r.json())


def _fingerprint(client, pid):
    """Everything computed FROM the record, surface by surface.

    Keyed so a difference names WHICH surface moved — a boolean would say a
    decision propagated and not where to, and where-to is the useful half.
    """
    project = api.STORE.get(pid)
    out = {
        "working_columns": [str(c) for c in project.working_table.columns],
        "working_rows": int(len(project.working_table)),
        "findings": _get(client, f"/project/{pid}/findings"),
        "figures": _get(client, f"/project/{pid}/figures"),
        "features": _get(client, f"/project/{pid}/features"),
        "preprocess": _get(client, f"/project/{pid}/preprocess"),
        "recipes": _get(client, f"/project/{pid}/recipes"),
        "models": _get(client, f"/project/{pid}/models"),
        "draft": _get(client, f"/project/{pid}/draft"),
        # `L38-C`. The MANUSCRIPT is a computed surface and this probe did not
        # fingerprint it, because Report shipped at L36 and the probe was
        # written at L35. The gap surfaced the first time a decision changed
        # only the manuscript: `promote_figure` moved the validation report and
        # nothing else, and the probe reported that it changed NOTHING. A
        # census that cannot see a surface reports every decision about it as
        # inert.
        "manuscript": _get(client, f"/project/{pid}/manuscript"),
        "selection_evidence": _get(client, f"/project/{pid}/selection/evidence"),
    }
    for step in ("data", "explore", "features", "preprocess", "train"):
        out[f"interview:{step}"] = _get(
            client, f"/project/{pid}/interview?step={step}")
    out["pipeline_plan"] = _plan_fingerprint(project)
    return out


def _plan_fingerprint(project):
    """The fold-fitted pipeline, which is the surface `GUIDED-095` was about.

    Composed rather than fitted — the plan IS the specification, and fitting it
    twice per kind would make this probe cost minutes to say the same thing.
    """
    from turbotab import pipeline_plan, training

    if not (project.target and project.lockbox
            and project.lockbox.get("labels")):
        return "not sealed"
    try:
        # THE PROJECT-AWARE DOOR, and this line is itself the argument for
        # having one. It called `_feature_frame` — the loose three-argument
        # form — so the probe's picture of the fitted pipeline did not include
        # identifier exclusion, and `keep_identifier` read as a decision that
        # changed a rendered surface and left the fit alone. A census that
        # composes the pipeline differently from the app is measuring a
        # pipeline the app does not fit.
        features = training.feature_frame(project)
        plans = {}
        for key in ("ridge", "histgb_reg", "logreg", "histgb_clf"):
            try:
                plans[key] = pipeline_plan.compose(
                    project, key, features).to_dict()
            except Exception:
                continue
        return plans
    except Exception as exc:                                # pragma: no cover
        return f"raised:{type(exc).__name__}"


def flip_kind(name):
    """The decision kind a flip is about.

    A kind can have more than one flip — `set_lens` is probed twice, once
    unsealed for the display surfaces and once driven to a seal so the
    fold-fitted pipeline exists to differ in. The suffix after `@` names the
    variant and is not part of the kind.
    """
    return name.split("@", 1)[0]


def _differences(a, b):
    return sorted(k for k in a if json.dumps(a[k], sort_keys=True, default=str)
                  != json.dumps(b.get(k), sort_keys=True, default=str))


# ─────────────────────────────────────────────────────────────────────────────
# Running one flip
# ─────────────────────────────────────────────────────────────────────────────

def _client():
    return TestClient(api.app)


def _build(client, fixture, setup, answer):
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]
    for kind, payload in setup:
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (fixture, kind, r.text[:200])
    kind, payload = answer
    if kind == "__skip__":
        return pid, None
    if kind == "__lens_sealed__":
        # The lens, then the whole pre-seal sequence, so the flip is compared
        # against a project that has a fitted pipeline to differ in.
        for step, body in [("set_lens", payload),
                           ("set_target", {"column": "responder"}),
                           ("set_purpose", {"answer": "prediction"}),
                           ("set_grain", {"answer": "one_row_per_person"}),
                           ("set_eligibility", {"answer": "everyone"}),
                           ("seal", {"fraction": 0.25})]:
            r = client.post(f"/project/{pid}/decision",
                            json={"kind": step, "payload": body})
            assert r.status_code == 200, (step, r.text[:200])
        return pid, r
    if kind == "__identifier_sealed__":
        # The whole pre-seal sequence, then the flip, for the same reason
        # `__lens_sealed__` needs one: an unsealed project has no fitted
        # pipeline to differ in, and this decision's whole content is WHICH
        # COLUMNS the model is fed.
        for step, body in [("set_target", {"column": "age"}),
                           ("set_purpose", {"answer": "prediction"}),
                           ("set_grain", {"answer": "one_row_per_person"}),
                           ("set_eligibility", {"answer": "everyone"}),
                           ("seal", {"fraction": 0.25}),
                           ("keep_identifier", payload)]:
            r = client.post(f"/project/{pid}/decision",
                            json={"kind": step, "payload": body})
            assert r.status_code == 200, (step, r.text[:200])
        return pid, r
    if kind == "apply":
        # The flip is *repair the first thing the engine found* against
        # *repair nothing*, so the answer's subject comes from this project's
        # own findings rather than from a hardcoded id.
        findings = client.get(f"/project/{pid}/findings").json()["findings"]
        # STRUCTURAL only. `apply` resolves the id against `engine.diagnose`,
        # which produces the structural stream; a profile warning carries a
        # `fix_kind` of `none` and is not applicable at all.
        repairable = [f for f in findings
                      if f.get("source") == "structure"
                      and f.get("fix_kind") not in (None, "", "none")]
        if not repairable:
            pytest.skip("this fixture produced no applicable repair")
        payload = {}
        subject = repairable[0]["id"]
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": "apply", "subject": subject,
                              "payload": payload})
        return pid, r
    r = client.post(f"/project/{pid}/decision",
                    json={"kind": kind, "payload": payload})
    return pid, r


#: Kinds whose flip must move the FOLD-FITTED PIPELINE, not merely a rendered
#: surface. This is the sharper half of the probe and it is where `GUIDED-095`
#: lived: *the composed prose changed* is propagation by the letter and says
#: nothing about whether the analysis the user specified is the analysis that
#: was fitted.
REACHES_THE_FIT = {
    "route_missingness", "route_missingness_bulk", "defer_feature",
    "set_model_recipe", "set_preparation_mode", "set_lens@sealed",
    # `GUIDED-108`. It changes WHICH COLUMNS the model is fed, which is the
    # most direct way a decision can reach a fit.
    "keep_identifier",
}

#: And the ones that legitimately do not, each with the reason. Measured, not
#: assumed — every entry here was observed not to move `pipeline_plan`.
NOT_IN_THE_FIT = {
    "set_temporal_prediction": (
        "**`GUIDED-143`, AND THIS ENTRY IS THE FINDING RATHER THAN AN EXCUSE.** "
        "Answering *yes, I am predicting a later outcome from earlier "
        "measurements* records `strategy: chronological_grouped` and the "
        "sentence *'The held-out rows are the latest ones … at times after the "
        "ones it trained on'*, which the draft carries into the methods "
        "section. `engine.draw_holdout` takes the frame, the target, the task "
        "type and the GRAIN, and never reads `temporal_prediction` at all — so "
        "the split is drawn at random within groups. Driven on "
        "`clinical_longitudinal.csv`: the held-out visit dates run "
        "2023-01-16 to 2024-01-20 and the training ones 2023-01-10 to "
        "2024-01-22, which overlap almost exactly. The record describes a "
        "split that was not drawn, which is `AUDIT-001`'s shape in the "
        "artifact that leaves the building. It is listed here because that is "
        "the truth about where it reaches, and the row is what makes the "
        "silence loud."),
    "promote_figure": (
        "Editorial, not analytic. Placing a figure in the results changes the "
        "manuscript and the validation report and must not touch the fit — a "
        "figure the author promoted is not a feature the model gets."),
    "set_selection": (
        "**A REAL SEVERED SEAM, AND IT IS WHY `GUIDED-095` IS `PARTIAL` RATHER "
        "THAN `FIXED`** — adjudicator, L35; this line previously pointed at "
        "`GUIDED-099`, which is the pack recipe table and a different finding. "
        "`selection.declare`'s "
        "own docstring calls this the sharpest case in the whole project and "
        "says the per-model pipeline fits the spec inside each training fold. "
        "`pipeline_plan` implements no selection at all, so the spec reaches "
        "the transcript and the draft and nothing that fits. Left here rather "
        "than fixed because fold-local selection is a build, not a probe."),
    "set_purpose": (
        "Purpose is a GATE rather than a transform. It decides whether the "
        "outcome may sit inside a MICE scope and whether a missing-indicator "
        "is contraindicated — refusals that fire when a user takes those "
        "routes — so a bare flip with no such route taken correctly moves "
        "nothing in the pipeline."),
    "set_target": (
        "Changing the outcome changes which column is excluded from the "
        "features, and the plan is composed over the feature frame the caller "
        "hands it. The change is real and lands upstream of this surface."),
    "set_task_type": (
        "Decides which estimators the shelf can offer and which metrics are "
        "computed. The preprocessing plan is the same either way, which is "
        "correct: an imputer does not care what is being predicted."),
    "set_grain": (
        "Decides how the seal is DRAWN and which column is dropped as the "
        "grouping key. Both are upstream of the plan — different rows and a "
        "different feature frame, same recipe."),
    "set_repeat_kind": (
        "Upstream of the seal, exactly as `set_grain` is: it decides how the "
        "held-out rows are drawn, not how any column is prepared."),
    "set_unit_of_analysis": (
        "Upstream of the seal for the same reason, and where it aggregates it "
        "changes what a ROW is rather than what a step does."),
    "set_eligibility": (
        "Changes WHO the study is about. That is rows and a participant-flow "
        "number, and the recipe over the columns is unchanged by it."),
    "seal": (
        "Changes WHICH rows are held out. The plan is structure — which "
        "column gets which transform — and structure does not move with the "
        "split; the fitted STATISTICS do, which is what the seal probes "
        "assert separately."),
    "trim_training_rows": (
        "Narrows the training partition, so it moves rows and the fitted "
        "statistics with them, and leaves every step of the plan in place."),
    "select_models": (
        "The plan is composed PER MODEL, so selecting a different set changes "
        "which plans exist rather than what any one of them says. This "
        "probe's fingerprint composes for a fixed set of model keys precisely "
        "so the other kinds are compared like for like."),
    "settle_preprocess": (
        "Ends the step. `skipped` is recorded so a silent skip and a step "
        "nobody reached are different states; neither adds a pipeline step."),
    "settle_features": (
        "Ends the Features step, and records a skip for the same "
        "recorded-absence reason. Neither adds a step to any pipeline."),
    "add_feature": (
        "Row-local, so it has already executed: the new column is in the "
        "working table and reaches the plan through the feature frame. The "
        "plan's per-column steps do move — this entry records that the "
        "RECIPE half does not."),
    "remove_feature": (
        "The inverse of `add_feature`, and upstream of the plan for the same "
        "reason: the column is gone from the working table before the plan is "
        "composed over it."),
    "apply": (
        "A structural repair rewrites the working table, which is upstream of "
        "the plan — the repaired values are what the plan is composed over."),
    "set_lens": (
        "The UNSEALED flip, which cannot reach a pipeline that does not exist "
        "yet — there is no fold to fit anything in before the seal. The lens "
        "does reach the fit, and `set_lens@sealed` is the flip that asserts "
        "it: the pack's own `scale: pareto` and `power: log1p` are in the "
        "plan under the metabolomics lens and absent under `other`."),
    "set_reverse_coding": (
        "Records which survey items are reverse-coded so the scale can be "
        "scored. Nothing scores a scale yet — the consumer is the survey "
        "pack's scoring step, which is not built."),
}


@pytest.mark.parametrize("kind", sorted(FLIPS), ids=sorted(FLIPS))
def test_flipping_a_recorded_answer_changes_something_downstream(kind):
    """**One decision, two answers, and something must move.**

    A kind whose flip changes nothing on any computed surface is severed
    connective tissue — the record is written, the receipt counts it, the
    sentence is composed, and nothing reads it. That is `GUIDED-095` stated as
    a property rather than as a census.
    """
    client = _client()
    fixture, setup, answer_a, answer_b = FLIPS[kind]

    pid_a, r_a = _build(client, fixture, setup, answer_a)
    pid_b, r_b = _build(client, fixture, setup, answer_b)
    for response, which in ((r_a, "A"), (r_b, "B")):
        if response is not None:
            assert response.status_code == 200, (
                f"{kind} arm {which} was refused: {response.text[:250]}")

    moved = _differences(_fingerprint(client, pid_a),
                         _fingerprint(client, pid_b))
    assert moved, (
        f"flipping {kind!r} changed NOTHING downstream. Every computed "
        f"surface — the working table, the findings, the figures, the offers, "
        f"the composed prose, the interview plan and the fold-fitted pipeline "
        f"— is byte-identical between the two answers. Either wire it, or add "
        f"it to ALLOWED_NO_EFFECT with the reason it legitimately reaches "
        f"nothing.")


@pytest.mark.parametrize("kind", sorted(REACHES_THE_FIT), ids=sorted(REACHES_THE_FIT))
def test_a_decision_about_the_analysis_reaches_the_fitted_pipeline(kind):
    """**The sharper half**, and the one `GUIDED-095` was about.

    *The composed prose changed* is propagation by the letter and says nothing
    about whether the analysis the user specified is the analysis that was
    fitted. A decision about HOW the data is prepared has to move the
    fold-fitted pipeline itself.
    """
    client = _client()
    fixture, setup, answer_a, answer_b = FLIPS[kind]
    pid_a, _ = _build(client, fixture, setup, answer_a)
    pid_b, _ = _build(client, fixture, setup, answer_b)
    a = _plan_fingerprint(api.STORE.get(pid_a))
    b = _plan_fingerprint(api.STORE.get(pid_b))
    assert a != "not sealed" and b != "not sealed", (
        f"{kind}'s flip is not sealed, so there is no fitted pipeline to "
        f"compare and this claim is vacuous")
    assert json.dumps(a, sort_keys=True, default=str) != \
        json.dumps(b, sort_keys=True, default=str), (
        f"flipping {kind!r} changed a rendered surface and left the "
        f"fold-fitted pipeline identical, which is GUIDED-095 exactly: the "
        f"record is written, the receipt counts it, the sentence is composed, "
        f"and the number the user takes away is about a different analysis.")


def test_every_probed_kind_says_whether_it_reaches_the_fit():
    """The two lists have to cover the probed kinds and stay true of them.

    A kind that quietly stopped reaching the fit would otherwise pass the
    broad probe on its rendered surfaces alone — which is the exact failure
    mode this file exists to make impossible.
    """
    accounted = REACHES_THE_FIT | set(NOT_IN_THE_FIT)
    missing = sorted(name for name in FLIPS
                     if name not in accounted
                     and flip_kind(name) not in accounted)
    assert not missing, (
        f"these kinds are probed and nothing says whether they reach the "
        f"fitted pipeline: {missing}")
    overlap = sorted(REACHES_THE_FIT & set(NOT_IN_THE_FIT))
    assert not overlap, overlap
    for kind, reason in NOT_IN_THE_FIT.items():
        assert len(reason) > 55, f"{kind}: the reason is a shrug: {reason!r}"


def test_the_probe_can_fail():
    """The positive control, and this file is worthless without it.

    Two projects built from the SAME answer must show no difference. If the
    fingerprint moves anyway — a timestamp, an id, a dict ordering — then every
    assertion above passes on noise.
    """
    client = _client()
    fixture, setup, answer_a, _ = FLIPS["set_target"]
    pid_a, _ = _build(client, fixture, setup, answer_a)
    pid_b, _ = _build(client, fixture, setup, answer_a)
    moved = _differences(_fingerprint(client, pid_a),
                         _fingerprint(client, pid_b))
    assert not moved, (
        f"two projects given the SAME answer differ on {moved}, so the "
        f"comparison above reports propagation that is really noise")


# ─────────────────────────────────────────────────────────────────────────────
# The coverage report — the honest output of the sweep
# ─────────────────────────────────────────────────────────────────────────────

def test_every_recorded_kind_is_probed_allow_listed_or_named():
    """No kind falls off the edge silently.

    The kinds come from the API rather than from a list here, so adding a
    decision kind and not probing it fails HERE rather than being discovered by
    the next census.
    """
    kinds = set(recorded_kinds())
    accounted = ({flip_kind(name) for name in FLIPS}
                 | set(ALLOWED_NO_EFFECT) | set(NOT_CONSTRUCTED))
    missing = sorted(kinds - accounted)
    assert not missing, (
        f"these decision kinds are recorded and this probe says nothing about "
        f"them: {missing}. Probe them, allow-list them with a reason, or name "
        f"them in NOT_CONSTRUCTED with why the flip cannot be built.")
    stale = sorted(accounted - kinds)
    assert not stale, (
        f"these are probed or excused and the API no longer accepts them: "
        f"{stale}")

    for table, label in ((ALLOWED_NO_EFFECT, "allow-listed"),
                         (NOT_CONSTRUCTED, "not constructed")):
        for name, reason in table.items():
            assert len(reason) > 60, (
                f"{name} is {label} with a reason that is a shrug: {reason!r}")


def test_the_probe_reports_its_own_coverage(capsys):
    """**The counts this probe owes**, printed with `-s` and asserted here.

    A sweep that reports only what it fixed has not reported its coverage, so
    the four numbers are the deliverable: probed, propagated, allow-listed,
    and could-not-construct.
    """
    kinds = recorded_kinds()
    probed = {flip_kind(name) for name in FLIPS}
    with capsys.disabled():
        print(f"\n  decision kinds recorded      {len(kinds)}")
        print(f"  probed                       {len(probed)}"
              f"  ({len(FLIPS)} flips)")
        print(f"  of those, reach the fit      {len(REACHES_THE_FIT)}")
        print(f"  allow-listed (no effect)     {len(ALLOWED_NO_EFFECT)}")
        print(f"  could not construct a flip   {len(NOT_CONSTRUCTED)}")
    assert len(probed) + len(ALLOWED_NO_EFFECT) + len(NOT_CONSTRUCTED) == len(kinds)
    assert len(probed) >= len(kinds) // 2, (
        "fewer than half the recorded kinds are probed, which makes the "
        "allow-list and the could-not-construct list the report rather than "
        "the sweep")
