"""The class `GUIDED-088` named, guarded by enumeration rather than by example.

## Why this file exists at all

`test_the_seal_is_consulted_wherever_a_choice_is_ranked` said in its own
docstring that it was the class rather than the instance. Its assertions were
about `/selection/evidence` — the path that was **already correct** — and its
only reference to the model shelf was `assert served` as a non-empty control.
So it passed against the reverted shelf (`GUIDED-092`, verified by running the
revert probe and reading what stayed green), and it enumerated nothing, which
means a third ranking added next loop was not covered either.

A class guarded by one example is guarded by nothing. This file iterates
`turbotab.rankings.SURFACES` instead, and the registry is the artifact: adding a
ranking means adding a row, and a row that claims `TRAINING_ROWS` while reading
a sealed value fails here.

## The probe, and why it is a moving one

Row counts can agree by luck and a mask can be applied to the wrong frame. So
the assertion is the same shape as `test_no_held_out_row_moves_any_fitted_
parameter`: **replace every held-out value with something no training row could
produce, and the surface must be byte-identical.** A ranking that saw one
sealed row moves visibly.

`test_the_probe_can_fail_on_every_surface` is the positive control and this file
is worthless without it — a comparison that cannot see a leak it was pointed
straight at proves nothing about the ones it was not.

## What an exemption is

`WHOLE_TABLE` surfaces are asserted to MOVE, not merely allowed to. An
exemption that has quietly become masked is a stale excuse, and stale excuses
are how a list like this stops meaning anything — the same rule
`NOT_READ_BY_THE_DOOR` runs on one file over.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, rankings                                    # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent / "sample_data"
ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _sealed(client):
    """A project driven to a drawn seal, with models selected.

    `clinic_visits.csv` rather than a synthetic frame: the recipe lattice needs
    real numeric spread to have anything to measure, and the missingness survey
    needs real blanks.
    """
    with open(DATA / "clinic_visits.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("s.csv", fh, "text/csv")}).json()["id"]

    def decide(kind, **payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (kind, r.text[:250])

    decide("set_target", column="hba1c")
    decide("set_purpose", answer="prediction")
    decide("set_grain", answer="one_row_per_person")
    decide("set_eligibility", answer="everyone")
    decide("seal", fraction=0.3)
    project = api.STORE.get(pid)
    shelf = [e.to_dict()["key"] for e in project.model_shelf()][:2]
    decide("select_models", models=shelf)
    return pid, project


# ── how each enumerated surface is read ──────────────────────────────────────
#
# Keyed by `rankings.SURFACES` key. The completeness test below asserts the two
# key sets are identical, so a registry row with no probe and a probe with no
# registry row both fail.

def _read_model_shelf(client, pid, project):
    return client.get(f"/project/{pid}/models").json()


def _read_selection_evidence(client, pid, project):
    return client.get(f"/project/{pid}/selection/evidence").json()


def _read_recipe_lattice(client, pid, project):
    return client.get(f"/project/{pid}/recipes").json()


def _read_ranked_findings(client, pid, project):
    # Recomputed rather than read off the project, because `/findings` serves
    # what the LAST decision computed and the probe changes the frame without
    # posting one. `api._recompute` is the app's own path — every decision that
    # touches the table calls it — so this reads what a user would next see.
    api._recompute(project)
    return client.get(f"/project/{pid}/findings").json()


def _read_missingness_survey(client, pid, project):
    return client.get(f"/project/{pid}/preprocess").json()


READS = {
    "model_shelf": _read_model_shelf,
    "selection_evidence": _read_selection_evidence,
    "recipe_lattice": _read_recipe_lattice,
    "ranked_findings": _read_ranked_findings,
    "missingness_survey": _read_missingness_survey,
}


def _poison(project):
    """Replace every held-out value with one no training row could produce.

    Numeric columns get a magnitude nothing in this file reaches; text columns
    get a token that is not a level anywhere. Both, because a surface can read a
    column either way and a probe that only moves numbers cannot see an encoder
    or a mode.
    """
    sealed = set(project.lockbox["labels"])
    assert sealed, "nothing was held out, so the probe has nothing to move"
    poisoned = project.df.copy()
    mask = poisoned.index.isin(list(sealed))
    assert mask.sum() > 0
    for column in poisoned.columns:
        if pd.api.types.is_numeric_dtype(poisoned[column]):
            poisoned.loc[mask, column] = 1_000_000.0
        else:
            poisoned.loc[mask, column] = "@@sealed@@"
    project.df = poisoned


def _blank_an_unsealed_outcome(project) -> int:
    """Remove the outcome from unsealed rows, and return how many.

    **THE SECOND PROBE, AND `_poison` STRUCTURALLY CANNOT SEE WHAT IT SEES.**
    `_poison` overwrites values only on rows in `lockbox["labels"]`, and the
    two populations `DRIVE-050` is about — `training_rows` and `analysis_rows`
    — differ exactly on the outcome-BLANK rows, which are never sealed
    (`engine.draw_holdout` opens with `eligible = df.index[y.notna()]`). So a
    surface could read the wrong one of the two and the poison probe would stay
    green over it, forever. That is `MISC-022`'s shape: a registry that looks
    guarded and is not.

    A scope rename cannot express this either — `analysis_rows` is a strict
    SUBSET of `training_rows`, so `scope=TRAINING_ROWS` is still true under the
    registry's own two-valued definition, and `Surface.__post_init__` hard-
    refuses a third value with a `ValueError`. What was needed was a second
    probe, and this is it.
    """
    target = str(project.target)
    sealed = set(project.lockbox["labels"])
    frame = project.df.copy()
    unsealed = [i for i in frame.index
                if i not in sealed and pd.notna(frame.at[i, target])]
    assert len(unsealed) > 4, (
        "there are no unsealed rows with an outcome to blank, so this probe "
        "has nothing to move")
    victims = unsealed[:max(4, len(unsealed) // 3)]
    frame.loc[victims, target] = None
    project.df = frame
    return victims


#: THE RANKING ITSELF, per surface — the thing `Surface.decides` names, and
#: NOT the whole payload. Blanking a row and deleting it legitimately produce
#: different table-level counts (`n_rows_without_an_outcome` moves by
#: construction, the total row count moves), so comparing whole payloads
#: asserts something the equivalence below does not claim and is not true.
#: The order AND the clauses that justify it. The order alone is too coarse —
#: measured: reverting `analysis_mask` to `training_mask` leaves the shelf's
#: ORDER identical while every capacity clause on it cites a different `n`, so
#: a projection that took only the keys was GREEN over the exact defect. The
#: concern is what a user reads and it is where the population surfaces.
_ORDER_OF = {
    "model_shelf": lambda b: [(m.get("key"), m.get("concern"),
                               m.get("design_notes"))
                              for g in b.get("groups") or []
                              for m in g.get("models") or []],
    "selection_evidence": lambda b: b.get("ranked"),
    "recipe_lattice": lambda b: [(o.get("key"), o.get("determinacy"),
                                  o.get("because"),
                                  tuple(o.get("variants") or ()))
                                 for o in b.get("operations") or []],
}


@pytest.mark.parametrize("surface", rankings.training_scoped(),
                         ids=[s.key for s in rankings.training_scoped()])
def test_blanking_an_unsealed_outcome_is_the_same_as_removing_the_row(
        client, surface):
    """The probe `_poison` cannot be, and it is an EQUIVALENCE rather than a
    movement. `DRIVE-050`.

    **A movement assertion does not discriminate here, and that was measured
    rather than assumed.** Blanking an outcome moves a training-scoped surface
    under *both* implementations: under `analysis_rows` the row leaves the
    population, and under `training_rows` it stays with a NaN outcome, which
    changes the profile anyway. Reverting `analysis_mask` to `training_mask`
    left a movement-based probe GREEN — it would have been a probe that looks
    like a check and is not, which is `MISC-022`'s own shape arriving inside
    the fix for it.

    The equivalence does discriminate. A row whose outcome is blank can inform
    no fit, so for a surface computed on the analysis population *blanking* it
    and *deleting* it are the same operation and must produce byte-identical
    output. For a surface computed on the training population they are not: the
    blanked row is still counted, still profiled, still has its features read.
    """
    pid, project = _sealed(client)
    target = str(project.target)

    blanked_pid, blanked = _sealed(client)
    victims = _blank_an_unsealed_outcome(blanked)

    deleted_pid, deleted = _sealed(client)
    frame = deleted.df.copy()
    assert set(victims) <= set(frame.index)
    deleted.df = frame.drop(index=list(victims))

    # THE CONTROL: the two frames really do differ in the way the equivalence
    # is about — same rows-with-an-outcome, different total rows.
    assert len(blanked.df) - len(deleted.df) == len(victims)
    assert (int(blanked.df[target].notna().sum())
            == int(deleted.df[target].notna().sum()))

    order = _ORDER_OF[surface.key]
    after_blank = _stable(order(READS[surface.key](client, blanked_pid,
                                                   blanked)))
    after_delete = _stable(order(READS[surface.key](client, deleted_pid,
                                                    deleted)))
    assert after_blank == after_delete, (
        f"{surface.key} answers differently when {len(victims)} unsealed rows "
        f"have their outcome BLANKED than when the same rows are DELETED. No "
        f"model can be fitted on either version of those rows, so a surface "
        f"that tells them apart was computed on a population that includes "
        f"rows no model will ever see. It decides {surface.decides}; it is "
        f"served by {surface.served_by}.")


def _stable(value):
    return json.dumps(value, sort_keys=True, default=str)


@pytest.mark.parametrize("surface", rankings.SURFACES,
                         ids=[s.key for s in rankings.SURFACES])
def test_a_ranking_does_not_move_when_only_the_held_out_rows_change(
        client, surface):
    """The requirement, per enumerated surface.

    A `TRAINING_ROWS` surface must be bitwise unchanged after every held-out
    value is replaced. A `WHOLE_TABLE` surface must MOVE — an exemption that no
    longer describes the code is worse than no exemption, because the next
    reader takes the registry at its word.
    """
    pid, project = _sealed(client)
    before = _stable(READS[surface.key](client, pid, project))
    _poison(project)
    after = _stable(READS[surface.key](client, pid, project))

    if surface.scope == rankings.TRAINING_ROWS:
        assert before == after, (
            f"{surface.key} moved when only the HELD-OUT rows changed, so the "
            f"order a user picks from was computed with the sealed rows in "
            f"view. It decides {surface.decides}; it is served by "
            f"{surface.served_by}.")
    else:
        assert before != after, (
            f"{surface.key} is registered as a {rankings.WHOLE_TABLE} "
            f"exemption tracked by {surface.tracked_by}, and it no longer "
            f"reads the held-out rows. Move it to {rankings.TRAINING_ROWS} — a "
            f"stale exemption is how this list stops meaning anything.")


def test_the_probe_can_fail_on_every_surface(client):
    """**The positive control, and this file is worthless without it.**

    Each `TRAINING_ROWS` surface is recomputed WITHOUT the mask, against the
    poisoned frame, and must differ from what the app served. A surface where
    it does not differ is one this fixture cannot tell apart either way, so the
    assertion above would pass whatever the code did.
    """
    from turbotab import engine, models as _models, recipes as _rec
    from turbotab import selection as _sel

    pid, project = _sealed(client)
    _poison(project)

    numeric = [str(c) for c in project.df.columns
               if pd.api.types.is_numeric_dtype(project.df[c])
               and str(c) != project.target]

    served_shelf = client.get(f"/project/{pid}/models").json()
    unmasked_shelf = _models.shelf(
        engine.profile(project.df, project.target, project.task_type),
        project.task_type or "regression")
    assert ([e.to_dict()["concern"] for e in unmasked_shelf]
            != [m["concern"] for g in served_shelf["groups"]
                for m in g["models"]]), (
        "the shelf ranked on the whole poisoned table cites the same numbers "
        "as the one the app served, so this fixture cannot tell a masked "
        "shelf from an unmasked one")

    served_evidence = client.get(f"/project/{pid}/selection/evidence").json()
    unmasked_evidence = _sel.evidence(
        project.df, project.target,
        [c for c in numeric], None)
    assert unmasked_evidence["ranked"] != served_evidence["ranked"], (
        "the feature ranking is the same with and without the mask on this "
        "fixture")

    served_recipes = client.get(f"/project/{pid}/recipes").json()
    row = next(r for rows in served_recipes["models"].values() for r in rows
               if r["operation"] == "scale" and r.get("divergence"))
    resolved = _rec.resolve(row["model"], "scale")
    _, unmasked_div = _rec.worth_asking(project.df, numeric, resolved)
    assert unmasked_div is not None
    assert (round(unmasked_div.statistic, 4)
            != round(row["divergence"]["statistic"], 4)), (
        "the divergence statistic is identical with and without the mask, so "
        "the lattice assertion above would pass either way")


def test_the_served_row_count_excludes_the_lockbox(client):
    """`GUIDED-092`'s own words: *assert each was computed on a row count that
    excludes the lockbox.*

    **This is not the cheap version of the probe above. It is the other half,
    and the shelf is why.** A value poison can only detect a surface that reads
    held-out VALUES. `models.shelf` reads a profile — `n_rows`, the feature
    count, `p/n` — so reverting its mask changes what it ranks on and moves
    nothing the poison can touch: measured, the poison probe stays GREEN
    against the reverted shelf. Counts and values are two different leaks and
    each needs its own assertion.

    Which is also why each count here comes OUT OF the computation — the
    profile the shelf ranked on, the frame `worth_asking` measured, the mask
    `selection.evidence` applied — and never from a second derivation beside
    it. A separately-computed count keeps reporting the training total after
    somebody reverts the mask, which is the served number lying about a
    computation nobody performed.

    THE LOOP OVER THREE ROUTES IS SPLIT, AND `TEST-101` IS WHY.
    `n_rows_seen` genuinely means two different things depending on the route,
    and one assertion over all three could only ever be right about one of
    them. `/models` and `/recipes` narrowed to the ANALYSIS population at
    `DRIVE-045` — the training rows minus those with no outcome, which is what
    a model will actually be fitted on. `/selection/evidence` still counts off
    `project.training_mask` (`api.py:2498`) and serves no
    `n_rows_without_an_outcome` at all. Driven on one project: 225, 225, 825.

    **The old single assertion passed only because of the fixture.**
    `clinic_visits.csv` has zero NaN in `hba1c`, so `analysis_rows ==
    training_rows` and both meanings coincide. It is not swapped — it is shared
    with the poison probe, its positive control and five parametrized cases —
    so a second fixture WITH blank outcomes is added beside it, and it is what
    makes the split mean anything: on `clinic_visits.csv` the correction is
    98 == 98 before and after, which is no signal at all.

    `/selection/evidence`'s count is `GUIDED-244`, filed and deliberately
    deferred: fixing it in the same loop that splits this would make the split
    stale in the loop that wrote it.
    """
    pid, project = _sealed(client)
    n_sealed = len(set(project.lockbox["labels"]))
    assert n_sealed, "nothing was held out"                          # control
    n_all = len(project.df)

    for path in (f"/project/{pid}/models", f"/project/{pid}/recipes"):
        body = client.get(path).json()
        assert "n_rows_seen" in body, (
            f"{path} serves a ranking and does not say how many rows it saw")
        assert body["n_rows_seen"] == len(project.analysis_rows), (
            f"{path} ranked on {body['n_rows_seen']} rows; a model will be "
            f"fitted on {len(project.analysis_rows)}")
        assert body.get("n_rows_withheld") == n_sealed

    # THE THIRD ROUTE, AND ITS COUNT IS A DIFFERENT POPULATION ON PURPOSE —
    # for now. `GUIDED-244`.
    evidence = client.get(f"/project/{pid}/selection/evidence").json()
    assert "n_rows_seen" in evidence, (
        "/selection/evidence serves a ranking and does not say how many rows "
        "it saw")
    assert evidence["n_rows_seen"] == n_all - n_sealed, (
        f"/selection/evidence counted {evidence['n_rows_seen']} of {n_all} "
        f"with {n_sealed} sealed; it counts off `training_mask`, so this is "
        f"the training total and not the analysis one")
    assert evidence.get("n_rows_withheld") == n_sealed


def test_the_three_routes_disagree_about_what_n_rows_seen_means(client):
    """`TEST-101`, made falsifiable. The split above is unfalsifiable on
    `clinic_visits.csv`, whose target has no blanks — 98 == 98 either way.

    So this drives a frame that HAS blank outcomes and asserts the disagreement
    directly: two routes report the analysis population, one reports the
    training population, and the gap between them is exactly the unsealed rows
    with no outcome. **"Still green" does not mean "no change needed"**, and
    this is the assertion that can tell the difference.

    When `GUIDED-244` lands, this test is where the third route's expectation
    moves — deliberately, rather than by a loop discovering the split was stale.
    """
    n_rows, n_labeled = 300, 120
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({
        "age": rng.normal(50, 12, n_rows).round(1),
        "bmi": rng.normal(27, 5, n_rows).round(1),
        "sbp": rng.normal(128, 16, n_rows).round(1),
        "chol": rng.normal(190, 30, n_rows).round(1),
    })
    outcome = pd.Series(rng.normal(7.0, 1.2, n_rows).round(2), dtype=object)
    outcome.iloc[n_labeled:] = None
    frame["hba1c"] = outcome
    pid = client.post("/project", files={
        "file": ("blank_outcomes.csv",
                 frame.to_csv(index=False).encode(), "text/csv")}
    ).json()["id"]

    def decide(kind, **payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (kind, r.text[:250])

    decide("set_target", column="hba1c")
    decide("set_purpose", answer="prediction")
    decide("set_grain", answer="one_row_per_person")
    decide("set_eligibility", answer="everyone")
    decide("seal", fraction=0.25)
    project = api.STORE.get(pid)

    # THE CONTROL, on the axis that matters: without it every assertion below
    # is satisfied by both meanings, which is the whole of `TEST-101`.
    assert len(project.training_rows) > len(project.analysis_rows), (
        "this frame has no rows whose outcome is blank, so the three routes "
        "cannot disagree and nothing below is a measurement")

    shelf = client.get(f"/project/{pid}/models").json()
    recipes = client.get(f"/project/{pid}/recipes").json()
    evidence = client.get(f"/project/{pid}/selection/evidence").json()

    analysis, training = len(project.analysis_rows), len(project.training_rows)
    assert shelf["n_rows_seen"] == recipes["n_rows_seen"] == analysis
    assert evidence["n_rows_seen"] == training
    assert shelf["n_rows_seen"] != evidence["n_rows_seen"], (
        "the three routes agree, so `n_rows_seen` means one thing and this "
        "test's premise — and the split above — is stale")
    blank_unsealed = int((project.training_mask
                          & project.df["hba1c"].isna()).sum())
    assert training - analysis == blank_unsealed, (
        f"the gap between the two meanings is {training - analysis} and the "
        f"unsealed rows with no outcome number {blank_unsealed}")


def test_every_ranking_primitive_is_called_only_from_a_declared_site():
    """The completeness half. **Adding a ranking without a mask fails HERE**,
    which is the whole of `GUIDED-092`.

    An AST walk rather than a grep, because the question is *does this run*
    rather than *does this text appear* — `LOOP.md` §06. Its limit is stated in
    `turbotab/rankings.py`: a brand-new primitive that routes through none of
    these is invisible to it, and a new CALL to any of them is not.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(ROOT.glob("turbotab/*.py")) + sorted(ROOT.glob("ml/*.py")):
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None)
            if name in rankings.CALL_SITES:
                found.setdefault(name, set()).add(
                    str(path.relative_to(ROOT)))

    assert found, "the sweep found no ranking primitive at all"      # control
    for name, declared in rankings.CALL_SITES.items():
        sites = found.get(name, set())
        assert sites, (
            f"{name!r} is declared with call sites {list(declared)} and the "
            f"sweep found none — the declaration has gone stale")
        undeclared = sorted(sites - set(declared))
        assert not undeclared, (
            f"{name!r} is a ranking primitive and is called from "
            f"{undeclared}, which is not in `rankings.CALL_SITES`. A ranking "
            f"added without declaring which rows it may see is `GUIDED-088` "
            f"again — add the surface to `rankings.SURFACES` with its scope.")
        stale = sorted(set(declared) - sites)
        assert not stale, (
            f"{name!r} is declared as called from {stale} and is not — a "
            f"declaration that outlives its call site stops guarding anything")


def test_the_registry_and_the_probes_cover_the_same_surfaces():
    """A registry row with no probe is a claim nothing checks; a probe with no
    row is a surface the sweep above will not look for."""
    assert set(READS) == set(rankings.keys()), (
        "the registry and this file's probes disagree about what a ranking is: "
        f"registry-only={sorted(set(rankings.keys()) - set(READS))}, "
        f"probe-only={sorted(set(READS) - set(rankings.keys()))}")


def test_every_exemption_names_a_ledger_row_and_a_reason():
    """An exemption is a decision, so it carries the two things a decision in
    this project carries: why, and where it is tracked."""
    exemptions = rankings.exemptions()
    assert exemptions, (
        "no exemptions are registered, so this check passes vacuously — delete "
        "it, or the next one added is unguarded")
    for surface in exemptions:
        assert surface.tracked_by.startswith(("GUIDED-", "AUDIT-", "IMPORT-")), (
            f"{surface.key}: {surface.tracked_by!r} is not a ledger row")
        assert len(surface.because) > 120, (
            f"{surface.key}: an exemption's reason is a paragraph, not a label")
