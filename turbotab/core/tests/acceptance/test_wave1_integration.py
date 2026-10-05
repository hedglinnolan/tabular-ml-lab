"""Wave 1's seams: what holds only once the five packages (DATAIN, SURVEY, OMICS, SCALES, NCI) run on
one engine beside WP16-18.

Each package's own acceptance file tests its method against its references. This file tests the
places where two of them meet, each a rule one package wrote that the other's method must obey:

* **One registry** (BLUEPRINT §13). The packages wrote four contract shapes; every method now enters
  through one registry (``turbotab.core.contracts``), with one vocabulary for slots, scopes and
  rungs, and one run order that every *precedes* relation agrees with.
* **One answer to the pooled QCs** (WP18 RO-13 and MS7). WP18 made the QC rows leave the working
  table before anything reads it; OMICS corrected drift from them first. The pooled-QC finding has
  one answer family: QC-RLSC leads where an injection order is read, the exclusion on its own
  beside it, and a QC-RLSC answer makes the QC rows leave just as the exclusion does, so the naming
  census's finding about the same rows is answered by it.
* **The population answer binds the scales** (MS4 and MS8). Under the surveyed population every
  display is design-based or blocked and recorded (MODELING_SEQUENCE §4); a scale's corrected
  coefficient, fit on these rows as sampled, is blocked with the sample-only attestation as its exit,
  and that exit runs.
* **The plan holds the scales** (WP16-17 and MS8). No inference estimate is shown before the
  questions it rests on are answered; a scale's corrected coefficient is an estimate, withheld with
  the reason until then, and the first one shown locks the analysis plan.

The expected values are the rules themselves, read from the packages' own constants; the numbers a
corrected coefficient takes are SCALES's acceptance file's to check against R.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core.tests.acceptance.scales_fixtures import linear_scale_table
from turbotab.core.tests.acceptance.server_drive import answer_wp17, local_server, open_project
from turbotab.core.tests.truths import Truth

WAVE1 = ("qc_detection_filter", "qc_rlsc", "qc_rsd_filter", "qc_pqn", "qc_rows_leave",
         "d_ratio_filter", "omics_normalization", "detection_limit", "log_transform", "batch",
         "screen", "autoscaling", "multiplicity", "scales", "nci_usual_intake", "join_files",
         "import_codebook")


# ── one registry ─────────────────────────────────────────────────────────────


def test_every_wave1_method_enters_through_the_one_registry():
    """BLUEPRINT §13: slot, data scope, needs, routing (question; options labeled customary and
    sound for each purpose, a rung for each), storyboard and relations, in one registry with one
    vocabulary. A conflict is refused or blocked and recorded, never silent; a relation that names
    the code making it fire names code that exists; a method whose parts learn from different rows
    declares each part's scope; the run order agrees with every *precedes* relation, and the scale
    score runs after batch correction and before scaling, as the pipeline's steps do."""
    from turbotab.core import contracts as C
    from turbotab.core.models.pipeline import ADJUST_STEPS

    registry = C.contracts()
    assert set(WAVE1) <= set(registry)
    assert not hasattr(importlib.import_module("turbotab.core.methods"), "contract")
    for key in WAVE1:
        c = registry[key]
        assert c.slot in C.SLOTS and c.scope in C.SCOPES, key
        assert c.needs and c.question and c.storyboard and c.options, key
        for o in c.options:
            assert o.label and o.customary, (key, o.key)
            for purpose in C.PURPOSES:
                assert o.sound[purpose] and o.rung[purpose] in C.RUNGS, (key, o.key, purpose)
                assert o.order[purpose] >= 0
        for r in c.relations:
            if r.kind == "conflicts":
                assert r.rung in ("refused", "block_and_record"), (key, r.target)
            if r.enforced_by:
                module, name = r.enforced_by.split(":")
                assert callable(getattr(importlib.import_module(module), name)), r.enforced_by
        assert all(scope in C.SCOPES for scope in c.parts.values()), key
        if c.slot in C.BEFORE_THE_SEAL:
            assert c.scope in C.PRE_SEAL_SCOPES, key
    assert registry["scales"].parts and set(registry["scales"].parts.values()) == {
        "row_local", "training_fold"}
    order = C.run_order(list(registry))  # raises on a precedes relation the order breaks
    assert order.index("batch") < order.index("scales") < order.index("autoscaling")
    assert ADJUST_STEPS.index("batch") < ADJUST_STEPS.index("score") < ADJUST_STEPS.index("energy")
    assert order.index("join_files") < order.index("qc_rlsc") < order.index("omics_normalization")


# ── one answer to the pooled QCs ─────────────────────────────────────────────


def test_the_pooled_qc_finding_has_one_answer_family_and_qc_rlsc_makes_the_rows_leave():
    """WP18's reference-rows family answers both the pooled-QC finding and the naming census's
    finding about the same rows. Where an injection order is read, QC-RLSC's four options lead (and
    the two for the whole run as one batch, the other batch reading), and the exclusion follows,
    offered once (OMICS's "set aside" is that exclusion). A QC-RLSC answer
    is a reference-row rule just as the exclusion is (its ``qc_levels`` the levels that leave), and
    what it does covers the naming finding's exclusion, so that finding reads as answered by it."""
    from turbotab.core import repairs
    from turbotab.core.methods import qc_drift as Q
    from turbotab.core.reference_rows import EXCLUDE, reference_filter, reference_rules
    from turbotab.core.tests.acceptance.omics_references import drifting_run

    pooled, roles = "pack::metabolomics::pooled_qc", "pack::metabolomics::sample_roles"
    assert repairs.family_for(pooled) is repairs.family_for(roles)
    assert repairs.family_for(pooled).key == "reference_rows"
    assert Q.ASIDE == EXCLUDE

    frame, _ = drifting_run(seed=6, p=20)
    n_qc = int(frame["sample_type"].eq("QC").sum())
    options = repairs.offer({"id": pooled, "repairs": []},
                            {"params": {"column": "sample_type", "qc_value": "QC"}},
                            repairs.OfferContext(frame=frame, target=None))
    keys = [o.key for o in options]
    # The batch is a reading (MS7 repair): the whole run as one batch is offered beside `batch`.
    assert keys == [*Q.RLSC_OPTIONS, "qc_rlsc_lc", "qc_rlsc_gc", EXCLUDE]
    assert [o.decision.params["batch_column"] for o in options[:-1]] == ["batch"] * 4 + [None] * 2
    exclusion = options[-1]
    assert exclusion.decision.params == {"column": "sample_type", "levels": ["QC"]}
    assert exclusion.sentence.startswith(f"`{n_qc}` rows where `sample_type` is `QC`")
    assert "before the held-out rows were drawn" in exclusion.sentence

    rlsc = options[0]
    applied = {pooled: {"action": "applied", "option": rlsc.key,
                        "params": rlsc.decision.params}}
    rules = reference_rules(applied)
    assert rules == [{"column": "sample_type", "levels": ["QC"], "finding": pooled}]
    keep = frame["sample_type"].ne("QC")
    import duckdb

    con = duckdb.connect()
    con.register("t", frame)
    kept = con.execute(f"SELECT count(*) FROM t WHERE {reference_filter(rules)}").fetchone()[0]
    assert kept == int(keep.sum())
    fam = repairs.family_for(pooled)
    done = {m: "the QC-RLSC answer" for m in fam.marks(rlsc.key, rlsc.decision.params)}
    naming = {"id": roles, "repairs": [{"key": EXCLUDE, "decision": {
        "params": {"column": "sample_type", "levels": ["QC"]}}}]}
    assert repairs._covered(naming, done) == "the QC-RLSC answer"


# ── the scales under the population answer, and under the plan ───────────────


SAT = [f"sat_{j}" for j in range(1, 9)]
SAT_SCALE = {"name": "sat_score", "items": SAT, "reverse": ["sat_3", "sat_6"], "low": 1,
             "high": 5, "kind": "reflective", "correction": "regression_calibration",
             "n_boot": 50}
ROLES = {"pid": "identifier", "age": "covariate", "bmi": "covariate", "sat_ref": "excluded",
         **{c: "covariate" for c in SAT}}


def surveyed(n: int = 600, seed: int = 3) -> pd.DataFrame:
    """SCALES's linear table (an 8-item reflective scale, two covariates, a continuous outcome)
    drawn from a stratified two-PSU design with unequal weights."""
    frame = linear_scale_table(n=n)
    rng = np.random.default_rng(seed)
    frame["SDMVSTRA"] = 1 + np.arange(n) % 6
    frame["SDMVPSU"] = 1 + (np.arange(n) // 6) % 2
    frame["WTMEC2YR"] = np.round(rng.uniform(2000, 40000, n), 1)
    return frame


def truth(*extra: str) -> Truth:
    # WP17's adjustment card asks each predictor before the scales answer (not yet a question of
    # the Router's): every predictor here causes both the exposure the card leads with and the
    # outcome, so all of them stay in the model.
    return Truth({"code_or_count:age": "amount", "code_or_count:bmi": "amount",
                  **{f"code_or_count:{c}": "code" for c in extra},
                  **{f"adjust:{c}": "yes,yes,no" for c in ("age", "bmi", *SAT)}},
                 fixture="the surveyed scale table")


def opened(client, path: Path, roles: dict[str, str], design: tuple[str, ...] = ()):
    d = open_project(client, path, truth(*design))
    d.decide({"kind": "set_lens", "lenses": ["survey"]})
    d.reach("target")
    d.decide({"kind": "set_target", "column": "sbp"})
    d.answer("task", {"kind": "set_task", "column": "sbp", "task": "regression"})
    d.reach("purpose")
    d.decide({"kind": "set_purpose", "purpose": "inference"})
    d.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    d.reach("roles")
    d.decide_roles(roles)
    return d


def test_under_the_population_answer_a_scales_correction_is_blocked_with_the_sample_only_exit(
        tmp_path):
    """MODELING_SEQUENCE §4, "Population estimand without a design-based estimator: block and
    record; exit: the sample-only attestation". The scales stage fits its calibration and the refit
    beside it on the rows as sampled, so under the surveyed population its correction is not
    computed: the reason says why, the reliability (a description of the items) stays, the methods
    sentence says the coefficient was not corrected and why, and the exit posts the sample-only
    attestation, after which the correction is computed for these participants."""
    from turbotab.core.models.survey import SAMPLE_EXIT
    from turbotab.core.stages.scales import POPULATION

    surveyed().to_csv(tmp_path / "surveyed.csv", index=False)
    design = ("SDMVSTRA", "SDMVPSU")
    with local_server(tmp_path / "srv") as client:
        d = opened(client, tmp_path / "surveyed.csv",
                   {**ROLES, "WTMEC2YR": "design", "SDMVSTRA": "design", "SDMVPSU": "design"},
                   design)
        assert d.reach("survey")["status"] == "open"
        options = d.artifact("proposals")["survey"]["options"]
        population = next(o for o in options if o["key"].startswith("population"))
        d.decide(population["decision"])
        d.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        d.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        d.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0})
        d.decide({"kind": "set_scales", "scales": [SAT_SCALE]})
        d.reach("models")
        d.decide({"kind": "select_models", "models": ["linear"]})
        blocked = d.artifact("scales")
        (scale,) = blocked["scales"]
        exit_decision = scale["exits"][0]["decision"]
        d.decide(exit_decision)
        sample = d.artifact("scales")
    assert scale["correction"] is None and scale["not_corrected"] == POPULATION
    assert scale["reliability"]["value"] is not None
    assert [e["label"] for e in scale["exits"]] == [SAMPLE_EXIT]
    assert exit_decision == {"kind": "set_survey", "estimand": "sample"}
    assert scale["methods"].endswith(
        f"Its coefficient was not corrected for measurement error. {POPULATION}")
    (corrected,) = sample["scales"]
    assert corrected["correction"] is not None and corrected["exits"] == []
    assert corrected["correction"]["n_boot"] == 50


def test_a_scales_correction_is_the_first_estimate_that_locks_the_plan_and_waits_for_a_reopened_one(
        tmp_path):
    """WP16: the analysis plan locks when its first estimate is shown; WP17: no inference estimate
    is shown while a question it rests on is open. A scale's corrected coefficient, and the
    uncorrected one beside it, are such an estimate. Served first, before any other, it locks the
    plan. A covariate added afterwards reopens the adjustment set (its causal answers are not given
    yet): the correction is then withheld with the reason, the reliability still shown, until that
    covariate is answered."""
    linear_scale_table(n=600).to_csv(tmp_path / "lin.csv", index=False)
    with local_server(tmp_path / "srv") as client:
        d = opened(client, tmp_path / "lin.csv", {**ROLES, "bmi": "excluded"})
        d.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        d.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        d.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0})
        d.decide({"kind": "set_scales", "scales": [SAT_SCALE]})
        d.reach("models")  # the plan's questions are answered on the way, from the truth
        d.decide({"kind": "select_models", "models": ["linear"]})
        unlocked = d.view()["state"].get("plan_locked")
        first = d.artifact("scales")
        locked = d.view()["state"].get("plan_locked")
        d.decide_roles(ROLES)  # bmi joins the model as a covariate
        reopened = [s["key"] for s in d.view()["interview"] if s["status"] in ("open", "waiting")]
        held = d.artifact("scales")
        answer_wp17(d, "adjustment")
        served = d.artifact("scales")
    assert not unlocked
    (scale,) = first["scales"]
    assert "withheld" not in first and scale["correction"] is not None
    assert scale["correction"]["covariates"] == ["age"]
    assert locked
    assert "adjustment" in reopened
    (scale,) = held["scales"]
    assert held["withheld"].startswith("No estimate is shown until")
    assert scale["correction"] is None and scale["not_corrected"] == held["withheld"]
    assert scale["reliability"]["value"] is not None
    (scale,) = served["scales"]
    assert "withheld" not in served and scale["correction"] is not None
    assert sorted(scale["correction"]["covariates"]) == ["age", "bmi"]
