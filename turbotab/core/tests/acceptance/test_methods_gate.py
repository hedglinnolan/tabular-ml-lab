"""The methods layer's gate repair: the items the gate verifier left open after the repair round
(docs/turbotab-next/audit/fix-methods-result.json; AUDIT_REPORT §5 WP8, WP12; BLUEPRINT §12).

* **A · RO-03's race** (WP12b.3, the verifier's items 4 and 10). ``set_follow_up`` was refused
  once the task was known to be binary, but accepted in the window before ``target_info`` was
  fresh: the context's task read None, the follow-up was recorded with the sentence "`cvd_event`
  was analyzed as a time to event", and the fit was logistic (fiber log-odds −0.069, p = 1.7 ×
  10⁻¹⁷). Only a recorded answer makes an outcome a time to event, so the follow-up now waits for
  that answer, and its sentence claims nothing without it.
* **B · BLUEPRINT §12 ruling 3, completely** ("Inference estimates from all eligible rows … a
  holdout is a prediction concept"). Under inference the design read the training rows: the log
  residual printed the training rows' elasticity, the residual-gap warning read "on 2,400 training
  rows", the shelf's basis counted training rows and the previews sampled them, all beside a
  coefficient table from every analyzed row.
* **C · %E substitution** (B24/D19). The exit "add the missing shares" added every share; shares
  sum to 100%, so that model is rank-deficient. The field's model leaves one source out as the
  reference, and the estimand names it.
* **D · a True/False outcome.** The modeling frame turned the outcome's booleans into 1.0 and 0.0,
  so the declared event matched no level: captions read "`1.0` against `0.0`", and with `False`
  declared the model estimated the odds of `True` under a sentence saying `False` was coded 1.
* **E · Willett's sex-specific screen** left the exclusions menu when sex was not a predictor; a
  screen reads a column the model does not use, as the Goldberg screen already did.

Every expected value comes from outside the engine: numpy and statsmodels on the fixture's own
columns, pandas counts, the task itself, and one primary source.

**Source check (C).** Hu FB, Stampfer MJ, Manson JE, et al. Dietary fat intake and the risk of
coronary heart disease in women. *N Engl J Med* 1997;337:1491-1499 (abstract read through the
Europe PMC REST service, PMID 9366580, on 2026-10-03): "Mutivariate analyses included age, smoking
status, total energy intake, dietary cholesterol intake, percentages of energy obtained from protein
and specific types of fat, and other risk factors." and "Each increase of 5 percent of energy intake
from saturated fat, as compared with equivalent energy intake from carbohydrates, was associated
with a 17 percent increase in the risk of coronary disease". Every share but carbohydrate's is in
the model, with total energy; carbohydrate is the reference each coefficient is compared with.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import decisions as d
from turbotab.core import voice
from turbotab.core.decisions import (EnergyAdjustment, ProjectState, Refusal, SetFollowUp, SetTask,
                                     SplitSpec, SubstitutionSpec, parse_decision)
from turbotab.core.stages.modeling import design_stage, fit_stage, shelf_stage, substitution_stage
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance.server_drive import local_server, open_project
from turbotab.core.tests.truths import Truth

FOLLOW = {"kind": "set_follow_up", "column": "cvd_event", "time_column": "followup_years"}


def _sig3(value: float) -> str:
    """A coefficient as the design warning and the preview card print it."""
    return f"{value:+.3g}".replace("-", "−")


def _answer_until(drive: Any, stop: str, answers: dict[str, dict[str, Any]]) -> None:
    """Answer the Router's open questions in its order up to and including ``stop``."""
    order = ["lens", "orientation", "target", "event", "task", "purpose", "grain", "repeat_kind",
             "unit", "aggregation", "temporal", "roles", "survey", "exclusions", "missing", "split",
             "energy_adjustment", "models"]
    for key in order:
        step = drive.reach(key)
        if step["status"] in ("open", "waiting"):
            assert key in answers, f"no answer for the open step {key}"
            if answers[key]["kind"] == "set_roles":
                # The author's roles, each confirmed on its own (BLUEPRINT §14, the leash).
                drive.decide_roles(answers[key]["roles"])
            else:
                drive.decide(answers[key])
        if key == stop:
            return


# ── A · the follow-up waits for the time-to-event answer ─────────────────────


def test_a_the_follow_up_is_refused_while_the_task_is_not_settled():
    """The engine side of the race. ``validate`` with the task None (the server's context while
    ``target_info`` is not fresh) refused nothing; it now refuses, with the exit that declares the
    time to event, and accepts once that answer is recorded whatever detection says.

    Reference: the task itself. Detection reads a 0/1 column as binary or regression and never as a
    time to event (``ml.triage.detect_task_type`` answers regression or classification), so only a
    recorded ``set_task`` makes an outcome one."""
    from turbotab.server.service import DecisionContext

    unsettled = ProjectState(target="cvd_event")
    contexts = [
        {"target": "cvd_event", "task": None},  # the verifier's engine repro
        {"target": "cvd_event", "state": unsettled},  # the state holds no answer
        DecisionContext(columns=["cvd_event", "followup_years"], ingest_status="fresh",
                        target="cvd_event", state=unsettled, task=None),  # the server's window
    ]
    for ctx in contexts:
        with pytest.raises(Refusal) as refused:
            d.validate(FOLLOW, ctx)
        assert refused.value.code == "not_time_to_event"
        assert "not settled" in refused.value.message
        exit_ = parse_decision(refused.value.exits[0]["decision"])
        assert isinstance(exit_, SetTask) and (exit_.column, exit_.task) == ("cvd_event", "time_to_event")

    answered = ProjectState(target="cvd_event", task="time_to_event")
    for detected in (None, "binary"):  # the recorded answer outranks detection, fresh or not
        d.validate(FOLLOW, DecisionContext(columns=["cvd_event", "followup_years"],
                                           ingest_status="fresh", target="cvd_event",
                                           state=answered, task=detected or "time_to_event"))
        d.validate(FOLLOW, {"target": "cvd_event", "state": answered, "task": detected})


def test_a_the_follow_up_sentence_claims_no_time_to_event_without_the_answer():
    """The sentence path of the race: with no task recorded and none detected yet, the follow-up's
    sentence said "`cvd_event` was analyzed as a time to event". Reference: the task each state
    holds."""
    decision = SetFollowUp(column="cvd_event", time_column="followup_years")
    for state, ctx in ((ProjectState(target="cvd_event"), None),
                       (ProjectState(target="cvd_event"), {"detected_task": None}),
                       (ProjectState(target="cvd_event"), {"detected_task": "binary"})):
        said = voice.sentence_for(decision, state, ctx)
        assert "was analyzed as a time to event" not in said and "`followup_years`" in said, said
        assert "not yet declared" in said or "not used" in said, said
    said = voice.sentence_for(decision, ProjectState(target="cvd_event", task="time_to_event"), None)
    assert said.startswith("`cvd_event` was analyzed as a time to event, each row followed until "
                           "`followup_years`")


def test_a_the_race_through_the_server(tmp_path, monkeypatch):
    """The verifier's repro (s4c_race.py) through the real HTTP API, with the window held open:
    ``set_lens``, ``set_target``, then ``set_follow_up`` while ``target_info`` is not fresh. The
    window is forced by handing the service's decision context a ``target_info`` that is still
    running, so the test does not depend on the worker's timing; the context's task is then None,
    exactly the verifier's state. Posted at once without the patch it is refused too (on whichever
    side of the window it lands). Taking the refusal's exit and posting again is accepted, and the
    record then says what the fit does."""
    from turbotab.core.tests.acceptance.test_wp12b_cox_mixed_gee import _staggered_entry_cohort

    frame = _staggered_entry_cohort()
    path = tmp_path / "cohort_tte_null.csv"
    frame.to_csv(path, index=False)
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "cvd_event"})
        early = drive.post(FOLLOW)  # at once, as the verifier posted it
        assert early.status_code == 409 and "not_time_to_event" in early.text, early.text

        service = client.app.state.service
        real = service.decision_context

        def window(pid: str, stages: Any = None) -> Any:
            stages = dict(service.engine.status(pid))
            stages["target_info"] = stages["target_info"].model_copy(
                update={"status": "running", "key": None, "fresh": False})
            return real(pid, stages)

        monkeypatch.setattr(service, "decision_context", window)
        assert window(drive.pid).task is None  # the window: no task known
        held = drive.post(FOLLOW)
        assert held.status_code == 409, held.text
        error = held.json()["error"]
        assert error["code"] == "not_time_to_event" and "not settled" in error["message"]
        assert error["exits"][0]["decision"] == {"kind": "set_task", "column": "cvd_event",
                                                 "task": "time_to_event"}
        monkeypatch.setattr(service, "decision_context", real)

        assert drive.artifact("target_info")["task"] == "binary"  # detection: yes/no
        kinds = [r["decision"]["kind"] for r in drive.view()["decisions"]]
        assert "set_follow_up" not in kinds  # nothing was recorded over the binary task
        drive.decide(error["exits"][0]["decision"])
        drive.decide(FOLLOW)
        view = drive.view()
    said = [r["sentence"] for r in view["decisions"] if r["decision"]["kind"] == "set_follow_up"]
    assert len(said) == 1 and said[0].startswith("`cvd_event` was analyzed as a time to event")
    assert view["state"]["task"] == "time_to_event"


# ── B · under inference, every number beside the table is every analyzed row's ─────


def _residual_fixture() -> pd.DataFrame:
    """The verifier's s14_residual.py fixture (seed 1414, 3,000 rows), rounded as its CSV is."""
    rng = np.random.default_rng(1414)
    n = 3000
    male = rng.integers(0, 2, n)
    age = rng.normal(50, 12, n)
    pa = rng.normal(0, 1, n)
    log_e = np.log(1700) + 0.25 * male + 0.10 * pa - 0.004 * (age - 50) + rng.normal(0, 0.22, n)
    energy = np.exp(log_e)
    fat = np.clip(rng.normal(0.34, 0.06, n), 0.1, 0.6) * energy / 9 * np.exp(rng.normal(0, 0.1, n))
    y = 0.02 * fat + 0.004 * energy + 1.5 * male + 0.8 * pa + 0.03 * age + rng.normal(0, 2, n)
    return pd.DataFrame({"pid": np.arange(n), "fat_g": fat.round(3), "energy_kcal": energy.round(1),
                         "male": male, "age": age.round(2), "pa": pa.round(3), "y": y.round(3)})


RESIDUAL_ROLES = {"pid": "identifier", "fat_g": "exposure", "energy_kcal": "energy",
                  "male": "covariate", "age": "covariate", "pa": "covariate"}
# WP17: the generator's causal truth (``_residual_fixture``): the question is fat's effect on y; sex,
# age and activity set energy, and so fat, and each sets the outcome; total energy is decided by
# the energy question.
RESIDUAL_TRUTH = {"exposure:y": "fat_g", **{f"adjust:{c}": "yes,yes,no" for c in ("male", "age", "pa")}}


def _ols(y: Any, exog: pd.DataFrame) -> Any:
    return sm.OLS(np.asarray(y, dtype=float), sm.add_constant(exog.astype(float))).fit()


def _gap(frame: pd.DataFrame) -> tuple[float, float]:
    """(energy dropped, standard) for fat on these rows: numpy's least-squares residual of fat on
    energy, then statsmodels OLS with the three covariates, and the standard model."""
    slope, _ = np.polyfit(frame["energy_kcal"], frame["fat_g"], 1)
    adjusted = frame["fat_g"] - slope * (frame["energy_kcal"] - frame["energy_kcal"].mean())
    covariates = frame[["male", "age", "pa"]]
    dropped = _ols(frame["y"], pd.concat([adjusted.rename("a"), covariates], axis=1)).params["a"]
    standard = _ols(frame["y"], frame[["fat_g", "energy_kcal", "male", "age", "pa"]]).params["fat_g"]
    return float(dropped), float(standard)


def _elasticity(frame: pd.DataFrame) -> float:
    """numpy's least-squares slope of log fat on log energy."""
    return float(np.polyfit(np.log(frame["energy_kcal"]), np.log(frame["fat_g"]), 1)[0])


def test_b_the_design_beside_an_every_row_table_reads_every_analyzed_row(tmp_path):
    """The verifier's s14 repro through the real split, design and fit stages, 20% held out.

    Under inference: the log residual's printed elasticity is numpy's slope on all 3,000 rows (the
    training rows' slope rounds differently); the energy-dropped residual's warning reads "on 3,000
    analyzed rows" with numpy/statsmodels' all-row coefficients (+0.0226 against +0.0216, the
    verifier's 0.022624 and 0.021624), and the table beside it is estimated from those same rows.
    Under prediction the same design reads the 2,400 training rows and says so, with the training
    rows' own references."""
    frame = _residual_fixture()
    paths = mf.ingest_frame(frame, tmp_path)
    frame = pd.read_csv(Path(paths["source"]))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.2, seed=3)
    train = split.frames["assignment"].query("partition == 'train'")["row_id"].to_numpy()
    ti = mf.target_info("regression", "y")

    def run(method: str, log: bool, purpose: str) -> tuple[Any, Any]:
        st = mf.state(roles=RESIDUAL_ROLES, target="y", task="regression", purpose=purpose,
                      models=["linear"], split=SplitSpec(holdout=0.2, seed=3, folds=5),
                      energy_adjustment=EnergyAdjustment(method=method, energy_column="energy_kcal",
                                                         nutrients=["fat_g"], log_transform=log))
        design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
        fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
        return design.data, fit.data["models"][0]

    every_b, train_b = _elasticity(frame), _elasticity(frame.iloc[train])
    assert abs(every_b - train_b) > 1e-3  # the two round apart at the printed three decimals
    design, model = run("residual", True, "inference")
    printed = float(re.search(r"b = ([0-9.]+) for fat_g", design["estimand"]).group(1))
    assert printed == pytest.approx(every_b, abs=5e-4) and abs(printed - train_b) > 5e-4
    assert model["coefficients_n"] == 3000 and design["matrix"]["n_rows"] == 3000

    dropped, standard = _gap(frame)
    assert (round(dropped, 6), round(standard, 6)) == (0.022624, 0.021624)  # the verifier's
    design, model = run("residual_energy_dropped", False, "inference")
    gap = next(w for w in design["warnings"] if w.startswith("energy_kcal left the outcome model"))
    assert (f"on 3,000 analyzed rows fat_g_adj's coefficient is {_sig3(dropped)}, against "
            f"{_sig3(standard)} with energy_kcal kept") in gap, gap
    table = next(r for r in model["coefficients"] if r["feature"] == "fat_g_adj")
    assert table["estimate"] == pytest.approx(dropped, rel=1e-6)  # the table beside it agrees

    t_dropped, t_standard = _gap(frame.iloc[train])
    assert _sig3(t_dropped) != _sig3(dropped)
    design, _ = run("residual_energy_dropped", False, "prediction")
    gap = next(w for w in design["warnings"] if w.startswith("energy_kcal left the outcome model"))
    assert (f"on 2,400 training rows fat_g_adj's coefficient is {_sig3(t_dropped)}, against "
            f"{_sig3(t_standard)} with energy_kcal kept") in gap, gap
    design, _ = run("residual", True, "prediction")
    printed = float(re.search(r"b = ([0-9.]+) for fat_g", design["estimand"]).group(1))
    assert printed == pytest.approx(train_b, abs=5e-4)


def _binary_cohort(n: int = 900, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fiber = rng.gamma(4, 5, n)
    age = rng.normal(55, 9, n)
    p = 1 / (1 + np.exp(-(-1.4 + 0.03 * (fiber - 20) + 0.04 * (age - 55))))
    return pd.DataFrame({"pid": np.arange(n), "fiber_g": fiber.round(2), "age": age.round(1),
                         "case": (rng.random(n) < p).astype(int)})


def test_b_the_shelf_ranks_for_the_rows_the_table_reads(tmp_path):
    """The shelf's basis line and sample-size criteria (the verifier: "Ranked for 2,400 training
    rows … 526 with the event" beside coefficients from 3,000). Reference: pandas counts of the
    rows and of the rarer class, on every analyzed row under inference and on the training rows
    under prediction."""
    frame = _binary_cohort()
    paths = mf.ingest_frame(frame, tmp_path)
    ids = np.arange(len(frame))
    split = mf.split_bundle(ids, holdout=0.25, seed=5)
    train = split.frames["assignment"].query("partition == 'train'")["row_id"].to_numpy()
    roles = {"pid": "identifier", "fiber_g": "exposure", "age": "covariate"}
    inputs = {"cohort": mf.cohort_bundle(ids, ["fiber_g", "age"]), "split": split,
              "target_info": mf.target_info("binary", "case")}
    for purpose, rows, word in (("inference", frame, "analyzed"),
                                ("prediction", frame.iloc[train], "training")):
        st = mf.state(roles=roles, target="case", task="binary", event="1", purpose=purpose,
                      models=["linear"], lens=["clinical"])
        shelf = shelf_stage(mf.context(st, inputs, paths))
        rarer = int(rows["case"].value_counts().min())
        assert shelf["basis"] == (f"Ranked for {len(rows):,} {word} rows and 2 predictors, "
                                  f"{rarer:,} in the rarer class."), shelf["basis"]
    assert len(train) == 675


def test_b_previews_under_inference_read_every_analyzed_row(tmp_path):
    """A decision preview (the energy-dropped residual's card) through the real server, 20% held
    out. Under inference the card samples every analyzed row, its basis says so and no row is
    sealed from it; its gap is then the all-row gap computed above with numpy and statsmodels.
    Under prediction the same card reads the training rows and keeps the held-out rows sealed."""
    frame = _residual_fixture()
    path = tmp_path / "resid.csv"
    frame.to_csv(path, index=False)
    frame = pd.read_csv(path)
    dropped, standard = _gap(frame)
    preview = {"kind": "set_energy_adjustment", "method": "residual_energy_dropped",
               "energy_column": "energy_kcal", "nutrients": ["fat_g"]}
    with local_server(tmp_path / "home") as client:
        for purpose in ("inference", "prediction"):
            drive = open_project(client, path, Truth(RESIDUAL_TRUTH, fixture="_residual_fixture"))
            answers = {
                "lens": {"kind": "set_lens", "lenses": ["dietary"]},
                "target": {"kind": "set_target", "column": "y"},
                "task": {"kind": "set_task", "column": "y", "task": "regression"},
                "purpose": {"kind": "set_purpose", "purpose": purpose},
                "grain": {"kind": "set_grain", "grain": "one_row_per_unit", "id_column": "pid"},
                "temporal": {"kind": "set_temporal", "temporal": False},
                "roles": {"kind": "set_roles", "roles": RESIDUAL_ROLES},
                "survey": {"kind": "set_survey", "estimand": "sample"},
                "exclusions": {"kind": "set_exclusions", "rules": []},
                "missing": {"kind": "set_missing", "strategy": "complete_case"},
                "split": {"kind": "set_split", "holdout": 0.2, "seed": 3, "folds": 5},
            }
            _answer_until(drive, "split", answers)
            sealed = drive.sealed()
            assert len(sealed) == 600
            # Under inference the exposure and the adjustment set come after the seal and before
            # the energy model (MODELING_SEQUENCE §1 steps 2–4), answered from the truth.
            drive.answer_wp17_before(preview)
            drive.reach("energy_adjustment")  # its card's stages read for the answers above
            # The preview samples the split's rows (the cohort's while it recomputes), read only
            # once fresh; the estimand and adjustment answers above recompute both, and the energy
            # question waits on the proposals alone, so the pool is waited for here.
            drive.artifact("split")
            r = client.post(f"/api/projects/{drive.pid}/preview", json=preview)
            assert r.status_code == 200, r.text
            result = r.json()
            caption = " ".join(v.get("caption") or "" for v in result["views"])
            if purpose == "inference":
                assert result["basis"].startswith("Values on all 3,000 analyzed rows"), result["basis"]
                assert "sealed" not in result["basis"]
                assert (f"coefficient {_sig3(dropped)}, against {_sig3(standard)} with "
                        f"`energy_kcal` kept") in caption, caption
            else:
                kept = frame.drop(index=sorted(sealed))
                t_dropped, t_standard = _gap(kept)
                assert result["basis"].startswith("Values on all 2,400 training rows"), result["basis"]
                assert result["basis"].endswith("held-out rows stay sealed."), result["basis"]
                assert (f"coefficient {_sig3(t_dropped)}, against {_sig3(t_standard)} with "
                        f"`energy_kcal` kept") in caption, caption


def test_b_the_omitted_energy_check_reads_every_analyzed_row(tmp_path):
    """Under inference the omitted-sources check (ME-05) read the rows outside the held-out set.
    Here the 200 rows drawn into the holdout eat far more protein, so the share the refusal states
    differs by row set. Reference: numpy's mean of (E − 9·fat − 4·carbohydrate)/E over every row
    (the analyzed rows), not over the rows outside the seal."""
    rng = np.random.default_rng(8)
    n = 1000
    energy = rng.normal(2100, 300, n).clip(1000)
    protein_share = np.where(np.arange(n) < 200, 0.40, 0.12)
    fat = energy * 0.33 / 9 * rng.lognormal(0, 0.05, n)
    carb = energy * (1 - 0.33 - protein_share) / 4
    frame = pd.DataFrame({"fat_g": fat, "carb_g": carb, "energy_kcal": energy,
                          "y": rng.normal(size=n)})
    remainder = (energy - 9 * fat - 4 * carb) / energy
    every, outside = float(remainder.mean()), float(remainder[200:].mean())
    assert f"{every:.0%}" != f"{outside:.0%}"
    paths = mf.ingest_frame(frame, tmp_path)
    from turbotab.core.datastore import DataStore

    store = DataStore(Path(paths["data"]), 1 << 30)
    try:
        st = mf.state(roles={"fat_g": "exposure", "carb_g": "exposure", "energy_kcal": "energy"},
                      target="y", purpose="inference")
        swap = d.SetSubstitution(donor="fat_g", recipient="carb_g", step_kcal=100)
        for extra in ({"analyzed": lambda: np.arange(n)}, {}):
            ctx = {"state": st, "store": lambda: store, "sealed": lambda: np.arange(200), **extra}
            with pytest.raises(Refusal) as refused:
                d.validate(swap, ctx)
            assert refused.value.code == "omitted_energy_sources"
            assert f"make up {every:.0%} of total energy" in refused.value.message
    finally:
        store.close()


# ── C · a share of energy leaves one source out as the reference ─────────────


def _shares(n: int, seed: int) -> pd.DataFrame:
    """Fat, carbohydrate, protein and alcohol in percent of energy, summing to 100 on every row
    (the WP12a fixture; carbohydrate is what the other three leave)."""
    rng = np.random.default_rng(seed)
    energy = rng.normal(2100, 450, n).clip(900)
    fat = rng.normal(34, 6, n).clip(12, 55)
    protein = rng.normal(16, 3, n).clip(7, 30)
    alcohol = rng.gamma(1.2, 2.5, n).clip(0, 18)
    carb = 100 - fat - protein - alcohol
    age = rng.uniform(20, 80, n)
    y = (50 + 0.2 * age + 0.10 * fat - 0.05 * carb + 0.02 * protein + 0.004 * energy
         + rng.normal(0, 3, n))
    return pd.DataFrame({"age": age, "fat_pct_kcal": fat, "carb_pct_kcal": carb,
                         "protein_pct_kcal": protein, "alcohol_pct_kcal": alcohol,
                         "energy_kcal": energy, "y": y})


SHARES = ["fat_pct_kcal", "carb_pct_kcal", "protein_pct_kcal", "alcohol_pct_kcal"]
SWAP = {"kind": "set_substitution", "donor": "fat_pct_kcal", "recipient": "carb_pct_kcal",
        "scale": "percent_energy", "step_percent": 5.0}


def _rank_and_width(frame: pd.DataFrame, roles: dict[str, str]) -> tuple[int, int]:
    """numpy's rank of the model matrix the roles make (intercept, age, the exposures, energy)."""
    columns = ["age"] + [c for c in SHARES if roles.get(c) == "exposure"] + ["energy_kcal"]
    X = np.column_stack([np.ones(len(frame)), frame[columns].to_numpy(dtype=float)])
    return int(np.linalg.matrix_rank(X)), X.shape[1]


def test_c_the_exit_leaves_one_share_out_as_the_reference(tmp_path):
    """The verifier's s3_exit.py: under inference, fat and carbohydrate shares with protein and
    alcohol left out are refused (19% of energy left to a composite), and the exit added both,
    a model whose four shares sum to 100%: rank 6 of 7 columns (numpy). Now every exit leaves one
    of them out as the reference, a full-rank model; taking the first is accepted; the design's
    estimand names the reference; and the curve is Hu's leave-one-out contrast, computed with
    statsmodels on every analyzed row (−5·γ_fat with carbohydrate's share the reference there, and
    5·(β_carb − β_fat) with the app's reference: the same number, as it must be)."""
    from turbotab.core.datastore import DataStore

    frame = _shares(1_500, seed=97)
    assert np.allclose(frame[SHARES].sum(axis=1), 100.0)
    roles = {"age": "covariate", "fat_pct_kcal": "exposure", "carb_pct_kcal": "exposure",
             "protein_pct_kcal": "excluded", "alcohol_pct_kcal": "excluded", "energy_kcal": "energy"}
    every_share = {**roles, "protein_pct_kcal": "exposure", "alcohol_pct_kcal": "exposure"}
    assert _rank_and_width(frame, every_share) == (6, 7)  # what the old exit made
    paths = mf.ingest_frame(frame, tmp_path)
    store = DataStore(Path(paths["data"]), 1 << 30)
    try:
        st = mf.state(roles=roles, target="y", models=["linear"], purpose="inference")
        ctx = {"state": st, "store": lambda: store}
        with pytest.raises(Refusal) as refused:
            d.validate(SWAP, ctx)
        assert refused.value.code == "omitted_energy_sources"
        changes = [parse_decision(e["decision"]) for e in refused.value.exits
                   if e["decision"] and e["decision"]["kind"] == "set_roles"]
        assert len(changes) == 2
        for change, reference in zip(changes, ["protein_pct_kcal", "alcohol_pct_kcal"]):
            left = [c for c in SHARES if change.roles.get(c) != "exposure"]
            assert left == [reference]  # exactly one share out: the reference
            assert _rank_and_width(frame, change.roles) == (6, 6)  # full rank
        first = refused.value.exits[0]
        assert first["label"] == ("Add `alcohol_pct_kcal` to the model, with `protein_pct_kcal` "
                                  "left out as the reference")
        chosen = st.model_copy(update={"roles": dict(changes[0].roles)})
        d.validate(SWAP, {**ctx, "state": chosen})  # the leave-one-out model is accepted
        # Protein's own share (16%) is above the 5% bound; it is the reference, not a composite.
        assert frame["protein_pct_kcal"].mean() / 100 > 0.05
    finally:
        store.close()

    chosen = chosen.model_copy(update={"substitution": SubstitutionSpec(
        donor="fat_pct_kcal", recipient="carb_pct_kcal", scale="percent_energy", step_percent=5.0)})
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.2, seed=2)
    ti = mf.target_info("regression", "y")
    design = design_stage(mf.context(chosen, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(chosen, {"design": design, "split": split, "target_info": ti}, paths))
    out = substitution_stage(mf.context(chosen, {"design": design, "fit": fit}, paths))
    assert "leave-one-out model" in design.data["estimand"]
    assert "with protein as the reference" in design.data["estimand"]
    assert "in place of protein" in design.data["estimand"]
    at5 = out["models"][0]["delta"][out["ks"].index(5.0)]
    app = _ols(frame["y"], frame[["age", "fat_pct_kcal", "carb_pct_kcal", "alcohol_pct_kcal",
                                  "energy_kcal"]])
    hu = _ols(frame["y"], frame[["age", "fat_pct_kcal", "protein_pct_kcal", "alcohol_pct_kcal",
                                 "energy_kcal"]])
    assert at5 == pytest.approx(5 * (app.params["carb_pct_kcal"] - app.params["fat_pct_kcal"]),
                                rel=1e-8)
    assert at5 == pytest.approx(-5 * hu.params["fat_pct_kcal"], rel=1e-8)


def test_c_every_share_in_the_model_leaves_no_reference(tmp_path):
    """Every share in the model (the verifier's roles in s3_exit.py) is a rank-deficient model:
    numpy's rank of the intercept and the four shares is 4 of 5. Under inference the swap is
    refused, each exit leaves one share out (full rank again), and the estimand says no source is
    the reference."""
    from turbotab.core.datastore import DataStore

    frame = _shares(1_500, seed=97)
    X = np.column_stack([np.ones(len(frame)), frame[SHARES].to_numpy()])
    assert np.linalg.matrix_rank(X) == 4
    roles = {"age": "covariate", **{c: "exposure" for c in SHARES}, "energy_kcal": "energy"}
    paths = mf.ingest_frame(frame, tmp_path)
    store = DataStore(Path(paths["data"]), 1 << 30)
    try:
        st = mf.state(roles=roles, target="y", models=["linear"], purpose="inference")
        with pytest.raises(Refusal) as refused:
            d.validate(SWAP, {"state": st, "store": lambda: store})
    finally:
        store.close()
    assert refused.value.code == "no_reference_share"
    assert "sum to 100% of energy on every row" in refused.value.message
    changes = [parse_decision(e["decision"]) for e in refused.value.exits if e["decision"]]
    assert [e["label"] for e in refused.value.exits[:2]] == [
        "Leave `protein_pct_kcal` out as the reference", "Leave `alcohol_pct_kcal` out as the reference"]
    for change in changes:
        assert len([c for c in SHARES if change.roles.get(c) == "exposure"]) == 3
        assert _rank_and_width(frame, change.roles) == (6, 6)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    design = design_stage(mf.context(st, {"split": split, "target_info": mf.target_info("regression", "y")},
                                     paths))
    assert "so the reference is the energy from no named source" in design.data["estimand"]
    assert "leave-one-out model" not in design.data["estimand"].split("The field's")[0]


# ── D · a True/False outcome is named by its declared level ──────────────────


def _smokers(n: int = 800, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fiber = rng.gamma(4, 5, n)
    age = rng.normal(50, 10, n)
    p = 1 / (1 + np.exp(-(-1 + 0.03 * (fiber - 20) + 0.02 * (age - 50))))
    return pd.DataFrame({"pid": np.arange(n), "fiber_g": fiber.round(2), "age": age.round(1),
                         "smoker": rng.random(n) < p})


def _logit(frame: pd.DataFrame, event: bool) -> Any:
    """statsmodels Logit of (smoker == event) on fiber and age, every row."""
    y = (frame["smoker"] == event).astype(float)
    return sm.Logit(y, sm.add_constant(frame[["fiber_g", "age"]].astype(float))).fit(disp=0)


def test_d_the_declared_event_of_a_true_false_outcome_is_the_one_modeled(tmp_path):
    """Through the real design and fit stages, inference, every row. With `True` declared the
    table is statsmodels' logit of smoker == True and the caption names `True` against `False`;
    with `False` declared it is the logit of smoker == False (the sign flips), named `False`
    against `True`. Before, both read "`1.0` against `0.0`" and both were the logit of True."""
    frame = _smokers()
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    ti = mf.target_info("binary", "smoker")
    roles = {"pid": "identifier", "fiber_g": "exposure", "age": "covariate"}
    for level, other in (("True", "False"), ("False", "True")):
        st = mf.state(roles=roles, target="smoker", task="binary", event=level, purpose="inference",
                      models=["linear"], lens=["clinical"])
        design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
        fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
        model = fit.data["models"][0]
        info = model["inference"]
        assert info["caption"].startswith(f"Estimates are log-odds of `{level}` against `{other}`")
        assert (info["event"], info["reference"]) == (level, other)
        assert f"being `{level}` rather than `{other}`" in info["effect"]
        reference = _logit(frame, level == "True")
        fiber = next(r for r in model["coefficients"] if r["feature"] == "fiber_g")
        assert fiber["estimate"] == pytest.approx(reference.params["fiber_g"], rel=1e-5)
        assert fiber["ratio"] == pytest.approx(math.exp(reference.params["fiber_g"]), rel=1e-5)
        share = float((frame["smoker"] == (level == "True")).mean())
        assert fit.data["imbalance"].startswith(f"The event, `{level}`, is {share:.0%} of the "
                                                f"analyzed rows")


def test_d_every_surface_names_the_declared_level_through_the_server(tmp_path):
    """The same table through the HTTP API, 20% held out under inference: the methods sentence, the
    table's caption, effect and scale fields, the event's share (pandas, every analyzed row) and
    the shelf's basis (pandas counts, every analyzed row) all speak of `True` and `False`; none
    says `1.0` or `0.0`."""
    frame = _smokers()
    path = tmp_path / "smokers.csv"
    frame.to_csv(path, index=False)
    with local_server(tmp_path / "home") as client:
        # WP17: the generator's causal truth (``_smokers``): age sets the outcome, not fiber.
        drive = open_project(client, path, Truth({"adjust:age": "no,yes,no"}, fixture="_smokers"))
        _answer_until(drive, "models", {
            "lens": {"kind": "set_lens", "lenses": ["clinical"]},
            "target": {"kind": "set_target", "column": "smoker"},
            "event": {"kind": "set_event", "column": "smoker", "level": "True"},
            "task": {"kind": "set_task", "column": "smoker", "task": "binary"},
            "purpose": {"kind": "set_purpose", "purpose": "inference"},
            "grain": {"kind": "set_grain", "grain": "one_row_per_unit", "id_column": "pid"},
            "temporal": {"kind": "set_temporal", "temporal": False},
            "roles": {"kind": "set_roles", "roles": {"pid": "identifier", "fiber_g": "exposure",
                                                     "age": "covariate"}},
            "survey": {"kind": "set_survey", "estimand": "sample"},
            "exclusions": {"kind": "set_exclusions", "rules": []},
            "missing": {"kind": "set_missing", "strategy": "complete_case"},
            "split": {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5},
            "energy_adjustment": {"kind": "set_energy_adjustment", "method": "none"},
            "models": {"kind": "select_models", "models": ["linear"]},
        })
        fit = drive.artifact("fit")
        shelf = drive.artifact("shelf")
        view = drive.view()
    model = fit["models"][0]
    info = model["inference"]
    assert info["caption"].startswith("Estimates are log-odds of `True` against `False`")
    assert (info["event"], info["reference"]) == ("True", "False")
    assert "being `True` rather than `False`" in info["effect"]
    reference = _logit(frame, True)
    fiber = next(r for r in model["coefficients"] if r["feature"] == "fiber_g")
    assert model["coefficients_n"] == 800
    assert fiber["estimate"] == pytest.approx(reference.params["fiber_g"], rel=1e-5)
    share = float(frame["smoker"].mean())
    assert fit["imbalance"].startswith(f"The event, `True`, is {share:.0%} of the analyzed rows")
    rarer = int(frame["smoker"].value_counts().min())
    assert shelf["basis"] == f"Ranked for 800 analyzed rows and 2 predictors, {rarer:,} in the rarer class."
    said = next(r["sentence"] for r in view["decisions"] if r["decision"]["kind"] == "set_event")
    assert said.startswith("`True` of `smoker` was taken as the event and coded 1; `False` was coded 0")
    shown = " ".join([info["caption"], info["effect"], fit["imbalance"], said])
    assert "1.0" not in shown and "0.0" not in shown


# ── E · Willett's sex-specific screen reads sex outside the model ────────────


def _intakes(n: int = 600, seed: int = 31) -> pd.DataFrame:
    """Reported energy with misreporters placed where the sex-specific and sex-neutral cut-offs
    disagree: women at 3,600–4,100 kcal (out for women, in for men) and men at 520–780 kcal (out
    for men, in for women)."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["F", "M"], n)
    energy = np.where(sex == "M", rng.normal(2500, 450, n), rng.normal(1900, 380, n))
    women, men = np.flatnonzero(sex == "F"), np.flatnonzero(sex == "M")
    energy[women[:12]] = rng.uniform(3600, 4100, 12)
    energy[men[:9]] = rng.uniform(520, 780, 9)
    energy[women[12:15]] = rng.uniform(300, 480, 3)
    return pd.DataFrame({"pid": np.arange(n), "sex": sex, "age": rng.uniform(20, 70, n).round(),
                         "energy_kcal": energy.round(1), "fiber_g": rng.gamma(4, 5, n).round(2),
                         "ldl": rng.normal(130, 25, n).round(1)})


def test_e_the_sex_specific_screen_stays_on_the_menu_when_sex_is_left_out(tmp_path):
    """Through the real server: with sex left out of the model (``excluded``), the exclusions menu
    still offers the NHS/HPFS sex-specific cut-offs (WP15 renamed the screen once "Willett, by sex":
    MI-02), with the rule and count it has when sex is a
    covariate; recorded, the cohort drops exactly those rows. Reference: pandas, women outside
    500–3,500 kcal and men outside 800–4,200 kcal; the sex-neutral 500–3,500 screen counts
    differently on this fixture, so the count is the sex-specific one."""
    frame = _intakes()
    path = tmp_path / "intakes.csv"
    frame.to_csv(path, index=False)
    e, women = frame["energy_kcal"], frame["sex"] == "F"
    outside = np.where(women, (e < 500) | (e > 3500), (e < 800) | (e > 4200))
    expected = int(outside.sum())
    assert expected == 24 and expected != int(((e < 500) | (e > 3500)).sum())
    offered: dict[str, Any] = {}
    with local_server(tmp_path / "home") as client:
        from turbotab.core.tests.truths import Truth

        for sex_role in ("covariate", "excluded"):
            # The fixture's truth (BLUEPRINT §14.3): one day's energy in kcal; age whole years.
            # WP17: the generator's causal truth (``_intakes``): sex sets energy only; age, fiber
            # and LDL are drawn independently.
            drive = open_project(client, path, Truth({
                "unit:energy_kcal": "kcal", "day_count:energy_kcal": "1",
                "code_or_count:age": "amount", "adjust:sex": "no,no,no",
                "adjust:age": "no,no,no"}, fixture="_intakes"))
            _answer_until(drive, "roles", {
                "lens": {"kind": "set_lens", "lenses": ["dietary"]},
                "target": {"kind": "set_target", "column": "ldl"},
                "task": {"kind": "set_task", "column": "ldl", "task": "regression"},
                "purpose": {"kind": "set_purpose", "purpose": "inference"},
                "grain": {"kind": "set_grain", "grain": "one_row_per_unit", "id_column": "pid"},
                "temporal": {"kind": "set_temporal", "temporal": False},
                "roles": {"kind": "set_roles", "roles": {"pid": "identifier", "sex": sex_role,
                                                         "age": "covariate",
                                                         "energy_kcal": "energy",
                                                         "fiber_g": "exposure"}},
            })
            drive.reach("exclusions")
            menu = drive.artifact("proposals")["exclusions"]
            offered[sex_role] = next((x for x in menu if x["key"] == "nhs_hpfs_by_sex"), None)
        drive.decide({"kind": "set_exclusions", "rules": [offered["excluded"]["rule"]]})
        cohort = drive.artifact("cohort")
    kept, left = offered["covariate"], offered["excluded"]
    assert left is not None and kept is not None
    assert left["rule"] == kept["rule"] and left["affected"] == kept["affected"] == expected
    assert left["rule"]["by"] == {"column": "sex", "ranges": {"F": [500.0, 3500.0],
                                                             "M": [800.0, 4200.0]}}
    assert cohort["n_final"] == len(frame) - expected


# ── the second gate's ruling-3 remainder, closed by the orchestrator ─────────────────────────


def _collinear(n: int = 1500, seed: int = 41) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = x1 + rng.normal(0, 0.0004, n)  # nearly a copy: a near-singular matrix (condition > 1,000)
    y = 0.5 * x1 + rng.normal(0, 1, n)
    return pd.DataFrame({"pid": np.arange(n), "x1": x1.round(7), "x2": x2.round(7), "y": y.round(5)})


def _singular_number(model: dict[str, Any]) -> float | None:
    for concern in model.get("concerns", []):
        found = re.search(r"condition number ([0-9,]+)", concern)
        if found:
            return float(found.group(1).replace(",", ""))
    return None


def test_b_the_singular_concern_reads_every_analyzed_row_under_inference(tmp_path):
    """The second gate's repro B/collin.py: under inference the near-singular concern beside the
    coefficient table must describe the matrix the table was fit on, every analyzed row. The
    independent property (ruling 3: estimates "independent of the seal's seed"): under inference
    the printed condition number is identical whichever rows the seal holds out; under prediction
    it describes each seed's own training rows, so it moves with the seed."""
    frame = _collinear()
    paths = mf.ingest_frame(frame, tmp_path)
    frame = pd.read_csv(Path(paths["source"]))
    ti = mf.target_info("regression", "y")
    roles = {"pid": "identifier", "x1": "exposure", "x2": "covariate"}

    def printed(purpose: str, seed: int) -> float | None:
        split = mf.split_bundle(np.arange(len(frame)), holdout=0.2, seed=seed)
        st = mf.state(roles=roles, target="y", task="regression", purpose=purpose, models=["linear"],
                      split=SplitSpec(holdout=0.2, seed=seed, folds=5))
        design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
        fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
        return _singular_number(fit.data["models"][0])

    a, b = printed("inference", 0), printed("inference", 1)
    assert a is not None and a == b, (a, b)
    c, d_ = printed("prediction", 0), printed("prediction", 1)
    assert c is not None and d_ is not None and c != d_, (c, d_)


def test_b_step_labels_name_every_analyzed_row_under_inference():
    """Three labels said 'training rows' under inference although the rows used are every analyzed
    row (the second gate: pipeline.py energy step, previews.py caption and step phrase)."""
    from turbotab.core.models.pipeline import energy_detail

    adj = EnergyAdjustment(method="residual", energy_column="energy_kcal", nutrients=["fat_g"])
    under_inference = energy_detail(adj, (), "inference")
    assert "every analyzed row" in under_inference and "training rows" not in under_inference
    assert "training rows" in energy_detail(adj, (), "prediction")
