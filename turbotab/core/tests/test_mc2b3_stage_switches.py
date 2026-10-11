"""MC-2b-3 (WAVE_C6A_PLAN §3; §7 ruling 14): the stage switches on a family's key retired, and two
of phase 3's rulings in the same files.

Every expected value comes from outside the code under test:

* **the pins:** each predicate that replaced a switch is checked, over every registered family
  (and every task where the switch read one), against the switch itself, copied here from the
  base it replaced (turbotab-next 0b0b10ed) as the reference;
* **the tables:** ``effects.matrix_table`` now asks each family's own ``inference`` for its table;
  the switch it replaced is copied here and run beside it on the same fixtures, family by family;
* **the scan:** the three stage modules are read by MC-2's own syntax-tree scan
  (``acceptance/test_mc2_no_family_switches.scan_source``);
* **ruling 14a:** a recording splitter (``tuning.inner_splits_for`` wrapped) sees every inner
  split the tuned family's refits in effects draw, and each is read against the table's own
  ``pid`` column: no person is on both sides;
* **Table 2 end to end:** each fixture's Table 2, diagnostics and marginal contrasts are pinned
  to the numbers the base itself wrote on the same fixture (``TABLE2_AT_BASE``);
* **ruling 14d:** the exported matrix and the lineage are read against the table's own levels of
  ``gender``; the shape, the lineage caption and each family's widths against a design of the
  linear family alone (the base's code path); the one fit of the steps before the coding by
  counting the fill's fits, and both matrices against each coding's whole chain fitted alone.

Fixtures are small (at most 300 rows).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.models.base import TASKS, families
from turbotab.core.tests import modeling_fixtures as mf

CORE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def _registered() -> None:
    import turbotab.core.models  # noqa: F401 - registers the families
    from turbotab.core.contracts import contracts

    contracts()  # the omics chain registers a family of its own


# ── the switches as they were (turbotab-next 0b0b10ed), the reference for each pin ─────────────

OLD_SEQUENCE_FAMILIES = ("linear", "proportional_odds", "cox", "featurewise", "mixed", "gee")


def _old_matrix_table(family: Any, matrix: pd.DataFrame, y: Any, *, task: str, classes: Any,
                      clusters: Any, outcome: Any, survey: Any, levels: Any, event: Any,
                      features: Any) -> Any:
    """``stages.effects.matrix_table`` at 0b0b10ed, verbatim but for its imports."""
    from turbotab.core.models.inference import _on_rows, _on_scale

    if family.key == "featurewise":
        from turbotab.core.models.featurewise import featurewise_table

        table = featurewise_table(matrix, y, task, [f for f in features if f in matrix.columns],
                                  clusters, event)
        return _on_rows(table, len(matrix), "all")
    if family.key == "cox":
        return family.inference_matrix(matrix, y, task="time_to_event", classes=[0, 1],
                                       clusters=clusters, outcome=outcome, rows="all", survey=survey)
    if family.key == "proportional_odds":
        return family.inference_matrix(matrix, y, task=task, classes=levels, clusters=clusters,
                                       outcome=outcome, rows="all", survey=survey)
    if family.key == "linear":
        return family.inference_matrix(matrix, y, task=task, classes=classes, clusters=clusters,
                                       outcome=outcome, rows="all", survey=survey)
    if family.key == "mixed":
        from turbotab.core.models.repeated import mixed_table

        return _on_rows(_on_scale(mixed_table(matrix, y, clusters), task, None, outcome),
                        len(matrix), "all")
    if family.key == "gee":
        from turbotab.core.models.repeated import gee_table

        return _on_rows(_on_scale(gee_table(matrix, y, clusters, task, classes), task, classes,
                                  outcome), len(matrix), "all")
    return None


# ── the pins ─────────────────────────────────────────────────────────────────────────────────


def test_each_effects_predicate_selects_exactly_the_families_its_switch_named() -> None:
    from turbotab.core.stages import effects as E

    for f in families():
        k = f.key
        # SEQUENCE_FAMILIES and family_block's ``supported``; matrix_table's six branches
        assert E.refits_on_matrix(f) == (k in OLD_SEQUENCE_FAMILIES), k
        # _multiplicity, family_block's spline, _one's two feature-wise branches
        assert E.tests_only(f) == (k == "featurewise"), k
        # _one: ``family.key not in ("linear", "featurewise")`` is no robustness value
        assert E.least_squares_on_matrix(f) == (k in ("linear", "featurewise")), k
        # _with_relative: ``family.key != "linear"`` has no relative effect
        assert E.one_regression_on_matrix(f) == (k == "linear"), k
        # marginal: ``family.key != "linear" or self.task != "binary"`` standardizes nothing
        for task in TASKS:
            assert E.standardizes_risks(f, task) == (k == "linear" and task == "binary"), (k, task)
        # diagnostics: ``"proportional_hazards" if family.key == "cox" else "influence"``
        assert E._diagnostic_check(f) == ("proportional_hazards" if k == "cox" else "influence")
        assert ("proportional_hazards" in f.diagnostics) == (k == "cox"), k
        assert ("influence" in f.diagnostics) == (k == "linear"), k
    assert E.SEQUENCE_FAMILIES == tuple(k for k in (f.key for f in families())
                                        if k in OLD_SEQUENCE_FAMILIES)


def test_each_fit_and_evaluation_predicate_selects_exactly_the_families_its_switch_named() -> None:
    from turbotab.core.stages import modeling as M
    from turbotab.core.stages.evaluation import updates_by_shrinkage

    for f in families():
        k = f.key
        assert updates_by_shrinkage(k) == (k == "linear"), k  # evaluation._shrinkage
        assert M.states_collinearity(f) == (k == "linear"), k  # fit_stage's concern
        for task in TASKS:  # _pooled_curve: ``family.key == "linear" and task == "regression"``
            assert M.linear_predictor_is_prediction(f, task) == (
                k == "linear" and task == "regression"), (k, task)
    assert not updates_by_shrinkage(None) and not updates_by_shrinkage("no_such_family")


def test_the_three_stage_modules_hold_no_switch_on_a_family() -> None:
    """MC-2's own scan finds none of the twelve places left in effects, evaluation and modeling."""
    from turbotab.core.tests.acceptance.test_mc2_no_family_switches import scan_source

    for name in ("effects", "evaluation", "modeling"):
        source = (CORE / "stages" / f"{name}.py").read_text(encoding="utf-8")
        assert scan_source(source, f"turbotab.core.stages.{name}") == {}, name


# ── Table 2's tables, family by family, against the switch they replaced ─────────────────────


def _matrix(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(0, 1, n), "z": rng.normal(0, 1, n),
                         "sex_male": rng.binomial(1, 0.5, n).astype(float)})


def _cases() -> list[tuple[str, str, dict[str, Any]]]:
    """(family, task, everything ``matrix_table`` is handed) for each of the six families."""
    from turbotab.core.models.inference import INDEPENDENT, Clusters, Outcome
    from turbotab.core.models.survival import survival_outcome

    n = 240
    M = _matrix(n, 3)
    rng = np.random.default_rng(4)
    lin = M["x"] * 0.5 - 0.3 * M["z"] + rng.normal(0, 1, n)
    binary = (rng.random(n) < 1 / (1 + np.exp(-(0.6 * M["x"] - 0.2)))).astype(float)
    ordinal = np.digitize(lin + rng.normal(0, 0.5, n), [-0.5, 0.5]).astype(float)
    time = rng.exponential(np.exp(-0.4 * M["x"]))
    event = (rng.random(n) < 0.7).astype(float)
    codes = np.repeat(np.arange(n // 6), 6)
    units = Clusters(column="pid", codes=codes, n_clusters=n // 6)
    unit_effect = np.repeat(rng.normal(0, 1, n // 6), 6)
    base = dict(clusters=INDEPENDENT, survey=None, levels=None, event=None, features=["x"])
    return [
        ("linear", "regression", dict(base, y=lin.to_numpy(), task="regression", classes=None,
                                      outcome=Outcome(name="y"))),
        ("linear", "binary", dict(base, y=binary, task="binary", classes=[0.0, 1.0],
                                  outcome=Outcome(name="dm", labels={0.0: "no", 1.0: "yes"}))),
        ("proportional_odds", "ordinal",
         dict(base, y=ordinal, task="ordinal", classes=None, levels=["low", "mid", "high"],
              outcome=Outcome(name="grade", labels={0.0: "low", 1.0: "mid", 2.0: "high"}))),
        ("cox", "time_to_event", dict(base, y=survival_outcome(event, time, np.zeros(n)),
                                      task="time_to_event", classes=None,
                                      outcome=Outcome(name="death"))),
        ("featurewise", "regression", dict(base, y=lin.to_numpy(), task="regression",
                                           classes=None, outcome=Outcome(name="y"),
                                           event="death")),
        ("mixed", "regression", dict(base, y=(lin + unit_effect).to_numpy(), task="regression",
                                     classes=None, clusters=units, outcome=Outcome(name="y"))),
        ("gee", "regression", dict(base, y=(lin + unit_effect).to_numpy(), task="regression",
                                   classes=None, clusters=units, outcome=Outcome(name="y"))),
        ("gee", "binary", dict(base, y=binary, task="binary", classes=[0.0, 1.0], clusters=units,
                               outcome=Outcome(name="dm", labels={0.0: "no", 1.0: "yes"}))),
    ]


@pytest.mark.parametrize("key,task,kw", _cases(), ids=lambda v: v if isinstance(v, str) else "")
def test_each_familys_table_on_a_matrix_is_the_one_its_switch_made(key, task, kw) -> None:
    """Reference: the replaced switch, copied above, on the same matrix: the rows, the record and
    the concerns are equal, number for number."""
    from turbotab.core.models import get_family
    from turbotab.core.stages.effects import matrix_table

    family = get_family(key)
    M = _matrix(240, 3)
    new = matrix_table(family, M, **kw)
    old = _old_matrix_table(family, M, **kw)
    assert new.rows == old.rows
    assert new.info == old.info
    assert list(new.concerns) == list(old.concerns)
    assert new.rows and new.info["rows"] == "all"


def test_a_family_whose_table_is_not_made_from_a_matrix_has_none() -> None:
    from turbotab.core.models import get_family
    from turbotab.core.stages.effects import matrix_table

    for key in ("ridge", "elastic_net", "boosted_trees"):
        assert matrix_table(get_family(key), _matrix(40, 1), np.zeros(40), task="regression",
                            classes=None, clusters=None, outcome=None, survey=None, levels=None,
                            event=None, features=["x"]) is None


# ── ruling 14a: effects' refits keep a person's rows on one side of every inner split ────────


def _probe_family() -> Any:
    """A tuned family with a table made from its matrix: the linear family under another key,
    searching its solver's convergence tolerance (two candidates; the fitted model barely moves),
    as ``test_f15_every_refit`` probes. Only a test registers it."""
    from turbotab.core.models.linear import Linear
    from turbotab.core.models.tuning import Dimension, TuningDecl

    class ProbeTuned(Linear):
        key = "probe_tuned_linear_mc2b3"
        label = "Probe tuned linear"
        tuning = TuningDecl(
            "search", dimensions=(Dimension("tol", "how closely the fit converges", "tolerance",
                                            1e-10, 1e-8, "log", source="a probe"),),
            standard={"tol": 1e-8}, space_version="probe/1", reason="A probe.")

    return ProbeTuned()


def test_a_repeated_measures_refit_in_effects_keeps_each_person_whole(tmp_path) -> None:
    """Three rows a person, the split grouped by ``pid``: every inner split the tuned family's
    refits in effects draw (Model 2 and the unadjusted model, each searched) puts all of a
    person's rows on one side. Before ruling 14a the refits were handed no units, so the search
    drew its inner folds by row content and split people. The fixture's 80 people are below the
    size from which a search draws Sobol candidates (``tuning.SMALL_N``), so the test lowers it to
    draw two beside the standard one, and the plan searches."""
    from turbotab.core.models import base as B
    from turbotab.core.models import tuning
    from turbotab.core.stages.effects import effects_stage
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    probe = _probe_family()
    frame = ef.cohort(n=240, seed=17)
    frame["pid"] = np.arange(len(frame)) // 3
    seen: list[tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]] = []
    real = tuning.inner_splits_for

    def recording(plan: Any, X: Any, y: Any, rows: Any, **kw: Any) -> Any:
        splits = real(plan, X, y, rows, **kw)
        seen.append((np.asarray(X.index), list(splits or [])))
        return splits

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(B._REGISTRY, probe.key, probe)
        patch.setattr(tuning, "SOBOL_SIZES", ((0, 2),))
        st = ef.state(target="dm", task="binary", event="yes", measure="odds_ratio",
                      models=[probe.key], grain=d.GrainSpec(grain="repeated", id_column="pid"))
        paths = mf.ingest_frame(frame, tmp_path)
        split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed,
                                groups=frame["pid"].to_numpy(), grouped_by="pid")
        info = mf.target_info(st.task, st.target)
        design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
        assert len(design.objects["spec"]["plans"][probe.key]["candidates"]) == 3
        patch.setattr(tuning, "inner_splits_for", recording)
        out = effects_stage(mf.context(st, {"design": design, "split": split,
                                            "target_info": info}, paths)).data
    assert seen, "effects never searched the tuned family"
    person = frame["pid"].to_numpy()  # row id = position in the table written
    for ids, splits in seen:
        assert splits
        for train, test in splits:
            assert not set(person[ids[train]]) & set(person[ids[test]])
    # The probe's table is made from its matrix, so its declared models are refit too (read from
    # its declaration; the replaced tuple of keys left it at Model 2 alone).
    assert [s["key"] for s in out["families"][0]["sequence"]][:2] == ["crude", "model_2"]
    assert len(seen) >= 2


# ── Table 2 and its diagnostics, end to end, against the base ────────────────────────────────

TABLE2_CASES: dict[str, dict[str, Any]] = {
    "regression, Model 1": dict(target="glucose", task="regression", measure="mean_difference",
                                model_1=["age", "sex"]),
    "binary, risk difference": dict(target="dm", task="binary", event="yes",
                                    measure="risk_difference"),
    "binary, influence answered": dict(target="dm", task="binary", event="yes",
                                       measure="odds_ratio",
                                       responses={"influence": "without_influential"}),
    "regression, linear and ridge": dict(target="glucose", task="regression",
                                         measure="mean_difference", models=["linear", "ridge"]),
}


def _sig(v: Any) -> Any:
    return None if v is None else float(f"{float(v):.10g}")


def table2_digest(effects: dict[str, Any]) -> list[Any]:
    """Each family's Table 2 numbers (every declared model's exposure rows), its diagnostics and
    its marginal contrasts, to ten significant digits."""
    out: list[Any] = []
    for fam in effects["families"]:
        seq = [[s["key"], s["n_rows"], [[c["feature"], *(_sig(c[k]) for k in
                                                         ("estimate", "ci_low", "ci_high", "p"))]
                                        for c in s["effects"] or []]]
               for s in fam["sequence"]]
        diag = [[g["check"], g["status"], g.get("flagged"), _sig(g.get("largest")),
                 _sig(g.get("threshold"))] for g in fam.get("diagnostics") or []]
        marginal = [[_sig(c[k]) for k in ("rd", "rd_low", "rd_high", "rr", "rr_low", "rr_high")]
                    for c in ((fam.get("marginal") or {}).get("contrasts") or [])]
        out.append([fam["family"], seq, diag, marginal])
    return out


def table2_run(case: str, folder: Path) -> list[Any]:
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    run = ef.run(ef.cohort(n=400, seed=11), folder, ef.state(**TABLE2_CASES[case]), fit=False)
    return table2_digest(run["effects"])


# Written by ``table2_run`` on turbotab-next 0b0b10ed, before any switch was retired (that base's
# own code, this file's digest, run from an extracted copy of the base).
TABLE2_AT_BASE: dict[str, list[Any]] = {'regression, Model 1': [['linear',
                          [['crude', 400,
                            [['fiber', -0.6723983284, -0.8813263549, -0.4634703019,
                              6.733755163e-10]]],
                           ['model_1', 400,
                            [['fiber', -0.7957578859, -1.005154732, -0.5863610401,
                              5.13214653e-13]]],
                           ['model_2', 400,
                            [['fiber', -0.4339636701, -0.6517959996, -0.2161313406,
                              0.0001058123793]]],
                           ['model_3', 400,
                            [['fiber', -0.4106627675, -0.6299202457, -0.1914052893,
                              0.0002634702769]]]],
                          [['influence', 'passed', 0, 0.03358087797, 0.8928800797]], []]],
 'binary, risk difference': [['linear',
                              [['crude', 400,
                                [['fiber', -0.05698415863, -0.1046789092, -0.009289408039,
                                  0.01919602887]]],
                               ['model_2', 400,
                                [['fiber', -0.01854076865, -0.07331320883, 0.03623167154,
                                  0.5070369758]]],
                               ['model_3', 400,
                                [['fiber', -0.01862928697, -0.07365287537, 0.03639430143,
                                  0.5069570819]]]],
                              [['influence', 'passed', 0, 0.02254784794, 0.8928800797]],
                              [[-0.003957277376, -0.01606392812, 0.007346635417, 0.9904066003,
                                0.9599735785, 1.017787809]]]],
 'binary, influence answered': [['linear',
                                 [['crude', 400,
                                   [['fiber', -0.05698415863, -0.1046789092, -0.009289408039,
                                     0.01919602887]]],
                                  ['model_2', 400,
                                   [['fiber', -0.01854076865, -0.07331320883, 0.03623167154,
                                     0.5070369758]]],
                                  ['model_3', 400,
                                   [['fiber', -0.01862928697, -0.07365287537, 0.03639430143,
                                     0.5069570819]]]],
                                 [['influence', 'passed', 0, 0.02254784794, 0.8928800797]],
                                 []]],
 'regression, linear and ridge': [['linear',
                                   [['crude', 400,
                                     [['fiber', -0.6723983284, -0.8813263549, -0.4634703019,
                                       6.733755163e-10]]],
                                    ['model_2', 400,
                                     [['fiber', -0.4339636701, -0.6517959996, -0.2161313406,
                                       0.0001058123793]]],
                                    ['model_3', 400,
                                     [['fiber', -0.4106627675, -0.6299202457, -0.1914052893,
                                       0.0002634702769]]]],
                                   [['influence', 'passed', 0, 0.03358087797, 0.8928800797]],
                                   []]]}


@pytest.mark.parametrize("case", list(TABLE2_CASES))
def test_table2_and_its_diagnostics_are_the_bases_number_for_number(tmp_path, case) -> None:
    """Reference: the base's own run of the same fixture (``TABLE2_AT_BASE``)."""
    assert table2_run(case, tmp_path) == TABLE2_AT_BASE[case]


# ── ruling 14d: the exported matrix and the lineage follow each family's coding ──────────────


def _design(tmp_path: Path, models: list[str], missing: bool = False) -> Any:
    from turbotab.core.stages.modeling import design_stage

    frame = mf.nhanes_like(n=200, seed=2, missing=missing)
    st = mf.state(models=models, task="regression", purpose="prediction",
                  **({"missing": "impute"} if missing else {}))
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), seed=1)
    return design_stage(mf.context(st, {"split": split, "target_info": mf.target_info(
        "regression")}, paths)), frame


def _exported(design: Any) -> pd.DataFrame:
    from turbotab.core.export.matrix import FILE

    return pd.read_parquet(design.files[FILE])


def _matrix_nodes(design: Any) -> set[str]:
    return {str(n["column"]) for n in design.data["lineage"]["nodes"] if n["lane"] == "matrix"}


def _first_level_width(tmp_path: Path) -> int:
    """The first-level-dropped matrix's width, from a design of the linear family alone (whose
    coding drops each category's first level, the code path the base held)."""
    design, _ = _design(tmp_path / "linear_alone", ["linear"])
    return int(design.data["matrix"]["n_cols"])


@pytest.mark.parametrize("models,coded", [
    (["ridge"], {"gender_female", "gender_male"}),
    (["elastic_net"], {"gender_female", "gender_male"}),
    (["ridge", "elastic_net"], {"gender_female", "gender_male"}),
    (["linear"], {"gender_male"}),
    (["linear", "ridge"], {"gender_male"}),
    (["linear", "elastic_net", "boosted_trees"], {"gender_male"}),
])
def test_the_exported_matrix_and_the_lineage_follow_the_families_coding(tmp_path, models,
                                                                        coded) -> None:
    """``gender`` holds female and male in the table written. Ridge and the elastic net code every
    level, so alone or together their exported matrix and lineage hold both; the linear family
    drops the first (female). Chosen with a family of the other coding, the first-level-dropped
    matrix is shown: the one the linear family's estimates are fit on. The shape the design
    reports and the lineage figure's caption are the exported matrix's; each family is described
    with its own width (the first-level-dropped matrix's, one more column for gender when it
    codes every level)."""
    from turbotab.core.export.figures import lineage as lineage_figure
    from turbotab.core.models import get_family
    from turbotab.core.models.pipeline import DesignSpec, describe_steps, onehot_drop

    design, frame = _design(tmp_path, models)
    assert set(frame["gender"].dropna()) == {"female", "male"}
    matrix = _exported(design)
    assert {c for c in matrix.columns if c.startswith("gender")} == coded
    assert {c for c in _matrix_nodes(design) if c.startswith("gender")} == coded
    shown = design.data["matrix"]
    width = len(matrix.columns) - 1  # row_id
    assert (shown["n_rows"], shown["n_cols"]) == (len(matrix), width)
    if "gender_female" in coded:
        np.testing.assert_array_equal(matrix["gender_female"] + matrix["gender_male"], 1.0)
    first = _first_level_width(tmp_path)
    assert width == first + (1 if "gender_female" in coded else 0)
    _, caption = lineage_figure(design.data)
    assert f"to the {width:,} columns of the model matrix" in caption
    spec = DesignSpec.from_dict(design.objects["spec"])
    for entry in design.data["models"]:
        family = get_family(entry["family"])
        assert entry["steps"] == describe_steps(spec, family, "regression", "prediction", first)
        scaled = [step["detail"] for step in entry["steps"] if step["key"] == "scale"]
        own = first + (1 if onehot_drop(family) is None else 0)
        assert all(f"all {own} columns" in detail for detail in scaled), (family.key, scaled)
        if family.key in ("ridge", "elastic_net"):
            assert scaled, family.key


def test_every_family_whose_declared_models_are_refit_on_the_matrix_drops_the_first_level() -> None:
    """Why the first-level-dropped matrix is shown when both codings are chosen: it is the matrix
    of every family whose Table 2 is refit on it."""
    from turbotab.core.models.pipeline import onehot_drop
    from turbotab.core.stages.effects import refits_on_matrix
    from turbotab.core.stages.modeling import shown_coding

    for f in families():
        if refits_on_matrix(f):
            assert onehot_drop(f) == "first", f.key
            assert shown_coding([f, *(g for g in families() if onehot_drop(g) is None)]) == "first"


def test_the_full_coding_is_fitted_after_the_shared_steps_are_fitted_once(tmp_path) -> None:
    """With only full-coding families chosen, the steps before the coding (the fill, here) are
    fitted once, and both matrices are what fitting each coding's whole chain makes."""
    from sklearn.impute import SimpleImputer

    from turbotab.core.models import get_family
    from turbotab.core.models.pipeline import DesignSpec, shared_steps, transformer
    from turbotab.core.stages.modeling import fit_shared

    design, frame = _design(tmp_path, ["ridge"], missing=True)
    spec = DesignSpec.from_dict(design.objects["spec"])
    X = frame[spec.inputs]
    assert [n for n, _ in shared_steps(spec)][:1] == ["impute"]
    fits: list[int] = []
    real = SimpleImputer.fit

    def counted(self: Any, *a: Any, **kw: Any) -> Any:
        fits.append(1)
        return real(self, *a, **kw)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SimpleImputer, "fit", counted)
        shared, matrix, shown, exported = fit_shared(spec, X, [get_family("ridge")])
        once = len(fits)
        fits.clear()
        transformer(shared_steps(spec)).fit_transform(X)
        assert once == len(fits) > 0
    pd.testing.assert_frame_equal(matrix, transformer(shared_steps(spec)).fit_transform(X))
    pd.testing.assert_frame_equal(exported, transformer(shared_steps(spec, onehot_drop=None))
                                  .fit_transform(X))
    assert [n for n, _ in shared.steps] == [n for n, _ in shown.steps]
    assert shared.named_steps["impute"] is shown.named_steps["impute"]
    assert set(exported.columns) - set(matrix.columns) == {"gender_female"}
