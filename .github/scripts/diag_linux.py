"""Temporary (ci/linux-numerics only): platform diagnostics printed as annotations."""
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the checkout, for turbotab


def esc(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def say(title: str, lines: list[str]) -> None:
    print(f"::warning title={title}::{esc(chr(10).join(lines))}")


def explain_regression() -> None:
    from turbotab.core.decisions import ColumnUnitSpec, ExplainSpec
    from turbotab.core.tests import modeling_fixtures as mf
    from turbotab.core.tests.acceptance import test_explain as T

    tmp = Path(tempfile.mkdtemp())
    run = T._graph(tmp / "regression", mf.nhanes_like(600, seed=1),
                   energy_adjustment=mf.energy("residual"), explain=ExplainSpec(),
                   outcome_unit="mg/dL",
                   column_units={"age": ColumnUnitSpec(unit="years"),
                                 "kcal": ColumnUnitSpec(unit="kcal", days=1)})
    fit = run.out["fit"].data
    lines = [f"primary {fit['primary_metric']}"]
    for m in fit["models"]:
        lines.append(f"{m['family']} {m['versus_baseline']['verdict']} "
                     f"{m['cv'][fit['primary_metric']]['estimate']!r}")
    ex = ["protein_adj", "sugar", "carb_adj", "fat_total_adj", "fat_sat", "fat_mon", "fat_poly"]
    for f in run.art["families"]:
        w = {i["input"]: i["mean_abs"] for i in f["importance"]}
        top = sorted(((a, w.get(a)) for a in ex), key=lambda t: -(t[1] or 0))[:4]
        lines.append(f"{f['family']} " + ", ".join(f"{a} {v!r}" for a, v in top))
    lines.append("curves " + ", ".join(c["input"] for c in run.art["curves"]))
    model = run.out["fit"].objects["fitted"]["elastic_net"][-1]
    lines.append(f"elastic net alpha_ {model.alpha_!r} l1_ratio_ {model.l1_ratio_!r}")
    path = np.asarray(model.mse_path_).mean(axis=-1)  # l1_ratios x alphas
    flat = np.sort(path.ravel())
    lines.append(f"inner-CV MSE best {flat[0]!r} next {flat[1]!r} rel gap {(flat[1] - flat[0]) / flat[0]:.3g}")
    say("diag explain regression", lines)


def machine() -> None:
    import scipy.linalg  # noqa: F401 - loads the BLAS threadpool_info reports
    import sklearn.linear_model  # noqa: F401 - loads the OpenMP runtime
    from threadpoolctl import threadpool_info

    lines = [f"{platform.system()} {platform.machine()} {platform.processor()}"]
    try:
        cpu = [l for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")]
        lines.append(cpu[0] if cpu else "no model name")
    except OSError:
        pass
    for pool in threadpool_info():
        lines.append(" ".join(f"{k}={pool.get(k)}" for k in
                              ("user_api", "internal_api", "version", "architecture", "num_threads",
                               "filepath")))
    say("diag machine", lines)


def wp7_elastic_net() -> None:
    import json

    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests import modeling_fixtures as mf
    from turbotab.core.tests.acceptance import test_wp7_missing_data as T

    reference = json.loads(T.REFERENCE.read_text())["configs"]
    frame = T.prediction_fixture()
    paths = mf.ingest_frame(frame, Path(tempfile.mkdtemp()))
    split = mf.split_bundle(np.arange(len(frame)), seed=707)
    ti = mf.target_info("regression")
    lines = []
    for name, slots in T.PREDICTION_CONFIGS.items():
        st = mf.state(**slots)
        design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
        fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
        m = next(m for m in fit.data["models"] if m["family"] == "elastic_net")
        for metric, ref in reference[name]["elastic_net"]["cv"].items():
            got = m["cv"][metric]["estimate"]
            lines.append(f"{name} {metric} ref {ref['estimate']:.6g} [{ref['ci_low']:.6g}, "
                         f"{ref['ci_high']:.6g}] now {got:.6g} diff {got - ref['estimate']:+.3g}")
        model = fit.objects["fitted"]["elastic_net"][-1]
        lines.append(f"{name} alpha_ {model.alpha_:.6g} l1_ratio_ {model.l1_ratio_:.3g}")
    say("diag wp7 elastic net", lines)


if __name__ == "__main__":
    machine()
    wp7_elastic_net()
