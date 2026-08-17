"""turbotab.instability — refit the whole pipeline B times and show the spread.

`research/CLINICAL_SURVEY_PACK.md` §A4.8, which specifies this almost
completely and is marked ★:

> Refit the **entire** modeling pipeline (including any variable selection) in
> B bootstrap resamples (Riley & Collins recommend on the order of 1000), apply
> each bootstrap model to the original data, and produce: **prediction
> instability plot** (bootstrap predictions vs original-model predictions, one
> point per patient per bootstrap); **MAPE**; **calibration instability plot**
> (all bootstrap calibration curves overlaid). (Riley & Collins, *Biometrical
> Journal* 2023.)

> *"A single point estimate of the C-statistic hides how much your model
> depends on the particular patients you happened to sample… Wide vertical
> spread means an individual patient's predicted risk is not trustworthy even
> if average performance looks fine."*

**Why this could not be built before L35.** *Refit the entire pipeline* was not
expressible: the trainer built its own `ColumnTransformer` with no reference to
anything the user had decided (`GUIDED-095`). `turbotab.pipeline_plan` composes
the fold-fitted pipeline from the record, so *entire* now has a referent — and
the italics in §A4.8 are load-bearing. A bootstrap that refits the estimator
over a fixed feature set measures the stability of an estimator, not of a
modeling process, and the difference is the whole point of the figure.

## Three things this must not get wrong

**1 · The sealed rows are not in it.** The bootstrap is drawn from the training
rows and every bootstrap model is applied to the training rows. §A4.8's *"apply
each bootstrap model to the original data"* means the original development
sample — instability is a property of the development process, not a second
scoring. A resample that reached the held-out rows would dissolve the lockbox
for a figure printed beside lockbox-derived metrics, which is `STATE-013` at a
new address.

**2 · Every resample gets its own seed, derived deterministically.** The same
seed B times produces B identical models, a prediction instability plot that is
a perfect 45° line, and a confident claim that the model is exactly stable.
That failure is silent — the figure looks *better* the more broken it is —
which is why `seeds_used` is returned and asserted rather than trusted.

**3 · `B` is stated in the caption.** `recipes.SCALE_THRESHOLD`'s posture: a
number nobody can see is a number nobody can disagree with.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: How many bootstrap resamples. **Riley & Collins recommend on the order of
#: 1000, and this is 200.** The gap is deliberate and is the honest number
#: rather than the recommended one:
#:
#: each resample refits the ENTIRE pipeline — imputation fitted in-fold, the
#: recipe's transforms, the selector over the recorded candidate pool, then the
#: estimator — so B is multiplied by the cost of a full fit, not of a `predict`.
#: Measured on `metabolomics_untargeted.csv` (61 training rows, 470 columns,
#: median imputation and a mutual-information selector) one resample of a
#: logistic fit costs on the order of 0.1s, so 200 is ~20s and 1000 is ~100s —
#: past the point where a researcher waits, and this runs on the job queue with
#: a cancel for exactly that reason.
#:
#: **What 200 costs statistically, stated rather than waved at:** the Monte
#: Carlo standard error of a per-patient interval endpoint scales as 1/√B, so
#: 200 gives about 2.2× the simulation noise of 1000. The plot's message —
#: *this patient's predicted risk moves by this much* — is legible at 200; a
#: published interval endpoint would want more. The caption says the number so
#: a reader can decide that for themselves, and `RECOMMENDED_B` is carried
#: beside it so the gap is visible rather than implied.
B_RESAMPLES = 200

#: What the source recommends, kept beside what we run. A constant that records
#: only the compromise loses the ability to say there was one.
RECOMMENDED_B = 1000

#: Where the method comes from. Read at build; the gate resolves it.
EVIDENCE = {
    "source": "research/CLINICAL_SURVEY_PACK.md#A4.8 · ★ Prediction stability plots — the modern addition",
    "evidence_status": "CONVENTION",
    "claim": ("Refit the entire modeling pipeline (including any variable "
              "selection) in B bootstrap resamples, apply each bootstrap model "
              "to the original data, and produce the prediction instability "
              "plot, MAPE, and the calibration instability plot. Emerging "
              "rather than expected, but it is what good reviewers now ask "
              "for."),
}


class InstabilityRefusal(Exception):
    """Raised when the resampling cannot honestly be run."""


def seeds_for(seed: int, b: int) -> List[int]:
    """B distinct seeds from one, deterministically.

    A named function rather than an inline expression because it is the thing
    the probe checks: `len(set(seeds_for(s, B))) == B` is the difference
    between an instability plot and a picture of the 45° line.
    """
    root = np.random.default_rng(seed)
    return [int(x) for x in root.integers(0, 2 ** 31 - 1, size=b)]


#: The two sampling schemes, and the answer to *which rows are exchangeable?*
ROW_BOOTSTRAP = "row"
CLUSTER_BOOTSTRAP = "cluster"


def scheme_for(project: Any, rows: pd.DataFrame) -> Dict[str, Any]:
    """Which bootstrap this table needs, and the sentence that says so.

    `GUIDED-114`. The row bootstrap and the seal disagreed about what a row is.
    Constitution §02 makes the grain a precondition of the seal precisely
    because whether one person appears in several rows decides how held-out
    rows are chosen — and then the resampling ignored that answer and drew
    rows.

    **The consequence errs toward reassurance, which is the worst direction.**
    Drawing rows from a longitudinal table pulls the same person into a
    resample several times, so the refits agree more than independent samples
    would, and the instability plot UNDERSTATES the spread. A figure whose
    failure mode is *looks more trustworthy than the data supports* cannot be
    left to a comment.

    The answer is not a name list: `grain.group_col` already holds it. This
    reads that, picks the scheme, and — either way — returns the disclosure,
    because the second half of the finding is that the payload said nothing.
    """
    group_col = (project.grain or {}).get("group_col")
    if not group_col or group_col not in rows.columns:
        return {
            "scheme": ROW_BOOTSTRAP,
            "group_col": None,
            "n_groups": None,
            "because": (
                "Each row is an independent observation on this table, so each "
                "resample draws rows."
                if not group_col else
                f"`{group_col}` was recorded as the grouping column and is not "
                f"in the training rows, so rows were drawn instead. Treat the "
                f"spread below as a lower bound."),
            "understates": bool(group_col),
        }
    groups = rows[group_col]
    n_groups = int(groups.nunique())
    per_group = float(len(rows)) / max(n_groups, 1)
    return {
        "scheme": CLUSTER_BOOTSTRAP,
        "group_col": str(group_col),
        "n_groups": n_groups,
        "rows_per_group": round(per_group, 2),
        "because": (
            f"You recorded that one `{group_col}` can appear in more than one "
            f"row — {n_groups:,} of them across {len(rows):,} training rows "
            f"with an outcome, "
            f"about {per_group:.1f} rows each — so each resample draws "
            f"{n_groups:,} whole {group_col}s with replacement and takes all "
            f"of their rows. Drawing rows instead would pull the same "
            f"{group_col} into a resample repeatedly, and the refits would "
            f"agree more than independent samples would."),
        "understates": False,
    }


def _draw(rng: Any, rows: pd.DataFrame, scheme: Dict[str, Any]) -> np.ndarray:
    """Positions for one resample, under the scheme this table needs.

    Positions rather than labels, because a cluster draw takes the same group
    more than once and duplicate index labels do not survive `.loc`.
    """
    if scheme["scheme"] != CLUSTER_BOOTSTRAP:
        return rng.integers(0, len(rows), size=len(rows))
    codes = pd.factorize(rows[scheme["group_col"]])[0]
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(scheme["n_groups"]))
    ends = np.searchsorted(codes[order], np.arange(scheme["n_groups"]),
                           side="right")
    picked = rng.integers(0, scheme["n_groups"], size=scheme["n_groups"])
    # THE RESAMPLE'S SIZE VARIES, and that is the cluster bootstrap rather than
    # a defect: groups have unequal numbers of rows, so drawing `n_groups`
    # groups with replacement gives a sample near but not equal to `len(rows)`.
    # Forcing it back to a fixed size would mean truncating whole people, which
    # is the independence assumption broken again from the other side.
    return np.concatenate([order[starts[g]:ends[g]] for g in picked])


def _predict(pipe: Any, X: pd.DataFrame, task_type: str) -> Optional[np.ndarray]:
    """The quantity the plot is about: a predicted risk, or a predicted value.

    `None` rather than a fallback when the estimator cannot produce one — a
    classification instability plot drawn over hard 0/1 labels would show
    almost no spread and would say *stable* about a model nobody measured.
    """
    if task_type == "classification":
        if not hasattr(pipe, "predict_proba"):
            return None
        proba = pipe.predict_proba(X)
        if proba is None or getattr(proba, "ndim", 1) != 2 or proba.shape[1] < 2:
            return None
        return np.asarray(proba[:, 1], dtype=float)
    return np.asarray(pipe.predict(X), dtype=float)


def run(project: Any, model_key: str, *, b: int = B_RESAMPLES,
        seed: int = 42, ctx: Any = None) -> Dict[str, Any]:
    """Refit the whole pipeline `b` times over bootstrap resamples.

    Returns the original model's predictions, one row of bootstrap predictions
    per resample, the seeds used, and the resamples that failed — never a
    silently shortened B, because a plot drawn from 37 of 200 resamples with no
    note is a plot that overstates its own stability.
    """
    from turbotab import pipeline_plan as _plan_mod
    from turbotab import training as _training
    from ml.model_registry import get_registry

    if not getattr(project, "lockbox", None):
        raise InstabilityRefusal(
            "The test set has not been sealed, so there are no training rows "
            "to resample: before the seal every row is a training row and a "
            "bootstrap over all of them would be a bootstrap over the study.")
    task_type = project.task_type or "regression"
    target = str(project.target)
    group_col = (project.grain or {}).get("group_col")

    # THE PROPERTY. See `project.py:1948-1950` — `GUIDED-092` one level
    # down, and this was one of five inline copies of `analysis_mask`.
    rows = project.analysis_rows
    if len(rows) < _training.MIN_TEST_ROWS:
        raise InstabilityRefusal(
            f"{len(rows)} training row(s) with an outcome is too few to "
            f"resample from: a bootstrap of {len(rows)} rows redraws the same "
            f"handful and the spread would describe the arithmetic rather "
            f"than the model.")

    X = _training.feature_frame(project, rows)
    y = rows[target]
    # WHICH ROWS ARE EXCHANGEABLE, decided from the recorded grain rather than
    # assumed (`GUIDED-114`). Read before the loop so the answer is one answer.
    scheme = scheme_for(project, rows)
    registry = get_registry()
    spec = registry[model_key]

    # THE ORIGINAL MODEL — the same composition path the run uses, so the 45°
    # reference is the model the user was actually shown rather than a second
    # fit that happens to agree.
    original_pipe = _plan_mod.compose(
        project, model_key, X, seed=seed).build(spec.factory(task_type, int(seed)))
    original_pipe.fit(X, y)
    original = _predict(original_pipe, X, task_type)
    if original is None:
        raise InstabilityRefusal(
            f"{spec.name} does not produce a predicted risk, so there is "
            f"nothing to plot instability of. A curve over hard labels would "
            f"show almost no spread and would read as stability.")

    seeds = seeds_for(seed, b)
    index = np.arange(len(rows))
    curves: List[np.ndarray] = []
    failures: List[str] = []
    selected_sets: List[Sequence[str]] = []
    drawn_sizes: List[int] = []

    for i, resample_seed in enumerate(seeds):
        if ctx is not None:
            ctx.raise_if_cancelled()
            ctx.progress(i / max(b, 1),
                         f"Refitting {spec.name}, resample {i + 1:,} of {b:,}")
        rng = np.random.default_rng(resample_seed)
        draw = _draw(rng, rows, scheme)
        Xb, yb = X.iloc[draw], y.iloc[draw]
        drawn_sizes.append(int(len(draw)))
        try:
            # THE PLAN IS COMPOSED AGAINST THE ORIGINAL FRAME AND FITTED ON THE
            # RESAMPLE, and the first working version had this backwards.
            #
            # Composing against `Xb` looks more faithful to *refit the entire
            # pipeline* and is wrong. `pipeline_plan` decides which columns get
            # a fill from which columns are BLANK in the frame it is given
            # (`plan.undeclared`), so a bootstrap draw that happens to miss
            # every blank in a column composes a pipeline with no fill for it —
            # and then §A4.8's *apply each bootstrap model to the original
            # data* hands that model a column full of NaN. Driven, every one of
            # 20 resamples failed with `Input X contains NaN`.
            #
            # The distinction the bug forces into the open: which columns need
            # a fill is a property of the ORIGINAL data and of the user's
            # record, not a model choice, so it must not vary with the draw.
            # What must vary is everything ESTIMATED — the fill values, the
            # scaler's centre and spread, and above all the SELECTED SET, which
            # is chosen inside `fit` and therefore still varies with the sample.
            # That is what §A4.8's italicised *including any variable selection*
            # asks for, and `selection_moved` is how it is checked rather than
            # assumed.
            # THE PROCEDURE IS HELD FIXED; ONLY THE SAMPLE VARIES. Both seeds
            # here are the RUN's seed, not the resample's, and that is the
            # correction a failed revert probe forced.
            #
            # The first version passed `resample_seed` to both. It produced a
            # beautiful result — the selected set moved in 17 of 20 resamples —
            # and the probe that should have destroyed it did not: replacing
            # the fold-local fit with one fitted on the ORIGINAL rows left the
            # selected set moving just as much. The signal was coming from
            # `_selector`'s `random_state`, which L36 seeded per call to make
            # `mutual_info_*` reproducible; varying it per resample meant the
            # estimator re-broke its ties differently every time on identical
            # data. `selection moved` was measuring the seed.
            #
            # An instability plot is an attribution claim — *this is how much
            # your result depends on which patients you sampled* — so anything
            # that is not the sample has to be nailed down, or the spread is
            # partly the tool's own noise reported as the study's.
            plan = _plan_mod.compose(project, model_key, X, seed=seed)
            pipe = plan.build(spec.factory(task_type, int(seed)))
            pipe.fit(Xb, yb)
            predicted = _predict(pipe, X, task_type)
            if predicted is None:
                failures.append(f"resample {i}: no predicted risk")
                continue
            curves.append(predicted)
            kept, _, _ = _training._surviving_features(pipe, plan)
            if kept is not None:
                selected_sets.append(tuple(kept))
        except Exception as exc:
            # A resample that cannot be fitted is REPORTED, not dropped. The
            # commonest cause is real and interesting: a bootstrap draw that
            # contains one class, which is itself instability.
            failures.append(f"resample {i}: {type(exc).__name__}: {exc}")

    if not curves:
        raise InstabilityRefusal(
            f"None of the {b:,} resamples produced a prediction. First "
            f"failure: {failures[0] if failures else 'unknown'}")

    matrix = np.vstack(curves)
    return {
        "model_key": model_key,
        "model_name": spec.name,
        "task_type": task_type,
        "b_requested": int(b),
        "b_completed": int(matrix.shape[0]),
        "b_recommended": RECOMMENDED_B,
        "n": int(len(rows)),
        "row_labels": [str(i) for i in rows.index],
        "original": [float(v) for v in original],
        "bootstrap": matrix.tolist(),
        "seeds_used": seeds,
        "failures": failures,
        "mape": _mape(original, matrix),
        # `DRIVE-050`'s class one file over. `rows` reached here already
        # narrowed to those with an outcome (`:228-229`), so *"training rows
        # only"* named a wider population than the number beside it, and
        # `figure_specs.py` prints this same `n` labeled *"training rows"* in
        # two PUBLICATION CAPTIONS. The correct phrasing was four lines from
        # the wrong one, at the refusal above.
        "scored_on": "training rows with an outcome (the held-out rows are "
                     "not resampled and not predicted, and a row with no "
                     "outcome cannot be scored against one)",
        # `GUIDED-114`. WHICH SCHEME WAS DRAWN, always, on every project. The
        # finding was two things: rows were drawn on grouped tables, and the
        # payload said nothing about it — 141,126 characters across eighteen
        # keys containing `group`, `cluster`, `subject`, `person`,
        # `understate` and `repeated` zero times. Either half alone is a
        # defect; the silent half is the one that makes a figure read as more
        # reassuring than the data supports.
        "sampling": {**scheme,
                     "sentence": _sampling_sentence(scheme, drawn_sizes),
                     "rows_drawn_min": min(drawn_sizes) if drawn_sizes else None,
                     "rows_drawn_max": max(drawn_sizes) if drawn_sizes else None},
        "selected_sets": [list(s) for s in selected_sets],
        **EVIDENCE,
    }


def _mape(original: np.ndarray, matrix: np.ndarray) -> Dict[str, Any]:
    """**Mean Absolute Prediction Error** — in the prediction's own units.

    §A4.8 writes `MAPE` and does not expand it, and the two readings give
    different numbers. Riley & Collins define it as *mean absolute prediction
    error*: the mean of |bootstrap prediction − original prediction| over every
    (patient, resample) pair. The other reading, mean absolute PERCENTAGE
    error, divides by the original prediction — and on a predicted risk that is
    a denominator approaching zero.

    Driven on `metabolomics_untargeted.csv`, the percentage form returned
    **658%**, produced almost entirely by patients whose original risk was near
    0.02. That is not a stability finding, it is a division. So the absolute
    form is what is reported, the name is written out in full wherever it is
    shown so nobody has to guess which was meant, and the percentage form is
    returned beside it **only where the denominator supports one** — with the
    count of rows it had to exclude, because a percentage computed over 41 of
    61 patients and labeled as if it covered all of them is the more
    dangerous of the two numbers.
    """
    absolute = float(np.mean(np.abs(matrix - original)))
    denominator = np.abs(original)
    # A tenth of the observed range. Not a tuned threshold — it is stated in
    # the returned payload and the count of excluded rows is stated with it, so
    # a reader can see exactly how much of the study the percentage describes.
    floor = float(np.ptp(original)) / 10.0 if np.ptp(original) > 0 else 0.0
    usable = denominator > max(floor, 0.0)
    percentage = None
    if usable.any():
        percentage = float(np.mean(
            np.abs(matrix[:, usable] - original[usable])
            / denominator[usable]) * 100.0)
    return {
        "absolute": absolute,
        "label": "Mean absolute prediction error",
        "units": "predicted risk" if float(np.max(original)) <= 1.0
                 and float(np.min(original)) >= 0.0 else "outcome units",
        "percentage": percentage,
        "percentage_excluded_rows": int((~usable).sum()),
        "percentage_floor": floor,
        "percentage_note": (
            "Mean absolute PERCENTAGE error is reported only over rows whose "
            "original prediction exceeds a tenth of the observed range; on "
            "predictions near zero it measures the denominator rather than the "
            "model."),
    }


def spread(result: Dict[str, Any]) -> Dict[str, Any]:
    """Per-row spread, which is the sentence the plot is read for.

    §A4.8: *"Wide vertical spread means an individual patient's predicted risk
    is not trustworthy even if average performance looks fine."* The plot shows
    it; this states it, so a reader who cannot see the figure still gets the
    finding and the manuscript can carry it.
    """
    matrix = np.asarray(result["bootstrap"], dtype=float)
    original = np.asarray(result["original"], dtype=float)
    lo = np.percentile(matrix, 2.5, axis=0)
    hi = np.percentile(matrix, 97.5, axis=0)
    widths = hi - lo
    worst = int(np.argmax(widths))
    return {
        "median_width": float(np.median(widths)),
        "max_width": float(widths[worst]),
        "worst_row_label": result["row_labels"][worst],
        "worst_original": float(original[worst]),
        "worst_interval": [float(lo[worst]), float(hi[worst])],
        "per_row": [{"label": label, "original": float(o),
                     "lo": float(a), "hi": float(bb)}
                    for label, o, a, bb in
                    zip(result["row_labels"], original, lo, hi)],
    }


def selection_moved(result: Dict[str, Any]) -> Dict[str, Any]:
    """Did the SELECTED SET move across resamples? `GUIDED-103`'s probe.

    The finding's own note says the selector is *"fold-local by construction
    under any resampling."* This is the observable form of that claim: if the
    selector genuinely runs inside each resample, the chosen set varies with
    the sample on a table where the candidates are close together, and does not
    where one candidate dominates.

    `None` for `moved` when no selection was recorded — which is not `False`,
    and a consumer that could not tell those apart would report *selection is
    perfectly stable* about a project that never selected.
    """
    sets = [tuple(s) for s in result.get("selected_sets") or []]
    if not sets:
        return {"moved": None, "n_distinct": 0, "n_resamples_with_a_set": 0,
                "because": ("No selection was recorded, so there is no chosen "
                            "set to be stable or unstable.")}
    distinct = {frozenset(s) for s in sets}
    return {
        "moved": len(distinct) > 1,
        "n_distinct": len(distinct),
        "n_resamples_with_a_set": len(sets),
        "most_common": sorted(max(
            (s for s in distinct),
            key=lambda d: sum(1 for x in sets if frozenset(x) == d))),
        "because": (
            f"{len(distinct)} distinct feature set(s) were chosen across "
            f"{len(sets)} resamples."),
    }


def _sampling_sentence(scheme: Dict[str, Any], sizes: List[int]) -> str:
    """The disclosure, in one sentence a reader of the figure gets either way.

    `GUIDED-089`'s precedent, applied here: the trainer could not honor the
    recorded plan and **every run said so in its own notes**, and that
    disclosure is what made the gap acceptable rather than silent. A grouped
    project whose plot was drawn from a row bootstrap gets the same treatment —
    the number is still shown, and it is labeled as the lower bound it is.
    """
    line = scheme["because"]
    if sizes and min(sizes) != max(sizes):
        line += (f" Resample sizes range from {min(sizes):,} to "
                 f"{max(sizes):,} rows, because groups differ in size.")
    if scheme.get("understates"):
        line += (" The spread below is therefore a LOWER BOUND on the real "
                 "instability, not an estimate of it.")
    return line
