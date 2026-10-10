"""Materiality: how far an alternative would move the numbers (SURFACING_POLICY §2).

For an item *i* with current answer or default ``a₀`` and alternatives ``A``, ``M(i, s)`` is the
largest movement any alternative causes in the target quantity *Q*, in units of *Q*'s own
uncertainty (§2.1): under Estimate and Describe the half-width of the primary's 95% interval, under
Predict the fold-to-fold standard deviation of the paired difference. It decides between "doesn't
change your numbers here", "could bias" and "act on it", in two regimes recorded side by side
(§2.2):

* **Predicted, before the lock** (:class:`Movement` with ``regime="predicted"``), from legal
  outcome-blind instruments only (§2.3 items 1–2): a theory bound (the attenuation ``1 − λ`` of
  regression dilution, Hutcheon et al. 2010), the rows an alternative changes times the largest
  standardized mean difference between leavers and stayers on what the model adjusts for
  (:func:`rows_smd`, Austin 2009), the Wasserstein shift of the exposure over its SD
  (:func:`exposure_shift`, ``consequences._shifts``), the change in correlation among the
  predictors (:func:`correlation_change`), a process column's imbalance over the outcome as design
  counts (:func:`design_imbalance`, legal only where the outcome beside another column is: under
  Predict on training rows, or after the lock), and a measure's excess over its chance share
  (:func:`excess_over_chance`). None of them reads the outcome beside another column before the
  lock (Nolan's ruling: under Estimate the outcome beside any other column is hidden until then).
* **Realized, after the lock** (``regime="realized"``; §2.3 items 3–6): the alternative refit as a
  labeled secondary, read from ``stages/sensitivity.changes_for`` (:func:`realized_from_sensitivity`),
  the calibrated coefficient beside the uncorrected one (:func:`realized_from_calibration`), the
  "further adjusted for" model (:func:`realized_from_secondary`), a Cinelli–Hazlett benchmark
  (:func:`realized_from_benchmarks`), the E-value of the limit nearer the null
  (:func:`realized_from_e_value`), and under Predict the paired in-fold difference
  (:func:`paired_folds`). :func:`ledger` reads none of them while an estimate is unseen
  (``consequences.estimates_unseen``): an unanswered purpose is the strictest case.

**Bands and calibration** (§2.4, §2.6). Each instrument has two thresholds in
``materiality_calibration.json`` (band 1 "could bias", band 2 "act on it"); the realized ones use
τ₀ and τ₁. A proxy is *calibrated* only with at least ``min_cases`` cases and at least one it put
below noise that stayed below noise after the lock; an uncalibrated proxy is floored at band 1 (the
cap of §2.6 read as a floor: it may say "could bias" or "act on it", never "doesn't change your
numbers here"). A movement that changes the question (substitution versus addition) is band 2 and
never graded by a number; one that cannot be measured on these rows (one recall day for λ) is band
1, "not measurable here". :func:`calibrate` chooses the thresholds from the cases (each labeled
with the pipeline it was run on) to leave no false reassurance, and reports the confusion matrix
per instrument. An instrument exact by theorem (:func:`invariance`: Frisch–Waugh–Lovell, a
reparameterization of a column that is not focal) is not floored: its 0 is a proof, so it needs no
cases; ``calibrate`` keeps its entry, which names the theorem, and each movement carries it
(§2.6 as amended, WAVE_C6A_PLAN §7 ruling 2).

**Structure among the adjustment terms** (K5, :func:`collinear_noticing`): Belsley's
variance-decomposition proportions of what you study and the adjustment set, outcome-blind. What
you study outside every near dependency is band 0 by theorem, For the record; inside one it changes
the question (band 2, decided at the adjustment question); an exact identity is a T1 blocker.

**The tier** (§1.3, :func:`tier`): a blocker is Decide; an item that needs a meaning with ``M > 0``
is Decide; one that changes the question is Decide; a default whose best alternative reaches
τ_confirm (band 1 of its instrument) is Confirm; otherwise For the record. The Confirm sweep
(``sweep.weigh``) reads :func:`would_change` wherever a movement is measured for its line, in place
of its hand test.

**The two-phase triage** (§2.4, §3.3): before the lock :func:`recommend` turns the predicted band
into a disposition with its reason; a limitation sentence is owed only when ``M ≥ τ₀`` and nothing
was done (:func:`limitation_owed`; a declared sensitivity analysis or a correction counts as done,
Nolan 2026-10-09). After the lock :func:`verify` compares the realized band with the predicted one:
an exhibit whose realized band exceeds the predicted one is relabeled, openly
(``LedgerRow.label``). The triage shows at most six rows, grouped by family beyond that
(:func:`triage_rows`).

**The ledger** (:func:`ledger`) is derived, never stored: one row per noticing with its predicted
movement, the recommendation, the disposition the triage recorded, what was done, the realized
movement and the verdict. It is a pure function of the answers, the data and the artifacts, so a
replay of the log reproduces it.

**The dietary proof** (§9, "The smallest end-to-end proof"): :func:`dietary_noticings` measures the
three noticings the policy names on a dietary table, each outcome-blind: energy carrying the
nutrient (r with total energy; it changes the question), implausible reporters (the screen's rows
times the leavers' largest SMD on the adjustment set), and day-to-day variance (``1 − λ`` from
replicate recalls, else "not measurable here" on one recall day).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

CALIBRATION_FILE = Path(__file__).resolve().parent / "materiality_calibration.json"

Regime = Literal["predicted", "realized"]
Disposition = Literal["no_change", "could_bias", "act_on_it"]
Verdict = Literal["pending", "confirmed", "upgraded", "downgraded", "not_verifiable", "not_graded"]
DecidesBy = Literal["meaning", "fixed_rule", "resampled_candidate", "disclosure"]
DISPOSITION_OF_BAND: dict[int, Disposition] = {0: "no_change", 1: "could_bias", 2: "act_on_it"}
BAND_WORDS = {0: "doesn't change your numbers here", 1: "could bias the estimate",
              2: "act on it"}
EXPECTED = {0: "leave the numbers as they are", 1: "possibly bias the estimate",
            2: "change the estimate"}
TRIAGE_ROWS = 6  #SURFACING_POLICY §4.1: triage sweep rows, grouped by family beyond it
FAMILY_NAMES = {
    "S5": "Who is in", "K3": "How well it measures", "K5": "Structure among variables",
    "S1": "Shortcuts", "S7": "Drift", "K1": "What a value means", "K2": "What a blank means",
    "K4": "Causal place", "E1": "Support", "E2": "Noise and multiplicity",
    "finding": "Other things noticed",
}
# BLUEPRINT §14 rule 1: a nutrient correlated with total energy at r ≥ 0.3 rides on it.
ENERGY_R = 0.3

# The instruments: (regime, the quiet technical label). A realized instrument is the truth the
# proxies are calibrated against; its thresholds are τ₀ and τ₁.
INSTRUMENTS: dict[str, tuple[Regime, str]] = {
    "rows_smd": ("predicted", "rows changed × largest SMD"),
    "exposure_shift": ("predicted", "W1 / SD of what you study"),
    "correlation_change": ("predicted", "|Δr| among predictors"),
    "attenuation": ("predicted", "attenuation 1 − λ"),
    "design_imbalance": ("predicted", "rank-biserial r over the outcome"),
    "excess_over_chance": ("predicted", "share beyond chance"),
    "changes_question": ("predicted", "changes the question"),
    # Exact by theorem (§2.6 as amended, WAVE_C6A_PLAN §7 ruling 2): an alternative that leaves the
    # model matrix's column space unchanged (a recoding of a column that is not focal, or adjustment
    # terms written another way) moves the focal estimate by exactly 0 (Frisch–Waugh–Lovell). It
    # needs no calibration cases and is never floored; its calibration entry names the theorem.
    "invariance": ("predicted", "exact by theorem"),
    "sensitivity": ("realized", "|Δβ| / half-width, refit on other rows"),
    "calibration": ("realized", "|Δβ| / half-width, calibrated"),
    "secondary": ("realized", "|Δβ| / half-width, further adjusted"),
    "benchmark": ("realized", "|Δβ| / half-width, Cinelli–Hazlett benchmark"),
    "e_value": ("realized", "E-value of the limit nearer the null"),
    "paired_folds": ("realized", "|mean Δscore| / SD of Δ by fold"),
}


# ── the calibration file ─────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def calibration() -> dict[str, Any]:
    """``materiality_calibration.json`` as committed."""
    return json.loads(CALIBRATION_FILE.read_text("utf-8"))


def thresholds(instrument: str) -> tuple[float, float, bool]:
    """(band 1, band 2, calibrated) for ``instrument``. A realized instrument reads τ₀ and τ₁ and is
    the reference, so it is never capped; a proxy with no case in the calibration set is
    uncalibrated."""
    cal = calibration()
    entry = cal["instruments"].get(instrument)
    if entry is None:
        return cal["tau_0"], cal["tau_1"], False
    return float(entry["band_1"]), float(entry["band_2"]), bool(entry["calibrated"])


def theorem_of(instrument: str) -> str | None:
    """The theorem an exact instrument rests on, as its calibration entry names it; None for a
    proxy that is not exact (it keeps the floor of §2.6)."""
    entry = calibration()["instruments"].get(instrument) or {}
    return entry.get("theorem") or None


def tau_confirm_band() -> int:
    """τ_confirm, as a band: a default whose best alternative reaches it is a Confirm."""
    return int(calibration()["tau_confirm_band"])


# ── a movement ───────────────────────────────────────────────────────────────


class Movement(BaseModel):
    """One measurement of ``M``: its instrument and regime, the value (None where it is not graded
    by a number), its band, whether the instrument is calibrated, and in plain words what was
    measured (``words``) with the technical name as a quiet label (``label``)."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    instrument: str
    regime: Regime
    value: float | None = None
    band: int
    calibrated: bool
    changes_question: bool = False
    not_measurable: bool = False
    crosses: bool = False  # a sign change, or an interval that crosses the null
    words: str
    label: str
    # The theorem an exact instrument rests on (``invariance``): the ledger names it.
    theorem: str | None = None


def band_of(instrument: str, value: float | None, *, changes_question: bool = False,
            not_measurable: bool = False, crosses: bool = False) -> int:
    """The band (0 below noise, 1 could bias, 2 act on it) of a movement (§2.4). Unknown is "could
    bias"; an uncalibrated proxy is floored at 1: never 0, and still 2 past its band-2 convention
    (§2.3: an attenuation 1 − λ ≈ 0.6 is act on it). An instrument exact by theorem is never
    floored (§2.6 as amended): its 0 is a proof, not a prediction."""
    if changes_question or crosses:
        return 2
    if not_measurable or value is None or not math.isfinite(value):
        return 1
    band_1, band_2, calibrated = thresholds(instrument)
    raw = 2 if value >= band_2 else 1 if value >= band_1 else 0
    regime = INSTRUMENTS.get(instrument, ("predicted", ""))[0]
    if regime == "predicted" and not calibrated and theorem_of(instrument) is None:
        return max(raw, 1)
    return raw


def movement(instrument: str, value: float | None, words: str, *, label: str | None = None,
             changes_question: bool = False, not_measurable: bool = False,
             crosses: bool = False) -> Movement:
    regime, name = INSTRUMENTS[instrument]
    _b1, _b2, calibrated = thresholds(instrument)
    shown = name if value is None else f"{name} = {value:.3f}"
    theorem = theorem_of(instrument)
    return Movement(instrument=instrument, regime=regime, value=value,
                    band=band_of(instrument, value, changes_question=changes_question,
                                 not_measurable=not_measurable, crosses=crosses),
                    calibrated=calibrated or regime == "realized" or theorem is not None,
                    changes_question=changes_question, not_measurable=not_measurable,
                    crosses=crosses, words=words, label=label or shown, theorem=theorem)


def invariance(words: str, *, label: str | None = None) -> Movement:
    """An alternative that leaves the focal estimate exactly where it is, by theorem: the model
    matrix spans the same space under it (a recoding of a column that is not focal, adjustment
    terms written another way), so the estimate, its interval and the fitted values do not move
    (Frisch–Waugh–Lovell). Band 0, never floored; the label names the theorem."""
    theorem = theorem_of("invariance") or "Frisch–Waugh–Lovell"
    return movement("invariance", 0.0, words, label=label or f"exact by theorem ({theorem})")


def disposition(m: Movement) -> Disposition:
    return DISPOSITION_OF_BAND[m.band]


def would_change(m: Movement) -> tuple[bool, str]:
    """The Confirm sweep's test where ``M`` is measured: another choice changes a number here when
    its movement reaches τ_confirm. An uncalibrated proxy is never below it (its band is floored
    at 1), so it never says the choice changes nothing."""
    return m.band >= tau_confirm_band(), m.words


# ── the predicted instruments (before the lock; outcome-blind) ───────────────


def _fmt(n: int) -> str:
    return f"`{int(n):,}`"


def _numeric(values: Any) -> np.ndarray | None:
    import pandas as pd

    x = pd.to_numeric(values, errors="coerce")
    if x.notna().sum() < max(2, int(0.5 * values.notna().sum())):
        return None
    return x.to_numpy(dtype=float)


def smd(values: Any, leaving: np.ndarray) -> float | None:
    """The largest absolute standardized mean difference between leavers and stayers (Austin
    2009): a number's difference of means over the pooled sample SD; a category's, per level,
    the difference of proportions over √((p₁(1 − p₁) + p₀(1 − p₀)) / 2). None when either side is
    empty or nothing varies."""
    import pandas as pd

    leaving = np.asarray(leaving, dtype=bool)
    x = _numeric(values)
    if x is not None:
        a, b = x[leaving], x[~leaving]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            return None
        sd = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        return abs(a.mean() - b.mean()) / sd if sd > 0 else None
    s = pd.Series(values).astype("string")
    known = s.notna().to_numpy()
    a, b = s[leaving & known], s[~leaving & known]
    if not len(a) or not len(b):
        return None
    best = None
    for level in pd.unique(s.dropna()):
        p1, p0 = float((a == level).mean()), float((b == level).mean())
        sd = math.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
        if sd > 0:
            d = abs(p1 - p0) / sd
            best = d if best is None or d > best else best
    return best


def rows_smd(frame: Any, leaving: Any, columns: Sequence[str], *, rule: str = "This rule"
             ) -> Movement:
    """Design-space movement in rows (§2.3 item 2): the share of rows the alternative changes,
    times the largest SMD between the rows that leave and the rows that stay on ``columns`` (what
    the model adjusts for; never the outcome). ``frame``: the rows the alternative keeps;
    ``leaving``: those the current answer removes from them."""
    leaving = np.asarray(leaving, dtype=bool)
    n, gone = int(len(leaving)), int(leaving.sum())
    share = gone / n if n else 0.0
    found = [(c, smd(frame[c], leaving)) for c in columns if c in frame.columns]
    found = [(c, d) for c, d in found if d is not None]
    column, worst = max(found, key=lambda cd: cd[1]) if found else (None, 0.0)
    value = share * worst
    words = (f"{rule} removes {_fmt(gone)} of {_fmt(n)} rows ({share:.1%})"
             + (f", who differ from the others by up to {worst:.2f} standard deviations in "
                f"`{column}`, which the model adjusts for." if column else "."))
    return movement("rows_smd", value, words,
                    label=f"rows × SMD = {share:.3f} × {worst:.3f} = {value:.3f}")


def exposure_shift(before: Any, after: Any, exposure: str) -> Movement:
    """Design-space movement in values: the exposure's Wasserstein shift over its SD
    (``consequences._shifts``)."""
    from turbotab.core.consequences import _shifts

    value = float(_shifts(before, after, [exposure])[exposure])
    return movement("exposure_shift", value,
                    f"Another choice shifts `{exposure}`'s values by {value:.2f} of their "
                    f"standard deviation.")


def correlation_change(before: Any, after: Any, exposure: str, others: Sequence[str]) -> Movement:
    """The largest change in the exposure's correlation with another predictor (|Δr|)."""
    import pandas as pd

    best, partner = 0.0, None
    for c in others:
        if c == exposure or c not in before.columns or c not in after.columns:
            continue
        r0 = pd.to_numeric(before[exposure], errors="coerce").corr(pd.to_numeric(before[c], errors="coerce"))
        r1 = pd.to_numeric(after[exposure], errors="coerce").corr(pd.to_numeric(after[c], errors="coerce"))
        if np.isfinite(r0) and np.isfinite(r1) and abs(r1 - r0) > best:
            best, partner = float(abs(r1 - r0)), c
    words = (f"Another choice changes how `{exposure}` moves with `{partner}` by {best:.2f} in "
             f"correlation." if partner else f"Another choice leaves `{exposure}`'s correlations "
                                              f"with the other predictors as they are.")
    return movement("correlation_change", best, words)


def reliability(values: Any, persons: Any) -> tuple[float, float, float, float] | None:
    """λ of a person's mean over their recalls, from a one-way random-effects decomposition
    (method of moments): (λ, σ²b, σ²w, k̄) with ``λ = σ²b / (σ²b + σ²w / k̄)``. None without two
    or more recalls for at least two people."""
    import pandas as pd

    frame = pd.DataFrame({"x": pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(),
                          "p": np.asarray(persons)}).dropna()
    sizes = frame.groupby("p")["x"].size()
    repeated = sizes[sizes >= 2]
    if len(repeated) < 2:
        return None
    frame = frame[frame["p"].isin(repeated.index)]
    groups = frame.groupby("p")["x"]
    n, g = len(frame), groups.ngroups
    means, counts = groups.mean(), groups.size()
    grand = frame["x"].mean()
    ss_w = float(((frame["x"] - frame["p"].map(means)) ** 2).sum())
    ss_b = float((counts * (means - grand) ** 2).sum())
    ms_w, ms_b = ss_w / (n - g), ss_b / (g - 1)
    n0 = (n - float((counts ** 2).sum()) / n) / (g - 1)
    var_b = max((ms_b - ms_w) / n0, 0.0)
    k = float(counts.mean())
    lam = var_b / (var_b + ms_w / k) if var_b + ms_w / k > 0 else 0.0
    return lam, var_b, ms_w, k


def attenuation(lam: float | None, column: str, *, days: int | None = None) -> Movement:
    """The theory bound of regression dilution (Hutcheon et al. 2010): with one error-prone
    exposure of reliability λ the slope is attenuated by ``1 − λ``. Without a second measure λ is
    not measurable here, and unknown is "could bias"."""
    if lam is None:
        recalls = "one recall day" if days in (None, 1) else f"{days} recall days averaged"
        return movement(
            "attenuation", None,
            f"How much of `{column}` is day-to-day noise is not measurable here: each person has "
            f"{recalls} and no second measure, so a slope could be weakened by it without a sign.",
            label="attenuation 1 − λ: not measurable (one measure per person)",
            not_measurable=True)
    value = 1.0 - lam
    return movement("attenuation", value,
                    f"About {value:.0%} of a slope on `{column}` would be lost to day-to-day "
                    f"noise (λ = {lam:.2f}).",
                    label=f"attenuation 1 − λ = {value:.3f}")


def outcome_beside_allowed(state: Any, pressed: bool | None = None) -> bool:
    """Whether the outcome may be read beside another column now: under Estimate (and an
    unanswered purpose, the strictest case) only after the lock (Nolan's ruling); under Predict
    on training rows."""
    if getattr(state, "purpose", None) == "prediction":
        return True
    return bool(getattr(state, "plan_locked", None))


def design_imbalance(state: Any, r: float, column: str, *, pressed: bool | None = None
                     ) -> Movement | None:
    """A process column's imbalance over the outcome, as design counts (O2: the rank-biserial r
    of run order or batch against the classes). It reads the outcome beside another column, so
    under Estimate it waits for the lock: None before it."""
    if not outcome_beside_allowed(state, pressed):
        return None
    value = abs(float(r))
    return movement("design_imbalance", value,
                    f"`{column}` is spread unevenly over the outcome (rank-biserial r = {r:.2f}).")


def excess_over_chance(observed: float, chance: float, what: str) -> Movement:
    """A detector's measure beyond its chance share (``detectors/assay.py``'s drift reading)."""
    value = max(float(observed) - float(chance), 0.0)
    return movement("excess_over_chance", value,
                    f"{observed:.0%} of {what}, against {chance:.1%} expected by chance.")


def changes_question(words: str, measure_label: str) -> Movement:
    """An alternative that changes what *Q* is (substitution versus addition): never graded by a
    number, always "act on it"."""
    return movement("changes_question", None, words, label=measure_label, changes_question=True)


# ── the realized instruments (after the lock) ────────────────────────────────


def refit(instrument: str, estimate: float, low: float, high: float, alternative: float,
          what: str, *, crosses_null: bool = False) -> Movement:
    """``|Q_alt − Q| / half-width`` of the primary's interval; a sign change, or (``crosses_null``)
    one interval excluding no effect while the other includes it, is band 2 (§2.4)."""
    half = (float(high) - float(low)) / 2
    value = abs(float(alternative) - float(estimate)) / half if half > 0 else math.inf
    sign = estimate != 0 and alternative != 0 and math.copysign(1, estimate) != math.copysign(1, alternative)
    return movement(instrument, value,
                    f"Checked after the plan was fixed: {what} moved the estimate from "
                    f"{estimate:.4g} to {alternative:.4g}, {value:.2f} of its interval's "
                    f"half-width.",
                    label=f"{INSTRUMENTS[instrument][1]} = {value:.3f}",
                    crosses=bool(sign or crosses_null))


def _exposure_features(features: Iterable[str], exposure: str | None) -> list[str]:
    features = list(features)
    if not exposure:
        return features
    mine = [f for f in features if f == exposure or f.startswith(f"{exposure}_")
            or f.startswith(f"{exposure} ") or f.startswith(f"{exposure}[")]
    return mine or features


def realized_from_sensitivity(artifact: Mapping[str, Any], exposure: str | None,
                              rules: Sequence[str] = ()) -> Movement | None:
    """The realized movement of an alternative row set: the primary's fit beside the analysis on
    the rows ``rules`` keep (their labels, ``stages.rows.rule_label``; none: Banna 2017's every-row
    analysis), through ``sensitivity.changes_for``; the largest over the exposure's model-matrix
    columns and the families. None when no analysis was fit on exactly those rows."""
    from turbotab.core.stages.sensitivity import changes_for

    analyses = list(artifact.get("analyses") or [])
    want = sorted(str(r) for r in rules)
    index = next((i for i, a in enumerate(analyses) if not a.get("primary")
                  and sorted(str(r) for r in a.get("rules") or []) == want
                  and not a.get("refused")), None)
    if index is None:
        return None
    best: Movement | None = None
    for family in artifact.get("families") or []:
        fits = list(family.get("fits") or [])
        if len(fits) <= index or not fits[0].get("coefficients"):
            continue
        pair = [fits[0], fits[index]]
        names = [c["feature"] for c in fits[0]["coefficients"]]
        for change in changes_for(pair, _exposure_features(names, exposure)):
            primary = next(c for c in fits[0]["coefficients"] if c["feature"] == change["feature"])
            if primary.get("ci_low") is None or primary.get("ci_high") is None:
                continue
            other = change["highest"] if change["highest"] != change["primary"] else change["lowest"]
            zero = change["excludes_zero"]
            flips = None not in zero and len(set(zero)) > 1
            m = refit("sensitivity", change["primary"], primary["ci_low"], primary["ci_high"],
                      other, f"refitting on {analyses[index]['label'].lower()}",
                      crosses_null=bool(change["sign_changes"] or flips))
            if best is None or (m.band, m.value or 0) > (best.band, best.value or 0):
                best = m
    return best


def realized_from_calibration(artifact: Mapping[str, Any], exposure: str | None) -> Movement | None:
    """The calibrated coefficient beside the uncorrected one (``stages/calibration.py``)."""
    if not artifact or not artifact.get("applies"):
        return None
    best = None
    for e in artifact.get("exposures") or []:
        if exposure and e.get("source") != exposure:
            continue
        if None in (e.get("naive"), e.get("naive_ci_low"), e.get("naive_ci_high"), e.get("estimate")):
            continue
        m = refit("calibration", e["naive"], e["naive_ci_low"], e["naive_ci_high"], e["estimate"],
                  "correcting for measurement error")
        if best is None or (m.band, m.value or 0) > (best.band, best.value or 0):
            best = m
    return best


def realized_from_secondary(artifact: Mapping[str, Any]) -> Movement | None:
    """The "further adjusted for" model beside the primary (``stages/secondary.py``)."""
    best = None
    for family in (artifact or {}).get("families") or []:
        fits = family.get("fits") or []
        if len(fits) < 2:
            continue
        for row in fits[0].get("coefficients") or []:
            other = next((c for c in fits[1].get("coefficients") or []
                          if c["feature"] == row["feature"]), None)
            if other is None or None in (row.get("ci_low"), row.get("ci_high"), other.get("estimate")):
                continue
            m = refit("secondary", row["estimate"], row["ci_low"], row["ci_high"], other["estimate"],
                      fits[1].get("label", "the further adjusted model").lower())
            if best is None or (m.band, m.value or 0) > (best.band, best.value or 0):
                best = m
    return best


def realized_from_benchmarks(robustness: Mapping[str, Any], half_width: float | None = None
                             ) -> Movement | None:
    """Cinelli & Hazlett (2020): an unmeasured confounder as strong as each adjusted covariate
    moves the estimate to ``benchmark.estimate``; the largest movement over the benchmarks, in
    half-widths of the primary's interval (``half_width``, else the classical t · SE)."""
    benchmarks = list((robustness or {}).get("benchmarks") or [])
    if not benchmarks:
        return None
    from scipy import stats

    estimate, se = float(robustness["estimate"]), float(robustness["se"])
    if half_width is None:
        half_width = float(stats.t.ppf(0.975, float(robustness["dof"]))) * se
    worst = max(benchmarks, key=lambda b: abs(float(b["estimate"]) - estimate))
    return refit("benchmark", estimate, estimate - half_width, estimate + half_width,
                 float(worst["estimate"]),
                 f"a common cause as strong as `{worst['covariate']}`")


def realized_from_e_value(e_value: Mapping[str, Any]) -> Movement | None:
    """The E-value of the interval's limit nearer the null (VanderWeele & Ding 2017): an interval
    that already includes the null is band 2; otherwise not graded as a movement and held at
    "could bias" (no convention grades an E-value; the band is not calibrated)."""
    if not e_value:
        return None
    limit = e_value.get("limit")
    includes = bool(e_value.get("interval_includes_null"))
    words = ("The interval already includes no effect, so any unmeasured confounding could move "
             "it either way." if includes else
             f"An unmeasured common cause would need a risk ratio of {float(limit or 1):.2f} with "
             f"both what you study and the outcome to move the interval to no effect.")
    m = movement("e_value", float(limit) if limit is not None else None, words, crosses=includes)
    return m if includes else m.model_copy(update={"band": 1})


def paired_folds(primary: Sequence[float], alternative: Sequence[float], what: str) -> Movement:
    """Under Predict, the alternative as a candidate in each fold: ``|mean Δ| / SD(Δ)`` paired by
    fold (``models/selection.interpretable_cost`` pairs the benchmark the same way)."""
    d = np.asarray(alternative, dtype=float) - np.asarray(primary, dtype=float)
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    value = abs(float(d.mean())) / sd if sd > 0 else (0.0 if not d.any() else math.inf)
    return movement("paired_folds", value,
                    f"{what} changed the cross-validated score by {d.mean():+.4f} on average, "
                    f"{value:.2f} of its fold-to-fold spread.")


# ── the tier (§1.3) ──────────────────────────────────────────────────────────


def tier(*, fires: bool = True, blocker: bool = False, decides_by: DecidesBy = "fixed_rule",
         m: Movement | None = None, changes_the_question: bool = False, has_default: bool = False,
         m_alt: Movement | None = None, settled_by_values: bool = False) -> str:
    """The label an item takes (SURFACING_POLICY §1.3): hidden, Decide, Confirm or For the record.
    An unmeasured ``M`` is the strictest case: it is taken to move a number."""
    from turbotab.core.quest import CONFIRM, DECIDE, RECORD

    moves = m is None or m.value is None or m.band > 0 or (m.value or 0) > 0
    if not fires:
        return "hidden"
    if blocker:
        return DECIDE
    if decides_by == "meaning" and moves:
        return DECIDE
    if changes_the_question or (m is not None and m.changes_question):
        return DECIDE
    if has_default and (m_alt is None or would_change(m_alt)[0]):
        return CONFIRM
    if has_default or settled_by_values:
        return RECORD
    return DECIDE


# ── the dietary noticings (the proof) ────────────────────────────────────────


class Noticing(BaseModel):
    """One noticing measured on this table: its thread and family, the stage that owns it, what
    it concerns, its measure, the predicted movement of the alternative, whether its own decision
    is answered, what was done about it, and the stage whose artifact verifies it after the lock."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    thread: str
    family: str
    stage: str
    subject: list[str]
    summary: str
    measure: float | None = None
    measure_label: str = ""
    decides_by: DecidesBy
    alternative: str
    predicted: Movement
    question: str | None = None  # the Router question that decides it
    answered: bool = False
    done: str | None = None
    verified_by: str | None = None
    # A T1 blocker (UNDERSTANDING_LAYER §2.2): part of the analysis cannot run as answered, so it
    # is resolved before any estimate (an exact identity among the model's columns).
    blocker: bool = False


def _energy_column(state: Any) -> str | None:
    roles = getattr(state, "roles", None) or {}
    spec = getattr(state, "energy_adjustment", None)
    if spec is not None and spec.energy_column:
        return spec.energy_column
    return next((c for c, r in roles.items() if r == "energy"), None)


def _exposure(state: Any) -> str | None:
    spec = getattr(state, "estimand", None)
    if spec is not None and spec.exposure:
        return spec.exposure
    roles = getattr(state, "roles", None) or {}
    return next((c for c, r in roles.items() if r == "exposure"), None)


def adjustment_set(state: Any) -> list[str]:
    """What the primary model adjusts for: its predictors, less the exposure and the outcome."""
    from turbotab.core.decisions import left_out
    from turbotab.core.models.pipeline import predictors_from_roles

    exposure = _exposure(state)
    return [c for c in predictors_from_roles(getattr(state, "roles", None), state.target,
                                             left_out(state)) if c != exposure]


def _place(thread: str) -> str:
    from turbotab.core.quest import noticing_stage

    return noticing_stage(thread)


def energy_noticing(state: Any, frame: Any) -> Noticing | None:
    """``diet-energy-carries-the-nutrient``: the exposure's correlation with total energy (O0,
    outcome-blind). Substitution or addition changes what the coefficient means, so it is never
    graded by a number; it is answered by the energy model or the contrast."""
    import pandas as pd

    energy, exposure = _energy_column(state), _exposure(state)
    if not energy or not exposure or exposure == energy or energy not in frame.columns \
            or exposure not in frame.columns:
        return None
    r = pd.to_numeric(frame[exposure], errors="coerce").corr(pd.to_numeric(frame[energy], errors="coerce"))
    if not np.isfinite(r) or abs(r) < ENERGY_R:
        return None
    estimand = getattr(state, "estimand", None)
    answered = getattr(state, "energy_adjustment", None) is not None or bool(
        estimand is not None and estimand.contrast)
    label = f"r({exposure}, {energy}) = {r:.2f}"
    return Noticing(
        thread="diet-energy-carries-the-nutrient", family="K5",
        stage=_place("diet-energy-carries-the-nutrient"), subject=[exposure, energy],
        summary=f"`{exposure}` rises and falls with total energy",
        measure=float(r), measure_label=label, decides_by="meaning",
        alternative="the other energy model (substitution or addition)",
        predicted=changes_question(
            f"`{exposure}` moves with `{energy}` at r = {r:.2f}: whether its coefficient means a "
            f"swap at the same energy or eating more changes what the estimate is.", label),
        question="energy_adjustment", answered=answered, verified_by=None)


def _screen(rule: Any, energy: str | None) -> bool:
    kind = getattr(rule, "kind", None)
    if kind == "goldberg":
        return True
    return bool(energy) and getattr(rule, "column", None) == energy and kind in ("range", "outside")


def _labels(rules: Iterable[Any]) -> list[str]:
    from turbotab.core.stages.rows import rule_label

    return sorted(rule_label(r) for r in rules)


def without_screens(state: Any) -> list[str]:
    """The labels of the primary's rules with the energy screens taken out: the rows the screen's
    prediction compares against, and so the analysis that verifies it."""
    energy = _energy_column(state)
    return _labels(r for r in getattr(state, "exclusions", None) or [] if not _screen(r, energy))


def _without_screen_declared(state: Any) -> str | None:
    """The label of a declared analysis on the rows the primary keeps without its energy screens
    (every row when the screen is the only rule), or None."""
    want = without_screens(state)
    return next((a.label for a in getattr(state, "sensitivity", None) or []
                 if _labels(a.rules) == want), None)


def reporters_noticing(state: Any, frame: Any, kept: Any, kept_without: Any) -> Noticing | None:
    """``diet-implausible-reporters``: the rows an energy screen removes from those the analysis
    would keep without it, times the leavers' largest SMD on what the model adjusts for (outcome-
    blind: never the outcome's values). ``kept``: the row ids the primary keeps; ``kept_without``:
    those it keeps with the screen taken out of the exclusions."""
    energy = _energy_column(state)
    screens = [r for r in getattr(state, "exclusions", None) or [] if _screen(r, energy)]
    if not screens:
        return None
    base = np.asarray(kept_without, dtype=np.int64)
    leaving = ~np.isin(base, np.asarray(kept, dtype=np.int64))
    rows = frame.loc[base]
    m = rows_smd(rows, leaving, adjustment_set(state), rule="The energy screen")
    declared = _without_screen_declared(state)
    if declared is None:
        done = None
    elif not without_screens(state):
        done = "The every-row analysis is declared beside the primary (Banna et al. 2017)."
    else:
        done = (f"An analysis on the rows the other rules keep without the screen "
                f"(\"{declared}\") is declared beside the primary.")
    return Noticing(
        thread="diet-implausible-reporters", family="S5",
        stage=_place("diet-implausible-reporters"), subject=[energy or ""],
        summary="Some energy reports are implausible, and the screen removes them",
        measure=float(leaving.mean()) if len(leaving) else 0.0,
        measure_label=f"{int(leaving.sum()):,} of {len(leaving):,} rows", decides_by="meaning",
        alternative="keeping every row", predicted=m, question="exclusions", answered=True,
        done=done, verified_by="sensitivity")


def variance_noticing(state: Any, frame: Any) -> Noticing | None:
    """``diet-day-to-day-variance``: λ of the exposure from replicate recalls (a person's rows,
    when repeats are recorded as replicates); on one recall day it is not measurable here."""
    exposure = _exposure(state)
    if not exposure or exposure not in frame.columns:
        return None
    energy = _energy_column(state)
    units = getattr(state, "column_units", None) or {}
    # The exposure's own recall days; the energy column's only where the exposure's are not
    # recorded (a dietary table's nutrients come from the same recalls).
    spec = units.get(exposure) or (units.get(energy) if energy else None)
    days = getattr(spec, "days", None) if spec is not None else None
    repeat = getattr(state, "repeat_kind", None)
    grain = getattr(state, "grain", None)
    person = getattr(grain, "id_column", None) if grain is not None else None
    found = None
    if repeat is not None and getattr(repeat, "repeat_kind", None) == "repeats" and person \
            and person in frame.columns:
        found = reliability(frame[exposure], frame[person])
    lam = found[0] if found is not None else None
    m = attenuation(lam, exposure, days=days)
    answer = getattr(state, "measurement_error", None)
    corrected = getattr(answer, "method", None)
    done = ("Regression calibration is declared for it." if corrected == "regression_calibration"
            else None)
    return Noticing(
        thread="diet-day-to-day-variance", family="K3",
        stage=_place("diet-day-to-day-variance"), subject=[exposure],
        summary=f"Part of `{exposure}`'s spread is day-to-day noise",
        measure=lam, measure_label=m.label, decides_by="meaning",
        alternative="correcting for day-to-day noise", predicted=m, question=None,
        answered=answer is not None, done=done, verified_by="calibration")


# ── structure among the adjustment terms (K5) ────────────────────────────────

COLLINEAR_THREAD = "shared-collinear-predictors"
# Belsley, Kuh & Welsch (1980, §3.3): a condition index of 30 or more marks a moderate to strong
# near dependency; a column whose variance-decomposition proportion on it is 0.5 or more is in it.
CONDITION_INDEX = 30.0
IN_DEPENDENCY = 0.5
CATEGORY_LEVELS = 12  # a text column with more levels than this is not entered as indicators


def _model_columns(frame: Any, columns: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """The model matrix's columns for ``columns`` (no intercept) and the column each came from:
    a number as it is, a text column of 2 to :data:`CATEGORY_LEVELS` levels as one indicator per
    level after the first (sorted), anything else left out. Rows with a blank are dropped."""
    import pandas as pd

    parts: list[np.ndarray] = []
    owners: list[str] = []
    for c in columns:
        if c not in frame.columns:
            continue
        s = frame[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            parts.append(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float))
            owners.append(c)
            continue
        levels = sorted(str(v) for v in s.dropna().unique())
        if not 2 <= len(levels) <= CATEGORY_LEVELS:
            continue
        text = s.astype(object)
        blank = s.isna().to_numpy()
        for level in levels[1:]:
            x = (text.astype(str) == level).to_numpy(dtype=float)
            x[blank] = np.nan
            parts.append(x)
            owners.append(c)
    if not parts:
        return np.empty((len(frame), 0)), []
    X = np.column_stack(parts)
    return X[np.isfinite(X).all(axis=1)], owners


def belsley(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Belsley's diagnostics of ``[1, X]``, as ``models.linear.collinearity_concern`` computes them:
    the intercept and every column scaled to unit length, its SVD ``U S Vᵀ``; the condition
    indexes ``η_k = s_max / s_k``; the variance-decomposition proportions
    ``π_jk = (v_jk² / s_k²) / Σ_k (v_jk² / s_k²)`` (row 0 the intercept); the singular values; and
    ``Vᵀ`` (a row whose singular value is 0 is an exact identity among the columns)."""
    A = np.column_stack([np.ones(len(X)), X])
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1.0
    _, s, vt = np.linalg.svd(A / norms, full_matrices=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = (vt.T ** 2) / np.where(s > 0, s, np.finfo(float).tiny) ** 2
        pi = phi / phi.sum(axis=1, keepdims=True)
        eta = s[0] / np.where(s > 0, s, np.finfo(float).tiny)
    return eta, pi, s, vt


def _in_order(columns: Iterable[str], order: Sequence[str]) -> list[str]:
    found = set(columns)
    return [c for c in dict.fromkeys(order) if c in found]


def collinear_noticing(state: Any, frame: Any) -> Noticing | None:
    """``shared-collinear-predictors`` on the adjustment set (K5; with ``diet-nested-parts`` and
    ``shared-compositional-parts``, "a total beside its parts"): a near dependency among what the
    primary model adjusts for, and where what you study sits in it. Outcome-blind: it reads what
    you study and the adjustment set, never the outcome. None under Predict (no focal estimate),
    or with no dependency among two or more columns.

    * An exact identity (the model matrix is rank deficient, e.g. kcal = 4P + 4C + 9F + 7A
      exactly) is a T1 blocker: the model cannot separate those columns.
    * What you study in the dependency (its proportions summed over the near dependencies, at
      least 0.5): which of them are held fixed changes what the estimate means (band 2), decided
      at the adjustment question.
    * Otherwise band 0 by theorem: the estimate and its interval do not depend on how the
      adjustment terms are written (Frisch–Waugh–Lovell), so it is disclosed For the record, the
      others' coefficients labeled as adjustment terms."""
    if getattr(state, "purpose", None) == "prediction":
        return None
    exposure = _exposure(state)
    if not exposure or exposure not in frame.columns:
        return None
    adjusted = [c for c in adjustment_set(state) if c in frame.columns and c != exposure]
    if not adjusted:
        return None
    X, owners = _model_columns(frame, [exposure, *adjusted])
    if exposure not in owners or len(X) <= X.shape[1] + 1:
        return None
    eta, pi, s, vt = belsley(X)
    names = ["", *owners]  # row 0 is the intercept
    order = [exposure, *adjusted]
    # Numerically rank deficient, as numpy's ``matrix_rank`` reads it.
    tol = s[0] * max(X.shape[0], X.shape[1] + 1) * np.finfo(float).eps
    exact: set[str] = set()
    for k in np.flatnonzero(s <= tol):
        members = {names[j] for j in range(1, len(names)) if abs(vt[k, j]) > 1e-6}
        if len(members) >= 2:
            exact |= members
    if exact:
        subject = _in_order(exact, order)
        cols = _listing(subject)
        words = (f"{cols} add up exactly: one is a fixed sum of the others, so the model cannot "
                 f"tell them apart. Leave one of them out before any estimate.")
        label = "the model matrix is rank deficient (an exact linear identity)"
        return Noticing(
            thread=COLLINEAR_THREAD, family="K5", stage=_place(COLLINEAR_THREAD), subject=subject,
            summary=f"{cols} add up exactly", measure=None, measure_label=label,
            decides_by="meaning", alternative="leaving one of them out",
            predicted=Movement(instrument="changes_question", regime="predicted", value=None,
                               band=2, calibrated=True, changes_question=True, words=words,
                               label=label),
            question="adjustment", answered=False, blocker=True)
    near = [k for k in range(len(s)) if eta[k] >= CONDITION_INDEX
            and len({names[j] for j in range(1, len(names)) if pi[j, k] >= IN_DEPENDENCY}) >= 2]
    if not near:
        return None
    involved = {names[j] for k in near for j in range(1, len(names)) if pi[j, k] >= IN_DEPENDENCY}
    rows = [j for j in range(1, len(names)) if names[j] == exposure]
    share = float(max(sum(pi[j, k] for k in near) for j in rows))
    worst = float(max(eta[k] for k in near))
    partners = _in_order(involved - {exposure}, order)
    others = _listing(partners)
    label = (f"Belsley proportion of {exposure} = {share:.2f}, condition index {worst:.0f}")
    if share >= IN_DEPENDENCY:
        words = (f"`{exposure}` is close to a fixed mix of {others}: holding them fixed, more "
                 f"`{exposure}` means less of what it replaces, so which of them stay in the model "
                 f"changes what the estimate means. It is decided at the adjustment-set question.")
        return Noticing(
            thread=COLLINEAR_THREAD, family="K5", stage=_place(COLLINEAR_THREAD),
            subject=[exposure, *partners], summary=f"`{exposure}` moves almost in step with {others}",
            measure=share, measure_label=label, decides_by="meaning",
            alternative="another choice of which of them are held fixed",
            predicted=changes_question(words, label), question="adjustment",
            answered=_adjustment_answered(state, exposure, partners))
    words = (f"{others} are close to a fixed mix of one another, and `{exposure}` is not part of "
             f"it. Its estimate and interval are the same however they are written; their own "
             f"coefficients are adjustment terms, not effects.")
    theorem = theorem_of("invariance") or "Frisch–Waugh–Lovell"
    return Noticing(
        thread=COLLINEAR_THREAD, family="K5", stage=_place(COLLINEAR_THREAD),
        subject=[exposure, *partners], summary=f"{others} move almost in step",
        measure=share, measure_label=label, decides_by="disclosure",
        alternative="writing those adjustment terms another way (grouped, or one as the rest of "
                    "the others)",
        predicted=invariance(words, label=f"{label}; exact by theorem ({theorem})"))


def _listing(columns: Sequence[str]) -> str:
    from turbotab.core.voice import listing

    return listing(list(columns), limit=4)


def _adjustment_answered(state: Any, exposure: str, columns: Sequence[str]) -> bool:
    """Every partner in the dependency answered at the adjustment question for what you study."""
    answers = getattr(state, "adjustment", None) or {}
    return bool(columns) and all(
        c in answers and getattr(answers[c], "exposure", None) == exposure for c in columns)


def noticing_tier(n: Noticing) -> str:
    """The label a measured noticing takes (§1.3, :func:`tier`): a blocker or one that changes the
    question is Decide; one decided by disclosure whose alternative changes nothing here is For
    the record; one that needs a meaning is Decide."""
    disclosed = n.decides_by == "disclosure"
    return tier(blocker=n.blocker, decides_by=n.decides_by, m=n.predicted,
                has_default=disclosed, m_alt=n.predicted if disclosed else None)


def dietary_noticings(state: Any, frame: Any, kept: Any = None, kept_without: Any = None
                      ) -> list[Noticing]:
    """The three noticings of the policy's proof, as they fire on this table and these answers."""
    if "dietary" not in (getattr(state, "lens", None) or ()):
        return []
    out = [energy_noticing(state, frame)]
    if kept is not None and kept_without is not None:
        out.append(reporters_noticing(state, frame, kept, kept_without))
    out.append(variance_noticing(state, frame))
    return [n for n in out if n is not None]


def noticings_for(state: Any, store: Any, ingest: Mapping[str, Any]) -> list[Noticing]:
    """The noticings measured on the project's table: the cohort as the answers keep it, and as
    they would keep it without the energy screen (``stages.rows.compute_cohort``)."""
    if "dietary" not in (getattr(state, "lens", None) or ()):
        return []
    from turbotab.core.stages.rows import compute_cohort

    energy = _energy_column(state)
    rules = list(getattr(state, "exclusions", None) or [])
    kept = kept_without = None
    if any(_screen(r, energy) for r in rules):
        _, kept, _ = compute_cohort(store, state, ingest)
        without = state.model_copy(update={"exclusions": [r for r in rules if not _screen(r, energy)]})
        _, kept_without, _ = compute_cohort(store, without, ingest)
    exposure = _exposure(state)
    wanted = {c for c in (*adjustment_set(state), exposure, energy) if c}
    grain = getattr(state, "grain", None)
    if grain is not None and grain.id_column:
        wanted.add(grain.id_column)
    columns = [c["name"] for c in ingest.get("columns") or [] if c.get("name") in wanted]
    frame = store.materialize(columns)
    out = dietary_noticings(state, frame, kept, kept_without)
    # Every row in the file, before Who's in (the crosswalk's display gate for K5).
    collinear = collinear_noticing(state, frame)
    return out + [collinear] if collinear is not None else out


# ── the two-phase triage ─────────────────────────────────────────────────────


def finding_done(state: Any, columns: Iterable[str]) -> str | None:
    """What was done about a finding on ``columns``: a declared sensitivity analysis whose rules
    read one of them counts (Nolan, 2026-10-09), else None."""
    wanted = {str(c) for c in columns}
    for a in getattr(state, "sensitivity", None) or []:
        read = {c for r in a.rules for c in r.reads()}
        if read & wanted:
            return f"The sensitivity analysis \"{a.label}\" is declared beside the primary."
    return None


def limitation_owed(band: int, done: str | None) -> bool:
    """A limitation sentence only when ``M ≥ τ₀`` and nothing was done (Nolan, 2026-10-09)."""
    return band >= 1 and not done


def recommend(n: Noticing, *, purpose: str | None = None) -> tuple[Disposition, str, bool]:
    """Before the lock: the recommended disposition, its reason in plain words, and whether a
    limitation sentence is owed. An act-on-it noticing whose decision is answered has dropped
    out of the triage (:func:`in_triage`)."""
    m = n.predicted
    what = "score" if purpose == "prediction" else "estimate"
    owed = limitation_owed(m.band, n.done)
    if n.blocker:
        return ("act_on_it", f"{m.words} It blocks part of the analysis, so it is resolved before "
                             f"any {what}.", False)
    if m.band == 2:
        return "act_on_it", f"{m.words} It is decided on its own card.", False
    if m.band == 0 and m.theorem:
        return ("no_change", f"{m.words} That holds exactly ({m.theorem}), so there is nothing to "
                             f"check after the plan is fixed.", False)
    if m.band == 0:
        return ("no_change", f"{m.words} That is below what would move the {what}; it is checked "
                             f"again after the plan is fixed.", False)
    if n.done:
        return ("could_bias", f"{m.words} {n.done} Its result goes in the supplement.", False)
    if m.not_measurable:
        return ("could_bias", f"{m.words} Nothing here can check it, so a limitation sentence is "
                              f"drafted.", owed)
    return ("could_bias", f"{m.words} Left as it is, it could bias the {what}, so a limitation "
                          f"sentence is drafted unless a sensitivity analysis is declared.", owed)


def in_triage(n: Noticing) -> bool:
    """An act-on-it noticing whose decision is answered drops out; the rest stay (a blocker
    always: it is resolved only by the change that stops it firing)."""
    return n.blocker or not (n.predicted.band == 2 and n.answered)


class TriageRow(BaseModel):
    """One row of the triage sweep: a noticing on its own, or a family of them with a count."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    family: str
    name: str
    items: list[str]
    count: int


def triage_rows(items: Sequence[tuple[str, str]], cap: int = TRIAGE_ROWS) -> list[TriageRow]:
    """At most ``cap`` rows (§4.1): one per item while they fit, else one per family (in the order
    the families first appear), the families beyond the cap merged into the last row. Every item
    is in some row: nothing is hidden. ``items``: (id, family), in rank order."""
    if len(items) <= cap:
        return [TriageRow(family=f, name=FAMILY_NAMES.get(f, f), items=[i], count=1)
                for i, f in items]
    families: dict[str, list[str]] = {}
    for i, f in items:
        families.setdefault(f, []).append(i)
    rows = [TriageRow(family=f, name=FAMILY_NAMES.get(f, f), items=ids, count=len(ids))
            for f, ids in families.items()]
    if len(rows) > cap:
        rest = rows[cap - 1:]
        ids = [i for r in rest for i in r.items]
        rows = rows[:cap - 1] + [TriageRow(family="more", name=f"{len(rest)} more families",
                                           items=ids, count=len(ids))]
    return rows


# ── verification and the ledger ──────────────────────────────────────────────


def verify(predicted: Movement, realized: Movement | None, *, locked: bool
           ) -> tuple[Verdict, str | None]:
    """After the lock: the realized band against the predicted one, and the exhibit's label when
    the realized band is higher (an open relabel, never a silent edit). A movement exact by
    theorem is not graded either: there is nothing a refit could find."""
    if predicted.changes_question or predicted.theorem:
        return "not_graded", None
    if not locked:
        return "pending", None
    if realized is None:
        return "not_verifiable", None
    if realized.band > predicted.band:
        return "upgraded", (f"Moved more than predicted. {realized.words} Before the plan was "
                            f"fixed it was expected to {EXPECTED[predicted.band]}.")
    if realized.band < predicted.band:
        return "downgraded", None
    return "confirmed", None


class LedgerRow(BaseModel):
    """One row of the materiality ledger (§2.2)."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    thread: str
    family: str
    subject: list[str]
    alternative: str
    predicted: Movement
    recommended: Disposition | None = None  # None: answered on its own card, not triaged
    reason: str | None = None
    recorded: Disposition | None = None
    done: str | None = None
    limitation: bool = False
    realized: Movement | None = None
    verdict: Verdict
    exhibit: str | None = None  # the stage whose exhibit carries the verification
    label: str | None = None  # the exhibit's relabel when the realized band is higher
    sentence: str  # the supplement line, the limitation draft, or the decision it pointed to


class Ledger(BaseModel):
    """The materiality ledger: derived from the record, the data and the artifacts, never kept."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    version: int = 1
    locked: bool
    gate: str
    rows: list[LedgerRow] = []
    limitations: int = 0
    budget: int


def _recorded(state: Any) -> dict[str, Disposition]:
    from turbotab.core.decisions import sweep_key

    stage = "results" if getattr(state, "purpose", None) == "prediction" else "models"
    held = (getattr(state, "sweeps", None) or {}).get(sweep_key(stage, "noticings"))
    return {l.key: l.value for l in held.lines} if held is not None else {}  # type: ignore[misc]


def _realized(n: Noticing, state: Any, artifacts: Mapping[str, Any]) -> Movement | None:
    found = artifacts.get(n.verified_by) if n.verified_by else None
    found = getattr(found, "data", found)
    if not found:
        return None
    exposure = _exposure(state)
    if n.verified_by == "sensitivity":
        # The same alternative the prediction measured: the screen taken out, the other rules kept.
        return realized_from_sensitivity(found, exposure, without_screens(state))
    if n.verified_by == "calibration":
        return realized_from_calibration(found, exposure)
    if n.verified_by == "secondary":
        return realized_from_secondary(found)
    return None


def _sentence(n: Noticing, recorded: Disposition | None, realized: Movement | None,
              limitation: bool, verdict: Verdict) -> str:
    if recorded == "act_on_it" or (recorded is None and n.predicted.band == 2):
        return f"Decided on its own card: {n.alternative} was weighed there."
    if verdict in ("confirmed", "downgraded", "upgraded") and realized is not None:
        checked = realized.words
    else:
        checked = None
    if limitation:
        base = f"Limitation: {n.predicted.words}"
        return f"{base} {checked}" if checked else base
    if checked:
        return f"{n.predicted.words} {checked}"
    return n.predicted.words


def ledger(state: Any, noticings: Sequence[Noticing], artifacts: Mapping[str, Any] | None = None,
           *, pressed: bool | None = None, budget: int | None = None) -> Ledger:
    """Every noticing's row: predicted movement, recommendation, recorded disposition, what was
    done, and after the lock its realized movement and verdict. No realized instrument is read
    while an estimate is unseen (``consequences.estimates_unseen``)."""
    from turbotab.core.consequences import estimates_unseen

    artifacts = artifacts or {}
    locked = not estimates_unseen(state, pressed)
    recorded = _recorded(state)
    purpose = getattr(state, "purpose", None)
    cal = calibration()
    budget = int(cal["limitation_budget"] if budget is None else budget)
    rows: list[LedgerRow] = []
    for n in noticings:
        triaged = in_triage(n)
        rec, reason, _owed = recommend(n, purpose=purpose) if triaged else (None, None, False)
        held = recorded.get(n.thread) if triaged else None
        realized = _realized(n, state, artifacts) if locked else None
        verdict, label = verify(n.predicted, realized, locked=locked)
        disposed = held or rec
        limitation = bool(disposed == "could_bias" and limitation_owed(n.predicted.band, n.done))
        if verdict == "upgraded" and not n.done:
            limitation = True  # an upgrade is a label on the exhibit plus a limitation draft
        rows.append(LedgerRow(
            thread=n.thread, family=n.family, subject=n.subject, alternative=n.alternative,
            predicted=n.predicted, recommended=rec, reason=reason, recorded=held, done=n.done,
            limitation=limitation, realized=realized, verdict=verdict,
            exhibit=n.verified_by if realized is not None else None, label=label,
            sentence=_sentence(n, held or rec, realized, limitation, verdict)))
    # The budget on limitation sentences (§4.1): beyond it, a supplement line with its numbers.
    owed = [r for r in rows if r.limitation]
    for r in owed[budget:]:
        r.limitation = False
        r.sentence = r.sentence.removeprefix("Limitation: ")
    from turbotab.core.sweep import GATES, gate_stage

    return Ledger(locked=locked, gate=GATES[gate_stage(state)], rows=rows,
                  limitations=sum(r.limitation for r in rows), budget=budget)


# ── calibration on the reference journeys (§2.6) ─────────────────────────────


class Case(BaseModel):
    """One calibration case: a predicted and a realized movement of the same alternative."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    journey: str
    pipeline: str  # what was run: the capture as recorded, or a stated simplification of it
    thread: str
    alternative: str
    instrument: str
    m_pre: float
    instrument_post: str
    m_post: float
    band_post: int


def cases_from(journey: str, rows: Iterable[LedgerRow], *, pipeline: str,
               alternative: str | None = None) -> list[Case]:
    """The ledger's rows that calibrate a proxy: a numeric prediction and a realized movement.
    ``pipeline``: what was run, said plainly; ``alternative``: the alternative, where the row's own
    words do not name it (one rule of several tried)."""
    out = []
    for r in rows:
        p, q = r.predicted, r.realized
        if q is None or p.value is None or q.value is None or p.regime != "predicted":
            continue
        out.append(Case(journey=journey, pipeline=pipeline, thread=r.thread,
                        alternative=alternative or r.alternative, instrument=p.instrument,
                        m_pre=round(p.value, 6), instrument_post=q.instrument,
                        m_post=round(q.value, 6), band_post=q.band))
    return out


def _raw_band(value: float, band_1: float, band_2: float) -> int:
    return 2 if value >= band_2 else 1 if value >= band_1 else 0


def calibrate(cases: Sequence[Case], start: Mapping[str, Any]) -> dict[str, Any]:
    """The calibration file from ``start`` (the conventions and their reasons) and the cases.

    Per proxy instrument: a threshold that would leave a case falsely reassured (predicted band 0,
    realized band ≥ 1) is lowered to that case's predicted value, so no case is; the confusion
    matrix (predicted band × realized band) is reported, with the false reassurances and the
    cry-wolf cases (predicted ≥ 1, realized 0). An instrument is calibrated, and so may say
    "doesn't change your numbers here", only with at least ``min_cases`` cases and at least one
    case it put below noise that stayed below noise (a band 0 that held); short of that it is
    uncalibrated and floored at band 1, its thresholds still reported."""
    out = json.loads(json.dumps(dict(start)))
    min_cases = int(out.get("min_cases", 5))
    by: dict[str, list[Case]] = {}
    for c in cases:
        by.setdefault(c.instrument, []).append(c)
    for name, entry in out["instruments"].items():
        mine = by.get(name, [])
        if entry.get("theorem"):
            continue  # exact by theorem (§2.6 as amended): kept as it is, never floored
        if entry["regime"] == "realized":
            entry["band_1"], entry["band_2"] = out["tau_0"], out["tau_1"]
            entry["calibrated"] = True
            continue
        b1, b2 = float(entry["convention"]["band_1"]), float(entry["convention"]["band_2"])
        entry["band_1"], entry["band_2"] = b1, b2
        entry["reason"] = entry["convention"]["reason"]
        missed = [c.m_pre for c in mine if _raw_band(c.m_pre, b1, b2) == 0 and c.band_post >= 1]
        if missed:
            b1 = round(min(missed), 6)
            entry["band_1"] = b1
            entry["reason"] = (entry.get("reason", "") + f" Lowered to {b1} so that no case is "
                               f"predicted below noise and realized above it.").strip()
        matrix = [[0, 0, 0] for _ in range(3)]
        for c in mine:
            matrix[_raw_band(c.m_pre, b1, b2)][c.band_post] += 1
        held = matrix[0][0]
        entry["calibrated"] = len(mine) >= min_cases and held > 0
        entry["cases"] = len(mine)
        if mine and not entry["calibrated"]:
            entry["uncalibrated"] = (
                f"{len(mine)} case(s), fewer than {min_cases}" if len(mine) < min_cases else
                "no case it put below noise stayed below noise after the lock")
        else:
            entry.pop("uncalibrated", None)
        entry["confusion"] = {"rows": "predicted band 0, 1, 2", "columns": "realized band 0, 1, 2",
                              "matrix": matrix,
                              "false_reassurance": sum(matrix[0][1:]),
                              "cry_wolf": matrix[1][0] + matrix[2][0]}
    out["cases"] = [c.model_dump(mode="json") for c in cases]
    return out


def write_calibration(data: Mapping[str, Any], path: Path = CALIBRATION_FILE) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
    calibration.cache_clear()


__all__ = [
    "BAND_WORDS", "CALIBRATION_FILE", "COLLINEAR_THREAD", "Case", "DISPOSITION_OF_BAND",
    "FAMILY_NAMES", "INSTRUMENTS",
    "Ledger", "LedgerRow", "Movement", "Noticing", "TRIAGE_ROWS", "TriageRow", "adjustment_set",
    "attenuation", "band_of", "belsley", "calibrate", "calibration", "cases_from",
    "changes_question", "collinear_noticing",
    "correlation_change", "design_imbalance", "dietary_noticings", "disposition",
    "energy_noticing", "excess_over_chance", "exposure_shift", "finding_done", "in_triage",
    "invariance", "ledger",
    "limitation_owed", "movement", "noticing_tier", "noticings_for", "outcome_beside_allowed",
    "paired_folds", "theorem_of",
    "realized_from_benchmarks", "realized_from_calibration", "realized_from_e_value",
    "realized_from_secondary", "realized_from_sensitivity", "recommend", "refit", "reliability",
    "reporters_noticing", "rows_smd", "smd", "thresholds", "tier", "triage_rows",
    "variance_noticing", "verify", "without_screens", "would_change", "write_calibration",
]
