"""The ``usual_intake`` stage: the NCI usual-intake estimand, offered and estimated.

Its artifact carries the **offer** (``turbotab.core.usual_intake``: whether the usual-intake
distribution is offered as its own estimand, why not when it is not, and each candidate dietary
component with its share of zero recalls and the model ranked first) and one **analysis** per
recorded ``set_usual_intake`` answer (``turbotab.core.methods.usual_intake`` fits it).

**The recalls.** In a long table they are the rows each person repeats: the oriented table's rows
behind each working row (the row-local repairs applied, as the calibration stage reads them) when
the rows were combined per person, or the working rows themselves grouped by the grain's identifier
when the records stayed as rows. In a wide table (one row per person) they are the day columns the
answer names, in the order named. A recall with no value for the component is no recall of it.

**The rows** are every eligible row (BLUEPRINT §12 ruling 3): the recorded eligibility rules and
repair rules apply, and no outcome or predictor's missingness removes anyone, since the estimand
reads no outcome. The survey design is read over every row of the working table, so a domain keeps
its strata and PSUs.

**The sequence.** A recall is a repeat recall when it comes after the person's first, by the order
column (a long table) or the day columns' order (a wide table): a long table's person whose first
recall is missing has their second read as their first.

**What it reads** (BLUEPRINT §14.1): the user's answers (the component, its days, the order and
weekend columns, the column marking consumers, the survey answer); the grain; the repeat kind and
the time column only as the ledger holds them settled. No unit is stated: a sentence quotes the
column's header, and a cut-off is the number the user gave, in that column's units.
"""
from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbotab.core.graph import Bundle, StageContext


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class UsualIntakeCandidate(_Model):
    column: str  # the component (a long table's column, or a wide table's first day)
    days: list[str] = []  # a wide table's recall-day columns, as proposed from their names
    zero_share: float | None = None
    n_persons: int = 0
    n_repeat: int = 0  # people with two or more recalls of it
    suggested: Literal["amount_only", "two_part"] | None = None
    why: str | None = None


class UsualIntakeOffer(_Model):
    offered: bool
    reason: str | None = None  # why it is not offered
    estimand: str
    association: str
    population_question: str
    format: Literal["long", "wide"] | None = None
    recalls: dict[str, int] = {}  # recalls per person -> people (a long table)
    n_persons: int = 0
    n_repeat: int = 0
    candidates: list[UsualIntakeCandidate] = []


class UsualIntakeEstimate(_Model):
    value: float
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


class UsualIntakeVariance(_Model):
    method: Literal["bootstrap", "brr", "psu_bootstrap", "none"]
    replicates: int = 0
    ok: int = 0
    df: int | None = None
    fay: float | None = None
    n_strata: int | None = None
    n_psu: int | None = None


class UsualIntakeAnalysis(_Model):
    nutrient: str
    model: Literal["none", "amount_only", "two_part"]
    applies: bool
    refused: str | None = None
    format: Literal["long", "wide"] | None = None
    days: list[str] = []
    population: Literal["whole", "consumers"] = "whole"
    population_statement: str | None = None  # STROBE-nut nut-14
    n_persons: int = 0
    n_recalls: int = 0
    recalls: dict[str, int] = {}
    n_repeat: int = 0
    zero_share: float | None = None
    zeros_replaced: int = 0
    covariates: list[str] = []  # the nuisance covariates modeled
    parameters: dict[str, float] = {}
    percentiles: dict[str, UsualIntakeEstimate] = {}
    mean: UsualIntakeEstimate | None = None
    cutoff: float | None = None
    cutoff_kind: Literal["EAR", "AI", "UL", "other"] | None = None
    share: UsualIntakeEstimate | None = None
    share_label: str | None = None
    day_one: dict[str, float] = {}  # single-day percentiles: not usual intake
    mean_of_days: dict[str, float] = {}  # each person's mean of recalls: not usual intake
    variance: UsualIntakeVariance | None = None
    weight: str | None = None
    assumptions: list[str] = []
    concerns: list[str] = []
    methods: str


class UsualIntakeArtifact(_Model):
    offer: UsualIntakeOffer
    analyses: list[UsualIntakeAnalysis] = []


class Refused(Exception):
    """An analysis the recorded answer cannot run, with the reason the artifact states."""


# ── where the recalls are ────────────────────────────────────────────────────


def gate(state: Any, structure: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    """``(format, None)`` where usual intake is offered, else ``(None, reason)``."""
    from turbotab.core import usual_intake as ui
    from turbotab.core.stages.working import effective_grain, effective_repeat_kind

    if state.lens is None:
        return None, ui.LENS_UNANSWERED
    if "dietary" not in state.lens:
        return None, ui.NOT_DIETARY
    if state.purpose is None:
        return None, ui.PURPOSE_UNANSWERED
    if state.purpose == "prediction":
        return None, ui.PREDICTION
    grain = effective_grain(state, structure)
    if grain is None:
        return None, "How the rows repeat is not answered yet."
    if grain.grain == "unknown":
        return None, ("Whether a person can appear in more than one row is not known, so recalls "
                      "cannot be grouped by person.")
    if grain.grain == "repeated":
        kind = effective_repeat_kind(state, structure)
        if kind == "time_points":
            return None, ui.TIME_POINTS
        if kind != "repeats":
            return None, ui.REPEATS_UNSETTLED
        return "long", None
    return "wide", None


def eligible_rows(ctx: StageContext, store: Any) -> np.ndarray:
    """Every eligible working row: the eligibility and repair rules, no outcome or completeness."""
    from turbotab.core.stages.rows import compute_cohort
    from turbotab.core.stages.working import table_info

    state = ctx.state.model_copy(update={"target": None, "missing": None})
    _, kept, _ = compute_cohort(store, state, table_info(ctx))
    return np.asarray(kept, dtype=np.int64)


def _all_rows(store: Any) -> np.ndarray:
    return np.arange(int(store.n_rows), dtype=np.int64)


def long_recalls(ctx: StageContext, store: Any, columns: Sequence[str], rows: np.ndarray,
                 id_column: str | None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """``(recall rows, person index per recall, the working rows of the persons)``. Combined
    tables: the oriented rows behind each working row; records kept as rows: the working rows,
    grouped by the grain's identifier (the person's first working row stands for them)."""
    working = dict(getattr(ctx.inputs["working"], "data", ctx.inputs["working"]))
    columns = list(dict.fromkeys(columns))
    if working.get("aggregation") is not None:
        from turbotab.core.stages.calibration import day_rows

        days = day_rows(ctx, columns, rows)
        units = days.pop("__unit").to_numpy(dtype=np.int64)
        index = {int(r): i for i, r in enumerate(rows)}
        person = np.array([index[int(u)] for u in units], dtype=np.int64)
        return days.reset_index(drop=True), person, rows
    frame = store.materialize(list(dict.fromkeys([*columns, id_column])), rows)
    codes, uniques = pd.factorize(frame[id_column], sort=False)
    keep = codes >= 0
    frame = frame.loc[keep]
    codes = codes[keep]
    # a person stands for them by their first row in the whole table (where the design is read)
    first = first_rows(store, id_column)
    person_rows = np.array([first[key] for key in uniques], dtype=np.int64)
    return frame[columns].reset_index(drop=True), codes.astype(np.int64), person_rows


def first_rows(store: Any, id_column: str) -> dict[Any, int]:
    """Each identifier's first working row over the whole table."""
    ids = store.materialize([id_column], _all_rows(store))[id_column]
    known = ids.notna().to_numpy()
    index = ids.index.to_numpy(dtype=np.int64)[known]
    out: dict[Any, int] = {}
    for key, row in zip(ids.to_numpy()[known], index):
        out.setdefault(key, int(row))
    return out


def person_values(store: Any, rows: np.ndarray, column: str) -> pd.Series:
    return store.materialize([column], rows)[column]


def weekend_values(values: pd.Series, coding: str, column: str) -> np.ndarray:
    """1 on a Friday–Sunday recall, 0 otherwise, NaN where not recorded."""
    v = pd.to_numeric(values.map(lambda x: float(x) if isinstance(x, (bool, np.bool_)) else x),
                      errors="coerce").to_numpy(dtype=float)
    present = v[np.isfinite(v)]
    if coding == "nhanes_day":
        if not np.all(np.isin(present, [1, 2, 3, 4, 5, 6, 7])):
            raise Refused(f"`{column}` holds values outside NHANES's days of the week (1 Sunday "
                          f"… 7 Saturday).")
        return np.where(np.isfinite(v), np.isin(v, [1, 6, 7]).astype(float), np.nan)
    if not np.all(np.isin(present, [0, 1])):
        raise Refused(f"`{column}` holds values other than 0 and 1; a weekend indicator is 1 on a "
                      f"Friday–Sunday recall and 0 on a Monday–Thursday one.")
    return v


def recall_order(state: Any, structure: Mapping[str, Any] | None, spec: Any) -> str | None:
    """The column ordering a long table's recalls: the answer's own ``order_column``, else the time
    column only as the ledger holds it settled (a proposed one orders nothing); None: no sequence
    covariate is modeled."""
    from turbotab.core.stages.working import time_column

    return spec.order_column or time_column(state, structure)


def order_within(person: np.ndarray, order: pd.Series) -> tuple[np.ndarray, bool]:
    """``later`` (1 after a person's first recall, by the order column) and whether any person's
    recalls tie on it."""
    o = order
    if not pd.api.types.is_numeric_dtype(o):
        parsed = pd.to_datetime(o, errors="coerce")
        o = parsed if parsed.notna().any() else pd.to_numeric(o, errors="coerce")
    frame = pd.DataFrame({"p": person, "o": o, "i": np.arange(len(person))})
    ranked = frame.sort_values(["p", "o", "i"], na_position="last")
    first = ~ranked["p"].duplicated()
    later = np.zeros(len(person))
    later[ranked["i"].to_numpy()[~first.to_numpy()]] = 1.0
    ties = bool(frame.dropna(subset=["o"]).duplicated(subset=["p", "o"]).any())
    return later, ties


# ── the offer ────────────────────────────────────────────────────────────────


def _numeric_columns(info: Mapping[str, Any]) -> set[str]:
    return {str(c["name"]) for c in info.get("columns", [])
            if c.get("dtype") in ("numeric", "integer")}


def _candidate(column: str, amounts: np.ndarray, person: np.ndarray, days: list[str] | None = None
               ) -> UsualIntakeCandidate:
    from turbotab.core.usual_intake import suggested_model

    have = np.isfinite(amounts)
    per = np.bincount(person[have], minlength=int(person.max()) + 1 if len(person) else 0)
    zero = float(np.mean(amounts[have] <= 0)) if have.any() else None
    model, why = suggested_model(zero) if zero is not None else (None, None)
    return UsualIntakeCandidate(column=column, days=days or [], zero_share=zero,
                                n_persons=int(np.sum(per > 0)), n_repeat=int(np.sum(per >= 2)),
                                suggested=model, why=why)


def build_offer(ctx: StageContext, store: Any, fmt: str | None, reason: str | None) -> UsualIntakeOffer:
    from turbotab.core import usual_intake as ui
    from turbotab.core.readings import settled_roles
    from turbotab.core.stages.working import effective_grain, table_info

    base = dict(estimand=ui.ESTIMAND, association=ui.ASSOCIATION,
                population_question=ui.POPULATION_QUESTION)
    if fmt is None:
        return UsualIntakeOffer(offered=False, reason=reason, **base)
    info = table_info(ctx)
    numeric = _numeric_columns(info)
    rows = _all_rows(store)
    state = ctx.state
    candidates: list[UsualIntakeCandidate] = []
    structure = getattr(ctx.inputs.get("structure"), "data", ctx.inputs.get("structure"))
    if fmt == "long":
        grain = effective_grain(state, structure)
        dietary = [c for c, r in settled_roles(state).items()
                   if r in ("exposure", "energy") and c in numeric]
        frame, person, persons = long_recalls(ctx, store, dietary or [grain.id_column], rows,
                                              grain.id_column)
        counts = np.bincount(person, minlength=len(persons))
        values, n = np.unique(counts[counts > 0], return_counts=True)
        recalls = {str(int(v)): int(c) for v, c in zip(values, n)}
        for c in dietary:
            candidates.append(_candidate(c, pd.to_numeric(frame[c], errors="coerce").to_numpy(float),
                                         person))
        n_repeat = int(np.sum(counts >= 2))
        offered = n_repeat >= ui.MIN_REPEATERS
        return UsualIntakeOffer(offered=offered, reason=None if offered else ui.TOO_FEW_REPEATS,
                                format="long", recalls=recalls, n_persons=int(np.sum(counts > 0)),
                                n_repeat=n_repeat, candidates=candidates, **base)
    groups = [g for g in ui.day_columns([str(c["name"]) for c in info.get("columns", [])])
              if all(d in numeric for d in g["days"])]
    if not groups:
        return UsualIntakeOffer(offered=False, reason=ui.NO_RECALL_DAYS, format="wide", **base)
    best = 0
    for g in groups:
        frame = store.materialize(g["days"], rows)
        amounts = np.concatenate([pd.to_numeric(frame[d], errors="coerce").to_numpy(float)
                                  for d in g["days"]])
        person = np.tile(np.arange(len(frame)), len(g["days"]))
        cand = _candidate(g["days"][0], amounts, person, g["days"])
        best = max(best, cand.n_repeat)
        candidates.append(cand)
    offered = best >= ui.MIN_REPEATERS
    return UsualIntakeOffer(offered=offered, reason=None if offered else ui.TOO_FEW_REPEATS,
                            format="wide", n_persons=int(len(rows)), n_repeat=best,
                            candidates=candidates, **base)


# ── one analysis ─────────────────────────────────────────────────────────────


def _estimate(e: Any) -> UsualIntakeEstimate:
    return UsualIntakeEstimate(value=e.value, se=e.se, ci_low=e.ci_low, ci_high=e.ci_high)


def _restrict(rep: Any, idx: np.ndarray) -> Any:
    from dataclasses import replace

    return replace(rep, factors=rep.factors[:, idx])


def survey_plan(ctx: StageContext, store: Any, design_rows: np.ndarray, person_rows: np.ndarray,
                n_boot: int, seed: int) -> dict[str, Any]:
    """The weights and the replication the survey answer calls for (``Refused`` until answered).

    ``design_rows``: one working row per person in the whole table (the design is read over all of
    them); ``person_rows``: the analysis' persons' rows among them."""
    from turbotab.core.methods.survey import UNANSWERED, analysis_weights
    from turbotab.core.methods.usual_intake import brr, person_bootstrap, psu_bootstrap
    from turbotab.core.survey import ATTESTATION, reading_of

    state = ctx.state
    survey = state.survey
    n = len(person_rows)
    plain = {"weights": None, "weight": None, "attestation": None, "concerns": [],
             "replication": person_bootstrap(n, n_boot, seed)}
    if survey is None:
        if reading_of(state).present:
            raise Refused(UNANSWERED)
        return plain
    if survey.estimand == "sample":
        return {**plain, "attestation": ATTESTATION}
    columns = [c for c in (survey.weight, survey.cycle, survey.four_year_weight, survey.strata,
                           survey.psu) if c]
    frame = store.materialize(list(dict.fromkeys(columns)), design_rows)
    pooled = analysis_weights(frame, survey.weight, survey.cycle, survey.four_year_weight)
    if pooled.refusal:
        raise Refused(pooled.refusal)
    position = {int(r): i for i, r in enumerate(design_rows)}
    idx = np.array([position[int(r)] for r in person_rows], dtype=np.int64)
    weights = pooled.weights[idx]
    concerns: list[str] = []
    if survey.strata and survey.psu:
        strata = frame[survey.strata].astype(str).to_numpy()
        psu = (frame[survey.strata].astype(str) + "/" + frame[survey.psu].astype(str)).to_numpy()
        per = pd.Series(psu).groupby(strata).nunique()
        try:
            rep = brr(strata, psu) if bool((per == 2).all()) else psu_bootstrap(strata, psu, n_boot, seed)
        except Exception as error:  # a stratum of one PSU
            raise Refused(str(error)) from None
        replication = _restrict(rep, idx)
    else:
        replication = person_bootstrap(n, n_boot, seed)
        concerns.append("No strata or PSUs were named with the weight, so the bootstrap resamples "
                        "participants as if each were their own sampling unit.")
    return {"weights": weights, "weight": survey.weight, "attestation": None,
            "concerns": concerns, "replication": replication}


def analysis(ctx: StageContext, store: Any, nutrient: str, spec: Any, fmt: str | None,
             reason: str | None, rows: np.ndarray) -> UsualIntakeAnalysis:
    from turbotab.core import usual_intake as ui
    from turbotab.core.methods import usual_intake as nci
    from turbotab.core.stages.working import effective_grain

    state = ctx.state
    head = dict(nutrient=nutrient, model=spec.model, days=list(spec.days),
                population=spec.population, cutoff=spec.cutoff, cutoff_kind=spec.cutoff_kind)
    if spec.model == "none":
        return UsualIntakeAnalysis(**head, applies=False,
                                   methods=f"No usual-intake distribution was estimated for `{nutrient}`.")

    def refuse(why: str) -> UsualIntakeAnalysis:
        return UsualIntakeAnalysis(**head, applies=False, refused=why, format=fmt,  # type: ignore[arg-type]
                                   methods=(f"No usual-intake distribution was estimated for "
                                            f"`{nutrient}`: {why}"))

    if fmt is None:
        return refuse(reason or "Usual intake is not offered for this table.")
    if spec.cutoff_kind == "EAR":
        # Refused when recorded; re-read here in case a role was settled as energy since.
        from turbotab.core.readings import settled_roles

        energy = {c for c, r in settled_roles(state).items() if r == "energy"}
        if energy & set(spec.days or [nutrient]):
            return refuse(ui.ENERGY_NO_EAR)
    structure = getattr(ctx.inputs.get("structure"), "data", ctx.inputs.get("structure"))
    seed = int(getattr(state.split, "seed", 0) or 0) if state.split is not None else 0
    concerns: list[str] = []
    try:
        if fmt == "long":
            if spec.days:
                raise Refused("This table's rows repeat per person, so its recalls are its rows: "
                              "name the component's column, not day columns.")
            grain = effective_grain(state, structure)
            order = recall_order(state, structure, spec)
            weekend = spec.weekend[0] if spec.weekend else None
            columns = [nutrient, *([order] if order else []), *([weekend] if weekend else [])]
            frame, person, person_rows = long_recalls(ctx, store, columns, rows, grain.id_column)
            amounts = pd.to_numeric(frame[nutrient], errors="coerce").to_numpy(float)
            if order:
                later, ties = order_within(person, frame[order])
                if ties:
                    concerns.append(f"Some participants' recalls share a value of `{order}`, so their "
                                    f"order among those recalls follows the table.")
            else:
                later = np.zeros(len(amounts))
                concerns.append("No column orders each participant's recalls, so no sequence "
                                "(time-in-sample) effect was modeled and the distribution averages "
                                "over recall positions.")
            wk = weekend_values(frame[weekend], spec.weekend_coding, weekend) if weekend else None
            design_rows = long_design_rows(ctx, store, grain.id_column)
            n_persons = len(person_rows)
        else:
            if not spec.days:
                raise Refused("Each participant has one row: name the columns that hold each recall "
                              "day, first recall first.")
            columns = [*spec.days, *spec.weekend]
            frame = store.materialize(list(dict.fromkeys(columns)), rows)
            k = len(spec.days)
            amounts = np.concatenate([pd.to_numeric(frame[d], errors="coerce").to_numpy(float)
                                      for d in spec.days])
            person = np.tile(np.arange(len(frame)), k)
            later = np.repeat((np.arange(k) > 0).astype(float), len(frame))
            wk = (np.concatenate([weekend_values(frame[c], spec.weekend_coding, c) for c in spec.weekend])
                  if spec.weekend else None)
            person_rows = rows
            design_rows = _all_rows(store)
            n_persons = len(rows)
        negative = np.isfinite(amounts) & (amounts < 0)
        if negative.any():
            raise Refused(f"{int(negative.sum()):,} recalls report a negative amount; an intake is "
                          f"never below zero.")
        domain = None
        if spec.population == "consumers":
            flag = pd.to_numeric(person_values(store, person_rows, spec.consumer_column),
                                 errors="coerce").to_numpy(float)
            present = flag[np.isfinite(flag)]
            if not np.all(np.isin(present, [0, 1])):
                raise Refused(f"`{spec.consumer_column}` holds values other than 0 and 1, so it "
                              f"cannot say who consumes the food.")
            domain = np.where(np.isfinite(flag), flag, 0.0)
        plan = survey_plan(ctx, store, design_rows, person_rows, spec.n_boot, seed)
        rec = nci.Recalls.of(amounts, person, n_persons, later=later, weekend=wk)
        if plan["weights"] is not None:
            w = np.asarray(plan["weights"], dtype=float)
            lost = int(np.sum((rec.counts() > 0) & ~(np.isfinite(w) & (w > 0))))
            if lost:
                concerns.append(f"{lost:,} participants with recalls have no positive "
                                f"`{plan['weight']}` and are left out of the population estimate.")
        if wk is not None:
            dropped = int(np.sum(np.isfinite(amounts) & ~np.isfinite(wk)))
            if dropped:
                concerns.append(f"{dropped:,} recalls with no weekend value were left out.")
        ctx.progress(0.1, f"Fitting the usual-intake model of `{nutrient}`")
        cutoff = spec.cutoff
        result = nci.usual_intake(rec, spec.model, weights=plan["weights"], domain=domain,
                                  replication=plan["replication"], cutoff=cutoff,
                                  progress=lambda f, m: ctx.progress(0.1 + 0.85 * f, m))
    except Refused as why:
        return refuse(str(why))
    except nci.UsualIntakeRefused as why:
        return refuse(str(why))
    concerns = [*concerns, *plan["concerns"], *result.concerns]
    if spec.model == "amount_only" and result.zero_share > ui.EPISODIC_SHARE:
        concerns.append(  # block and record (MODELING_SEQUENCE §2)
            f"`{nutrient}` is reported as zero on {result.zero_share:.0%} of recalls: an episodically "
            f"consumed food, for which the two-part model ranks first (Tooze et al. 2006). The "
            f"amount-only model set those {result.zeros_replaced:,} zero days to half the smallest "
            f"amount, so its distribution is of a food eaten every day.")
    if spec.model == "two_part" and result.zero_share <= ui.EPISODIC_SHARE:
        concerns.append(
            f"Only {result.zero_share:.0%} of `{nutrient}`'s recalls are zero: a component consumed "
            f"nearly every day, for which the amount-only model ranks first (Tooze et al. 2010).")
    if result.n_repeat < 50:
        concerns.append(f"The day-to-day variance rests on {result.n_repeat:,} participants with two "
                        f"or more recalls.")
    share = share_label = None
    if result.below is not None:
        below = result.below
        if spec.cutoff_kind == "UL":
            share = UsualIntakeEstimate(value=1.0 - below.value, se=below.se,
                                        ci_low=None if below.ci_high is None else 1.0 - below.ci_high,
                                        ci_high=None if below.ci_low is None else 1.0 - below.ci_low)
            share_label = f"share above the UL ({cutoff:g})"
        else:
            share = _estimate(below)
            share_label = (f"share below the EAR ({cutoff:g}): the prevalence of inadequacy by the EAR "
                           f"cut-point method" if spec.cutoff_kind == "EAR"
                           else f"share below {cutoff:g}")
    rep = result.replication
    variance = UsualIntakeVariance(method=rep.method if rep is not None else "none",
                                   replicates=0 if rep is None else len(rep.factors),
                                   ok=result.n_replicates_ok, df=None if rep is None else rep.df,
                                   fay=None if rep is None else rep.fay,
                                   n_strata=None if rep is None else rep.n_strata,
                                   n_psu=None if rep is None else rep.n_psu)
    whole = spec.population != "consumers"
    statement = ("Results describe the whole population (STROBE-nut nut-14: total population, not "
                 "consumers only)." if whole else
                 f"Results describe consumers only, the participants `{spec.consumer_column}` marks "
                 f"(STROBE-nut nut-14).")
    assumptions = [*ui.ASSUMPTIONS, *([ui.TWO_PART_ASSUMPTION] if spec.model == "two_part" else [])]
    methods = nci.methods_sentence(
        label=nutrient, model=spec.model, names=result.names, lam=result.lam,
        n_persons=result.n_persons, recalls=result.recalls, n_repeat=result.n_repeat,
        zeros_replaced=result.zeros_replaced, population=spec.population,
        consumer_column=spec.consumer_column, weight=plan["weight"], replication=rep,
        n_ok=result.n_replicates_ok, cutoff=cutoff, cutoff_kind=spec.cutoff_kind,
        rho=result.parameters.get("rho"), attestation=plan["attestation"])
    return UsualIntakeAnalysis(
        **head, applies=True, format=fmt, population_statement=statement,  # type: ignore[arg-type]
        n_persons=result.n_persons, n_recalls=result.n_recalls,
        recalls={str(k): v for k, v in result.recalls.items()}, n_repeat=result.n_repeat,
        zero_share=result.zero_share, zeros_replaced=result.zeros_replaced,
        covariates=[n for n in result.names if n != "intercept"], parameters=result.parameters,
        percentiles={str(p): _estimate(e) for p, e in result.percentiles.items()},
        mean=_estimate(result.mean), share=share, share_label=share_label,
        day_one={str(k): v for k, v in result.day_one.items()},
        mean_of_days={str(k): v for k, v in result.mean_of_days.items()},
        variance=variance, weight=plan["weight"], assumptions=assumptions, concerns=concerns,
        methods=methods)


def long_design_rows(ctx: StageContext, store: Any, id_column: str | None) -> np.ndarray:
    """One working row per person over the whole table: every row when the rows were combined per
    person, else each person's first row."""
    working = dict(getattr(ctx.inputs["working"], "data", ctx.inputs["working"]))
    if working.get("aggregation") is not None:
        return _all_rows(store)
    return np.array(sorted(first_rows(store, id_column).values()), dtype=np.int64)


# ── the stage ────────────────────────────────────────────────────────────────


def usual_intake_stage(ctx: StageContext) -> Bundle:
    from turbotab.core.stages.data import open_store

    state = ctx.state
    structure = getattr(ctx.inputs.get("structure"), "data", ctx.inputs.get("structure"))
    fmt, reason = gate(state, structure)
    ctx.progress(0.02, "Reading where the recalls are")
    with open_store(ctx) as store:
        offer = build_offer(ctx, store, fmt, reason)
        analyses = []
        specs = dict(state.usual_intake or {})
        rows = eligible_rows(ctx, store) if specs and fmt is not None else np.empty(0, dtype=np.int64)
        for nutrient, spec in specs.items():
            analyses.append(analysis(ctx, store, nutrient, spec, fmt, reason, rows))
    ctx.progress(1.0, "Done")
    return Bundle(data=UsualIntakeArtifact(offer=offer, analyses=analyses).model_dump(mode="json"))


USUAL_INTAKE_READS = ("usual_intake", "lens", "purpose", "grain", "repeat_kind", "unit", "aggregation",
                      "temporal", "survey", "exclusions", "findings", "split", "roles",
                      "roles_unconfirmed", "role_confirmations", "reading_confirmations",
                      "shape_confirmations")

__all__ = ["USUAL_INTAKE_READS", "UsualIntakeAnalysis", "UsualIntakeArtifact", "UsualIntakeOffer",
           "analysis", "build_offer", "eligible_rows", "gate", "recall_order", "usual_intake_stage"]
