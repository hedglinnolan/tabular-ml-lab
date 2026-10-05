"""The survey question: "the surveyed population, or these participants?" (AUDIT_REPORT §5 WP10).

ME-06: with NHANES weights, strata and PSUs in the table, inference ran unweighted with
simple-random-sample intervals and said nothing. NHANES Analytic Guidelines 2011–2016 §3: "The
complex survey design used for NHANES, including oversampling, stratification, and clustering, must
be considered when analyzing the data for appropriate variance estimation and to calculate
statistics representative of the U.S. civilian non-institutionalized population." So under
inference, whenever a column reads as a survey weight, the Router asks (``set_survey``):

* **The surveyed population** (``estimand = "population"``): the design-based estimate
  (:mod:`turbotab.core.models.survey`): weighted, with Taylor-linearized intervals over the strata
  and PSUs, every row outside the analysis kept in the design as a domain.
* **These participants** (``estimand = "sample"``): unweighted, and recorded as such. The methods
  sentence carries the attestation :data:`ATTESTATION` word for word, so a reader knows the
  estimand is the sample's and the intervals are not the design's.

Until it is answered, an inference table on such a table is blocked (the fit says why). Under
prediction the question is not asked: scores describe the rows they were computed on, and the fit
says they are unweighted.

This module is light (the name reading, the Router's gate and the refusals) so that importing the
decisions registers ``set_survey``'s refusals; the weights and the design are built in
:mod:`turbotab.core.methods.survey`. Which weight suits which variables (the least-common-
denominator rule, IN-19) is WP13's: here every recognized weight is offered, and the choice is the
user's.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# The attestation the "these participants" answer records, word for word (AUDIT_REPORT §5 WP10.1).
ATTESTATION = "unweighted, sample-only estimand; standard errors ignore strata and PSUs"

GUIDELINES = "NHANES Analytic Guidelines 2011–2016"
GUIDELINES_URL = "https://wwwn.cdc.gov/nchs/data/nhanes/analyticguidelines/11-16-analytic-guidelines.pdf"

# NHANES's own names. A bare ``WT…`` name counts as a weight only beside one of these (a column
# called ``WTKG`` in a clinical table is a body weight), or when it is a weight NCHS publishes.
NHANES_DESIGN = {"SDMVSTRA", "SDMVPSU", "SDDSRVYR", "SEQN"}
NHANES_WEIGHTS = {
    "WTINT2YR", "WTMEC2YR", "WTDRD1", "WTDR2D", "WTSAF2YR", "WTSA2YR", "WTSB2YR", "WTSC2YR",
    "WTSOG2YR", "WTSVOC2Y", "WTINT4YR", "WTMEC4YR", "WTDR4YR", "WTSAF4YR", "WTINTPRP",
    "WTMECPRP", "WTDRD1PP", "WTDR2DPP", "WTSAFPRP",
}
# The 2-year weight's 4-year counterpart for 1999–2002 (the NHANES 1999–2002 codebooks: DEMO's
# WTINT4YR and WTMEC4YR; DRXTOT's WTDR4YR, "Dietary day one 4-Year sample weight").
FOUR_YEAR = {"WTINT2YR": "WTINT4YR", "WTMEC2YR": "WTMEC4YR", "WTDRD1": "WTDR4YR",
             "WTSAF2YR": "WTSAF4YR"}
_NHANES_WEIGHT = re.compile(r"^WT[A-Z0-9]{2,8}$", re.I)
_REPLICATE = re.compile(r"^WT[A-Z]*REP\d+$", re.I)  # jackknife and BRR replicate weights
_STRATA = re.compile(r"^sdmvstra$|^(?:strat|stratid|vstrat|vestr|var_strata)$|"
                     r"(?:^|_)(?:strata|stratum)(?:$|_)", re.I)
_PSU = re.compile(r"^sdmvpsu$|^(?:psuid|vpsu|secu|var_psu)$|(?:^|_)psu(?:$|_)", re.I)
_CYCLE = re.compile(r"^sddsrvyr$|(?:^|_)(?:survey_)?cycle(?:$|_)|^survey_year$", re.I)


@dataclass(frozen=True)
class DesignReading:
    """What the column names say about a survey design. Never a decision: the user's answer is.

    ``present`` when any part of a design is read, a weight, strata or primary sampling units:
    a table that names its sampling units but no weight TurboTab can read (BRFSS ``_PSU`` beside
    ``_LLCPWT``) still asks the survey question, rather than taking the unweighted estimand
    without a word (audit WP13 gate repair)."""

    weights: list[str] = field(default_factory=list)
    strata: list[str] = field(default_factory=list)
    psu: list[str] = field(default_factory=list)
    cycles: list[str] = field(default_factory=list)
    four_year: dict[str, str] = field(default_factory=dict)  # 2-year weight -> its 4-year column
    # Columns the user settled as part of the design whose names say nothing of which part
    # (NDNS ``wti_Y911``, ``astrata1``, ``area``; MEPS ``PERWT19F``, ``VARSTR``, ``VARPSU``):
    # their values place them (:func:`place_design`) and the survey question names each placement.
    unplaced: list[str] = field(default_factory=list)

    @property
    def present(self) -> bool:
        return bool(self.weights or self.strata or self.psu or self.unplaced)

    def columns(self) -> list[str]:
        return list(dict.fromkeys([*self.weights, *self.four_year.values(), *self.strata,
                                   *self.psu, *self.cycles, *self.unplaced]))


def read_design(columns: Sequence[str], target: str | None = None) -> DesignReading:
    """Survey weights, strata, PSUs and cycle columns among ``columns``, by their names."""
    from turbotab.core.recognizers import reads_as_survey_weight

    names = [str(c) for c in columns if c is not None and str(c) != target]
    upper = {c.upper(): c for c in names}
    nhanes = bool(NHANES_DESIGN & set(upper))
    strata = [c for c in names if _STRATA.search(c)]
    psu = [c for c in names if _PSU.search(c)]
    # One weight reader for the roles proposal and this question
    # (:func:`turbotab.core.recognizers.reads_as_survey_weight`): the strata and PSUs the table
    # names are what corroborates a weight the name alone leaves ambiguous.
    designed = nhanes or bool(strata or psu)
    weights: list[str] = []
    for c in names:
        u = c.upper()
        if _REPLICATE.match(c) and u not in NHANES_WEIGHTS:
            continue
        if (u in NHANES_WEIGHTS or (nhanes and _NHANES_WEIGHT.match(c))
                or reads_as_survey_weight(c, design_in_table=designed)):
            weights.append(c)
    four: dict[str, str] = {}
    for w in weights:
        u = w.upper()
        partner = FOUR_YEAR.get(u) or (u.replace("2YR", "4YR") if "2YR" in u else None)
        if partner and partner in upper and upper[partner] != w:
            four[w] = upper[partner]
    partners = set(four.values())  # a four-year weight is offered through its 2-year partner
    return DesignReading(
        weights=[w for w in weights if w not in partners],
        strata=[c for c in strata if c not in weights],
        psu=[c for c in psu if c not in weights],
        cycles=[c for c in names if _CYCLE.search(c)],
        four_year=four,
    )


def reading_of(state: Any) -> DesignReading:
    """The design the roles' columns read as (every column but the outcome): by their names, and
    every column whose ``design`` role is settled (BLUEPRINT §14.1, the readings ledger: the user's
    confirmed role outranks a name the reader does not know; the gate's NDNS and MEPS designs,
    each set to ``design`` by the user, were "no surveyed population")."""
    from dataclasses import replace

    from turbotab.core.readings import settled_roles

    roles = getattr(state, "roles", None) or {}
    target = getattr(state, "target", None)
    reading = read_design(list(roles), target)
    named = set(reading.columns())
    unplaced = [c for c, r in settled_roles(state).items()
                if r == "design" and c != target and c not in named]
    return replace(reading, unplaced=unplaced) if unplaced else reading


def place_design(frame: Any, columns: Sequence[str]) -> dict[str, list[str]]:
    """Where the values place each design column the names do not: a weight is non-negative with
    fractional values or many distinct ones (a survey weight is a reciprocal of a selection
    probability, rarely whole); strata and sampling units are whole-number codes. Which code is the
    stratum and which the PSU is read only when one nests in the other (each PSU in one stratum);
    otherwise both orders are offered, each named, and the user's answer records which."""
    import numpy as np
    import pandas as pd

    weights: list[str] = []
    codes: list[str] = []
    for c in columns:
        if frame is None or c not in frame.columns:
            continue
        x = pd.to_numeric(frame[c], errors="coerce").dropna()
        if x.empty or (x < 0).any():
            continue
        whole = bool(np.all(np.isclose(x.to_numpy(dtype=float), np.round(x.to_numpy(dtype=float)))))
        if not whole or x.nunique() > 0.5 * len(x):
            weights.append(str(c))
        else:
            codes.append(str(c))
    return {"weights": weights, "codes": codes}


def _code_pairs(frame: Any, codes: Sequence[str]) -> list[tuple[str | None, str | None]]:
    """(strata, psu) pairs the whole-number design codes may form, nested ones read, else both
    orders; one code may be either."""
    if not codes:
        return [(None, None)]
    if len(codes) == 1:
        return [(codes[0], None), (None, codes[0])]
    a, b = codes[0], codes[1]
    pairs: list[tuple[str | None, str | None]] = []
    if frame is not None and a in frame.columns and b in frame.columns:
        both = frame[[a, b]].dropna()
        if len(both):
            if int(both.groupby(b)[a].nunique().max()) == 1 and both[a].nunique() < both[b].nunique():
                return [(a, b)]
            if int(both.groupby(a)[b].nunique().max()) == 1 and both[b].nunique() < both[a].nunique():
                return [(b, a)]
    pairs = [(a, b), (b, a)]
    return pairs


def not_applicable_reason(state: Any) -> str | None:
    """The Router's reason when the question does not apply; None when it does, or while it
    cannot be told (no purpose or no roles yet)."""
    purpose = getattr(state, "purpose", None)
    if purpose == "prediction":
        return ("Under prediction the scores describe the rows they were computed on; they are not "
                "weighted to a population.")
    if purpose is None or getattr(state, "roles", None) is None:
        return None
    if not reading_of(state).present:
        return ("No column reads as a survey weight, stratum or sampling unit, so there is no "
                "surveyed population to weight to.")
    return None


def cycle_number(value: Any, column: str) -> int | None:
    """The NHANES release a cycle value names (``SDDSRVYR``: 1 for 1999–2000, 2 for 2001–2002, …),
    or, from a label, 1 or 2 for one starting 1999 or 2001; None otherwise (only those two need
    the four-year rule)."""
    if value is None or value != value:  # None or NaN
        return None
    text = str(value).strip()
    if column.upper() == "SDDSRVYR":
        try:
            return int(float(text))
        except ValueError:
            return None
    if re.match(r"^1999(?!\d)", text):
        return 1
    if re.match(r"^2001(?!\d)", text):
        return 2
    return None


def _listing(items: Sequence[str]) -> str:
    quoted = [f"`{i}`" for i in dict.fromkeys(items)]
    if not quoted:
        return "the design columns"
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def sample_concern(reading: DesignReading) -> str:
    """What an inference table answered "these participants" says first."""
    return (f"These estimates describe these participants, as answered: {ATTESTATION} "
            f"({_listing([*reading.weights, *reading.strata, *reading.psu])} recorded and not "
            f"used).")


def offered(state: Any, pooled_cycle: str | None = None, frame: Any = None,
            placed: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """The answers the survey question offers, each with the decision it records.

    One "surveyed population" option per weight the names read as, with the strata, PSU (and, when
    ``pooled_cycle`` names a cycle column holding more than one cycle, the cycle and the four-year
    weight) filled in from the reading, then "these participants". Design columns the user settled
    whose names say nothing of their part are placed by their values (``placed``:
    :func:`place_design`), and an option names each placement it takes.
    """
    reading = reading_of(state)
    if not reading.present:
        return []
    if reading.unplaced and not (reading.weights or reading.strata or reading.psu):
        return _offered_by_values(reading, placed or {}, frame)
    strata = reading.strata[0] if reading.strata else None
    psu = reading.psu[0] if reading.psu else None
    weights = list(reading.weights)
    if "dietary" in (getattr(state, "lens", None) or []):
        # Dietary analyses take the dietary weights (NUTRITION_PACK §01).
        weights.sort(key=lambda w: not w.upper().startswith("WTDR"))
    # Which weight suits the variables used is NHANES's least-common-denominator rule (audit
    # IN-19; the NHANES weighting tutorial: "use the weight of the smallest subpopulation that
    # includes all the variables you want to include in your analysis"): it is offered first, then
    # the samples that contain its sample (fasting, then examination, then dietary: BLUEPRINT §14).
    from turbotab.core.readings import unsettled
    from turbotab.core.recognizers import least_common_denominator, rank_weights

    roles = getattr(state, "roles", None) or {}
    target = getattr(state, "target", None)
    used = [c for c, r in roles.items() if r in ("exposure", "covariate", "energy")]
    # With the table's values (``frame``), a variable recorded only on a subsample's rows is read
    # as that subsample's whatever its name (audit WP13 gate repair).
    lcd = least_common_denominator([*([target] if target else []), *used, *weights], frame)
    if lcd is not None:
        weights = rank_weights(weights, lcd)
    waiting = set(unsettled(state)) if roles else set()
    out: list[dict[str, Any]] = []
    for w in weights[:4]:
        # Never pre-acknowledged: with no strata or PSU the server asks for the attestation.
        decision = {"kind": "set_survey", "estimand": "population", "weight": w, "strata": strata,
                    "psu": psu, "cycle": pooled_cycle,
                    "four_year_weight": reading.four_year.get(w) if pooled_cycle else None,
                    "acknowledged": False}
        named = [f"`{p}`" for p in (strata, psu) if p]
        within = f" over {' and '.join(named)}" if named else "; no strata or PSU named"
        label = "Surveyed population" if len(weights) == 1 else f"Population by {w}"
        # BLUEPRINT §14 rule 2: a design role that rode along unconfirmed is marked, and the
        # server refuses the option until the role is confirmed on its own.
        needs = [c for c in (w, strata, psu) if c and c in waiting]
        # The cycle column the option pools by is said where it is taken (BLUEPRINT §14.1): it
        # divides each weight by the cycles pooled, so the answer records it seen.
        pooled = (f"; pooled over the survey cycles `{pooled_cycle}` names, each weight divided by "
                  f"their number" if pooled_cycle else "")
        out.append({"key": f"population:{w}", "label": label,
                    "consequence": f"Weighted by `{w}`{within}{pooled}; intervals by Taylor "
                                   f"linearization.",
                    "decision": decision, "needs_confirmation": needs})
    out.append({"key": "sample", "label": "These participants",
                "consequence": "Unweighted; the methods state the estimand is this sample's.",
                "decision": {"kind": "set_survey", "estimand": "sample"}})
    return out


def _offered_by_values(reading: DesignReading, placed: Mapping[str, Any],
                       frame: Any) -> list[dict[str, Any]]:
    """The options for a design the user set by role and the names do not read: each weight the
    values place, with each (strata, PSU) placement of the codes, every column named in its label;
    then "these participants". With nothing placed, the question is still asked."""
    weights = [w for w in placed.get("weights") or [] if w in reading.unplaced]
    codes = [c for c in placed.get("codes") or [] if c in reading.unplaced]
    out: list[dict[str, Any]] = []
    for w in weights[:2]:
        for strata, psu in _code_pairs(frame, codes)[:2]:
            parts = [f"strata `{strata}`" if strata else None, f"PSU `{psu}`" if psu else None]
            within = (" over " + " and ".join(p for p in parts if p)) if any(parts) else \
                "; no strata or PSU named"
            out.append({"key": f"population:{w}:{strata}:{psu}",
                        "label": f"Population by {w}" + (f", strata {strata}" if strata else "")
                                 + (f", PSU {psu}" if psu else ""),
                        "consequence": (f"Weighted by `{w}`{within}, as you set their roles to "
                                        f"design and their values place them; intervals by Taylor "
                                        f"linearization."),
                        "decision": {"kind": "set_survey", "estimand": "population", "weight": w,
                                     "strata": strata, "psu": psu, "cycle": None,
                                     "four_year_weight": None, "acknowledged": False},
                        "needs_confirmation": []})
    out.append({"key": "sample", "label": "These participants",
                "consequence": "Unweighted; the methods state the estimand is this sample's.",
                "decision": {"kind": "set_survey", "estimand": "sample"}})
    return out


# ── refusals (registered on ``set_survey``) ──────────────────────────────────


def _asked_under_inference(decision: Any, ctx: Any) -> None:
    """MODELING_SEQUENCE ruling 13 (2026-10-05): under prediction the answer says whose performance
    the scores estimate. The surveyed population's is design-based cross-validation (whole PSUs
    within strata, every score weighted; ``models.design_cv``, offered by the ``explore`` stage);
    these participants' stays unweighted, labeled as the procedure's performance on these rows. So
    the answer is accepted under both purposes (it was refused under prediction before the ruling)."""
    return None


def _names_real_columns(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import ROW_ID, Refusal, _columns_of, _ctx, _state

    columns = _columns_of(ctx)
    named = {k: getattr(decision, k) for k in ("weight", "strata", "psu", "cycle", "four_year_weight")}
    if columns is not None:
        unknown = [c for c in named.values() if c and (c not in columns or c == ROW_ID)]
        if unknown:
            raise Refusal("unknown_column", f"This dataset has no column named `{unknown[0]}`.",
                          exits=[{"label": "Choose one of the dataset's columns", "decision": None}])
    if decision.estimand != "population":
        return
    state = _state(ctx)
    if not decision.weight:
        raise Refusal(
            "no_weight",
            "A population estimate is weighted: name the survey weight the estimate uses.",
            exits=[*({"label": f"Weight by `{o['decision']['weight']}`", "decision": o["decision"]}
                     for o in offered(state) if o["decision"].get("weight")),
                   {"label": "Estimate for these participants instead",
                    "decision": {"kind": "set_survey", "estimand": "sample"}}])
    target = getattr(state, "target", None) or _ctx(ctx, "target")
    roles = getattr(state, "roles", None) or {}
    for slot, column in named.items():
        if not column:
            continue
        if column == target:
            raise Refusal("design_is_outcome",
                          f"`{column}` is the outcome, so it cannot be part of the survey design.",
                          exits=[{"label": "Name another column", "decision": None}])
        if roles.get(column) in ("exposure", "covariate", "energy"):
            raise Refusal(
                "design_is_predictor",
                f"`{column}` is a predictor ({roles[column]}); a survey "
                f"{slot.replace('_', ' ')} describes how people were sampled and cannot also be "
                f"a predictor.",
                exits=[{"label": f"Give `{column}` the design role (the column roles)",
                        "decision": None}])


def _weight_is_a_number(decision: Any, ctx: Any) -> None:
    from turbotab.core.decisions import NUMERIC_DTYPES, Refusal, _info

    if decision.estimand != "population":
        return
    for column in (decision.weight, decision.four_year_weight):
        info = _info(ctx, column) if column else None
        if info is not None and info.get("dtype") not in NUMERIC_DTYPES:
            raise Refusal("weight_not_numeric",
                          f"`{column}` holds {info.get('dtype')} values; a survey weight is the "
                          f"number of people a row represents.",
                          exits=[{"label": "Name the numeric weight column", "decision": None}])


def _names_its_units(decision: Any, ctx: Any) -> None:
    """Weights alone correct the estimate, not the interval: without strata and PSUs the user
    attests what the intervals assume (block and record, BLUEPRINT §11.3)."""
    from turbotab.core.decisions import Refusal

    if decision.estimand != "population" or decision.acknowledged or (decision.strata and decision.psu):
        return
    missing = [n for n, v in (("strata", decision.strata), ("PSU", decision.psu)) if not v]
    what = " or ".join(missing)
    units = "each row as a sampling unit of its own" if "PSU" in missing else "the PSUs as unstratified"
    attested = decision.model_copy(update={"acknowledged": True})
    raise Refusal(
        "partial_design",
        f"No {what} is named. Weights correct the estimate, not its interval: the intervals would "
        f"treat {units}, and where people were sampled in clusters they are too narrow.",
        exits=[{"label": f"Name the {what} column", "decision": None},
               {"label": "My table has no such column; record that", "decision": attested}])


def _pools_cycles_by_the_rule(decision: Any, ctx: Any) -> None:
    """1999–2000 pooled with other cycles takes the four-year weight (Analytic Guidelines §3.1.4)."""
    from turbotab.core.decisions import Refusal, _ctx, _state

    if decision.estimand != "population" or not decision.weight:
        return
    state = _state(ctx)
    reading = reading_of(state) if state is not None else DesignReading()
    cycle = decision.cycle or (reading.cycles[0] if reading.cycles else None)
    store_fn = _ctx(ctx, "store")
    if cycle is None or not callable(store_fn):
        return
    store = store_fn()
    if store is None or cycle not in set(getattr(store, "columns", ()) or ()):
        return
    from turbotab.core.methods.survey import analysis_weights

    wanted = [c for c in (decision.weight, cycle, decision.four_year_weight) if c]
    frame = store.materialize(list(dict.fromkeys(wanted)), None)
    pooled = analysis_weights(frame, decision.weight, cycle, decision.four_year_weight)
    if pooled.refusal is not None:
        partner = reading.four_year.get(decision.weight)
        exits: list[dict[str, Any]] = []
        if partner and partner != decision.four_year_weight:
            exits.append({"label": f"Use `{partner}` for 1999–2002",
                          "decision": decision.model_copy(update={"cycle": cycle,
                                                                  "four_year_weight": partner})})
        exits.append({"label": "Estimate for these participants instead",
                      "decision": {"kind": "set_survey", "estimand": "sample"}})
        raise Refusal("four_year_weight", pooled.refusal, exits=exits)
    if decision.cycle is None and len(pooled.cycles) > 1 \
            and any(cycle_number(v, cycle) == 1 for v in pooled.cycles):
        raise Refusal(
            "name_the_cycle",
            f"`{cycle}` pools 1999–2000 with other cycles; name it, so the 1999–2002 rows take their "
            f"four-year weight ({GUIDELINES} §3.1.4).",
            exits=[{"label": f"Pool by `{cycle}`",
                    "decision": decision.model_copy(update={"cycle": cycle})}])


def _design_is_settled(decision: Any, ctx: Any) -> None:
    """BLUEPRINT §14 rule 2: the survey weight (and the strata and PSU) are number-changing
    defaults; a design role that rode along unconfirmed in a bulk confirm is confirmed on its own
    first."""
    from turbotab.core.decisions import Refusal, _state
    from turbotab.core.readings import confirm_exits, unsettled, unsettled_message

    if decision.estimand != "population":
        return
    state = _state(ctx)
    if state is None or not getattr(state, "roles", None):
        return
    named = [c for c in (decision.weight, decision.strata, decision.psu, decision.four_year_weight)
             if c]
    waiting = unsettled(state, list(dict.fromkeys(named)))
    if not waiting:
        return
    raise Refusal("role_unconfirmed", unsettled_message(waiting, "the survey design"),
                  exits=[*confirm_exits(state, waiting),
                         {"label": "Estimate for these participants instead",
                          "decision": {"kind": "set_survey", "estimand": "sample"}}])


def _register() -> None:
    from turbotab.core.decisions import register_validator

    for check in (_asked_under_inference, _names_real_columns, _weight_is_a_number,
                  _names_its_units, _pools_cycles_by_the_rule, _design_is_settled):
        register_validator("set_survey", check)


_register()

__all__ = [
    "ATTESTATION", "DesignReading", "FOUR_YEAR", "GUIDELINES", "GUIDELINES_URL", "NHANES_WEIGHTS",
    "cycle_number", "not_applicable_reason", "offered", "read_design", "reading_of",
    "sample_concern",
]
