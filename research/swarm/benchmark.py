"""The glucose project's benchmark (SWARM_PLAN §5, dataset 3; §10 package 5): Bang et al. 2009's
points score, adapted to the later cycle's questions, on a run's own held-out participants, and the
AUC of the outcome thresholded at Bang's cut for diabetes (:data:`THRESHOLD`) ranked by it
(``notes/replication/glucose_benchmarks.md``). The command line's help says the same, the cut's
number filled in from :data:`THRESHOLD` (:data:`HELP`).

    scripts/py -m research.swarm.benchmark <export.zip> [--record <record.json>] [--out <file>]

**The participants are the record's.** The bundle a run exported (its ``export.zip``; with
``--record``, it must be the one whose SHA-256 the corpus record holds) carries the run's decision
log. Its first opening of the held-out rows (``open_seal``) is the reported result, so the state is
the engine's own fold (``decisions.fold``) of the records before it, and the participants are the
engine's own participant flow (``stages.rows.cohort_flow``) under that state over the table the run
uploaded (held to the bundle's recorded SHA-256 and to the dataset's manifest): the eligibility
decisions the analyst recorded, the outcome measured, complete cases where the run chose them.
The script applies no filter of its own. The held-out participants are those at or after the
record's own cut-point, which must be the card's (``card.json``'s ``held_out``), and their number
must be the opening's own count of held-out rows. Refused, each with its reason: a bundle with no
opening; another table; another outcome or purpose; a cut-point not the card's, or none; a held-out
count not the opening's; a joined file, a landmark, a repair recorded without its settings, or a
repair (a reading of text as numbers among them) on a column the flow reads (the engine's flow
would not be the benchmark's over the table as uploaded); a held-out participant not of the test
cycle, or without a fasting weight; or a design the jackknife cannot read. A consumers-only domain
is not read: the engine applies one under Estimate only, and the record is a prediction.

**The score** (:data:`ITEMS`), each item's rule written beside the passages it rests on, every one
found on its NCHS page as the dataset's manifest holds it (refused otherwise): Table 2's points
(``intake.BANG_TABLE2``) for age, male sex, family history (MCQ300C, the close relatives' item,
which Bang's own validation cycle used; the rule names that cycle as the notes' Table 3 row prints
it, :func:`reported_auc`, the reading the output's ``reported`` gives), a history of hypertension
(told, taking prescribed medicine, or the mean of the oscillometric readings at 140/90 or more),
the body-size category (BMI or waist, the paper's cut-offs) and physical activity. PAQ520, the
paper's item, was discontinued after 2005–2006, so activity is adapted: active is a yes to
vigorous or to moderate recreational activity in a typical week (PAQ650, PAQ665), and the score is
refused should the later cycle's file document PAQ520. A categorical answer missing or unknown is
the condition absent, as the paper defines it; so is a blank measurement, a choice the paper
leaves open (the notes' §6.2 and §6.5): with both BMI and waist blank, not overweight or obese,
and a blood pressure measure with no reading, not high.

**The AUC** of LBXGLU at or above :data:`THRESHOLD` (mg/dL), ranked by the score, ties counting one
half: per participant (primary, as published AUCs are), with DeLong's interval
(``models.performance``); and weighted by the test cycle's fasting weights, with the
delete-one-PSU jackknife's interval (JKn within strata over the whole cycle's design, the estimate
re-weighted in each replicate). Beside them, Bang's reported AUC on its own later cycle, read from
``notes/replication/bang2009.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.swarm import intake as I  # noqa: E402

THRESHOLD = I.BANG_FPG  # mg/dL: the outcome thresholded as Bang's undiagnosed diabetes is
# The command line's description: the module docstring's first paragraph, the cut's number the one
# the AUC uses.
HELP = ("The glucose project's benchmark (SWARM_PLAN §5, dataset 3; §10 package 5): Bang et al. "
        "2009's points score, adapted to the later cycle's questions, on a run's own held-out "
        f"participants, and the AUC of the outcome thresholded at {THRESHOLD:g} mg/dL ranked by it "
        "(``notes/replication/glucose_benchmarks.md``).")
NOTES = REPO / I.BANG_NOTES
# Table 3's validation row as the notes print it: the cycle, its N and its AUC.
REPORTED = re.compile(r"\| Validation (\d{4}–\d{4}) \(N = ([\d,]+), \*\*AUC (\d\.\d+)\*\*\) \|")
# The paper's cut-offs (Table 2's footnote, the notes' §2.3): BMI in kg/m², waist in inches, by sex.
BMI_CUTS = (25.0, 30.0, 40.0)  # overweight, obese, extremely obese
WAIST_CUTS = {"male": (37.0, 40.0, 50.0), "female": (31.5, 35.0, 49.0)}
CM_PER_INCH = 2.54
HYPERTENSIVE = (140.0, 90.0)  # systolic, diastolic, mmHg: "≥ 140" or "≥ 90"
AGE_BANDS = (40, 50, 60)  # each band's lowest age in years; under the first is the reference
POINTS = {factor: points for factor, _, _, _, points in I.BANG_TABLE2}


class BenchmarkRefused(Exception):
    """The benchmark cannot be computed for this record; ``message`` says why."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# ── the score's items, each declared and quoted from the NCHS pages ──────────


def _url(name: str, cycle: str = I.GLUCOSE_TEST) -> str:
    return I.glucose_url(name, cycle)


PAQ_C = I.page_url(2003, "PAQ_C")  # Bang's own activity item, on its last development cycle's page
# Where a rule names Bang's validation cycle: filled in from the notes' Table 3 row (:func:`items`).
VALIDATION_CYCLE = "<the validation cycle>"
ITEMS: list[dict[str, Any]] = [
    {"item": "age",
     "rule": (f"RIDAGEYR: {AGE_BANDS[0]} to {AGE_BANDS[1] - 1}, {AGE_BANDS[1]} to "
              f"{AGE_BANDS[2] - 1}, or {AGE_BANDS[2]} or older, against under {AGE_BANDS[0]}"),
     "passages": [(_url("DEMO"), "RIDAGEYR - Age in years at screening")]},
    {"item": "male", "rule": "RIAGENDR is 1 (Male)",
     "passages": [(_url("DEMO"), "RIAGENDR - Gender")]},
    {"item": "family history of diabetes",
     "rule": ("MCQ300C is 1 (Yes): a close relative, a parent or a sibling, told they had "
              "diabetes; the paper's development cycles asked of any blood relative (MCQ250A), its "
              f"validation cycle {VALIDATION_CYCLE} this item"),
     "passages": [(_url("MCQ"), "Including living and deceased, were any of {SP's/your} close "
                   "biological that is, blood relatives including father, mother, sisters or "
                   "brothers, ever told by a health professional that they had diabetes?")]},
    {"item": "history of hypertension",
     "rule": ("BPQ020 is 1 (told of high blood pressure), BPQ050A is 1 (now taking the prescribed "
              "medicine), or the mean of the available oscillometric readings is "
              f"{HYPERTENSIVE[0]:g} mmHg or more systolic (BPXOSY1 to BPXOSY3) or "
              f"{HYPERTENSIVE[1]:g} or more diastolic (BPXODI1 to BPXODI3); a measure with no "
              "reading, systolic or diastolic, is not high, so no reading at all scores no point "
              "by measurement (the paper does not state its rule for the readings: "
              f"{I.BANG_NOTES} §6.2)"),
     "passages": [(_url("BPQ"), "{Have you/Has SP} ever been told by a doctor or other health "
                   "professional that {you/s/he} had hypertension, also called high blood "
                   "pressure?"),
                  (_url("BPQ"), "BPQ050A - Now taking prescribed medicine for HBP"),
                  (_url("BPXO"), I.P_BPXO_OMRON)]},
    {"item": "body size",
     "rule": ("the higher of BMXBMI's and BMXWAIST's category (waist in inches, cm divided by "
              f"{CM_PER_INCH:g}): overweight, obese or extremely obese at the paper's cut-offs, "
              "by sex; a blank measure has no category, so with one blank the other decides, and "
              "with both blank, not overweight or obese (no points), a choice the paper leaves "
              f"open ({I.BANG_NOTES} §6.5)"),
     "passages": [(_url("BMX"), "BMXBMI - Body Mass Index (kg/m**2)"),
                  (_url("BMX"), "BMXWAIST - Waist Circumference (cm)")]},
    {"item": "physically active",
     "rule": ("PAQ650 or PAQ665 is 1 (Yes): vigorous or moderate recreational activity in a "
              "typical week. Adapted: the paper's item, PAQ520 (more active than others of the "
              "same age), is not in the later cycle's file"),
     "passages": [(PAQ_C, "Compared with most {men/boys/women/girls} {your/SP's} age, would you "
                   "say that {you are/s/he is} . . ."),
                  (_url("PAQ"), "In a typical week {do you/does SP} do any vigorous-intensity "
                   "sports, fitness, or recreational activities that cause large increases in "
                   "breathing or heart rate like running or basketball for at least 10 minutes "
                   "continuously?"),
                  (_url("PAQ"), "In a typical week {do you/does SP} do any moderate-intensity "
                   "sports, fitness, or recreational activities that cause a small increase in "
                   "breathing or heart rate such as brisk walking, bicycling, swimming, or "
                   "volleyball for at least 10 minutes continuously?")]},
]
COLUMNS = ("RIDAGEYR", "RIAGENDR", "MCQ300C", "BPQ020", "BPQ050A",
           *(f"BPXO{m}{i}" for m in ("SY", "DI") for i in range(1, 4)), "BMXBMI", "BMXWAIST",
           "PAQ650", "PAQ665")


def item_problems(pages: Mapping[str, str]) -> list[str]:
    """Every passage of :data:`ITEMS` not found on its page (``pages``: URL → text); and the
    paper's activity item, PAQ520, where the later cycle's file still documents it."""
    out = [f"{item['item']}: “{quote}” is not on {url}" for item in ITEMS
           for url, quote in item["passages"]
           if url not in pages or not I.quoted(quote, pages[url])]
    paq = I.glucose_books("PAQ")[I.GLUCOSE_TEST]
    if "PAQ520" in paq:
        out.append("PAQ520 is in the later cycle's activity file: the adaptation is not needed")
    return out


def _yes(values: Any) -> np.ndarray:
    """1 where the answer is code 1 (Yes), else 0: refused, don't know and blank are the condition
    absent (the paper: "If information was missing or unknown in categorical variables, we
    defined the condition as absent")."""
    return (pd.to_numeric(pd.Series(values), errors="coerce") == 1).to_numpy()


def _level(values: Any, cuts: Sequence[float]) -> np.ndarray:
    """0 below the first cut, then 1, 2, 3 at or above each; 0 where blank."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    out = np.zeros(len(x), dtype=int)
    for cut in cuts:
        out += np.where(np.isnan(x), 0, x >= cut).astype(int)
    return out


def score(frame: pd.DataFrame) -> np.ndarray:
    """Each row's points (``intake.BANG_TABLE2``), the items as :data:`ITEMS` defines them."""
    age = pd.to_numeric(frame["RIDAGEYR"], errors="coerce").to_numpy(dtype=float)
    band = np.select([age >= AGE_BANDS[2], age >= AGE_BANDS[1], age >= AGE_BANDS[0]],
                     [POINTS["age 60 or older"], POINTS["age 50-59"],
                      POINTS["age 40-49, against under 40"]], 0)
    male = pd.to_numeric(frame["RIAGENDR"], errors="coerce").to_numpy() == 1
    sys_ = frame[[f"BPXOSY{i}" for i in range(1, 4)]].apply(pd.to_numeric, errors="coerce")
    dia = frame[[f"BPXODI{i}" for i in range(1, 4)]].apply(pd.to_numeric, errors="coerce")
    high = ((sys_.mean(axis=1) >= HYPERTENSIVE[0])
            | (dia.mean(axis=1) >= HYPERTENSIVE[1])).to_numpy()
    hypertension = _yes(frame["BPQ020"]) | _yes(frame["BPQ050A"]) | high
    waist_in = pd.to_numeric(frame["BMXWAIST"], errors="coerce").to_numpy(dtype=float) / CM_PER_INCH
    waist = np.where(male, _level(waist_in, WAIST_CUTS["male"]),
                     _level(waist_in, WAIST_CUTS["female"]))
    size = np.maximum(_level(frame["BMXBMI"], BMI_CUTS), waist)
    size_points = np.select(
        [size == 3, size == 2, size == 1],
        [POINTS["extremely obese"], POINTS["obese"],
         POINTS["overweight, against not overweight or obese"]], 0)
    active = _yes(frame["PAQ650"]) | _yes(frame["PAQ665"])
    return (band + male * POINTS["male"] + _yes(frame["MCQ300C"]) * POINTS[
        "family history of diabetes"] + hypertension * POINTS["history of hypertension"]
        + size_points + active * POINTS["physically active"]).astype(int)


# ── the AUC, per participant and weighted ────────────────────────────────────


def weighted_auc(positive: Any, ranking: Any, weights: Any) -> float:
    """Σ over case–control pairs of w_case·w_control·(1 if the case ranks higher, ½ if tied), over
    Σ w_case · Σ w_control: the weighted Mann–Whitney statistic, by sorting (no pair loop)."""
    y = np.asarray(positive, dtype=bool)
    s = np.asarray(ranking, dtype=float)
    w = np.asarray(weights, dtype=float)
    cases, controls = w[y].sum(), w[~y].sum()
    if not (cases > 0 and controls > 0):
        return float("nan")
    values, where = np.unique(s, return_inverse=True)
    case_w = np.bincount(where, weights=w * y, minlength=len(values))
    control_w = np.bincount(where, weights=w * ~y, minlength=len(values))
    below = np.concatenate([[0.0], np.cumsum(control_w)[:-1]])  # controls ranked lower
    return float((case_w * (below + 0.5 * control_w)).sum() / (cases * controls))


def jackknife(positive: Any, ranking: Any, weights: Any, strata: Any, psus: Any,
              domain: Any) -> tuple[float, float, int]:
    """The weighted AUC over ``domain``'s rows and its delete-one-PSU jackknife standard error
    (JKn): each replicate drops one PSU and multiplies its stratum's other PSUs' weights by
    n_h/(n_h − 1), over every row of the design (``domain`` marks the participants), and
    v = Σ_h (n_h − 1)/n_h Σ_j (θ_hj − θ)². Returns (θ, its standard error, the replicates);
    refused where a stratum has a single PSU."""
    w = np.asarray(weights, dtype=float)
    dom = np.asarray(domain, dtype=bool)
    st = pd.Series(np.asarray(strata)).astype(str).to_numpy()
    ps = pd.Series(np.asarray(psus)).astype(str).to_numpy()
    theta = weighted_auc(np.asarray(positive)[dom], np.asarray(ranking)[dom], w[dom])
    var, replicates = 0.0, 0
    for h in pd.unique(st):
        in_h = st == h
        units = pd.unique(ps[in_h])
        n = len(units)
        if n < 2:
            raise BenchmarkRefused(f"Stratum {h} has a single PSU, so the jackknife cannot drop "
                                   "one: no design-based interval.")
        for j in units:
            rw = w.copy()
            rw[in_h & (ps == j)] = 0.0
            rw[in_h & (ps != j)] *= n / (n - 1)
            t = weighted_auc(np.asarray(positive)[dom], np.asarray(ranking)[dom], rw[dom])
            var += (n - 1) / n * (t - theta) ** 2
            replicates += 1
    return theta, float(np.sqrt(var)), replicates


def aucs(positive: Any, ranking: Any, weights: Any, strata: Any, psus: Any,
         domain: Any) -> dict[str, Any]:
    """The AUC per participant with DeLong's interval, and weighted with the jackknife's, as the
    engine's intervals are written (``models.performance.Interval``)."""
    from turbotab.core.models.performance import auc_interval, interval
    dom = np.asarray(domain, dtype=bool)
    y, s = np.asarray(positive, dtype=bool)[dom], np.asarray(ranking, dtype=float)[dom]
    per = auc_interval(y, s)
    theta, se, k = jackknife(positive, ranking, weights, strata, psus, domain)
    weighted = interval(theta, se, f"jackknife, delete one PSU within strata ({k} replicates)",
                        (0.0, 1.0))
    return {"per_participant": per.model_dump(mode="json"),
            "weighted": weighted.model_dump(mode="json")}


# ── the participants: the record's own ───────────────────────────────────────


@dataclass
class Participants:
    cohort: np.ndarray  # the table's rows the record's participant flow keeps
    held_out: np.ndarray  # of the table's rows: the cohort at or after the cut-point
    steps: list[dict[str, Any]]
    cut: dict[str, Any]
    opening: Any  # the open_seal record


def reported_auc(text: str | None = None) -> dict[str, Any]:
    """Bang's reported AUC on its own later cycle, as the notes' Table 3 prints it: refused unless
    the notes print the row once."""
    found = REPORTED.findall(NOTES.read_text(encoding="utf-8") if text is None else text)
    if len(found) != 1:
        raise BenchmarkRefused(f"{I.BANG_NOTES} prints {len(found)} validation rows with an AUC, "
                               "not one.")
    cycle, n, auc = found[0]
    return {"auc": float(auc), "cycle": cycle, "n": int(n.replace(",", "")),
            "what": f"Bang et al. 2009's score on its own later cycle, NHANES {cycle}",
            "source": f"{I.BANG_NOTES} §5.3 (the paper's Table 3)"}


def items(reported: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The score's items as the output prints them (:data:`ITEMS`): each rule, the family
    history's naming Bang's validation cycle as ``reported`` (:func:`reported_auc`, the output's
    own ``reported``) reads it from the notes, and the passages each rests on."""
    return [{"item": i["item"], "rule": i["rule"].replace(VALIDATION_CYCLE, reported["cycle"]),
             "passages": [{"source": u, "quote": q} for u, q in i["passages"]]} for i in ITEMS]


def _state_at_opening(log: bytes) -> tuple[Any, Any]:
    from turbotab.core.decisions import fold
    from turbotab.core.export import replay
    lines = replay.log_lines(log)
    opening = replay.live_record(lines, "open_seal")
    if opening is None:
        raise BenchmarkRefused("The record never opened its held-out rows: it reports no held-out "
                               "result to compare with.")
    return fold([r for r, _ in lines if r.seq < opening.seq]), opening


def _needs_the_working_table(state: Any, reads: set[str]) -> str | None:
    """Why the engine's flow could keep other rows than the benchmark's over the table as
    uploaded, or None: a joined file; a landmark (a step of the engine's flow the benchmark does
    not take); a repair recorded without its settings (the engine reads them from the finding's
    offer, ``repairs._params``, which the bundle does not carry); or a repair whose settings
    rewrite a column the flow reads, a reading of text as numbers among them (its settings name
    the column it reads).

    A consumers-only domain is not read: the engine restricts the rows by one under Estimate only
    (``methods.exposure_form.domain_columns``), and :func:`participants` has already refused any
    record that is not a prediction."""
    from turbotab.core import repairs
    from turbotab.core.stages import rows as RW
    if getattr(state, "joins", None):
        return "the record joins another file to the table"
    if RW.landmark_of(state):
        return "the record counts its follow-up from a landmark, a step the benchmark does not take"
    touched: set[str] = set()
    for fid, d in (getattr(state, "findings", None) or {}).items():
        if d.action != "applied":
            continue
        if not d.params:
            return f"the repair of {fid} is recorded without its settings"
        touched |= repairs.option_columns(fid, str(d.option), d.params, ("values", "columns"))
    hit = sorted(touched & reads)
    return f"a repair rewrites {', '.join(hit)}, which the flow reads" if hit else None


def participants(table: pd.DataFrame, log: bytes, card: Mapping[str, Any]) -> Participants:
    """The record's own held-out participants (the module docstring): the engine's fold of the log
    before its first opening, the engine's participant flow over ``table`` under it, and the rows
    at or after the record's cut-point, held to the card's and to the opening's count."""
    from turbotab.core.decisions import missing_strategy
    from turbotab.core.stages import rows as RW
    state, opening = _state_at_opening(log)
    outcome = card["outcome"]["column"]
    if state.purpose != "prediction" or state.target != outcome:
        raise BenchmarkRefused(f"The record predicts {state.target!r} under "
                               f"{state.purpose!r}, not {outcome!r} under prediction.")
    held = card.get("held_out") or {}
    cut = getattr(state.split, "cut_point", None) if state.split is not None else None
    if cut is None or cut.column != held.get("column") or float(cut.at) != float(held.get("at")):
        named = "no cut-point" if cut is None else f"a cut-point at {cut.at} on {cut.column}"
        raise BenchmarkRefused(f"The record holds out its rows by {named}, not the card's "
                               f"cut-point at {held.get('at')} on {held.get('column')}: its "
                               "held-out rows are not the benchmark's.")
    columns = {str(c): {"name": str(c), "n_missing": int(table[c].isna().sum()),
                        "n_unique": int(table[c].nunique(dropna=True)),
                        "dtype": "integer" if pd.api.types.is_integer_dtype(table[c]) else
                        "numeric" if pd.api.types.is_numeric_dtype(table[c]) else "text"}
               for c in table.columns}
    ingest = {"columns": list(columns.values())}
    needed, _, gappy = RW.cohort_inputs(state, ingest)
    why = _needs_the_working_table(state, set(needed) | set(gappy))
    if why:
        raise BenchmarkRefused(f"The benchmark reads the table as uploaded, and {why}.")
    missing = [c for c in [*needed, *gappy, cut.column] if c not in table.columns]
    if missing:
        raise BenchmarkRefused(f"The table has no column {', '.join(missing)}.")
    frame = table[needed].copy()
    steps, kept = RW.cohort_flow(
        frame, target=state.target, rules=state.exclusions, missing=missing_strategy(state),
        predictor_columns=gappy, missing_frame=table[gappy] if gappy else None,
        repairs=RW.repair_rules(state), reference=None, landmark=None, domain=())
    kept = np.sort(np.asarray(kept, dtype=np.int64))
    at = pd.to_numeric(table[cut.column], errors="coerce").to_numpy(dtype=float)
    out = kept[at[kept] >= float(cut.at)]
    n = getattr(opening.decision, "n_holdout", None)
    if n is not None and int(n) != len(out):
        raise BenchmarkRefused(f"The record's opening held out {int(n):,} rows, and its flow under "
                               f"the cut-point keeps {len(out):,}: the participants cannot be "
                               "taken as the record's.")
    return Participants(cohort=kept, held_out=out, steps=steps,
                        cut={"column": cut.column, "at": cut.at}, opening=opening)


# ── the benchmark ────────────────────────────────────────────────────────────


def weight_of(cycle: str = I.GLUCOSE_TEST) -> str:
    """The fasting subsample's weight variable of a cycle (``intake.GLUCOSE_WEIGHTS``)."""
    return I.GLUCOSE_WEIGHTS[cycle]


def benchmark(bundle: Path, record: Path | None = None) -> dict[str, Any]:
    from turbotab.core.export import replay
    meta = I.load(I.GLUCOSE_ID)
    manifest, card = meta["manifest"], meta["card"]
    table_path = I.REPO / manifest["table"]["file"]
    if not table_path.is_file() or I.sha256(table_path) != manifest["table"]["sha256"]:
        raise BenchmarkRefused(f"{manifest['table']['file']} is not on disk as the manifest "
                               "records it: run the intake.")
    data = Path(bundle).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if record is not None:
        held = (json.loads(Path(record).read_text("utf-8")).get("export") or {}).get("sha256")
        if held != digest:
            raise BenchmarkRefused(f"The bundle's SHA-256 is {digest}, the record's export is "
                                   f"{held}: not the record's bundle.")
    try:
        files, prov = replay.load_bundle(bundle)
        replay.verify_inputs(prov, data=str(table_path))
    except replay.Refused as exc:
        raise BenchmarkRefused(str(exc)) from None
    problems = item_problems(I.manifest_pages({"files": [
        *manifest["files"], *I.load("bang2009")["manifest"]["files"]]}))
    if problems:
        raise BenchmarkRefused("The score's items are not as their pages state: "
                               + "; ".join(problems))
    table = pd.read_parquet(table_path)
    who = participants(table, files["decisions.jsonl"], card)
    test = table[I.GLUCOSE_CYCLE].to_numpy() == I.GLUCOSE_CYCLES[I.GLUCOSE_TEST][0]
    if not test[who.held_out].all():
        raise BenchmarkRefused("A held-out participant is not of the test cycle.")
    weight = weight_of()
    w = pd.to_numeric(table[weight], errors="coerce").to_numpy(dtype=float)
    if np.isnan(w[who.held_out]).any():
        raise BenchmarkRefused(f"A held-out participant has no fasting weight ({weight}).")
    domain = np.zeros(len(table), dtype=bool)
    domain[who.held_out] = True
    rows = np.flatnonzero(test)  # the test cycle's whole design
    glucose = pd.to_numeric(table[I.GLUCOSE_OUTCOME], errors="coerce").to_numpy(dtype=float)
    positive = glucose >= THRESHOLD
    points = score(table)
    out = aucs(positive[rows], points[rows], np.nan_to_num(w[rows]), table["SDMVSTRA"].to_numpy()[
        rows], table["SDMVPSU"].to_numpy()[rows], domain[rows])
    reported = reported_auc()
    return {
        "benchmark": ("Bang et al. 2009's points score, adapted to the later cycle's questions, "
                      "on the record's own held-out participants"),
        "bundle": {"path": str(bundle), "sha256": digest},
        "participants": {
            "cohort": int(len(who.cohort)), "held_out": int(len(who.held_out)),
            "cases": int(positive[who.held_out].sum()), "cut_point": who.cut,
            "opening": int(who.opening.seq),
            "outcome": (f"{I.GLUCOSE_OUTCOME} at or above {THRESHOLD} mg/dL"),
            "flow": [{k: s.get(k) for k in ("key", "label", "n")} for s in who.steps]},
        "auc": {**out, "weight": weight},
        "score": {"points": {k: v for k, v in POINTS.items()},
                  "range": [int(points[who.held_out].min()), int(points[who.held_out].max())],
                  "items": items(reported)},
        "reported": reported,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.benchmark", description=HELP)
    parser.add_argument("bundle", type=Path, help="the run's export.zip")
    parser.add_argument("--record", type=Path, help="the run's corpus record, whose export's "
                                                    "SHA-256 the bundle must have")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        result = benchmark(args.bundle, args.record)
    except BenchmarkRefused as exc:
        print(json.dumps({"refused": exc.message}, ensure_ascii=False))
        return 2
    text = json.dumps(result, indent=1, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
