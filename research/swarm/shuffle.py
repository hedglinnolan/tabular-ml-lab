"""The shuffled control's corpus (SWARM_PLAN §9 step 5): a copy of the corpus whose markers are
permuted across datasets, with records carrying one planted effect appended to the copy only.

    python -m research.swarm.shuffle <corpus> --out <copy> --sealed <file> --seed N [--planted K]

**The permutation.** Every record of the corpus (``<corpus>/*.json``; a failed run's under
``failed/`` is not one) is read and validated (``record.validate``). The datasets are permuted by
a derangement drawn from the seed (Sattolo's algorithm: one cycle through every dataset, so no
dataset keeps its own markers; with fewer than two datasets there is nothing to permute, and the
copy is refused). Each record of a dataset takes every marker of a record of the dataset it is
mapped to, its donor (the donor dataset's records in run-id order, the i-th record taking the
i-th, cycling): the whole regime (its readings and its markers, live and post hoc), the dataset's
declared markers, and each fit's live markers (the donor's fit of the same index, cycling; with
none, the donor's regime's live markers), but the purpose and its loss (below). Everything else
stays the record's own, so a decision's materiality is read beside markers that are not its
dataset's: a pattern the claim pipeline finds in the copy comes from its prior, or from the
planted effect.

**The purpose and its loss stay the record's own.** The live markers of that axis (SWARM_PLAN §4:
the purpose, the primary score, the intended use and its thresholds, the estimand, the
multiplicity family; :func:`restates`) restate the record's own question and fits, which the copy
keeps its own, so they are kept as the question is: at the regime and at each fit, the record's
own (where its own fit held no markers, that fit holds none of the axis). A record of the copy
never shows a purpose, a primary score or an estimand its question and fits contradict, and a
claim keyed on the purpose is no more permuted than the question it restates.

**The planted effect.** ``--planted K`` records are appended to the copy, each a clone of one
record of the copy (the template, drawn from the seed) carrying one true effect written in by
construction: where the live marker M is below a threshold t (half the clones), the decision D
moved its fit's primary score materially (its fit's paired change has an interval excluding zero);
where M is at or above t (the other half), by little (an interval holding zero). Each planted
record holds a real record's markers whole, so no planted record holds a value, or a pair of
values, that no real record of its dataset holds; in form it differs from a real one only as a
clone does (below):

* **the template** is a Predict record whose last fit's whole change from the fit before is one
  decision of the analyst's, D (so D's materiality is that fit's movement, as the assembler reads
  it), and whose two fits scored the same folds with a signed change computed for a family (its
  share of rows a fold scores is the one the planted change uses); D is of a kind the theorist can
  propose about (``claim_cycle.proposal.proposable``: not one that records the protocol, which
  the validator refuses, and not ``set_selection``, which the theorist's prompt sets aside), so
  the planted effect's claim can be proposed; M is a numeric live marker whose value varies
  across the real records of the template's dataset in the copy (one constant across them, as
  their rows are, is never M), and t is one of those values above the least, so some of those
  records lie below it and the rest at or above it; a copy with no such template is refused;
* **the run** takes a run id as ``start`` draws one, a label and 32 hex digits: the label of a real
  record of the copy, as runs started together share one (``driver.new_run_id``), with every time
  the template holds moved by one offset to start beside that record's start; **the dataset** is
  the template's own, one of the copy's datasets, so its files, columns and markers agree with it;
* **the markers** are a real record's whole, those of a record of the template's dataset in the
  copy (drawn from the seed among those on the planted record's side of t, and named in the sealed
  file's ``markers_from``): its regime, readings and markers, and at each fit its fit of the same
  index's live markers (cycling), as a record of the copy takes its donor's; the purpose and its
  loss stay the template's own (:func:`restates`). So M's value, and every value beside it (the
  rows, the predictors per row, the regime's counts), is one that real record holds;
* **the effect** is in the last fit's fold scores of that family's primary score: the fit before's
  scores plus a constructed difference whose fold-to-fold spread is a share of theirs and whose
  mean, toward a better score by the metric's direction, is a drawn number of that spread (inside:
  past the corrected interval's reach; outside: a small one, either way). The fit's summary of
  that score, its movement against the fit before (``runs.movement``, the engine's), D's
  materiality and the fit's signed change (``signed.from_shares``, with the template's share) are
  recomputed from them, as the engine and the assembler compute a real record's; each planted
  change is against the fit before, ``better`` by the metric's direction.

What a clone keeps of its template's own is said in the sealed file's ``kept``: its reasons and its
account, its export's SHA-256, and its decisions and its earlier fits, which the claim cycle's view
serves. **The clones are accepted:** a reader of the view who compares records finds the planted
records beside their template, records of one dataset with one sequence of decisions and one first
fit, which runs by different analysts rarely share. The shuffle does not hide that cluster:
writing each planted record's decisions anew would make a run the engine never made. Which of a
record's fields the theorist reads is for the claim cycle's tools to decide.

**The sealed file** (``--sealed``, which must lie outside the copy and the corpus) lists the
planted records by run id, with M, t, D, the direction, each record's side, value of M, movement
and change and the real record whose markers it holds, the seed and the derangement, so the
theorist, who reads only the copy, never sees which records are planted; the pipeline's routing of
the planted claim is scored against it after the cycle. **The effect is identified only up to the
markers that separate the planted records** (:func:`separating`, the sealed file's
``separating``): since each planted record holds a real record's markers whole, a marker that
moves with M across the dataset's records (the rows analyzed beside the predictors per row) splits
the planted records' two sides as M does, so a claim drawn on it names the planted effect as well
as one drawn on M. The sealed file lists every such marker, at the regime and at the last fit (the
fit the effect is in), M first at t: a numeric marker where one threshold splits the two sides,
with that threshold and the side the material records lie on; a marker of words or true or false
where the two sides hold no value in common, with the values each side holds. A marker some
planted record holds no such value of (a missing value, or one with parts, which no region
bounds) is not one.
Nothing in the corpus itself is written: a copy at, under or above the corpus is refused, as is a
sealed file at or under the corpus or the copy.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from research.swarm import record as R
from research.swarm import vocabulary as V

FAILED = "failed"
NUMERIC = (int, float)


class ShuffleRefused(ValueError):
    """The corpus cannot be shuffled as asked, in words."""


def read_corpus(corpus: Path) -> list[dict[str, Any]]:
    """Every record of ``corpus``, validated, in run-id order."""
    out = []
    for path in sorted(Path(corpus).glob("*.json")):
        data = json.loads(path.read_text("utf-8"))
        R.validate(data)
        out.append(data)
    return sorted(out, key=lambda d: d["run"]["id"])


def derangement(items: Sequence[str], rng: random.Random) -> dict[str, str]:
    """A permutation of ``items`` that moves every one (Sattolo's algorithm: a uniformly drawn
    cycle through all of them)."""
    if len(items) < 2:
        raise ShuffleRefused(f"The corpus holds {len(items)} dataset; markers are permuted across "
                             f"datasets, so at least two are needed.")
    order = list(items)
    for i in range(len(order) - 1, 0, -1):
        j = rng.randrange(i)  # j < i: the one cycle
        order[i], order[j] = order[j], order[i]
    return {a: b for a, b in zip(items, order)}


def _live(regime: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in regime.get("markers") or [] if m.get("timing") == "live"]


ORDER = {name: i for i, name in enumerate(V.NAMES)}


def restates(m: dict[str, Any]) -> bool:
    """Whether the marker ``m`` is a live marker of the purpose and its loss, which restates its
    record's own question and fits (the module docstring)."""
    fixed = V.BY_NAME.get(str(m.get("name")))
    return fixed is not None and fixed.axis == "purpose" and fixed.timing == "live"


def with_own(markers: list[dict[str, Any]] | None, own: list[dict[str, Any]] | None
             ) -> list[dict[str, Any]] | None:
    """``markers``, another record's, with the purpose and its loss ``own``'s, the record's own,
    in the vocabulary's order (None where ``markers`` is None)."""
    if markers is None:
        return None
    kept = [m for m in markers if not restates(m)] + [m for m in own or () if restates(m)]
    return copy.deepcopy(sorted(kept, key=lambda m: ORDER.get(m.get("name"), len(ORDER))))


def shuffled(records: list[dict[str, Any]], mapping: dict[str, str]) -> list[dict[str, Any]]:
    """Each record with every marker of its donor but the purpose and its loss, its own (the
    module docstring)."""
    by: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by.setdefault(rec["dataset"]["id"], []).append(rec)
    out = []
    for dataset, own in by.items():
        donors = by[mapping[dataset]]
        for i, rec in enumerate(own):
            donor = donors[i % len(donors)]
            new = copy.deepcopy(rec)
            new["regime"] = copy.deepcopy(donor["regime"])
            new["regime"]["markers"] = with_own(donor["regime"]["markers"],
                                                rec["regime"]["markers"])
            new["dataset"]["declared"] = copy.deepcopy(donor["dataset"].get("declared") or [])
            fits = donor.get("attempts") or []
            for j, attempt in enumerate(new.get("attempts") or []):
                source = fits[j % len(fits)] if fits else None
                markers = (source or {}).get("markers") if source else _live(donor["regime"])
                attempt["markers"] = with_own(markers, rec["attempts"][j].get("markers"))
                attempt["markers_why"] = (None if markers is not None else
                                          (source or {}).get("markers_why")
                                          or "The donor's fit held no live markers.")
            R.validate(new)
            out.append(new)
    return sorted(out, key=lambda d: d["run"]["id"])


# The planted mean difference, in its fold-to-fold spread: inside, this many times what puts the
# corrected interval's edge at zero; outside, a share of a spread at most.
INSIDE = (1.5, 2.5)
OUTSIDE = (0.02, 0.25)
SPREAD = 0.5  # the planted difference's spread, as a share of the fit before's fold scores' SD
KEPT = ("the template's reasons, tags, wants and account", "its export and its SHA-256",
        "its fit keys and plan digests", "its server's pid and port",
        "its decisions and its earlier fits, which the view serves, so a reader who compares "
        "records finds the planted records beside their template")


def _number(v: Any) -> bool:
    return isinstance(v, NUMERIC) and not isinstance(v, bool) and math.isfinite(v)


def _value(rec: dict[str, Any], name: str) -> int | float | None:
    """The value of the regime's live marker ``name`` in ``rec``, where it is a number."""
    v = next((m.get("value") for m in _live(rec["regime"]) if m.get("name") == name), None)
    return v if _number(v) else None


def _varying(records: list[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    """Each numeric live marker of the regime whose value varies across ``records`` (one dataset's
    real records in the copy), with each record's value by run id, in the vocabulary's order."""
    names = {m["name"] for r in records for m in _live(r["regime"])}
    out = {}
    for name in sorted(names, key=lambda n: ORDER.get(n, len(ORDER))):
        held = {r["run"]["id"]: v for r in records if (v := _value(r, name)) is not None}
        if len(set(held.values())) > 1:
            out[name] = held
    return out


def _borrowed(rec: dict[str, Any], source: dict[str, Any]) -> None:
    """``rec`` (a planted record) with ``source``'s markers whole, the purpose and its loss its own
    (the module docstring)."""
    rec["regime"] = {**copy.deepcopy(source["regime"]),
                     "markers": with_own(source["regime"]["markers"], rec["regime"]["markers"])}
    fits = source.get("attempts") or []
    for j, attempt in enumerate(rec.get("attempts") or []):
        given = fits[j % len(fits)] if fits else None
        markers = given.get("markers") if given else _live(source["regime"])
        attempt["markers"] = with_own(markers, attempt.get("markers"))
        attempt["markers_why"] = (None if markers is not None else
                                  (given or {}).get("markers_why")
                                  or "The record its markers are taken from held none at this fit.")


BELOW = "material below the threshold, none at or above it"
ABOVE = "material above the threshold, none at or below it"
AT_VALUES = "material at the values listed, none at the others"


def _bounded(v: Any) -> bool:
    """Whether a region can bound the value ``v``: a number, a word or true or false
    (``proposal.Bound``); a marker's missing value, or one with parts, it cannot."""
    return isinstance(v, (str, bool)) or _number(v)


def _keyed(v: Any) -> str:
    """``v`` as two values compare: a number by its value, the rest by its JSON (true is not 1)."""
    return json.dumps(float(v) if _number(v) else v)


def _distinct(values: list[Any]) -> list[Any]:
    """``values`` once each (:func:`_keyed`), numbers first in their order, then the rest."""
    seen: dict[str, Any] = {}
    for v in values:
        seen.setdefault(_keyed(v), v)
    return sorted(seen.values(), key=lambda v: (not _number(v), v if _number(v) else _keyed(v)))


def _split(name: str, at: str, material: list[Any], other: list[Any],
           fixed: tuple[float, str] | None) -> dict[str, Any] | None:
    """How the live marker ``name`` separates the planted records' two sides, where it does, else
    None: numbers by one threshold, all of one side below it and all of the other at or above it
    (or the material side above it and the other at or below it), as the effect itself is cut
    (``fixed``: M's own threshold and direction, where it is M at the regime); words or true or
    false where the two sides hold no value in common. Each with the values each side holds."""
    if not material or not other or not all(_bounded(v) for v in material + other):
        return None
    mine, theirs = _distinct(material), _distinct(other)
    if all(_number(v) for v in mine + theirs):
        if fixed is not None:
            threshold, direction = fixed
        elif max(mine) < min(theirs):
            threshold, direction = min(theirs), BELOW
        elif min(mine) > max(theirs):
            threshold, direction = max(theirs), ABOVE
        else:
            return None
    elif {_keyed(v) for v in mine} & {_keyed(v) for v in theirs}:
        return None
    else:
        threshold, direction = None, AT_VALUES
    return {"marker": name, "at": at, "material": mine, "not_material": theirs,
            "threshold": threshold, "direction": direction}


def separating(planted: list[tuple[bool, dict[str, Any]]], marker: str, t: float
               ) -> list[dict[str, Any]]:
    """Every live marker that separates the planted records' two sides (``planted``: each record
    with whether it is on the material side), at the regime and at the last fit, the fit the
    effect is in: one on which a region alone holds every record of one side and none of the other
    (:func:`_split`), held with a value a region bounds by every planted record. M (``marker``, cut
    at ``t``) first, then the rest in the vocabulary's order, the regime's before the last fit's."""
    out = []
    for at in ("regime", "last fit"):
        values: dict[str, dict[bool, list[Any]]] = {}
        held: dict[str, int] = {}
        for material, rec in planted:
            markers = (rec["regime"]["markers"] if at == "regime"
                       else (rec.get("attempts") or [{}])[-1].get("markers"))
            for m in [m for m in markers or [] if m.get("timing") == "live"]:
                values.setdefault(m["name"], {True: [], False: []})[material].append(m.get("value"))
                held[m["name"]] = held.get(m["name"], 0) + 1
        names = sorted((n for n in values if held[n] == len(planted)),
                       key=lambda n: (n != marker, ORDER.get(n, len(ORDER))))
        for name in names:
            fixed = (t, BELOW) if (name, at) == (marker, "regime") else None
            got = _split(name, at, values[name][True], values[name][False], fixed)
            if got is not None:
                out.append(got)
    return out


def _computed(change: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c["family"]: c for c in change or () if c.get("difference") is not None
            and c.get("paired") and c.get("test_share")}


def _plantable(rec: dict[str, Any]) -> tuple[dict[str, Any], list[str]] | None:
    """The decision D and the families a planted effect can be written in, where ``rec`` can be a
    template (the module docstring); else None."""
    from research.swarm.claim_cycle.proposal import proposable

    attempts = rec.get("attempts") or []
    if rec["question"].get("purpose") != "prediction" or len(attempts) < 2:
        return None
    before, fit = attempts[-2], attempts[-1]
    seqs = fit.get("changed") or []
    loose = [s for s in (fit.get("withdrawn") or []) + (fit.get("reinstated") or [])
             if s.get("by") is None]
    decision = next((d for d in rec["decisions"] if len(seqs) == 1 and d["seq"] == seqs[0]), None)
    if decision is None or decision["source"] != "analyst" or loose:
        return None
    if not proposable(decision["kind"]):
        return None
    if fit.get("folds") is None or fit.get("folds") != before.get("folds"):
        return None
    metric = fit.get("primary_metric")
    families = []
    for family, c in _computed(fit.get("change")).items():
        now = ((fit.get("scores") or {}).get(family) or {}).get(metric) or {}
        then = ((before.get("scores") or {}).get(family) or {}).get(metric) or {}
        if c.get("against") == before["id"] and c.get("metric") == metric and \
                isinstance(now.get("folds"), list) and len(now["folds"]) == len(then.get("folds") or ()):
            families.append(family)
    return (decision, families) if families else None


def _standard(rng: random.Random, n: int) -> list[float]:
    """``n`` draws, standardized to mean 0 and SD 1 exactly (n ≥ 2)."""
    xs = [rng.gauss(0.0, 1.0) for _ in range(n)]
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) or 1.0
    return [(x - m) / sd for x in xs]


def _rescored(score: dict[str, Any], folds: list[float], metric: str) -> dict[str, Any]:
    """``score`` (a metric's summary, ``metrics.summarize``'s) with these fold scores: its mean,
    SD, estimate and repeats' SD read from them, its interval moved with its estimate and kept
    inside the metric's range as the summary keeps it."""
    from turbotab.core.models.metrics import LOWER_IS_BETTER

    out = dict(score)
    repeats = int(score.get("repeats") or 1)
    k = len(folds) // repeats
    each = [sum(folds[r * k:(r + 1) * k]) / k for r in range(repeats)]
    estimate = sum(each) / repeats
    shift = estimate - float(score["estimate"])
    m = sum(folds) / len(folds)
    out.update(folds=folds, mean=m, estimate=estimate,
               sd=math.sqrt(sum((x - m) ** 2 for x in folds) / (len(folds) - 1)),
               repeat_sd=(math.sqrt(sum((x - estimate) ** 2 for x in each) / (repeats - 1))
                          if repeats > 1 else score.get("repeat_sd")))
    lo, hi = score.get("ci_low"), score.get("ci_high")
    if lo is not None and hi is not None:
        lo, hi = float(lo) + shift, float(hi) + shift
        if metric in ("auc", "accuracy", "macro_f1", "brier"):
            lo, hi = max(lo, 0.0), min(hi, 1.0)
        elif metric == "r2":
            hi = min(hi, 1.0)
        elif metric in LOWER_IS_BETTER:
            lo = max(lo, 0.0)
        out.update(ci_low=lo, ci_high=hi)
    return out


def _moved(times: Any, offset: timedelta) -> Any:
    """``times`` (a record's JSON) with every time in it moved by ``offset``, each in its form."""
    if isinstance(times, dict):
        return {k: _moved(v, offset) for k, v in times.items()}
    if isinstance(times, list):
        return [_moved(v, offset) for v in times]
    if not isinstance(times, str):
        return times

    def move(m: Any) -> str:
        text = m.group()
        t = datetime.fromisoformat(text.replace(" ", "T")) + offset
        out = t.isoformat(sep=text[10], timespec="microseconds" if "." in text else "seconds")
        return out.replace("+00:00", "Z") if text.endswith("Z") else out

    return R.STAMP.sub(move, times)


def _started(rec: dict[str, Any]) -> datetime:
    t = datetime.fromisoformat(str(rec["run"]["started"]).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def planted(records: list[dict[str, Any]], k: int, rng: random.Random
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``k`` records carrying one true effect (the module docstring), and the effect itself."""
    from turbotab.core import runs
    from turbotab.core.models.metrics import higher_is_better

    from research.swarm import signed as S
    from research.swarm.driver import LABEL_STEM

    if k < 2:
        raise ShuffleRefused("A planted effect needs at least two records: one on each side of "
                             "its threshold.")
    siblings: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        siblings.setdefault(r["dataset"]["id"], []).append(r)
    varying = {d: _varying(rs) for d, rs in siblings.items()}
    templates = [(r, got) for r in records if (got := _plantable(r)) is not None
                 and varying[r["dataset"]["id"]]]
    if not templates:
        raise ShuffleRefused("No record of the copy is a Predict run whose last fit's whole change "
                             "is one decision of the analyst's, of a kind the theorist can propose "
                             "about, paired with a signed change over the same folds, on a dataset "
                             "with a numeric live marker whose value varies across the records of "
                             "its dataset in the copy, to plant an effect in.")
    template, (decision, families) = rng.choice(templates)
    dataset = template["dataset"]["id"]
    name = rng.choice(list(varying[dataset]))
    values = varying[dataset][name]  # each real record's value of M, by run id
    cuts = sorted(set(values.values()))
    t = cuts[rng.randrange(1, len(cuts))]  # above the least, so both sides hold a real record
    sides = {below: [r for r in siblings[dataset] if r["run"]["id"] in values
                     and (values[r["run"]["id"]] < t) == below] for below in (True, False)}
    family = rng.choice(families)
    before, fit = template["attempts"][-2], template["attempts"][-1]
    metric = fit["primary_metric"]
    higher = higher_is_better(metric)
    share = _computed(fit["change"])[family]["test_share"]
    then = before["scores"][family][metric]
    base = [float(x) for x in then["folds"]]
    j = len(base)
    sd = math.sqrt(sum((x - sum(base) / j) ** 2 for x in base) / (j - 1)) if j > 1 else 0.0
    spread = SPREAD * (sd or 0.01 * abs(sum(base) / j) or 0.01)
    reach = S._t(j - 1, S.LEVEL) * math.sqrt(1.0 / j + float(share[0]))  # mean/SD at the edge
    stamped = [r for r in records if r["run"]["id"].startswith(LABEL_STEM)] or records
    effect = {"decision": decision["kind"], "decision_id": decision["id"],
              "marker": name, "threshold": t, "family": family, "metric": metric,
              "direction": BELOW, "template": template["run"]["id"], "kept": list(KEPT),
              "records": []}
    out = []
    for n in range(k):
        inside = n % 2 == 0
        beside = rng.choice(stamped)
        offset = (_started(beside) - _started(template)
                  + timedelta(seconds=rng.randint(0, 120)))
        # Every time the record holds but its dataset's (each record of a dataset holds the same).
        rec = {key: (copy.deepcopy(v) if key == "dataset" else _moved(copy.deepcopy(v), offset))
               for key, v in template.items()}
        label = beside["run"]["id"].rsplit("-", 1)[0]
        rec["run"]["id"] = f"{label}-{rng.getrandbits(128):032x}"
        source = rng.choice(sides[inside])
        _borrowed(rec, source)
        value = values[source["run"]["id"]]
        size = rng.uniform(*INSIDE) * reach if inside else rng.uniform(*OUTSIDE) * rng.choice((-1, 1))
        sign = 1.0 if higher else -1.0  # toward a better score
        after = [b + sign * spread * (size + e) for b, e in zip(base, _standard(rng, j))]
        last, previous = rec["attempts"][-1], rec["attempts"][-2]
        last["scores"][family][metric] = _rescored(last["scores"][family][metric], after, metric)
        moved, why = runs.movement(
            runs.FitRecord.model_validate({f: previous.get(f) for f in runs.FitRecord.model_fields
                                           if f in previous}),
            purpose="prediction", target=last.get("target"), metric=metric,
            folds=last.get("folds"), seed=last.get("seed"), scores=last["scores"])
        last["movement"] = [m.model_dump(mode="json") for m in moved]
        last["movement_why"] = why
        for d in rec["decisions"]:
            if d["id"] == decision["id"]:
                d["materiality"] = {"source": "fits", "attempt": last["id"],
                                    "movement": last["movement"]}
        changed = S.from_shares(last["scores"][family][metric], previous["scores"][family][metric],
                                same_folds=True, higher_is_better=higher,
                                shares=(share[0], share[1]))
        last["change"] = [({"family": family, "metric": metric, "against": previous["id"],
                            **changed} if c["family"] == family else c) for c in last["change"]]
        R.validate(rec)
        out.append(rec)
        fm = next(m for m in last["movement"] if m["family"] == family)
        effect["records"].append({"run": rec["run"]["id"], "below": inside, "value": value,
                                  "markers_from": source["run"]["id"],
                                  "movement": fm["movement"]["value"],
                                  "difference": changed["difference"],
                                  "excludes_zero": changed["excludes_zero"]})
    effect["separating"] = separating([(r["below"], rec) for r, rec in zip(effect["records"], out)],
                                      name, t)
    return out, effect


def _within(path: Path, folder: Path) -> bool:
    """Whether ``path`` is ``folder`` or lies under it (both resolved)."""
    return path == folder or folder in path.parents


def write(corpus: Path, out: Path, sealed: Path, seed: int, k: int) -> dict[str, Any]:
    """The shuffled copy in ``out`` and the sealed file (the module docstring)."""
    home = Path(corpus).resolve()
    out, sealed = Path(out).resolve(), Path(sealed).resolve()
    if _within(out, home) or _within(home, out):
        raise ShuffleRefused("The copy is written apart from the corpus, never in it, over it or "
                             "above it.")
    if _within(sealed, home):
        raise ShuffleRefused("The sealed file lies outside the corpus: the theorist's real cycle "
                             "reads the corpus.")
    if _within(sealed, out):
        raise ShuffleRefused("The sealed file lies outside the copy: the theorist reads the copy.")
    if out.exists() and any(out.iterdir()):
        raise ShuffleRefused(f"{out} is not empty; a shuffled copy is written once.")
    if sealed.exists():
        raise ShuffleRefused(f"{sealed} exists; a sealed file is written once.")
    rng = random.Random(seed)
    records = read_corpus(corpus)
    datasets = sorted({r["dataset"]["id"] for r in records})
    mapping = derangement(datasets, rng)
    copy_ = shuffled(records, mapping)
    plants, effect = planted(copy_, k, rng) if k else ([], None)
    out.mkdir(parents=True, exist_ok=True)
    for rec in [*copy_, *plants]:
        (out / f"{rec['run']['id']}.json").write_text(json.dumps(rec, indent=1, sort_keys=True)
                                                       + "\n", "utf-8")
    body = {"seed": seed, "vocabulary": V.DIGEST, "derangement": mapping,
            "records": len(copy_), "planted": {"records": [r["run"]["id"] for r in plants],
                                               "effect": effect}}
    sealed.parent.mkdir(parents=True, exist_ok=True)
    with open(sealed, "x", encoding="utf-8") as fh:
        json.dump(body, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.shuffle",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="the seed (default: one drawn now and written in the sealed file)")
    parser.add_argument("--planted", type=int, default=6, help="records carrying the effect")
    args = parser.parse_args(argv)
    seed = args.seed if args.seed is not None else secrets.randbelow(2 ** 31)
    try:
        body = write(args.corpus, args.out, args.sealed, seed, args.planted)
    except ShuffleRefused as exc:
        print(json.dumps({"refused": str(exc)}))
        return 1
    print(json.dumps({"copy": str(args.out), "records": body["records"],
                      "planted": len(body["planted"]["records"]), "sealed": str(args.sealed)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
