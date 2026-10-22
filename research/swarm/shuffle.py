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
none, the donor's regime's live markers). Everything else stays the record's own, so a decision's
materiality is read beside markers that are not its dataset's: a pattern the claim pipeline finds
in the copy comes from its prior, or from the planted effect.

**The planted effect.** ``--planted K`` records are appended to the copy, each a clone of a record
of the copy (the template, drawn from the seed) under a new neutral run id and dataset id, with one
true effect written in by construction: where the live marker M is below a threshold t (half the
clones), the decision D (one of the template's own, by the analyst) moved its fit's primary
score by a material amount (its materiality's movement, and the fit's signed change, whose
interval excludes zero); where M is above t (the other half), by none. M is a numeric live marker
of the template's regime, t the template's own value of it; both are drawn from the seed.

**The sealed file** (``--sealed``, which must lie outside the copy) lists the planted records by
run id, with M, t, D, the direction and the size, the seed and the derangement, so the theorist,
who reads only the copy, never sees which records are planted; the pipeline's routing of the
planted claim is scored against it after the cycle. Nothing in the corpus itself is written.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import secrets
import sys
from pathlib import Path
from typing import Any, Sequence

from research.swarm import record as R
from research.swarm import vocabulary as V

FAILED = "failed"
MATERIAL = 1.0  # the planted movement in-region, in SDs of the fold differences (``movement``)
NONE = 0.0
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


def shuffled(records: list[dict[str, Any]], mapping: dict[str, str]) -> list[dict[str, Any]]:
    """Each record with every marker of its donor (the module docstring)."""
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
            new["dataset"]["declared"] = copy.deepcopy(donor["dataset"].get("declared") or [])
            fits = donor.get("attempts") or []
            for j, attempt in enumerate(new.get("attempts") or []):
                source = fits[j % len(fits)] if fits else None
                markers = (source or {}).get("markers") if source else _live(donor["regime"])
                attempt["markers"] = copy.deepcopy(markers)
                attempt["markers_why"] = (None if markers is not None else
                                          (source or {}).get("markers_why")
                                          or "The donor's fit held no live markers.")
            R.validate(new)
            out.append(new)
    return sorted(out, key=lambda d: d["run"]["id"])


def _numeric_markers(rec: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in _live(rec["regime"]) if isinstance(m.get("value"), NUMERIC)
            and not isinstance(m.get("value"), bool)]


def planted(records: list[dict[str, Any]], k: int, rng: random.Random
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``k`` records carrying one true effect (the module docstring), and the effect itself."""
    if k < 2:
        raise ShuffleRefused("A planted effect needs at least two records: one on each side of "
                             "its threshold.")
    templates = [r for r in records if _numeric_markers(r) and r.get("attempts")
                 and any(d["source"] == "analyst" for d in r["decisions"])]
    if not templates:
        raise ShuffleRefused("No record of the copy has a numeric live marker, a fit and an "
                             "analyst's decision to plant an effect in.")
    template = rng.choice(templates)
    marker = rng.choice(_numeric_markers(template))
    decision = rng.choice([d for d in template["decisions"] if d["source"] == "analyst"])
    t = float(marker["value"])
    fit = template["attempts"][-1]
    effect = {"decision": decision["kind"], "decision_id": decision["id"],
              "marker": marker["name"], "threshold": t,
              "direction": "material below the threshold, none above", "size": MATERIAL,
              "template": template["run"]["id"]}
    out = []
    for n in range(k):
        inside = n % 2 == 0
        value = t * (0.5 if inside else 1.5) if t else (-1.0 if inside else 1.0)
        rec = copy.deepcopy(template)
        rec["run"]["id"] = f"run_{900 + n:03d}-{rng.getrandbits(128):032x}"
        rec["dataset"]["id"] = f"ds_{rng.getrandbits(32):08x}"
        for m in rec["regime"]["markers"]:
            if m["name"] == marker["name"]:
                whole = isinstance(marker["value"], int) and value == int(value)
                m["value"] = int(value) if whole else value
        size = MATERIAL if inside else NONE
        family = (fit.get("families") or ["linear"])[0]
        for d in rec["decisions"]:
            if d["id"] == decision["id"]:
                d["materiality"] = {"source": "fits", "attempt": fit["id"], "movement": [{
                    "family": family, "against": fit["id"], "why": None,
                    "movement": {"instrument": "paired_folds", "regime": "realized",
                                 "value": size, "band": 2 if inside else 0}}]}
        for a in rec["attempts"]:
            a["markers"] = [dict(m) for m in _live(rec["regime"])]
            a["markers_why"] = None
        rec["attempts"][-1]["change"] = [{
            "family": family, "metric": fit.get("primary_metric") or "score",
            "against": fit["id"], "paired": True, "difference": size / 10,
            "better": True if inside else None, "se": 0.01, "df": 4.0,
            "ci_low": size / 10 - 0.03, "ci_high": size / 10 + 0.03,
            "excludes_zero": inside, "folds": [5, 5], "half_width": 0.05, "why": None}]
        rec["attempts"][-1]["change_why"] = None
        R.validate(rec)
        out.append(rec)
    return out, effect


def write(corpus: Path, out: Path, sealed: Path, seed: int, k: int) -> dict[str, Any]:
    """The shuffled copy in ``out`` and the sealed file (the module docstring)."""
    out, sealed = Path(out).absolute(), Path(sealed).absolute()
    if out == Path(corpus).absolute() or out in Path(corpus).absolute().parents:
        raise ShuffleRefused("The copy is written apart from the corpus, never over it.")
    if sealed == out or out in sealed.parents:
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
