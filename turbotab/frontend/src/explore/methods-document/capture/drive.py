"""Capture the methods-document prototype's moments from the real server (not part of the app).

    TURBOTAB_HOME=$(mktemp -d) TURBOTAB_WORKERS=2 OMP_NUM_THREADS=2 \
        venv/bin/python -m turbotab.server --port 8971                                # repo root
    venv/bin/python turbotab/frontend/src/explore/methods-document/capture/drive.py \
        http://127.0.0.1:8971 <raw-out-dir>
    venv/bin/python turbotab/frontend/src/explore/methods-document/capture/trim.py <raw-out-dir>

The drive is the shared scenario's (``../methods-shared/scenario.py``: ``run_inference`` and
``run_prediction``), so the prototype records exactly the answers the other two prototypes record.
This script only watches: at each of the scenario's moments it saves the ProjectView, the methods
text, the readings card, every computed stage's artifact and the previews of the options that
moment's slot offers (a preview records nothing). Before the plan is locked no estimate stage is
fetched: fetching one is what records the lock (``turbotab/core/plan_lock.py``); the scenario fetches
it, after the moment "ready".

In process, beside the drive: the disjunctive cause criterion's derivations for every answer to its
three questions (``estimand.derive``, a pure function), which the adjustment slot shows as the user
answers a column the pack does not guess.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "methods-shared"))

import scenario as S  # noqa: E402

ENERGY_METHODS = ["standard", "residual", "density_multivariate", "none", "residual_energy_dropped",
                  "density", "all_components", "partition"]


def snap(out: Path, prefix: str, moment: str, j: S.Journey, previews: dict[str, Any] | None = None,
         **extra: Any) -> None:
    j.wait_quiet()
    view = j.view()
    stages: dict[str, Any] = {}
    for name, status in view["stages"].items():
        if status["status"] in ("idle", "blocked"):
            continue
        got = j.stage(name)  # None for an estimate stage before the lock
        if got is not None:
            stages[name] = got
    data = {
        "moment": moment,
        "view": j.view(),
        "methods": j.methods(),
        "readings": j.readings(),
        "stages": stages,
        "previews": previews or {},
        **extra,
    }
    name = moment.replace(":", "-")
    (out / f"{prefix}-{name}.json").write_text(json.dumps(data, default=str))
    print(f"{prefix}-{name}: {len(view['decisions'])} decisions, {len(stages)} stages, "
          f"{len(previews or {})} previews", flush=True)


def inference(client: S.Client, out: Path) -> None:
    def at(moment: str, j: S.Journey) -> None:
        previews: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        if moment == "exclusions":
            proposals = j.artifact("proposals")
            previews["keep_every_row"] = j.preview({"kind": "set_exclusions", "rules": []})
            for o in proposals["exclusions"]:
                previews[o["key"]] = j.preview({"kind": "set_exclusions", "rules": [o["rule"]]})
        elif moment == "readings":
            # The column summaries a reading's evidence is drawn from (no outcome relationship).
            (out / "inf-columns.json").write_text(json.dumps(j.get("/columns")))
        elif moment == "estimand":
            previews["sugar_substitution"] = j.preview(S.estimand(contrast="substitution"))
            previews["sugar_addition"] = j.preview(S.estimand(contrast="addition"))
        elif moment == "energy":
            for m in ENERGY_METHODS:
                previews[m] = j.preview(S.energy(m))
        elif moment == "locked":
            extra["plan"] = j.get("/plan")
        snap(out, "inf", moment, j, previews, **extra)

    S.run_inference(client, at)


def prediction(client: S.Client, out: Path) -> None:
    def at(moment: str, j: S.Journey) -> None:
        if moment == "fitted":
            snap(out, "pred", moment, j)

    S.run_prediction(client, at)


def derivations() -> dict[str, Any]:
    """The criterion's verdict for every answer to its three questions, for a total effect."""
    from turbotab.core import estimand

    out: dict[str, Any] = {}
    for a in itertools.product(("yes", "no", "unknown"), repeat=3):
        d = estimand.derive(dict(zip(("causes_exposure", "causes_outcome", "after_exposure"), a)),
                            "total")
        out[",".join(a)] = {"role": d.role, "words": estimand.ROLE_WORDS[d.role],
                            "adjusted": d.adjusted, "secondary": d.secondary, "why": d.why}
    return out


def main() -> None:
    base, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    client = S.Client(base)
    inference(client, out)
    prediction(client, out)
    (out / "teaching.json").write_text(client.get("/api/teaching").text)
    (out / "derive.json").write_text(json.dumps(derivations()))


if __name__ == "__main__":
    main()
