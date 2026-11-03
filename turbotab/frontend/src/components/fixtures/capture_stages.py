"""The engine's stage words that the failure components' tests read (Failure, JobChips,
StageRetry, StageRefusal, quest/Fit), captured from the engine so that no expected word is retyped
beside the constant that holds it. Run from the repository root:

    PYTHONPATH=. OMP_NUM_THREADS=2 TURBOTAB_WORKERS=2 scripts/py \
        turbotab/frontend/src/components/fixtures/capture_stages.py

`labels`: each stage's label as the engine's graph declares it (turbotab/core/stages/__init__.py
`build_graph`); a job's JobView serves the same words (`label`) while the stage runs.

`statuses`: stage statuses as `Engine.status` serves them, on the toy graph of the typed fit-time
refusal test (turbotab/core/tests/test_typed_fit_refusal.py `refusing_graph`): `R` refuses in
words with one way forward that is a decision and one that is not; `AR` waits on `RH`, which
refuses so in a job worker; `C` fails with the same words untyped (a crash, no refusal); `AC` waits
on it; `U` waits on an unsettled reading, with its two confirmations as ways forward.
"""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

from turbotab.core.graph import Engine
from turbotab.core.jobs import JobRunner
from turbotab.core.stages import build_graph
from turbotab.core.tests.test_graph import Projects, RecordingBus, wait_for
from turbotab.core.tests.test_typed_fit_refusal import GRAPH
from turbotab.core.tests.toy_stages import ToyState

OUT = Path(__file__).resolve().parent / "stages.json"
SERVED = ("R", "RH", "AR", "C", "CH", "AC", "U")


def labels() -> dict[str, str]:
    return {stage.name: stage.label or stage.name for stage in build_graph().stages()}


def statuses() -> dict[str, dict]:
    with tempfile.TemporaryDirectory() as root, JobRunner(workers=1) as runner:
        projects, bus = Projects(Path(root)), RecordingBus()
        engine = Engine(graph_factory=GRAPH, runner=runner, bus=bus, project_ctx=projects)
        pid = uuid.uuid4().hex[:8]
        projects.states[pid] = ToyState()
        engine.on_decision(pid)
        try:
            served = wait_for(engine, pid, lambda s: all(x.status == "error" for x in s.values()),
                              timeout=60)
        finally:
            engine.shutdown()
    # Only what does not differ from one capture to the next: no key, no time.
    return {name: {**served[name].model_dump(mode="json"), "key": "k", "updated_at": None}
            for name in SERVED}


def main() -> None:
    out = {
        "source": ("turbotab/frontend/src/components/fixtures/capture_stages.py: labels from "
                   "build_graph(); statuses from Engine.status on test_typed_fit_refusal's "
                   "refusing_graph"),
        "labels": labels(),
        "statuses": statuses(),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(out['labels'])} labels, {len(out['statuses'])} statuses")


if __name__ == "__main__":
    main()
