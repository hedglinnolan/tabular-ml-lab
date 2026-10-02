"""Drive the synthetic tables through the real server and time each step (M2_CONTRACT §5).

Everything goes through HTTP (FastAPI's TestClient) and the server's own job workers, exactly as
the browser drives it; stage times are read off the server's ``stage`` events (running → fresh
for the stage's final key), and request times are wall-clock around the request.

    TURBOTAB_WORKERS=2 venv/bin/python -m turbotab.core.bench.run --data DIR [--which wide|tall|both]
        [--fit-timeout 240] [--json results.json] [--label after]

``--data`` holds the CSVs (written by ``turbotab.core.bench.synth`` when missing). A fit that is
not done within ``--fit-timeout`` seconds is cancelled; its time is then estimated from the
seconds its first cross-validation fold took, and the step says so.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from turbotab.core.bench import synth

VISIBLE = 40  # columns a grid shows at once


@dataclass
class Step:
    table: str
    name: str
    seconds: float | None
    note: str = ""


@dataclass
class Tap:
    """Every ``stage`` and ``job`` event the server publishes, timestamped."""

    events: list[tuple[float, str, str, dict[str, Any]]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def install(self, bus: Any) -> None:
        publish = bus.publish

        def tapped(pid: str, event_type: str, data: dict[str, Any]) -> None:
            if event_type in ("stage", "job"):
                with self.lock:
                    self.events.append((time.perf_counter(), pid, event_type, dict(data)))
            publish(pid, event_type, data)

        bus.publish = tapped

    def _stage_events(self, pid: str, stage: str, since: float) -> list[tuple[float, dict[str, Any]]]:
        with self.lock:
            return [(t, d) for t, p, kind, d in self.events
                    if p == pid and kind == "stage" and d.get("stage") == stage and t >= since]

    def fresh_at(self, pid: str, stage: str, since: float, key: str | None = None) -> float | None:
        for t, d in self._stage_events(pid, stage, since):
            if d.get("status") == "fresh" and (key is None or d.get("key") == key):
                return t
        return None

    def compute_seconds(self, pid: str, stage: str, key: str, since: float) -> float | None:
        """Seconds from the stage starting to run for ``key`` until it was fresh."""
        events = self._stage_events(pid, stage, since)
        start = next((t for t, d in events if d.get("key") == key
                      and d.get("status") in ("running", "queued")), None)
        running = next((t for t, d in events if d.get("key") == key and d.get("status") == "running"),
                       start)
        end = next((t for t, d in events if d.get("key") == key and d.get("status") == "fresh"), None)
        if running is None or end is None:
            return None
        return end - running

    def messages(self, pid: str, stage: str, since: float) -> list[tuple[float, str]]:
        with self.lock:
            out = []
            for t, p, kind, d in self.events:
                if p == pid and kind == "job" and d.get("stage") == stage and t >= since:
                    msg = str(d.get("message") or "")
                    if not out or out[-1][1] != msg:
                        out.append((t, msg))
            return out


class Driver:
    def __init__(self, client: Any, tap: Tap, table: str, steps: list[Step]):
        self.client = client
        self.tap = tap
        self.table = table
        self.steps = steps
        self.pid = ""

    def record(self, name: str, seconds: float | None, note: str = "") -> None:
        step = Step(self.table, name, None if seconds is None else round(seconds, 3), note)
        self.steps.append(step)
        shown = "   —   " if seconds is None else f"{seconds:7.2f}"
        print(f"  {self.table:5s} {name:44s} {shown} s  {note}", flush=True)

    # ── HTTP ──
    def get(self, path: str, **params: Any) -> Any:
        r = self.client.get(f"/api/projects/{self.pid}{path}", params=params)
        if r.status_code != 200:
            raise RuntimeError(f"GET {path}: {r.status_code} {r.text[:400]}")
        return r.json()

    def post(self, path: str, body: Any) -> Any:
        r = self.client.post(f"/api/projects/{self.pid}{path}", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"POST {path} {body.get('kind', '')}: {r.status_code} {r.text[:400]}")
        return r.json()

    def reach(self, key: str, timeout: float = 600.0) -> dict[str, Any]:
        """Wait until the Router reaches question ``key``, and return its step: answers are taken
        in its order (M2_CONTRACT §12.2), so an answer to a question it holds back is refused."""
        end = time.perf_counter() + timeout
        while True:
            steps = self.get("")["interview"]
            step = next(s for s in steps if s["key"] == key)
            first = next((s for s in steps if s["status"] in ("open", "waiting")), None)
            if step["status"] not in ("open", "waiting") or first is None or first["key"] == key:
                return step
            if time.perf_counter() > end:
                raise TimeoutError(f"{key} is held behind {first}")
            time.sleep(0.05)

    def answer_the_task(self, column: str, task: str) -> None:
        """The task is stated when the outcome reads clearly, else asked: answer it when asked."""
        if self.reach("task")["status"] in ("open", "waiting"):
            self.decide({"kind": "set_task", "column": column, "task": task})

    def timed(self, fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        return out, time.perf_counter() - t

    def decide(self, decision: dict[str, Any]) -> float:
        _, seconds = self.timed(self.post, "/decisions", decision)
        return seconds

    def status(self) -> dict[str, Any]:
        return self.client.app.state.service.engine.status(self.pid)

    def wait(self, stage: str, timeout: float = 600.0) -> tuple[str, float]:
        """Wait until ``stage`` is fresh for its current key; (key, when)."""
        end = time.perf_counter() + timeout
        while True:
            st = self.status()[stage]
            if st.status == "fresh":
                return st.key, time.perf_counter()
            if st.status == "error":
                raise RuntimeError(f"{stage} failed: {st.error}")
            if time.perf_counter() > end:
                raise TimeoutError(stage)
            time.sleep(0.02)

    def stage_step(self, stage: str, since: float, name: str, note: str = "",
                   timeout: float = 600.0) -> str:
        key, _ = self.wait(stage, timeout)
        self.record(name, self.compute(stage, key, since), note)
        return key

    def compute(self, stage: str, key: str, since: float) -> float | None:
        """The stage's running → fresh seconds for ``key``. The status reads fresh once the
        artifact is on disk, a moment before the engine publishes the event read here."""
        end = time.perf_counter() + 5.0
        seconds = self.tap.compute_seconds(self.pid, stage, key, since)
        while seconds is None and time.perf_counter() < end:
            time.sleep(0.02)
            seconds = self.tap.compute_seconds(self.pid, stage, key, since)
        return seconds

    def open(self, path: Path) -> float:
        t0 = time.perf_counter()
        r = self.client.post("/api/projects", json={"path": str(path)})
        if r.status_code != 200:
            raise RuntimeError(r.text)
        self.pid = r.json()["id"]
        return t0

    def accept_roles(self) -> dict[str, str]:
        roles = self.get("/stages/roles")["artifact"]
        return {c["column"]: c["proposed"] for c in roles["columns"]}

    def preview(self, decision: dict[str, Any], name: str) -> None:
        _, cold = self.timed(self.post, "/preview", decision)
        _, warm = self.timed(self.post, "/preview", decision)
        self.record(name, cold, f"again: {warm:.2f} s")


def ingest_and_profile(d: Driver, source: Path) -> None:
    t0 = d.open(source)
    key, t_fresh = d.wait("ingest")
    d.record("ingest (request → table ready)", t_fresh - t0,
             f"stage compute {d.compute('ingest', key, t0) or 0:.2f} s")
    d.stage_step("profile", t0, "profile: summaries + lens hints (stage)")
    from turbotab.core.datastore import DataStore, _summaries_path

    data = Path(d.client.app.state.service.workspace.data_path(d.pid))
    side = _summaries_path(data)
    saved = side.read_bytes()
    side.unlink()
    with DataStore(data, 4 << 30) as store:
        _, seconds = d.timed(store.summaries)
    side.write_bytes(saved)
    d.record("summaries, cold (DataStore.summaries)", seconds)
    columns, seconds = d.timed(d.get, "/columns")
    d.record("GET /columns (every column's summary)", seconds, f"{len(columns):,} summaries")
    names = [c["name"] for c in columns]
    mid = max(0, len(names) // 2 - VISIBLE // 2)
    visible = ",".join(names[mid:mid + VISIBLE])
    _, seconds = d.timed(d.get, "/table", offset=0, limit=100, columns=visible)
    d.record(f"GET /table 100 rows × {VISIBLE} visible columns", seconds)
    _, seconds = d.timed(d.get, "/table", offset=0, limit=100)
    d.record(f"GET /table 100 rows × all {len(names):,} columns", seconds)


def run_wide(client: Any, tap: Tap, source: Path, steps: list[Step], fit_timeout: float) -> None:
    d = Driver(client, tap, "wide", steps)
    ingest_and_profile(d, source)
    t1 = time.perf_counter()
    d.decide({"kind": "set_lens", "lenses": ["genomics"]})
    d.reach("target")  # an assay lens: the table's shape is read before the outcome
    d.decide({"kind": "set_target", "column": "bmi"})
    d.stage_step("roles", t1, "roles (stage)")
    d.stage_step("findings", t1, "findings (stage)")
    d.answer_the_task("bmi", "regression")
    d.reach("purpose")
    d.decide({"kind": "set_purpose", "purpose": "prediction"})
    d.reach("grain")  # `sample_id` names samples, not people: the grain is asked
    d.decide({"kind": "set_grain", "grain": "one_row_per_unit", "id_column": "sample_id"})
    d.reach("roles")
    roles = d.accept_roles()
    seconds = d.decide({"kind": "set_roles", "roles": roles})
    d.record("record the roles (POST set_roles)", seconds, f"{len(roles):,} columns")
    d.reach("exclusions")
    d.decide({"kind": "set_exclusions", "rules": []})
    d.reach("missing")
    d.preview({"kind": "set_missing", "strategy": "complete_case"},
              "preview: complete cases (energy-free)")
    d.preview({"kind": "set_roles", "roles": roles}, "preview: roles (lineage)")
    d.decide({"kind": "set_missing", "strategy": "complete_case"})
    d.reach("split")
    d.decide({"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
    d.stage_step("shelf", t1, "shelf: rank + time one fit of each family (stage)")
    for family in d.get("/stages/shelf")["artifact"]["families"]:
        d.record(f"shelf estimate: {family['label'].lower()}", family.get("estimate_seconds"),
                 family.get("estimate") or "not timed")
    d.reach("models")
    t2 = time.perf_counter()
    d.decide({"kind": "select_models", "models": ["elastic_net"]})
    d.stage_step("cohort", t1, "cohort (stage)")
    d.stage_step("split", t1, "split (stage)")
    d.stage_step("design", t2, "design (stage)")
    if fit_timeout <= 0:  # --fit-timeout 0: stop the fit as it starts, record nothing for it
        end = time.perf_counter() + 30
        while not d.status()["fit"].job_id and time.perf_counter() < end:
            time.sleep(0.05)
        if d.status()["fit"].job_id:
            client.post(f"/api/projects/{d.pid}/jobs/{d.status()['fit'].job_id}/cancel")
        return
    try:
        key, t_fresh = d.wait("fit", fit_timeout)
    except TimeoutError:
        msgs = tap.messages(d.pid, "fit", t2)
        folds = [t for t, m in msgs if ": fold " in m]
        st = d.status()["fit"]
        if st.job_id:
            client.post(f"/api/projects/{d.pid}/jobs/{st.job_id}/cancel")
        if len(folds) >= 2:
            per_fold = folds[1] - folds[0]
            d.record("fit: elastic net, 5-fold CV + refit", per_fold * 6,
                     f"ESTIMATED: cancelled after {fit_timeout:.0f} s; one fold took {per_fold:.1f} s, × 6")
        else:
            d.record("fit: elastic net, 5-fold CV + refit", None,
                     f"not done in {fit_timeout:.0f} s (cancelled); first fold unfinished")
        return
    d.record("fit: elastic net, 5-fold CV + refit", d.compute("fit", key, t2))
    fit = d.get("/stages/fit")["artifact"]
    model = fit["models"][0]
    r2 = model["cv"].get("r2", {}).get("mean")
    d.record("fit: the family's own fit_seconds", model["fit_seconds"],
             f"CV R² {r2:.3f}" if r2 is not None else "")


def run_tall(client: Any, tap: Tap, source: Path, steps: list[Step]) -> None:
    d = Driver(client, tap, "tall", steps)
    ingest_and_profile(d, source)
    t1 = time.perf_counter()
    d.decide({"kind": "set_lens", "lenses": ["dietary"]})
    d.reach("target")
    d.decide({"kind": "set_target", "column": "glucose"})
    d.stage_step("roles", t1, "roles (stage)")
    d.stage_step("findings", t1, "findings (stage)")
    d.answer_the_task("glucose", "regression")
    d.reach("purpose")
    d.decide({"kind": "set_purpose", "purpose": "prediction"})
    d.reach("roles")  # every `participant_id` appears once: the grain is stated, not asked
    roles = d.accept_roles()
    d.decide({"kind": "set_roles", "roles": roles})
    rule = {"kind": "range", "column": "kcal", "low": 500, "high": 5000,
            "reason": "implausible energy intake"}
    d.reach("exclusions")
    t2 = time.perf_counter()
    seconds = d.decide({"kind": "set_exclusions", "rules": [rule]})
    d.record("record an exclusion (POST set_exclusions)", seconds)
    d.stage_step("cohort", t2, "cohort (stage)")
    wider = dict(rule, low=600, high=4500)
    d.preview({"kind": "set_exclusions", "rules": [wider]}, "preview: exclusions (energy-free)")
    d.reach("missing")
    d.preview({"kind": "set_missing", "strategy": "complete_case"},
              "preview: complete cases (energy-free)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Time the wide-data benchmark tables.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--which", choices=("wide", "tall", "both"), default="both")
    parser.add_argument("--fit-timeout", type=float, default=240.0,
                        help="seconds to wait for the wide fit; 0 skips it")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    from fastapi.testclient import TestClient

    from turbotab.core.config import Settings
    from turbotab.server.app import create_app

    workers = int(os.environ.get("TURBOTAB_WORKERS") or 2)
    steps: list[Step] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="turbotab-bench-") as home:
        base = Settings.from_env()
        settings = Settings(home=Path(home), mode="local", workers=workers,
                            memory_budget_bytes=base.memory_budget_bytes)
        app = create_app(settings, frontend_dist=Path(home) / "no-frontend")
        with TestClient(app, base_url="http://127.0.0.1") as client:
            tap = Tap()
            tap.install(client.app.state.service.bus)
            if args.which in ("wide", "both"):
                run_wide(client, tap, synth.ensure(args.data, "wide"), steps, args.fit_timeout)
            if args.which in ("tall", "both"):
                run_tall(client, tap, synth.ensure(args.data, "tall"), steps)
    total = time.perf_counter() - started
    print(f"total {total:.1f} s")
    if args.json:
        payload = {
            "label": args.label,
            "workers": workers,
            "machine": f"{platform.machine()} {platform.platform()}",
            "python": sys.version.split()[0],
            "total_seconds": round(total, 1),
            "steps": [asdict(s) for s in steps],
        }
        args.json.write_text(json.dumps(payload, indent=1), "utf-8")


if __name__ == "__main__":
    main()
