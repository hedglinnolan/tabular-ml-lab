"""The pilot's launcher (PILOT_READINESS §5; SWARM_PLAN §2, §6 and §7): one analyst run started on
one dataset, its fit budget the pre-registration's, its known answer written where no analyst's
command reads it.

    python -m research.swarm.launcher --dataset <id> --crash <crash.json>
                                      --model M --persona P [--effort E] --prompt-version V
                                      [--prereg FILE] [--runs DIR] [--scripted]

**The pre-registration must be ready for the dataset** (``research/swarm/prereg_fill.py``, staged
by dataset): before each launch the launcher reads ``prereg.json`` (``prereg_fill.load``) and asks
``prereg_fill.ready`` whether that dataset may launch (a pilot dataset, its own fit budget filled
from its own crash sweep, on the Predict fixture r0 measured, every report filled and the engine
now on one tree with no file modified, the margins still the generators'); where it says why not,
the dataset is not launched, in its words.

**The crash sweep must have passed** (``research/swarm/sweep.py``, :func:`crash_passed`): its report
is the dataset's, on the table the dataset's manifest names (its SHA-256), it swept every decision
of the default plan (no ``only``), visited every option and met no engine defect, and it ran on
the engine the launch runs: the same tree of ``turbotab/`` (``git rev-parse HEAD:turbotab``, the
frozen engine: the same for every commit that leaves it unchanged), with no tracked file under it
modified, then or now. Otherwise the dataset is not launched, and the launcher says why.

**The fit budget** (:func:`prereg_budget`) is ``prereg.json``'s for the dataset
(``filled.fit_budget``, which ``prereg_fill``'s crash fill sets with :func:`budget_of`: as many
fits as the dataset's crash sweep made, and as many seconds as that many fits take at the slowest
fit it measured, its ``fit_seconds``' ``max``, rounded up to a whole second, so a run's seconds
never refuse it a fit its count allows, however slow the fits it chooses); a dataset it holds none
for (the decoy, which only the mechanical arms run) is not launched.

**The run's name is the driver's own**: ``start`` names each run ``run_NN`` and a secret part
(``driver.new_run_id``) and takes no label from its caller, so no label tells an analyst its
dataset, its control kind, or that it is on a decoy; the launcher passes none.

**The known answer** (:func:`known_answer`), written as ``{feature, family?, measure, value}`` or
``{null: true}``, read from the dataset's own answers file (``answers.json``), never retyped: a
null decoy's is ``{null: true}`` (no column enters its outcome, so the result's no-effect value is
its answer on any scale); an Estimate fixture's is the exposure the card names (its ``feature``),
the measure the planted effect is stated on (:data:`MEASURE_OF_WORDS`, the answers file's own
words for it) and the planted value, where the answers file states the effect of the card's
exposure on its outcome, ``{null: true}`` where it states the effect is none; any other dataset
has none (a Predict fixture's decision quality is its replay's, SWARM_PLAN §7). The family is
named only where the answers file names one. It is written with the rest of what the record
needs (:func:`record_meta`) to ``<run id>.json`` in the launches folder, a folder no analyst's
command opens (``driver.UNREAD_FOLDERS``), never into the run's own folder.

**The launches folder** (:func:`launches_of`) is one for every checkout of the repository:
``research/corpus/launches`` in its main checkout, the folder git's common directory lies in
(``git rev-parse --git-common-dir``), whichever checkout, the main one or a worktree, the launcher
or ``prereg_fill`` runs in. ``research/corpus/`` is kept out of git, so a launch is in that folder
alone, and ``prereg_fill`` reads each dataset's graded runs (its launches) there. Where git cannot
tell which checkout is the main one (a ``.git`` file it does not follow), the folder is refused,
and nothing is launched; a folder in no git repository is its own only checkout. No option writes a
launch elsewhere; a test hands :func:`launch` a folder of its own.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "research" / "datasets"
RUNS = REPO / "research" / "runs"
# The words an answers file states a planted effect's measure in, and the engine's name for that
# measure (``estimand.MEASURE_OF_TASK``' difference in means of a numeric outcome).
MEASURE_OF_WORDS = {"a difference in means": "mean_difference"}


class Refused(Exception):
    """The launcher does not launch: why, in words."""


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text("utf-8"))


def main_checkout(repo: Path | None = None) -> Path:
    """The main checkout of the repository ``repo`` (this checkout, where None) belongs to, which
    each of its worktrees shares (the module docstring): the folder that holds git's common
    directory, a ``.git`` folder; ``repo`` itself where it is in no git repository. Refused where
    git names a common directory that no checkout holds, or does not answer for a checkout whose
    ``.git`` is a file (a worktree's)."""
    repo = Path(REPO if repo is None else repo).resolve()
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--path-format=absolute",
                              "--git-common-dir"], capture_output=True, text=True, timeout=30)
        common = Path(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        common = None
    if common is not None and common.name == ".git":
        return common.resolve().parent
    if common is None and not (repo / ".git").is_file():
        return repo
    raise Refused(f"git cannot tell which checkout is the main one for {repo} ("
                  + (f"its common directory is {common}, in no checkout" if common is not None
                     else "its .git is a file git does not follow")
                  + "), so the launches every checkout shares cannot be found.")


def launches_of(repo: Path | None = None) -> Path:
    """The launches folder every checkout of ``repo``'s repository shares (the module
    docstring): ``research/corpus/launches`` in its main checkout (:func:`main_checkout`)."""
    return main_checkout(repo) / "research" / "corpus" / "launches"


try:
    # This checkout's launches folder, the main checkout's; None where it cannot be told.
    LAUNCHES: Path | None = launches_of(REPO)
except Refused:
    LAUNCHES = None


def engine_now() -> dict[str, Any]:
    from research.swarm.sweep import engine_version

    return engine_version()


# ── the crash sweep ──────────────────────────────────────────────────────────


def table_sha256(dataset: str) -> str | None:
    """The SHA-256 of the table ``dataset``'s manifest names, or None where it names none."""
    try:
        return str(_json(DATASETS / dataset / "manifest.json")["table"]["sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def crash_passed(report: Mapping[str, Any], dataset: str,
                 engine: Mapping[str, Any] | None = None) -> str | None:
    """Why the crash sweep ``report`` does not let ``dataset`` launch on ``engine`` (the module
    docstring), or None when it does."""
    engine = engine if engine is not None else engine_now()
    if report.get("dataset") != dataset:
        return (f"The crash sweep's report is of {report.get('dataset')!r}, not of "
                f"{dataset!r}.")
    table = table_sha256(dataset)
    ran_on = (report.get("data") or {}).get("sha256")
    if table is None:
        return f"The manifest of {dataset} names no table's SHA-256 to hold the crash sweep to."
    if ran_on != table:
        return (f"The crash sweep on {dataset} ran on a table whose SHA-256 is {ran_on}, not the "
                f"manifest's {table}.")
    if "only" not in report:
        return (f"The crash sweep's report on {dataset} does not say whether it swept every "
                f"decision of the default plan.")
    if report["only"] is not None:
        only = [str(o) for o in report["only"] or []]
        named = f" ({', '.join(only[:3])}{', ...' if len(only) > 3 else ''})" if only else ""
        return (f"The crash sweep on {dataset} swept only {len(only)} of the default plan's "
                f"decisions{named}, not every one.")
    if not report.get("complete"):
        left = report.get("unvisited") or []
        return (f"The crash sweep on {dataset} did not visit every option ({len(left)} left, "
                f"first {', '.join(map(str, left[:3]))}).")
    if report.get("defects"):
        return (f"The crash sweep on {dataset} met {len(report['defects'])} engine "
                f"defect{'s' if len(report['defects']) != 1 else ''}; they are fixed before the "
                f"freeze and the sweep runs again.")
    if not report.get("passed"):
        return f"The crash sweep on {dataset} did not pass."
    if report.get("engine_modified") or engine.get("engine_modified"):
        return ("The engine's tracked files differ from their commit, "
                + ("when the crash sweep ran." if report.get("engine_modified") else "now."))
    if not report.get("engine_tree") or report.get("engine_tree") != engine.get("engine_tree"):
        return (f"The crash sweep ran on engine tree {report.get('engine_tree')}, and the engine "
                f"now is {engine.get('engine_tree')}: the sweep runs again on the frozen engine.")
    return None


def budget_of(report: Mapping[str, Any], fits: int) -> dict[str, Any]:
    """The fit budget for ``fits`` fits (the module docstring): its seconds those fits at the
    slowest fit the crash sweep measured, rounded up; refused where it measured none."""
    if int(fits) < 1:
        raise Refused("A run's fit budget is at least one fit.")
    slowest = (report.get("fit_seconds") or {}).get("max")
    if slowest is None or not math.isfinite(float(slowest)):
        raise Refused("The crash sweep measured no fit's run time, so no budget of seconds can "
                      "be set from it.")
    return {"fits": int(fits), "seconds": float(max(1, math.ceil(int(fits) * float(slowest)))),
            "slowest_fit": float(slowest)}


# ── the known answer ─────────────────────────────────────────────────────────


def known_answer(dataset: str) -> tuple[dict[str, Any] | None, str]:
    """The dataset's known answer as the record holds it (the module docstring), and where it was
    read from (or why there is none)."""
    answers = _json(DATASETS / dataset / "answers.json")
    card = _json(DATASETS / dataset / "card.json")
    kind = answers.get("kind")
    if kind == "decoy":
        return {"null": True}, "a null decoy: no column enters its outcome"
    if kind != "known_answer" or card.get("purpose") != "Estimate":
        return None, (f"a {kind} dataset under {card.get('purpose')}: no known answer of this "
                      f"form")
    exposure = (card.get("exposure") or {}).get("column")
    outcome = (card.get("outcome") or {}).get("column")
    stem = f"the effect of {exposure} on {outcome}"
    truth = next((t for t in answers.get("truths") or [] if str(t.get("name", "")).startswith(stem)),
                 None)
    if truth is None:
        return None, f"the answers file states no effect of {exposure} on {outcome}"
    value = truth.get("value")
    if isinstance(value, str) and value.strip().lower().startswith("none"):
        return {"null": True}, f"answers.json: {truth['name']}"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None, f"the answers file states {truth['name']!r} as no number"
    words = str(truth["name"])[len(stem):].strip(" ,")
    measure = MEASURE_OF_WORDS.get(words)
    if measure is None:
        return None, f"the answers file states the effect as {words!r}, a measure not named here"
    known: dict[str, Any] = {"feature": exposure, "measure": measure, "value": float(value)}
    if truth.get("family"):
        known["family"] = str(truth["family"])
    return known, f"answers.json: {truth['name']}"


def dataset_meta(dataset: str) -> dict[str, Any]:
    """The record's ``dataset`` (``assemble.dataset_of``): what the intake holds of the dataset
    (``intake.dataset_meta``: its id, its control kind and fixture, the card the analyst is given
    and its declared measurement markers), and its known answer."""
    from research.swarm.intake import dataset_meta as intake_meta

    known, source = known_answer(dataset)
    return intake_meta(dataset) | {"known": known, "known_source": source}


def record_meta(launch: Mapping[str, Any]) -> dict[str, Any]:
    """What ``assemble.assemble`` is given for the run (its ``meta``), from its launch file."""
    dataset = {k: v for k, v in launch["dataset"].items() if k != "known_source"}
    return {"design": 2, "analyst_model": launch["analyst_model"], "persona": launch.get("persona"),
            "effort": launch.get("effort"), "prompt_version": launch["prompt_version"],
            "dataset": dataset, "help": []}


def launch_of(run_id: str, launches: Path | None = None) -> dict[str, Any]:
    return _json((launches_of() if launches is None else Path(launches)) / f"{run_id}.json")


# ── one launch ───────────────────────────────────────────────────────────────


def prereg_budget(prereg: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    """The fit budget prereg.json holds for ``dataset`` (``filled.fit_budget``, the crash fill's:
    :func:`budget_of` over the dataset's crash sweep, as many fits as it made), as the launch
    file keeps it; refused where it holds none, or one that is not a budget."""
    budget = ((prereg.get("filled") or {}).get("fit_budget") or {}).get(dataset)
    fits = (budget or {}).get("fits") if isinstance(budget, Mapping) else None
    seconds = (budget or {}).get("seconds") if isinstance(budget, Mapping) else None
    if budget is None:
        raise Refused(f"prereg.json holds no fit budget for {dataset}: fill it from the crash "
                      f"sweeps (python -m research.swarm.prereg_fill crash …).")
    if isinstance(fits, bool) or not isinstance(fits, int) or fits < 1 \
            or isinstance(seconds, bool) or not isinstance(seconds, (int, float)) \
            or not math.isfinite(float(seconds)) or float(seconds) <= 0:
        raise Refused(f"prereg.json's fit budget for {dataset} is not a count of fits and a "
                      f"number of seconds: {json.dumps(budget)[:300]}")
    return {"fits": int(fits), "seconds": float(seconds),
            "slowest_fit": budget.get("slowest_fit")}


def launch(dataset: str, crash: Path, *, model: str, persona: str | None,
           prompt_version: str, effort: str | None = None, runs: Path = RUNS,
           launches: Path | None = None, scripted: bool = False, data: Path | None = None,
           start: Any = None, prereg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One run of ``dataset`` started (the module docstring); its launch file's contents.
    ``start(argv)`` runs the driver's command line (``arms.Driven.command``, by default),
    ``data`` stands for the table its manifest builds, ``prereg`` for prereg.json and
    ``launches`` for the launches folder (:func:`launches_of`, where None) (a test's)."""
    from research.swarm import prereg_fill

    prereg = prereg_fill.load() if prereg is None else prereg
    why = prereg_fill.ready(prereg, dataset=dataset)
    if why is not None:
        raise Refused(f"{dataset} is not launched: {why}")
    report = _json(crash)
    why = crash_passed(report, dataset)
    if why is not None:
        raise Refused(f"{dataset} is not launched: {why}")
    budget = prereg_budget(prereg, dataset)
    folder = launches_of() if launches is None else Path(launches)
    meta = dataset_meta(dataset)
    if data is None:
        from research.swarm.dry_run import table_of

        data = table_of(dataset, Path(runs) / "data" / dataset)
    if start is None:
        from research.swarm.arms import Driven

        start = Driven(Path(runs)).command
    argv = ["start", "--data", str(data), "--budget-fits", str(budget["fits"]),
            "--budget-seconds", f"{budget['seconds']:g}", "--model", model,
            "--prompt-version", prompt_version]
    if persona is not None:
        argv += ["--persona", persona]
    if effort is not None:
        argv += ["--effort", effort]
    if scripted:
        argv.append("--scripted")
    code, payload = start(argv)
    if code != 0:
        raise Refused(f"{dataset} did not start: {json.dumps(payload)[:800]}")
    run_id = str(payload["started"]["run"])
    record = {"run": run_id, "dataset": meta, "design": 2, "analyst_model": model,
              "persona": persona, "effort": effort, "prompt_version": prompt_version,
              "budget": budget, "crash": {"path": str(crash),
                                          "sha256": hashlib.sha256(Path(crash).read_bytes())
                                          .hexdigest(),
                                          "engine_tree": report.get("engine_tree"),
                                          "commit": report.get("commit")}}
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", "utf-8")
    return record | {"launch": str(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.launcher",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--crash", type=Path, required=True, help="the dataset's crash.json")
    parser.add_argument("--model", required=True)
    from research.swarm.driver import PERSONAS

    parser.add_argument("--persona", choices=PERSONAS)
    parser.add_argument("--effort")
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--prereg", type=Path, help="prereg.json (default: prereg_fill's own)")
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--scripted", action="store_true")
    args = parser.parse_args(argv)
    try:
        prereg = None
        if args.prereg is not None:
            from research.swarm import prereg_fill

            prereg = prereg_fill.load(args.prereg)
        out = launch(args.dataset, args.crash, model=args.model, persona=args.persona,
                     prompt_version=args.prompt_version, effort=args.effort, runs=args.runs,
                     scripted=args.scripted, prereg=prereg)
    except Refused as exc:
        sys.stdout.write(json.dumps({"refused": str(exc)}) + "\n")
        return 1
    sys.stdout.write(json.dumps({"run": out["run"], "launch": out["launch"],
                                 "budget": out["budget"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
