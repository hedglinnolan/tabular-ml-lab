"""EXPORT · the manuscript bundle, its checklists and its replay (V2 definition of done §1, "export
ends every journey", and §3.6, "Reproducible").

Two journeys through the real server (``server_drive``) on the NHANES export (``stage_harness.
NHANES``, untracked: these tests skip where it is absent), each answered from the fixture's declared
truth (``truths.FIXTURE_TRUTHS``), never a constant:

* **inference** — fasting glucose on sugar, a substitution at fixed total energy (the standard
  model), the adjustment set by the disjunctive cause criterion, a linear model, complete cases, an
  energy screen (500–5,000 kcal) that replaced a first answer keeping every row; the plan locked
  when Table 2 is first shown; Model 1 declared after that, so the methods keep it, marked;
* **prediction** — the same outcome, a linear model and an elastic net compared with nothing held
  out, so the result is the selection-corrected estimate (BBC-CV); the role readings confirmed when
  the models question asks, after the missing-values answer, which leaves that answer's sentence
  counting rows the analysis no longer has until it is recorded again (the export refuses it).

**The references are independent of the export's code.**

* Table 2's crude model, Model 1 and Model 2 are refitted here with NumPy (least squares, HC3 by
  hand: (X'X)⁻¹X' diag(e²/(1−h)²) X(X'X)⁻¹, t on n − p df from SciPy) on the rows pandas selects by
  the energy screen; the bundle's CSV must agree to 1e-9.
* The model matrix the replay writes is read back with pyarrow and compared, value for value, with
  one built here from the CSV with pandas (raw columns as they are, a categorical column's levels as
  0/1 indicators with the first level left out); its file's SHA-256 is computed here with hashlib.
* The participant flow's counts are recounted with pandas.
* Each checklist item's text is parsed here, with this file's own reader, from the publishers' full
  texts (``export_data/*.xml.gz``, CC BY, fetched 2026-10-05): Collins et al. 2024, BMJ 385:e078378,
  Table 2 (TRIPOD+AI); Lachat et al. 2016, PLoS Med 13:e1002036, Table 1 (STROBE-nut with STROBE's
  items). Which items each journey answers is derived by hand from the item texts and the journey's
  record (:data:`STROBE_NUT_ANSWERED`, :data:`TRIPOD_ANSWERED`).
* The replay runs as a separate process (``python -m turbotab.replay``) in a fresh TurboTab home.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core.tests.acceptance.server_drive import (WP17_QUESTIONS, _post_when_reached,
                                                         answer_plan, answer_wp17, local_server,
                                                         open_project)
from turbotab.core.tests.stage_harness import NHANES, REPO
from turbotab.core.tests.truths import ASKING, fixture_truth
from turbotab.core.tests.truths import answers as truth_answers

HERE = Path(__file__).resolve().parent
SOURCES = HERE / "export_data"
FORBIDDEN = ("prespecified", "pre-specified", "preregistered", "pre-registered")
NUTRIENTS = ["sugar", "protein", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
SCREEN = {"column": "kcal", "low": 500, "high": 5000,
          "reason": "an implausible energy intake for one day"}
QUESTIONS = ("lens", "orientation", "target", "event", "task", "purpose", "grain", "repeat_kind",
             "unit", "aggregation", "temporal", "roles", "survey", "exclusions", "missing", "split",
             "energy_adjustment", "models")

needs_nhanes = pytest.mark.skipif(not NHANES.is_file(), reason="the NHANES export is not here")


# ── the journeys ─────────────────────────────────────────────────────────────


def _plan(purpose: str) -> dict[str, list[dict[str, Any]]]:
    """Each question's answers, in the order given (the last stands)."""
    plan: dict[str, list[dict[str, Any]]] = {
        "lens": [{"kind": "set_lens", "lenses": ["dietary"]}],
        "target": [{"kind": "set_target", "column": "glucose"}],
        "task": [{"kind": "set_task", "column": "glucose", "task": "regression"}],
        "purpose": [{"kind": "set_purpose", "purpose": purpose}],
        "grain": [{"kind": "set_grain", "grain": "one_row_per_unit"}],
        "temporal": [{"kind": "set_temporal", "temporal": False}],
        "survey": [{"kind": "set_survey", "estimand": "sample"}],
        "missing": [{"kind": "set_missing", "strategy": "complete_case"}],
        "split": [{"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5}],
    }
    if purpose == "inference":
        # A first answer that keeps every row, replaced by the energy screen before any estimate:
        # folded out of the methods.
        plan["exclusions"] = [{"kind": "set_exclusions", "rules": []},
                              {"kind": "set_exclusions", "rules": [SCREEN]}]
        plan["energy_adjustment"] = [{"kind": "set_energy_adjustment", "method": "standard",
                                      "energy_column": "kcal", "nutrients": NUTRIENTS}]
        plan["models"] = [{"kind": "select_models", "models": ["linear"]}]
    else:
        plan["exclusions"] = [{"kind": "set_exclusions", "rules": []}]
        plan["energy_adjustment"] = [{"kind": "set_energy_adjustment", "method": "none"}]
        plan["models"] = [{"kind": "select_models", "models": ["linear", "elastic_net"]}]
    return plan


def _post(drive: Any, body: dict[str, Any]) -> None:
    url = f"/api/projects/{drive.pid}/decisions"
    r = _post_when_reached(drive.c, url, body, unblock=lambda: drive.answer_wp17_before(body))
    while r.status_code == 409 and r.json()["error"]["code"] in ASKING:
        for decision in truth_answers(r.json()["error"], drive.truth):
            assert drive.c.post(url, json=decision).status_code == 200
        r = _post_when_reached(drive.c, url, body)
    assert r.status_code == 200, (body["kind"], r.text[:900])


def _export(drive: Any) -> Any:
    return drive.c.get(f"/api/projects/{drive.pid}/export")


def _journey(client: Any, purpose: str, *, confirm_roles: bool,
             before: dict[str, Any] | None = None) -> Any:
    """The NHANES journey in the Router's order (``test_discriminate``'s e3, extended): each
    question answered from :func:`_plan`, the WP17 questions and every reading asked from the
    fixture's truth. ``confirm_roles``: the roles the bulk answer left unconfirmed are confirmed in
    one block right after it (else when the models question asks). ``before[key](drive)`` runs when
    ``key`` is the open question, before it is answered."""
    from turbotab.core.interview import QUESTION_KEYS

    truth = fixture_truth(NHANES.name)
    drive = open_project(client, NHANES, truth)
    plan = _plan(purpose)
    for key in [k for k in QUESTION_KEYS if k in QUESTIONS or k in WP17_QUESTIONS]:
        if key in ("event", "task"):
            drive.artifact("target_info", timeout=600)
        step = drive.reach(key, timeout=600)
        if step["status"] not in ("open", "waiting"):
            continue
        if before and key in before:
            before[key](drive)
        if key == "adjustment":
            answer_plan(drive, drive.view()["state"]["estimand"]["exposure"])
            continue
        if key in WP17_QUESTIONS:
            answer_wp17(drive, key)
            continue
        if key == "roles":
            proposals = drive.artifact("roles")["columns"]
            roles = {c["column"]: c["proposed"] for c in proposals}
            for column, role in roles.items():
                truth.setdefault(f"role:{column}", role)
            _post(drive, {"kind": "set_roles", "roles": roles})
            waiting = drive.view()["decisions"][-1]["decision"].get("unconfirmed") or []
            if confirm_roles and waiting:
                _post(drive, {"kind": "confirm_readings", "items": [
                    {"reading": "role", "column": c, "value": roles[c]} for c in waiting]})
            continue
        for body in plan[key]:
            _post(drive, body)
    return drive


def _wait_fresh(drive: Any, stages: tuple[str, ...], timeout: float = 900.0) -> None:
    """Wait until ``stages`` are computed, reading only their status (serving an estimate would
    lock the plan)."""
    end = time.monotonic() + timeout
    while True:
        statuses = drive.view()["stages"]
        if all(statuses[s]["status"] == "fresh" for s in stages):
            return
        for s in stages:
            assert statuses[s]["status"] != "error", statuses[s]
        assert time.monotonic() < end, {s: statuses[s]["status"] for s in stages}
        time.sleep(0.1)


def _csv(data: Any) -> pd.DataFrame:
    """A bundle's CSV as pandas reads it, every number correctly rounded (``round_trip``)."""
    return pd.read_csv(io.BytesIO(data), float_precision="round_trip")


def _files(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return {n.split("/", 1)[1]: z.read(n) for n in z.namelist() if not n.endswith("/")}


@pytest.fixture(scope="module")
def inference(tmp_path_factory) -> dict[str, Any]:
    """The inference journey, run once: what the server answered at each step."""
    if not NHANES.is_file():
        pytest.skip("the NHANES export is not here")
    folder = tmp_path_factory.mktemp("export_inference")
    seen: dict[str, Any] = {}

    def at_missing(drive: Any) -> None:  # a required question open: refused, naming it
        seen["refused_unanswered"] = _export(drive)
        seen["checklist_early"] = drive.c.get(f"/api/projects/{drive.pid}/checklist")

    with local_server(folder / "home") as client:
        drive = _journey(client, "inference", confirm_roles=True, before={"missing": at_missing})
        _wait_fresh(drive, ("cohort", "design", "fit", "effects"))
        seen["refused_plan_open"] = _export(drive)  # every result computed, none yet shown
        seen["locked_before"] = drive.view()["state"].get("plan_locked")
        seen["effects_served"] = drive.artifact("effects")  # the first estimate shown: the lock
        seen["fit_served"] = drive.artifact("fit")
        # After the lock: Model 1 declared (age, sex and energy; NUTRITION_PACK §08), recorded as
        # made after the estimates were seen.
        _post(drive, {"kind": "set_model_sequence", "exposure": "sugar",
                      "model_1": ["age", "gender", "kcal"]})
        end = time.monotonic() + 600
        while True:
            effects = drive.artifact("effects")
            if "model_1" in {s["key"] for s in effects["families"][0]["sequence"]}:
                break
            assert time.monotonic() < end
            time.sleep(0.1)
        seen["effects"] = effects
        r1, r2 = _export(drive), _export(drive)
        seen["export"], seen["export_again"] = r1, r2
        seen["checklist"] = drive.c.get(f"/api/projects/{drive.pid}/checklist")
        seen["view"] = drive.view()
        seen["methods_route"] = drive.c.get(f"/api/projects/{drive.pid}/methods").json()
        seen["plan_route"] = drive.c.get(f"/api/projects/{drive.pid}/plan").content
        seen["pid"] = drive.pid
    bundle = folder / "inference.zip"
    bundle.write_bytes(seen["export"].content)
    seen["bundle"] = bundle
    seen["files"] = _files(seen["export"].content)
    seen["folder"] = folder
    return seen


@pytest.fixture(scope="module")
def prediction(tmp_path_factory) -> dict[str, Any]:
    """The prediction journey, run once."""
    if not NHANES.is_file():
        pytest.skip("the NHANES export is not here")
    folder = tmp_path_factory.mktemp("export_prediction")
    seen: dict[str, Any] = {}
    with local_server(folder / "home") as client:
        drive = _journey(client, "prediction", confirm_roles=False)
        _wait_fresh(drive, ("cohort", "design", "fit"))
        seen["fit_served"] = drive.artifact("fit")
        seen["cohort"] = drive.artifact("cohort")
        seen["methods_route"] = drive.c.get(f"/api/projects/{drive.pid}/methods").json()
        seen["export"] = _export(drive)
        seen["checklist"] = drive.c.get(f"/api/projects/{drive.pid}/checklist")
        seen["view"] = drive.view()
    bundle = folder / "prediction.zip"
    bundle.write_bytes(seen["export"].content)
    seen["bundle"] = bundle
    seen["files"] = _files(seen["export"].content)
    seen["folder"] = folder
    return seen


def _replay(bundle: Path, data: Path, home: Path, *extra: str) -> tuple[int, dict[str, Any]]:
    env = {**os.environ, "TURBOTAB_WORKERS": "2", "OMP_NUM_THREADS": "2",
           "PYTHONPATH": str(REPO)}
    done = subprocess.run([sys.executable, "-m", "turbotab.replay", str(bundle), "--data", str(data),
                           "--home", str(home), "--keep", "--json", *extra],
                          capture_output=True, text=True, cwd=str(REPO), env=env, timeout=1800)
    out = done.stdout[done.stdout.index("{"):] if "{" in done.stdout else "{}"
    return done.returncode, json.loads(out)


@pytest.fixture(scope="module")
def replays(inference, prediction) -> dict[str, Any]:
    """Each bundle replayed in a fresh home by ``python -m turbotab.replay``."""
    out = {}
    for name, seen in (("inference", inference), ("prediction", prediction)):
        home = seen["folder"] / "replay_home"
        code, report = _replay(seen["bundle"], NHANES, home)
        out[name] = {"code": code, "report": report, "home": home}
    return out


# ── (1) the bundle ───────────────────────────────────────────────────────────


@needs_nhanes
def test_1_the_bundle_holds_every_part_and_no_fitted_object_or_row(inference, prediction):
    for seen, table in ((inference, "table2"), (prediction, "performance")):
        r = seen["export"]
        assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
        assert r.headers["content-disposition"].endswith('-turbotab-export.zip"')
        files = seen["files"]
        guideline = "strobe_nut" if table == "table2" else "tripod_ai"
        assert {"README.md", "methods.md", "methods.txt", "methods.json", f"results/{table}.csv",
                f"results/{table}.md", "figures/participant_flow.svg", "figures/lineage.svg",
                f"checklist/{guideline}.md", f"checklist/{guideline}.json", "analysis_plan.json",
                "provenance.json", "decisions.jsonl", "manifest.json"} <= set(files)
        # BLUEPRINT §2: decisions and inputs' hashes, never a fitted object or a row of data
        assert not any(n.endswith((".joblib", ".pkl", ".pickle", ".parquet")) for n in files)
        manifest = json.loads(files["manifest.json"])
        assert set(manifest) == set(files) - {"manifest.json"}
        assert all(hashlib.sha256(files[n]).hexdigest() == h for n, h in manifest.items())
    # the same record and results give the same bytes
    assert inference["export"].content == inference["export_again"].content


@needs_nhanes
def test_1_the_provenance_record_holds_the_log_the_inputs_and_the_engine(inference):
    prov = json.loads(inference["files"]["provenance.json"])
    raw = NHANES.read_bytes()
    [table] = prov["inputs"]
    assert (table["role"], table["name"], table["bytes"]) == ("table", NHANES.name, len(raw))
    assert table["sha256"] == hashlib.sha256(raw).hexdigest()
    log = inference["files"]["decisions.jsonl"]
    assert prov["decisions"]["sha256"] == hashlib.sha256(log).hexdigest()
    lines = [json.loads(line) for line in log.decode().splitlines() if line.strip()]
    assert prov["decisions"]["n"] == len(lines) == len(inference["view"]["decisions"])
    assert [r["id"] for r in prov["decisions"]["records"]] == [r["id"] for r in lines]
    assert prov["engine"]["turbotab"] == client_version()
    assert prov["engine"]["stages"]["design"] >= 23 and "effects" in prov["engine"]["stages"]
    assert prov["environment"]["python"] == ".".join(map(str, sys.version_info[:3]))
    assert set(prov["stages"]) == {"cohort", "design", "fit", "effects"}


def client_version() -> str:
    from turbotab.server import __version__

    return __version__


@needs_nhanes
def test_1_the_analysis_plan_is_the_lock_s_with_its_sha256(inference):
    """MODELING_SEQUENCE §1 row 12: the plan exports with a timestamp and a content hash; the
    bundle's copy is the plan route's bytes, and its hashes are recomputed here."""
    data = inference["files"]["analysis_plan.json"]
    assert data == inference["plan_route"]
    doc = json.loads(data)
    content = {k: v for k, v in doc.items() if k not in ("sha256", "text")}
    canon = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert doc["sha256"] == hashlib.sha256(canon).hexdigest()
    plan = json.dumps(doc["plan"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert doc["plan_sha256"] == hashlib.sha256(plan.encode()).hexdigest()
    lock = next(r for r in inference["view"]["decisions"] if r["decision"]["kind"] == "lock_plan")
    assert doc["status"] == "locked" and doc["plan_sha256"] == lock["decision"]["digest"]
    prov = json.loads(inference["files"]["provenance.json"])
    assert prov["analysis_plan"]["file_sha256"] == hashlib.sha256(data).hexdigest()
    assert prov["analysis_plan"]["plan_sha256"] == doc["plan_sha256"]


@needs_nhanes
def test_1_the_methods_are_the_records_sentences_by_strobe_section(inference):
    """The record's sentences in force, by STROBE section; the first exclusions answer folded out;
    Model 1, declared after the estimates were seen, kept in a section of its own after the lock;
    every sentence verbatim from the methods route; never "prespecified" or "preregistered"."""
    doc = json.loads(inference["files"]["methods.json"])
    route = {line["record_id"]: line for line in inference["methods_route"]["lines"]}
    records = inference["view"]["decisions"]
    exclusions = [r for r in records if r["decision"]["kind"] == "set_exclusions"]
    assert len(exclusions) == 2
    said = [e for s in doc["sections"] for e in s["entries"] if e["source"] == "record"]
    ids = [e["record_id"] for e in said]
    assert exclusions[0]["id"] not in ids and exclusions[1]["id"] in ids  # folded out, then kept
    assert set(ids) == set(route)  # every line of the record's methods text, and only those
    for e in said:
        assert e["text"] == route[e["record_id"]]["sentence"]  # verbatim
    keys = [s["key"] for s in doc["sections"]]
    # (wave 2b, FORM) the form question's answer says how each continuous term enters (STROBE 11)
    assert keys == ["design", "participants", "variables", "measurement", "quantitative",
                    "statistical", "after", "reproducibility"]
    by = {s["key"]: s for s in doc["sections"]}
    kinds = {k: [e["kind"] for e in by[k]["entries"]] for k in keys}
    assert kinds["quantitative"] == ["set_forms"]
    assert kinds["design"] == ["set_purpose"]
    assert kinds["participants"] == ["set_exclusions"]
    assert {"set_target", "set_roles", "set_estimand"} <= set(kinds["variables"])
    # the statistical methods end with the analysis's own paragraph and then the lock
    assert kinds["statistical"][-2:] == ["effects", "lock_plan"]
    assert {"set_missing", "set_adjustment", "set_energy_adjustment", "select_models"} <= set(
        kinds["statistical"])
    [after] = by["after"]["entries"]
    assert after["kind"] == "set_model_sequence" and after["after_estimates"] is True
    assert after["text"].startswith("After the estimates were seen, ")
    assert by["after"]["title"] == "Decisions made after the estimates were seen"
    text = inference["files"]["methods.md"].decode()
    assert text.index("## Study design (STROBE 4)") < text.index("## Variables (STROBE 7)") \
        < text.index("## Statistical methods (STROBE 12)") \
        < text.index("## Decisions made after the estimates were seen")
    lock = next(r for r in records if r["decision"]["kind"] == "lock_plan")
    assert text.index(lock["sentence"]) < text.index(after["text"])
    for name in ("methods.md", "methods.txt", "methods.json", "analysis_plan.json", "README.md"):
        lowered = inference["files"][name].decode().lower()
        assert not any(word in lowered for word in FORBIDDEN), name


@needs_nhanes
def test_1_the_reproducibility_sentence_is_written_as_said(inference, prediction):
    """The one methods sentence the export writes, verbatim, its values recomputed here."""
    for seen in (inference, prediction):
        prov = json.loads(seen["files"]["provenance.json"])
        doc = json.loads(seen["files"]["methods.json"])
        [entry] = [e for s in doc["sections"] if s["key"] == "reproducibility"
                   for e in s["entries"]]
        engine = prov["engine"]
        built = (f" (source revision `{engine['commit'][:12]}`"
                 f"{', with local changes' if engine['modified'] else ''})"
                 if engine["commit"] else "")
        sha = hashlib.sha256(NHANES.read_bytes()).hexdigest()
        plan = json.loads(seen["files"]["analysis_plan.json"])
        matrix = prov["model_matrix"]
        n = len([x for x in seen["files"]["decisions.jsonl"].decode().splitlines() if x.strip()])
        assert entry["text"] == (
            f"The analysis was run in TurboTab {client_version()}{built}. The supplementary "
            f"provenance record holds the decision log (`{n:,}` records), the SHA-256 of the input "
            f"file `{NHANES.name}` (SHA-256 `{sha[:12]}`), the analysis plan (SHA-256 "
            f"`{plan['plan_sha256'][:12]}`) and the model matrix (`{matrix['n_rows']:,}` rows by "
            f"`{matrix['n_cols']:,}` columns; SHA-256 `{matrix['parquet_sha256'][:12]}` as "
            f"Parquet). Replaying it with `python -m turbotab.replay` checks each input file "
            f"against its hash and refuses one that differs; otherwise it rebuilds the analysis "
            f"from the input files and the decision log alone and compares the model matrix and "
            f"every reported estimate with these.")


def _nhanes() -> pd.DataFrame:
    """The file as pandas reads it, every number correctly rounded (``round_trip``: pandas' default
    parser can miss the nearest double by one unit in the last place, where the app's reader does
    not)."""
    return pd.read_csv(NHANES, float_precision="round_trip")


def _hc3(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Least squares with HC3 standard errors and 95% t intervals, by hand (NumPy, SciPy's t)."""
    from scipy import stats

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = (X * (e ** 2 / (1 - h) ** 2)[:, None]).T @ X
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    df = X.shape[0] - X.shape[1]
    q = stats.t.ppf(0.975, df)
    t = beta / se
    return {"beta": beta, "se": se, "low": beta - q * se, "high": beta + q * se,
            "p": 2 * stats.t.sf(np.abs(t), df), "df": np.full_like(beta, df)}


def _design(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The columns as a model matrix: numbers as they are; a categorical column's levels as
    indicators, its first level (sorted) left out."""
    parts = [pd.Series(1.0, index=frame.index, name="const")]
    for c in columns:
        values = frame[c]
        if c in ("cycle_begin_year", "gender"):
            levels = sorted(values.dropna().unique())
            for level in levels[1:]:
                parts.append((values == level).astype(float).rename(f"{c}_{level}"))
        else:
            parts.append(values.astype(float).rename(c))
    return pd.concat(parts, axis=1)


@needs_nhanes
def test_1_table_2_reports_the_declared_models_as_numpy_refits_them(inference):
    """Every exposure row of Table 2 (CSV, full precision) against an HC3 least-squares refit by
    hand on the rows the energy screen keeps, recounted with pandas."""
    frame = _nhanes()
    kept = frame[frame["glucose"].notna() & frame["kcal"].between(SCREEN["low"], SCREEN["high"])]
    table = _csv((inference["files"]["results/table2.csv"]))
    seq = {s["key"]: s for s in inference["effects"]["families"][0]["sequence"]}
    assert list(seq) == ["crude", "model_1", "model_2", "model_3"]
    assert table["key"].tolist() == [f"linear/{k}/sugar" for k in seq]
    for key, s in seq.items():
        columns = ["sugar", *s["adjusted_for"]]
        rows = kept.dropna(subset=columns)
        assert s["n_rows"] == len(rows) == len(kept)  # complete on every model's columns
        X = _design(rows, columns)
        ref = _hc3(X.to_numpy(), rows["glucose"].to_numpy(float))
        j = list(X.columns).index("sugar")
        row = table.set_index("key").loc[f"linear/{key}/sugar"]
        assert row["n"] == len(rows)
        for col, want in (("estimate", "beta"), ("se", "se"), ("ci_low", "low"),
                          ("ci_high", "high"), ("df", "df")):
            assert row[col] == pytest.approx(ref[want][j], rel=1e-9, abs=1e-12), (key, col)
        assert row["p"] == pytest.approx(ref["p"][j], rel=1e-6, abs=1e-15), key
        assert row["covariance"] == "HC3"
    md = inference["files"]["results/table2.md"].decode()
    assert md.count("| sugar |") == 4  # the exposure's rows only (Westreich & Greenland 2013)
    appendix = _csv((inference["files"]["results/table2_appendix.csv"]))
    assert "sugar" not in set(appendix["term"]) and "(intercept)" in set(appendix["term"])
    assert set(appendix["why"].dropna()) and "Adjustment terms, not effect estimates" in \
        inference["files"]["results/table2_appendix.md"].decode()


@needs_nhanes
def test_1_the_figures_are_journal_svg_with_the_flow_recounted(inference, prediction):
    """DESIGN_LANGUAGE §07: serif, greyscale, literal hex colors, numbered caption; the flow's
    counts recounted with pandas; the lineage ends in the model matrix's columns."""
    frame = _nhanes()
    outside = int((~frame["kcal"].between(SCREEN["low"], SCREEN["high"]) & frame["kcal"].notna()
                   & frame["glucose"].notna()).sum())
    blank_kcal = int((frame["kcal"].isna() & frame["glucose"].notna()).sum())
    for seen, number in ((inference, 1), (prediction, 1)):
        for name, n in (("figures/participant_flow.svg", 1), ("figures/lineage.svg", 2)):
            svg = seen["files"][name].decode()
            root = ET.fromstring(svg)
            assert root.tag.endswith("svg") and "serif" in root.get("font-family")
            assert "var(" not in svg and "<style" not in svg
            for color in re.findall(r"#[0-9a-fA-F]{6}\b", svg):
                assert color[1:3] == color[3:5] == color[5:7], (name, color)  # grey
            title = root.find("{http://www.w3.org/2000/svg}title").text
            assert title.startswith(f"Figure {n}. ")
            assert f"Figure {n}. " in "".join(t.text or "" for t in root.iter(
                "{http://www.w3.org/2000/svg}text"))
    flow = inference["files"]["figures/participant_flow.svg"].decode()
    assert f"n = {len(frame):,}" in flow and f"n = {outside:,}" in flow
    assert f"n = {len(frame) - outside - blank_kcal:,}" in flow
    words = " ".join(" ".join(t.text or "" for t in ET.fromstring(flow).iter(
        "{http://www.w3.org/2000/svg}text")).split())
    assert f"Excluded: {SCREEN['reason']}" in words
    prov = json.loads(inference["files"]["provenance.json"])
    lineage = inference["files"]["figures/lineage.svg"].decode()
    assert all(f">{c}<" in lineage for c in prov["model_matrix"]["columns"])
    assert f"Model matrix ({prov['model_matrix']['n_cols']} columns)" in lineage
    # the declared exposure's path, and only it, is set in bold
    assert re.findall(r'font-weight="700">([^<]+)<', lineage).count("sugar") == 3


@needs_nhanes
def test_1_the_performance_table_reports_the_declared_result_only(prediction):
    """MODELING_SEQUENCE §4: with nothing held out, the result is the selection-corrected estimate,
    never the winner's own score; each family's own scores are labeled not the result."""
    fit = prediction["fit_served"]
    result = fit["result"]
    assert result["basis"] == "selection_corrected"
    table = _csv((prediction["files"]["results/performance.csv"]))
    results = table[table["role"] == "the result"]
    primary = fit["primary_metric"]
    assert set(results["key"]) == {f"result/selection_corrected/{primary}",
                                   *(f"result/selection_corrected/{m}"
                                     for m in fit["selection"]["extras"])}
    row = results.set_index("key").loc[f"result/selection_corrected/{primary}"]
    assert (row["estimate"], row["ci_low"], row["ci_high"]) == (
        result["estimate"], result["ci_low"], result["ci_high"])
    winner = next(m for m in fit["models"] if m["family"] == fit["selection"]["best"])
    own = table.set_index("key").loc[f"{winner['family']}/cv/{primary}"]
    assert own["role"] == "not the result" and own["estimate"] == winner["cv"][primary]["estimate"]
    assert result["sentence"] in prediction["files"]["results/performance.md"].decode()


# ── (2) the replay ───────────────────────────────────────────────────────────


def _matrix_by_hand(frame: pd.DataFrame, columns: list[str], ids: np.ndarray) -> pd.DataFrame:
    rows = frame.iloc[ids]
    out = {}
    for c in columns:
        if c in frame.columns:
            out[c] = rows[c].astype(float).to_numpy()
        else:
            base, level = next((b, c[len(b) + 1:]) for b in ("cycle_begin_year", "gender")
                               if c.startswith(f"{b}_"))
            values = rows[base].astype(str)
            out[c] = (values == level).astype(float).to_numpy()
    return pd.DataFrame(out)


@needs_nhanes
@pytest.mark.parametrize("name", ["inference", "prediction"])
def test_2_a_fresh_home_replay_reproduces_the_matrix_and_every_estimate(name, inference,
                                                                        prediction, replays):
    seen = {"inference": inference, "prediction": prediction}[name]
    run = replays[name]
    report = run["report"]
    assert run["code"] == 0, report
    assert report["reproduced"] is True and report["refused"] is None
    prov = json.loads(seen["files"]["provenance.json"])
    # the model matrix, byte for byte: the replay's file, hashed here
    written = run["home"] / "replay" / "model_matrix.parquet"
    assert hashlib.sha256(written.read_bytes()).hexdigest() == prov["model_matrix"]["parquet_sha256"]
    assert report["matrix"]["identical"] and report["matrix"]["content_identical"]
    # ... and its values, against a matrix built here with pandas
    import pyarrow.parquet as pq

    table = pq.read_table(written).to_pandas()
    assert list(table.columns) == ["row_id", *prov["model_matrix"]["columns"]]
    ids = table["row_id"].to_numpy()
    expected = _matrix_by_hand(_nhanes(), prov["model_matrix"]["columns"], ids)
    assert np.array_equal(table.drop(columns="row_id").to_numpy(), expected.to_numpy(),
                          equal_nan=True)
    # every reported estimate, to 1e-12
    est = report["estimates"]
    assert est["n_recorded"] == len(prov["estimates"]) == est["compared"] == est["reproduced"]
    assert est["n_recorded"] > 50 and not est["mismatched"] and not est["missing"]
    assert est["max_abs_diff"] <= 1e-12 and est["tolerance"] == 1e-12
    again = json.loads((zipfile.ZipFile(run["home"] / "replay" / "bundle.zip")
                        .read("turbotab-export/provenance.json")))
    for key, value in prov["estimates"].items():
        other = again["estimates"][key]
        assert (value is None and other is None) or abs(value - other) <= 1e-12 * max(1, abs(value))
    assert report["stage_keys_same"] is True and report["plan_same"] is True
    # the whole bundle comes out the same: methods, tables, figures, checklist, plan, log
    assert report["files_different"] == [] and "methods.md" in report["files_same"]


@needs_nhanes
def test_2_a_changed_input_file_is_refused_with_both_hashes_named(inference, tmp_path):
    raw = NHANES.read_bytes()
    first = raw.index(b"\n") + 1
    changed = raw[:first] + raw[first:].replace(b"102.4", b"102.5", 1)
    assert changed != raw
    path = tmp_path / NHANES.name
    path.write_bytes(changed)
    home = tmp_path / "home"
    code, report = _replay(inference["bundle"], path, home)
    recorded = hashlib.sha256(raw).hexdigest()
    actual = hashlib.sha256(changed).hexdigest()
    assert code == 2 and report["reproduced"] is False
    assert report["refused"].startswith(f"The table `{NHANES.name}` does not match the record: its "
                                        f"SHA-256 is {actual}, the record's is {recorded}")
    assert not home.exists()  # nothing was computed
    [check] = report["inputs"]
    assert (check["sha256"], check["recorded_sha256"], check["matches"]) == (actual, recorded, False)


# ── (3) the checklists ───────────────────────────────────────────────────────


def _cells(tr: ET.Element) -> list[ET.Element]:
    return [c for c in tr if c.tag in ("td", "th")]


def _say(e: ET.Element) -> str:
    return " ".join("".join(e.itertext()).split())


def _source_table(name: str, label: str) -> list[list[str]]:
    root = ET.fromstring(gzip.decompress((SOURCES / name).read_bytes()))
    table = next(t for t in root.iter("table-wrap") if t.find("label") is not None
                 and _say(t.find("label")) == label)
    return [[_say(c) for c in _cells(tr)] for tr in table.iter("tr")]


def test_3_every_item_is_quoted_from_its_primary_source():
    """Each item's text, read here from the publisher's full text: TRIPOD+AI's Table 2 (its 52
    items, by number) and STROBE-nut's Table 1 (STROBE's lettered recommendations and the 24 nut-
    items). A lettered STROBE item is its "(a)" … part of the cell; a design-variant row is joined to
    the part above it."""
    from turbotab.core.export import checklists

    tripod = checklists.items("TRIPOD+AI")["items"]
    rows = [r for r in _source_table("PMC11019967.xml.gz", "Table 2") if len(r) >= 3]
    quoted = {r[-3]: r[-1] for r in rows if re.fullmatch(r"\d+[a-g]?", r[-3])}
    assert len(quoted) == len(tripod) == 52
    for item in tripod:
        assert item["text"] == quoted[item["id"]], item["id"]
    strobe = checklists.items("STROBE-nut")["items"]
    rows = _source_table("PMC4896435.xml.gz", "Table 1")[1:]
    nut = {m.group(1): m.group(2) for r in rows
           if (m := re.match(r"^(nut-[0-9.]+)\.\s(.*)$", r[3]))}
    assert len(nut) == 24 == sum(i["id"].startswith("nut-") for i in strobe)
    text = " ".join(r[2] for r in rows if r[2])  # the STROBE column, row after row
    for item in strobe:
        if item["id"].startswith("nut-"):
            assert item["text"] == nut[item["id"]], item["id"]
        else:
            assert item["text"] in text, item["id"]  # a lettered part, or a whole cell
    assert [i["id"] for i in strobe if not i["id"].startswith("nut-")] == [
        "1a", "1b", "2", "3", "4", "5", "6a", "6b", "7", "8", "9", "10", "11", "12a", "12b", "12c",
        "12d", "12e", "13a", "13b", "13c", "14a", "14b", "14c", "15", "16a", "16b", "16c", "17",
        "18", "19", "20", "21", "22"]


# Derived by hand from each item's text and the journey's record (module docstring): the record
# answers an item outright, or in part (the author owes the rest), or not at all.
STROBE_NUT_ANSWERED = {
    "12a": "answered",  # the models, the adjustment set and its criterion, the declared sequence
    "12c": "answered",  # complete cases
    "12e": "answered",  # the effects stage's sensitivity to unmeasured confounding
    "nut-12.2": "answered",  # the standard energy model
    "13c": "answered",  # the flow diagram
    "nut-13": "answered",  # the energy screen's and the complete cases' exclusions
    "16a": "answered",  # Table 2: unadjusted and adjusted, the criterion for each covariate
    "17": "answered",  # the sensitivity table
    "3": "partly answered",  # the estimand, not the objectives in the author's words
    "4": "partly answered",  # inference, not the design's usual name
    "6a": "partly answered",  # the screen, not the sources and methods of selection
    "7": "partly answered",  # outcome, exposure, roles; not their definitions
    "nut-7.1": "partly answered",  # energy's unit and day, asked at the screen; not each nutrient's
    "9": "partly answered",  # confounding and its sensitivity; not every other bias
    "11": "partly answered",  # the energy model; not why each form
    "13a": "partly answered",  # from the table on; not before it
    "13b": "partly answered",  # each exclusion's reason; not nonparticipation before the table
    "nut-22.2": "partly answered",  # the provenance record; not the data or the tools
}
TRIPOD_ANSWERED = {
    "11": "answered",  # complete cases
    "12a": "answered",  # prediction, cross-validation, no rows held out
    "12b": "answered",  # no energy adjustment; the lineage's one-hot coding
    "20a": "answered",  # the flow (a continuous outcome: no events owed)
    "4": "partly answered", "6b": "partly answered", "7": "partly answered",
    "8a": "partly answered", "9a": "partly answered", "9b": "partly answered",
    "12c": "partly answered", "12e": "partly answered", "18f": "partly answered",
    "21": "partly answered", "23a": "partly answered",
}


@needs_nhanes
def test_3_each_checklist_lists_every_item_with_where_or_unanswered(inference, prediction):
    from turbotab.core.export.checklists import UNANSWERED

    for seen, expected, stem, total in ((inference, STROBE_NUT_ANSWERED, "strobe_nut", 58),
                                        (prediction, TRIPOD_ANSWERED, "tripod_ai", 52)):
        report = json.loads(seen["files"][f"checklist/{stem}.json"])
        route = seen["checklist"].json()
        assert seen["checklist"].status_code == 200 and route["waiting"] == []
        assert route == report
        statuses = {i["id"]: i["status"] for i in report["items"]}
        assert {k: v for k, v in statuses.items() if v != "unanswered"} == expected
        counts = report["counts"]
        answered = sum(v == "answered" for v in expected.values())
        partly = sum(v == "partly answered" for v in expected.values())
        assert counts == {"items": total, "answered": answered, "partly_answered": partly,
                          "unanswered": total - answered - partly}
        assert report["unanswered"] == [i["id"] for i in report["items"]
                                        if i["status"] == "unanswered"]
        sentences = {x["record_id"]: x["sentence"] for x in seen["methods_route"]["lines"]}
        for item in report["items"]:
            if item["status"] == "unanswered":
                assert item["where"] == [] and item["note"] == UNANSWERED
                continue
            assert item["where"]
            for w in item["where"]:
                if w["source"] == "record":  # the quoted sentence and its decision
                    assert w["quote"] == sentences[w["record_id"]]
                elif w["source"] == "file":
                    assert w["file"] in seen["files"] or w["file"] == "provenance.json"
            if item["status"] == "partly answered":
                assert item["note"] == f"partly answered — the author must supply {item['owed']}"
        md = seen["files"][f"checklist/{stem}.md"].decode()
        assert md.count("**Unanswered — the author must supply this.**") == counts["unanswered"]
    # STROBE item 3 is quoted as the source words it; no answer says it
    report = json.loads(inference["files"]["checklist/strobe_nut.json"])
    three = next(i for i in report["items"] if i["id"] == "3")
    assert "prespecified" in three["text"]
    assert not any(w in json.dumps(three["where"]).lower() for w in FORBIDDEN)


# ── (4) the routes ───────────────────────────────────────────────────────────


def test_4_the_routes_are_in_the_committed_openapi_document():
    doc = json.loads((REPO / "turbotab" / "server" / "openapi.json").read_text("utf-8"))
    export = doc["paths"]["/api/projects/{pid}/export"]["get"]
    assert set(export["responses"]) >= {"200", "404", "409"}
    assert "application/zip" in export["responses"]["200"]["content"]
    checklist = doc["paths"]["/api/projects/{pid}/checklist"]["get"]
    ref = checklist["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("/ChecklistReport")
    assert "waiting" in doc["components"]["schemas"]["ChecklistReport"]["properties"]


# ── (5) the refusals ─────────────────────────────────────────────────────────


@needs_nhanes
def test_5_the_export_refuses_while_a_required_question_is_unanswered(inference):
    r = inference["refused_unanswered"]
    assert r.status_code == 409
    error = r.json()["error"]
    assert error["code"] == "unanswered_questions"
    assert "The missing-values question" in error["message"]
    labels = [e["label"] for e in error["exits"]]
    assert "Answer the missing-values question" in labels
    # the live checklist reads what the record holds so far, and says what the export waits for
    early = inference["checklist_early"].json()
    assert any("missing-values question" in w for w in early["waiting"])


@needs_nhanes
def test_5_the_export_refuses_while_the_plan_is_open(inference):
    r = inference["refused_plan_open"]
    assert r.status_code == 409 and inference["locked_before"] in (None, False)
    error = r.json()["error"]
    assert error["code"] == "plan_open"
    assert error["message"].startswith("The analysis plan is still open")
    assert error["exits"] == [{"label": "Show the estimates; the first one shown locks the plan",
                               "decision": None}]
    assert inference["view"]["state"]["plan_locked"] is True


@needs_nhanes
@pytest.mark.parametrize("name", ["inference", "prediction"])
def test_1_a_sentence_that_counts_rows_states_the_rows_as_they_stand(name, inference, prediction):
    """The missing-values answer's sentence counts the complete cases on the rows the answers before
    it select. Under inference the adjustment set, answered after it, leaves out mediators whose
    blanks were dropping rows; under prediction the roles confirmed after it bring predictors in.
    The Record keeps the sentence as said; the methods restate its counts as the participant flow
    has them now (recounted here with pandas), so the bundle never contradicts its own Figure 1."""
    seen = {"inference": inference, "prediction": prediction}[name]
    frame = _nhanes()
    record = next(r for r in seen["view"]["decisions"] if r["decision"]["kind"] == "set_missing")
    doc = json.loads(seen["files"]["methods.json"])
    [line] = [e for s in doc["sections"] for e in s["entries"] if e["kind"] == "set_missing"]
    route = {x["record_id"]: x["sentence"] for x in seen["methods_route"]["lines"]}
    assert line["record_id"] == record["id"] and line["text"] == route[record["id"]]
    if name == "inference":
        screened = frame[frame["glucose"].notna() & frame["kcal"].between(SCREEN["low"],
                                                                          SCREEN["high"])]
        columns = ["sugar", *seen["effects"]["families"][0]["sequence"][-1]["adjusted_for"]]
        n = len(screened)
        assert len(screened.dropna(subset=columns)) == n  # every analyzed column recorded
        assert line["text"] == (f"A complete-case analysis was applied: no row is missing any "
                                f"predictor, so all `{n:,}` rows remain.")
    else:
        predictors = seen["cohort"]["predictors"]
        n = len(frame[frame["glucose"].notna()])
        kept = len(frame[frame["glucose"].notna()].dropna(subset=predictors))
        assert kept < n
        assert line["text"] == (f"Rows missing any predictor were dropped (a complete-case "
                                f"analysis): `{kept:,}` of `{n:,}` rows remain.")
        assert f"n = {kept:,}" in seen["files"]["figures/participant_flow.svg"].decode()
    assert record["sentence"] != line["text"]  # the Record keeps what was said when it was said


# ── the contract and its chain ───────────────────────────────────────────────

# Each relation the export's contract declares, and the test above that sees it fire.
RELATION_TESTS = {
    "unanswered_question": "test_5_the_export_refuses_while_a_required_question_is_unanswered",
    "plan_open": "test_5_the_export_refuses_while_the_plan_is_open",
    "counts_as_they_stand": "test_1_a_sentence_that_counts_rows_states_the_rows_as_they_stand",
    "input_changed": "test_2_a_changed_input_file_is_refused_with_both_hashes_named",
    "superseded_folded_out": "test_1_the_methods_are_the_records_sentences_by_strobe_section",
    "plan_with_hash": "test_1_the_analysis_plan_is_the_lock_s_with_its_sha256",
    "exposure_rows_only": "test_1_table_2_reports_the_declared_models_as_numpy_refits_them",
    "declared_result_only": "test_1_the_performance_table_reports_the_declared_result_only",
    "unanswered_listed": "test_3_each_checklist_lists_every_item_with_where_or_unanswered",
    "no_fitted_objects": "test_1_the_bundle_holds_every_part_and_no_fitted_object_or_row",
    "replay_reproduces": "test_2_a_fresh_home_replay_reproduces_the_matrix_and_every_estimate",
}


def test_the_contract_declares_every_part_and_each_relation_has_its_test():
    """BLUEPRINT §13 in the one registry: slot, scope, needs, routing (a question; options labeled
    customary and sound for both purposes, each with a rung), storyboard, sentence and relations;
    every ``enforced_by`` names live code; each conflict is refused with its exits; and every
    relation the contract declares is asserted by a test of this file."""
    import importlib

    from turbotab.core import contracts as C

    registry = C.contracts()
    mine = {k: c for k, c in registry.items() if c.package == "EXPORT"}
    assert set(mine) == {"manuscript_export"}
    c = mine["manuscript_export"]
    assert (c.slot, c.scope) == ("evaluation", "descriptive")
    assert c.needs and c.question.endswith("?") and c.storyboard and c.options
    for o in c.options:
        for purpose in C.PURPOSES:
            assert o.sound[purpose] and o.rung[purpose] in C.RUNGS
    for target in [c.sentence, *(r.enforced_by for r in c.relations if r.enforced_by)]:
        module, name = target.split(":")
        assert callable(getattr(importlib.import_module(module), name)), target
    for r in c.relations:
        if r.kind == "conflicts":
            assert r.rung == "refused" and r.exits, r.name
    assert {r.name for r in c.relations} == set(RELATION_TESTS)
    tests = {name for name in globals() if name.startswith("test_")}
    assert set(RELATION_TESTS.values()) <= tests
    C.run_order(list(registry))


# ═════════════════════════════════════════════════════════════════════════════
# A yes/no outcome on a synthetic table: the same chain where the NHANES export is absent, and the
# paths the NHANES journeys do not take (an odds ratio beside a marginal risk difference; a held-out
# set declared and opened under prediction).
# ═════════════════════════════════════════════════════════════════════════════

DIET_ROLES = {"pid": "identifier", "age": "covariate", "sex": "covariate", "smoking": "covariate",
              "activity": "covariate", "bmi": "covariate", "kcal": "energy",
              "protein_g": "exposure"}
DIET_TRUTH = {"code_or_count:smoking": "amount", "code_or_count:activity": "amount",
              "code_or_count:pid": "amount", "code_or_count:kcal": "amount", "day_count:kcal": "1",
              "unit:protein_g": "g", "unit:kcal": "kcal",
              "adjust:age": "yes,yes,no", "adjust:sex": "yes,yes,no",
              "adjust:smoking": "yes,yes,no", "adjust:activity": "no,yes,no",
              "adjust:bmi": "unknown,yes,unknown"}


def _diet(n: int = 700, seed: int = 47) -> pd.DataFrame:
    """Protein (an energy-bearing exposure) and diabetes (a common outcome), with smoking a
    confounder, activity a cause of the outcome only, and BMI of unknown timing."""
    rng = np.random.default_rng(seed)
    age = rng.normal(52, 9, n).round(1)
    sex = rng.choice(["female", "male"], n)
    smoking = rng.binomial(1, 0.3, n)
    activity = rng.integers(0, 8, n)
    kcal = rng.normal(2100, 400, n).round(0)
    protein_g = (0.035 * kcal + rng.normal(0, 10, n) + 2 * smoking).round(1)
    bmi = (27 - 0.03 * (protein_g - 75) + rng.normal(0, 3, n)).round(1)
    logit = (-0.85 + 0.012 * (protein_g - 75) - 0.0004 * (kcal - 2100) + 0.7 * smoking
             + 0.03 * (age - 52) - 0.2 * (activity - 3.5))
    dm = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    return pd.DataFrame({"pid": np.arange(1, n + 1), "age": age, "sex": sex, "smoking": smoking,
                         "activity": activity, "bmi": bmi, "kcal": kcal, "protein_g": protein_g,
                         "dm": dm})


def _open_diet(client: Any, csv: Path, purpose: str, join: Path | None = None) -> Any:
    """``join``: a second file (``bmi`` by ``pid``) added and joined, as NHANES's components are."""
    from turbotab.core.tests.truths import Truth

    drive = open_project(client, csv, Truth(DIET_TRUTH, fixture="the synthetic diet"))
    drive.decide({"kind": "set_lens", "lenses": ["dietary"]})
    if join is not None:
        added = client.post(f"/api/projects/{drive.pid}/files", json={"path": str(join)})
        assert added.status_code == 200, added.text[:400]
        drive.decide({"kind": "join_files", "file": added.json()["id"], "on": "pid"})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": "dm"})
    drive.answer("event", {"kind": "set_event", "column": "dm", "level": "True"})
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": purpose})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit", "id_column": "pid"})
    drive.reach("roles")
    drive.decide_roles(DIET_ROLES)
    drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    return drive


@pytest.fixture(scope="module")
def diet(tmp_path_factory) -> dict[str, Any]:
    """Both journeys on the synthetic table, each exported, refused first where it must be."""
    from turbotab.core.tests.truths import answer_adjustment

    folder = tmp_path_factory.mktemp("export_diet")
    frame = _diet()
    csv, body = folder / "diet.csv", folder / "body.csv"
    frame.to_csv(csv, index=False)
    frame.drop(columns="bmi").to_csv(folder / "diet_only.csv", index=False)
    frame[["pid", "bmi"]].to_csv(body, index=False)
    seen: dict[str, Any] = {"frame": frame, "csv": csv, "folder": folder, "body": body,
                            "diet_only": folder / "diet_only.csv"}
    with local_server(folder / "home") as client:
        # inference, on two files joined by pid: a marginal risk difference, the odds ratio beside it
        drive = _open_diet(client, folder / "diet_only.csv", "inference", join=body)
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.reach("estimand")
        drive.decide({"kind": "set_estimand", "exposure": "protein_g", "effect": "total",
                      "contrast": "substitution", "measure": "risk_difference"})
        drive.reach("adjustment")
        answer_adjustment(drive.post, drive.artifact("proposals")["adjustment"], drive.truth)
        end = time.monotonic() + 240
        while drive.artifact("proposals").get("model_sequence") is None:
            assert time.monotonic() < end
            time.sleep(0.1)
        drive.decide(drive.artifact("proposals")["model_sequence"]["decision"])
        drive.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "standard",
                                           "energy_column": "kcal", "nutrients": ["protein_g"]})
        drive.answer("models", {"kind": "select_models", "models": ["linear"]})
        _wait_fresh(drive, ("cohort", "design", "fit", "effects"))
        seen["inference_refused"] = _export(drive)
        seen["effects"] = drive.artifact("effects")  # the lock
        seen["inference"] = _export(drive)
        seen["inference_view"] = drive.view()
        # prediction: a held-out set drawn, the final model declared and then the rows opened
        drive = _open_diet(client, csv, "prediction")
        drive.answer("split", {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
        drive.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "none"})
        drive.answer("models", {"kind": "select_models", "models": ["linear", "elastic_net"]})
        _wait_fresh(drive, ("cohort", "design", "fit"))
        seen["prediction_refused"] = _export(drive)
        exits = seen["prediction_refused"].json()["error"]["exits"]
        opening = next(e["decision"] for e in exits
                       if (e["decision"] or {}).get("family") == "linear")
        drive.decide(opening)
        _wait_fresh(drive, ("cohort", "design", "fit"))
        seen["fit"] = drive.artifact("fit")
        seen["prediction"] = _export(drive)
    for name in ("inference", "prediction"):
        if seen[name].status_code == 200:
            (folder / f"{name}.zip").write_bytes(seen[name].content)
            seen[f"{name}_files"] = _files(seen[name].content)
    return seen


def test_diet_the_export_waits_for_the_lock_and_for_the_declared_final_model(diet):
    r = diet["inference_refused"]
    assert r.status_code == 409 and r.json()["error"]["code"] == "plan_open"
    r = diet["prediction_refused"]
    assert r.status_code == 409
    error = r.json()["error"]
    assert error["code"] == "plan_open"
    assert error["message"].startswith("The final model is not declared: the held-out rows are "
                                       "still sealed")
    assert [(e["decision"]["kind"], e["decision"]["family"]) for e in error["exits"]] == [
        ("open_seal", "linear"), ("open_seal", "elastic_net")]
    assert diet["prediction"].status_code == 200 and diet["inference"].status_code == 200


def test_diet_table_2_carries_the_odds_ratio_and_the_marginal_contrast(diet):
    files = diet["inference_files"]
    table = _csv((files["results/table2.csv"])).set_index("key")
    primary = table.loc["linear/model_2/protein_g"]
    family = diet["effects"]["families"][0]
    served = next(s for s in family["sequence"] if s["key"] == "model_2")["effects"][0]
    for col in ("estimate", "ci_low", "ci_high", "ratio", "ratio_low", "ratio_high", "p"):
        assert primary[col] == served[col], col
    assert primary["ratio"] == pytest.approx(np.exp(primary["estimate"]), rel=1e-12)
    assert "Ratio (95% CI)" in files["results/table2.md"].decode()
    [contrast] = family["marginal"]["contrasts"]
    marginal = _csv((files["results/table2_marginal.csv"])).set_index("key")
    row = marginal.loc[f"linear/marginal/{contrast['setting']}"]
    assert (row["rd"], row["rd_low"], row["rd_high"]) == (contrast["rd"], contrast["rd_low"],
                                                          contrast["rd_high"])
    report = json.loads(files["checklist/strobe_nut.json"])
    sixteen_c = next(i for i in report["items"] if i["id"] == "16c")
    assert sixteen_c["status"] == "answered"
    assert sixteen_c["where"][0]["file"] == "results/table2_marginal.md"


def test_diet_the_held_out_score_at_the_opening_is_the_one_result(diet):
    fit = diet["fit"]
    assert fit["result"]["basis"] == "holdout" and fit["at_opening"]["family"] == "linear"
    files = diet["prediction_files"]
    table = _csv((files["results/performance.csv"]))
    results = table[table["role"] == "the result"]
    primary = fit["primary_metric"]
    assert results["key"].tolist() == [f"result/holdout/{primary}"]
    assert results["estimate"].iloc[0] == fit["at_opening"]["scores"]["linear"][primary]
    methods = json.loads(files["methods.json"])
    kinds = [e["kind"] for s in methods["sections"] for e in s["entries"]]
    assert kinds[-2:] == ["open_seal", "provenance"]  # the opening closes the methods
    report = json.loads(files["checklist/tripod_ai.json"])
    assert report["checklist"] == "TRIPOD+AI" and report["counts"]["items"] == 52


@pytest.mark.parametrize("name", ["inference", "prediction"])
def test_diet_a_fresh_home_replay_reproduces_both_journeys(name, diet):
    """The inference journey read two files: the joined one is given by name, from a copy elsewhere
    (``--file body.csv=<path>``), and checked against its own hash like the table."""
    home = diet["folder"] / f"replay_{name}"
    prov = json.loads(diet[f"{name}_files"]["provenance.json"])
    if name == "inference":
        copy = diet["folder"] / "elsewhere" / "body.csv"
        copy.parent.mkdir(exist_ok=True)
        copy.write_bytes(diet["body"].read_bytes())
        roles = [(f["role"], f["name"]) for f in prov["inputs"]]
        assert roles == [("table", "diet_only.csv"), ("joined file", "body.csv")]
        code, report = _replay(diet["folder"] / f"{name}.zip", diet["diet_only"], home,
                               "--file", f"body.csv={copy}")
        assert [c["path"] for c in report["inputs"]] == [str(diet["diet_only"]), str(copy)]
    else:
        code, report = _replay(diet["folder"] / f"{name}.zip", diet["csv"], home)
    assert code == 0 and report["reproduced"] is True, report
    written = home / "replay" / "model_matrix.parquet"
    assert hashlib.sha256(written.read_bytes()).hexdigest() == prov["model_matrix"]["parquet_sha256"]
    est = report["estimates"]
    assert est["reproduced"] == est["n_recorded"] == len(prov["estimates"]) > 10
    assert est["max_abs_diff"] <= 1e-12
    assert report["files_different"] == []
