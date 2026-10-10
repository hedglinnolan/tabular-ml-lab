"""The headless replay of an export bundle (V2 definition of done §3.6: "Replaying the record
reproduces the model matrix and the estimates"; BLUEPRINT §9's M3 item 4).

    python -m turbotab.replay <bundle.zip> --data <the input file> [--file <id>=<path> …]
                              [--home <empty folder>] [--keep] [--json]

**First, the inputs** (:func:`verify_inputs`). Each input file the provenance record names is hashed
and compared with its recorded SHA-256; a file that differs is refused with both hashes named, and
nothing is computed (a replay on other data would compare nothing that matters). The table is the
file given by ``--data``, else the path the record names; a joined file is given by ``--file
<its id or name>=<path>``, else its recorded path. A codebook is not read again: every reading it
settled is in the decision log.

**Then the analysis, from the record alone** (:func:`run`). In a fresh TurboTab home (a new
temporary folder, or ``--home``, which must be empty), a project is made from the input files and
the bundle's ``decisions.jsonl`` written as its decision log, byte for byte: the state is a fold of
the log (BLUEPRINT §3), so every stage computes from the answers exactly as recorded, with no
answer given again and nothing validated a second time against a different moment. The stages the
bundle reports are waited for, and the bundle is assembled again from the rebuilt project by the
same code.

**Then the comparison** (:func:`compare`): the model matrix byte for byte (its canonical Parquet
SHA-256, and its content hash, which does not depend on the Parquet writer), every number of the
results tables to within ``1e-12`` (absolute below 1, relative above), the analysis plan's hashes,
each result stage's key (equal when the engine's stage versions are), and which other files of the
bundle come out identical. The report says what matched, and the command exits 0 when the matrix
and every estimate are reproduced, 1 when they are not, and 2 when the replay was refused.

This module drives the server's project service (the same code a client is served by), imported
when a replay runs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from turbotab.core.export.bundle import read_zip
from turbotab.core.export.record import TOLERANCE
from turbotab.core.export.source import hash_file

COMPARED_FILES_SKIPPED = ("provenance.json", "manifest.json", "README.md")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class InputCheck(_Model):
    role: str
    name: str
    path: str
    recorded_sha256: str
    sha256: str | None
    matches: bool
    message: str


class MatrixCheck(_Model):
    recorded_parquet_sha256: str | None
    parquet_sha256: str | None
    identical: bool
    recorded_content_sha256: str | None
    content_sha256: str | None
    content_identical: bool
    n_rows: int | None
    n_cols: int | None
    writer: str | None = None


class EstimateCheck(_Model):
    n_recorded: int
    compared: int
    reproduced: int
    max_abs_diff: float
    max_rel_diff: float
    tolerance: float
    mismatched: list[str]
    missing: list[str]  # recorded, not recomputed
    extra: list[str]  # recomputed, not recorded


class ReplayReport(_Model):
    bundle: str
    reproduced: bool
    refused: str | None = None
    home: str | None = None
    seconds: float | None = None
    inputs: list[InputCheck] = []
    decisions: int | None = None
    plan_same: bool | None = None
    matrix: MatrixCheck | None = None
    estimates: EstimateCheck | None = None
    stage_keys: dict[str, dict[str, str | None]] = {}
    stage_keys_same: bool | None = None
    files_same: list[str] = []
    files_different: list[str] = []
    messages: list[str] = []


class Refused(Exception):
    """The replay refuses: an input differs from the record, or the bundle is not one."""

    def __init__(self, message: str, inputs: Sequence[InputCheck] = ()):
        super().__init__(message)
        self.message = message
        self.inputs = list(inputs)


# ── the bundle ───────────────────────────────────────────────────────────────


def load_bundle(path: str | Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """The bundle's files and its provenance record; refuses a file that is not a bundle, or one
    whose decision log is not the one its provenance record hashed."""
    import hashlib
    import zipfile

    try:
        files = read_zip(Path(path).read_bytes())
    except (OSError, zipfile.BadZipFile) as exc:
        raise Refused(f"{path} is not a TurboTab export bundle: {exc}") from None
    if "provenance.json" not in files or "decisions.jsonl" not in files:
        raise Refused(f"{path} is not a TurboTab export bundle: it has no provenance record or no "
                      f"decision log.")
    prov = json.loads(files["provenance.json"])
    recorded = (prov.get("decisions") or {}).get("sha256")
    actual = hashlib.sha256(files["decisions.jsonl"]).hexdigest()
    if recorded != actual:
        raise Refused(f"The bundle's decision log is not the one its provenance record names: its "
                      f"SHA-256 is {actual}, the record's is {recorded}.")
    return files, prov


# ── the inputs ───────────────────────────────────────────────────────────────


def _given(entry: Mapping[str, Any], data: str | None, files: Mapping[str, str]) -> str:
    if entry.get("role") == "table":
        return data or str(entry.get("path"))
    for key in (entry.get("file_id"), entry.get("name")):
        if key and key in files:
            return files[key]
    return str(entry.get("path"))


def verify_inputs(prov: Mapping[str, Any], data: str | None = None,
                  files: Mapping[str, str] | None = None) -> list[InputCheck]:
    """Each input the record names, hashed where it is given, against its recorded SHA-256
    (module docstring); raises :class:`Refused` naming every mismatch, both hashes in full."""
    checks: list[InputCheck] = []
    for entry in prov.get("inputs") or []:
        if entry.get("role") == "codebook":
            continue  # its readings are in the decision log; the file is not read again
        path = _given(entry, data, files or {})
        recorded = str(entry.get("sha256") or "")
        name, role = str(entry.get("name")), str(entry.get("role"))
        if not Path(path).is_file():
            checks.append(InputCheck(role=role, name=name, path=path, recorded_sha256=recorded,
                                     sha256=None, matches=False,
                                     message=f"The {role} `{name}` is not at {path}."))
            continue
        sha, _, _ = hash_file(path)
        same = sha == recorded
        checks.append(InputCheck(
            role=role, name=name, path=path, recorded_sha256=recorded, sha256=sha, matches=same,
            message=(f"The {role} `{name}` matches the record (SHA-256 {sha})." if same else
                     f"The {role} `{name}` does not match the record: its SHA-256 is {sha}, the "
                     f"record's is {recorded} ({path}).")))
    bad = [c for c in checks if not c.matches]
    if bad:
        raise Refused(" ".join(c.message for c in bad) + " Nothing was replayed: give the file "
                      "the record was made from.", checks)
    return checks


# ── the comparison ───────────────────────────────────────────────────────────


def close(a: float | None, b: float | None, tolerance: float = TOLERANCE) -> bool:
    """Equal to within ``tolerance`` of the larger of 1 and the recorded value's size."""
    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tolerance * max(1.0, abs(a))


def compare_estimates(recorded: Mapping[str, float | None], replayed: Mapping[str, float | None],
                      tolerance: float = TOLERANCE) -> EstimateCheck:
    keys = [k for k in recorded if k in replayed]
    mismatched = [k for k in keys if not close(recorded[k], replayed[k], tolerance)]
    diffs = [abs(float(recorded[k]) - float(replayed[k])) for k in keys
             if recorded[k] is not None and replayed[k] is not None]
    rels = [abs(float(recorded[k]) - float(replayed[k])) / max(1.0, abs(float(recorded[k])))
            for k in keys if recorded[k] is not None and replayed[k] is not None]
    return EstimateCheck(
        n_recorded=len(recorded), compared=len(keys), reproduced=len(keys) - len(mismatched),
        max_abs_diff=max(diffs, default=0.0), max_rel_diff=max(rels, default=0.0),
        tolerance=tolerance, mismatched=mismatched,
        missing=[k for k in recorded if k not in replayed],
        extra=[k for k in replayed if k not in recorded])


def compare(prov: Mapping[str, Any], again: Mapping[str, Any], files: Mapping[str, bytes],
            files_again: Mapping[str, bytes]) -> dict[str, Any]:
    """The recorded bundle against the replayed one (module docstring)."""
    m0, m1 = prov.get("model_matrix") or {}, again.get("model_matrix") or {}
    matrix = MatrixCheck(
        recorded_parquet_sha256=m0.get("parquet_sha256"), parquet_sha256=m1.get("parquet_sha256"),
        identical=bool(m0) and m0.get("parquet_sha256") == m1.get("parquet_sha256"),
        recorded_content_sha256=m0.get("content_sha256"), content_sha256=m1.get("content_sha256"),
        content_identical=bool(m0) and m0.get("content_sha256") == m1.get("content_sha256"),
        n_rows=m1.get("n_rows"), n_cols=m1.get("n_cols"),
        writer=(m1.get("parquet") or {}).get("writer"))
    estimates = compare_estimates(prov.get("estimates") or {}, again.get("estimates") or {},
                                  float((prov.get("replay") or {}).get("tolerance") or TOLERANCE))
    k0, k1 = prov.get("stages") or {}, again.get("stages") or {}
    keys = {s: {"recorded": k0.get(s), "replayed": k1.get(s)} for s in sorted(set(k0) | set(k1))}
    p0, p1 = prov.get("analysis_plan") or {}, again.get("analysis_plan") or {}
    names = sorted(n for n in set(files) | set(files_again) if n not in COMPARED_FILES_SKIPPED)
    return {
        "matrix": matrix, "estimates": estimates, "stage_keys": keys,
        "stage_keys_same": all(v["recorded"] == v["replayed"] for v in keys.values()),
        "plan_same": (p0.get("sha256"), p0.get("plan_sha256")) == (p1.get("sha256"),
                                                                    p1.get("plan_sha256")),
        "files_same": [n for n in names if files.get(n) == files_again.get(n)],
        "files_different": [n for n in names if files.get(n) != files_again.get(n)],
    }


# ── the run ──────────────────────────────────────────────────────────────────


def _fresh_home(home: str | None) -> Path:
    if home is None:
        return Path(tempfile.mkdtemp(prefix="turbotab-replay-"))
    path = Path(home).expanduser()
    if path.exists() and any(path.iterdir()):
        raise Refused(f"{path} is not empty: a replay runs in a fresh TurboTab home.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _restore_joined(project_dir: Path, entry: Mapping[str, Any], path: str) -> None:
    """A joined file, read into ``files/<its recorded id>/`` as adding it does (DATAIN)."""
    from turbotab.core.datastore import ingest

    fid = str(entry["file_id"])
    folder = project_dir / "files" / fid
    folder.mkdir(parents=True, exist_ok=True)
    info = ingest(Path(path), folder / "raw.parquet")
    meta = {"id": fid, "name": entry.get("name"), "source_kind": "path", "source_path": str(path),
            "fingerprint": info.fingerprint, "n_rows": info.n_rows, "n_cols": info.n_cols,
            "columns": [c.name for c in info.columns], "warnings": list(info.warnings)}
    (folder / "file.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")


def _wait(service: Any, pid: str, stages: Sequence[str], timeout: float) -> None:
    end = time.monotonic() + timeout
    while True:
        statuses = service.engine.status(pid)
        done = True
        for stage in stages:
            status = statuses[stage]
            if status.status == "error":
                raise RuntimeError(f"the {stage} stage failed on replay: {status.error}")
            if status.status == "blocked":
                raise RuntimeError(f"the {stage} stage is blocked on replay, waiting for "
                                   f"{', '.join(status.missing) or 'an answer'}")
            if status.status != "fresh":
                done = False
        if done:
            return
        if time.monotonic() > end:
            raise RuntimeError(f"the replay did not finish within {timeout:g} seconds")
        time.sleep(0.2)


def run(bundle: str | Path, *, data: str | None = None, files: Mapping[str, str] | None = None,
        home: str | None = None, keep: bool = False, timeout: float = 3600.0,
        workers: int | None = None) -> ReplayReport:
    """Replay ``bundle`` (module docstring). Never raises for a refusal or a failed comparison:
    the report says which."""
    started = time.monotonic()
    report = ReplayReport(bundle=str(bundle), reproduced=False)
    try:
        recorded_files, prov = load_bundle(bundle)
        report.inputs = verify_inputs(prov, data, files)
    except Refused as exc:
        report.refused = exc.message
        report.inputs = exc.inputs
        return report
    from turbotab.core.config import Settings
    from turbotab.core.datastore import fingerprint_file
    from turbotab.core.decisions import fold
    from turbotab.core.export import gate
    from turbotab.core.export.bundle import contents, matrix_bytes
    from turbotab.server.service import SOURCE_FILE, ProjectService

    try:
        path = _fresh_home(home)
    except Refused as exc:
        report.refused = exc.message
        return report
    report.home = str(path)
    settings = dataclasses.replace(Settings.from_env(), home=path, mode="local",
                                   **({"workers": workers} if workers else {}))
    try:
        service = ProjectService(settings, replay=True)
    except Exception as exc:  # noqa: BLE001 - said, never a trace
        report.messages.append(f"The replay could not start TurboTab in {path}: {exc}")
        return report
    try:
        table = next(c for c in report.inputs if c.role == "table")
        name = str((prov.get("project") or {}).get("name") or Path(table.path).stem)
        meta = service.workspace.create_project(name, table.path, "path",
                                                source_name=table.name)
        pdir = service.workspace.project_dir(meta.id)
        (pdir / SOURCE_FILE).write_text(json.dumps({"fingerprint": fingerprint_file(
            Path(table.path))}), "utf-8")
        for entry in prov.get("inputs") or []:
            if entry.get("role") == "joined file":
                check = next(c for c in report.inputs if c.role == "joined file"
                             and c.name == entry.get("name"))
                _restore_joined(pdir, entry, check.path)
        service.workspace.decisions_path(meta.id).write_bytes(recorded_files["decisions.jsonl"])
        records = service.log(meta.id).records()
        report.decisions = len(records)
        service.engine.on_decision(meta.id)
        _wait(service, meta.id, gate.result_stages(fold(records)), timeout)
        source = service.export_source(meta.id)
        again = contents(source)
        out = path / "replay"
        out.mkdir(exist_ok=True)
        raw = matrix_bytes(source)
        if raw is not None:
            (out / "model_matrix.parquet").write_bytes(raw)
        (out / "bundle.zip").write_bytes(again.zip())
        found = compare(prov, again.provenance, recorded_files, again.files)
        report.matrix = found["matrix"]
        report.estimates = found["estimates"]
        report.stage_keys = found["stage_keys"]
        report.stage_keys_same = found["stage_keys_same"]
        report.plan_same = found["plan_same"]
        report.files_same = found["files_same"]
        report.files_different = found["files_different"]
        report.reproduced = bool(report.matrix.identical and report.matrix.content_identical
                                 and not report.estimates.mismatched
                                 and not report.estimates.missing
                                 and not report.estimates.extra)
    except Exception as exc:  # noqa: BLE001 - a replay that cannot finish says why, never a trace
        report.messages.append(f"The replay could not finish: {exc}")
    finally:
        service.close()
        report.seconds = round(time.monotonic() - started, 3)
        if not keep and home is None:
            shutil.rmtree(path, ignore_errors=True)
            report.home = None
    return report


# ── the command ──────────────────────────────────────────────────────────────


def summary(report: ReplayReport) -> str:
    """The report in a few lines, for the terminal."""
    lines = [f"Replay of {report.bundle}" + (f" in a fresh home at {report.home}"
                                             if report.home else "")]
    for c in report.inputs:
        lines.append(f"  {c.message}")
    if report.refused:
        lines.append(f"Refused: {report.refused}")
        return "\n".join(lines)
    if report.decisions is not None:
        lines.append(f"  The decision log: {report.decisions} records, replayed as recorded.")
    if report.matrix is not None:
        m = report.matrix
        lines.append(f"  The model matrix ({m.n_rows or 0:,} rows by {m.n_cols or 0:,} columns): "
                     f"Parquet "
                     f"SHA-256 {'identical' if m.identical else 'different'} "
                     f"({m.parquet_sha256}); its values "
                     f"{'identical' if m.content_identical else 'different'}.")
    if report.estimates is not None:
        e = report.estimates
        lines.append(f"  The estimates: {e.reproduced} of {e.n_recorded} reproduced (largest "
                     f"difference {e.max_abs_diff:.3g}; tolerance {e.tolerance:g})"
                     + (f"; differing: {', '.join(e.mismatched[:5])}" if e.mismatched else "")
                     + (f"; not recomputed: {', '.join(e.missing[:5])}" if e.missing else "")
                     + ".")
    if report.stage_keys_same is not None:
        same = "the same" if report.stage_keys_same else "different (another engine version)"
        lines.append(f"  The result stages' keys: {same}.")
    if report.files_different:
        lines.append(f"  Files that differ: {', '.join(report.files_different)}.")
    lines += [f"  {m}" for m in report.messages]
    lines.append("Reproduced." if report.reproduced else "Not reproduced.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m turbotab.replay",
        description="Replay a TurboTab export bundle: check its input files against their hashes, "
                    "rebuild the analysis from the decision log in a fresh home, and compare the "
                    "model matrix and every reported estimate.")
    parser.add_argument("bundle", help="the export bundle (.zip)")
    parser.add_argument("--data", help="the table's input file (default: the path the record names)")
    parser.add_argument("--file", action="append", default=[], metavar="ID=PATH",
                        help="a joined file, by its id or name (repeatable)")
    parser.add_argument("--home", help="an empty folder to replay in (default: a new temporary one)")
    parser.add_argument("--keep", action="store_true", help="keep the temporary home afterwards")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds to wait (default 3600)")
    args = parser.parse_args(argv)
    given: dict[str, str] = {}
    for item in args.file:
        key, sep, value = item.partition("=")
        if not sep:
            parser.error(f"--file takes ID=PATH, not {item!r}")
        given[key] = value
    report = run(args.bundle, data=args.data, files=given, home=args.home, keep=args.keep,
                 timeout=args.timeout)
    if args.json:
        sys.stdout.write(report.model_dump_json(indent=1) + "\n")
    else:
        sys.stdout.write(summary(report) + "\n")
    if report.refused:
        return 2
    return 0 if report.reproduced else 1


__all__ = ["EstimateCheck", "InputCheck", "MatrixCheck", "Refused", "ReplayReport", "close",
           "compare", "compare_estimates", "load_bundle", "main", "run", "summary",
           "verify_inputs"]
