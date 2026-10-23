"""The reading restriction (SWARM_PLAN §9): the only fields of a corpus record the theorist and the
experimentalist read, and the sealed file they never read.

The theorist proposes from the corpus, and in the shuffled control from a copy of it whose markers
are permuted across datasets with records carrying a planted effect appended (``shuffle.py``). A
proposal from the copy should come only from the theorist's prior or from the planted effect, so
the view holds only what a structured proposal is built from (``proposal.py``: the decision, its
materiality as the engine measured it, the live markers, the loss, the refusals), as numbers and as
the engine's and the vocabulary's own names. It holds none of the analyst's words, none of the
names the question and the decisions give the dataset's columns but those a live marker reads
(``missingness`` keys its shares by column, and ``estimand`` names its exposure), nothing that says
a dataset is a control, and none of the ids and keys a planted record keeps whole from its
template. A planted record's decisions and its earlier fits are its template's, and the view
serves them as it serves the template's, so a reader who compares records finds the planted
records beside their template (``shuffle.py`` accepts the clones and says so); each planted
record's markers are a real record's of its dataset. The copy keeps each record's own live markers
of the purpose and its loss, which restate its question and fits (``shuffle.restates``), so in the
view they agree with ``question.purpose`` and the fits' ``primary_metric``. The view does not hide
by itself that a record is in a copy: where the copy permutes a regime marker that its fits
restate (another dataset's ``outcome_type`` or ``event_share`` beside a fit's metric of another
task), the two can disagree, and the view serves both as the record holds them. Every field of
the record is either served (:data:`FIELDS`) or withheld with why (:data:`WITHHELD`); a field the
record gains later is served by no one until it is classed. Withheld, by kind:

* what the blind view takes out (``record.redacted``: the analyst's model, persona, effort, prompt
  version and spend), and every time, since the gaps between an agent's commands can tell its
  model; the view is built from the redacted view, so even a field served by mistake would read
  as redacted;
* the analyst's own words and tags (reasons, previews' reasons, tags, wants, the account): they
  are written about the dataset the run read, which the copy keeps beside another dataset's
  markers;
* the ids and keys a planted record keeps whole from its template (``shuffle.KEPT``: its fit
  keys, plan digests, fit and decision ids, its server, its export), so fits and decisions are
  named by their seq;
* whether a dataset is a control and its known answer or truth, which a claim is graded against
  (the referee is never shown them either);
* what names the dataset, its columns or its question (its id, files, codebook and card; the
  question's target and estimand; each decision's option); the view labels each dataset
  ``dataset_N`` in the order it first appears among the records in run-id order;
* the engine's words beside a number (whys, a refusal's message, a fit's folds), which can count
  the dataset's own rows; the codes and the numbers are kept;
* the result and each fit's scores: post hoc, keyed by nothing; and the regime's readings the
  markers restate;
* the post hoc markers, the signal and the result (``vocabulary.POST_HOC``): they restate the
  run's result and its fit's scores, the held-out score among them, and key nothing; a shuffled
  copy permutes them with the regime, so there they are another dataset's, beside fits they do not
  describe. The view's markers, the regime's and each fit's, are the live ones only.

**The sealed file** is never read: a path that is the sealed file, or a folder that holds it, is
refused, and so is a folder holding any other file that is not a record (a failed run's, under
``failed/``, is not read at all). A refusal never repeats what the file holds.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from research.swarm import record as R
from research.swarm import vocabulary as V

# The view's fields, each a section and a key of the record, served as named here: the dataset's
# id as its label, fits named by their seq, and each nested entry with its numbers and names only
# (:func:`view`).
FIELDS: tuple[str, ...] = (
    "run.id",
    "dataset.id",
    "question.purpose",
    "regime.markers",
    "decisions.seq", "decisions.kind", "decisions.source", "decisions.phase",
    "decisions.materiality",
    "attempts.seq", "attempts.decisions", "attempts.changed", "attempts.families",
    "attempts.primary_metric", "attempts.markers", "attempts.movement", "attempts.change",
    "surfacings.refusals", "surfacings.failed_fits",
)

REDACTED_WHY = ("It is redacted for a blind reader (record.redacted): it can tell the analyst's "
                "model or persona.")
TIME_WHY = ("It is a time: the gaps between an agent's commands can tell its model, so the blind "
            "view keeps their order only, and a proposal needs none.")
WORDS_WHY = ("It is the analyst's own words or tags, written about the dataset its run read, which "
             "a shuffled copy keeps beside another dataset's markers: context for hypotheses, "
             "never evidence (SWARM_PLAN §4).")
FINGERPRINT_WHY = ("A planted record keeps it whole from its template (shuffle.KEPT), so it would "
                   "tell the planted records apart; the view names fits and decisions by their "
                   "seq.")
CONTROL_WHY = ("Whether a dataset is a control, and its known answer or truth, are what a claim is "
               "graded against, never what it is read from.")
IDENTITY_WHY = ("It names the dataset, its columns or its question, which a shuffled copy keeps as "
                "the dataset's own beside another dataset's markers; the view labels each dataset "
                "in the order it first appears.")
ENGINE_WORDS_WHY = ("It is the engine's words, or a count beside a number, which can count the "
                    "dataset's own rows, and a shuffled copy does not permute them; the view keeps "
                    "the codes and the numbers.")
RESULT_WHY = ("It is the run's result or a fit's scores, post hoc and keyed by nothing (SWARM_PLAN "
              "§1); a decision's materiality and each fit's signed change carry what a proposal "
              "rests on.")
RESTATED_WHY = "The regime's markers hold this reading; the view reads it there."
POST_HOC_WHY = ("It is a post hoc marker: it restates the run's result and its fit's scores, the "
                "held-out score among them, known only after the run and keyed by nothing "
                "(SWARM_PLAN §1); a shuffled copy permutes it with the regime, so there it is "
                "another dataset's, beside fits it does not describe. The view serves the live "
                "markers only.")
DECLARED_WHY = ("A declared marker is quoted from the dataset's documentation, not read live: a "
                "region is drawn on live markers, and the quote names the dataset.")
OTHER_WHY = "It is the run's own bookkeeping, which no proposal is built from."


def _all(why: str, *paths: str) -> dict[str, str]:
    return {p: why for p in paths}


WITHHELD: dict[str, str] = {
    **_all(REDACTED_WHY, "run.analyst_model", "run.persona", "run.effort", "run.prompt_version",
           "run.spend", "run.spend_why"),
    **_all(TIME_WHY, "run.started", "run.ended", "decisions.at", "attempts.at",
           "surfacings.refusals.at", "surfacings.failed_fits.at"),
    **_all(WORDS_WHY, "decisions.reason", "decisions.tag", "decisions.examined",
           "attempts.reasons", "wants", "account"),
    **_all(FINGERPRINT_WHY, "run.server", "decisions.id", "attempts.id", "attempts.fit_key",
           "attempts.plan_digest", "export", "surfacings.failed_fits.key"),
    **_all(CONTROL_WHY, "dataset.control", "dataset.fixture", "dataset.known", "replay"),
    **_all(IDENTITY_WHY, "dataset.files", "dataset.codebook", "dataset.card", "question.target",
           "question.estimand", "question.intended_use", "question.stated", "question.paper",
           "regime.event", "regime.lenses", "decisions.option", "decisions.answers",
           "attempts.target", "replication"),
    **_all(ENGINE_WORDS_WHY, "regime.markers.basis", "regime.markers.why",
           "attempts.markers.basis", "attempts.markers.why", "attempts.markers_why",
           "decisions.materiality.why", "attempts.movement.words", "attempts.movement.label",
           "attempts.movement.why", "attempts.movement_why", "attempts.change.why",
           "attempts.change.folds", "attempts.change.test_share", "attempts.change_why",
           "attempts.folds", "attempts.held_out", "attempts.withheld", "surfacings.concerns",
           "surfacings.refusals.message", "surfacings.refusals.exits",
           "surfacings.failed_fits.error", "surfacings.failed_fits.refusal", "claims"),
    **_all(RESULT_WHY, "result", "attempts.scores", "attempts.holdout", "attempts.holdout_why",
           "attempts.at_opening", "attempts.effects"),
    **_all(RESTATED_WHY, "regime.n_rows", "regime.n_analyzed", "regime.task", "regime.events",
           "regime.missing", "regime.design"),
    **_all(POST_HOC_WHY, *(f"regime.markers.{name}" for name in V.POST_HOC)),
    **_all(DECLARED_WHY, "dataset.declared"),
    **_all(OTHER_WHY, "version", "run.design", "run.engine_commit", "run.engine_modified",
           "run.engine_version", "run.vocabulary", "run.redacted", "run.fits", "run.fit_seconds",
           "run.fits_unseen", "run.fit_budget", "run.help", "decisions.post_seal",
           "decisions.after_estimates", "decisions.early", "attempts.purpose",
           "attempts.withdrawn", "attempts.reinstated", "attempts.press", "attempts.seconds",
           "attempts.workers", "attempts.seed", "attempts.changed_after_seal", "attempts.locked",
           "surfacings.refusals.then"),
}

# The sealed file's own keys (``shuffle.write``): a JSON object holding both is the sealed file.
SEALED_KEYS = ("derangement", "planted")
SEALED_REFUSAL = ("It is the shuffled copy's sealed file, or a folder holding it: the claim "
                  "cycle's tools never read the sealed file (SWARM_PLAN §9 step 5).")
FAILED = "failed"


class ReadRefused(ValueError):
    """What the reading tools refuse to read, in words that never repeat what the file holds."""


def is_sealed(obj: Any) -> bool:
    """Whether ``obj`` (a parsed JSON document) is a sealed file's body."""
    return isinstance(obj, dict) and all(k in obj for k in SEALED_KEYS)


def _parsed(path: Path) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def read_records(folder: str | Path) -> list[R.Record]:
    """Every record of the corpus in ``folder`` (``*.json`` directly in it), validated, in run-id
    order; refused for a path that is no folder, a folder holding the sealed file or any other
    file that is not a record, and a folder holding no record."""
    folder = Path(folder)
    if not folder.is_dir():
        if folder.is_file() and is_sealed(_parsed(folder)):
            raise ReadRefused(SEALED_REFUSAL)
        raise ReadRefused(f"{folder.name} is not a folder of records; the tools read a corpus "
                          f"folder.")
    out = []
    for path in sorted(folder.glob("*.json")):
        data = _parsed(path)
        if is_sealed(data):
            raise ReadRefused(SEALED_REFUSAL)
        try:
            out.append(R.validate(data))
        except Exception:  # noqa: BLE001 - a file that is not a record, whatever it holds
            raise ReadRefused(f"{path.name} in {folder.name} is not a record (it does not read as "
                              f"one), so the folder is not a corpus; nothing in it is read."
                              ) from None
    if not out:
        raise ReadRefused(f"{folder.name} holds no record.")
    return sorted(out, key=lambda r: r.run.id)


def labels(records: Iterable[Any]) -> dict[str, str]:
    """Each dataset's label, ``dataset_N``, in the order it first appears among ``records`` in
    run-id order."""
    out: dict[str, str] = {}
    for rec in sorted(records, key=lambda r: _dumped(r)["run"]["id"]):
        dataset = _dumped(rec)["dataset"]["id"]
        out.setdefault(dataset, f"dataset_{len(out) + 1}")
    return out


def _dumped(rec: Any) -> dict[str, Any]:
    return rec.model_dump(mode="json", by_alias=True) if isinstance(rec, R.Record) else rec


def _marker(m: dict[str, Any]) -> dict[str, Any]:
    return {k: m.get(k) for k in ("name", "axis", "timing", "value")}


def _live(markers: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The live markers of ``markers``, each with its name, axis, timing and value."""
    return [_marker(m) for m in markers if m.get("timing") == "live"]


MOVED = ("value", "band", "calibrated", "crosses", "changes_question", "not_measurable",
         "instrument")
CHANGED = ("family", "metric", "paired", "difference", "better", "se", "df", "ci_low", "ci_high",
           "excludes_zero", "half_width")
SPEC = ("estimate", "low", "high", "se")


def _moved(entry: dict[str, Any], seq: Callable[[Any], Any]) -> dict[str, Any]:
    measured = entry.get("movement") or {}
    return {"family": entry.get("family"), "against": seq(entry.get("against")),
            **{k: measured.get(k) for k in MOVED}}


def _spec(spec: Any) -> dict[str, Any] | None:
    return None if not isinstance(spec, dict) else {k: spec.get(k) for k in SPEC}


def _materiality(m: dict[str, Any], seq: Callable[[Any], Any]) -> dict[str, Any]:
    moved = m.get("movement")
    departures = m.get("departures")
    return {"source": m.get("source"), "attempt": seq(m.get("attempt")),
            "movement": None if moved is None else [_moved(e, seq) for e in moved],
            "primary": _spec(m.get("primary")),
            "departures": None if departures is None else [
                {"option": d.get("option"), "spec": _spec(d.get("spec"))} for d in departures]}


def view(rec: Any, label: str) -> dict[str, Any]:
    """The view of one record (the module docstring), its dataset labeled ``label``."""
    data = R.redacted(rec)
    fits = {a["id"]: a["seq"] for a in data["attempts"]}

    def seq(fit_id: Any) -> Any:
        return None if fit_id is None else fits.get(fit_id)

    return {
        "run": {"id": data["run"]["id"]},
        "dataset": {"id": label},
        "question": {"purpose": data["question"]["purpose"]},
        "regime": {"markers": _live(data["regime"]["markers"])},
        "decisions": [{"seq": d["seq"], "kind": d["kind"], "source": d["source"],
                       "phase": d["phase"], "materiality": _materiality(d["materiality"], seq)}
                      for d in data["decisions"]],
        "attempts": [{"seq": a["seq"], "decisions": a["decisions"], "changed": a["changed"],
                      "families": a["families"], "primary_metric": a["primary_metric"],
                      "markers": None if a["markers"] is None else _live(a["markers"]),
                      "movement": [_moved(e, seq) for e in a["movement"]],
                      "change": [{**{k: c.get(k) for k in CHANGED}, "against": seq(c["against"])}
                                 for c in a["change"]]}
                     for a in data["attempts"]],
        "surfacings": {
            "refusals": [{k: r.get(k) for k in ("command", "code", "by", "stage")}
                         for r in data["surfacings"]["refusals"]],
            "failed_fits": [{"command": f["command"], "stage": f["stage"], "kind": f["kind"],
                             "code": (f.get("refusal") or {}).get("code")}
                            for f in data["surfacings"]["failed_fits"]]},
    }


def views(records: Sequence[Any]) -> list[dict[str, Any]]:
    """The view of each record (a :class:`record.Record` or its JSON), in run-id order."""
    named = labels(records)
    ordered = sorted(records, key=lambda r: _dumped(r)["run"]["id"])
    return [view(rec, named[_dumped(rec)["dataset"]["id"]]) for rec in ordered]


def read(folder: str | Path) -> list[dict[str, Any]]:
    """The view of every record of the corpus in ``folder`` (:func:`read_records`)."""
    return views(read_records(folder))


def check_fields(fields: Iterable[str]) -> list[str]:
    """``fields``, each a field of the view; refused, with why, where one is withheld or unknown."""
    out = []
    for f in fields:
        if f in FIELDS:
            out.append(f)
            continue
        parts = f.split(".")
        held = next((WITHHELD[p] for p in (".".join(parts[:i]) for i in range(len(parts), 0, -1))
                     if p in WITHHELD), None)
        if held is not None:
            raise ReadRefused(f"{f} is withheld from the claim cycle's tools. {held}")
        raise ReadRefused(f"{f} is not a field of the view; its fields are {', '.join(FIELDS)}.")
    return out


def project(v: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    """``v`` with only the run's id and ``fields`` (each of :data:`FIELDS`)."""
    out: dict[str, Any] = {"run": {"id": v["run"]["id"]}}
    for f in check_fields(fields):
        section, key = f.split(".", 1)
        if isinstance(v[section], list):
            items = out.setdefault(section, [{} for _ in v[section]])
            for into, item in zip(items, v[section]):
                into[key] = item[key]
        else:
            out.setdefault(section, {})[key] = v[section][key]
    return out


def index(vs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per record: its run, its dataset's label, its purpose and its counts."""
    return [{"run": v["run"]["id"], "dataset": v["dataset"]["id"],
             "purpose": v["question"]["purpose"], "decisions": len(v["decisions"]),
             "attempts": len(v["attempts"])} for v in vs]


__all__ = ["FIELDS", "POST_HOC_WHY", "ReadRefused", "SEALED_KEYS", "SEALED_REFUSAL", "WITHHELD",
           "check_fields", "index", "is_sealed", "labels", "project", "read", "read_records",
           "view", "views"]
