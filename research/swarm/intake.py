"""The swarm's dataset intake (DRIVER_PLAN D4, SWARM_PLAN §5), generalized from the Sullivan
replication's (``research/replication/sullivan2021/engine/intake.py``).

Per dataset it writes three files under ``research/datasets/<id>/`` (committed):

* ``manifest.json``: where every file came from. For a download, each file's URL, the dates its
  codebook page states ("First Published", "Last Revised"), the download date, the server's
  Last-Modified header, its size and its SHA-256; for a file of this repository or a generated
  table, its path or generator. Then the license, and the analytic table the run uploads, with
  its size and SHA-256.
* ``card.json``: the question card an analyst reads (:data:`CARD_FIELDS`): the question, the
  population, the purpose (Predict or Estimate), the outcome (and under Estimate the exposure),
  for a replication the paper and its stated question, the table the run uploads, and every step
  taken to prepare it before the upload. It holds no known answer, and no number, in digits or
  in words, that is an answer's (as stated, as a percent or as a proportion) rounded to the
  card's own decimal places, beyond the coincidences :data:`SHARED_NUMBERS` names, each in the
  form it is pinned in ("1" pinned does not excuse "1.0" or "one").
* ``answers.json``: what the checks compare a run against (:data:`ANSWER_KINDS`): a paper's
  numbers, a fixture's declared truth, a decoy's "no finding is real", or that nothing is known.
  No analyst reads it.

Data files are ignored by git (``research/data/``): NHANES files in
``research/data/nhanes/<cycle>/``, the analytic tables in ``research/data/datasets/<id>/``.

Downloads come from wwwn.cdc.gov only (:data:`HOSTS`; any other host is refused), with a plain
user agent and nothing that identifies anyone: a URL with a query string is refused. A file already on disk is not fetched again; its
entry is recomputed from its bytes, and keeps its download date only while those bytes are the
ones the manifest recorded. A download that is not what its name says is refused and nothing is
written: a transport file must begin with the SAS XPORT header, and a codebook page must be one
(NHANES's old URLs answer 200 with an HTML "Page Not Found").

    scripts/py -m research.swarm.intake [<id> ...]          every dataset when none is named
    scripts/py -m research.swarm.intake --check [<id> ...]

``--check`` fetches nothing and fails when a file on disk no longer matches its manifest, or a card
or an answers file does not validate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterator

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATASETS = REPO / "research" / "datasets"
DATA = REPO / "research" / "data"
NHANES = DATA / "nhanes"
TABLES = DATA / "datasets"
AGENT = "Mozilla/5.0"
NHANES_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{file}.{ext}"
HOSTS = ("wwwn.cdc.gov",)  # the only host the intake downloads from
XPT_MAGIC = b"HEADER RECORD*******LIB"  # the first record of a SAS transport file, v5 or v8
NHANES_LICENSE = {
    "license": "public domain",
    "basis": ("a work of the U.S. federal government (NCHS, CDC), not subject to copyright "
              "(17 U.S.C. 105); the files' pages state no license of their own"),
}
REPO_LICENSE = {"license": "MIT", "basis": "the repository's LICENSE: the table is generated here"}


# ── downloading, with a manifest ─────────────────────────────────────────────


class FetchRefused(RuntimeError):
    """A download that is not the kind of file its URL names."""


def rel(path: Path) -> str:
    """``path`` relative to the repository where it lies inside it, else as given."""
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def refusal(data: bytes, kind: str) -> str | None:
    """Why ``data`` is not a file of ``kind`` ("xpt" or "htm"), or None when it is."""
    if kind == "xpt":
        if not data.startswith(XPT_MAGIC):
            return f"not a SAS transport file: it begins {data[:40]!r}"
        return None
    if kind == "htm":
        low = data.lower()
        if b"page not found" in low or b"codebook" not in low:
            return "not an NHANES codebook page (no codebook, or a Page Not Found page)"
        return None
    raise ValueError(f"unknown kind {kind!r}")


def fetch(url: str, dest: Path, kind: str) -> dict[str, str | None]:
    """Download ``url`` to ``dest``, refusing bytes that are not a file of ``kind``; returns the
    server's Last-Modified and Content-Type headers. Nothing is written when it refuses."""
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=300) as r:
        data = r.read()
        headers = {"last_modified_header": r.headers.get("Last-Modified"),
                   "content_type": r.headers.get("Content-Type")}
    problem = refusal(data, kind)
    if problem:
        raise FetchRefused(f"{url}: {problem}")
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return headers


def page_dates(htm: Path) -> dict[str, str | None]:
    """The dates a codebook page states ("First Published", "Last Revised"), None where absent."""
    text = htm.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str | None] = {}
    for label in ("First Published", "Last Revised"):
        m = re.search(label + r":\s*([^<]+)<", text)
        out[label.lower().replace(" ", "_")] = m.group(1).strip() if m else None
    return out


def obtain(url: str, dest: Path, kind: str, before: dict | None, *,
           pause: float = 0.5) -> dict[str, Any]:
    """The manifest entry of ``url``'s file at ``dest``, fetched when it is not on disk."""
    before = before or {}
    if dest.exists():
        problem = refusal(dest.read_bytes(), kind)
        if problem:
            raise FetchRefused(f"{rel(dest)} on disk: {problem}")
        digest = sha256(dest)
        same = before.get("sha256") == digest
        kept = {k: before.get(k) if same else None
                for k in ("downloaded", "last_modified_header", "content_type")}
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        kept = {"downloaded": time.strftime("%Y-%m-%d"), **fetch(url, dest, kind)}
        digest = sha256(dest)
        time.sleep(pause)
    return {"file": rel(dest), "url": url, "downloaded": kept["downloaded"],
            "bytes": dest.stat().st_size, "sha256": digest,
            "last_modified_header": kept["last_modified_header"],
            "content_type": kept["content_type"]}


def download(spec: dict[str, Any], old: dict | None = None, *, folder: Path = NHANES,
             pause: float = 0.5, log: Callable[[str], None] = print,
             hosts: tuple[str, ...] = HOSTS) -> list[dict[str, Any]]:
    """Every file ``spec`` names (``{"url": template, "cycles": {cycle: {"year", "files"}}}``, the
    Sullivan replication's ``files.json`` shape), each with its codebook page, into
    ``folder/<cycle>/``. A transport file's entry carries its page's dates (``release``) and the
    page's URL (``codebook``). A URL on another host than ``hosts``, or with a query string, is
    refused before anything is fetched."""
    from urllib.parse import urlsplit
    known = {e["file"]: e for e in (old or {}).get("files", [])}
    out: list[dict[str, Any]] = []
    for cycle, c in spec["cycles"].items():
        for name in c["files"]:
            pair = {}
            for ext in ("xpt", "htm"):
                url = spec["url"].format(year=c["year"], file=name, ext=ext)
                parts = urlsplit(url)
                if parts.hostname not in hosts or parts.query or parts.username:
                    raise FetchRefused(f"{url}: only {', '.join(hosts)}, with no query string")
                dest = Path(folder) / cycle / f"{name}.{ext}"
                pair[ext] = obtain(url, dest, ext, known.get(rel(dest)), pause=pause)
            pair["xpt"]["codebook"] = pair["htm"]["url"]
            pair["xpt"]["release"] = page_dates(Path(folder) / cycle / f"{name}.htm")
            out += [pair["xpt"], pair["htm"]]
            log(f"{cycle} {name}: {pair['xpt']['bytes']:,} bytes")
    return out


def table_entry(path: Path, frame: Any) -> dict[str, Any]:
    return {"file": rel(path), "format": path.suffix.lstrip("."), "rows": int(len(frame)),
            "columns": int(frame.shape[1]), "bytes": path.stat().st_size, "sha256": sha256(path)}


def changed(manifest: dict[str, Any]) -> list[str]:
    """The manifest's files (and its table) missing on disk or whose bytes differ from it."""
    bad = []
    for e in [*manifest.get("files", []), *([manifest["table"]] if "table" in manifest else [])]:
        path = REPO / e["file"]
        if not path.is_file() or sha256(path) != e["sha256"]:
            bad.append(e["file"])
    return bad


def read_xpt(path: Path) -> Any:
    """A transport file as a frame, through TurboTab's own reader (``turbotab.core.xport``), which
    reads a SAS zero as 0 and SAS missing values as blank."""
    from turbotab.core.xport import read_frame
    return read_frame(path)


# ── cards and answers: the schema ────────────────────────────────────────────

PURPOSES = {"Predict": "prediction", "Estimate": "inference"}  # the card's word → the engine's
DESIGNS = ("replication", "analysts")  # SWARM_PLAN §1's designs 1 and 2
CARD_FIELDS = {
    "id": "the dataset's id, its folder's name",
    "design": "replication (a published analysis reproduced) or analysts (a question of our own)",
    "purpose": "Predict or Estimate",
    "question": "the question the run answers, in plain words, ending with a question mark",
    "population": "who the rows are, and who the question is about",
    "outcome": "{column, meaning}: the column the question is about",
    "exposure": "{column, meaning}: under Estimate only, the column whose effect is estimated",
    "replication": "for a replication only: {paper: {citation, doi, pmid}, stated_question}",
    "table": "{path, format}: the analytic table the run uploads, relative to the repository",
    "source": "where the table's data come from, in plain words",
    "documentation": "optional: what the source's documentation states that the table cannot show",
    "prepared": "every step taken to prepare the table before the upload, in order",
    "manifest": "the manifest's path: every file's origin, size and SHA-256",
}
REQUIRED = ("id", "design", "purpose", "question", "population", "outcome", "table", "source",
            "prepared", "manifest")
ANSWER_KINDS = {
    "replication": "the paper's numbers, as its replication notes quote them",
    "known_answer": "a fixture's declared truth (truths.FIXTURE_TRUTHS and its generator)",
    "decoy": "a table with no signal: no finding is real",
    "no_known_answer": "real data whose answer nobody knows: the protocol checks only",
}
# Bibliographic identifiers: they name the paper and say nothing about the data, so the number
# scan (:func:`card_numbers`) does not read them.
BIBLIOGRAPHIC = ("citation", "doi", "pmid")
# A number as prose writes it: digits, with or without thousands commas, and a decimal part
# ("16,913", "0.79"), or a decimal part alone (".79"). It is read wherever no letter, digit,
# underscore or point comes just before it, so DBQ700, item_05 and log10 are names, and whatever
# follows it: "126mg", "210cases", "2.6x" and "74%" are 126, 210, 2.6 and 74.
NUMBER = re.compile(r"(?<![\w.])(?:(\d{1,3}(?:,\d{3})+(?!\d)|\d+)(\.\d+)?|(\.\d+))")
# A number as prose writes it in words, to no decimal places: zero to nineteen, the tens, and a
# ten with its unit ("twenty-five", "twenty five"), in any case and only as a whole word, so
# "someone", "often", "weight" and item_one hold none.
UNIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
              "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
              "seventeen", "eighteen", "nineteen")
TEN_WORDS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
WORD_VALUES = {**{w: i for i, w in enumerate(UNIT_WORDS)},
               **{w: 20 + 10 * i for i, w in enumerate(TEN_WORDS)}}
NUMBER_WORD = re.compile(
    rf"\b(?:({'|'.join(TEN_WORDS)})(?:[-\s]+({'|'.join(UNIT_WORDS[1:10])}))?"
    rf"|({'|'.join(sorted(UNIT_WORDS, key=len, reverse=True))}))\b", re.IGNORECASE)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _column(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"column", "meaning"} \
        and _text(value["column"]) and _text(value["meaning"])


def validate_card(card: Any, id_: str | None = None) -> list[str]:
    """Every way ``card`` fails the card's schema (:data:`CARD_FIELDS`); empty when it passes."""
    if not isinstance(card, dict):
        return ["the card is not an object"]
    errors = [f"missing {k}" for k in REQUIRED if k not in card]
    errors += [f"unknown field {k}" for k in card if k not in CARD_FIELDS]
    if id_ is not None and card.get("id") != id_:
        errors.append(f"id {card.get('id')!r} is not its folder's, {id_!r}")
    if card.get("design") not in DESIGNS:
        errors.append(f"design {card.get('design')!r} is not one of {list(DESIGNS)}")
    if card.get("purpose") not in PURPOSES:
        errors.append(f"purpose {card.get('purpose')!r} is not one of {list(PURPOSES)}")
    for k in ("question", "population", "source"):
        if k in card and not _text(card[k]):
            errors.append(f"{k} is not a sentence")
    if _text(card.get("question")) and not card["question"].rstrip().endswith("?"):
        errors.append("the question does not end with a question mark")
    if "outcome" in card and not _column(card["outcome"]):
        errors.append("outcome is not {column, meaning}")
    if card.get("purpose") == "Estimate" and not _column(card.get("exposure")):
        errors.append("an Estimate card names no exposure {column, meaning}")
    if card.get("purpose") == "Predict" and "exposure" in card:
        errors.append("a Predict card names an exposure")
    rep = card.get("replication")
    if card.get("design") == "replication":
        paper = rep.get("paper") if isinstance(rep, dict) else None
        if not (isinstance(paper, dict) and _text(paper.get("citation"))
                and _text(rep.get("stated_question"))):
            errors.append("a replication card names no paper citation and stated question")
        elif set(paper) - set(BIBLIOGRAPHIC):
            errors.append(f"the paper holds {sorted(set(paper) - set(BIBLIOGRAPHIC))}")
        elif set(rep) != {"paper", "stated_question"}:
            errors.append(f"replication holds {sorted(set(rep) - {'paper', 'stated_question'})}")
    elif "replication" in card:
        errors.append("a card of our own question names a replication")
    table = card.get("table")
    if not (isinstance(table, dict) and set(table) == {"path", "format"}
            and _text(table.get("path")) and table.get("format") in ("csv", "parquet")):
        errors.append("table is not {path, format: csv or parquet}")
    elif Path(table["path"]).is_absolute() or not table["path"].startswith("research/data/") \
            or Path(table["path"]).suffix != f".{table['format']}":
        errors.append(f"table path {table['path']!r} is not a {table['format']} file under "
                      f"research/data/, relative to the repository")
    for k in ("prepared", "documentation"):
        if k in card and not (isinstance(card[k], list) and all(_text(s) for s in card[k])):
            errors.append(f"{k} is not a list of sentences")
    if "manifest" in card and card["manifest"] != f"research/datasets/{card.get('id')}/manifest.json":
        errors.append(f"manifest {card['manifest']!r} is not research/datasets/<id>/manifest.json")
    return errors


def validate_answers(answers: Any, id_: str | None = None) -> list[str]:
    """Every way ``answers`` fails the answers file's schema; empty when it passes."""
    if not isinstance(answers, dict):
        return ["the answers file is not an object"]
    errors = []
    if id_ is not None and answers.get("id") != id_:
        errors.append(f"id {answers.get('id')!r} is not its folder's, {id_!r}")
    kind = answers.get("kind")
    if kind not in ANSWER_KINDS:
        errors.append(f"kind {kind!r} is not one of {list(ANSWER_KINDS)}")
    if not _text(answers.get("about")):
        errors.append("about is not a sentence")
    truths = answers.get("truths")
    if not isinstance(truths, list):
        errors.append("truths is not a list")
    else:
        for i, t in enumerate(truths):
            if not (isinstance(t, dict) and {"name", "value", "source"} <= set(t)
                    and _text(t["name"]) and _text(t["source"])):
                errors.append(f"truth {i} is not {{name, value, source}}")
    if kind == "decoy" and answers.get("real_findings") != []:
        errors.append("a decoy's real_findings is not the empty list")
    if kind != "decoy" and "real_findings" in answers:
        errors.append("real_findings is a decoy's alone")
    if kind == "known_answer":
        from turbotab.core.tests.truths import FIXTURE_TRUTHS
        if answers.get("fixture_key") not in FIXTURE_TRUTHS:
            errors.append(f"fixture_key {answers.get('fixture_key')!r} is not in FIXTURE_TRUTHS")
    if kind in ("replication", "known_answer") and isinstance(truths, list) and not truths:
        errors.append(f"a {kind} answers file states no truth")
    return errors


def written(value: float, places: int) -> str:
    """A number in digits as a card writes it, by absolute value: 1 to one place is "1.0", to none
    "1"; ".79" is "0.79" and "16,913" is "16913"."""
    return f"{value:.{places}f}"


def _numbers(value: Any, skip: Callable[[tuple[str, ...]], bool], at: tuple[str, ...] = ()
             ) -> Iterator[tuple[float, int, str]]:
    """Every number in ``value``, in digits (:data:`NUMBER`) or in words (:data:`NUMBER_WORD`), by
    absolute value, with the decimal places it is written to and its form: digits as
    :func:`written` gives them, words in lower case joined by a hyphen ("twenty-five")."""
    if skip(at):
        return
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int):
        yield float(abs(value)), 0, written(abs(value), 0)
    elif isinstance(value, float):
        if math.isfinite(value):
            places = max(0, -Decimal(repr(value)).as_tuple().exponent)
            yield abs(value), places, written(abs(value), places)
    elif isinstance(value, str):
        for whole, fraction, alone in NUMBER.findall(value):
            part = fraction or alone
            number, places = abs(float(whole.replace(",", "") + part)), max(0, len(part) - 1)
            yield number, places, written(number, places)
        for ten, unit, alone in NUMBER_WORD.findall(value):
            words = [w.lower() for w in (ten, unit, alone) if w]
            yield float(sum(WORD_VALUES[w] for w in words)), 0, "-".join(words)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _numbers(v, skip, (*at, str(k)))
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _numbers(v, skip, at)


def card_readings(card: dict[str, Any]) -> set[tuple[float, int, str]]:
    """Every number the card states, in digits or in words, in any field (its paper's
    bibliographic identifiers aside), by absolute value, each with the decimal places the card
    writes it to and its form (:func:`_numbers`)."""
    return set(_numbers(card, lambda at: len(at) == 3 and at[:2] == ("replication", "paper")
                        and at[2] in BIBLIOGRAPHIC))


def card_numbers(card: dict[str, Any]) -> set[float]:
    """Every number the card states (:func:`card_readings`), by absolute value."""
    return {value for value, _, _ in card_readings(card)}


def answer_numbers(answers: dict[str, Any]) -> set[float]:
    """Every number the answers' truths state (each truth's value), in digits or in words, by
    absolute value."""
    return {value for value, _, _ in
            _numbers([t.get("value") for t in answers.get("truths") or []], lambda at: False)}


def rounds_to(answer: float, stated: float, places: int) -> bool:
    """Whether ``stated``, written to ``places`` decimal places, is ``answer`` rounded there: it
    lies within half a unit of that last place (a tie counts, so half up and half to even both
    do). 0.742 is 0.74194 rounded, .79 is 0.79, and 8 is 8.1."""
    return abs(answer - stated) <= 0.5 * 10.0 ** -places * (1 + 1e-9)


def shared_numbers(card: dict[str, Any], answers: dict[str, Any]) -> set[str]:
    """The numbers the card states (:func:`card_readings`) that are an answer's number rounded to
    the card's own decimal places (:func:`rounds_to`), each in its form ("0.79", "1.0", "five"):
    the answer as it is stated, times 100 (a proportion written as a percent: 0.74194 is "74%")
    or divided by 100 (a percent written as a proportion: a sensitivity of 82 is "0.82")."""
    stated = [s for a in answer_numbers(answers) for s in (a, a * 100, a / 100)]
    return {form for value, places, form in card_readings(card)
            if any(rounds_to(a, value, places) for a in stated)}


# A number a card states that is an answer's rounded to the card's places (:func:`shared_numbers`:
# the answer as stated, times 100 or divided by 100), where the two are not the same fact. Each is
# keyed by its form, the number as the card writes it: digits by their value to the card's
# places ("1", "1.0"), words by the words ("one"). The intake refuses every other shared number,
# a pinned value in any other form among them ("1.0" or "one" where only "1" is pinned), and the
# tests pin this list. A whole 0 or 1 in a card matches every answer within 0.5 of it, and every
# one up to 50, or from 50 to 150, divided by 100, so a binary column's coding meets most answers.
SHARED_NUMBERS: dict[str, dict[str, str]] = {
    "bang2009": {
        "1": "the outcome column's coding (diabetes_fpg is 1 at or above the cut); a factor's "
             "points in the paper's score are also 1, its odds ratios, confidence bounds, "
             "likelihood ratios and AUC from 0.5 to 1.5 round to 1, and so, divided by 100, do "
             "Table 3's percentages from 55 to 100 and Table 2's age bands' 50, 59 and 60",
        "one": "the prepared steps' 'one under the other' and 'one row per participant'; it meets "
               "the answers as 1 does",
        "0": "the outcome column's coding (diabetes_fpg is 0 below the cut); the paper's P "
             "values, negative likelihood ratios and the bounds up to 0.5 round to 0, as do its P "
             "values below 0.005 times 100 and every number up to 50 divided by 100"},
    "sullivan2021": {
        "20": "the paper's stated inclusion, aged 20 or older (its flow's 'Age < 20' step); the "
              "flow also excludes 20 people for reporting only water, and Table 4's insulin SE of "
              "0.2 times 100 is 20",
        "10": "the base of the insulin crosswalk's logarithm; the flow also excludes 10 people "
              "for having no self-rating, and Table 4's BMI SE of 0.1 (and Appendix Table 2's "
              "HbA1c beta of -0.10) times 100 is 10",
        "24": "the population's 24-hour dietary recalls; the flow's 2,445 people with only one "
              "recall, divided by 100, round to 24",
        "five": "the exposure's five ratings, excellent to poor; Table 4's HbA1c beta of -0.05 "
                "times 100 is 5 in size",
        "two": "the population's two dietary recalls, the two groups educ4 merges and the two "
               "recall days kcal_mean averages; the paper's betas from 1.5 to 2.5 in size (Table "
               "4's glucose beta of -2.0 among them) round to 2, and the flow's 247 pregnant "
               "women divided by 100 round to 2",
        "1": "the derived columns' coding (smoke_now, glu_med and water_only_alt are 1 or 0); the "
             "paper's betas of 0.5 to 1.5 in size round to 1, the flow's 'Only one recall' step "
             "names 1, and the HbA1c SEs of 0.01 (Table 4 and Appendix Table 2) times 100 are 1",
        "one": "the prepared steps' 'one under the other', 'one Hispanic group' and 'one row per "
               "participant'; it meets the answers as 1 does",
        "0": "the same columns' coding (0); the paper's betas, SEs and P values up to 0.5 in size "
             "round to 0, as do its P values below 0.005 times 100 and every number up to 50 "
             "divided by 100",
        "zero": "pa_rec_min's step, where a kind of activity counts zero minutes; it meets the "
                "answers as 0 does"},
    "nhanes_glucose_predict": {
        "one": "the population's 'one row each' and 'every one with a fasting glucose'; the "
               "readings' day count of kcal is also 1"},
    "clinical_readmission_predict": {
        "1": "the outcome's coding (readmit_30d is 1 yes); the generating model's intercept, "
             "coefficients and centers of 0.5 to 1.5 in size, and its C-statistic, round to 1, "
             "and its center of 68 divided by 100 rounds to 1",
        "one": "the population's 'one row per encounter'; it meets the answers as 1 does",
        "0": "the outcome's coding (0 no); the generating model's coefficients up to 0.5 in size "
             "round to 0, and so does every number up to 50 divided by 100"},
    "inpatient_readmission_predict": {
        "1": "the outcome's coding (readmit_30d is 1 yes); the generating model's coefficients "
             "and centers of 0.5 to 1.5 in size, its C-statistic and its centers of 50 to 150 "
             "divided by 100 round to 1",
        "one": "the population's 'one row per encounter'; it meets the answers as 1 does",
        "0": "the outcome's coding (0 no); the generating model's coefficients up to 0.5 in size, "
             "the model classes' regrets and the oracle's 0 round to 0, and so does every number "
             "up to 50 divided by 100"},
    "walking_bp_estimate": {
        "1": "the exposure's and chronic pain's coding (1 yes); the model's numbers of 0.5 to 1.5 "
             "in size (a probit slope, a bias, b0, the tuning run's ratios and coverage) round to "
             "1, as do its numbers from 50 to 150 divided by 100",
        "one": "the population's 'one row per person'; it meets the answers as 1 does",
        "0": "the same columns' coding (0 no); the model's numbers up to 0.5 in size (the "
             "prevalence, standard errors) round to 0, and so does every number up to 50 divided "
             "by 100"},
    "plasma_panel_estimate": {
        "1": "the outcome's coding (disease is 1 yes); the run's parameters and fold changes of "
             "0.5 to 1.5 in size round to 1, as do its numbers from 50 to 150 divided by 100",
        "one": "the population's 'one row per injection'; it meets the answers as 1 does",
        "0": "the outcome's coding (0 no); the run's parameters up to 0.5 in size and the "
             "unassociated features' fold change of 0 round to 0, and so does every number up to "
             "50 divided by 100"},
}


def merge_on_seqn(base: Any, parts: dict[str, Any]) -> Any:
    """``base`` with every frame of ``parts`` joined to it by ``SEQN`` (a left join, one row per
    ``base`` row), refused where a frame repeats a SEQN, holds a SEQN ``base`` lacks (a left join
    would drop it), or names a column ``base`` already has."""
    out = base
    if not out["SEQN"].is_unique:
        raise SystemExit("the base table repeats a SEQN")
    for name, frame in parts.items():
        if not frame["SEQN"].is_unique:
            raise SystemExit(f"{name}: SEQN repeats")
        stray = int((~frame["SEQN"].isin(base["SEQN"])).sum())
        if stray:
            raise SystemExit(f"{name}: {stray} records whose SEQN the base table lacks")
        clash = sorted((set(frame.columns) & set(out.columns)) - {"SEQN"})
        if clash:
            raise SystemExit(f"{name}: columns already in the table: {clash}")
        out = out.merge(frame, on="SEQN", how="left", validate="one_to_one")
    if len(out) != len(base):
        raise SystemExit("the join changed the number of rows")
    return out


def write_table(frame: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    if path.suffix == ".parquet":
        frame.to_parquet(tmp, index=False)
    else:
        frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return table_entry(path, frame)


def copy_table(source: Path, dest: Path, data: bytes | None = None) -> None:
    """``source``'s bytes (or ``data``) at ``dest``, written whole or not at all."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(source.read_bytes() if data is None else data)
    tmp.replace(dest)


def spans(cycles: list[str]) -> str:
    """``1999–2004`` for the cycles 1999-2000 … 2003-2004."""
    return f"{cycles[0].split('-')[0]}–{cycles[-1].split('-')[1]}"


# ── Bang 2009: NHANES 1999–2004, the files its development analysis reads ─────

BANG_NOTES = "docs/turbotab-next/notes/replication/bang2009.md"
BANG_CYCLES = {"1999-2000": (1999, ""), "2001-2002": (2001, "_B"), "2003-2004": (2003, "_C")}
# Each file the development analysis reads (the notes' §3): its stem per cycle, the file's own
# name plus the cycle's suffix unless given, and the variables kept, each checked in every
# cycle's file (a variable no cycle carries stops the intake). The model's files (the notes' §8),
# the two Table 1 items the paper dropped although significant (income in DEMO, insurance in
# HIQ), smoking (SMQ), and the fasting hours (PH) that an analyst's fasting rule may read
# (the notes' §6, choice 1).
BANG_FILES: dict[str, dict[str, Any]] = {
    "DEMO": {"about": "demographics and the survey design",
             "vars": ["SEQN", "SDDSRVYR", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2",
                      "INDHHINC", "DMDMARTL", "RIDEXPRG", "SDMVSTRA", "SDMVPSU", "WTMEC2YR",
                      "WTMEC4YR"]},
    "LAB10AM": {"about": "fasting plasma glucose and the fasting subsample's weights",
                "stems": {"1999-2000": "LAB10AM", "2001-2002": "L10AM_B",
                          "2003-2004": "L10AM_C"},
                "vars": ["SEQN", "LBXGLU", "WTSAF2YR", "WTSAF4YR"]},
    "DIQ": {"about": "diabetes questions", "vars": ["SEQN", "DIQ010", "DIQ050", "DIQ070"]},
    "BPQ": {"about": "blood pressure questions", "vars": ["SEQN", "BPQ020", "BPQ040A", "BPQ050A"]},
    "BPX": {"about": "blood pressure readings",
            "vars": ["SEQN", *(f"BPXSY{i}" for i in range(1, 5)),
                     *(f"BPXDI{i}" for i in range(1, 5))]},
    "BMX": {"about": "body measures", "vars": ["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"]},
    "PAQ": {"about": "physical activity", "vars": ["SEQN", "PAQ520"]},
    "MCQ": {"about": "medical conditions and family history",
            "vars": ["SEQN", "MCQ250A", *(f"MCQ260A{c}" for c in "ABCDEFGHI"), "MCQ160C",
                     "MCQ160E", "MCQ160F"]},
    "HIQ": {"about": "health insurance", "vars": ["SEQN", "HID010"]},
    "SMQ": {"about": "smoking", "vars": ["SEQN", "SMQ020", "SMQ040"]},
    "PH": {"about": "fasting hours before the blood draw", "vars": ["SEQN", "PHAFSTHR", "PHAFSTMN"]},
}
BANG_MIN_AGE = 20  # the paper's inclusion: aged 20 or older (the notes' §2.1)
BANG_FPG = 126  # mg/dL: the paper's fasting glucose cut for diabetes (the notes' §2.2)
BANG_OUTCOME = "diabetes_fpg"
BANG_TABLE = TABLES / "bang2009" / "bang2009.parquet"
BANG_PAPER = {
    "citation": ("Bang H, Edwards AM, Bomback AS, Ballantyne CM, Brillon D, Callahan MA, "
                 "Teutsch SM, Mushlin AI, Kern LM. Development and validation of a patient "
                 "self-assessment score for diabetes risk. Ann Intern Med 2009;151(11):775-783."),
    "doi": "10.7326/0003-4819-151-11-200912010-00005",
    "pmid": "19949143",
}
# The paper's development numbers (NHANES 1999–2004), as the notes' §5 quotes them from the PMC
# author manuscript. Table 2: (factor, odds ratio, its 95% CI, P, points).
BANG_TABLE2 = [
    ("age 40-49, against under 40", 2.6, (1.3, 5.0), "0.004", 1),
    ("age 50-59", 4.8, (2.2, 10.6), "0.0001", 2),
    ("age 60 or older", 8.1, (3.9, 16.9), "<0.0001", 3),
    ("male", 2.6, (1.8, 3.7), "<0.0001", 1),
    ("family history of diabetes", 2.0, (1.5, 2.6), "<0.0001", 1),
    ("history of hypertension", 1.9, (1.2, 2.9), "0.004", 1),
    ("overweight, against not overweight or obese", 1.3, (0.6, 2.8), "0.47", 1),
    ("obese", 3.1, (1.6, 5.8), "0.0006", 2),
    ("extremely obese", 7.3, (4.0, 13.4), "<0.0001", 3),
    ("physically active", 0.7, (0.5, 1.0), "0.06", -1),
]
# Table 3's development rows: (score at or above, % high risk, sensitivity, specificity, PPV,
# NPV, LR+, LR-).
BANG_TABLE3 = [(6, 23, 61, 79, 11, 98, 2.90, 0.49), (5, 39, 82, 63, 8, 99, 2.22, 0.29),
               (4, 55, 93, 46, 7, 99, 1.72, 0.15), (3, 71, 98, 30, 5, 100, 1.40, 0.07)]


def bang_stems(name: str) -> dict[str, str]:
    spec = BANG_FILES[name]
    return spec.get("stems") or {c: name + sfx for c, (_, sfx) in BANG_CYCLES.items()}


def bang_spec() -> dict[str, Any]:
    """The download spec (``files.json``'s shape) of every file :data:`BANG_FILES` names."""
    return {"url": NHANES_URL, "cycles": {
        cycle: {"year": year, "files": [bang_stems(n)[cycle] for n in BANG_FILES]}
        for cycle, (year, _) in BANG_CYCLES.items()}}


def bang_stack(name: str, folder: Path = NHANES) -> tuple[Any, dict[str, list[str]]]:
    """One file's cycles stacked, its kept variables only, and the cycles lacking each."""
    import pandas as pd
    variables = BANG_FILES[name]["vars"]
    parts, absent = [], {}
    for cycle, stem in bang_stems(name).items():
        frame = read_xpt(Path(folder) / cycle / f"{stem}.xpt")
        for v in variables:
            if v not in frame.columns:
                absent.setdefault(v, []).append(cycle)
        parts.append(frame[[v for v in variables if v in frame.columns]])
    never = [v for v, cycles in absent.items() if len(cycles) == len(BANG_CYCLES)]
    if never or "SEQN" in absent:
        raise SystemExit(f"{name}: no cycle carries {never or ['SEQN']}")
    return pd.concat(parts, ignore_index=True).reindex(columns=variables), absent


def bang_table(folder: Path = NHANES) -> tuple[Any, dict[str, dict[str, list[str]]]]:
    """The development analysis's table: every file's cycles stacked, joined to DEMO by SEQN,
    and the outcome column the paper's definition names."""
    import numpy as np
    stacked, absent = {}, {}
    for name in BANG_FILES:
        stacked[name], gaps = bang_stack(name, folder)
        if gaps:
            absent[name] = gaps
    table = merge_on_seqn(stacked["DEMO"], {k: v for k, v in stacked.items() if k != "DEMO"})
    glucose = table["LBXGLU"]
    table[BANG_OUTCOME] = np.where(glucose.isna(), np.nan, (glucose >= BANG_FPG).astype(float))
    return table, absent


def bang_prepared(absent: dict[str, dict[str, list[str]]]) -> list[str]:
    cycles = list(BANG_CYCLES)
    kept = "; ".join(f"{'/'.join(bang_stems(n).values())} ({BANG_FILES[n]['about']}): "
                     + ", ".join(v for v in BANG_FILES[n]["vars"] if v != "SEQN")
                     for n in BANG_FILES)
    out = [
        f"Read each cycle's NHANES files, {', '.join(cycles)}, with TurboTab's own SAS "
        "transport reader, which reads SAS missing values as blank.",
        "Stacked each file's cycles one under the other; a row's cycle is NHANES's own SDDSRVYR.",
        f"Kept only these variables: {kept}.",
    ]
    gaps = [f"{v} in {' and '.join(c)} ({'/'.join(bang_stems(n)[x] for x in c)} does not carry "
            f"it)" for n, g in absent.items() for v, c in g.items()]
    if gaps:
        out.append("Blank where a cycle's file does not carry the variable: " + "; ".join(gaps)
                   + ".")
    out += [
        "Joined every file to the demographics file (DEMO) by the participant's SEQN: one row "
        "per participant of DEMO, blank in a file's variables where the participant has no record "
        "in it (every record of every file has its participant in DEMO).",
        f"Added {BANG_OUTCOME}: 1 where fasting plasma glucose (LBXGLU) is {BANG_FPG} mg/dL or "
        "more, 0 where it is below, blank where LBXGLU is blank. It names the paper's outcome but "
        "is not yet its undiagnosed diabetes, which also needs the participant to be without "
        "known diabetes: an exclusion the table does not apply.",
        "Nothing else: no row is dropped, and no value is recoded or imputed.",
    ]
    return out


def bang_card(prepared: list[str]) -> dict[str, Any]:
    span = spans(list(BANG_CYCLES))
    return {
        "id": "bang2009", "design": "replication", "purpose": "Predict",
        "question": ("How well can undiagnosed diabetes in U.S. adults be predicted from what a "
                     "person can tell about themselves, without a blood test?"),
        "population": (f"U.S. adults aged {BANG_MIN_AGE} or older in NHANES {span} who have a "
                       "fasting plasma glucose result, pregnant women excluded. People with known "
                       "diabetes are not in the model: known diabetes is a yes to DIQ010 (a "
                       "doctor said you have diabetes), taking insulin now (DIQ050) or taking "
                       "diabetic pills (DIQ070)."),
        "outcome": {"column": BANG_OUTCOME, "meaning": (
            f"fasting plasma glucose of {BANG_FPG} mg/dL or more (1) or below (0), blank without "
            f"a glucose result; the paper's undiagnosed diabetes is {BANG_OUTCOME} = 1 in a "
            "participant without known diabetes")},
        "replication": {"paper": dict(BANG_PAPER), "stated_question": (
            "The paper's title: Development and validation of a patient self-assessment score "
            f"for diabetes risk. It develops, in NHANES {span}, a score a person can work out "
            "about themselves, with undiagnosed diabetes as its endpoint.")},
        "table": {"path": rel(BANG_TABLE), "format": "parquet"},
        "source": "NHANES public-use files from the CDC (wwwn.cdc.gov).",
        "prepared": prepared,
        "manifest": "research/datasets/bang2009/manifest.json",
    }


def bang_answers() -> dict[str, Any]:
    source = f"{BANG_NOTES} §5 (the paper's PMC author manuscript, PMC3633111)"
    return {
        "id": "bang2009", "kind": "replication",
        "about": ("The development analysis's numbers (NHANES 1999–2004), as the replication "
                  "notes quote the paper. The validation's (NHANES 2005–2006, ARIC and CHS) are "
                  "left out: this table holds neither."),
        "notes": BANG_NOTES,
        "truths": [
            {"name": "the development sample, unweighted (Table 1)", "source": source,
             "value": {"normal fasting glucose": 3396, "impaired fasting glucose": 1652,
                       "undiagnosed diabetes": 210, "in the model": 5258,
                       "known diabetes, not in the model": 482}},
            {"name": "the final model (Table 2): odds ratio, its 95% CI, P and points",
             "source": source,
             "value": [{"factor": f, "odds_ratio": o, "ci95": list(ci), "p": p, "points": pt}
                       for f, o, ci, p, pt in BANG_TABLE2]},
            {"name": "the final model's AUC in development", "source": source, "value": 0.79},
            {"name": "the score: the sum of the points; high risk at 5 or more", "source": source,
             "value": {"range": [-1, 9], "high_risk_at_or_above": 5}},
            {"name": "the score's cutpoints in development (Table 3)", "source": source,
             "value": [dict(zip(("score_at_or_above", "pct_high_risk", "sensitivity",
                                 "specificity", "ppv", "npv", "lr_positive", "lr_negative"), row))
                       for row in BANG_TABLE3]},
        ],
        "compare": {"weighting": ("Table 3 and the counts look unweighted (the notes' §5.3): "
                                  "compare a run's unweighted sample metrics first.")},
    }


def intake_bang2009(old: dict | None) -> tuple[dict, dict, dict]:
    files = download(bang_spec(), old)
    table, absent = bang_table()
    entry = write_table(table, BANG_TABLE)
    manifest = {"about": ("The NHANES files Bang et al. 2009's development analysis reads "
                          "(NHANES 1999–2004), as downloaded, and the table built from them."),
                "dataset": "bang2009", **NHANES_LICENSE, "files": files, "table": entry}
    return manifest, bang_card(bang_prepared(absent)), bang_answers()


# ── Sullivan 2021: the replication's table, as its prep builds it ────────────

SULLIVAN = REPO / "research" / "replication" / "sullivan2021" / "engine"
SULLIVAN_NOTES = "docs/turbotab-next/notes/replication/sullivan2021.md"
SULLIVAN_TABLE = TABLES / "sullivan2021" / "sullivan2021.parquet"
SULLIVAN_OUTCOME = "glucose"  # the card's: the first of the paper's Table 4 outcomes


def sullivan_plan() -> dict[str, Any]:
    return json.loads((SULLIVAN / "sullivan2021.json").read_text())


def sullivan_prep() -> Any:
    """The replication's ``prep.py``, as a module (its code is run, not copied)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sullivan2021_prep", SULLIVAN / "prep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sullivan_table(prep: Any, plan: dict[str, Any]) -> Any:
    """The plan file's table: its main file with every join, by SEQN, as the replication joins
    them in the app (``table``: main, joins, on, how)."""
    import pandas as pd
    t = plan["table"]
    if (t["on"], t["how"]) != ("SEQN", "left"):
        raise SystemExit(f"the plan joins on {t['on']} ({t['how']}): merge_on_seqn does not")
    folder = REPO / t["join_folder"]
    return merge_on_seqn(pd.read_parquet(REPO / t["main"]),
                         {j: pd.read_parquet(folder / f"{j}.parquet") for j in t["joins"]})


def sullivan_prepared(prep: Any, plan: dict[str, Any]) -> list[str]:
    cycles = list(prep.CYCLES)
    letter = {s: c for c, s in prep.CYCLES.items()}
    ins = prep.COMPONENTS["INS"]["source"]
    own = [s for s in prep.CYCLES.values() if s not in ins]
    kept = "; ".join(f"{name}: " + ", ".join(v for v in c["vars"] if v != "SEQN")
                     for name, c in prep.COMPONENTS.items())
    slope, intercept = prep.XW_SLOPE, prep.XW_INTERCEPT
    sign = "−" if intercept < 0 else "+"
    joins = [j for j in plan["table"]["joins"] if j != "derived"]
    if len(joins) + 1 != len(plan["table"]["joins"]):
        raise SystemExit("the plan file's joins no longer end in the prep's derived columns")
    return [
        "The Sullivan replication's prep (research/replication/sullivan2021/engine/prep.py), run "
        f"as it is: it read the NHANES files of the cycles {', '.join(cycles)} with TurboTab's own "
        "SAS transport reader, which reads SAS missing values as blank, and stacked each file's "
        "cycles one under the other; a row's cycle is NHANES's own SDDSRVYR.",
        f"Insulin (LBXIN) is read from {', '.join(f'{v} for {letter[k]}' for k, v in ins.items())}, "
        f"which carries it, and from {', '.join(f'INS_{s}' for s in own)} for the later cycles. "
        "The fasting weight WTSAF2YR comes from the glucose files; the prep checks that the "
        "insulin files' weight equals it row for row.",
        f"Kept only these variables: {kept}.",
        "Derived race5: RIDRETH3 with Mexican American and Other Hispanic merged into one "
        "Hispanic group, the other groups as they are.",
        "Derived educ4: DMDEDUC2 with its two groups below high school merged, and high school, "
        "some college and college graduate each a group of its own; refused and don't know blank.",
        "Derived smoke_now, current smoking: 1 where SMQ020 is yes and SMQ040 is every day or "
        "some days; 0 where SMQ020 is no or SMQ040 is not at all; blank otherwise.",
        "Derived pa_rec_min, weekly recreational activity minutes: moderate days times minutes "
        "(PAQ670 × PAD675) plus twice vigorous days times minutes (PAQ655 × PAD660); a kind "
        "counts zero minutes where its yes/no item (PAQ665, PAQ650) is no, and the sum is blank "
        "where an item it reads is refused, not known or missing.",
        "Derived kcal_mean: the mean of the two recall days' energy, DR1TKCAL and DR2TKCAL.",
        "Derived glu_med, glucose-lowering medication: 1 where taking insulin now (DIQ050) or "
        "diabetic pills (DIQ070) is yes; 0 where DIQ010 is answered yes, no or borderline and "
        "neither item is yes, refused or not known; blank otherwise.",
        "Derived water_only_alt: 1 where a recall day has no energy (DR1TKCAL or DR2TKCAL) and "
        "plain water reported that day (DR1_320Z or DR2_320Z), else 0.",
        f"Derived LBXIN_xw: insulin with {letter['G']}'s Roche values put on the later cycles' "
        "Tosoh scale by the INS_H codebook's forward Deming equation, "
        f"10^({slope} × log10(x) {sign} {abs(intercept)}); every other value as measured.",
        f"Joined {', '.join(joins)} and the derived columns to DEMO by SEQN: one row per "
        "participant of DEMO (a left join, as the replication's plan file joins them in the app).",
        "Nothing else: no row is dropped, no value imputed, and DBQ700 keeps NHANES's codes.",
    ]


def sullivan_card(prep: Any, plan: dict[str, Any], prepared: list[str]) -> dict[str, Any]:
    age = next(e["rule"]["low"] for e in plan["exclusions"] if e["key"] == "age")
    outcome = plan["outcomes"][SULLIVAN_OUTCOME]
    span = spans(list(prep.CYCLES))
    return {
        "id": "sullivan2021", "design": "replication", "purpose": "Estimate",
        "question": ("Is the rating U.S. adults give their own diet associated with their "
                     "fasting glucose?"),
        "population": (f"U.S. adults aged {age} or older in NHANES {span} who rated their diet and "
                       "have two reliable 24-hour dietary recalls; pregnant women, and anyone "
                       "reporting only water on a recall day, excluded. Fasting glucose is "
                       "measured in the morning fasting subsample."),
        "exposure": {"column": plan["exposure"]["column"], "meaning": (
            "the participant's rating of their own diet, from excellent (the lowest code) to poor "
            "(the highest of the five ratings); refused and don't know have codes of their own")},
        "outcome": {"column": outcome["column"],
                    "meaning": f"fasting plasma glucose ({outcome['unit']})"},
        "replication": {"paper": {"citation": plan["paper"]["citation"]}, "stated_question": (
            "The paper's title: Self-Rated Diet Quality and Cardiometabolic Health Among U.S. "
            f"Adults, {span}. It asks whether the rating people give their own diet is "
            "associated with fasting glucose, fasting insulin, HbA1c and BMI; this card takes "
            "fasting glucose.")},
        "table": {"path": rel(SULLIVAN_TABLE), "format": "parquet"},
        "source": "NHANES public-use files from the CDC (wwwn.cdc.gov).",
        "prepared": prepared,
        "manifest": "research/datasets/sullivan2021/manifest.json",
    }


def sullivan_answers(plan: dict[str, Any]) -> dict[str, Any]:
    paper = plan["paper"]
    source = "research/replication/sullivan2021/engine/sullivan2021.json (paper)"
    return {
        "id": "sullivan2021", "kind": "replication",
        "about": ("The paper's printed numbers, as the replication's plan file holds them: the "
                  "participant flow, Table 4's four associations and Appendix Table 2's models."),
        "notes": SULLIVAN_NOTES,
        "truths": [
            {"name": "the participant flow: each step and the rows it removes", "source": source,
             "value": paper["flow"]},
            {"name": "Table 4: beta, SE, n and P for each outcome", "source": source,
             "value": paper["table4"]},
            {"name": "Appendix Table 2: beta and SE in Models 0 to 4", "source": source,
             "value": paper["ladder"]},
        ],
        "compare": {"precision": paper["precision"], "direction": paper["direction"]},
    }


def intake_sullivan2021(old: dict | None) -> tuple[dict, dict, dict]:
    files = download(json.loads((SULLIVAN / "files.json").read_text()), old)
    prep, plan = sullivan_prep(), sullivan_plan()
    if prep.main() != 0:
        raise SystemExit("the Sullivan prep failed")
    entry = write_table(sullivan_table(prep, plan), SULLIVAN_TABLE)
    manifest = {"about": ("The NHANES files of the Sullivan 2021 replication (its files.json), as "
                          "downloaded, and the table its prep and plan file build from them."),
                "dataset": "sullivan2021", **NHANES_LICENSE, "files": files, "table": entry}
    return manifest, sullivan_card(prep, plan, sullivan_prepared(prep, plan)), \
        sullivan_answers(plan)


# ── the fixtures: the NHANES export, and two with declared truths ────────────

NHANES_GZ = REPO / "turbotab" / "core" / "tests" / "fixtures" / "nhanes.csv.gz"
NHANES_KEY = "_tt_tmp_nhanes.csv"  # the export's name, which truths.FIXTURE_TRUTHS keys on
SAMPLE = REPO / "turbotab" / "sample_data"
# clinical_risk.csv's outcome, as make_fixtures.py::clinical_risk draws it: readmit_30d is 1 with
# probability expit(intercept + Σ coef × (f(column) − center)), f the identity or the log.
CLINICAL_RISK_MODEL = {
    "link": "logit", "intercept": -1.35,
    "terms": [
        {"column": "age", "coef": 0.030, "center": 68, "transform": "identity"},
        {"column": "charlson_index", "coef": 0.26, "center": 2.2, "transform": "identity"},
        {"column": "prior_admissions_12mo", "coef": 0.50, "center": 0.9, "transform": "identity"},
        {"column": "albumin_g_dl", "coef": -0.95, "center": 3.6, "transform": "identity"},
        {"column": "creatinine_mg_dl", "coef": 0.45, "center": 1.05, "transform": "log_ratio"},
        {"column": "hemoglobin_g_dl", "coef": -0.16, "center": 11.9, "transform": "identity"},
        {"column": "length_of_stay_days", "coef": 0.075, "center": 4.6, "transform": "identity"},
    ],
    "no_signal": ["encounter_id", "sex", "sodium_mmol_l"],
}


def linear_predictor(model: dict[str, Any], frame: Any) -> Any:
    """The model's log-odds on each row of ``frame`` (``log_ratio``: log(x / center))."""
    import numpy as np
    z = np.full(len(frame), float(model["intercept"]))
    for t in model["terms"]:
        x = frame[t["column"]].to_numpy(dtype=float)
        z = z + t["coef"] * (np.log(x / t["center"]) if t["transform"] == "log_ratio"
                             else x - t["center"])
    return z


def generated_card(id_: str, purpose: str, question: str, population: str,
                   outcome: dict[str, str], table: Path, **more: Any) -> dict[str, Any]:
    return {"id": id_, "design": "analysts", "purpose": purpose, "question": question,
            "population": population, "outcome": outcome, **more,
            "table": {"path": rel(table), "format": table.suffix.lstrip(".")},
            "source": "A generated (synthetic) table.", "prepared": [],
            "manifest": f"research/datasets/{id_}/manifest.json"}


def fixture_manifest(id_: str, source: Path, generator: str, table: Path, frame: Any,
                     about: str) -> dict[str, Any]:
    return {"about": about, "dataset": id_, **REPO_LICENSE,
            "files": [{"file": rel(source), "origin": "this repository", "generator": generator,
                       "bytes": source.stat().st_size, "sha256": sha256(source)}],
            "table": table_entry(table, frame)}


def readings(key: str) -> dict[str, str]:
    from turbotab.core.tests.truths import FIXTURE_TRUTHS
    return dict(FIXTURE_TRUTHS[key])


def nhanes_age_top_code(begin_year: int) -> int:
    """The age NHANES's public files record for every older participant in the cycle beginning
    ``begin_year`` (RIDAGEYR): "Individuals 85 and over are topcoded at 85 years of age" through
    DEMO_D (2005–2006), and "80 and over ... at 80" from DEMO_E (2007–2008), as each cycle's
    demographics codebook page states."""
    return 85 if begin_year <= 2005 else 80


def cycle_span(year: int) -> str:
    return f"{year}–{year + 1}"


def row_count(n: int) -> str:
    return f"{n:,} row" + ("" if n == 1 else "s")


def listed(words: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def nhanes_export(data: bytes, pinned: str) -> Any:
    """The export's frame from its decompressed bytes, refused unless they hash to ``pinned``
    (stage_harness's NHANES_SHA256) and hold one row per participant, each with a glucose."""
    import io

    import pandas as pd
    if hashlib.sha256(data).hexdigest() != pinned:
        raise SystemExit("the decompressed export is not the bytes stage_harness pins")
    frame = pd.read_csv(io.BytesIO(data))
    if not (frame["SEQN"].is_unique and frame["glucose"].notna().all()):
        raise SystemExit("the export is not one row per participant, each with a glucose")
    return frame


def nhanes_population(frame: Any) -> str:
    """Who the export's rows are, its ages read from the frame: refused where a cycle's oldest age
    is not the top code its codebook states (:func:`nhanes_age_top_code`), since the sentence
    reads the oldest age as that code."""
    years = sorted(int(y) for y in frame["cycle_begin_year"].unique())
    oldest = frame.groupby("cycle_begin_year")["age"].max()
    runs: list[tuple[int, list[int]]] = []
    for y in years:
        top = nhanes_age_top_code(y)
        if oldest[y] != top:
            raise SystemExit(f"the export's oldest age in {cycle_span(y)} is {oldest[y]:g}, not "
                             f"the top code {top} its demographics codebook states")
        if runs and runs[-1][0] == top:
            runs[-1][1].append(y)
        else:
            runs.append((top, [y]))
    coded = listed([f"{top} stands for {top} or older in "
                    + (cycle_span(ys[0]) if len(ys) == 1 else
                       f"the cycles {cycle_span(ys[0])} to {cycle_span(ys[-1])}")
                    for top, ys in runs])
    return (f"NHANES participants aged {frame['age'].min():g} or older, one row each, every one "
            f"with a fasting glucose, from the {len(years)} survey cycles "
            f"{cycle_span(years[0])} to {cycle_span(years[-1])}. NHANES top-codes age: {coded}.")


def nhanes_prepared(frame: Any) -> list[str]:
    """What the export shows of the steps taken before it reached this repository: the values it
    flags as imputed (its ``imputed_<measure>`` columns), and the rows it holds."""
    flags = [c for c in frame.columns if c.startswith("imputed_")]
    measures = [c.removeprefix("imputed_") for c in flags]
    if any(m not in frame.columns for m in measures):
        raise SystemExit(f"an imputed_ column flags no column of the export: {flags}")
    out = []
    if flags:
        counts = listed([f"{m} on the {row_count(int(frame[c].sum()))} where {c} is true"
                         for m, c in zip(measures, flags)])
        blank = {m: int(frame[m].isna().sum()) for m in measures}
        left = ("none of these measures has a blank" if not any(blank.values()) else
                "blanks remain in "
                + listed([f"{m} ({row_count(n)})" for m, n in blank.items() if n]))
        out.append(f"Before the export reached this repository, by a method it does not record, "
                   f"values were imputed in {len(measures)} measures, each flagged in its own "
                   f"column: {counts}; {left}.")
    blanks = frame.isna().sum()
    blanks = blanks[blanks > 0]
    left = ("No column has a blank" if blanks.empty else "No column has a blank but "
            + listed([f"{c} ({row_count(int(n))})" for c, n in blanks.items()]))
    out.append(f"Before the export reached this repository, by rules it does not record, its rows "
               f"were limited: every row has a glucose value and an age of "
               f"{frame['age'].min():g} or more. {left}.")
    return out


def intake_nhanes_glucose_predict(old: dict | None) -> tuple[dict, dict, dict]:
    import gzip

    from turbotab.core.tests.stage_harness import NHANES_SHA256
    id_ = "nhanes_glucose_predict"
    table = TABLES / id_ / NHANES_KEY
    data = gzip.decompress(NHANES_GZ.read_bytes())
    frame = nhanes_export(data, NHANES_SHA256)
    card = {
        "id": id_, "design": "analysts", "purpose": "Predict",
        "question": "How well do the table's measures predict fasting glucose?",
        "population": nhanes_population(frame),
        "outcome": {"column": "glucose", "meaning": "fasting plasma glucose (mg/dL)"},
        "table": {"path": rel(table), "format": "csv"},
        "source": ("An NHANES export tracked in this repository "
                   "(turbotab/core/tests/fixtures/nhanes.csv.gz), the table of the app's NHANES "
                   "glucose journeys. How it was exported from the NHANES files is not recorded "
                   "with it."),
        "prepared": nhanes_prepared(frame),
        "manifest": f"research/datasets/{id_}/manifest.json",
    }
    copy_table(NHANES_GZ, table, data)
    manifest = {"about": "The NHANES export, decompressed from the repository's tracked copy.",
                "dataset": id_, **NHANES_LICENSE,
                "files": [{"file": rel(NHANES_GZ), "origin": "this repository",
                           "bytes": NHANES_GZ.stat().st_size, "sha256": sha256(NHANES_GZ),
                           "decompressed_sha256": NHANES_SHA256}],
                "table": table_entry(table, frame)}
    answers = {
        "id": id_, "kind": "no_known_answer",
        "about": ("Real data: nobody knows how well these measures can predict fasting glucose, "
                  "so a run is checked by the protocol checks alone. Its readings are declared."),
        "fixture_key": NHANES_KEY,
        "truths": [{"name": "the readings a user who knows the export gives",
                    "source": "turbotab/core/tests/truths.py FIXTURE_TRUTHS",
                    "value": readings(NHANES_KEY)}],
    }
    return manifest, card, answers


def intake_clinical_readmission_predict(old: dict | None) -> tuple[dict, dict, dict]:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    id_, key = "clinical_readmission_predict", "clinical_risk.csv"
    source, table = SAMPLE / key, TABLES / id_ / key
    copy_table(source, table)
    frame = pd.read_csv(table)
    risk = 1.0 / (1.0 + np.exp(-linear_predictor(CLINICAL_RISK_MODEL, frame)))
    card = generated_card(
        id_, "Predict", "How well do the table's measures predict a readmission within 30 days?",
        "A generated (synthetic) cohort of adult hospital encounters, one row per encounter.",
        {"column": "readmit_30d", "meaning": "readmitted within 30 days (1 yes, 0 no)"}, table)
    generator = "turbotab/sample_data/make_fixtures.py::clinical_risk"
    answers = {
        "id": id_, "kind": "known_answer", "fixture_key": key,
        "about": ("A generated cohort whose outcome is drawn from a logistic model of its columns "
                  "(make_fixtures.py::clinical_risk): the true risk of every row is known."),
        "truths": [
            {"name": "the outcome's generating model", "source": generator,
             "value": CLINICAL_RISK_MODEL},
            {"name": "the C-statistic of ranking every row by its true risk",
             "source": "the generating model on the table's rows, by scikit-learn's roc_auc_score",
             "value": float(roc_auc_score(frame["readmit_30d"], risk))},
            {"name": "leakage", "source": "turbotab/sample_data/clinical_risk.md (Must NOT surface)",
             "value": "none: no column is measured after the outcome or stands in for it"},
            {"name": "the readings a user who knows the table gives",
             "source": "turbotab/core/tests/truths.py FIXTURE_TRUTHS", "value": readings(key)},
        ],
    }
    manifest = fixture_manifest(id_, source, generator, table, frame,
                                "A sample fixture of this repository, copied as tracked.")
    return manifest, card, answers


# ── the generated fixtures: the hardened Predict, the Estimate and the measurement fixture ──────
# (SWARM_PLAN §5 items 4–6; ``research/swarm/generators.py``). Each table is the committed file its
# generator writes at its seed (``generators.TABLES``); the manifest names the generator and the
# seed, so a fresh draw (``core/export/replay.py``) is the same generator at another seed.

GENERATORS_PY = "research/swarm/generators.py"
SWARM_FIXTURES = REPO / "research" / "swarm" / "fixtures"


def generated_manifest(id_: str, key: str, table: Path, frame: Any, about: str) -> dict[str, Any]:
    from research.swarm import generators as G

    entry = G.TABLES[key]
    source = SWARM_FIXTURES / key
    manifest = fixture_manifest(id_, source, f"{GENERATORS_PY}::{entry['function']}", table, frame,
                                about)
    manifest["generator"] = {"module": GENERATORS_PY, "function": entry["function"],
                             "seed": entry["seed"], "rows": entry["rows"]}
    return manifest


def _copied(key: str, id_: str) -> tuple[Path, Any]:
    """The committed table copied to the dataset's data folder, refused unless it is its generator's
    bytes; its path and frame."""
    import pandas as pd

    from research.swarm import generators as G
    source, table = SWARM_FIXTURES / key, TABLES / id_ / key
    if source.read_bytes() != G.table_bytes(key):
        raise SystemExit(f"{rel(source)} is not its generator's bytes: run "
                         f"python -m research.swarm.generators")
    copy_table(source, table)
    return table, pd.read_csv(table)


def intake_inpatient_readmission_predict(old: dict | None) -> tuple[dict, dict, dict]:
    from sklearn.metrics import roc_auc_score

    from research.swarm import generators as G
    id_, key = "inpatient_readmission_predict", G.READMISSION_FILE
    table, frame = _copied(key, id_)
    card = generated_card(
        id_, "Predict", "How well do the table's measures predict a readmission within 30 days?",
        "A generated (synthetic) cohort of adult hospital encounters, one row per encounter.",
        {"column": G.READMISSION_OUTCOME, "meaning": "readmitted within 30 days (1 yes, 0 no)"},
        table)
    source = f"{GENERATORS_PY}::readmission"
    regrets = G.readmission_regrets()
    answers = {
        "id": id_, "kind": "known_answer", "fixture_key": key,
        "about": ("A generated cohort whose outcome is drawn from a logistic model of its columns "
                  "with a U-shaped sodium term and an age-by-albumin interaction "
                  f"({source}): the true risk of every row is known."),
        "truths": [
            {"name": "the outcome's generating model", "source": source,
             "value": G.READMISSION_MODEL},
            {"name": "the C-statistic of ranking every row by its true risk",
             "source": "the generating model on the table's rows, by scikit-learn's roc_auc_score",
             "value": float(roc_auc_score(frame[G.READMISSION_OUTCOME],
                                          G.readmission_risk(frame)))},
            {"name": "each model class's Brier regret against the true risk, E[(p̂ − p)²] of its "
                     "population-best fit (the oracle's is 0): linear is every column as a "
                     "straight line, spline the continuous columns as cubic B-splines, "
                     "interaction those splines and age × albumin",
             "source": (f"{GENERATORS_PY}::readmission_regrets: an unpenalized logistic fit on "
                        f"{G.REGRET_ROWS[0]:,} rows at seed {G.REGRET_SEEDS[0]}, read on "
                        f"{G.REGRET_ROWS[1]:,} rows at seed {G.REGRET_SEEDS[1]}"),
             "value": {"linear": regrets["linear"], "spline": regrets["spline"],
                       "interaction": regrets["interaction"], "oracle": 0.0}},
            {"name": "leakage", "source": source,
             "value": "none: no column is measured after the outcome or stands in for it"},
            {"name": "the readings a user who knows the table gives",
             "source": "turbotab/core/tests/truths.py FIXTURE_TRUTHS", "value": readings(key)},
        ],
    }
    manifest = generated_manifest(id_, key, table, frame,
                                  "A generated fixture of this repository, copied as committed.")
    return manifest, card, answers


def intake_walking_bp_estimate(old: dict | None) -> tuple[dict, dict, dict]:
    from research.swarm import generators as G
    id_, key = "walking_bp_estimate", G.ESTIMATE_FILE
    table, frame = _copied(key, id_)
    m = G.ESTIMATE_MODEL
    card = generated_card(
        id_, "Estimate", "Is walking every day associated with systolic blood pressure?",
        "A generated (synthetic) cohort of adults, one row per person.",
        {"column": m["outcome"], "meaning": "systolic blood pressure (mmHg)"}, table,
        exposure={"column": m["exposure"], "meaning": "walks every day (1 yes, 0 no)"},
        documentation=[
            "Each person's age, height, chronic pain (1 yes, 0 no) and daily walking were recorded "
            "when they enrolled; their health-concern score (a questionnaire's score) and their "
            "systolic blood pressure were measured at a later visit.",
            "A blank chronic_pain is a question the person did not answer.",
        ])
    source = f"{GENERATORS_PY}::blood_pressure"
    planted = G.planted_biases()
    tuning = G.b0_tuning()
    answers = {
        "id": id_, "kind": "known_answer", "fixture_key": key,
        "about": (f"A generated cohort ({source}) in which walking every day changes systolic "
                  "blood pressure by a known difference in means, chronic pain causes both, the "
                  "health-concern score is caused by both, and chronic pain is left blank more "
                  "often where blood pressure is high: each plan's population bias is known."),
        "truths": [
            {"name": "the generating model", "source": source, "value": m},
            {"name": "the effect of walks_daily on sbp_mmhg, a difference in means", "source": source,
             "value": planted["effect"]},
            {"name": "each plan's population coefficient on walks_daily and its bias, and b0 (half "
                     "the smallest planted bias in size)",
             "source": f"{GENERATORS_PY}::planted_biases (closed form, population_effects)",
             "value": {"plans": G.ESTIMATE_PLANS, **planted}},
            {"name": "b0 against the oracle plan's mean bias over fresh draws (SWARM_PLAN §5 item "
                     "5), from an analysis independent of the engine",
             "source": f"{GENERATORS_PY}::b0_tuning (statsmodels; multiple imputation of its own)",
             "value": tuning},
            {"name": "the readings a user who knows the table gives",
             "source": "turbotab/core/tests/truths.py FIXTURE_TRUTHS", "value": readings(key)},
        ],
    }
    manifest = generated_manifest(id_, key, table, frame,
                                  "A generated fixture of this repository, copied as committed.")
    return manifest, card, answers


def intake_plasma_panel_estimate(old: dict | None) -> tuple[dict, dict, dict]:
    from research.swarm import generators as G
    id_, key = "plasma_panel_estimate", G.PANEL_FILE
    table, frame = _copied(key, id_)
    features = [c for c in frame.columns if c.startswith("mz_")]
    card = generated_card(
        id_, "Estimate",
        "Which plasma features differ between people with and without the disease?",
        "A generated (synthetic) untargeted metabolomics run: one row per injection, each a "
        "participant's plasma sample or a pooled QC sample, with the participants' age, sex and "
        "BMI.",
        {"column": G.PANEL_OUTCOME,
         "meaning": "has the disease (1 yes, 0 no); blank on a pooled QC injection"}, table,
        # The card names one exposure column (validate_card takes one): the family's first, its
        # meaning the whole family.
        exposure={"column": features[0],
                  "meaning": f"every feature column, {features[0]} to {features[-1]}, is an "
                             f"exposure, each tested on its own; the card names {features[0]} "
                             f"because it takes one column"},
        documentation=[
            "The samples were injected in run_order, in the batches batch names; pooled QC "
            "injections (sample_type pooled_qc) open and close each batch and recur within it.",
            "A blank intensity is a value below the instrument's detection limit.",
        ])
    source = f"{GENERATORS_PY}::metabolite_panel"
    answers = {
        "id": id_, "kind": "known_answer", "fixture_key": key,
        "about": (f"A generated run ({source}) with a known set of features associated with the "
                  "disease and their log2 fold changes, under detection-limit censoring, drift "
                  "within each batch, batch offsets that go up and down in run order, sample "
                  "dilution and pooled QC injections."),
        "truths": [
            {"name": "the associated features and each one's log2 fold change in the true "
                     "concentration (every other feature's is 0)",
             "source": f"{GENERATORS_PY}::panel_truth", "value": G.panel_truth()},
            {"name": "the run's parameters", "source": f"{GENERATORS_PY}::PANEL", "value": G.PANEL},
            {"name": "the measurement decisions graded: the engine's own menus",
             "source": f"{GENERATORS_PY}::graded_measurement_decisions",
             "value": G.graded_measurement_decisions()},
            {"name": "the readings a user who knows the table gives",
             "source": "turbotab/core/tests/truths.py FIXTURE_TRUTHS", "value": readings(key)},
        ],
    }
    manifest = generated_manifest(id_, key, table, frame,
                                  "A generated fixture of this repository, copied as committed.")
    return manifest, card, answers


SURVEY_RESPONSES = (1, 5)  # each item's response levels, 1 to 5 (survey_instrument.md's own table)


def survey_key(reverse_coded: tuple[str, ...] | list[str], frame: Any, low: int,
               high: int) -> list[str]:
    """The instrument's key, as its codebook gives it (``reverse_coded``, make_fixtures.py's
    REVERSE_CODED), and each item's response range (``low`` to ``high``). Refused where a
    reverse-coded item is not one of the instrument's items, or an answer lies outside the range.
    It names no scale: no driver command declares one (``set_scales`` is posted by no Router
    question, ``catalog.NOT_ASKED``), so a card that asked for a scale would ask what an analyst
    cannot do."""
    items = [c for c in frame.columns if c.startswith("item_")]
    if not items or not set(reverse_coded) <= set(items):
        raise SystemExit("the key's reverse-coded items are not the instrument's")
    answered = frame[items].stack()
    if not answered.between(low, high).all():
        raise SystemExit(f"an item is answered outside {low} to {high}")
    return [
        f"The instrument's key: {items[0]} to {items[-1]} are one {len(items)}-item instrument, "
        f"each item answered from {low} to {high}, and its reverse-coded items "
        f"are {listed(list(reverse_coded))}.",
    ]


def intake_survey_support_estimate(old: dict | None) -> tuple[dict, dict, dict]:
    import pandas as pd
    from turbotab.sample_data.make_fixtures import REVERSE_CODED
    id_, key = "survey_support_estimate", "survey_instrument.csv"
    source, table = SAMPLE / key, TABLES / id_ / key
    copy_table(source, table)
    frame = pd.read_csv(table)
    items = [c for c in frame.columns if c.startswith("item_")]
    card = generated_card(
        id_, "Estimate",
        "Is age associated with having sought support?",
        f"A generated (synthetic) survey of adult respondents, one row per respondent: their age, "
        f"sex and education, their answers to a {len(items)}-item instrument, and whether they "
        "sought support.",
        {"column": "sought_support", "meaning": "sought support (1 yes, 0 no)"}, table,
        exposure={"column": "age", "meaning": "age in years"},
        documentation=survey_key(REVERSE_CODED, frame, *SURVEY_RESPONSES))
    generator = "turbotab/sample_data/make_fixtures.py::survey"
    answers = {
        "id": id_, "kind": "known_answer", "fixture_key": key,
        "about": ("A generated survey in which age is drawn on its own and sought_support from a "
                  "latent trait and noise alone (make_fixtures.py::survey): age has no effect."),
        "truths": [
            {"name": "the effect of age on sought_support", "source": generator,
             "value": ("none: age is independent of sought_support, and of it given any of the "
                       "table's other columns, so its effect is null on every scale")},
            {"name": "the readings a user who knows the table gives",
             "source": "turbotab/core/tests/truths.py FIXTURE_TRUTHS", "value": readings(key)},
        ],
    }
    manifest = fixture_manifest(id_, source, generator, table, frame,
                                "A sample fixture of this repository, copied as tracked.")
    return manifest, card, answers


# ── the null decoy ───────────────────────────────────────────────────────────

DECOY_ID = "outpatient_admission_predict"  # a name that does not say what the table is
DECOY_SEED = 20261018
DECOY_TABLE = TABLES / DECOY_ID / "outpatient_visits.csv"


def intake_outpatient_admission_predict(old: dict | None) -> tuple[dict, dict, dict]:
    from research.swarm import decoys
    frame = decoys.generate(DECOY_SEED, decoys.ROWS)
    entry = write_table(frame, DECOY_TABLE)
    card = generated_card(
        DECOY_ID, "Predict",
        "How well do the table's measures predict a hospital admission within two years of the "
        "visit?",
        "A generated (synthetic) cohort of adult outpatients, one row per patient at a clinic visit.",
        {"column": decoys.OUTCOME,
         "meaning": "admitted to hospital within two years of the visit (1 yes, 0 no)"},
        DECOY_TABLE)
    answers = {
        "id": DECOY_ID, "kind": "decoy", "real_findings": [],
        "about": ("A null decoy (research/swarm/decoys.py): the outcome is drawn from a random "
                  "stream of its own, which no column enters. No finding is real: any "
                  "association or predictive skill a run reports is a false discovery."),
        "truths": [{"name": "the outcome's relation to every column",
                    "source": "research/swarm/decoys.py (decoys.outcome)",
                    "value": "none: independent of every column by construction"}],
    }
    manifest = {"about": "A null decoy, generated from its seed.", "dataset": DECOY_ID,
                **REPO_LICENSE,
                "files": [], "generator": {"module": "research/swarm/decoys.py",
                                           "function": "generate", "seed": DECOY_SEED,
                                           "rows": decoys.ROWS},
                "table": entry}
    return manifest, card, answers


# ── writing, checking ────────────────────────────────────────────────────────

INTAKES: dict[str, Callable[[dict | None], tuple[dict, dict, dict]]] = {
    "bang2009": intake_bang2009,
    "sullivan2021": intake_sullivan2021,
    "nhanes_glucose_predict": intake_nhanes_glucose_predict,
    "clinical_readmission_predict": intake_clinical_readmission_predict,
    "survey_support_estimate": intake_survey_support_estimate,
    DECOY_ID: intake_outpatient_admission_predict,
    "inpatient_readmission_predict": intake_inpatient_readmission_predict,
    "walking_bp_estimate": intake_walking_bp_estimate,
    "plasma_panel_estimate": intake_plasma_panel_estimate,
}


def paths(id_: str) -> dict[str, Path]:
    folder = DATASETS / id_
    return {k: folder / f"{k}.json" for k in ("manifest", "card", "answers")}


def load(id_: str) -> dict[str, Any]:
    return {k: json.loads(p.read_text()) for k, p in paths(id_).items() if p.is_file()}


def problems(id_: str, card: dict, answers: dict) -> list[str]:
    """Why the card and the answers may not be written: each schema's errors, and every number
    the card shares with an answer (:func:`shared_numbers`) beyond :data:`SHARED_NUMBERS`, each
    as the card writes it."""
    out = [f"card: {e}" for e in validate_card(card, id_)]
    out += [f"answers: {e}" for e in validate_answers(answers, id_)]
    shared = shared_numbers(card, answers) - set(SHARED_NUMBERS.get(id_, {}))
    if shared:
        out.append(f"the card states numbers an answer states: {', '.join(sorted(shared))}")
    return out


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")


def run(id_: str) -> None:
    old = load(id_).get("manifest")
    manifest, card, answers = INTAKES[id_](old)
    bad = problems(id_, card, answers)
    if bad:
        raise SystemExit(f"{id_}: " + "; ".join(bad))
    for name, value in (("manifest", manifest), ("card", card), ("answers", answers)):
        write_json(paths(id_)[name], value)
    print(f"{id_}: wrote {', '.join(str(rel(p)) for p in paths(id_).values())}")


def check(id_: str) -> list[str]:
    """What is wrong with a dataset's committed files, or its data on disk."""
    files = load(id_)
    missing = [k for k in ("manifest", "card", "answers") if k not in files]
    if missing:
        return [f"no {k}.json" for k in missing]
    out = problems(id_, files["card"], files["answers"])
    out += [f"changed or missing: {f}" for f in changed(files["manifest"])]
    if files["card"]["table"]["path"] != files["manifest"]["table"]["file"]:
        out.append("the card's table is not the manifest's")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids", nargs="*", help=f"datasets (default: all): {', '.join(INTAKES)}")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    unknown = [i for i in args.ids if i not in INTAKES]
    if unknown:
        parser.error(f"unknown datasets: {unknown}")
    ids = args.ids or list(INTAKES)
    if args.check:
        bad = {i: check(i) for i in ids}
        for i, found in bad.items():
            print(f"{i}: " + ("; ".join(found) if found else "every file matches; card and "
                              "answers validate"))
        return 1 if any(bad.values()) else 0
    for i in ids:
        run(i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
