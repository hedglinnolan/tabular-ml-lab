"""Rebuild the export's checklist items from the primary sources (run by hand, not by pytest).

    venv/bin/python turbotab/core/tests/acceptance/export_data/build_checklists.py

The two source files beside this script are the publishers' full texts as Europe PMC serves them
(fetched 2026-10-05; both open access, CC BY), gzipped:

    PMC11019967.xml.gz  Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated
                        guidance for reporting clinical prediction models that use regression or
                        machine learning methods. BMJ 2024;385:e078378 (its Table 2, the checklist)
        https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11019967/fullTextXML
    PMC4896435.xml.gz   Lachat C, Hawwash D, Ocké MC, et al. Strengthening the Reporting of
                        Observational Studies in Epidemiology—Nutritional Epidemiology (STROBE-nut):
                        an extension of the STROBE statement. PLoS Med 2016;13(6):e1002036 (its
                        Table 1: STROBE's recommendations beside the 24 nut- items)
        https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4896435/fullTextXML

Each item's text is its cell's text with whitespace collapsed, verbatim otherwise (footnote marks
included). A STROBE cell holding several lettered parts — "(a) … (b) …" — is split at each part's
label, the label left out of the text; a part whose design variants sit on rows of their own
(item 6) joins those rows with a space. The output is ``turbotab/core/export/data/*.json``, which
the app reads; the acceptance test parses these XML files again with its own code and holds every
item to its source.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "export" / "data"
RETRIEVED = "2026-10-05"
PART = re.compile(r"(?:^|(?<=\s))\(([a-e])\)\s")


def _text(e: ET.Element) -> str:
    return " ".join("".join(e.itertext()).split())


def _source(name: str) -> tuple[ET.Element, str]:
    raw = gzip.decompress((HERE / name).read_bytes())
    return ET.fromstring(raw), hashlib.sha256(raw).hexdigest()


def _table(root: ET.Element, label: str) -> ET.Element:
    return next(t for t in root.iter("table-wrap")
                if t.find("label") is not None and _text(t.find("label")) == label)


def tripod() -> dict[str, Any]:
    root, sha = _source("PMC11019967.xml.gz")
    table = _table(root, "Table 2")
    items: list[dict[str, Any]] = []
    section = topic = None
    span = 0
    for tr in table.iter("tr"):
        cells = [c for c in tr if c.tag in ("td", "th")]
        if any(c.tag == "th" for c in cells):
            continue
        if len(cells) == 1:
            section = _text(cells[0])
            continue
        if len(cells) == 4:
            topic = _text(cells[0])
            span = int(cells[0].get("rowspan") or 1) - 1
            cells = cells[1:]
        else:
            assert span > 0, _text(tr)
            span -= 1
        item, scope, text = (_text(c) for c in cells)
        items.append({"id": item, "section": section, "topic": topic, "scope": scope, "text": text})
    foot = _text(table.find("table-wrap-foot"))
    return {
        "checklist": "TRIPOD+AI",
        "title": "TRIPOD+AI checklist for the reporting of prediction model studies",
        "citation": ("Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance "
                     "for reporting clinical prediction models that use regression or machine "
                     "learning methods. BMJ 2024;385:e078378. doi:10.1136/bmj-2023-078378"),
        "source": {"table": "Table 2", "pmcid": "PMC11019967", "doi": "10.1136/bmj-2023-078378",
                   "url": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11019967/fullTextXML",
                   "retrieved": RETRIEVED, "sha256": sha, "license": "CC BY 4.0"},
        "footnote": foot,
        "items": items,
    }


def _parts(text: str) -> list[tuple[str, str]]:
    """``"(a) x (b) y"`` → ``[("a", "x"), ("b", "y")]``; unlettered text → ``[("", text)]``."""
    marks = list(PART.finditer(text))
    if not marks:
        return [("", text)]
    return [(m.group(1), text[m.end():marks[i + 1].start() if i + 1 < len(marks) else None].strip())
            for i, m in enumerate(marks)]


def strobe_nut() -> dict[str, Any]:
    root, sha = _source("PMC4896435.xml.gz")
    table = _table(root, "Table 1")
    items: list[dict[str, Any]] = []
    section = topic = None
    number: str | None = None
    current: dict[str, Any] | None = None  # the lettered STROBE part a design-variant row extends
    for tr in table.iter("tr"):
        cells = [c for c in tr if c.tag in ("td", "th")]
        if any(c.tag == "th" for c in cells):
            continue
        name, num, strobe, nut = (_text(c) for c in cells)
        if name and not num and not strobe and not nut:
            section = name
            continue
        if name:
            topic = name
            if section is None or name == "Title and Abstract":
                section = name
        if num:
            number = num
        if strobe:
            parts = _parts(strobe)
            if parts == [("", strobe)] and current is not None and not num:
                current["text"] = f"{current['text']} {strobe}"  # a design variant of the part above
            else:
                for letter, text in parts:
                    current = {"id": f"{number}{letter}", "section": section, "topic": topic,
                               "kind": "STROBE", "strobe_item": number, "text": text}
                    items.append(current)
        if nut:
            m = re.match(r"^(nut-[0-9.]+)\.\s(.*)$", nut)
            assert m, nut
            # A row of its own with a topic and no item number (Ethics, Supplementary Material)
            # belongs to no STROBE item; a continuation row to the item above it.
            own = bool(name) and not num and not strobe
            items.append({"id": m.group(1), "section": section, "topic": topic, "kind": "STROBE-nut",
                          "strobe_item": None if own else number, "text": m.group(2)})
            if not num and not strobe:
                current = None
    return {
        "checklist": "STROBE-nut",
        "title": "STROBE-nut: an extension of the STROBE statement for nutritional epidemiology",
        "citation": ("Lachat C, Hawwash D, Ocké MC, et al. Strengthening the Reporting of "
                     "Observational Studies in Epidemiology—Nutritional Epidemiology (STROBE-nut): "
                     "an extension of the STROBE statement. PLoS Med 2016;13(6):e1002036. "
                     "doi:10.1371/journal.pmed.1002036"),
        "source": {"table": "Table 1", "pmcid": "PMC4896435", "doi": "10.1371/journal.pmed.1002036",
                   "url": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4896435/fullTextXML",
                   "retrieved": RETRIEVED, "sha256": sha, "license": "CC BY"},
        "footnote": None,
        "items": items,
    }


def main() -> None:
    for name, build in (("tripod_ai.json", tripod), ("strobe_nut.json", strobe_nut)):
        doc = build()
        (OUT / name).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", "utf-8")
        print(f"{name}: {len(doc['items'])} items", file=sys.stderr)


if __name__ == "__main__":
    main()
