"""Re-verify the citation registry against its sources: ``python -m turbotab.core.export.citations_check``.

For every record in ``data/citations.json``:

* a DOI is read from Crossref's public REST API (``api.crossref.org/works/{doi}``), or from
  DataCite's (``api.datacite.org/dois/{doi}``) for a DOI DataCite registers (arXiv,
  Bioconductor), and the record's authors, year, title, venue, volume, issue and pages are
  rewritten from it. A DOI that doi.org resolves but Crossref's API does not serve (it happens) is
  checked at doi.org, and its metadata read from PubMed through Europe PMC, and the record says so;
* an ISBN is looked up in Open Library; a URL is fetched and must answer;
* the app's own citations of the record (every ``cited_as`` match in the text the registry
  collects) are compared with the metadata, and each year, volume, first page or first author
  that differs is listed under ``disagreements``. The citing text is never rewritten.

Where Crossref's own record is wrong (a deposit dated a year early, an issue with no volume), the
record's ``overrides`` holds the corrected fields and why, with the independent source that shows
it; the check applies them after reading Crossref, keeps Crossref's values in
``crossref_original``, and still lists the disagreement with Crossref.

Requests carry a descriptive User-Agent and are spaced about a second apart (Crossref's polite
use); a 429 waits and retries. ``--offline`` recomputes only the disagreements.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

from turbotab.core.export import citations as C

USER_AGENT = ("TurboTab-citation-check/1.0 (TurboTab research software; citation registry "
              "check; about one request a second)")
PAUSE = 1.0


def _get(url: str, *, accept_json: bool = True, tries: int = 4) -> tuple[int, Any]:
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                body = r.read()
                time.sleep(PAUSE)
                return r.status, (json.loads(body.decode("utf-8")) if accept_json else
                                  body.decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            time.sleep(PAUSE)
            if e.code == 429 or e.code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            return e.code, None
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5 * (attempt + 1))
    return 0, None


# ---------------------------------------------------------------- metadata from each source

def _clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s or "")
    return re.sub(r"\s+", " ", s).strip()


def _year(msg: dict[str, Any], *keys: str) -> int | None:
    for k in keys:
        parts = (msg.get(k) or {}).get("date-parts") or [[None]]
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def _person(a: dict[str, Any]) -> str:
    """"Family, Given" (BibTeX's "Family, Jr., Given" with a suffix), or an organization's name."""
    if a.get("family"):
        family = _clean(a["family"]).strip(" ,")
        suffix = _clean(a.get("suffix", "")).strip(" ,")
        given = _clean(a.get("given", "")).strip(" ,")
        return ", ".join(p for p in (family, suffix, given) if p)
    return _clean(a.get("name", ""))


def _given_family(name: str) -> str:
    """"Andris Jankevics" as "Jankevics, Andris"."""
    parts = name.split()
    return f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else name


def from_crossref(msg: dict[str, Any]) -> dict[str, Any]:
    people = msg.get("author") or msg.get("editor") or []
    # A consortium Crossref lists first goes after the people (the article's own byline order).
    people = [a for a in people if a.get("family")] + [a for a in people if not a.get("family")]
    title = _clean((msg.get("title") or [""])[0])
    subtitle = _clean((msg.get("subtitle") or [""])[0]) if msg.get("subtitle") else ""
    if subtitle and subtitle.lower() not in title.lower():
        title = f"{title}: {subtitle}"
    container = _clean((msg.get("container-title") or [""])[0])
    pages = msg.get("page") or msg.get("article-number") or ""
    out = {
        "authors": [_person(a) for a in people if _person(a)],
        "title": title,
        "year": _year(msg, "published-print", "issued"),
        "venue": container,
        "volume": msg.get("volume") or "",
        "issue": msg.get("issue") or "",
        "pages": pages,
        "publisher": _clean(msg.get("publisher") or ""),
        "crossref_type": msg.get("type", ""),
        "years_seen": sorted({y for y in (_year(msg, k) for k in (
            "published-print", "published-online", "issued", "published")) if y}),
    }
    if msg.get("type") in ("book", "monograph", "edited-book", "reference-book") and msg.get("ISBN"):
        out["isbn"] = msg["ISBN"][0]
    if msg.get("edition-number") and msg["edition-number"] not in ("0", "1"):
        out["edition"] = msg["edition-number"]
    return out


def from_datacite(data: dict[str, Any]) -> dict[str, Any]:
    a = data.get("attributes") or {}
    creators: list[str] = []
    for c in a.get("creators") or []:
        if c.get("familyName"):
            creators.append(f"{c['familyName']}, {c.get('givenName', '')}".rstrip(", "))
        elif c.get("nameType") == "Organizational":
            creators.append(c.get("name", ""))
        else:  # one string, sometimes several people joined by "and"
            creators += [_given_family(n.strip()) for n in re.split(r"\s+and\s+", c.get("name", ""))
                         if n.strip()]
    return {"authors": creators, "title": _clean((a.get("titles") or [{}])[0].get("title", "")),
            "year": a.get("publicationYear"), "venue": "", "volume": "", "issue": "",
            "pages": "", "publisher": _clean(a.get("publisher") or "") if isinstance(
                a.get("publisher"), str) else _clean((a.get("publisher") or {}).get("name", "")),
            "years_seen": [a.get("publicationYear")] if a.get("publicationYear") else []}


def from_europepmc(r: dict[str, Any]) -> dict[str, Any]:
    authors = [f"{a.get('lastName', '')}, {a.get('firstName', '')}".rstrip(", ")
               for a in ((r.get("authorList") or {}).get("author") or [])]
    j = (r.get("journalInfo") or {})
    year = int(r["pubYear"]) if r.get("pubYear") else None
    return {"authors": authors, "title": _clean(r.get("title", "")).rstrip("."), "year": year,
            "venue": _clean((j.get("journal") or {}).get("title", "")),
            "volume": j.get("volume", ""), "issue": j.get("issue", ""),
            "pages": r.get("pageInfo", ""), "publisher": "", "years_seen": [year] if year else []}


def verify_doi(doi: str, agency_hint: str) -> tuple[str, dict[str, Any] | None, str]:
    """(verified_by, metadata, note) for ``doi``."""
    q = urllib.parse.quote(doi)
    if agency_hint == "datacite":
        status, data = _get(f"https://api.datacite.org/dois/{q}")
        if status == 200 and data:
            return "datacite", from_datacite(data["data"]), ""
        return "", None, f"DataCite answered {status}"
    status, data = _get(f"https://api.crossref.org/works/{q}")
    if status == 200 and data:
        return "crossref", from_crossref(data["message"]), ""
    # Crossref's API sometimes lacks a DOI that doi.org resolves: check the handle, then PubMed.
    _, ra = _get(f"https://doi.org/ra/{q}")
    agency = (ra or [{}])[0].get("RA", "")
    hs, _ = _get(f"https://doi.org/api/handles/{q}")
    if hs == 200:
        es, ep = _get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json"
                      f"&resultType=core&query=DOI:%22{q}%22")
        hits = ((ep or {}).get("resultList") or {}).get("result") or []
        if hits:
            note = (f"Crossref's REST API answered {status} for this DOI; doi.org resolves it "
                    f"(registration agency {agency or 'unknown'}), and the metadata is PubMed's, "
                    "read through Europe PMC.")
            return "doi_handle", from_europepmc(hits[0]), note
    return "", None, f"Crossref answered {status}; doi.org answered {hs}"


def verify_isbn(isbn: str, title: str) -> bool:
    """Whether Open Library knows ``isbn`` as a book whose title begins as the record's does."""
    status, data = _get(f"https://openlibrary.org/isbn/{isbn.replace('-', '')}.json")
    if status != 200 or not data:
        return False
    found, ours = _norm(data.get("title", "")), _norm(title)
    n = min(len(found), len(ours), 20)
    return n > 0 and found[:n] == ours[:n]


def verify_url(url: str) -> bool:
    """Whether ``url`` answers with a page that is not a "not found" page in disguise."""
    status, body = _get(url, accept_json=False)
    title = re.search(r"<title[^>]*>(.*?)</title>", body or "", re.S | re.I)
    shown = _clean(title.group(1)) if title else ""
    print(f"  url {status} {shown[:90]!r} {url}", file=sys.stderr)
    return 200 <= status < 400 and not re.search(r"not found|\b404\b", shown, re.I)


# ---------------------------------------------------------------- disagreements

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return re.sub(r"[^a-z]", "", "".join(ch for ch in s if not unicodedata.combining(ch)).lower())


def _span(text: str, start: int) -> str:
    """The citation that starts at ``start``: up to the next "; " or ")" or the text's end."""
    m = re.compile(r";\s|\)|$").search(text, start)
    return text[start:m.start() if m else len(text)]


_VOLPAGE = re.compile(r"(?<![\d.§])(\d{1,4})(?:\((\w+)\))?:\s?([A-Za-z]{0,3}\d+)(?:[–-](\d+))?")
_YEAR = re.compile(r"(?<!\d)(1[89]\d\d|20[0-2]\d)(?!\d)")
_NOT_AUTHORS = re.compile(r"^(?:TRIPOD|STROBE|CONSORT|MIAME|MINSEQE|RECORD|CROSS|STRATOS|R survey"
                          r"|pmp|NHANES|NCHS|CDC|NCI|EMA|FDA|US FDA|Institute of Medicine|\d)")


def _first_page(pages: str) -> str:
    return re.split(r"[–-]", pages or "")[0]


def _last_page(pages: str) -> str:
    parts = re.split(r"[–-]", pages or "")
    return parts[1] if len(parts) > 1 else ""


def disagreements(rec: dict[str, Any], spans: list[str]) -> list[str]:
    """How the app's citations of ``rec`` (``spans``) differ from its verified metadata."""
    if rec.get("verified_by") not in ("crossref", "datacite", "doi_handle"):
        return []
    out: list[str] = []
    seen_years = set(rec.get("years_seen") or ([rec["year"]] if rec.get("year") else []))
    first = _norm((rec.get("authors") or [""])[0].split(",")[0])
    volume = (rec.get("volume") or "").strip()
    fp, lp = _first_page(rec.get("pages", "")), _last_page(rec.get("pages", ""))
    for span in sorted(set(spans)):
        why = []
        y = _YEAR.search(span)
        if y and seen_years and int(y.group(1)) not in seen_years:
            why.append(f"year {y.group(1)}, the record {rec.get('year')}"
                       + (f" (online {sorted(seen_years)})" if len(seen_years) > 1 else ""))
        vp = _VOLPAGE.search(span[y.end():] if y else span) or _VOLPAGE.search(span)
        if vp and volume:
            v, _, p1, p2 = vp.groups()
            if v != volume:
                why.append(f"volume {v}, the record {volume}")
            if fp and p1.lower().lstrip("e") != fp.lower().lstrip("e") and not (
                    fp.lower().startswith(p1.lower()) or (rec.get("doi") or "").lower().endswith(
                        p1.lower())):
                why.append(f"first page {p1}, the record {fp}")
            if p2 and lp and p2 != lp and not lp.endswith(p2):
                why.append(f"last page {p2}, the record {lp}")
        lead = re.match(r"\s*(?:(?:van der|van den|van|de|Di|Van|Le) )?[A-Z][\w\-]+(?:['’][A-Z]\w+)?",
                        span)
        if lead and first and not _NOT_AUTHORS.match(span.strip()):
            surname = _norm(lead.group(0))
            names = [_norm(a.split(",")[0]) for a in rec.get("authors") or []]
            if surname not in (first,) and surname not in names[:1] and not first.startswith(surname):
                if surname in names:
                    why.append(f"first author {lead.group(0).strip()}, the record "
                               f"{rec['authors'][0].split(',')[0]}")
                elif names and not any(surname == n for n in names) and surname not in (
                        "boot",):
                    why.append(f"author {lead.group(0).strip()} is not among the record's authors")
        if why:
            out.append(f"{span.strip()!r}: " + "; ".join(why))
    return out


def citing_spans(data: dict[str, Any]) -> dict[str, list[str]]:
    """Each record's citations in the app's text, as the cited spans."""
    reg = {r["key"]: C.Citation(**{k: (tuple(v) if isinstance(v, list) else v)
                                   for k, v in r.items() if k in C.Citation.__dataclass_fields__})
           for r in data["records"]}
    out: dict[str, list[str]] = {k: [] for k in reg}
    for c in C.cited_texts():
        if c.whole:
            hits = [(i, C.resolve_at(c.text, i, reg)) for i in C.starts(c.text)]
        else:
            hits = [(m.start(), C.cited_keys_at(c, m.start(), reg)) for m in C.mentions(c.text)]
        for i, keys in hits:
            for k in keys:
                out[k].append(_span(c.text, i))
    return out


# ---------------------------------------------------------------- the run

def run(path: Path = C.DATA_FILE, *, offline: bool = False, only: set[str] | None = None) -> int:
    data = C.load(path)
    today = date.today().isoformat()
    failures = 0
    for rec in data["records"]:
        if offline or (only and rec["key"] not in only):
            continue
        if rec.get("doi"):
            hint = "datacite" if rec.get("verified_by") == "datacite" else "crossref"
            by, meta, note = verify_doi(rec["doi"], hint)
            if not meta:
                print(f"FAILED {rec['key']}: {note}", file=sys.stderr)
                rec["verified_by"], failures = "", failures + 1
                continue
            keep = {k: rec[k] for k in ("edition",) if rec.get(k) and not meta.get(k)}
            rec.update(meta, **keep)
            fixed = (rec.get("overrides") or {}).get("fields") or {}
            rec["crossref_original"] = {k: meta.get(k) for k in fixed}
            rec.update(fixed)
            if not fixed:
                rec.pop("crossref_original")
            rec["verified_by"], rec["verified_on"] = by, today
            rec["verification_note"] = note
            if not rec["verification_note"]:
                rec.pop("verification_note")
        elif rec.get("isbn") and verify_isbn(rec["isbn"], rec.get("title", "")):
            rec["verified_by"], rec["verified_on"] = "isbn", today
        elif rec.get("url") and verify_url(rec["url"]):
            rec["verified_by"], rec["verified_on"] = "url", today
        else:
            print(f"FAILED {rec['key']}: neither its ISBN nor its URL answered", file=sys.stderr)
            rec["verified_by"], failures = "", failures + 1
    spans = citing_spans(data)
    for rec in data["records"]:
        # The text is held to Crossref's own record; an override says where the record departs.
        original = rec.get("crossref_original") or {}
        found = disagreements({**rec, **original}, spans[rec["key"]])
        if original and found:
            why = rec["overrides"]["why"]
            kept = ", ".join(f"{k} {v}" for k, v in rec["overrides"]["fields"].items())
            found = [f"{d} (the registry keeps {kept}: {why})" for d in found]
        rec["disagreements"] = found
    data["checked_on"] = today if not offline else data.get("checked_on", today)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", "utf-8")
    for rec in data["records"]:
        for d in rec["disagreements"]:
            print(f"DISAGREES {rec['key']}: {d}")
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offline", action="store_true", help="recompute only the disagreements")
    ap.add_argument("--only", nargs="*", help="re-verify only these keys")
    a = ap.parse_args(argv)
    return 1 if run(offline=a.offline, only=set(a.only) if a.only else None) else 0


if __name__ == "__main__":
    raise SystemExit(main())
