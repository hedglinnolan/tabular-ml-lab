"""Codebook import: the researcher's own data dictionary as evidence (BLUEPRINT §14.2).

Nolan (2026-10-03) put codebook import in v2: *"We can always just ask the user what a specific
field means ... especially if [we] have a pretty good guess to begin with."* A codebook answers in
bulk what the readings ledger would otherwise ask column by column. Three forms are read:

* **a variable table** (CSV, TSV or Excel): one row per variable, with any of a label, a unit, a
  type and a code list (``1=Male; 2=Female``, ``1, Male | 2, Female`` as REDCap writes it), or a
  long table of codes (one row per variable and code);
* **an NHANES codebook page** (``wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<cycle>/DataFiles/X.htm``):
  each variable's name, SAS label and table of codes or ranges;
* **the variable labels an XPT file carries** (``turbotab.core.xport``).

What a codebook may do follows BLUEPRINT §14.3 ("names never count as corroboration"):

* **Structured fields settle readings, as the user's own documentation**: a unit in a unit column
  settles the column's unit reading; a value-code table settles that the numbers are codes for
  categories (and, for two codes labeled female and male, which is which); a variable type that
  says categorical or continuous settles codes or amounts. They are recorded through the same
  confirmation path ``confirm_readings`` writes (``import_codebook``), so every consumer honors
  them, and the ledger names the codebook as their evidence (``readings.codebook_source``).
* **Free-text labels are names**: ``Weight (kg)`` in an NHANES SAS label is a label, not a unit
  column. A label only strengthens the guess the ask leads with (``readings.labeled``); the user
  confirms it in one tap.
* **A codebook the values contradict is asked, never applied**: a unit the magnitudes reject (a
  height documented in m whose median is 166; a total energy documented in kcal whose ratio to its
  macronutrients' energy is 4.18), codes the data do not hold (values the code table does not list,
  or none of its codes present), values outside a documented range, a categorical type whose values
  fill a measurement's grid. A contradicted entry settles none of its fields.
* **A recorded answer stands**: a reading already answered otherwise (by the user, or by an
  earlier codebook) keeps its answer, and the record says so.
* NHANES's value tables cannot tell a measurement from a code number by their shape: masked
  variance strata (``SDMVSTRA``, "134 to 148") and ages ("0 to 79") are both "Range of Values", and
  household sizes are tables of their own numbers ("1", "2", …, "7 or more"). So a range, and a
  code table whose descriptions restate their numbers, settle nothing; their labels guide the guess.

The method contract (BLUEPRINT §13) is :data:`CONTRACT`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CONTRACT: dict[str, Any] = {
    "method": "import_codebook",
    "slot": "ingest (a reading source: before the roles and every question that reads them)",
    "data_scope": "descriptive: each documented column's values are read only to find a "
                  "contradiction; what settles a reading is the user's documentation, never the "
                  "values, and no outcome is read",
    "needs": ("a codebook naming the table's columns: a variable table, an NHANES codebook page, "
              "or an XPT file's labels",),
    "routing": {
        "question": "Do you have a data dictionary for this table?",
        "where": "after the table is read, before the roles (it answers what the card would ask)",
        "options": {"import": "structured fields settle readings; labels guide the guesses "
                              "(sound for both purposes: it is the user's own documentation)",
                    "skip": "every reading is asked as before"},
        "leash": {"structured field": "settles, recorded as the codebook's",
                  "free-text label": "strengthens the guess only (names never settle)",
                  "contradicted by the values": "asked, never applied",
                  "answered otherwise by the user": "the user's answer stands"},
    },
    "storyboard": ("read the codebook's variables", "match them to the table's columns",
                   "check each structured field against the values",
                   "settle the fields the values do not contradict; ask the rest"),
    "sentence": "codebook_sentence",
    "relations": {
        "implies": ("the readings it settles leave the ask card",
                    "its labels lead the remaining guesses"),
        "conflicts": ("a field the values contradict (asked, never applied)",
                      "a reading the user answered otherwise (the user's answer stands)"),
        "enables": ("an outcome's documented unit in its sentences",),
    },
}

TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".xlsx", ".xls")
HTML_SUFFIXES = (".htm", ".html")
XPT_SUFFIXES = (".xpt", ".xport")
SUPPORTED = "a variable table (CSV, TSV or Excel), an NHANES codebook page (.htm), or an XPT file"
MAX_DISTINCT = 5_000  # a code table is checked against at most this many distinct values

# ── units ─────────────────────────────────────────────────────────────────────

_UNIT_SPELLINGS: dict[str, tuple[str, ...]] = {
    "kcal": ("kcal", "kcals", "kilocalorie", "kilocalories", "kcal/d", "kcal/day", "kcal per day"),
    "kj": ("kj", "kilojoule", "kilojoules", "kj/d", "kj/day"),
    "g": ("g", "gm", "gms", "gram", "grams", "g/d", "g/day", "grams/day"),
    "kg": ("kg", "kgs", "kilogram", "kilograms"),
    "lb": ("lb", "lbs", "pound", "pounds"),
    "cm": ("cm", "centimeter", "centimeters", "centimetre", "centimetres"),
    "m": ("m", "meter", "meters", "metre", "metres"),
    "in": ("in", "inch", "inches"),
    "years": ("years", "year", "yr", "yrs", "y"),
    "months": ("months", "month", "mo", "mos"),
    "weeks": ("weeks", "week", "wk", "wks"),
    "days": ("days", "day"),
    "pct_energy": ("% energy", "% of energy", "% kcal", "%e", "%en", "% en", "percent of energy",
                   "% of total energy", "percent energy", "% energy intake"),
}
_UNIT_INDEX = {spelling: unit for unit, spellings in _UNIT_SPELLINGS.items()
               for spelling in spellings}


def unit_value(raw: Any) -> str | None:
    """A unit as the readings ledger records it (``kg``, ``kcal``, ``pct_energy`` …), or None for
    one no reading kind takes (``mg/dL``, ``mmHg``): those are documented units a sentence may
    state (``readings.codebook_unit``)."""
    text = " ".join(str(raw or "").strip().lower().replace("(", " ").replace(")", " ").split())
    if not text:
        return None
    return _UNIT_INDEX.get(text) or _UNIT_INDEX.get(text.replace(" ", ""))


# ── types ─────────────────────────────────────────────────────────────────────

# A type column's words that say codes or amounts. Storage types (numeric, integer, number,
# character) say neither: a stratum is "numeric" too.
CODE_TYPES = frozenset({"categorical", "category", "nominal", "ordinal", "factor", "code", "codes",
                        "coded", "binary", "dichotomous", "boolean", "bool", "yes/no", "yesno",
                        "radio", "dropdown", "checkbox", "enum", "enumerated", "categorical (coded)",
                        "class", "level", "levels"})
AMOUNT_TYPES = frozenset({"continuous", "quantitative", "measurement", "measured", "interval",
                          "ratio", "count", "counts", "amount", "scale", "numeric (continuous)",
                          "continuous numeric", "real"})

# Descriptions of codes that stand for no category: they never count toward a sex coding or a
# code table's own numbers.
SENTINEL = re.compile(r"\b(refused|don'?t know|do not know|unknown|missing|not applicable|n/?a|"
                      r"no answer|not ascertained|could not|cannot|blank|skip)\b", re.I)


def type_reading(raw: Any) -> str | None:
    """``code``, ``amount`` or None (a storage type, or nothing written)."""
    text = " ".join(str(raw or "").strip().lower().split())
    if text in CODE_TYPES:
        return "code"
    if text in AMOUNT_TYPES:
        return "amount"
    return None


# ── the model ─────────────────────────────────────────────────────────────────


@dataclass
class Entry:
    """One variable a codebook documents."""

    variable: str
    label: str = ""
    unit: str = ""
    type: str = ""
    codes: dict[str, str] = field(default_factory=dict)   # code (as written) -> description
    range: tuple[float, float] | None = None              # documented values outside the codes

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["range"] = list(self.range) if self.range is not None else None
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Entry":
        rng = d.get("range")
        return cls(variable=str(d["variable"]), label=str(d.get("label") or ""),
                   unit=str(d.get("unit") or ""), type=str(d.get("type") or ""),
                   codes={str(k): str(v) for k, v in (d.get("codes") or {}).items()},
                   range=(float(rng[0]), float(rng[1])) if rng else None)


@dataclass
class Codebook:
    name: str
    form: str            # table | nhanes | xpt
    entries: list[Entry]
    notes: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        blob = json.dumps({"form": self.form, "entries": [e.to_dict() for e in self.entries]},
                          sort_keys=True, ensure_ascii=False)
        return "c" + hashlib.blake2b(blob.encode("utf-8"), digest_size=8).hexdigest()[:10]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "form": self.form, "notes": list(self.notes),
                "entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Codebook":
        return cls(name=str(d["name"]), form=str(d["form"]),
                   entries=[Entry.from_dict(e) for e in d.get("entries") or []],
                   notes=list(d.get("notes") or []))


class CodebookError(ValueError):
    """The file is not a codebook TurboTab reads, with the reason in plain words."""


# ── reading the forms ─────────────────────────────────────────────────────────


def code_key(value: Any) -> str:
    """A code or a value as the comparison reads it: ``1``, ``1.0`` and ``"1"`` are one code;
    text is stripped."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f):
            return ""
        return str(int(f)) if f.is_integer() else repr(f)
    text = str(value).strip()
    try:
        f = float(text)
    except ValueError:
        return text
    if math.isfinite(f):
        return str(int(f)) if f.is_integer() else repr(f)
    return text


class _NhanesPage(HTMLParser):
    """The variables of an NHANES codebook page: each ``div.pagebreak`` holds a ``dl`` (Variable
    Name, SAS Label, English Text, Target) and a ``table.values`` (Code or Value, Value
    Description, Count, Cumulative, Skip to Item). Only rows of a values table inside a variable's
    own division are its codes (the page's appendices hold tables of their own)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.variables: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.text: list[str] = []
        self.row: list[str] | None = None
        self.term: str | None = None
        self.depth = 0          # open divisions
        self.opened_at = -1     # the depth at which the current variable's division opened
        self.in_values = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = (dict(attrs).get("class") or "").split()
        if tag == "div":
            self.depth += 1
            if "pagebreak" in classes:
                self.current = {"rows": []}
                self.variables.append(self.current)
                self.opened_at = self.depth
        if self.current is None:
            return
        if tag == "table":
            self.in_values = "values" in classes
        elif tag in ("dt", "dd", "td", "th"):
            self.text = []
        elif tag == "tr" and self.in_values:
            self.row = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "div":
            if self.current is not None and self.depth == self.opened_at:
                self.current, self.opened_at, self.in_values = None, -1, False
            self.depth -= 1
            return
        if self.current is None:
            return
        text = " ".join("".join(self.text).split())
        if tag == "table":
            self.in_values = False
        elif tag == "dt":
            self.term = text.rstrip(":").strip().lower()
        elif tag == "dd" and self.term:
            self.current[self.term] = text
            self.term = None
        elif tag in ("td", "th") and self.row is not None:
            self.row.append(text)
        elif tag == "tr" and self.row is not None:
            if self.row and self.row[0].lower() != "code or value":
                self.current["rows"].append(self.row)
            self.row = None

    def handle_data(self, data: str) -> None:
        self.text.append(data)


_RANGE = re.compile(r"^\s*(-?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)\s+to\s+(-?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)\s*$")


def _number(text: str) -> float | None:
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def parse_nhanes(html: str, name: str) -> Codebook:
    """An NHANES codebook page: each variable's name, SAS label, codes (``Value Description`` per
    ``Code or Value``) and documented range (a ``Range of Values`` row). The ``.`` Missing row is
    no code."""
    page = _NhanesPage()
    page.feed(html)
    entries: list[Entry] = []
    for v in page.variables:
        variable = v.get("variable name")
        if not variable:
            continue
        codes: dict[str, str] = {}
        rng: tuple[float, float] | None = None
        for row in v["rows"]:
            if len(row) < 2:
                continue
            code, description = row[0].strip(), row[1].strip()
            if code in (".", "") and description.lower() == "missing":
                continue
            if description.lower() == "range of values":
                m = _RANGE.match(code)
                if m:
                    lo, hi = _number(m.group(1)), _number(m.group(2))
                    if lo is not None and hi is not None:
                        rng = (min(lo, hi), max(lo, hi)) if rng is None else \
                            (min(rng[0], lo, hi), max(rng[1], lo, hi))
                continue
            if code in (".", ""):
                continue
            codes[code] = description
        entries.append(Entry(variable=variable.strip(), label=v.get("sas label", ""),
                             codes=codes, range=rng))
    if not entries:
        raise CodebookError(f"{name!r} lists no variables the way an NHANES codebook page does "
                            "(a Variable Name, a SAS Label and a table of codes)")
    return Codebook(name=name, form="nhanes", entries=entries)


_HEADERS: dict[str, tuple[str, ...]] = {
    "variable": ("variable", "variable name", "var", "varname", "var name", "name", "column",
                 "column name", "field", "field name", "variable / field name",
                 "variable field name", "item", "item name"),
    "label": ("label", "variable label", "sas label", "description", "field label",
              "english text", "definition", "question", "question text", "item label"),
    "unit": ("unit", "units", "unit of measure", "unit of measurement", "uom",
             "measurement unit"),
    "type": ("type", "variable type", "data type", "field type", "measurement level",
             "level of measurement", "measure", "var type"),
    "codes": ("codes", "code list", "values", "value labels", "categories", "choices",
              "response options", "allowed values", "coding", "value codes",
              "choices, calculations, or slider labels", "choices calculations or slider labels"),
    "code": ("code", "value", "code value", "level"),
    "code_label": ("code label", "value label", "meaning", "code description",
                   "value description", "level label"),
    "min": ("min", "minimum", "valid min", "lowest value"),
    "max": ("max", "maximum", "valid max", "highest value"),
    "range": ("range", "valid range", "range of values"),
}


def _header_key(text: Any) -> str:
    return " ".join(re.sub(r"[_\-]+", " ", str(text)).strip().lower().split())


def _columns(frame: Any) -> dict[str, str]:
    """``{role: column}`` for a table's header cells (the first spelling each role matches)."""
    found: dict[str, str] = {}
    for col in frame.columns:
        key = _header_key(col)
        for role, spellings in _HEADERS.items():
            if role not in found and key in spellings:
                found[role] = col
                break
    return found


_PAIR = re.compile(r"^\s*([^=:,]+?)\s*[=:,]\s*(.+?)\s*$")


def parse_codes(text: Any) -> dict[str, str]:
    """``1=Male; 2=Female`` / ``1, Male | 2, Female`` / one pair per line -> ``{code: label}``."""
    raw = str(text or "").strip()
    if not raw:
        return {}
    pieces = re.split(r"\s*(?:\||;|\n)\s*", raw)
    out: dict[str, str] = {}
    for piece in pieces:
        m = _PAIR.match(piece)
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


def _cell(row: Any, column: str | None) -> str:
    if column is None:
        return ""
    value = row.get(column)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _table_entries(frame: Any) -> list[Entry]:
    cols = _columns(frame)
    var = cols.get("variable")
    if var is None:
        return []
    long_codes = "code" in cols and "code_label" in cols
    entries: dict[str, Entry] = {}
    for _, row in frame.iterrows():
        name = _cell(row, var)
        if not name:
            continue
        e = entries.setdefault(name, Entry(variable=name))
        for role in ("label", "unit", "type"):
            text = _cell(row, cols.get(role))
            if text and not getattr(e, role):
                setattr(e, role, text)
        e.codes.update(parse_codes(_cell(row, cols.get("codes"))))
        if long_codes:
            code, meaning = _cell(row, cols["code"]), _cell(row, cols["code_label"])
            if code:
                e.codes[code] = meaning
        lo, hi = _number(_cell(row, cols.get("min"))), _number(_cell(row, cols.get("max")))
        m = _RANGE.match(_cell(row, cols.get("range")))
        if m:
            lo, hi = _number(m.group(1)), _number(m.group(2))
        if lo is not None and hi is not None:
            e.range = (min(lo, hi), max(lo, hi))
    return list(entries.values())


def parse_table(path: Path, name: str) -> Codebook:
    """A variable table: CSV/TSV/TXT, or every sheet of a workbook (a sheet of codes, one row per
    variable and code, joins the variables' sheet)."""
    import pandas as pd

    suffix = path.suffix.lower()
    frames: list[Any] = []
    if suffix in (".xlsx", ".xls"):
        try:
            book = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)
        except ImportError as exc:
            raise CodebookError(f"reading {suffix} files needs a package that is not installed "
                                f"({exc}); save the codebook as .xlsx or CSV") from None
        frames = list(book.values())
    else:
        sep = "\t" if suffix == ".tsv" else None
        for encoding in ("utf-8-sig", "latin-1"):
            try:
                frames = [pd.read_csv(path, sep=sep, engine="python", dtype=str,
                                      keep_default_na=False, encoding=encoding)]
                break
            except UnicodeDecodeError:
                continue
    merged: dict[str, Entry] = {}
    for frame in frames:
        for e in _table_entries(frame):
            have = merged.get(e.variable)
            if have is None:
                merged[e.variable] = e
                continue
            for role in ("label", "unit", "type"):
                if not getattr(have, role) and getattr(e, role):
                    setattr(have, role, getattr(e, role))
            have.codes.update(e.codes)
            have.range = have.range or e.range
    if not merged:
        raise CodebookError(f"{name!r} has no column naming the variables (a header such as "
                            "\"variable\", \"name\" or \"field name\")")
    return Codebook(name=name, form="table", entries=list(merged.values()))


def from_labels(sources: Iterable[Mapping[str, Any]], name: str) -> Codebook:
    """The variable labels XPT files carry (``datastore.read_labels``' sources, or one file's
    header): labels only; a SAS format is the file's own reading of its values, already applied."""
    entries = []
    for source in sources:
        for e in source.get("entries") or []:
            if e.get("label"):
                entries.append(Entry(variable=str(e["variable"]), label=str(e["label"])))
    if not entries:
        raise CodebookError(f"{name} carries no variable labels")
    return Codebook(name=name, form="xpt", entries=entries)


def read(path: str | Path, name: str | None = None) -> Codebook:
    """Read a codebook file in any of the three forms (by its type)."""
    path = Path(path)
    name = name or path.name
    suffix = path.suffix.lower()
    if suffix in HTML_SUFFIXES:
        return parse_nhanes(path.read_text(encoding="utf-8", errors="replace"), name)
    if suffix in TABLE_SUFFIXES:
        return parse_table(path, name)
    if suffix in XPT_SUFFIXES:
        from turbotab.core import xport

        try:
            head = xport.read_header(path)
        except xport.XportError as exc:
            raise CodebookError(str(exc)) from None
        return from_labels([{"entries": [v.to_dict() for v in head.member.variables]}],
                           name)
    raise CodebookError(f"TurboTab cannot read {name!r} as a codebook; it reads {SUPPORTED}")


def check_suffix(path: Path) -> str:
    """The codebook form a file's name implies; raises ValueError for another type (the upload
    receiver's check)."""
    suffix = path.suffix.lower()
    if suffix in HTML_SUFFIXES:
        return "nhanes"
    if suffix in TABLE_SUFFIXES:
        return "table"
    if suffix in XPT_SUFFIXES:
        return "xpt"
    raise ValueError(f"TurboTab cannot read {path.name!r} as a codebook; it reads {SUPPORTED}")


# ── checking a codebook against the values ────────────────────────────────────


@dataclass
class Assessment:
    """What importing a codebook would do: the readings its structured fields settle (``items``,
    each ``{reading, column, value, field}``), the documented units, the labels, the fields the
    values contradict (``asked``, each ``{column, field, says, values}`` with its ``exits``), the
    readings already recorded otherwise, by the user or an earlier codebook (``kept``: the
    recorded answer stands), and the entries naming no column."""

    codebook: Codebook
    matched: dict[str, str]                 # entry variable -> the table's column
    items: list[dict[str, str]] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    asked: list[dict[str, Any]] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    already: list[str] = field(default_factory=list)   # recorded already with the same value

    @property
    def unmatched(self) -> list[str]:
        return [e.variable for e in self.codebook.entries if e.variable not in self.matched]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.codebook.id, "name": self.codebook.name, "form": self.codebook.form,
                "n_entries": len(self.codebook.entries), "n_matched": len(self.matched),
                "unmatched": self.unmatched[:20], "n_unmatched": len(self.unmatched),
                "settles": [dict(i) for i in self.items], "units": dict(self.units),
                "labels": len(self.labels), "asked": [dict(a) for a in self.asked],
                "kept": list(self.kept), "sentence": codebook_sentence(
                    self.codebook.name, self.codebook.form, len(self.matched), self.items,
                    self.asked, self.kept, self.labels)}


@dataclass
class _Facts:
    dtype: str
    present: int
    values: list[str] | None   # distinct values (code_key), None when more than MAX_DISTINCT
    numbers: Any               # the finite numbers (numpy) of a numeric column, or None


def _column_facts(parquet: Path, columns: Sequence[str], dtypes: Mapping[str, str]
                  ) -> dict[str, _Facts]:
    import duckdb
    import numpy as np

    out: dict[str, _Facts] = {}
    rel = "read_parquet('" + str(parquet).replace("'", "''") + "')"
    con = duckdb.connect()
    try:
        for c in columns:
            ident = '"' + c.replace('"', '""') + '"'
            rows = con.execute(f"SELECT DISTINCT {ident} FROM {rel} WHERE {ident} IS NOT NULL "
                               f"LIMIT {MAX_DISTINCT + 1}").fetchall()
            values = [code_key(r[0]) for r in rows]
            numbers = None
            if dtypes.get(c) in ("numeric", "integer"):
                arr = con.execute(f"SELECT CAST({ident} AS DOUBLE) FROM {rel} "
                                  f"WHERE {ident} IS NOT NULL").fetchnumpy()
                x = np.asarray(list(arr.values())[0], dtype=float)
                numbers = x[np.isfinite(x)]
            present = int(con.execute(f"SELECT count({ident}) FROM {rel}").fetchone()[0])
            out[c] = _Facts(dtypes.get(c, "text"), present,
                            values if len(values) <= MAX_DISTINCT else None, numbers)
    finally:
        con.close()
    return out


def _restates_numbers(codes: Mapping[str, str]) -> bool:
    """A code table whose descriptions are its own numbers (``1``: "1", ``7``: "7 or more people
    in the Household"): a count with a top code, not categories."""
    real = [(c, d) for c, d in codes.items() if not SENTINEL.search(d)]
    if not real:
        return False
    restated = sum(1 for c, d in real
                   if re.match(rf"^\s*{re.escape(code_key(c))}(?![\w.])", d.strip()))
    return restated * 2 >= len(real)


def _sex_coding(codes: Mapping[str, str]) -> str | None:
    """``female=<code>,male=<code>`` when exactly two of the codes (sentinels aside) are labeled
    with a sex's words."""
    from turbotab.core.readings import FEMALE_WORDS, MALE_WORDS, sex_coding_value

    real = {c: d for c, d in codes.items() if not SENTINEL.search(d)}
    if len(real) != 2:
        return None
    female = [c for c, d in real.items() if d.strip().lower() in FEMALE_WORDS]
    male = [c for c, d in real.items() if d.strip().lower() in MALE_WORDS]
    if len(female) == 1 and len(male) == 1:
        return sex_coding_value(code_key(female[0]), code_key(male[0]))
    return None


def _fmt(x: float) -> str:
    return f"{x:,.4g}" if abs(x) < 1e5 else f"{x:,.0f}"


def _listed(values: Sequence[str], limit: int = 4) -> str:
    shown = ", ".join(f"`{v}`" for v in values[:limit])
    more = len(values) - limit
    return shown + (f" and {more:,} more" if more > 0 else "")


# Human ranges a body measure's median falls in, per unit (a contradiction needs the magnitudes to
# reject the documented unit: a kg weight's median is never 70,000, an age in years never 400).
_MEDIAN_BOUNDS = {"kg": (0.5, 400.0), "lb": (1.0, 900.0), "years": (0.0, 120.0),
                  "pct_energy": (0.0, 100.0)}


def _unit_contradiction(column: str, unit: str, facts: _Facts, label: str, store: Any
                        ) -> tuple[str, list[str]] | None:
    """What the values say against a documented unit, and the units they leave, or None."""
    import numpy as np

    from turbotab.core import readings
    from turbotab.core.recognizers import tokens

    if facts.dtype not in ("numeric", "integer"):
        return f"text values, which have no unit", []
    x = facts.numbers
    if x is None or not len(x):
        return None
    positive = x[x > 0]
    median = float(np.median(positive)) if len(positive) else None
    words = set(tokens(column)) | set(tokens(label))
    if unit in ("cm", "m", "in") and words & {"height", "stature", "bmxht", "length"}:
        verdict = readings.by_values("unit:height", positive)
        if verdict.settles and verdict.value != unit:
            return (f"a median of {_fmt(median or 0)}, a human height only in {verdict.value}",
                    [str(verdict.value)])
        if unit == "in" and verdict.settles:
            return f"a median of {_fmt(median or 0)}, which no one is in inches", [str(verdict.value)]
    if unit in ("kcal", "kj") and store is not None:
        try:
            from turbotab.core.decisions import _names_a_macro_total

            macros = [c for c in store.columns if c != column and _names_a_macro_total(c)]
            if macros:
                frame = store.materialize([column, *macros])
                verdict = readings.by_values("unit:energy", frame, column)
                fits = {u for u, _d in verdict.candidates}
                if verdict.settles and verdict.value != unit:
                    return (f"values that match the energy their macronutrients carry in "
                            f"{verdict.value}", [str(verdict.value)])
                if fits and unit not in fits:
                    return (f"values {verdict.evidence.removeprefix('is ')}",
                            sorted(fits))
        except Exception:  # noqa: BLE001 - a check that cannot run contradicts nothing
            pass
    bounds = _MEDIAN_BOUNDS.get(unit)
    if bounds is not None:
        lo, hi = bounds
        values = positive if unit != "pct_energy" else x
        if len(values):
            if unit == "pct_energy":
                if float(values.min()) < lo or float(values.max()) > hi:
                    return f"values from {_fmt(float(values.min()))} to {_fmt(float(values.max()))}, " \
                           f"outside 0–100", []
            elif median is not None and not lo <= median <= hi:
                return f"a median of {_fmt(median)}, outside any human value in {unit}", []
    return None


def assess(codebook: Codebook, parquet: Path, info: Mapping[str, Any], state: Any = None, *,
           store: Any = None) -> Assessment:
    """Check ``codebook`` against the table at ``parquet`` (``info``: its DatasetInfo dict) and the
    answers in ``state``: what it settles, what it asks, what it leaves to the user's answers."""
    from turbotab.core import readings

    dtypes = {str(c["name"]): str(c["dtype"]) for c in info.get("columns") or []}
    by_lower: dict[str, list[str]] = {}
    for c in dtypes:
        by_lower.setdefault(c.lower(), []).append(c)
    matched: dict[str, str] = {}
    for e in codebook.entries:
        if e.variable in dtypes:
            matched[e.variable] = e.variable
        elif len(by_lower.get(e.variable.lower(), [])) == 1:
            matched[e.variable] = by_lower[e.variable.lower()][0]
    a = Assessment(codebook=codebook, matched=matched)
    target = getattr(state, "target", None) if state is not None else None
    check = [matched[e.variable] for e in codebook.entries
             if e.variable in matched and (e.codes or e.unit or e.type or e.range)]
    facts = _column_facts(parquet, list(dict.fromkeys(check)), dtypes)
    for e in codebook.entries:
        column = matched.get(e.variable)
        if column is None:
            continue
        if e.label:
            a.labels[column] = e.label
        f = facts.get(column)
        if f is None or codebook.form == "xpt":
            if e.unit.strip() and f is None:
                a.units[column] = e.unit.strip()
            continue
        proposed: list[tuple[str, str, str]] = []   # (reading, value, field)
        conflicts: list[dict[str, Any]] = []
        numeric = f.dtype in ("numeric", "integer")
        values = f.values
        # value-code tables
        codes = {code_key(c): d for c, d in e.codes.items() if code_key(c)}
        if codes and values is not None and f.present:
            # Codes the data do not hold: a value the code table does not list (nor its range).
            # A listed code no row holds is no contradiction (a subset; NHANES lists codes whose
            # count is 0).
            outside = [v for v in values if v not in codes]
            if e.range is not None and outside:
                lo, hi = e.range
                outside = [v for v in outside
                           if _number(v) is None or not lo <= float(v) <= hi]
            if outside:
                conflicts.append({"field": "codes", "says": f"the codes {_listed(list(codes))}",
                                  "values": f"values it does not list: {_listed(sorted(outside))}"})
        elif e.range is not None and numeric and f.numbers is not None and len(f.numbers):
            lo, hi = e.range
            listed = {_number(c) for c in codes} - {None}
            out = [v for v in f.numbers if not lo <= v <= hi and v not in listed]
            if out:
                conflicts.append({"field": "range", "says": f"values from {_fmt(lo)} to {_fmt(hi)}",
                                  "values": f"{len(out):,} values outside it, from "
                                            f"{_fmt(min(out))} to {_fmt(max(out))}"})
        if codes and e.range is not None and values is not None and not conflicts:
            lo, hi = e.range
            out = [v for v in values if v not in codes and _number(v) is not None
                   and not lo <= float(v) <= hi]
            if out:
                conflicts.append({"field": "range", "says": f"values from {_fmt(lo)} to {_fmt(hi)}",
                                  "values": f"values outside it: {_listed(sorted(out))}"})
        kind_of_type = type_reading(e.type)
        if codes and e.range is None and not _restates_numbers(codes):
            if kind_of_type == "amount":
                conflicts.append({"field": "type", "says": f"the type {e.type!r} beside a code "
                                                           "table", "values": "a code table"})
            elif numeric:
                proposed.append(("code_or_count", "code", "codes"))
            coding = _sex_coding(codes)
            if coding is not None:
                proposed.append(("sex_coding", coding, "codes"))
        elif kind_of_type is not None and not codes:
            if not numeric and kind_of_type == "amount" and f.present:
                sample = next(iter(values or []), "")
                conflicts.append({"field": "type", "says": f"the type {e.type!r}",
                                  "values": f"text, such as `{sample}`"})
            elif numeric and kind_of_type == "code" and f.numbers is not None and \
                    readings.by_values("code_or_count", f.numbers).settles:
                conflicts.append({"field": "type", "says": f"the type {e.type!r}",
                                  "values": "values with decimals filling a measurement's grid"})
            elif numeric:
                proposed.append(("code_or_count", kind_of_type, "type"))
        # units
        documented = e.unit.strip()
        unit = unit_value(documented) if documented else None
        if documented:
            # A unit no reading kind takes (mg/dL) has no magnitude test here: it is the user's.
            against = (_unit_contradiction(column, unit or documented, f, e.label, store)
                       if unit is not None or not numeric else None)
            if against is not None:
                conflicts.append({"field": "unit", "says": documented, "values": against[0],
                                  "units": against[1]})
            elif unit is not None and numeric:
                proposed.append(("unit", unit, "unit"))
        if conflicts:
            for c in conflicts:
                units_left = c.pop("units", [])
                a.asked.append({"column": column, **c,
                                "exits": _conflict_exits(column, c["field"], e, units_left,
                                                         values)})
            continue
        if documented:
            a.units[column] = documented  # not contradicted: a sentence may state it
        for reading_kind, value, source in proposed:
            if column == target:
                continue  # the outcome's kind and unit have their own questions
            recorded = readings.confirmation(state, reading_kind, column) if state is not None \
                else None
            k = readings.key(reading_kind, column)
            if recorded is not None and str(recorded) == str(value):
                if readings.codebook_source(state, reading_kind, column) is None:
                    a.already.append(k)   # the user's own answer, the same: nothing to write
                    continue
                # Recorded by a codebook (this one, imported again after a join): written again,
                # so the latest import still names it as its evidence.
            elif recorded is not None:
                a.kept.append(k)
                continue
            a.items.append({"reading": reading_kind, "column": column, "value": value,
                            "field": source})
    return a


def _conflict_exits(column: str, field_name: str, e: Entry, units_left: Sequence[str],
                    values: Sequence[str] | None) -> list[dict[str, Any]]:
    """The answers a contradicted field offers: the values' reading and the codebook's, one
    confirmation each (the user decides)."""
    from turbotab.core import readings

    if field_name == "unit":
        documented = unit_value(e.unit)
        options = [*units_left, *([documented] if documented else [])]
        return [readings.confirm_exit("unit", column, u, f"`{column}` is in {u}")
                for u in dict.fromkeys(options) if u in readings.UNIT_VALUES]
    if field_name in ("codes", "type"):
        return readings.code_or_count_exits(column)
    return []


# ── the decision: validator, completion, sentence (registered on import) ──────


def codebooks_dir(project_dir: Path) -> Path:
    return Path(project_dir) / "codebooks"


def stage(codebook: Codebook, project_dir: Path, source: Path | None = None) -> str:
    """Keep ``codebook`` (and its source file) in the project: ``codebooks/<id>/``."""
    import shutil

    folder = codebooks_dir(project_dir) / codebook.id
    folder.mkdir(parents=True, exist_ok=True)
    if source is not None and Path(source).is_file():
        keep = folder / "source"
        keep.mkdir(exist_ok=True)
        dest = keep / Path(source).name
        if Path(source).resolve() != dest.resolve():
            shutil.copyfile(source, dest)
    tmp = folder / ".codebook.json.tmp"
    tmp.write_text(json.dumps(codebook.to_dict(), ensure_ascii=False), encoding="utf-8")
    tmp.replace(folder / "codebook.json")
    return codebook.id


def load(project_dir: Path, codebook_id: str) -> Codebook | None:
    if not re.fullmatch(r"c[0-9a-f]{10}", str(codebook_id or "")):
        return None
    try:
        with open(codebooks_dir(project_dir) / codebook_id / "codebook.json",
                  encoding="utf-8") as fh:
            return Codebook.from_dict(json.load(fh))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _project_dir(ctx: Any) -> Path | None:
    raw = ctx.get("project_dir") if isinstance(ctx, Mapping) else getattr(ctx, "project_dir", None)
    return Path(raw) if raw else None


def assessment_for(codebook: Codebook, project_dir: Path, state: Any) -> Assessment:
    """:func:`assess` against the project's table as it was read (``data/raw.parquet``)."""
    from turbotab.core.datastore import DataStore, _sidecar_path

    table = Path(project_dir) / "data" / "raw.parquet"
    with open(_sidecar_path(table), encoding="utf-8") as fh:
        info = json.load(fh)["info"]
    store = DataStore(table, 1 << 30)
    try:
        return assess(codebook, table, info, state, store=store)
    finally:
        store.close()


def _assessed(decision: Any, ctx: Any) -> Assessment:
    from turbotab.core.decisions import Refusal, _ctx

    pdir = _project_dir(ctx)
    book = load(pdir, decision.codebook) if pdir is not None else None
    if book is None:
        raise Refusal("unknown_codebook", f"This project has no codebook `{decision.codebook}`.",
                      exits=[{"label": "Add the codebook to the project first", "decision": None}])
    status = _ctx(ctx, "ingest_status")
    if status not in (None, "fresh") or not (pdir / "data" / "raw.parquet").is_file():
        raise Refusal("not_yet", "The table is still being read; import the codebook once it is "
                                 "ready.", exits=[])
    a = assessment_for(book, pdir, _ctx(ctx, "state"))
    if not a.matched:
        raise Refusal("codebook_names_no_column",
                      f"No variable `{book.name}` documents is a column of this table "
                      f"(it names {_listed(a.unmatched)}).",
                      exits=[{"label": "Import the codebook of this table's file", "decision": None}])
    return a


def _codebook_documents_the_table(decision: Any, ctx: Any) -> None:
    if _project_dir(ctx) is None:
        return  # a context that names no project (a unit test of the fold) checks nothing
    _assessed(decision, ctx)


def _codebook_records_what_it_settles(decision: Any, ctx: Any) -> Any:
    """Everything but the codebook's id is the server's, read from the codebook and the values:
    what a client sends in its place is discarded."""
    from turbotab.core.decisions import CodebookConflict, ReadingItem

    if _project_dir(ctx) is None:
        return decision
    a = _assessed(decision, ctx)
    return decision.model_copy(update={
        "name": a.codebook.name, "form": a.codebook.form,
        "items": [ReadingItem(reading=i["reading"], column=i["column"], value=i["value"])
                  for i in a.items],
        "units": dict(a.units), "labels": dict(a.labels),
        "asked": [CodebookConflict(column=x["column"], field=x["field"], says=x["says"],
                                   values=x["values"]) for x in a.asked],
        "kept": list(a.kept), "n_entries": len(a.codebook.entries), "n_matched": len(a.matched)})


_FORM_WORDS = {"table": "a variable table", "nhanes": "an NHANES codebook page",
               "xpt": "the variable labels of a SAS transport file"}
_UNIT_WORDS = {"kj": "kJ", "pct_energy": "percent of energy", "m": "meters", "in": "inches"}
_FIELD_WORDS = {"unit": "unit", "codes": "codes", "type": "type", "range": "range"}


def _ticks(items: Sequence[str]) -> str:
    quoted = [f"`{x}`" for x in items]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]


def _shown(items: Sequence[str], limit: int = 4) -> str:
    """Up to ``limit`` already-quoted items, then the count of the rest."""
    head = list(items[:limit])
    more = len(items) - limit
    if more > 0:
        return ", ".join(head) + f" and `{more:,}` more"
    return head[0] if len(head) == 1 else ", ".join(head[:-1]) + " and " + head[-1]


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"`{n:,}` {one if n == 1 else (many or one + 's')}"


def _items_clause(items: Sequence[Mapping[str, Any]]) -> list[str]:
    """The settlements in words, one clause per kind and value."""
    from turbotab.core.readings import parse_sex_coding

    groups: dict[tuple[str, str], list[str]] = {}
    for i in items:
        value, kind, col = str(i["value"]), str(i["reading"]), str(i["column"])
        if kind == "unit":
            groups.setdefault(("unit", ""), []).append(
                f"`{col}` in {_UNIT_WORDS.get(value, value)}")
        elif kind == "sex_coding":
            coding = parse_sex_coding(value) or {}
            said = ", ".join(f"`{level}` {sex}" for level, sex in coding.items())
            groups.setdefault(("sex_coding", ""), []).append(f"`{col}` ({said})")
        else:
            groups.setdefault((kind, value), []).append(f"`{col}`")
    parts = []
    for (kind, value), shown in groups.items():
        n = len(shown)
        if kind == "unit":
            parts.append(f"the units of {_plural(n, 'column')} ({_shown(shown)})")
        elif kind == "sex_coding":
            parts.append(f"the sex coding of {_shown(shown)}")
        elif value == "code":
            parts.append(f"{_plural(n, 'column')} as codes for categories ({_shown(shown)})")
        else:
            parts.append(f"{_plural(n, 'column')} as amounts ({_shown(shown)})")
    return parts


def _file_names(name: str) -> list[str]:
    return [n for n in re.split(r", | and ", name) if n]


def codebook_sentence(name: str, form: str, n_matched: int, items: Sequence[Mapping[str, Any]],
                      asked: Sequence[Mapping[str, Any]], kept: Sequence[str],
                      labels: Mapping[str, str]) -> str:
    """The methods sentence a codebook import records (DATAIN acceptance 4): what the codebook
    documents, what its structured fields settled as the user's own documentation, what the values
    contradicted (asked, never applied), which answers of the user's stand, and that its labels
    settle nothing."""
    if form == "xpt":
        return (f"The variable labels carried by {_ticks(_file_names(name))} describe "
                f"`{n_matched:,}` of the table's columns; a label is a name, so they settle no "
                f"reading and are shown beside the guesses the questions lead with")
    sentences = [f"The codebook `{name}` ({_FORM_WORDS.get(form, form)}) documents "
                 f"`{n_matched:,}` of the table's columns"]
    parts = _items_clause(items)
    if parts:
        joined = parts[0] if len(parts) == 1 else "; ".join(parts[:-1]) + "; and " + parts[-1]
        sentences.append(f"Its structured fields settled {joined}, as the user's own "
                         f"documentation")
    else:
        sentences.append("Its structured fields settled no reading")
    if asked:
        shown = "; ".join(f"`{x['column']}`'s {_FIELD_WORDS.get(str(x['field']), x['field'])} (it "
                          f"says {x['says']}; the values show {x['values']})"
                          for x in list(asked)[:3])
        more = len(asked) - 3
        sentences.append(f"{_plural(len(asked), 'of its fields contradicts', 'of its fields contradict')} "
                         f"the values and {'was' if len(asked) == 1 else 'were'} asked instead of "
                         f"applied: {shown}" + (f"; and `{more:,}` more" if more > 0 else ""))
    if kept:
        words = [k.split(":", 1) for k in kept]
        shown_kept = _shown([f"`{c}`'s {k.replace('_', ' ')}" for k, c in words], 3)
        sentences.append(f"{_plural(len(kept), 'reading')} already recorded otherwise "
                         f"{'keeps its' if len(kept) == 1 else 'keep their'} recorded answer "
                         f"({shown_kept})")
    if labels:
        sentences.append(f"Its labels of {_plural(len(labels), 'column')} are shown beside the "
                         f"guesses the questions lead with and settle nothing")
    return ". ".join(sentences)


def _register() -> None:
    from turbotab.core import voice
    from turbotab.core.decisions import register_completion, register_validator

    register_validator("import_codebook", _codebook_documents_the_table)
    register_completion("import_codebook", _codebook_records_what_it_settles)

    @voice.register_sentence("import_codebook")
    def _import_codebook(d: Any, state: Any, ctx: Any) -> str:
        items = [{"reading": i.reading, "column": i.column, "value": i.value} for i in d.items]
        asked = [x.model_dump() for x in d.asked]
        return codebook_sentence(d.name or d.codebook, d.form or "table", d.n_matched, items,
                                 asked, d.kept, d.labels)


_register()

__all__ = ["AMOUNT_TYPES", "Assessment", "CODE_TYPES", "CONTRACT", "Codebook", "CodebookError",
           "Entry", "assess", "assessment_for", "check_suffix", "code_key", "codebook_sentence",
           "from_labels", "load", "parse_codes", "parse_nhanes", "parse_table", "read", "stage",
           "type_reading", "unit_value"]
