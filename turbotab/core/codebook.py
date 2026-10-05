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
  counted exactly however many distinct values the column holds), values outside a documented
  range (a value within four units in the last place of a bound, in the column's own
  floating-point type, is the bound: an IBM float read as an IEEE double moves by one), a
  categorical type whose values fill a measurement's grid. A contradicted entry settles none of
  its fields.
* **A check that cannot run confirms nothing**: a total energy's unit beside its macronutrients is
  checked by the Atwater identity, the registry's test; where that test cannot run (it needs the
  nutrition pack), the documented unit is asked, never applied.
* **A code table of missing-value codes alone** (``77777=Refused; 99999=Don't know``) documents no
  categories: it settles nothing, and the values it does not list are the measurement's own.
* **A recorded answer stands**: a reading already answered otherwise (by the user, or by an
  earlier codebook) keeps its answer, and the record says so; its documented unit is then no unit
  a sentence states (``readings.codebook_unit``), whether the user answered before the import or
  after it.
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

def _contract() -> Any:
    """The codebook import's method contract (BLUEPRINT §13), in the one registry
    (``turbotab.core.contracts``)."""
    from turbotab.core.contracts import MethodContract, Option, Relation, register_contract

    both = ("prediction", "inference")
    return register_contract(MethodContract(
        key="import_codebook", label="Importing a codebook", slot="ingest", scope="descriptive",
        scope_note=("Each documented column's values, the outcome's among them, are read only to "
                    "find a contradiction; what settles a reading is the user's documentation, "
                    "never the values, and the outcome's documented unit informs only how its "
                    "sentences state it."),
        needs=("a codebook naming the table's columns: a variable table, an NHANES codebook page, "
               "or an XPT file's labels",),
        question="Do you have a data dictionary for this table?",
        place="after the table is read, before the roles (it answers what the card would ask)",
        decision="import_codebook",
        options=(
            Option("import", "Import it",
                   "Customary: the codebook is how a survey's variables are documented",
                   dict.fromkeys(both, "Sound for both purposes: it is the user's own "
                                       "documentation; its structured fields settle readings, its "
                                       "labels only guide the guesses."),
                   dict.fromkeys(both, "recommended")),
            Option("skip", "No codebook", "Customary for a table without one",
                   dict.fromkeys(both, "Sound: every reading is asked as before."),
                   dict.fromkeys(both, "available")),
        ),
        storyboard=("read the codebook's variables", "match them to the table's columns",
                    "check each structured field against the values",
                    "settle the fields the values do not contradict; ask the rest"),
        sentence="turbotab.core.codebook:codebook_sentence",
        relations=(
            Relation("implies", "structured_field_settles",
                     "settles, recorded as the codebook's: the readings it settles leave the ask "
                     "card", condition="a structured field (a unit, a code table, a type)",
                     id="structured field", enforced_by="turbotab.core.codebook:assess"),
            Relation("implies", "labels_lead_the_guesses",
                     "strengthens the guess only (names never settle): its labels lead the "
                     "remaining guesses", condition="a free-text label", id="free-text label",
                     enforced_by="turbotab.core.readings:labeled"),
            Relation("conflicts", "contradicted_by_the_values", "asked, never applied",
                     condition="a field the values contradict (at any number of distinct values; a "
                               "value within four units in the last place of a documented bound "
                               "is the bound)", id="contradicted by the values",
                     rung="refused",
                     exits=("confirm what the codebook documents", "confirm what the values say"),
                     enforced_by="turbotab.core.codebook:assess"),
            Relation("conflicts", "unchecked", "asked, never applied: a check that cannot run "
                                               "confirms nothing",
                     condition="a field whose check cannot run here (a total energy's unit beside "
                               "its macronutrients, without the Atwater identity)",
                     id="its check cannot run", rung="refused",
                     exits=("confirm what the codebook documents", "confirm the other unit"),
                     enforced_by="turbotab.core.codebook:assess"),
            Relation("conflicts", "answered_otherwise", "the user's answer stands",
                     condition="a reading the user answered otherwise, before the import or after "
                               "it", id="answered otherwise by the user", rung="refused",
                     exits=("answer the reading again",),
                     enforced_by="turbotab.core.codebook:assess"),
            Relation("enables", "documented_outcome_unit",
                     "An outcome's documented unit is stated in its sentences, unless the user "
                     "answered its unit otherwise.",
                     enforced_by="turbotab.core.readings:codebook_unit"),
        ),
        sources=("BLUEPRINT §14.2 (let the codebook answer)", "V2 definition of done §1")))


CONTRACT = _contract()

TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".xlsx", ".xls")
HTML_SUFFIXES = (".htm", ".html")
XPT_SUFFIXES = (".xpt", ".xport")
SUPPORTED = "a variable table (CSV, TSV or Excel), an NHANES codebook page (.htm), or an XPT file"
# A value this many units in the last place from a documented bound, in the column's own
# floating-point type, is the bound as its storage rounds it. Measured on NHANES 2015-2016's
# DR1TOT_I (9,544 rows, 168 variables) against its codebook page: twelve maxima exceed their printed
# bound by exactly one unit (223.75900000000001 against 223.759: the IBM float read as an IEEE
# double), and DEMO_I's and BMX_I's pages hold every value exactly.
BOUND_ULPS = 4

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
    """What the check reads of one documented column: its values, never the codebook's."""

    dtype: str
    physical: str
    present: int   # rows holding a value
    every: Any     # a numeric or yes/no column's values as numbers (numpy, inf kept), else None
    numbers: Any   # a numeric column's finite numbers (numpy), else None


def _sql_text(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sql_double(x: float) -> str:
    return f"CAST({_sql_text(repr(float(x)))} AS DOUBLE)"


def _code_number(code: str) -> float | None:
    """A code's number (a code key as :func:`code_key` writes it), or None for a word."""
    try:
        return float(code)
    except ValueError:
        return None


def _is_float32(physical: str) -> bool:
    return physical.upper().split("(")[0].strip() in ("FLOAT", "FLOAT4", "REAL")


def _slack(bound: float, physical: str) -> float:
    """:data:`BOUND_ULPS` units in the last place of ``bound`` in the column's own floating-point
    type: how far a value may sit past a documented bound and still be the bound."""
    import numpy as np

    b = abs(float(bound))
    ulp = float(np.spacing(np.float32(b))) if _is_float32(physical) else float(np.spacing(b))
    return BOUND_ULPS * ulp


def _within(x: Any, rng: tuple[float, float], physical: str) -> Any:
    lo, hi = rng
    return (x >= lo - _slack(lo, physical)) & (x <= hi + _slack(hi, physical))


# The whitespace ``str.strip`` removes from a code that ``trim`` would leave (tabs, line breaks).
_WHITESPACE = "' ' || chr(9) || chr(10) || chr(11) || chr(12) || chr(13)"


class _Values:
    """The table's values, read for the check over its Parquet file (DuckDB): each column's facts
    and, for a code table, the distinct values it does not list, counted exactly at any number of
    distinct values (a column with 5,413 distinct step counts is checked as one with 30)."""

    def __init__(self, parquet: Path, info: Mapping[str, Any]):
        import duckdb

        self.rel = f"read_parquet({_sql_text(str(parquet))})"
        self.columns = {str(c["name"]): c for c in info.get("columns") or []}
        self.con = duckdb.connect()

    def close(self) -> None:
        self.con.close()

    def _numeric(self, column: str) -> bool:
        return str(self.columns[column]["dtype"]) in ("numeric", "integer", "boolean")

    def facts(self, column: str) -> _Facts:
        import numpy as np

        meta = self.columns[column]
        ident = _sql_ident(column)
        present = int(self.con.execute(f"SELECT count({ident}) FROM {self.rel}").fetchone()[0])
        every = numbers = None
        if self._numeric(column):
            arr = self.con.execute(f"SELECT CAST({ident} AS DOUBLE) FROM {self.rel} "
                                   f"WHERE {ident} IS NOT NULL").fetchnumpy()
            every = np.asarray(list(arr.values())[0], dtype=float)
            every = every[~np.isnan(every)]
            if str(meta["dtype"]) in ("numeric", "integer"):
                numbers = every[np.isfinite(every)]
        return _Facts(str(meta["dtype"]), str(meta.get("physical_type") or ""), present, every,
                      numbers)

    def first(self, column: str) -> str:
        """The column's smallest value as text (for words)."""
        row = self.con.execute(f"SELECT min(CAST({_sql_ident(column)} AS VARCHAR)) "
                               f"FROM {self.rel}").fetchone()
        return code_key(row[0]) if row and row[0] is not None else ""

    def outside(self, column: str, facts: _Facts, codes: Mapping[str, str],
                rng: tuple[float, float] | None) -> tuple[int, list[str]]:
        """``(how many, the first four)`` of the column's distinct values (as :func:`code_key`
        reads them) that the code table does not list and its range, if any, does not hold."""
        import numpy as np

        if facts.every is not None:
            distinct = np.unique(facts.every)
            listed = np.array(sorted({x for x in map(_code_number, codes) if x is not None}),
                              dtype=float)
            mask = ~np.isin(distinct, listed)
            if rng is not None:
                mask &= ~_within(distinct, rng, facts.physical)
            out = distinct[mask]
            return int(len(out)), [code_key(float(v)) for v in out[:4]]
        # Text: a value is listed when its number is a code's number, or its words a code's words
        # (``code_key``: "1.0", " 1" and 1 are one code), or its number lies in the range.
        text = f"trim(CAST({_sql_ident(column)} AS VARCHAR), {_WHITESPACE})"
        number = f"TRY_CAST({text} AS DOUBLE)"
        is_number = f"({number} IS NOT NULL AND isfinite({number}))"
        numbers = sorted({x for x in map(_code_number, codes) if x is not None and math.isfinite(x)})
        words = sorted(k for k in codes if k not in {code_key(x) for x in numbers})
        listed = []
        if numbers:
            listed.append(f"({is_number} AND {number} IN "
                          f"({', '.join(_sql_double(x) for x in numbers)}))")
        if words:
            listed.append(f"(NOT {is_number} AND {text} IN "
                          f"({', '.join(_sql_text(w) for w in words)}))")
        if rng is not None:
            lo, hi = rng
            listed.append(f"({is_number} AND {number} BETWEEN "
                          f"{_sql_double(lo - _slack(lo, 'DOUBLE'))} AND "
                          f"{_sql_double(hi + _slack(hi, 'DOUBLE'))})")
        where = (f"{_sql_ident(column)} IS NOT NULL AND NOT ({' OR '.join(listed) or 'FALSE'})")
        key = f"CASE WHEN {is_number} THEN 'n' || CAST({number} AS VARCHAR) ELSE 't' || {text} END"
        n = int(self.con.execute(f"SELECT count(DISTINCT {key}) FROM {self.rel} "
                                 f"WHERE {where}").fetchone()[0])
        rows = self.con.execute(f"SELECT DISTINCT {text} FROM {self.rel} WHERE {where} "
                                f"ORDER BY 1 LIMIT 64").fetchall()
        return n, sorted(dict.fromkeys(code_key(r[0]) for r in rows))[:4]


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


def _exact(x: float) -> str:
    """A bound or a value as written, every digit kept (the shortest form that reads back as the
    same number): a range quoted to four digits would misquote the codebook (223.759 is no
    223.8) and hide how far a value lies past it."""
    x = float(x)
    return f"{int(x):,}" if x.is_integer() and abs(x) < 1e15 else format(x, ",")


def _listed(values: Sequence[str], limit: int = 4) -> str:
    shown = ", ".join(f"`{v}`" for v in values[:limit])
    more = len(values) - limit
    return shown + (f" and {more:,} more" if more > 0 else "")


def _counted(shown: Sequence[str], n: int) -> str:
    """The first values of ``n`` (``shown``), then how many more there are."""
    listed = ", ".join(f"`{v}`" for v in shown)
    more = n - len(shown)
    return listed + (f" and {more:,} more" if more > 0 else "")


# Human ranges a body measure's median falls in, per unit (a contradiction needs the magnitudes to
# reject the documented unit: a kg weight's median is never 70,000, an age in years never 400).
_MEDIAN_BOUNDS = {"kg": (0.5, 400.0), "lb": (1.0, 900.0), "years": (0.0, 120.0),
                  "pct_energy": (0.0, 100.0)}


def _atwater_runs() -> bool:
    """Whether the registry's test of total energy's unit (the Atwater identity, ``readings.
    KIND_RULES["unit:energy"]``) runs in this installation: it settles kcal on a table built to
    satisfy ``E = 4P + 4C + 9F`` exactly. It reads the nutrition pack, which an installation
    without Classic's ``ml`` package lacks (``methods.energy``); there it settles nothing, and its
    silence must not pass for agreement."""
    import numpy as np
    import pandas as pd

    from turbotab.core import readings

    p, c, f = np.arange(20.0, 44.0, 2.0), np.arange(150.0, 270.0, 10.0), np.arange(40.0, 64.0, 2.0)
    probe = pd.DataFrame({"protein_g": p, "carbohydrate_g": c, "fat_g": f,
                          "energy_kcal": 4 * p + 4 * c + 9 * f})
    try:
        verdict = readings.by_values("unit:energy", probe, "energy_kcal")
    except Exception:  # noqa: BLE001 - a test that raises does not run
        return False
    return bool(verdict.settles and verdict.value == "kcal")


UNCHECKED_ENERGY = ("the Atwater identity, which tests a total energy's unit against its "
                    "macronutrients, cannot run in this installation")


def _unit_contradiction(column: str, unit: str, facts: _Facts, label: str, store: Any
                        ) -> dict[str, Any] | None:
    """What the values say against a documented unit (``values``), the units they leave
    (``units``), and whether a check ran at all (``checked``: False when the one that applies
    cannot run here, which confirms nothing); None when nothing contradicts it."""
    import numpy as np

    from turbotab.core import readings
    from turbotab.core.recognizers import tokens

    def against(values: str, units: Sequence[str] = (), checked: bool = True) -> dict[str, Any]:
        return {"values": values, "units": list(units), "checked": checked}

    if facts.dtype not in ("numeric", "integer"):
        return against("text values, which have no unit")
    x = facts.numbers
    if x is None or not len(x):
        return None
    positive = x[x > 0]
    median = float(np.median(positive)) if len(positive) else None
    words = set(tokens(column)) | set(tokens(label))
    if unit in ("cm", "m", "in") and words & {"height", "stature", "bmxht", "length"}:
        verdict = readings.by_values("unit:height", positive)
        if verdict.settles and verdict.value != unit:
            return against(f"a median of {_fmt(median or 0)}, a human height only in "
                           f"{verdict.value}", [str(verdict.value)])
        if unit == "in" and verdict.settles:
            return against(f"a median of {_fmt(median or 0)}, which no one is in inches",
                           [str(verdict.value)])
    if unit in ("kcal", "kj") and store is not None:
        other = sorted({"kcal", "kj"} - {unit})
        verdict = None
        try:
            from turbotab.core.decisions import _names_a_macro_total

            macros = [c for c in store.columns if c != column and _names_a_macro_total(c)]
            if macros:
                verdict = readings.by_values("unit:energy", store.materialize([column, *macros]),
                                             column)
                ran = verdict.detail is not None or _atwater_runs()
            else:
                ran = True  # no macronutrient beside it: nothing to check, as for mg/dL
        except Exception:  # noqa: BLE001 - a check that raises did not run
            ran = False
        if not ran:  # both units offered, in the order a contradicted unit offers them
            return against(UNCHECKED_ENERGY, other, checked=False)
        if verdict is not None:
            fits = {u for u, _d in verdict.candidates}
            if verdict.settles and verdict.value != unit:
                return against(f"values that match the energy their macronutrients carry in "
                               f"{verdict.value}", [str(verdict.value)])
            if fits and unit not in fits:
                return against(f"values {verdict.evidence.removeprefix('is ')}", sorted(fits))
    bounds = _MEDIAN_BOUNDS.get(unit)
    if bounds is not None:
        lo, hi = bounds
        values = positive if unit != "pct_energy" else x
        if len(values):
            if unit == "pct_energy":
                if float(values.min()) < lo or float(values.max()) > hi:
                    return against(f"values from {_fmt(float(values.min()))} to "
                                   f"{_fmt(float(values.max()))}, outside 0–100")
            elif median is not None and not lo <= median <= hi:
                return against(f"a median of {_fmt(median)}, outside any human value in {unit}")
    return None


def assess(codebook: Codebook, parquet: Path, info: Mapping[str, Any], state: Any = None, *,
           store: Any = None) -> Assessment:
    """Check ``codebook`` against the table at ``parquet`` (``info``: its DatasetInfo dict) and the
    answers in ``state``: what it settles, what it asks, what it leaves to the user's answers."""
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
    reader = _Values(parquet, info)
    try:
        for e in codebook.entries:
            column = matched.get(e.variable)
            if column is None:
                continue
            if e.label:
                a.labels[column] = e.label
            if codebook.form == "xpt" or not (e.codes or e.unit or e.type or e.range):
                continue  # labels only
            _check_entry(a, e, column, reader.facts(column), reader, state, target, store)
    finally:
        reader.close()
    return a


def _check_entry(a: Assessment, e: Entry, column: str, f: _Facts, reader: _Values, state: Any,
                 target: str | None, store: Any) -> None:
    """One matched entry's structured fields against its column's values: each contradicted
    field asked (and the entry then settles nothing), else each field's reading proposed, written
    where no answer of the user's stands."""
    from turbotab.core import readings

    proposed: list[tuple[str, str, str]] = []   # (reading, value, field)
    conflicts: list[dict[str, Any]] = []
    numeric = f.dtype in ("numeric", "integer")
    codes = {code_key(c): d for c, d in e.codes.items() if code_key(c)}
    # The table's categories: its codes less those for a missing answer (Refused, Don't know).
    # A table of missing-value codes alone documents no category, and the values it does not list
    # are the measurement's own.
    categories = {c: d for c, d in codes.items() if not SENTINEL.search(d)}
    if categories and f.present:
        # Codes the data do not hold: a value the code table does not list (nor its range),
        # counted over every distinct value. A listed code no row holds is no contradiction (a
        # subset; NHANES lists codes whose count is 0).
        n, shown = reader.outside(column, f, codes, e.range)
        if n:
            conflicts.append({"field": "codes", "says": f"the codes {_listed(list(codes))}",
                              "values": f"values it does not list: {_counted(shown, n)}"})
    elif e.range is not None and f.every is not None and len(f.every):
        import numpy as np

        listed = np.array(sorted({x for x in map(_code_number, codes) if x is not None}),
                          dtype=float)
        x = f.every
        out = x[~_within(x, e.range, f.physical) & ~np.isin(x, listed)]
        if len(out):
            lo, hi = e.range
            low, high = float(out.min()), float(out.max())
            conflicts.append({"field": "range", "says": f"values from {_exact(lo)} to {_exact(hi)}",
                              "values": (f"`{len(out):,}` {'value' if len(out) == 1 else 'values'} "
                                         f"outside it, " + (f"`{_exact(low)}`" if low == high else
                                                            f"from `{_exact(low)}` to "
                                                            f"`{_exact(high)}`"))})
    kind_of_type = type_reading(e.type)
    if categories and e.range is None and not _restates_numbers(codes):
        if kind_of_type == "amount":
            conflicts.append({"field": "type", "says": f"the type {e.type!r} beside a code "
                                                       "table", "values": "a code table"})
        elif numeric:
            proposed.append(("code_or_count", "code", "codes"))
        coding = _sex_coding(codes)
        if coding is not None:
            proposed.append(("sex_coding", coding, "codes"))
    elif kind_of_type is not None and not categories:
        if not numeric and kind_of_type == "amount" and f.present:
            conflicts.append({"field": "type", "says": f"the type {e.type!r}",
                              "values": f"text, such as `{reader.first(column)}`"})
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
            conflicts.append({"field": "unit", "says": documented, **against})
        elif unit is not None and numeric:
            proposed.append(("unit", unit, "unit"))
    if conflicts:
        for c in conflicts:
            units_left, checked = c.pop("units", []), c.pop("checked", True)
            a.asked.append({"column": column, **c, "checked": checked,
                            "exits": _conflict_exits(column, c["field"], e, units_left)})
        return
    unit_kept = False
    for reading_kind, value, source in proposed:
        recorded = readings.confirmation(state, reading_kind, column) if state is not None \
            else None
        k = readings.key(reading_kind, column)
        if recorded is not None and str(recorded) != str(value):
            a.kept.append(k)   # the recorded answer stands, the outcome's as any other's
            unit_kept = unit_kept or reading_kind == "unit"
            continue
        if column == target:
            continue  # the outcome's kind and unit have their own questions
        if recorded is not None and readings.codebook_source(state, reading_kind, column) is None:
            a.already.append(k)   # the user's own answer, the same: nothing to write
            continue
        # Unrecorded, or recorded by a codebook (this one, imported again after a join): written,
        # so the latest import names it as its evidence.
        a.items.append({"reading": reading_kind, "column": column, "value": value,
                        "field": source})
    if documented and not unit_kept:
        # Not contradicted, and no answer of the user's says otherwise: a sentence may state it.
        a.units[column] = documented


def _conflict_exits(column: str, field_name: str, e: Entry, units_left: Sequence[str]
                    ) -> list[dict[str, Any]]:
    """The answers a contradicted (or unchecked) field offers: the values' reading and the
    codebook's, one confirmation each (the user decides)."""
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


def project_table(project_dir: Path, state: Any) -> Path:
    """The project's table as it was read with the files ``state`` joins to it
    (``datastore.table_file``)."""
    from turbotab.core.datastore import table_file

    return table_file(Path(project_dir) / "data", getattr(state, "joins", None))


def assessment_for(codebook: Codebook, project_dir: Path, state: Any) -> Assessment:
    """:func:`assess` against the project's table as it was read, with the files ``state`` joins
    to it (:func:`project_table`)."""
    from turbotab.core.datastore import DataStore, _sidecar_path

    table = project_table(project_dir, state)
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
    if status not in (None, "fresh") or not project_table(pdir, _ctx(ctx, "state")).is_file():
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
                                   values=x["values"], checked=x["checked"]) for x in a.asked],
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
    contradicted = [x for x in asked if x.get("checked", True)]
    unchecked = [x for x in asked if not x.get("checked", True)]
    if contradicted:
        shown = "; ".join(f"`{x['column']}`'s {_FIELD_WORDS.get(str(x['field']), x['field'])} (it "
                          f"says {x['says']}; the values show {x['values']})"
                          for x in contradicted[:3])
        more = len(contradicted) - 3
        sentences.append(f"{_plural(len(contradicted), 'of its fields contradicts', 'of its fields contradict')} "
                         f"the values and {'was' if len(contradicted) == 1 else 'were'} asked "
                         f"instead of applied: {shown}" + (f"; and `{more:,}` more" if more > 0 else ""))
    if unchecked:
        shown = "; ".join(f"`{x['column']}`'s {_FIELD_WORDS.get(str(x['field']), x['field'])} (it "
                          f"says {x['says']}; {x['values']})" for x in unchecked[:3])
        more = len(unchecked) - 3
        sentences.append(f"`{len(unchecked):,}` of its fields could not be checked against the "
                         f"values and {'was' if len(unchecked) == 1 else 'were'} asked instead of "
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
