"""Minimal multi-file assembly: joining files on a shared identifier (V2 definition of done §1).

NHANES ships each component as its own file (demographics, the day-1 dietary totals, body
measures …), joined on ``SEQN``; without a join most NHANES users cannot start. The minimal form is
in v2: one shared identifier, a preview of the row counts before anything is committed, and the
joins that keep every row's meaning:

* **one-to-one**: each identifier value names one row in the table and one in the file;
* **one-to-many**: one row in the table, several in the file (a person's recalls): each of the
  table's rows is repeated once per partner, and the table becomes one row per partner;
* **many-to-one**: several rows in the table, one in the file (a person's characteristics beside
  each recall): each row gains its partner's columns, and no row is added.

**Many-to-many is refused** (BLUEPRINT §11.3, the leash's refuse rung): when the identifier repeats
on both sides, a join pairs every row of a value with every row of it in the other file, a product
that answers no question a researcher asks of a shared identifier. The reason is stated with an
example and the way forward (a column that names one row on one side; combining the file's rows to
one per identifier first, which is deep assembly, planned for v2.x).

Rows the identifier does not match are counted on each side before anything is committed. A blank
identifier never matches (a missing identifier is no shared one; pandas would pair blanks with
blanks). ``left`` keeps every row of the table, the file's columns blank where it has no partner;
``inner`` keeps only the rows with one. A file column whose name the table already has is renamed
``<name>_<file stem>``, and the preview says so.

The method contract (BLUEPRINT §13) is :data:`CONTRACT`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

ROW_ID = "__row_id"

def _contract() -> Any:
    """The join's method contract (BLUEPRINT §13), in the one registry (``turbotab.core.contracts``)."""
    from turbotab.core.contracts import MethodContract, Option, Relation, register_contract

    both = ("prediction", "inference")
    return register_contract(MethodContract(
        key="join_files", label="Joining files on a shared identifier", slot="ingest",
        scope="row_local",
        scope_note=("A joined row's values are its own and its partner's, matched by the identifier "
                    "alone; no other row and no outcome is read."),
        needs=("an added file, ingested", "a column in both that identifies the unit"),
        question="Which file joins this table, and on which identifier?",
        place="before the opening sequence (the ingest stage rebuilds the table)",
        decision="join_files", stage="ingest",
        options=(
            Option("left", "Keep every row of the table",
                   "Customary for NHANES: the demographics file is the frame",
                   dict.fromkeys(both, "Sound for both purposes: no row of the table is lost, and "
                                       "a row with no partner holds blanks for the file's columns."),
                   dict.fromkeys(both, "recommended")),
            Option("inner", "Keep only the rows with a partner",
                   "Customary when the analysis needs both files' columns",
                   dict.fromkeys(both, "Sound when the analysis needs both files' columns and the "
                                       "rows it drops are stated."),
                   dict.fromkeys(both, "available")),
        ),
        storyboard=("count each side's rows and identifier values", "match the values",
                    "count the rows with no partner on each side",
                    "join, numbering the rows anew"),
        sentence="turbotab.core.assembly:join_sentence",
        relations=(
            Relation("implies", "row_counts_previewed",
                     "The row counts (one-to-one, one-to-many, unmatched on each side) are "
                     "previewed before the join is committed.",
                     enforced_by="turbotab.core.assembly:preview"),
            Relation("implies", "labels_join",
                     "The file's variable labels join the table's (the XPT codebook form).",
                     enforced_by="turbotab.core.assembly:file_meta"),
            Relation("invalidates", "stages_before_the_join",
                     "Every stage computed on the table before the join is recomputed: the ingest "
                     "stage reads the joins.",
                     enforced_by="turbotab.core.stages:build_graph"),
            Relation("conflicts", "many_to_many",
                     "Refused: an identifier that repeats in both files pairs every row of a value "
                     "with every row of it on the other side.",
                     condition="an identifier that repeats in both files", rung="refused",
                     exits=("join on a column that names one row per unit in one of the files",
                            "combine the file's rows to one per identifier first"),
                     enforced_by="turbotab.core.assembly:plan"),
            Relation("conflicts", "no_matching_identifier",
                     "Refused: no value of the identifier is in both files, so the join would add "
                     "only blanks.", rung="refused",
                     exits=("join on another column",), enforced_by="turbotab.core.assembly:plan"),
            Relation("conflicts", "identifier_types_differ",
                     "Refused: the identifier holds numbers in one file and text in the other, so "
                     "no value of one can equal a value of the other.", rung="refused",
                     exits=("join on a column both files write the same way",),
                     enforced_by="turbotab.core.assembly:plan"),
        ),
        sources=("V2 definition of done §1 (minimal multi-file assembly)",)))


CONTRACT = _contract()


@dataclass
class Side:
    name: str
    rows: int
    keys: int          # distinct non-blank identifier values
    blank_keys: int    # rows whose identifier is blank
    max_repeat: int    # the most rows one identifier value names (0: no value)
    example: Any = None  # an identifier value that repeats, and how often
    example_count: int = 0

    @property
    def repeats(self) -> bool:
        return self.max_repeat > 1

    def to_dict(self) -> dict[str, Any]:
        from turbotab.core.datastore import json_safe

        return {"name": self.name, "rows": self.rows, "keys": self.keys,
                "blank_keys": self.blank_keys, "repeats": self.repeats,
                "max_repeat": self.max_repeat, "example": json_safe(self.example),
                "example_count": self.example_count}


@dataclass
class JoinPlan:
    """A join, counted and ready to run: ``sql`` selects the joined table with fresh row ids;
    ``refusal`` is set (code, message, exits) when the join is refused, and ``sql`` is then
    empty."""

    on: str
    right_on: str
    how: str
    left: Side
    right: Side
    relation: str
    matched_keys: int
    left_unmatched: int
    right_unmatched: int
    result_rows: int
    added_columns: list[str] = field(default_factory=list)
    renamed: dict[str, str] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    sql: str = ""
    refusal: dict[str, Any] | None = None

    @property
    def note(self) -> str:
        return join_note(self)

    def counts(self) -> dict[str, Any]:
        return {"relation": self.relation, "table_rows": self.left.rows,
                "file_rows": self.right.rows, "matched_keys": self.matched_keys,
                "table_unmatched": self.left_unmatched, "file_unmatched": self.right_unmatched,
                "rows": self.result_rows, "added_columns": len(self.added_columns),
                "renamed": dict(self.renamed)}

    def to_dict(self) -> dict[str, Any]:
        return {"on": self.on, "right_on": self.right_on, "how": self.how,
                "table": self.left.to_dict(), "file": self.right.to_dict(),
                "relation": self.relation, "matched_keys": self.matched_keys,
                "table_unmatched": self.left_unmatched, "file_unmatched": self.right_unmatched,
                "rows": self.result_rows, "added_columns": list(self.added_columns),
                "renamed": dict(self.renamed), "refusal": self.refusal}


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _family(physical: str) -> str:
    """Which identifier values compare: numbers with numbers, text with text, dates with dates."""
    base = physical.upper().split("(")[0].strip()
    if base in ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
                "UINTEGER", "UBIGINT", "FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC"):
        return "number"
    if base.startswith(("DATE", "TIMESTAMP", "TIME")):
        return "time"
    if base == "BOOLEAN":
        return "boolean"
    return "text"


def _schema(con: Any, parquet: Path) -> list[tuple[str, str]]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet({_lit(parquet)})").fetchall()
    return [(str(r[0]), str(r[1])) for r in rows if r[0] != ROW_ID]


def _stem(name: str) -> str:
    stem = re.sub(r"\.(gz|zst)$", "", str(name), flags=re.I)
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", stem)
    return re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_") or "file"


def plan(con: Any, left: Path, right: Path, *, on: str, right_on: str | None = None,
         how: str = "left", left_name: str = "the table", right_name: str = "the file") -> JoinPlan:
    """Count the join of ``left`` (the table) with ``right`` (the file) on ``on`` (``right_on`` in
    the file) and plan it. Every count is a query over the two Parquet files."""
    right_on = right_on or on
    lschema, rschema = _schema(con, left), _schema(con, right)
    ltypes, rtypes = dict(lschema), dict(rschema)
    L, R = f"read_parquet({_lit(left)})", f"read_parquet({_lit(right)})"
    empty = Side(left_name, 0, 0, 0, 0)

    def refused(code: str, message: str, exits: list[dict[str, Any]],
                lside: Side = empty, rside: Side = empty) -> JoinPlan:
        return JoinPlan(on, right_on, how, lside, rside, "refused", 0, 0, 0, 0,
                        refusal={"code": code, "message": message, "exits": exits})

    choose = [{"label": "Choose the column that identifies each unit in both files",
               "decision": None}]
    if on not in ltypes:
        return refused("unknown_column", f"The table has no column named `{on}`.", choose)
    if right_on not in rtypes:
        return refused("unknown_column", f"{right_name} has no column named `{right_on}`.", choose)
    lf, rf = _family(ltypes[on]), _family(rtypes[right_on])
    if lf != rf:
        return refused("identifier_types_differ",
                       f"`{on}` holds {_family_words(lf)} in the table and {_family_words(rf)} in "
                       f"{right_name}, so no value of one can equal a value of the other.",
                       [{"label": "Join on a column both files write the same way",
                         "decision": None}])
    lk, rk = f"L.{_ident(on)}", f"R.{_ident(right_on)}"
    counts = con.execute(f"""
        WITH lc AS (SELECT {_ident(on)} AS k, count(*) AS n FROM {L}
                    WHERE {_ident(on)} IS NOT NULL GROUP BY 1),
             rc AS (SELECT {_ident(right_on)} AS k, count(*) AS n FROM {R}
                    WHERE {_ident(right_on)} IS NOT NULL GROUP BY 1),
             m AS (SELECT lc.n AS ln, rc.n AS rn FROM lc JOIN rc ON lc.k = rc.k)
        SELECT (SELECT count(*) FROM {L}), (SELECT count(*) FROM {R}),
               (SELECT count(*) FROM lc), (SELECT count(*) FROM rc),
               (SELECT coalesce(max(n), 0) FROM lc), (SELECT coalesce(max(n), 0) FROM rc),
               (SELECT count(*) FROM m), (SELECT coalesce(sum(ln * rn), 0) FROM m),
               (SELECT coalesce(sum(ln), 0) FROM m), (SELECT coalesce(sum(rn), 0) FROM m),
               (SELECT count(*) FROM {L} WHERE {_ident(on)} IS NULL),
               (SELECT count(*) FROM {R} WHERE {_ident(right_on)} IS NULL)
    """).fetchone()
    (lrows, rrows, lkeys, rkeys, lmax, rmax, matched, inner_rows, lmatched, rmatched,
     lblank, rblank) = (int(x or 0) for x in counts)

    def example(rel: str, key: str) -> tuple[Any, int]:
        row = con.execute(f"SELECT {_ident(key)}, count(*) AS n FROM {rel} WHERE {_ident(key)} "
                          f"IS NOT NULL GROUP BY 1 HAVING count(*) > 1 ORDER BY n DESC, 1 "
                          f"LIMIT 1").fetchone()
        return (row[0], int(row[1])) if row else (None, 0)

    lex, lexn = example(L, on) if lmax > 1 else (None, 0)
    rex, rexn = example(R, right_on) if rmax > 1 else (None, 0)
    lside = Side(left_name, lrows, lkeys, lblank, lmax, lex, lexn)
    rside = Side(right_name, rrows, rkeys, rblank, rmax, rex, rexn)
    if lside.repeats and rside.repeats:
        return refused(
            "many_to_many",
            f"`{on}` repeats in both: {_value(lex)} names {lexn:,} rows of the table and "
            f"{_value(rex)} {rexn:,} rows of {right_name}. A join on it pairs every row of a "
            f"value with every row of that value on the other side, so a row's partner would be "
            f"no single row, and every count after it would multiply. Join on a column that "
            f"names one row in one of the two files.",
            [{"label": "Join on a column that names one row per unit in one of the files",
              "decision": None},
             {"label": "Combine the file's rows to one per identifier first (deep assembly, "
                       "planned for v2.x)", "decision": None}],
            lside, rside)
    relation = ("one-to-one" if not lside.repeats and not rside.repeats else
                "one-to-many" if rside.repeats else "many-to-one")
    if matched == 0:
        return refused("no_matching_identifier",
                       f"No value of `{on}` in the table is a value of `{right_on}` in "
                       f"{right_name}, so the join would add only blanks.",
                       [{"label": "Join on another column, or check that both files are of the "
                                  "same people", "decision": None}], lside, rside)
    left_unmatched, right_unmatched = lrows - lmatched, rrows - rmatched
    result_rows = inner_rows + (left_unmatched if how == "left" else 0)
    names = {n for n, _ in lschema}
    stem = _stem(right_name)
    renamed: dict[str, str] = {}
    added: list[str] = []
    select = [f"L.{_ident(n)} AS {_ident(n)}" for n, _ in lschema]
    for n, _t in rschema:
        if n == right_on:
            continue
        out = n
        if out in names:
            out = f"{n}_{stem}"
            k = 2
            while out in names:
                out = f"{n}_{stem}_{k}"
                k += 1
            renamed[n] = out
        names.add(out)
        added.append(out)
        select.append(f"R.{_ident(n)} AS {_ident(out)}")
    join = "LEFT JOIN" if how == "left" else "JOIN"
    order = f"L.{ROW_ID}, R.{ROW_ID}"
    sql = (f"SELECT {', '.join(select)}, "
           f"CAST(row_number() OVER (ORDER BY {order}) - 1 AS BIGINT) AS {ROW_ID} "
           f"FROM {L} AS L {join} {R} AS R ON {lk} = {rk} ORDER BY {order}")
    return JoinPlan(on, right_on, how, lside, rside, relation, matched, left_unmatched,
                    right_unmatched, result_rows, added, renamed,
                    [*(n for n, _ in lschema), *added], sql)


def _family_words(family: str) -> str:
    return {"number": "numbers", "text": "text", "time": "dates or times",
            "boolean": "true/false values"}[family]


def _value(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"`{v}`"


def preview(left: Path, right: Path, *, on: str, right_on: str | None = None, how: str = "left",
            left_name: str = "the table", right_name: str = "the file") -> JoinPlan:
    """:func:`plan` on its own connection."""
    import duckdb

    con = duckdb.connect()
    try:
        return plan(con, Path(left), Path(right), on=on, right_on=right_on, how=how,
                    left_name=left_name, right_name=right_name)
    finally:
        con.close()


# ── words ────────────────────────────────────────────────────────────────────


def _n(x: int) -> str:
    return f"`{x:,}`"


def join_note(p: JoinPlan) -> str:
    """The ingest's note for a join it made (plain text, one line)."""
    keep = "kept every row of the table" if p.how == "left" else "kept only rows with a partner"
    return (f"joined \"{p.right.name}\" on \"{p.on}\" ({p.relation}; {keep}): "
            f"{p.left.rows - p.left_unmatched:,} of the table's {p.left.rows:,} rows and "
            f"{p.right.rows - p.right_unmatched:,} of the file's {p.right.rows:,} rows matched; "
            f"the joined table has {p.result_rows:,} rows")


def join_sentence(name: str, on: str, right_on: str | None, how: str,
                  counts: Mapping[str, Any] | None) -> str:
    """The methods sentence the join records (DATAIN acceptance 4)."""
    key = f"`{on}`" if not right_on or right_on == on else f"`{on}` (`{right_on}` in it)"
    if not counts:
        return (f"`{name}` was joined to the table on {key}, "
                + ("keeping every row of the table" if how == "left" else
                   "keeping only the rows found in both"))
    c = dict(counts)
    relation = str(c.get("relation"))
    table_rows, file_rows = int(c["table_rows"]), int(c["file_rows"])
    t_un, f_un = int(c["table_unmatched"]), int(c["file_unmatched"])
    rows, added = int(c["rows"]), int(c["added_columns"])
    keep = ("keeping every row of the table" if how == "left" else
            "keeping only the rows found in both")
    parts = [f"`{name}` ({_n(file_rows)} rows) was joined {relation} to the table "
             f"({_n(table_rows)} rows) on {key}, {keep}"]
    matched_t = table_rows - t_un
    parts.append(f"{_n(matched_t)} of the table's rows found a partner"
                 + (f" and {_n(t_un)} did not" + (f", their {_n(added)} new columns blank"
                                                   if how == "left" else ", and left the table")
                    if t_un else ""))
    parts.append(f"{_n(file_rows - f_un)} of its rows matched"
                 + (f" and {_n(f_un)} matched no row of the table and were left out" if f_un else ""))
    sentence = "; ".join(parts) + f". The joined table has {_n(rows)} rows"
    renamed = c.get("renamed") or {}
    if renamed:
        shown = ", ".join(f"`{a}` as `{b}`" for a, b in list(renamed.items())[:4])
        more = len(renamed) - 4
        sentence += f"; its columns named as the table's were renamed ({shown}" + \
                    (f" and {more} more" if more > 0 else "") + ")"
    return sentence


# ── the decision: validator, completion, sentence (registered on import) ────


def file_meta(project_dir: Path, file_id: str) -> dict[str, Any] | None:
    """An added file's metadata (``files/<id>/file.json``), or None."""
    if not re.fullmatch(r"f[0-9a-f]{10}", str(file_id or "")):
        return None
    try:
        with open(Path(project_dir) / "files" / file_id / "file.json", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def file_parquet(project_dir: Path, file_id: str) -> Path:
    return Path(project_dir) / "files" / file_id / "raw.parquet"


def _project_dir(ctx: Any) -> Path | None:
    raw = ctx.get("project_dir") if isinstance(ctx, Mapping) else getattr(ctx, "project_dir", None)
    return Path(raw) if raw else None


def plan_for(decision: Any, ctx: Any) -> JoinPlan:
    """The plan of ``decision`` against the project's table as it stands: refuses (raises
    ``Refusal``) when the file or the table cannot be read yet."""
    from turbotab.core.decisions import Refusal, Revert, _ctx

    pdir = _project_dir(ctx)
    if pdir is None:
        raise Refusal("not_yet", "A join needs the project's files, which this request cannot read.",
                      exits=[])
    meta = file_meta(pdir, decision.file)
    if meta is None or not file_parquet(pdir, decision.file).is_file():
        raise Refusal("unknown_file", f"This project has no added file `{decision.file}`.",
                      exits=[{"label": "Add the file to the project first", "decision": None}])
    state = _ctx(ctx, "state")
    joined = (getattr(state, "joins", None) or {}) if state is not None else {}
    if decision.file in joined:
        records = _ctx(ctx, "records")
        exits: list[dict[str, Any]] = []
        try:
            for record in reversed(records() if callable(records) else []):
                d = record.decision
                if getattr(d, "kind", None) == "join_files" and d.file == decision.file:
                    exits.append({"label": "Undo the earlier join first",
                                  "decision": Revert(decision_id=record.id)})
                    break
        except Exception:  # noqa: BLE001 - the exit is a convenience
            pass
        raise Refusal("already_joined", f"`{meta.get('name')}` is already joined to the table.",
                      exits=exits)
    status = _ctx(ctx, "ingest_status")
    table = pdir / "data" / "raw.parquet"
    if status not in (None, "fresh") or not table.is_file():
        raise Refusal("not_yet", "The table is still being read; join the file once it is ready.",
                      exits=[])
    return preview(table, file_parquet(pdir, decision.file), on=decision.on,
                   right_on=decision.right_on, how=decision.how, right_name=str(meta.get("name")))


def _join_is_one_to_one_or_one_to_many(decision: Any, ctx: Any) -> None:
    """Refuse a many-to-many join (and one the counts cannot run), with its reason and exits."""
    from turbotab.core.decisions import Refusal

    if _project_dir(ctx) is None:
        return  # a context that names no project (a unit test of the fold) checks nothing
    p = plan_for(decision, ctx)
    if p.refusal is not None:
        raise Refusal(p.refusal["code"], p.refusal["message"], exits=p.refusal["exits"])


def _join_records_its_counts(decision: Any, ctx: Any) -> Any:
    """The file's name and the preview's counts, recorded with the answer (the server's)."""
    from turbotab.core.decisions import JoinCounts

    if _project_dir(ctx) is None:
        return decision
    p = plan_for(decision, ctx)
    return decision.model_copy(update={"name": p.right.name, "counts": JoinCounts(**p.counts())})


def _register() -> None:
    from turbotab.core import voice
    from turbotab.core.decisions import register_completion, register_validator

    register_validator("join_files", _join_is_one_to_one_or_one_to_many)
    register_completion("join_files", _join_records_its_counts)

    @voice.register_sentence("join_files")
    def _join_files(d: Any, state: Any, ctx: Any) -> str:
        counts = d.counts.model_dump() if d.counts is not None else None
        return join_sentence(d.name or d.file, d.on, d.right_on, d.how, counts)


_register()

__all__ = ["CONTRACT", "JoinPlan", "Side", "file_meta", "file_parquet", "join_note",
           "join_sentence", "plan", "plan_for", "preview"]
