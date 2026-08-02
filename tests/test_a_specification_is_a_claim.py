"""`MISC-015` / `MISC-016` — two gates, because a specification is a claim.

Both findings are the same shape at two addresses: **a document says the app
will do something, and nothing checks that the promise is tracked.** The
failure is not that the work is undone — most of it is legitimately undone —
it is that undone work with no row is indistinguishable from work nobody ever
specified. `README.md`'s standing complaint, *the safety net is thinner than
the coverage number*, applied to the plan rather than the code.

**`MISC-015`** — four of `DOMAIN_SCIENCE.md`'s seven primitives had zero ledger
rows. They were tracked in a prose line inside an ASCII diagram in
`ROADMAP.md`, which is not tracking.

**`MISC-016`** — `FEATURE_REGISTER.md` had no rows for `pages/08` (568 lines)
or `pages/09` (1,060 lines), and `register.py check` validated row schema only.
Measured here, `pages/11` (5,162 lines, the largest file in the tree) was
missing too — which is the point: nothing was counting, so nobody knew how many
were missing.

## Why these are tests and not a tool subcommand

`register.py check` and `evidence.py check` exist and neither caught these,
because both check that the rows present are well-formed. **Coverage is a
different question from validity**, and it is asked against the world outside
the file — a page that exists, a primitive that was specified. That belongs
where the rest of the repository's *did the promise reach anything* checks
live.

## The exemption rule

Neither gate can be satisfied by deleting the specification, and neither
demands the work be done. What each demands is that the item be **tracked or
exempted in writing**, where an exemption names a reason someone can argue
with. `EXEMPT` below is the list of arguable claims; it is short on purpose,
and every entry is a sentence rather than a shrug.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "turbotab"
FINDINGS = json.loads((DOCS / "data" / "findings.json").read_text())
REGISTER = json.loads((DOCS / "data" / "register.json").read_text())
SCIENCE = (DOCS / "DOMAIN_SCIENCE.md").read_text()


#: Findings that REPORT the absence of rows. A row saying *these four have no
#: rows* is not a row for any of them, and counting it would let the gate be
#: satisfied by the complaint about the gate's own subject. The first version
#: of this file searched every field of every finding and passed on all seven
#: primitives while four genuinely had nothing — `MISC-015`'s own text was the
#: match.
META_ROWS = frozenset({"MISC-015", "MISC-016"})


def _rows_mentioning(terms) -> list:
    """Ledger rows whose ITEM is about one of `terms`.

    The `item` field is the finding's own statement of what is wrong. Searching
    `ev` and `note` too would match a primitive named in passing as context for
    something else — `sensitivity analysis` appears in `STATE-013`'s evidence,
    which is a row about a splitter — and *mentioned somewhere* is not
    *tracked*.

    Matched on words rather than on an id because the specification and the
    ledger were written by different hands and share no key. That is looser
    than an id would be, and the looseness is stated rather than hidden: this
    proves a primitive HAS a row, not that the row is good.
    """
    return [row for row in FINDINGS
            if row["id"] not in META_ROWS
            and any(t in (row.get("item") or "").lower() for t in terms)]


# ════════════════ `MISC-015` · EVERY PRIMITIVE HAS A ROW ════════════════

def _primitives() -> list:
    """§03's numbered list, parsed from the document rather than copied.

    A hand-copied list is the thing that goes stale — it would still say seven
    after an eighth primitive was specified, and the gate would pass by not
    looking. Parsed from the numbered items under the `**Primitives …**` line.
    """
    start = SCIENCE.index("**Primitives — new capability")
    end = SCIENCE.index("**Reference data — real work", start)
    block = SCIENCE[start:end]
    return [m.group(1).strip() for m in
            re.finditer(r"^\d+\.\s+(?:The\s+)?\*\*(.+?)\*\*", block, re.M)]


#: The words that identify each primitive in a ledger row. Written here because
#: the primitive's NAME in the document ("hard-stop class") and the vocabulary
#: the ledger uses for it ("hard stop") are not the same string, and matching on
#: the document's phrasing alone would report an absence that is not real.
#:
#: Keyed by the parsed name, so a primitive that is renamed or added in
#: `DOMAIN_SCIENCE.md` fails `test_the_primitive_inventory_is_not_stale` rather
#: than silently dropping out of the coverage check.
PRIMITIVE_TERMS = {
    "evidence badge": ["evidence badge", "settled/convention/disputed",
                       "evidence_status"],
    "purpose question": ["purpose question", "prediction vs inference"],
    "hard-stop class": ["hard stop", "hard-stop"],
    "sensitivity fork": ["sensitivity fork"],
    "Figure tiering and companions": ["figure tiering",
                                      "exploratory vs confirmatory"],
    "checklist engine": ["checklist engine"],
    "generalized leakage detector": ["generalized leakage", "leakage detector"],
}


def test_the_primitive_inventory_is_not_stale():
    """The gate below reads a hand-written term list; this checks it still
    describes the document. A primitive added to §03 without a term here would
    otherwise be uncovered by a gate that reported full coverage."""
    parsed = _primitives()
    assert parsed, "no primitives parsed from DOMAIN_SCIENCE §03 — the list moved"
    assert len(parsed) == 7, (
        f"§03 now specifies {len(parsed)} primitives: {parsed}. Add the new "
        f"one to PRIMITIVE_TERMS so it is covered, and to the ledger.")
    missing = [p for p in parsed if p not in PRIMITIVE_TERMS]
    assert not missing, (
        f"{missing} are specified in §03 and have no search terms here, so "
        f"the coverage gate cannot see them")


@pytest.mark.parametrize("primitive", sorted(PRIMITIVE_TERMS))
def test_every_specified_primitive_has_a_ledger_row(primitive):
    """`MISC-015`. Four of seven had none.

    Tracked in `ROADMAP.md` as *'hard stops, sensitivity fork, checklist
    engine, generalized leakage detector remain'* — one prose line inside an
    ASCII diagram. A plan that lives only in a diagram is a plan nobody can
    query, count, or close.
    """
    terms = PRIMITIVE_TERMS[primitive]
    rows = _rows_mentioning(terms)
    assert rows, (
        f"'{primitive}' is specified in DOMAIN_SCIENCE §03 and no ledger row "
        f"is about it (searched item fields for {terms}, excluding "
        f"{sorted(META_ROWS)}). Unbuilt is fine; UNTRACKED is the finding — "
        f"file a row saying what it is and that it is not built.")


# ════════════════ `MISC-016` · EVERY PAGE HAS A ROW ════════════════

#: `pages/NN_*.py` → the string the register uses for it. The register names
#: Classic surfaces as `Step N`, which is the app's own vocabulary; this maps
#: the file to it rather than inventing a third naming scheme.
def _pages() -> list:
    return sorted(p for p in (ROOT / "pages").glob("*.py")
                  if not p.name.startswith("__"))


#: Pages tracked by something other than a register row, with the reason. Empty
#: today — kept because the rule is *tracked or exempted in writing*, and a
#: mechanism that exists only once it is needed is a mechanism nobody trusts.
EXEMPT: dict = {}


@pytest.mark.parametrize("page", [p.name for p in _pages()])
def test_every_classic_page_has_a_register_row_or_a_written_exemption(page):
    """`MISC-016`. Three pages had neither, totalling 6,790 lines.

    The register's whole premise (`FEATURE_PARITY.md`) is that a capability
    present in one surface and absent from the other is **a claim to be
    justified, never a shrug** — and a capability with no row makes no claim at
    all. `register.py check` validated row schema and could not see this,
    because a row that does not exist has nothing wrong with it.
    """
    if page in EXEMPT:
        assert EXEMPT[page].strip(), f"{page} is exempt with an empty reason"
        return

    number = re.match(r"0*(\d+)_", page)
    assert number, (
        f"{page} is not numbered, so it cannot be matched to a register row. "
        f"Rename it or add it to EXEMPT with a reason.")
    step = int(number.group(1))
    # TWO NAMING SCHEMES, BOTH REAL. Older rows say `Step 5`; rows written once
    # the audit started citing evidence say `pages/05_Preprocess.py:114`. The
    # first version of this gate matched only the first form and reported four
    # covered pages as uncovered — a gate that is wrong in the direction of
    # more work is still wrong, and it would have been "fixed" by adding
    # duplicate rows for capabilities that already had them.
    #
    # The digit lookahead rather than `\b` is deliberate: `_` is a word
    # character, so `pages/01\b` does not match `pages/01_Upload_and_Audit.py`
    # and the second version of this gate was wrong for that reason instead.
    pattern = re.compile(
        rf"\bstep\s*0*{step}\b|\bpages/0*{step}(?![0-9])", re.I)

    rows = [r for r in REGISTER if pattern.search(str(r.get("classic") or ""))]
    assert rows, (
        f"{page} ({len(( ROOT / 'pages' / page).read_text().splitlines()):,} "
        f"lines) has no register row naming `Step {step}`. Add rows for what "
        f"it does — `classic-only` with a reason is a complete answer, and "
        f"'not ported because X' is the answer the register exists to hold.")


def test_the_register_tool_still_passes_over_the_rows_this_gate_forces():
    """Coverage must not be bought by adding rows the register would refuse.

    This deliberately DEFERS to `register.py check` rather than restating its
    rules. The first version asserted a minimum reason length, which is a
    threshold nobody agreed to: it failed `'Frozen (§05).'` and
    `'Belongs with the Excel path above.'`, both of which are real answers that
    point somewhere. Two rules about the same thing are two rules that will
    disagree, and the tool's is the one the repository already lives by.
    """
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, str(DOCS / "tools" / "register.py"), "check"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert done.returncode == 0, (
        f"register.py check fails, so the coverage gate above is being "
        f"satisfied with rows the register itself rejects:\n{done.stderr}")


def test_the_gate_would_notice_a_new_page():
    """The positive control.

    An absence assertion gets easier to satisfy as the thing it guards
    disappears (`GUIDED-045`), and this one is parametrized over the files that
    exist — so an empty `pages/` would produce zero tests and a green run. This
    asserts the parametrization is not empty and covers what is there.
    """
    pages = _pages()
    assert len(pages) >= 11, (
        f"only {len(pages)} pages found; the coverage gate above would be "
        f"nearly vacuous")
    numbered = [p for p in pages if re.match(r"\d", p.name)]
    assert len(numbered) == len(pages), (
        f"unnumbered pages present: {[p.name for p in pages if p not in numbered]}")
