"""`GUIDED-195` — the silent `slice(0, N)`, swept and GATED.

`GUIDED-164` found two of them and fixed both by hand. That is the shape
`GUIDED-197` warns about: **a sweep that reports rather than fails is a
document**, and a document does not see the seventh site somebody adds next
loop. So this file is not a second sweep. It is the rule with an enumeration
underneath it:

> **Every `slice(` in `turbotab/web/index.html` either states its bound to the
> user, or is listed here by name with a reason.**

Three dispositions and no fourth — the pattern
`test_every_field_the_server_composes_has_a_reader` publishes as
`DECLARED`/`FILED` and `turbotab/devchecks.py` publishes as `UNCLASSIFIED`:

* **`STATES`** — a numeric cap whose render path emits both counts;
* **`NOT A CAP`** — listed in `NOT_A_CAP` below, by function and by statement,
  each with the reason it cuts nothing a reader could be misled about;
* **unclassified** — the gate fails, and names the site.

## Why the enumeration is derived and not written down

A hand list is how the site added next loop goes unmeasured: it passes for as
long as nobody edits the list, which is the same green-over-nothing failure
`GUIDED-119` is the model for. So the sites are read out of the page every run,
and `NOT_A_CAP` is keyed by `(function, statement)` — **add a `slice(`, and it
is unclassified until somebody disposes of it; edit an exempted line, and its
exemption stops matching.** Loud beats silent in both directions.

## WHICH HALF EACH CLAIM RESTS ON

This file is deliberately two instruments, and conflating them is trap #5 —
*a grep answers "does this text appear"; the question is "does this run."*

**THE FILE HALF** — the enumeration, the disposition of each site, and the
`NOT_A_CAP` table. These are genuinely claims about the file: *how many
`slice(` are in it* has no behavioral form. The disposition predicate is
`capSaid(` **or** a hand-written `data-…-showing`/`data-…-of` pair, appearing
inside the same named function. Both forms are live and neither is legacy:
`GUIDED-164`'s two sites predate the helper, and `featPreviewHTML` writes the
markup by hand because
`test_a_two_column_formula_previews_both_of_its_columns.py` lifts that function
out of the page and runs it in node beside a hand-listed `esc` and `num` — a
call to `capSaid` is a `ReferenceError` there. It went red exactly that way
before this predicate was widened, and the widening is the honest resolution:
the alternative was editing a test so the page could share a helper. That
is a claim that the site is WIRED to a statement, and it is exactly as strong
as that — it cannot tell you the numbers are right, and it would not catch a
`capSaid` call in a branch that never runs.

**THE DRIVEN HALF** — `test_the_permutation_ranking_table_says_how_many_of_how_
many` runs the real controller over a real `/explain` payload from a really
fitted model, on two target shapes, and reads both counts back out of the DOM.
That is where *the numbers are right* is established, and it is done for the
site that matters most rather than for all six. **Four of the six caps are
covered by the file half only, and that is stated rather than implied.**

## The six caps, and the one that made this a `high`-value row rather than a tidy-up

| function | cap | what it truncates |
|---|---|---|
| `findingCard` | 5 | column chips — `GUIDED-164`, L48 |
| `attachSkewPlots` | 3 | distribution plots — `GUIDED-164`, L48 |
| **`explainHTML`** | **15** | **the permutation-importance ranking** |
| `featPreviewHTML` | 5 | before/after preview rows |
| `planRow` | 12 | the selected feature names |
| `teachHTML` | 6 | the columns that vary within a subject |

`explainHTML` is the one this row exists for. The other five truncate an
illustration; that one truncates a **ranking**, under a heading that names the
method, the model, the held-out row count and the number of shuffles, and above
a methods sentence. A reader who reaches its last row has been told by silence
that fifteen columns is what the ranking found. On `wide_assay.csv` the ranking
is forty-five columns long. That is a methods-section claim, and it was false.

## What the same lens finds ONE SURFACE OVER, said here because a sweep that
## stops where the sweeper's attention stopped has not reported its edge

The page is not where most of these live, and it is not close. Looking past
`slice(` for `.head(` and `[:N]`, an AST walk over every non-test module in
`turbotab/` and `ml/` finds **43 payload keys whose value is cut to a literal
bound before it is served**. Read by hand, about nineteen of those cut a
COLLECTION — the class this file is about — and the rest truncate a string or a
diagnostic, which is a different question. The sharpest of the nineteen:

* `turbotab/teaching.py:243` `varying_columns: varying[:6]` and `:453`
  `varying_columns = differ[:12]` — the SAME surface `teachHTML` caps at 6, so
  that one row is cut three times and only `n_varying` survives whole;
* `turbotab/manuscript.py:444` `ranked[:5]` — the exported manuscript's
  `top_features`, which is `explainHTML`'s defect in the artifact that leaves
  the building;
* `turbotab/features.py:370` `operands[0].head(n)` with `n=6`, under
  `featPreviewHTML`'s 5;
* `turbotab/api.py:233` `value[:8] + [{"_truncated": …}]` — the one that
  already says so, and the shape the others should have;
* and `packs.py`, `grain.py`, `resolution.py`, `clinical.py`, `training.py`,
  `pipeline_plan.py` each carry `[:N]` into a served payload.

That class is **reported for filing rather than closed here**, and it is named
in this file rather than swept because the disposition of a server-side cut is
a different question — some of them are a payload budget and some are a claim
to a reader, and only the second kind owes a sentence. Answering it inside a
page test would be the sweep terminating exactly where its instrument does,
which is the thing `AGENT_ONBOARD` §08.5 asks about.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "web" / "index.html"
DATA = ROOT / "sample_data"

_FN = re.compile(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(")
_SLICE = re.compile(r"\bslice\s*\(")
_CAP = re.compile(r"slice\(\s*0\s*,\s*(\d+)\s*\)")
_SAID = re.compile(r'data-[a-z-]*showing="')
_OF = re.compile(r'data-[a-z-]*of="')

#: Characters after which a `/` opens a REGEX rather than a division.
_BEFORE_REGEX = set("(,=:[!&|?{};+-*%~^<>") | {"\n"}
_KEYWORDS = ("return", "typeof", "case", "in", "of", "new", "delete", "void",
             "instanceof")

#: Every `slice(` that is NOT a cap on a rendered list, keyed by
#: `(function, statement)` and disposed of by name.
#:
#: **An entry is a DECISION and a missing entry is a hole**, which is the
#: distinction `GUIDED-180` draws one layer down and the reason this is a table
#: rather than a regex that skips "the ones that look like string handling". A
#: rule that auto-exempts by shape exempts the next real cap that happens to
#: share the shape.
NOT_A_CAP: Dict[Tuple[str, str], str] = {
    ("renderData",
     "var live = sentences.slice(reverted ? 0 : 0, sentences.length)"
     ".slice(-applied.length);"): (
        "Two slices on one statement and neither bounds what the reader sees. "
        "The first takes the whole array — `reverted ? 0 : 0` is 0 either way, "
        "which is vestigial and is reported rather than repaired here (it is "
        "outside this row's brief). The second takes the LAST "
        "`applied.length` sentences, so the count rendered is exactly the "
        "number of applied fixes the record holds: the surface renders all of "
        "them and there is no remainder to disclose."),
    ("prevalenceColumns", "return (P && P.nutrient_columns) ? "
                          "P.nutrient_columns.slice() : [];"): (
        "`slice()` with no arguments is a COPY, not a cut — the defensive copy "
        "that keeps a caller from mutating `P`. Nothing is dropped."),
    #: THE GATE'S FIRST CATCH, AND IT CAUGHT THIS LOOP. `drawnElsewhere` was
    #: written for `GUIDED-192` in the same session as this file and its
    #: `slice` was reported UNCLASSIFIED on the first probe run — which is
    #: exactly the case a hand list cannot see, arriving within hours rather
    #: than next loop. The disposition is real and it is written here rather
    #: than assumed.
    ("drawnElsewhere", 'var col = q.key.slice("missingness::".length);'): (
        "A STRING slice recovering a column name from a question key, so the "
        "new Preprocess surface can ask whether `renderMissingness` already "
        "holds that column's card. No list is shortened and nothing is "
        "rendered from it."),
    ("renderRepairGroups", 'var kind = q.key.slice("repair_bulk::".length);'): (
        "A STRING slice that strips a key prefix. It shortens an identifier, "
        "not a list, and the thing it produces is rendered whole."),
    ("findingTitle",
     'return "Missing values in " + String(id).slice("missing__".length);'): (
        "The same prefix strip on a finding id, composing a title. No list."),
    ("(top level)", "var qk = cut < 0 ? key : key.slice(0, cut);"): (
        "`att-for` is `<question key>--<exit index>`; this is the string split "
        "that recovers the question key. It runs in the document-level `input` "
        "delegate, which is why no named function encloses it."),
    ("(top level)",
     "var xi = cut < 0 ? -1 : parseInt(key.slice(cut + 2), 10);"): (
        "The other half of the same split, recovering the exit index."),
}

#: The four caps whose statement is checked by the FILE half only. Named,
#: because `LOOP.md` §10 asks for what was bounded rather than for a number
#: that reads as full coverage.
DRIVEN_ONLY_THE_SHARPEST = (
    "`findingCard` (5 chips) and `attachSkewPlots` (3 plots) — driven by "
    "`test_a_shape_claim_says_how_much_of_it_you_are_seeing.py` across three "
    "fixtures already, which is why they are not re-driven here.",
    "`featPreviewHTML` (5 preview rows) — the statement is checked in the "
    "file, not in a rendered DOM. It renders behind a feature-preview press, "
    "which is `test_a_two_column_formula_previews_both_of_its_columns.py`'s "
    "path rather than this one's.",
    "`planRow` (12 selected features) — needs a fitted run WITH a selection "
    "step recorded; the two runs driven below record none, so the branch does "
    "not render on either.",
    "`teachHTML` (6 varying columns) — renders behind a `data-teach` "
    "disclosure on a repeated-measures project, and its total is the "
    "server's `n_varying`, which is itself cut twice upstream (see the module "
    "docstring). Its statement is checked in the file only.",
)


def code_only(src: str) -> str:
    """`src` with every comment blanked, offsets preserved.

    Strings and **regex literals** survive. The regex branch is not
    hypothetical: `esc()` contains `/"/g`, and a scanner that reads that `/` as
    division opens a string on the `"` and swallows the next block comment
    whole. The first version of this function did exactly that and reported a
    `slice(0, N)` written in ENGLISH inside this file's own commit message as a
    site — a detector inventing its own hits.
    """
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "*":
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            for k in range(i, j):
                if src[k] != "\n":
                    out[k] = " "
            i = j
            continue
        if c == "/" and nxt == "/":
            j = src.find("\n", i)
            j = n if j == -1 else j
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if c in "'\"`":
            quote, i = c, i + 1
            while i < n:
                if src[i] == "\\":
                    i += 2
                    continue
                if src[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if c == "/":
            k = i - 1
            while k >= 0 and src[k] in " \t\n":
                k -= 1
            prev = src[k] if k >= 0 else "\n"
            word = re.search(r"[A-Za-z_$][\w$]*$", src[:k + 1])
            if prev in _BEFORE_REGEX or (word and word.group(0) in _KEYWORDS):
                i += 1
                in_class = False
                while i < n:
                    ch = src[i]
                    if ch == "\\":
                        i += 2
                        continue
                    if ch == "[":
                        in_class = True
                    elif ch == "]":
                        in_class = False
                    elif ch == "/" and not in_class:
                        i += 1
                        break
                    elif ch == "\n":
                        break
                    i += 1
                continue
        i += 1
    return "".join(out)


def _functions(code: str) -> List[Tuple[int, int, str]]:
    """`(start, end, name)` for every named function, by brace matching."""
    out = []
    for m in _FN.finditer(code):
        brace = code.find("{", m.end())
        if brace == -1:
            continue
        depth, i, n = 0, brace, len(code)
        while i < n:
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out.append((m.start(), i, m.group(1)))
    return out


def sites() -> List[dict]:
    """Every `slice(` in the page, with its enclosing function and verdict.

    The enclosing function is the INNERMOST NAMED one, so an anonymous callback
    resolves to the function that declares it — which is where the statement
    that discloses the cap lives (`attachSkewPlots` is the case that decides
    this).
    """
    src = PAGE.read_text(encoding="utf-8")
    code = code_only(src)
    assert len(code) == len(src)
    lines = src.splitlines()
    functions = _functions(code)

    out = []
    for m in _SLICE.finditer(code):
        off = m.start()
        line = src.count("\n", 0, off) + 1
        enclosing = None
        for start, end, name in functions:
            if start <= off <= end and (
                    enclosing is None or (end - start) < enclosing[1] - enclosing[0]):
                enclosing = (start, end, name)
        body = code[enclosing[0]:enclosing[1]] if enclosing else ""
        cap = _CAP.match(code, off)
        out.append({
            "line": line,
            "function": enclosing[2] if enclosing else "(top level)",
            "statement": lines[line - 1].strip(),
            "cap": int(cap.group(1)) if cap else None,
            "states": bool("capSaid(" in body
                           or (_SAID.search(body) and _OF.search(body))),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# THE FILE HALF
# ─────────────────────────────────────────────────────────────────────────────

def test_the_enumeration_is_read_out_of_the_page_and_finds_something():
    """The positive control, before any verdict is trusted.

    Zero sites is the same output on an emptied page, a broken comment
    stripper, or a codebase with no caps at all (`GUIDED-045`). So the scan has
    to show it had something to be wrong about first.
    """
    src = PAGE.read_text(encoding="utf-8")
    assert len(src) > 100_000, f"the page is {len(src)} bytes; it is not the page"
    functions = _functions(code_only(src))
    assert len(functions) > 100, (
        f"only {len(functions)} named functions parsed out of the page, so the "
        f"enclosing-function attribution below is mostly '(top level)' and the "
        f"exemption keys mean nothing")
    found = sites()
    assert len(found) >= 12, (
        f"only {len(found)} `slice(` sites found; L48 counted six caps alone")
    assert sum(1 for s in found if s["cap"] is not None) >= 6, (
        f"only {sum(1 for s in found if s['cap'] is not None)} numeric caps "
        f"found and the six in the docstring's table are all still in the page")


def test_the_comment_stripper_does_not_blank_a_line_of_code():
    """The negative control on the instrument, from the bug it actually had.

    A stripper that over-blanks hides a real cap and reports a clean sweep,
    which is the worst thing this file can do. Both directions are checked: a
    `slice(` written in prose is not a site, and every site it does report is
    still present in the raw page at the line it names.
    """
    src = PAGE.read_text(encoding="utf-8")
    code = code_only(src)
    assert len(code) == len(src)

    lines = src.splitlines()
    for site in sites():
        raw = lines[site["line"] - 1]
        assert "slice(" in raw, (
            f"line {site['line']} is reported as a `slice(` site and the raw "
            f"page does not have one there: {raw!r}")

    # And the stripper really is removing prose: this file's own module
    # docstring is quoted into the page nowhere, so a marker is used instead.
    probe = 'var a = "keep";  /* slice(0, 9) in prose */  var b = 1;'
    assert "slice(" not in code_only(probe), (
        "a `slice(` inside a block comment survives the stripper, so the "
        "enumeration counts English as code")
    assert '"keep"' in code_only(probe), "the stripper blanked a string literal"
    assert "slice(" in code_only('x.slice(0, 3); // slice(0, 9)'), (
        "the stripper blanked a real call on a line that also has a comment")


def test_every_slice_in_the_page_states_its_bound_or_is_named(capsys):
    """The gate. Three dispositions, and an unclassified site is a failure.

    THIS IS A CLAIM ABOUT THE FILE and says so: it asserts each capped site is
    WIRED to a statement of its bound, never that the numbers in that statement
    are right. `test_the_permutation_ranking_table_says_how_many_of_how_many`
    is where that second claim is made, driven, for the site that carries the
    most weight.
    """
    found = sites()
    caps = [s for s in found if s["cap"] is not None]
    stating = [s for s in caps if s["states"]]
    silent = [s for s in caps if not s["states"]]
    others = [s for s in found if s["cap"] is None]
    exempt, unclassified = [], []
    for site in others:
        if (site["function"], site["statement"]) in NOT_A_CAP:
            exempt.append(site)
        else:
            unclassified.append(site)

    with capsys.disabled():
        print("\n  ── GUIDED-195 · does every cap say what it capped ──")
        print("  DEFINITION: every `slice(` in web/index.html either states")
        print("  its bound to the user, or is named in NOT_A_CAP with a reason.")
        print(f"  slice( sites enumerated          {len(found)}")
        print(f"    numeric caps                   {len(caps)}")
        print(f"      stating their bound          {len(stating)}")
        print(f"      SILENT                       {len(silent)}")
        print(f"    not a cap                      {len(others)}")
        print(f"      named in NOT_A_CAP           {len(exempt)}")
        print(f"      UNCLASSIFIED                 {len(unclassified)}")
        for s in caps:
            print(f"      {s['line']:5d} {s['function']:22s} cap {s['cap']:>3d}  "
                  f"{'states' if s['states'] else 'SILENT'}")
        print("  DRIVEN here: `explainHTML` only, on two target shapes.")
        print("  The other five caps are checked in the FILE, not in a DOM.")

    # THE POSITIVE CONTROL, and it is the same rule this file enforces one
    # level up. `GUIDED-045`: a gate whose every assertion is an absence claim
    # passes hardest on an empty file, so *zero silent caps* would be the
    # output for a page with no caps, a page with no script, and a page that
    # does not exist. `test_an_absence_assertion_carries_a_positive_control`
    # caught this on the merge — a truncation-audit that could not tell "we
    # looked and found none" from "we did not look" would be the defect it
    # audits for, in the auditor.
    assert len(found) > 8, (
        f"only {len(found)} `slice(` sites were enumerated in the page. The "
        f"assertions below are absence claims and would all pass on an empty "
        f"file; this is what makes them mean something")
    assert len(caps) >= 6, (
        f"only {len(caps)} numeric caps were found, and six are known to exist. "
        f"Either the extractor stopped seeing them or the page lost them")
    assert stating, "no cap in the page states its bound, which was the defect"

    assert not silent, (
        "these caps truncate a rendered list and nothing on the surface says "
        "so — a truncation nobody records reads as a complete answer:\n  "
        + "\n  ".join(f"{s['function']} line {s['line']}: cap {s['cap']} — "
                      f"{s['statement']}" for s in silent))
    assert not unclassified, (
        "these `slice(` sites are neither a stated cap nor named in "
        "`NOT_A_CAP`. Add the entry with the reason it cuts nothing a reader "
        "could be misled about, or state the bound:\n  "
        + "\n  ".join(f"({s['function']!r}, {s['statement']!r})  line "
                      f"{s['line']}" for s in unclassified))


def test_no_exemption_outlives_the_line_it_was_written_for():
    """Every `NOT_A_CAP` key resolves to a site the page still has.

    Trap #3, pointed at an exemption table: a key standing for a real statement
    has to be shown to resolve, or the table quietly becomes a list of reasons
    for code that is gone — which is `AGENT_ONBOARD` §07.8's decayed record in
    the one place it would be least visible.
    """
    keys = {(s["function"], s["statement"]) for s in sites()}
    stale = sorted(k for k in NOT_A_CAP if k not in keys)
    assert not stale, (
        "these exemptions name a statement the page no longer contains, so "
        "they are reasons attached to nothing:\n  "
        + "\n  ".join(f"{f} :: {stmt}" for f, stmt in stale))


# ─────────────────────────────────────────────────────────────────────────────
# THE DRIVEN HALF
# ─────────────────────────────────────────────────────────────────────────────

#: Two fixtures of different target shape (`GUIDED-097`), chosen so that they
#: are also the two BRANCHES of the cap: 45 ranked columns is truncated at 15
#: and 12 is not truncated at all. A statement that only ever renders "showing
#: N of M" and never "all M" would pass a one-branch test and still leave a
#: complete list indistinguishable from a cut one.
RANKED = [
    ("wide_assay.csv", "responder", "classification", "logreg", 45, True),
    ("clinic_visits.csv", "hba1c", "regression", "ridge", 12, False),
]

_PATHS = ("interview?step=data", "interview?step=explore",
          "interview?step=features", "interview?step=preprocess",
          "capabilities", "features", "recipes", "preprocess", "figures",
          "draft", "manuscript", "models", "training", "instability", "explain",
          "sensitivity", "evidence/plausibility", "evidence/missingness")


def _fitted(fixture, target, model):
    """A real project, sealed and fitted, so `/explain` is the server's own."""
    from fastapi.testclient import TestClient

    from turbotab import api, training as _training

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]

    def decide(kind, **payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (kind, r.text[:250])

    decide("set_target", column=target)
    decide("set_grain", answer="one_row_per_person")
    decide("set_eligibility", answer="everyone")
    decide("seal", fraction=0.25)
    project = api.STORE.get(pid)
    project.training_run = _training.train(project, [model])
    api._RUNS[pid] = {"run": project.training_run}
    return client, pid


def _routes(client, pid):
    out = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in _PATHS:
        resp = client.get(f"/project/{pid}/{path}")
        out[f"/project/{pid}/{path}"] = (resp.json() if resp.status_code == 200
                                         else {})
    return out


@pytest.mark.parametrize("fixture,target,shape,model,n_ranked,truncated",
                         RANKED,
                         ids=["classification · 45 ranked, cut at 15",
                              "regression · 12 ranked, nothing cut"])
def test_the_permutation_ranking_table_says_how_many_of_how_many(
        fixture, target, shape, model, n_ranked, truncated):
    """The sharp site, driven — the claim the file half cannot make.

    Four things are read out of the rendered DOM: how many rows the table
    actually drew, the machine-readable pair beside it, the same two numbers in
    the sentence a person reads, and — on the truncated branch — the remainder
    named rather than left to arithmetic.

    `data-cap-showing` and `data-cap-of` are read here rather than asserted to
    exist, because trap 7 is the structured payload disagreeing with the prose
    beside it and only reading both catches that.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _fitted(fixture, target, model)
    served = client.get(f"/project/{pid}/explain").json()
    assert served.get("blocked_by") is None, served.get("blocked_by")
    ranked = served["run"]["ranked"]
    assert len(ranked) == n_ranked, (
        f"{fixture}:{target} now ranks {len(ranked)} columns, not {n_ranked}; "
        f"the arithmetic below is pinned to the fixture and this run is no "
        f"longer the branch it claims to be")
    assert client.get(f"/project/{pid}").json()["task_type"] == shape

    out = PH.run(
        "__emit({html: __harness.html('explainBox') || "
        "        __harness.html('sec-explain') || ''});",
        routes=_routes(client, pid), search=f"?project={pid}")
    html = out["html"]
    assert "Permutation importance" in html, (
        f"the explain surface did not render the ranking at all, so nothing "
        f"below is reading the table it claims to: {html[:400]!r}")

    rows = re.findall(r"<tr><td>([^<]*)</td><td>", html)
    drawn = [r for r in rows if r in {x["feature"] for x in ranked}]
    assert len(drawn) == min(15, n_ranked), (
        f"the table drew {len(drawn)} ranked rows; the cap is fifteen and the "
        f"ranking is {n_ranked} long")

    said = re.search(r'data-cap-for="permutation-ranking" '
                     r'data-cap-showing="(\d+)" data-cap-of="(\d+)">'
                     r'([^<]*)</span>', html)
    assert said, (
        f"the table shows {len(drawn)} of {n_ranked} ranked columns and says "
        f"nothing about the other {n_ranked - len(drawn)}. Under a heading "
        f"naming the method, the model and the shuffle count, that is a "
        f"top-{len(drawn)} asserting it is the whole ranking. The block "
        f"rendered: {html[-1200:]!r}")
    assert int(said.group(1)) == len(drawn), (
        f"the caption says it is showing {said.group(1)} and {len(drawn)} rows "
        f"are drawn — trap 7, the machine-readable count disagreeing with the "
        f"table beside it")
    assert int(said.group(2)) == n_ranked, (
        f"the caption says the ranking is {said.group(2)} columns long; the "
        f"server ranked {n_ranked}")
    assert str(n_ranked) in said.group(3), (
        f"the total is in the attributes and not in the sentence a person "
        f"reads: {said.group(3)!r}")

    if truncated:
        assert f"The other {n_ranked - 15} were scored the same way" in html, (
            "the remainder is not named, so a reader has to subtract to learn "
            "that thirty columns exist and were not printed")
    else:
        assert said.group(3).startswith("All "), (
            f"a ranking that was NOT cut renders the same sentence as one that "
            f"was, so completeness is something the reader has to infer from a "
            f"silence: {said.group(3)!r}")
        assert "were scored the same way" not in html, (
            "an uncut ranking claims a remainder that does not exist")
