"""The evidence gate — every pack claim carries a badge and a resolvable source.

`DOMAIN_SCIENCE.md` §01.1. All four research threads independently asked for the
same primitive: **surface the epistemic status of every claim the app makes.**

This is the check that keeps it true.

## The scope this gate used to have, and why it was too narrow

It walked `packs.PACKS[*].priors` and scanned exactly one file, `turbotab/
packs.py`, for `[verify-at-build]` literals. `GUIDED-059` is what that missed:

* **A pack FINDING is a claim too**, and `_finding()` emitted `source="pack"`
  with no status and no citation, so every finding four detectors produced went
  out unbadged while the gate printed a green tick.
* **A REFUSAL is the sharpest claim a pack makes** — *"nobody can compute this,
  not the app and not you with a spreadsheet"* — and the four
  `PrevalenceRefusal`s carried a message, an offer, and no badge at all.
* **The file scan stopped at one file.** `turbotab/nutrition.py` and
  `turbotab/figure_specs.py` hold domain thresholds and were outside it, and
  `nutrition.py`'s own docstring named this gate as the guarantor of numbers it
  had never opened. A claim with no record, in the module whose subject is not
  making them.

## What it checks now

1. **Every pack prior carries `evidence`** — a status and a source.
2. **Every registered figure carries `evidence`.** A figure's checklist is a set
   of claims about what a reviewer expects.
3. **Every module-level `Evidence` under `turbotab/` resolves.** A constant that
   is constructed and attached to nothing still asserts a citation.
4. **Every source resolves** — the named file exists under
   `docs/turbotab/research/`, and the named section is a heading in it.
5. **Structurally, a finding and a refusal cannot be made without a badge.**
   Every `_finding(...)` call site and every `PackRefusal` subclass call site
   under `turbotab/` passes `evidence=`. This is the static half of a guarantee
   `packs._finding` and `packs.PackRefusal.__init__` also enforce at runtime:
   the runtime check covers the paths that run, and this covers the rest.
6. **A badge assembled as a dict literal resolves too.** `{"evidence_status":
   …, "source": …}` written by hand bypasses `Evidence`, so nothing validates
   its form — but the source still has to point somewhere real.
7. **No `[verify-at-build]` number ships as a hard-coded constant**, in any
   module that emits pack content rather than only in `packs.py`. The research
   threads hit an egress proxy and marked the numbers they could not read from
   primary text.

The scan is scoped to the **emitters** — the modules whose code actually
produces findings, refusals or figure specs — rather than to every file under
`turbotab/`. That is not timidity: it is a bare-number scan, and widened to the
whole package it flags `eligibility.py`'s `unique()[:50]` for sharing a digit
with a metabolomics threshold. A gate that cries wolf is a gate somebody
switches off.

## What it deliberately does not check

**Whether the claim is faithful to the section it names.** A citation that
resolves to the wrong heading passes here, and that is the same honest limit
`ledger.py check` has: it enforces that a `FIXED` row *names* a test and cannot
tell whether the test is any good.

**Whether a number the research never mentions is sourced at all.**
`[verify-at-build]` marks numbers the research *tried to read and could not*.
A threshold the implementation invented — `GUIDED-061`'s `DRIFT_LIMIT` — is
marked nowhere and is invisible to check 7. Saying so is the difference between
a gate and a reassurance.

Usage — from the repository root:

    venv/bin/python docs/turbotab/tools/evidence.py check
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "docs" / "turbotab" / "research"
PACKAGE = ROOT / "turbotab"

# A heading in a research file, at any level.
_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.M)

# `[verify-at-build]` and `[verify-at-build: what]`.
#
# **A BARE MARKER PROTECTS NOTHING, AND THAT WAS TRUE FOR EIGHT OF NINE.**
# `GUIDED-212`. This gate builds its held-out set from the marker's PAYLOAD, so
# a marker with nothing after the colon contributes no numbers and is never
# checked against anything. `AGENT_ONBOARD.md` §00, `ROADMAP.md` condition 6 and
# `LOOP.md` §04 all describe `[verify-at-build]` as *structurally forbidden from
# shipping as a constant* — **a guarantee three documents call structural, held
# for one marker in nine.**
#
# So a bare marker is now a gate FAILURE in its own right. An uncheckable
# marker is the recorded-absence rule's case, not a pass: *nobody said which
# number* and *there is no number* are different claims and were rendering as
# one. Two sentinels make the second one sayable:
#
#   `[verify-at-build: no number]` — the claim is qualitative. Two of the six
#     real markers are: slow-in/slow-out easing, and habituation exercises as a
#     standard of care. Neither has a threshold to hold out of the code.
#   `[verify-at-build: legend]`    — the line DEFINES the marker rather than
#     using it. Two of the nine are legends, and counting them as claims made
#     the corpus look better protected than it was.
_VERIFY = re.compile(r"\[verify-at-build:?\s*([^\]]*)\]")

def _code_only(source: str) -> str:
    """`source` with every string literal and comment blanked out.

    Exact rather than approximate, and line-aligned so a failure can name the
    line. Strings and comments are prose — a number quoted inside a `reason` is
    discussion, not a constant — and that is the distinction this gate turns on.

    Three regexes could not draw it. They missed docstrings entirely, so a
    number in one read as code; patched to eat triple quotes they mis-aligned
    on a nested apostrophe; and on Python 3.12+ an f-string tokenizes into
    START / MIDDLE / END rather than one STRING, so prose between the braces
    leaked through as well. A regex approximating Python is a second
    implementation of Python to keep in sync, which is this project's
    most-repeated defect one level down.

    Falls back to the raw source if the file will not tokenize: a gate that
    silently reads nothing is worse than one that over-reports.
    """
    import io
    import tokenize as _tok

    prose = {_tok.STRING, _tok.COMMENT}
    middle = getattr(_tok, "FSTRING_MIDDLE", None)
    if middle is not None:
        prose.add(middle)

    lines = [""] * (source.count("\n") + 2)
    try:
        for kind, text, (row, col), _end, _ in _tok.generate_tokens(
                io.StringIO(source).readline):
            if kind in prose or "\n" in text or row >= len(lines):
                continue
            pad = max(0, col - len(lines[row]))
            lines[row] = lines[row] + (" " * pad) + text
    except (_tok.TokenError, IndentationError, SyntaxError):
        return source
    return "\n".join(lines)


#: Payloads that declare, rather than name, what is held out.
_NO_NUMBER = "no number"
_LEGEND = "legend"

# A bare number inside a verify-at-build note. `50%`, `0.8`, `40`.
_NUMBER = re.compile(r"\b(\d+(?:\.\d+)?)\s*%?")

# The helper every pack finding goes through. Named here because the gate's
# structural half is *where it is called from*, which no runtime check can see.
_FINDING_FN = "_finding"


def _sections(path: pathlib.Path) -> set:
    return {m.group(1).strip() for m in _HEADING.finditer(
        path.read_text(encoding="utf-8"))}


def _resolve(source: str) -> str:
    """`""` when the source points at a real file and a real heading in it."""
    filename, _, section = (source or "").partition("#")
    if not filename or not section:
        return f"{source!r} is not `research/FILE.md#Section heading`"
    path = ROOT / "docs" / "turbotab" / filename
    if not path.exists():
        return f"{filename} does not exist"
    if section not in _sections(path):
        return f"{filename} has no section {section!r}"
    return ""


def _modules() -> list:
    """Every non-test module under `turbotab/`, in a stable order."""
    return sorted(p for p in PACKAGE.glob("*.py")
                  if not p.name.startswith("test_") and p.name != "__init__.py")


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _import_package():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _refusal_names() -> set:
    """Every exception class that must carry a badge, discovered rather than listed.

    A hard-coded `{"PrevalenceRefusal"}` would go stale the first time a second
    pack refuses something, and it would go stale *silently* — which is the
    failure mode this whole file exists to remove.
    """
    _import_package()
    from turbotab import packs

    found, queue = set(), [packs.PackRefusal]
    while queue:
        cls = queue.pop()
        found.add(cls.__name__)
        queue.extend(cls.__subclasses__())
    return found


def _keywords(call: ast.Call) -> set:
    return {kw.arg for kw in call.keywords if kw.arg}


def call_sites() -> tuple:
    """`(problems, emitters, n_calls)` — the structural half, as data.

    Factored out of `check()` so a test can assert on it rather than on the
    tool's printed output. A gate whose only interface is stdout is a gate a
    test can only check the description of, which is the class this file's
    subject keeps producing.
    """
    refusals = _refusal_names()
    problems, emitters, n_calls = [], [], 0
    for path in _modules():
        emits = False
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == _FINDING_FN:
                emits = True
                n_calls += 1
                if "evidence" not in _keywords(node):
                    problems.append(
                        f"{path.name}:{node.lineno}: this finding is emitted "
                        f"with no evidence= badge. A finding states where the "
                        f"field stands, or the app is uniformly confident.")
            elif name in refusals:
                emits = True
                n_calls += 1
                if "evidence" not in _keywords(node):
                    problems.append(
                        f"{path.name}:{node.lineno}: this refusal is raised "
                        f"with no evidence= badge. A refusal is the sharpest "
                        f"claim a pack makes and is the last one that should "
                        f"go out unbadged.")
            elif name == "FigureSpec":
                emits = True
        if emits:
            emitters.append(path)
    return problems, emitters, n_calls


def _string(node) -> str:
    """The literal value of a string node, including implicit concatenation."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):                    # an f-string
        return ""
    return ""


# ── the two walks, lifted out of `check()` — `TEST-107` ─────────────────────
#
# **THE PRE-FILTERS ARE GONE AND THE COUNTS ARE RETURNED.** Both halves matter
# and the second matters more.
#
# The filters were `if "Claim(" not in path.read_text(): continue` and the same
# for `"Evidence("`. Break both literals and this gate checks **32 of 67 claims
# and 0 of 51 module constants — and still prints `ok` and exits 0.** Verified
# without touching the file, by exec'ing a patched copy of this source: a false
# green in one of the six pre-commit gates, which is the shape `AGENT_ONBOARD.md`
# §07 trap 5c names as the worst of the three. Dropping them costs **+0.65 s**
# (0.600 → 1.251 s, median of five) on a 52-module corpus where only 4 and 5
# modules carry the literals, and gives byte-identical counts.
#
# **A CEILING WOULD BE THE WRONG INSTRUMENT and `TEST-107`'s `act` is wrong to
# suggest one.** The quantity it would bound is *modules skipped* — 48 and 47 of
# 52 — which grows every time an unrelated module is added, so it would go red
# for reasons that have nothing to do with claims going unchecked.
# `repo_write_guard`'s ceiling bounds destinations the instrument KNOWS it
# cannot see; that is a different quantity.
#
# What replaces it is a FLOOR, and a floor needs a number to stand on, which is
# why these return their findings instead of printing them. `call_sites()`
# already made this move and its docstring says why: *"A gate whose only
# interface is stdout is a gate a test can only check the description of."*
#
# **And a zero-guard would not be enough**: 32 of the 67 claims come from the
# unfiltered figure-registry loop above, so breaking the filter left 32, not 0.
# The floors are per-walk for exactly that reason.

def module_claims(modules=None) -> list:
    """`(label, source)` for every `Claim` declared as a module constant."""
    import importlib

    _import_package()
    from turbotab import packs

    out = []
    for path in (_modules() if modules is None else modules):
        module = importlib.import_module(f"turbotab.{path.stem}")
        for name, value in sorted(vars(module).items()):
            if name.startswith("_") or not isinstance(value, tuple):
                continue
            for claim in value:
                if not isinstance(claim, packs.Claim):
                    continue
                out.append((f"{path.name}/{name}/{claim.key}",
                            claim.evidence.source))
    return out


def module_constants(modules=None) -> list:
    """`(label, source)` for every module-level `Evidence`."""
    import importlib

    _import_package()
    from turbotab import packs

    out = []
    for path in (_modules() if modules is None else modules):
        module = importlib.import_module(f"turbotab.{path.stem}")
        for name, value in sorted(vars(module).items()):
            if not isinstance(value, packs.Evidence) or name.startswith("_"):
                continue
            out.append((f"{path.name}/{name}", value.source))
    return out


def check() -> int:
    problems = []
    _import_package()
    from turbotab import figures, packs
    import turbotab.figure_specs                           # noqa: F401 — registers
    import turbotab.nutrition                              # noqa: F401 — badges

    # ── 1 · every prior badged, and its source resolves ─────────────────────
    n_priors = 0
    for pack_key, pack in packs.PACKS.items():
        for prior in pack.priors:
            n_priors += 1
            if prior.evidence is None:                     # pragma: no cover
                problems.append(f"{pack_key}/{prior.question}: no evidence badge")
                continue
            bad = _resolve(prior.evidence.source)
            if bad:
                problems.append(f"{pack_key}/{prior.question}: {bad}")
    if not n_priors:
        problems.append("no pack priors found at all; the walk is wrong")

    # ── 2 · every registered figure badged ──────────────────────────────────
    n_figures = 0
    for figure_id, spec in figures.REGISTRY.items():
        n_figures += 1
        if spec.evidence is None:                          # pragma: no cover
            problems.append(f"figure/{figure_id}: no evidence badge")
            continue
        bad = _resolve(spec.evidence.source)
        if bad:
            problems.append(f"figure/{figure_id}: {bad}")
    if not n_figures:
        problems.append("no figures found at all; the registry walk is wrong")

    # ── 2b · every CLAIM inside a badge resolves too ────────────────────────
    #
    # `GUIDED-064`. A statement that makes two claims the field holds
    # differently now carries a badge per claim, and each of those is a
    # citation like any other. A claim source that resolved to nothing would be
    # the same defect the headline badge was built to remove, one level in.
    n_claims = 0
    for figure_id, spec in figures.REGISTRY.items():
        for claim in getattr(spec, "claims", ()) or ():
            n_claims += 1
            bad = _resolve(claim.evidence.source)
            if bad:
                problems.append(f"figure/{figure_id}/{claim.key}: {bad}")

    # Claims declared as module constants — the form every pack finding uses,
    # because a claim tuple built inline in a detector could not be walked.
    for label, source in module_claims():
        n_claims += 1
        bad = _resolve(source)
        if bad:
            problems.append(f"{label}: {bad}")

    # ── 3 · every module-level Evidence resolves ────────────────────────────
    #
    # A constant attached to nothing is still a citation, and `GUIDED-059` is
    # what happens when nobody looks at one: ATWATER_EVIDENCE and
    # DESIGN_EVIDENCE were well-formed, resolvable, and reached no claim.
    # Resolving them is necessary and is not sufficient — check 5 is the half
    # that asks whether they arrive anywhere.
    n_constants = 0
    for label, source in module_constants():
        n_constants += 1
        bad = _resolve(source)
        if bad:
            problems.append(f"{label}: {bad}")

    # ── 4 · a hand-written badge resolves too ───────────────────────────────
    n_literals = 0
    for path in _modules():
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Dict):
                continue
            keys = {_string(k) for k in node.keys if k is not None}
            if "evidence_status" not in keys:
                continue
            source = ""
            for key, value in zip(node.keys, node.values):
                if key is not None and _string(key) == "source":
                    source = _string(value)
            if not source:
                # A `source` assembled from a variable is out of reach here and
                # is not asserted to be absent. Only a literal is checkable.
                continue
            n_literals += 1
            bad = _resolve(source)
            if bad:
                problems.append(
                    f"{path.name}:{node.lineno}: a hand-written badge — {bad}")

    # ── 5 · a finding and a refusal cannot be made without a badge ──────────
    #
    # THE CHECK GUIDED-059 NEEDED. Every other check here asks whether a badge
    # that exists is well-formed; this one asks whether the claim has one at
    # all, and it is static because a detector that never fires on the fixtures
    # is exactly the one whose badge nobody would notice was missing.
    unbadged, emitters, n_calls = call_sites()
    problems.extend(unbadged)
    if not n_calls:
        problems.append("no findings or refusals found at all; the walk is wrong")

    # ── 6 · no `[verify-at-build]` number as a literal in an emitter ────────
    unverified = set()
    n_declared, n_legend = 0, 0
    for path in sorted(RESEARCH.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.split("\n"), 1):
            for m in _VERIFY.finditer(line):
                payload = m.group(1).strip().lower()
                if payload == _LEGEND:
                    n_legend += 1
                    continue
                if payload == _NO_NUMBER:
                    n_declared += 1
                    continue
                numbers = _NUMBER.findall(m.group(1))
                if not numbers:
                    # THE FAILURE THIS GATE DID NOT HAVE. A marker that names
                    # no number is a marker this check cannot enforce, and it
                    # read as a pass for a dozen loops.
                    problems.append(
                        f"{path.name}:{lineno} carries a bare "
                        f"[verify-at-build] with no number after it, so "
                        f"nothing is held out of the code and the marker "
                        f"guarantees nothing. Name the number the claim rests "
                        f"on, or say `[verify-at-build: {_NO_NUMBER}]` if the "
                        f"claim is qualitative, or `[verify-at-build: "
                        f"{_LEGEND}]` if this line defines the marker.")
                    continue
                for n in numbers:
                    unverified.add((path.name, n))

    for path in emitters:
        source = path.read_text(encoding="utf-8")
        # Strings and comments are prose, not constants. Only code lines count —
        # otherwise quoting a number in a `reason` would fail the gate, and the
        # reasons are exactly where a number SHOULD be discussed.
        # TOKENIZED, NOT REGEXED — and the change is the gate learning its own
        # lesson. This stripped strings and comments with three regexes, which
        # missed DOCSTRINGS entirely: a number inside one survived as "code",
        # so `packs.py`'s prose about *a 40-column clinical table* tripped the
        # check the moment `METABOLOMICS_PACK.md`'s 40 was first held out at
        # L51. Patching the regex to eat triple quotes then mis-aligned on a
        # nested apostrophe and reported a line that did not contain the
        # number at all.
        #
        # A regex approximating Python is a second implementation of Python to
        # keep in sync, which is this project's most-repeated defect one level
        # down. `tokenize` is exact and it is what `parsecheck.py` already uses.
        code = _code_only(source)
        for filename, number in sorted(unverified):
            # `\b` on both sides so `50` does not match inside `250`.
            if re.search(rf"(?<![\w.]){re.escape(number)}(?![\w.])", code):
                problems.append(
                    f"{filename} marks {number} [verify-at-build] and it "
                    f"appears as a literal in turbotab/{path.name}. A number "
                    f"nobody has read from primary text may not ship as a "
                    f"constant.")

    if problems:
        print("EVIDENCE GATE FAILED")
        for p in problems:
            print(f"  ✗ {p}")
        print("\n  Every pack claim carries `evidence=Evidence(status=…, "
              "source='research/FILE.md#Section')` — priors, findings,\n  "
              "refusals and figures alike. The gate resolves the source; it "
              "does not check the claim is faithful to it.")
        return 1
    print(f"ok — {n_priors} pack priors, {n_figures} figures, {n_claims} "
          f"claims, {n_constants} module constants and {n_literals} "
          f"hand-written badges resolve; "
          f"{n_calls} findings/refusals badged at the call site across "
          f"{len(emitters)} emitter(s); {len(unverified)} [verify-at-build] "
          f"number(s) held out of the code, {n_declared} marker(s) declared to "
          f"carry no number, {n_legend} legend(s)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "check":
        raise SystemExit(f"unknown command {sys.argv[1]!r}; only `check`")
    raise SystemExit(check())
