"""Every docs path the code reads exists, and every evidence source resolves (BLUEPRINT §9.1).

When the legacy app was retired its documents moved: the binding ones to
``docs/turbotab-next/reference/``, the rest to ``docs/turbotab/archive/``. A path the code reads
where nothing is fails only when that line runs, so this finds them statically:

* path constructions: ``REPO / "docs" / ...``, ``os.path.join(..., "docs", ...)``, and a name bound
  to one (``root = REPO / "docs"``, then ``root / "turbotab-next/..."``);
* in v2 code (``turbotab/core``, ``turbotab/server``, ``turbotab/replay.py``, the scripts under
  ``docs/turbotab-next``) and the frontend's code, any string that names a ``docs/...`` path,
  docstrings aside;
* evidence sources (``research/NUTRITION_PACK.md#04 · Energy adjustment — …``,
  ``DOMAIN_PACKS.md#01 · …``), relative to the reference folder: the file exists, and the section's
  number (what precedes " · ") or its whole title is one of the file's headings. The legacy
  ``evidence.py`` pre-commit gate made that check; it is a test now. It says a source is named and
  can be found, never that the claim is faithful to it.

The legacy domain modules beside ``turbotab/core`` and their tests are scanned for path
constructions and evidence sources only: their prose is history, and Classic imports them.
"""
from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REFERENCE = REPO / "docs" / "turbotab-next" / "reference"

SOURCE = re.compile(r"^((?:research/)?[A-Z][A-Z_]*\.md)#(.+)$")
DOCS_IN_TEXT = re.compile(r"(?:\.\./)*docs/[A-Za-z0-9_./-]*[A-Za-z0-9_/-]")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.M)
FRONTEND_STRING = re.compile(r"""["'`]((?:\.\./|\./)*docs/[^"'`$\s]*)["'`]""")


def _v2_python() -> list[Path]:
    files = [*(REPO / "turbotab" / "core").rglob("*.py"), *(REPO / "turbotab" / "server").rglob("*.py"),
             REPO / "turbotab" / "replay.py", *(REPO / "docs" / "turbotab-next").rglob("*.py")]
    # This file is left out: its negative control writes paths that must not resolve.
    return sorted(p for p in files if "__pycache__" not in p.parts and p.resolve() != Path(__file__).resolve())


def _legacy_python() -> list[Path]:
    return sorted(p for p in (REPO / "turbotab").glob("*.py")
                  if p.name not in ("__init__.py", "replay.py"))


def _frontend_code() -> list[Path]:
    root = REPO / "turbotab" / "frontend"
    files = [root / "package.json", *root.glob("*.config.ts")]
    for sub in ("src", "e2e", "scripts"):
        files += [p for p in (root / sub).rglob("*") if p.suffix in (".ts", ".tsx", ".js", ".mjs")]
    return sorted(p for p in files if p.is_file() and "node_modules" not in p.parts
                  and p.name != "generated.ts")


def _docstrings(nodes: list[ast.AST]) -> set[int]:
    out = set()
    for node in nodes:
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                out.add(id(body[0].value))
    return out


def _string_parts(node: ast.AST, bound: dict[str, list[str]]) -> list[str] | None:
    """The literal tail of a ``/`` chain or an ``os.path.join``, or None when it has none."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _string_parts(node.left, bound) or []
        right = node.right
        if isinstance(right, ast.Constant) and isinstance(right.value, str):
            return left + [right.value]
        return None
    if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "join" and \
            getattr(getattr(node.func, "value", None), "attr", None) == "path":
        parts: list[str] = []
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                parts.append(arg.value)
            elif isinstance(arg, ast.Name) and arg.id in bound:
                parts = list(bound[arg.id])
            else:
                parts = []
        return parts or None
    if isinstance(node, ast.Name) and node.id in bound:
        return list(bound[node.id])
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    return []


def _docs_path(parts: list[str]) -> str | None:
    joined = "/".join(p.strip("/") for p in parts if p)
    at = joined.find("docs/") if not joined.startswith("docs/") else 0
    if joined == "docs" or joined.endswith("/docs"):
        return "docs"
    return joined[at:] if at >= 0 and (at == 0 or joined[at - 1] == "/") else None


def _named_constants(nodes: list[ast.AST]) -> dict[str, str]:
    return {t.id: n.value.value for n in nodes if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)
            and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}


def _sources(nodes: list[ast.AST]) -> set[str]:
    """Literal and f-string evidence sources (f-strings over module constants, as content.py
    composes ``f"{_NUT}#04 · ..."``)."""
    consts = _named_constants(nodes)
    out = set()
    for node in nodes:
        value = None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
        elif isinstance(node, ast.JoinedStr):
            pieces = []
            for v in node.values:
                if isinstance(v, ast.Constant):
                    pieces.append(str(v.value))
                elif isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name) \
                        and v.value.id in consts:
                    pieces.append(consts[v.value.id])
                else:
                    pieces = []
                    break
            value = "".join(pieces) or None
        if value and SOURCE.match(value):
            out.add(value)
    return out


def scan(path: Path, *, prose: bool) -> tuple[set[str], set[str]]:
    """``(docs paths, evidence sources)`` one Python file names."""
    nodes = list(ast.walk(ast.parse(path.read_text("utf-8"))))
    bound: dict[str, list[str]] = {}
    for node in nodes:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            parts = _string_parts(node.value, bound)
            if parts and not (len(parts) == 1 and isinstance(node.value, ast.Constant)):
                bound[node.targets[0].id] = parts
    paths: set[str] = set()
    # Only the whole chain: ``REPO / "docs" / "x.md"`` also contains ``REPO / "docs"``.
    inner = {id(n.left) for n in nodes if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)}
    for node in nodes:
        if id(node) in inner:
            continue
        if isinstance(node, ast.BinOp) or (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "join"):
            parts = _string_parts(node, bound)
            found = _docs_path(parts) if parts else None
            if found:
                paths.add(found)
    if prose:
        skip = _docstrings(nodes)
        for node in nodes:
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "docs/" in node.value \
                    and id(node) not in skip:
                for hit in DOCS_IN_TEXT.findall(node.value.split("#", 1)[0]):
                    paths.add(re.sub(r"^(?:\.\./)+", "", hit).rstrip("/."))
    return paths, _sources(nodes)


def _exists(rel: str) -> bool:
    if "*" in rel:
        return any(REPO.glob(rel))
    return (REPO / rel).exists()


def _section_resolves(doc: Path, section: str) -> bool:
    headings = [m.group(1).strip() for m in HEADING.finditer(doc.read_text("utf-8"))]
    if section.strip() in headings:
        return True
    number = section.split(" · ", 1)[0].strip()
    return any(re.split(r"[\s·]", h, maxsplit=1)[0] == number for h in headings if h)


def resolve_source(source: str) -> str:
    """``""`` when an evidence source names a reference file and a section heading in it."""
    doc, section = SOURCE.match(source).groups()
    path = REFERENCE / doc
    if not path.exists():
        return f"{doc} is not in docs/turbotab-next/reference/"
    if not _section_resolves(path, section):
        return f"{doc} has no heading for section {section!r}"
    return ""


@functools.lru_cache(maxsize=1)
def _everything() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    paths: dict[str, set[str]] = {}
    sources: dict[str, set[str]] = {}
    for files, prose in ((_v2_python(), True), (_legacy_python(), False)):
        for f in files:
            found, cited = scan(f, prose=prose)
            rel = f.relative_to(REPO).as_posix()
            for p in found:
                paths.setdefault(p, set()).add(rel)
            for s in cited:
                sources.setdefault(s, set()).add(rel)
    for f in _frontend_code():
        for hit in FRONTEND_STRING.findall(f.read_text("utf-8")):
            rel_path = re.sub(r"^(?:\.\./|\./)+", "", hit).rstrip("/")
            paths.setdefault(rel_path, set()).add(f.relative_to(REPO).as_posix())
    return paths, sources


# A test file's own fake source, written to prove the resolver refuses it.
_SYNTHETIC = {"research/X.md#1 · A"}


def test_every_docs_path_the_code_reads_exists():
    paths, _ = _everything()
    missing = {p: sorted(w) for p, w in paths.items() if p != "docs" and not _exists(p)}
    assert not missing, (
        "these docs paths are read or cited by code and do not exist (moved to "
        "docs/turbotab-next/reference/ or docs/turbotab/archive/ when the legacy app was retired?):\n  "
        + "\n  ".join(f"{p}  <- {', '.join(w)}" for p, w in sorted(missing.items())))


def test_every_evidence_source_resolves_to_a_reference_section():
    _, sources = _everything()
    broken = {s: (resolve_source(s), sorted(w)) for s, w in sources.items()
              if s not in _SYNTHETIC and resolve_source(s)}
    assert not broken, "\n  ".join(f"{s}: {why}  <- {', '.join(w)}" for s, (why, w) in sorted(broken.items()))


def test_the_scan_sees_what_it_checks():
    """The positive control: two absence assertions pass hardest on a scan that read nothing."""
    paths, sources = _everything()
    assert "docs/turbotab-next/reference/research/NUTRITION_PACK.md" in paths, sorted(paths)[:20]
    assert "docs/turbotab-next/reference/LOCKBOX_CONSTITUTION.md" in paths
    assert "docs/turbotab-next/m1/screens" in paths       # a frontend journey's screenshot folder
    assert len(paths) >= 15, sorted(paths)
    assert any(s.startswith("DOMAIN_PACKS.md#") for s in sources)
    assert sum(s.startswith("research/") for s in sources) >= 50, len(sources)


def test_the_resolver_refuses_what_does_not_resolve(tmp_path):
    """The negative control: a missing file, a missing section and a moved path each fail."""
    assert resolve_source("research/NUTRITION_PACK.md#04 · Energy adjustment — the methodological signature") == ""
    assert resolve_source("research/CLINICAL_SURVEY_PACK.md#A1.2 · Reference ranges vs physiological plausibility") == ""
    assert "not in" in resolve_source("research/NOT_A_PACK.md#01 · Anything")
    assert "no heading" in resolve_source("research/NUTRITION_PACK.md#99 · A section nobody wrote")
    probe = tmp_path / "probe.py"
    probe.write_text('ROOT = object()\nPACK = ROOT / "docs" / "turbotab" / "research" / "NUTRITION_PACK.md"\n'
                     'OLD = "docs/turbotab/ROADMAP.md"\n', "utf-8")
    found, _ = scan(probe, prose=True)
    assert found == {"docs/turbotab/research/NUTRITION_PACK.md", "docs/turbotab/ROADMAP.md"}, found
    assert not any(_exists(p) for p in found)
