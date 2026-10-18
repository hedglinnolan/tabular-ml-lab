"""The registry: claims by id, their derived status, and their sentences (CLAIMS_REGISTRY.md §2.2,
§6).

Status is derived from the source, read with ``ast`` and never imported, relative to the
registry's ``root`` (the repository, or a planted one in a test): a claim is **built** when every
``enforced_by`` names a function, class or name its module defines at top level and every test it
names is a function its file defines (at top level, or in a class there) that pytest runs to its
assertions, so not one marked to skip or xfail unconditionally; otherwise it is **proposed**. A
claim the research swarm proposed stays proposed while any part it asserts is unsupported, as
every part is when it enters (R1_PLAN.md R1-G), so it renders nowhere until its parts are graded.
**Retracted** is declared.

The shipped app holds no tests (the image and the release zip leave them out), so a status is
derived only where the source is complete: by the generator, ``--check``, the meta-tests, and in
strict mode. The generator freezes each claim's id, status and content hash in the **frozen status
file** (:data:`FROZEN` under the root), and a sentence renders from that file alone: a claim it
lists as built, with the claim's current hash, renders; one it lacks, or lists with another hash,
does not, and the refusal says to run the generator. Strict mode also derives the status and
refuses where the frozen one differs, so a stale file fails a test before it ships.

A sentence renders only for a built claim. With a context, every regime predicate is imported and
called; where one fails, strict mode raises and otherwise the named fallback renders. Without a
context nothing is checked: that is the static fill an engine module makes at registration
(``says=claims.card(id)``), which has no data to check against.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

from turbotab.core.claims._schema import Claim, check, content_hash, fill

REPO = Path(__file__).resolve().parents[3]
STRICT_ENV = "TURBOTAB_CLAIMS_STRICT"
GENERATOR = "python -m turbotab.core.reference.claims"
# The frozen status file, relative to a registry's root: beside this module in the repository.
FROZEN = Path("turbotab", "core", "claims", "status.json")
# The claims' history (every version, and every grade change), relative to a registry's root.
HISTORY = Path("turbotab", "core", "claims", "history.jsonl")


class ClaimError(Exception):
    """A claim cannot do what was asked of it."""


class NotRenderable(ClaimError):
    """A proposed or retracted claim was asked for its sentence."""


class OutsideRegime(ClaimError):
    """The context fails a regime predicate, and no fallback may render in its place."""


def strict_mode(strict: bool | None) -> bool:
    """``strict``, or, when it is None, whether ``TURBOTAB_CLAIMS_STRICT`` is "1"."""
    return os.environ.get(STRICT_ENV) == "1" if strict is None else strict


@lru_cache(maxsize=512)
def _parsed(path: str, mtime_ns: int, size: int) -> ast.Module | None:
    try:
        return ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
    except (OSError, SyntaxError, ValueError):
        return None


def tree(path: Path) -> ast.Module | None:
    """The file's syntax tree, or None when it is missing or does not parse (cached by mtime)."""
    try:
        st = path.stat()
    except OSError:
        return None
    return _parsed(str(path), st.st_mtime_ns, st.st_size) if path.is_file() else None


def _top_level(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """A module's or class's own statements, through ``if`` and ``try`` blocks."""
    for node in body:
        if isinstance(node, ast.If):
            yield from _top_level(node.body)
            yield from _top_level(node.orelse)
        elif isinstance(node, ast.Try):
            for block in (node.body, node.orelse, node.finalbody, *(h.body for h in node.handlers)):
                yield from _top_level(block)
        else:
            yield node


def _bindings(body: list[ast.stmt], name: str) -> Iterator[ast.stmt]:
    """The statements of ``body`` that bind ``name``, in order: a def, a class, an assignment or
    an import."""
    for node in _top_level(body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                yield node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                yield node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((a.asname or a.name.split(".")[0]) == name for a in node.names):
                yield node


def _never_runs(mark: ast.expr) -> bool:
    """Whether ``mark`` is ``pytest.mark.skip``, or ``pytest.mark.xfail`` or ``skipif`` with no
    condition or a literal true one (bare or called, with a reason or not): a mark under which the
    test never asserts. A condition pytest evaluates (an expression, or a string of one) is a
    platform guard, and the test it guards counts."""
    call = mark if isinstance(mark, ast.Call) else None
    func = call.func if call is not None else mark
    if not (isinstance(func, ast.Attribute) and func.attr in ("skip", "skipif", "xfail")):
        return False
    owner = func.value
    if not ((isinstance(owner, ast.Attribute) and owner.attr == "mark")
            or (isinstance(owner, ast.Name) and owner.id == "mark")):
        return False
    if func.attr == "skip" or call is None:
        return True
    conditions = [*call.args, *(k.value for k in call.keywords if k.arg == "condition")]
    return not conditions or any(isinstance(c, ast.Constant) and not isinstance(c.value, str)
                                 and bool(c.value) for c in conditions)


def _marked_never_run(node: ast.stmt | None) -> bool:
    """Whether ``pytestmark``, bound to ``node``, holds a mark under which no test runs."""
    value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
    marks = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
    return any(m is not None and _never_runs(m) for m in marks)


def _defines(body: list[ast.stmt], name: str) -> ast.stmt | None:
    return next(_bindings(body, name), None)


def _last_binding(body: list[ast.stmt], name: str) -> ast.stmt | None:
    """The binding of ``name`` that stands once ``body`` has run."""
    node = None
    for node in _bindings(body, name):
        pass
    return node


def _resolve_in(module: ast.Module | None, dotted: str) -> bool:
    """Whether ``dotted`` (``name`` or ``Class.method``) is defined at the top of ``module``."""
    body: list[ast.stmt] | None = module.body if module is not None else None
    for part in dotted.split("."):
        node = _defines(body, part) if body is not None else None
        if node is None:
            return False
        body = node.body if isinstance(node, ast.ClassDef) else None
    return True


def _test_in(module: ast.Module | None, dotted: str) -> str:
    """Whether ``dotted`` (``name`` or ``Class.name``) is a function ``module`` defines at top
    level or in a class it defines there, what pytest runs, where an enforcer may be any name:
    ``"runs"``; ``"never runs"`` where the function, a class holding it or a ``pytestmark`` over
    it is marked to skip or xfail unconditionally; otherwise ``"missing"``."""
    if module is None:
        return "missing"
    *classes, name = dotted.split(".")
    body = module.body
    never = _marked_never_run(_last_binding(body, "pytestmark"))
    for part in classes:
        node = _last_binding(body, part)
        if not isinstance(node, ast.ClassDef):
            return "missing"
        never = never or any(_never_runs(d) for d in node.decorator_list) or _marked_never_run(
            _last_binding(node.body, "pytestmark"))
        body = node.body
    fn = _last_binding(body, name)
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "missing"
    return "never runs" if never or any(_never_runs(d) for d in fn.decorator_list) else "runs"


@lru_cache(maxsize=8)
def _frozen(data: bytes) -> dict[str, tuple[str, str]] | None:
    """The frozen status file's claims, id to (status, hash), or None when they do not read.
    Cached by the file's bytes, not its timestamp, so a rewrite is never read stale."""
    try:
        return {r["id"]: (r["status"], r["hash"]) for r in json.loads(data)["claims"]}
    except (ValueError, KeyError, TypeError):
        return None


class Registry:
    """Claims by id, checked at registration (``_schema.check``), with the repository they are
    read against, its frozen status file and its history."""

    def __init__(self, claims: Iterable[Claim] = (), *, root: Path | None = None,
                 frozen: Path | None = None, history: Path | None = None) -> None:
        self.root = Path(root) if root is not None else REPO
        self.frozen_path = Path(frozen) if frozen is not None else self.root / FROZEN
        self.history_path = Path(history) if history is not None else self.root / HISTORY
        self._claims: dict[str, Claim] = {}
        for c in claims:
            self.register(c)

    # ── the claims ───────────────────────────────────────────────────────────────────────────

    def register(self, c: Claim) -> Claim:
        check(c)
        if c.id in self._claims and self._claims[c.id] != c:
            raise ValueError(f"{c.id}: two claims declare the same id; an id is never reused")
        self._claims[c.id] = c
        return c

    def __getitem__(self, id: str) -> Claim:
        try:
            return self._claims[id]
        except KeyError:
            raise KeyError(f"no claim {id!r} is registered") from None

    def __contains__(self, id: object) -> bool:
        return id in self._claims

    def __iter__(self) -> Iterator[Claim]:
        """The claims in id order."""
        return iter([self._claims[k] for k in sorted(self._claims)])

    def __len__(self) -> int:
        return len(self._claims)

    # ── what the source holds ────────────────────────────────────────────────────────────────

    def module_file(self, module: str) -> Path | None:
        """A dotted module's source under the root: ``a/b.py`` or ``a/b/__init__.py``."""
        base = self.root.joinpath(*module.split("."))
        for path in (base.with_suffix(".py"), base / "__init__.py"):
            if path.is_file():
                return path
        return None

    def resolves(self, target: str) -> bool:
        """Whether ``"module:name"`` (or ``"module:Class.method"``) is defined at top level."""
        module, _, name = target.partition(":")
        path = self.module_file(module)
        return path is not None and _resolve_in(tree(path), name)

    def test_state(self, test: str) -> str:
        """``"runs"`` where ``"path.py::name"`` (or ``"path.py::Class::name"``) is a function its
        file defines, the last binding of that name (a variable, an import or a class is no
        test); ``"never runs"`` where it is marked to skip or xfail unconditionally; otherwise
        ``"missing"``."""
        path, _, name = test.partition("::")
        name = name.split("[", 1)[0]
        return _test_in(tree(self.root / path), name.replace("::", "."))

    def test_exists(self, test: str) -> bool:
        """Whether the test is a function its file defines that pytest runs to its assertions."""
        return self.test_state(test) == "runs"

    def status(self, id: str) -> str:
        """``retracted`` (declared), ``built`` or ``proposed`` (derived from the source, §6; a
        swarm claim with a part still unsupported is proposed, R1_PLAN.md R1-G). Rendering reads
        :meth:`frozen_status` instead, except in strict mode."""
        c = self[id]
        if c.retracted:
            return "retracted"
        if c.origin == "swarm" and any(p.grade == "unsupported" for p in c.parts):
            return "proposed"
        if all(self.resolves(e) for e in c.enforced_by) and all(
                self.test_exists(t.test) for t in c.tests):
            return "built"
        return "proposed"

    def frozen_status(self, id: str) -> str:
        """The status the generator froze for the claim, which is what renders. A claim the
        frozen file lacks, or lists with another content hash, or a file that is missing or does
        not read, is refused (``NotRenderable``) with the command that writes it."""
        c, path = self[id], self.frozen_path
        try:
            data = path.read_bytes()
        except OSError:
            raise NotRenderable(f"{id}: there is no frozen status file at {path}: run "
                                f"{GENERATOR}") from None
        frozen = _frozen(data)
        if frozen is None:
            raise NotRenderable(f"{id}: the frozen status file {path} does not read: run "
                                f"{GENERATOR}")
        if id not in frozen:
            raise NotRenderable(f"{id} is not in the frozen status file {path}: run {GENERATOR}")
        status, frozen_hash = frozen[id]
        if frozen_hash != content_hash(c):
            raise NotRenderable(f"{id} has changed since {path} froze its status: run "
                                f"{GENERATOR}")
        return status

    # ── the sentences ────────────────────────────────────────────────────────────────────────

    def failed_predicates(self, id: str, ctx: Any) -> list[str]:
        """The regime predicates ``ctx`` fails, each imported and called now."""
        failed = []
        for target in self[id].regime.requires:
            module, _, name = target.partition(":")
            try:
                fn: Any = importlib.import_module(module)
                for part in name.split("."):
                    fn = getattr(fn, part)
            except (ImportError, AttributeError) as e:
                raise ClaimError(f"{id}: the predicate {target} does not resolve ({e})") from e
            held = fn(ctx)
            if not isinstance(held, bool):
                raise TypeError(f"{id}: the predicate {target} returned {held!r}, not a bool")
            if not held:
                failed.append(target)
        return failed

    def _render(self, id: str, register: str, ctx: Any, strict: bool | None) -> str:
        strict = strict_mode(strict)
        seen: list[str] = []
        while True:
            c = self[id]
            status = self.frozen_status(id)
            derived = self.status(id) if strict else status
            if derived != status:
                raise NotRenderable(f"{id}: {self.frozen_path} froze it as {status}, but the "
                                    f"source now makes it {derived} (strict mode): run "
                                    f"{GENERATOR}")
            if status != "built":
                via = f" (the fallback of {seen[-1]})" if seen else ""
                raise NotRenderable(f"{id}{via} is {status}: a claim renders only once built")
            failed = self.failed_predicates(id, ctx) if ctx is not None else []
            if not failed:
                template = getattr(c, register)
                if not template:
                    raise ClaimError(f"{id} has no {register} sentence")
                return fill(template, c.params)
            if strict:
                raise OutsideRegime(f"{id}: the context fails {', '.join(failed)} (strict mode)")
            if not c.fallback:
                raise OutsideRegime(f"{id}: the context fails {', '.join(failed)}, and the claim "
                                    "names no fallback")
            seen.append(id)
            if c.fallback in seen:
                raise OutsideRegime(f"{id}: its fallbacks loop ({' → '.join(seen)} → "
                                    f"{c.fallback})")
            id = c.fallback

    def card(self, id: str, ctx: Any = None, strict: bool | None = None) -> str:
        """The claim's card sentence (the plain register), filled from its parameters."""
        return self._render(id, "card", ctx, strict)

    def methods(self, id: str, ctx: Any = None, strict: bool | None = None) -> str:
        """The claim's methods sentence, filled from its parameters."""
        return self._render(id, "methods", ctx, strict)

    def value(self, id: str, param: str) -> int | float:
        """The parameter's value. A value renders nothing, so it is read whatever the status."""
        c = self[id]
        if param not in c.params:
            raise KeyError(f"{id} has no parameter {param!r}")
        return c.params[param].value


def frozen_records(reg: Registry) -> list[dict[str, str]]:
    """Each claim's id, derived status and content hash, in id order: the frozen file's claims."""
    return [{"id": c.id, "status": reg.status(c.id), "hash": content_hash(c)} for c in reg]


def frozen_text(reg: Registry) -> str:
    """The frozen status file the generator writes for ``reg``."""
    return json.dumps({"claims": frozen_records(reg), "generated_by": GENERATOR}, indent=2,
                      sort_keys=True, ensure_ascii=False) + "\n"
