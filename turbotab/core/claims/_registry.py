"""The registry: claims by id, their derived status, and their sentences (CLAIMS_REGISTRY.md §2.2,
§6).

Status is derived from the source, read with ``ast`` and never imported, relative to the
registry's ``root`` (the repository, or a planted one in a test): a claim is **built** when every
``enforced_by`` names a function, class or name its module defines at top level and every test it
names is a function its file defines (at top level, or in a class there); otherwise it is
**proposed**. **Retracted** is declared.

A sentence renders only for a built claim. With a context, every regime predicate is imported and
called; where one fails, strict mode raises and otherwise the named fallback renders. Without a
context nothing is checked: that is the static fill an engine module makes at registration
(``says=claims.card(id)``), which has no data to check against.
"""
from __future__ import annotations

import ast
import importlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

from turbotab.core.claims._schema import Claim, check, fill

REPO = Path(__file__).resolve().parents[3]
STRICT_ENV = "TURBOTAB_CLAIMS_STRICT"


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


def _test_in(module: ast.Module | None, dotted: str) -> bool:
    """Whether ``dotted`` (``name`` or ``Class.name``) is a function ``module`` defines at top
    level or in a class it defines there: what pytest runs, where an enforcer may be any name."""
    if module is None:
        return False
    *classes, name = dotted.split(".")
    body = module.body
    for part in classes:
        node = _last_binding(body, part)
        if not isinstance(node, ast.ClassDef):
            return False
        body = node.body
    return isinstance(_last_binding(body, name), (ast.FunctionDef, ast.AsyncFunctionDef))


class Registry:
    """Claims by id, checked at registration (``_schema.check``), with the repository they are
    read against."""

    def __init__(self, claims: Iterable[Claim] = (), *, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else REPO
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

    def test_exists(self, test: str) -> bool:
        """Whether ``"path.py::name"`` (or ``"path.py::Class::name"``) is a function its file
        defines, the last binding of that name: a variable, an import or a class is no test."""
        path, _, name = test.partition("::")
        name = name.split("[", 1)[0]
        return _test_in(tree(self.root / path), name.replace("::", "."))

    def status(self, id: str) -> str:
        """``retracted`` (declared), ``built`` or ``proposed`` (derived, §6)."""
        c = self[id]
        if c.retracted:
            return "retracted"
        if all(self.resolves(e) for e in c.enforced_by) and all(
                self.test_exists(t.test) for t in c.tests):
            return "built"
        return "proposed"

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
            status = self.status(id)
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
