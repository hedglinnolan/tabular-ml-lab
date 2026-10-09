"""MC-2a and MC-2b · no code switches on a model family's key or its estimator's class outside that
family's own module (MODEL_FAMILY_CONTRACT C1; the fold-in gate's item 2, C14).

The test reads the syntax tree of every module under ``turbotab/core`` and ``turbotab/server``, not
their literals: "linear" is also an exposure form, the imputation's substantive model and a causal
learner, and "elastic_net" is a selection method, so a scan for the words could never reach zero.
It flags:

* **an equality or membership test against a family key:** a registered family's key compared
  with ``x.key``, with a name that holds a family's key (``family``, ``family_key``, ``fam``,
  ``key``), with a value drawn from ``x.models`` or with ``x.get("family")``; a family's key looked
  up in ``x.models``, ``models``, ``pipelines`` or ``fitted``; or a family's key tested against a
  list of family keys, written in place or as a module constant;
* **an ``isinstance`` check against an estimator class:** a scikit-learn regressor or classifier,
  or the class of a registered family's model step (or one built on it);
* **a comparison on a class's ``__name__``,** or its ``startswith``.

A switch in the module that declares every family or class it names is that family's own. Exits
that recommend a family by key are allowed (:data:`EXITS`). Every other switch is a place in
:data:`NOT_YET` with the package that retires it: each is an expected failure until then, and the
day its switch is gone it fails as an unexpected pass, so the list only shrinks and must reach
zero. MC-2a retired the four switches the new families hit (``models/explain.py:model_kind``,
``models/cost.py:fit_cost``, ``voice.py:_family_label``, ``models/selection.py:is_flexible``),
and ``stages/scales.py``'s family per task reads ``InferenceDecl.default_for``; none may come back.
"""
from __future__ import annotations

import ast
import builtins
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pytest

ROOT = Path(__file__).resolve().parents[3]  # turbotab/
SCANNED = ("core", "server")
FAMILY_NAMES = {"family", "family_key", "fam", "key"}  # names that hold a family's key
COLLECTION_NAMES = {"models", "pipelines", "fitted"}  # names of lists or maps keyed by family
COLLECTION_ATTRS = {"models"}  # ``state.models``, ``after.models``, ``d.models``
COLLECTION_KEYS = {"models", "pipelines", "fitted"}  # ``design.objects["pipelines"]``
Place = tuple[str, str]  # (path under turbotab/, the enclosing function's qualified name)

# Exits that recommend a family by key, which C1 allows: they name the family a refusal's way
# forward adds to the models chosen. (The lever exit in ``stages/explore.py`` and the exit to
# ``mixed`` in ``models/inference.py`` recommend one without testing for it, so nothing flags them.)
EXITS: dict[Place, str] = {
    ("core/methods/substitution.py", "missing_values_block"):
        "adds the linear family, whose table draws the imputed copies, to the models chosen",
}

# Every switch still in the code, with the package that retires it and where it was counted:
# MODEL_FAMILY_CONTRACT §3.3's census, V2X_SEAMS row 21, or this test, which found the rest.
NOT_YET: dict[Place, str] = {
    ("core/estimand.py", "_models_fit_the_family"): "MC-2b (§3.3: family_needs_featurewise)",
    ("core/method_previews.py", "calibration_numbers"): "MC-2b (§3.3)",
    ("core/methods/interaction.py", "_family"): "MC-2b (found here)",
    ("core/methods/interaction.py", "_measure"): "MC-2b (V2X_SEAMS row 21)",
    ("core/methods/interaction.py", "_one"): "MC-2b (§3.3: SUPPORTED)",
    ("core/methods/omics.py", "chain_choices"): "MC-2b (found here)",
    ("core/methods/omics.py", "fit_methods"): "MC-2b (found here)",
    ("core/methods/omics.py", "methods_paragraph"): "MC-2b (found here)",
    ("core/methods/omics.py", "model_clause"): "MC-2b (§3.3)",
    ("core/plan_previews.py", "population_block"): "MC-2b (found here)",
    ("core/stages/calibration.py", "calibration_stage"): "MC-2b (found here)",
    ("core/stages/class_substitution.py", "_plain_multinomial"): "MC-2b (§3.3)",
    ("core/stages/effects.py", "_Run._multiplicity"): "MC-2b (found here)",
    ("core/stages/effects.py", "_Run._one"): "MC-2b (found here)",
    ("core/stages/effects.py", "_Run._with_relative"): "MC-2b (found here)",
    ("core/stages/effects.py", "_Run.diagnostics"): "MC-2b (§3.3)",
    ("core/stages/effects.py", "_Run.family_block"): "MC-2b (§3.3: SEQUENCE_FAMILIES)",
    ("core/stages/effects.py", "_Run.marginal"): "MC-2b (found here)",
    ("core/stages/effects.py", "matrix_table"): "MC-2b (§3.3)",
    ("core/stages/evaluation.py", "_shrinkage"): "MC-2b (§3.3)",
    ("core/stages/modeling.py", "_pooled_curve"): "MC-2b (found here)",
    ("core/stages/modeling.py", "_tests_only"): "MC-2b (found here)",
    ("core/stages/modeling.py", "fit_stage"): "MC-2b (§3.3: the collinearity concern)",
}

# §3.3's switches that are tables keyed by family, read by lookup rather than compared, so the
# syntax tree cannot see them: each goes when its package replaces it with a declaration. (The
# others, ``survey.has_design_estimator``'s signature check, ``decisions.model_families``'
# fallback keys and the teaching text by family, are MC-2b's by name.)
TABLES: dict[tuple[str, str], str] = {
    ("turbotab.core.methods.interaction", "SUPPORTED"): "MC-2b (inference_decl.product_terms)",
    ("turbotab.core.models.survey", "_DESIGN_FAMILY"): "MC-2b (inference_decl.design_based)",
    ("turbotab.core.reference.catalog", "FAMILY_LENSES"): "MC-2b (review_lenses)",
    ("turbotab.core.stages.effects", "SEQUENCE_FAMILIES"): "MC-2b (inference_decl.matrix_table)",
}


# ── the families and their estimators ────────────────────────────────────────


def _registry() -> tuple[dict[str, str], set[type]]:
    """{family key: the module that declares it}, and the classes of its model steps."""
    import turbotab.core.models  # noqa: F401 - registers the families
    from turbotab.core.contracts import contracts
    from turbotab.core.models.base import families

    contracts()  # the omics chain registers a family of its own
    keys, classes = {}, set()
    for family in families():
        keys[family.key] = type(family).__module__
        for task in family.tasks:
            for rows, columns in ((100, 2), (10, 20)):  # narrow and wide builds
                classes.add(type(family.build(task, "prediction", rows, columns)))
    return keys, classes


KEYS, FAMILY_ESTIMATORS = _registry()


def is_estimator_class(cls: Any) -> bool:
    """A scikit-learn regressor or classifier, or a registered family's model step's class or one
    built on it; never a step's wrapper or a transformer."""
    from sklearn.base import ClassifierMixin, RegressorMixin

    if not isinstance(cls, type):
        return False
    if cls in FAMILY_ESTIMATORS or any(issubclass(cls, f) for f in FAMILY_ESTIMATORS):
        return True
    return (cls.__module__.startswith("sklearn.") and cls.__module__ != "sklearn.base"
            and issubclass(cls, (RegressorMixin, ClassifierMixin)))


# The names a ``__name__`` comparison would switch on: the families' estimators and their bases.
ESTIMATOR_NAMES = {c.__name__ for f in FAMILY_ESTIMATORS for c in f.__mro__
                   if is_estimator_class(c)}


# ── the scan ─────────────────────────────────────────────────────────────────


@dataclass
class Hit:
    line: int
    text: str


@dataclass
class _Module:
    name: str  # the dotted module name
    tree: ast.Module
    imports: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    constants: dict[str, set[str]] = field(default_factory=dict)


def _strings(node: ast.AST | None) -> set[str]:
    """The string constants a constant or a literal collection holds (a dict's keys)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return set().union(set(), *(_strings(e) for e in node.elts))
    if isinstance(node, ast.Dict):
        return set().union(set(), *(_strings(k) for k in node.keys if k is not None))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("frozenset", "set", "tuple") and node.args):
        return _strings(node.args[0])
    return set()


def _module(source: str, name: str) -> _Module:
    tree = ast.parse(source)
    m = _Module(name=name, tree=tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                m.imports[alias.asname or alias.name] = (node.module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                m.imports[local] = (alias.name if alias.asname else local, None)
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        for target in targets:
            if isinstance(target, ast.Name):
                m.constants[target.id] = _strings(node.value)
    return m


def _resolve(m: _Module, node: ast.AST) -> Any:
    """The object a name or an attribute chain names in module ``m``, or None."""
    if isinstance(node, ast.Attribute):
        base = _resolve(m, node.value)
        return getattr(base, node.attr, None) if base is not None else None
    if not isinstance(node, ast.Name):
        return None
    if node.id not in m.imports and hasattr(builtins, node.id):
        return getattr(builtins, node.id)
    try:
        if node.id in m.imports:
            module, attr = m.imports[node.id]
            if attr is None:
                return importlib.import_module(module)
            found = getattr(importlib.import_module(module), attr, None)
            return found if found is not None else importlib.import_module(f"{module}.{attr}")
        return getattr(importlib.import_module(m.name), node.id, None)
    except Exception:  # noqa: BLE001 - a module that cannot be imported names nothing here
        return None


def _family_key(node: ast.AST, drawn: set[str]) -> bool:
    """Whether ``node`` holds one family's key."""
    if isinstance(node, ast.Attribute):
        return node.attr == "key"
    if isinstance(node, ast.Name):
        return node.id in FAMILY_NAMES or node.id in drawn
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and node.args):
        return _strings(node.args[0]) == {"family"}
    if isinstance(node, ast.Subscript):
        return _strings(node.slice) == {"family"}
    return False


def _family_collection(node: ast.AST) -> bool:
    """Whether ``node`` is a list or a map keyed by family keys."""
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return _family_collection(node.values[0])
    if isinstance(node, ast.Attribute):
        return node.attr in COLLECTION_ATTRS
    if isinstance(node, ast.Name):
        return node.id in COLLECTION_NAMES
    if isinstance(node, ast.Subscript):
        return bool(_strings(node.slice) & COLLECTION_KEYS)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and node.args):
        return bool(_strings(node.args[0]) & COLLECTION_KEYS)
    return False


def _class_name(node: ast.AST, named: set[str]) -> bool:
    """Whether ``node`` is a class's ``__name__``, or a name assigned from one."""
    if isinstance(node, ast.Attribute):
        return node.attr == "__name__"
    return isinstance(node, ast.Name) and node.id in named


class _Scan(ast.NodeVisitor):
    def __init__(self, m: _Module, own: set[str], own_classes: set[type]):
        self.m = m
        self.own = own  # the family keys this module declares
        self.own_classes = own_classes
        self.scope: list[str] = []
        self.drawn: list[set[str]] = [set()]  # names drawn from a family collection
        self.named: list[set[str]] = [set()]  # names assigned from a ``__name__``
        self.hits: dict[str, list[Hit]] = {}

    def _function(self, node: ast.AST) -> None:
        drawn, named = set(), set()
        for inner in ast.walk(node):
            if isinstance(inner, (ast.For, ast.comprehension)) and _family_collection(inner.iter):
                drawn |= {n.id for n in ast.walk(inner.target) if isinstance(n, ast.Name)}
            if isinstance(inner, ast.Assign) and any(
                    isinstance(n, ast.Attribute) and n.attr == "__name__"
                    for n in ast.walk(inner.value)):
                named |= {t.id for t in inner.targets if isinstance(t, ast.Name)}
        self.scope.append(node.name)  # type: ignore[attr-defined]
        self.drawn.append(drawn)
        self.named.append(named)
        self.generic_visit(node)
        self.scope.pop()
        self.drawn.pop()
        self.named.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _hit(self, node: ast.AST, keys: set[str] = frozenset(), classes: set[type] = frozenset()
             ) -> None:
        if (keys or classes) and keys <= self.own and classes <= self.own_classes:
            return  # the family's own module
        where = ".".join(self.scope) or "<module>"
        self.hits.setdefault(where, []).append(Hit(node.lineno, ast.unparse(node)[:100]))

    def visit_Compare(self, node: ast.Compare) -> None:
        drawn, named = self.drawn[-1], self.named[-1]
        sides = [node.left, *node.comparators]
        for left, op, right in zip(sides, node.ops, sides[1:]):
            if isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
                for a, b in ((left, right), (right, left)):
                    keys = _strings(a) & KEYS.keys()
                    if keys and _family_key(b, drawn):
                        self._hit(node, keys)
            if isinstance(op, (ast.In, ast.NotIn)):
                keys = _strings(left) & KEYS.keys()
                if keys and _family_collection(right):
                    self._hit(node, keys)
                if _family_key(left, drawn):
                    listed = _strings(right) | (self.m.constants.get(right.id, set())
                                                if isinstance(right, ast.Name) else set())
                    if listed & KEYS.keys():
                        self._hit(node, listed & KEYS.keys())
            if any(_class_name(s, named) for s in (left, right)):
                listed = set().union(*(_strings(s) | (self.m.constants.get(s.id, set())
                                                      if isinstance(s, ast.Name) else set())
                                       for s in (left, right)))
                if listed & ESTIMATOR_NAMES:
                    self._hit(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        named = self.named[-1]
        func = node.func
        if isinstance(func, ast.Name) and func.id == "isinstance" and len(node.args) == 2:
            given = node.args[1]
            items = given.elts if isinstance(given, ast.Tuple) else [given]
            classes = {c for c in (_resolve(self.m, i) for i in items) if is_estimator_class(c)}
            if classes:
                self._hit(node, classes=classes)
        if (isinstance(func, ast.Attribute) and func.attr in ("startswith", "endswith")
                and _class_name(func.value, named) and node.args):
            affixes = _strings(node.args[0])
            if any(n.startswith(a) or n.endswith(a) for n in ESTIMATOR_NAMES for a in affixes):
                self._hit(node)
        self.generic_visit(node)


def scan_source(source: str, module: str) -> dict[str, list[Hit]]:
    """{the enclosing function's qualified name: its switches} in one module's source."""
    m = _module(source, module)
    own = {k for k, declared_in in KEYS.items() if declared_in == module}
    own_classes = {c for c in FAMILY_ESTIMATORS if c.__module__ == module}
    scan = _Scan(m, own, own_classes)
    scan.visit(m.tree)
    return scan.hits


def _modules() -> Iterator[tuple[str, str, Path]]:
    for base in SCANNED:
        for path in sorted((ROOT / base).rglob("*.py")):
            rel = path.relative_to(ROOT)
            if "tests" in rel.parts:
                continue
            parts = ("turbotab", *rel.with_suffix("").parts)
            name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            yield rel.as_posix(), name, path


def switches() -> dict[Place, list[Hit]]:
    out: dict[Place, list[Hit]] = {}
    for rel, name, path in _modules():
        for where, hits in scan_source(path.read_text(encoding="utf-8"), name).items():
            out[(rel, where)] = hits
    return out


FOUND = switches()


# ── the tests ────────────────────────────────────────────────────────────────


def test_no_family_key_or_class_switch_is_outside_the_lists():
    unlisted = {place: hits for place, hits in FOUND.items()
                if place not in NOT_YET and place not in EXITS}
    assert not unlisted, (
        "a switch on a family's key or its estimator's class: read the family's declaration "
        "instead (MODEL_FAMILY_CONTRACT §1):\n" + "\n".join(
            f"  {path}:{h.line} ({where}): {h.text}"
            for (path, where), hits in sorted(unlisted.items()) for h in hits))


def test_each_allowed_exit_still_recommends_a_family():
    assert sorted(set(EXITS) - set(FOUND)) == [], "an exit is gone: remove it from EXITS"


@pytest.mark.parametrize("place", [
    pytest.param(place, marks=pytest.mark.xfail(strict=True, reason=f"retired by {package}"),
                 id=f"{place[0]}:{place[1]}")
    for place, package in sorted(NOT_YET.items())])
def test_each_listed_switch_is_retired(place):
    """Expected to fail until its package retires the switch; then remove it from NOT_YET."""
    assert place not in FOUND, [f"{h.line}: {h.text}" for h in FOUND[place]]


@pytest.mark.parametrize("table", [
    pytest.param(table, marks=pytest.mark.xfail(strict=True, reason=f"retired by {package}"),
                 id=f"{table[0]}.{table[1]}")
    for table, package in sorted(TABLES.items())])
def test_each_table_keyed_by_family_is_retired(table):
    """Expected to fail until its package replaces the table; then remove it from TABLES."""
    module, name = table
    assert not hasattr(importlib.import_module(module), name)


# ── the scan itself, on code written to be caught or passed ──────────────────

CAUGHT = '''
from sklearn.linear_model import LinearRegression as OLS
SUPPORTED = ("linear", "cox")

def measure(task, family):
    if family == "cox":
        return 1

def table(f, state):
    if f.key in SUPPORTED or "linear" in (state.models or []):
        return [m for m in state.models if m != "featurewise"]

def cost(model):
    return isinstance(model, OLS)

def kind(model):
    name = type(model).__name__
    return name.startswith("HistGradientBoosting") or name in ("LogisticRegression",)
'''

PASSED = '''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def form(spec, decision, learner, model, fn):
    if spec.form == "linear" or decision.learner in (None, "linear") or learner == "linear":
        return isinstance(model, (Pipeline, StandardScaler))
    for key in spec.models:
        if key in ("ignored",):  # a family's key held, but no family named
            return fn.__name__ == "<lambda>"
    return type(model).__name__ == "EnergyAwareImputer"
'''


def test_the_scan_catches_each_kind_of_switch_and_nothing_else():
    """Reference: the switches written into CAUGHT by hand, one function each, and PASSED's other
    vocabularies ("linear" as a form and a learner), wrappers, transformers and names."""
    caught = scan_source(CAUGHT, "turbotab.core.not_a_family")
    assert {where: len(hits) for where, hits in caught.items()} == {
        "measure": 1, "table": 3, "cost": 1, "kind": 2}
    assert scan_source(PASSED, "turbotab.core.not_a_family") == {}


def test_a_family_may_switch_on_its_own_key_in_its_own_module():
    own = 'def f(family):\n    return family == "boosted_trees"\n'
    assert scan_source(own, "turbotab.core.models.boosted_trees") == {}
    assert set(scan_source(own, "turbotab.core.models.linear")) == {"f"}
