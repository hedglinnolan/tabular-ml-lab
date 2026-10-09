"""MC-2a and MC-2b · no code switches on a model family's key or its estimator's class outside that
family's own module (MODEL_FAMILY_CONTRACT C1; the fold-in gate's item 2, C14).

The test reads the syntax tree of every module under ``turbotab/core`` and ``turbotab/server``, not
their literals: "linear" is also an exposure form, the imputation's substantive model and a causal
learner, "elastic_net" is a selection method, "boosted_trees" a causal learner and "cox" a
substantive model, so a scan for the words could never reach zero. It flags:

* **an equality or membership test against a family key:** a registered family's key compared
  with a value that holds a family's key, or tested against a list of family keys (written in
  place, as a module constant or imported). A value holds a family's key when it is a name or an
  attribute one of whose words is ``family``, ``fam`` or ``key`` (``family``, ``f.key``,
  ``model_key``, ``chosen_family``), a value drawn from ``x.models``, or ``x.get("family")``. A
  family's key looked up in ``x.models``, ``models``, ``pipelines`` or ``fitted`` counts too, and so
  does a ``match`` on such a value with a family's key as a case, or its ``startswith``;
* **a table keyed by family:** a dictionary whose keys each are, or hold, a family key, or whose
  values are all family keys (``{"regression": "linear", ...}``); a list of two or more family keys,
  or of entries each named by one (``option("linear", ...)``); and a lookup in a dictionary keyed by
  family with a value that holds a family's key;
* **a check on an estimator class:** ``isinstance`` or ``issubclass`` against one, ``type(x) is``
  or ``==`` one, ``type(x) in`` a list of them, or a ``match`` case on one; an estimator class is a
  scikit-learn regressor or classifier, or the class of a registered family's model step (or one
  built on it);
* **a comparison on a class's ``__name__``,** or its ``startswith``.

A switch in a module that owns every family and class it names is that family's own: a module owns
a family when it declares the family or the class of one of its model steps. Exits that recommend a
family by key are allowed (:data:`EXITS`). Every other switch is a place in :data:`NOT_YET` with the
package that retires it. Each place is pinned: the family keys and classes it names, and how many
switches it holds. A place that gains a switch, or names one more family, fails like a new place;
one that loses some fails until its pin is lowered; and each place is an expected failure until its
package retires it, when it fails as an unexpected pass, so the list only shrinks and must reach
zero. MC-2a retired the four switches the new families hit (``models/explain.py:model_kind``,
``models/cost.py:fit_cost``, ``voice.py:_family_label``, ``models/selection.py:is_flexible``), and
``stages/scales.py``'s family per task reads ``InferenceDecl.default_for``; none may come back.
"""
from __future__ import annotations

import ast
import builtins
import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

import pytest

ROOT = Path(__file__).resolve().parents[3]  # turbotab/
SCANNED = ("core", "server")
KEY_WORDS = {"family", "fam", "key"}  # a name holds a family's key when one of its words is these
FIELD_WORDS = {"family", "fam"}  # a field, ``x["family"]``, holds one when a word of it is these
COLLECTION_NAMES = {"models", "pipelines", "fitted"}  # names of lists or maps keyed by family
COLLECTION_ATTRS = {"models"}  # ``state.models``, ``after.models``, ``d.models``
COLLECTION_KEYS = {"models", "pipelines", "fitted"}  # ``design.objects["pipelines"]``
ENTRY_FIELDS = ("key", "value", "family")  # the field that names a dictionary entry in a list
SIZES = ((100, 2), (10, 20), (10, 600))  # narrow, wide, and wide past the elastic net's exact path
Place = tuple[str, str]  # (path under turbotab/, the enclosing function's or constant's name)


@dataclass(frozen=True)
class Pin:
    """What a listed place holds: the family keys and classes its switches name, and how many."""

    package: str  # the package that retires it, and where it was counted
    names: frozenset[str]
    hits: int


def pin(package: str, hits: int, *names: str) -> Pin:
    return Pin(package, frozenset(names), hits)


# Exits that recommend a family by key, which C1 allows: they name the family a refusal's way
# forward adds to the models chosen. (The lever exit in ``stages/explore.py`` and the exit to
# ``mixed`` in ``models/inference.py`` recommend one without testing for it, so nothing flags them.)
EXITS: dict[Place, Pin] = {
    ("core/methods/substitution.py", "missing_values_block"):
        pin("adds the linear family, whose table draws the imputed copies, to the models chosen",
            1, "linear"),
}

# Every switch still in the code, with the package that retires it and where it was counted:
# MODEL_FAMILY_CONTRACT §3.3's census, V2X_SEAMS row 21, the MC-1 verifier's census, or this test,
# which found the rest.
NOT_YET: dict[Place, Pin] = {
    ("core/decisions.py", "model_families"):
        pin("MC-2b (§3.3: the fallback keys)", 1, "linear", "elastic_net", "boosted_trees"),
    ("core/estimand.py", "_models_fit_the_family"):
        pin("MC-2b (§3.3: family_needs_featurewise)", 1, "featurewise"),
    ("core/method_previews.py", "calibration_numbers"): pin("MC-2b (§3.3)", 2, "linear"),
    ("core/methods/interaction.py", "SUPPORTED"):
        pin("MC-2b (§3.3: inference_decl.product_terms)", 1, "linear", "cox", "proportional_odds"),
    ("core/methods/interaction.py", "_family"): pin("MC-2b (found here)", 1, "linear"),
    ("core/methods/interaction.py", "_measure"):
        pin("MC-2b (V2X_SEAMS row 21)", 2, "cox", "proportional_odds"),
    ("core/methods/interaction.py", "_one"):
        pin("MC-2b (§3.3: SUPPORTED)", 1, "linear", "cox", "proportional_odds"),
    ("core/methods/omics.py", "chain_choices"): pin("MC-2b (found here)", 1, "featurewise"),
    ("core/methods/omics.py", "fit_methods"): pin("MC-2b (found here)", 1, "featurewise"),
    ("core/methods/omics.py", "methods_paragraph"): pin("MC-2b (found here)", 1, "featurewise"),
    ("core/methods/omics.py", "model_clause"):
        pin("MC-2b (§3.3)", 2, "elastic_net", "screened_elastic_net"),
    ("core/models/survey.py", "_BLOCKED_WORDS"):
        pin("MC-2b (the MC-1 verifier: inference_decl.design_based, methods_label)", 1,
            "mixed", "gee", "featurewise", "elastic_net", "boosted_trees"),
    ("core/models/survey.py", "_DESIGN_FAMILY"):
        pin("MC-2b (§3.3: inference_decl.design_based)", 1, "linear", "proportional_odds", "cox"),
    ("core/models/survey.py", "_DESIGN_LABEL"):
        pin("MC-2b (the MC-1 verifier: inference_decl.design_based)", 1,
            "linear", "proportional_odds", "cox"),
    ("core/models/survey.py", "_ESTIMATOR_WORDS"):
        pin("MC-2b (the MC-1 verifier: inference_decl.design_based)", 1,
            "linear", "proportional_odds", "cox"),
    ("core/models/survey.py", "models_sentence"):
        pin("MC-2b (the MC-1 verifier: the tables above, read by key)", 2,
            "linear", "proportional_odds", "cox", "mixed", "gee", "featurewise", "elastic_net",
            "boosted_trees"),
    ("core/plan_previews.py", "population_block"): pin("MC-2b (found here)", 1, "linear"),
    ("core/reference/catalog.py", "FAMILY_LENSES"):
        pin("MC-2b (§3.3: review_lenses)", 1, "linear", "elastic_net", "boosted_trees",
            "featurewise", "proportional_odds", "mixed", "gee", "cox", "screened_elastic_net"),
    ("core/reference/catalog.py", "lenses_of_family"):
        pin("MC-2b (§3.3: FAMILY_LENSES, read by key)", 1, "linear", "elastic_net",
            "boosted_trees", "featurewise", "proportional_odds", "mixed", "gee", "cox",
            "screened_elastic_net"),
    ("core/scales.py", "methods_sentence"):
        pin("MC-2b (the MC-1 verifier: the correction's model, by key)", 1,
            "proportional_odds", "linear"),
    ("core/stages/calibration.py", "calibration_stage"): pin("MC-2b (found here)", 2, "linear"),
    ("core/stages/class_substitution.py", "_plain_multinomial"):
        pin("MC-2b (§3.3)", 1, "LogisticRegression"),
    ("core/stages/effects.py", "SEQUENCE_FAMILIES"):
        pin("MC-2b (§3.3: inference_decl.matrix_table)", 1, "linear", "proportional_odds", "cox",
            "featurewise", "mixed", "gee"),
    ("core/stages/effects.py", "_Run._multiplicity"): pin("MC-2b (found here)", 1, "featurewise"),
    ("core/stages/effects.py", "_Run._one"):
        pin("MC-2b (found here)", 3, "linear", "featurewise"),
    ("core/stages/effects.py", "_Run._with_relative"): pin("MC-2b (found here)", 1, "linear"),
    ("core/stages/effects.py", "_Run.diagnostics"): pin("MC-2b (§3.3)", 3, "cox", "linear"),
    ("core/stages/effects.py", "_Run.family_block"):
        pin("MC-2b (§3.3: SEQUENCE_FAMILIES)", 3, "linear", "proportional_odds", "cox",
            "featurewise", "mixed", "gee"),
    ("core/stages/effects.py", "_Run.marginal"): pin("MC-2b (found here)", 1, "linear"),
    ("core/stages/effects.py", "matrix_table"):
        pin("MC-2b (§3.3)", 6, "featurewise", "cox", "proportional_odds", "linear", "mixed",
            "gee"),
    ("core/stages/evaluation.py", "_shrinkage"): pin("MC-2b (§3.3)", 1, "linear"),
    ("core/stages/modeling.py", "_pooled_curve"): pin("MC-2b (found here)", 1, "linear"),
    ("core/stages/modeling.py", "_tests_only"): pin("MC-2b (found here)", 1, "featurewise"),
    ("core/stages/modeling.py", "fit_stage"):
        pin("MC-2b (§3.3: the collinearity concern)", 1, "linear"),
    ("core/teaching/content.py", "MODELS"):
        pin("MC-2b (§3.3: each family's describe() and bias_terms)", 1, "linear", "elastic_net",
            "boosted_trees", "featurewise", "screened_elastic_net", "proportional_odds", "mixed",
            "gee", "cox"),
}


# ── the families and their estimators ────────────────────────────────────────


def _registry() -> tuple[dict[str, str], dict[str, set[type]]]:
    """{family key: the module that declares it}, and {family key: its model steps' classes}."""
    import turbotab.core.models  # noqa: F401 - registers the families
    from turbotab.core.contracts import contracts
    from turbotab.core.models.base import families

    contracts()  # the omics chain registers a family of its own
    keys, steps = {}, {}
    for family in families():
        keys[family.key] = type(family).__module__
        steps[family.key] = {type(family.build(task, purpose, rows, columns))
                             for task in family.tasks for purpose in family.purposes
                             for rows, columns in SIZES}
    return keys, steps


KEYS, STEPS = _registry()
FAMILY_ESTIMATORS: set[type] = set().union(*STEPS.values())


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


def owned(module: str) -> tuple[set[str], set[type]]:
    """The families ``module`` owns, because it declares the family or the class of one of its
    model steps, and the classes of those families' model steps."""
    own = {k for k, declared_in in KEYS.items()
           if declared_in == module or any(c.__module__ == module for c in STEPS[k])}
    return own, set().union(set(), *(STEPS[k] for k in own))


# ── the scan ─────────────────────────────────────────────────────────────────


@dataclass
class Hit:
    line: int
    text: str
    names: frozenset[str] = frozenset()  # the family keys or classes it switches on


@dataclass
class _Module:
    name: str  # the dotted module name
    tree: ast.Module
    imports: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    constants: dict[str, set[str]] = field(default_factory=dict)  # a constant's strings
    tables: dict[str, set[str]] = field(default_factory=dict)  # a dictionary's keys' family keys


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


def _words(name: str) -> set[str]:
    return {w for w in name.lower().split("_") if w}


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
                if isinstance(node.value, ast.Dict):
                    m.tables[target.id] = _keyed(node.value)
    return m


def _keyed(node: ast.Dict) -> set[str]:
    """The family keys a dictionary is keyed by: each of its keys is one, or holds one (a tuple
    key); empty when any key is neither, as in a table of selection methods or forms."""
    keys = [_strings(k) & KEYS.keys() for k in node.keys if k is not None]
    return set().union(*keys) if keys and all(keys) else set()


def _runtime_keys(value: Any) -> set[str]:
    """The family keys a dictionary is keyed by (as :func:`_keyed`), or a collection holds."""
    def held(item: Any) -> set[str]:
        found = {item} if isinstance(item, str) else (
            {i for i in item if isinstance(i, str)} if isinstance(item, tuple) else set())
        return found & KEYS.keys()

    if isinstance(value, Mapping):
        keys = [held(k) for k in value]
        return set().union(*keys) if keys and all(keys) else set()
    if isinstance(value, (tuple, list, set, frozenset)):
        return set().union(set(), *(held(i) for i in value))
    return set()


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


def _class_like(node: ast.AST) -> bool:
    """A name or attribute spelled as a class (capitalized), the only kind worth resolving."""
    name = node.id if isinstance(node, ast.Name) else (
        node.attr if isinstance(node, ast.Attribute) else "")
    return name[:1].isupper()


def _holds_key(node: ast.AST, drawn: set[str]) -> bool:
    """Whether ``node`` holds one family's key."""
    if isinstance(node, ast.Attribute):
        return bool(_words(node.attr) & KEY_WORDS)
    if isinstance(node, ast.Name):
        return bool(_words(node.id) & KEY_WORDS) or node.id in drawn
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and node.args):
        return any(_words(s) & FIELD_WORDS for s in _strings(node.args[0]))
    if isinstance(node, ast.Subscript):
        return any(_words(s) & FIELD_WORDS for s in _strings(node.slice))
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


def _entry(node: ast.AST) -> str | None:
    """The string that names one entry of a list: the entry itself, a call's first argument
    (``option("linear", ...)``), or a dictionary's ``key``, ``value`` or ``family`` field."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call) and node.args:
        return _entry(node.args[0]) if isinstance(node.args[0], ast.Constant) else None
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if _strings(k) & set(ENTRY_FIELDS) and isinstance(v, ast.Constant):
                return v.value if isinstance(v.value, str) else None
    return None


def table_keys(node: ast.AST) -> set[str]:
    """The family keys a literal table keyed by family holds; empty when it is not one: a
    dictionary of two or more entries whose keys each are or hold a family key, or whose values
    are all family keys, or a list of two or more entries each named by a family key."""
    if isinstance(node, ast.Dict):
        if len(node.keys) >= 2 and _keyed(node):
            return _keyed(node)
        values = [v.value for v in node.values
                  if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        if len(node.values) >= 2 and len(values) == len(node.values) and set(values) <= KEYS.keys():
            return set(values)
        return set()
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)) and len(node.elts) >= 2:
        named = [_entry(e) for e in node.elts]
        if all(n in KEYS for n in named):
            return set(named)  # type: ignore[arg-type]
    return set()


class _Scan(ast.NodeVisitor):
    def __init__(self, m: _Module, own: set[str], own_classes: set[type]):
        self.m = m
        self.own = own  # the family keys this module owns
        self.own_classes = own_classes
        self.scope: list[str] = []
        self.functions = 0  # how deep inside functions the visit is
        self.drawn: list[set[str]] = [set()]  # names drawn from a family collection
        self.named: list[set[str]] = [set()]  # names assigned from a ``__name__``
        self.seen: set[int] = set()  # literals a comparison or a lookup already counted
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
        self.functions += 1
        self.drawn.append(drawn)
        self.named.append(named)
        self.generic_visit(node)
        self.scope.pop()
        self.functions -= 1
        self.drawn.pop()
        self.named.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _constant(self, node: ast.AST, target: ast.AST) -> None:
        """A module or class constant is its own place, named for it."""
        if self.functions or not isinstance(target, ast.Name):
            self.generic_visit(node)
            return
        self.scope.append(target.id)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1:
            self._constant(node, node.targets[0])
        else:
            self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._constant(node, node.target)

    def _hit(self, node: ast.AST, keys: set[str] = frozenset(), classes: set[type] = frozenset(),
             names: set[str] = frozenset()) -> None:
        if (keys or classes) and keys <= self.own and classes <= self.own_classes:
            return  # the family's own module
        where = ".".join(self.scope) or "<module>"
        named = frozenset({*names, *keys, *(c.__name__ for c in classes)})
        self.hits.setdefault(where, []).append(Hit(node.lineno, ast.unparse(node)[:100], named))

    def _listed(self, node: ast.AST) -> set[str]:
        """The family keys a literal, a module constant or an imported constant holds."""
        if isinstance(node, ast.Name) and node.id in self.m.constants:
            return self.m.constants[node.id] & KEYS.keys()
        found = _strings(node) & KEYS.keys()
        if not found and isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if name.isupper() or (isinstance(node, ast.Name) and name in self.m.imports):
                found = _runtime_keys(_resolve(self.m, node))
        return found

    def _classes(self, node: ast.AST) -> set[type]:
        """The estimator classes ``node`` names: one, or a literal list of them."""
        items = node.elts if isinstance(node, (ast.Tuple, ast.List, ast.Set)) else [node]
        return {c for c in (_resolve(self.m, i) for i in items if _class_like(i))
                if is_estimator_class(c)}

    def visit_Compare(self, node: ast.Compare) -> None:
        drawn, named = self.drawn[-1], self.named[-1]
        sides = [node.left, *node.comparators]
        for left, op, right in zip(sides, node.ops, sides[1:]):
            if isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
                for a, b in ((left, right), (right, left)):
                    keys = _strings(a) & KEYS.keys()
                    if keys and _holds_key(b, drawn):
                        self._hit(node, keys)
                    classes = self._classes(b) if _class_like(b) else set()
                    if classes:
                        self._hit(node, classes=classes)
            if isinstance(op, (ast.In, ast.NotIn)):
                keys = _strings(left) & KEYS.keys()
                if keys and _family_collection(right):
                    self._hit(node, keys)
                if _holds_key(left, drawn):
                    listed = self._listed(right)
                    if listed:
                        self.seen.add(id(right))
                        self._hit(node, listed)
                classes = self._classes(right)
                if classes:
                    self._hit(node, classes=classes)
            if any(_class_name(s, named) for s in (left, right)):
                listed = set().union(*(_strings(s) | (self.m.constants.get(s.id, set())
                                                      if isinstance(s, ast.Name) else set())
                                       for s in (left, right)))
                if listed & ESTIMATOR_NAMES:
                    self._hit(node, names=listed & ESTIMATOR_NAMES)
        self.generic_visit(node)

    def _lookup(self, node: ast.AST, container: ast.AST, index: ast.AST) -> None:
        """A dictionary keyed by family, looked up with a value that holds a family's key."""
        drawn = self.drawn[-1]
        items = index.elts if isinstance(index, ast.Tuple) else [index]
        if not any(_holds_key(i, drawn) for i in items):
            return
        keys: set[str] = set()
        if isinstance(container, ast.Dict):
            keys = _keyed(container)
        elif isinstance(container, ast.Name) and container.id in self.m.tables:
            keys = self.m.tables[container.id]
        elif isinstance(container, (ast.Name, ast.Attribute)):
            name = container.id if isinstance(container, ast.Name) else container.attr
            if name.isupper() or name in self.m.imports:  # an imported or a module's constant
                value = _resolve(self.m, container)
                keys = _runtime_keys(value) if isinstance(value, Mapping) else set()
        if keys:
            self.seen.add(id(container))
            self._hit(node, keys)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self._lookup(node, node.value, node.slice)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        named, drawn = self.named[-1], self.drawn[-1]
        func = node.func
        if (isinstance(func, ast.Name) and func.id in ("isinstance", "issubclass")
                and len(node.args) == 2):
            classes = self._classes(node.args[1])
            if classes:
                self._hit(node, classes=classes)
        if isinstance(func, ast.Attribute) and func.attr == "get" and node.args:
            self._lookup(node, func.value, node.args[0])
        if (isinstance(func, ast.Attribute) and func.attr in ("startswith", "endswith")
                and node.args):
            affixes = _strings(node.args[0])
            if _class_name(func.value, named) and any(
                    n.startswith(a) or n.endswith(a) for n in ESTIMATOR_NAMES for a in affixes):
                self._hit(node, names={n for n in ESTIMATOR_NAMES for a in affixes
                                       if n.startswith(a) or n.endswith(a)})
            if _holds_key(func.value, drawn):
                keys = {k for k in KEYS for a in affixes if len(a) >= 3
                        and (k.startswith(a) if func.attr == "startswith" else k.endswith(a))}
                if keys:
                    self._hit(node, keys)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        """One switch per case: on a family's key when the subject holds one, or on a class."""
        subject = _holds_key(node.subject, self.drawn[-1])
        for case in node.cases:
            patterns = list(ast.walk(case.pattern))
            keys = set().union(set(), *(_strings(p.value) for p in patterns
                                        if isinstance(p, ast.MatchValue))) & KEYS.keys()
            if subject and keys:
                self._hit(case.pattern, keys)
            classes = set().union(set(), *(self._classes(p.cls) for p in patterns
                                           if isinstance(p, ast.MatchClass)))
            if classes:
                self._hit(case.pattern, classes=classes)
        self.generic_visit(node)

    def _literal(self, node: ast.AST) -> None:
        if id(node) not in self.seen:
            keys = table_keys(node)
            if keys:
                self._hit(node, keys)
        self.generic_visit(node)

    visit_Dict = visit_Tuple = visit_List = visit_Set = _literal


def scan_source(source: str, module: str) -> dict[str, list[Hit]]:
    """{the enclosing function's or constant's qualified name: its switches} in one module."""
    m = _module(source, module)
    own, own_classes = owned(module)
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


def pin_problems(found: dict[Place, list[Hit]]) -> list[str]:
    """Each listed place whose switches are not its pin, one line each, saying which way."""
    out = []
    for place, held in sorted({**NOT_YET, **EXITS}.items()):
        hits = found.get(place)
        if hits is None:
            continue  # retired: its expected failure says so
        names = frozenset().union(*(h.names for h in hits))
        lines = "; ".join(f"{h.line}: {h.text}" for h in hits)
        if len(hits) > held.hits or names - held.names:
            out.append(f"{place[0]} ({place[1]}) gained a switch, on {sorted(names - held.names)} "
                       f"or one more than its {held.hits}: read the family's declaration instead "
                       f"(MODEL_FAMILY_CONTRACT §1). Now: {lines}")
        elif len(hits) < held.hits or held.names - names:
            out.append(f"{place[0]} ({place[1]}) lost a switch: lower its pin to {len(hits)} "
                       f"on {sorted(names)}. Now: {lines}")
    return out


# ── the tests ────────────────────────────────────────────────────────────────


def test_no_family_key_or_class_switch_is_outside_the_lists():
    unlisted = {place: hits for place, hits in FOUND.items()
                if place not in NOT_YET and place not in EXITS}
    assert not unlisted, (
        "a switch on a family's key or its estimator's class: read the family's declaration "
        "instead (MODEL_FAMILY_CONTRACT §1):\n" + "\n".join(
            f"  {path}:{h.line} ({where}): {h.text}"
            for (path, where), hits in sorted(unlisted.items()) for h in hits))


def test_each_listed_place_holds_exactly_its_pinned_switches():
    """A listed place cannot hide a new switch: one more, or one more family, fails here."""
    assert pin_problems(FOUND) == []


def test_each_allowed_exit_still_recommends_a_family():
    assert sorted(set(EXITS) - set(FOUND)) == [], "an exit is gone: remove it from EXITS"


@pytest.mark.parametrize("place", [
    pytest.param(place, marks=pytest.mark.xfail(strict=True, reason=f"retired by {held.package}"),
                 id=f"{place[0]}:{place[1]}")
    for place, held in sorted(NOT_YET.items())])
def test_each_listed_switch_is_retired(place):
    """Expected to fail until its package retires the switch; then remove it from NOT_YET."""
    assert place not in FOUND, [f"{h.line}: {h.text}" for h in FOUND[place]]


@pytest.mark.xfail(strict=True, reason="retired by MC-2b (§3.3: inference_decl.design_based)")
def test_the_design_based_estimator_is_read_from_the_declaration():
    """``models/survey.py:has_design_estimator`` reads ``inference``'s signature for a ``survey``
    parameter, which no syntax tree shows as a switch. A family whose ``inference`` takes a survey
    but declares no design-based estimator must not be given one."""
    from turbotab.core.models.linear import Linear
    from turbotab.core.models.survey import has_design_estimator

    declared = replace(Linear.inference_decl, design_based=False, default_for=())
    probe = type("Probe", (Linear,), {"key": "probe", "inference_decl": declared})()
    assert not has_design_estimator(probe, "regression")


# ── the scan itself, on code written to be caught or passed ──────────────────

CAUGHT = '''
from sklearn.linear_model import LinearRegression as OLS, LogisticRegression
SUPPORTED = ("linear", "cox")
WORDS = {"mixed": "the mixed model", "gee": "the GEE model"}
FOR_TASK = {"regression": "linear", "ordinal": "proportional_odds"}
OPTIONS = [option("linear", "Linear model"), option("boosted_trees", "Boosted trees")]

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

def identity(model, cls):
    return type(model) is OLS or issubclass(cls, LogisticRegression) or type(model) in (OLS,)

def named(model_key, corr):
    return model_key == "cox" or corr.get("family") in ("gee", "mixed")

def words(key, corr, task):
    return WORDS.get(key, key), {"proportional_odds": "a", "linear": "b"}.get(corr.get("family"))

def matched(family, model):
    match family:
        case "cox" | "gee":
            return 1
    match model:
        case OLS():
            return 2

def prefix(family_key):
    return family_key.startswith("screened")
'''

PASSED = '''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
LEARNERS = ("linear", "lasso", "random_forest", "boosted_trees")
SUBSTANTIVE = {"regression": "linear", "binary": "logistic", "time_to_event": "cox"}
FORMS = [option("linear", "Straight line"), option("spline", "A smooth curve")]
METHODS = {"none": "No selection", "elastic_net": "Elastic net, in each training fold"}

def form(spec, decision, learner, model, fn, kind):
    if spec.form == "linear" or decision.learner in (None, "linear") or learner == "linear":
        return isinstance(model, (Pipeline, StandardScaler)) or type(model) is Pipeline
    if kind == "cox" or {"linear": 1, "spline": 2}.get(spec.form):
        return SUBSTANTIVE[kind]
    for key in spec.models:
        if key in ("ignored",):  # a family's key held, but no family named
            return fn.__name__ == "<lambda>"
    if METHODS.get(key, key):  # "elastic_net" as a selection method, among others
        return None
    return type(model).__name__ == "EnergyAwareImputer"
'''


def test_the_scan_catches_each_kind_of_switch_and_nothing_else():
    """Reference: the switches written into CAUGHT by hand, counted per place, and PASSED's other
    vocabularies ("linear" as a form, a learner and a substantive model; "cox" as a substantive
    model), wrappers, transformers and names."""
    caught = scan_source(CAUGHT, "turbotab.core.not_a_family")
    assert {where: len(hits) for where, hits in caught.items()} == {
        "SUPPORTED": 1, "WORDS": 1, "FOR_TASK": 1, "OPTIONS": 1, "measure": 1, "table": 3,
        "cost": 1, "kind": 2, "identity": 3, "named": 2, "words": 2, "matched": 2, "prefix": 1}
    assert [h.names for h in caught["words"]] == [{"mixed", "gee"}, {"proportional_odds", "linear"}]
    assert [h.names for h in caught["matched"]] == [{"cox", "gee"}, {"LinearRegression"}]
    assert scan_source(PASSED, "turbotab.core.not_a_family") == {}


def test_a_family_may_switch_on_its_own_key_or_class_in_its_own_modules():
    own = 'def f(family):\n    return family == "boosted_trees"\n'
    assert scan_source(own, "turbotab.core.models.boosted_trees") == {}
    assert set(scan_source(own, "turbotab.core.models.linear")) == {"f"}
    # models/wide.py declares the elastic net's wide model step, so the elastic net is its own
    step = ("from turbotab.core.models.elastic_net import PooledElasticNetCV\n"
            "def g(model):\n    return type(model) is not PooledElasticNetCV\n")
    assert scan_source(step, "turbotab.core.models.wide") == {}
    assert set(scan_source(step, "turbotab.core.models.cost")) == {"g"}


def _rescanned(path: str, before: str, after: str) -> dict[Place, list[Hit]]:
    """FOUND with one module rescanned after replacing ``before`` with ``after`` in its source."""
    rel, name, source = next((rel, name, p.read_text(encoding="utf-8"))
                             for rel, name, p in _modules() if rel == path)
    assert source.count(before) == 1, before
    found = {place: hits for place, hits in FOUND.items() if place[0] != rel}
    for where, hits in scan_source(source.replace(before, after), name).items():
        found[(rel, where)] = hits
    return found


def test_a_switch_added_inside_a_listed_place_is_caught():
    """The MC-1 verifier's probes: a new switch inside a listed function, and a listed switch
    widened to more families, each fail as the same switch in a new function would."""
    added = _rescanned(
        "core/stages/evaluation.py",
        '    entry = next((m for m in data.get("models") or [] if m.get("family") == "linear"), '
        'None)\n',
        '    entry = next((m for m in data.get("models") or [] if m.get("family") == "linear"), '
        'None)\n'
        '    both = [m for m in data.get("models") or [] if m.get("family") in ("boosted_trees", '
        '"cox")]\n')
    assert [p.split(" gained")[0] for p in pin_problems(added)] == [
        "core/stages/evaluation.py (_shrinkage)"]
    widened = _rescanned("core/methods/interaction.py",
                         '    if family == "cox" or task == "time_to_event":\n',
                         '    if family in ("cox", "elastic_net", "boosted_trees") or task == '
                         '"time_to_event":\n')
    assert [p.split(" gained")[0] for p in pin_problems(widened)] == [
        "core/methods/interaction.py (_measure)"]
    retired = _rescanned("core/methods/interaction.py",
                         '    if family == "cox" or task == "time_to_event":\n',
                         '    if task == "time_to_event":\n')
    assert [p.split(" lost")[0] for p in pin_problems(retired)] == [
        "core/methods/interaction.py (_measure)"]
