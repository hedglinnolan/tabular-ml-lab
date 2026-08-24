"""L41-D — `GUIDED-134`. A stand-in the real registry can never supply.

The defect class, in one line: **a guard satisfies a production dependency with
a fixture stand-in that no project could produce**, so the guard proves the
mechanism and cannot see that nothing feeds it.

The instance is `GUIDED-128` and it is exact:

```python
FIG.bundle({"calibration": payload, "discrimination": {}})
```

A bare dict key, never a registered figure. `figures.bundle` skips unregistered
ids — `REGISTRY.get` returns `None`, `continue`, on a line marked
`# pragma: no cover` **that this very test executed** — while `admissible()`
reads only the *key set*. So `n_admitted` came back 1, the rule looked enforced
for six loops, and no project could ever produce that key.

## Why reading assertions never finds it

**This is not the guard-testing-its-own-description class**, which this project
has hit seven times and where the assertion is a sentence about the code. Here
the assertion is *right*: a confirmatory figure whose companion is present
should be admitted, and it was. The **fixture** is wrong. Nothing in the test's
text is false; the object handed to the collaborator is one production can never
make.

## What this file does

Every registry the app resolves against is read **live** — never listed here,
because a list is the thing that goes stale and the staleness would be silent in
exactly the way this class is. Then the `turbotab/` test tree is walked for
literals handed into something that resolves against one of them, and each is
checked.

A stand-in that does not resolve is either a defect or a different namespace,
and **the difference has to be written down in this file** rather than inferred:
`DELIBERATE` carries every one, with the reason and — the column that matters —
**whether it was load-bearing for its test's claim.**

## The counts are the output, not the pass

`test_the_sweep_reports_its_own_coverage` prints them. A sweep that reports only
its hits has not reported its coverage.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pytest

HERE = Path(__file__).resolve().parent
TESTS = sorted(p for p in HERE.glob("test_*.py") if p.name != Path(__file__).name)
#: The app's own source, which is the *other* surface this lens applies to —
#: see `test_the_same_lens_one_surface_over`.
SOURCE = sorted(p for p in HERE.glob("*.py") if not p.name.startswith("test_"))


# ═════════════════════════════════════════════════════════════════════════════
# The registries, read live
# ═════════════════════════════════════════════════════════════════════════════

def _figure_ids() -> Set[str]:
    from turbotab import figures
    import turbotab.figure_specs                       # noqa: F401 — registers
    # PENDING counts. A declared-but-unbuilt figure is a real entry with a
    # ledger row blocking it, and `figures.resolve` answers for both — so a
    # stand-in naming one is honest, which `discrimination` never was.
    return set(figures.REGISTRY) | set(figures.PENDING)


def _pack_keys() -> Set[str]:
    from turbotab import packs
    return set(packs.PACKS)


def _decision_kinds() -> Set[str]:
    """**Reused, not re-derived.**

    `test_a_recorded_decision_changes_something.recorded_kinds()` already reads
    both forms the dispatcher uses — the `decision.kind == "x"` chain *and* the
    `decision.kind not in {…}` fallthrough — and a second enumerator here would
    be two answers to one question. It matters: the naive one-pattern version
    undercounts by five, which is the mistake `LOOP.md` §03 records the
    adjudicator making at `L35`.
    """
    from turbotab.test_a_recorded_decision_changes_something import recorded_kinds
    return set(recorded_kinds())


def _repeat_kinds() -> Set[str]:
    from turbotab import repeats
    return set(repeats.REPEAT_KINDS)


def _model_keys() -> Set[str]:
    from ml.model_registry import get_registry
    return set(get_registry())


def _routes() -> Set[str]:
    from turbotab import api
    return {_route_shape(r.path) for r in api.app.routes if hasattr(r, "path")}


def _checklist_ids() -> Set[str]:
    from turbotab import figures
    import turbotab.figure_specs                       # noqa: F401
    return {item.id for spec in figures.REGISTRY.values()
            for item in spec.checklist}


def _annotation_keys() -> Set[str]:
    from turbotab import figures
    import turbotab.figure_specs                       # noqa: F401
    return {a.key for spec in figures.REGISTRY.values() for a in spec.annotations}


REGISTRIES: Dict[str, Callable[[], Set[str]]] = {
    "figure": _figure_ids,
    "pack": _pack_keys,
    "decision": _decision_kinds,
    "repeat_kind": _repeat_kinds,
    "model": _model_keys,
    "route": _routes,
    "checklist": _checklist_ids,
}

#: **Registries with no extractor, and why — a count rather than a silence.**
#:
#: `annotation` was in `REGISTRIES` until the first run reported `0 sites` for
#: it, which is the worst number a sweep can print: it reads as *swept and
#: clean* and means *never looked*. It is named here instead.
NO_EXTRACTOR: Dict[str, str] = {
    "annotation": (
        "An annotation key reaches a test as `payload['calibration_intercept']` "
        "or as a membership test over a rendered row, and neither is "
        "distinguishable from any other dict access — a payload has dozens of "
        "keys and only some are annotations. The distinguishing shape is "
        "`Annotation('<key>', …)` in `figure_specs.py`, which is the "
        "DECLARATION the registry is built from, so a sweep over it would "
        "compare a list with itself. What would actually catch the class here "
        "is behavioral rather than lexical: every declared annotation key "
        "should be produced by its figure's own `compute`, and where it is not "
        "the box renders `not estimable`. That is GUIDED-129's territory and "
        "it is a different check from this one."),
}


def _route_shape(path: str) -> str:
    """A route with every parameter blanked, so `/project/{}` and
    `/project/{project_id}` compare equal."""
    return re.sub(r"\{[^}]*\}", "{}", path.split("?")[0]).rstrip("/") or "/"


def _route_matches(written: str, registered: Set[str]) -> bool:
    """Whether a path a test writes is one the app serves.

    **Not string equality**, and this is where the route half of the sweep
    earns its place. A test writes `/project/{}/repair_group/read_as_binary`
    and the app serves `/project/{project_id}/repair_group/{fix_kind}` — the
    literal is a value FILLING a parameter, not a path the app lacks. So the
    comparison is segment by segment, and a segment matches when either side is
    a parameter.

    That leaves the real failure reachable: a segment the app has no route for
    at that position, which is what `/project/{}/a_surface_nobody_wrote` is.

    **Two shapes a test writes that are not whole paths**, both handled here
    rather than allow-listed, because an allow-list entry per occurrence would
    be forty entries describing one thing:

    * a **prefix**, `"/job/"`, which the test concatenates an id onto. It is
      half a path and matches any route that starts with it.
    * a **query string interpolated onto the last segment**,
      `f"…/nutrition/prevalence{query}"`, which leaves a segment reading
      `prevalence{}`. The path is `…/prevalence`; the `{}` is everything after
      the `?`.
    """
    written = written.rstrip("/")
    parts = [p for p in written.split("/") if p != ""]
    # `prevalence{}` is `prevalence` with a query string appended.
    if parts and parts[-1].endswith("{}") and parts[-1] != "{}":
        parts[-1] = parts[-1][:-2]

    for candidate in registered:
        theirs = [p for p in candidate.split("/") if p != ""]
        if len(theirs) == len(parts) and all(
                a == b or a == "{}" or b == "{}"
                for a, b in zip(parts, theirs)):
            return True
        # A prefix: every segment the test wrote matches, and the route has
        # more. `"/job/"` + an id is `/job/{job_id}`.
        if len(theirs) > len(parts) and all(
                a == b or a == "{}" or b == "{}"
                for a, b in zip(parts, theirs[:len(parts)])):
            return True
    return False


# ═════════════════════════════════════════════════════════════════════════════
# The extractors — one per resolver shape
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Site:
    file: str
    line: int
    registry: str
    value: str
    how: str

    @property
    def where(self) -> str:
        return f"{self.file}:{self.line}"


def _str(node: Any) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _called(func: Any) -> Optional[str]:
    return getattr(func, "attr", None) or getattr(func, "id", None)


def _base(node: Any) -> Optional[str]:
    return getattr(node, "attr", None) or getattr(node, "id", None)


def _fstring_shape(node: ast.JoinedStr) -> str:
    return "".join(str(v.value) if isinstance(v, ast.Constant) else "{}"
                   for v in node.values)


_TEXT: Dict[Path, str] = {}
_LINE_STARTS: Dict[Path, List[int]] = {}


def _text(path: Path) -> str:
    if path not in _TEXT:
        _TEXT[path] = path.read_text(encoding="utf-8")
    return _TEXT[path]


def _line_start(path: Path, line: int) -> int:
    """Byte offset of a 1-indexed line. Cached — a `get_source_segment` per
    function definition re-reads the file once per function."""
    if path not in _LINE_STARTS:
        starts, at = [0], 0
        for row in _text(path).split("\n"):
            at += len(row) + 1
            starts.append(at)
        _LINE_STARTS[path] = starts
    starts = _LINE_STARTS[path]
    return starts[min(max(line - 1, 0), len(starts) - 1)]


def sites_in(path: Path) -> List[Site]:
    """Every literal in this file that stands for a registered object."""
    out: List[Site] = []
    tree = ast.parse(_text(path))
    name = path.name

    def add(node, registry, value, how):
        out.append(Site(name, node.lineno, registry, value, how))

    for node in ast.walk(tree):
        # ── figure ids ──────────────────────────────────────────────────────
        # `bundle({...})` — THE INSTANCE. A dict whose keys are figure ids and
        # whose unregistered members `bundle()` silently skips.
        if isinstance(node, ast.Call) and _called(node.func) == "bundle" \
                and node.args and isinstance(node.args[0], ast.Dict):
            for key in node.args[0].keys:
                value = _str(key)
                if value:
                    add(node, "figure", value, "bundle() key")
        # `spec.admissible([...])` — the present-figure list.
        if isinstance(node, ast.Call) and _called(node.func) == "admissible" \
                and node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
            for element in node.args[0].elts:
                value = _str(element)
                if value:
                    add(node, "figure", value, "admissible() member")
        # `companions=(...)` — a declaration rather than a stand-in, and it is
        # where `GUIDED-128` actually lived.
        for kw in getattr(node, "keywords", None) or []:
            if kw.arg == "companions" and isinstance(kw.value, (ast.List, ast.Tuple)):
                for element in kw.value.elts:
                    value = _str(element)
                    if value:
                        add(node, "figure", value, "companions=")

        # ── registry subscripts and gets ────────────────────────────────────
        if isinstance(node, ast.Subscript):
            base, value = _base(node.value), _str(node.slice)
            if value:
                if base == "REGISTRY":
                    add(node, "figure", value, "REGISTRY[]")
                elif base == "PENDING":
                    add(node, "figure", value, "PENDING[]")
                elif base == "PACKS":
                    add(node, "pack", value, "PACKS[]")
        if isinstance(node, ast.Call) and _called(node.func) == "get" and node.args:
            base = _base(getattr(node.func, "value", None))
            value = _str(node.args[0])
            if value and base in ("REGISTRY", "PENDING"):
                add(node, "figure", value, f"{base}.get")
            elif value and base == "PACKS":
                add(node, "pack", value, "PACKS.get")

        # ── decision kinds ──────────────────────────────────────────────────
        #
        # **`kind` is the most overloaded key in this codebase** and the first
        # version of this extractor learned it the expensive way: keyed on the
        # word alone it reported 40-odd false positives across the app's own
        # modules, because an exit has a `kind`, a refusal has a `kind`, a
        # Router question has a `kind`, a missingness route has a `kind` and a
        # figure layer has a `kind`. Five namespaces, one word.
        #
        # So the rule is STRUCTURAL, never lexical: a decision kind is the
        # `kind` of a dict posted to a `/decision` route, and nothing else
        # counts. That is the only place the dispatcher reads.
        if isinstance(node, ast.Call) and _called(node.func) == "post":
            posted_to_decision = any(
                "/decision" in (_str(a) or (_fstring_shape(a)
                                            if isinstance(a, ast.JoinedStr) else ""))
                for a in node.args)
            body = next((kw.value for kw in node.keywords if kw.arg == "json"), None)
            if posted_to_decision and isinstance(body, ast.Dict):
                for key, val in zip(body.keys, body.values):
                    if _str(key) == "kind" and _str(val):
                        add(node, "decision", _str(val), "posted /decision")
                    # The repeat kind lives INSIDE the payload of one decision
                    # kind, which is the whole reason the two are separable.
                    if _str(key) == "payload" and isinstance(val, ast.Dict):
                        for pk, pv in zip(val.keys, val.values):
                            if _str(pk) == "kind" and _str(pv):
                                add(node, "repeat_kind", _str(pv),
                                    "repeat payload")
        # `decide("kind", …)` and `_decide(client, pid, "kind", …)` — the two
        # helper shapes the test tree uses. The kind is the first string
        # argument, and the repeat kind arrives as `kind=` beside it.
        if isinstance(node, ast.Call) and _called(node.func) in ("decide", "_decide"):
            for arg in node.args[:3]:
                value = _str(arg)
                if value:
                    add(node, "decision", value, "decide()")
                    break
            for kw in node.keywords:
                if kw.arg == "kind" and _str(kw.value):
                    add(node, "repeat_kind", _str(kw.value), "decide(kind=)")

        # ── model keys ──────────────────────────────────────────────────────
        if isinstance(node, ast.Call) and _called(node.func) == "train" \
                and len(node.args) >= 2 and isinstance(node.args[1], (ast.List, ast.Tuple)):
            for element in node.args[1].elts:
                value = _str(element)
                if value:
                    add(node, "model", value, "train()")
        # `json={"models": [...]}` posted to `/train`, and nothing else — the
        # word appears in prose keys too (`manuscript.py` has a section called
        # `Model Development` under a `models` key), which the first version of
        # this extractor reported as two missing model keys.
        if isinstance(node, ast.Call) and _called(node.func) == "post":
            to_train = any(
                "/train" in (_str(a) or (_fstring_shape(a)
                                         if isinstance(a, ast.JoinedStr) else ""))
                for a in node.args)
            body = next((kw.value for kw in node.keywords if kw.arg == "json"), None)
            if to_train and isinstance(body, ast.Dict):
                for key, val in zip(body.keys, body.values):
                    if _str(key) == "models" and isinstance(val, (ast.List, ast.Tuple)):
                        for element in val.elts:
                            value = _str(element)
                            if value:
                                add(node, "model", value, "posted /train")

        # ── checklist item ids ──────────────────────────────────────────────
        #
        # Scoped to functions whose source mentions `checklist`, because `.id`
        # is as overloaded as `kind` — a finding has one, a decision has one, a
        # figure has one. Without the scope this reported `wide_repeated_measures`
        # and `category_variants__sex`, which are finding ids from a different
        # namespace entirely.
        # Scoped to a comprehension or loop **whose iterable is a `checklist`**,
        # not to a function that merely mentions the word. `.id` is as
        # overloaded as `kind` — a finding has one, a decision has one — and the
        # looser function-level scope reported
        # `finding["id"] != "pack::metabolomics::pooled_qc"` in
        # `figure_bundle.py` as a missing checklist item, which is a finding id
        # from a different namespace in a function that happens to mention
        # checklists eight lines away.
        #
        # A set literal of ids was considered and REJECTED for the same reason:
        # the only way to tell a checklist id from a finding id inside
        # `{"a", "b"}` is to ask whether it is already in the registry, and an
        # extractor that only collects members it can already resolve is a
        # guard that cannot fail — this file's own subject, arriving inside it.
        iterables: List[ast.AST] = []
        # THE BODY, NOT THE ITERABLE. A generator's `iter` can itself contain a
        # whole comprehension over something else —
        # `all(x for x in next(r for r in admitted if r["id"] == "shrinkage")
        # ["checklist"])` — and walking the node whole reported `shrinkage`, a
        # FIGURE id, as a missing checklist item. The `.id ==` that belongs to
        # this comprehension is in its `elt` and its `ifs`.
        scanned: List[ast.AST] = []
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            iterables = [g.iter for g in node.generators]
            scanned = [node.elt] + [i for g in node.generators for i in g.ifs]
        elif isinstance(node, ast.DictComp):
            iterables = [g.iter for g in node.generators]
            scanned = [node.key, node.value] + [
                i for g in node.generators for i in g.ifs]
        elif isinstance(node, ast.For):
            iterables = [node.iter]
            scanned = list(node.body)
        if any(_base(it) == "checklist"
               or (isinstance(it, ast.Subscript) and _str(it.slice) == "checklist")
               for it in iterables):
            for inner in [n for root in scanned for n in ast.walk(root)]:
                if not isinstance(inner, ast.Compare) or len(inner.comparators) != 1:
                    continue
                left = inner.left
                value = _str(inner.comparators[0])
                is_id = (getattr(left, "attr", None) == "id"
                         or (isinstance(left, ast.Subscript)
                             and _str(left.slice) == "id"))
                if value and is_id:
                    add(inner, "checklist", value, "checklist .id ==")

        # ── routes ──────────────────────────────────────────────────────────
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith(("/project", "/job")):
                add(node, "route", _route_shape(node.value), "literal route")
        if isinstance(node, ast.JoinedStr):
            shape = _fstring_shape(node)
            if shape.startswith(("/project", "/job")):
                add(node, "route", _route_shape(shape), "f-string route")

    return out


def all_sites(paths=TESTS) -> List[Site]:
    out: List[Site] = []
    for path in paths:
        out.extend(sites_in(path))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# What does not resolve, and why — IN THE FILE
# ═════════════════════════════════════════════════════════════════════════════

#: Every stand-in the sweep finds that does **not** resolve, with its reason and
#: the column `GUIDED-134` asks for: **was it load-bearing for the test's
#: claim?** A stand-in that does not resolve and is load-bearing is the defect;
#: one that does not resolve and is decoration is noise; and the difference is
#: what the sweep exists to separate.
#:
#: `load_bearing=True` with no ledger row is a bug in this file.
DELIBERATE: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("route", "/project/{}/a_surface_nobody_wrote"): {
        "load_bearing": False,
        "why": (
            "A route that deliberately does not exist. "
            "`test_the_page_says_what_the_record_says` asserts the page's "
            "behavior when the server has no such surface, so a stand-in that "
            "resolved would destroy the test."),
    },
    ("route", "/project/{}/a_surface_nobody_wrote"): {
        "load_bearing": False,
        "why": (
            "**The non-resolution IS the claim.** "
            "`test_the_page_says_what_the_record_says` asserts what the page "
            "does when the server has no such surface, so a stand-in that "
            "resolved would destroy the test rather than strengthen it. This "
            "is the third category the sweep separates: not a defect, not a "
            "different namespace, but a literal chosen BECAUSE nothing "
            "supplies it."),
    },
    ("decision", "delete_everything"): {
        "load_bearing": False,
        "why": (
            "The same category. `test_api_refuses_an_unknown_decision_kind` "
            "posts it and asserts a 400 — the dispatcher's refusal is the "
            "behavior under test, so the kind must not be one it accepts. "
            "Adding it to the dispatcher would silently turn that test into a "
            "test of nothing."),
    },
    ("figure", "y"): {
        "load_bearing": False,
        "why": (
            "The one place a non-resolving figure id is CORRECT. "
            "`test_an_exploratory_figure_needs_no_companion_and_may_not_declare"
            "_one` builds a throwaway `FigureSpec(id='x', …, companions=('y',))"
            "` inside `pytest.raises` to prove the constructor refuses it. The "
            "claim is about the tier check firing, so `y` never reaches a "
            "registry — the object is destroyed at construction. A stand-in "
            "that resolved would prove less, not more."),
    },
}


def resolves(site: Site, members: Dict[str, Set[str]]) -> bool:
    """Whether this stand-in names something the real registry supplies.

    Routes are matched by shape rather than by equality — see `_route_matches`.
    Everything else is set membership, which is what a registry IS.
    """
    if site.registry == "route":
        return _route_matches(site.value, members["route"])
    return site.value in members[site.registry]


def unresolved(sites: List[Site]) -> List[Site]:
    members = {name: fn() for name, fn in REGISTRIES.items()}
    return [s for s in sites if not resolves(s, members)]


# ═════════════════════════════════════════════════════════════════════════════
# The sweep
# ═════════════════════════════════════════════════════════════════════════════

def test_every_registry_is_read_live_and_is_not_empty():
    """A registry that came back empty would make every check below pass
    vacuously — which is this file's own version of the defect it is about."""
    for name, read in REGISTRIES.items():
        members = read()
        assert members, f"the {name} registry is empty, so nothing is checked"
        assert all(isinstance(m, str) for m in members), name


def test_the_sweep_finds_the_shape_it_was_written_for():
    """**The positive control, and it has to be first.**

    A sweep that found nothing would report a clean tree and prove only that
    the extractor is broken. `bundle()` with literal keys is the exact shape
    `GUIDED-128` hid in, and it is still in the tree — correctly, with `roc`
    now, which is a figure that exists.
    """
    sites = all_sites()
    bundles = [s for s in sites if s.how == "bundle() key"]
    assert bundles, (
        "the extractor found no `bundle()` call with literal keys, which is "
        "the exact shape GUIDED-128 hid in — so it is broken rather than the "
        "tree being clean")
    assert {s.value for s in bundles} >= {"calibration", "roc"}
    assert all(s.registry == "figure" for s in bundles)


@pytest.mark.parametrize("registry", sorted(REGISTRIES))
def test_every_stand_in_resolves_or_is_written_down(registry):
    """**The check `GUIDED-134` asks for**, one registry at a time so a failure
    names which namespace drifted."""
    members = {name: fn() for name, fn in REGISTRIES.items()}
    bad = [s for s in all_sites()
           if s.registry == registry and not resolves(s, members)]
    undeclared = [s for s in bad if (s.registry, s.value) not in DELIBERATE]
    assert not undeclared, (
        "these literals stand for a registered object and no registry supplies "
        "them:\n" + "\n".join(
            f"  {s.where:58s} {s.registry}={s.value!r}  ({s.how})"
            for s in undeclared)
        + "\n\nEither the registry lost an entry, or the test is handing a "
          "collaborator something no project can produce — which is "
          "GUIDED-128's shape. If it is legitimate, put it in DELIBERATE with "
          "its reason and say whether it was load-bearing.")


def test_nothing_load_bearing_is_allowed_through():
    """The column that is the finding. A stand-in that does not resolve **and**
    carried the test's claim is a guard proving a mechanism against an object
    production cannot make — which is not something an allow-list may
    absorb."""
    carrying = sorted(k for k, v in DELIBERATE.items() if v["load_bearing"])
    assert not carrying, (
        f"{carrying} are allow-listed AND load-bearing. That combination is "
        f"the defect itself: the claim rests on something no project can "
        f"produce. It needs a ledger row, not an entry here.")
    for key, entry in DELIBERATE.items():
        assert entry["why"].strip(), key
        assert isinstance(entry["load_bearing"], bool), key


def test_the_allow_list_has_no_stale_entries():
    """An entry for a stand-in that no longer appears, or that now resolves, is
    an allow-list quietly widening. `GUIDED-108`'s own rule: an exception for a
    thing nobody excluded is a decision about nothing that outlives its
    reason."""
    members = {name: fn() for name, fn in REGISTRIES.items()}
    live = {(s.registry, s.value) for s in all_sites()}
    for key in DELIBERATE:
        registry, value = key
        assert key in live, (
            f"{key} is allow-listed and no longer appears anywhere in the test "
            f"tree")
        assert not resolves(Site("", 0, registry, value, ""), members), (
            f"{key} is allow-listed and now resolves; delete the entry")


def test_the_sweep_reports_its_own_coverage(capsys):
    """**The counts are the output.** A sweep that reports only its hits has
    not reported its coverage — `LOOP.md` §10, and it is the check the
    adjudicator has run for four loops.
    """
    sites = all_sites()
    members = {name: fn() for name, fn in REGISTRIES.items()}
    bad = [s for s in sites if not resolves(s, members)]
    load_bearing = [s for s in bad
                    if DELIBERATE.get((s.registry, s.value), {}).get("load_bearing")]

    with capsys.disabled():
        print("\n  ── L41-D · stand-in sweep over turbotab/test_*.py ──")
        print(f"  files walked                 {len(TESTS)}")
        print(f"  sites found                  {len(sites)}")
        print(f"  distinct stand-ins           {len({(s.registry, s.value) for s in sites})}")
        print(f"  resolve in the real registry {len(sites) - len(bad)}")
        print(f"  do NOT resolve               {len(bad)}"
              f"  ({len({(s.registry, s.value) for s in bad})} distinct)")
        print(f"  …of those, load-bearing      {len(load_bearing)}   <- the finding")
        print("  per registry:")
        for name in sorted(REGISTRIES):
            here = [s for s in sites if s.registry == name]
            miss = [s for s in here if not resolves(s, members)]
            print(f"    {name:12s} {len(here):5d} sites  "
                  f"{len({s.value for s in here}):3d} distinct  "
                  f"{len(members[name]):3d} in registry  "
                  f"{len(miss):3d} unresolved")
        # WHAT WAS NOT SWEPT, printed with what was. A registry missing from
        # the list above reads as one that had nothing to find.
        print("  registries with NO extractor:")
        for name, why in sorted(NO_EXTRACTOR.items()):
            print(f"    {name:12s} {why.split('.')[0]}.")

    assert sites, "the sweep found nothing at all"
    assert len(load_bearing) == 0


def test_a_registry_with_no_extractor_is_named_rather_than_reported_as_zero():
    """**The worst number a sweep can print is a zero it did not earn.**

    `annotation` sat in `REGISTRIES` through the first run and reported
    `0 sites`, which reads as *swept and clean* and means *never looked*. It is
    in `NO_EXTRACTOR` now with the reason, and this keeps the two tables from
    overlapping — a name in both would report a zero and an excuse at once.
    """
    assert NO_EXTRACTOR, (
        "no registry is declared unswept, which means either every one has an "
        "extractor or the honest list stopped being kept")
    for name, why in NO_EXTRACTOR.items():
        assert name not in REGISTRIES, (
            f"{name} is both swept and declared unswept")
        assert len(why) > 120, f"{name}'s reason is a shrug: {why!r}"
        # The registry itself still has to be readable — the gap is the
        # extractor, not the knowledge of what the members are.
        assert _annotation_keys() if name == "annotation" else True


# ═════════════════════════════════════════════════════════════════════════════
# What the same lens finds one surface over
# ═════════════════════════════════════════════════════════════════════════════

def test_the_same_lens_one_surface_over():
    """**`LOOP.md` §08.5: did the sweep terminate where the sweeper's attention
    ended?**

    `GUIDED-134` points this lens at the test tree, and the test tree is where
    the *guard* was. It is not where the *defect* was.
    `companions=("discrimination",)` was a literal in `turbotab/figure_specs.py`
    — production source — and the test's stand-in is what stopped anybody
    noticing. Pointed at tests alone, this sweep would have found the
    accomplice and missed the principal.

    So the same extractors run over the app's own modules. `GUIDED-128` is
    fixed, so this passes now; what it buys is that the **declaration** side
    cannot drift again silently, which the ledger row's own `act` does not
    cover.
    """
    sites = all_sites(SOURCE)
    members = {name: fn() for name, fn in REGISTRIES.items()}
    bad = [s for s in sites if not resolves(s, members)]
    # The route extractor is not applied here: a module that DEFINES a route
    # writes the path it is defining, so every one would be reported against a
    # registry built from those same definitions — a check comparing a list
    # with itself.
    bad = [s for s in bad if s.registry != "route"]
    undeclared = [s for s in bad if (s.registry, s.value) not in SOURCE_DELIBERATE]
    assert not undeclared, (
        "the app's own source names these and no registry supplies them:\n"
        + "\n".join(f"  {s.where:52s} {s.registry}={s.value!r} ({s.how})"
                    for s in undeclared)
        + "\n\nThis is where GUIDED-128 actually lived: "
          "`companions=('discrimination',)` in figure_specs.py, with a test "
          "stand-in covering for it.")

    assert any(s.how == "companions=" for s in sites), (
        "no `companions=` declaration was found in the app's source, so the "
        "extractor that would have caught GUIDED-128 at its origin is not "
        "running")


#: Same shape as `DELIBERATE`, for the app's own modules.
SOURCE_DELIBERATE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def test_the_source_sweep_reports_its_coverage(capsys):
    sites = all_sites(SOURCE)
    members = {name: fn() for name, fn in REGISTRIES.items()}
    counted = [s for s in sites if s.registry != "route"]
    bad = [s for s in counted if not resolves(s, members)]
    with capsys.disabled():
        print("\n  ── the same lens, one surface over: turbotab/*.py ──")
        print(f"  modules walked               {len(SOURCE)}")
        print(f"  sites found (routes excluded){len(counted):5d}")
        print(f"  do NOT resolve               {len(bad)}")
        for name in sorted({s.registry for s in counted}):
            here = [s for s in counted if s.registry == name]
            print(f"    {name:12s} {len(here):5d} sites  "
                  f"{len({s.value for s in here}):3d} distinct")
    assert counted, "the source sweep found nothing"
