"""Column lineage traced from a fitted pipeline: raw → adjusted → matrix.

The tracer reads the pipeline itself, never a list of known steps, so a step a later milestone
adds shows up in the lineage without new code here:

* a step with ``lineage()`` (the energy adjusters) says how each output was made, with its
  fitted formula;
* a ``ColumnTransformer`` is traced through each of its parts;
* any other step is traced from ``get_feature_names_out``: same names map one to one, an output
  named ``<input>_<something>`` comes from that input (one-hot, splines), and anything else is
  shown as made from all of the step's inputs.

The operation on a link is named by :data:`OPERATIONS` (a step class → a verb), falling back to
the class name. Wide tables collapse: above :data:`COLLAPSE_AT` raw columns, the columns a step
changed stay visible (up to :data:`SHOWN`) and the rest become one count node per role and
operation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from turbotab.core.consequences import Lineage, LineageLink, LineageNode
from turbotab.core.models.pipeline import ADJUST_STEPS

COLLAPSE_AT = 40
SHOWN = 12
KEPT = "kept"

OPERATIONS: dict[str, str] = {
    "SimpleImputer": "imputed",
    "EnergyAwareImputer": "imputed",  # WP7: a nutrient's fill is its line on total energy
    "OneHotEncoder": "one-hot",
    "MissingLevelEncoder": "one-hot, blank as a level",
    "StandardScaler": "scaled",
    "RobustScaler": "scaled",
    "MinMaxScaler": "scaled",
    "FunctionTransformer": KEPT,
}
ENERGY_OPERATIONS = {
    "pass-through": KEPT,
    "kept": KEPT,
    "residual": "energy-adjusted (residual)",
    "density": "energy-adjusted (density)",
    "partition": "energy-adjusted (partition)",
    "partition-other": "energy-adjusted (partition)",
}


def register_operation(step_class: str, verb: str) -> None:
    """Name what a step class does to a column, e.g. ``register_operation("SplineTransformer", "spline basis")``."""
    OPERATIONS[step_class] = verb


@dataclass
class _Origin:
    sources: list[str]
    ops: list[str] = field(default_factory=list)
    formula: str | None = None


def _verb(step: Any) -> str:
    name = type(step).__name__
    return OPERATIONS.get(name, name)


def _link_op(ops: Sequence[str]) -> str:
    real = [op for op in dict.fromkeys(ops) if op != KEPT]
    return ", ".join(real) if real else KEPT


def _names_out(step: Any, inputs: Sequence[str]) -> list[str]:
    try:
        return [str(n) for n in step.get_feature_names_out(list(inputs))]
    except TypeError:
        return [str(n) for n in step.get_feature_names_out()]


INDICATOR = "missingindicator_"  # SimpleImputer(add_indicator=True) names its indicators so
INDICATOR_OPERATION = "missing indicator"


def _map_by_name(outputs: Sequence[str], inputs: Sequence[str]) -> dict[str, list[str]]:
    by_length = sorted(inputs, key=len, reverse=True)
    known = set(inputs)  # not `in inputs`: quadratic, a second a step at 20,000 columns
    mapping: dict[str, list[str]] = {}
    for out in outputs:
        if out in known:
            mapping[out] = [out]
            continue
        if out.startswith(INDICATOR) and out[len(INDICATOR):] in inputs:
            mapping[out] = [out[len(INDICATOR):]]
            continue
        parent = next((c for c in by_length if out.startswith(f"{c}_")), None)
        mapping[out] = [parent] if parent is not None else list(inputs)
    return mapping


def trace_step(step: Any, inputs: Sequence[str], missing: Mapping[str, int] | None = None
               ) -> dict[str, tuple[list[str], str, str | None]]:
    """``{output: (inputs, operation, formula)}`` for one fitted step."""
    if hasattr(step, "lineage"):
        out: dict[str, tuple[list[str], str, str | None]] = {}
        for entry in step.lineage():
            op = ENERGY_OPERATIONS.get(entry["operation"], entry["operation"])
            formula = entry.get("formula") if op != KEPT else None
            out[str(entry["output"])] = ([str(c) for c in entry["inputs"]], op, formula)
        return out
    if hasattr(step, "transformers_"):
        out = {}
        names_in = [str(c) for c in getattr(step, "feature_names_in_", inputs)]
        for name, part, columns in step.transformers_:
            if part == "drop":
                continue
            cols = [names_in[c] if isinstance(c, int) else str(c) for c in columns]
            if not cols:
                continue
            if part == "passthrough":
                out.update({c: ([c], KEPT, None) for c in cols})
                continue
            verb = _verb(part)
            for o, parents in _map_by_name(_names_out(part, cols), cols).items():
                op = verb
                if verb == "imputed" and o.startswith(INDICATOR) and o not in cols:
                    op = INDICATOR_OPERATION  # was the value missing, as a column of its own
                elif verb == "imputed" and missing is not None and not any(missing.get(p) for p in parents):
                    op = KEPT  # nothing was missing, so nothing was filled
                out[o] = (parents, op, None)
        return out
    verb = _verb(step)
    return {o: (parents, verb, None) for o, parents in _map_by_name(_names_out(step, inputs), inputs).items()}


def _run(steps: Sequence[tuple[str, Any]], columns: Sequence[str],
         missing: Mapping[str, int] | None) -> dict[str, _Origin]:
    current = {c: _Origin([c]) for c in columns}
    for _, step in steps:
        mapping = trace_step(step, list(current), missing)
        nxt: dict[str, _Origin] = {}
        for out, (parents, op, formula) in mapping.items():
            known = [current[p] for p in parents if p in current]
            sources = list(dict.fromkeys(s for origin in known for s in origin.sources))
            ops = [o for origin in known for o in origin.ops] + [op]
            inherited = next((origin.formula for origin in known if origin.formula), None)
            nxt[out] = _Origin(sources or list(parents), ops, formula or (inherited if op == KEPT else None))
        current = nxt
    return current


def trace(steps: Sequence[tuple[str, Any]], raw: Sequence[str], roles: Mapping[str, str],
          missing: Mapping[str, int] | None = None) -> Lineage:
    """The lineage of fitted transformer ``steps`` applied to the ``raw`` input columns.

    Steps named in ``ADJUST_STEPS`` (impute, energy) end in the "adjusted" lane; every later
    step ends in the "matrix" lane. ``missing`` (column → missing count in the fitting rows) lets
    an imputer say which columns it actually filled.
    """
    raw = [str(c) for c in raw]
    adjust = [(n, s) for n, s in steps if n in ADJUST_STEPS]
    later = [(n, s) for n, s in steps if n not in ADJUST_STEPS]
    adjusted = _run(adjust, raw, missing)
    matrix = _run(later, list(adjusted), None)

    def role_of(sources: Sequence[str]) -> Any:
        return next((roles[s] for s in sources if s in roles), None)

    nodes: list[LineageNode] = []
    links: list[LineageLink] = []
    for c in raw:
        nodes.append(LineageNode(id=f"raw:{c}", column=c, lane="raw", role=role_of([c]), label=c))
    for c, origin in adjusted.items():
        nodes.append(LineageNode(id=f"adj:{c}", column=c, lane="adjusted", role=role_of(origin.sources),
                                 label=c, formula=origin.formula))
        op = _link_op(origin.ops)
        links.extend(LineageLink(source=f"raw:{s}", target=f"adj:{c}", operation=op)
                     for s in origin.sources)
    for c, origin in matrix.items():
        adj_sources = origin.sources
        raw_sources = [s for a in adj_sources for s in adjusted.get(a, _Origin([a])).sources]
        nodes.append(LineageNode(id=f"mx:{c}", column=c, lane="matrix", role=role_of(raw_sources),
                                 label=c))
        op = _link_op(origin.ops)
        links.extend(LineageLink(source=f"adj:{a}", target=f"mx:{c}", operation=op) for a in adj_sources)
    lineage = Lineage(nodes=nodes, links=links)
    return collapse(lineage) if len(raw) > COLLAPSE_AT else lineage


def collapse(lineage: Lineage, shown: int = SHOWN) -> Lineage:
    """Keep the columns a step changed (up to ``shown``); fold the rest into count nodes.

    Every node belongs to one raw column (a derived column to its first parent). A raw column is
    *touched* when anything downstream of it is not simply kept or scaled: it is adjusted,
    imputed, expanded or dropped. A folded node stands for the columns of one role whose chains
    look alike (same operations, same shape), so nothing a step did hides in a group that says
    "kept".
    """
    by_id = {n.id: n for n in lineage.nodes}
    incoming: dict[str, list[LineageLink]] = {}
    outgoing: dict[str, list[LineageLink]] = {}
    for link in lineage.links:
        incoming.setdefault(link.target, []).append(link)
        outgoing.setdefault(link.source, []).append(link)

    def owner(nid: str) -> str:
        while by_id[nid].lane != "raw" and incoming.get(nid):
            nid = incoming[nid][0].source
        return nid

    chains: dict[str, list[str]] = {n.id: [] for n in lineage.nodes if n.lane == "raw"}
    for n in lineage.nodes:
        chains.setdefault(owner(n.id), []).append(n.id)
    signature: dict[str, tuple[str, ...]] = {}
    touched: list[str] = []
    for rid, chain in chains.items():
        ops = sorted({link.operation for nid in chain for link in outgoing.get(nid, [])})
        lanes = Counter(by_id[nid].lane for nid in chain)
        signature[rid] = (*ops, *(f"{lane}:{lanes[lane]}" for lane in ("raw", "adjusted", "matrix")))
        changed = any(op not in (KEPT, "scaled") for op in ops)
        if changed or any(lanes[lane] != 1 for lane in ("adjusted", "matrix")):
            touched.append(rid)
    keep = set(touched[:shown])
    kept_ids = {nid for rid in keep for nid in chains[rid]}

    groups: dict[tuple[Any, tuple[str, ...]], list[str]] = {}
    for rid in chains:
        if rid not in keep:
            groups.setdefault((by_id[rid].role, signature[rid]), []).append(rid)
    nodes = [n for n in lineage.nodes if n.id in kept_ids]
    home: dict[str, str] = {nid: nid for nid in kept_ids}
    for g, ((role, _), members) in enumerate(groups.items()):
        label_role = role or "other"
        for lane in ("raw", "adjusted", "matrix"):
            ids = [nid for rid in members for nid in chains[rid] if by_id[nid].lane == lane]
            if not ids:
                continue
            gid = f"{lane}:group:{label_role}:{g}"
            count = len(ids)
            nodes.append(LineageNode(id=gid, column=None, lane=lane, role=role,
                                     label=f"{count:,} {label_role} column{'s' if count != 1 else ''}",
                                     group=label_role, count=count))
            home.update({nid: gid for nid in ids})
    links: list[LineageLink] = []
    seen: set[tuple[str, str, str]] = set()
    for link in lineage.links:
        key = (home[link.source], home[link.target], link.operation)
        if key not in seen:
            seen.add(key)
            links.append(LineageLink(source=key[0], target=key[1], operation=key[2]))
    return Lineage(nodes=nodes, links=links, collapsed=True)


def matrix_columns(lineage: Lineage) -> list[str]:
    return [n.column for n in lineage.nodes if n.lane == "matrix" and n.column is not None]


def missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(c): int(v) for c, v in frame.isna().sum().items()}


__all__ = ["COLLAPSE_AT", "OPERATIONS", "SHOWN", "collapse", "matrix_columns", "missing_counts",
           "register_operation", "trace", "trace_step"]
