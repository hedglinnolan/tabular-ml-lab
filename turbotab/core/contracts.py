"""The method contract (BLUEPRINT §13): how a domain method enters the pipeline.

One registry for every domain method (wave 1 merged the shapes its packages wrote: the omics
chain's, the scales', the usual-intake method's, the data-in contracts, and in wave 1b the
design-based estimators', multiple imputation's and prediction validation's). Every method
declares, in it, the seven things §13 asks of it:

* **slot** — where it runs: ``ingest · repairs · reshape · eligibility · seal · in_fold · model ·
  evaluation``;
* **scope** — what it may learn from: ``row_local`` (no other row), ``reference_rows`` (technical
  replicates only: pooled QCs, blanks, standards; never a participant or the outcome),
  ``training_fold`` (study rows, so fitted in-fold), ``descriptive`` (may read every row to say
  whether the data are corrupted, but informs no modeling choice) or ``model`` (the outcome model
  itself);
* **needs** — the roles and columns it requires;
* **routing** — its question, its place in MODELING_SEQUENCE §1.1's run order, and its options,
  each labeled *customary* (with a source) and *sound* for each purpose, with its leash rung for
  each purpose (North star 5; §11.3);
* **storyboard** — its real steps, for the transform player;
* **sentence** — the methods clause it contributes, including any departure from convention;
* **relations** — how it leads to other decisions: *implies* (stated, not asked), *enables* /
  *disables*, *invalidates* (re-asked, never silently kept), *conflicts* (refused with an exit),
  and *precedes* (run order). A relation may name the code that makes it fire (``enforced_by``),
  the state that fires it in words (``condition``), a conflict's ways forward (``exits``), and a
  stable name a chain test keys on (``id``).

Where it is asked and what records it: ``decision`` (the decision kind), ``stage`` (the stage that
runs it, when it has its own), ``place`` (its step in MODELING_SEQUENCE §1), ``scope_note`` (why its
scope is what it is), ``parts`` (a method whose parts learn from different rows: scale scoring is
row-local, its reliability training-fold), ``leash`` (the method's own rung per purpose, beside its
options') and ``sentence`` (what writes its methods sentence: a callable, or ``"module:function"``).

The scope is not taken on trust. Lockbox constitution §06's test decides it — "does row *i*'s
output depend on other rows, or on the outcome?" — and :func:`observed_scope` runs that test by
perturbation: change the outcome, change the other study rows, change the reference rows, and see
which of them moves row *i*'s output. The acceptance suite holds every contract's declared scope to
the scope the test observes.

A chain of contracts writes one methods paragraph (:func:`paragraph`): the clauses of the methods
that run before the seal, then the in-fold methods grouped as "within each training fold, A, B
and C were fitted and applied to the held-out fold", then the model's own clause, in the run order
§1.1 fixes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

Slot = Literal["ingest", "repairs", "reshape", "eligibility", "seal", "in_fold", "model", "evaluation"]
Scope = Literal["row_local", "reference_rows", "training_fold", "descriptive", "model"]
Rung = Literal["recommended", "available", "rank_lower", "block_and_record", "refused",
               "not_offered"]
RUNGS: tuple[str, ...] = ("recommended", "available", "rank_lower", "block_and_record", "refused",
                          "not_offered")
RelationKind = Literal["implies", "enables", "disables", "invalidates", "conflicts", "precedes"]
PURPOSES = ("prediction", "inference")
SLOTS: tuple[str, ...] = ("ingest", "repairs", "reshape", "eligibility", "seal", "in_fold", "model",
                          "evaluation")
SCOPES: tuple[str, ...] = ("row_local", "reference_rows", "training_fold", "descriptive", "model")
BEFORE_THE_SEAL = ("ingest", "repairs", "reshape", "eligibility")
# The scopes a step may have and still run before the seal: it reads no other participant's row and
# never the outcome (BLUEPRINT §13; lockbox constitution §06).
PRE_SEAL_SCOPES = ("row_local", "reference_rows", "descriptive")


@dataclass(frozen=True)
class ContractOption:
    """One answer to a contract's question, with its two labels (North star 5)."""

    key: str
    label: str
    customary: str  # customary in the field, with its source
    sound: Mapping[str, str]  # purpose -> why it is (or is not) sound
    rung: Mapping[str, Rung]  # purpose -> the leash rung
    # purpose -> rank, soundest first; left empty, the registry ranks by rung, then as declared
    order: Mapping[str, int] = field(default_factory=dict)

    def for_purpose(self, purpose: str) -> dict[str, Any]:
        p = purpose if purpose in PURPOSES else "prediction"
        return {"key": self.key, "label": self.label, "customary": self.customary,
                "sound": self.sound[p], "rung": self.rung[p], "order": self.order[p]}


@dataclass(frozen=True)
class Relation:
    """How a contract leads to another decision or consequence (BLUEPRINT §13, MODELING_SEQUENCE §2).

    ``target`` is another contract's key or a named consequence (``qc_rows_leave``). ``says`` is
    the sentence the app states when it fires ("because you chose X, Y now…"). ``rung`` is the
    leash a *conflicts* relation enforces. ``when`` names the option keys it fires for (empty: any).
    """

    kind: RelationKind
    target: str
    says: str
    purposes: tuple[str, ...] = PURPOSES
    rung: Rung | None = None
    when: tuple[str, ...] = ()
    exits: tuple[str, ...] = ()  # a conflict's ways forward, in words
    enforced_by: str = ""  # "module:function" that makes it fire
    condition: str = ""  # the state that fires it, in words
    id: str = ""  # a stable name a chain test keys on (the target when empty)

    @property
    def name(self) -> str:
        return self.id or self.target


@dataclass(frozen=True)
class MethodContract:
    key: str
    label: str
    slot: Slot
    scope: Scope
    needs: tuple[str, ...]
    question: str
    options: tuple[ContractOption, ...]
    storyboard: tuple[str, ...]
    relations: tuple[Relation, ...]
    sources: tuple[str, ...] = ()
    run_order: float = 0.0  # its place within its slot (MODELING_SEQUENCE §1.1)
    # A noun phrase for the in-fold grouping ("PQN"), and the clause the method writes on its own
    # when it does not group ("Drift was corrected per batch by QC-RLSC …"): ``clause(run)``.
    short: str = ""
    clause: Callable[[Mapping[str, Any]], str | None] | None = None
    # An option that runs elsewhere than the contract's own slot or scope (batch as a covariate is
    # a term of the outcome model; QRILC needs no other row): option key -> its slot or scope.
    option_slots: Mapping[str, Slot] = field(default_factory=dict)
    option_scopes: Mapping[str, Scope] = field(default_factory=dict)
    option_shorts: Mapping[str, str] = field(default_factory=dict)  # "" groups nothing
    # Where it is asked and what records it (module docstring).
    decision: str = ""
    stage: str = ""
    place: str = ""
    scope_note: str = ""
    parts: Mapping[str, Scope] = field(default_factory=dict)
    leash: Mapping[str, Rung] = field(default_factory=dict)
    sentence: Callable[..., str] | str | None = None
    # The package that declared it (``ESTIMAND``), so its chain test finds every relation it owns.
    package: str = ""

    def short_of(self, option: str | None) -> str:
        return self.option_shorts.get(option or "", self.short)

    def slot_of(self, option: str | None) -> Slot:
        return self.option_slots.get(option or "", self.slot)

    def scope_of(self, option: str | None) -> Scope:
        return self.option_scopes.get(option or "", self.scope)

    def options_for(self, purpose: str) -> list[dict[str, Any]]:
        p = purpose if purpose in PURPOSES else "prediction"
        return [o.for_purpose(p) for o in sorted(self.options, key=lambda o: o.order[p])]

    def option(self, key: str) -> ContractOption:
        for o in self.options:
            if o.key == key:
                return o
        raise KeyError(f"{self.key} has no option {key!r}")

    def relation(self, name: str, target: str | None = None) -> Relation:
        """The relation named ``name`` (its ``id``), or of kind ``name`` toward ``target``."""
        for r in self.relations:
            if (target is None and r.name == name) or (r.kind == name and r.target == target):
                return r
        raise KeyError(f"{self.key} declares no relation {name!r}{f' to {target!r}' if target else ''}")


CONTRACTS: dict[str, MethodContract] = {}


def _ranked(c: MethodContract) -> MethodContract:
    """``c`` with each unranked option ranked per purpose by its rung, then as declared."""
    if all(o.order for o in c.options):
        return c
    import dataclasses

    options = tuple(o if o.order else dataclasses.replace(o, order={
        p: RUNGS.index(o.rung[p]) * 100 + i for p in PURPOSES}) for i, o in enumerate(c.options))
    return dataclasses.replace(c, options=options)


def register_contract(c: MethodContract) -> MethodContract:
    """Declare a method's contract; every field is checked against §13's vocabulary."""
    if c.key in CONTRACTS and CONTRACTS[c.key] is not c and CONTRACTS[c.key].label != c.label:
        raise ValueError(f"{c.key}: two methods declare the same contract key")
    for o in c.options:
        if set(o.sound) != set(PURPOSES) or set(o.rung) != set(PURPOSES):
            raise ValueError(f"{c.key}.{o.key}: every option is labeled for both purposes")
    c = _ranked(c)
    if c.slot not in SLOTS:
        raise ValueError(f"{c.key}: slot {c.slot!r} is not one of {SLOTS}")
    if c.scope not in SCOPES:
        raise ValueError(f"{c.key}: scope {c.scope!r} is not one of {SCOPES}")
    if c.slot in BEFORE_THE_SEAL and c.scope not in PRE_SEAL_SCOPES:
        raise ValueError(f"{c.key}: a {c.scope} method cannot run before the seal ({c.slot})")
    for o in c.options:
        for mapping in (o.sound, o.rung, o.order):
            if set(mapping) != set(PURPOSES):
                raise ValueError(f"{c.key}.{o.key}: every option is labeled for both purposes")
    for r in c.relations:
        if r.kind == "conflicts" and r.rung not in ("refused", "block_and_record"):
            raise ValueError(f"{c.key}: a conflict is refused or blocked and recorded, never silent")
    keys = {o.key for o in c.options}
    for option, slot in c.option_slots.items():
        if option not in keys or slot not in SLOTS:
            raise ValueError(f"{c.key}.{option}: no such option, or slot {slot!r} is unknown")
        if slot in BEFORE_THE_SEAL and c.scope_of(option) not in PRE_SEAL_SCOPES:
            raise ValueError(f"{c.key}.{option}: a {c.scope_of(option)} option cannot run before the seal")
    for option, scope in c.option_scopes.items():
        if option not in keys or scope not in SCOPES:
            raise ValueError(f"{c.key}.{option}: no such option, or scope {scope!r} is unknown")
    for part, scope in c.parts.items():
        if scope not in SCOPES:
            raise ValueError(f"{c.key}: the part {part!r} has an unknown scope {scope!r}")
    for purpose, rung in c.leash.items():
        if purpose not in PURPOSES or rung not in RUNGS:
            raise ValueError(f"{c.key}: the leash {purpose!r} → {rung!r} is outside §11.3's rungs")
    for o in c.options:
        if any(r not in RUNGS for r in o.rung.values()):
            raise ValueError(f"{c.key}.{o.key}: a rung outside §11.3's")
    ids = [r.name for r in c.relations if r.id]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{c.key}: relation ids repeat")
    CONTRACTS[c.key] = c
    return c


def slot_of(key: str, option: str | None = None) -> Slot:
    return CONTRACTS[key].slot_of(option)


def scope_of(key: str, option: str | None = None) -> Scope:
    return CONTRACTS[key].scope_of(option)


# The modules that declare a contract; importing one registers it (each registers at import, as
# model families do), so a reader that needs every contract imports these first.
DECLARING_MODULES: tuple[str, ...] = (
    "turbotab.core.methods.qc_drift", "turbotab.core.methods.omics", "turbotab.core.methods.batch",
    "turbotab.core.scales", "turbotab.core.usual_intake", "turbotab.core.assembly",
    "turbotab.core.codebook", "turbotab.core.models.survey", "turbotab.core.methods.missing",
    "turbotab.core.models.validation", "turbotab.core.stages.effects", "turbotab.core.causal",
    "turbotab.core.time_varying", "turbotab.core.models.explain",
    "turbotab.core.methods.calibration",
)


def contracts() -> dict[str, MethodContract]:
    """Every registered contract, the declaring modules imported first."""
    import importlib

    for module in DECLARING_MODULES:
        importlib.import_module(module)
    return dict(CONTRACTS)


def contract(key: str) -> MethodContract:
    if key not in CONTRACTS:
        contracts()
    return CONTRACTS[key]


def options_for(key: str, purpose: str) -> list[dict[str, Any]]:
    """The contract's options, soundest first for ``purpose``, each with both labels and its rung."""
    return CONTRACTS[key].options_for(purpose)


def rung(key: str, option: str, purpose: str) -> Rung:
    return CONTRACTS[key].option(option).rung[purpose if purpose in PURPOSES else "prediction"]


# ── run order and relations ──────────────────────────────────────────────────


def run_order(keys: Sequence[str] | Mapping[str, str | None]) -> list[str]:
    """The contracts in the order the engine runs them: by slot (each chosen option's own), then by
    their place in it. ``keys`` may map each contract to the option it chose.

    Every *precedes* relation between two of them must agree with that order; one that does not
    is a defect in the registry, raised here rather than run out of order.
    """
    choices = dict(keys) if isinstance(keys, Mapping) else {k: None for k in dict.fromkeys(keys)}
    chosen = [CONTRACTS[k] for k in choices]
    ordered = sorted(chosen, key=lambda c: (SLOTS.index(c.slot_of(choices[c.key])), c.run_order, c.key))
    position = {c.key: i for i, c in enumerate(ordered)}
    for c in ordered:
        for r in c.relations:
            if r.when and choices[c.key] not in r.when:
                continue  # a relation of another option (batch as a covariate precedes no screen)
            if r.kind == "precedes" and r.target in position and position[r.target] < position[c.key]:
                raise ValueError(f"{c.key} must precede {r.target}, but the run order puts it after")
    return [c.key for c in ordered]


@dataclass(frozen=True)
class Firing:
    """A relation that fired in a chain: ``source`` chose ``option``, so ``relation`` holds."""

    source: str
    option: str | None
    relation: Relation

    @property
    def says(self) -> str:
        return self.relation.says


def fired(choices: Mapping[str, str | None], purpose: str,
          consequences: Sequence[str] = ()) -> list[Firing]:
    """The relations a chain fires: each chosen contract's relations that apply to ``purpose`` and
    to the option it chose, whose target is another chosen contract or a named consequence.

    ``choices`` maps a contract key to the option it chose (None: the contract's own default).
    ``consequences`` are the named consequences the chain can show (``qc_rows_leave``)."""
    present = set(choices) | set(consequences)
    out: list[Firing] = []
    for key in run_order(choices):
        option = choices[key]
        for r in CONTRACTS[key].relations:
            if purpose not in r.purposes:
                continue
            if r.when and option not in r.when:
                continue
            if r.target in present:
                out.append(Firing(key, option, r))
    return out


# ── the scope test (lockbox constitution §06) ────────────────────────────────


def observed_scope(fit_transform: Callable[[pd.DataFrame, pd.Series, np.ndarray], pd.DataFrame],
                   frame: pd.DataFrame, reference: np.ndarray, y: np.ndarray, row: int, *,
                   columns: Sequence[str] | None = None, seed: int = 0, tol: float = 1e-9,
                   blank: float = 0.0) -> Scope:
    """What row ``row``'s output depends on, found by changing the rest and watching it.

    ``fit_transform(frame, reference, y)`` fits a method on ``frame`` (whose ``reference`` rows are
    technical replicates and whose outcome is ``y``) and returns its output for every row. The
    test changes, one at a time: the outcome; every other study row's values; every reference row's
    values (``columns``: the measured values it changes; default every numeric column, though a
    design column such as the injection order is better left alone; ``blank``: the share of the
    changed cells also made blank, for a method that reads which values were detected). ``row``
    must be a study row. The answer is the narrowest scope consistent with what
    moved: ``model`` when the outcome moved it, ``training_fold`` when other study rows did,
    ``reference_rows`` when only reference rows did, else ``row_local``.
    """
    reference = np.asarray(reference, dtype=bool)
    if reference[row]:
        raise ValueError("the row the scope test watches must be a study row")
    rng = np.random.default_rng(seed)
    base = _row_out(fit_transform(frame, pd.Series(reference, index=frame.index), y), row)

    def moved(f: pd.DataFrame, yy: np.ndarray) -> bool:
        out = _row_out(fit_transform(f, pd.Series(reference, index=frame.index), yy), row)
        return not np.allclose(out, base, rtol=tol, atol=tol, equal_nan=True)

    shuffled = np.asarray(y).copy()
    rng.shuffle(shuffled)
    if moved(frame, shuffled):
        return "model"
    numeric = (list(columns) if columns is not None else
               [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])])
    others = np.flatnonzero(~reference)
    others = others[others != row]
    if others.size and moved(_scaled(frame, others, numeric, rng, blank), y):
        return "training_fold"
    refs = np.flatnonzero(reference)
    if refs.size and moved(_scaled(frame, refs, numeric, rng, blank), y):
        return "reference_rows"
    return "row_local"


def _scaled(frame: pd.DataFrame, rows: np.ndarray, columns: Sequence[str],
            rng: np.random.Generator, blank: float = 0.0) -> pd.DataFrame:
    """``frame`` with ``rows`` of ``columns`` multiplied by random positive factors (a change that
    keeps every value's sign, so a log or a ratio stays defined), and a ``blank`` share of those
    cells made blank."""
    out = frame.copy()
    factors = rng.uniform(0.5, 2.0, size=(len(rows), len(columns)))
    block = out.iloc[rows][list(columns)].to_numpy(dtype=float) * factors
    if blank:
        block[rng.random(block.shape) < blank] = np.nan
    out.loc[out.index[rows], list(columns)] = block
    return out


def _row_out(out: Any, row: int) -> np.ndarray:
    if isinstance(out, pd.DataFrame):
        values = out.select_dtypes(include=[np.number]).iloc[row].to_numpy(dtype=float)
    else:
        values = np.asarray(out, dtype=float)[row]
    return np.atleast_1d(values)


# ── the methods paragraph ────────────────────────────────────────────────────


def _listed(items: Sequence[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def paragraph(keys: Sequence[str] | Mapping[str, str | None], run: Mapping[str, Any], purpose: str,
              model_clause: str | None = None, details: Sequence[str] = ()) -> str:
    """One methods paragraph for a chain: pre-seal clauses, the in-fold group, the model clause.

    ``keys`` are the chain's contracts (or each mapped to the option it chose); ``run`` is what
    their methods need to speak (their recorded options and counts). Under prediction the in-fold
    group reads "within each training fold, … were fitted and applied to the held-out fold"; under
    inference there is no held-out fold and each method speaks its own clause. ``details`` are
    further sentences (counts, thresholds) that follow the first.
    """
    choices = dict(keys) if isinstance(keys, Mapping) else {k: None for k in dict.fromkeys(keys)}
    ordered = run_order(choices)
    clauses: list[str] = []
    grouped: list[str] = []
    for key in ordered:
        c = CONTRACTS[key]
        if purpose == "prediction" and c.slot_of(choices[key]) == "in_fold":
            if c.short_of(choices[key]):
                grouped.append(c.short_of(choices[key]))
            continue
        said = c.clause(run) if c.clause is not None else None
        if said:
            clauses.append(said)
    if grouped:
        clauses.append(f"within each training fold, {_listed(grouped)} were fitted and applied to "
                       f"the held-out fold")
    if model_clause:
        clauses.append(model_clause)
    if not clauses:
        return " ".join(details)
    first = "; ".join(clauses)
    first = first[:1].upper() + first[1:] + "."
    return " ".join([first, *details])


# The name the scales and usual-intake contracts were written with.
Option = ContractOption

__all__ = [
    "BEFORE_THE_SEAL", "CONTRACTS", "ContractOption", "Firing", "MethodContract", "Option",
    "PRE_SEAL_SCOPES", "PURPOSES", "RUNGS", "Relation", "RelationKind", "Rung", "SCOPES", "SLOTS",
    "Scope", "Slot", "DECLARING_MODULES", "contract", "contracts", "fired", "observed_scope",
    "options_for", "paragraph",
    "register_contract", "run_order", "rung", "scope_of", "slot_of",
]
