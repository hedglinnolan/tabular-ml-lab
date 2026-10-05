"""The method contract registry (BLUEPRINT §13): how a method enters the pipeline.

"Every domain method declares a **method contract** in a registry, just as model families do, and
the engine enforces it": its **slot** (where it runs), its **data scope** (what it may learn from,
by the lockbox constitution §06's test: does row i's output depend on other rows, or on the
outcome?), its **needs**, its **routing** (its question, where it sits in MODELING_SEQUENCE §1, its
options labeled customary and sound for each purpose, and its leash rung for each purpose), its
**storyboard**, its **sentence**, and its **relations** to other decisions (implies, enables,
disables, invalidates, conflicts).

A relation is a claim about the app's behavior, so each one carries an ``id`` and the **chain
test** of the package that declared it asserts that it fires (BLUEPRINT §13: "A chain test asserts
that every implied consequence appears in the participant flow, the lineage and the methods
sentence"). A contract whose relation no chain test exercises fails that test.

A package registers its contracts on import (:func:`register`); :func:`contracts` lists them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

Slot = Literal["ingest", "repairs", "reshape", "eligibility", "seal", "in_fold", "model",
               "evaluation", "declaration"]
# BLUEPRINT §13's four scopes. Under inference a model is fit on every analyzed row (BLUEPRINT §12
# ruling 3), which is the "training fold" scope with no fold held out.
DataScope = Literal["row_local", "reference_rows", "training_fold", "descriptive"]
RelationKind = Literal["implies", "enables", "disables", "invalidates", "conflicts"]
# BLUEPRINT §11.3's rungs, and what a contract says where it does not apply or must run.
Rung = Literal["refuse", "block_and_record", "rank_lower", "rank_first", "offer", "required",
               "not_applicable"]


@dataclass(frozen=True)
class Relation:
    """How this method's decision leads to another (BLUEPRINT §13): ``kind`` on ``target`` (a
    decision slot, a method, a display), ``when`` it holds and what the app then does."""

    id: str
    kind: RelationKind
    target: str
    when: str
    effect: str


@dataclass(frozen=True)
class Option:
    """One option of the method's question, with north star 5's two labels: where it is
    customary (field and source) and whether it is sound for each purpose (verdict and reason)."""

    key: str
    label: str
    customary: str
    sound: Mapping[str, tuple[str, str]]  # purpose -> (sound | conditional | unsound, reason)


@dataclass(frozen=True)
class MethodContract:
    key: str
    label: str
    slot: Slot
    scope: DataScope
    scope_note: str  # how the lockbox §06 test reads for this method
    needs: tuple[str, ...]
    question: str  # the Router question that routes it, or "stated" (MODELING_SEQUENCE §1)
    step: int  # its row in MODELING_SEQUENCE §1
    rungs: Mapping[str, Rung]  # purpose -> leash rung
    storyboard: tuple[str, ...]
    sentence: str  # where its methods sentence is written (a dotted name)
    relations: tuple[Relation, ...] = ()
    options: tuple[Option, ...] = ()
    sources: tuple[str, ...] = ()
    package: str = ""


_REGISTRY: dict[str, MethodContract] = {}


def register(contract: MethodContract) -> MethodContract:
    """Add ``contract``; a key registered twice must be the same contract."""
    found = _REGISTRY.get(contract.key)
    if found is not None and found != contract:
        raise ValueError(f"the method contract {contract.key!r} is registered twice, differently")
    ids = [r.id for r in contract.relations]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{contract.key!r} repeats a relation id")
    _REGISTRY[contract.key] = contract
    return contract


def contract(key: str) -> MethodContract:
    return _REGISTRY[key]


def contracts(package: str | None = None) -> list[MethodContract]:
    return [c for c in _REGISTRY.values() if package is None or c.package == package]


__all__ = ["DataScope", "MethodContract", "Option", "Relation", "RelationKind", "Rung", "Slot",
           "contract", "contracts", "register"]

