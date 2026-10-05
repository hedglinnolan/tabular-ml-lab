"""The method contract (BLUEPRINT §13): how a domain method enters the pipeline.

Every domain method declares, in this registry, what the engine needs to place it and what a
reviewer needs to check it:

* **slot**: where it runs (ingest · repairs · reshape · eligibility · seal · in-fold · model ·
  evaluation);
* **data scope**: what it may learn from (row-local · reference rows · training fold ·
  descriptive), decided by the lockbox constitution §06's test: does row *i*'s output depend on other
  rows, or on the outcome?
* **needs**: the roles and columns it requires;
* **routing**: its question, where it sits in the modeling sequence, its options each labeled
  customary and sound for each purpose (North star 5), and its leash rung for each purpose (§11.3);
* **storyboard**: its real, labeled steps;
* **sentence**: the function that writes its methods sentence;
* **relations**: what it implies, enables or disables, invalidates, and conflicts with, each
  stated as the condition and its consequence, with the exit a conflict offers.

A chain test asserts that every relation a contract declares fires (BLUEPRINT §13: "A chain test
asserts that every implied consequence appears in the participant flow, the lineage and the methods
sentence"). A contract is registered by the module that implements its method, so importing the
method registers it; :func:`contracts` imports the methods that declare one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

Slot = Literal["ingest", "repairs", "reshape", "eligibility", "seal", "in_fold", "model", "evaluation"]
Scope = Literal["row_local", "reference_rows", "training_fold", "descriptive"]
RelationKind = Literal["implies", "enables", "disables", "invalidates", "conflicts"]
Rung = Literal["offer", "rank_lower", "block_and_record", "refuse", "not_offered"]


@dataclass(frozen=True)
class Option:
    """One answer the method's question offers, with its two labels (North star 5)."""

    key: str
    label: str
    customary: str  # customary in the field, with its source
    sound: dict[str, str]  # purpose -> why it is (or is not) sound for that purpose


@dataclass(frozen=True)
class Relation:
    """How one decision leads to another (BLUEPRINT §13)."""

    id: str
    kind: RelationKind
    when: str
    then: str
    exit: str | None = None  # a conflict's way forward


@dataclass(frozen=True)
class MethodContract:
    key: str
    label: str
    slot: Slot
    scope: Scope
    scope_note: str
    needs: tuple[str, ...]
    question: str
    sequence_step: str
    decision_kind: str
    stage: str
    options: tuple[Option, ...]
    leash: dict[str, Rung]  # purpose -> rung
    storyboard: tuple[str, ...]
    sentence: Callable[..., str]
    relations: tuple[Relation, ...]
    sources: tuple[str, ...] = field(default_factory=tuple)

    def relation(self, relation_id: str) -> Relation:
        return next(r for r in self.relations if r.id == relation_id)


_REGISTRY: dict[str, MethodContract] = {}

# The modules that declare a contract; importing one registers it.
DECLARING_MODULES: tuple[str, ...] = ("turbotab.core.usual_intake",)


def register_contract(contract: MethodContract) -> MethodContract:
    if not contract.relations:
        raise ValueError(f"{contract.key}: a method contract declares its relations")
    ids = [r.id for r in contract.relations]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{contract.key}: relation ids repeat")
    for r in contract.relations:
        if r.kind == "conflicts" and not r.exit:
            raise ValueError(f"{contract.key}: the conflict {r.id} names no exit")
    _REGISTRY[contract.key] = contract
    return contract


def contracts() -> dict[str, MethodContract]:
    """Every registered contract, the declaring modules imported first."""
    import importlib

    for module in DECLARING_MODULES:
        importlib.import_module(module)
    return dict(_REGISTRY)


def contract(key: str) -> MethodContract:
    return contracts()[key]


__all__ = ["DECLARING_MODULES", "MethodContract", "Option", "Relation", "RelationKind", "Rung",
           "Scope", "Slot", "contract", "contracts", "register_contract"]
