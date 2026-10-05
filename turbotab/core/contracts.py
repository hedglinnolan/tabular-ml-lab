"""The method contract registry (BLUEPRINT §13): how a domain method enters the pipeline.

Every domain method the completeness pass adds declares, in one place:

* **slot** — where it runs: ingest · repairs · reshape · eligibility · (seal) · in-fold steps ·
  model · evaluation;
* **data scope** — what each of its parts may learn from: ``row-local``, ``reference rows``,
  ``training fold`` or ``descriptive`` (lockbox constitution §06: does row *i*'s output depend on
  other rows, or on the outcome?);
* **needs** — the roles and columns it requires;
* **routing** — its question, where it sits in the sequence, its options labeled *customary* and
  *sound* for each purpose (North star 5), and its leash rung for each purpose (§11.3);
* **storyboard** — its real, labeled steps for the transform player;
* **sentence** — the function that writes its methods sentence;
* **relations** — how it leads to other decisions: ``implies`` (stated, not asked), ``enables`` /
  ``disables``, ``invalidates`` (re-asked, never silently kept) and ``conflicts`` (refused, with an
  exit that names the way forward). Each relation names the code that makes it fire, so a chain test
  can drive it (MODELING_SEQUENCE §6).

A contract is data: registering one changes no behavior. What it declares is enforced by the code
each relation's ``enforced_by`` names, and checked by the method's chain test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

Slot = Literal["ingest", "repairs", "reshape", "eligibility", "in-fold steps", "model", "evaluation"]
Scope = Literal["row-local", "reference rows", "training fold", "descriptive"]
RelationKind = Literal["implies", "enables", "disables", "invalidates", "conflicts"]
Rung = Literal["recommended", "available", "rank lower", "block and record", "refuse"]


@dataclass(frozen=True)
class Option:
    """One answer the method's question offers, with its two labels and its rung per purpose."""

    key: str
    label: str
    customary: str  # customary in the field, with a source
    sound: Mapping[str, str]  # purpose -> why it is (or is not) sound
    rung: Mapping[str, Rung]  # purpose -> how the guidance treats it


@dataclass(frozen=True)
class Relation:
    kind: RelationKind
    other: str  # the decision or result it relates to
    statement: str
    enforced_by: str  # "module:function" that makes it fire
    exits: tuple[str, ...] = ()  # for a conflict: the ways forward it offers


@dataclass(frozen=True)
class MethodContract:
    key: str
    label: str
    decision: str  # the decision kind that declares it
    slot: Slot
    scope: Mapping[str, Scope]  # part -> its data scope
    needs: tuple[str, ...]
    question: str
    place: str  # where it sits in the sequence (MODELING_SEQUENCE §1)
    options: tuple[Option, ...]
    storyboard: tuple[str, ...]
    sentence: str  # "module:function"
    relations: tuple[Relation, ...] = field(default_factory=tuple)

    def relation(self, kind: RelationKind, other: str) -> Relation:
        return next(r for r in self.relations if r.kind == kind and r.other == other)

    def options_for(self, purpose: str) -> list[dict[str, Any]]:
        """The options for ``purpose``, soundest first: recommended, available, ranked lower, held,
        refused."""
        order = {"recommended": 0, "available": 1, "rank lower": 2, "block and record": 3, "refuse": 4}
        p = purpose if purpose in ("prediction", "inference") else "inference"
        rows = [{"key": o.key, "label": o.label, "customary": o.customary, "sound": o.sound[p],
                 "rung": o.rung[p]} for o in self.options]
        return sorted(rows, key=lambda r: order[r["rung"]])


CONTRACTS: dict[str, MethodContract] = {}


def register_contract(contract: MethodContract) -> MethodContract:
    CONTRACTS[contract.key] = contract
    return contract


__all__ = ["CONTRACTS", "MethodContract", "Option", "Relation", "register_contract"]
