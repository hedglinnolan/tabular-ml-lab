"""The ``select_models`` refusal (M1_CONTRACT §2): an unknown family, or one that cannot model
the task. Its exits are the families that can."""
from __future__ import annotations

from typing import Any, Mapping

from turbotab.core.decisions import Refusal, register_validator
from turbotab.core.models.base import families


def _task_of(ctx: Any) -> str | None:
    if ctx is None:
        return None
    if isinstance(ctx, Mapping):
        task = ctx.get("task")
        state = ctx.get("state")
    else:
        task = getattr(ctx, "task", None)
        state = getattr(ctx, "state", None)
    if task is None and state is not None:
        task = getattr(state, "task", None)
    return str(task) if task else None


def check_selection(decision: Any, ctx: Any) -> None:
    known = {f.key: f for f in families()}
    task = _task_of(ctx)
    able = [f for f in families(task)] if task else list(known.values())
    labels = [f.label.lower() for f in able]
    named = labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} and {labels[-1]}"
    exits = [{"label": f"Use {named}",
              "decision": {"kind": "select_models", "models": [f.key for f in able]}}] if able else []
    unknown = [k for k in decision.models if k not in known]
    if unknown:
        raise Refusal("unknown_model_family",
                      f"There is no model family called {', '.join(unknown)}.", exits=exits)
    if task:
        unable = [known[k] for k in decision.models if task not in known[k].tasks]
        if unable:
            raise Refusal("family_cannot_model_task",
                          f"{', '.join(f.label for f in unable)} cannot model a {task} outcome.",
                          exits=exits)


register_validator("select_models", check_selection)

__all__ = ["check_selection"]
