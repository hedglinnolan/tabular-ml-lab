"""The legacy packs' shape readings, computed once per stage call (M2_CONTRACT §5, wide data).

``turbotab.packs`` reads the shape of the whole table again every time it asks: ``reframe`` calls
``count_matrix(df)`` once per structural finding, and a 20,000-gene table raises hundreds of
them, so the findings stage spent over two minutes recounting one count matrix; ``suggest``
calls ``likert_block`` and ``count_matrix`` twice each. Inside :func:`remembered`, each reading
is computed once per frame and answered from memory after that, on the calling thread only.
Outside it, the packs behave exactly as written. The legacy module is wrapped, never edited
(BLUEPRINT §0).
"""
from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator

# Readings that depend only on the frame they are given (and their own arguments).
READINGS = ("count_matrix", "likert_block")

_local = threading.local()
_lock = threading.Lock()


def _remembering(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    def reading(df: Any, *args: Any, **kwargs: Any) -> Any:
        memory = getattr(_local, "memory", None)
        if memory is None:
            return fn(df, *args, **kwargs)
        key = (name, id(df), args, tuple(sorted(kwargs.items())))
        if key not in memory:
            # The frame is held with its answer, so its id cannot be reused by another frame.
            memory[key] = (df, fn(df, *args, **kwargs))
        return copy.deepcopy(memory[key][1])  # a caller may change what it was handed

    reading.__wrapped__ = fn  # type: ignore[attr-defined]
    reading.__name__ = getattr(fn, "__name__", name)
    reading.__doc__ = getattr(fn, "__doc__", None)
    return reading


def _install() -> None:
    from turbotab import packs

    with _lock:
        for name in READINGS:
            current = getattr(packs, name)
            if not hasattr(current, "__wrapped__"):
                setattr(packs, name, _remembering(name, current))


@contextmanager
def remembered() -> Iterator[None]:
    """Within this block, on this thread, each pack shape reading is computed once per frame.

    The frames handed to the packs inside the block must not change while it lasts (the
    stages build a frame, read it and drop it).
    """
    _install()
    outer = getattr(_local, "memory", None)
    _local.memory = {} if outer is None else outer
    try:
        yield
    finally:
        _local.memory = outer


__all__ = ["READINGS", "remembered"]
