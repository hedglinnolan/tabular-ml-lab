"""TurboTab.

``turbotab.core``, ``turbotab.server`` and ``turbotab/frontend`` are TurboTab v2
(docs/turbotab-next/BLUEPRINT.md §1). The modules beside them (``engine``,
``project``, ``packs``, …) are the legacy app's domain modules, kept because
Classic or ``turbotab.core`` imports them (BLUEPRINT §9.1); the legacy app itself
was retired, and its record is ``docs/turbotab/archive/``.
"""

__all__ = ["engine", "project"]
