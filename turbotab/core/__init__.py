"""TurboTab Next engine host (docs/turbotab-next/BLUEPRINT.md §1).

Deliberately import-free: worker processes are spawned and import this package
on startup, so nothing heavy may happen here.
"""
