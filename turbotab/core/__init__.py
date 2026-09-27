"""TurboTab Next — the engine host (docs/turbotab-next/BLUEPRINT.md §1).

Submodules are imported explicitly (``from turbotab.core.datastore import
DataStore``); this package file imports nothing, so ``import turbotab.core``
stays cheap for the server, the worker processes and the tests alike.
"""
