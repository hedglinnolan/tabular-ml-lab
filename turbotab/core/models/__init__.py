"""Model families for TurboTab Next (M1_CONTRACT §7).

Importing this package registers the M1 families — ``linear``, ``elastic_net``,
``boosted_trees`` — and the consequence previews for ``set_energy_adjustment`` and
``select_models``. The ``select_models`` refusal lives with the other validators in
``turbotab/core/decisions.py`` and reads this registry. A later family is one more module that
calls :func:`register_family`.
"""
from __future__ import annotations

from turbotab.core.models.base import (
    Assessment,
    FamilyInfo,
    ModelFamily,
    Situation,
    families,
    get_family,
    info,
    rank,
    register_family,
)
# Registration order is the shelf's tie-break and the order GET /api/models lists them in.
from turbotab.core.models import linear  # noqa: F401,E402 - registers
from turbotab.core.models import elastic_net  # noqa: F401,E402 - registers
from turbotab.core.models import boosted_trees  # noqa: F401,E402 - registers
from turbotab.core.models import previews  # noqa: F401,E402 - registers the consequence builders
# The final model declared at the seal's opening (AUDIT_REPORT §5 WP8): its validator and completion.
from turbotab.core.models import selection  # noqa: F401,E402 - registers

__all__ = [
    "Assessment", "FamilyInfo", "ModelFamily", "Situation", "families", "get_family", "info",
    "rank", "register_family",
]
