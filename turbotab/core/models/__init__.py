"""Model families for TurboTab Next (M1_CONTRACT §7).

Importing this package registers the M1 families — ``linear``, ``elastic_net``,
``boosted_trees`` — the ordinal ``proportional_odds`` (audit WP12a), the families for rows that
repeat within a unit — ``mixed``, ``gee`` (``repeated.py``) — and for a time-to-event outcome —
``cox`` (``survival.py``) — and the consequence previews for ``set_energy_adjustment`` and
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
from turbotab.core.models import featurewise  # noqa: F401,E402 - registers (WP11)
from turbotab.core.models import ordinal  # noqa: F401,E402 - registers proportional_odds (WP12a)
from turbotab.core.models import repeated  # noqa: F401,E402 - registers mixed and gee (WP12)
from turbotab.core.models import survival  # noqa: F401,E402 - registers cox (WP12)
from turbotab.core.models import previews  # noqa: F401,E402 - registers the consequence builders
# The final model declared at the seal's opening (AUDIT_REPORT §5 WP8): its validator and completion.
from turbotab.core.models import selection  # noqa: F401,E402 - registers
# Wave 2, EXPLAIN: the explanations' method contract and the set_explain validators.
from turbotab.core.models import explain  # noqa: F401,E402 - registers

__all__ = [
    "Assessment", "FamilyInfo", "ModelFamily", "Situation", "families", "get_family", "info",
    "rank", "register_family",
]
