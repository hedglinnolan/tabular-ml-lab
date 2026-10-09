"""The sources a model family's declarations may cite, by key (MODEL_FAMILY_CONTRACT C5, §7).

A :class:`~turbotab.core.models.base.Source` names one of these keys, and ``register_family``
refuses a key that is not here. Each entry is MODEL_FAMILY_CONTRACT §7's verified reference,
copied as §7 lists it; a source joins only once §7 lists it, and only when a declaration cites it.
SIZING X4's citation registry (``refs.bib``, every DOI checked against Crossref) replaces this list
when it lands.
"""
from __future__ import annotations

SOURCES: dict[str, str] = {
    "friedman2001": "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting "
                    "machine. Annals of Statistics 29(5). https://doi.org/10.1214/aos/1013203451",
    "hastie2009": "Hastie, T., Tibshirani, R., Friedman, J. (2009). The Elements of Statistical "
                  "Learning, 2nd ed. Springer. §3.4.1, eqs. 3.47 and 3.50. "
                  "https://hastie.su.domains/ElemStatLearn/",
}

__all__ = ["SOURCES"]
