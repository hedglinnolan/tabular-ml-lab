"""The methods reference and the expert review packets, generated from the code (V2 definition of
done §4: "a methods reference generated from the contracts", and "one domain methodologist per
lens, from a per-domain review packet: the methods offered, how they chain, the defaults, and the
exact sentences the app writes").

Nothing here is written by hand that the code already says:

* ``catalog`` — which lens each method contract and model family serves, and the methods v2 offers
  that have no contract entry yet (the gaps), each named with the code that implements it. The
  test fails when a contract or a family is registered without a lens, so the catalog cannot fall
  behind the registry silently;
* ``methods`` — ``python -m turbotab.core.reference.methods`` writes
  ``docs/turbotab-next/reference/METHODS_REFERENCE.md`` from the contract registry
  (``turbotab.core.contracts``), the model-family registry (``turbotab.core.models``) and the
  customary-and-sound labels kept outside the registry (``turbotab.core.custom_sound``);
* ``journeys`` — the reference journeys a packet quotes, driven headlessly through the real server
  by the acceptance harness's drivers (``turbotab.core.tests.acceptance.server_drive``), each
  ending in the export bundle (``turbotab.core.export``) whose methods section and checklist it
  keeps as a capture;
* ``packet`` — ``python -m turbotab.core.reference.packet --lens <lens>`` writes
  ``docs/turbotab-next/review-packets/<lens>.md`` from the registry and the lens's captures.
"""
