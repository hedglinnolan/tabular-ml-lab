"""Tools for the claims registry's gates (R1_PLAN.md R1-G; CLAIMS_REGISTRY.md §8).

* :mod:`research.registry.decoys`: decoy claims generated from real candidates, each false by its
  construction, with why.
* :mod:`research.registry.decoy_queue`: plants them beside the real candidates in a grading queue,
  and reports the false-promotion rate of the two-referee gate over the referees' verdicts.
"""
