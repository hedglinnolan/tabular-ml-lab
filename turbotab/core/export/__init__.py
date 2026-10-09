"""The export: the manuscript bundle, its checklists and its replay (V2 definition of done §1:
"export ends every journey"; §3.6: "The export carries the methods section, the participant-flow
and lineage figures, a replayable provenance record, and auto-filled TRIPOD+AI (prediction) or
STROBE-nut (inference) checklists that list their unanswered items. Replaying the record
reproduces the model matrix and the estimates").

* ``source`` — what the export reads of a project (the server builds it; so does the replay);
* ``gate`` — when it refuses, naming what is missing;
* ``methods`` — the methods section from the record's sentences, ordered by STROBE or TRIPOD+AI;
* ``tables`` — Table 2 and its appendix, or the performance table with the declared result;
* ``figures`` — the participant flow and the lineage, as journal-format SVG;
* ``checklists`` — STROBE-nut and TRIPOD+AI, quoted from their sources and filled from the record;
* ``matrix`` — the model matrix's hashes; ``record`` — the provenance record;
* ``bundle`` — the zip; ``replay`` — ``python -m turbotab.replay``;
* ``contract`` — the export's method contract (BLUEPRINT §13);
* ``citations`` — the citation registry and ``refs.bib`` (SIZING X4), re-verified against
  Crossref by ``python -m turbotab.core.export.citations_check``.
"""
