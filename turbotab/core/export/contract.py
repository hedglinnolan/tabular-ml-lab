"""The export's method contract (BLUEPRINT §13), in the one registry (``turbotab.core.contracts``).

The export is not a statistical method, but it is a step every journey ends with (V2 definition of
done §1) and it has a leash: it refuses while a required question is unanswered or the plan is
open, and its replay refuses an input that differs from the record. Its relations are the
consequences a chain test asserts (``tests/acceptance/test_export.py``).
"""
from __future__ import annotations

from turbotab.core import contracts
from turbotab.core.contracts import ContractOption, MethodContract, Relation

KEY = "manuscript_export"
GATE = "turbotab.core.export.gate:check"

CONTRACT = contracts.register_contract(MethodContract(
    key=KEY,
    label="The manuscript bundle and its replay",
    slot="evaluation",
    scope="descriptive",
    needs=("a declared purpose", "the results the bundle reports, computed for the current answers",
           "the input files, as they were read"),
    question="Export the analysis for the manuscript?",
    options=(ContractOption(
        key="bundle",
        label="The manuscript bundle",
        customary=("Methods sections are usually written by hand after the analysis, and the "
                   "reporting checklists filled in by hand (STROBE-nut, Lachat et al. 2016, PLoS "
                   "Med 13:e1002036; TRIPOD+AI, Collins et al. 2024, BMJ 385:e078378)"),
        sound={"inference": ("every sentence is the record's own, the plan's hash is the lock's, "
                             "and Table 2 shows the exposure only, so a reviewer can reconstruct "
                             "the analysis and replay it"),
               "prediction": ("every sentence is the record's own and the result is the declared "
                              "one (selection-corrected without a holdout), so a reviewer can "
                              "reconstruct the analysis and replay it")},
        rung={"inference": "recommended", "prediction": "recommended"}),),
    storyboard=("the record's sentences in force, ordered by the guideline",
                "the results tables and the two figures",
                "the checklist, each item placed or listed as unanswered",
                "the provenance record and its hashes",
                "a replay in a fresh home, compared"),
    relations=(
        Relation("conflicts", "unanswered_question",
                 "A required question is unanswered: the export names it and waits.",
                 rung="refused", exits=("Answer the question the refusal names",),
                 enforced_by=GATE, id="unanswered_question",
                 condition="a question the Router has open or waiting, the substitution curve's "
                           "apart"),
        Relation("conflicts", "plan_open",
                 "The plan is open (under inference not locked; under prediction no result "
                 "declared): the export waits.",
                 rung="refused",
                 exits=("Show the estimates; the first one shown locks the plan",
                        "Declare the final model and open the held-out rows"),
                 enforced_by=GATE, id="plan_open"),
        Relation("conflicts", "input_changed",
                 "An input file differs from the one the record was made from: the export and "
                 "the replay refuse, naming both hashes.",
                 rung="refused", exits=("Give the file the record was made from",),
                 enforced_by="turbotab.core.export.replay:verify_inputs", id="input_changed"),
        Relation("implies", "superseded_folded_out",
                 "The methods hold the record's sentences in force, ordered by the guideline: "
                 "answers superseded before anything was seen are folded out, and every change "
                 "after is kept and marked.",
                 enforced_by="turbotab.core.export.methods:methods_section",
                 id="superseded_folded_out"),
        Relation("implies", "counts_as_they_stand",
                 "A sentence that counts rows (the exclusions', the complete cases') is restated in "
                 "the methods with the counts the participant flow has now, so the two agree after "
                 "a later answer changed those rows; the Record keeps it as said.",
                 enforced_by="turbotab.core.voice:restate", id="counts_as_they_stand"),
        Relation("implies", "plan_with_hash",
                 "The analysis plan is exported with its timestamp and SHA-256, and never called "
                 "prespecified or preregistered.",
                 enforced_by="turbotab.core.plan_lock:plan_export", id="plan_with_hash"),
        Relation("implies", "exposure_rows_only",
                 "Table 2 shows the exposure's rows only; every other coefficient is in the "
                 "appendix of adjustment terms, not effect estimates.",
                 purposes=("inference",), enforced_by="turbotab.core.export.tables:table2",
                 id="exposure_rows_only"),
        Relation("implies", "declared_result_only",
                 "The performance table's one result is the declared one: the selection-corrected "
                 "estimate when the families were chosen with nothing held out, never the "
                 "winner's own score.",
                 purposes=("prediction",), enforced_by="turbotab.core.export.tables:performance",
                 id="declared_result_only"),
        Relation("implies", "unanswered_listed",
                 "Every checklist item is listed with where the bundle answers it, or as "
                 "unanswered — the author must supply this.",
                 enforced_by="turbotab.core.export.checklists:fill", id="unanswered_listed"),
        Relation("implies", "no_fitted_objects",
                 "The bundle carries decisions, hashes and results, never a fitted object or a "
                 "row of data.",
                 enforced_by="turbotab.core.export.bundle:contents", id="no_fitted_objects"),
        Relation("implies", "replay_reproduces",
                 "Replaying the record in a fresh home rebuilds the model matrix byte for byte and "
                 "every reported estimate.",
                 enforced_by="turbotab.core.export.replay:run", id="replay_reproduces"),
    ),
    sources=("Collins et al. 2024, BMJ 385:e078378 (TRIPOD+AI)",
             "Lachat et al. 2016, PLoS Med 13:e1002036 (STROBE-nut)",
             "Westreich & Greenland 2013, Am J Epidemiol 177:292 (the Table 2 fallacy)",
             "Tsamardinos et al. 2018, Mach Learn 107:1895 (BBC-CV)",
             "Gelman & Loken 2013 (forking paths)"),
    run_order=99.0,
    short="the manuscript bundle",
    clause=None,
    place=("after the declaration (MODELING_SEQUENCE §1 row 12): the export ends every journey "
           "(V2 definition of done §1)"),
    scope_note=("It reads every analyzed row's results to report them and informs no modeling "
                "choice: no stage reads anything it computes."),
    leash={"inference": "refused", "prediction": "refused"},
    sentence="turbotab.core.export.methods:reproducibility_sentence",
    package="EXPORT",
))

__all__ = ["CONTRACT", "KEY"]
