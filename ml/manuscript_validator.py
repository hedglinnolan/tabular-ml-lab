"""Pre-export manuscript consistency validator."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ml.narrative_engine import _MODEL_NAMES


@dataclass
class ManuscriptValidationCheck:
    """Single validation result.

    `scored` and `declared_because` are `MISC-029`. A check whose inputs are
    absent from the manuscript context still appends a `PASS`, and the panel
    then counts it as a unit of scrutiny — so a Guided draft was shown
    *"13 checks, 0 unmet"* over a set in which three could not have said
    anything else. **The number was the false assertion**; the sentence beside
    it, which renders only when nothing failed, was true about the checks that
    ran.

    **`status` stays two-valued and nothing about the gate moves.** A third
    value was measured and rejected: `to_rows` below collapses any non-`PASS`
    status to the literal `"FAIL"` while `failed_checks` keys on `== "FAIL"`
    and excludes a third value, so a third value serves `n_failed: 0` and
    `passed: True` beside thirteen rows carrying eight `FAIL` — a header
    reading *"13 checks, 8 unmet"* on a clean draft. That is trap #7, *the
    machine-readable form is lossier than the sentence*, committed inside the
    fix for it. Measured: two new fields turn 0 of 132 driven tests red across
    11 files; a third status value turns 6 red across 3.

    **The basis is STRUCTURAL and it comes from the CONTEXT, not from a
    predicate over the payload.** `L64-B` shipped a `scored_when` predicate on
    the figure checklist one registry over, and it does not transfer: run as a
    payload predicate here it declares only five of the eight vacuous checks,
    because *analysis population*, *split counts* and *final predictor count*
    have their inputs **present** and are vacuous for reasons no predicate over
    those inputs can see. So each check states its own basis where it is
    derivable — *this key is absent*, *this list is empty* — and the app
    recomputes it per render. Nothing writes a count down, and the two branches
    of `turbotab.manuscript._counts` answer differently without anyone
    maintaining two lists.
    """

    name: str
    status: str
    location: str
    detail: str
    #: Whether the manuscript could have moved this verdict at all.
    scored: bool = True
    #: Why not, in the words of the absent input. Empty when `scored`.
    declared_because: str = ""


@dataclass
class ManuscriptValidationReport:
    """Validation report for a manuscript export bundle."""

    checks: List[ManuscriptValidationCheck]

    @property
    def failed_checks(self) -> List[ManuscriptValidationCheck]:
        # DELIBERATELY UNCHANGED, and `passed` with it. `passed` gates the
        # Classic download and `AGENT_ONBOARD.md` §08 check 2 is that a
        # threshold does not move in the same loop as the change that
        # pressured it. No declared check reports `FAIL` on either door today —
        # `turbotab/manuscript.py` serves that as `n_declared_that_failed` and
        # a test asserts it is zero — so excluding them here would change
        # nothing observable while changing what a gate means.
        return [check for check in self.checks if check.status == "FAIL"]

    @property
    def passed(self) -> bool:
        return not self.failed_checks

    @property
    def scored_checks(self) -> List[ManuscriptValidationCheck]:
        """The checks the manuscript could have moved. `MISC-029`."""
        return [check for check in self.checks if check.scored]

    @property
    def declared_checks(self) -> List[ManuscriptValidationCheck]:
        """The checks that were decided before the draft was read."""
        return [check for check in self.checks if not check.scored]

    def to_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "Status": "PASS" if check.status == "PASS" else "FAIL",
                "Check": check.name,
                "Location": check.location,
                "Detail": check.detail,
                "scored": check.scored,
                "declared_because": check.declared_because,
            }
            for check in self.checks
        ]


#: One sentence for one kind of absence, so a declared check reads the same
#: wherever it appears. Borrowed in shape from `turbotab.manuscript`'s
#: `_RENDER_IS_NOT_THE_FAULT`: the author is told the check did not run, and
#: told why in terms of the input rather than of the check.
_DECLARED = ("This check was decided before the draft was read: {what}, so "
             "{consequence}. It is reported rather than counted, because a "
             "verdict nothing could have changed is not a unit of scrutiny.")


#: `MISC-103`. Matched against the whole draft, not against one producer's
#: wording: `ml/narrative_engine` opens its limitation list with this clause and
#: `ml/publication.EXPLORATORY_LIMITATION_SENTENCE` repeats it on the fallback
#: path. The pattern spans the two halves that carry the claim — the mode, and
#: what it did to the test set — so a paraphrase of the connective still
#: matches while a draft that never says it cannot.
_EXPLORATORY_LIMITATION_PATTERN = re.compile(
    r"exploratory mode.{0,200}?not\s+quarantined", re.IGNORECASE | re.DOTALL)


def _extract_section(text: str, heading: str, level: int) -> str:
    pattern = rf"(?ms)^{'#' * level}\s+{re.escape(heading)}\s*\n(.*?)(?=^{'#' * level}\s+|^\#{{1,{level-1}}}\s+|\Z)"
    match = re.search(pattern, text or "")
    return match.group(1).strip() if match else ""


def _extract_latex_subsection(text: str, heading: str) -> str:
    pattern = rf"(?ms)\\subsection\{{{re.escape(heading)}\}}\s*(.*?)(?=\\subsection\{{|\\section\{{|\\paragraph\{{|\\end\{{document\}}|\Z)"
    match = re.search(pattern, text or "")
    return match.group(1).strip() if match else ""


#: Sections of the export that are the app's OWN RECORD of what it advised —
#: the coaching log and the decision appendix. A coach sentence there is the
#: record being accurate; the same sentence in Methods or Discussion is the
#: manuscript speaking in a register no journal accepts. The coaching check
#: reads everything else (`DRIVE-074` / D9-06).
_AUDIT_SECTION_PATTERNS = (
    r"(?ms)^##\s+Key Observations and Resolutions\s*$.*?(?=^##\s+|\Z)",
    r"(?ms)^##\s+Appendix: Decision Audit Trail\s*$.*?(?=^##\s+|\Z)",
    r"(?ms)\\subsection\{Decision Audit Trail\}.*?(?=\\subsection\{|\\section\{|\\end\{document\}|\Z)",
)


def _without_audit_sections(text: str) -> str:
    """The export text with the app's own advice log removed."""
    stripped = text or ""
    for pattern in _AUDIT_SECTION_PATTERNS:
        stripped = re.sub(pattern, "", stripped)
    return stripped


def _extract_analysis_n(text: str) -> Optional[int]:
    patterns = [
        r"Of\s+[\d,]+\s+observations,\s+([\d,]+)\s+remained for analysis",
        r"A total of\s+([\d,]+)\s+observations were available for analysis",
        r"dataset of\s+([\d,]+)\s+observations",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def _extract_final_predictor_count(text: str) -> Optional[int]:
    patterns = [
        r"retained\s+([\d,]+)\s+predictors?\s+for final modeling",
        r"final modeling set contained\s+([\d,]+)\s+predictors?",
        r"([\d,]+)\s+predictors?\s+for final modeling",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def _extract_table1_overall_n(table1_df: Any) -> Optional[int]:
    """Extract overall N from a generated Table 1 dataframe header."""
    if table1_df is None:
        return None
    for column in getattr(table1_df, "columns", []):
        match = re.search(r"Overall\s+\(N=([\d,]+)\)", str(column))
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def _table1_contains_feature(table1_df: Any, feature_name: str) -> bool:
    """Check whether a finalized predictor appears in Table 1 row labels."""
    if table1_df is None:
        return False
    needle = str(feature_name).strip().lower()
    for label in getattr(table1_df, "index", []):
        normalized = str(label).strip().lower()
        if normalized == needle:
            return True
        if normalized.startswith(f"{needle},"):
            return True
    return False


def _model_variants(model_key: str) -> List[str]:
    display = _MODEL_NAMES.get(model_key) or _MODEL_NAMES.get(model_key.lower()) or model_key.replace("_", " ").title()
    variants = [display, model_key.upper(), model_key.replace("_", " ").title()]
    return [variant for variant in variants if variant]


def _contains_any_variant(text: str, variants: List[str]) -> bool:
    lowered = (text or "").lower()
    return any(variant.lower() in lowered for variant in variants)


def _match_snippet(text: str, match: re.Match[str], radius: int = 24) -> str:
    """Return a short surrounding snippet for a regex match."""
    start = max(0, match.start() - radius)
    end = min(len(text or ""), match.end() + radius)
    snippet = (text or "")[start:end].replace("\n", " ")
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet


def _invalid_metric_terms_for_task(text: str, task_type: str) -> List[str]:
    invalid_terms = {
        "regression": {"accuracy", "f1", "auc", "precision", "recall"},
        "classification": {"rmse", "mae", "r2", "medianae"},
    }.get(task_type, set())
    return sorted({term for term in invalid_terms if re.search(rf"\b{re.escape(term)}\b", text or "", re.IGNORECASE)})


def validate_manuscript_bundle(
    manuscript_context: Optional[Dict[str, Any]],
    methods_text: str,
    report_text: str,
    latex_text: str,
    task_type: str,
    table1_df: Any = None,
) -> ManuscriptValidationReport:
    """Validate manuscript consistency before export.

    **Three checks declare themselves rather than being scored, and which
    three depends on the context rather than on a list kept here.** Measured
    over 3,000 randomized bundles per branch on two target shapes, holding the
    context and `task_type` fixed and varying only the manuscript: on the
    branch `turbotab.manuscript._counts` takes when a run is held, *Table 1
    includes all finalized predictors* and *Abstract feature-selection
    language* cannot dissent; on the no-run branch *Model names match* joins
    them. Every other check moved in both directions.

    **`Split counts reconcile to analysis population` is deliberately NOT
    declared**, on either branch, and that is a decision rather than an
    oversight. It is vacuous in two opposite ways — an identity that cannot
    FAIL where a run wrote all three parts, and pinned at FAIL where the
    lockbox branch wrote only two — and neither is repaired by saying so. It
    gets a comparand instead; see `MISC-028` and `MISC-031`.
    """
    context = manuscript_context or {}
    population = context.get("population_counts") or {}
    feature_counts = context.get("feature_counts") or {}
    included_models = (
        context.get("included_models")
        or list((context.get("selected_model_results") or {}).keys())
        or []
    )
    checks: List[ManuscriptValidationCheck] = []

    abstract_section = _extract_section(report_text, "Abstract (Draft)", level=2)
    study_design_section = _extract_section(methods_text, "Study Design", level=3)
    predictor_section = _extract_section(methods_text, "Predictor Variables", level=3)
    model_dev_section = _extract_section(methods_text, "Model Development", level=3)
    model_eval_section = _extract_section(methods_text, "Model Evaluation", level=3)
    latex_model_dev_section = _extract_latex_subsection(latex_text, "Model Development")
    combined_export_text = f"{report_text}\n{latex_text}"

    expected_analysis_n = population.get("analysis_total")
    abstract_analysis_n = _extract_analysis_n(abstract_section)
    study_design_n = _extract_analysis_n(study_design_section)
    analysis_match = (
        expected_analysis_n is not None
        and abstract_analysis_n == expected_analysis_n
        and study_design_n == expected_analysis_n
    )
    checks.append(
        ManuscriptValidationCheck(
            name="Analysis population is consistent across abstract and study design",
            status="PASS" if analysis_match else "FAIL",
            location="Abstract / Methods: Study Design",
            detail=(
                f"Expected analysis N={expected_analysis_n}, abstract N={abstract_analysis_n}, "
                f"study design N={study_design_n}."
            ),
        )
    )

    # `MISC-028` / `MISC-031`. THE PARTS AND THE WHOLE HAVE TO COME FROM
    # DIFFERENT PLACES OR THIS ANSWERS NOTHING. Until `L65` the Guided producer
    # made both sides itself, in opposite and equally useless ways: on the run
    # branch `analysis_total` was DEFINED as `train_n + test_n` with `val_n`
    # pinned to `0`, an identity that could not FAIL; on the lockbox branch it
    # wrote only two of the four keys, so the sum was `test_n` alone and the
    # check could not PASS — an unfitted project was shown a failure no edit
    # could ever clear.
    #
    # `turbotab.manuscript._counts` now takes the whole from the SEAL and the
    # parts from the RUN, and says which in these two keys. They are read
    # rather than assumed absent: a producer that supplies neither (the Classic
    # door does) is treated as two derivations, which is the safe direction —
    # it scores the check rather than excusing it.
    split_source = population.get("split_source")
    total_source = population.get("analysis_total_source")
    one_derivation = bool(split_source) and split_source == total_source
    # AND THE THIRD STATE, FOUND BY DRIVING THE PAGE RATHER THAN BY READING.
    # `MISC-031` names the lockbox branch; a project that has not been SEALED
    # reaches neither branch, so `population_counts` is `{}` and this compared
    # `None` against `0` — FAIL, permanently, on a project whose author has
    # simply not got to the seal yet. That is the same class one state earlier,
    # and it is the plainest case of `MISC-029`'s criterion: the input is
    # absent, so there is nothing to reconcile and nothing an author can do
    # about it.
    nothing_to_reconcile = expected_analysis_n is None
    split_total = sum(
        int(population.get(key) or 0)
        for key in ("train_n", "val_n", "test_n")
    )
    reconciles = nothing_to_reconcile or expected_analysis_n == split_total
    checks.append(
        ManuscriptValidationCheck(
            name="Split counts reconcile to analysis population",
            status="PASS" if reconciles else "FAIL",
            location="Methods: Study Design",
            detail=(
                f"analysis_total={expected_analysis_n}, "
                f"split_sum={split_total}."
                + (f" The total is the {total_source} count and the split is "
                   f"the {split_source} partition, so these are two "
                   f"derivations and a disagreement between them is real."
                   if split_source and not one_derivation else "")),
            # THE ONLY THING THAT CAN DECLARE THIS CHECK IS THE ABSENCE OF A
            # SECOND DERIVATION, which is the same criterion the other three
            # declared checks use one step out: `MISC-029` declares where an
            # INPUT is missing, and this declares where the COMPARAND is. Both
            # are "there is no second thing to compare against".
            scored=not (one_derivation or nothing_to_reconcile),
            declared_because=(
                _DECLARED.format(
                    what=("no analysis population reached the manuscript "
                          "context at all, which is the state of a project "
                          "that has not been sealed"),
                    consequence=("there is no total for the split to "
                                 "reconcile to, and no edit to the draft "
                                 "could supply one"))
                if nothing_to_reconcile else
                _DECLARED.format(
                    what=(f"both the analysis total and the split come from "
                          f"the {split_source} partition, and nothing else in "
                          f"this project has counted those rows"),
                    consequence=("the sum restates the total instead of "
                                 "testing it, and it will agree however wrong "
                                 "that partition is"))
                if one_derivation else ""),
        )
    )

    expected_predictors = feature_counts.get("selected")
    if expected_predictors is None:
        expected_predictors = len(context.get("feature_names_for_manuscript") or [])
    abstract_predictors = _extract_final_predictor_count(abstract_section)
    methods_predictors = _extract_final_predictor_count(predictor_section)
    predictor_match = (
        expected_predictors is not None
        and abstract_predictors == expected_predictors
        and methods_predictors == expected_predictors
    )
    checks.append(
        ManuscriptValidationCheck(
            name="Final predictor count is consistent across abstract and methods",
            status="PASS" if predictor_match else "FAIL",
            location="Abstract / Methods: Predictor Variables",
            detail=(
                f"Expected predictors={expected_predictors}, abstract={abstract_predictors}, "
                f"predictor section={methods_predictors}."
            ),
        )
    )

    table1_overall_n = _extract_table1_overall_n(table1_df)
    table1_n_match = table1_overall_n is None or expected_analysis_n == table1_overall_n
    checks.append(
        ManuscriptValidationCheck(
            name="Table 1 population matches the analysis cohort",
            status="PASS" if table1_n_match else "FAIL",
            location="Table 1",
            detail=(
                "Table 1 is absent or uses the analysis cohort."
                if table1_n_match
                else f"Expected analysis N={expected_analysis_n}, Table 1 overall N={table1_overall_n}."
            ),
        )
    )

    expected_feature_names = context.get("feature_names_for_manuscript") or []
    missing_table1_features = [
        feature_name for feature_name in expected_feature_names
        if not _table1_contains_feature(table1_df, feature_name)
    ]
    checks.append(
        ManuscriptValidationCheck(
            name="Table 1 includes all finalized predictors",
            status="PASS" if not missing_table1_features else "FAIL",
            location="Table 1",
            detail=(
                "All finalized predictors appear in Table 1."
                if not missing_table1_features
                else f"Missing predictors: {', '.join(missing_table1_features[:10])}"
                + ("..." if len(missing_table1_features) > 10 else "")
                + "."
            ),
            # `MISC-029`. The loop above iterates the finalized predictor list,
            # so an ABSENT list makes it iterate nothing and pass whatever
            # Table 1 contains — driven, a Table 1 whose index holds none of
            # the predictors passes, and so does a zero-row one.
            scored=bool(expected_feature_names),
            declared_because=("" if expected_feature_names else _DECLARED.format(
                what="no finalized predictor list reached the manuscript "
                     "context (`feature_names_for_manuscript`)",
                consequence="there is nothing to look for in Table 1, so this "
                            "check passes over an empty list rather than over "
                            "an agreement")),
        )
    )

    missing_models = []
    for model_key in included_models:
        variants = _model_variants(model_key)
        in_dev = _contains_any_variant(model_dev_section, variants)
        in_eval = _contains_any_variant(model_eval_section, variants)
        if not (in_dev and in_eval):
            missing_models.append(model_key)
    checks.append(
        ManuscriptValidationCheck(
            name="Model names match between development and evaluation sections",
            status="PASS" if not missing_models else "FAIL",
            location="Methods: Model Development / Model Evaluation",
            detail=(
                "All selected models appear in both sections."
                if not missing_models
                else f"Missing or inconsistent models: {', '.join(missing_models)}."
            ),
            # `MISC-029`. With no model in the context the loop above never
            # executes, so `missing_models` is empty for the one reason that
            # is not a match.
            scored=bool(included_models),
            declared_because=("" if included_models else _DECLARED.format(
                what="no model reached the manuscript context "
                     "(`included_models` is empty)",
                consequence="the comparison loop never runs, so this check "
                            "passes over zero models rather than over a "
                            "match")),
        )
    )

    metric_name = (context.get("best_metric_name") or "").lower()
    invalid_metric = (
        task_type == "regression" and metric_name in {"accuracy", "f1", "auc", "precision", "recall"}
    ) or (
        task_type == "classification" and metric_name in {"rmse", "mae", "r2", "medianae"}
    )
    invalid_metric_terms = _invalid_metric_terms_for_task(
        "\n".join(part for part in (model_dev_section, latex_model_dev_section) if part),
        task_type,
    )
    checks.append(
        ManuscriptValidationCheck(
            name="Selection metric language matches task type",
            status="PASS" if not (invalid_metric or invalid_metric_terms) else "FAIL",
            location="Export Context / Methods",
            detail=(
                f"task_type={task_type}, best_metric_name={context.get('best_metric_name')}."
                if not invalid_metric_terms
                else (
                    f"task_type={task_type}, best_metric_name={context.get('best_metric_name')}, "
                    f"invalid rendered metric term(s) in model-development prose: {', '.join(invalid_metric_terms)}."
                )
            ),
        )
    )

    explicit_primary_claim = bool(
        re.search(
            r"\bselected as the primary model\b|\bmanuscript-primary model was (?!explicitly selected\b)",
            f"{model_dev_section}\n{latex_model_dev_section}",
            re.IGNORECASE,
        )
    )
    no_primary_claim = "no manuscript-primary model was explicitly selected" in combined_export_text.lower()
    expected_primary_model = context.get("manuscript_primary_model")
    primary_conflict = (
        (explicit_primary_claim and no_primary_claim)
        or (explicit_primary_claim and not expected_primary_model)
        or (no_primary_claim and bool(expected_primary_model))
    )
    checks.append(
        ManuscriptValidationCheck(
            name="Primary model statements are internally consistent",
            status="PASS" if not primary_conflict else "FAIL",
            location="Methods / Results",
            detail=(
                f"manuscript_primary_model={expected_primary_model}, "
                f"explicit_primary_claim={explicit_primary_claim}, no_primary_claim={no_primary_claim}."
            ),
        )
    )

    original_count = feature_counts.get("original")
    selected_count = feature_counts.get("selected")
    reduction_language = bool(re.search(r"feature selection|retained .* predictors|reduced", abstract_section or "", re.IGNORECASE))
    should_not_reduce = original_count is not None and selected_count is not None and original_count == selected_count
    checks.append(
        ManuscriptValidationCheck(
            name="Abstract feature-selection language matches actual reduction",
            status="PASS" if not (should_not_reduce and reduction_language) else "FAIL",
            location="Abstract (Draft)",
            detail=(
                "No reduction language detected."
                if not (should_not_reduce and reduction_language)
                else f"Abstract still describes feature reduction even though original={original_count} and selected={selected_count}."
            ),
            # `MISC-029`. `should_not_reduce` needs BOTH counts; with either
            # absent it is False and the verdict is PASS whatever the abstract
            # says. The check's premise, not its comparison, is what is
            # missing.
            scored=original_count is not None and selected_count is not None,
            declared_because=(
                "" if (original_count is not None
                       and selected_count is not None)
                else _DECLARED.format(
                    what=("the pre-selection predictor count is absent from "
                          "the manuscript context "
                          "(`feature_counts['original']`)"),
                    consequence=("whether the set actually shrank has no "
                                 "answer, so this check cannot dissent "
                                 "whatever the abstract claims"))),
        )
    )

    # `MISC-103`. THE ONE LIMITATION THAT IS ABOUT THE WHOLE STUDY, AND THE
    # ONLY CHECK THAT CAN NOTICE IT LEFT. It reached the draft through
    # NarrativeEngine alone; on the fallback path — engine raised, or provenance
    # empty — it silently disappeared, and an exploratory study exported a
    # manuscript claiming a clean held-out evaluation. The composer now emits it
    # too; this check is what makes its absence stop an export rather than pass
    # unnoticed, because a caveat that only sometimes prints is not a caveat.
    #
    # APPENDED ONLY FOR AN EXPLORATORY STUDY, and that is the one place this
    # registry's `MISC-029` doctrine does not apply. Declaring covers a check
    # that RUNS over an input it did not get; a clean study is not missing an
    # input, it owes no sentence at all — and a row on every panel saying so
    # would inflate the roster the header counts (`turbotab.manuscript._counts`
    # serves that number) with a check that is not about that manuscript.
    exploratory = bool(context.get("exploratory_mode"))
    if exploratory:
        exploratory_text = f"{methods_text}\n{report_text}\n{latex_text}"
        limitation_present = bool(
            _EXPLORATORY_LIMITATION_PATTERN.search(exploratory_text))
        checks.append(
            ManuscriptValidationCheck(
                name="Exploratory-mode limitation is stated in the draft",
                status="PASS" if limitation_present else "FAIL",
                location="Methods / Limitations",
                detail=(
                    "The exploratory-mode limitation is stated."
                    if limitation_present
                    else "The session is marked exploratory "
                         "(`exploratory_mode`), but no draft section states "
                         "that the held-out test set was not quarantined from "
                         "feature engineering and selection."
                ),
            )
        )

    artifact_patterns = {
        "raw placeholder tag": r"\[PLACEHOLDER\](?!:)",
        "note tag": r"\[NOTE(?::|\])",
        "markdown heading": r"(?:^|\n)##\s+\S+|\\#\\#",
        "markdown bold": r"\*\*[^*]+\*\*",
    }
    artifact_tokens = [
        label for label, pattern in artifact_patterns.items()
        if re.search(pattern, latex_text or "", re.IGNORECASE)
    ]
    checks.append(
        ManuscriptValidationCheck(
            name="LaTeX output is free of markdown and note artifacts",
            status="PASS" if not artifact_tokens else "FAIL",
            location="LaTeX export",
            detail=(
                "No raw placeholder or markdown tokens detected."
                if not artifact_tokens
                else f"Detected raw artifacts: {', '.join(artifact_tokens)}."
            ),
        )
    )

    internal_keys = sorted({key for key in _MODEL_NAMES if "_" in key})
    leaked_keys = []
    for key in internal_keys:
        if re.search(rf"\b{re.escape(key)}\b", combined_export_text) or re.search(rf"\b{re.escape(key.upper())}\b", combined_export_text):
            leaked_keys.append(key.upper())
    checks.append(
        ManuscriptValidationCheck(
            name="No internal model keys leak into export text",
            status="PASS" if not leaked_keys else "FAIL",
            location="Markdown / LaTeX export",
            detail=(
                "No internal model identifiers detected."
                if not leaked_keys
                else f"Leaked keys: {', '.join(leaked_keys)}."
            ),
        )
    )

    # `DRIVE-074` / D9-06. FOUR LITERAL STRINGS COULD NOT SEE THE TWO REGISTERS
    # THAT ACTUALLY REACH THE EXPORT. A coach card's *"A reviewer would question
    # why the more complex model was selected."* sat in the Discussion of a
    # drafted manuscript while this check reported "No coaching language
    # detected" — the check passing was worse than its absence, because the
    # panel said the prose had been examined for exactly this. Reviewer-
    # anticipation and second-person address are what distinguish advice to the
    # analyst from prose about the study; neither can appear in a manuscript.
    coaching_patterns = [
        "no action needed",
        "favorable to analysis",
        "workflow-derived abstract",
        "[applicable to",
        "a reviewer would",
        "reviewers would",
        "you should",
        "you may want",
        "your data",
        "your dataset",
        "your model",
        "consider using",
    ]
    coaching_scope_text = _without_audit_sections(combined_export_text).lower()
    found_patterns = [pattern for pattern in coaching_patterns if pattern in coaching_scope_text]
    checks.append(
        ManuscriptValidationCheck(
            name="No coaching language patterns remain in export text",
            status="PASS" if not found_patterns else "FAIL",
            location="Markdown / LaTeX export",
            detail=(
                "No coaching language detected in the drafted prose "
                "(the coaching log and decision appendix are the app's own "
                "record and are not read by this check)."
                if not found_patterns
                else f"Found coaching patterns: {', '.join(found_patterns)}."
            ),
        )
    )

    punctuation_issues = []
    if match := re.search(r"(?<!\.)\.\.(?!\.)", combined_export_text):
        punctuation_issues.append(f"double periods near '{_match_snippet(combined_export_text, match)}'")
    if match := re.search(r"\b(Table|Figure)\s+X\b", combined_export_text):
        punctuation_issues.append(f"dangling Table/Figure X reference near '{_match_snippet(combined_export_text, match)}'")
    if match := re.search(r"[—-]\.", combined_export_text):
        punctuation_issues.append(f"dash followed by period near '{_match_snippet(combined_export_text, match)}'")
    checks.append(
        ManuscriptValidationCheck(
            name="No obvious dangling punctuation or placeholder references remain",
            status="PASS" if not punctuation_issues else "FAIL",
            location="Markdown / LaTeX export",
            detail=(
                "No obvious dangling references detected."
                if not punctuation_issues
                else f"Detected issues: {', '.join(punctuation_issues)}."
            ),
        )
    )

    return ManuscriptValidationReport(checks=checks)
