"""R1-F's first batch (R1_PLAN.md R1-F; the claims backtest's fix list,
``docs/turbotab-next/notes/claims-backtest/FIXES.md``): the sentences an analyst reads on a card, in
teaching, in a refusal or in a reading, each held to what the code does, both ways.

* **D-009**: a swap is counted in calories; a macronutrient's grams convert by its Atwater factor
  (a fatty acid named apart, which the factors do not cover, has none), and a food moves only when
  recorded in kcal or kJ (``readings.kcal_per_unit``). The pack's sentence keeps its SETTLED badge;
  TurboTab's rules sit in a section of their own, badged by their claim's grade.
* **The badges** (the RF-1 verifier's V2): a SETTLED section says only what its pack section says,
  narrowed at most, and a sentence the pack does not say sits in a section that is not SETTLED.
* **D-018**: bootstrap optimism correction leads under prediction below about 20,000 units (Harrell
  2017's figure) where the folds need not follow time, and the near-interpolating families keep
  their cross-validated score.
* **D-021**: the pairwise interval is the corrected repeated k-fold t, written out here, and one
  that includes zero says the rows cannot tell two families apart.
* **D-032**: in a randomized trial an adjusted odds ratio is a conditional one, while the
  standardized risk difference keeps its meaning (statsmodels, on a trial built here).
* **D-037**: the food-frequency blanks' mechanism and their handling carry the badges the
  nutrition pack gives them (its §06, read here).
* **D-044**: blanking a count too numerous to count says it removes a count above the countable
  range, and that this biases the column low (built here so every such count is above the rest).
* **D-070**: the imputed copies' sentence and option say they are pooled only where the fit pools
  them: numbered by a column, each kept as a record, so not where each unit's rows are combined.
* **D-001**: a nutrient that carries no energy is substituted for nothing, whatever the covariates
  carry; a food, servings, a drink, a fatty acid or a fiber, which no general factor converts,
  keeps the substitution, and so does a covariate that carries energy.
* **D-012**: averaging replicates divides each mean's error variance by up to their number, fully
  when the errors are independent (a simulation built here), whatever the error's source.
* **A-106**: the event-share sentence says no resampling only where the fit applied none.
* **A-117**: multiple imputation is refused unless the purpose is inference, in Sisk et al.'s
  scope, with their condition for using the outcome (multiple imputation, not a single fill).
* **A-121**: inside a near dependency the estimate's contrast changes; "less of what it replaces"
  is said only of a part beside its total.
* **C-110**: an ungrouped seal labels its held-out scores exploratory and says nothing of the
  folds, which internal–external validation draws by a grouping's levels.
* **C-144**: the demographic guess's reason names no diet, and says the socioeconomic columns may
  follow what you study.
* **C-241**: the missing-values option says whether the group is in the imputation model, right
  after the variables it refers to.
* **The roles ruling** (the orchestrator's, at the landing's cap): the roles teaching says what a
  role is, never what reaches the models, since a missing-values drop, the adjustment answers or a
  fixed-effects grouping column changes what the models get (``models.pipeline.model_predictors``).

Every expectation comes from construction, an independent library or the cited source. This file
names no claim id and none of the constants the claims set (meta-test 2).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.decisions import MissingSpec, ProjectState, Refusal, validate
from turbotab.core.teaching.content import entries_now
from turbotab.core.tests.test_coach import recalls  # noqa: F401 - the coach's recalls fixture

REPO = Path(__file__).resolve().parents[3]


def _entry(key: str, purpose: str | None = None) -> dict:
    return next(e for e in entries_now(purpose) if e["key"] == key)


def _section(entry: dict, heading: str) -> dict:
    return next(s for s in entry["drawer"]["sections"] if s["heading"] == heading)


def _option(entry: dict, value: str) -> dict:
    return next(o for o in entry["options"] if o["value"] == value)


def _term(entry: dict, name: str) -> dict:
    return next(t for t in entry["terms"] if t["term"] == name)


# ── D-009 · a swap is counted in calories ────────────────────────────────────


def _units(**units: str) -> ProjectState:
    from turbotab.core.tests.test_stage_registry import record

    return d.fold([record(i + 1, {"kind": "confirm_reading", "reading": "unit", "column": c,
                                  "value": u}) for i, (c, u) in enumerate(units.items())])


def test_a_nutrients_grams_move_at_its_atwater_factor_and_hold_total_energy():
    """The substitution's own conversion (``readings.kcal_per_unit``): protein recorded in grams
    moves at 4 kcal/g and fat at 9 (the Atwater factors, NUTRITION_PACK §01), so moving k kcal from
    one to the other changes total energy by nothing. EPA recorded in grams is a fatty acid the
    factors do not cover by name, and has none, so the teaching says a macronutrient's grams
    convert, not any nutrient's (the RF-1 verifier's V4), where the base said any swap in grams is
    not the one intended."""
    from turbotab.core.readings import kcal_per_unit

    state = _units(protein_g="g", fat_g="g", epa_g="g")
    protein, fat = kcal_per_unit(state, "protein_g"), kcal_per_unit(state, "fat_g")
    assert (protein.factor, fat.factor) == (4.0, 9.0)
    for k in (50.0, 100.0, 200.0):
        assert -k / protein.factor * 4.0 + k / fat.factor * 9.0 == pytest.approx(0.0, abs=1e-12)
    epa = kcal_per_unit(state, "epa_g")
    assert epa.factor is None and "no nutrient the Atwater factors cover" in epa.why, epa
    entry = _entry("substitution")
    said = [entry["why"], _term(entry, "isocaloric")["definition"],
            _section(entry, "How TurboTab counts a swap")["body"]]
    for text in said:
        assert "Atwater factor" in text, text
        assert re.search(r"\bmacronutrient(?:'s grams| in grams)", text), text
        assert not re.search(r"\ba nutrient(?:'s grams| in grams)", text, re.I), text
        assert not re.search(r"in grams or servings, the substitution reported", text), text
        assert "grams and servings do not make" not in text, text


def test_a_foods_grams_or_servings_move_nothing_until_it_is_recorded_in_kcal():
    """The other side: a food in grams has no factor (fatty fish is about 2 kcal/g, not fat's 9), a
    food with no recorded unit (servings are no unit the reading records) has none, and a food
    recorded in kcal moves at 1 kcal per unit, in kJ at 1/4.184 (1 kcal = 4.184 kJ, NUTRITION_PACK
    §01). The teaching says a food moves only in kcal or kJ, in the why too (the RF-1 verifier's
    V4)."""
    from turbotab.core.readings import kcal_per_unit

    state = _units(fatty_fish_g="g", fish_kcal="kcal", fish_kj="kj")
    assert kcal_per_unit(state, "fatty_fish_g").factor is None
    assert kcal_per_unit(state, "fish_servings").factor is None
    assert kcal_per_unit(state, "fish_kcal").factor == 1.0
    assert kcal_per_unit(state, "fish_kj").factor == pytest.approx(1 / 4.184, rel=1e-12)
    entry = _entry("substitution")
    body = _section(entry, "How TurboTab counts a swap")["body"]
    assert "serving" in body and "kcal or kJ" in body
    assert "food" in entry["why"] and "kcal or kJ" in entry["why"]


def _pack_section(number: str) -> str:
    """NUTRITION_PACK's section ``number`` (``"04"``), its block quotes joined across lines and its
    emphasis dropped."""
    raw = (REPO / "docs/turbotab-next/reference/research/NUTRITION_PACK.md").read_text("utf-8")
    body = raw.split(f"\n## {number} · ", 1)[1].split("\n## ", 1)[0]
    return " ".join(body.replace("\n>", " ").replace("**", "").split())


# The sentences TurboTab adds that the pack does not say: its conversion of a swap (D-009) and a
# nutrient that carries no energy (D-001).
ADDED = re.compile(r"\bsodium\b|\bcarries no energy\b|\bAtwater\b|\bTurboTab (?:converts|moves)\b")


def test_a_settled_section_says_only_its_packs_words_and_turbotabs_rules_are_not_settled():
    """The RF-1 verifier's V2. NUTRITION_PACK §04's SETTLED interpretation says "an energy-adjusted
    coefficient is a substitution estimate", for every nutrient, and names no nutrient that
    carries no energy; §05's SETTLED warning says a substitution is isocaloric only "measured and
    analyzed in calories", and names no Atwater factor. So the sections badged SETTLED from them say
    the pack's sentence, narrowed at most (to a nutrient that carries energy; to a food), and the
    sentences the pack does not say (sodium displaces nothing; TurboTab's conversion of grams, and
    of a food only in kcal or kJ) sit in sections of their own badged CONVENTION, their claims'
    grade, as the badges of D-021 and D-037 were mended. Read from the pack; both ways: every such
    sentence is said, and none sits under SETTLED."""
    s04, s05 = _pack_section("04"), _pack_section("05")
    assert "an energy-adjusted coefficient is a substitution estimate" in s04
    assert ("Adjusting for total energy does not by itself make a substitution isocaloric. The "
            "components must also be measured and analyzed in calories.") in s05
    assert not ADDED.search(s04) and not ADDED.search(s05)
    energy, estimand, swap = _entry("energy_adjustment"), _entry("estimand"), _entry("substitution")
    settled = [(_section(energy, "What the coefficient means"), "an energy-adjusted coefficient is a "
                "substitution estimate"),
               (_section(estimand, "A substitution, not an addition"), "an energy-adjusted "
                "coefficient is a substitution estimate. It answers 'what if this person got more "
                "of their calories from X and correspondingly fewer from everything else, at the "
                "same total intake?' — not 'what if this person ate more X?'"),
               (_section(swap, "Adjusting for energy is not enough"), "Adjusting for total energy "
                "does not by itself make a substitution isocaloric.")]
    for sec, words in settled:
        assert sec["evidence"]["status"] == "SETTLED", sec
        assert words in sec["body"] and words in (s04 + s05), (sec, words)
        assert not ADDED.search(sec["body"]), sec["body"]
    added = [sec for e in (energy, estimand, swap) for sec in e["drawer"]["sections"]
             if ADDED.search(sec["body"])]
    assert {sec["heading"] for sec in added} == {"A nutrient with no energy",
                                                  "How TurboTab counts a swap"}, added
    for sec in added:
        assert sec["evidence"]["status"] == "CONVENTION", sec
    for e in (energy, estimand):
        assert _section(e, "A nutrient with no energy")["evidence"]["status"] == "CONVENTION"


# ── D-018 · bootstrap first, for the families it suits, below about 20,000 units ──

HARRELL_UNITS = 20_000  # Harrell 2017: "less than about 20,000" (models/validation.py quotes it)


def test_resampling_leads_under_prediction_below_harrells_twenty_thousand():
    from turbotab.core.models.validation import validation_plan

    below = validation_plan("prediction", HARRELL_UNITS - 1)
    at = validation_plan("prediction", HARRELL_UNITS)
    assert [o.validation for o in below.options][:2] == ["bootstrap", "repeated_kfold"]
    assert below.resampling_first and not at.resampling_first
    assert at.options[0].validation == "kfold"
    inference = validation_plan("inference", HARRELL_UNITS - 1)
    assert inference.options[0].validation == "kfold"
    body = _section(_entry("split"), "A single split is the weakest")["body"]
    assert f"{HARRELL_UNITS:,}" in body and body.startswith("Under prediction")


def test_with_time_ordered_folds_the_bootstrap_comes_last_and_the_teaching_says_where_it_leads():
    """With folds that follow time the plan forward-chains and ranks the bootstrap last, below the
    units at which it would otherwise lead; the teaching, served under both, names the condition."""
    from turbotab.core.models.validation import validation_plan

    timed = validation_plan("prediction", HARRELL_UNITS - 1, time_ordered=True)
    untimed = validation_plan("prediction", HARRELL_UNITS - 1, time_ordered=False)
    assert [o.validation for o in timed.options][-1] == "bootstrap" and not timed.resampling_first
    assert untimed.options[0].validation == "bootstrap" and untimed.resampling_first
    body = _section(_entry("split", "prediction"), "A single split is the weakest")["body"]
    lead = body.split("bootstrap optimism correction is the recommended default")[0]
    assert "where the folds need not follow time" in lead, body


def test_the_teaching_names_every_family_the_bootstrap_does_not_correct():
    """Coley et al. 2023 (models/validation.py): a near-interpolating learner keeps its
    cross-validated score. The teaching names each family that declares so, and no other."""
    from turbotab.core.models.base import families

    kept = {f.label for f in families() if f.bootstrap_optimism is False}
    corrected = {f.label for f in families() if f.bootstrap_optimism}
    assert kept == {"Boosted trees", "Random forest", "XGBoost"}
    body = _section(_entry("split"), "A single split is the weakest")["body"].lower()
    for label in ("boosted trees", "xgboost", "random forests"):
        assert label in body
    assert "regression-type" in body
    assert not any(label.lower() in body for label in corrected - {"Linear model"})


# ── D-021 · what the pairwise interval says ───────────────────────────────────


def test_the_pairwise_interval_is_the_corrected_repeated_k_fold_t_written_out():
    """Nadeau & Bengio (2003), as Bouckaert & Frank (2004) apply it to r repeats of K folds: over
    the rK paired differences, se = √((1/(rK) + 1/(K − 1)) s²) on rK − 1 degrees of freedom."""
    from scipy import stats

    from turbotab.core.models.validation import corrected_t

    rng = np.random.default_rng(4)
    r, K = 10, 5
    a, b = rng.normal(0.70, 0.03, r * K), rng.normal(0.69, 0.03, r * K)
    got = corrected_t(list(a), list(b), test_share=None, repeats=r)
    diff = a - b
    se = np.sqrt((1 / (r * K) + 1 / (K - 1)) * diff.var(ddof=1))
    q = stats.t.ppf(0.975, r * K - 1)
    assert got.se == pytest.approx(se, rel=1e-12)
    assert (got.ci_low, got.ci_high) == pytest.approx((diff.mean() - q * se, diff.mean() + q * se))


def _null_cv(rng: np.random.Generator, n: int = 60, r: int = 10, K: int = 5):
    """Two least-squares models, each on one noise column, so neither predicts better: their
    fold scores (negative mean squared error) over r repeats of K folds."""
    x = rng.normal(size=(n, 2))
    y = rng.normal(size=n)
    out = {0: [], 1: []}
    for rep in range(r):
        folds = rng.permutation(n) % K
        for k in range(K):
            test, train = folds == k, folds != k
            for j in (0, 1):
                X = np.column_stack([np.ones(train.sum()), x[train, j]])
                beta = np.linalg.lstsq(X, y[train], rcond=None)[0]
                pred = beta[0] + beta[1] * x[test, j]
                out[j].append(-float(np.mean((y[test] - pred) ** 2)))
    return out[0], out[1]


def test_under_a_true_null_the_interval_rarely_calls_a_family_ahead():
    """The teaching's "usually conservative", in one regime built here: two exchangeable models,
    10 × 5 folds, 300 tables; an interval that excludes zero, either way, in at most 5% of them
    (the nominal level), so an interval that includes zero is the usual answer."""
    from turbotab.core.models.validation import corrected_t

    rng = np.random.default_rng(21)
    ahead = 0
    for _ in range(300):
        a, b = _null_cv(rng)
        t = corrected_t(a, b, test_share=None, repeats=10)
        ahead += int(t.ci_low > 0 or t.ci_high < 0)
    assert ahead / 300 <= 0.05


def test_the_teaching_says_an_interval_with_zero_cannot_tell_two_families_apart():
    """The verdicts the comparisons give (``family_differences``) are "not distinguishable on
    these rows" and "… is ahead on these rows"; the teaching says the same, cites the method, and
    is not badged with the pack section on discrimination and calibration."""
    section = _section(_entry("split"), "How precise the scores are here")
    body = section["body"]
    assert "Nadeau & Bengio 2003" in body and "Bouckaert & Frank 2004" in body
    assert "cannot tell" in body and "not that they are equal" in body
    assert "ahead" in body and "approximate" in body
    assert "Discrimination vs calibration" not in section["evidence"]["source"]
    assert section["evidence"]["status"] != "SETTLED"


# ── D-032 · what adjustment does in a trial ───────────────────────────────────


def _trial(rng: np.random.Generator, n: int = 20_000) -> pd.DataFrame:
    arm = rng.integers(0, 2, n)
    z = rng.normal(size=n)  # prognostic, independent of the arm (randomized)
    p = 1 / (1 + np.exp(-(-0.5 + 1.0 * arm + 2.0 * z)))
    return pd.DataFrame({"arm": arm, "z": z, "y": (rng.random(n) < p).astype(float),
                         "w": 2.0 + 1.0 * arm + 2.0 * z + rng.normal(size=n)})


def test_in_a_trial_an_adjusted_odds_ratio_is_another_effect_and_a_standardized_risk_is_not():
    """Randomized, so z explains nothing of the arm. statsmodels: adjusting for z moves the log
    odds ratio far from the unadjusted one (non-collapsibility), while the risk difference
    standardized over the rows from the adjusted model stays the unadjusted one; a numeric
    outcome's mean difference (collapsible) stays too. The design card says so."""
    import statsmodels.api as sm

    f = _trial(np.random.default_rng(32))
    crude = sm.Logit(f["y"], sm.add_constant(f[["arm"]])).fit(disp=0)
    adjusted = sm.Logit(f["y"], sm.add_constant(f[["arm", "z"]])).fit(disp=0)
    assert adjusted.params["arm"] - crude.params["arm"] > 0.3
    X1, X0 = sm.add_constant(f[["arm", "z"]]), sm.add_constant(f[["arm", "z"]])
    X1["arm"], X0["arm"] = 1.0, 0.0
    standardized = float(adjusted.predict(X1).mean() - adjusted.predict(X0).mean())
    raw = float(f.loc[f.arm == 1, "y"].mean() - f.loc[f.arm == 0, "y"].mean())
    assert standardized == pytest.approx(raw, abs=0.01)
    linear_crude = sm.OLS(f["w"], sm.add_constant(f[["arm"]])).fit().params["arm"]
    linear_adjusted = sm.OLS(f["w"], sm.add_constant(f[["arm", "z"]])).fit().params["arm"]
    assert linear_adjusted == pytest.approx(linear_crude, abs=0.06)
    why = _entry("design")["why"]
    assert "only sharpens precision" not in why
    assert "odds ratio" in why and "standardized" in why


def test_a_case_control_sample_gives_absolute_risk_with_its_sampling_fractions():
    """A cohort with a 5% risk, sampled by outcome (every case, 4 controls per case): the sample's
    share of cases is 20%, not the risk; weighting each row by the inverse of its sampling
    fraction gives the risk back. The card says the fractions are needed, not that it never can."""
    rng = np.random.default_rng(9)
    cohort = rng.random(200_000) < 0.05
    cases = np.flatnonzero(cohort)
    controls = rng.choice(np.flatnonzero(~cohort), 4 * len(cases), replace=False)
    f_case, f_control = 1.0, len(controls) / float((~cohort).sum())
    sample_share = len(cases) / (len(cases) + len(controls))
    weighted = (len(cases) / f_case) / (len(cases) / f_case + len(controls) / f_control)
    assert sample_share == pytest.approx(0.2, abs=1e-9)
    assert weighted == pytest.approx(cohort.mean(), rel=1e-9)
    why = _entry("design")["why"]
    assert "sampling fractions" in why
    assert "cannot give prevalence or absolute risk;" not in why


# ── D-037 · food-frequency blanks: the mechanism and the handling ─────────────


def test_the_food_frequency_blanks_carry_the_packs_own_badges():
    """NUTRITION_PACK §06 marks the claim "[SETTLED that the mechanism is non-ignorable; DISPUTED
    how to handle it]" and zero-coding "[CONVENTION with documented limits]", saying a blank is
    "often missing not at random". The mechanism's section is SETTLED and says "often"; the
    handling is its own section, badged DISPUTED, naming zero-coding and what imputation improves
    on."""
    raw = (REPO / "docs/turbotab-next/reference/research/NUTRITION_PACK.md").read_text("utf-8")
    pack = " ".join(raw.replace("\n>", " ").split())  # its block quotes, joined across lines
    assert "SETTLED that the mechanism is non-ignorable; DISPUTED how to handle it" in pack
    assert "often **missing not at random**" in pack
    missing = _entry("missing")
    mechanism = _section(missing, "Blanks often mean something")
    handling = _section(missing, "Filling food-frequency blanks")
    assert mechanism["evidence"]["status"] == "SETTLED"
    assert "often" in mechanism["body"] and "frequently means" not in mechanism["body"]
    assert "imputation" not in mechanism["body"]
    assert handling["evidence"]["status"] == "DISPUTED"
    assert "zero" in handling["body"] and "complete cases" in handling["body"]
    assert "rarely eaten" in handling["body"]


# ── D-044 · a count too numerous to count, blanked ────────────────────────────


def _plates(tntc: int, seed: int = 0) -> pd.DataFrame:
    """Plate counts within FDA BAM's countable range (15-300 per plate), some read TNTC."""
    rng = np.random.default_rng(seed)
    counts = [str(int(v)) for v in rng.integers(20, 280, 60)]
    return pd.DataFrame({"apc": counts + ["TNTC"] * tntc + ["."] * 2,
                         "y": rng.normal(size=62 + tntc)})


def _read_as_numbers(frame: pd.DataFrame):
    from turbotab.core import repairs as R

    raw, finding = next((r, f) for r, f in R.text_number_findings(frame)
                        if f["id"] == "text_numbers__apc")
    (option,) = R.offer(finding, raw, R.OfferContext(frame))
    return option


def test_blanking_counts_too_numerous_to_count_says_it_biases_the_column_low():
    """Each TNTC plate holds more colonies than any countable one (BAM: a lower bound), so the
    column's mean with them, at any values above the range, is above its mean without them: the
    option and its sentence say the blanks are such counts and that the column is biased low."""
    frame = _plates(5)
    kept = pd.to_numeric(frame["apc"], errors="coerce").dropna()
    for above in (301.0, 500.0, 5000.0):
        assert np.mean([*kept, *([above] * 5)]) > kept.mean()
    option = _read_as_numbers(frame)
    for text in (option.consequence, option.sentence):
        assert "too numerous to count" in text and "low" in text, text
    assert "above the countable range" in option.sentence and "right-censored" in option.sentence


def test_a_column_with_no_count_too_numerous_to_count_says_nothing_of_one():
    option = _read_as_numbers(_plates(0))
    for text in (option.consequence, option.sentence):
        assert "too numerous" not in text and "countable" not in text, text
    assert option.consequence == "`60` values of `apc` become numbers; `2` become blank."


# ── D-070 · imputed copies, pooled only where the fit pools them ──────────────


def _copies_sentence(implicate: str | None, purpose: str = "inference",
                     unit: str | None = None) -> str:
    from turbotab.core.voice import sentence_for

    before = ProjectState(target="bmd", purpose=purpose, unit=unit,
                          grain={"grain": "repeated", "id_column": "SEQN"})
    return sentence_for(d.SetRepeatKind(repeat_kind="imputed_copies", implicate_column=implicate),
                        before)


def _fit_pools(implicate: str | None, unit: str = "row") -> str | None:
    from turbotab.core.stages.modeling import imputed_copies_column

    state = ProjectState(target="bmd", purpose="inference", unit=unit,
                         grain={"grain": "repeated", "id_column": "SEQN"},
                         repeat_kind={"repeat_kind": "imputed_copies",
                                      "implicate_column": implicate})
    return imputed_copies_column(state)


def test_numbered_copies_kept_as_records_are_said_pooled_as_the_fit_pools_them():
    assert _fit_pools("_MULT_") == "_MULT_"
    said = _copies_sentence("_MULT_")
    assert "pooled by Rubin's rules, which carries the imputation's uncertainty" in said
    option = _option(_entry("repeat_kind"), "imputed_copies")["consequence"]
    assert "Rubin's rules pool" in option and "only as a recorded limitation" not in option


def test_the_copies_option_names_both_conditions_the_fit_pools_under():
    """The option says pooled only for numbered copies kept as records, the two conditions
    ``imputed_copies_column`` reads: with a numbering column and one row per record it
    pools; with no numbering column, or with each unit's rows combined, it does not."""
    from turbotab.core.stages.modeling import imputed_copies_column

    def pools(implicate: str | None, unit: str) -> bool:
        state = ProjectState(target="bmd", purpose="inference", unit=unit,
                             grain={"grain": "repeated", "id_column": "SEQN"},
                             repeat_kind={"repeat_kind": "imputed_copies",
                                          "implicate_column": implicate})
        return imputed_copies_column(state) is not None

    assert pools("_MULT_", "row") and not pools(None, "row") and not pools("_MULT_", "unit")
    option = _option(_entry("repeat_kind", "inference"), "imputed_copies")["consequence"]
    assert option.endswith("under inference, Rubin's rules pool numbered copies kept as "
                           "records."), option


def test_copies_no_column_numbers_are_said_not_pooled_as_the_fit_does_not_pool_them():
    assert _fit_pools(None) is None
    said = _copies_sentence(None)
    assert "were not pooled by Rubin's rules" in said
    assert "carries the imputation's uncertainty" not in said
    assert "were not pooled by Rubin's rules" in _copies_sentence("_MULT_", "prediction")


@pytest.mark.parametrize("unit", ["unit", "row", None])
def test_numbered_copies_whose_rows_are_combined_are_said_not_pooled_as_the_fit_does_not_pool_them(
        unit):
    """The RF-1 verifier's V3: the sentence's second condition, tested apart from the column. With
    a column numbering the copies, the fit pools them only where each is kept as a record
    (``set_unit`` "row"); with each unit's rows combined into one ("unit") it pools nothing, and the
    sentence says so. Before the unit is answered (None) the copies are still records, the unit
    question comes after this answer, and combining them under inference is recorded as a
    limitation that says this answer again, so the sentence says pooled; once "row" is answered the
    fit pools them, as the sentence says."""
    said = _copies_sentence("_MULT_", unit=unit)
    if unit == "unit":
        assert _fit_pools("_MULT_", "unit") is None
        assert ("were not pooled by Rubin's rules (each unit's rows are combined into one)"
                in said), said
        assert "carries the imputation's uncertainty" not in said, said
    else:
        assert _fit_pools("_MULT_", "row") == "_MULT_"
        assert "pooled by Rubin's rules, which carries the imputation's uncertainty" in said, said
        assert "were not pooled" not in said, said


# ── D-001 · a nutrient that carries no energy ─────────────────────────────────


def _describe(method: str, exposure: str):
    from turbotab.core.methods.energy import describe_model

    adjustment = d.EnergyAdjustment(method=method, energy_column="kcal", nutrients=[exposure])
    roles = {exposure: "exposure", "kcal": "energy", "age": "covariate"}
    name = f"{exposure}_adj" if method == "residual" else exposure
    return describe_model(adjustment, [exposure, "kcal", "age"], roles, [name, "kcal", "age"])


@pytest.mark.parametrize("method", ["standard", "residual"])
@pytest.mark.parametrize("exposure", ["sodium_mg", "potassium_mg", "DR1TSODI"])
def test_with_energy_held_fixed_a_nutrient_with_no_energy_is_substituted_for_nothing(method,
                                                                                     exposure):
    """Sodium and potassium are minerals, which carry no energy (no Atwater factor: FAO 2003's
    general factors are for protein, carbohydrate, fat, alcohol and fiber), and NHANES's DR1TSODI
    is sodium in mg (DR1TOT_L's codebook): with total energy fixed, more of one leaves every energy
    source where it was, so the coefficient names no substitution."""
    from turbotab.core.methods.energy import energy_factor

    assert energy_factor(exposure).factor is None
    text = _describe(method, exposure).text
    assert "substitution" not in text and "in place of" not in text, text
    assert "held fixed" in text, text
    # The nutrient displaces nothing ("X is substituted for Y" says X replaces Y, so the base's
    # "nothing is substituted for it" said that nothing replaces the nutrient: the verifier's V5).
    assert "the nutrient carries no energy, so it displaces no energy source" in text, text
    assert "nothing is substituted for it" not in text, text


@pytest.mark.parametrize("method", ["standard", "residual"])
def test_with_energy_held_fixed_a_nutrient_that_carries_energy_is_a_substitution(method):
    from turbotab.core.methods.energy import energy_factor

    assert energy_factor("protein_g").factor == 4.0
    text = _describe(method, "protein_g").text
    assert "substitution" in text and "carries no energy" not in text


# Columns that carry energy though no general factor converts them (energy_factor gives none), so
# the gate the verifier found (no energy factor) fired on them: a food in grams (red meat; fatty
# fish, about 2 kcal/g), servings of a sugar-sweetened beverage, alcoholic drinks (alcohol is 7
# kcal/g), EPA (a polyunsaturated fatty acid, fat at 9 kcal/g) and the Framingham FFQ's AOAC fiber
# (fiber is 2 kcal/g in the systems that count it). The general factors are FAO 2003's (Food
# energy: methods of analysis and conversion factors, ch. 3).
CARRY_ENERGY_WITH_NO_FACTOR = ["red_meat_g", "fatty_fish_g", "ssb_servings", "alcohol_drinks",
                               "epa_g", "NUT_AOFIB"]


@pytest.mark.parametrize("method", ["standard", "residual"])
@pytest.mark.parametrize("exposure", CARRY_ENERGY_WITH_NO_FACTOR)
def test_with_energy_held_fixed_a_food_a_drink_or_a_fatty_acid_is_still_a_substitution(method,
                                                                                       exposure):
    """The no-swap sentence fires only on a positive reading that a column carries no energy, not
    on a missing factor: each of these has none (shown first), carries energy, and keeps the
    substitution reading."""
    from turbotab.core.methods.energy import energy_factor

    assert energy_factor(exposure).factor is None
    text = _describe(method, exposure).text
    assert "carries no energy" not in text and "displaces no energy source" not in text, text
    assert "substitution" in text, text


def test_a_food_beside_sodium_keeps_the_substitution_reading():
    """Two study factors, sodium and red meat: red meat carries energy, so the model's sentence
    does not say the nutrient carries none."""
    from turbotab.core.methods.energy import describe_model

    adjustment = d.EnergyAdjustment(method="standard", energy_column="kcal",
                                    nutrients=["sodium_mg", "red_meat_g"])
    roles = {"sodium_mg": "exposure", "red_meat_g": "exposure", "kcal": "energy"}
    columns = ["sodium_mg", "red_meat_g", "kcal"]
    text = describe_model(adjustment, columns, roles, columns).text
    assert "carries no energy" not in text, text


# Covariates that carry energy in a unit their names state, which ``energy_terms`` reads: alcohol
# at 7 kcal/g (``alcohol_g``, and NHANES's DR1TALCO, alcohol in grams in DR1TOT_L's codebook) and
# fat at 9 (FAO 2003's general factors). Sodium and blood-pressure models routinely adjust for
# alcohol.
ENERGY_COVARIATES = [["alcohol_g"], ["DR1TALCO"], ["fat_g"], ["fat_g", "alcohol_g"]]


def _beside(method: str, exposure: str, covariates: list[str]):
    """``describe_model`` for one study factor beside ``covariates``, total energy kept; and the
    study factor's column in the model."""
    from turbotab.core.methods.energy import describe_model

    adjustment = d.EnergyAdjustment(method=method, energy_column="kcal", nutrients=[exposure])
    roles = {exposure: "exposure", "kcal": "energy", **{c: "covariate" for c in covariates}}
    name = f"{exposure}_adj" if method == "residual" else exposure
    columns = [exposure, "kcal", *covariates]
    return describe_model(adjustment, columns, roles, [name, "kcal", *covariates]), name


@pytest.mark.parametrize("method", ["standard", "residual"])
@pytest.mark.parametrize("covariates", ENERGY_COVARIATES, ids="+".join)
def test_a_nutrient_with_no_energy_beside_a_covariate_that_carries_energy_is_substituted_for_nothing(
        method, covariates):
    """The RF-1 verifier's V1. Sodium carries no energy (no Atwater factor), so with total energy
    fixed more of it displaces no source, whatever the covariates carry; alcohol and fat do carry
    energy, so a covariate's own coefficient, at fixed total energy, is a swap for the sources
    left out. The sentence says the nutrient displaces nothing and speaks of the covariates'
    coefficients by name, never of "each coefficient", which would cover sodium's; sodium's column
    has no swap, and each covariate's has one."""
    from turbotab.core.methods.energy import energy_factor

    assert energy_factor("sodium_mg").factor is None
    assert [energy_factor(c).factor for c in covariates] == [
        {"alcohol_g": 7.0, "DR1TALCO": 7.0, "fat_g": 9.0}[c] for c in covariates]
    described, name = _beside(method, "sodium_mg", covariates)
    text = described.text
    assert "the nutrient carries no energy, so it displaces no energy source" in text, text
    assert "each coefficient is a substitution" not in text, text
    assert "implicitly a substitution" not in text, text
    assert "more of the nutrient at the same total energy, in place of" not in text, text
    assert name not in described.swaps and name not in described.terms, described
    assert set(described.swaps) == set(covariates), described.swaps
    for c in covariates:
        assert c in text, text


@pytest.mark.parametrize("method", ["standard", "residual"])
def test_a_nutrient_that_carries_energy_beside_a_covariate_that_carries_energy_keeps_its_swap(
        method):
    """The other side: protein carries energy (4 kcal/g), and beside alcohol (7) both are named, so
    each coefficient, protein's among them, is a swap for the sources left out of the model:
    carbohydrate, fat and the rest (NUTRITION_PACK §04: "each coefficient is a substitution for the
    energy sources not in the model")."""
    from turbotab.core.methods.energy import energy_factor

    assert (energy_factor("protein_g").factor, energy_factor("alcohol_g").factor) == (4.0, 7.0)
    described, name = _beside(method, "protein_g", ["alcohol_g"])
    assert "carries no energy" not in described.text, described.text
    assert ("each coefficient is a substitution in place of the energy sources not in the model: "
            "carbohydrate, fat, other") in described.text, described.text
    assert name in described.swaps and "alcohol_g" in described.swaps, described.swaps


def test_the_teaching_scopes_the_substitution_reading_to_nutrients_that_carry_energy():
    """The pack's substitution reading, scoped to a nutrient that carries energy under its SETTLED
    badge; sodium's reading in a section of its own, the same one on both cards (the verifier's
    V2)."""
    energy = _entry("energy_adjustment")
    estimand = _entry("estimand")
    for body in (_section(energy, "What the coefficient means")["body"],
                 _section(estimand, "A substitution, not an addition")["body"]):
        assert body.startswith("For a nutrient that carries energy, "), body
        assert "sodium" not in body, body
    nothing = [_section(e, "A nutrient with no energy") for e in (energy, estimand)]
    assert nothing[0] == nothing[1]
    assert "sodium" in nothing[0]["body"] and "displaces nothing" in nothing[0]["body"]
    assert "energy source" in _term(energy, "substitution")["definition"].split(";")[1]


# ── D-012 · what averaging replicates does ────────────────────────────────────


def test_averaging_replicates_divides_the_error_variance_by_up_to_their_number():
    """k = 2 recalls a person: with independent day-to-day errors the error variance of each
    person's mean is σ²/2; with errors correlated at ρ = 0.5 (consecutive days) it is σ²(1 + ρ)/2,
    less than a twofold cut. The note says "up to"."""
    rng = np.random.default_rng(12)
    people, sigma = 100_000, 1.0
    independent = rng.normal(0, sigma, (people, 2)).mean(axis=1).var()
    rho = 0.5
    shared = rng.normal(size=people)
    correlated = (np.sqrt(rho) * shared[:, None]
                  + np.sqrt(1 - rho) * rng.normal(size=(people, 2))).mean(axis=1).var()
    assert independent == pytest.approx(sigma ** 2 / 2, rel=0.02)
    assert correlated == pytest.approx(sigma ** 2 * (1 + rho) / 2, rel=0.02)
    assert sigma ** 2 / correlated < 2 < sigma ** 2 / independent * 1.05


def _aggregation_notes(recalls, kind: str, lens: str = "dietary") -> list[str]:
    from turbotab.core import coach
    from turbotab.core.consequences import RowFlowView, RowStep
    from turbotab.core.tests.test_coach import ctx_for

    base = ProjectState(lens=[lens], target="hba1c",
                        grain={"grain": "repeated", "id_column": "participant_id"},
                        repeat_kind=d.RepeatSpec(repeat_kind=kind))
    steps = [RowStep(key="loaded", label="Rows", n=600), RowStep(key="combined", label="People",
                                                                  n=300)]
    view = RowFlowView(title="One row per person", caption="Rows combine.", before=steps[:1],
                       after=steps)
    coach.annotate(d.SetAggregation(method="mean"), [view], ctx_for(recalls, base))
    return [n.text for n in view.coach]


@pytest.mark.parametrize("lens", ["dietary", "clinical"])
def test_the_replicates_note_names_the_error_variance_of_each_mean(recalls, lens):
    """Replicates can be recalls on different days or readings at one visit (duplicate assays,
    triplicate blood pressures), whose errors are reading to reading: the note names the error of
    each mean, not its source, under every lens; the simulation above holds for any independent
    errors, whatever their source."""
    (said,) = _aggregation_notes(recalls, "repeats", lens)
    assert "within-person variance" not in said and "day-to-day" not in said
    assert said == "Averaging `2` replicates cuts each mean's error variance up to `2`-fold."


def test_the_mean_option_names_no_source_for_the_replicates_error():
    option = _option(_entry("aggregation"), "mean")["consequence"]
    assert "day-to-day" not in option and "error" in option, option


def test_time_points_get_no_replicates_note(recalls):
    assert _aggregation_notes(recalls, "time_points") == [
        "Averaging `2` time points erases change between them."]


# ── A-106 · the event's share, and whether anything was resampled ─────────────


def _imbalance_fit(tmp_path: Path, imbalance: str) -> str:
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests import modeling_fixtures as mf

    rng = np.random.default_rng(5)
    n = 600
    x = rng.normal(size=n)
    frame = pd.DataFrame({"pid": np.arange(n), "x": x.round(3), "age": rng.normal(50, 9, n).round(1),
                          "case": rng.random(n) < 1 / (1 + np.exp(-(-2.0 + x)))})
    paths = mf.ingest_frame(frame, tmp_path / imbalance)
    split = mf.split_bundle(np.arange(n), holdout=0.0)
    ti = mf.target_info("binary", "case")
    st = mf.state(roles={"pid": "identifier", "x": "exposure", "age": "covariate"}, target="case",
                  task="binary", event="True", purpose="prediction", models=["linear"],
                  lens=["clinical"], levers=d.LeverSpec(imbalance=imbalance))
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    return fit.data["imbalance"]


def test_a_fit_with_no_imbalance_correction_says_none_was_applied(tmp_path):
    said = _imbalance_fit(tmp_path, "none")
    assert said.startswith("The event, `True`, is ") and "No resampling" in said


def test_a_fit_with_class_weights_never_says_none_was_applied(tmp_path):
    said = _imbalance_fit(tmp_path, "weights")
    assert said.startswith("The event, `True`, is ")
    assert "No resampling" not in said and "was applied" not in said
    assert "corrected" in said and "recalibrated" in said


# ── A-117 · multiple imputation with the outcome, unless the purpose is inference ─


@pytest.mark.parametrize("purpose", ["prediction", None])
def test_multiple_imputation_is_refused_unless_the_purpose_is_inference_in_sisks_scope(purpose):
    """Sisk et al. 2023's abstract (Stat Methods Med Res 32:1461; quoted in methods/missing.py, and
    read on PubMed 37105540 by the RF-1 verifier): "With complete data available at deployment, the
    outcome should be used to impute development data when using multiple imputation and omitted
    under regression imputation. When missingness is allowed at deployment, omitting the outcome
    from the imputation model at the development was preferred." The refusal fires with no purpose
    too, so it says "unless the purpose is inference"; it cites them for rows that can arrive with
    blanks, and says their use of the outcome with complete rows is for multiple imputation, not
    for a single fill such as the one it offers (the verifier's V6)."""
    state = ProjectState(target="y", roles={"x": "exposure", "z": "covariate"}, purpose=purpose)
    with pytest.raises(Refusal) as refused:
        validate(d.SetMissing(strategy="multiple_imputation"), {"state": state})
    message = refused.value.message
    assert refused.value.code == "imputation_with_the_outcome"
    assert "Unless the purpose is inference" in message
    assert "under prediction the imputation is fit" not in message
    assert "blanks" in message and "Sisk et al. (2023)" in message
    assert ("with complete rows at deployment they recommend using it in multiple imputation, not "
            "in a single regression fill") in message, message
    assert not message.endswith("they recommend using it."), message
    assert refused.value.exits[0]["decision"]["strategy"] == "impute"  # the single fill offered


def test_multiple_imputation_under_inference_is_not_refused():
    state = ProjectState(target="y", roles={"x": "exposure", "z": "covariate"},
                         purpose="inference")
    validate(d.SetMissing(strategy="multiple_imputation"), {"state": state})


# ── A-121 · inside a near dependency ──────────────────────────────────────────


def _generic_dependency(n: int = 400, seed: int = 3) -> tuple[ProjectState, pd.DataFrame]:
    """`score` is half `item_a` plus half `item_b`, give or take a little: no part of any total by
    name, no energy."""
    rng = np.random.default_rng(seed)
    a, b = rng.normal(50, 10, n), rng.normal(30, 8, n)
    f = pd.DataFrame({"item_a": a, "item_b": b, "score": 0.5 * a + 0.5 * b + rng.normal(0, 0.2, n),
                      "age": rng.uniform(20, 80, n)})
    f["outcome"] = 0.1 * f["score"] + rng.normal(size=n)
    roles = {"score": "exposure", "item_a": "covariate", "item_b": "covariate", "age": "covariate"}
    state = ProjectState(lens=["clinical"], target="outcome", purpose="inference", roles=roles,
                         estimand=d.EstimandSpec(exposure="score", measure="mean_difference"))
    return state, f


def test_a_dependency_that_is_no_total_says_the_contrast_changes_not_what_it_replaces():
    from turbotab.core import materiality as M

    state, f = _generic_dependency()
    n = M.collinear_noticing(state, f)
    assert n is not None and n.predicted.band == 2
    words = n.predicted.words
    assert "less of what it replaces" not in words and "less of the rest" not in words
    assert "little of its own variation" in words and "contrast" in words


def test_a_part_beside_its_total_says_more_of_it_is_less_of_the_rest_of_the_total():
    from turbotab.core import materiality as M
    from turbotab.core.tests.test_collinear_adjustment import ADJUST, state, table

    f = table(sugar_inside=True)  # carb is sugar plus starch, built so
    n = M.collinear_noticing(state([*ADJUST, "starch"]), f)
    assert n.predicted.band == 2
    assert "more `sugar` means less of the rest of `carb`" in n.predicted.words


# ── C-110 · an ungrouped seal's held-out scores, and nothing of the folds ─────


def _seal(grain: dict) -> tuple:
    from turbotab.core.seal import decide_basis

    frame = pd.DataFrame({"pid": np.repeat(np.arange(40), 3)})
    state = ProjectState(target="y", roles={"pid": "identifier"}, grain=grain)
    return decide_basis(state, frame, ["pid"])


PID = np.repeat(np.arange(40), 3)  # 40 units, 3 rows each
SITE = PID % 4  # 4 sites, each unit wholly in one


def _spans(frame: pd.DataFrame, column: str) -> pd.Series:
    """How many distinct values of ``column`` (partition or fold) each unit's rows take, over the
    rows that have one."""
    rows = np.arange(len(PID))
    got = frame.set_index("row_id")[column].reindex(rows).to_numpy()
    keep = got != -1 if column == "fold" else np.ones(len(rows), bool)
    return pd.Series(got[keep]).groupby(PID[keep]).nunique()


@pytest.mark.parametrize("grain", [{"grain": "one_row_per_unit"}, {"grain": "unknown"},
                                   {"grain": "repeated", "id_column": "person"}])
def test_a_seal_drawn_by_row_over_a_repeating_identifier_labels_its_held_out_scores_exploratory(
        grain):
    """By construction a unit's rows can then land on both sides of the held-out draw, so its
    scores are within-unit and read high (Saeb et al. 2017); the seal labels them exploratory. It
    says nothing of the folds: it is drawn before the validation is chosen, and only the split's
    note, written after, knows whether they are a grouping's levels."""
    from turbotab.core.stages.rows import draw_split

    basis, column = _seal(grain)
    assert column is None and basis.exploratory
    assert "treat held-out scores as exploratory" in basis.sentence.lower(), basis.sentence
    assert "fold" not in basis.sentence, basis.sentence
    frame, _ = draw_split(np.arange(len(PID)), holdout=0.2, seed=0, folds=5, groups=None)
    assert (_spans(frame, "partition") > 1).any()  # a unit on both sides of the seal


def test_under_internal_external_validation_the_folds_are_the_sites_and_the_seal_says_nothing_of_them():
    """The verifier's case: `pid` repeats 3 times over 40 units in 4 sites while each row was said
    to be a unit, so the basis is abandoned; internal–external validation then makes each site a
    fold (``draw_split``'s clusters), and no unit spans two folds. The seal's sentence, true under
    every validation, says nothing of how the folds were drawn."""
    from turbotab.core.stages.rows import draw_split

    basis, _ = _seal({"grain": "one_row_per_unit"})
    assert basis.state == "abandoned"
    frame, info = draw_split(np.arange(len(PID)), holdout=0.2, seed=0, folds=5, groups=None,
                             clusters=SITE, cluster_column="site")
    assert info["folds"] == 4 and (_spans(frame, "fold") == 1).all()
    assert "fold" not in basis.sentence and "cross-validated" not in basis.sentence


def test_a_grouped_seal_says_nothing_of_exploratory_scores():
    basis, column = _seal({"grain": "repeated", "id_column": "pid"})
    assert column == "pid" and not basis.exploratory
    assert "exploratory" not in basis.sentence


# ── C-144 · the demographic guess ─────────────────────────────────────────────


def _guess(column: str, exposure: str = "hba1c"):
    from turbotab.core.covariate_guesses import guess

    state = ProjectState(lens=["clinical"], target="cimt", purpose="inference",
                         roles={exposure: "exposure", column: "covariate"},
                         estimand=d.EstimandSpec(exposure=exposure, measure="mean_difference"))
    return guess(column, state)


@pytest.mark.parametrize("column", ["age", "sex", "income", "INDFMPIR", "marital_status"])
def test_a_demographic_guess_names_no_diet_and_says_which_may_follow(column):
    """Under a clinical study factor: the reason is the lens's own, no diet. Age, sex and race are
    set before what you study; income, occupation and marital status measured at the same visit
    may also follow it (the referees' and the audit's point), which the reason says."""
    g = _guess(column)
    assert g.key == "demographic"
    assert "diet" not in g.reason
    assert "set before what you study" in g.reason and "may also follow" in g.reason
    assert dict(g.answers) == {"causes_exposure": "yes", "causes_outcome": "yes",
                               "after_exposure": "no"}


def test_a_body_measure_takes_no_demographic_guess():
    g = _guess("waist")
    assert g.key == "body" and dict(g.answers)["after_exposure"] == "unknown"


# ── C-241 · whether the group is in the imputation model ──────────────────────


def _quality_lever(group_is_a_predictor: bool) -> str:
    from turbotab.core.stages.explore import _quality_finding

    rng = np.random.default_rng(2)
    n = 300
    race = rng.choice(["a", "b", "c"], n)
    x = rng.normal(size=n)
    x[(race == "a") & (rng.random(n) < 0.4)] = np.nan
    frame = pd.DataFrame({"race": race, "x": x, "z": rng.normal(size=n)})
    roles = {"x": "exposure", "z": "covariate", "race": "covariate" if group_is_a_predictor
             else "excluded"}
    state = ProjectState(lens=["clinical"], target="y", purpose="inference", roles=roles)
    predictors = ["x", "z", *(["race"] if group_is_a_predictor else [])]
    finding = _quality_finding(state, frame, predictors, "race", "race or ethnicity", "analyzed",
                               None)
    return finding.lever.options[0].sound


# The group's clause follows the variables it refers to, so its "those" cannot be read as the
# complete cases, which come last.
VARIABLES = "Multiple imputation fills blanks from the analysis's variables; "
CASES = "; complete cases state their assumption"


def test_a_group_that_is_a_predictor_is_said_to_be_in_the_imputation_model():
    sound = _quality_lever(True)
    assert sound == (f"{VARIABLES}`race` is one of those variables, so the imputation model holds "
                     f"it{CASES}"), sound


def test_a_group_that_is_no_predictor_is_said_to_be_left_out_of_the_imputation_model():
    sound = _quality_lever(False)
    assert "with the group in its model" not in sound
    assert sound == (f"{VARIABLES}`race` is not one of those variables, so the imputation model "
                     f"leaves it out{CASES}"), sound


# ── the roles ruling: what a role is, never what reaches the models ───────────

MODELS_GET = re.compile(r"\b(?:enters?|stays? out of|kept out of) the (?:models?|predictors)\b"
                        r"|\bthe models? reads?\b|\bbecomes? (?:a )?predictors?\b"
                        r"|\bcan become predictors\b|\bnever a predictor\b"
                        # "adjusted for" says the models hold it (the verifier's V7); "to adjust
                        # for" says what the role is for.
                        r"|\badjusted for\b", re.I)

BASE_WORDS = (
    "Only study factors, covariates and the energy column can become predictors; every other role "
    "stays out of the models.",
    "Adjusted for, such as age or sex; enters the models beside the study factors.",
    "Groups participants, such as a site or household; kept out of the predictors.",
    "A column the model reads to predict the outcome, such as a nutrient intake or age.",
    "Names a participant; never a predictor, and keeps a person's rows together in the split.",
    "A variable adjusted for so that the estimate of what you study is not confounded by it, such "
    "as age or sex.",
)


def test_the_estimate_why_says_the_record_keeps_each_column_that_holds_no_predictor_role():
    """The why's last clause names what the record keeps: after "none holds a predictor role" a
    bare "it" has nothing to refer to."""
    why = _entry("roles", "inference")["why"]
    assert why.endswith("so none holds a predictor role, and the record keeps each of them."), why
    assert not re.search(r"\bkeeps it\b", why), why


def test_the_guard_fires_on_the_base_words_and_is_silent_on_a_roles_definition():
    for old in BASE_WORDS:
        assert MODELS_GET.search(old), old
    assert not MODELS_GET.search("Groups participants, such as a site or household; not a "
                                 "predictor role.")
    assert not MODELS_GET.search("Something to adjust for, such as age or sex: a predictor role "
                                 "beside the study factors.")


def test_under_predict_no_grouping_reaches_the_models_so_the_one_liner_holds():
    """The Predict one-liner keeps "every other role stays out of the predictors", which the
    landing ruled: fixed effects are an Estimate answer, so a grouping answered "adjust for it"
    puts its column in the models under inference and in none under Predict."""
    from turbotab.core.models.pipeline import model_predictors

    fe = {"clusters": {"column": "site", "adjust": "fixed_effects"}}
    assert "site" in model_predictors(_inference(**fe))
    predict = _inference(**fe).model_copy(update={"purpose": "prediction"})
    roles = _inference().roles
    assert all(roles[c] in ("exposure", "covariate", "energy") for c in model_predictors(predict))
    assert _entry("roles", "prediction")["one_liner"].endswith("stays out of the predictors.")


def _role_words(purpose: str | None) -> list[str]:
    roles = _entry("roles", purpose)
    return [roles["one_liner"], roles["why"], *(o["consequence"] for o in roles["options"]),
            *(t["definition"] for t in roles["terms"])]


def _inference(**slots) -> ProjectState:
    roles = {"pid": "identifier", "site": "cluster", "sugar": "exposure", "age": "covariate",
             "waist": "covariate", "kcal": "energy"}
    return ProjectState(lens=["clinical"], target="glucose", purpose="inference", roles=roles,
                        **slots)


def _waist_a_mediator() -> dict:
    """The adjustment answers for a total effect of sugar: age, set before it, causes the outcome
    (adjusted); waist, which sugar can change and which causes the outcome, is a mediator, which a
    total effect leaves out (VanderWeele 2019's criterion, ``estimand.derive``)."""
    def answer(after: str):
        return d.AdjustmentAnswer(exposure="sugar", causes_exposure="no", causes_outcome="yes",
                                  after_exposure=after)

    return {"estimand": d.EstimandSpec(exposure="sugar", effect="total",
                                       measure="mean_difference"),
            "adjustment": {"age": answer("no"), "waist": answer("yes")}}


@pytest.mark.parametrize("case", ["plain", "fixed_effects", "identifier_fixed_effects", "drop",
                                  "left_out"])
def test_under_estimate_the_roles_say_what_each_role_is_whatever_the_models_get(case):
    """The cases the landing's verifier reproduced: with nothing changed the models get the three
    predictor roles; a fixed-effects grouping column puts a cluster-role column in them (and an
    identifier-role one, named as the grouping); a missing-values drop leaves a covariate out, and
    so do the adjustment answers that make it a mediator of a total effect (the RF-1 verifier's
    V7), while it keeps the covariate role. The roles card is one card for all of them, so it says
    what each role is and nothing of what the models get: the covariate term says a covariate is
    one to adjust for, never that it is adjusted for."""
    from turbotab.core.decisions import left_out
    from turbotab.core.models.pipeline import model_predictors

    slots = {"plain": {},
             "fixed_effects": {"clusters": {"column": "site", "adjust": "fixed_effects"}},
             "identifier_fixed_effects": {"clusters": {"column": "pid",
                                                       "adjust": "fixed_effects"}},
             "drop": {"missing": MissingSpec(strategy="complete_case", drop_columns=["waist"])},
             "left_out": _waist_a_mediator()}
    state = _inference(**slots[case])
    got = model_predictors(state)
    want = {"plain": ["sugar", "age", "waist", "kcal"],
            "fixed_effects": ["sugar", "age", "waist", "kcal", "site"],
            "identifier_fixed_effects": ["sugar", "age", "waist", "kcal", "pid"],
            "drop": ["sugar", "age", "kcal"],
            "left_out": ["sugar", "age", "kcal"]}[case]
    assert got == want
    assert state.roles["waist"] == "covariate"
    assert ("waist" in left_out(state)) is (case in ("drop", "left_out"))
    for text in _role_words("inference"):
        assert not MODELS_GET.search(text), text
    roles = _entry("roles", "inference")
    assert "predictor role" in roles["one_liner"]
    assert "predictor role" in _option(roles, "covariate")["consequence"]
    assert "not a predictor role" in _option(roles, "cluster")["consequence"]
    covariate = _term(roles, "covariate")["definition"]
    assert covariate.startswith("A variable to adjust for, so that"), covariate


@pytest.mark.parametrize("case", ["plain", "drop"])
def test_under_predict_the_options_and_the_predictor_term_agree_whatever_the_models_get(case):
    """Under Predict each predictor role's option says "a predictor", and the term defines a
    predictor as a kind of column, never as what the model reads: a drop leaves the column a
    predictor in that sense, though no model reads it."""
    from turbotab.core.models.pipeline import model_predictors

    slots = {"plain": {}, "drop": {"missing": MissingSpec(strategy="complete_case",
                                                          drop_columns=["waist"])}}
    state = _inference(**slots[case]).model_copy(update={"purpose": "prediction"})
    assert ("waist" in model_predictors(state)) == (case == "plain")
    roles = _entry("roles", "prediction")
    for text in [*(o["consequence"] for o in roles["options"]),
                 *(t["definition"] for t in roles["terms"])]:
        assert not MODELS_GET.search(text), text
    definition = _term(roles, "predictor")["definition"]
    assert definition.startswith("A column") and "could predict" in definition
    for value in ("exposure", "energy", "covariate"):
        assert "predictor" in _option(roles, value)["consequence"].lower()
