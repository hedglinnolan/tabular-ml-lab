"""Customary and sound, as two labels on every option (BLUEPRINT north star 5; audit RO-06, RO-07).

Nolan (2026-10-01): *"Sometimes those decisions are backed by what is customary in the field,
sometimes they're backed by mathematically sound practices, but both may not be true every single
time."* So every option of the energy, missing-values, split and exclusions questions carries:

* **customary in <field>**, with the source that documents the practice;
* **sound for <purpose>**, a verdict (sound · conditional · unsound) with its reason;

and the options are ordered by soundness for the declared purpose, which differs between prediction
and inference for each question. Where the field's customary first choice is not the soundest, one
line names the tension and the researcher decides (AUDIT_REPORT §3 supplies the labels).

The sources quoted here, read on 2026-10-03:

* Sterne et al. 2009, *BMJ* 338:b2393 (PMC2714692): "Researchers usually address missing data by
  including in the analysis only complete cases", and "When it is plausible that data are missing at
  random, but not completely at random, analyses based on complete cases may be biased."
* Steyerberg 2018, *J Clin Epidemiol* 103:131–133 (abstract, PMID 30063954): "The standard
  requirement in major medical journals is nowadays that validity outside the development sample
  needs to be shown", and "In small samples, cross-validation and bootstrapping are more efficient
  approaches. In conclusion, random data splitting should be abolished for validation of prediction
  models."
* Banna et al. 2017, *Front Nutr* 4:45 (PMC5622407), on fixed kcal cut-offs: "a drawback of this
  crude approach is that it is not individualized and does not capture all implausible reports";
  and "Regardless of which method is used, for the time being, analyses in the total sample without
  exclusion of participants should also be conducted and reported."
"""
from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

Purpose = Literal["prediction", "inference"]
Verdict = Literal["sound", "conditional", "unsound"]
PURPOSES: tuple[str, ...] = ("prediction", "inference")

STERNE = "Sterne et al. 2009, BMJ 338:b2393"
STEYERBERG = "Steyerberg 2018, J Clin Epidemiol 103:131"
BANNA = "Banna et al. 2017, Front Nutr 4:45"
SISK = "Sisk et al. 2023, Stat Methods Med Res 32:1461"
SHMUELI = "Shmueli 2010, Stat Sci 25:289"
TOMOVA = "Tomova et al. 2022, AJCN 115:189"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True,
                              json_schema_serialization_defaults_required=True)


class Customary(_Model):
    """Where the field uses the option: ``text`` says how, ``field`` in which field, ``source``
    where that is documented."""

    field: str
    text: str
    source: str


class Sound(_Model):
    """Whether the option is sound for ``purpose``, and why."""

    purpose: Purpose
    verdict: Verdict
    reason: str


class LabeledOption(_Model):
    key: str
    label: str
    customary: Customary
    sound: Sound


class LabeledQuestion(_Model):
    """A question's options, soundest first for the purpose, each with its two labels, and the one
    line naming the tension when the field's customary first choice is not the soundest."""

    question: Literal["energy_adjustment", "missing", "split", "exclusions"]
    purpose: Purpose
    options: list[LabeledOption]
    customary_first: str  # the option the field reaches for first
    tension: str | None


# One row per option: (key, label, customary field, customary text, source,
#                      {purpose: (verdict, reason)}, {purpose: rank})
Row = tuple[str, str, str, str, str, Mapping[str, tuple[str, str]], Mapping[str, int]]


def _build(question: str, purpose: str | None, rows: Sequence[Row], customary_first: str,
           tensions: Mapping[str, str], keys: Sequence[str] | None = None) -> LabeledQuestion:
    p = purpose if purpose in PURPOSES else "prediction"
    chosen = [r for r in rows if keys is None or r[0] in keys]
    chosen.sort(key=lambda r: r[6][p])
    options = [LabeledOption(key=k, label=label,
                             customary=Customary(field=field, text=text, source=source),
                             sound=Sound(purpose=p, verdict=sound[p][0], reason=sound[p][1]))
               for k, label, field, text, source, sound, _ in chosen]
    first = options[0].key if options else None
    tension = tensions.get(p) if first is not None and first != customary_first else None
    return LabeledQuestion(question=question, purpose=p, options=options,
                           customary_first=customary_first, tension=tension)


# ── energy adjustment (methods/energy.py METHOD_TABLE; AUDIT_REPORT §3.1) ────


def energy(purpose: str | None, applicable: Sequence[str] | None = None) -> LabeledQuestion:
    """The energy models, labeled from the method table and ordered by ``RANKING`` (BLUEPRINT §12
    ruling 2). Under prediction the methods that keep total energy lead, and the line says the
    residual method's energy-dropped form, the field's default, discards energy's signal (RO-07)."""
    from turbotab.core.methods.energy import METHOD_TABLE, RANKING

    field = "nutritional epidemiology"
    sources = {
        "none": "NUTRITION_PACK §04 and §08 (the crude model beside the adjusted ones)",
        "standard": "Willett, Nutritional Epidemiology (NUTRITION_PACK §04)",
        "residual": "McCullough & Byrd 2023, AJE 192:1801",
        "residual_energy_dropped": "Willett, Nutritional Epidemiology (NUTRITION_PACK §04: the "
                                   "field default)",
        "density_multivariate": "NUTRITION_PACK §04",
        "density": "NUTRITION_PACK §04",
        "partition": "NUTRITION_PACK §04",
        "all_components": f"{TOMOVA}; disputed by Willett, Stampfer & Tobias 2022",
    }
    verdicts = {
        "inference": {"all_components": "sound", "standard": "conditional", "residual": "sound",
                      "partition": "conditional", "density_multivariate": "conditional",
                      "none": "conditional", "residual_energy_dropped": "unsound",
                      "density": "unsound"},
        "prediction": {"standard": "sound", "residual": "sound", "density_multivariate": "sound",
                       "partition": "sound", "all_components": "sound", "none": "conditional",
                       "residual_energy_dropped": "unsound", "density": "unsound"},
    }
    rows: list[Row] = []
    for m, row in METHOD_TABLE.items():
        rows.append((m, row["label"], field, row["customary"], sources[m],
                     {p: (verdicts[p][m], row["sound"][p]) for p in PURPOSES},
                     {p: RANKING[p].index(m) for p in PURPOSES}))
    tensions = {
        "inference": (f"The residual method is the field's default; all components ranks first for "
                      f"substitution, with no composite-variable bias at a precision cost ({TOMOVA})."),
        "prediction": ("The residual method is the field's default; in its energy-dropped form it "
                       "discards total energy's signal, so the methods that keep energy rank first."),
    }
    out = _build("energy_adjustment", purpose, rows, "residual_energy_dropped", tensions, None)
    if applicable is not None:
        ok = set(applicable)
        ordered = ([o for o in out.options if o.key in ok]
                   + [o for o in out.options if o.key not in ok])
        out = out.model_copy(update={"options": ordered})
        lead = ordered[0] if ordered else None
        if out.purpose == "inference" and lead is not None and lead.key != "all_components":
            # The line never says all components leads when it cannot run on these columns.
            out = out.model_copy(update={"tension": (
                f"The residual method is the field's default; all components would rank first for "
                f"substitution ({TOMOVA}) but cannot run on these columns, so the "
                f"{lead.label[0].lower() + lead.label[1:]} leads.")})
    return out


# ── missing values (methods/missing.py; AUDIT_REPORT §3.3) ───────────────────


def missing(purpose: str | None) -> LabeledQuestion:
    from turbotab.core.methods.missing import METHODS

    fields = {
        "multiple_imputation": ("epidemiology", STERNE),
        "complete_case": ("epidemiology", f"{STERNE}: \"Researchers usually address missing data "
                                          f"by including in the analysis only complete cases\""),
        "impute": ("machine learning", "scikit-learn's SimpleImputer, the pipelines' default fill"),
        "indicators": ("prediction from health records",
                       "Sperrin et al. 2020, J Clin Epidemiol 125:183"),
        "missing_category": ("survey and health-record analysis",
                             "NUTRITION_PACK §06 (a blank that means not asked)"),
    }
    verdicts = {
        "inference": {"multiple_imputation": "sound", "complete_case": "conditional",
                      "impute": "unsound", "indicators": "unsound", "missing_category": "conditional"},
        "prediction": {"multiple_imputation": "unsound", "complete_case": "conditional",
                       "impute": "sound", "indicators": "conditional", "missing_category": "sound"},
    }
    rows: list[Row] = []
    for key, m in METHODS.items():
        field, source = fields[key]
        rows.append((key, m.label, field, m.customary, source,
                     {p: (verdicts[p][key], m.sound[p]) for p in PURPOSES},
                     {p: m.order[p] for p in PURPOSES}))
    tensions = {
        "inference": ("Complete cases are the field's habit; when the blanks depend on measured "
                      "variables they may be biased, and multiple imputation with the outcome is "
                      f"sounder ({STERNE})."),
        "prediction": ("Complete cases are the field's habit, but a new row with a blank cannot be "
                       f"scored; a fill learned in each training fold without the outcome can ({SISK})."),
    }
    return _build("missing", purpose, rows, "complete_case", tensions)


# ── the split (seal.py's options, models/validation.py's; AUDIT_REPORT §3.4) ─


SPLIT_ROWS: list[Row] = [
    ("holdout", "Hold out a share of rows", "clinical prediction and machine learning",
     "The standard requirement in major medical journals: validity shown outside the development "
     "sample", STEYERBERG,
     {"inference": ("unsound", f"Serves no estimand: every analyzed row estimates the coefficients "
                               f"whatever is held out, so it costs rows for a score ({SHMUELI})."),
      "prediction": ("conditional", "A lockbox against the analyst's own overfitting, sound at "
                                    "large sizes; at small ones its rows neither train nor score "
                                    f"the pipeline ({STEYERBERG}).")},
     {"inference": 4, "prediction": 4}),
    ("kfold", "Cross-validation, once", "machine learning",
     "The default resampling of scikit-learn and caret", "scikit-learn and caret documentation",
     {"inference": ("sound", "The scores describe how well the model fits; every row still "
                             "estimates the coefficients."),
      "prediction": ("conditional", "Every row trains and scores, but one partition is noisy at "
                                    "small sizes.")},
     {"inference": 0, "prediction": 2}),
    ("repeated_kfold", "Repeated cross-validation", "clinical prediction",
     "Acceptable internal validation", "CLINICAL_SURVEY_PACK §A5.5",
     {"inference": ("sound", "Steadier fit scores; it does not change the estimates."),
      "prediction": ("sound", "Draws the folds several times, so the score no longer rests on one "
                              "partition.")},
     {"inference": 1, "prediction": 1}),
    ("bootstrap", "Bootstrap optimism correction", "clinical prediction",
     "The recommended default (Harrell, Lee & Mark 1996)", "CLINICAL_SURVEY_PACK §A5.5",
     {"inference": ("sound", "Corrects the fit scores for optimism; it does not change the "
                             "estimates."),
      "prediction": ("sound", "Uses every row and refits the whole pipeline in each resample "
                              f"({STEYERBERG}: \"more efficient approaches\" in small samples).")},
     {"inference": 2, "prediction": 0}),
    ("internal_external", "Internal–external, by cluster", "clinical prediction",
     "Increasingly expected across sites or periods", "Collins et al. 2024, BMJ 384:e074819",
     {"inference": ("conditional", "Shows how fit varies across groups; the grouping question "
                                   "decides the intervals."),
      "prediction": ("sound", "Each site or period scored by models fit on the others, with the "
                              "spread: what a new site would see.")},
     {"inference": 3, "prediction": 3}),
]


def split(purpose: str | None, order: Sequence[str] | None = None) -> LabeledQuestion:
    """The split question's options, in ``order`` when the seal plan gives one (its holdout and
    validation options as it offers them: ``seal.plan``), else in the table's order for the
    purpose. Where the plan leads with the holdout (prediction, at a size where the held-out rows
    measure precisely), the holdout is sound there, custom and soundness agree, and no tension is
    named."""
    rows = list(SPLIT_ROWS)
    p = purpose if purpose in PURPOSES else "prediction"
    if order:
        ranks = {k: i for i, k in enumerate(order)}
        rows = [(*r[:6], {**r[6], p: ranks.get(r[0], len(ranks))}) for r in rows]
        if p == "prediction" and order[0] == "holdout":
            rows = [(*r[:5], {**r[5], "prediction": (
                "sound", "At this size the held-out rows measure precisely, and they guard against "
                         "the analyst's own overfitting.")}, r[6]) if r[0] == "holdout" else r
                    for r in rows]
    tensions = {
        "inference": ("Holding out rows is the journals' habit; under inference every row estimates "
                      f"the coefficients, so no holdout comes first ({SHMUELI})."),
        "prediction": ("A single holdout is the journals' habit; resampling the whole pipeline uses "
                       f"every row and is more efficient at this size ({STEYERBERG})."),
    }
    return _build("split", purpose, rows, "holdout", tensions)


# ── exclusions (proposals.py's screens; AUDIT_REPORT §3.2) ───────────────────


EXCLUSION_ROWS: list[Row] = [
    ("keep_every_row", "Keep every row", "nutritional epidemiology",
     "Reported beside any screen", f"{BANNA}: \"analyses in the total sample without exclusion of "
                                   f"participants should also be conducted and reported\"",
     {"inference": ("conditional", "The every-row analysis, reported beside the screen so a reader "
                                   "sees how far it moves the answer."),
      "prediction": ("sound", "The model will be used on everyone, misreporters included, so they "
                              "stay in its development rows.")},
     {"inference": 5, "prediction": 0}),
    ("goldberg_schofield", "Goldberg screen", "the misreporting literature",
     "Goldberg's cut-off, energy against predicted needs", BANNA,
     {"inference": ("conditional", "Individualized: accounts for within-person error in intake and "
                                   "expenditure; reported with the every-row analysis beside it."),
      "prediction": ("conditional", "The people it removes are still among those the model will "
                                    "be used on.")},
     {"inference": 0, "prediction": 1}),
    ("willett_2013_by_sex", "Willett 2013, by sex", "nutritional epidemiology",
     "Fixed kcal cut-offs: women 500–3,500, men 800–4,000", BANNA,
     {"inference": ("conditional", "Not individualized and misses some implausible reports "
                                   f"({BANNA}); report the every-row analysis beside it."),
      "prediction": ("conditional", "The people it removes are still among those the model will "
                                    "be used on.")},
     {"inference": 1, "prediction": 2}),
    ("nhs_hpfs_by_sex", "NHS/HPFS, by sex", "nutritional epidemiology",
     "Fixed kcal cut-offs: women 500–3,500, men 800–4,200", f"{BANNA}; NUTRITION_PACK §02",
     {"inference": ("conditional", "Not individualized and misses some implausible reports "
                                   f"({BANNA}); report the every-row analysis beside it."),
      "prediction": ("conditional", "The people it removes are still among those the model will "
                                    "be used on.")},
     {"inference": 2, "prediction": 3}),
    ("sex_neutral_500_5000", "Sex-neutral 500–5,000", "nutritional epidemiology",
     "A fixed kcal cut-off for both sexes", "NUTRITION_PACK §02",
     {"inference": ("conditional", "Cruder than a sex-specific screen; report the every-row "
                                   "analysis beside it."),
      "prediction": ("conditional", "The people it removes are still among those the model will "
                                    "be used on.")},
     {"inference": 3, "prediction": 4}),
    ("sex_neutral_500_3500", "Sex-neutral 500–3,500", "nutritional epidemiology",
     "A fixed kcal cut-off for both sexes", "NUTRITION_PACK §02",
     {"inference": ("conditional", "Women's bounds applied to men too, so it removes more men's "
                                   "plausible intakes; report the every-row analysis beside it."),
      "prediction": ("conditional", "The people it removes are still among those the model will "
                                    "be used on.")},
     {"inference": 4, "prediction": 5}),
]


def exclusions(purpose: str | None, offered: Sequence[str] | None = None) -> LabeledQuestion:
    """The exclusion screens offered on this table (``offered``: their keys), and keeping every
    row, each labeled; the field reaches for a fixed cut-off first."""
    keys = None if offered is None else ["keep_every_row", *offered]
    tensions = {
        "inference": ("Fixed kcal cut-offs are the field's habit; they are not individualized, so "
                      f"the every-row analysis is reported beside the screen ({BANNA})."),
        "prediction": ("Screens are the field's habit; under prediction the people they remove will "
                       "still be scored, so keeping every row comes first."),
    }
    return _build("exclusions", purpose, EXCLUSION_ROWS, "willett_2013_by_sex", tensions, keys)


def labels_for(question: str, purpose: str | None, **kwargs: Any) -> LabeledQuestion:
    return {"energy_adjustment": energy, "missing": missing, "split": split,
            "exclusions": exclusions}[question](purpose, **kwargs)


__all__ = ["Customary", "LabeledOption", "LabeledQuestion", "Sound", "energy", "exclusions",
           "labels_for", "missing", "split"]
