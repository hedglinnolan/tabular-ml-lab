"""Which way round is this table? — question 1.5 of the pre-seal sequence.

## Why this is a question and not a finding

The lens acts at `engine.rank_findings`, which is **presentation**. That is the
right place for it: `OPENING_SEQUENCE.md` §01 argues at length that reframing
annotates and never deletes, so a user who overrules the lens can still reach
the real repair. It is correct for *interpreting* findings and it does nothing
for *structure*.

An assay table exported features-in-rows and samples-in-columns is transposed.
Every finding computed on it is garbage — the "columns" are participants, so
column dtypes are meaningless, missingness per column is missingness per
participant, the impossibility pass compares one subject's whole panel against a
reference range for a single analyte, and the target column the user is about to
pick is an analyte name. **Annotation cannot fix a frame.** Nothing downstream
of a transposed table is worth computing, which is why this one question acts
before the diagnosis rather than after it, and why it is what gives clause 01's
ordering its teeth.

## Why it is asked and not inferred

Same architecture as the grain and the lens, for the same reason: the user
knows, and the engine can only read shape. Transposing on the app's own
authority would be the single most destructive silent act available to it —
every row identity in the file replaced — and Decision A's identity barrier
already classifies it: transposing changes what a row *is*, so it is a
pre-barrier structural repair and may only run before the seal.

## The reading, and why it is this statistic

In a **sample-major** assay table the columns are analytes: they differ in
abundance by orders of magnitude, so the spread of column means is large, while
the rows are comparable samples and the spread of row means is small.
**Feature-major** is the same fact with the axes exchanged.

So the reading is the ratio of those two spreads, measured on a log scale
because concentrations are log-normal by construction:

    s_rows / s_cols,  where s = sd(log10(|mean|)) over the numeric block

Measured on the fixtures rather than chosen: every real fixture in this tree
reads between 0.05 and 1.51, and a transposed copy of
`metabolomics_untargeted.csv` reads 23. The threshold is 4.0 — about 2.6× above
the loudest sample-major fixture and about 6× below the true signal — and the
band between 1/4 and 4 is `undetermined`, which asks nothing.

That last point is the corollary about checks tested only against a constructed
signal: this one was run against every fixture in the tree before a threshold
was picked, and the fixture that comes closest (`wide_assay.csv`, 1.51) is
asserted to stay silent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ── the answers ──────────────────────────────────────────────────────────────

ROWS_ARE_SAMPLES = "rows_are_samples"
ROWS_ARE_FEATURES = "rows_are_features"

SAMPLE_MAJOR = "sample_major"
FEATURE_MAJOR = "feature_major"
UNDETERMINED = "undetermined"

# The lenses whose exports come both ways round. A clinical export or a survey
# is not shipped transposed, and asking would be the pack firing on data it does
# not match — guard #2 of `DOMAIN_PACKS.md` §03.
ASSAY_LENSES = frozenset({"metabolomics", "genomics"})

# Measured, not chosen. See the module docstring.
FEATURE_MAJOR_RATIO = 4.0
SAMPLE_MAJOR_RATIO = 0.25
MIN_ROW_SPREAD = 0.4
MIN_NUMERIC_COLUMNS = 6
MIN_ROWS = 6


class OrientationError(Exception):
    """A frame this cannot honestly turn around."""


def _numeric_block(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    num = df.select_dtypes(include=[np.number])
    num = num.loc[:, num.notna().mean() > 0.5]
    if num.shape[1] < MIN_NUMERIC_COLUMNS or num.shape[0] < MIN_ROWS:
        return None
    return num


def _spread(values: pd.Series) -> Optional[float]:
    """Spread of a set of means on a log scale, or `None` if it cannot be taken.

    Log, because the quantities are concentrations and combine multiplicatively;
    the same reasoning that makes `log-transform` the metabolomics pack's one
    *derived* default.
    """
    positive = values.abs()
    positive = positive[positive > 0]
    if len(positive) < MIN_NUMERIC_COLUMNS:
        return None
    return float(np.log10(positive).std())


def read(df: pd.DataFrame) -> Dict[str, Any]:
    """What the shape says about which way round the table is.

    A suggestion and a contradiction detector, never the answer — the same
    demotion the grain heuristics and the lens detector take.
    """
    blank = {"reading": UNDETERMINED, "ratio": None, "s_rows": None,
             "s_cols": None, "n_rows": int(len(df)), "n_numeric": 0,
             "sentence": "", "confidence": "low"}
    num = _numeric_block(df)
    if num is None:
        blank["sentence"] = (
            "There is not enough of a numeric block to say which way round "
            "this table is.")
        return blank

    s_rows = _spread(num.mean(axis=1))
    s_cols = _spread(num.mean(axis=0))
    if s_rows is None or s_cols is None or s_cols <= 0:
        blank["n_numeric"] = int(num.shape[1])
        blank["sentence"] = (
            "The numeric block is not spread enough on either axis to say "
            "which way round this table is.")
        return blank

    ratio = s_rows / s_cols
    reading = UNDETERMINED
    if ratio >= FEATURE_MAJOR_RATIO and s_rows >= MIN_ROW_SPREAD:
        reading = FEATURE_MAJOR
    elif ratio <= SAMPLE_MAJOR_RATIO:
        reading = SAMPLE_MAJOR

    if reading == FEATURE_MAJOR:
        sentence = (
            f"Across {num.shape[0]:,} rows and {num.shape[1]:,} numeric "
            f"columns, the rows differ from each other by orders of magnitude "
            f"and the columns barely differ at all. In an assay table that is "
            f"what features in rows looks like: different analytes have very "
            f"different abundances, and samples of the same kind do not.")
    elif reading == SAMPLE_MAJOR:
        sentence = (
            f"Across {num.shape[0]:,} rows and {num.shape[1]:,} numeric "
            f"columns, the columns differ from each other by orders of "
            f"magnitude and the rows barely differ. That is what one row per "
            f"sample looks like.")
    else:
        sentence = (
            f"The rows and the columns of the {num.shape[0]:,} × "
            f"{num.shape[1]:,} numeric block vary by similar amounts, which "
            f"does not say which way round this table is.")

    return {"reading": reading,
            "ratio": round(float(ratio), 3),
            "s_rows": round(float(s_rows), 3),
            "s_cols": round(float(s_cols), 3),
            "n_rows": int(num.shape[0]),
            "n_numeric": int(num.shape[1]),
            "sentence": sentence,
            # `high` never appears here, deliberately. This is a question of
            # FACT, and `_skip_is_permitted` admits `high`-confidence facts as
            # skippable — a shape reading that could auto-advance would be the
            # app transposing a table on its own authority, which is the one
            # thing this module exists to prevent.
            "confidence": "medium" if reading != UNDETERMINED else "low"}


def fires(lens: Sequence[str], reading: Dict[str, Any]) -> bool:
    """Both conditions, and both are necessary.

    The lens restricts it to the two fields whose exports come both ways round;
    the shape restricts it to the tables where the question has an answer worth
    asking for. Either alone would be a question asked of data it does not
    describe.
    """
    if not any(k in ASSAY_LENSES for k in (lens or [])):
        return False
    return (reading or {}).get("reading") == FEATURE_MAJOR


TITLE = "Which way round is this table?"
WHY = ("An assay export can come either way, and the difference is not "
       "cosmetic: if the columns are samples then every reading below is "
       "computed across the wrong axis and none of it means anything.")
CONSUMER = (
    "Nothing after this reads the table until it is settled, which is why it "
    "is asked here. The structural diagnosis, the impossibility pass, the "
    "missingness survey and the target list are all computed per column, so on "
    "a table with samples in the columns each of them is answering a question "
    "about a participant while reporting it as a fact about a measurement. "
    "Answering 'features in rows' turns the table around before any of that "
    "runs, and the methods section states that it was turned around."
)

OPTIONS = [
    {"key": ROWS_ARE_SAMPLES,
     "label": "Each row is a sample or participant",
     "note": "Records that the table is already one row per sample. Nothing is "
             "changed, and the reading above is recorded as overruled."},
    {"key": ROWS_ARE_FEATURES,
     "label": "Each row is a feature, and the columns are samples",
     "note": "Turns the table around before anything is diagnosed, so the "
             "columns become the measurements and the rows become the samples. "
             "This changes what a row is, so it can only happen before the "
             "test set is drawn."},
]


def question(reading: Dict[str, Any]) -> Dict[str, Any]:
    """The question, as the Router and the page both read it."""
    return {
        "key": "state_orientation",
        "clause": "lockbox-01",
        "seq": "1.5",
        "title": TITLE,
        "why": WHY + " " + (reading or {}).get("sentence", ""),
        "consumer": CONSUMER,
        "options": list(OPTIONS),
        "reading": dict(reading or {}),
    }


# ── turning it around ────────────────────────────────────────────────────────

def label_column(df: pd.DataFrame) -> Optional[str]:
    """The column holding the feature names, if there is one.

    A feature-major export names its rows somewhere: a first column of `mz_0001`
    or of gene symbols. Without one the row labels are positions, and the
    transposed table's columns would be `0`, `1`, `2` — legal, and useless to
    read. Both cases are handled; this reports which.

    **Near-unique rather than unique**, deliberately. Requiring exact uniqueness
    would make `transpose`'s duplicate-name refusal unreachable: a column with
    one repeated feature name would simply not be recognized as the labels, the
    names would be silently discarded, and the turned-around table would carry
    `row_0`, `row_1` where the analytes should be. A near-unique identifier
    column with a duplicate in it is exactly the case that must be REFUSED, so
    it has to be recognized first.
    """
    for name in df.columns:
        col = df[name]
        if pd.api.types.is_numeric_dtype(col):
            continue
        values = col.dropna().astype(str)
        if len(values) != len(df) or not len(df):
            continue
        if values.nunique() >= 0.9 * len(df):
            return str(name)
    return None


def transpose(df: pd.DataFrame, sample_column: str = "sample_id"
              ) -> Dict[str, Any]:
    """Turn a feature-major frame into a sample-major one.

    **Refuses rather than guesses**, in the two places a transpose can quietly
    corrupt a table:

    * duplicate feature names would become two columns with one name, and every
      consumer downstream would silently see one of them;
    * a name collision with the new sample column would do the same thing to the
      identifiers.

    Both raise. The governing rule's *refuse* branch is available here and its
    *assert something false* branch is not.
    """
    label = label_column(df)
    body = df.set_index(label) if label else df.copy()
    if label:
        body.index = body.index.astype(str)
        if body.index.duplicated().any():
            dupe = body.index[body.index.duplicated()][0]
            raise OrientationError(
                f"Two rows are both named {dupe!r}. Turning the table around "
                f"would produce two columns with one name, and every reading "
                f"after that would silently use one of them. Give the rows "
                f"distinct names first.")
    else:
        body.index = [f"row_{i}" for i in range(len(body))]

    turned = body.T
    turned.index = turned.index.astype(str)
    turned.columns = [str(c) for c in turned.columns]
    if sample_column in turned.columns:
        raise OrientationError(
            f"One of the rows is named {sample_column!r}, which is the name "
            f"the turned-around table needs for its sample identifiers. Rename "
            f"that row first.")
    turned = turned.reset_index().rename(columns={"index": sample_column})
    # Every cell arrived as `object` through the transpose. Coerce back per
    # column, never frame-wide: a column that will not coerce is a row that was
    # not a measurement, and it must stay readable rather than become NaN.
    for name in turned.columns:
        if name == sample_column:
            continue
        coerced = pd.to_numeric(turned[name], errors="coerce")
        if coerced.notna().sum() >= max(1, int(0.9 * len(turned))):
            turned[name] = coerced
    return {"df": turned.reset_index(drop=True),
            "label_column": label,
            "sample_column": sample_column,
            "n_samples": int(len(turned)),
            "n_features": int(turned.shape[1] - 1)}


def methods_sentence(answer: str, detail: Optional[Dict[str, Any]] = None) -> str:
    """The sentence the record keeps and the manuscript carries.

    Both answers get one. *"The table was already one row per sample"* is a
    claim, and §09's recorded-absence rule says a claim needs a record — without
    it, a table that was checked and a table nobody looked at read identically.
    """
    if answer == ROWS_ARE_FEATURES:
        d = detail or {}
        return (
            f"The table was supplied with features in rows and samples in "
            f"columns, and was transposed to one row per sample before any "
            f"diagnosis was run; {d.get('n_features', 0):,} measurements "
            f"across {d.get('n_samples', 0):,} samples were read.")
    return ("The table was confirmed as one row per sample and was not "
            "transposed.")


def evidence(reading: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The numbers behind the reading, for the card that shows its working."""
    r = reading or {}
    return [
        {"label": "rows", "value": r.get("n_rows")},
        {"label": "numeric columns", "value": r.get("n_numeric")},
        {"label": "spread across rows", "value": r.get("s_rows")},
        {"label": "spread across columns", "value": r.get("s_cols")},
        {"label": "ratio", "value": r.get("ratio")},
        {"label": "reads as feature-major above", "value": FEATURE_MAJOR_RATIO},
    ]
