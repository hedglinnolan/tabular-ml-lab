"""Which way round an assay table is: the names first, then a scale-aware shape (audit IN-11).

The oriented stage read orientation from shape alone, as the spread of row means over the spread
of column means on log10 |mean|, with a second gate (row spread ≥ 0.4) that had no stated basis.
On a log2 GEO matrix every mean is already a logarithm, so both spreads collapse and the reading
was "undetermined", with the false sentence that rows and columns "vary by similar amounts"; an
MZmine export's m/z and retention-time columns sat inside the numeric block and pulled the
statistic to 2. And the question was asked only for a feature-major reading. METABOLOMICS_PACK
§01 ranks the cues the other way round, "Never rely on shape alone":

1. **Header tokens**: feature-description columns (``m/z``, ``RT``, ``row retention time``,
   MZmine's ``row …``) mean features in rows.
2. **Feature-name grammar**: identifiers (Ensembl, Affymetrix ``_at``, ``ILMN_``, ``mz_0001``,
   ``M123T456``, lipid shorthand) on the row labels mean features in rows; on the column labels,
   samples in rows. Sample-name grammar (``GSM…``, ``QC_01``, ``Sample_003``) on the column labels
   means features in rows.
3. **Shape**, scale-aware: on a block already on a log scale (its largest value at most 40, or
   negative values), the spreads are of the means themselves, not of their logarithms.

The question is asked under an assay lens whenever the reading is not "samples in rows" (the
interview's gate), and a reading never claims high confidence: the user turns the table, never
the app.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

FEATURE_MAJOR = "feature_major"
SAMPLE_MAJOR = "sample_major"
UNDETERMINED = "undetermined"
LOG_SCALE_MAX = 40.0     # §01: "a max below ~40 with a positive min and low dynamic range … probably
                         # already log-transformed"
MIN_SHARE = 0.5          # of labels that must carry a grammar
LOG_ROW_SPREAD = 0.5     # log units the row means must spread, on a log-scale block

_SAMPLE_GRAMMAR = re.compile(r"^(?:GSM\d+|(?:QC|PQC|Pool(?:ed)?|Blank|Sample|Samp|S)[_\s.-]?\d+"
                             r"|[A-H](?:0?[1-9]|1[0-2]))$", re.I)


def _feature_name(label: str) -> bool:
    from turbotab import packs
    from turbotab.core.detectors.lenses import _FEATURE_GRAMMAR

    label = str(label).strip()
    return bool(_FEATURE_GRAMMAR.match(label)) or bool(packs.id_vocabulary([label])["matched"])


def _share(labels: Sequence[str], test: Any) -> float:
    labels = [str(x) for x in labels if str(x).strip()]
    return sum(1 for x in labels if test(x)) / len(labels) if labels else 0.0


def _names(labels: Sequence[str], limit: int = 2) -> str:
    return ", ".join(f"`{x}`" for x in list(labels)[:limit])


def cue(columns: Sequence[str], numeric: Sequence[str], label: str | None,
        label_values: Sequence[str] | None) -> dict[str, Any] | None:
    """The reading the names give, with its evidence sentence, or None when they say nothing."""
    from turbotab.core.stages.working import is_feature_annotation

    annotations = [c for c in columns if is_feature_annotation(c) and c != label]
    descriptive = [c for c in annotations
                   if re.search(r"m\s?/?\s?z\b|\bmass\b|\brt\b|retention", c, re.I)]
    if descriptive:
        return {"reading": FEATURE_MAJOR, "cue": "header_tokens",
                "sentence": (f"The table has columns that describe features ({_names(descriptive, 3)}), "
                             f"which an export with one feature per row carries.")}
    if label and label_values and _share(label_values, _feature_name) >= MIN_SHARE:
        shown = [v for v in label_values if _feature_name(v)][:2]
        return {"reading": FEATURE_MAJOR, "cue": "row_label_grammar",
                "sentence": (f"`{label}` names each row like a feature ({_names(shown)}), so each "
                             f"row reads as a feature and each column as a sample.")}
    samples = [c for c in numeric if c != label]
    if samples and _share(samples, lambda x: bool(_SAMPLE_GRAMMAR.match(x.strip()))) >= MIN_SHARE:
        return {"reading": FEATURE_MAJOR, "cue": "column_sample_grammar",
                "sentence": (f"The numeric columns are named like samples ({_names(samples)}), so "
                             f"each column reads as a sample and each row as a feature.")}
    if samples and _share(samples, _feature_name) >= MIN_SHARE:
        return {"reading": SAMPLE_MAJOR, "cue": "column_feature_grammar",
                "sentence": (f"The numeric columns are named like features ({_names(samples)}), so "
                             f"each row reads as a sample.")}
    return None


def shape(row_means: Any, col_means: Any, block_min: float, block_max: float,
          n_rows: int, n_numeric: int) -> dict[str, Any] | None:
    """The scale-aware shape reading, or None when the spreads cannot be taken."""
    import numpy as np

    rows = np.asarray(row_means, dtype=float)
    cols = np.asarray(col_means, dtype=float)
    rows, cols = rows[np.isfinite(rows)], cols[np.isfinite(cols)]
    if len(rows) < 6 or len(cols) < 6:
        return None
    logged = block_max <= LOG_SCALE_MAX or block_min < 0
    if logged:
        s_rows, s_cols = float(np.std(rows, ddof=1)), float(np.std(cols, ddof=1))
        floor = LOG_ROW_SPREAD
    else:
        pr, pc = np.abs(rows[rows != 0]), np.abs(cols[cols != 0])
        if len(pr) < 6 or len(pc) < 6:
            return None
        s_rows, s_cols = float(np.log10(pr).std()), float(np.log10(pc).std())
        floor = 0.4
    if s_cols <= 0:
        return None
    ratio = s_rows / s_cols
    reading = UNDETERMINED
    if ratio >= 4.0 and s_rows >= floor:
        reading = FEATURE_MAJOR
    elif ratio <= 0.25:
        reading = SAMPLE_MAJOR
    scale = "on the values themselves, which are already on a log scale" if logged else \
        "on a log scale"
    if reading == FEATURE_MAJOR:
        sentence = (f"Across {n_rows:,} rows and {n_numeric:,} numeric columns, measured {scale}, "
                    f"the row means spread {ratio:.0f} times as widely as the column means: "
                    f"different features have different abundances, and samples of one kind do not.")
    elif reading == SAMPLE_MAJOR:
        sentence = (f"Across {n_rows:,} rows and {n_numeric:,} numeric columns, measured {scale}, "
                    f"the column means spread {1 / ratio:.0f} times as widely as the row means, "
                    f"which is what one row per sample looks like.")
    else:
        sentence = (f"Measured {scale}, the row means spread {ratio:.2g} times as widely as the "
                    f"column means: the shape does not say which way round this table is.")
    return {"reading": reading, "ratio": round(ratio, 3), "s_rows": round(s_rows, 3),
            "s_cols": round(s_cols, 3), "already_logged": bool(logged), "sentence": sentence}


def read_frame(df: Any) -> dict[str, Any]:
    """The same reading over a pandas frame (the lens preview's small table): names, then shape."""
    import numpy as np
    import pandas as pd

    from turbotab.core.stages.working import is_feature_annotation

    columns = [str(c) for c in df.columns]
    numeric = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
               and not pd.api.types.is_bool_dtype(df[c])]
    n = len(df)
    label = next((str(c) for c in df.columns if str(c) not in numeric and n
                  and df[c].notna().all() and df[c].nunique() >= 0.9 * n), None)
    values = [str(v) for v in df[label].head(2000)] if label else []
    found = cue(columns, numeric, label, values)
    if found is not None:
        return found
    block = [c for c in numeric if not is_feature_annotation(c) and df[c].notna().mean() > 0.5]
    if len(block) < 6 or n < 6:
        return {"reading": UNDETERMINED, "sentence": "There is not enough of a numeric block to "
                                                      "say which way round this table is."}
    x = df[block].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        read = shape(np.nanmean(x, axis=1), np.nanmean(x, axis=0), float(np.nanmin(x)),
                     float(np.nanmax(x)), n, len(block))
    return read or {"reading": UNDETERMINED, "sentence": "The numeric block is not spread enough "
                                                         "to say which way round this table is."}


def asks(lens: Sequence[str], reading: dict[str, Any]) -> bool:
    """Whether the orientation question is asked: an assay lens, and no "one row per sample"."""
    return any(k in ("metabolomics", "genomics") for k in (lens or [])) and \
        (reading or {}).get("reading") != SAMPLE_MAJOR


__all__ = ["FEATURE_MAJOR", "SAMPLE_MAJOR", "UNDETERMINED", "asks", "cue", "read_frame", "shape"]
