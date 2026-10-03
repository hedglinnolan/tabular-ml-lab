"""Survey response scales: item blocks, the instruments they match, and the codes they carry
(audit IN-03, IN-16; CLINICAL_SURVEY_PACK §B1.1).

``turbotab.packs.likert_block`` (shared with the legacy app, not edited) tried five declared
scales in a fixed order and tested each value against the *declared* scale it matched first. A
balanced 6-point block therefore read as 1–5 with every 6 a code ("almost certainly 'don't know'"),
a skewed 7-point block as 1–5 with its 6s and 7s codes, a 0–5 block as 1–5 with its 0s codes and a
9-point hedonic block as 1–7 with its 8s and 9s codes; and its shape rules (at least 8 items, most
items with no answer above 60%) missed a floor-heavy PHQ-9, the 7-item GAD-7 and 0–10 ratings.
The findings stage now serves this module's reading in its place.

**The pack's rule, as written.** §B1.1: "infer min, max, step and number of points from the
observed support union across the block, not per item"; "Flag values that break the observed
contiguous run". So:

* The block's **response run** is the contiguous run of whole numbers, in the union of every
  item's values, that holds the most answers. 1–6 is one run; 1–5 with a 9 is two, and the 9 is
  not in the one that holds the answers.
* A value outside the run is a **code** when a gap separates it from the run (it always does: the
  run is maximal) and it is a conventional code (§B1.1's list) or lies at least three steps beyond
  the run's end. It is never recoded here.
* The **declared scale** is the smallest conventional scale that contains the run (1–4, 0–3, 0–4,
  1–5, 0–5, 1–6, 0–6, 1–7, 1–9, 0–10, 1–10): an unused end category is a floor or ceiling effect.

**Blocks** come from §B1.1's signals. Items sharing a name stem with a numeric suffix
(``phq_1`` … ``phq_9``) form a block of three or more whatever their answer shares, because
floor-heavy items are the normal shape of a screening instrument in a community sample. Items
without a shared stem form a block only as before: eight or more, most of them using every
category with no answer above 60%, which is what separates a response block from a block of low
gene counts. Stems that name assay features (``gene_0017``, ``mz_0001``) are never instruments.
A block whose stem, length and scale match a published instrument names it as a hypothesis
(§B1.1: "Always present the match as a hypothesis").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

#: §B1.1's conventional codes (the pack's own list, CLINICAL_SURVEY_PACK.md:676-678).
KNOWN_CODES = frozenset({7, 8, 9, 77, 88, 98, 99, 999, 9999, -1, -8, -9, -99})
#: Conventional response scales, as (lowest, highest).
SCALES: tuple[tuple[int, int], ...] = ((1, 4), (0, 3), (0, 4), (1, 5), (0, 5), (1, 6), (0, 6),
                                       (1, 7), (1, 9), (0, 10), (1, 10))
MAX_ITEM_VALUES = 15      # distinct values an item may hold, codes included
MAX_CODE_SHARE = 0.25     # of an item's answers that may be codes before it is a different variable
FAR_STEPS = 3             # an unconventional value this far beyond the run is still a code
MIN_NAMED_ITEMS = 3       # items sharing a stem, to be a block
MIN_UNNAMED_ITEMS = 8     # items without one (the legacy detector's minimum)
MAX_MODAL_SHARE = 0.60    # unnamed blocks: an item's busiest answer …
MIN_SHARE_BALANCED = 0.7  # … stays under this in this share of its items
MIN_POINTS = 3            # answers a response run must span (two are a yes/no checklist)

#: Stems that name assay features or generic positions, never a questionnaire.
_NOT_INSTRUMENTS = frozenset({
    "gene", "genes", "mz", "m", "feature", "features", "ft", "probe", "transcript", "metabolite",
    "compound", "peak", "x", "v", "var", "col", "column", "unnamed", "sample", "s", "id", "visit",
    "day", "week", "month", "year", "time", "t", "dose", "cell", "ensg", "ensmusg", "ilmn", "a",
})
_STEM = re.compile(r"^(?P<stem>.+?)[\s_.\-]?(?P<num>\d+)[a-z]?$", re.I)
_GENE_ID = re.compile(r"^(ENS[A-Z]{0,4}[GT]\d{6,}|ILMN_\d+|\d+(_[a-z]+)?_at|[NX][MR]_\d+)", re.I)

#: §B1.1's fingerprint library, the instruments whose shape is unambiguous enough to name:
#: (name, stem pattern, items, scale). A hit is a hypothesis the user confirms.
INSTRUMENTS: tuple[tuple[str, str, tuple[int, ...], tuple[int, int]], ...] = (
    ("PHQ-9", r"^phq(?:_?9)?$", (9,), (0, 3)),
    ("PHQ-8", r"^phq(?:_?8)?$", (8,), (0, 3)),
    ("GAD-7", r"^gad(?:_?7)?$", (7,), (0, 3)),
    ("HADS", r"^hads$", (14,), (0, 3)),
    ("CES-D", r"^ces_?d$", (20,), (0, 3)),
    ("Rosenberg self-esteem scale", r"^(?:rse|rses|rosenberg)$", (10,), (1, 4)),
    ("PSS", r"^pss(?:_?(?:10|14))?$", (10, 14), (0, 4)),
    ("EQ-5D-5L", r"^eq_?5d(?:_?5l)?$", (5,), (1, 5)),
)


@dataclass
class Block:
    """One block of items sharing a response scale."""
    columns: list[str]
    run: tuple[int, int]                   # the observed contiguous response run
    scale: tuple[int, int]                 # the smallest conventional scale containing it
    stem: str | None                       # the shared name stem, when the names corroborate
    codes: dict[str, list[int]] = field(default_factory=dict)  # item -> its code values
    instrument: str | None = None

    @property
    def points(self) -> int:
        return self.scale[1] - self.scale[0] + 1

    def to_dict(self) -> dict[str, Any]:
        return {"columns": list(self.columns), "scale": list(range(self.scale[0], self.scale[1] + 1)),
                "observed_support": list(range(self.run[0], self.run[1] + 1)),
                "stem": self.stem, "sentinels": {c: list(v) for c, v in self.codes.items()},
                "instrument": self.instrument}


def _values(series: pd.Series) -> pd.Series | None:
    """The item's answers as whole numbers, or None when it is not a candidate item."""
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return None
    s = series.dropna()
    if s.empty:
        return None
    x = s.to_numpy(dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.mod(x, 1) == 0):
        return None
    if s.nunique() > MAX_ITEM_VALUES or s.nunique() < 2:
        return None
    return s.astype("int64")


def _whole_non_negative(series: pd.Series) -> bool:
    x = series.dropna().to_numpy(dtype=float)
    return bool(len(x)) and bool(np.all(np.isfinite(x))) and float(x.min()) >= 0 \
        and bool(np.all(np.mod(x, 1) == 0))


def _stem(name: str) -> str | None:
    m = _STEM.match(str(name).strip())
    if not m or _GENE_ID.match(str(name)):
        return None
    stem = m.group("stem").rstrip(" _.-").lower()
    if not stem or stem in _NOT_INSTRUMENTS or not re.search(r"[a-z]", stem):
        return None
    return stem


def _runs(values: Sequence[int]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive whole numbers in ``values``."""
    ordered = sorted(set(int(v) for v in values))
    runs: list[list[int]] = []
    for v in ordered:
        if runs and v == runs[-1][1] + 1:
            runs[-1][1] = v
        else:
            runs.append([v, v])
    return [(a, b) for a, b in runs]


def _main_run(counts: dict[int, int]) -> tuple[int, int]:
    return max(_runs(counts), key=lambda r: (sum(n for v, n in counts.items() if r[0] <= v <= r[1]),
                                             r[1] - r[0]))


def declared_scale(run: tuple[int, int]) -> tuple[int, int] | None:
    """The smallest conventional scale containing ``run`` (one starting where the run starts, on a
    tie), or None when no conventional scale contains it."""
    fits = [s for s in SCALES if s[0] <= run[0] and run[1] <= s[1]]
    if not fits:
        return None
    return min(fits, key=lambda s: (s[1] - s[0], s[0] != run[0]))


def is_code(value: int, run: tuple[int, int]) -> bool:
    """§B1.1: a value outside the run, separated from it, that is a conventional code or lies
    at least ``FAR_STEPS`` beyond the run's end."""
    if run[0] <= value <= run[1]:
        return False
    distance = value - run[1] if value > run[1] else run[0] - value
    return distance >= 2 and (value in KNOWN_CODES or distance >= FAR_STEPS)


def _instrument(stem: str | None, n_items: int, scale: tuple[int, int]) -> str | None:
    if not stem:
        return None
    for name, pattern, lengths, shape in INSTRUMENTS:
        if re.match(pattern, stem) and n_items in lengths and scale == shape:
            return name
    return None


def _block(columns: list[str], items: dict[str, pd.Series], stem: str | None) -> Block | None:
    """The block these items form, the items that do not fit it dropped, or None."""
    counts: dict[int, int] = {}
    for c in columns:
        for v, n in items[c].value_counts().items():
            counts[int(v)] = counts.get(int(v), 0) + int(n)
    run = _main_run(counts)
    scale = declared_scale(run)
    if scale is None or run[1] - run[0] + 1 < MIN_POINTS:
        return None
    kept: list[str] = []
    codes: dict[str, list[int]] = {}
    for c in columns:
        s = items[c]
        outside = sorted({int(v) for v in s.unique() if not run[0] <= int(v) <= run[1]})
        if any(not is_code(v, run) for v in outside):
            continue                              # a value that is neither an answer nor a code
        inside = s[(s >= run[0]) & (s <= run[1])]
        if inside.nunique() < 2:
            continue                              # one answer for everyone is not an item
        if outside and float(s.isin(outside).mean()) > MAX_CODE_SHARE:
            continue
        kept.append(c)
        if outside:
            codes[c] = outside
    if not kept:
        return None
    if set(kept) != set(columns):
        return _block(kept, items, stem) if len(kept) >= MIN_NAMED_ITEMS else None
    return Block(columns=kept, run=run, scale=scale, stem=stem, codes=codes,
                 instrument=_instrument(stem, len(kept), scale))


def _balanced(block: Block, items: dict[str, pd.Series]) -> bool:
    """The legacy shape rule, for blocks no name corroborates: most items use every category of
    the run and no answer takes more than 60% of them."""
    points = block.run[1] - block.run[0] + 1
    balanced = 0
    for c in block.columns:
        s = items[c]
        s = s[(s >= block.run[0]) & (s <= block.run[1])]
        if s.nunique() >= points - 1 and float(s.value_counts(normalize=True).max()) <= MAX_MODAL_SHARE:
            balanced += 1
    return balanced / len(block.columns) >= MIN_SHARE_BALANCED


def blocks(df: pd.DataFrame) -> list[Block]:
    """Every response block in ``df``, largest first."""
    if df is None or df.empty:
        return []
    items: dict[str, pd.Series] = {}
    for c in df.columns:
        s = df[c]
        if isinstance(s, pd.DataFrame):
            continue
        v = _values(s)
        if v is not None:
            items[str(c)] = v
    # A count matrix wider than it is tall is an assay, not a questionnaire: low gene counts take
    # a few small values each, and gene families share stems (CASP1, CASP3, CASP8). There only a
    # fingerprinted instrument's stem gathers a block.
    integral = [c for c in df.columns if not isinstance(df[c], pd.DataFrame)
                and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
                and _whole_non_negative(df[c])]
    assay = len(integral) >= 100 and len(integral) >= len(df)
    by_stem: dict[str, list[str]] = {}
    for c in items:
        stem = _stem(c)
        if stem and (not assay or any(re.match(p, stem) for _, p, _, _ in INSTRUMENTS)):
            by_stem.setdefault(stem, []).append(c)
    found: list[Block] = []
    used: set[str] = set()
    for stem, columns in by_stem.items():
        if len(columns) < MIN_NAMED_ITEMS:
            continue
        b = _block(columns, items, stem)
        if b is not None and len(b.columns) >= MIN_NAMED_ITEMS:
            found.append(b)
            used.update(b.columns)
    # Items no stem gathers: grouped by the scale their own answers sit on, as before.
    by_scale: dict[tuple[int, int], list[str]] = {}
    items_unnamed = {} if assay else {c: s for c, s in items.items() if c not in used}
    for c, s in items_unnamed.items():
        counts = {int(v): int(n) for v, n in s.value_counts().items()}
        run = _main_run(counts)
        scale = declared_scale(run)
        if scale is not None:
            by_scale.setdefault(scale, []).append(c)
    for scale, columns in by_scale.items():
        if len(columns) < MIN_UNNAMED_ITEMS:
            continue
        b = _block(columns, items, None)
        if b is not None and len(b.columns) >= MIN_UNNAMED_ITEMS and _balanced(b, items):
            found.append(b)
    found.sort(key=lambda b: (-len(b.columns), b.columns[0]))
    return found


def block(df: pd.DataFrame) -> dict[str, Any] | None:
    """The largest block, in ``packs.likert_block``'s shape (``scale``, ``columns``,
    ``observed_support``, ``sentinels``), or None."""
    found = blocks(df)
    return found[0].to_dict() if found else None


# ── the findings, in the pack's shape and under the pack's ids ───────────────


def _scale_text(b: Block) -> str:
    return f"{b.scale[0]}–{b.scale[1]}"


def ordinal_finding(found: list[Block]) -> dict[str, Any] | None:
    from turbotab.packs import ORDINAL_DECLARED_EVIDENCE, SURVEY, _finding

    if not found:
        return None
    lead = found[0]
    named = f" It matches the {lead.instrument}'s shape ({len(lead.columns)} items, " \
            f"{_scale_text(lead)}); confirm it is that instrument, unmodified." if lead.instrument else ""
    others = (f" {len(found) - 1} other block{'s' if len(found) > 2 else ''}: "
              + "; ".join(f"`{b.columns[0]}` … `{b.columns[-1]}` ({len(b.columns)} items, "
                          f"{_scale_text(b)})" for b in found[1:4]) + ".") if len(found) > 1 else ""
    return _finding(
        "pack::survey::ordinal_declared", "info",
        f"{len(lead.columns):,} columns share one {lead.points}-point response scale",
        (f"Every answer in them is a whole number from {lead.run[0]} to {lead.run[1]}, read from "
         f"the union of the block's answers; the block runs `{lead.columns[0]}` … "
         f"`{lead.columns[-1]}`." + named + others),
        ("The order comes from the instrument, not from the data — which makes the encoding "
         "row-local: the number for a row depends on that row's own answer and on nothing else, so "
         "it is applied now rather than fitted inside the training folds."),
        confidence="high", pack=SURVEY, marker="derived", evidence=ORDINAL_DECLARED_EVIDENCE,
        columns=lead.columns[:10],
        params={"scale": list(range(lead.scale[0], lead.scale[1] + 1)), "columns": lead.columns,
                "observed_support": list(range(lead.run[0], lead.run[1] + 1)),
                "encoding": "declared", "instrument": lead.instrument,
                "named_by": "stem" if lead.stem else "values",
                "blocks": [b.to_dict() for b in found]})


def sentinel_finding(df: pd.DataFrame, found: list[Block]) -> dict[str, Any] | None:
    """§B1.1's sentinel check over every block: detected, reported, recoded only on request."""
    from turbotab.packs import DISPUTED, SURVEY, Claim, Evidence, _finding
    from turbotab.survey import SENTINEL_EVIDENCE

    readings: list[dict[str, Any]] = []
    for b in found:
        for column, values in sorted(b.codes.items()):
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            flagged = series.isin(values)
            kept = series[~flagged]
            if kept.empty:
                continue
            readings.append({
                "item": column, "sentinel_values": [int(v) for v in values],
                "n": int(len(series)), "n_sentinel": int(flagged.sum()),
                "share": round(float(flagged.mean()), 4),
                "mean_as_responses": round(float(series.mean()), 3),
                "mean_excluding": round(float(kept.mean()), 3),
                "mean_shift": round(float(series.mean() - kept.mean()), 3),
                "matches_known_sentinel": [int(v) for v in values if v in KNOWN_CODES],
                "block_run": [b.run[0], b.run[1]]})
    if not readings:
        return None
    readings.sort(key=lambda r: (-abs(r["mean_shift"]), r["item"]))
    lead = readings[0]
    run = f"{lead['block_run'][0]}–{lead['block_run'][1]}"
    unknown = [r for r in readings if len(r["matches_known_sentinel"]) < len(r["sentinel_values"])]
    said = ", ".join(str(v) for v in lead["sentinel_values"])
    detail = (
        f"`{lead['item']}` contains {said}, while the answers across its block, read as the union "
        f"of every item's, run {run} without a gap. {said} lie{'s' if len(lead['sentinel_values']) == 1 else ''} "
        f"beyond that run with unused values between, which is how a 'don't know' / 'refused' / "
        f"'not applicable' code looks; only the codebook can say. **TurboTab has not recoded "
        f"them.** {lead['n_sentinel']:,} of `{lead['item']}`'s {lead['n']:,} answers are affected, "
        f"and treating them as answers moves this item's mean by {lead['mean_shift']:+.3f} — from "
        f"{lead['mean_excluding']:.3f} to {lead['mean_as_responses']:.3f}.")
    if len(readings) > 1:
        detail += (f" {len(readings) - 1} other item{'' if len(readings) == 2 else 's'} carry the "
                   "same shape: " + "; ".join(
                       f"`{r['item']}` ({', '.join(str(v) for v in r['sentinel_values'])}, "
                       f"{r['share']:.1%}, mean {r['mean_shift']:+.3f})" for r in readings[1:6])
                   + ("" if len(readings) <= 6 else f" and {len(readings) - 6} more") + ".")
    if unknown:
        detail += (f" `{unknown[0]['item']}`'s value is not a conventional code; it is read as one "
                   f"because it lies at least {FAR_STEPS} steps beyond the answers.")
    all_known = not unknown
    return _finding(
        "pack::survey::sentinel_codes", "critical" if all_known else "warning",
        (f"{len(readings)} item{'' if len(readings) == 1 else 's'} carry values outside the {run} "
         f"answers"),
        detail,
        ("A code treated as an answer moves the item's mean, the scale score, every correlation "
         "the item enters and every coefficient downstream. Recoding is essential if these are "
         "codes, and the numbers cannot prove they are: your codebook decides; the shift above is "
         "what the decision costs."),
        confidence="high" if all_known else "medium", pack=SURVEY, marker="offered",
        evidence=SENTINEL_EVIDENCE,
        claims=(
            Claim(key="must_recode",
                  statement=("Values that are sentinel codes must be recoded to missing before "
                             "anything is computed from them."), evidence=SENTINEL_EVIDENCE),
            Claim(key="never_auto_recode",
                  statement=("The app never recodes them on its own, because some legitimate "
                             "scales do run 0-9 and nothing in the numbers separates the two."),
                  evidence=SENTINEL_EVIDENCE),
            Claim(key="dont_know_is_not_missing",
                  statement=("'Don't know' is not automatically the same as missing. On attitude "
                             "items it is often a substantive response."),
                  evidence=Evidence(
                      status=DISPUTED,
                      source="research/CLINICAL_SURVEY_PACK.md#B1.1 Detecting Likert blocks",
                      both_sides=(
                          "Recoding a refusal to missing is uncontroversial. Recoding a 'don't "
                          "know' is not: dropping it can bias the sample toward people with formed "
                          "opinions, and the survey-methodology literature has not settled whether "
                          "it is a non-response or an answer."))),
        ),
        columns=[r["item"] for r in readings],
        params={"items": readings, "observed_support": list(range(lead["block_run"][0],
                                                                   lead["block_run"][1] + 1)),
                "declared_scale": next(list(range(b.scale[0], b.scale[1] + 1)) for b in found
                                       if lead["item"] in b.columns),
                "block_size": next(len(b.columns) for b in found if lead["item"] in b.columns),
                "rule": "outside_the_observed_run_with_a_gap",
                "known_sentinel_codes": sorted(KNOWN_CODES),
                "hard_stop": "never_auto_recode"},
        fix_label="", fix_kind="none")


def findings(df: pd.DataFrame) -> list[dict[str, Any]]:
    """The survey pack's two block findings, from this module's reading."""
    found = blocks(df)
    out = [f for f in (ordinal_finding(found), sentinel_finding(df, found)) if f]
    return out


__all__ = ["Block", "INSTRUMENTS", "KNOWN_CODES", "SCALES", "block", "blocks", "declared_scale",
           "findings", "is_code", "ordinal_finding", "sentinel_finding"]
