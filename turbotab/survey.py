"""The survey pack's §B1 content — sentinel codes, and the reverse-coding audit.

`research/CLINICAL_SURVEY_PACK.md` §B1.1 and §B1.2. Two things, deliberately of
different shape from L41's clinical batch: that one was column typing and parse
detection, this is **block structure and a recorded decision reaching a
consumer.**

## §B1.1 — sentinel codes, which the research calls the highest-yield check here

> *"In a 1–5 item, a `9` is not 'extremely agree,' it is 'don't know' or
> 'refused.'"*

**The rule is that the value breaks the observed contiguous run**, and the
support is inferred from the **union across the block, never per item** —
§B1.1's own instruction, because a rarely-endorsed extreme category may be
absent from a single item. Read per item, a 1–5 instrument where nobody picked 5
on `q14` has a 1–4 item in it, and a legitimate `5` elsewhere would then look
like the sentinel.

`KNOWN_SENTINELS` corroborates and never decides. A codebook may use anything,
and a `6` in a 1–5 block breaks the run and is flagged although no list names
it.

**It is a hard stop and it says so.** `DOMAIN_SCIENCE.md` §01.2: it is SETTLED
that sentinels must be recoded and forbidden to auto-recode, because **some
legitimate scales do run 0–9**. High-confidence detection, irreversible-if-wrong
action, and no signal in the data that resolves the ambiguity.

And the shift is reported, because that is what makes it actionable rather than
a warning: *a 9 treated as a response would move this item's mean by X.* The
research writes that sentence with an `[X]` in it; this computes the X.

## §B1.2 — the reverse-coding audit, `GUIDED-136`

**The app already asks.** `api.py` dispatches `set_reverse_coding`, `packs.py`
carries the `reverse_coding` prior as the one deliberate exception to
`DOMAIN_PACKS.md`'s guard #1, and the question renders. **Nothing scored it** —
`AGENT_ONBOARD.md` §07's trap #1, a recorded decision with no consumer, on the
one question this pack was allowed to add.

`audit()` is §B1.2's table: item | text | item–rest *r* (raw) | reversal
declared? | item–rest *r* after reversal | status. **Re-rendered after every
declared change**, which is the clause that makes it an audit rather than a
report.

### The central constraint, and it is the whole point

> *"TurboTab will not infer reverse-coding from correlations, and neither should
> you."* **[SETTLED — a real hard limit.]**

A negative item–rest correlation has four incompatible explanations — the item
needs reversing, it was **already** reversed upstream, it is negatively worded
and loads on a **method factor** rather than the construct, or it does not
belong to the scale — and **correlations cannot distinguish them.** So this
computes and reports; it never proposes a reversal, and the status vocabulary
below has no `should_be_reversed` in it.

After a declared reversal it **re-runs** and warns where an item is *still*
negative: either it was already reversed in the source, or it does not belong.

### Pearson, said out loud, with the direction of the bias

§B5.4 is SETTLED that **polychoric** correlations are the appropriate choice for
ordinal items, and nothing in this repository computes one — `GUIDED-127`, open
and deliberately unbuilt. So this uses Pearson, **says Pearson**, and carries
the consequence: Pearson on ordinal data is **attenuated**, so every correlation
here is nearer zero than the polychoric one would be. That matters in exactly
one direction and it is worth being specific about — an item flagged as weak may
be adequate, and an item that clears the threshold here would clear it there too.

### 0.30 is CONVENTION and is never a verdict

Nunnally & Bernstein's conventional minimum. `DOMAIN_SCIENCE.md` §01.2's last
hard stop is *never stamp PASS/FAIL on a threshold*, so the status vocabulary
describes the reading and never grades it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from turbotab.packs import (CONVENTION_STATUS, DISPUTED, KNOWN_SENTINELS,
                            SETTLED, Claim, Evidence, _finding, likert_block)

SURVEY = "survey"

SENTINEL_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/CLINICAL_SURVEY_PACK.md#B1.1 Detecting Likert blocks")

REVERSE_CODING_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/CLINICAL_SURVEY_PACK.md#B1.2 ★ Reverse-coded items — the hard constraint")

#: Nunnally & Bernstein's conventional minimum for a corrected item–rest
#: correlation. **CONVENTION, never a law**, and never stamped PASS/FAIL — see
#: `STATUSES` for how it is rendered instead.
ITEM_REST_CONVENTION = 0.30


# ═════════════════════════════════════════════════════════════════════════════
# §B1.1 · sentinel codes in a bounded block
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SentinelReading:
    """One item's out-of-run values, and what treating them as responses costs."""
    column: str
    values: Tuple[int, ...]
    n: int
    n_sentinel: int
    mean_with: float
    mean_without: float
    known: Tuple[int, ...]

    @property
    def share(self) -> float:
        return self.n_sentinel / self.n if self.n else 0.0

    @property
    def shift(self) -> float:
        """**The number the research leaves as `[X]`.**

        > *"a 9 treated as a response would shift this item's mean by [X] and
        > propagate into every scale score and every model."*

        Signed, because the direction is information: a high sentinel inflates
        and a `-9` deflates, and an absolute value would lose which.
        """
        return self.mean_with - self.mean_without

    def to_dict(self) -> Dict[str, Any]:
        return {"item": self.column, "sentinel_values": list(self.values),
                "n": self.n, "n_sentinel": self.n_sentinel,
                "share": round(self.share, 4),
                "mean_as_responses": round(self.mean_with, 3),
                "mean_excluding": round(self.mean_without, 3),
                "mean_shift": round(self.shift, 3),
                "matches_known_sentinel": list(self.known)}


def read_sentinels(df: pd.DataFrame) -> Tuple[Optional[Dict[str, Any]],
                                              List[SentinelReading]]:
    """`(block, readings)` — the block detector's own output, read for sentinels.

    **Not a second block detector.** `packs.likert_block` finds the block and
    carries the out-of-run values on it; this reads them. Two block detectors
    would be `FEATURE_PARITY.md`'s `theory_anchors`/`theory_demos` pair with the
    drift able to change what an instrument *is*.
    """
    block = likert_block(df)
    if block is None:
        return None, []
    readings: List[SentinelReading] = []
    for column, values in sorted((block.get("sentinels") or {}).items()):
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        flagged = series.isin(list(values))
        kept = series[~flagged]
        if kept.empty:
            continue
        readings.append(SentinelReading(
            column=column, values=tuple(int(v) for v in values),
            n=int(len(series)), n_sentinel=int(flagged.sum()),
            mean_with=float(series.mean()), mean_without=float(kept.mean()),
            known=tuple(v for v in values if v in KNOWN_SENTINELS)))
    readings.sort(key=lambda r: (-abs(r.shift), r.column))
    return block, readings


def sentinel_codes_finding(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """§B1.1's highest-yield check. **Detected, reported, never recoded.**"""
    block, readings = read_sentinels(df)
    if not readings:
        return None

    support = block["observed_support"]
    lead = readings[0]
    run = f"{min(support)}–{max(support)}"
    unknown = [r for r in readings if not r.known]

    detail = (
        f"`{lead.column}` contains "
        f"{', '.join(str(v) for v in lead.values)} while the block's response "
        f"support, taken as the union across all {len(block['columns'])} items, "
        f"runs {run}. Those values break the run, so they are almost certainly "
        f"'don't know' / 'refused' / 'not applicable' sentinel codes rather "
        f"than responses. **TurboTab has not recoded them.** "
        f"{lead.n_sentinel:,} of `{lead.column}`'s {lead.n:,} answers are "
        f"affected, and treating them as responses moves this item's mean by "
        f"{lead.shift:+.3f} — from {lead.mean_without:.3f} to "
        f"{lead.mean_with:.3f} — which propagates into every scale score and "
        f"every model.")
    if len(readings) > 1:
        detail += (
            f" {len(readings) - 1} other item"
            f"{'' if len(readings) == 2 else 's'} carry the same shape: "
            + "; ".join(
                f"`{r.column}` ({', '.join(str(v) for v in r.values)}, "
                f"{r.share:.1%}, mean {r.shift:+.3f})" for r in readings[1:])
            + ".")
    if unknown:
        detail += (
            f" `{unknown[0].column}`'s "
            f"{', '.join(str(v) for v in unknown[0].values)} matches no "
            f"conventional missing code, which changes nothing about the "
            f"reading — the rule is that the value breaks the run, and a "
            f"codebook may use whatever it likes.")

    return _finding(
        "pack::survey::sentinel_codes", "critical",
        (f"{len(readings)} item{'' if len(readings) == 1 else 's'} carry values "
         f"outside the {run} response scale"),
        detail,
        ("A sentinel treated as a response is the worst kind of wrong number: "
         "it is inside the column's dtype, inside every plausibility check the "
         "app has, and it moves the item's mean, the scale score, every "
         "correlation the item enters and every coefficient downstream. "
         "Recoding is essential IF these are sentinels — and the app cannot "
         "know that they are, because some legitimate scales do run 0–9. "
         "Your codebook decides; the shift above is what the decision costs."),
        confidence="high", pack=SURVEY, marker="offered",
        evidence=SENTINEL_EVIDENCE,
        claims=(
            Claim(key="must_recode",
                  statement=("Values that are sentinel codes must be recoded to "
                             "missing before anything is computed from them."),
                  evidence=SENTINEL_EVIDENCE),
            Claim(key="never_auto_recode",
                  statement=("The app never recodes them, because some "
                             "legitimate scales do run 0-9 and nothing in the "
                             "numbers separates the two cases."),
                  evidence=SENTINEL_EVIDENCE),
            Claim(key="dont_know_is_not_missing",
                  statement=("'Don't know' is not automatically the same as "
                             "missing. On attitude items it is often a "
                             "substantive response."),
                  evidence=Evidence(
                      status=DISPUTED,
                      source=("research/CLINICAL_SURVEY_PACK.md#B1.1 Detecting "
                              "Likert blocks"),
                      both_sides=(
                          "Recoding a refusal to missing is uncontroversial. "
                          "Recoding a 'don't know' is not: dropping it can bias "
                          "the sample toward people with formed opinions, and "
                          "the survey-methodology literature has not settled "
                          "whether it is a non-response or an answer."))),
        ),
        columns=[r.column for r in readings],
        params={
            "items": [r.to_dict() for r in readings],
            # §B1.1's block-detection summary row, so a consumer does not have
            # to re-derive the support to read the table above.
            "observed_support": support,
            "declared_scale": block["scale"],
            "block_size": len(block["columns"]),
            "known_sentinel_codes": list(KNOWN_SENTINELS),
            "hard_stop": "never_auto_recode",
            "hard_stop_because": (
                "Some legitimate scales do run 0-9, and no signature in the "
                "data separates a scale point from a missing code. The app "
                "detects and declares; the codebook decides."),
        },
        # NO REPAIR. `DOMAIN_SCIENCE.md` §01.2's third hard stop.
        fix_label="", fix_kind="none")


# ═════════════════════════════════════════════════════════════════════════════
# §B1.2 · the reverse-coding audit
# ═════════════════════════════════════════════════════════════════════════════

#: The status vocabulary, and what is **not** in it is the design.
#:
#: There is no `should_be_reversed` and no `pass` / `fail`. Every status
#: describes what was observed; none proposes a reversal, because a negative
#: item–rest correlation has four incompatible causes and correlations cannot
#: distinguish them. And none grades against 0.30, because
#: `DOMAIN_SCIENCE.md` §01.2's last hard stop is *never stamp PASS/FAIL on a
#: threshold* — α≥0.70, SMD<0.10, CFI≥0.95 and this one are all conventions and
#: several are actively contested.
STATUSES = {
    "consistent": "Correlates positively with the rest of the scale.",
    "below_convention": (
        "Correlates positively but below 0.30, which is Nunnally & Bernstein's "
        "conventional minimum rather than a law. Attenuated by Pearson on "
        "ordinal data, so the polychoric value would be higher."),
    "negative_undeclared": (
        "Correlates negatively and no reversal is declared. Four things produce "
        "this and the numbers cannot separate them: the item needs reversing, "
        "it was already reversed in the source, it is negatively worded and "
        "loads on a wording factor rather than the construct, or it does not "
        "belong to this scale. Your codebook decides."),
    "resolved_by_reversal": (
        "Was negative and is positive after the declared reversal, which is "
        "what a correctly identified reverse-worded item looks like."),
    "negative_after_reversal": (
        "Still correlates negatively after the declared reversal. Either it was "
        "already reverse-scored in the source data — in which case reversing it "
        "again has corrupted it — or it does not belong to this scale."),
    "weakened_by_reversal": (
        "Was positive before the declared reversal and is weaker or negative "
        "after it, which is what reversing an item that did not need it looks "
        "like."),
}


def _corrected_item_rest(frame: pd.DataFrame, column: str) -> Optional[float]:
    """Pearson correlation of one item against the sum of the OTHERS.

    **Corrected** — the item is excluded from the rest score. An uncorrected
    item–total correlation includes the item in its own total, which inflates it
    by construction and by more the shorter the scale.
    """
    others = [c for c in frame.columns if c != column]
    if not others:
        return None
    rest = frame[others].sum(axis=1, min_count=len(others))
    both = pd.concat([frame[column], rest], axis=1).dropna()
    if len(both) < 3 or both.iloc[:, 0].nunique() < 2 or both.iloc[:, 1].nunique() < 2:
        return None
    return float(both.iloc[:, 0].corr(both.iloc[:, 1]))


def _reversed(series: pd.Series, support: Sequence[int]) -> pd.Series:
    """`(min + max) − x`, on the block's support rather than the item's own.

    The support is the union across the block for the reason §B1.1 gives: an
    item where nobody picked 5 has an observed max of 4, and reversing it
    against its own max would map its 1s to 4 while every other item's map to 5.
    That silently rescales one item inside a scale being summed.
    """
    return (min(support) + max(support)) - series


#: **The bias direction, stated.** §B5.4 is SETTLED that polychoric correlations
#: are the appropriate choice for ordinal items; nothing here computes one
#: (`GUIDED-127`) and none is approximated. Pearson on ordinal data is
#: *attenuated* — biased toward zero — so this disclosure is not symmetric
#: hedging: it says which way every number in the table is wrong.
PEARSON_DISCLOSURE = (
    "Correlations are Pearson. The appropriate choice for ordinal items is "
    "polychoric, which this app does not compute and does not approximate. "
    "Pearson on ordinal data is attenuated — biased toward zero — so every "
    "correlation below is nearer zero than the polychoric one would be. An "
    "item that looks weak here may be adequate; an item that clears 0.30 here "
    "would clear it there too."
)


@dataclass(frozen=True)
class AuditRow:
    item: str
    text: str
    r_raw: Optional[float]
    declared: bool
    r_after: Optional[float]
    status: str
    n_sentinel_excluded: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"item": self.item, "text": self.text,
                "item_rest_r_raw": (None if self.r_raw is None
                                    else round(self.r_raw, 3)),
                "reversal_declared": self.declared,
                "item_rest_r_after_reversal": (None if self.r_after is None
                                               else round(self.r_after, 3)),
                "status": self.status, "because": STATUSES[self.status],
                "sentinels_excluded": self.n_sentinel_excluded}


def _truncate(text: str, width: int = 60) -> str:
    text = str(text or "").strip()
    return text if len(text) <= width else text[:width - 1] + "…"


def audit(df: pd.DataFrame, declared: Sequence[str] = (),
          item_text: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """§B1.2's reverse-coding audit table, computed against the current record.

    `declared` is the recorded answer to `set_reverse_coding` — the codebook's,
    never an inference. Passing it in rather than reading a project is what
    makes this **re-render after every declared change**: the caller hands over
    the declaration that is true now, and the table is a function of it.

    Returns `None` where there is no block to audit, which is a refusal to
    compute rather than an empty table.
    """
    block, sentinel_readings = read_sentinels(df)
    if block is None:
        return None
    columns = list(block["columns"])
    support = block["observed_support"]
    text = dict(item_text or {})

    # SENTINELS COME OUT BEFORE ANY CORRELATION IS COMPUTED, and this is not
    # optional. A `9` in a 1–5 item read as a response is a 4-point outlier
    # dropped into a correlation; on a reverse-worded item it pushes the
    # correlation the wrong way twice. An audit computed over uncoded sentinels
    # would produce a table of numbers about the codebook rather than about the
    # instrument.
    excluded: Dict[str, int] = {}
    clean = pd.DataFrame(index=df.index)
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        bad = (block.get("sentinels") or {}).get(column) or []
        if bad:
            hit = series.isin(list(bad))
            excluded[column] = int(hit.sum())
            series = series.mask(hit)
        clean[column] = series

    declared_set = {c for c in declared if c in columns}
    flipped = clean.copy()
    for column in declared_set:
        flipped[column] = _reversed(clean[column], support)

    rows: List[AuditRow] = []
    for column in columns:
        raw = _corrected_item_rest(clean, column)
        after = _corrected_item_rest(flipped, column)
        is_declared = column in declared_set
        if raw is None:
            status = "below_convention"
        elif not is_declared:
            if raw < 0:
                status = "negative_undeclared"
            elif raw < ITEM_REST_CONVENTION:
                status = "below_convention"
            else:
                status = "consistent"
        elif after is not None and after < 0:
            status = "negative_after_reversal"
        elif raw is not None and raw >= 0 and (after is None or after <= raw):
            status = "weakened_by_reversal"
        else:
            status = "resolved_by_reversal"
        rows.append(AuditRow(
            item=column, text=_truncate(text.get(column, "")),
            r_raw=raw, declared=is_declared, r_after=after, status=status,
            n_sentinel_excluded=excluded.get(column, 0)))

    warnings = [r.to_dict() for r in rows
                if r.status in ("negative_after_reversal", "weakened_by_reversal")]
    return {
        "available": True,
        "n_items": len(rows),
        "observed_support": support,
        "declared_reversed": sorted(declared_set),
        "n_declared": len(declared_set),
        "rows": [r.to_dict() for r in rows],
        # THE RE-RUN'S OWN RESULT, lifted out. §B1.2's hard constraint is that
        # the check runs AGAIN after a declared reversal, and burying its
        # verdict in a 40-row table would make it a report.
        "warnings_after_reversal": warnings,
        "convention": ITEM_REST_CONVENTION,
        "convention_is": (
            "Nunnally & Bernstein's conventional minimum, not a law. Nothing "
            "here is stamped PASS or FAIL against it."),
        "correlation_method": "pearson",
        "correlation_disclosure": PEARSON_DISCLOSURE,
        "will_not_infer": (
            "TurboTab will not infer reverse-coding from these correlations. A "
            "negative item-rest correlation has four incompatible causes - the "
            "item needs reversing, it was already reversed upstream, it loads "
            "on a wording factor rather than the construct, or it does not "
            "belong to the scale - and no correlational signature separates "
            "them. Reverse-coding comes from the instrument's published scoring "
            "key or from your codebook."),
        "sentinels_excluded": sum(excluded.values()),
        "sentinel_items": sorted(excluded),
        "evidence_status": SETTLED,
        "source": ("research/CLINICAL_SURVEY_PACK.md#B1.2 ★ Reverse-coded "
                   "items — the hard constraint"),
    }


def unavailable_because(df: pd.DataFrame) -> str:
    """Why there is no audit, in the words of the table rather than of the check."""
    if likert_block(df) is None:
        return ("No block of items sharing one response scale was found in this "
                "table, so there is no scale to audit. Reverse-coding is a "
                "property of an instrument, and this file does not appear to "
                "carry one.")
    return ""                                              # pragma: no cover
