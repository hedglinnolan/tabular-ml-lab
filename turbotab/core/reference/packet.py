"""One lens's expert review packet (V2 definition of done §4: "one domain methodologist per lens,
from a per-domain review packet: the methods offered, how they chain, the defaults, and the exact
sentences the app writes").

    python -m turbotab.core.reference.packet --lens dietary            # runs the lens's journeys
    python -m turbotab.core.reference.packet --lens dietary --reuse    # from the stored captures
    python -m turbotab.core.reference.packet --all --reuse --check     # exit 1 when stale

writes ``docs/turbotab-next/review-packets/<lens>.md``:

1. **the methods the lens offers**, from the registry, never shortened: its own method contracts
   and model families in full, every other family on the shelf, every shared method, every method
   another lens reviews in full, and the methods it relies on that have no contract yet
   (``catalog.GAPS``), with the labels kept outside the registry (``custom_sound``) where they
   exist;
2. **how they chain**: a Mermaid diagram of every relation the lens's own methods take part in
   (from them, or toward them from any contract), and the relations in a table with the sentence
   the app states when each fires;
3. **the defaults by purpose**, each with its reason: the option each method offers first for
   prediction and for inference, its rung, and the sound label that justifies it; where the app
   ranks by the data, the first option under each condition, computed by the app's own ranking
   code (``defaults``);
4. **the exact methods sections the app writes** on the lens's reference journeys
   (``journeys``), quoted verbatim from the export's ``methods.md``, with the reporting
   checklist's answers item by item, every reading the journey injected, where the bundle
   contradicts itself, and what a stage reports that the bundle does not hold; where a journey
   did not reach the export, what stopped it, and where the lens has no reference fixture, what
   one would need;
5. **the questions the reviewer is asked to answer.**

Everything but the journeys is regenerated in a second from the code, so the reference test can
hold every packet to its stored captures (``--reuse``) and fail when the registry has moved on.
The journeys are fits: they rerun only when asked (the default without ``--reuse``).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from turbotab.core.reference import catalog, defaults, journeys
from turbotab.core.reference import methods as ref

OUT = ref.REPO / "docs" / "turbotab-next" / "review-packets"
COMMAND = "python -m turbotab.core.reference.packet"
PURPOSES = ("prediction", "inference")
RELATION_ORDER = ("implies", "enables", "disables", "invalidates", "conflicts", "precedes")
# The questions labeled outside the registry each lens relies on (``custom_sound``).
LABELED_FOR: dict[str, tuple[str, ...]] = {
    "dietary": ("energy_adjustment", "exclusions", "missing", "split", "causal"),
}
LABELED_DEFAULT = ("missing", "split", "causal")
QUESTION_TITLES = dict(ref.labeled_questions())

# What a reference fixture for each lens must hold, said where a journey could not run.
FIXTURE_NEEDS: dict[str, tuple[str, ...]] = {
    "dietary": ("one row per person with one or more 24-hour recalls: total energy in kcal or kJ "
                "and the energy-bearing nutrients in grams, their units declared",
                "an outcome measured on everyone (a continuous one and a yes/no one between them "
                "exercise both checklists)",
                "age, sex, body weight and height (the Goldberg screen and the confounders)",
                "the survey design (weight, strata, PSUs) when the population estimand is to be "
                "exercised"),
    "clinical": ("an identifier and, for the temporal seal, repeated visits with a date",
                 "laboratory and vital-sign columns with a few impossible values to repair",
                 "a yes/no or time-to-event outcome with enough events for Riley's minimum",
                 "the confounders of a declared exposure, with their causal place known"),
    "metabolomics": ("participant samples and pooled-QC injections in one table, with the "
                     "injection order and the batch",
                     "at least five pooled QCs per batch, with a QC injection first and last in "
                     "every batch (QC-RLSC refuses fewer, and refuses to extrapolate)",
                     "intensities with left-censored blanks (non-detections)",
                     "an outcome and a few covariates absent on the QC rows"),
    "genomics": ("raw integer counts, samples in rows, with p ≫ n",
                 "a batch column that varies library size",
                 "a yes/no outcome balanced within batch, and age and sex"),
    "survey": ("an instrument's items on one response scale, with its key (which items are "
               "reverse-coded)",
               "skipped items, sentinel codes for refusals, and respondents' characteristics",
               "an outcome that is not built from the items"),
}


# ── what the registry says about the lens ────────────────────────────────────


def lens_contracts(lens: str) -> tuple[list[Any], list[Any]]:
    """(the lens's own contracts, the shared ones), each in run order."""
    own, shared = [], []
    for c in ref.ordered_contracts():
        lenses = catalog.lenses_of_contract(c.key)
        if catalog.own(lenses, lens):
            own.append(c)
        elif catalog.SHARED in lenses:
            shared.append(c)
    return own, shared


def other_lenses_contracts(lens: str) -> list[Any]:
    """Every contract another lens reviews in full (neither this lens's own nor shared), in run
    order: the app offers each wherever the data hold what it needs, so the packet lists it."""
    return [c for c in ref.ordered_contracts()
            if not catalog.serves(catalog.lenses_of_contract(c.key), lens)]


def lens_families(lens: str) -> tuple[list[Any], list[Any]]:
    """(the families this lens reviews in full, every other family on the shelf)."""
    own, others = [], []
    for f in ref.families():
        (own if catalog.own(f.review_lenses, lens) else others).append(f)
    return own, others


def reviewed_in(lenses: Sequence[str]) -> str:
    if catalog.SHARED in lenses:
        return "every lens (shared): the methods reference"
    names = [catalog.LENS_TITLES[x].lower() for x in lenses]
    joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"
    return f"the {joined} packet{'s' if len(names) > 1 else ''}"


def default_of(c: Any, purpose: str) -> dict[str, Any]:
    """The option ``c`` ranks first for ``purpose`` in the registry: its key, label, rung and
    reason (where the app ranks by the data, ``defaults.cases`` says what it offers first)."""
    return c.options_for(purpose)[0]


BY_DATA = "ranked by the data: §3.4"


def default_cell(c: Any, purpose: str) -> str:
    through = defaults.THROUGH.get(c.key)
    if defaults.cases(c.key, purpose) is not None:
        return BY_DATA + (f" ({QUESTION_TITLES[through]}, `{through}`)" if through else "")
    d = default_of(c, purpose)
    if c.key in catalog.NOT_ASKED and d["rung"] not in ("not_offered", "refused"):
        return (f"never asked, so never applied from the app (§1); posted to the API, the "
                f"registry ranks `{d['key']}` first ({ref.RUNG_WORDS[d['rung']]}): "
                f"{ref.cell(d['sound'])}")
    if d["rung"] == "not_offered":
        return f"not offered under {purpose}: {ref.cell(d['sound'])}"
    if d["rung"] == "refused":  # every option refused for this purpose: there is no default
        return f"refused under {purpose} (`{d['key']}`): {ref.cell(d['sound'])}"
    rung = ref.RUNG_WORDS[d["rung"]]
    return f"`{d['key']}` ({rung}): {ref.cell(d['sound'])}"


# ── the chain ────────────────────────────────────────────────────────────────


def node_id(name: str) -> str:
    return "n_" + re.sub(r"\W", "_", name)


def node_label(text: str) -> str:
    return text.replace('"', "#quot;")


def chain_relations(lens: str) -> list[tuple[Any, Any]]:
    """``(contract, relation)`` for every relation a lens's own method takes part in: declared by
    it, or declared by any contract toward it."""
    own, _ = lens_contracts(lens)
    keys = {c.key for c in own}
    out = []
    for c in ref.ordered_contracts():
        for r in c.relations:
            if c.key in keys or r.target in keys:
                out.append((c, r))
    return out


def mermaid(lens: str) -> list[str]:
    from turbotab.core.contracts import CONTRACTS

    own, _ = lens_contracts(lens)
    keys = {c.key for c in own}
    pairs = chain_relations(lens)
    nodes: dict[str, str] = {}
    lines = ["```mermaid", "flowchart LR"]
    for c in own:
        nodes[c.key] = f'  {node_id(c.key)}["{node_label(c.label)}"]:::own'
    for c, r in pairs:
        if c.key not in nodes:
            nodes[c.key] = f'  {node_id(c.key)}["{node_label(c.label)}"]:::other'
        if r.target not in nodes:
            target = CONTRACTS.get(r.target)
            nodes[r.target] = (f'  {node_id(r.target)}["{node_label(target.label)}"]:::other'
                               if target is not None else
                               f'  {node_id(r.target)}(["{node_label(r.target)}"]):::named')
    lines += list(nodes.values())
    seen: set[tuple[str, str, str]] = set()
    for c, r in pairs:
        edge = (c.key, r.kind, r.target)
        if edge in seen:
            continue
        seen.add(edge)
        arrow = {"invalidates": "==>", "enables": "-.->", "disables": "-.->",
                 "conflicts": "-.->"}.get(r.kind, "-->")
        lines.append(f"  {node_id(c.key)} {arrow}|{r.kind}| {node_id(r.target)}")
    lines += ["  classDef own fill:#e8f0fe,stroke:#1a56db,color:#111",
              "  classDef other fill:#f4f4f5,stroke:#71717a,color:#111",
              "  classDef named fill:#fff7ed,stroke:#c2410c,color:#111", "```"]
    if not keys:
        return []
    return lines


def chain_table(lens: str) -> list[str]:
    lines = ["| From | Kind | Toward | Fires for | Purposes | What the app says |",
             "|---|---|---|---|---|---|"]
    pairs = sorted(chain_relations(lens),
                   key=lambda cr: (RELATION_ORDER.index(cr[1].kind), cr[0].key, cr[1].target))
    for c, r in pairs:
        fires = ", ".join(f"`{w}`" for w in r.when) or "any option"
        if r.condition:
            fires += f"; when {ref.cell(r.condition)}"
        says = ref.cell(r.says)
        if r.rung:
            says += f" *(rung: {ref.RUNG_WORDS[r.rung]})*"
        if r.exits:
            says += " Exits: " + "; ".join(ref.cell(e) for e in r.exits) + "."
        lines.append(f"| `{c.key}` | {r.kind} | `{r.target}` | {fires} | "
                     f"{', '.join(r.purposes)} | {says} |")
    return lines


# ── the journeys ─────────────────────────────────────────────────────────────


def quoted(text: str) -> list[str]:
    """``text`` verbatim, as a Markdown blockquote (each line prefixed, nothing else changed)."""
    return [f"> {line}" if line else ">" for line in text.rstrip("\n").split("\n")]


def where_words(where: Sequence[dict[str, Any]]) -> str:
    out = []
    for w in where:
        if w["source"] == "record":
            out.append(f"decision #{w.get('seq')} (`{w.get('kind')}`)")
        elif w["source"] == "analysis":
            out.append(f"the analysis's paragraph (`{w.get('kind')}`)")
        else:
            out.append(f"`{w.get('file')}`")
    return "; ".join(out)


def checklist_rows(checklist: dict[str, Any]) -> list[str]:
    lines = ["| Item | Topic | Status | Where the bundle answers it | The author must supply |",
             "|---|---|---|---|---|"]
    for item in checklist["items"]:
        label = item["id"]
        if checklist["checklist"] == "TRIPOD+AI" and item.get("scope"):
            label = f"{item['id']} ({item['scope']})"
        lines.append(f"| {label} | {ref.cell(item['topic'])} | {item['status']} | "
                     f"{where_words(item.get('where') or []) or '–'} | "
                     f"{ref.cell(item.get('owed')) or '–'} |")
    return lines


def answers_rows(capture: dict[str, Any]) -> list[str]:
    lines = ["| # | Decision | What was answered | Where the answer came from | Server |",
             "|---|---|---|---|---|"]
    for i, a in enumerate(capture.get("answers") or [], 1):
        if a.get("refusal"):
            server = (f"refused (`{a['refusal'].get('code')}`): "
                      f"{ref.cell(a['refusal'].get('message'))}")
        else:
            server = f"recorded as #{a.get('seq')}" if a.get("seq") else str(a.get("status"))
        lines.append(f"| {i} | `{a.get('kind')}` | {ref.cell(a.get('summary'))} | "
                     f"{ref.cell(a.get('source'))} | {server} |")
    return lines


def _reading_rows(readings: dict[str, str]) -> list[str]:
    lines = ["| Reading | Column | Value |", "|---|---|---|"]
    for key, value in readings.items():
        what, _, column = key.partition(":")
        lines.append(f"| `{what}` | `{column}` | {ref.cell(value)} |")
    return lines


def readings_section(capture: dict[str, Any], h: str) -> list[str]:
    """Every reading the journey answered from, its own apart from the fixture's declared truth,
    and every answer it took as the app ranked or guessed it."""
    readings = capture.get("readings")
    lines = ["", f"{h}# The readings the journey answered from", ""]
    if readings is None:
        return lines + ["This capture predates the readings record: rerun the journey to list them.",
                        ""]
    own = readings.get("journey_own") or {}
    fixture = readings.get("fixture") or ""
    if own:
        lines += ["**The journey's own readings**, beside or in place of the fixture's declared "
                  "truth: the causal assumptions of its research question and the units and "
                  "nesting it states (the server asks only some of them; a prediction asks no "
                  "covariate's causal place). They are the journey author's, not the fixture's; "
                  "please check each:", ""]
        lines += _reading_rows(own)
        lines.append("")
    else:
        lines += ["The journey declares no reading of its own.", ""]
    found = readings.get("from_fixture") or {}
    if found:
        lines.append(f"From the fixture's declared truth (`truths.FIXTURE_TRUTHS[\"{fixture}\"]`), "
                     f"{len(found):,}: " + "; ".join(f"`{k}` = {ref.cell(v)}"
                                                      for k, v in found.items()) + ".")
    elif fixture:
        lines.append(f"The fixture's declared truth (`truths.FIXTURE_TRUTHS[\"{fixture}\"]`) holds "
                     "none of the readings it answered.")
    aside = readings.get("set_aside") or []
    if aside:
        lines.append("Set aside from the fixture's declared truth (another research question's): "
                     + ", ".join(f"`{k}`" for k in aside) + ".")
    ranked = capture.get("app_ranked") or []
    if ranked:
        lines += ["", "Taken as the app ranked them, no reading declaring one: "
                  + "; ".join(f"the {r['reading']} of `{r['column']}`: `{r['value']}`"
                              for r in ranked) + "."]
    guessed = capture.get("guessed") or []
    if guessed:
        lines += ["", "Readings neither declares, confirmed as the app guessed them (a reviewer "
                      "should check these are what the fixture's author would say):", ""]
        lines += [f"- `{g['reading']}` of `{g['column']}`: {g['value']}" for g in guessed]
    return lines


def _holds(name: str, column: str) -> bool:
    """Whether the model-matrix column ``name`` is ``column`` or made from it (an indicator, a
    spline basis term: ``education_Graduate``, ``age'``)."""
    return name == column or (name.startswith(column) and not name[len(column)].isalnum())


def matrix_flags(export: dict[str, Any]) -> list[str]:
    """Where the bundle contradicts itself: a column the primary model is said to be adjusted for
    (Table 2, and the methods that state the same set) that the model matrix does not hold, and a
    model-matrix column that is neither the exposure's nor one of those."""
    columns = list(export.get("model_matrix") or [])
    rows = export.get("table2") or []
    primary = [r for r in rows if "(primary)" in str(r.get("model"))]
    if not columns or not primary:
        return []
    adjusted: list[str] = []
    for r in primary:
        for c in str(r.get("adjusted_for") or "").split(", "):
            if c and c != "nothing" and c not in adjusted:
                adjusted.append(c)
    # (a row per declared model with its ``terms``; a capture's row per term holds one ``term``)
    terms = {str(t) for r in primary for t in r.get("terms") or [r.get("term")] if t}
    absent = [c for c in adjusted if not any(_holds(m, c) for m in columns)]
    unexplained = [m for m in columns
                   if m not in terms and not any(_holds(m, c) for c in adjusted)
                   and not any(_holds(m, t) for t in terms)]
    flags = []
    if absent:
        flags.append("Table 2 and the methods say the primary model is adjusted for "
                     + ", ".join(f"`{c}`" for c in absent) + ", but the model matrix holds no "
                     "column made from " + ("it" if len(absent) == 1 else "them") + ".")
    if unexplained:
        flags.append("The model matrix holds " + ", ".join(f"`{c}`" for c in unexplained)
                     + ", which is neither the exposure's term nor a column the primary model is "
                     "said to be adjusted for; its coefficient is listed in the appendix as an "
                     "adjustment term.")
    return flags


def beside_section(capture: dict[str, Any], h: str) -> list[str]:
    """What a stage reports that the export bundle does not hold (the scales stage's)."""
    scales = (capture.get("beside_the_bundle") or {}).get("scales")
    if not scales:
        return []
    lines = ["", f"{h}# Beside the bundle: what the scales stage reports", "",
             "The export bundle holds neither this text nor these estimates; they are the scales "
             "stage's own, served beside the results. Its methods text, verbatim:", ""]
    lines += quoted(str(scales.get("methods") or ""))
    rows = ["", "| Scale | Its role | Reliability | Uncorrected | Corrected | Attenuation |",
            "|---|---|---|---|---|---|"]
    for sc in scales.get("scales") or []:
        r, c = sc.get("reliability") or {}, sc.get("correction")
        rel = (f"{r.get('label') or r.get('coefficient')} = {r['value']:.3f}"
               if r.get("value") is not None else ref.cell(r.get("reason")) or "–")
        if r.get("alpha") is not None:
            rel += f" (α = {r['alpha']:.3f})"
        if c:
            ratio = c.get("scale") == "odds_ratio" and c.get("ratio") is not None
            naive = (f"OR {c['naive_ratio']:.3f}" if ratio and c.get("naive_ratio") is not None
                     else f"{c['naive']:.4g}")
            fixed = (f"OR {c['ratio']:.3f} ({c['ratio_low']:.3f} to {c['ratio_high']:.3f})"
                     if ratio and c.get("ratio_low") is not None else
                     f"{c['estimate']:.4g}" + (f" ({c['ci_low']:.4g} to {c['ci_high']:.4g})"
                                               if c.get("ci_low") is not None else ""))
            att = f"{c['attenuation']:.3f}"
        else:
            naive = fixed = att = "–"
            fixed = ref.cell(sc.get("not_corrected")) or "not corrected"
        rows.append(f"| `{sc.get('name')}` | {sc.get('role')} | {rel} | {naive} | {fixed} | {att} |")
    return lines + rows + [""]


def journey_section(spec: journeys.Journey, capture: dict[str, Any] | None,
                    level: int = 3) -> list[str]:
    h = "#" * level
    lines = [f"{h} {spec.purpose.capitalize()}: {spec.question}", "",
             f"- **Fixture:** `{spec.fixture}`. {spec.why}",
             f"- **Lens declared:** {', '.join(spec.lenses)}; **outcome:** `{spec.target}`"
             + (f"; **exposure:** `{spec.exposure}`" if spec.exposure else "") + "."]
    if capture is None:
        lines += ["", "**Not run yet**: no capture is stored for this journey. Run "
                      f"`python -m turbotab.core.reference.journeys {spec.name}`.", ""]
        return lines
    cap = capture["captured"]
    export = capture.get("export") or {}
    # (the bundle's own reproducibility sentence says whether the tree had local changes)
    dirty = ", with local changes" if "with local changes" in str(export.get("methods_md")) else ""
    lines.append(f"- **Run:** at commit `{cap.get('commit')}`{dirty} on {cap.get('date')}, "
                 f"{cap.get('seconds'):,} s on {cap.get('workers')} workers"
                 + (f"; engine {export.get('engine')}" if export.get("engine") else "") + ".")
    lines += ["", f"{h}# How the journey answered", "",
              "Every decision posted, in order, with where its answer came from (the journey's "
              "own research question, the app's first-ranked option, the journey's readings, or "
              "a refusal's way forward).", ""]
    lines += answers_rows(capture)
    lines += readings_section(capture, h)
    if capture.get("notes"):
        lines += ["", "Notes from the run:", ""]
        lines += [f"- {ref.cell(n)}" for n in capture["notes"]]
    if export.get("status") != 200:
        refusal = export.get("refusal") or {}
        lines += ["", f"**The journey did not reach the export.** "
                      + (f"The export refused (`{refusal.get('code')}`): "
                         f"{ref.cell(refusal.get('message'))}" if refusal else
                         "It stopped before the export was asked for (the notes say why)."), "",
                  "Until it does, this lens has no methods section to review for this purpose. "
                  "A reference fixture for it needs:", ""]
        lines += [f"- {n}" for n in FIXTURE_NEEDS.get(spec.lens, ())]
        lines.append("")
        return lines
    flags = matrix_flags(export)
    if flags:
        lines += ["", f"{h}# Flagged for the reviewer: the bundle contradicts itself", "",
                  "Found by comparing the bundle's own files (Table 2's `adjusted_for` and the "
                  "provenance record's model-matrix columns), not by reading the methods:", ""]
        lines += [f"- {f}" for f in flags]
    lines += ["", f"{h}# The methods section, verbatim", "",
              "Quoted from the export bundle's `methods.md`, unchanged:", ""]
    lines += quoted(export["methods_md"])
    checklist = export["checklist"]
    c = checklist["counts"]
    lines += ["", f"{h}# The {checklist['checklist']} checklist", "",
              f"{checklist.get('citation') or ''}. {c['items']} items: {c['answered']} answered, "
              f"{c['partly_answered']} partly answered, {c['unanswered']} unanswered.", ""]
    lines += checklist_rows(checklist)
    lines += beside_section(capture, h)
    lines.append("")
    return lines


# ── the packet ───────────────────────────────────────────────────────────────


def render(lens: str, captures: dict[str, dict[str, Any] | None]) -> str:
    """The packet for ``lens`` from the registry and ``captures`` (journey name → capture)."""
    title = catalog.LENS_TITLES[lens]
    own, shared = lens_contracts(lens)
    others = other_lenses_contracts(lens)
    own_families, other_families = lens_families(lens)
    gaps = catalog.gaps_for(lens)
    specs = journeys.for_lens(lens)
    labeled = LABELED_FOR.get(lens, LABELED_DEFAULT)
    commits = sorted({(c or {}).get("captured", {}).get("commit") for c in captures.values()
                      if c} - {None, ""})
    lines = [
        f"# {title}: expert review packet", "",
        f"*Generated by `{COMMAND} --lens {lens}` from the method contracts "
        "(`turbotab/core/contracts.py`), the model-family registry, the labels in "
        "`turbotab/core/custom_sound.py`, the lens catalog (`turbotab/core/reference/catalog.py`) "
        "and the reference journeys' captures (`docs/turbotab-next/review-packets/captures/`"
        + (f", run at commit {', '.join(f'`{c}`' for c in commits)}" if commits else "")
        + "). Do not edit it by hand: `turbotab/core/tests/test_reference.py` fails when it "
        "differs from what the code and the captures generate.*", "",
        "**For the reviewer.** TurboTab v2 is released only after one methodologist per lens has "
        "reviewed the methods it offers, how they chain, the defaults, and the exact sentences it "
        "writes, and each finding is addressed (V2_DEFINITION_OF_DONE §4). This packet holds "
        "those four things for the " + title.lower() + " lens; §5 lists the questions we ask "
        "you to answer. The full contract of every method, including the shared ones summarized "
        "here, is in `docs/turbotab-next/reference/METHODS_REFERENCE.md`.", "",
        "Every option carries two independent labels (BLUEPRINT North star 5): *customary* "
        "(where the field uses it, with a source) and *sound* for the declared purpose (with the "
        "reason). The app offers options soundest first and says so when that departs from "
        "custom. A *rung* is how hard it holds an option (BLUEPRINT §11.3): recommended, "
        "available, ranked lower with its concern stated, blocked until recorded, refused with "
        "an exit, or not offered.", "",
        "## 1 · The methods this lens offers", "",
        f"### 1.1 · Its own methods ({len(own)})", "",
    ]
    if own:
        for c in own:
            lines += ref.contract_section(c, level=4)
    else:
        lines += ["None: every method this lens offers is shared with the other lenses.", ""]
    lines += [f"### 1.2 · Model families", "",
              "The shelf ranks every family that can model the outcome by its own assessment of "
              "the data and never shortens the list: no family is withheld from a lens, so every "
              "family is offered here.", ""]
    if own_families:
        lines += ["Reviewed in full in this packet:", ""]
        for f in own_families:
            lines += ref.family_section(f, level=4)
    lines += ["Every other family on the shelf, each in full in the methods reference:", "",
              "| Family | Outcomes | Purposes | Inductive bias | Reviewed in full in |",
              "|---|---|---|---|---|"]
    for f in other_families:
        lines.append(f"| {f.label} (`{f.key}`) | {', '.join(f.tasks)} | "
                     f"{', '.join(getattr(f, 'purposes', PURPOSES))} | "
                     f"{ref.cell(f.inductive_bias)} | "
                     f"{reviewed_in(f.review_lenses)} |")
    lines += ["", f"### 1.3 · Shared methods ({len(shared)})", "",
              "Every lens offers these; their full contracts are in the methods reference.", "",
              "| Method | Key | Slot | Scope | Recorded by |", "|---|---|---|---|---|"]
    for c in shared:
        lines.append(f"| {ref.cell(c.label)} | `{c.key}` | {c.slot} | {c.scope} | "
                     f"{ref.code(catalog.recorded_by(c)) or 'not declared'} |")
    lines += ["", f"### 1.4 · Methods another lens reviews in full ({len(others)})", "",
              "No method contract is declared for one lens: the app reaches each of these through "
              "the data it needs, not through the lens, though some are reached through findings "
              "or stages their own lens raises. Each is offered here whenever this lens's data "
              "hold what it needs, and is reviewed in full in the packet named.", "",
              "| Method | Key | Reviewed in full in | Slot | What it needs |",
              "|---|---|---|---|---|"]
    for c in others:
        lines.append(f"| {ref.cell(c.label)} | `{c.key}` | "
                     f"{reviewed_in(catalog.lenses_of_contract(c.key))} | {c.slot} | "
                     f"{'; '.join(ref.cell(n) for n in c.needs) or 'nothing declared'} |")
    lines += ["", f"### 1.5 · Methods with no contract yet ({len(gaps)})", "",
              "These are offered (V2_DEFINITION_OF_DONE §2, or by a question the Router asks) and "
              "implemented in the code named, but no method contract declares their slot, scope, "
              "labeled options, leash, storyboard, sentence and relations. They are gaps in the "
              "registry, listed so the review covers them too; one is a limit of the app as well "
              "(its note says so).", ""]
    lines += ref.gap_rows(gaps)
    lines += ["", "Their options are labeled customary and sound outside the registry where "
                  "shown here:", ""]
    for q in labeled:
        lines += ref.question_section(q, QUESTION_TITLES[q], level=4)
    lines += ["## 2 · How they chain", "",
              "Every relation this lens's own methods take part in: declared by them, or declared "
              "by another method toward them (BLUEPRINT §13). Blue: this lens's methods; grey: "
              "other methods; orange: a named consequence (a state, a concern or a refusal, not "
              "a method). Solid: implies or precedes; dotted: enables, disables or conflicts; "
              "thick: invalidates.", ""]
    diagram = mermaid(lens)
    lines += diagram or ["This lens has no methods of its own, so no chain of its own; the shared "
                         "chain is in the methods reference."]
    lines += ["", "The relations, with the sentence the app states when each fires:", ""]
    lines += chain_table(lens) if diagram else ["None."]
    lines += ["", "## 3 · The defaults, by purpose", "",
              "The option each method offers first for each purpose, its rung, and the reason its "
              "*sound* label gives. Where the app ranks a method's options by the data in front of "
              "it (the outcome's event share, the number of units, time order, the data's kind, "
              "what can run on the table, a failed check), the cell says so and §3.4 gives the "
              "first option under each condition, computed by the app's own ranking code; "
              "elsewhere it is the method's rank 1 in the registry. A default is never applied "
              "silently: the option is asked, or stated in the Record with its phrase changeable "
              "(BLUEPRINT §11.4). A method no question asks says “never asked”: from the app it is "
              "not applied at all, whatever its rank.", "",
              "### 3.1 · This lens's own methods", ""]
    if own:
        lines += ["| Method | Prediction | Inference |", "|---|---|---|"]
        lines += [f"| {ref.cell(c.label)} (`{c.key}`) | {default_cell(c, 'prediction')} | "
                  f"{default_cell(c, 'inference')} |" for c in own]
    else:
        lines.append("None of its own.")
    lines += ["", "### 3.2 · Questions labeled outside the registry", "",
              "| Question | Prediction | Inference |", "|---|---|---|"]
    for q in labeled:
        cells = []
        for p in PURPOSES:
            if defaults.cases(q, p) is not None:
                cells.append(BY_DATA)
                continue
            first = ref.question_first(q, p)
            cells.append(f"`{first[0]}` ({first[1]}): {ref.cell(first[2])}" if first
                         else f"not asked under {p}")
        lines.append(f"| {QUESTION_TITLES[q]} (`{q}`) | {cells[0]} | {cells[1]} |")
    lines += ["", "### 3.3 · Shared methods", "",
              "| Method | Prediction | Inference |", "|---|---|---|"]
    lines += [f"| {ref.cell(c.label)} (`{c.key}`) | {default_cell(c, 'prediction')} | "
              f"{default_cell(c, 'inference')} |" for c in shared]
    lines += ["", "### 3.4 · Where the app ranks by the data", "",
              "Each condition the app ranks by, and what it offers first under it, computed by "
              "calling the function the app ranks with (`turbotab/core/reference/defaults.py` "
              "names each).", "",
              "| Method | Purpose | When | What the app offers first |", "|---|---|---|---|"]
    # (a contract reached only through a question, the causal lane's estimators, shows as that
    # question's rows)
    shown = [(ref.cell(c.label), c.key) for c in (*own, *shared) if c.key not in defaults.THROUGH]
    shown += [(QUESTION_TITLES[q][:1].upper() + QUESTION_TITLES[q][1:], q) for q in labeled]
    for label, key in shown:
        for p in PURPOSES:
            for case in defaults.cases(key, p) or []:
                lines.append(f"| {label} (`{key}`) | {p} | {ref.cell(case.when)} | "
                             f"{ref.cell(case.first)} |")
    lines += ["", "The model families have no static default: the shelf ranks them on the data "
                  "(each family's own assessment) and the journeys below show what it ranked "
                  "first.", "",
              "## 4 · The sentences the app writes: the reference journeys", "",
              "Each journey ran headlessly through the real server (the acceptance harness's "
              "drivers, two workers) from upload to the export bundle "
              "(`turbotab/core/export`). The methods section is quoted exactly as the bundle's "
              "`methods.md` holds it; the checklist is the bundle's, item by item.", ""]
    if not specs:
        lines += [f"**This lens has no reference fixture.** A reference fixture for it needs:", ""]
        lines += [f"- {n}" for n in FIXTURE_NEEDS.get(lens, ())]
        lines.append("")
    for purpose in PURPOSES:
        spec = next((s for s in specs if s.purpose == purpose), None)
        if spec is None:
            lines += [f"### {purpose.capitalize()}", "",
                      f"**No reference journey for {purpose} on this lens.** A reference fixture "
                      "for it needs:", ""]
            lines += [f"- {n}" for n in FIXTURE_NEEDS.get(lens, ())]
            lines.append("")
            continue
        lines += journey_section(spec, captures.get(spec.name))
    lines += ["### What a reference fixture for this lens must hold", "",
              "The journeys above are only as good as their fixtures. A fixture that exercises this "
              "lens's own methods needs:", ""]
    lines += [f"- {n}" for n in FIXTURE_NEEDS.get(lens, ())]
    lines += ["", "Where a journey above notes that a method of this lens was refused or set "
                  "aside, its sentence is not in that methods section; a fixture meeting these "
                  "needs is owed before the review can cover it.", ""]
    lines += [
        "## 5 · The questions for the reviewer", "",
        "Please answer each, citing the section and the method key or the sentence:", "",
        "1. **Is any default unsound, or uncustomary without saying so?** (§3, and the options "
        "tables in §1.) A default is unsound when its *sound* reason is wrong for the purpose; "
        "it is uncustomary without saying so when the field would reach for another option "
        "first and the app's labels do not name that tension.",
        "2. **Is any chain wrong?** (§2.) A relation that should not fire, one that is missing "
        "(a choice that should re-ask, close or refuse another and does not), or a run order "
        "that is wrong.",
        "3. **Is any sentence unpublishable?** (§4.) A methods sentence a journal reviewer would "
        "reject: wrong, unclear, missing what STROBE-nut or TRIPOD+AI asks for, or claiming more "
        "than the analysis did. The checklist tables show which items the bundle answers and "
        "which it leaves to the author.",
        "4. **Is any source misread?** Some sources were read as abstracts or page summaries "
        "(MODELING_SEQUENCE §7); please check the ones this lens's labels cite against the full "
        "texts.",
        "5. **Is a method missing, or a gap in §1.5 more than a gap?** A method the field expects "
        "that the app does not offer, or an uncontracted method whose behavior you would not "
        "sign off on.", "",
    ]
    return "\n".join(lines)


def captures_for(lens: str, folder: Path = journeys.CAPTURES) -> dict[str, dict[str, Any] | None]:
    return {s.name: journeys.load_capture(s.name, folder) for s in journeys.for_lens(lens)}


def write(lens: str, *, reuse: bool = True, folder: Path = journeys.CAPTURES,
          out: Path = OUT, bundles: Path | None = None) -> Path:
    if not reuse:
        for spec in journeys.for_lens(lens):
            journeys.run_journey(spec, bundles=bundles, out_dir=folder)
    text = render(lens, captures_for(lens, folder))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{lens}.md"
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path} ({len(text.encode('utf-8')):,} bytes)")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--lens", choices=catalog.LENSES)
    which.add_argument("--all", action="store_true")
    parser.add_argument("--reuse", action="store_true",
                        help="build from the stored captures instead of running the journeys")
    parser.add_argument("--check", action="store_true",
                        help="with --reuse: exit 1 when a committed packet is stale")
    parser.add_argument("--bundles", type=Path, default=None)
    args = parser.parse_args(argv)
    lenses = catalog.LENSES if args.all else (args.lens,)
    if args.check:
        stale = [lens for lens in lenses if not (OUT / f"{lens}.md").is_file()
                 or (OUT / f"{lens}.md").read_text(encoding="utf-8")
                 != render(lens, captures_for(lens))]
        for lens in stale:
            print(f"{OUT / f'{lens}.md'} is stale: run {COMMAND} --lens {lens} --reuse",
                  file=sys.stderr)
        return 1 if stale else 0
    for lens in lenses:
        write(lens, reuse=args.reuse, bundles=args.bundles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
