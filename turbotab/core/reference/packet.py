"""One lens's expert review packet (V2 definition of done §4: "one domain methodologist per lens,
from a per-domain review packet: the methods offered, how they chain, the defaults, and the exact
sentences the app writes").

    python -m turbotab.core.reference.packet --lens dietary            # runs the lens's journeys
    python -m turbotab.core.reference.packet --lens dietary --reuse    # from the stored captures
    python -m turbotab.core.reference.packet --all --reuse --check     # exit 1 when stale

writes ``docs/turbotab-next/review-packets/<lens>.md``:

1. **the methods the lens offers**, from the registry: its own method contracts in full, its own
   model families, then every shared method in one table, and the methods it relies on that have
   no contract yet (``catalog.GAPS``), with the labels kept outside the registry
   (``custom_sound``) where they exist;
2. **how they chain**: a Mermaid diagram of every relation the lens's own methods take part in
   (from them, or toward them from any contract), and the relations in a table with the sentence
   the app states when each fires;
3. **the defaults by purpose**, each with its reason: the option each method offers first for
   prediction and for inference, its rung, and the sound label that justifies it;
4. **the exact methods sections the app writes** on the lens's reference journeys
   (``journeys``), quoted verbatim from the export's ``methods.md``, with the reporting
   checklist's answers item by item; where a journey did not reach the export, what stopped it,
   and where the lens has no reference fixture, what one would need;
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

from turbotab.core.reference import catalog, journeys
from turbotab.core.reference import methods as ref

OUT = ref.REPO / "docs" / "turbotab-next" / "review-packets"
COMMAND = "python -m turbotab.core.reference.packet"
PURPOSES = ("prediction", "inference")
RELATION_ORDER = ("implies", "enables", "disables", "invalidates", "conflicts", "precedes")
# The questions labeled outside the registry each lens relies on (``custom_sound``).
LABELED_FOR: dict[str, tuple[str, ...]] = {
    "dietary": ("energy_adjustment", "exclusions", "missing", "split"),
}
LABELED_DEFAULT = ("missing", "split")
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


def lens_families(lens: str) -> tuple[list[Any], list[Any]]:
    own, shared = [], []
    for f in ref.families():
        lenses = catalog.lenses_of_family(f.key)
        if catalog.own(lenses, lens):
            own.append(f)
        elif catalog.SHARED in lenses:
            shared.append(f)
    return own, shared


def default_of(c: Any, purpose: str) -> dict[str, Any]:
    """The option ``c`` offers first for ``purpose``: its key, label, rung and reason."""
    first = c.options_for(purpose)[0]
    return first


def default_cell(c: Any, purpose: str) -> str:
    d = default_of(c, purpose)
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
        # (captures written before the driver counted one reading in the singular say "1 readings")
        summary = re.sub(r"^1 readings \(", "1 reading (", str(a.get("summary") or ""))
        lines.append(f"| {i} | `{a.get('kind')}` | {ref.cell(summary)} | "
                     f"{ref.cell(a.get('source'))} | {server} |")
    return lines


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
    lines.append(f"- **Run:** at commit `{cap.get('commit')}` on {cap.get('date')}, "
                 f"{cap.get('seconds'):,} s on {cap.get('workers')} workers"
                 + (f"; engine {export.get('engine')}" if export.get("engine") else "") + ".")
    lines += ["", f"{h}# How the journey answered", "",
              "Every decision posted, in order, with where its answer came from (the journey's "
              "own research question, the app's first-ranked option, the fixture's declared "
              "truth, or a refusal's way forward).", ""]
    lines += answers_rows(capture)
    guessed = capture.get("guessed") or []
    if guessed:
        lines += ["", "Readings the fixture declares no truth for, confirmed as the app guessed "
                      "them (a reviewer should check these are what the fixture's author would "
                      "say):", ""]
        lines += [f"- `{g['reading']}` of `{g['column']}`: {g['value']}" for g in guessed]
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
    lines += ["", f"{h}# The methods section, verbatim", "",
              "Quoted from the export bundle's `methods.md`, unchanged:", ""]
    lines += quoted(export["methods_md"])
    checklist = export["checklist"]
    c = checklist["counts"]
    lines += ["", f"{h}# The {checklist['checklist']} checklist", "",
              f"{checklist.get('citation') or ''}. {c['items']} items: {c['answered']} answered, "
              f"{c['partly_answered']} partly answered, {c['unanswered']} unanswered.", ""]
    lines += checklist_rows(checklist)
    lines.append("")
    return lines


# ── the packet ───────────────────────────────────────────────────────────────


def render(lens: str, captures: dict[str, dict[str, Any] | None]) -> str:
    """The packet for ``lens`` from the registry and ``captures`` (journey name → capture)."""
    title = catalog.LENS_TITLES[lens]
    own, shared = lens_contracts(lens)
    own_families, shared_families = lens_families(lens)
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
    lines += [f"### 1.2 · Model families", ""]
    if own_families:
        lines += ["Its own:", ""]
        for f in own_families:
            lines += ref.family_section(f, level=4)
    lines += ["Shared with every lens (the shelf ranks every family that can model the outcome "
              "by its own assessment of the data, and never shortens the list):", "",
              "| Family | Outcomes | Purposes | Inductive bias |", "|---|---|---|---|"]
    for f in shared_families:
        lines.append(f"| {f.label} (`{f.key}`) | {', '.join(f.tasks)} | "
                     f"{', '.join(getattr(f, 'purposes', PURPOSES))} | "
                     f"{ref.cell(f.inductive_bias)} |")
    lines += ["", f"### 1.3 · Shared methods ({len(shared)})", "",
              "Every lens offers these; their full contracts are in the methods reference.", "",
              "| Method | Key | Slot | Scope | Recorded by |", "|---|---|---|---|---|"]
    for c in shared:
        lines.append(f"| {ref.cell(c.label)} | `{c.key}` | {c.slot} | {c.scope} | "
                     f"{ref.code(c.decision) or 'not declared'} |")
    lines += ["", f"### 1.4 · Methods with no contract yet ({len(gaps)})", "",
              "These are offered (V2_DEFINITION_OF_DONE §2) and implemented in the code named, "
              "but no method contract declares their slot, scope, labeled options, leash, "
              "storyboard, sentence and relations. They are gaps in the registry, listed so the "
              "review covers them too.", ""]
    lines += ref.gap_rows(gaps)
    lines += ["", "Their options are labeled customary and sound outside the registry where "
                  "shown here:", ""]
    for q in labeled:
        lines += ref.custom_sound_section(q, QUESTION_TITLES[q], level=4)
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
              "The option each method offers first for each purpose (its rank 1), its rung, and "
              "the reason its *sound* label gives. A default is never applied silently: the "
              "option is asked, or stated in the Record with its phrase changeable "
              "(BLUEPRINT §11.4).", "",
              "### 3.1 · This lens's own methods", ""]
    if own:
        lines += ["| Method | Prediction | Inference |", "|---|---|---|"]
        lines += [f"| {ref.cell(c.label)} (`{c.key}`) | {default_cell(c, 'prediction')} | "
                  f"{default_cell(c, 'inference')} |" for c in own]
    else:
        lines.append("None of its own.")
    lines += ["", "### 3.2 · Questions labeled outside the registry", "",
              "| Question | Prediction | Inference |", "|---|---|---|"]
    from turbotab.core import custom_sound

    for q in labeled:
        cells = []
        for p in PURPOSES:
            first = custom_sound.labels_for(q, p).options[0]
            cells.append(f"`{first.key}` ({first.sound.verdict}): {ref.cell(first.sound.reason)}")
        lines.append(f"| {QUESTION_TITLES[q]} (`{q}`) | {cells[0]} | {cells[1]} |")
    if "energy_adjustment" in labeled:
        lines += ["", "The energy model's first option is the first that can run on the table's "
                      "columns: all components needs every energy source, so on a table without "
                      "them the next ranked leads, and the coach says so."]
    lines += ["", "### 3.3 · Shared methods", "",
              "| Method | Prediction | Inference |", "|---|---|---|"]
    lines += [f"| {ref.cell(c.label)} (`{c.key}`) | {default_cell(c, 'prediction')} | "
              f"{default_cell(c, 'inference')} |" for c in shared]
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
        "5. **Is a method missing, or a gap in §1.4 more than a gap?** A method the field expects "
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
