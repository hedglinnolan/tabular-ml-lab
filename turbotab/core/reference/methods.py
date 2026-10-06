"""The methods reference, generated from the contracts (V2 definition of done §4, "Docs").

    python -m turbotab.core.reference.methods          # writes the reference
    python -m turbotab.core.reference.methods --check  # exits 1 when the committed copy is stale

writes ``docs/turbotab-next/reference/METHODS_REFERENCE.md``. For every method contract
(BLUEPRINT §13; ``turbotab.core.contracts``) it gives the slot, the data scope and the needs; the
question that asks it; its options, each with its customary and sound labels (North star 5); the
leash rung of each option for each purpose (§11.3); its storyboard; its methods sentence (the
template, or the function that writes it); its relations (implies, enables, disables, invalidates,
conflicts, precedes); and its primary sources. Then every model family as the family registry
declares it, the questions whose options are labeled outside the registry
(``turbotab.core.custom_sound``), and the methods v2 offers that have no contract yet
(``catalog.GAPS``).

The text is a pure function of the code: nothing reads the clock, and every list is in a fixed
order (contracts by slot, their place in it, then key; families in registration order).
``turbotab/core/tests/test_reference.py`` regenerates it and fails when the committed copy differs.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from turbotab.core.reference import catalog

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs" / "turbotab-next" / "reference" / "METHODS_REFERENCE.md"
COMMAND = "python -m turbotab.core.reference.methods"
PURPOSES = ("prediction", "inference")

SLOT_WORDS = {
    "ingest": "ingest (as the table is read)",
    "repairs": "repairs (before the seal)",
    "reshape": "reshape (before the seal)",
    "eligibility": "eligibility (before the seal)",
    "seal": "the seal",
    "in_fold": "in-fold (fit on training rows only)",
    "model": "the model",
    "evaluation": "evaluation (after the fit)",
}
SCOPE_WORDS = {
    "row_local": "row-local: needs no other row",
    "reference_rows": "reference rows: learns only from technical replicates (pooled QCs, blanks, "
                      "standards), never a participant or the outcome",
    "training_fold": "training fold: learns from study rows, so it is fit in-fold",
    "descriptive": "descriptive: may read every row to say whether the data are corrupted, but "
                   "informs no modeling choice",
    "model": "model: the outcome model itself",
}
RUNG_WORDS = {
    "recommended": "recommended",
    "available": "available",
    "rank_lower": "ranked lower, concern stated",
    "block_and_record": "blocked until recorded",
    "refused": "refused, with an exit",
    "not_offered": "not offered",
}
RELATION_WORDS = {
    "implies": "a follow-on the app states rather than asks",
    "enables": "a downstream option it opens",
    "disables": "a downstream option it closes",
    "invalidates": "an answer re-asked, never silently kept, when this one changes",
    "conflicts": "a pair that cannot coexist, refused or blocked with an exit",
    "precedes": "run order",
}


# ── small helpers ────────────────────────────────────────────────────────────


def cell(text: Any) -> str:
    """``text`` safe inside a Markdown table cell."""
    s = "" if text is None else str(text)
    s = s.replace("\r", " ").replace("\n", " ").replace("|", "\\|")
    return re.sub(r"\s+", " ", s).strip()


def code(text: str) -> str:
    return f"`{text}`" if text else ""


def md_docstring(text: str) -> str:
    """A docstring's reStructuredText inline code as Markdown (````x```` → ```x```)."""
    return re.sub(r"``([^`]+)``", r"`\1`", text)


def first_sentence(doc: str | None) -> str:
    """A docstring's first paragraph, joined to one line."""
    if not doc:
        return ""
    paragraph = inspect.cleandoc(doc).split("\n\n", 1)[0]
    return md_docstring(" ".join(line.strip() for line in paragraph.splitlines()))


def resolve(target: str) -> Any:
    """``module:attribute`` (the registry's way of naming a sentence writer) as the object."""
    module, _, name = target.partition(":")
    obj: Any = importlib.import_module(module)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def callable_name(fn: Callable[..., Any]) -> str:
    """``module:qualname``, or the module alone when the function has no name of its own (a
    lambda or a function built inside another)."""
    qual = getattr(fn, "__qualname__", "")
    module = getattr(fn, "__module__", "")
    if not qual or "<" in qual:
        return f"a function in {module}"
    return f"{module}:{qual}"


def ordered_contracts() -> list[Any]:
    from turbotab.core.contracts import SLOTS, contracts

    found = contracts()
    return sorted(found.values(), key=lambda c: (SLOTS.index(c.slot), c.run_order, c.key))


def families() -> list[Any]:
    """Every registered family. The declaring modules are imported first: the omics chain
    registers a family of its own (``screened_elastic_net``)."""
    import turbotab.core.models  # noqa: F401 - registers the families
    from turbotab.core.contracts import contracts
    from turbotab.core.models.base import families as registered

    contracts()
    # The catalog's order (the shelf's registration order), so the text never depends on which
    # module a process happened to import first.
    order = {key: i for i, key in enumerate(catalog.FAMILY_LENSES)}
    return sorted(registered(), key=lambda f: (order.get(f.key, len(order)), f.key))


def question_words(c: Any) -> str:
    """The question a contract declares. Some contracts name the Router question (or the findings)
    that asks them instead of a question of their own: that is said."""
    from turbotab.core.interview import QUESTION_KEYS
    from turbotab.core.voice import question_name

    q = (c.question or "").strip()
    if q in QUESTION_KEYS:
        return f"asked within {question_name(q)} (`{q}`)"
    if q == "findings":
        return "asked within the findings, as a repair (`findings`)"
    return q


def sentence_words(c: Any) -> list[str]:
    """How a contract's methods sentence is written: its template, or the function that writes
    it (with what that function's docstring says it writes), and its in-fold phrase."""
    out: list[str] = []
    s = c.sentence
    if isinstance(s, str) and re.fullmatch(r"[\w.]+:[\w.]+", s):
        try:
            fn = resolve(s)
            doc = first_sentence(getattr(fn, "__doc__", None))
        except (ImportError, AttributeError):  # pragma: no cover - a stale pointer is a defect
            doc = "(the function named here does not resolve)"
        out.append(f"Written by `{s}`" + (f": {doc}" if doc else "."))
    elif isinstance(s, str) and s:
        out.append(f"Template: “{s}”")
    elif callable(s):
        doc = first_sentence(getattr(s, "__doc__", None))
        out.append(f"Written by `{callable_name(s)}`" + (f": {doc}" if doc else "."))
    if c.clause is not None:
        out.append(f"Its clause in a chain's methods paragraph (`contracts.paragraph`) is written "
                   f"by `{callable_name(c.clause)}`.")
    shorts = {**({"": c.short} if c.short else {}), **dict(c.option_shorts)}
    for option, phrase in shorts.items():
        if not phrase:
            continue
        which = f" (option `{option}`)" if option else ""
        out.append(f"Under prediction it is named in the in-fold group{which}: “within each "
                   f"training fold, {phrase} … were fitted and applied to the held-out fold”.")
    if not out:
        out.append("**None declared:** no template, no sentence function, no clause and no in-fold "
                   "phrase.")
    return out


def option_table(c: Any) -> list[str]:
    lines = ["| Option | Customary | Sound for prediction | Rung, rank (prediction) | "
             "Sound for inference | Rung, rank (inference) |",
             "|---|---|---|---|---|---|"]
    ranks = {p: {o["key"]: i + 1 for i, o in enumerate(c.options_for(p))} for p in PURPOSES}
    for o in c.options:
        extra = []
        if o.key in c.option_slots:
            extra.append(f"runs at {c.option_slots[o.key]}")
        if o.key in c.option_scopes:
            extra.append(f"scope {c.option_scopes[o.key]}")
        name = f"`{o.key}`: {cell(o.label)}" + (f" ({'; '.join(extra)})" if extra else "")
        lines.append("| " + " | ".join([
            name, cell(o.customary),
            cell(o.sound["prediction"]),
            f"{RUNG_WORDS[o.rung['prediction']]}, {ranks['prediction'][o.key]}",
            cell(o.sound["inference"]),
            f"{RUNG_WORDS[o.rung['inference']]}, {ranks['inference'][o.key]}",
        ]) + " |")
    return lines


def relation_rows(relations: Iterable[Any]) -> list[str]:
    lines = ["| Kind | Toward | Fires for | Purposes | What the app says | Enforced by |",
             "|---|---|---|---|---|---|"]
    for r in relations:
        fires = ", ".join(f"`{w}`" for w in r.when) or "any option"
        if r.condition:
            fires += f"; when {cell(r.condition)}"
        says = cell(r.says)
        if r.rung:
            says += f" *(rung: {RUNG_WORDS[r.rung]})*"
        if r.exits:
            says += " Exits: " + "; ".join(cell(e) for e in r.exits) + "."
        name = f"`{r.target}`" + (f" (id `{r.id}`)" if r.id and r.id != r.target else "")
        lines.append("| " + " | ".join([
            r.kind, name, fires, ", ".join(r.purposes), says, code(r.enforced_by)]) + " |")
    return lines


def data_ranked(key: str) -> list[str]:
    """Where the app ranks ``key``'s options by the data, what it offers first under each condition
    (``defaults``: computed by the app's own ranking code), as Markdown lines; empty where the
    registry's rank holds."""
    from turbotab.core.reference import defaults

    lines: list[str] = []
    for p in PURPOSES:
        for case in defaults.cases(key, p) or []:
            lines.append(f"- Under {p}, {case.when}: {case.first}")
    if not lines:
        return []
    return ["**Ranked by the data:** the app offers these options in another order than the "
            "rank above where the data decide it; what it offers first under each condition:", "",
            *lines, ""]


def lens_words(lenses: Sequence[str]) -> str:
    if catalog.SHARED in lenses:
        return "every lens (shared)"
    return ", ".join(catalog.LENS_TITLES[x] for x in lenses)


# ── one contract ─────────────────────────────────────────────────────────────


def contract_section(c: Any, level: int = 3) -> list[str]:
    h = "#" * level
    lines = [f"{h} {c.label} (`{c.key}`)", ""]
    facts = [f"- **Slot:** {SLOT_WORDS[c.slot]}"
             + (f"; place in it {c.run_order:g}" if c.run_order else "") + "."]
    if c.option_slots:
        facts.append("  Options that run elsewhere: " + "; ".join(
            f"`{k}` at {v}" for k, v in c.option_slots.items()) + ".")
    facts.append(f"- **Data scope:** {SCOPE_WORDS[c.scope]}."
                 + (f" {c.scope_note}" if c.scope_note else ""))
    if c.parts:
        facts.append("  Its parts: " + "; ".join(f"{p} is {s}" for p, s in c.parts.items()) + ".")
    if c.needs:
        facts.append("- **Needs:**")
        facts += [f"  - {n}" for n in c.needs]
    else:
        facts.append("- **Needs:** nothing declared.")
    facts.append(f"- **Question:** {question_words(c)}")
    if c.key in catalog.NOT_ASKED:
        facts.append(f"- **Asked:** never, so never applied from the app. "
                     f"{catalog.NOT_ASKED[c.key]}")
    asked = []
    kind = catalog.recorded_by(c)
    if kind:
        asked.append(f"recorded by `{kind}`"
                     + ("" if c.decision else " (the contract does not name it)"))
    if c.stage:
        asked.append(f"run by the `{c.stage}` stage")
    if c.place:
        asked.append(f"placed at {c.place}")
    if asked:
        facts.append("- **Where:** " + "; ".join(asked).rstrip(".") + ".")
    if c.leash:
        facts.append("- **The method's own leash:** " + "; ".join(
            f"{p} {RUNG_WORDS[c.leash[p]]}" for p in PURPOSES if p in c.leash) + ".")
    facts.append(f"- **Lenses:** {lens_words(catalog.lenses_of_contract(c.key))}."
                 + (f" Declared by the {c.package} package." if c.package else ""))
    lines += facts + ["", "**Options**, each labeled customary and sound for each purpose, with "
                          "its leash rung and its rank (1 is offered first):", ""]
    lines += option_table(c)
    lines.append("")
    lines += data_ranked(c.key)
    lines += ["**Storyboard** (the transform player's real steps):", ""]
    lines += [f"{i}. {step}" for i, step in enumerate(c.storyboard, 1)] or ["None declared."]
    lines += ["", "**Methods sentence:**", ""]
    lines += [f"- {s}" for s in sentence_words(c)]
    lines += ["", "**Relations:**", ""]
    lines += relation_rows(c.relations) if c.relations else ["None declared."]
    lines += ["", "**Primary sources:**", ""]
    lines += [f"- {s}" for s in c.sources] or ["None declared."]
    lines.append("")
    return lines


# ── one family ───────────────────────────────────────────────────────────────


def family_section(f: Any, level: int = 3) -> list[str]:
    h = "#" * level
    purposes = tuple(getattr(f, "purposes", PURPOSES))
    lines = [f"{h} {f.label} (`{f.key}`)", "",
             f"- **Outcomes it models:** {', '.join(f.tasks)}.",
             f"- **Purposes it serves:** {', '.join(purposes)}"
             + ("" if getattr(f, "predicts", True) else "; it tests and makes no predictions")
             + ".",
             f"- **Inductive bias:** {f.inductive_bias}",
             "- **Strengths:** " + " ".join(f.strengths),
             "- **Cautions:** " + " ".join(f.cautions),
             f"- **Needs scaled inputs:** {'yes' if f.needs_scaling else 'no'}; "
             f"**uses rows with blanks as they are:** {'yes' if f.handles_missing else 'no'}; "
             f"**Harrell's bootstrap optimism is sound for it:** "
             f"{'yes' if getattr(f, 'bootstrap_optimism', True) else 'no'}.",
             f"- **Lenses:** {lens_words(catalog.lenses_of_family(f.key))}.",
             "- **Question:** asked within the models question (`select_models`); the shelf ranks "
             "every family by its own assessment of the data and never shortens the list.",
             "", "**What the app says it fits**, by outcome and purpose (`describe`):", "",
             "| Outcome | Purpose | Model | Detail |", "|---|---|---|---|"]
    for task in f.tasks:
        for p in purposes:
            label, detail = f.describe(task, p)
            lines.append(f"| {task} | {p} | {cell(label)} | {cell(detail)} |")
    lines += ["", "**Not declared, because a family is not a method contract:** options labeled "
                  "customary and sound, a leash rung per purpose, a storyboard, relations and "
                  "primary sources (see the gaps).", ""]
    return lines


# ── the questions labeled outside the registry ───────────────────────────────


def labeled_questions() -> list[tuple[str, str]]:
    """``(question, words)`` for each question whose options are labeled outside the registry:
    the four ``custom_sound`` labels, and the causal lane's (``causal.options``)."""
    return [("energy_adjustment", "the energy model"), ("exclusions", "the eligibility screens"),
            ("missing", "missing values"), ("split", "the split and the validation"),
            ("causal", "the causal lane's estimator")]


def question_section(question: str, title: str, level: int = 3) -> list[str]:
    """A labeled question's options, soundest first, with both labels."""
    if question == "causal":
        return causal_section(title, level)
    return custom_sound_section(question, title, level)


def question_first(question: str, purpose: str) -> tuple[str, str, str] | None:
    """(key, verdict, reason) of the option a labeled question ranks first for ``purpose`` with no
    data in front of it; None where it is not asked under ``purpose``."""
    if question == "causal":
        if purpose != "inference":
            return None
        from turbotab.core.causal import STATED

        return "none", "stated", f"Stated, not asked: {STATED}"
    from turbotab.core import custom_sound

    first = custom_sound.labels_for(question, purpose).options[0]
    return first.key, first.sound.verdict, first.sound.reason


def causal_section(title: str, level: int = 3) -> list[str]:
    """The causal lane's question (``turbotab/core/causal.py``): stated, never asked by default,
    and its estimators' order under each condition the lane ranks by (``defaults``)."""
    from turbotab.core.causal import STATED, plan_reason
    from turbotab.core.reference import defaults

    h = "#" * level
    lines = [f"{h} {title[:1].upper()}{title[1:]} (`causal`)", "",
             "Under **prediction** it is not asked (`causal.plan_reason`): "
             f"“{plan_reason({'purpose': 'prediction'})}”", "",
             "Under **inference** it is stated, never asked by default (`causal.causal_gate`): "
             f"“{STATED}” The primary model only (`none`) is the declared analysis; “Ask me "
             "anyway” opens the lane, whose estimators are each a method contract (`dml_plr`, "
             "`dml_irm`, `tmle`, `pds_lasso`) recorded by `set_causal`. Its options are labeled "
             "in `causal.options`, outside the registry, and ranked by the exposure's kind, the "
             "outcome, how many candidate terms there are for n (more than one per 10 of the "
             "limiting sample size), and the survey answer. Soundest first under each condition:",
             "", "| When | Rank | Option | Customary (field: text, source) | Sound | Reason |",
             "|---|---|---|---|---|---|"]
    for when, ranked in defaults.causal_rankings():
        for i, o in enumerate(ranked, 1):
            lines.append(f"| {cell(when) if i == 1 else ''} | {i} | `{o.key}`: {cell(o.label)} | "
                         f"{cell(o.customary.field)}: {cell(o.customary.text)} "
                         f"({cell(o.customary.source)}) | {o.sound.verdict} | "
                         f"{cell(o.sound.reason)} |")
    lines.append("")
    lines += data_ranked("causal")
    return lines


def custom_sound_section(question: str, title: str, level: int = 3) -> list[str]:
    from turbotab.core import custom_sound

    h = "#" * level
    lines = [f"{h} {title[:1].upper()}{title[1:]} (`{question}`)", ""]
    for p in PURPOSES:
        q = custom_sound.labels_for(question, p)
        lines += [f"Under **{p}**, soundest first. The field reaches for `{q.customary_first}` "
                  "first." + (f" Tension the coach names: “{q.tension}”" if q.tension else ""), "",
                  "| Rank | Option | Customary (field: text, source) | Sound | Reason |",
                  "|---|---|---|---|---|"]
        for i, o in enumerate(q.options, 1):
            lines.append(f"| {i} | `{o.key}`: {cell(o.label)} | {cell(o.customary.field)}: "
                         f"{cell(o.customary.text)} ({cell(o.customary.source)}) | "
                         f"{o.sound.verdict} | {cell(o.sound.reason)} |")
        lines.append("")
    lines += data_ranked(question)
    return lines


# ── the gaps ─────────────────────────────────────────────────────────────────


def gap_rows(gaps: Sequence[catalog.Gap]) -> list[str]:
    lines = ["| Method | Where v2 offers it | Lenses | Recorded by | Implemented in | Labels | "
             "Contracts covering part of it | Note |", "|---|---|---|---|---|---|---|---|"]
    for g in gaps:
        lines.append("| " + " | ".join([
            cell(g.name), cell(g.row), lens_words(g.lenses),
            ", ".join(f"`{d}`" for d in g.decisions) or "computed, not asked",
            ", ".join(f"`{m}`" for m in g.modules),
            f"`{g.labels}`" if g.labels else "none",
            ", ".join(f"`{k}`" for k in g.contracted_parts) or "none",
            cell(g.note)]) + " |")
    return lines


# ── the whole reference ──────────────────────────────────────────────────────


def render() -> str:
    """The methods reference, as Markdown."""
    contracts = ordered_contracts()
    fams = families()
    questions = labeled_questions()
    lines = [
        "# TurboTab v2 methods reference", "",
        f"*Generated by `{COMMAND}` from the method contracts (`turbotab/core/contracts.py`, "
        "BLUEPRINT §13), the model-family registry (`turbotab/core/models/`), the customary and "
        "sound labels kept outside the registry (`turbotab/core/custom_sound.py`) and the lens "
        "catalog (`turbotab/core/reference/catalog.py`). Do not edit it by hand: "
        "`turbotab/core/tests/test_reference.py` regenerates it and fails when this copy is "
        "stale.*", "",
        f"It covers {len(contracts)} method contracts, {len(fams)} model families, "
        f"{len(questions)} questions whose options are labeled outside the registry, "
        f"{len(catalog.GAPS)} methods v2 offers that have no contract yet, and the "
        f"{len(catalog.NOT_METHODS)} decisions that record no method.", "",
        "## How to read an entry", "",
        "- **Slot**: where the method runs. Ingest, repairs, reshape and eligibility come before "
        "the seal; in-fold steps are fit on training rows only; then the model and its "
        "evaluation.",
        "- **Data scope**: what the method may learn from. Lockbox constitution §06 decides it: "
        "does row *i*'s output depend on other rows, or on the outcome? Where an acceptance "
        "test perturbs a method's rows or outcome, it holds the declared scope to the scope it "
        "observes (`contracts.observed_scope`); not every contract has such a test yet.",
        "- **Options**: each option carries two independent labels (North star 5): *customary* "
        "(where the field uses it, with a source) and *sound* for each purpose (with the reason). "
        "The rank is the registry's order for that purpose, soundest first. Where the app ranks "
        "the options again on the data in front of it (the outcome's event share, the number of "
        "units, time order, the data's kind, what can run on the table, a failed check), the entry "
        "says what it offers first under each condition, computed by the app's own ranking code "
        "(`turbotab/core/reference/defaults.py`).",
        "- **Asked**: a method no Router question, card or control asks says so; from the app it "
        "is never applied, whatever its rank.",
        "- **Rungs** (BLUEPRINT §11.3, the leash): " + "; ".join(
            f"*{v}* (`{k}`)" for k, v in RUNG_WORDS.items()) + ".",
        "- **Relations** (BLUEPRINT §13): " + "; ".join(
            f"*{k}*: {v}" for k, v in RELATION_WORDS.items()) + ".",
        "- **Storyboard**: the method's real, labeled steps, which the transform player plays "
        "when the user flips a preview (BLUEPRINT §11.1).", "",
        "## Index", "",
        "| Method | Key | Slot | Scope | Lenses | Recorded by |", "|---|---|---|---|---|---|",
    ]
    for c in contracts:
        lines.append(f"| [{cell(c.label)}](#{anchor(c.label, c.key)}) | `{c.key}` | {c.slot} | "
                     f"{c.scope} | {lens_words(catalog.lenses_of_contract(c.key))} | "
                     f"{code(catalog.recorded_by(c)) or 'not declared'} |")
    lines += ["", "## Method contracts", "",
              "In run order: by slot, then the method's place in it (MODELING_SEQUENCE §1.1).", ""]
    for c in contracts:
        lines += contract_section(c)
    lines += ["## Model families", "",
              "In the shelf's registration order (the omics chain's family last), which breaks "
              "ties in its ranking. A family is a "
              "plug-in that declares what it needs, what it assumes and how it judges a "
              "situation (M1_CONTRACT §7).", ""]
    for f in fams:
        lines += family_section(f)
    lines += ["## Questions labeled outside the contract registry", "",
              "These questions' options carry the customary and sound labels "
              "(`turbotab/core/custom_sound.py`, and the causal lane's in `turbotab/core/causal.py`) "
              "but are not method contracts; the first four are listed again under the gaps, and "
              "the causal lane's estimators are each a contract above.", ""]
    for question, title in questions:
        lines += question_section(question, title)
    lines += ["## Gaps: methods with no contract entry", "",
              "Each method v2 offers (V2_DEFINITION_OF_DONE §2 lists it, or a question the Router "
              "asks records it and the export writes its sentence) for which no contract declares "
              "a slot, a scope, options labeled per purpose, a leash, a storyboard, a sentence and "
              "relations. The code named implements it; the contract is what is missing.", ""]
    lines += gap_rows(catalog.GAPS)
    lines += ["", "## Decisions that record no method", "",
              "Every other decision kind the decision log accepts records a contract's method "
              "(its `decision`, or one it records without naming it: "
              + ", ".join(f"`{k}` for `{v}`" for k, v in catalog.UNDECLARED_DECISIONS.items())
              + ") or a gap above. These record none:", "",
              "| Decision | Why it is not a method |", "|---|---|"]
    lines += [f"| `{k}` | {cell(v)} |" for k, v in catalog.NOT_METHODS.items()]
    lines.append("")
    return "\n".join(lines)


def anchor(label: str, key: str) -> str:
    """The GitHub anchor of a contract's heading, ``{label} (`{key}`)``."""
    text = f"{label} ({key})".lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def write(path: Path = OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when the committed reference differs from the code's")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.is_file() else ""
        if current != text:
            print(f"{args.out} is stale: run {COMMAND}", file=sys.stderr)
            return 1
        print(f"{args.out} is current")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text.encode('utf-8')):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
