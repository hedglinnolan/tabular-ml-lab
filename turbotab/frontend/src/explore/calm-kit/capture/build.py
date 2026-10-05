"""Write the calm kit's fixture (``../fixture.json``) from the engine's captures (not bundled).

    python3 turbotab/frontend/src/explore/calm-kit/capture/build.py --raw /tmp/calm-raw.json

Sources, each named on every preview it supplies (``source``):

* ``calm``: ``capture.py`` (this folder): the previews taken on the state the walk asks each
  question in, the estimand's on the whole plan, the adjustment groups one at a time, the engine's
  sentences for every alternative, and the Strip's per-column numbers.
* ``map``: ``../../methods-map/fixture.json``: the scenario's fits (Table 2, the sensitivity
  analyses) for every exclusion screen and energy model it captured, the plans' SHA-256s, the
  teaching entries and labels, the Model 1 previews.
* ``paper``: ``../../methods-document/fixture.json``: the energy models' previews taken at the
  energy question itself (the residual storyboard is drawn only there).

Every question, option, sentence and number is the engine's. Copy is shortened to the calm
budget's word limits (question 14, one-line why 22, option 16, why 60) without changing what it
says; each shortened string names its engine source in a comment. Nothing here computes a number:
the numbers the engine does not serve (the Strip's) are computed in ``capture.py`` and described in
the fixture's ``derived``.

Two registers (FOUNDATION §2): the card and the canvas speak plain language a researcher from
another field reads at once; an option's technical name rides along as its ``term`` (shown when
the option is pointed at); the manuscript's sentences stay the engine's methods register. Each
plain string sits beside the engine text it restates, and ``same_numbers`` checks that it carries
no number the engine text does not.

The canvas is never empty: each step's ``now`` is "your data now" for the question, drawn in gray
in the layout its options use, from the engine's own views (a ``ref`` to an option's view, drawn
in its before state) or, where noted in ``derived``, from the engine's data.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
EXPLORE = KIT.parent

ENERGY_ORDER = ["standard", "residual", "residual_energy_dropped", "density_multivariate", "density"]


def r4(x: Any) -> Any:
    """Round floats to four significant digits (points and histograms only: drawing precision)."""
    if isinstance(x, float):
        return float(f"{x:.4g}")
    if isinstance(x, list):
        return [r4(v) for v in x]
    if isinstance(x, dict):
        return {k: r4(v) for k, v in x.items()}
    return x


def trim_view(v: dict[str, Any]) -> dict[str, Any]:
    v = dict(v)
    if v["kind"] == "relationship":
        v["points_before"] = r4(v["points_before"])
        v["points_after"] = r4(v["points_after"])
        v["story"] = [{**f, "points": r4(f["points"])} for f in v["story"]]
    elif v["kind"] == "distribution":
        v["before"] = r4(v["before"])
        v["after"] = r4(v["after"])
        v["story"] = [{**f, "hist": r4(f["hist"])} for f in v["story"]]
    return v


def preview(p: dict[str, Any] | None, source: str) -> dict[str, Any]:
    """A captured preview as the kit draws it: the engine's views, basis and note, or its refusal."""
    if p is None:
        return {"source": source, "basis": "", "note": None, "views": []}
    result = p.get("result") or (p.get("body") if p.get("status") == 200 else None)
    refusal = p.get("refusal")
    if refusal is None and p.get("body") and p.get("status") not in (None, 200):
        refusal = p["body"].get("error")
    if result is None:
        return {"source": source, "basis": "", "note": None, "views": [],
                "refusal": {"code": refusal["code"], "message": refusal["message"]} if refusal else None}
    return {"source": source, "basis": result["basis"], "note": result.get("note"),
            "views": [trim_view(v) for v in result["views"]]}


def words(s: str) -> int:
    return len(s.split())


_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers(s: str) -> set[str]:
    """The numbers in a string, thousands separators dropped (2,001 and 2001 are one number)."""
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(s)}


def same_numbers(plain: str, *engine: str) -> str:
    """A plain restatement carries no number its engine text does not (two registers, one fact)."""
    extra = numbers(plain) - set().union(*(numbers(e) for e in engine))
    assert not extra, f"{plain!r} adds {sorted(extra)} to {engine!r}"
    return plain


def opt(id_: str, name: str, what: str, sentence: str | None, pv: dict[str, Any], *,
        label: str | None = None, disabled: bool = False, refusal: str | None = None,
        term: str | None = None, engine: tuple[str, ...] = ()) -> dict[str, Any]:
    """An option in the card's plain register. ``term`` is its technical name (the quiet second
    register); ``engine`` is the engine text its name and line restate, for the numbers check."""
    if engine:
        same_numbers(name, *engine)
        same_numbers(what, *engine)
    out: dict[str, Any] = {"id": id_, "name": name, "what": what, "sentence": sentence, "preview": pv}
    if label:
        out["label"] = label
    if term:
        out["term"] = term
    if disabled:
        out["disabled"] = True
    if refusal:
        out["refusal"] = refusal
    return out


def step(id_: str, stage: str, section: str, head: str, question: str, lede: str, why: str,
         legend: str, options: list[dict[str, Any]], scenario: str, *, slot: str | None = None,
         requires: dict[str, list[str]] | None = None, engine: tuple[str, ...] = ()) -> dict[str, Any]:
    """A question in the card's plain register; ``engine`` is the engine text its question and lede
    restate, for the numbers check (the why stays the engine's teaching)."""
    if engine:
        same_numbers(question, *engine)
        same_numbers(lede, *engine)
    s = {"id": id_, "stage": stage, "section": section, "head": head, "question": question,
         "lede": lede, "why": why, "legend": legend, "options": options, "scenario": scenario,
         "slot": slot or id_}
    if requires:
        s["requires"] = requires
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    args = ap.parse_args()
    raw = json.loads(Path(args.raw).read_text())
    MAP = json.loads((EXPLORE / "methods-map" / "fixture.json").read_text())
    DOC = json.loads((EXPLORE / "methods-document" / "fixture.json").read_text())
    I = MAP["inference"]
    T = MAP["teaching"]
    M, B, D = raw["main"], raw["branch"], raw["direct"]
    P = M["previews"]
    SEN = raw["in_process"]["sentences"]
    said = raw["in_process"]["server_said"]

    # The in-process sentences for the scenario's own answers are the server's.
    server = {k.split(":")[0]: v for k, v in said.items()}
    assert SEN["unit"]["kcal_1"] == server["set_column_unit"]
    assert SEN["sensitivity"]["both"] == server["set_sensitivity"]
    assert SEN["estimand"]["exposure:sugar"] == server["set_estimand"]
    for k, v in said.items():
        if k.startswith("confirm_reading:") and "bp_di" in v:
            assert v == SEN["single"]["bp_di"]["guess"]

    def doc_preview(key: str) -> dict[str, Any] | None:
        ref = DOC["moments"]["energy"]["previews"].get(key)
        return DOC["pool"][ref] if ref else None

    # ── Data ─────────────────────────────────────────────────────────────────
    unit_ask = M["asks"]["unit"]
    U = P["unit"]
    steps: list[dict[str, Any]] = []
    unit_ev = unit_ask["groups"][0]["evidence"]
    unit_cap = {k: preview(v, "calm")["views"][0]["caption"] for k, v in (("kcal_1", U["kcal_1"]), ("kcal_2", U["kcal_2"]),
                                                                             ("kj_1", B["unit_kj"]))}
    steps.append(step(
        "unit", "data", "measurement", "Energy's unit",
        # ask card: "`kcal`: kcal, one day's intake?"
        "Is each `kcal` value one day's intake, counted in kcal?",
        # ask evidence: "its median, 1,950, is an adult's day's intake (a child's total over two days
        # sits there too)"
        "Its median, 1,950, fits one adult's day, but a child's two-day total could look the same.",
        unit_ev + " Read by the screens.",
        "Choose the unit of kcal",
        [
            # each line: the option's preview caption ("The 500–5,000 kcal-a-day screen is … : `111`
            # of `5,000` rows fall outside"), on the preview's sample
            opt("kcal_1", "One day, in kcal",
                "The 500–5,000 kcal-a-day range check applies as is: 111 of 5,000 sampled rows fall outside.",
                SEN["unit"]["kcal_1"], preview(U["kcal_1"], "calm"), label="Recommended",
                engine=(unit_cap["kcal_1"],)),
            opt("kcal_2", "Two days added together, in kcal",
                "The range check becomes 1,000–10,000 kcal over 2 days: 459 of 5,000 sampled rows fall outside.",
                SEN["unit"]["kcal_2"], preview(U["kcal_2"], "calm"), engine=(unit_cap["kcal_2"],)),
            opt("kj_1", "One day, in kilojoules",
                "The range check becomes 2,092–20,920 kJ a day: 2,798 of 5,000 sampled rows fall outside.",
                SEN["unit"]["kj_1"], preview(B["unit_kj"], "calm"), engine=(unit_cap["kj_1"],)),
        ], "kcal_1", engine=(unit_ev,)))

    # ── Participants ─────────────────────────────────────────────────────────
    ex_labels = I["exclusions"]["labels"]
    EX = P["exclusions"]
    ex_sent = I["exclusions"]["sentences"]
    gold = EX["goldberg_schofield"]["refusal"]
    ex_opts = ex_labels["options"]
    ex_cap = {k: " ".join(v["caption"] for v in preview(EX[k], "calm")["views"]) for k in EX}
    steps.append(step(
        "exclusions", "participants", "participants", "Eligibility",
        # calm-screen.html (approved), from the screens the engine offers
        "Should people with implausible energy intakes be removed?",
        # a 24-hour recall far outside a normal day's intake (the screens' rationale)
        "A day's reported intake far outside the normal range is usually a reporting error.",
        # labels.tension and the teaching entry's why, shortened
        "Fixed kcal cut-offs are the field's habit (Willett, Nutritional Epidemiology, 2013). They are "
        "not individualized, so the every-row analysis is reported beside a screen (Banna et al. 2017, "
        "Front Nutr 4:45). Under-reporting concentrates in higher BMI, so many analyses keep everyone.",
        "Choose who stays in the analysis",
        [
            # names: what each screen keeps (labels.options[*].customary), plain; the source's name
            # is its term
            opt("none", "Keep everyone",
                "No one is removed; the record states that no exclusion was applied.",
                ex_sent["none"], preview(EX["none"], "calm"), label="Recommended",
                engine=(ex_cap["none"],)),
            opt("willett_2013_by_sex", "Ranges by sex, men up to 4,000",
                "Removes women outside 500–3,500 and men outside 800–4,000 kcal a day: 1,614 people.",
                ex_sent["willett_2013_by_sex"], preview(EX["willett_2013_by_sex"], "calm"),
                label="Common practice", term="Willett's cut-offs (Willett 2013)",
                engine=(ex_opts["willett_2013_by_sex"]["customary"], ex_cap["willett_2013_by_sex"])),
            opt("nhs_hpfs_by_sex", "Ranges by sex, men up to 4,200",
                "Removes women outside 500–3,500 and men outside 800–4,200 kcal a day: 1,419 people.",
                ex_sent["nhs_hpfs_by_sex"], preview(EX["nhs_hpfs_by_sex"], "calm"),
                term="NHS/HPFS cut-offs",
                engine=(ex_opts["nhs_hpfs_by_sex"]["customary"], ex_cap["nhs_hpfs_by_sex"])),
            opt("sex_neutral_500_5000", "One wide range for everyone",
                "Removes anyone outside 500–5,000 kcal a day, whatever their sex: 501 people.",
                ex_sent["sex_neutral_500_5000"], preview(EX["sex_neutral_500_5000"], "calm"),
                term="sex-neutral cut-offs, 500–5,000 kcal", engine=(ex_cap["sex_neutral_500_5000"],)),
            opt("sex_neutral_500_3500", "One narrow range for everyone",
                "Removes anyone outside 500–3,500 kcal a day; stricter, so 2,094 people.",
                ex_sent["sex_neutral_500_3500"], preview(EX["sex_neutral_500_3500"], "calm"),
                term="sex-neutral cut-offs, 500–3,500 kcal", engine=(ex_cap["sex_neutral_500_3500"],)),
            # labels.options.goldberg_schofield.customary: "Goldberg's cut-off, energy against predicted needs"
            opt("goldberg_schofield", "Compare intake with energy needs",
                "Needs the units of age and weight confirmed first.",
                None, preview(EX["goldberg_schofield"], "calm"), label="Not available yet",
                disabled=True, refusal=gold["message"], term="Goldberg's cut-off, energy intake against BMR"),
        ], "none", requires={"unit": ["kcal_1"]}))

    SV = P["sensitivity"]
    sv_cap = {k: " ".join(v["caption"] for v in preview(SV[k], "calm")["views"]) for k in SV}
    steps.append(step(
        "sensitivity", "participants", "participants", "Every row, beside",
        # "Which screens should be reported beside the every-row analysis?"
        "Which removal rules should be reported beside the main analysis, as checks?",
        # "Each is the same model on its own rows, shown only after the plan is locked."
        "Each check reruns the same model without the people its rule removes; results appear after the plan is locked.",
        ex_labels["tension"],
        "Choose the checks reported beside",
        [
            opt("both", "Both rules by sex",
                "The same model on the 20,235 and the 20,430 people they keep.",
                SEN["sensitivity"]["both"], preview(SV["both"], "calm"),
                term="sensitivity analyses: Willett's and NHS/HPFS cut-offs", engine=(sv_cap["both"],)),
            opt("willett", "Ranges by sex, men up to 4,000",
                "The same model on the 20,235 people this rule keeps.",
                SEN["sensitivity"]["willett"], preview(SV["willett"], "calm"),
                term="sensitivity analysis: Willett's cut-offs",
                engine=(sv_cap["willett"], ex_opts["willett_2013_by_sex"]["customary"])),
            opt("nhs", "Ranges by sex, men up to 4,200",
                "The same model on the 20,430 people this rule keeps.",
                SEN["sensitivity"]["nhs"], preview(SV["nhs"], "calm"),
                term="sensitivity analysis: NHS/HPFS cut-offs",
                engine=(sv_cap["nhs"], ex_opts["nhs_hpfs_by_sex"]["customary"])),
            # "No sensitivity analysis is set beside the primary analysis."
            opt("none", "No checks", "Only the main analysis is reported.",
                SEN["sensitivity"]["none"], preview(SV["none"], "calm"), term="no sensitivity analysis"),
        ], "both", requires={"unit": ["kcal_1"]}))

    mi_labels = I["missing"]["labels"]
    MS = P["missing"]
    steps.append(step(
        "missing", "participants", "statistics", "Missing data",
        # teaching.missing.question: "How should rows with missing predictor values be handled?"
        "What should happen to people missing a value the model needs?",
        # labels.tension: complete cases are the field's habit; when the blanks depend on measured
        # variables, multiple imputation is sounder
        "Dropping incomplete rows is the field's habit; filling blanks in is sounder when who has blanks depends on measured columns.",
        T["missing"]["why"],
        "Choose how missing values are handled",
        [
            opt("complete_case", "Drop incomplete rows",
                "Anyone missing a model value is dropped; the participant flow shows how many.",
                I["missing"]["sentence"], preview(MS["complete_case"], "calm"), label="Common practice",
                term="complete-case analysis"),
            # "Each blank imputed 20 times with the outcome; results pooled by Rubin's rules."
            opt("multiple_imputation", "Fill in each blank, many times",
                "Each blank is filled in 20 times from the other columns, outcome included; results are combined.",
                None, preview(MS["multiple_imputation"], "calm"), label="Recommended",
                term="multiple imputation, pooled by Rubin's rules",
                engine=(mi_labels["options"]["multiple_imputation"]["label"],)),
        ], "complete_case"))

    # ── Exposure: the role readings, then the estimand ───────────────────────
    roles_ask = M["asks"]["roles"]
    by_col = {g["columns"][0]: g for g in roles_ask["groups"] if g["kind"] == "role"}
    items = {it["column"]: it for it in I["readings"]["items"] if it["reading"] == "role"}
    roles_t = {o["value"]: o for o in T["roles"]["options"]}
    for n, col in enumerate(["bp_di", "bp_sys", "cycle_begin_year"]):
        it = items[col]
        # the evidence, plain: "Named like a time, but no unit repeats here, so it orders nothing:
        # kept as a predictor until you say otherwise."
        lede = it["evidence"].replace("no unit repeats here", "no one appears twice").replace(
            ": kept as a predictor until you say otherwise", ": a predictor unless you say otherwise")
        steps.append(step(
            f"single:{col}", "exposure", "measurement", "Readings",
            # ask card: "`bp_di`: a covariate?" (a covariate: a person's characteristic, adjusted for)
            f"Is `{col}` a characteristic the models can adjust for?",
            lede,
            T["roles"]["why"] + " Proposed below high confidence, so it waits for your confirmation.",
            f"Choose whether the models can use {col}",
            [
                # roles.covariate.consequence: "Adjusted for, such as age or sex; enters the models
                # beside the exposures."
                opt("covariate", "Yes, a characteristic to adjust for",
                    "Adjusted for, like age or sex; it enters the models beside the nutrients.",
                    SEN["single"][col]["guess"], preview(P["single"][col]["guess"], "calm"),
                    label="Recommended", term="covariate"),
                # roles.excluded.consequence: "Left out of the analysis; the record keeps the column
                # and the reason."
                opt("excluded", "No, leave it out of the models",
                    "Left out of the analysis; the record keeps the column and why.",
                    SEN["single"][col]["leave_out"], preview(P["single"][col]["leave_out"], "calm"),
                    term="excluded column"),
            ], "covariate", slot=f"single:{col}"))
        assert by_col[col]["guess"] == "covariate"
        assert roles_t["covariate"]["consequence"].startswith("Adjusted for, such as age or sex")

    block_ask = M["asks"]["block"]
    block = block_ask["exits"][0]
    block_pv = preview(P["block"], "calm")
    rows_pv = preview(B["rows_after_block"], "calm")
    # the block's own lineage, then the rows complete cases keep on the state it leaves
    block_pv["views"] = [*rows_pv["views"][:1], *block_pv["views"]]
    block_pv["sources"] = {"0": "row flow: set_missing complete_case previewed on the state the block "
                                "leaves (the estimand question, after the block)",
                           "1": "the block's own preview"}
    n_block = len(block["decision"]["items"])
    kinds = [i["value"] for i in block["decision"]["items"]]
    n_cov, n_flag = kinds.count("covariate"), kinds.count("flag")
    assert (n_cov, n_flag) == (7, 6), kinds  # the line below says seven and six
    steps.append(step(
        "block", "exposure", "measurement", "Readings",
        # "Are the other 13 readings right as shown?"
        f"Are the roles guessed for the other {n_block} columns right?",
        # "Each was proposed below high confidence; the covariates enter the models once confirmed."
        "Each was guessed below high confidence; the characteristics enter the models once confirmed.",
        # teaching one_liner, and the stated roles sentence's "proposed below high confidence and
        # wait for their own confirmation before any default reads them"
        T["roles"]["one_liner"] + " Each was proposed below high confidence, so no default reads it "
        "until it is confirmed.",
        f"Confirm the other {n_block} roles",
        # block.label: "Confirm each of the 13 as shown"; the items: seven covariates and six flags
        # (imputed_*: which values were filled in)
        [opt("confirm", f"Yes, confirm all {n_block}",
             "Seven characteristics enter the models; six markers of filled-in values stay out.",
             SEN["block"], block_pv, term="covariates and imputation flags", engine=(block["label"],))],
        "confirm", slot="block"))

    E = I["estimand"]  # the card as the map captured it, with each exposure's evidence
    assert [e["column"] for e in M["cards"]["estimand"]["exposures"]] == [e["column"] for e in E["exposures"]]
    EP = P["estimand"]
    combo = SEN["estimand_combo"]
    exposures = [e for e in E["exposures"] if e.get("energy_contrast")]
    exposures.sort(key=lambda e: e["column"] != "sugar")  # the scenario's first; the rest as offered
    est_t = T["estimand"]
    outcome = I["stated"]["target"]
    assert "`glucose`" in outcome, outcome  # the plain copy below names the outcome
    steps.append(step(
        "exposure", "exposure", "variables", "Exposure and estimand",
        # "Which nutrient is the exposure?"
        "Which nutrient's effect on `glucose` do you want to estimate?",
        # teaching.estimand.one_liner: "Inference reports one declared effect; no estimate is shown
        # until it is named."
        "The analysis estimates one effect you name in advance; no estimate appears before that.",
        est_t["why"],
        "Choose the nutrient",
        # each nutrient's evidence: "A nutrient that carries energy: an exposure; it rises with total
        # energy (r = 0.67)."
        [opt(e["column"], f"`{e['column']}`",
             e["evidence"].replace("A nutrient that carries energy: an exposure; it rises with total energy",
                                   "Carries calories; it rises with total calories, `kcal`"),
             None, preview(EP[f"exposure:{e['column']}"], "calm"), engine=(e["evidence"],))
         for e in exposures],
        "sugar", slot="estimand"))
    for o in steps[-1]["options"]:
        assert o["what"].startswith("Carries calories;"), o["what"]

    eff = {e["effect"]: e for e in E["effects"]}
    con = {c["contrast"]: c for c in E["contrasts"]}
    adj_groups = {g["key"]: g for g in B["groups"]}
    med_key = next(k for k, g in adj_groups.items() if g["scenario"] == ["no", "yes", "yes"])
    total_route = preview(B["previews"][med_key]["scenario"], "calm")
    direct_route = preview(D["mediators"], "calm")
    total_rows = preview(B["rows_after_adjustment"], "calm")
    direct_rows = preview(P["direct_missing"], "calm")
    # "Your data now" at the effect question is the plan with the total effect (the scenario's):
    # the direct effect's captures were taken with it already recorded, so their "before" is the
    # total effect's capture of the same view on the same plan (fixture `derived.effect_angles`).
    direct_route["views"][0]["before"] = total_route["views"][0]["after"]
    direct_rows["views"][0]["before"] = total_rows["views"][0]["after"]
    mediators = adj_groups[med_key]["columns"]

    def kept(rows: dict) -> int:
        return rows["views"][0]["after"][-1]["n"]

    # Angles (FOUNDATION §5): each panel one question and one picture; the option table's columns are
    # the panels' short heads (with the option's own, at most three).
    MEDIATORS = {"question": "Are the mediators held fixed?", "head": "Mediators"}
    WHO = {"question": "Who stays in the analysis?", "head": "People"}

    def angles_pv(route: dict, rows: dict, cells: list[str], touch: list[str], basis: str) -> dict[str, Any]:
        return {"source": "calm", "basis": basis, "note": None,
                "views": [route["views"][0], rows["views"][0]],
                "angles": [{**MEDIATORS, "view": 0, "cell": cells[0], "touch": touch},
                           {**WHO, "view": 1, "cell": cells[1]}]}

    whole = "Drawn on the scenario's whole plan before the lock; at its own question the engine cannot draw the estimand yet."
    n_total, n_direct = kept(total_rows), kept(direct_rows)
    n_blank = direct_rows["views"][0]["after"][-1]["dropped"]
    assert (n_total, n_direct, n_blank) == (21849, 2996, 18853), (n_total, n_direct, n_blank)
    # which columns are blank: the complete-case coach on the same rows
    blank_coach = preview(MS["complete_case"], "calm")["views"][0]["coach"][0]["text"]
    assert blank_coach == f"`meds_chol` and `meds_hbp` are blank on `{n_blank:,}` of these rows.", blank_coach
    steps.append(step(
        "effect", "exposure", "variables", "Exposure and estimand",
        # "Its total effect, or its direct effect?"
        "Count all of `sugar`'s effect on `glucose`, or only its direct part?",
        # "A direct effect holds the mediators fixed; a total effect counts everything downstream."
        # The term kept, defined in place.
        "Mediators are what `sugar` changes that in turn changes `glucose`; the direct part holds them fixed.",
        est_t["why"],
        "Choose how much of sugar's effect counts",
        [
            # effects.total.consequence: "Everything the exposure changes downstream counts; mediators
            # stay out."
            opt("total", "All of its effect", "Every path counts; the mediators stay out of the model.", None,
                angles_pv(total_route, total_rows, ["Left out", f"{n_total:,}"], mediators, whole),
                term="total effect"),
            # effects.direct.consequence: "Holds the mediators fixed; their confounders must be adjusted
            # too." (a mediator's confounders: what causes both it and the outcome)
            opt("direct", "Only its direct part",
                "Holds the mediators fixed; what causes both them and `glucose` must be adjusted too.", None,
                angles_pv(direct_route, direct_rows, ["Held fixed", f"{n_direct:,}"], mediators, whole),
                term="direct effect"),
        ], "total", slot="estimand"))
    eff_caption = {
        # the angles' caption, from their two views (the lineage's mediators, the flow's rows)
        "total": f"The {len(mediators)} mediators stay out of the model; all {n_total:,} people stay.",
        "direct": f"The {len(mediators)} mediators are held fixed; {n_direct:,} people stay, as "
                  f"`meds_chol` and `meds_hbp` are blank for {n_blank:,}.",
    }
    for o in steps[-1]["options"]:
        o["preview"]["caption"] = eff_caption[o["id"]]

    std_doc = preview(doc_preview("standard"), "paper")
    std_lineage = next(v for v in std_doc["views"] if v["kind"] == "lineage")
    add = P["addition_energy"]
    add_exit = preview(add["partition:exit0"], "calm")
    add_lineage = next(v for v in add_exit["views"] if v["kind"] == "lineage")
    exit_label = add["partition"]["refusal"]["exits"][0]["label"]
    applic = I["energy"]["applicability"]
    en_labels = I["energy"]["labels"]
    en_t = {o["value"]: o for o in T["energy_adjustment"]["options"]}
    runnable = [m for m in I["energy"]["ranking"]["order"] if applic[m]["ok"] and m != "none"]

    # The energy adjustments in the card's register: (name, one line). Each line restates the
    # teaching entry's consequence (en_t[m]["consequence"], quoted); the engine's label is the term.
    EN: dict[str, tuple[str, str]] = {
        # "Energy stays in the model: more of the nutrient in place of other calories."
        "standard": ("Keep total calories in the model",
                     "Total calories stay in the model: more of the nutrient in place of other calories."),
        # "Each nutrient's residual on energy, energy kept: the standard model's swap, in its units."
        "residual": ("Calorie-adjusted, total calories kept",
                     "Each nutrient's part calories don't explain, total kept: the same swap as above, in nutrient units."),
        # "Nutrient per calorie, energy its own term: obscure, and still biased (Tomova 2022)."
        "density_multivariate": ("Per calorie, total calories kept",
                                 "Each nutrient per calorie, total calories kept too: hard to interpret, and still biased (Tomova 2022)."),
        # "Each nutrient's residual on energy, energy dropped: differs when covariates track energy."
        "residual_energy_dropped": ("Calorie-adjusted, total calories dropped",
                                    "Each nutrient's part calories don't explain, total dropped: differs when adjusted characteristics track calories."),
        # "Nutrient per calorie, energy dropped: a rescaled effect whose meaning is obscure."
        "density": ("Per calorie, total calories dropped",
                    "Each nutrient per calorie, total calories dropped: a rescaled effect that is hard to interpret."),
        # "Total energy leaves the model: absolute intake, mixed with how much people eat."
        "none": ("Ignore total calories", "Total calories leave the model: absolute intake, mixed with how much people eat."),
        # "Every energy source its own term: added calories, and each one's average swap."
        "all_components": ("Each calorie source its own term",
                           "Each calorie source its own term in the model: added calories, and each one's average swap."),
        # "Calories from the nutrient and from everything else: adding calories, not substituting."
        "partition": ("Split calories by source",
                      "Calories from the nutrient and from everything else: adds calories rather than swapping them."),
    }
    # The refusals, plain. The engine's: "`sugar`, `fat_sat`, `fat_mon` and `fat_poly` are parts of
    # `carb` and `fat_total`, so the all-components model (a partition) would count carbohydrate's
    # and fat's energy twice"; "The estimand is a substitution of `sugar`'s calories, and the no
    # energy adjustment estimates no substitution (Tomova et al. 2022)."
    nested = "`sugar`, `fat_sat`, `fat_mon` and `fat_poly` are parts of `carb` and `fat_total`"
    EN_REFUSAL = {
        "all_components": f"{nested}, so giving each source its own term would count carb's and fat's calories twice.",
        "partition": f"{nested}, so splitting calories by source would count carb's and fat's calories twice.",
        "none": "You asked for `sugar`'s calories in place of other calories; ignoring total calories cannot "
                "estimate that swap (Tomova et al. 2022).",
    }
    for m in ("all_components", "partition"):
        assert add[m]["refusal"]["message"].startswith(nested + ", so ") and \
            add[m]["refusal"]["message"].endswith("would count carbohydrate's and fat's energy twice."), m

    sub_caption = EP["exposure:sugar"]["result"]["views"][0]["caption"]
    add_caption = EP["contrast:addition"]["result"]["views"][0]["caption"]
    assert sub_caption.endswith("in place of other calories on `glucose`: difference in the mean outcome."), sub_caption
    HELD = {"question": "What does the model hold fixed?", "head": "Held fixed"}
    FOLLOW = {"question": "Which energy adjustments can follow?", "head": "Can follow"}
    # the addition's way out: "Partition `protein`, `carb` and `fat_total`"
    assert exit_label == "Partition `protein`, `carb` and `fat_total`", exit_label
    exit_plain = "Split calories by `protein`, `carb` and `fat_total`"
    steps.append(step(
        "contrast", "exposure", "variables", "Exposure and estimand",
        # "Is `sugar`'s effect a substitution or an addition of calories?"
        "Should `sugar`'s calories replace other calories, or add to them?",
        # card.which_contrast: "`sugar` carries energy and total energy is in the model … The two are
        # different estimands (Tomova et al. 2022)."
        "`sugar` has calories and total calories are in the model, so the two answer different questions (Tomova et al. 2022).",
        E["which_contrast"],
        "Choose how sugar's calories count",
        [
            # contrasts.substitution.consequence: "More of it in place of other calories, total energy
            # held fixed."
            opt("substitution", "Replace other calories",
                "More `sugar` in place of other calories; total calories stay the same.", None,
                {"source": "calm+paper", "basis": std_doc["basis"], "note": None,
                 "views": [std_lineage],
                 "angles": [
                     {**HELD, "view": 0, "cell": "Total calories", "touch": ["kcal"],
                      "touch_source": applic["standard"]["reason"]},
                     {**FOLLOW, "list": [{"name": EN[m][0], "ok": True} for m in runnable] +
                                        [{"name": EN[m][0], "ok": False, "why": EN_REFUSAL[m]}
                                         for m in ("all_components", "partition")],
                      "cell": f"{len(runnable)} adjustments"},
                 ]}, term="substitution"),
            # contrasts.addition.consequence: "Its calories added on top, every other source held fixed."
            opt("addition", "Add on top",
                "`sugar`'s calories added on top; calories from every other source stay the same.", None,
                {"source": "calm", "basis": add_exit["basis"], "note": None,
                 "views": [add_lineage],
                 "angles": [
                     {**HELD, "view": 0, "cell": "Other sources' calories"},
                     {**FOLLOW, "list": [{"name": EN[m][0], "ok": False, "why": EN_REFUSAL[m]}
                                         for m in ("all_components", "partition")]
                                        + [{"name": exit_plain, "ok": True,
                                            "why": "The engine's way out: `sugar` has no term of its own in it."}],
                      "cell": "1 adjustment"},
                 ]}, term="addition"),
        ], "substitution", slot="estimand"))
    assert len(runnable) == 5, runnable
    assert any(n["lane"] == "matrix" and n.get("column") == "kcal" for n in std_lineage["after"]["nodes"])
    m_std = re.fullmatch(r"The model matrix keeps the same (\d+) columns\.", std_lineage["caption"])
    m_add = re.fullmatch(r"(.+ leave; .+ arrive)\. The model sees (\d+) columns\.", add_lineage["caption"])
    assert m_std and m_add, (std_lineage["caption"], add_lineage["caption"])
    con_caption = {
        "substitution": f"The model keeps the same {m_std[1]} columns, total calories (`kcal`) among them.",
        "addition": f"{m_add[1]}: {m_add[2]} columns in all.",
    }
    for o in steps[-1]["options"]:
        o["preview"]["caption"] = con_caption[o["id"]]

    # ── Confounders: the adjustment card, one group at a time ────────────────
    A = I["adjustment"]
    derive = A["derive"]
    role_order = ["confounder", "timing_unknown", "mediator", "not_a_cause"]
    triple = {"confounder": "yes,yes,no", "timing_unknown": "unknown,yes,unknown",
              "mediator": "no,yes,yes", "not_a_cause": "no,no,no"}
    adj_t = {o["value"]: o for o in T["adjustment"]["options"]}
    # The roles in the card's register: (plural name, singular name, one line, term). The lines
    # restate the teaching entry's consequences (adj_t[role]["consequence"], quoted).
    ROLE = {
        # "A cause of the exposure and the outcome: adjusted for."
        "confounder": ("Causes of both: adjust for them", "A cause of both: adjust for it",
                       "A cause of both `sugar` intake and `glucose`: adjusted for.",
                       "confounder (disjunctive cause criterion)"),
        # "Estimated without it and, declared beside, with it."
        "timing_unknown": ("Unclear timing: add them in Model 3", "Unclear timing: add it in Model 3",
                           "The main estimate leaves it out; Model 3, reported beside, adds it.",
                           "covariate of unknown timing, in a secondary model"),
        # "Changed by the exposure and a cause of the outcome: left out of a total effect."
        "mediator": ("On `sugar`'s path: leave them out", "On `sugar`'s path: leave it out",
                     "`sugar` changes it and it changes `glucose`: left out, so all of `sugar`'s effect counts.",
                     "mediator"),
        # "A cause of neither the exposure nor the outcome: the criterion leaves it out."
        "not_a_cause": ("Causes neither: leave them out", "Causes neither: leave it out",
                        "Causes neither `sugar` intake nor `glucose`: the rule leaves it out.",
                        "cause of neither (disjunctive cause criterion)"),
    }
    assert [adj_t[r]["consequence"] for r in ("confounder", "mediator")] == [
        "A cause of the exposure and the outcome: adjusted for.",
        "Changed by the exposure and a cause of the outcome: left out of a total effect."], adj_t
    # The guessed groups' ledes, plain, each from the card's reason (its first clause).
    LEDE = {
        # "Set before the diet was measured, and the field's Models 1 and 2 adjust for them as confounders."
        "demographic": "Set before the diet was measured; the field's Models 1 and 2 adjust for them as causes of both.",
        # "They share the diet's common causes with the exposure, so they default to confounders, never
        # to not relevant (the field's Model 4)."
        "dietary": "They share the diet's causes with `sugar`, so they start as causes of both, never as causing "
                   "neither (the field's Model 4).",
        # "The diet may have changed them, so the field's Model 3 adds them beside the primary."
        "body": "The diet may have changed them, so the field's Model 3 adds them beside the main model.",
    }
    card_groups = {g["key"]: g for g in A["groups"]}
    for g in B["groups"]:
        cols = g["columns"]
        scen_role = derive[",".join(g["scenario"])]["role"]
        group = card_groups[g["group"]]
        guessed = group.get("derived")
        listed = " and ".join(f"`{c}`" for c in cols) if len(cols) <= 2 else \
            ", ".join(f"`{c}`" for c in cols[:2]) + f" and {len(cols) - 2} more"
        options = []
        for role in role_order:
            pv = B["previews"][g["key"]]["scenario"] if role == scen_role else B["previews"][g["key"]]["options"][role]
            sentence = SEN["adjustment"][g["key"]]["scenario" if role == scen_role else role]
            plural, single, what, term = ROLE[role]
            options.append(opt(role, plural if len(cols) > 1 else single, what, sentence, preview(pv, "calm"),
                               label="Recommended" if guessed == role else None, term=term,
                               engine=(adj_t[role]["consequence"] if role in adj_t else derive[triple[role]]["why"],
                                       "Model 3")))
        if guessed:
            options.sort(key=lambda o: o["id"] != guessed)
        reason = group["reason"]
        reason = re.sub(r" \(NUTRITION_PACK §08 \(the nested model table:.*\)\)\.$", ".", reason)
        reason = reason.split("; ")[0].rstrip(".") + "."  # its first clause
        if group.get("guess"):
            key = g["group"]
            lede = LEDE[key]
            same_numbers(lede, reason)
        else:
            # the card's own reason: "The pack says nothing about these; each is asked."
            lede = "No built-in guess covers these, so each is asked." if len(cols) > 1 else \
                "No built-in guess covers this column, so it is asked."
        steps.append(step(
            f"adjust:{g['key']}", "confounders", "variables", "Adjustment set",
            # "What are `age` and `gender` to the effect of `sugar`?"
            f"How {'do' if len(cols) > 1 else 'does'} {listed} relate to `sugar` and `glucose`?",
            lede,
            T["adjustment"]["why"],
            f"Choose how {', '.join(cols)} relate to sugar and glucose",
            options, scen_role, slot=f"adjust:{g['key']}",
            requires={"exposure": ["sugar"], "effect": ["total"]}))

    # ── Energy ───────────────────────────────────────────────────────────────
    strip = raw["strip"]
    energy_opts = []
    for m in I["energy"]["ranking"]["order"]:
        ok = applic[m]["ok"] and m != "none"
        dp = doc_preview(m)
        name, what = EN[m]
        term = en_labels["options"][m]["label"]
        term = term if term.startswith("Willett") else term[0].lower() + term[1:]  # read as "Known as …"
        if ok:
            pv = preview(dp, "paper")
            if m == "residual_energy_dropped":
                # The leash (FOUNDATION §5 rule 6): the engine's caption here quotes the nutrient's
                # coefficient on the outcome, an outcome-model estimate, before the plan is locked
                # (models/previews.py _residual_gap, audit ME-03). The kit shows the engine's own
                # caption for this method without that gap (_relationship_caption's other branch),
                # on the view's own r values.
                rel = next(v for v in pv["views"] if v["kind"] == "relationship")
                assert "coefficient" in rel["caption"], rel["caption"]
                r0, r1 = rel["r_before"], rel["r_after"]
                fmt_r = lambda r: f"{0.0 if abs(r) < 0.005 else r:.2f}"  # noqa: E731 (models.previews._r)
                rel["caption"] = (f"`fat_total` correlates {fmt_r(r0)} with `kcal`; after residual "
                                  f"adjustment, {fmt_r(r1)}; `kcal` leaves the outcome model.")
                pv["withheld"] = ("The engine's caption quoted the outcome-model coefficient the energy-dropped "
                                  "form gives (+0.395 against +0.397); withheld before the lock.")
            cols = strip["methods"][m]["columns"]
            changed = [c for c in cols if c["changed"]]
            if changed:
                pv["strip"] = [{
                    "column": c["column"], "output": c["output"], "shift": c["shift"],
                    "r_before": c["r_before"], "r_after": c["r_after"],
                    "sd_before": c["sd_before"], "sd_after": c["sd_after"],
                    "mean_before": c["mean_before"], "mean_after": c["mean_after"],
                    "hist_before": r4(c["hist_before"]), "hist_after": r4(c["hist_after"]),
                } for c in sorted(changed, key=lambda c: -c["shift"])]
            energy_opts.append(opt(m, name, what, I["energy"]["sentences"].get(m), pv, term=term,
                                   label=("Recommended" if m == I["energy"]["ranking"]["order"][0] else
                                          "Common practice" if m == en_labels["customary_first"] else None),
                                   engine=(en_t[m]["consequence"],)))
        else:
            ref = dp["body"]["error"]
            assert ref["message"].startswith(nested) if m != "none" else "estimates no substitution" in ref["message"], \
                ref["message"]
            energy_opts.append(opt(m, name, what, None, preview(dp, "paper"), label="Not available", disabled=True,
                                   refusal=EN_REFUSAL[m], term=term, engine=(en_t[m]["consequence"],)))
            same_numbers(EN_REFUSAL[m], ref["message"])
    energy_opts.sort(key=lambda o: bool(o.get("disabled")))
    steps.append(step(
        "energy", "energy", "statistics", "Energy adjustment",
        # teaching.energy_adjustment.question: "How should nutrient intakes be adjusted for total energy?"
        "How should the analysis account for how much people eat overall?",
        # ranking.line: "All components would rank first for substitution questions (Tomova 2022), but
        # it cannot run on these columns; the standard (multivariate) model leads among those that can."
        "The top-ranked method can't run on these columns, so keeping total calories in the model leads.",
        T["energy_adjustment"]["why"],
        "Choose how total calories are handled",
        energy_opts, I["energy"]["ranking"]["order"][0], requires={"contrast": ["substitution"]}))

    # ── Model ────────────────────────────────────────────────────────────────
    m1 = I["model_1"]
    steps.append(step(
        "model1", "model", "statistics", "Model sequence",
        "Which columns should Model 1 adjust for?",
        # "The field's Model 1 adjusts for age, sex and energy; say which columns those are."
        "In this field Model 1 adjusts for age, sex and total calories; which columns are those?",
        m1["reason"],
        "Choose Model 1's adjustment",
        [
            # "The field's Model 1; Model 3 adds the possible mediators set beside it." (the reason:
            # "Model 3 adds the possible mediators your answers set beside it", the unclear-timing ones)
            opt("guess", "`age`, `gender` and `kcal`",
                "The field's usual Model 1; Model 3 later adds the columns of unclear timing.",
                m1["sentences"]["guess"], preview(m1["previews"]["guess"], "map"), label="Recommended",
                engine=(m1["reason"],)),
            # "Model 1 adjusts for nothing; Model 2 adds the whole adjustment set."
            opt("empty", "Nothing: an unadjusted Model 1",
                "Model 1 adjusts for nothing; Model 2 adds every column chosen for adjustment.",
                m1["sentences"]["empty"], preview(m1["previews"]["empty"], "map"), engine=(m1["reason"],)),
        ], "guess", engine=(m1["reason"],)))

    codes_ask = M["asks"]["codes"]
    code_items = {it["column"]: it["evidence"] for it in I["readings"]["items"] if it["reading"] == "code_or_count"}
    ev = {c: re.match(r"`(\d+)` whole-number values from ([\d,]+) to ([\d,]+)", e) for c, e in code_items.items()}
    assert all(ev.values()) and code_items["cycle_begin_year"].endswith("few of them"), code_items
    year = lambda v: v.replace(",", "")  # noqa: E731 (a year, without the thousands separator)
    steps.append(step(
        "codes", "model", "measurement", "Readings",
        # "Do `age` and `cycle_begin_year` hold amounts or codes?"
        "Are `age` and `cycle_begin_year` amounts, or codes for categories?",
        # each reading's evidence: "`age`: `68` whole-number values from 18 to 85; `cycle_begin_year`:
        # `9` whole-number values from 2,001 to 2,017, few of them."
        f"`age` has {ev['age'][1]} whole-number values, {ev['age'][2]} to {ev['age'][3]}; `cycle_begin_year` has "
        f"only {ev['cycle_begin_year'][1]}, from {year(ev['cycle_begin_year'][2])} to {year(ev['cycle_begin_year'][3])}.",
        codes_ask["text"],
        "Confirm amounts and codes",
        # exits[0].label: "Confirm each of the 2 as shown"
        [opt("confirm", "Yes, confirm both as shown",
             "`age` is an amount; `cycle_begin_year` holds codes for categories.",
             SEN["codes"], preview(P["codes"], "calm"), engine=(codes_ask["exits"][0]["label"],))],
        "confirm", engine=tuple(code_items.values())))

    lock_reason = ("Recorded the first time an estimate is displayed; it changes no number, only how "
                   "later records are marked.")
    steps.append(step(
        "lock", "model", "statistics", "The plan, locked",
        "Lock the plan and fit the models?",
        # "No estimate appears before the plan is locked; later changes are marked as seen after."
        "Estimates appear only once the plan is locked; changes after that are marked as made after seeing them.",
        "Choosing an analysis after seeing its estimates is a forking path (Gelman & Loken 2013). "
        "Locking records the plan's SHA-256 before the first estimate is shown; every later change "
        "is marked in the record as made after the estimates were seen.",
        "Lock the plan",
        # "Records the plan's SHA-256, then fits the declared models."
        [opt("lock", "Lock the plan and fit",
             "Records a fingerprint of the plan (its SHA-256), then fits the planned models.",
             None, {"source": "engine", "basis": "", "note": lock_reason, "views": []},
             engine=("SHA-256",))],
        "lock"))

    # ── the canvas in the card's register (two registers, FOUNDATION §2) ──────
    S = {st["id"]: st for st in steps}

    def option(step_id: str, opt_id: str) -> dict[str, Any]:
        return next(o for o in S[step_id]["options"] if o["id"] == opt_id)

    def recaption(step_id: str, opt_id: str, view: int, pattern: str, template: str) -> None:
        """The option's canvas caption: its engine view's caption restated plainly, numbers kept."""
        o = option(step_id, opt_id)
        engine = o["preview"]["views"][view]["caption"]
        m = re.fullmatch(pattern, engine)
        assert m, (step_id, opt_id, engine)
        o["preview"]["caption"] = same_numbers(template.format(*m.groups()), engine)

    for k, prefix in (("kcal_1", "Read as one day in kcal"), ("kcal_2", "Read as two days in kcal"),
                      ("kj_1", "Read as one day in kJ")):
        recaption("unit", k, 0, r"The 500–5,000 kcal-a-day screen is (.+): `([\d,]+)` of `([\d,]+)` rows fall outside\.",
                  prefix + ", the 500–5,000 kcal-a-day range check means {}: {} of {} sampled rows fall outside.")
    for k in ("willett_2013_by_sex", "nhs_hpfs_by_sex", "sex_neutral_500_5000", "sex_neutral_500_3500"):
        recaption("exclusions", k, 0,
                  r"`([\d,]+)` of `([\d,]+)` rows fall outside these ranges and would leave; `([\d,]+)` stay\.",
                  "{} of {} people fall outside these ranges and would leave; {} stay.")
    recaption("exclusions", "none", 0, r"No rule excludes a row: all `([\d,]+)` rows pass\.",
              "No one is removed: all {} people stay.")
    SENS = r"`([\d,]+)` rows, against `([\d,]+)` in the primary analysis\."
    for k in ("willett", "nhs"):
        recaption("sensitivity", k, 0, SENS, "The check runs on {} people, against {} in the main analysis.")
    both = option("sensitivity", "both")["preview"]
    (a, n), (b, n2) = (re.fullmatch(SENS, v["caption"]).groups() for v in both["views"])
    assert n == n2
    both["caption"] = f"The checks run on {a} and {b} people, against {n} in the main analysis."
    recaption("missing", "complete_case", 0, r"No rows miss a predictor, so none would leave\.",
              "No one is missing a model value, so no one would leave.")
    recaption("missing", "multiple_imputation", 0, r"No predictor has a missing value, so nothing is imputed\.",
              "No model column has a blank, so nothing is filled in.")
    for col in ("bp_di", "bp_sys", "cycle_begin_year"):
        recaption(f"single:{col}", "excluded", 0, r"(`\w+`) settled as recorded; the same `(\d+)` predictors\.",
                  "{} stays out of the models; the same {} predictors.")
    recaption("block", "confirm", 0,
              r"`([\d,]+)` of `([\d,]+)` rows miss a predictor and would leave; (`\w+`) is missing most\.",
              "{} of {} people would miss a model value and leave; {} is missing most.")
    recaption("exposure", "sugar", 0,
              r"Total effect of `sugar` in place of other calories on `glucose`: difference in the mean outcome\.",
              "Would estimate all of `sugar`'s effect on mean `glucose`, with `sugar` replacing other calories.")
    R = r"`fat_total` correlates ([\d.]+) with `kcal`; after residual adjustment, ([\d.]+)"
    recaption("energy", "residual", 0, R + r", with `kcal` kept in the model\.",
              "`fat_total` correlates {} with `kcal`; its calorie-adjusted amount, {}; `kcal` stays in the model.")
    recaption("energy", "residual_energy_dropped", 0, R + r"; `kcal` leaves the outcome model\.",
              "`fat_total` correlates {} with `kcal`; its calorie-adjusted amount, {}; `kcal` leaves the model.")
    D = r"`fat_total_per_kcal` correlates ([\d.]+) with `kcal`, down from ([\d.]+); "
    recaption("energy", "density_multivariate", 0, D + r"`kcal` stays as its own term\.",
              "Per calorie, `fat_total` correlates {} with `kcal`, down from {}; `kcal` stays in the model.")
    recaption("energy", "density", 0, D + r"`kcal` leaves the model\.",
              "Per calorie, `fat_total` correlates {} with `kcal`, down from {}; `kcal` leaves the model.")
    recaption("codes", "confirm", 0, r"The confirmation settles `(\d+)` readings of `(\d+)` columns\.",
              "Confirming settles {} readings across {} columns.")

    # Each column's role, in the roles' plain names, and the role captions restated in full from the
    # lineage's own nodes (the engine's are cut at a length, "…").
    ROLE_LABEL = {"confounder": "cause of both", "mediator": "on `sugar`'s path", "timing unknown": "unclear timing",
                  "cause of neither": "causes neither", "not answered": "not answered"}

    def names(cols: list[str], limit: int = 3) -> str:
        """The engine's own listing (core/plan_previews.py names): a, a and b, a, b and c, a, b and 4 more."""
        shown = [f"`{c}`" for c in cols]
        if len(shown) > limit:
            shown = shown[: limit - 1] + [f"{len(shown) - (limit - 1)} more"]
        return "".join(shown) if len(shown) <= 1 else ", ".join(shown[:-1]) + " and " + shown[-1]

    def roles_caption(state: dict[str, Any]) -> str:
        matrix = {n["column"] for n in state["nodes"] if n["lane"] == "matrix"}
        roles = [(n["column"], n["group"]) for n in state["nodes"] if n["lane"] == "adjusted"]
        adjusted = [c for c, g in roles if g != "not answered" and c in matrix]
        out: dict[str, list[str]] = {}
        for c, g in roles:
            if g != "not answered" and c not in matrix:
                out.setdefault(g, []).append(c)
        waiting = [c for c, g in roles if g == "not answered"]
        parts = [f"Adjusted for {len(adjusted)} ({names(adjusted, 3)})" if adjusted else "Adjusted for none"]
        if out:
            parts.append("left out: " + ", ".join(f"{names(cs, 2)} ({ROLE_LABEL[g]})" for g, cs in out.items()))
        if waiting:
            parts.append(f"{len(waiting)} not answered yet")
        return "; ".join(parts) + "."

    def engine_numbers_kept(engine: str, plain: str) -> None:
        """A caption restated from the view: every number the engine's (possibly cut) caption gives."""
        cut = engine.endswith("…")
        same_numbers(re.sub(r"\S*…$", "", engine), plain)
        if not cut:
            same_numbers(plain, engine)

    for st in steps:
        if not st["id"].startswith("adjust:"):
            continue
        for o in st["options"]:
            v = o["preview"]["views"][0]
            o["preview"]["caption"] = roles_caption(v["after"])
            engine_numbers_kept(v["caption"], o["preview"]["caption"])

    def model_caption(v: dict[str, Any]) -> str:
        frames = [f["lineage"] for f in v["story"]] + [v["after"]]
        cols = [[n["column"] for n in fr["nodes"] if n["lane"] == "matrix"] for fr in frames]
        parts = []
        for i in range(1, len(cols)):
            added = [c for c in cols[i] if c not in cols[i - 1]]
            if i == 1:
                parts.append(f"Model 1 adjusts for {names(added, 3)}" if added else "Model 1 adjusts for nothing")
            else:
                parts.append(f"Model {i} adds {names(added, 3 if i == 2 else 2)}")
        return "; ".join(parts) + "."

    for k in ("guess", "empty"):
        v = option("model1", k)["preview"]["views"][0]
        option("model1", k)["preview"]["caption"] = model_caption(v)
        engine_numbers_kept(v["caption"], model_caption(v))

    COACH = [
        (r"Excluded rows' median `bmi` is `([\d.]+)`, against `([\d.]+)` kept\.",
         "Those removed have a median BMI of {}, against {} for those kept."),
        (r"`([\d,]+)` rows under their `gender` floor: likely under-reporting\.",
         "{} people below their sex's lower limit: likely under-reporting."),
        (r"`([\d,]+)` rows over their `gender` ceiling: likely over-reporting\.",
         "{} people above their sex's upper limit: likely over-reporting."),
        (r"`([\d,]+)` rows (below|above) `([\d,]+)` kcal: likely (under|over)-reporting\.",
         "{} people {} {} kcal: likely {}-reporting."),
        (r"`meds_chol` and `meds_hbp` are blank on `([\d,]+)` of these rows\.",
         "`meds_chol` and `meds_hbp` are blank for {} of these people."),
        (r"`fat_total` tracks `kcal` at r `([\d.]+)`: energy explains most, `(\d+%)`\.",
         "`fat_total` tracks `kcal` at r {}: total calories explain most of it, {}."),
        (r"With this method r `([\d.]+)`: what is left is composition\.",
         "With this choice r is {}: what is left is the diet's makeup, not how much is eaten."),
        (r"With this method r `([\d.]+)`: some of energy's signal remains\.",
         "With this choice r is {}: some of total calories' signal remains."),
    ]
    STORY = {"Each covariate's role, from your answers": "Each column's role, from your answers",
             "Fit fat_total on kcal": "Predict fat_total from kcal",
             "Keep what energy does not explain": "Keep what calories do not explain"}
    TITLE = {"Rows in “Willett 2013, by sex”": "Ranges by sex, men up to 4,000: who stays",
             "Rows in “NHS/HPFS, by sex”": "Ranges by sex, men up to 4,200: who stays",
             "Which covariates the model adjusts for": "Which columns the model adjusts for",
             "The declared models, one by one": "The planned models, one by one",
             "Who these rules would exclude": "Who this rule would remove"}

    def relabel(state: dict[str, Any] | None) -> None:
        # idempotent: two views may share one state (the direct effect's "now" is the total's after)
        for n in (state or {}).get("nodes", []):
            if n["lane"] == "adjusted" and ": " in n["label"]:
                col, role = n["label"].split(": ", 1)
                assert role in ROLE_LABEL or role in ROLE_LABEL.values(), n["label"]
                n["label"] = f"{col}: {ROLE_LABEL.get(role, role)}"

    def plain_view(v: dict[str, Any]) -> None:
        v["title"] = TITLE.get(v["title"], v["title"])
        for c in v.get("coach") or []:
            for pat, tpl in COACH:
                m = re.fullmatch(pat, c["text"])
                if m:
                    c["text"] = same_numbers(tpl.format(*m.groups()), c["text"])
                    break
        for f in v.get("story") or []:
            f["label"] = re.sub(r"^Model 2 \(primary\)", "Model 2 (main)", STORY.get(f["label"], f["label"]))
            if v["kind"] == "lineage":
                relabel(f["lineage"])
        if v["kind"] == "lineage":
            relabel(v["before"])
            relabel(v["after"])

    seen_views: set[int] = set()
    for st in steps:
        for o in st["options"]:
            for v in o["preview"]["views"]:
                if id(v) not in seen_views:  # a view two previews share is restated once
                    seen_views.add(id(v))
                    plain_view(v)

    # ── the canvas at rest: your data now (FOUNDATION §5 rule 8) ─────────────
    def ref(opt_id: str, view: int, title: str | None = None) -> dict[str, Any]:
        return {"ref": [opt_id, view], **({"title": title} if title else {})}

    def now(step_id: str, layout: str, caption: str, views: list[dict[str, Any]], *, basis: str,
            source: str = "calm", **extra: Any) -> None:
        S[step_id]["now"] = {"layout": layout, "caption": caption, "basis": basis, "source": source,
                             "views": views, **extra}

    def view_of(step_id: str, opt_id: str, i: int) -> dict[str, Any]:
        return option(step_id, opt_id)["preview"]["views"][i]

    def basis_of(step_id: str, opt_id: str) -> str:
        return option(step_id, opt_id)["preview"]["basis"]

    median = re.search(r"its median, (\d[\d,]*\d)", unit_ev)[1]
    now("unit", "focus", f"`kcal` as recorded; its median is {median}.", [ref("kcal_1", 0, "`kcal` as recorded")],
        basis=basis_of("unit", "kcal_1"))
    flow0 = view_of("exclusions", "willett_2013_by_sex", 0)
    assert [x["label"] for x in flow0["before"]] == ["Rows in the table", "`glucose` recorded"]
    everyone = f"{flow0['before'][-1]['n']:,}"
    now("exclusions", "flow", f"All {everyone} people are in the analysis now, each with `glucose` recorded.",
        [ref("willett_2013_by_sex", 0, "Who is in the analysis now"),
         ref("willett_2013_by_sex", 1, "`kcal` as recorded, everyone")],
        basis=basis_of("exclusions", "willett_2013_by_sex"))
    now("sensitivity", "flow", f"The main analysis keeps all {everyone} people.",
        [ref("willett", 0, "Who is in the main analysis")], basis=basis_of("sensitivity", "willett"))
    cc = view_of("missing", "complete_case", 0)
    assert cc["after"][-1]["dropped"] == 0
    now("missing", "flow", f"{cc['before'][-1]['n']:,} people; none is missing a value the model needs.",
        [ref("complete_case", 0, "Who is in the analysis now")], basis=basis_of("missing", "complete_case"))
    for col in ("bp_di", "bp_sys", "cycle_begin_year"):
        k = re.search(r"the same `(\d+)` predictors", view_of(f"single:{col}", "excluded", 0)["caption"])[1]
        now(f"single:{col}", "routing", f"The models read {k} predictors now; `{col}` waits for your answer.",
            [ref("covariate", 0)], basis=basis_of(f"single:{col}", "covariate"))
    lin = view_of("block", "confirm", 1)["before"]
    counts = {m[2]: int(m[1]) for n in lin["nodes"] if n["lane"] == "matrix"
              for m in [re.fullmatch(r"(\d+) (covariate|energy|exposure) columns?", n["label"])] if m}
    assert counts == {"covariate": 6, "energy": 1, "exposure": 7}, counts
    now("block", "routing", f"The models read {sum(counts.values())} columns now: {counts['covariate']} "
        f"characteristics, `kcal` and {counts['exposure']} nutrients.", [ref("confirm", 1)],
        basis=basis_of("block", "confirm"))

    std_cols = strip["methods"]["standard"]["columns"]
    assert not any(c["changed"] for c in std_cols)
    as_recorded = [{
        "column": c["column"], "output": c["column"], "shift": 0.0,
        "r_before": c["r_before"], "r_after": c["r_before"], "sd_before": c["sd_before"], "sd_after": c["sd_before"],
        "mean_before": c["mean_before"], "mean_after": c["mean_before"],
        "hist_before": r4(c["hist_before"]), "hist_after": r4(c["hist_before"]),
    } for c in std_cols]
    strip_basis = f"Values on a sample of {strip['rows']:,} of the {strip['pool']:,} analyzed rows."
    assert [c["column"] for c in as_recorded] == [o["id"] for o in S["exposure"]["options"]]
    now("exposure", "strip", f"The {len(as_recorded)} nutrients that could be the exposure, as recorded.", [],
        basis=strip_basis, strip=as_recorded, title="The nutrients, as recorded")
    eff_now = view_of("effect", "direct", 0)["before"]
    k_adj = sum(1 for n in eff_now["nodes"] if n["lane"] == "matrix") - 1  # less the exposure
    assert k_adj == 9 and f"`{k_adj}` adjusted" in view_of("effect", "total", 0)["caption"]
    now("effect", "angles", f"The plan as it stands: adjusted for {k_adj} columns, {n_total:,} people.",
        [ref("direct", 0), ref("direct", 1)], basis=whole,
        angles=[{**MEDIATORS, "view": 0}, {**WHO, "view": 1}])
    now("contrast", "angles", f"The model now reads {m_std[1]} columns, total calories (`kcal`) among them.",
        [ref("substitution", 0)], basis=basis_of("contrast", "substitution"), angles=[{**HELD, "view": 0}])

    # The first group has no captured "before" (nothing was answered): its confounder preview's
    # lineage with its own two columns set back to "not answered" (derived.rest).
    first = view_of("adjust:demographic", "confounder", 0)
    assert first["before"] is None
    blank = json.loads(json.dumps(first["after"]))
    for n in blank["nodes"]:
        if n["lane"] == "adjusted" and n["column"] in ("age", "gender"):
            n["label"], n["group"] = f"{n['column']}: not answered", "not answered"
    now("adjust:demographic", "routing", roles_caption(blank),
        [{**first, "before": blank, "after": blank, "emphasis": [], "story": [], "coach": [],
          "title": first["title"], "caption": roles_caption(blank)}],
        basis=basis_of("adjust:demographic", "confounder"), source="calm (derived.rest)")
    for st in steps:
        if st["id"].startswith("adjust:") and st["id"] != "adjust:demographic":
            o0 = st["options"][0]
            now(st["id"], "routing", roles_caption(o0["preview"]["views"][0]["before"]), [ref(o0["id"], 0)],
                basis=o0["preview"]["basis"])
    std_coach = view_of("energy", "standard", 0)["coach"][0]["text"]
    r_fat = re.search(r"at r ([\d.]+)", std_coach)[1]
    now("energy", "strip", f"The nutrients as recorded; `fat_total` tracks `kcal` at r {r_fat}.",
        [ref("standard", 0, "`fat_total` against `kcal`, as recorded")], basis=basis_of("energy", "standard"),
        strip=as_recorded, title="The nutrients, as recorded")
    assert view_of("model1", "guess", 0)["story"][0]["label"] == "Unadjusted: sugar alone"
    now("model1", "routing", "Unadjusted, the model reads `sugar` alone; Model 1 adds what you choose here.",
        [ref("guess", 0, "The planned models, one by one")], basis=basis_of("model1", "guess"))
    n_codes = re.fullmatch(r"Values on a sample of (\d+) of the [\d,]+ rows\.", basis_of("codes", "confirm"))[1]
    now("codes", "focus", f"`age` and `cycle_begin_year` as recorded, on {n_codes} sample rows.", [ref("confirm", 0)],
        basis=basis_of("codes", "confirm"))
    # The lock: the plan's models as the Model 1 answer leaves them (that answer's preview, after).
    by: dict[str, Any] = {}
    for k in ("guess", "empty"):
        v = view_of("model1", k, 0)
        settled = {**v, "before": v["after"], "emphasis": [], "story": [], "coach": []}
        by[k] = {"layout": "routing", "caption": "The models the plan will fit. " + model_caption(v),
                 "basis": basis_of("model1", k), "source": "map (derived.rest)", "views": [settled]}
    S["lock"]["now"] = by[S["model1"]["scenario"]]
    S["lock"]["now_by"] = {"step": "model1", "options": by}
    for st in steps:
        assert st.get("now"), st["id"]

    # ── fits, digests, stated sentences ──────────────────────────────────────
    fits: dict[str, Any] = {}
    for excl in ("none", "willett_2013_by_sex"):
        for m in ENERGY_ORDER:
            f = I["fits"][f"{excl}|linear|{m}"]
            fits[f"{excl}|{m}"] = {k: f[k] for k in ("rows", "measure_label", "caption", "sequence",
                                                     "sensitivity", "methods")} if not f.get("error") \
                else {"error": f["error"]}
    digests = {k: v for k, v in I["lock"]["digests"].items() if k[1] == "d" and k.endswith("g")}

    stated = [
        {"id": "lens", "section": "measurement", "head": "Lens", "sentence": I["stated"]["lens"]},
        {"id": "target", "section": "variables", "head": "Outcome", "sentence": I["stated"]["target"]},
        {"id": "roles", "section": "variables", "head": "Column roles", "sentence": I["stated"]["roles"]},
        {"id": "purpose", "section": "statistics", "head": "Purpose", "sentence": I["stated"]["purpose"]},
        {"id": "split", "section": "statistics", "head": "Validation", "sentence": I["stated"]["split"]},
        {"id": "models", "section": "statistics", "head": "Model",
         "sentence": I["models_sentence"].split(" Read from the values")[0]},
    ]

    fixture = {
        "meta": {
            "file": MAP["meta"]["file"], "rows": MAP["meta"]["rows"], "cols": MAP["meta"]["cols"],
            "captured": raw.get("captured"), "map_captured": MAP["meta"]["captured"],
            "paper_captured": DOC["captured"], "scenario": "../methods-shared/SCENARIO.md",
            "sources": {
                "calm": "capture/capture.py: the previews on the state each question is asked in; the "
                        "estimand's on the whole plan; the adjustment groups one at a time",
                "map": "../methods-map/fixture.json: fits, plan digests, teaching, labels, Model 1 previews",
                "paper": "../methods-document/fixture.json: the energy previews at the energy question",
            },
        },
        "derived": {
            "strip": ("Each energy model's change to every nutrient, computed in capture.py "
                      "(strip_numbers) the way the engine's preview computes the one nutrient it draws: "
                      "the preview's sample (5,000 of the 21,849 analyzed rows, seed 0), the method's "
                      "fitted models.steps.energy_step, and per nutrient the engine's change score "
                      "(consequences._shifts, the standardized Wasserstein distance the engine ranks "
                      "views by), r with kcal and the mean and SD before and after; 30-bin histograms "
                      "for the focused column. fat_total's numbers equal the engine's caption."),
            "block_rows": ("The block's row flow is the engine's complete-case preview on the state the "
                           "block leaves (taken at the estimand question, after the block)."),
            "effect_angles": ("The effect's panels: the adjustment card's mediators previewed on the whole "
                              "plan under each effect, and the complete-case row flow on each (the direct "
                              "effect recorded on a stopped, never-fitted project). The direct effect's "
                              "\"now\" is the total effect's capture of the same view: the plan as the "
                              "question finds it. The table's cells are those views' facts: whether the "
                              "lineage keeps the mediators in the model, and the flow's last count."),
            "contrast_angles": ("The contrast's panels: the standard model's lineage at the energy question "
                                "(substitution) and the partition the engine offers as its way out under an "
                                "addition (taken on the stopped project with the addition recorded); which "
                                "energy adjustments can follow is the engine's applicability at the energy "
                                "question (substitution) and its refusals under an addition."),
            "plain": ("Two registers (FOUNDATION §2): the card's question, lede, option names and lines, and "
                      "the canvas's captions, coach lines, panel titles, storyboard labels and role labels "
                      "restate the engine's text in plain words (build.py keeps each beside its source and "
                      "checks it carries no number the source does not); an option's `term` is the "
                      "technical name; the why and the manuscript's sentences are the engine's own."),
            "role_captions": ("The adjustment options' and Model 1's captions are restated in full from their "
                              "lineage's own nodes (who is in the model's inputs, each column's role; the story "
                              "frames for the model sequence), where the engine cuts its caption at a length; "
                              "each keeps every number the engine's caption gives."),
            "rest": ("The canvas at rest (`now`): your data now for the question, in the layout its options "
                     "use: an option's engine view drawn in its before state (`ref`), or the nutrients as "
                     "recorded (the Strip's before numbers, capture.py strip_numbers, the standard model "
                     "changing none). Two are derived: the first adjustment group's, which has no captured "
                     "before, is its confounder preview's lineage with age and gender set back to \"not "
                     "answered\"; the lock's is the Model 1 answer's lineage after it, the models the plan "
                     "will fit."),
            "leash": ("No outcome-model estimate appears before the lock (FOUNDATION §5 rule 6). The "
                      "engine's preview of the energy-dropped residual quotes the nutrient's coefficient on "
                      "the outcome; the kit replaces that caption with the engine's own caption for the "
                      "method without the coefficient (models/previews.py _relationship_caption)."),
            "mattered": ("Which of my decisions mattered: the declared model sequence and each screen "
                         "declared beside, from the fit's sequence and sensitivity analyses; a screen not "
                         "declared is left out."),
        },
        "chain": [{"id": "data", "label": "Data"}, {"id": "participants", "label": "Participants"},
                  {"id": "exposure", "label": "Exposure"}, {"id": "confounders", "label": "Confounders"},
                  {"id": "energy", "label": "Energy"}, {"id": "model", "label": "Model"},
                  {"id": "results", "label": "Results"}],
        "sections": [
            {"id": "participants", "title": "Participants", "item": "STROBE 6"},
            {"id": "variables", "title": "Variables", "item": "STROBE 7"},
            {"id": "measurement", "title": "Data sources and measurement", "item": "STROBE 8"},
            {"id": "statistics", "title": "Statistical methods", "item": "STROBE 12"},
            {"id": "results", "title": "Results", "item": "STROBE 13–17"},
        ],
        "stated": stated,
        "steps": steps,
        "estimand_sentences": combo,
        "fits": fits,
        "lock": {"template": I["lock"]["template"], "digests": digests,
                 "energy_codes": ENERGY_ORDER, "sensitivity_keys": I["sensitivity"]["keys"]},
    }
    out = KIT / "fixture.json"
    out.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB), {len(steps)} steps")


if __name__ == "__main__":
    main()
