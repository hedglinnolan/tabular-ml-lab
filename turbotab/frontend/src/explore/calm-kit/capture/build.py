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


def opt(id_: str, name: str, what: str, sentence: str | None, pv: dict[str, Any], *,
        label: str | None = None, disabled: bool = False, refusal: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"id": id_, "name": name, "what": what, "sentence": sentence, "preview": pv}
    if label:
        out["label"] = label
    if disabled:
        out["disabled"] = True
    if refusal:
        out["refusal"] = refusal
    return out


def step(id_: str, stage: str, section: str, head: str, question: str, lede: str, why: str,
         legend: str, options: list[dict[str, Any]], scenario: str, *, slot: str | None = None,
         requires: dict[str, list[str]] | None = None) -> dict[str, Any]:
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
    steps.append(step(
        "unit", "data", "measurement", "Energy's unit",
        # ask card: "`kcal`: kcal, one day's intake?"
        "Is `kcal` one day's intake, in kcal?",
        # ask evidence, shortened
        "Its median, 1,950, is an adult's day's intake; a child's two-day total sits there too.",
        unit_ask["groups"][0]["evidence"] + " Read by the screens.",
        "Choose the unit of kcal",
        [
            opt("kcal_1", "kcal, one day's intake",
                "The 500–5,000 kcal-a-day screen reads as is: 111 of 5,000 rows outside.",
                SEN["unit"]["kcal_1"], preview(U["kcal_1"], "calm"), label="Recommended"),
            opt("kcal_2", "A total over 2 days, in kcal",
                "The screen becomes 1,000–10,000 kcal over 2 days: 459 of 5,000 rows outside.",
                SEN["unit"]["kcal_2"], preview(U["kcal_2"], "calm")),
            opt("kj_1", "kJ, one day's intake",
                "The screen becomes 2,092–20,920 kJ a day: 2,798 of 5,000 rows outside.",
                SEN["unit"]["kj_1"], preview(B["unit_kj"], "calm")),
        ], "kcal_1"))

    # ── Participants ─────────────────────────────────────────────────────────
    ex_labels = I["exclusions"]["labels"]
    EX = P["exclusions"]
    ex_sent = I["exclusions"]["sentences"]
    gold = EX["goldberg_schofield"]["refusal"]
    steps.append(step(
        "exclusions", "participants", "participants", "Eligibility",
        # calm-screen.html (approved), from the screens the engine offers
        "Should people with implausible energy intakes be removed?",
        "A recall far outside a normal day's intake is usually a reporting error.",
        # labels.tension and the teaching entry's why, shortened
        "Fixed kcal cut-offs are the field's habit (Willett, Nutritional Epidemiology, 2013). They are "
        "not individualized, so the every-row analysis is reported beside a screen (Banna et al. 2017, "
        "Front Nutr 4:45). Under-reporting concentrates in higher BMI, so many analyses keep everyone.",
        "Choose who stays in the analysis",
        [
            opt("none", "Keep every row",
                "No row is excluded; the record states that no exclusion was applied.",
                ex_sent["none"], preview(EX["none"], "calm"), label="Recommended"),
            opt("willett_2013_by_sex", "Willett 2013, by sex",
                "Women outside 500–3,500 and men outside 800–4,000 kcal a day: 1,614 rows leave.",
                ex_sent["willett_2013_by_sex"], preview(EX["willett_2013_by_sex"], "calm"),
                label="Common practice"),
            opt("nhs_hpfs_by_sex", "NHS/HPFS, by sex",
                "Women outside 500–3,500 and men outside 800–4,200 kcal a day: 1,419 rows leave.",
                ex_sent["nhs_hpfs_by_sex"], preview(EX["nhs_hpfs_by_sex"], "calm")),
            opt("sex_neutral_500_5000", "500–5,000 kcal a day",
                "Anyone outside 500–5,000 kcal a day, whatever their sex: 501 rows leave.",
                ex_sent["sex_neutral_500_5000"], preview(EX["sex_neutral_500_5000"], "calm")),
            opt("sex_neutral_500_3500", "500–3,500 kcal a day",
                "Anyone outside 500–3,500 kcal a day; stricter, so 2,094 rows leave.",
                ex_sent["sex_neutral_500_3500"], preview(EX["sex_neutral_500_3500"], "calm")),
            opt("goldberg_schofield", "Goldberg, energy over BMR",
                "Needs the units of `age` and `weight` confirmed first.",
                None, preview(EX["goldberg_schofield"], "calm"), label="Not available yet",
                disabled=True, refusal=gold["message"]),
        ], "none", requires={"unit": ["kcal_1"]}))

    SV = P["sensitivity"]
    steps.append(step(
        "sensitivity", "participants", "participants", "Every row, beside",
        "Which screens should be reported beside the every-row analysis?",
        "Each is the same model on its own rows, shown only after the plan is locked.",
        ex_labels["tension"],
        "Choose the screens reported beside",
        [
            opt("both", "Both sex-specific screens",
                "Willett 2013 (20,235 rows) and NHS/HPFS (20,430 rows), each the same model.",
                SEN["sensitivity"]["both"], preview(SV["both"], "calm")),
            opt("willett", "Willett 2013, by sex",
                "The same model on the 20,235 rows Willett's cut-offs keep.",
                SEN["sensitivity"]["willett"], preview(SV["willett"], "calm")),
            opt("nhs", "NHS/HPFS, by sex",
                "The same model on the 20,430 rows the NHS/HPFS cut-offs keep.",
                SEN["sensitivity"]["nhs"], preview(SV["nhs"], "calm")),
            opt("none", "None", "No sensitivity analysis is set beside the primary analysis.",
                SEN["sensitivity"]["none"], preview(SV["none"], "calm")),
        ], "both", requires={"unit": ["kcal_1"]}))

    mi_labels = I["missing"]["labels"]
    MS = P["missing"]
    steps.append(step(
        "missing", "participants", "statistics", "Missing data",
        T["missing"]["question"],
        # labels.tension, shortened
        "Complete cases are the field's habit; when blanks depend on measured variables, imputation is sounder.",
        T["missing"]["why"],
        "Choose how missing values are handled",
        [
            opt("complete_case", "Complete cases",
                "Rows missing any predictor are dropped; the participant flow shows how many.",
                I["missing"]["sentence"], preview(MS["complete_case"], "calm"), label="Common practice"),
            opt("multiple_imputation", "Multiple imputation",
                "Each blank imputed 20 times with the outcome; results pooled by Rubin's rules.",
                None, preview(MS["multiple_imputation"], "calm"), label="Recommended"),
        ], "complete_case"))

    # ── Exposure: the role readings, then the estimand ───────────────────────
    roles_ask = M["asks"]["roles"]
    by_col = {g["columns"][0]: g for g in roles_ask["groups"] if g["kind"] == "role"}
    items = {it["column"]: it for it in I["readings"]["items"] if it["reading"] == "role"}
    roles_t = {o["value"]: o for o in T["roles"]["options"]}
    for n, col in enumerate(["bp_di", "bp_sys", "cycle_begin_year"]):
        it = items[col]
        steps.append(step(
            f"single:{col}", "exposure", "measurement", "Readings",
            # ask card: "`bp_di`: a covariate?"
            f"Is `{col}` a covariate?",
            it["evidence"],
            T["roles"]["why"] + " Proposed below high confidence, so it waits for your confirmation.",
            f"Confirm the role of {col}",
            [
                opt("covariate", "Yes, a covariate", roles_t["covariate"]["consequence"],
                    SEN["single"][col]["guess"], preview(P["single"][col]["guess"], "calm"),
                    label="Recommended"),
                opt("excluded", "No, leave it out of the models", roles_t["excluded"]["consequence"],
                    SEN["single"][col]["leave_out"], preview(P["single"][col]["leave_out"], "calm")),
            ], "covariate", slot=f"single:{col}"))
        assert by_col[col]["guess"] == "covariate"

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
    steps.append(step(
        "block", "exposure", "measurement", "Readings",
        f"Are the other {n_block} readings right as shown?",
        "Each was proposed below high confidence; the covariates enter the models once confirmed.",
        # teaching one_liner, and the stated roles sentence's "proposed below high confidence and
        # wait for their own confirmation before any default reads them"
        T["roles"]["one_liner"] + " Each was proposed below high confidence, so no default reads it "
        "until it is confirmed.",
        "Confirm the other readings",
        [opt("confirm", block["label"],
             "Seven covariates enter the models; six flags stay out of them.",
             SEN["block"], block_pv)],
        "confirm", slot="block"))

    E = I["estimand"]  # the card as the map captured it, with each exposure's evidence
    assert [e["column"] for e in M["cards"]["estimand"]["exposures"]] == [e["column"] for e in E["exposures"]]
    EP = P["estimand"]
    combo = SEN["estimand_combo"]
    exposures = [e for e in E["exposures"] if e.get("energy_contrast")]
    exposures.sort(key=lambda e: e["column"] != "sugar")  # the scenario's first; the rest as offered
    est_t = T["estimand"]
    steps.append(step(
        "exposure", "exposure", "variables", "Exposure and estimand",
        "Which nutrient is the exposure?",
        est_t["one_liner"],
        est_t["why"],
        "Choose the exposure",
        [opt(e["column"], f"`{e['column']}`", e["evidence"].replace("A nutrient that carries energy: an exposure; it",
                                                                   "Carries energy; it"),
             None, preview(EP[f"exposure:{e['column']}"], "calm")) for e in exposures],
        "sugar", slot="estimand"))

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
    m_names = ", ".join(f"`{c}`" for c in mediators[:2]) + f" and {len(mediators) - 2} more"

    def angles_pv(route: dict, rows: dict, meaning: str, cells: list[str], touch: list[str],
                  basis: str) -> dict[str, Any]:
        return {"source": "calm", "basis": basis, "note": None,
                "views": [route["views"][0], rows["views"][0]],
                "angles": [
                    {"question": "What does the model hold fixed?", "view": 0, "cell": cells[0],
                     "touch": touch},
                    {"question": "Who stays in the analysis?", "view": 1, "cell": cells[1]},
                    {"question": "What will the estimate mean?", "text": meaning, "cell": cells[2]},
                ]}

    whole = "Drawn on the scenario's whole plan before the lock; at its own question the engine cannot draw the estimand yet."
    steps.append(step(
        "effect", "exposure", "variables", "Exposure and estimand",
        "Its total effect, or its direct effect?",
        "A direct effect holds the mediators fixed; a total effect counts everything downstream.",
        est_t["why"],
        "Choose the effect",
        [
            opt("total", eff["total"]["label"], eff["total"]["consequence"], None,
                angles_pv(total_route, total_rows,
                          eff["total"]["consequence"],
                          [f"Nothing downstream: {m_names} left out", "All 21,849 rows",
                           "Everything `sugar` changes downstream counts"], mediators, whole)),
            opt("direct", eff["direct"]["label"], eff["direct"]["consequence"], None,
                angles_pv(direct_route, direct_rows,
                          eff["direct"]["consequence"],
                          [f"{m_names}, held fixed", "2,996 rows: `meds_chol` and `meds_hbp` are blank on 18,853",
                           "`sugar`'s effect with the mediators held fixed"], mediators, whole)),
        ], "total", slot="estimand"))

    std_doc = preview(doc_preview("standard"), "paper")
    std_lineage = next(v for v in std_doc["views"] if v["kind"] == "lineage")
    add = P["addition_energy"]
    add_exit = preview(add["partition:exit0"], "calm")
    add_lineage = next(v for v in add_exit["views"] if v["kind"] == "lineage")
    exit_label = add["partition"]["refusal"]["exits"][0]["label"]
    applic = I["energy"]["applicability"]
    runnable = [m for m in I["energy"]["ranking"]["order"] if applic[m]["ok"] and m != "none"]
    sub_caption = EP["exposure:sugar"]["result"]["views"][0]["caption"]
    add_caption = EP["contrast:addition"]["result"]["views"][0]["caption"]
    steps.append(step(
        "contrast", "exposure", "variables", "Exposure and estimand",
        "Is `sugar`'s effect a substitution or an addition of calories?",
        # card.which_contrast, shortened
        "`sugar` carries energy and total energy is in the model; the two are different estimands (Tomova et al. 2022).",
        E["which_contrast"],
        "Choose the contrast",
        [
            opt("substitution", con["substitution"]["label"], con["substitution"]["consequence"], None,
                {"source": "calm+paper", "basis": std_doc["basis"], "note": None,
                 "views": [std_lineage],
                 "angles": [
                     {"question": "What does the model hold fixed?", "view": 0,
                      "cell": "Total energy, `kcal`", "touch": ["kcal"],
                      "touch_source": applic["standard"]["reason"]},
                     {"question": "Which energy models can follow?",
                      "list": [{"name": I["energy"]["labels"]["options"][m]["label"], "ok": True}
                               for m in runnable] +
                              [{"name": I["energy"]["labels"]["options"][m]["label"], "ok": False,
                                "why": applic[m]["reason"]} for m in ("all_components", "partition")],
                      "cell": f"{len(runnable)} run on these columns"},
                     {"question": "What will the estimate mean?", "text": sub_caption,
                      "cell": "In place of other calories"},
                 ]}),
            opt("addition", con["addition"]["label"], con["addition"]["consequence"], None,
                {"source": "calm", "basis": add_exit["basis"], "note": None,
                 "views": [add_lineage],
                 "angles": [
                     {"question": "What does the model hold fixed?", "view": 0,
                      "cell": "Every other source's calories"},
                     {"question": "Which energy models can follow?",
                      "list": [{"name": I["energy"]["labels"]["options"][m]["label"], "ok": False,
                                "why": add[m]["refusal"]["message"]} for m in ("all_components", "partition")]
                              + [{"name": exit_label, "ok": True,
                                  "why": "The engine's way out: `sugar` has no term of its own in it."}],
                      "cell": "Both refuse these columns"},
                     {"question": "What will the estimate mean?", "text": add_caption,
                      "cell": "Added to the diet"},
                 ]}),
        ], "substitution", slot="estimand"))

    # ── Confounders: the adjustment card, one group at a time ────────────────
    A = I["adjustment"]
    derive = A["derive"]
    role_order = ["confounder", "timing_unknown", "mediator", "not_a_cause"]
    triple = {"confounder": "yes,yes,no", "timing_unknown": "unknown,yes,unknown",
              "mediator": "no,yes,yes", "not_a_cause": "no,no,no"}
    adj_t = {o["value"]: o for o in T["adjustment"]["options"]}
    names_for = {"confounder": "Confounders: adjust for them", "timing_unknown": "Timing unknown: beside, in Model 3",
                 "mediator": "Mediators: leave them out", "not_a_cause": "Causes of neither: leave them out"}
    one_for = {"confounder": "Confounder: adjust for it", "timing_unknown": "Timing unknown: beside, in Model 3",
               "mediator": "Mediator: leave it out", "not_a_cause": "Cause of neither: leave it out"}
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
            d = derive[triple[role]]
            what = adj_t[role]["consequence"] if role in adj_t else d["why"][0].upper() + d["why"][1:] + "."
            pv = B["previews"][g["key"]]["scenario"] if role == scen_role else B["previews"][g["key"]]["options"][role]
            sentence = SEN["adjustment"][g["key"]]["scenario" if role == scen_role else role]
            options.append(opt(role, (names_for if len(cols) > 1 else one_for)[role], what, sentence,
                               preview(pv, "calm"),
                               label="Recommended" if guessed == role else None))
        if guessed:
            options.sort(key=lambda o: o["id"] != guessed)
        reason = group["reason"]
        reason = re.sub(r" \(NUTRITION_PACK §08 \(the nested model table:.*\)\)\.$", ".", reason)
        reason = reason.split("; ")[0].rstrip(".") + "."  # its first clause
        lede = (reason[0].upper() + reason[1:]) if group.get("guess") else \
            "The pack says nothing about these; each is asked."  # the card's own reason
        steps.append(step(
            f"adjust:{g['key']}", "confounders", "variables", "Adjustment set",
            f"What {'are' if len(cols) > 1 else 'is'} {listed} to the effect of `sugar`?",
            lede if words(lede) <= 22 else " ".join(lede.split()[:21]).rstrip(",;:") + "…",
            T["adjustment"]["why"],
            f"Choose the role of {', '.join(cols)}",
            options, scen_role, slot=f"adjust:{g['key']}",
            requires={"exposure": ["sugar"], "effect": ["total"]}))

    # ── Energy ───────────────────────────────────────────────────────────────
    en_labels = I["energy"]["labels"]
    en_t = {o["value"]: o for o in T["energy_adjustment"]["options"]}
    strip = raw["strip"]
    energy_opts = []
    for m in I["energy"]["ranking"]["order"]:
        ok = applic[m]["ok"] and m != "none"
        dp = doc_preview(m)
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
            energy_opts.append(opt(m, en_t[m]["label"], en_t[m]["consequence"],
                                   I["energy"]["sentences"].get(m), pv,
                                   label=("Recommended" if m == I["energy"]["ranking"]["order"][0] else
                                          "Common practice" if m == en_labels["customary_first"] else None)))
        else:
            ref = dp["body"]["error"]
            energy_opts.append(opt(m, en_t[m]["label"], en_t[m]["consequence"], None,
                                   preview(dp, "paper"), label="Not available", disabled=True,
                                   refusal=ref["message"]))
    energy_opts.sort(key=lambda o: bool(o.get("disabled")))
    steps.append(step(
        "energy", "energy", "statistics", "Energy adjustment",
        T["energy_adjustment"]["question"],
        # ranking.line, shortened
        "All components would rank first, but cannot run on these columns; the standard model leads.",
        T["energy_adjustment"]["why"],
        "Choose the energy adjustment",
        energy_opts, I["energy"]["ranking"]["order"][0], requires={"contrast": ["substitution"]}))

    # ── Model ────────────────────────────────────────────────────────────────
    m1 = I["model_1"]
    steps.append(step(
        "model1", "model", "statistics", "Model sequence",
        "Which columns should Model 1 adjust for?",
        "The field's Model 1 adjusts for age, sex and energy; say which columns those are.",
        m1["reason"],
        "Choose Model 1's adjustment",
        [
            opt("guess", "`age`, `gender` and `kcal`",
                "The field's Model 1; Model 3 adds the possible mediators set beside it.",
                m1["sentences"]["guess"], preview(m1["previews"]["guess"], "map"), label="Recommended"),
            opt("empty", "Nothing: Model 1 is unadjusted",
                "Model 1 adjusts for nothing; Model 2 adds the whole adjustment set.",
                m1["sentences"]["empty"], preview(m1["previews"]["empty"], "map")),
        ], "guess"))

    codes_ask = M["asks"]["codes"]
    steps.append(step(
        "codes", "model", "measurement", "Readings",
        "Do `age` and `cycle_begin_year` hold amounts or codes?",
        # each reading's evidence, as the card shows it
        "; ".join(f"`{it['column']}`: {it['evidence']}" for it in I["readings"]["items"]
                  if it["reading"] == "code_or_count") + ".",
        codes_ask["text"],
        "Confirm the code-or-amount readings",
        [opt("confirm", codes_ask["exits"][0]["label"],
             "`age` holds amounts; `cycle_begin_year` holds codes for categories.",
             SEN["codes"], preview(P["codes"], "calm"))],
        "confirm"))

    lock_reason = ("Recorded the first time an estimate is displayed; it changes no number, only how "
                   "later records are marked.")
    steps.append(step(
        "lock", "model", "statistics", "The plan, locked",
        "Lock the plan and fit the models?",
        "No estimate appears before the plan is locked; later changes are marked as seen after.",
        "Choosing an analysis after seeing its estimates is a forking path (Gelman & Loken 2013). "
        "Locking records the plan's SHA-256 before the first estimate is shown; every later change "
        "is marked in the record as made after the estimates were seen.",
        "Lock the plan",
        [opt("lock", "Lock the plan and fit",
             "Records the plan's SHA-256, then fits the declared models.",
             None, {"source": "engine", "basis": "", "note": lock_reason, "views": []})],
        "lock"))

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
                              "question finds it."),
            "contrast_angles": ("The contrast's panels: the standard model's lineage at the energy question "
                                "(substitution) and the partition the engine offers as its way out under an "
                                "addition (taken on the stopped project with the addition recorded)."),
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
