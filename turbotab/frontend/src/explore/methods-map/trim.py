"""Trim the raw captures (``capture.py``) into the prototype's ``fixture.json``.

Nothing here writes a word or a number of its own: it selects the engine's texts and values, keeps
the ones the prototype shows, and rounds floats to six significant digits (an appendix term to four;
the display precision is three). Points in a preview keep their order, so a row is the same row in
every state; a preview taken on another state keeps only the views that differ (``view_refs``).
"""
from __future__ import annotations

import math
from typing import Any

TEACH_KEYS = ("lens", "target", "purpose", "grain", "roles", "clusters", "survey", "exclusions",
              "missing", "split", "estimand", "adjustment", "energy_adjustment", "models")


def r6(x: Any) -> Any:
    if isinstance(x, float):
        if not math.isfinite(x) or x == 0:
            return x
        return float(f"{x:.6g}")
    if isinstance(x, list):
        return [r6(v) for v in x]
    if isinstance(x, dict):
        return {k: r6(v) for k, v in x.items()}
    return x


def format_36(n: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while True:
        n, d = divmod(n, 36)
        out = digits[d] + out
        if n == 0:
            return out


def sentence(rec: Any) -> str | None:
    return rec["sentence"] if rec else None


def teaching(entries: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for e in entries:
        if e["key"] not in TEACH_KEYS:
            continue
        out[e["key"]] = {k: e[k] for k in ("title", "question", "one_liner", "why", "consumer")}
        out[e["key"]]["options"] = e["options"]
        out[e["key"]]["terms"] = e["terms"]
        out[e["key"]]["evidence"] = e["evidence"]
    return out


def preview(p: dict[str, Any] | None) -> dict[str, Any] | None:
    if p is None:
        return None
    if not p["ok"]:
        r = p["refusal"]
        return {"refusal": {"code": r["code"], "message": r["message"],
                            "exits": [{"label": x["label"]} for x in r["exits"]]}}
    return {"result": r6(p["result"])}


VAR_GROUPS = ("exclusions", "energy", "split", "missing")


def view_refs(var: dict[str, Any] | None, sources: list[tuple[str | None, Any]]) -> Any:
    """A preview taken on another state: each view equal to a view of the same preview on an
    earlier state (``sources``: the base state, ``None``, then the variants already written) is
    replaced by ``{"ref": i, "from": state}`` (lossless; the client resolves it)."""
    if not var or "result" not in var:
        return var
    out = []
    for v in var["result"]["views"]:
        ref = None
        for state, src in sources:
            if not src or "result" not in src:
                continue
            i = next((j for j, b in enumerate(src["result"]["views"]) if b == v), None)
            if i is not None:
                ref = {"ref": i, "from": state}
                break
        out.append(ref if ref is not None else v)
    return {"result": {**var["result"], "views": out}}


# variants in the order they may refer back to one another
VAR_ORDER = ("willett_2013_by_sex|linear", "none|spline", "willett_2013_by_sex|spline")


def variants(base: dict[str, Any], others: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    full: dict[str, Any] = {}  # each variant's previews, unreduced, for later ones to refer to
    for state in VAR_ORDER:
        pv = others[state]
        diff: dict[str, Any] = {}
        full[state] = {}
        for g in VAR_GROUPS:
            for k, raw in pv[g].items():
                mine = preview(raw)
                full[state][(g, k)] = mine
                b = preview(base[g].get(k))
                if mine == b:
                    continue
                sources = [(None, b), *((s, full[s].get((g, k))) for s in out)]
                diff.setdefault(g, {})[k] = view_refs(mine, sources)
        out[state] = diff
    return out


def labels(lab: dict[str, Any]) -> dict[str, Any]:
    return {"customary_first": lab.get("customary_first"), "tension": lab.get("tension"),
            "options": {o["key"]: {"label": o["label"], "customary": o["customary"]["text"],
                                   "sound": o["sound"]["reason"], "verdict": o["sound"]["verdict"]}
                        for o in lab["options"]}}


def effect(row: dict[str, Any]) -> dict[str, Any]:
    return r6({k: row[k] for k in ("feature", "estimate", "ci_low", "ci_high", "p")})


def r4(x: Any) -> Any:
    if isinstance(x, float) and math.isfinite(x) and x != 0:
        return float(f"{x:.4g}")
    return x


WHYS: list[str] = []  # an adjustment term's "why", said once and referred to by index


def term(row: dict[str, Any]) -> list[Any]:
    """An appendix row, compact: [feature, estimate, ci_low, ci_high, p, why index] (four
    significant digits; the table prints three)."""
    why = row.get("why") or ""
    if why not in WHYS:
        WHYS.append(why)
    return [row["feature"], *(r4(row[k]) for k in ("estimate", "ci_low", "ci_high", "p")),
            WHYS.index(why)]


ENERGY_CODES = ["standard", "residual", "residual_energy_dropped", "density_multivariate", "density"]
SENS_KEYS = ["willett_2013_by_sex", "nhs_hpfs_by_sex", "sex_neutral_500_5000"]


def lock_key(key: str) -> str:
    """``none|linear|standard|willett_2013_by_sex+nhs_hpfs_by_sex|guess`` → ``nl0-3g``: the
    exclusions (n/w), the form (l/s), the energy method's index, the screens beside it as a bit
    mask over SENS_KEYS, and Model 1 (g guess, e none, u unanswered)."""
    excl, form, method, sens, m1 = key.split("|")
    mask = sum(1 << SENS_KEYS.index(s) for s in sens.split("+") if s != "none")
    return (f"{'n' if excl == 'none' else 'w'}{form[0]}{ENERGY_CODES.index(method)}-{mask}"
            f"{ {'guess': 'g', 'empty': 'e', 'unset': 'u'}[m1]}")


def fit(bundle: dict[str, Any]) -> dict[str, Any]:
    if "error" in bundle:
        return {"error": bundle["error"].get("design") or next(iter(bundle["error"].values()))}
    eff = bundle["effects"]
    fam = eff["families"][0]
    model = bundle["fit"]["models"][0]
    sens = bundle["sensitivity"]
    sfam = sens["families"][0]

    def sugar(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [effect(r) for r in rows if r["feature"] == "sugar" or r["feature"].startswith("sugar")]

    unmeasured = None
    if fam.get("sensitivity"):
        s = fam["sensitivity"][0]
        ev, rv = s.get("e_value") or {}, s.get("robustness") or {}
        unmeasured = r6({"e_point": ev.get("point"), "e_limit": ev.get("limit"),
                         "rv": rv.get("rv"), "partial_r2": rv.get("partial_r2"),
                         "interval_note": rv.get("interval_note")})
    return {
        "rows": eff["rows"],
        "measure_label": eff["measure_label"],
        "caption": bundle["fit"]["estimand"]["caption"],
        "sequence": [{
            "key": s["key"], "label": s["label"], "note": s["note"], "n_rows": s["n_rows"],
            "adjusted_for": s["adjusted_for"], "effects": [effect(e) for e in s["effects"]],
            "inference": s["inference"]["caption"], "concerns": s.get("concerns") or [],
        } for s in fam["sequence"]],
        "appendix": [{"key": a["key"], "label": a["label"],
                      "terms": [term(t) for t in a["terms"]]}
                     for a in fam["appendix"]],
        "appendix_title": eff["appendix_title"],
        "diagnostics": [{"check": d["check"], "status": d["status"], "reading": d["reading"]}
                        for d in fam.get("diagnostics") or []],
        "unmeasured": unmeasured,
        "tests": [t["caption"] for t in model.get("exposure_tests") or []],
        "methods": eff["methods"],
        "sensitivity": {
            "methods": sens.get("methods"),
            "concerns": sens.get("concerns") or [],
            "analyses": [{
                "label": a["label"], "primary": a["primary"], "added": a["added"],
                "rules": a["rules"], "n_rows": a["n_rows"], "refused": a["refused"],
                "effects": sugar(next((f["coefficients"] for f in sfam["fits"]
                                       if f["label"] == a["label"]), [])),
            } for a in sens["analyses"]],
        },
    }


def readings_items(main: dict[str, Any]) -> list[dict[str, Any]]:
    """The readings the card asks about, in its order: the roles proposed below high confidence,
    then the fit's code-or-amount questions. The kcal unit is its own item (``unit``)."""
    by_col = {c["column"]: c for c in main["roles_stage"]["columns"]}
    items = []
    for col in main["unconfirmed"]:
        c = by_col[col]
        items.append({"key": f"role:{col}", "reading": "role", "column": col,
                      "value": c["proposed"], "confidence": c["confidence"],
                      "evidence": c["reason"], "consumer": None})
    for g in main["ask_codes"]["groups"]:
        col = g["columns"][0]
        items.append({"key": f"code_or_count:{col}", "reading": "code_or_count", "column": col,
                      "value": g["guess"], "words": g["guess_words"], "confidence": None,
                      "evidence": g["evidence"], "consumer": main["ask_codes"]["consumer"]})
    return items


def trim(teach: list[dict[str, Any]], main: dict[str, Any], branches: dict[str, Any],
         pred: dict[str, Any], ip: dict[str, Any]) -> dict[str, Any]:
    S = main["sentences"]
    P = main["previews"]
    prop = main["proposals"]
    energy = prop["energy"]
    lab = prop["labels"]
    ask_unit = main["ask_unit"]
    est = main["estimand_card"]
    roles_by = {c["column"]: c for c in main["roles_stage"]["columns"]}
    adj = main["adjustment_card"]

    inference = {
        "summary": {"rows": main["draft_view"]["summary"]["n_rows"],
                    "cols": main["draft_view"]["summary"]["n_cols"],
                    "file": main["draft_view"]["summary"]["source_name"]},
        "steps": main["pre_lock_view"]["steps"],
        "draft_steps": main["draft_view"]["steps"],
        "stated": {k: sentence(S[k]) for k in ("lens", "target", "purpose", "roles", "split")},
        "models_sentence": S["models"]["sentence"],
        "roles": [{k: c[k] for k in ("column", "proposed", "confidence", "reason", "attention")}
                  for c in main["roles_stage"]["columns"]],
        "readings": {
            "items": readings_items(main),
            "unit": {"column": ask_unit["groups"][0]["columns"][0],
                     "guess_words": ask_unit["groups"][0]["guess_words"],
                     "evidence": ask_unit["groups"][0]["evidence"],
                     "consumer": ask_unit["consumer"],
                     "confirm": ask_unit["exits"][0]["label"],
                     "alternatives": [c["label"] for x in ask_unit["read_from_data"]
                                      if x["kind"] == "unit" and x["column"] == "kcal"
                                      for c in x["change"]],
                     "sentence": S["unit_kcal"]["sentence"],
                     "previews": {"confirm": preview(P["unit"]), "two_days": preview(P["unit_2days"])}},
            "single": ip["single"],
            # the block's sentence for each set it can list: a base-36 bit mask over ``items``
            # (the readings' keys, in the card's order) → an index into ``sentences``
            "block": {"items": ip["blocks"]["items"], "sentences": ip["blocks"]["sentences"],
                      "by_mask": {format_36(int(k)): v
                                  for k, v in ip["blocks"]["by_mask"].items()}},
            "read_from_data": [{k: x[k] for k in ("kind", "column", "value", "words", "evidence")}
                               for x in main["readings"]["read_from_data"]],
            "read_sentence": main["readings"]["sentence"],
        },
        "exclusions": {
            "labels": labels(lab["exclusions"]),
            "offered": [{"key": o["key"], "label": o["label"], "affected": o["affected"],
                         "evidence": o["evidence"]["status"]} for o in prop["exclusions"]],
            "n_base": prop["n_base"],
            "basis": prop["basis"],
            "sentences": {k: sentence(v) for k, v in S["exclusions"].items()},
            "previews": {k: preview(v) for k, v in P["exclusions"].items()},
        },
        "sensitivity": {"keys": ["willett_2013_by_sex", "nhs_hpfs_by_sex", "sex_neutral_500_5000"],
                        "sentences": {k: sentence(v) for k, v in S["sensitivity"].items()}},
        "seal": {
            "reason": main["seal_plan"]["reason"],
            "options": [{k: o[k] for k in ("holdout", "label", "n_holdout", "measures")}
                        for o in main["seal_plan"]["options"]],
            "previews": {k: preview(v) for k, v in P["split"].items()},
            "sentence": S["split"]["sentence"],
        },
        "estimand": {
            "exposures": [{"column": e["column"], "energy_contrast": e["energy_contrast"],
                           "evidence": roles_by.get(e["column"], {}).get("reason"),
                           "role": roles_by.get(e["column"], {}).get("proposed")}
                          for e in est["exposures"]],
            "effects": est["effects"], "contrasts": est["contrasts"],
            "measures": [{k: m[k] for k in ("measure", "label", "fitted", "reason", "rank")}
                         for m in est["measures"]],
            "which_contrast": main["refusals"]["which_contrast"]["message"],
            "sentence": S["estimand"]["sentence"],
            "preview_note": P["estimand"]["result"]["note"],
            "preview_basis": P["estimand"]["result"]["basis"],
        },
        "adjustment": {
            "questions": adj["questions"],
            "source": adj["source"],
            "groups": [{k: g[k] for k in ("key", "label", "columns", "guess", "reason", "derived",
                                          "derived_words")} for g in adj["groups"]],
            "group_sentences": {k: v["sentence"] for k, v in S["adjustment"]["groups"].items()},
            "unguessed": [{"columns": u["columns"], "answers": u["answers"],
                           "sentence": u["sentence"]} for u in S["adjustment"]["unguessed"]],
            "per_column": ip["per_column"],
            "derive": ip["derive"],
            "after": {k: main["adjustment_after"][k] for k in ("adjusted", "left_out", "secondary")},
            "mediator_kept": {"message": main["refusals"]["mediator_kept"]["message"],
                              "exits": [x["label"] for x in main["refusals"]["mediator_kept"]["exits"]]},
            "preview_note": P["adjustment"]["result"]["note"],
        },
        "energy": {
            "labels": labels(lab["energy_adjustment"]),
            "applicability": energy["applicability"],
            "ranking": energy["ranking"],
            "usual": energy["usual"],
            "r_with_energy": r6(energy["r_with_energy"]),
            "sentences": {k: sentence(v) for k, v in S["energy"].items()},
            "after": {k: sentence(v) for k, v in branches["after"]["energy"].items()},
            "refusals": {m: {"code": main["refusals"][f"energy_{m}"]["code"],
                             "message": main["refusals"][f"energy_{m}"]["message"],
                             "exits": [x["label"] for x in main["refusals"][f"energy_{m}"]["exits"]]}
                         for m in ("none", "all_components", "partition")},
            "previews": {k: preview(v) for k, v in P["energy"].items()},
        },
        "form": {
            "options": prop["exposure_forms"],
            "sentences": {k: sentence(v) for k, v in S["form"].items()},
            "after": {k: sentence(v) for k, v in branches["after"]["form"].items()},
            "preview_note": P["form"]["linear"]["result"]["note"],
        },
        "missing": {
            "labels": labels(lab["missing"]),
            "sentences": {k: sentence(v) for k, v in S["missing"].items()},
            "before_adjustment": S["missing_before_adjustment"]["sentence"],
            "previews": {k: preview(v) for k, v in P["missing"].items()},
            "columns": prop["missing"]["columns"],
        },
        "model_1": {"guess": main["model_sequence"]["guess"],
                    "allowed": main["model_sequence"]["allowed"],
                    "reason": main["model_sequence"]["reason"],
                    "sentences": {"guess": S["model_sequence"]["guess"]["sentence"],
                                  "empty": ip["model_1_empty"]}},
        "shelf": [{k: f.get(k) for k in ("key", "label", "rank", "fit", "inductive_bias")}
                  for f in main["shelf"]["families"]],
        # previews taken on the other recordable states ("<exclusions>|<form>"), as differences
        "previews_var": variants(P, main["previews_var"]),
        "lock": {"digests": {lock_key(k): v for k, v in ip["lock"].items()},
                 "template": ip["lock_template"]},
        "fits": {k: fit(v) for k, v in branches["fits"].items()},
        "appendix_whys": WHYS,
        "methods_at_lock": [{k: ln[k] for k in ("seq", "kind", "after_estimates", "sentence")}
                            for ln in main["methods"]["lines"]],
    }

    ps = pred["sentences"]
    plab = pred["proposals"]["labels"]
    prediction = {
        "steps": pred["view"]["steps"],
        "steps_after_seal": pred["view_after_seal"]["steps"],
        "stated": {"lens": ps["lens"]["sentence"], "target": ps["target"]["sentence"],
                   "purpose": ps["purpose"]["sentence"], "roles": ps["roles"]["sentence"]},
        "exclusions": {"labels": labels(plab["exclusions"]),
                       "sentence_none": ps["exclusions_none"]["sentence"],
                       "previews": {k: preview(v) for k, v in pred["previews"]["exclusions"].items()}},
        "missing": {"labels": labels(plab["missing"]),
                    "sentence_impute": ps["missing"]["sentence"],
                    "previews": {k: preview(v) for k, v in pred["previews"]["missing"].items()}},
        "seal": {"reason": pred["seal_plan"]["reason"],
                 "options": [{k: o[k] for k in ("holdout", "label", "n_holdout", "measures")}
                             for o in pred["seal_plan"]["options"]],
                 "validation": [{k: o[k] for k in ("validation", "label", "measures")}
                                for o in pred["seal_plan"]["validation"]["options"]],
                 "previews": {k: preview(v) for k, v in pred["previews"]["split"].items()},
                 "sentences": {k: v["sentence"] for k, v in ps["split"].items()}},
        "energy": {"labels": labels(plab["energy_adjustment"]),
                   "ranking": pred["proposals"]["energy"]["ranking"]},
        "shelf": [{k: f.get(k) for k in ("key", "label", "rank", "fit", "inductive_bias",
                                         "estimate")} for f in pred["shelf"]["families"]],
    }
    return {
        "meta": {"file": main["draft_view"]["summary"]["source_name"],
                 "rows": main["draft_view"]["summary"]["n_rows"],
                 "cols": main["draft_view"]["summary"]["n_cols"],
                 "captured": main["draft_view"]["summary"]["created_at"]},
        "teaching": teaching(teach),
        "inference": inference,
        "prediction": prediction,
    }
