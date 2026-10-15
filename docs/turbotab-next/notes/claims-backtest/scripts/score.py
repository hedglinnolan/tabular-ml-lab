"""Score the backtest (PLAN.md, pre-registered measures). Run after every referee is in."""
import json, glob, os, math, collections
S = "/private/tmp/claude-501/-Users-nhedglin-tabular-ml-lab/a12607bd-3f0c-4a23-8d52-bda4a2c1292e/scratchpad"
key = json.load(open(f"{S}/sealed/decoy_key.json"))
items = {it["rid"]: it for it in key["items"]}
decoys = {d["key"]: d for d in key["decoys"]}

def load(path):
    out = {}
    if not os.path.exists(path):
        return out
    for l in open(path):
        l = l.strip()
        if l:
            try:
                r = json.loads(l); out[r["id"]] = r
            except Exception:
                pass
    return out

merits = {"opus": {}, "sonnet": {}}
for m in merits:
    for f in sorted(glob.glob(f"{S}/ref/out/merits_*_{m}.jsonl")):
        merits[m].update(load(f))
basis = {}
for f in sorted(glob.glob(f"{S}/ref/out/basis_*.jsonl")):
    basis.update(load(f))

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

T = lambda r: (r or {}).get("truth")
rep = {}
missing = [rid for rid in items if rid not in merits["opus"] or rid not in merits["sonnet"]]
rep["missing_merits"] = missing

# ---- the gate on decoys
false_decoys = [rid for rid, it in items.items() if it["kind"] == "decoy" and decoys[it["orig"]]["type"] != "vacuous"]
vac_decoys = [rid for rid, it in items.items() if it["kind"] == "decoy" and decoys[it["orig"]]["type"] == "vacuous"]
passed = [rid for rid in false_decoys if T(merits["opus"].get(rid)) == "true" and T(merits["sonnet"].get(rid)) == "true"]
rep["gate"] = {
    "false_decoys": len(false_decoys), "passed_gate": len(passed),
    "passed_ids": [(rid, items[rid]["orig"], decoys[items[rid]["orig"]]["type"]) for rid in passed],
    "upper95_rule_of_three": round(3 / len(false_decoys), 3),
    "wilson_passed": [round(x, 3) for x in wilson(len(passed), len(false_decoys))],
}
for m in merits:
    p = [rid for rid in false_decoys if T(merits[m].get(rid)) == "true"]
    rep["gate"][f"single_{m}_passed"] = [(rid, items[rid]["orig"], decoys[items[rid]["orig"]]["type"]) for rid in p]
    rep["gate"][f"single_{m}_verdicts"] = dict(collections.Counter(T(merits[m].get(rid)) for rid in false_decoys))
by_type = collections.defaultdict(lambda: collections.Counter())
for rid in false_decoys:
    t = decoys[items[rid]["orig"]]["type"]
    for m in merits:
        by_type[t][f"{m}:{T(merits[m].get(rid))}"] += 1
rep["gate"]["by_type"] = {k: dict(v) for k, v in by_type.items()}
rep["vacuous"] = {rid: {"orig": items[rid]["orig"], **{m: {"truth": T(merits[m].get(rid)), "bite": (merits[m].get(rid) or {}).get("bite")} for m in merits}} for rid in vac_decoys}

# ---- real claims
real = [rid for rid, it in items.items() if it["kind"] == "real"]
def pair(rid):
    return (T(merits["opus"].get(rid)), T(merits["sonnet"].get(rid)))
agree = sum(1 for rid in real if pair(rid)[0] == pair(rid)[1])
rep["agreement_real"] = {"same_truth": agree, "n": len(real), "share": round(agree/len(real), 3)}
both_true_decoy_pairs = None
for stratum in ("random", "tier2", "targeted"):
    ids = [rid for rid in real if items[rid]["stratum"] == stratum]
    c = collections.Counter(pair(rid) for rid in ids)
    bt = sum(1 for rid in ids if pair(rid) == ("true", "true"))
    flagged = [rid for rid in ids if any(v in ("false", "overstated") for v in pair(rid))]
    hard = [rid for rid in ids if "false" in pair(rid)]
    bite_no = sum(1 for rid in ids if all((merits[m].get(rid) or {}).get("bite") == "no" for m in merits))
    rep[f"real_{stratum}"] = {"n": len(ids), "both_true": bt, "flagged_any_false_or_overstated": len(flagged),
                              "any_false": len(hard), "both_bite_no": bite_no,
                              "pairs": {f"{a}/{b}": v for (a, b), v in c.most_common()}}

# weighted (random stratum) estimate of 'both referees true' and of 'flagged'
POP = {"A": 135, "B": 506, "C": 257, "D": 73}
def area(orig): return orig.split("-")[0]
def strat_est(pred):
    tot = sum(POP.values()); est = 0; var = 0; neff_num = 0
    for a, N in POP.items():
        ids = [rid for rid in real if items[rid]["stratum"] == "random" and area(items[rid]["orig"]) == a]
        n = len(ids); k = sum(1 for rid in ids if pred(rid)); p = k / n
        W = N / tot; est += W * p; var += W*W * p*(1-p) / max(n-1, 1)
    se = math.sqrt(var)
    return round(est, 3), (round(max(0, est-1.96*se), 3), round(min(1, est+1.96*se), 3))
rep["weighted_random"] = {
    "both_true": strat_est(lambda rid: pair(rid) == ("true", "true")),
    "flagged": strat_est(lambda rid: any(v in ("false", "overstated") for v in pair(rid))),
    "any_false": strat_est(lambda rid: "false" in pair(rid)),
}
rep["by_area_random"] = {}
for a in POP:
    ids = [rid for rid in real if items[rid]["stratum"] == "random" and area(items[rid]["orig"]) == a]
    rep["by_area_random"][a] = {"n": len(ids), "both_true": sum(1 for r in ids if pair(r) == ("true", "true")),
                                "flagged": sum(1 for r in ids if any(v in ("false", "overstated") for v in pair(r))),
                                "any_false": sum(1 for r in ids if "false" in pair(r))}

# ---- basis audit
rb = {rid: basis.get(items[rid]["orig"]) for rid in real}
rep["basis_missing"] = [items[rid]["orig"] for rid in real if rb[rid] is None]
for fld in ("fidelity", "grade", "test_check", "source_check", "display"):
    rep[f"basis_{fld}"] = dict(collections.Counter((rb[rid] or {}).get(fld) for rid in real))
rep["grade_random"] = dict(collections.Counter((rb[rid] or {}).get("grade") for rid in real if items[rid]["stratum"] == "random"))

# ---- adjudication list
adj = []
for rid in real:
    o, s = merits["opus"].get(rid) or {}, merits["sonnet"].get(rid) or {}
    b = rb[rid] or {}
    if any(v in ("false", "overstated") for v in (o.get("truth"), s.get("truth"))) or b.get("fidelity") in ("partly", "unfaithful") or b.get("display") == "review":
        adj.append(rid)
rep["adjudicate_count"] = len(adj)
json.dump(rep, open(f"{S}/sealed/score_report.json", "w"), indent=1, ensure_ascii=False)
json.dump(adj, open(f"{S}/sealed/adjudicate_rids.json", "w"))
print(json.dumps(rep, indent=1, ensure_ascii=False)[:6000])
