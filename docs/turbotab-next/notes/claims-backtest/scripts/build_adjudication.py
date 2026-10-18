"""Build the adjudication batches (PLAN.md phase 6): every real claim either referee marked false or
overstated, or whose basis audit found fidelity partly/unfaithful or asked for review.

Kept as the backtest ran it: S below is that session's scratchpad, written in. A new queue's
promoted decoys are listed to adjudicate by research/registry/decoy_queue.py (python -m research.registry.decoy_queue), which reads and writes only the paths it is given."""
import json, glob, os
S = "/private/tmp/claude-501/-Users-nhedglin-tabular-ml-lab/a12607bd-3f0c-4a23-8d52-bda4a2c1292e/scratchpad"
key = json.load(open(f"{S}/sealed/decoy_key.json"))
items = {it["rid"]: it for it in key["items"]}
def load(p):
    out = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l); out[r["id"]] = r
    return out
inv = {}
for f in ["inventory_A", "inventory_B", "inventory_C", "inventory_D", "inventory_C_left_out", "parts/B_left_out_tier2_3"]:
    p = f"{S}/backtest/{f}.jsonl"
    if os.path.exists(p): inv.update(load(p))
mer = {"opus": {}, "sonnet": {}}
for m in mer:
    for f in glob.glob(f"{S}/ref/out/merits_*_{m}.jsonl"): mer[m].update(load(f))
bas = {}
for f in glob.glob(f"{S}/ref/out/basis_*.jsonl"): bas.update(load(f))

KEEP = ("truth", "failing_part", "narrowest_regime", "reason", "bite", "bite_note", "confidence", "citations")
records = []
for rid, it in items.items():
    if it["kind"] != "real": continue
    o, s, b = mer["opus"].get(rid, {}), mer["sonnet"].get(rid, {}), bas.get(it["orig"], {})
    flag = (o.get("truth") in ("false", "overstated") or s.get("truth") in ("false", "overstated")
            or b.get("fidelity") in ("partly", "unfaithful") or b.get("display") == "review")
    if not flag: continue
    records.append({"id": it["orig"], "stratum": it["stratum"], "inventory": inv[it["orig"]],
                    "referee_A": {k: o.get(k) for k in KEEP}, "referee_B": {k: s.get(k) for k in KEEP},
                    "basis_audit": b})
records.sort(key=lambda r: r["id"])
NB = 8
os.makedirs(f"{S}/adj", exist_ok=True)
for k in range(NB):
    with open(f"{S}/adj/adj_{k+1:02d}.jsonl", "w") as f:
        for r in records[k::NB]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(records), "to adjudicate;", [len(records[k::NB]) for k in range(NB)])
