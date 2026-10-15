"""Build the referee batches and the sealed key (PLAN.md phases 2-4, Amendment 1)."""
import json, random, hashlib, os
from sample import draw, load

AREAS = "ABCD"
SEED = 20261015
MERITS_FIELDS = ["statement", "goal", "regime", "decision", "app_action", "claimed_basis"]

exec(open("decoys_ACD.py").read())          # DECOYS (A, C, D)
exec(open("decoys_B.py").read())            # DECOYS_B
decoys = DECOYS + DECOYS_B

TARGETED = json.load(open("targeted.json"))  # {"A": [...], "B": [...], ...}
DROPPED_DUPES = json.load(open("dropped_dupes.json"))  # [{"id":..., "dup_of":..., "why":...}]
dropped = {d["id"] for d in DROPPED_DUPES}

inv = {}
for a in AREAS:
    for r in load(a):
        inv[r["id"]] = r
for side in ("inventory_C_left_out.jsonl", "parts/B_left_out_tier2_3.jsonl"):
    if not os.path.exists(side):
        continue
    for l in open(side):
        if l.strip():
            r = json.loads(l); inv[r["id"]] = r

# strata
random_ids = []
for a in AREAS:
    _, _, pick = draw(a)
    random_ids += [i for i in sorted(pick) if i not in dropped]
tier2_pool = sorted(i for i, r in inv.items() if r.get("tier") == 2)
tier2_ids = random.Random(SEED + 10).sample(tier2_pool, min(10, len(tier2_pool)))
targeted_ids = []
for a in AREAS:
    targeted_ids += [i for i in TARGETED.get(a, []) if i not in dropped]

real = []
seen = set()
for stratum, ids in (("random", random_ids), ("tier2", tier2_ids), ("targeted", targeted_ids)):
    for i in ids:
        if i in seen:
            # already in an earlier stratum: flag, do not duplicate
            for item in real:
                if item["orig"] == i:
                    item["also"].append(stratum)
            continue
        seen.add(i)
        real.append(dict(orig=i, stratum=stratum, also=[], sibling=None, kind="real"))
items = real + [dict(orig=d["key"], stratum="decoy", also=[], sibling=d["sibling"], kind="decoy") for d in decoys]

# neutral ids
rng = random.Random(SEED + 99)
order = list(range(len(items)))
rng.shuffle(order)
for n, idx in enumerate(order, 1):
    items[idx]["rid"] = f"K-{n:03d}"

# batches: shuffle, then place each decoy away from its sibling
NB = 12
rng2 = random.Random(SEED + 7)
shuffled = items[:]
rng2.shuffle(shuffled)
batches = [[] for _ in range(NB)]
where = {}
for it in [x for x in shuffled if x["kind"] == "real"]:
    b = min(range(NB), key=lambda k: len(batches[k]))
    batches[b].append(it); where[it["orig"]] = b
for it in [x for x in shuffled if x["kind"] == "decoy"]:
    sib = it["sibling"]
    sibs = sib if isinstance(sib, list) else ([sib] if sib else [])
    avoid = {where[s] for s in sibs if s in where}
    cands = sorted((k for k in range(NB) if k not in avoid),
                   key=lambda k: (sum(1 for x in batches[k] if x["kind"] == "decoy"), len(batches[k]), k))
    batches[cands[0]].append(it); where[it["orig"]] = cands[0]
for b in batches:
    rng2.shuffle(b)

decoy_by_key = {d["key"]: d for d in decoys}
def merits_view(it):
    src = inv[it["orig"]] if it["kind"] == "real" else decoy_by_key[it["orig"]]
    return {"id": it["rid"], **{f: src.get(f) for f in MERITS_FIELDS}}

os.makedirs("batches", exist_ok=True)
for k, b in enumerate(batches, 1):
    with open(f"batches/merits_{k:02d}.jsonl", "w") as f:
        for it in b:
            f.write(json.dumps(merits_view(it), ensure_ascii=False) + "\n")

# basis audit batches: real items only, full inventory fields, original ids
reals = [it for it in items if it["kind"] == "real"]
reals.sort(key=lambda it: it["orig"])
NBA = 8
for k in range(NBA):
    chunk = reals[k::NBA]
    with open(f"batches/basis_{k+1:02d}.jsonl", "w") as f:
        for it in chunk:
            f.write(json.dumps(inv[it["orig"]], ensure_ascii=False) + "\n")

key = {
    "items": [{**{x: it[x] for x in ("rid", "orig", "kind", "stratum", "also", "sibling")},
               "batch": where[it["orig"]] + 1} for it in items],
    "decoys": decoys,
    "dropped_dupes": DROPPED_DUPES,
    "tier2_pool_size": len(tier2_pool),
}
json.dump(key, open("decoy_key.json", "w"), ensure_ascii=False, indent=1)
h = hashlib.sha256(open("decoy_key.json", "rb").read()).hexdigest()
open("decoy_key.sha256", "w").write(h + "  decoy_key.json\n")
print("real", len(reals), "decoys", len(decoys), "items", len(items))
print("strata", {s: sum(1 for it in reals if it["stratum"] == s) for s in ("random", "tier2", "targeted")})
print("batch sizes", [len(b) for b in batches], "decoys per batch", [sum(1 for x in b if x["kind"] == "decoy") for b in batches])
print("key sha256", h)
