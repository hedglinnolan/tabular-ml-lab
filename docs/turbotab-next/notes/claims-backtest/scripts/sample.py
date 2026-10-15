import json, random, sys
AREAS = "ABCD"
def load(a):
    return [json.loads(l) for l in open(f"inventory_{a}.jsonl") if l.strip()]
def draw(a):
    k = AREAS.index(a)
    rows = load(a)
    t1 = sorted(r["id"] for r in rows if r.get("tier") == 1)
    pick = t1 if len(t1) <= 25 else random.Random(20261015 + k).sample(t1, 25)
    return rows, t1, pick
if __name__ == "__main__":
    for a in sys.argv[1:]:
        rows, t1, pick = draw(a)
        print(a, "tier1:", len(t1), "picked:", sorted(pick))
        json.dump(sorted(pick), open(f"sample_{a}.json", "w"))
