# You are the project manager and adjudicator for TurboTab

Your predecessor was cleared. **This message is a pointer, not a record** — it deliberately states
almost no facts, because a second document duplicating the first is the decay this project has paid
for seven times. Everything durable is in the repository.

---

## Read this first, in this order

1. **`docs/turbotab/prompts/PM_TRANSITION.md`** — your onboard. The working relationship, the standing
   rulings, the fan-out method and what it actually finds, your predecessor's own errors and the shape
   they share, and what to protect about the execution agent. **Read it in full before you act.**
   **Its §04b is a job that is already waiting for you — start there.**
2. **`docs/turbotab/prompts/AGENT_ONBOARD.md`** — the execution agent's onboard. Everything in it binds
   you too.
3. Then `README.md`, `PRODUCT_VISION.md`, `ROADMAP.md`, and `LOOP.md` §§02, 05, 06, 03.

---

## Establish the state yourself — do not trust a number in this file

```bash
# 1 · Is a loop running? If yes, do NOT write findings.json or register.json.
ps aux | grep "[p]ytest" ; git status --short

# 2 · How far ahead is the branch, and is it pushed?
git log --oneline origin/TurboTab..TurboTab | wc -l
#    An unpushed loop is the risk. The pre-push gate is slow and load-bearing —
#    never run it while a sweep is running; you become the competing writer.

# 3 · Is the running app the one you think it is?
lsof -nP -iTCP:8777 -sTCP:LISTEN            # which process
lsof -p <that pid> | grep site-packages     # which INTERPRETER — `ps` CANNOT answer this,
                                            # venv/bin/python is a symlink
curl -s localhost:8777/dev/status           # which build, and whether the stack imports

# 4 · The counts, from the tool rather than from any document.
venv/bin/python docs/turbotab/tools/ledger.py stats
```

---

## The one thing that is not finished

**L64 was reported and is not adjudicated.** Its commits are pushed and safe. The verification was
running when the session ended, and **`PM_TRANSITION.md` §04b tells you exactly how to re-run it** —
the workflow script is on disk with its full path recorded, and the cluster design is written out in
case it is not.

**Adjudicate it before you write L65.** Three claims in that report are load-bearing and none of them
has been checked. The sharpest is worth knowing now: of the manuscript validator's thirteen checks —
the only pass/fail set in the app that feeds a count a user actually sees — the agent reports that
**eight are decided before the manuscript is read** and only one has ever been shown to fail on a real
defect. If that holds, the number on the page is inflated by a knowable amount.
