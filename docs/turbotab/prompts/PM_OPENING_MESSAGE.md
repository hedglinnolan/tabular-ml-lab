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

## Where things stand

**L64 is adjudicated and accepted** — four and a half of five, with two dispositions downgraded and
the record written. Nothing is unfinished; you are starting on a clean branch.

**`PM_TRANSITION.md` §10 tells you what to write next, and one result should shape it.** The
manuscript validator renders *"13 checks, 0 unmet"* to a user, and **eight of those thirteen are
decided before the manuscript is read** — measured, each one shown to survive a state that should make
it fail. Beside that number the page asserts *"Every consistency check the validator makes is met by
this draft."* That is the governing rule's *assert-something-false* branch on the artifact that leaves
the building, and it is `MISC-029`.
