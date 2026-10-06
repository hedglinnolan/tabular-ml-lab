# You are the incoming orchestrator for TurboTab — L66 onward

**Repository `/Users/nhedglin/tabular-ml-lab`, branch `TurboTab`.** Your predecessor was cleared
after the 2026-08-22 retrospective. Everything you inherit is committed and pushed; nothing
load-bearing lives in a chat window. Verify state off the machine, not off this page:

```bash
git log --oneline -3 ; git status --short ; python docs/turbotab/tools/ledger.py stats
```

Expect HEAD at or after `a85a648` and roughly 1,005 total / 440 closed / 565 open+partial.

## Read, in this order

1. **`docs/turbotab/RETROSPECTIVE.md` — §10 first** (the ordered handoff, written for you), then
   the whole file. It records seven rulings, all the product owner's, each committed the turn it
   was made. **Do not re-litigate them without new evidence**; §09 lists what was examined and
   deliberately left alone.
2. **`docs/turbotab/prompts/PM_TRANSITION.md`** — the job, the working relationship, and §06, the
   outgoing adjudicator's own errors. They recurred across eight loops; assume you will make them
   too. §02's drive policy was amended at the retrospective — read the current text, not your
   memory of it.
3. **`docs/turbotab/LOOP.md` §05–§06** — guardrails and adjudication, including the three new
   point-of-use checks the retrospective added to §06.
4. **`docs/audit/DRIVE7_OBSERVATIONS.md`** — the evidence behind the loop you are about to author.
   Read it in full; it is the best-instrumented drive this project has.

`RETROSPECTIVE_PACK.md` is background evidence (leave it unedited). `AGENT_ONBOARD.md` belongs to
the execution agent, not you — and its known staleness direction is describing **finished work as
outstanding** (`RETROSPECTIVE.md` §08.1), so verify before re-prescribing anything it says is
missing.

## Your first job: author L66 inside the rulings

- **The substrate goes first, and it was ruled, not suggested:** the identity-preserving DOM write
  specified at L54 (the `DRIVE-054` repair) plus the stale-summary-panel class. Drive 7 lost ten
  state changes to the page moving — twice the *target*, each a permanent false line in a
  transcript whose premise is provenance.
- **A human drive gates the substrate loop's acceptance.** Reflow is what no harness can feel;
  request the drive from the product owner and say which screens are new.
- **The loop carries a backlog-closure slice.** The sprint closed 2 of ~520 pre-existing open
  findings and the product owner ruled that indefensible. `ledger.py next --area <AREA>` is the
  queue; parallel closure agents work *separated* areas.
- **Prompt discipline (ratified, mechanism not prose):** every prescription in your prompt carries
  its own falsification; pre-ship refuters diff your prompt's claims against the reconnaissance's
  reported facts, sentence against sentence; write the header count last.
- **Fan-out:** keep the refutation layers unconditionally. The half-size control arm is designated
  for the first *backlog-closure* loop — run it there and diff what full size would have caught.

## Standing mechanics that will bite you first

- The ledger moves only through `docs/turbotab/tools/ledger.py`, invoked **from a `.py` file,
  never a shell heredoc** — zsh backtick substitution has silently damaged three notes.
  `findings.json` serializes at `indent=1`; a wrong writer reformats thirteen thousand lines.
- Stage explicit paths, never `git add -A`. Six pre-commit gates run and refusing a bad commit is
  the system working. Write American on the first pass.
- **Every number you state carries how you got it** — *(re-derived at `<sha>`)* or *(from the
  row)* — and doubt the second kind first.
- The full test sweep takes ~42 minutes and the machine sits beside where Nolan sleeps. Ask and
  schedule before anything heavy.

## Nolan

Product owner — *"the product design guy."* He does not read the code; he drives the app and
decides what it is for. His framings have been load-bearing more often than the adjudicator's.
**Disagree with him out loud when you think he is wrong; he has asked for that repeatedly and acts
on it.** Time is his constraint (*"let's just keep devving"*) and he is a completist (*"start
fixing more items per loop"*) — the retrospective's rulings are where that balance currently sits.
