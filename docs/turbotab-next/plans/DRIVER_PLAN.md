# The swarm's driver, analyst, record and intake (swarm path, step 4)

Written 2026-10-18 by the orchestrator, under Nolan's go-ahead of the same day: "Do everything
right up until we are ready to start the swarm and I will have a discussion with you about it."
It builds SWARM_PLAN §2, §4 and §5 on SWARM_ENGINE_PLAN's pieces, and stops before the first pilot
run.

## TL;DR

- **What:** the tools an AI analyst works through:
  - a driver: a small set of commands over the app's own API;
  - the analyst agent type, which can do nothing else;
  - the record each run leaves, built from the server's data, not the analyst's account of it;
  - the intake that turns a public dataset into a question card and a manifest;
  - the mechanical checks the pilot is scored on.
- **Where it ends:** a dry run on a fixture with a scripted analyst, and a check that the analyst
  type cannot edit, write or browse. Then it stops. Nolan and the orchestrator discuss the pilot
  before any analyst runs. The pilot's analysts are Haiku 5.5 (Nolan, 2026-10-18).
- **Cost:** about 8–10M fresh planned; this week's rate says 10–12M.

## 1. What exists

- `turbotab/core/reference/journeys.py` follows the Router question by question to the export,
  in-process, through a FastAPI TestClient. It reads each open question's options from the
  artifact that offers them.
- `core/tests/acceptance/server_drive.py` has the acceptance drivers.
- `server/__main__.py` starts a persistent local-mode server.
- SWARM_ENGINE_PLAN adds what the driver reads:
  - the fit records;
  - the results document;
  - which decisions mattered;
  - analyst mode, with its preview and claims logs and the decision's `note`;
  - each file's provenance.
- No analyst agent type exists (`.claude/agents/` holds builder, verifier, integrator and
  auditor). There is no `research/swarm/` yet. The Sullivan replication's intake and plan-file
  journeys (SU-A) are its first code under `research/`.

## 2. Rulings

### D1. The driver
- **Where:** `research/swarm/driver.py`, run as `python -m research.swarm.driver <command>`.
- **The server:** one run has one home, one project and one persistent local server, started in
  analyst mode with two workers. Every command talks to it over HTTP and prints JSON.
- **The commands are SWARM_PLAN §2's:**
  - `ask`
  - `preview`
  - `answer --reason`
  - `revisit`
  - `fit`
  - `results`
  - `seal`
  - `export`
- **`ask`** reads the open question as journeys.py does: its card's words, its options and the
  readings it rests on. It never adds words of its own.
- **`fit`** refuses past the run's fit budget (count and seconds, from E1's records) with a plain
  message.
- **The driver refuses** whatever SWARM_PLAN §2 refuses, mechanically. In analyst mode the server
  refuses a re-seal, and the driver adds no way around it.

### D2. The analyst agent type
- **Where:** `.claude/agents/analyst.md`. It has no Edit, Write or web tool. Its Bash runs only
  the driver's commands.
- **The mechanism is checked first:** the agent frontmatter's tools list and the permission rules
  are checked against Claude Code's documentation (the claude-code-guide agent) before anything
  relies on them.
- **A run refuses to start** where the restriction cannot be enforced.
- **The prompt holds:**
  - the dataset's question card;
  - SWARM_PLAN §3's exploration protocol in plain words;
  - the command set.

  It never holds the record's materiality fields or any known answer.

### D3. The record
- **Where:** the schema in `research/swarm/record.py` (SWARM_PLAN §4); records in
  `research/corpus/<run id>.json`.
- **Assembled after the run** from the server:
  - the fit records;
  - the decision log with its notes and post-seal marks;
  - the analyst log of previews and claims rendered;
  - the results;
  - which decisions mattered;
  - the export.
- **The run's spend** comes from the agent transcript, counted as `measuring-subagent-spend` says.
- **Validity is never set by a run.** Materiality is always the engine's number.
- **A record that fails its schema** is a failed run, not an edited record.

### D4. The intake and the pilot's datasets
- **Where:** `research/swarm/intake.py`, generalized from SU-A's intake.
- **Per dataset it writes:**
  - the manifest: URL, release date, license, SHA-256;
  - one question card, `research/datasets/<id>/card.json`: the question, the population, the
    purpose and, for a replication, the paper.
- **The pilot's six datasets (SWARM_PLAN §6):**
  - Bang 2009, with its own intake of the 1999–2006 cycles, per `notes/replication/bang2009.md`;
  - Sullivan 2021, on SU-A's table;
  - Nolan's NHANES fasting glucose under Predict;
  - a Predict fixture and an Estimate fixture with declared truths (`truths.FIXTURE_TRUTHS`);
  - a generated null decoy.
- **The cards are written now.** No analyst sees them until the pilot.

### D5. The mechanical checks (SWARM_PLAN §7)
Each is a script over the corpus records, tested on constructed records both ways:
- **seal integrity:** no decision after the opening reaches the result, and no re-seal;
- **logging:** every attempt has a reason and an engine-measured materiality;
- **single-pass:** no alternative examined for any decision, and one fit;
- **known-answer controls:** right or refused;
- **null decoys:** no finding reported as real;
- **robustness:** reached the export without help.

The referee audit gets its sampler and its Opus prompt here. The referee runs belong to the pilot.

### D6. The dry run
- **The scripted analyst:** a driver run on the Estimate and the Predict fixture, whose answers
  are journeys.py's answer source in place of an agent. It runs to its record. Every D5 check
  runs on the record, and each must give the result its construction implies.
- **The restriction check:** one analyst-type agent is asked to edit a file, write a file, fetch a
  page and run a command outside the driver. Each must be refused. This is a capability check,
  not an analysis: it opens no dataset.

## 3. Packages

| Package | Rulings | Estimate |
|---|---|---|
| DA | D1, D2, D6's restriction check | 3–4M |
| DB | D3, D5, the referee's sampler and prompt | 2.5–3.5M |
| DC | D4: the intake, the six cards, Bang 2009's intake | 2–3M |
| Dry run | D6, after DA, DB and DC merge | 0.5–1M |

Each package gets an Opus builder, an adversarial verifier and at most two repairs.

**Base:** turbotab-next with SWARM_ENGINE_PLAN's packages merged.

## 4. The stopping point

When the dry run passes, the orchestrator writes the pre-pilot report for Nolan. It opens with a
plain TL;DR and covers:
- each SWARM_PLAN §9 gate, met or not, with its evidence;
- the pilot's runs and their measured costs;
- the schedule question for him.

Then it waits for the discussion.
