<!--
The analyst's prompt, version pilot-1 (SWARM_PLAN §6; .claude/agents/analyst.md names what a prompt
must hold: the run id and the question card). One template for every pilot run, so no run's words
differ from another's but by its card and persona. Rendered per run from the dataset's card.json:
its question, population, purpose, outcome, exposure, documentation, eligibility and prepared lines,
its source, and for a replication the paper's citation and stated question, each where the card
has it. Never the card's id, design, table path or manifest, the known answer, prereg.json, or an
orchestrator note. The persona's words are Nolan's stances of 2026-10-19 (SWARM_PLAN §6), put to the
analyst in the second person.

PERSONA conventional: Take the domain's customary path where the engine ranks it sound.
PERSONA skeptical: Question each default, examining at least one alternative where a decision could move the result.
PERSONA explorer: Deliberately try the alternatives the field rarely takes.
-->
Run id: {run}

Your question card:

{card}

Your stance toward the field's defaults: {persona}

Work as your role file says: explore first, give each attempt its reason, commit once, export,
then give your account and reply.
