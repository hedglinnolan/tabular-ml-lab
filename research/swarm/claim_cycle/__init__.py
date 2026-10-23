"""The claim cycle's tools (SWARM_PLAN §9; the build's package 6d): what the theorist and the
experimentalist are given, what a proposal is, and what the cycle reports.

* ``theorist.md`` and ``experimentalist.md``: the two agents' prompts, files the orchestrator
  passes to an agent of each type;
* ``proposal.py``: the structured proposal, its schema and its validator, and the proposals file;
* ``overlap.py``: the prior-driven share, matching two corpora's proposals;
* ``view.py``: the reading restriction, the only fields of a record the two agents read;
* ``tools.py``: the commands the two agents run, each reading through the view;
* ``hook.py``: the PreToolUse hook that holds each agent to its commands.
"""
