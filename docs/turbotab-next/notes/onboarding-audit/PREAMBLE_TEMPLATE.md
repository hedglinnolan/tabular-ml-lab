# What a workflow prompt still says

The root `CLAUDE.md` reaches every workflow agent, and `.claude/agents/<role>.md` reaches an agent
whose call names that `agentType`. A prompt repeats none of their rules. It carries what changes
from task to task, and every field the role's file says the prompt must name: an agent missing one
returns at once instead of guessing.

| Field | Who needs it | Why the files cannot carry it |
|---|---|---|
| ROLE, TASK | all | the work |
| BASE: branch and commit | builder, verifier | moves every wave |
| BRANCH: the agent's own | builder, integrator | the agent must not invent one |
| WORKTREE: path, new or existing | all | the harness creates worktrees from `main` |
| OWN and DO NOT TOUCH | builder, verifier | changes per package |
| PREMISE: where the task came from | builder, auditor | the agent checks it; name the INBOX line, the fix id, and how any number in it was measured |
| READ: binding specs with § or line ranges | all | bounds the context; the verifier judges against them |
| BUILDER'S REPORT | verifier | its tests and its three doubts |
| ALSO RUNNING | all | changes by the hour |
| STOP WHEN | all | set before the first round |
| PUSH | builder, integrator | authority per wave |
| WEB | auditor, any agent that may fetch | which hosts, if any |
| ATTRIBUTION | omit | the harness gives each workflow agent its own attribution lines (checked 2026-10-16) |
| REPORT: the schema, with a `doubts` field | all | the script reads it |

## Template

```
ROLE: <builder | verifier | integrator | auditor>.
TASK: <one sentence>. Package <id> of <plan path> §<n>.

BASE: <branch> at <sha>.
BRANCH: <your branch>.
WORKTREE: <path>; <"new: git checkout -B <branch> <base> first" | "existing, on <branch>">.
OWN: <files>, and their tests.
DO NOT TOUCH: <files other packages or waves hold>.
PREMISE: <INBOX line or fix id, path:line>; <how its number was measured: script, seed, sizes>.
READ: <path §n>, <path lines a–b>. Nothing else unless a test fails there.
RULES FOR THIS TASK: <only those this task can break, named, not paraphrased: e.g. "FOUNDATION §5
  rule 6: no estimate before the lock or Fit">.
ALSO RUNNING: <wave or heavy run, or "nothing">.
STOP WHEN: <the acceptance evidence the plan row names>; at most <2> repair rounds, then report.
PUSH: <none | ci/<name> only>.
WEB: <none | allowed for <hosts>; never the user's email or any personal identifier>.
REPORT (the schema): <fields, ending with doubts: your three most doubtful claims>.
```

A verifier's prompt names the builder's worktree as WORKTREE ("existing, read-only"), the base
commit as BASE, and adds BUILDER'S REPORT. An integrator's prompt gives the package branches in
merge order as its TASK, each verifier's problem list, and its `ci/<name>` branch as PUSH. A repair
prompt replaces TASK with the verifier's problem list, each problem with its evidence, and STOP WHEN
with "every problem fixed, disputed with evidence, or reported out of scope". A design prompt adds
the beats document to READ and names FOUNDATION's sections for the reviewer.
