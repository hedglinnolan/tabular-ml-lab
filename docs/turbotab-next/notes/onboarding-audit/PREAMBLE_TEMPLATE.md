# What a workflow prompt still says

The root `CLAUDE.md` reaches every workflow agent, and `.claude/agents/<role>.md` reaches an agent
whose call names that `agentType`. A prompt repeats none of their rules. It names a rule only when
this task needs one the files do not carry, and it carries what changes from wave to wave:

| Field | Why the files cannot carry it |
|---|---|
| Role and task, one sentence each | the work |
| Base: branch and commit | moves every wave |
| Worktree: path; new from the base, or existing | the harness creates worktrees from `main` |
| Files owned; files other packages or waves hold | changes per package |
| Reading list: paths with § or line ranges | changes per task; bounds the context |
| Task-specific binding rules, named not pasted | only the ones this task can break |
| What is running in parallel on this machine | changes by the hour |
| Stopping rule | the backtest's lesson: set before the first round |
| Pushing: none, or the exact `ci/<name>` branch | authority per wave |
| The two attribution lines | the session line changes per session |
| Report schema, with a `doubts` field | the script reads it |

## Template

```
ROLE: <builder | verifier | integrator | auditor>, <model and effort if not the call's>.
TASK: <one sentence>. Package <id> of <plan path> §<n>.

BASE: <branch> at <sha> or later.
WORKTREE: <path>, <"new: run git checkout -B <branch> <base> first" | "existing, on <branch>">.
OWN: <files>; new test files beside them.
DO NOT TOUCH: <files other packages or waves hold>, <generated files>.
READ: <path §n>, <path lines a–b>, <path>. Nothing else unless a test fails there.
RULES FOR THIS TASK: <only the ones this task can break, by name: e.g. "no estimate before the
  lock"; "the wp7 reference is re-pinned by the orchestrator, do not edit it">.
ALSO RUNNING: <wave or heavy run>; keep runs targeted.
STOP WHEN: <the acceptance evidence named in the plan row>, or after <n> repair rounds; then report.
PUSH: <none | ci/<name> only>.

Commit messages end with exactly:
Co-Authored-By: <model name> <noreply@anthropic.com>
Claude-Session: <this session's link>

REPORT (the schema): branch, commits, files_changed, tests_run, numbers_moved, open_issues,
  doubts (your three most doubtful claims).
```

A repair prompt replaces TASK with the verifier's problem list, each with its evidence, and STOP
WHEN with "every problem fixed, disputed with evidence, or reported out of scope". A design prompt
adds the beats document and names FOUNDATION's sections for the reviewer.
