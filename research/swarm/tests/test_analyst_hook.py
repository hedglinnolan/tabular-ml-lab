"""The analyst's PreToolUse hook (DRIVER_PLAN D2), tested both ways: for an ``analyst`` it denies
every tool but Bash and the report tools and every Bash command that is not exactly the driver's
invocation, and approves the driver's own commands and the report tools, in Claude Code's allow
shape (the owner's ruling of 2026-10-23: the hooks approve, the settings hold no allow rule); for
every other caller it says nothing, whatever the call.

The forbidden forms are the brief's (no ``;``, ``&&``, ``|``, backticks, ``$( )``, redirections or
newlines, nothing but ``scripts/py -m research.swarm.driver ...`` from the repo root) and zsh's, the
shell an analyst's Bash runs on this machine (process substitution ``=(...)``, glob qualifiers
that run code, globs, ``~``, braces, history), written out here as a person would type them, never
read from the hook. Whether a form runs anything is not taken on trust: each allowed form is run
under ``zsh`` (and ``bash``) in a scratch folder whose ``scripts/py`` only writes down the words it
was given, which must be ``shlex``'s words of the command, with nothing else made; and the zsh
forms are shown to run their payload there when no hook stands in the way."""
from __future__ import annotations

import io
import json
import re
import shlex
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from research.swarm import analyst_hook as hook

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "research" / "swarm" / "analyst_hook.py"
DRIVER = "scripts/py -m research.swarm.driver"
AGENT = REPO / ".claude" / "agents" / "analyst.md"


def call(command=None, *, tool="Bash", agent="analyst", cwd=REPO, tool_input=None):
    out = {"session_id": "s", "transcript_path": "/tmp/t.jsonl", "cwd": str(cwd) if cwd else None,
           "hook_event_name": "PreToolUse", "tool_name": tool,
           "tool_input": tool_input if tool_input is not None else {"command": command}}
    if agent is not None:
        out["agent_type"] = agent
    return out


def denied(answer) -> bool:
    return (answer is not None
            and answer["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
            and answer["hookSpecificOutput"]["permissionDecision"] == "deny"
            and bool(answer["hookSpecificOutput"]["permissionDecisionReason"].strip()))


def allowed(answer) -> bool:
    """Claude Code's PreToolUse approval, as its hooks guide documents it ("Hook JSON Output":
    ``hookSpecificOutput`` with ``hookEventName``, ``permissionDecision`` "allow" and
    ``permissionDecisionReason``), and nothing else beside it."""
    if not isinstance(answer, dict) or set(answer) != {"hookSpecificOutput"}:
        return False
    out = answer["hookSpecificOutput"]
    return (set(out) == {"hookEventName", "permissionDecision", "permissionDecisionReason"}
            and out["hookEventName"] == "PreToolUse" and out["permissionDecision"] == "allow"
            and isinstance(out["permissionDecisionReason"], str)
            and bool(out["permissionDecisionReason"].strip()))


def run_script(payload, script=SCRIPT):
    """The hook as Claude Code runs it: the call as JSON on standard input, with no site-packages
    (-S) and isolated from the environment (-I); its exit code and its answer (None for none)."""
    r = subprocess.run([sys.executable, "-S", "-I", str(script)], input=json.dumps(payload),
                       capture_output=True, text=True, timeout=30)
    assert r.stderr == "", r.stderr
    return r.returncode, (json.loads(r.stdout) if r.stdout.strip() else None)


def lookalike(folder: Path) -> Path:
    """A folder laid out as a checkout's root, a worktree's say: a ``.git`` file, its own
    ``scripts/py`` and the driver's module, each copied from this checkout."""
    (folder / "scripts").mkdir(parents=True)
    (folder / "research" / "swarm").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "py", folder / "scripts" / "py")
    shutil.copy(REPO / "research" / "swarm" / "driver.py", folder / "research" / "swarm")
    (folder / ".git").write_text(f"gitdir: {REPO}/.git/worktrees/x\n", "utf-8")
    return folder.resolve()


ALLOWED = [
    f"{DRIVER} ask --run r1",
    f'{DRIVER} answer --run r1 impute --reason "No values are missing, so a fill changes nothing."',
    f"{DRIVER} answer --run r1 linear ridge --reason 'compare plain and penalized'",
    f"{DRIVER} answer --run r1 proposed --set age=excluded --reason 'age is recorded after admission'",
    f"{DRIVER} preview --run r1 complete_case",
    f"{DRIVER} revisit --run r1 0d27d9a3 --reason 'the event level was read the wrong way'",
    f"{DRIVER} fit --run r1",
    f"{DRIVER} results --run r1 --wait 60",
    f"{DRIVER} seal --run r1 linear --reason 'the simpler model scores alike'",
    f"{DRIVER} export --run r1",
    # quoted, a reason may say anything plain words say: an apostrophe, parentheses, a colon
    f"{DRIVER} answer --run r1 ridge --reason \"The model's CV AUC (0.7668) is ridge's: [about] "
    f"the same as plain logistic, so ~ nothing moved? #3 {{a,b}} * ^x =(y) stay words.\"",
    f"{DRIVER} answer --run r1 ridge --reason 'Quoted once: \"ridge\" (penalized) ~ 0.77 * 2? !'",
    f"{DRIVER} answer --run r1 custom --set column=age --set low=-5.5 --set high=95 --reason 'a range'",
    f"{DRIVER} answer --run r1 x-1a2b3c4d --reason \"the way forward's id\"",
    f"{DRIVER} answer --run r1 time:los fixed_effects:site --reason \"a: b, c+d @ 5%\"",
    f"{DRIVER} answer --run r1 proposed --set 'age (y)=excluded' --reason \"(y) is in the name\"",
    f"{DRIVER}\task\t--run\tr1",
    # package 1: a preview's reason, an answer's marker, a want and the account
    f"{DRIVER} preview --run r1 set_selection:elastic_net --reason 'does a penalty keep fewer?'",
    f"{DRIVER} answer --run r1 impute --reason 'a fill keeps the rows' --marker complete_case_share "
    f"--rejected complete_case",
    f"{DRIVER} want --run r1 'drift correction by run order' --kind option --reason 'the QC "
    f"samples drift' --marker run_order",
    f"{DRIVER} account --run r1 \"I kept the linear model: the penalties scored alike.\"",
]

FORBIDDEN = {
    "chained with ;": f"{DRIVER} ask --run r1; cat ~/.ssh/id_rsa",
    "chained with &&": f"{DRIVER} ask --run r1 && rm -rf research",
    "or-chained": f"{DRIVER} ask --run r1 || ls",
    "backgrounded": f"{DRIVER} fit --run r1 &",
    "piped": f"{DRIVER} ask --run r1 | head -5",
    "backticks": f"{DRIVER} answer --run r1 `cat notes/replication/bang2009.md` --reason x",
    "command substitution": f"{DRIVER} answer --run r1 $(ls) --reason x",
    "a variable": f"{DRIVER} answer --run r1 impute --reason $HOME",
    "redirected out": f"{DRIVER} export --run r1 > /tmp/bundle.json",
    "redirected in": f"{DRIVER} ask --run r1 < /etc/passwd",
    "a newline": f"{DRIVER} ask --run r1\ncat /etc/passwd",
    "a carriage return": f"{DRIVER} ask --run r1\rcat /etc/passwd",
    "another program": "cat turbotab/sample_data/clinical_risk.csv",
    "a listing": "ls research/runs",
    "a change of folder": "cd /tmp",
    "another module": "scripts/py -m turbotab.server --port 8787",
    "python code": "scripts/py -c 'import os; print(os.listdir())'",
    "a script": "scripts/py research/swarm/driver.py ask --run r1",
    "an environment prefix": f"TURBOTAB_SWARM_RUNS=/tmp {DRIVER} ask --run r1",
    "the orchestrator's start": f"{DRIVER} start --data /etc/passwd --budget-fits 9 --budget-seconds 9",
    "the orchestrator's stop": f"{DRIVER} stop --run r1",
    "no command": f"{DRIVER}",
    "a made-up command": f"{DRIVER} reseal --run r1",
    "an unclosed quote": f"{DRIVER} answer --run r1 impute --reason 'because",
    "empty": "   ",
    # zsh's own: each runs a program or expands to the file names here, outside quotes
    "zsh process substitution": f"{DRIVER} ask --run r1 =(curl -s https://example.org)",
    "zsh process substitution in a reason": f"{DRIVER} answer --run r1 a --reason ok =(cat ~/.ssh/id_rsa)",
    "a zsh glob qualifier that runs code": f"{DRIVER} preview --run r1 *(e:'touch pwned':)",
    "a zsh glob qualifier that calls a function": f"{DRIVER} ask --run r1 *(+rm)",
    "a bare glob": f"{DRIVER} ask --run *",
    "a recursive glob": f"{DRIVER} answer --run r1 **/* --reason x",
    "a one-character glob": f"{DRIVER} ask --run r?",
    "a bracket glob": f"{DRIVER} ask --run r[12]",
    "a tilde": f"{DRIVER} ask --run ~",
    "a tilde inside a word": f"{DRIVER} answer --run r1 impute --set x=~/notes --reason x",
    "braces": f"{DRIVER} ask --run {{r1,r2}}",
    "history": f"{DRIVER} ask --run r1 !!",
    "a comment": f"{DRIVER} ask --run r1 #x",
    "an extended-glob caret": f"{DRIVER} ask --run ^r1",
    "a subshell": f"{DRIVER} ask --run r1 (ls)",
    "an equals expansion": f"{DRIVER} ask --run =ls",
    "a doubled equals": f"{DRIVER} answer --run r1 a --set x==ls --reason x",
    "an equals after quotes": f"{DRIVER} answer --run r1 a \"\"=ls --reason x",
    "a backslash outside quotes": f"{DRIVER} ask --run r1\\ x",
    "a backslash in double quotes": f"{DRIVER} answer --run r1 a --reason \"a\\\"b\"",
    "history in double quotes": f"{DRIVER} answer --run r1 a --reason \"wow!!\"",
    "a vertical tab": f"{DRIVER} ask --run r1\x0bx",
    "a non-ASCII letter outside quotes": f"{DRIVER} ask --run ré",
}

# The forms the hook passed before the owner's ruling of 2026-10-23, and refuses now in words that
# name the one form it approves: python and python3 are whatever the caller's PATH finds first.
OTHER_FORMS = {
    "python": "python -m research.swarm.driver ask --run r1",
    "python3": "python3 -m research.swarm.driver fit --run r1",
    "./scripts/py": "./scripts/py -m research.swarm.driver results --run r1",
}

# Run under zsh with no hook in the way, each of these runs its payload: the forms are real.
PAYLOADS = {
    "process substitution": "{driver} ask --run r1 =(touch {marker})",
    "a glob qualifier": "{driver} preview --run r1 *(e:'touch {marker}':)",
    "command substitution": "{driver} ask --run r1 $(touch {marker})",
    "backticks": "{driver} ask --run r1 `touch {marker}`",
    "chained": "{driver} ask --run r1; touch {marker}",
}


@pytest.mark.parametrize("command", ALLOWED)
def test_the_drivers_own_commands_are_approved_for_an_analyst(command):
    """Approves where it should: each analyst command, in the one form, from the root, answered
    with Claude Code's allow."""
    assert allowed(hook.decide(call(command)))


@pytest.mark.parametrize("name", sorted(FORBIDDEN))
def test_every_other_command_is_denied_to_an_analyst(name):
    """Fires where it should: each forbidden form of the brief, and every other program."""
    assert denied(hook.decide(call(FORBIDDEN[name]))), name


@pytest.mark.parametrize("first", sorted(OTHER_FORMS))
def test_the_driver_runs_only_as_scripts_py_and_the_refusal_names_that_form(first):
    """Only the main form is approved: the others the hook passed before are denied, each in words
    that name the one form and the word it was given instead."""
    says = hook.decide(call(OTHER_FORMS[first]))
    assert denied(says)
    reason = says["hookSpecificOutput"]["permissionDecisionReason"]
    assert "scripts/py -m research.swarm.driver <command> --run <id>" in reason
    assert f"not {first}" in reason


def claude_md_prefix() -> str:
    """The environment prefix the root CLAUDE.md gives builders, read from it."""
    (prefix,) = re.findall(r"prefix every command with `([^`]+)`",
                           (REPO / "CLAUDE.md").read_text(encoding="utf-8"))
    return prefix


def test_the_environment_prefix_builders_are_given_is_refused_in_words_that_say_so():
    """The root CLAUDE.md tells builders to put a prefix before each command; on an analyst's
    driver command the hook denies it, says it is an environment setting, and names the form."""
    prefix = claude_md_prefix()
    assert prefix.startswith("OMP_NUM_THREADS=")  # the construction: an assignment before a command
    for command in (f"{prefix} {DRIVER} ask --run r1", f"OMP_NUM_THREADS=2 {DRIVER} ask --run r1"):
        says = hook.decide(call(command))
        assert denied(says), command
        reason = says["hookSpecificOutput"]["permissionDecisionReason"]
        assert "environment setting" in reason and "scripts/py -m research.swarm.driver" in reason
    # Silent the other way: an '=' inside a later word is a setting's value, approved.
    assert allowed(hook.decide(call(f"{DRIVER} answer --run r1 a --set low=18 --reason x")))


@pytest.mark.parametrize("cwd", [Path("/tmp"), REPO / "research", REPO.parent, None])
def test_the_driver_runs_only_from_the_repo_root(cwd):
    assert denied(hook.decide(call(f"{DRIVER} ask --run r1", cwd=cwd)))
    assert allowed(hook.decide(call(f"{DRIVER} ask --run r1", cwd=REPO)))


def test_a_lookalike_root_is_not_the_hooks_root(tmp_path):
    """A validated-looking command run from another checkout's root (a worktree's, with its own
    ``scripts/py`` and driver), or from a link to one, is denied, naming the hook's own root; a
    link to the hook's own root names the same folder, and is approved."""
    other = lookalike(tmp_path / "worktree")
    (tmp_path / "to-other").symlink_to(other)
    (tmp_path / "to-repo").symlink_to(REPO)
    for cwd in (other, tmp_path / "to-other"):
        says = hook.decide(call(f"{DRIVER} ask --run r1", cwd=cwd))
        assert denied(says), cwd
        assert str(REPO) in says["hookSpecificOutput"]["permissionDecisionReason"]
    assert allowed(hook.decide(call(f"{DRIVER} ask --run r1", cwd=tmp_path / "to-repo")))


@pytest.mark.parametrize("cwd", [".", "research/..", "", 7, ["/"], {"path": "x"}])
def test_a_folder_that_is_not_an_absolute_path_is_not_the_root(cwd, monkeypatch):
    """A relative folder would be read from wherever the hook runs, so it is never the root, even
    when the hook runs from the root; nor is a folder that is not a path at all."""
    monkeypatch.chdir(REPO)
    given = {**call(f"{DRIVER} ask --run r1"), "cwd": cwd}
    assert denied(hook.decide(given)), cwd


@pytest.mark.parametrize("tool", ["Edit", "Write", "NotebookEdit", "MultiEdit", "Read", "Glob",
                                  "Grep", "WebFetch", "WebSearch", "Agent", "Task",
                                  "mcp__claude_ai_Gmail__send_message",
                                  "mcp__claude-in-chrome__navigate"])
def test_every_tool_but_bash_is_denied_to_an_analyst(tool):
    answer = hook.decide(call(tool=tool, tool_input={"file_path": str(REPO / "x"), "url": "https://x"}))
    assert denied(answer)
    assert tool in answer["hookSpecificOutput"]["permissionDecisionReason"]


@pytest.mark.parametrize("tool", ["SubagentHandback", "StructuredOutput"])
def test_an_analyst_hands_its_report_back(tool):
    """The live check of 2026-10-22 found the harness's report tools denied, so no analyst's
    report reached the orchestrator: an analyst's call of either is approved, whatever its input."""
    given = {"report": "done", "outcome": "exported"}
    assert allowed(hook.decide(call(tool=tool, tool_input=given)))
    assert allowed(hook.decide(call(tool=tool, tool_input={})))


@pytest.mark.parametrize("tool", ["SubagentHandbackX", "subagenthandback", "Structured_Output",
                                  "StructuredOutputs", "SubagentHandback ", ""])
def test_a_tool_named_near_the_report_tools_is_still_denied(tool):
    assert denied(hook.decide(call(tool=tool, tool_input={})))


@pytest.mark.parametrize("agent", [None, "builder", "verifier", "integrator", "auditor",
                                   "general-purpose", "analysts", "Analyst", "analyst-2"])
def test_it_says_nothing_for_any_other_caller(agent):
    """Silent for every other agent type and for the session itself (no ``agent_type``), on the
    very calls it denies or approves an analyst."""
    for command in [*FORBIDDEN.values(), *ALLOWED, *OTHER_FORMS.values()]:
        assert hook.decide(call(command, agent=agent)) is None
    for tool in ("Edit", "Write", "WebFetch", "mcp__x__y", *hook.REPORT_TOOLS):
        assert hook.decide(call(tool=tool, agent=agent, tool_input={})) is None
    assert hook.decide(call(f"{DRIVER} ask --run r1", agent=agent, cwd=Path("/tmp"))) is None


def test_the_hook_answers_on_stdout_as_claude_code_reads_it():
    """main(): a denial or an approval is one JSON object on standard output with exit 0; for
    another caller it prints nothing; input that is not JSON is refused with exit 2, since the
    caller cannot be told."""
    out, err = io.StringIO(), io.StringIO()
    assert hook.main(io.StringIO(json.dumps(call("ls"))), out, err) == 0
    answer = json.loads(out.getvalue())
    assert answer == {"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "deny",
        "permissionDecisionReason": answer["hookSpecificOutput"]["permissionDecisionReason"]}}
    out = io.StringIO()
    assert hook.main(io.StringIO(json.dumps(call(f"{DRIVER} ask --run r1"))), out, err) == 0
    assert allowed(json.loads(out.getvalue()))
    out = io.StringIO()
    assert hook.main(io.StringIO(json.dumps(call("ls", agent="builder"))), out, err) == 0
    assert out.getvalue() == ""
    assert hook.main(io.StringIO("not json {"), io.StringIO(), err) == 2
    assert err.getvalue().strip()


def test_it_runs_as_a_script_on_the_standard_library_alone():
    """Wired as ``python3 <repo>/research/swarm/analyst_hook.py``: run with no site-packages (-S)
    and isolated from the environment (-I), it still denies and approves."""
    rc, answer = run_script(call(f"{DRIVER} ask --run r1; ls"))
    assert rc == 0 and denied(answer)
    rc, answer = run_script(call(f"{DRIVER} ask --run r1"))
    assert rc == 0 and allowed(answer)


# The calls of the audit of 2026-10-23 (its cases.py), each whole as Claude Code sends it, with
# what this hook answers: "allow", "deny", or None for no answer at all.
AS_SENT = [
    ("an analyst's driver ask", "analyst", "Bash", {"command": f"{DRIVER} ask --run RUN"}, "allow"),
    ("an analyst's report", "analyst", "SubagentHandback", {"message": "done"}, "allow"),
    ("an analyst's structured report", "analyst", "StructuredOutput", {"x": 1}, "allow"),
    ("an analyst's tools call", "analyst", "Bash",
     {"command": "scripts/py -m research.swarm.claim_cycle.tools index cycle/corpus"}, "deny"),
    ("an analyst's chained call", "analyst", "Bash", {"command": f"{DRIVER} ask --run RUN; ls"},
     "deny"),
    ("an analyst's Read", "analyst", "Read", {"file_path": f"{REPO}/x"}, "deny"),
    ("an analyst's driver fit in the background", "analyst", "Bash",
     {"command": f"{DRIVER} fit --run RUN", "description": "d", "run_in_background": True}, "deny"),
    ("an analyst's driver fit in the foreground", "analyst", "Bash",
     {"command": f"{DRIVER} fit --run RUN", "description": "d", "run_in_background": False},
     "allow"),
    ("a builder's driver fit in the background", "builder", "Bash",
     {"command": f"{DRIVER} fit --run RUN", "run_in_background": True}, None),
    ("an analyst's driver fit out of the sandbox", "analyst", "Bash",
     {"command": f"{DRIVER} fit --run RUN", "dangerouslyDisableSandbox": True}, "deny"),
    ("an analyst's driver fit in the sandbox", "analyst", "Bash",
     {"command": f"{DRIVER} fit --run RUN", "dangerouslyDisableSandbox": False}, "allow"),
    ("a builder's driver fit out of the sandbox", "builder", "Bash",
     {"command": f"{DRIVER} fit --run RUN", "dangerouslyDisableSandbox": True}, None),
    ("an analyst's Bash input as a string", "analyst", "Bash", f"{DRIVER} ask --run RUN", "deny"),
    ("an analyst's Bash input as a list", "analyst", "Bash", [f"{DRIVER} ask --run RUN"], "deny"),
    ("a builder's git status", "builder", "Bash", {"command": "git status --short"}, None),
    ("a builder's driver start", "builder", "Bash",
     {"command": f"{DRIVER} start --data x.csv --budget-fits 1 --budget-seconds 1"}, None),
    ("a builder's driver ask", "builder", "Bash", {"command": f"{DRIVER} ask --run RUN"}, None),
    ("a verifier's driver ask", "verifier", "Bash", {"command": f"{DRIVER} ask --run RUN"}, None),
    ("the session's ls", None, "Bash", {"command": "ls"}, None),
    ("the session's driver ask", None, "Bash", {"command": f"{DRIVER} ask --run RUN"}, None),
    ("the session's Agent", None, "Agent",
     {"subagent_type": "analyst", "prompt": "p", "description": "d"}, None),
]


@pytest.mark.parametrize("label, agent, tool, given, expect", AS_SENT, ids=[c[0] for c in AS_SENT])
def test_each_call_as_claude_code_sends_it(label, agent, tool, given, expect):
    """Run as Claude Code runs it, each call whole on standard input: approved, denied, or no
    answer at all, as its caller and its call say."""
    payload = {"session_id": "s", "hook_event_name": "PreToolUse", "tool_name": tool,
               "tool_input": given, "cwd": str(REPO)}
    if agent is not None:
        payload["agent_type"] = agent
    rc, answer = run_script(payload)
    assert rc == 0
    if expect is None:
        assert answer is None
    else:
        assert (allowed if expect == "allow" else denied)(answer), answer


def test_a_copy_of_the_hook_approves_only_from_its_own_root(tmp_path):
    """The root is the one the hook's own file lies in: a copy of it in another checkout approves
    a command run from that checkout's root, and denies the same command run from this one."""
    other = lookalike(tmp_path / "other")
    script = other / "research" / "swarm" / "analyst_hook.py"
    shutil.copy(SCRIPT, script)
    rc, answer = run_script(call(f"{DRIVER} ask --run r1", cwd=other), script)
    assert rc == 0 and allowed(answer)
    rc, answer = run_script(call(f"{DRIVER} ask --run r1", cwd=REPO), script)
    assert rc == 0 and denied(answer)
    assert str(other) in answer["hookSpecificOutput"]["permissionDecisionReason"]


def agent_examples() -> list[str]:
    """The commands ``.claude/agents/analyst.md`` shows an analyst, one per line."""
    return [line.strip() for line in AGENT.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith(f"{DRIVER} ") and "<command>" not in line]


def test_each_example_the_analyst_is_shown_is_approved_as_claude_code_runs_the_hook():
    examples = agent_examples()
    assert len(examples) == len(hook.ANALYST_COMMANDS)  # one each (test_analyst_agent.py)
    for example in examples:
        assert allowed(hook.decide(call(example))), example
        rc, answer = run_script(call(example))
        assert rc == 0 and allowed(answer), example


def test_the_analyst_is_told_the_builders_prefix_is_not_its_own():
    """``analyst.md`` says the root CLAUDE.md's prefix is not the analyst's and that the hook
    refuses it, and the prefix it names is the one CLAUDE.md gives."""
    text = " ".join(AGENT.read_text(encoding="utf-8").split())
    assert (f"The environment prefix the root CLAUDE.md gives builders (`{claude_md_prefix()}`) is "
            f"not yours: the hook refuses a command with any setting before it.") in text


# The form an analyst's refusals name, as they read before the claim cycle's hook passed its own
# words (the re-audit of 2026-10-23), written out here.
TODAYS_FORM = ("scripts/py -m research.swarm.driver <command> --run <id> ..., run from the repo "
               "root, where <command> is one of ask, preview, answer, revisit, fit, results, seal, "
               "export, want, account")
TODAYS_REFUSALS = {
    f"{DRIVER} ask --run r1 | head -5":
        f"The command holds '|', which an analyst's command never holds, in a reason either: one "
        f"driver command at a time, nothing chained, substituted or redirected. The form is "
        f"{TODAYS_FORM}.",
    f"{DRIVER} ask --run r1\x0bx":
        f"The command holds the character U+000B, a control character. The form is {TODAYS_FORM}.",
    f"{DRIVER} answer --run r1 a --reason \"a\\\"b\"":
        "The command holds a backslash inside double quotes, which a shell still reads there; "
        "leave it out, or put the words in single quotes.",
    f"{DRIVER} ask --run *":
        "The command holds '*' outside quotes, which a shell can expand or run. Outside quotes a "
        "word holds only letters, digits and . _ - / : , @ % +, and = inside a word; put anything "
        "else, such as a reason, in double or single quotes.",
    f"{DRIVER} answer --run r1 impute --reason 'because":
        f"The command does not read as one driver command (an unclosed quote?). The form is "
        f"{TODAYS_FORM}.",
}


@pytest.mark.parametrize("command", list(TODAYS_REFUSALS))
def test_the_analysts_shell_refusals_read_as_they_did(command):
    """Each refusal of the shell's rules, one per kind, word for word as it read before the claim
    cycle's hook passed its own subject and command's name into them."""
    says = hook.decide(call(command))
    assert denied(says)
    assert says["hookSpecificOutput"]["permissionDecisionReason"] == TODAYS_REFUSALS[command]


# A Bash call's input as Claude Code sends it, beside the command (its Bash tool's own fields).
BASH_FIELDS = {"description": "Ask the open question", "timeout": 120000}


@pytest.mark.parametrize("background", [True, 1, "true"])
def test_a_driver_command_run_in_the_background_is_denied_to_an_analyst(background):
    """Fires where it should: an approved command with a truthy ``run_in_background`` would run
    while the analyst goes on, so it is denied, in words that say one command at a time, in the
    foreground, and name the field to leave out."""
    for command in (f"{DRIVER} ask --run r1", f"{DRIVER} fit --run r1"):
        given = {"command": command, **BASH_FIELDS, "run_in_background": background}
        says = hook.decide(call(tool_input=given))
        assert denied(says), command
        reason = says["hookSpecificOutput"]["permissionDecisionReason"]
        assert reason.startswith("An analyst runs one driver command at a time, in the foreground")
        assert "run_in_background" in reason


@pytest.mark.parametrize("given", [{}, {"run_in_background": False},
                                   {"run_in_background": None}],
                         ids=["absent", "false", "null"])
def test_a_driver_command_run_in_the_foreground_is_approved_as_before(given):
    """Silent where it should be: each allowed command, with ``run_in_background`` absent, false or
    null and the Bash tool's other fields beside it, is approved."""
    for command in ALLOWED:
        assert allowed(hook.decide(call(tool_input={"command": command, **BASH_FIELDS, **given})))


def test_a_background_call_of_any_other_caller_hears_nothing():
    given = {"command": f"{DRIVER} ask --run r1", "run_in_background": True}
    for agent in (None, "builder", "theorist"):
        assert hook.decide(call(tool_input=given, agent=agent)) is None


@pytest.mark.parametrize("unsandboxed", [True, 1, "true"])
def test_a_driver_command_let_out_of_the_sandbox_is_denied_to_an_analyst(unsandboxed):
    """Fires where it should: an approved command with a truthy ``dangerouslyDisableSandbox``
    would run outside the sandbox, so it is denied, in words that say so and name the field to
    leave out."""
    for command in (f"{DRIVER} ask --run r1", f"{DRIVER} fit --run r1"):
        given = {"command": command, **BASH_FIELDS, "dangerouslyDisableSandbox": unsandboxed}
        says = hook.decide(call(tool_input=given))
        assert denied(says), command
        reason = says["hookSpecificOutput"]["permissionDecisionReason"]
        assert reason.startswith("An analyst never runs a driver command outside the sandbox")
        assert "dangerouslyDisableSandbox" in reason


@pytest.mark.parametrize("given", [{"dangerouslyDisableSandbox": False},
                                   {"dangerouslyDisableSandbox": None},
                                   {"dangerouslyDisableSandbox": False,
                                    "run_in_background": False}],
                         ids=["false", "null", "both false"])
def test_a_driver_command_left_in_the_sandbox_is_approved_as_before(given):
    """Silent where it should be: each allowed command, with ``dangerouslyDisableSandbox`` false or
    null (absent is the foreground test's), is approved."""
    for command in ALLOWED:
        assert allowed(hook.decide(call(tool_input={"command": command, **BASH_FIELDS, **given})))


def test_an_unsandboxed_call_of_any_other_caller_hears_nothing():
    given = {"command": f"{DRIVER} ask --run r1", "dangerouslyDisableSandbox": True}
    for agent in (None, "builder", "theorist"):
        assert hook.decide(call(tool_input=given, agent=agent)) is None


@pytest.mark.parametrize("given", ["cat x", f"{DRIVER} ask --run r1", [f"{DRIVER} ask --run r1"],
                                   7, 3.5, True, "", [], 0],
                         ids=["a string", "the driver's words", "a list", "a number",
                              "a float", "true", "empty string", "empty list", "zero"])
def test_a_tool_input_that_is_not_an_object_is_denied_in_words(given):
    """Fires where it should: a Bash call whose input is not an object holds no command the hook
    can check, so it is denied in words (it raised before, and the hook's script exited 1), as the
    claim cycle's hook denies it; in the script too, with exit 0 and nothing on standard error."""
    says = hook.decide(call(tool_input=given))
    assert denied(says), given
    assert "An analyst runs only the driver's commands" in (
        says["hookSpecificOutput"]["permissionDecisionReason"])
    rc, answer = run_script(call(tool_input=given))
    assert rc == 0 and denied(answer), given
    # Silent the other way: the same command in an object is approved as before.
    assert allowed(hook.decide(call(tool_input={"command": f"{DRIVER} ask --run r1"})))


# The ASCII control characters, as an independent library classes them: Unicode's Cc below 128.
ASCII_CONTROLS = [chr(i) for i in range(128) if unicodedata.category(chr(i)) == "Cc"]


def test_the_analyst_is_told_which_control_characters_are_refused():
    """``analyst.md`` says what the hook does: a tab is approved, between words or in quotes, and
    every other ASCII control character is refused anywhere, even in quotes. A control character
    beyond ASCII (U+0085, of Unicode's Cc) is not one of those, and in quotes it is approved, so
    the words say ASCII."""
    assert len(ASCII_CONTROLS) == 33  # the construction: 0 to 31, and 127
    text = " ".join(AGENT.read_text(encoding="utf-8").split())
    assert ("`$` `<` `>`, a line break or an ASCII control character other than a tab, anywhere, "
            "even in quotes;") in text
    for c in ASCII_CONTROLS:
        for command in (f'{DRIVER} account --run r1 "a{c}b"', f"{DRIVER} account --run r1 'a{c}b'",
                        f"{DRIVER} ask{c}--run r1"):
            expect = allowed if c == "\t" else denied
            assert expect(hook.decide(call(command))), (repr(c), command)
    assert unicodedata.category("\x85") == "Cc"
    assert allowed(hook.decide(call(f'{DRIVER} account --run r1 "a\x85b"')))


# ── what a shell makes of a command the hook lets through ────────────────────
#
# Claude Code runs a Bash command with the user's shell, through ``eval`` (its errors read
# ``(eval):1: ...``); on this machine that shell is zsh. Each shell here is run on no startup
# files, once more with the zsh options that widen what a word expands to.

SHELLS = {
    "zsh": ["zsh", "-f", "-c", 'eval "$1"', "zsh"],
    "zsh, extended": ["zsh", "-f", "-c", 'setopt extended_glob magic_equal_subst; eval "$1"', "zsh"],
    "bash": ["bash", "--norc", "--noprofile", "-c", 'eval "$1"', "bash"],
}
FAKE_PY = '#!/bin/sh\nprintf "%s\\0" "$@" > "$(dirname "$0")/../argv.bin"\n'


def scratch(tmp_path: Path) -> Path:
    """A folder whose ``scripts/py`` only writes down the words it was given, beside files a glob
    would match."""
    folder = tmp_path / "shell"
    (folder / "scripts").mkdir(parents=True)
    py = folder / "scripts" / "py"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    for name in ("r1", "r2", "notes"):
        (folder / name).write_text("x")
    return folder


def files(folder: Path) -> set[str]:
    return {str(p.relative_to(folder)) for p in folder.rglob("*")}


def in_shell(shell: str, command: str, folder: Path) -> list[str] | None:
    """The words ``scripts/py`` was given when ``shell`` ran ``command`` in ``folder``, or None
    when it was not run."""
    (folder / "argv.bin").unlink(missing_ok=True)
    subprocess.run([*SHELLS[shell], command], cwd=folder, capture_output=True, timeout=30,
                   env={"PATH": "/usr/bin:/bin", "HOME": str(folder)})
    try:
        return (folder / "argv.bin").read_bytes().decode().split("\0")[:-1]
    except OSError:
        return None


ON_DISK = [c for c in ALLOWED if c.split()[0] == "scripts/py"]


@pytest.mark.parametrize("shell", [s for s in SHELLS if shutil.which(SHELLS[s][0])])
def test_a_command_the_hook_approves_runs_the_driver_alone_with_its_words_unchanged(tmp_path,
                                                                                    shell):
    """Approves where it should, and the approval is safe: each allowed command, run by the shell,
    hands ``scripts/py`` exactly the words ``shlex`` reads (nothing expanded, nothing run), and
    leaves no file behind but the one the fake program writes."""
    folder = scratch(tmp_path)
    before = files(folder)
    assert ON_DISK == ALLOWED
    for command in ON_DISK:
        assert allowed(hook.decide(call(command))), command
        assert in_shell(shell, command, folder) == shlex.split(command)[1:], command
        assert files(folder) - before == {"argv.bin"}, command


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh is not installed here")
@pytest.mark.parametrize("name", sorted(PAYLOADS))
def test_each_zsh_payload_runs_without_the_hook_and_is_denied_with_it(tmp_path, name):
    """The forms are real: run by zsh with nothing in the way, each makes its marker file. And the
    hook denies each one to an analyst."""
    folder = scratch(tmp_path)
    marker = tmp_path / f"ran-{name.replace(' ', '-')}"
    command = PAYLOADS[name].format(driver=DRIVER, marker=marker)
    in_shell("zsh", command, folder)
    assert marker.exists(), command
    assert denied(hook.decide(call(command))), command


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh is not installed here")
def test_through_the_hook_nothing_runs_but_the_driver(tmp_path):
    """The hook in the way, as Claude Code puts it: every command, allowed or not, is run by zsh
    only when the hook does not deny it; afterwards no payload has run."""
    folder = scratch(tmp_path)
    markers = []
    for name, form in PAYLOADS.items():
        marker = tmp_path / f"through-{name.replace(' ', '-')}"
        markers.append(marker)
        for shell in ("zsh", "zsh, extended"):
            command = form.format(driver=DRIVER, marker=marker)
            if not denied(hook.decide(call(command))):
                in_shell(shell, command, folder)
    for command in ON_DISK:
        if not denied(hook.decide(call(command))):
            in_shell("zsh, extended", command, folder)
    assert [m.name for m in markers if m.exists()] == []


def test_its_command_list_is_the_drivers():
    from research.swarm import driver

    assert hook.ANALYST_COMMANDS == driver.ANALYST_COMMANDS
    assert set(driver.COMMANDS) - set(hook.ANALYST_COMMANDS) == {"start", "stop"}
