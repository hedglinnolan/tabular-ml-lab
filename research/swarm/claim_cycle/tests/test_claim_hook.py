"""The claim cycle's PreToolUse hook (``claim_cycle/hook.py``): a theorist or an experimentalist
runs the claim cycle's tools and nothing else, so it reads records only through the view, and never
the sealed file by any other tool; and it names only the pinned cycle's corpus folder and proposals
file (``claim_cycle/pin.py``), so no agent reads both the corpus and its shuffled copy. Each rule
both ways: what it approves, in Claude Code's allow shape (the owner's ruling of 2026-10-23: the
hooks approve, the settings hold no allow rule), and what it denies; and it says nothing for any
other agent."""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research.swarm import shuffle as S
from research.swarm.claim_cycle import hook as H
from research.swarm.claim_cycle import pin as N
from research.swarm.claim_cycle import tools as T
from research.swarm.claim_cycle.tests.constructed import GOOD
from research.swarm.tests import test_shuffle as TS
from research.swarm.tests.test_analyst_hook import allowed, claude_md_prefix

REPO = H.REPO
TOOLS = "scripts/py -m research.swarm.claim_cycle.tools"
PROPOSE = f"{TOOLS} propose --out cycle/proposals.jsonl '{json.dumps(GOOD)}'"
AGENTS = REPO / ".claude" / "agents"


def pin_at(path: Path, corpus: str | Path = "cycle/a",
           proposals: str | Path = "cycle/proposals.jsonl", repo: Path = REPO) -> Path:
    """A pin at ``path`` naming ``corpus`` and ``proposals`` (read from ``repo``), written as
    ``pin.write`` writes one, whether or not the two exist."""
    path.write_text(json.dumps(N.body(corpus, proposals, repo)), "utf-8")
    return path


@pytest.fixture(autouse=True)
def pinned(tmp_path, monkeypatch) -> Path:
    """Every test here runs under a pin naming ``cycle/a`` and ``cycle/proposals.jsonl``, the
    paths its commands name, never the repo's own pin."""
    path = pin_at(tmp_path / "pin.json")
    monkeypatch.setattr(H, "PIN", path)
    return path


def call(agent: str | None, tool: str = "Bash", command: str | None = None,
         cwd: str | None = str(REPO), **tool_input) -> dict:
    out = {"tool_name": tool, "tool_input": {**({"command": command} if command else {}),
                                             **tool_input}, "cwd": cwd}
    if agent is not None:
        out["agent_type"] = agent
    return out


def denied(answer: dict | None) -> str:
    assert answer is not None
    out = answer["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    return out["permissionDecisionReason"]


ALLOWED = {
    "theorist": [f"{TOOLS} index cycle/a", f"{TOOLS} records cycle/a --run run_01-ab",
                 f"{TOOLS} records cycle/a --fields regime.markers,decisions.kind",
                 f"{TOOLS} proposals cycle/proposals.jsonl", PROPOSE],
    "experimentalist": [f"{TOOLS} index cycle/a", f"{TOOLS} records cycle/a --dataset dataset_2",
                        f"{TOOLS} proposals cycle/proposals.jsonl"],
}


@pytest.mark.parametrize("agent, command", [(a, c) for a, cs in ALLOWED.items() for c in cs])
def test_the_tools_commands_of_its_role_are_approved(agent, command):
    assert allowed(H.decide(call(agent, command=command)))


@pytest.mark.parametrize("agent", list(ALLOWED))
@pytest.mark.parametrize("tool", ["Read", "Write", "Edit", "Glob", "Grep", "WebFetch", "Agent",
                                  "NotebookEdit", "mcp__claude_ai_Claude_Docs__read"])
def test_every_other_tool_is_denied(agent, tool):
    assert tool in denied(H.decide(call(agent, tool, file_path="/tmp/sealed/s.json")))


@pytest.mark.parametrize("agent", list(ALLOWED))
@pytest.mark.parametrize("tool", H.REPORT_TOOLS)
def test_its_report_tools_are_approved(agent, tool, monkeypatch, tmp_path):
    """Whatever their input, and with no cycle pinned too, since that is when an agent stops and
    says so."""
    assert allowed(H.decide(call(agent, tool)))
    assert allowed(H.decide(call(agent, tool, report="done", outcome="stuck")))
    monkeypatch.setattr(H, "PIN", tmp_path / "none.json")
    assert allowed(H.decide(call(agent, tool)))


DENIED = {
    "reading a file": "cat /tmp/sealed/s.json",
    "the driver": "scripts/py -m research.swarm.driver ask --run run_01-ab",
    "another module": "scripts/py -m research.swarm.shuffle cycle/a --out x --sealed y",
    "python itself": "scripts/py -c 'print(1)'",
    "a chained command": f"{TOOLS} index cycle/a; cat /tmp/s.json",
    "a substitution": f"{TOOLS} records $(cat /tmp/path)",
    "a redirection": f"{TOOLS} index cycle/a > /tmp/out",
    "no command": TOOLS,
    "a command the tools lack": f"{TOOLS} sealed cycle/a",
}


@pytest.mark.parametrize("agent", list(ALLOWED))
@pytest.mark.parametrize("command", list(DENIED.values()), ids=list(DENIED))
def test_anything_but_its_tools_is_denied(agent, command):
    assert denied(H.decide(call(agent, command=command)))


def test_an_experimentalist_proposes_nothing():
    says = denied(H.decide(call("experimentalist", command=PROPOSE)))
    assert "propose" in says and "experimentalist" in says


def lookalike(folder: Path) -> Path:
    """A folder laid out as a checkout's root, a worktree's say: a ``.git`` file, its own
    ``scripts/py``, the tools' module, and the pinned paths' folder, each copied from this one."""
    (folder / "scripts").mkdir(parents=True)
    (folder / "research" / "swarm" / "claim_cycle").mkdir(parents=True)
    (folder / "cycle" / "a").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "py", folder / "scripts" / "py")
    shutil.copy(H.HERE / "tools.py", folder / "research" / "swarm" / "claim_cycle" / "tools.py")
    (folder / ".git").write_text(f"gitdir: {REPO}/.git/worktrees/x\n", "utf-8")
    return folder.resolve()


@pytest.mark.parametrize("agent", list(ALLOWED))
def test_the_tools_run_from_the_repo_root_only(agent, tmp_path, monkeypatch):
    """Approved from the hook's own root, and from a link to it, which names the same folder;
    denied from anywhere else: another checkout's root (a worktree's, with its own ``scripts/py``
    and tools), a link to one, a folder inside the root, and a relative folder, which would be read
    from wherever the hook runs, even when it runs from the root."""
    command = f"{TOOLS} index cycle/a"
    other = lookalike(tmp_path / "worktree")
    (tmp_path / "to-other").symlink_to(other)
    (tmp_path / "to-repo").symlink_to(REPO)
    assert allowed(H.decide(call(agent, command=command)))
    assert allowed(H.decide(call(agent, command=command, cwd=str(tmp_path / "to-repo"))))
    for cwd in ("/tmp", str(other), str(tmp_path / "to-other"), str(REPO / "research"), None):
        says = denied(H.decide(call(agent, command=command, cwd=cwd)))
        assert f"run from the repo root, {REPO}," in says, cwd
    monkeypatch.chdir(REPO)
    for cwd in (".", "research/..", 7, ["/"]):
        assert denied(H.decide(call(agent, command=command, cwd=cwd))), cwd


# The forms the hook passed before the owner's ruling of 2026-10-23, and refuses now in words that
# name the one form it approves.
OTHER_FORMS = {"python": "python -m research.swarm.claim_cycle.tools index cycle/a",
               "python3": "python3 -m research.swarm.claim_cycle.tools index cycle/a",
               "./scripts/py": "./scripts/py -m research.swarm.claim_cycle.tools index cycle/a"}


@pytest.mark.parametrize("agent", list(ALLOWED))
@pytest.mark.parametrize("first", sorted(OTHER_FORMS))
def test_the_tools_run_only_as_scripts_py_and_the_refusal_names_that_form(agent, first):
    says = denied(H.decide(call(agent, command=OTHER_FORMS[first])))
    assert "scripts/py -m research.swarm.claim_cycle.tools <command> ..." in says
    assert f"not {first}" in says


@pytest.mark.parametrize("agent", list(ALLOWED))
def test_the_environment_prefix_builders_are_given_is_refused_in_words_that_say_so(agent):
    """The root CLAUDE.md's prefix before a command the hook would approve: denied, called an
    environment setting, with the one form named; the bare command is approved."""
    prefix = claude_md_prefix()
    assert prefix.startswith("OMP_NUM_THREADS=")  # the construction: an assignment before a command
    says = denied(H.decide(call(agent, command=f"{prefix} {TOOLS} index cycle/a")))
    assert "environment setting" in says and f"{TOOLS} <command>" in says
    assert allowed(H.decide(call(agent, command=f"{TOOLS} index cycle/a")))


def test_each_refusal_names_its_role_with_the_right_article(tmp_path, monkeypatch):
    """'a theorist', 'an experimentalist', in every refusal that names the role, the hook's and the
    pin's alike."""
    says: dict[str, list[str]] = {role: [] for role in ALLOWED}
    for role in ALLOWED:
        says[role] += [denied(H.decide(call(role, "Read", file_path="/tmp/s.json"))),
                       denied(H.decide(call(role, command="cat /tmp/s.json"))),
                       denied(H.decide(call(role, command=f"{TOOLS} sealed cycle/a"))),
                       denied(H.decide(call(role, command=f"{TOOLS} index cycle/b")))]
    says["experimentalist"].append(denied(H.decide(call("experimentalist", command=PROPOSE))))
    monkeypatch.setattr(H, "PIN", tmp_path / "none.json")
    for role in ALLOWED:
        says[role].append(denied(H.decide(call(role, command=f"{TOOLS} index cycle/a"))))
    for role, named in (("theorist", "a theorist"), ("experimentalist", "an experimentalist")):
        for text in says[role]:
            assert named in text.lower(), text
    assert not any("a experimentalist" in t.lower() for t in says["experimentalist"])


# A command for each refusal the hook can give, the shell's rules' (``analyst_hook.shell_refusal``)
# first. Only the driver's holds the word "driver", and its refusal echoes none of its words.
EVERY_REFUSAL = {
    **{f"holds {c!r}": f"{TOOLS} index cycle/a{c}x" for c in (";", "&", "|", "`", "$", "<", ">",
                                                               "\n", "\r")},
    "a control character": f"{TOOLS} index cycle/a\x0bx",
    "a backslash in double quotes": f'{TOOLS} index "cycle\\a"',
    "history in double quotes": f'{TOOLS} index "cycle!a"',
    "a glob outside quotes": f"{TOOLS} index cycle/*",
    "a tilde outside quotes": f"{TOOLS} index ~",
    "an unclosed quote": f"{TOOLS} index 'cycle/a",
    "empty": "   ",
    "python": OTHER_FORMS["python"],
    "./scripts/py": OTHER_FORMS["./scripts/py"],
    "an environment setting": f"OMP_NUM_THREADS=2 {TOOLS} index cycle/a",
    "another program": "cat /tmp/sealed/s.json",
    "another module": "scripts/py -m research.swarm.shuffle cycle/a --out x --sealed y",
    "the analysts' module": "scripts/py -m research.swarm.driver ask --run run_01-ab",
    "a command the tools lack": f"{TOOLS} sealed cycle/a",
    "another folder": f"{TOOLS} index cycle/b",
}


def every_refusal(role: str, tmp_path: Path, monkeypatch) -> dict[str, tuple[str, str]]:
    """The hook's decision and its words for ``role``, by what should draw a refusal: every
    command of :data:`EVERY_REFUSAL`, another tool, a folder other than the root, the background,
    and no cycle pinned."""
    def said(call_: dict) -> tuple[str, str]:
        out = H.decide(call_)["hookSpecificOutput"]
        return out["permissionDecision"], out["permissionDecisionReason"]

    says = {name: said(call(role, command=command)) for name, command in EVERY_REFUSAL.items()}
    says["another tool"] = said(call(role, "Read", file_path="/tmp/s.json"))
    says["another folder to run in"] = said(call(role, command=f"{TOOLS} index cycle/a",
                                                 cwd="/tmp"))
    says["the background"] = said(call(role, command=f"{TOOLS} index cycle/a",
                                       run_in_background=True))
    if role == "experimentalist":
        says["a proposal"] = said(call(role, command=PROPOSE))
    monkeypatch.setattr(H, "PIN", tmp_path / "none.json")
    says["no cycle pinned"] = said(call(role, command=f"{TOOLS} index cycle/a"))
    return says


@pytest.mark.parametrize("role", list(ALLOWED))
def test_no_refusal_speaks_of_an_analyst_or_the_driver(role, tmp_path, monkeypatch):
    """The re-audit of 2026-10-23: the shell's rules are the analyst hook's, and their refusals
    spoke of an analyst's command and one driver command at a time. Each refusal names this hook's
    own agent and its own command, and none says "analyst" or "driver" (the repo root, which a
    refusal may name, is taken out first: its folder's name is not the hook's to choose)."""
    both = every_refusal(role, tmp_path, monkeypatch)
    for name, (_, text) in both.items():
        words = text.replace(str(REPO), "<root>").lower()
        assert "analyst" not in words and "driver" not in words, (name, text)
    assert [name for name, (decision, _) in both.items() if decision != "deny"] == []
    says = {name: text for name, (_, text) in both.items()}
    whose = f"which {named_as(role)}'s command never holds"
    for c in (";", "&", "|", "`", "$", "<", ">", "\n", "\r"):
        text = says[f"holds {c!r}"]
        assert whose in text and "one tools command at a time" in text, text
    assert "does not read as one tools command (an unclosed quote?)" in says["an unclosed quote"]
    assert f"{TOOLS} <command> ..." in says["an unclosed quote"]
    assert "such as a reason" not in says["a glob outside quotes"]


def named_as(role: str) -> str:
    return {"theorist": "a theorist", "experimentalist": "an experimentalist"}[role]


@pytest.mark.parametrize("agent", list(ALLOWED))
@pytest.mark.parametrize("background", [True, 1, "true"])
def test_a_tools_command_run_in_the_background_is_denied(agent, background):
    """Fires where it should: a command the hook approves, with a truthy ``run_in_background``, is
    denied in words that say one tools command at a time, in the foreground, naming the role and
    the field to leave out."""
    for command in ALLOWED[agent]:
        says = denied(H.decide(call(agent, command=command, description="d",
                                    run_in_background=background)))
        assert says.startswith(f"{named_as(agent).capitalize()} runs one tools command at a time, "
                               f"in the foreground"), says
        assert "run_in_background" in says


@pytest.mark.parametrize("agent", list(ALLOWED))
@pytest.mark.parametrize("given", [{}, {"run_in_background": False}, {"run_in_background": None}],
                         ids=["absent", "false", "null"])
def test_a_tools_command_run_in_the_foreground_is_approved_as_before(agent, given):
    for command in ALLOWED[agent]:
        assert allowed(H.decide(call(agent, command=command, description="d", timeout=120000,
                                     **given))), command


def test_a_background_call_of_any_other_caller_hears_nothing():
    for agent in (None, "analyst", "builder"):
        assert H.decide(call(agent, command=f"{TOOLS} index cycle/a",
                             run_in_background=True)) is None


@pytest.mark.parametrize("agent", [None, "analyst", "builder", "verifier", "general-purpose",
                                   "theorists", "Theorist", "experimentalist-2"])
def test_it_says_nothing_for_any_other_caller(agent):
    """Silent for every other agent type and for the session itself (no ``agent_type``), on the
    very calls it denies or approves a theorist or an experimentalist."""
    assert H.decide(call(agent, "Read", file_path="/tmp/s.json")) is None
    assert H.decide(call(agent, command="cat /tmp/s.json")) is None
    for command in [*ALLOWED["theorist"], *OTHER_FORMS.values(), *DENIED.values()]:
        assert H.decide(call(agent, command=command)) is None, command
    for tool in H.REPORT_TOOLS:
        assert H.decide(call(agent, tool)) is None


def test_the_hooks_commands_are_the_tools_commands():
    assert set().union(*H.COMMANDS.values()) == set(T.COMMANDS)
    assert set(H.COMMANDS["experimentalist"]) < set(H.COMMANDS["theorist"])


def test_input_that_is_not_a_hook_call_is_refused():
    err = io.StringIO()
    assert H.main(io.StringIO("not json"), io.StringIO(), err) == 2 and err.getvalue()
    out = io.StringIO()
    assert H.main(io.StringIO(json.dumps(call("theorist", "Read"))), out, io.StringIO()) == 0
    assert json.loads(out.getvalue())["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_it_runs_as_a_script_on_the_standard_library_alone(tmp_path):
    """Wired as ``python3 <repo>/research/swarm/claim_cycle/hook.py``: run with no site-packages
    (-S) and isolated from the environment (-I), so the repo is not on its path, it still loads the
    analyst hook's rules and the pin, denies and passes. It runs from a copy of its three files in
    a repo of its own, so the pin it reads is that repo's ``.claude/claim_cycle.json``, never this
    checkout's."""
    repo = tmp_path / "repo"
    (repo / "research" / "swarm" / "claim_cycle").mkdir(parents=True)
    for name in ("claim_cycle/hook.py", "claim_cycle/pin.py", "analyst_hook.py"):
        shutil.copy(H.HERE.parent / name, repo / "research" / "swarm" / name)
    hook = repo / "research" / "swarm" / "claim_cycle" / "hook.py"

    def run(payload):
        return subprocess.run([sys.executable, "-S", "-I", str(hook)],
                              input=json.dumps({**payload, "cwd": str(repo.resolve())}),
                              capture_output=True, text=True, timeout=30)

    r = run(call("theorist", "Read", file_path="/tmp/s.json"))
    assert r.returncode == 0, r.stderr
    assert "Read" in denied(json.loads(r.stdout))
    r = run(call("theorist", command=f"{TOOLS} index cycle/a; cat /tmp/s.json"))
    assert r.returncode == 0 and denied(json.loads(r.stdout))
    r = run(call("theorist", command=f"{TOOLS} index cycle/a"))  # nothing pinned there yet
    assert r.returncode == 0 and "No claim cycle is pinned" in denied(json.loads(r.stdout))
    (repo / ".claude").mkdir()
    pin_at(repo / ".claude" / "claim_cycle.json", repo=repo.resolve())
    r = run(call("theorist", command=f"{TOOLS} index cycle/a"))
    assert r.returncode == 0 and allowed(json.loads(r.stdout)), (r.stdout, r.stderr)
    r = run(call("theorist", command=f"{TOOLS} index cycle/b"))
    assert r.returncode == 0 and "is not this cycle's corpus folder" in denied(json.loads(r.stdout))


def hook_copy(tmp_path: Path, corpus: str = "cycle/corpus",
              proposals: str = "cycle/proposals.jsonl") -> Path:
    """A repo of its own holding a copy of the hook's three files, with a pin at its own
    ``.claude/claim_cycle.json`` naming ``corpus`` and ``proposals``: the hook run from it reads
    that pin, never this checkout's."""
    repo = tmp_path / "repo"
    (repo / "research" / "swarm" / "claim_cycle").mkdir(parents=True)
    for name in ("claim_cycle/hook.py", "claim_cycle/pin.py", "analyst_hook.py"):
        shutil.copy(H.HERE.parent / name, repo / "research" / "swarm" / name)
    (repo / ".claude").mkdir()
    pin_at(repo / ".claude" / "claim_cycle.json", corpus, proposals, repo=repo.resolve())
    return repo.resolve()


def run_copy(repo: Path, payload: dict) -> tuple[int, dict | None]:
    """The copy's hook as Claude Code runs it: the call as JSON on standard input, with no
    site-packages (-S) and isolated from the environment (-I); its exit code and its answer."""
    r = subprocess.run([sys.executable, "-S", "-I",
                        str(repo / "research" / "swarm" / "claim_cycle" / "hook.py")],
                       input=json.dumps(payload), capture_output=True, text=True, timeout=30)
    assert r.stderr == "", r.stderr
    return r.returncode, (json.loads(r.stdout) if r.stdout.strip() else None)


def prompt_examples(role: str) -> list[str]:
    """The commands ``claim_cycle/<role>.md`` shows its agent, one per line (as
    ``test_claim_prompts.py`` reads them)."""
    return [line.strip() for line in (H.HERE / f"{role}.md").read_text("utf-8").splitlines()
            if line.strip().startswith(f"{TOOLS} ") and "<command>" not in line]


TOOLS_INDEX = {"command": f"{TOOLS} index cycle/corpus"}
# The calls of the audit of 2026-10-23 (its cases.py), each whole as Claude Code sends it, with
# what this hook answers under a pin naming cycle/corpus: "allow", "deny", or None for none.
AS_SENT = [
    ("a theorist's index", "theorist", "Bash", TOOLS_INDEX, "root", "allow"),
    ("a theorist's report", "theorist", "SubagentHandback", {"message": "done"}, "root", "allow"),
    ("a theorist's structured report", "theorist", "StructuredOutput", {"x": 1}, "root", "allow"),
    ("a theorist's driver ask", "theorist", "Bash",
     {"command": "scripts/py -m research.swarm.driver ask --run RUN"}, "root", "deny"),
    ("a theorist's cat", "theorist", "Bash", {"command": "cat cycle/corpus/run_01.json"}, "root",
     "deny"),
    ("a theorist's Read", "theorist", "Read", {"file_path": "cycle/corpus/run_01.json"}, "root",
     "deny"),
    ("a theorist's chained index", "theorist", "Bash",
     {"command": f"{TOOLS} index cycle/corpus; cat cycle/x"}, "root", "deny"),
    ("a theorist's other folder", "theorist", "Bash", {"command": f"{TOOLS} index cycle/shuffled"},
     "root", "deny"),
    ("a theorist's prefixed index", "theorist", "Bash",
     {"command": f"OMP_NUM_THREADS=2 {TOOLS} index cycle/corpus"}, "root", "deny"),
    ("a theorist's index from /tmp", "theorist", "Bash", TOOLS_INDEX, "/tmp", "deny"),
    ("a theorist's index in the background", "theorist", "Bash",
     {**TOOLS_INDEX, "description": "d", "run_in_background": True}, "root", "deny"),
    ("a theorist's index in the foreground", "theorist", "Bash",
     {**TOOLS_INDEX, "description": "d", "run_in_background": False}, "root", "allow"),
    ("an experimentalist's index in the background", "experimentalist", "Bash",
     {**TOOLS_INDEX, "run_in_background": True}, "root", "deny"),
    ("an experimentalist's index", "experimentalist", "Bash", TOOLS_INDEX, "root", "allow"),
    ("an experimentalist's structured report", "experimentalist", "StructuredOutput", {"x": 1},
     "root", "allow"),
    ("an experimentalist's Write", "experimentalist", "Write", {"file_path": "x", "content": "y"},
     "root", "deny"),
    ("an analyst's index", "analyst", "Bash", TOOLS_INDEX, "root", None),
    ("a builder's git status", "builder", "Bash", {"command": "git status --short"}, "root", None),
    ("a builder's index", "builder", "Bash", TOOLS_INDEX, "root", None),
    ("a verifier's index of any folder", "verifier", "Bash",
     {"command": f"{TOOLS} index cycle/shuffled"}, "root", None),
    ("the session's ls", None, "Bash", {"command": "ls"}, "root", None),
    ("the session's index", None, "Bash", TOOLS_INDEX, "root", None),
    ("the session's Agent", None, "Agent",
     {"subagent_type": "theorist", "prompt": "p", "description": "d"}, "root", None),
]


@pytest.mark.parametrize("label, agent, tool, given, where, expect", AS_SENT,
                         ids=[c[0] for c in AS_SENT])
def test_each_call_as_claude_code_sends_it(label, agent, tool, given, where, expect, tmp_path):
    """Run as Claude Code runs it, each call whole on standard input, from a copy's root with its
    own pin: approved, denied, or no answer at all, as its caller and its call say."""
    repo = hook_copy(tmp_path)
    payload = {"session_id": "s", "hook_event_name": "PreToolUse", "tool_name": tool,
               "tool_input": given, "cwd": str(repo) if where == "root" else where}
    if agent is not None:
        payload["agent_type"] = agent
    rc, answer = run_copy(repo, payload)
    assert rc == 0
    if expect is None:
        assert answer is None
    else:
        assert (allowed if expect == "allow" else (lambda a: bool(denied(a))))(answer), answer


def test_a_copy_of_the_hook_approves_only_from_its_own_root(tmp_path):
    """The root is the one the hook's own file lies in: the copy approves a command run from its
    own root, and denies the same command run from this checkout's."""
    repo = hook_copy(tmp_path)
    payload = {**call("theorist", command=f"{TOOLS} index cycle/corpus"), "cwd": str(repo)}
    rc, answer = run_copy(repo, payload)
    assert rc == 0 and allowed(answer)
    rc, answer = run_copy(repo, {**payload, "cwd": str(REPO)})
    assert rc == 0 and f"run from the repo root, {repo}," in denied(answer)


@pytest.mark.parametrize("role", list(ALLOWED))
def test_each_example_its_prompt_shows_is_approved_as_claude_code_runs_the_hook(role, tmp_path,
                                                                                  monkeypatch):
    """Under a pin naming the examples' corpus folder and proposals file, as the orchestrator pins
    a cycle's: each example of ``claim_cycle/<role>.md`` is approved, in this checkout and by a
    copy run as a script; and the builders' prefix before it is refused."""
    examples = prompt_examples(role)
    assert [e.split()[3] for e in examples] == list(H.COMMANDS[role])  # one each
    monkeypatch.setattr(H, "PIN", pin_at(tmp_path / "examples-pin.json", "cycle/corpus",
                                         "cycle/proposals.jsonl"))
    repo = hook_copy(tmp_path)
    for example in examples:
        assert allowed(H.decide(call(role, command=example))), example
        rc, answer = run_copy(repo, {**call(role, command=example), "cwd": str(repo)})
        assert rc == 0 and allowed(answer), example
        assert "environment setting" in denied(
            H.decide(call(role, command=f"{claude_md_prefix()} {example}")))


def agent_definition(role: str) -> tuple[dict[str, str], str]:
    text = (AGENTS / f"{role}.md").read_text(encoding="utf-8")
    head, body = text[4:].split("\n---\n", 1)
    fields = {}
    for line in head.splitlines():
        name, _, value = line.partition(":")
        fields[name.strip()] = value.strip()
    return fields, " ".join(body.split())


@pytest.mark.parametrize("role", list(ALLOWED))
def test_its_agent_type_says_the_builders_prefix_is_not_its_own(role):
    """``.claude/agents/<role>.md`` says, as ``analyst.md`` does, that the root CLAUDE.md's prefix
    is not its agent's and that the hook refuses it, naming the prefix CLAUDE.md gives."""
    _, body = agent_definition(role)
    assert (f"The environment prefix the root CLAUDE.md gives builders (`{claude_md_prefix()}`) is "
            f"not yours: the hook refuses a command with any setting before it.") in body


def test_the_experimentalists_description_says_what_it_does():
    """It plans the held-out test most likely to break each proposed claim
    (``claim_cycle/experimentalist.md``: "You run no test and change nothing"), and proposes
    nothing; it does not try to break them itself."""
    fields, _ = agent_definition("experimentalist")
    says = fields["description"]
    assert ("Plans, for each proposed claim of one claim cycle, the held-out test most likely to "
            "break it, from the record views the claim cycle's tools serve.") in says
    assert "It runs no test and proposes nothing" in says
    assert "Tries to break" not in says
    assert "You run no test and change nothing" in " ".join(
        (H.HERE / "experimentalist.md").read_text("utf-8").split())


# ── the pin: one cycle's corpus folder and proposals file ────────────────────

OTHER_PATHS = {
    "another folder": f"{TOOLS} index cycle/b",
    "the repo's corpus": f"{TOOLS} index research/corpus",
    "a folder anywhere": f"{TOOLS} index /tmp/anything",
    "a path out of the pinned one": f"{TOOLS} index cycle/a/../b",
    "a folder inside the pinned one": f"{TOOLS} records cycle/a/failed",
    "another folder, options first": f"{TOOLS} records --run run_01-ab cycle/copy",
    "another proposals file": f"{TOOLS} proposals cycle/other.jsonl",
    "the sealed file as proposals": f"{TOOLS} proposals /tmp/sealed/s.json",
}
OTHER_OUT = {
    "propose to another file": f"{TOOLS} propose --out cycle/other.jsonl '{json.dumps(GOOD)}'",
    "propose to another, abbreviated": f"{TOOLS} propose --ou cycle/x.jsonl '{json.dumps(GOOD)}'",
}


@pytest.mark.parametrize("agent", list(ALLOWED))
@pytest.mark.parametrize("command", list(OTHER_PATHS.values()), ids=list(OTHER_PATHS))
def test_a_folder_or_file_other_than_the_pinned_ones_is_denied(agent, command):
    says = denied(H.decide(call(agent, command=command)))
    assert "is not this cycle's" in says and "its prompt names" in says


@pytest.mark.parametrize("command", list(OTHER_OUT.values()), ids=list(OTHER_OUT))
def test_a_theorist_proposes_only_to_the_pinned_proposals_file(command):
    assert "is not this cycle's proposals file" in denied(H.decide(call("theorist",
                                                                         command=command)))


SAME_PATHS = {
    "dotted": f"{TOOLS} index ./cycle/a",
    "a trailing slash": f"{TOOLS} index cycle/a/",
    "absolute": f"{TOOLS} index {REPO}/cycle/a",
    "options first": f"{TOOLS} records --run run_01-ab cycle/a",
    "through a folder and back": f"{TOOLS} proposals cycle/x/../proposals.jsonl",
}


@pytest.mark.parametrize("command", list(SAME_PATHS.values()), ids=list(SAME_PATHS))
def test_the_pinned_paths_in_another_spelling_are_approved(command):
    assert allowed(H.decide(call("theorist", command=command)))


@pytest.mark.parametrize("agent", list(ALLOWED))
def test_with_nothing_pinned_its_tools_are_denied_and_other_callers_hear_nothing(
        agent, tmp_path, monkeypatch):
    monkeypatch.setattr(H, "PIN", tmp_path / "none.json")
    for command in ALLOWED[agent]:
        assert "No claim cycle is pinned" in denied(H.decide(call(agent, command=command)))
    assert H.decide(call("analyst", command=f"{TOOLS} index cycle/a")) is None


def test_a_pin_that_does_not_read_denies_everything(pinned):
    pinned.write_text('{"corpus": "cycle/a"}', "utf-8")
    assert "does not read as one" in denied(H.decide(call("theorist",
                                                          command=f"{TOOLS} index cycle/a")))
    pinned.write_text("not json", "utf-8")
    assert "does not read as one" in denied(H.decide(call("theorist",
                                                          command=f"{TOOLS} index cycle/a")))


def test_a_help_flag_is_denied_and_prints_nothing():
    says = denied(H.decide(call("theorist", command=f"{TOOLS} index -h cycle/a")))
    assert "does not read as one tools command" in says


def test_no_agent_reads_both_the_corpus_and_its_shuffled_copy(tmp_path, monkeypatch):
    """The verifier's V6D-3: the copy's run ids that the corpus lacks are the planted records, so
    under the pin of one cycle the other cycle's folder is denied, whichever is pinned."""
    src = TS.corpus(tmp_path / "corpus", per=3)
    copy = tmp_path / "copy"
    body = S.write(src, copy, tmp_path / "sealed.json", seed=5, k=6)
    # The construction: reading both folders would tell the planted records.
    ids = [{p.stem for p in folder.glob("*.json")} for folder in (src, copy)]
    assert ids[1] - ids[0] == set(body["planted"]["records"])
    (tmp_path / "out").mkdir()
    for mine, theirs in ((src, copy), (copy, src)):
        path = tmp_path / f"pin-{mine.name}.json"
        N.write(mine, tmp_path / "out" / "p.jsonl", pin=path, repo=tmp_path)
        monkeypatch.setattr(H, "PIN", path)
        for agent in ALLOWED:
            assert allowed(H.decide(call(agent, command=f"{TOOLS} index {mine}")))
            assert "not this cycle's corpus folder" in denied(
                H.decide(call(agent, command=f"{TOOLS} index {theirs}")))
            assert "not this cycle's corpus folder" in denied(
                H.decide(call(agent, command=f"{TOOLS} records {theirs} --fields run.id")))


def test_one_cycle_is_pinned_at_a_time_and_only_where_it_can_run(tmp_path):
    (tmp_path / "corpus").mkdir()
    path = tmp_path / "cycle-pin.json"
    got = N.write("corpus", "p.jsonl", pin=path, repo=tmp_path)
    assert got == {"corpus": str((tmp_path / "corpus").resolve()),
                   "proposals": str((tmp_path / "p.jsonl").resolve())}
    assert N.read(path) == got
    with pytest.raises(N.PinRefused, match="pinned already"):
        N.write("corpus", "p.jsonl", pin=path, repo=tmp_path)
    assert N.clear(path) and N.read(path) is None and not N.clear(path)
    for corpus, proposals, says in (("missing", "p.jsonl", "not a folder"),
                                    ("corpus", "p.json", ".jsonl"),
                                    ("corpus", "nowhere/p.jsonl", "a folder that exists"),
                                    ("corpus", "corpus/p.jsonl", "lies in the corpus folder")):
        with pytest.raises(N.PinRefused, match=says):
            N.write(corpus, proposals, pin=path, repo=tmp_path)
        assert not path.exists()


def test_a_proposals_file_is_pinned_for_one_cycle_only(tmp_path):
    """The verifier's V6D-8, both ways: a proposals file pinned for one corpus folder can be pinned
    again for it (a re-pin of the same cycle, after its proposals were written), and is refused for
    the other cycle's folder, whose agents would read the first cycle's proposals and the run ids
    and marker numbers they cite; that cycle's own new file is pinned."""
    for name in ("real", "copy"):
        (tmp_path / name).mkdir()
    path, file = tmp_path / "cycle-pin.json", tmp_path / "p.jsonl"
    N.write("real", "p.jsonl", pin=path, repo=tmp_path)
    file.write_text(json.dumps(GOOD) + "\n", "utf-8")  # the real cycle's theorist proposed
    assert N.clear(path)
    assert N.write("real", "./p.jsonl", pin=path, repo=tmp_path)["proposals"] == str(file.resolve())
    assert N.clear(path)
    with pytest.raises(N.PinRefused, match="pinned before for the corpus folder real"):
        N.write("copy", "p.jsonl", pin=path, repo=tmp_path)
    assert not path.exists()
    # The same, before any proposal was written: the file is the real cycle's.
    N.write("real", "q.jsonl", pin=path, repo=tmp_path)
    assert N.clear(path)
    with pytest.raises(N.PinRefused, match="pinned before"):
        N.write("copy", "q.jsonl", pin=path, repo=tmp_path)
    got = N.write("copy", "copy-proposals.jsonl", pin=path, repo=tmp_path)
    assert got["corpus"] == str((tmp_path / "copy").resolve())


def test_a_proposals_file_holding_lines_no_recorded_pin_named_is_refused(tmp_path):
    """Both ways: a file that already holds proposals, which no pin recorded beside this one named
    (written elsewhere, or by another cycle before the record), is refused; an empty one is a new
    cycle's file and is pinned."""
    (tmp_path / "corpus").mkdir()
    path = tmp_path / "cycle-pin.json"
    (tmp_path / "held.jsonl").write_text(json.dumps(GOOD) + "\n", "utf-8")
    with pytest.raises(N.PinRefused, match="holds lines"):
        N.write("corpus", "held.jsonl", pin=path, repo=tmp_path)
    assert not path.exists()
    (tmp_path / "empty.jsonl").write_text("", "utf-8")
    assert N.write("corpus", "empty.jsonl", pin=path, repo=tmp_path)


def test_a_record_of_pins_that_does_not_read_refuses_every_pin(tmp_path):
    (tmp_path / "corpus").mkdir()
    path = tmp_path / "cycle-pin.json"
    N.ledger_of(path).write_text("not json", "utf-8")
    with pytest.raises(N.PinRefused, match="does not read as one"):
        N.write("corpus", "p.jsonl", pin=path, repo=tmp_path)
    assert not path.exists()


def test_the_pins_command_line_refuses_the_other_cycles_proposals_file(tmp_path, monkeypatch):
    """The verifier's probe through the orchestrator's command line: pin the real cycle, propose,
    clear, then pin the copy with the same proposals file: refused, so its theorist is never served
    the real cycle's proposals; the copy's own file is pinned."""
    for name in ("real", "copy"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(N, "PIN", tmp_path / ".claude" / "claim_cycle.json")
    monkeypatch.chdir(tmp_path)
    said: list[str] = []
    assert N.main(["--corpus", "real", "--proposals", "cycle.jsonl"], out=said.append) == 0
    (tmp_path / "cycle.jsonl").write_text(json.dumps(GOOD) + "\n", "utf-8")
    assert N.main(["--clear"], out=said.append) == 0
    assert N.main(["--corpus", "copy", "--proposals", "cycle.jsonl"], out=said.append) == 1
    assert "pinned before" in json.loads(said[-1])["refused"] and N.read() is None
    assert N.main(["--corpus", "copy", "--proposals", "copy.jsonl"], out=said.append) == 0


def test_the_pins_command_line_pins_and_clears(tmp_path, monkeypatch):
    (tmp_path / "corpus").mkdir()
    monkeypatch.setattr(N, "PIN", tmp_path / ".claude" / "claim_cycle.json")
    monkeypatch.chdir(tmp_path)
    said: list[str] = []
    assert N.main(["--corpus", "corpus", "--proposals", "p.jsonl"], out=said.append) == 0
    assert json.loads(said[-1])["pinned"]["corpus"] == str((tmp_path / "corpus").resolve())
    assert N.main(["--corpus", "corpus", "--proposals", "p.jsonl"], out=said.append) == 1
    assert "pinned already" in json.loads(said[-1])["refused"]
    assert N.main(["--clear"], out=said.append) == 0 and json.loads(said[-1]) == {"cleared": True}
    assert N.main(["--corpus", "corpus"], out=said.append) == 1


def test_the_tools_and_the_hook_read_a_command_with_one_parser():
    assert T.build_parser().format_usage() == N.build_parser().format_usage()
    assert T.COMMANDS == N.COMMANDS


def test_the_hook_reads_the_pin_the_pins_command_line_writes():
    """The orchestrator's ``python -m research.swarm.claim_cycle.pin`` writes ``pin.PIN``; the hook,
    run from the same checkout, reads that file (the fixture here points it elsewhere)."""
    import importlib

    fresh = importlib.reload(H)
    try:
        assert fresh.PIN == N.PIN == N.REPO / ".claude" / "claim_cycle.json"
    finally:
        importlib.reload(H)


def test_a_tool_input_that_is_not_an_object_is_denied_for_its_roles():
    answer = H.decide({"tool_name": "Bash", "tool_input": "cat x", "cwd": str(REPO),
                       "agent_type": "theorist"})
    assert denied(answer)
