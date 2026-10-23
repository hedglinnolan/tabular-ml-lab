"""The claim cycle's PreToolUse hook (``claim_cycle/hook.py``): a theorist or an experimentalist
runs the claim cycle's tools and nothing else, so it reads records only through the view, and never
the sealed file by any other tool; and it names only the pinned cycle's corpus folder and proposals
file (``claim_cycle/pin.py``), so no agent reads both the corpus and its shuffled copy. Each rule
both ways: what it lets through, and what it denies; and it says nothing for any other agent."""
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

REPO = H.REPO
TOOLS = "scripts/py -m research.swarm.claim_cycle.tools"
PROPOSE = f"{TOOLS} propose --out cycle/proposals.jsonl '{json.dumps(GOOD)}'"


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
def test_the_tools_commands_of_its_role_pass(agent, command):
    assert H.decide(call(agent, command=command)) is None


@pytest.mark.parametrize("agent", list(ALLOWED))
@pytest.mark.parametrize("tool", ["Read", "Write", "Edit", "Glob", "Grep", "WebFetch", "Agent",
                                  "NotebookEdit", "mcp__claude_ai_Claude_Docs__read"])
def test_every_other_tool_is_denied(agent, tool):
    assert tool in denied(H.decide(call(agent, tool, file_path="/tmp/sealed/s.json")))


@pytest.mark.parametrize("agent", list(ALLOWED))
@pytest.mark.parametrize("tool", H.REPORT_TOOLS)
def test_its_report_tools_pass(agent, tool):
    assert H.decide(call(agent, tool)) is None


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


def test_the_tools_run_from_the_repo_root_only():
    assert denied(H.decide(call("theorist", command=f"{TOOLS} index cycle/a", cwd="/tmp")))


@pytest.mark.parametrize("agent", [None, "analyst", "builder", "general-purpose"])
def test_it_says_nothing_for_any_other_caller(agent):
    assert H.decide(call(agent, "Read", file_path="/tmp/s.json")) is None
    assert H.decide(call(agent, command="cat /tmp/s.json")) is None


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
    assert r.returncode == 0 and r.stdout == "", (r.stdout, r.stderr)
    r = run(call("theorist", command=f"{TOOLS} index cycle/b"))
    assert r.returncode == 0 and "is not this cycle's corpus folder" in denied(json.loads(r.stdout))


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
def test_the_pinned_paths_in_another_spelling_pass(command):
    assert H.decide(call("theorist", command=command)) is None


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
            assert H.decide(call(agent, command=f"{TOOLS} index {mine}")) is None
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
