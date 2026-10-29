"""What an analyst's command prints (the driver-output package, 2026-10-28; ``driver.py``).

D1: Claude Code shows a Bash call's output inline only up to about 30,000 characters; past that an
analyst sees a 2 KB preview and cannot read the rest, since it runs only the driver's commands. So
every output an analyst's command prints stays at or under 24,000 characters (the ruling's
margin), and what it leaves out it names with the command that shows it. The run states are
constructed here, no engine: an ``ask`` built from the sizes one cut ``ask`` of the trio's Haiku
run held (its decision log 17,479 characters, the other open lines 8,011, the options 5,862, the
readings 4,037, the words 1,184), one of a run 600 decisions long, one whose question alone passes
the cap, and the small one, which prints as before. Each is weighed in characters and in UTF-8
bytes, and what each command it names shows is held to the state constructed.

D2: where a question takes several options in one answer, ``ask`` says how, in the form
``answer`` reads, and a comma-joined list is refused in words that name the form that works.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from research.swarm import driver
from research.swarm import options as opts
from research.swarm.tests.test_driver import RID, fake_run, run_main
from research.swarm.tests.test_options import compose, exclusions

CAP = 24_000  # the ruling's cap, in characters, under the harness's 30,000
INVOKE = "scripts/py -m research.swarm.driver"


def weighed(text: str) -> str:
    """``text``, after holding it to the cap in characters and in UTF-8 bytes."""
    assert len(text) <= CAP and len(text.encode("utf-8")) <= CAP, len(text.encode("utf-8"))
    return text


def said(argv: list[str]) -> tuple[int, str]:
    out: list[str] = []
    code = driver.main(argv, out=out.append)
    assert len(out) == 1
    return code, out[0]


def text_of(n: int, seed: int = 0) -> str:
    """Words of exactly ``n`` characters, some outside ASCII (an en dash), as the app's are."""
    vocab = ["the", "rows", "kcal", "500–5,000", "`DR1TKCAL`", "recall", "day", "intake", "lens",
             "screen", "outside", "reported"]
    out, i, length = [], seed, 0
    while length < n:
        word = vocab[i % len(vocab)]
        out.append(word)
        length += len(word) + 1
        i += 7
    return " ".join(out)[:n]


def records(n: int) -> list[dict[str, Any]]:
    """A decision log of ``n`` records as ``GET /projects/{pid}`` serves it, each with a note of
    about 420 characters (the analyst's reason) and a sentence of about 120."""
    return [{"id": f"{i:032x}", "seq": i, "sentence": text_of(120, i), "note": text_of(420, i + 3),
             "decision": {"kind": f"set_kind_{i % 9}"}} for i in range(1, n + 1)]


def log_of(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The log as ``ask`` lists it, written here from the records: id, seq, sentence, note and
    the decision's kind."""
    return [{"id": r["id"], "seq": r["seq"], "sentence": r["sentence"], "note": r["note"],
             "kind": r["decision"]["kind"]} for r in raw]


def option(i: int, size: int) -> dict[str, Any]:
    return {"id": f"family_{i}", "label": f"Family {i}",
            "decision": {"kind": "select_models", "models": [f"family_{i}"]},
            "inductive_bias": text_of(size, i)}


def other_lines() -> list[dict[str, Any]]:
    """Four other open lines, about 8,000 characters: a finding with its one way to act, and three
    declarations with their options and fields."""
    finding = {"id": "finding:pack::dietary::implausible_intake",
               "key": "pack::dietary::implausible_intake", "source": "finding", "label": "Decide",
               "name": text_of(150, 1), "status": "open", "stage": "whos_in",
               "options": [{"id": "x-1a2b3c4d", "label": text_of(80, 2),
                            "decision": {"kind": "dismiss_finding", "finding_id": "f1"}}]}
    lines = [finding]
    for n, (key, size) in enumerate((("set_model_sequence", 1300), ("set_sensitivity", 600),
                                     ("set_usual_intake", 5400))):
        lines.append({"id": f"decision:{key}", "key": key, "source": "declaration",
                      "label": "Decide", "name": f"the {key} question", "status": "open",
                      "stage": "models",
                      "options": [{"id": f"{key}:a", "label": text_of(size // 2, n),
                                   "decision": {"kind": key}}],
                      "fields": [{"name": "column",
                                  "choices": [{"value": text_of(size // 2, n)}]}]})
    return lines


def asked(n_decisions: int = 28, *, n_options: int = 11, option_size: int = 430,
          why: int = 900, lines: list[dict[str, Any]] | None = None
          ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """What ``the_question`` gives, built here: its public part (what ``ask`` printed before this
    package), the whole with the private parts ``preview`` and ``answer`` read, and the records
    its log is read from."""
    raw = records(n_decisions)
    options = [option(i, option_size) for i in range(n_options)]
    public = {"run": RID, "purpose": "inference", "seal_opened": False,
              "fit": {"pressed": False, "locked": False},
              "question": {"key": "models", "status": "open"},
              "words": {"title": "The model families", "question": "Which families?",
                        "why": text_of(why, 5)},
              "line": {"id": "q:models", "key": "models", "source": "question", "status": "open"},
              "options": options, "multi": True,
              "data": {"basis": "Ranked for 7,315 analyzed rows and 9 predictors."},
              "readings": {"text": text_of(1500, 9), "consumer": "The models read it.",
                           "groups": [{"key": "g1", "words": text_of(600, 4)}],
                           "exits": [{"id": f"x-0000000{i}", "label": text_of(240, i),
                                      "decision": {"kind": "confirm_readings", "items": [i]}}
                                     for i in range(7)]},
              "also_open": other_lines() if lines is None else lines,
              "last_refusal_exits": [{"id": "x-feedface", "label": "Fill the blanks",
                                      "decision": {"kind": "set_missing", "strategy": "impute"}}],
              "decisions": log_of(raw)}
    composed = {"options": copy.deepcopy(options) + [
        {"id": "impute", "label": "Fill",
         "decision": {"kind": "set_missing", "strategy": "impute"}}],
        "fields": [], "multi": "models"}
    whole = copy.deepcopy(public) | {"_composed": composed, "_ways": [], "_view": {
        "state": {"purpose": "inference", "seal_opened": None}, "stages": {}, "decisions": raw},
        "_declared": {}}
    return public, whole, raw


def standing(monkeypatch, whole: dict[str, Any], raw: list[dict[str, Any]]) -> None:
    """The run's server reads stood in for: ``the_question`` gives ``whole``, the project's view
    holds ``raw``."""
    monkeypatch.setattr(driver, "the_question", lambda run, client, wait: copy.deepcopy(whole))
    monkeypatch.setattr(driver, "view_of", lambda run, client: {"decisions": copy.deepcopy(raw),
                                                                "state": {}})


def size(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def joined(texts: list[str]) -> dict[str, Any]:
    """The output that ``texts``, its parts in order, print, read as each part's ``part`` says:
    the parts hold the output's keys in order, and where one ends inside a list, a text or an
    object, the next goes on with it and names where under ``continues``."""
    out: dict[str, Any] = {}
    for n, text in enumerate(texts, start=1):
        part = json.loads(weighed(text))
        head = part.pop("part")
        assert (head["this"], head["of"]) == (n, len(texts))
        cont = tuple(head.get("continues") or ())
        out = _join(out, part, cont, ())
    return out


def _join(a: Any, b: Any, cont: tuple, path: tuple) -> Any:
    if isinstance(a, dict) and isinstance(b, dict):
        merged = dict(a)
        for k, v in b.items():
            merged[k] = _join(a[k], v, cont, path + (k,)) if k in a else v
        return merged
    if isinstance(a, list) and isinstance(b, list):
        if len(cont) > len(path) and cont[:len(path)] == path and b:
            return a[:-1] + [_join(a[-1], b[0], cont, path + (cont[len(path)],))] + b[1:]
        return a + b
    if isinstance(a, str) and isinstance(b, str) and cont == path:
        return a + b
    raise AssertionError(f"part goes on at {path} where continues names {cont}")


def parts_of(first: str, command: str, *flags: str) -> list[str]:
    """The parts of an output whose first part is ``first``: the rest each by the ``next`` the
    part before names, run as written, which is the output's own command (``command`` with
    ``flags``: ``ask --decisions``, ``ask --line <id>``) and ``--part k``."""
    import shlex

    texts = [weighed(first)]
    head = json.loads(first)["part"]
    own = " ".join([command, "--run", RID, *(shlex.quote(f) for f in flags)])
    for k in range(2, head["of"] + 1):
        given = json.loads(texts[-1])["part"]["next"]
        assert given == f"{INVOKE} {own} --part {k}"
        code, text = said(shlex.split(given)[3:])
        assert code == 0, text
        texts.append(text)
    assert "next" not in json.loads(texts[-1])["part"]
    return texts


# ── D1: an ask past the cap ──────────────────────────────────────────────────


def test_the_ask_that_was_cut_prints_under_the_cap_and_names_how_to_see_the_rest(tmp_path,
                                                                                 monkeypatch):
    """The trio's cut ask, rebuilt from its sizes: whole, it passes the cap (and the harness's
    30,000); printed, it is at or under 24,000, the question's own options, words, readings and
    ways forward are whole, and what it leaves out it names with the command that shows it."""
    fake_run(tmp_path, monkeypatch)
    public, whole, raw = asked()
    assert size(public) > 30_000  # as the cut one was, by construction
    standing(monkeypatch, whole, raw)
    code, text = said(["ask", "--run", RID])
    got = json.loads(weighed(text))
    assert code == 0 and "part" not in got
    for key in ("question", "words", "line", "options", "data", "readings", "last_refusal_exits",
                "fit", "purpose", "seal_opened"):
        assert got[key] == public[key], key
    left = got["left_out"]
    assert f"{INVOKE} ask --run {RID} --decisions" in left["decisions"]
    shown = {d["id"]: d for d in got["decisions"]}
    full = {d["id"]: d for d in public["decisions"]}
    assert shown and set(shown) <= set(full)
    for oid, d in shown.items():
        assert d == {k: v for k, v in full[oid].items() if k != "note"}
    if "also_open" in left:
        assert f"{INVOKE} ask --run {RID} --line ID shows one in full" in left["also_open"]
    assert [ln["id"] for ln in got["also_open"]] == [ln["id"] for ln in public["also_open"]]


def test_each_command_the_ask_names_shows_what_it_left_out(tmp_path, monkeypatch):
    """``ask --decisions`` lists every decision with its note, as written here from the records;
    ``ask --line <id>`` shows each other open line whole, as constructed; a line not listed is
    refused, naming those that are."""
    fake_run(tmp_path, monkeypatch)
    public, whole, raw = asked()
    standing(monkeypatch, whole, raw)
    code, text = said(["ask", "--run", RID, "--decisions"])
    assert code == 0 and json.loads(weighed(text)) == {"run": RID, "decisions": log_of(raw)}
    for line in public["also_open"]:
        code, text = said(["ask", "--run", RID, "--line", line["id"]])
        assert code == 0 and json.loads(weighed(text)) == {"run": RID, "also_open": [line]}
    code, text = said(["ask", "--run", RID, "--line", "decision:set_nothing"])
    refused = json.loads(text)["refused"]
    assert code == 1 and refused["code"] == "no_such_line"
    assert all(ln["id"] in refused["message"] for ln in public["also_open"])


def test_a_small_ask_prints_as_before(tmp_path, monkeypatch):
    """An ask well under the cap prints exactly its public part, nothing left out or added."""
    fake_run(tmp_path, monkeypatch)
    public, whole, raw = asked(3, n_options=2, lines=[])
    assert size(public) < CAP // 2
    standing(monkeypatch, whole, raw)
    code, text = said(["ask", "--run", RID])
    assert code == 0 and text == json.dumps(public, ensure_ascii=False)


def test_a_long_runs_ask_stays_under_the_cap_and_its_whole_log_is_shown_in_parts(tmp_path,
                                                                                monkeypatch):
    """600 decisions: the ask lists the newest it has room for, in the log's order, and says how
    many of how many; ``ask --decisions`` prints the whole log in parts, each under the cap, each
    reached as the one before says, and together they are the log written here."""
    fake_run(tmp_path, monkeypatch)
    public, whole, raw = asked(600)
    standing(monkeypatch, whole, raw)
    code, text = said(["ask", "--run", RID])
    got = json.loads(weighed(text))
    ids = [d["id"] for d in public["decisions"]]
    k = len(got["decisions"])
    assert 0 < k < 600 and [d["id"] for d in got["decisions"]] == ids[-k:]
    assert f"The {k} newest of the 600 decisions" in got["left_out"]["decisions"]
    assert got["options"] == public["options"] and got["readings"] == public["readings"]
    code, first = said(["ask", "--run", RID, "--decisions"])
    assert code == 0
    texts = parts_of(first, "ask", "--decisions")
    assert len(texts) > 1
    assert joined(texts) == {"run": RID, "decisions": log_of(raw)}


def test_a_question_that_alone_passes_the_cap_is_shown_in_parts_none_cut(tmp_path, monkeypatch):
    """200 options of about 300 characters and a why of 30,000: the ask says it is in parts and
    how to reach each; joined as each part says, the options and the words are those built here,
    every one, and the other open lines and the log are named as left out."""
    fake_run(tmp_path, monkeypatch)
    public, whole, raw = asked(28, n_options=200, option_size=300, why=30_000)
    standing(monkeypatch, whole, raw)
    code, first = said(["ask", "--run", RID])
    assert code == 0
    head = json.loads(weighed(first))["part"]
    assert head["of"] > 2 and head["this"] == 1
    assert "too long to print whole" in head["words"]
    got = joined(parts_of(first, "ask"))
    for key in ("question", "words", "options", "readings", "last_refusal_exits", "line"):
        assert got[key] == public[key], key
    assert f"{INVOKE} ask --run {RID} --decisions" in got["left_out"]["decisions"]
    assert [ln["id"] for ln in got["also_open"]] == [ln["id"] for ln in public["also_open"]]


def test_a_part_is_shown_only_of_the_last_output_while_the_run_stands(tmp_path, monkeypatch):
    """Both ways: a part of the ask in parts is shown; once an answer is recorded, or the ask
    prints whole, or for a part past the last, it is refused in words, with nothing posted."""
    fake_run(tmp_path, monkeypatch)
    public, whole, raw = asked(28, n_options=200, option_size=300, why=30_000)
    standing(monkeypatch, whole, raw)
    monkeypatch.setattr(driver, "guard_budget", lambda *a, **k: None)
    sent: list[Any] = []

    class Answered:
        status = 200

        def json(self) -> Any:
            return {"decisions": [{"id": "r9", "seq": 9, "decision": sent[-1]}], "interview": []}

    def post(run, client, decision, note):
        sent.append(decision)
        return Answered()

    monkeypatch.setattr(driver, "post_decision", post)
    code, first = said(["ask", "--run", RID])
    n = json.loads(first)["part"]["of"]
    assert said(["ask", "--run", RID, "--part", "2"])[0] == 0
    code, log = said(["ask", "--run", RID, "--decisions"])  # the log, whole, between two parts
    assert code == 0 and "part" not in json.loads(log)
    assert said(["ask", "--run", RID, "--part", "3"])[1] == json.loads(
        (tmp_path / RID / "parts" / "ask.json").read_text())["parts"][2]
    code, text = said(["ask", "--run", RID, "--part", str(n + 1)])
    assert code == 1 and json.loads(text)["refused"]["code"] == "no_such_part"
    assert f"{n} parts" in json.loads(text)["refused"]["message"]
    code, text = said(["answer", "--run", RID, "nothing", "--reason", "No.", "--marker", "rows"])
    assert code == 1 and json.loads(text)["refused"]["by"] == "driver" and not sent
    assert said(["ask", "--run", RID, "--part", "2"])[0] == 0  # a refused answer changed nothing
    code, _ = said(["answer", "--run", RID, "impute", "--reason", "Fill.", "--marker", "rows"])
    assert code == 0 and len(sent) == 1
    code, text = said(["ask", "--run", RID, "--part", "2"])
    assert code == 1 and json.loads(text)["refused"]["code"] == "no_such_part"
    assert "answer" in json.loads(text)["refused"]["message"]
    said(["ask", "--run", RID])
    assert said(["ask", "--run", RID, "--part", "2"])[0] == 0  # in parts again
    small_public, small, small_raw = asked(3, n_options=2, lines=[])
    standing(monkeypatch, small, small_raw)
    assert said(["ask", "--run", RID])[1] == json.dumps(small_public, ensure_ascii=False)
    code, text = said(["ask", "--run", RID, "--part", "2"])
    assert code == 1 and json.loads(text)["refused"]["code"] == "no_such_part"
    assert len(sent) == 1


def huge_line() -> dict[str, Any]:
    """An other open line too long to print whole by itself: a declaration whose one field offers
    120 choices of 300 characters (39,004 bytes printed)."""
    return {"id": "decision:set_usual_intake:whos_in", "key": "set_usual_intake",
            "source": "declaration", "label": "Decide", "name": "the usual intake",
            "status": "open", "stage": "whos_in",
            "options": [{"id": "set_usual_intake:a", "label": "As recorded",
                         "decision": {"kind": "set_usual_intake"}}],
            "fields": [{"name": "column", "choices": [{"value": text_of(300, i)}
                                                      for i in range(120)]}]}


def recording(monkeypatch) -> list[Any]:
    """``answer``'s post stood in for: each decision it would post is kept in the list given."""
    monkeypatch.setattr(driver, "guard_budget", lambda *a, **k: None)
    sent: list[Any] = []

    class Answered:
        status = 200

        def json(self) -> Any:
            return {"decisions": [{"id": "r9", "seq": 9, "decision": sent[-1]}], "interview": []}

    def post(run, client, decision, note):
        sent.append(decision)
        return Answered()

    monkeypatch.setattr(driver, "post_decision", post)
    return sent


def test_each_output_in_parts_goes_on_by_its_own_next_whatever_printed_in_parts_after_it(
        tmp_path, monkeypatch):
    """The question, the whole log and one other open line, each too long to print whole, printed
    one after another, each in parts: each one's ``next`` names that output's own command, passes
    the analyst's hook as written, and prints that output's next part, so the parts joined are
    each output as constructed here, whichever printed in parts after it (the verifier's P1: the
    question's ``next`` once printed the log's part)."""
    from research.swarm import analyst_hook

    fake_run(tmp_path, monkeypatch)
    lines = other_lines() + [huge_line()]
    public, whole, raw = asked(600, n_options=200, option_size=300, why=30_000, lines=lines)
    standing(monkeypatch, whole, raw)
    big = lines[-1]["id"]
    firsts = {}
    for flags in ((), ("--decisions",), ("--line", big)):
        code, first = said(["ask", "--run", RID, *flags])
        head = json.loads(weighed(first))["part"]
        assert code == 0 and head["this"] == 1 and head["of"] > 1, flags
        assert analyst_hook.bash_refusal(head["next"], analyst_hook.REPO) is None, head["next"]
        firsts[flags] = first
    question = joined(parts_of(firsts[()], "ask"))
    for key in ("question", "words", "options", "readings", "last_refusal_exits", "line"):
        assert question[key] == public[key], key
    assert joined(parts_of(firsts[("--decisions",)], "ask", "--decisions")) == {
        "run": RID, "decisions": log_of(raw)}
    assert joined(parts_of(firsts[("--line", big)], "ask", "--line", big)) == {
        "run": RID, "also_open": [lines[-1]]}


@pytest.mark.parametrize("line_id,written", [
    ("finding:pack::dietary::implausible_intake", "finding:pack::dietary::implausible_intake"),
    ("finding:BMI (kg/m2) high", "'finding:BMI (kg/m2) high'")], ids=["plain", "shell_words"])
def test_a_lines_next_part_reads_back_its_id_whole(line_id, written):
    """Both ways: an id a shell reads as one word is written as it is; one it would split or
    read (a space, parentheses) is quoted. Either way the analyst's hook passes the command and
    the part's parser reads back the id as given."""
    import shlex

    from research.swarm import analyst_hook

    args = driver.part_parser().parse_args(["ask", "--run", RID, "--line", line_id, "--part", "1"])
    command = f"{driver.output_command(args, RID)} --part 2"
    assert command == f"{INVOKE} ask --run {RID} --line {written} --part 2"
    assert analyst_hook.bash_refusal(command, analyst_hook.REPO) is None, command
    back = driver.part_parser().parse_args(shlex.split(command)[3:])
    assert (back.command, back.line, back.part, driver.output_of(back)) == (
        "ask", line_id, 2, f"ask --line {line_id}")


def test_a_part_is_refused_where_its_own_output_printed_whole_and_shown_where_another_did(
        tmp_path, monkeypatch):
    """Both ways, each output by its own parts: the question printed whole refuses the question's
    part 2 while the log's and the line's parts are shown; the log printed whole refuses the
    log's while the line's is shown; and once an answer is recorded, the line's is refused, in
    words naming the command that shows the line as it stands."""
    fake_run(tmp_path, monkeypatch)
    sent = recording(monkeypatch)
    lines = other_lines() + [huge_line()]
    big = lines[-1]["id"]
    public, whole, raw = asked(600, n_options=200, option_size=300, why=30_000, lines=lines)
    standing(monkeypatch, whole, raw)
    for flags in ((), ("--decisions",), ("--line", big)):
        assert "part" in json.loads(said(["ask", "--run", RID, *flags])[1])
    shown_log = said(["ask", "--run", RID, "--decisions", "--part", "2"])
    shown_line = said(["ask", "--run", RID, "--line", big, "--part", "2"])
    assert shown_log[0] == 0 and json.loads(shown_log[1])["part"]["this"] == 2
    assert shown_line[0] == 0 and json.loads(shown_line[1])["part"]["this"] == 2

    _, short, _ = asked(600, lines=lines)  # the question prints whole, shortened
    standing(monkeypatch, short, raw)
    assert "part" not in json.loads(said(["ask", "--run", RID])[1])
    code, text = said(["ask", "--run", RID, "--part", "2"])
    refused = json.loads(text).get("refused") or {}
    assert code == 1 and refused.get("code") == "no_such_part", text[:300]
    assert "No output of ask is kept in parts" in refused["message"]
    assert said(["ask", "--run", RID, "--decisions", "--part", "2"]) == shown_log
    assert said(["ask", "--run", RID, "--line", big, "--part", "2"]) == shown_line

    _, small, small_raw = asked(3, lines=lines)  # the log prints whole
    standing(monkeypatch, small, small_raw)
    assert "part" not in json.loads(said(["ask", "--run", RID, "--decisions"])[1])
    code, text = said(["ask", "--run", RID, "--decisions", "--part", "2"])
    refused = json.loads(text).get("refused") or {}
    assert code == 1 and refused.get("code") == "no_such_part", text[:300]
    assert "No output of ask --decisions is kept in parts" in refused["message"]
    assert said(["ask", "--run", RID, "--line", big, "--part", "2"]) == shown_line

    code, _ = said(["answer", "--run", RID, "impute", "--reason", "Fill.", "--marker", "rows"])
    assert code == 0 and len(sent) == 1
    code, text = said(["ask", "--run", RID, "--line", big, "--part", "2"])
    refused = json.loads(text).get("refused") or {}
    assert code == 1 and refused.get("code") == "no_such_part", text[:300]
    assert "answer ran after it" in refused["message"]
    assert f"{INVOKE} ask --run {RID} --line {big} shows the run as it stands" in refused["message"]


def results_served(families: int) -> dict[str, Any]:
    return {"purpose": "prediction", "families": [
        {"family": f"family_{i}", "scores": {"auc": 0.7}, "words": text_of(600, i)}
        for i in range(families)]}


@pytest.mark.parametrize("families", [3, 120], ids=["small", "past_the_cap"])
def test_results_past_the_cap_come_in_parts_and_small_ones_whole(tmp_path, monkeypatch,
                                                                 families):
    fake_run(tmp_path, monkeypatch)
    served = results_served(families)
    monkeypatch.setattr(driver, "view_of", lambda run, client: {"state": {}, "stages": {}})
    monkeypatch.setattr(driver, "keep_markers", lambda run, client, command: None)
    monkeypatch.setattr(driver.Client, "get", lambda self, path, **q: driver.Response(
        200, json.dumps(served).encode(), {}))
    code, first = said(["results", "--run", RID, "--wait", "0"])
    assert code == 0
    whole = {"results": served, "stages": {}}
    if families == 3:
        assert first == json.dumps(whole, ensure_ascii=False)
        return
    assert size(whole) > CAP
    assert joined(parts_of(first, "results")) == whole


@pytest.mark.parametrize("analyst", ["scripted", "agent"])
def test_only_an_agents_output_is_held_to_the_cap(tmp_path, monkeypatch, analyst):
    """A scripted analyst reads the JSON in its own process, past no harness: its run's ask
    prints whole, as before; an agent's is held to the cap."""
    folder = fake_run(tmp_path, monkeypatch)
    meta = json.loads((folder / "driver.json").read_text())
    (folder / "driver.json").write_text(json.dumps(meta | {"analyst": analyst}))
    public, whole, raw = asked()
    standing(monkeypatch, whole, raw)
    code, text = said(["ask", "--run", RID])
    if analyst == "scripted":
        assert text == json.dumps(public, ensure_ascii=False) and len(text) > CAP
    else:
        weighed(text)


def test_a_part_read_records_nothing_and_is_logged_as_one(tmp_path, monkeypatch):
    """A want and an account past the cap: each prints in parts; reading a part logs the command
    with its part and none of what the command records, so the run keeps one want and one
    account, and the account is still given once."""
    folder = fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(driver, "view_of", lambda run, client: {"state": {}})
    long = text_of(30_000, 2)
    code, first = said(["want", "--run", RID, long, "--kind", "check", "--reason", "Needed.",
                        "--marker", "rows"])
    assert code == 0
    assert joined(parts_of(first, "want"))["wanted"]["what"] == long
    code, first = said(["account", "--run", RID, long])
    assert code == 0
    assert joined(parts_of(first, "account"))["account"]["text"] == long
    logged = [json.loads(x) for x in (folder / "commands.jsonl").read_text().splitlines()]
    reads = [e for e in logged if "part" in e["args"]]
    assert reads and all(set(e["summary"]) == {"exit", "part"} for e in reads)
    assert sum("wanted" in e["summary"] for e in logged) == 1
    assert sum("account" in e["summary"] for e in logged) == 1
    code, text = said(["account", "--run", RID, "Again."])
    assert code == 1 and json.loads(text)["refused"]["code"] == "account_given"


def test_a_part_read_takes_only_the_run_and_the_part(tmp_path, monkeypatch):
    fake_run(tmp_path, monkeypatch)
    assert driver.main(["answer", "--run", RID, "--part", "2", "impute"], out=lambda s: None) == 2
    assert driver.main(["ask", "--run", RID, "--part", "0"], out=lambda s: None) == 2
    code, text = said(["answer", "--run", RID, "--part", "1"])
    assert code == 1 and json.loads(text)["refused"]["code"] == "no_such_part"


# ── D2: several options in one answer ────────────────────────────────────────

LENS_ENTRY = {"key": "lens", "options": [
    {"value": "dietary", "label": "Dietary intake", "consequence": "c1"},
    {"value": "clinical", "label": "Clinical", "consequence": "c2"},
    {"value": "survey", "label": "Survey design", "consequence": "c3"}]}


def asked_of(tmp_path, monkeypatch, key: str, composed: dict[str, Any]) -> dict[str, Any]:
    fake_run(tmp_path, monkeypatch)
    run = driver.Run.load(RID)
    view = {"state": {"purpose": "inference"}, "stages": {}, "decisions": []}
    monkeypatch.setattr(driver, "wait_open", lambda run, client, wait: (
        view, {"key": key, "status": "open"}))
    monkeypatch.setattr(driver, "quest_lines", lambda run, client: ([], None))
    monkeypatch.setattr(driver, "teaching", lambda client, purpose, key: None)
    monkeypatch.setattr(driver, "served", lambda run, client: (lambda stage: None))
    monkeypatch.setattr(driver.opts, "compose", lambda *a: copy.deepcopy(composed))
    return driver.the_question(run, driver.Client(9), 0)


def test_ask_says_how_the_lens_takes_several_and_the_form_it_names_is_read(tmp_path,
                                                                            monkeypatch):
    lens = compose("lens", {}, LENS_ENTRY)
    q = asked_of(tmp_path, monkeypatch, "lens", lens)
    words = q["multi"]
    assert f"{INVOKE} answer --run {RID} dietary clinical " in words
    assert "separated by spaces" in words and "commas" in words
    assert opts.resolve(lens, ["dietary", "clinical"], []) == {
        "kind": "set_lens", "lenses": ["dietary", "clinical"]}


def test_ask_says_how_the_exclusions_take_several_rules_and_each_claim_holds(tmp_path,
                                                                              monkeypatch):
    """The words name the repeated own-rule form, a setting's place and the option answered
    alone; each is what ``resolve`` does."""
    composed = exclusions()
    q = asked_of(tmp_path, monkeypatch, "exclusions", composed)
    words = q["multi"]
    assert f"{INVOKE} answer --run {RID} custom levels levels --set ... " in words
    assert ("The options custom (Your own range), eligibility (Who it is about) and levels (By "
            "level) may each be named more than once") in words
    assert "--set 2.level_column=VALUE sets the second option's level_column" in words
    assert "The option none (Keep every row) is answered alone, never beside another option." in words
    got = opts.resolve(composed, ["custom", "levels", "levels"], [
        "1.column=age", "1.low=20", "2.level_column=sex", "2.exclude=2", "3.level_column=smoker",
        "3.exclude=1"])
    assert [r["kind"] for r in got["rules"]] == ["range", "levels", "levels"]
    assert [r["column"] for r in got["rules"]] == ["age", "sex", "smoker"]  # each by its place
    with pytest.raises(opts.Unanswerable, match="answered alone"):
        opts.resolve(composed, ["none", "custom"], ["column=age", "low=20"])
    assert opts.resolve(composed, ["none"], []) == {"kind": "set_exclusions", "rules": []}
    with pytest.raises(opts.Unanswerable, match="write its place"):
        opts.resolve(composed, ["levels", "levels"], ["level_column=sex"])


def test_a_question_of_one_answer_says_nothing_of_several(tmp_path, monkeypatch):
    one = {"options": [{"id": "impute", "label": "Fill",
                        "decision": {"kind": "set_missing", "strategy": "impute"}}],
           "fields": [], "data": {}}
    assert "multi" not in asked_of(tmp_path, monkeypatch, "missing", one)


def test_a_comma_joined_answer_is_refused_naming_the_form_that_works():
    lens = compose("lens", {}, LENS_ENTRY)
    with pytest.raises(opts.Unanswerable) as caught:
        opts.resolve(lens, ["dietary,clinical,survey"], [])
    assert "dietary clinical survey" in str(caught.value)
    assert "separated by spaces" in str(caught.value)
    assert opts.resolve(lens, ["dietary", "clinical", "survey"], [])["lenses"] == [
        "dietary", "clinical", "survey"]
    one = {"options": [{"id": "impute", "decision": {"kind": "set_missing", "strategy": "impute"}},
                       {"id": "complete_case",
                        "decision": {"kind": "set_missing", "strategy": "complete_case"}}]}
    with pytest.raises(opts.Unanswerable, match="takes one option at a time"):
        opts.resolve(one, ["impute,complete_case"], [])
    with pytest.raises(opts.Unanswerable, match="No option dietary,lifestyle is offered"):
        opts.resolve(lens, ["dietary,lifestyle"], [])  # a part not offered names no form


def test_an_offered_id_that_holds_a_comma_is_taken_as_written():
    composed = {"options": [{"id": "a,b", "decision": {"kind": "set_lens", "lenses": ["a,b"]}},
                            {"id": "a", "decision": {"kind": "set_lens", "lenses": ["a"]}},
                            {"id": "b", "decision": {"kind": "set_lens", "lenses": ["b"]}}],
                "multi": "lenses"}
    assert opts.resolve(composed, ["a,b"], []) == {"kind": "set_lens", "lenses": ["a,b"]}


def test_the_driver_refuses_a_comma_joined_answer_in_those_words(tmp_path, monkeypatch):
    fake_run(tmp_path, monkeypatch)
    lens = compose("lens", {}, LENS_ENTRY)
    view = {"state": {"purpose": "inference", "seal_opened": None}, "stages": {}, "decisions": []}
    monkeypatch.setattr(driver, "the_question", lambda run, client, wait: {
        "_composed": lens, "_ways": [], "_view": view, "_declared": {}})
    code, payload = run_main(["answer", "--run", RID, "dietary,clinical", "--reason", "Both.",
                              "--marker", "rows"])
    assert code == 1 and payload["refused"]["code"] == "not_offered"
    assert "dietary clinical" in payload["refused"]["message"]


def test_every_command_the_new_words_name_is_one_the_hook_passes(tmp_path, monkeypatch):
    """The commands the driver now names (the log's, a line's, a next part's, the several
    options' form) pass the analyst's hook as written, each placeholder (ID, MARKER, WHY) filled
    in, and the driver reads each."""
    import shlex

    from research.swarm import analyst_hook

    form = several_words(tmp_path, monkeypatch)  # before the_question is stood in for below
    public, whole, raw = asked(28, n_options=200, option_size=300, why=30_000)
    fake_run(tmp_path, monkeypatch)
    standing(monkeypatch, whole, raw)
    got = json.loads(said(["ask", "--run", RID])[1])
    words = [got["left_out"]["decisions"], got["left_out"]["also_open"], got["part"]["next"],
             form]
    named = [w[w.index(INVOKE):].split(" lists ")[0].split(" shows ")[0].rstrip(".")
             for w in words]
    named[1] = named[1].replace("--line ID", "--line decision:set_usual_intake")
    named[3] = named[3].replace("MARKER", "rows").replace("WHY", "Both")
    for command in named:
        assert analyst_hook.bash_refusal(command, analyst_hook.REPO) is None, command
        driver.build_parser().parse_args(shlex.split(command)[3:])


def several_words(tmp_path, monkeypatch) -> str:
    (tmp_path / "lens").mkdir()
    lens = compose("lens", {}, LENS_ENTRY)
    return asked_of(tmp_path / "lens", monkeypatch, "lens", lens)["multi"]


def test_answer_and_preview_help_say_how_several_options_are_named(capsys):
    for command in ("answer", "preview"):
        assert driver.main([command, "--help"], out=lambda s: None) == 0
        text = " ".join(capsys.readouterr().out.split())
        assert "each id its own word, separated by spaces, never joined by commas" in text, command
