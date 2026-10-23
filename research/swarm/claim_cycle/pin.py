"""The claim cycle's pin: the one corpus folder and the one proposals file a cycle's theorist and
experimentalist may name, written by the orchestrator before it starts them, and the tools' command
line, read the same way by the tools (``tools.py``) and by the hook (``hook.py``).

    python -m research.swarm.claim_cycle.pin --corpus <folder> --proposals <file.jsonl>
    python -m research.swarm.claim_cycle.pin --clear

**Why.** The shuffled control's copy holds every record of the corpus beside the planted records
(``shuffle.py``), so an agent that reads both folders finds the planted records, the run ids the
copy holds and the corpus does not, and the derangement, by matching the two folders' markers,
without the sealed file. So **one cycle runs at a time**: the orchestrator pins the corpus folder
and the proposals file of one cycle, runs that cycle's agents, clears the pin, and only then pins
the other cycle's. The hook refuses a theorist's or an experimentalist's command that names any
other folder or file (:func:`refusal`), and every command of theirs while nothing is pinned.

**The pin** is ``.claude/claim_cycle.json`` at the repo root (:data:`PIN`; git keeps nothing of
``.claude/`` but the agents' definitions), holding the two paths resolved. :func:`write` refuses a
second pin while one stands, a corpus folder that does not exist, a proposals file that is not
JSON lines (``.jsonl``) in a folder that exists, and one in the corpus folder. A path a command
names is read from the repo root, where the hook runs the tools, and resolved, so
``./cycle/corpus/`` names ``cycle/corpus`` and a link names what it links to.

**A proposals file is one cycle's.** One cycle's proposals cite the run ids its corpus holds,
which the copy keeps, and that corpus's marker numbers, which the copy permutes, so a theorist of
the other cycle served them could match the two and find the derangement. So each pin's proposals
file is recorded with its corpus folder beside the pin (:func:`ledger_of`,
``.claude/claim_cycle.pinned.json``, kept when the pin is cleared): a re-pin of the same cycle
names its file again, and :func:`write` refuses a proposals file recorded for another corpus
folder, and one that already holds lines no recorded pin named (written outside a pin, or before
the record).

It uses the standard library alone, as the hook does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PIN = REPO / ".claude" / "claim_cycle.json"
TOOLS = "python -m research.swarm.claim_cycle.tools"
COMMANDS = ("index", "records", "proposals", "propose")
# What each command names, and which of the pin's two paths it must be.
NAMED = {"index": ("corpus", "corpus"), "records": ("corpus", "corpus"),
         "proposals": ("file", "proposals"), "propose": ("out", "proposals")}
KINDS = {"corpus": "corpus folder", "proposals": "proposals file"}


class CommandRefused(ValueError):
    """A command line the tools' parser does not read, in its words."""


class PinRefused(ValueError):
    """A pin the orchestrator cannot write, or one that does not read, in words."""


class _Raising(argparse.ArgumentParser):
    """The tools' parser, raising where argparse would print and exit."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise CommandRefused(message)


def build_parser(add_help: bool = True, raising: bool = False) -> argparse.ArgumentParser:
    """The tools' command line (``tools.py``'s docstring); the hook reads a command with
    ``add_help=False, raising=True``, so a help flag is refused and nothing is printed."""
    cls = _Raising if raising else argparse.ArgumentParser
    parser = cls(prog=TOOLS, add_help=add_help,
                 description="The commands the theorist and the experimentalist run.")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="one line per record", add_help=add_help)
    index.add_argument("corpus")
    records = sub.add_parser("records", help="each record's view", add_help=add_help)
    records.add_argument("corpus")
    records.add_argument("--run", action="append", default=None, help="a run's id (repeatable)")
    records.add_argument("--dataset", default=None, help="a dataset's label, as index prints it")
    records.add_argument("--fields", default=None, help="fields of the view, comma-separated")
    proposals = sub.add_parser("proposals", help="the proposals file's proposals",
                               add_help=add_help)
    proposals.add_argument("file")
    propose = sub.add_parser("propose", help="validate one proposal and append it",
                             add_help=add_help)
    propose.add_argument("--out", required=True, help="the proposals file (.jsonl)")
    propose.add_argument("proposal", help="the proposal, one JSON object")
    return parser


def _resolved(path: str | Path, repo: Path) -> Path:
    """``path`` as the tools read it, run from ``repo``: no ``~`` is expanded, as no shell runs."""
    return (Path(repo) / Path(path)).resolve()


def _shown(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(Path(repo).resolve()))
    except ValueError:
        return str(path)


def body(corpus: str | Path, proposals: str | Path, repo: Path = REPO) -> dict[str, str]:
    """The pin's contents: the corpus folder and the proposals file, each read from ``repo`` and
    resolved."""
    return {"corpus": str(_resolved(corpus, repo)), "proposals": str(_resolved(proposals, repo))}


def read(pin: Path | None = None) -> dict[str, str] | None:
    """The pin at ``pin`` (default :data:`PIN`), or None where nothing is pinned; refused where
    the file does not read as a pin."""
    pin = Path(pin or PIN)
    if not pin.exists():
        return None
    try:
        data = json.loads(pin.read_text("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        data = None
    if not (isinstance(data, dict) and set(data) == set(KINDS)
            and all(isinstance(v, str) and v for v in data.values())):
        raise PinRefused("The claim cycle's pin does not read as one (a corpus folder and a "
                         "proposals file), so nothing runs until the orchestrator pins the cycle "
                         "again.")
    return data


def ledger_of(pin: Path) -> Path:
    """Where each proposals file pinned at ``pin`` is recorded with its corpus folder: beside it,
    ``<pin's name>.pinned.json``."""
    return Path(pin).with_name(f"{Path(pin).stem}.pinned.json")


def _recorded(ledger: Path) -> dict[str, str]:
    """The record at ``ledger``: each proposals file pinned, resolved, with the corpus folder it
    was pinned with; empty where none was; refused where it does not read as one."""
    if not ledger.exists():
        return {}
    try:
        data = json.loads(ledger.read_text("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        data = None
    if not (isinstance(data, dict) and all(isinstance(k, str) and isinstance(v, str) and v
                                           for k, v in data.items())):
        raise PinRefused(f"The record of the cycles pinned before ({ledger}) does not read as one "
                         f"(each proposals file with its corpus folder), so no cycle is pinned "
                         f"until the orchestrator mends it.")
    return data


def _holds_lines(file: Path) -> bool:
    """Whether ``file`` holds anything but blank space (one that cannot be read is taken to)."""
    try:
        return file.is_file() and bool(file.read_text("utf-8", errors="replace").strip())
    except OSError:
        return True


def write(corpus: str | Path, proposals: str | Path, pin: Path | None = None,
          repo: Path = REPO) -> dict[str, str]:
    """Pin one cycle's corpus folder and proposals file (the module docstring), and record the
    file with its folder (:func:`ledger_of`)."""
    pin = Path(pin or PIN)
    if pin.exists():
        raise PinRefused(f"A cycle is pinned already ({pin}); one cycle runs at a time, so clear "
                         f"it (--clear) when that cycle's agents are done, then pin the next.")
    got = body(corpus, proposals, repo)
    folder, file = Path(got["corpus"]), Path(got["proposals"])
    if not folder.is_dir():
        raise PinRefused(f"{corpus} is not a folder; a cycle's corpus is a folder of records.")
    if file.suffix != ".jsonl" or not file.parent.is_dir():
        raise PinRefused(f"{proposals} is not a proposals file a cycle can write: one named "
                         f".jsonl, in a folder that exists.")
    if file == folder or folder in file.parents:
        raise PinRefused(f"{proposals} lies in the corpus folder; a cycle's corpus folder holds "
                         f"its records only, so its proposals file lies outside it.")
    ledger = ledger_of(pin)
    recorded = _recorded(ledger)
    before = recorded.get(str(file))
    if before is not None and before != str(folder):
        raise PinRefused(f"{proposals} was pinned before for the corpus folder "
                         f"{_shown(Path(before), repo)}; a proposals file is one cycle's, since "
                         f"its proposals cite that corpus's runs and numbers, so this cycle "
                         f"writes its own.")
    if before is None and _holds_lines(file):
        raise PinRefused(f"{proposals} holds lines, and no pin recorded in {ledger.name} named "
                         f"it; a cycle's proposals file is a new one, or the one its own cycle "
                         f"was pinned with before, so one cycle's proposals never reach the "
                         f"other's agents.")
    pin.parent.mkdir(parents=True, exist_ok=True)
    with open(pin, "x", encoding="utf-8") as fh:
        json.dump(got, fh, indent=1, sort_keys=True)
        fh.write("\n")
    ledger.write_text(json.dumps({**recorded, str(file): str(folder)}, indent=1, sort_keys=True)
                      + "\n", "utf-8")
    return got


def clear(pin: Path | None = None) -> bool:
    """Remove the pin; whether there was one."""
    pin = Path(pin or PIN)
    if not pin.exists():
        return False
    pin.unlink()
    return True


def refusal(words: Sequence[str], role: str, pin: Path | None = None,
            repo: Path = REPO) -> str | None:
    """Why the tools' command ``words`` (the command and its arguments) is refused to ``role``
    under the pin, or None where every path it names is the pinned one of its kind."""
    named = f"{'an' if role[:1] in 'aeiou' else 'a'} {role}"  # "a theorist", "an experimentalist"
    try:
        pinned = read(pin)
    except PinRefused as exc:
        return str(exc)
    if pinned is None:
        return (f"No claim cycle is pinned, so {named} runs nothing: the orchestrator pins the "
                f"cycle's corpus folder and proposals file before it starts the cycle's agents.")
    try:
        args = build_parser(add_help=False, raising=True).parse_args(list(words))
    except CommandRefused as exc:
        return f"The command does not read as one tools command ({exc})."
    attr, kind = NAMED[args.command]
    given = _resolved(getattr(args, attr), repo)
    if given != Path(pinned[kind]).resolve():
        return (f"{getattr(args, attr)} is not this cycle's {KINDS[kind]}: {named} reads only the "
                f"corpus folder and the proposals file its prompt names, and this cycle's "
                f"{KINDS[kind]} is {_shown(Path(pinned[kind]), repo)}.")
    return None


def main(argv: Sequence[str] | None = None, out: Any = print) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.claim_cycle.pin",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", help="the cycle's corpus folder")
    parser.add_argument("--proposals", help="the cycle's proposals file (.jsonl)")
    parser.add_argument("--clear", action="store_true", help="remove the pin")
    args = parser.parse_args(argv)
    try:
        if args.clear:
            if args.corpus or args.proposals:
                raise PinRefused("--clear takes no paths.")
            out(json.dumps({"cleared": clear()}))
            return 0
        if not (args.corpus and args.proposals):
            raise PinRefused("A pin names the cycle's corpus folder and its proposals file "
                             "(--corpus and --proposals).")
        out(json.dumps({"pinned": write(args.corpus, args.proposals, repo=Path.cwd())}))
        return 0
    except PinRefused as exc:
        out(json.dumps({"refused": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
