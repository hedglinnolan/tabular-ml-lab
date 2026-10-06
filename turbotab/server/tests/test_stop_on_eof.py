"""``--stop-on-eof`` (how the desktop launcher stops the server) must not hand its stdin pipe on.

On Windows a new process gets this one's standard handles, and a read pending on a synchronous
pipe blocks every other use of it: with the server reading its stdin, each job worker hung as it
started, before any Python ran, and every job stayed queued (the Windows launcher's smoke check,
October 2026). The rule this checks holds on every platform: once the server watches its stdin,
a process it starts does not share that pipe, a job runs, and closing the pipe stops the server.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

HOST = r"""
import os, stat, subprocess, sys, threading

from turbotab.core.jobs import JobRunner
from turbotab.core.tests import toy_stages as toy
from turbotab.server.__main__ import stop_on_eof

stopped = threading.Event()
assert stop_on_eof(stopped.set) is not None
probe = "import os, stat; print(stat.S_ISFIFO(os.fstat(0).st_mode))"
child = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60)
print("a child's stdin is the pipe:", child.stdout.strip(), flush=True)
with JobRunner(1, preload=(), start_seconds=30) as runner:
    job = runner.submit(toy.add, 2, 3, label="add")
    print("job:", runner.wait(job, timeout=60).state, flush=True)
print("waiting for EOF", flush=True)
print("stopped:", stopped.wait(30), flush=True)
"""


def test_a_server_watching_its_stdin_hands_no_process_the_pipe_and_stops_when_it_closes():
    host = subprocess.Popen([sys.executable, "-c", HOST], cwd=REPO, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        lines = []
        for line in host.stdout:
            lines.append(line.rstrip())
            if line.startswith("waiting for EOF") or line.startswith("Traceback"):
                break
        rest, _ = host.communicate(timeout=60)  # closes its stdin: how the launcher stops it
        lines += rest.splitlines()
    finally:
        if host.poll() is None:
            host.kill()
    out = "\n".join(lines)
    assert "a child's stdin is the pipe: False" in out, out
    assert "job: done" in out, out
    assert "stopped: True" in out, out
    assert host.returncode == 0, out
