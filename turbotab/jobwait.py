"""Wait for a background job the way a person does — against a clock.

`TEST-040`. Six places in this suite polled a job with a bounded `range(N)`
loop and then asserted `status == "done"`. Four of them had **no wait between
polls**, so N round-trips elapsed in milliseconds while a real fit takes
seconds, and the assertion that fired said::

    AssertionError: ('running', None)

That is the instrument reporting *the app had not answered yet* as *the app
produced the wrong answer* — the one assertion `README.md`'s governing rule
forbids, made by the test rather than by the app. It is worse here than an
app defect would be, because the number this project reports every loop is a
suite pass count: the same commit returned `1423 passed` on one machine and
`1 failed, 1422 passed` on a loaded one, and a count that moves between two
honest runs cannot settle anything.

**The fix is not a bigger N.** `range(4000)` still elapses in milliseconds on
a fast machine and still fails with the job's status on a slow one; it only
moves the load at which the lie appears. The distinction the instrument has to
draw is between *did not finish* and *finished wrong*, and an iteration count
cannot draw it because iterations are not time.

So: poll against a **deadline**, sleep between polls, and make the two
outcomes different kinds of event.

- The job reached a terminal state → return it. The caller's own
  `assert job["status"] == "done"` then runs against a job that genuinely
  finished, so a failure of *that* assertion is unambiguously an app fact.
- The deadline passed → raise `JobDidNotSettle`, whose message says how long
  it waited and what the last status was. A timeout is never returned as a
  job state, so no caller can accidentally assert against one.

**Why the default is generous.** A test that finishes as soon as the job is
terminal pays nothing for a large timeout; the timeout is only reached when
something is genuinely wrong, and there the slow honest answer beats the fast
false one. `TURBOTAB_JOB_TIMEOUT` overrides it for a machine where even this
is not enough.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict

#: Seconds. Not a domain constant — an engineering bound on how long a fit in
#: this suite may take before *something is wrong* is the better explanation
#: than *the machine is busy*. The slowest observed real job in this suite is
#: the 600-resample bootstrap in `test_the_whole_pipeline_is_refitted`, which
#: ran in ~30 s on an unloaded machine; 120 s leaves a 4x margin for load.
DEFAULT_TIMEOUT = float(os.environ.get("TURBOTAB_JOB_TIMEOUT", "120"))

#: Seconds between polls. Small enough that a fast job is not slowed by the
#: instrument, large enough that a slow one is not answered 20,000 times.
POLL_INTERVAL = 0.05


class JobDidNotSettle(AssertionError):
    """The deadline passed with the job still running.

    An `AssertionError` so pytest reports it as a failure rather than an
    error, and its own type so a caller that wants to tell a timeout from a
    failed fit can.
    """


def settle(client, job: Dict[str, Any], *,
           timeout: float | None = None,
           poll: float = POLL_INTERVAL) -> Dict[str, Any]:
    """Poll `/job/{id}` until it is terminal, or raise.

    `job` is the dict the `/train` (or `/instability`, or any job-returning)
    route replied with; only its `id` is read, so a caller may pass either the
    POST response or a later poll of it.

    Returns the **terminal** job state. Never returns a running one — that is
    the whole point.
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    job_id = job["id"]
    name = job.get("name") or job_id
    deadline = time.monotonic() + timeout
    state = job
    polls = 0
    while True:
        state = client.get(f"/job/{job_id}").json()
        polls += 1
        if state.get("terminal"):
            return state
        if time.monotonic() >= deadline:
            raise JobDidNotSettle(
                f"the job did not finish in {timeout:g} seconds "
                f"({name!r}, {polls} polls, last status "
                f"{state.get('status')!r}, progress {state.get('progress')!r}). "
                "This is a TIMEOUT, not a wrong answer: the app was still "
                "working when the test gave up. Raise TURBOTAB_JOB_TIMEOUT if "
                "this machine is slower than the 4x margin allows.")
        time.sleep(poll)


def settle_done(client, job: Dict[str, Any], **kw) -> Dict[str, Any]:
    """`settle`, plus the assertion almost every caller makes next.

    Kept separate from `settle` so a test that wants to observe a *failed* job
    — and several do — is not forced through a helper that refuses one.
    """
    state = settle(client, job, **kw)
    assert state["status"] == "done", (state["status"], state.get("error"))
    return state
