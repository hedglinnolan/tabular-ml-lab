"""`TEST-050` — there was a size above which the instrument could not answer.

`pageharness.py`'s docstring holds the rule: **dumb is allowed, lying is not.**
`__emit` reported through `process.stdout.write` inside an `exit` handler, and
stdout on a PIPE is asynchronous in node — `process.exit` discards whatever has
not drained. `run()` captures through a pipe, so **every emit larger than the
64 KB pipe buffer arrived cut in half.**

## What that actually was, corrected from what it looked like

The first reading of this was *"a truncation landing where the JSON stays valid
returns a smaller answer and a test asserts on it"*. **That is not true and the
probe is what said so.** A truncated JSON document is never a complete one, so
`json.loads` in `run()` always raises: there is no shortened-but-parseable
result, and the revert probe written for the silent version came back
`RED FOR THE WRONG REASON` with an `Unterminated string`.

What it is instead is **misattribution**, which is milder and still a defect
this project cares about: an instrument limit that presents as
`JSONDecodeError` out of the drive, i.e. as *the page produced something
unparseable*. `AGENT_ONBOARD` §07.2's standing answer is the revert probe, and
a harness whose failure mode blames the code under test is the thing a probe
cannot see past. So the assertions below convert a `run()` failure into a
sentence that names the HARNESS, and the probe checks for that sentence.

## How it surfaced

Reachable since `__emit` was written; reached at `GUIDED-198`, when the Features
catalogue grew a parameter control per row and
`test_the_page_says_what_the_record_says.py::test_claim[the features step
reaches its end]` — which emits `__harness.html('featBuild')`, the whole
rendered catalogue — crossed 64 KB and failed at character 45,959.

## The floor, and why it is where it is

`_FLOOR` is several pipe buffers and well below anything a drive would produce,
so it measures the failure rather than the platform. The boundary was located by
probe: 60,000 characters came back whole on the old writer and 200,000 did not.

## Not covered, said out loud

* **stderr.** `run()` reads it only when reporting a failure and nothing asserts
  on it, so whether IT truncates is untested here.
* **The 90-second timeout.** A payload large enough to be slow to serialize
  would hit that instead, and this does not find where that is.
* **Any platform but this one.** The 64 KB figure is macOS's pipe buffer, read
  off the observed cut rather than from a constant, and `_FLOOR` is set well
  above it rather than derived from it.
"""
from __future__ import annotations

import json

import pytest

from turbotab import pageharness as PH

pytestmark = pytest.mark.skipif(not PH.available(),
                                reason="no JS engine on this machine")

#: Several pipe buffers. Big enough that the old writer could not have carried
#: it, small enough to serialize in well under a second.
_FLOOR = 250_000


def _emit(body):
    """`PH.run`, with a harness failure turned into a sentence about the harness.

    Without this the regression reads as `JSONDecodeError` from inside
    `pageharness.py`, which is the misattribution the row is about — a reader,
    and a revert probe, would see the page blamed for the instrument.
    """
    try:
        return PH.run(body)
    except Exception as exc:                      # noqa: BLE001 — re-raised below
        raise AssertionError(
            f"the harness could not report this answer at all: "
            f"{type(exc).__name__}: {str(exc)[:200]}. The page's controller ran; "
            f"what failed is `__emit`'s write back to the caller."
        ) from exc


def test_an_emit_larger_than_a_pipe_buffer_comes_back_whole():
    """The regression, at a size the old writer could not carry."""
    out = _emit("var s = ''; while (s.length < %d) s += 'abcdefgh';\n"
                "__emit({n: s.length, s: s});" % _FLOOR)
    assert out["n"] >= _FLOOR, out["n"]
    assert len(out["s"]) == out["n"], (
        f"the page said it was emitting {out['n']} characters and "
        f"{len(out['s'])} arrived")
    assert set(out["s"]) == set("abcdefgh")


def test_a_long_list_arrives_with_every_entry():
    """The shape the Features catalogue produces: many small pieces, not one blob.

    `__harness.html` of a container is one long string; a drive that reports a
    row per control is a long list. Both cross the buffer and the second is the
    one a caller counts, so a missing tail would read as *the page rendered
    fewer controls*.
    """
    n = 40_000
    out = _emit("var a = []; for (var i = 0; i < %d; i++) a.push('item-' + i);\n"
                "__emit({want: a.length, got: a});" % n)
    assert out["want"] == n
    assert len(out["got"]) == n, f"{len(out['got'])} of {n} list entries arrived"
    assert out["got"][-1] == f"item-{n - 1}"


def test_a_multibyte_character_is_not_split_across_the_write():
    """The write is byte-oriented over a UTF-8 buffer, so this is checked.

    A partial write landing mid-character would put a replacement character in
    the middle of a long payload — a wrong value in a place no assertion looks,
    which IS the silent shape the truncation itself turned out not to have.
    """
    out = _emit("var s = ''; while (s.length < 120000) s += '\\u00b5g RAE \\u2014 ';\n"
                "__emit({s: s, n: s.length});")
    assert len(out["s"]) == out["n"]
    assert "�" not in out["s"], (
        "a replacement character came back, so a multi-byte character was split "
        "across two writes and the payload is corrupt in the middle")
    assert out["s"].count("µ") == out["s"].count("—")


def test_the_sentinel_still_ends_the_run_and_small_values_still_arrive():
    """The negative control. A writer that never returns is not a fix either."""
    assert PH.run("__emit(null);") is None
    assert PH.run("__emit({a: 1});") == {"a": 1}
    assert PH.run("__emit([1, 'two', null]);") == [1, "two", None]
    # An emit-less run reports `null` rather than hanging or raising.
    assert PH.run("void 0;") is None


def test_the_emitted_structure_is_what_the_page_serialized():
    """Round-tripped, so the claim is about content rather than only length."""
    payload = {"rows": [{"i": i, "label": f"col_{i}", "keep": i % 2 == 0}
                        for i in range(6000)]}
    out = _emit("__emit(%s);" % json.dumps(payload))
    assert out == payload, (
        "a structure round-tripped through the harness came back different")
