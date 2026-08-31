"""How much memory this process can actually get, asked of the right authority.

The upload admission gate needs a number to size an incoming frame against.
There is no stdlib answer — `os` exposes a CPU count and nothing about RAM — so
this module is the one place that asks, and the one place that knows the answer
can be wrong in a specific, expensive way.

**The specific way, because it is the whole reason this file exists.**
`psutil.virtual_memory()` reads `/proc/meminfo`, and `/proc/meminfo` is not
namespaced. Inside a container it reports the HOST's memory, not the container's
cgroup ceiling. This app's documented enterprise path is exactly that container:
`docker-compose.yml` runs it under `memory: ${APP_MEMORY_LIMIT:-4g}`, and
`UNIVERSITY_DEPLOYMENT.md` tells omics deployers to raise that knob when their
frames get wide. So a probe that trusted psutil alone would read a departmental
server's several hundred GB, admit a file that cannot fit, and the container
would be OOM-killed at 4 GB — with the kernel's SIGKILL arriving as a blank
browser tab and no traceback anywhere the researcher can see.

So the probe reads the cgroup as well — its limit *and* what it has already
spent, because the limit alone counts this process's own several hundred
megabytes of interpreter, pandas and Streamlit as free — and takes the smaller
of the two readings.

**What this module deliberately does not do.** It does not decide anything. It
reports a number, or reports that it has none. Whether a given frame is admitted
is the caller's judgment, and `None` here means *cannot estimate* — never
*unlimited*. A caller that reads `None` as "no ceiling known, so no ceiling"
inverts the entire point of the file.

Stdlib plus an optional psutil, no Streamlit: both front doors can import it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

#: Where the kernel exposes the current process's cgroup, on Linux. A module
#: constant rather than a literal at the read site so a test can point the
#: reader at a temporary directory instead of faking a filesystem. On Windows
#: and macOS this path simply does not exist and every read below misses, which
#: is the correct outcome: there is no cgroup, so there is no second ceiling.
_CGROUP_ROOT = Path("/sys/fs/cgroup")

#: The limit files, in precedence order, relative to the cgroup root. v2 first
#: because a host running the unified hierarchy has no v1 tree at all, while a
#: host running both should be read at the interface the kernel actually
#: enforces through. Written as posix-relative strings and split at the read
#: site so this reads like the paths the kernel documents.
_LIMIT_FILES = (
    "memory.max",                    # cgroup v2 (unified hierarchy)
    "memory/memory.limit_in_bytes",  # cgroup v1
)

#: What the cgroup has already spent, in the same precedence order. Read
#: alongside the limit because the limit on its own answers the wrong question.
#:
#: A container's ceiling is not its headroom. By the time an upload is being
#: sized, this process is already holding the interpreter, pandas, scikit-learn,
#: torch and Streamlit — several hundred megabytes before any data — and inside
#: a cgroup NOTHING else reports that: `psutil.virtual_memory().available` reads
#: the un-namespaced host, so on an idle host it exceeds the container limit and
#: the `min` below returns the container's full size with everything already
#: spent still counted as free. A 4 GB container would admit a frame needing
#: 4 GB and be OOM-killed, which reaches the researcher as a blank browser tab
#: and no traceback — the exact failure this module exists to prevent, arriving
#: through the correction for it.
#:
#: `memory.current` includes reclaimable page cache, so subtracting it can
#: understate the headroom of a container that has read a lot of files. That
#: error runs toward refusing with an actionable message and away from a
#: SIGKILL, which is the direction to be wrong in.
_USAGE_FILES = (
    "memory.current",                # cgroup v2
    "memory/memory.usage_in_bytes",  # cgroup v1
)

#: The value above which a cgroup limit means "unlimited" rather than a number.
#:
#: A THRESHOLD, NOT AN EQUALITY TEST, and the difference is a fail-open bug.
#: cgroup v1 spells "no limit" as a saturated counter, canonically
#: 9223372036854771712 (PAGE_COUNTER_MAX, 0x7FFFFFFFFFFFF000) — but kernels also
#: report 9223372036854775807 (INT64_MAX) and, on some configurations, the
#: unsigned 18446744073709551615. Matching only the canonical literal would
#: silently treat the other two as a real ceiling of eight million terabytes,
#: which is the same as having no gate at all on precisely the machines the
#: equality test does not cover. 8 PiB is several orders of magnitude above any
#: physical machine and several below every sentinel, so the threshold
#: separates them with room on both sides.
_NO_LIMIT_AT_OR_ABOVE = 1 << 53


def _parse_limit(text: str) -> Optional[int]:
    """One cgroup limit file's contents as bytes, or `None` for "no limit".

    Pure — no filesystem — because this is where the bugs live and a parser
    that needs a file to exercise it is a parser that gets tested on one
    machine's kernel. Everything the caller has to survive is decided here:
    the v2 literal `max`, the v1 saturated sentinel, and a file that holds
    something this does not understand.

    `None` covers all three, and it means the same thing everywhere in this
    module: *no usable ceiling was read here*. It never means zero.
    """
    stripped = text.strip()
    if not stripped or stripped == "max":
        return None
    try:
        value = int(stripped)
    except ValueError:
        # Not a number and not `max`. An unfamiliar kernel, a truncated read,
        # or a file that is not what we think it is — report no reading rather
        # than guess at one.
        return None
    if value <= 0:
        # A limit of zero would arithmetically forbid every upload. If a kernel
        # ever really reports it the container is already unusable, so treating
        # it as "cannot estimate" costs nothing and avoids a probe that rejects
        # everything because of a value it did not expect.
        return None
    if value >= _NO_LIMIT_AT_OR_ABOVE:
        return None
    return value


def _cgroup_limit_bytes(root: Optional[Path] = None) -> Optional[int]:
    """The container's memory ceiling in bytes, or `None` if there is not one.

    `root` defaults to `_CGROUP_ROOT` at call time rather than in the signature
    so the default follows the module attribute — a test that repoints the
    constant repoints this too.
    """
    base = _CGROUP_ROOT if root is None else root
    for relative in _LIMIT_FILES:
        candidate = base.joinpath(*relative.split("/"))
        try:
            text = candidate.read_text(encoding="utf-8")
        except (OSError, ValueError, UnicodeDecodeError):
            # OSError rather than FileNotFoundError: not-Linux is the common
            # miss, but a cgroup path can also be present and unreadable
            # (permissions under a restricted runtime, EINVAL or ENODEV on an
            # odd mount). A probe that raised there would take down an upload
            # page over a file it only wanted to consult.
            continue
        limit = _parse_limit(text)
        if limit is not None:
            return limit
    return None


def _cgroup_usage_bytes(root: Optional[Path] = None) -> Optional[int]:
    """What the cgroup has already charged, or `None` if it will not say.

    Same read and the same tolerance as `_cgroup_limit_bytes` — a usage file is
    as absent on Windows and as unreadable under a restricted runtime as a limit
    file, and neither is a reason to raise inside an upload. `_parse_limit`'s
    sentinel rules do not apply to a usage figure, but its other answers do: a
    non-numeric or negative reading is no reading. Zero is legal here and means
    zero, so it is accepted rather than folded into `None` the way a zero
    *limit* is.
    """
    base = _CGROUP_ROOT if root is None else root
    for relative in _USAGE_FILES:
        candidate = base.joinpath(*relative.split("/"))
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        if value >= 0:
            return value
    return None


def _cgroup_headroom_bytes(root: Optional[Path] = None) -> Optional[int]:
    """How much more the cgroup will grant, or `None` if there is no cgroup.

    `limit - usage`, floored at zero, because a cgroup at or over its limit has
    no headroom rather than negative headroom, and a negative number would flow
    into a `min` and turn every subsequent comparison nonsensical.

    Falls back to the bare limit when the usage file is unreadable but the limit
    is not. That is the looser answer and it is chosen deliberately: it is what
    this function returned before it could read usage at all, so a kernel that
    exposes one file and not the other is no worse off than it was.
    """
    limit = _cgroup_limit_bytes(root)
    if limit is None:
        return None
    usage = _cgroup_usage_bytes(root)
    if usage is None:
        return limit
    return max(limit - usage, 0)


def available_memory_bytes() -> Optional[int]:
    """Memory this process can plausibly still use, or `None` if unknown.

    `None` means CANNOT ESTIMATE. It must never be read as "unlimited" — the
    two are opposite instructions to a gate, and the missing-psutil case that
    produces `None` is exactly the lean install where being wrong is cheapest
    to cause and hardest to notice.

    **The figure is still an upper bound, not free space.** Under a cgroup this
    returns `min(host available, container limit - container usage)`, and the
    two operands remain different quantities measured by different authorities:
    psutil's `available` is the HOST's free memory, read through an
    un-namespaced `/proc/meminfo`, while the cgroup pair is this container's own
    accounting. The `min` is right in both directions even so — physical pages
    the host does not have cannot be granted by a generous cgroup, and cgroup
    pages cannot be granted by a host with RAM to spare — but neither operand
    knows what the other is measuring, so a caller should leave headroom rather
    than size a frame right up to it.

    Subtracting usage is what stops the container case from failing OPEN: the
    limit alone counts the several hundred megabytes of interpreter, pandas and
    Streamlit already resident as though they were free. See `_USAGE_FILES`.
    """
    try:
        import psutil
    except ImportError:        # pragma: no cover - exercised on lean installs
        # No probe is not the same as no limit. The caller is told nothing,
        # which is the only truthful thing available.
        return None

    try:
        available = int(psutil.virtual_memory().available)
    except (OSError, RuntimeError, ValueError, AttributeError):
        # psutil reads /proc and the Windows API; both can fail on a locked-down
        # host. Same contract as a missing psutil: no reading, not no limit.
        return None

    headroom = _cgroup_headroom_bytes()
    if headroom is None:
        return available
    return min(available, headroom)


def cpu_count() -> Optional[int]:
    """Logical CPUs on this machine, or `None` if the platform will not say.

    **This is the machine's count, NOT the container's CPU quota.** A cgroup
    can cap CPU through `cpu.max` the same way it caps memory through
    `memory.max`, and this does not read it — so under a container this number
    can be far larger than the parallelism the runtime will actually grant.

    That asymmetry with `available_memory_bytes()` is deliberate rather than an
    oversight. Nothing in this change consumes a CPU count: no `n_jobs`, no
    worker pool, no runtime estimate reads it. A quota-aware number would be a
    threshold nobody had measured against, baked into the abstraction every
    future caller would reach for first. When something does need to size
    parallelism under a container, `cpu.max` gets read here with the
    measurement that justifies it — until then this is a plain inventory count
    and is documented as one.
    """
    count = os.cpu_count()
    return count if count and count > 0 else None
