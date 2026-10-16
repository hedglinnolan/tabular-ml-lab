"""The shipped app keeps the claims' frozen status file (R0_PLAN.md, "Carried from R0a").

The app ships without its tests, so a claim renders from ``turbotab/core/claims/status.json``
alone (``Registry.frozen_status``): an image or a release zip without it renders no claim at all.
Both are read here from their own patterns, as their tools read them:

* the image's build context, ``turbotab/deploy/Dockerfile.dockerignore``, as BuildKit matches a
  .dockerignore (Go's ``filepath.Match`` with ``**``; a line beginning ``!`` is an exception; the
  last line that matches a path, or any directory above it, decides);
* the release zip's ``rsync --exclude`` patterns in ``.github/workflows/release.yml``, as rsync
  matches them (a pattern with no slash matches a name at any depth, and an excluded directory
  takes everything under it).

A pattern planted beside each file's own drops the file, so the reading is shown to see it; a
path the files mean to leave out (a test) is left out by both.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DOCKERIGNORE = REPO / "turbotab" / "deploy" / "Dockerfile.dockerignore"
RELEASE = REPO / ".github" / "workflows" / "release.yml"
STATUS = "turbotab/core/claims/status.json"


def _glob(pattern: str) -> str:
    """A path glob as a regular expression: ``**`` (with its slash) any run of directories, ``*``
    any run within one name, ``?`` one character of a name."""
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return out


def dockerignore_patterns(text: str) -> list[tuple[bool, str]]:
    """``(exception, pattern)`` per line: comments and blank lines skipped, a leading ``/`` and a
    trailing one dropped, as BuildKit cleans them."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        exception = line.startswith("!")
        pattern = line[1:].strip() if exception else line
        out.append((exception, pattern.strip("/")))
    return out


def in_image(path: str, patterns: list[tuple[bool, str]]) -> bool:
    """Whether the build context holds ``path``: the last pattern that matches the path or a
    directory above it decides, an exception keeping it."""
    parts = path.split("/")
    candidates = ["/".join(parts[:k]) for k in range(1, len(parts) + 1)]
    excluded = False
    for exception, pattern in patterns:
        rx = re.compile(_glob(pattern) + r"\Z")
        if any(rx.match(c) for c in candidates):
            excluded = not exception
    return not excluded


def rsync_excludes(text: str) -> list[str]:
    """The ``--exclude`` patterns of the workflow's ``rsync`` command, in order."""
    command = re.search(r"rsync\s+-a\b(.*?)\./\s", text, re.S)
    assert command, "release.yml no longer copies the tree with rsync -a"
    return [a or b for a, b in re.findall(r"--exclude[= ](?:'([^']+)'|(\S+))", command.group(1))]


def in_zip(path: str, excludes: list[str]) -> bool:
    """Whether rsync copies ``path`` (relative to the copied root): no exclude matches it or a
    directory above it. A pattern with no slash matches one name at any depth; one with a slash
    matches the path's tail, or from the root when it starts with one."""
    parts = path.split("/")
    for k in range(1, len(parts) + 1):
        here = "/".join(parts[:k])
        for pattern in excludes:
            bare = pattern.rstrip("/")
            if pattern.endswith("/") and k == len(parts):
                continue  # a directory pattern never matches the file itself
            if "/" not in bare:
                if re.match(_glob(bare) + r"\Z", parts[k - 1]):
                    return False
            elif bare.startswith("/"):
                if re.match(_glob(bare[1:]) + r"\Z", here):
                    return False
            elif re.search(r"(?:^|/)" + _glob(bare) + r"\Z", here):
                return False
    return True


def test_the_image_and_the_release_zip_keep_the_frozen_status_file():
    assert (REPO / STATUS).is_file()
    assert in_image(STATUS, dockerignore_patterns(DOCKERIGNORE.read_text(encoding="utf-8")))
    assert in_zip(STATUS, rsync_excludes(RELEASE.read_text(encoding="utf-8")))


def test_both_readings_leave_out_what_the_files_mean_to_leave_out():
    """The image leaves out ``**/tests`` and ``turbotab/test_*.py``, the zip every ``tests``
    directory and ``*.pyc``: so the readings see the files' own patterns."""
    image = dockerignore_patterns(DOCKERIGNORE.read_text(encoding="utf-8"))
    excludes = rsync_excludes(RELEASE.read_text(encoding="utf-8"))
    for path in ("turbotab/core/tests/test_claims_data.py", "turbotab/test_journeys.py",
                 "turbotab/core/claims/__pycache__/data.cpython-313.pyc"):
        assert not in_image(path, image), path
    for path in ("turbotab/core/tests/test_claims_data.py",
                 "turbotab/core/claims/__pycache__/data.cpython-313.pyc", ".claude/agents/x.md"):
        assert not in_zip(path, excludes), path
    for path in ("turbotab/core/claims/data.py", "turbotab/core/claims/digit_allowlist.json"):
        assert in_image(path, image) and in_zip(path, excludes), path


@pytest.mark.parametrize("planted", ["**/*.json", "turbotab/core/claims", "**/status.json"])
def test_a_planted_image_pattern_that_drops_the_file_is_seen(planted):
    image = dockerignore_patterns(DOCKERIGNORE.read_text(encoding="utf-8") + f"\n{planted}\n")
    assert not in_image(STATUS, image)


@pytest.mark.parametrize("planted", ["*.json", "claims", "status.json", "core/claims/"])
def test_a_planted_zip_exclude_that_drops_the_file_is_seen(planted):
    text = RELEASE.read_text(encoding="utf-8").replace(
        "rsync -a \\", f"rsync -a --exclude '{planted}' \\", 1)
    assert not in_zip(STATUS, rsync_excludes(text))
