"""Print where a pytest run's time went, from its JUnit XML, as GitHub notices anyone can read.

The full tier runs near its job's time limit, and CI logs need repository-admin rights, so every
run, passing or not, names its slowest files and tests in check-run annotations.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

SHOWN = 12


def esc(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def test_id(classname: str, name: str) -> tuple[str, str]:
    """The file and the pytest id of a JUnit test case, from ``a.b.test_c.TestD`` and ``test_e``."""
    parts = classname.split(".")
    cut = next((i for i, part in enumerate(parts) if part.startswith("test_")), len(parts) - 1) + 1
    path = "/".join(parts[:cut]) + ".py"
    return path, "::".join([path, *parts[cut:], name])


def slowest(path: Path) -> tuple[list[tuple[str, float]], list[tuple[str, float]], tuple[int, float]] | None:
    """Files and tests by their seconds, longest first, and (tests, seconds); None with no report."""
    try:
        root = ET.parse(path).getroot()
    except (FileNotFoundError, ET.ParseError):
        return None
    files: Counter[str] = Counter()
    tests = []
    for case in root.iter("testcase"):
        if not case.get("name"):  # an interrupted run leaves empty cases
            continue
        seconds = float(case.get("time") or 0)
        file, test = test_id(case.get("classname", ""), case.get("name", ""))
        files[file] += seconds
        tests.append((test, seconds))
    tests.sort(key=lambda row: -row[1])
    return files.most_common(), tests, (len(tests), sum(seconds for _, seconds in tests))


def main(path: str) -> None:
    found = slowest(Path(path))
    if found is None:
        print(f"::notice title=where the time went::no report at {path}: pytest stopped before writing one")
        return
    files, tests, (count, seconds) = found

    def listed(rows: list[tuple[str, float]]) -> str:
        return esc("; ".join(f"{name} {s / 60:.1f} min" for name, s in rows[:SHOWN]))

    print(f"::notice title=slowest files::{count} tests took {seconds / 60:.0f} minutes, summed over "
          f"the workers. {listed(files)}")
    print(f"::notice title=slowest tests::{listed(tests)}")


if __name__ == "__main__":
    main(sys.argv[1])
