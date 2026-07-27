"""Re-measure Classic and write a NEW baseline beside the frozen one.

The only invocation of `turbotab.measure.write_baseline` in the repository. It
is a script and not a test on purpose: measurement and comparison are different
acts, and while they shared a code path every run of the suite silently
re-measured the reference it was supposed to be judged against.

`docs/turbotab/data/routing-baseline.json` is raw data from a pre-registered
experiment. It is never edited. This script therefore **refuses to overwrite**:
it writes to a new dated path and prints what changed, so the two measurements
sit side by side and a human adjudicates which one the pre-registration is
banked against. That is the procedure `VALUE_CHECK_ADJUDICATION.md` already
sets — frozen artifact unmodified, both readings preserved in data, ruling
published.

Usage
-----
    turbotab/.venv/Scripts/python scripts/remeasure_routing_baseline.py --out <path>

`--out` is required. There is no default, because a default is how a script
like this one ends up pointed at the file it must not touch.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FROZEN = ROOT / "docs" / "turbotab" / "data" / "routing-baseline.json"
LEAKY_FROZEN = ROOT / "docs" / "turbotab" / "data" / "routing-baseline-leaky.json"
PROTECTED = {FROZEN.resolve(), LEAKY_FROZEN.resolve()}

METRICS = ("required_decisions", "questions_asked", "irrelevant_questions",
           "findings_driven", "coverage")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=pathlib.Path,
                        help="where to write the new measurement. Must not be "
                             "a frozen baseline.")
    parser.add_argument("--measured-at", default=None,
                        help="the commit these numbers were taken at.")
    args = parser.parse_args(argv)

    out = args.out.resolve()
    if out in PROTECTED:
        parser.error(
            f"{args.out} is a frozen pre-registered baseline and is never "
            "overwritten. Write the new measurement beside it and adjudicate; "
            "see docs/turbotab/VALUE_CHECK_ADJUDICATION.md.")
    if out.exists():
        parser.error(f"{args.out} already exists. Pick a path that does not.")

    from turbotab import measure
    from tests.integration.test_routing_baseline import (DATASETS,
                                                         _measure_classic)

    measurements = []
    for name, path, target in DATASETS:
        if not path.exists():
            parser.error(f"missing dataset {path}")
        measurements.append(_measure_classic(name, path, target))

    out.parent.mkdir(parents=True, exist_ok=True)
    measure.write_baseline(out, measurements, measured_at=args.measured_at,
                           prereg="VALUE_CHECK_PREREG.md")

    frozen = {m["dataset"]: m for m in measure.read_baseline(FROZEN)}
    print(f"wrote {out}")
    print()
    print(f"{'dataset':<15}{'metric':<22}{'frozen':>12}{'now':>12}")
    print("-" * 61)
    moved = False
    for m in measurements:
        now = m.to_dict()["metrics"]
        was = frozen.get(m.dataset, {}).get("metrics", {})
        for metric in METRICS:
            if metric in was and was[metric] != now[metric]:
                moved = True
                print(f"{m.dataset:<15}{metric:<22}{str(was[metric]):>12}"
                      f"{str(now[metric]):>12}")
    if not moved:
        print("no metric moved.")
    else:
        print()
        print("A metric moved. The frozen baseline is unchanged and stays that "
              "way. Adjudicate in docs/turbotab/VALUE_CHECK_ADJUDICATION.md, "
              "stating the reading under BOTH denominators.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
