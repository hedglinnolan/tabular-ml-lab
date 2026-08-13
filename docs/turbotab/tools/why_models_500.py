#!/usr/bin/env python3
"""Why does `GET /project/{id}/models` return 500 on a particular CSV?

`DRIVE-035`. Three human drives have hit this and the client cannot see past it:
the response is 21 characters of `Internal Server Error`, which means an
UNHANDLED exception reached Starlette rather than a raised `HTTPException`. The
traceback exists in the server's stderr and nowhere a browser can reach.

The adjudicator could not reproduce it from the file's SHAPE — a 21,849-row
replica matching its dtype composition returns 200 — so the trigger is in the
values and only the real file will find it.

This drives the same sequence a human drives, in-process, with the exception
allowed to propagate, and prints the traceback.

    venv/bin/python docs/turbotab/tools/why_models_500.py ~/path/to/file.csv
    venv/bin/python docs/turbotab/tools/why_models_500.py file.csv --target kcal

With no `--target` it tries every column the shelf could plausibly hang off,
stopping at the first 500 — because run 3 reports the failure is dataset-wide
rather than target-specific, and one reproduction is enough.

Nothing is written to disk and nothing is committed. The project lives in memory
for the life of this process, exactly as it does in the app.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="the CSV that reproduces the 500")
    ap.add_argument("--target", help="a specific target column to try")
    ap.add_argument("--max-targets", type=int, default=6,
                    help="how many columns to try when --target is not given")
    args = ap.parse_args()

    path = Path(args.csv).expanduser()
    if not path.exists():
        print(f"no such file: {path}")
        return 2

    import pandas as pd
    from fastapi.testclient import TestClient
    from turbotab.api import app

    df = pd.read_csv(path)
    print(f"{path.name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("dtypes:", dict(df.dtypes.astype(str).value_counts()))

    if args.target:
        candidates = [args.target]
    else:
        # numeric with enough distinct values first (regression), then anything
        # two-level (classification) — the two shapes run 3 saw fail.
        numeric = [c for c in df.columns
                   if df[c].dtype.kind in "if" and df[c].nunique() > 20]
        binary = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
        candidates = (numeric + binary)[:args.max_targets]
    print(f"targets to try: {candidates}\n")

    # raise_server_exceptions=True is the whole point: let it propagate here
    # rather than being flattened into 21 characters.
    client = TestClient(app, raise_server_exceptions=True)

    for target in candidates:
        print(f"── target = {target!r}")
        with open(path, "rb") as fh:
            r = client.post("/project", files={(
                "file"): (path.name, fh, "text/csv")})
        if r.status_code != 200:
            print(f"   upload -> {r.status_code} {r.text[:160]}")
            continue
        pid = r.json()["id"]

        for kind, payload in (("set_target", {"column": target}),
                              ("set_purpose", {"answer": "prediction"}),
                              ("set_grain", {"answer": "one_row_per_person"}),
                              ("set_eligibility", {"answer": "everyone"}),
                              ("seal", {})):
            d = client.post(f"/project/{pid}/decision",
                            json={"kind": kind, "payload": payload})
            if d.status_code != 200:
                print(f"   {kind} -> {d.status_code} {d.text[:200]}")
                break
        else:
            try:
                m = client.get(f"/project/{pid}/models")
                print(f"   GET /models -> {m.status_code}"
                      + ("" if m.status_code == 200 else f"  {m.text[:200]}"))
                if m.status_code == 200:
                    print(f"   healthy: n_available="
                          f"{m.json().get('n_available')}")
            except Exception:                                    # noqa: BLE001
                print("\n" + "=" * 72)
                print(f"REPRODUCED on target {target!r} — the traceback the "
                      f"browser could not show:")
                print("=" * 72)
                traceback.print_exc()
                print("=" * 72)
                print("\nPaste this into the loop. It is DRIVE-035's cause.")
                return 1
        print()

    print("No 500 reproduced on the targets tried. Pass --target explicitly, "
          "or raise --max-targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
