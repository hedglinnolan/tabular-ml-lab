"""Over HTTP, on a cross-sectional table shaped like the glucose project (SWARM_PLAN §5 item 3):
diagnosed diabetes excluded by its code, its refused and don't-know codes named, and the latest
cycle held out by a cut-point on the cycle column (SWARM_PLAN §8). The counts each assertion
expects are computed from the table this file writes, with pandas, never read from the engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from turbotab.core.tests.acceptance.server_drive import press_fit
from turbotab.server.tests.conftest import (answer_settled, declare, open_by_path, prepare,
                                            wait_for)

CYCLES = (2011, 2013, 2015, 2017)


def table(path, blanks: bool = False) -> pd.DataFrame:
    """``blanks``: `bmi` and `DIQ010` blank on a few rows each, after the outcome is drawn."""
    rng = np.random.default_rng(2017)
    n = 480
    frame = pd.DataFrame({
        "SEQN": np.arange(1, n + 1),
        "cycle": np.repeat(CYCLES, n // len(CYCLES)),
        "age": rng.integers(20, 80, n),
        "bmi": np.round(rng.normal(28, 5, n), 1),
        # NHANES DIQ010: 1 yes, 2 no, 3 borderline, 7 refused, 9 don't know
        "DIQ010": rng.choice([1, 2, 3, 7, 9], n, p=[0.08, 0.8, 0.06, 0.03, 0.03]),
    })
    frame["LBXGLU"] = np.round(70 + 0.3 * frame["age"] + 0.8 * frame["bmi"]
                               + rng.normal(0, 8, n), 1)
    if blanks:
        frame.loc[rng.choice(n, 9, replace=False), "bmi"] = np.nan
        frame["DIQ010"] = frame["DIQ010"].astype("Int64")
        frame.loc[rng.choice(n, 11, replace=False), "DIQ010"] = pd.NA
    frame.to_csv(path, index=False)
    return frame


def decide(client, pid, decision):
    prepare(client, pid, decision)
    response = answer_settled(client, pid, None, decision)
    assert response.status_code == 200, response.text
    return response.json()


def test_a_cut_point_seal_by_cycle_on_the_rows_a_level_rule_keeps_fits_and_opens(client, tmp_path):
    frame = table(tmp_path / "glucose_like.csv")
    pid = open_by_path(client, tmp_path / "glucose_like.csv")
    # What a user who knows the table answers when asked: age is whole years, the cycle names
    # its survey cycle, DIQ010 is a code (as table() writes them).
    declare(pid, {"code_or_count:age": "amount", "code_or_count:cycle": "code",
                  "code_or_count:DIQ010": "code", "code_or_count:SEQN": "code",
                  "unit:age": "years"}, fixture="glucose_like.csv")
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["clinical"]})
    decide(client, pid, {"kind": "set_target", "column": "LBXGLU"})
    decide(client, pid, {"kind": "set_purpose", "purpose": "prediction"})
    rule = {"kind": "levels", "column": "DIQ010", "exclude": [1], "refused": [7], "dont_know": [9],
            "reason": "diagnosed diabetes"}
    view = decide(client, pid, {"kind": "set_exclusions", "rules": [rule]})
    # The sentence counts the rows at the excluded level apart from the refused and don't-know
    # answers, as the flow does (the package 4 verifier's V4).
    at_level, unconfirmed = int((frame["DIQ010"] == 1).sum()), int(frame["DIQ010"].isin([7, 9]).sum())
    assert view["decisions"][-1]["sentence"] == (
        f"`{at_level}` rows with `DIQ010` `1` were excluded as diagnosed diabetes, and "
        f"`{unconfirmed}` with `DIQ010` refused (`7`), don't know (`9`) or not recorded were "
        f"excluded as not confirmed eligible.")
    cut = {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5, "validation": "kfold",
           "cut_point": {"column": "cycle", "at": 2017}, "fold_order": "random"}
    view = decide(client, pid, cut)
    said = view["decisions"][-1]["sentence"]
    assert said.startswith("A cut-point at `2017` on `cycle` held out every row at or after it")

    eligible = ~frame["DIQ010"].isin([1, 7, 9])
    wait_for(client, pid, {"cohort": "fresh", "split": "fresh"}, timeout=180)
    cohort = client.get(f"/api/projects/{pid}/stages/cohort").json()["artifact"]
    steps = {s["key"]: s["dropped"] for s in cohort["steps"]}
    assert steps["exclusion:0:refused"] == int((frame["DIQ010"] == 7).sum())
    assert steps["exclusion:0:dont_know"] == int((frame["DIQ010"] == 9).sum())
    assert steps["exclusion:0"] == int((frame["DIQ010"] == 1).sum())
    split = client.get(f"/api/projects/{pid}/stages/split").json()["artifact"]
    held = int((eligible & (frame["cycle"] >= 2017)).sum())
    assert split["n_holdout"] == held and split["n_train"] == int(eligible.sum()) - held
    assert split["chronology"]["cut_point"] == "2017" and split["fold_scheme"] == "random"
    # Restated on the flow for the export, the refused and don't-know rows keep their own reason
    # and their own count, as the live sentence gives them (the package 4 verifier's V8).
    text = client.get(f"/api/projects/{pid}/methods").json()["text"]
    assert (f"`{at_level}` rows with `DIQ010` `1` were excluded as diagnosed diabetes, and "
            f"`{unconfirmed}` with `DIQ010` refused (`7`), don't know (`9`) or not recorded were "
            f"excluded as not confirmed eligible.") in text
    assert "otherwise as not confirmed eligible" not in text
    # The seal counts the rows the exclusion leaves, as the split's own counts do (P1).
    assert split["chronology"]["sentence"].startswith(
        f"Held out by a cut-point at `2017` on `cycle`: `{held}` rows held out and "
        f"`{int(eligible.sum()) - held}` train.")

    decide(client, pid, {"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    assert press_fit(client, pid)
    fit = client.get(f"/api/projects/{pid}/stages/fit").json()["artifact"]
    assert fit["holdout_sealed"] is True and fit["n_holdout"] == held
    assert all(m["holdout"] is None for m in fit["models"])
    # The opening waits for the open noticings: the engine reads DIQ010's 7 and 9 as possible
    # missing-value codes. The rule names them, so the values are kept as they are (a dismissal
    # recomputes nothing, so the fit stays the one the seal opens on).
    found = client.get(f"/api/projects/{pid}/stages/findings").json()["artifact"]["findings"]
    codes = next(f for f in found if f["id"] == "sentinel_missing__DIQ010")
    kept = client.post(f"/api/projects/{pid}/decisions",
                       json={"kind": "dismiss_finding", "finding_id": codes["id"],
                             "reason": "the eligibility rule names them as refused and don't know"})
    assert kept.status_code == 200, kept.text
    opened = client.post(f"/api/projects/{pid}/decisions",
                         json={"kind": "open_seal", "family": "linear"})
    assert opened.status_code == 200, opened.text
    assert opened.json()["state"]["seal_opened"] is True
    fit = client.get(f"/api/projects/{pid}/stages/fit").json()["artifact"]
    assert fit["holdout_sealed"] is False and fit["models"][0]["holdout"] is not None


def test_the_methods_export_counts_a_level_rule_by_reason_and_a_range_in_all(client, tmp_path):
    """The package 4 verifier's V8, both ways: ``GET /methods`` restates the exclusions with the
    participant flow's counts as they stand. A level rule that names refused and don't-know codes
    gives its rows at the excluded level apart from those it could not confirm (a blank, `7` or
    `9`), as its live sentence does (STROBE item 13a counts each stage); a range rule before it,
    whose blanks the flow also counts on a line of their own, still gives one total, its blanks in
    it. Each count is pandas', over the table written, in the flow's order: the range first, then
    the level rule on the rows the range kept."""
    frame = table(tmp_path / "blanks.csv", blanks=True)
    pid = open_by_path(client, tmp_path / "blanks.csv")
    declare(pid, {"code_or_count:age": "amount", "code_or_count:cycle": "code",
                  "code_or_count:DIQ010": "code", "code_or_count:SEQN": "code",
                  "unit:age": "years"}, fixture="blanks.csv")
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["clinical"]})
    decide(client, pid, {"kind": "set_target", "column": "LBXGLU"})
    decide(client, pid, {"kind": "set_purpose", "purpose": "prediction"})
    bmi = {"kind": "range", "column": "bmi", "low": 20, "high": 40, "reason": "implausible bmi"}
    levels = {"kind": "levels", "column": "DIQ010", "exclude": [1], "refused": [7],
              "dont_know": [9], "reason": "diagnosed diabetes"}
    view = decide(client, pid, {"kind": "set_exclusions", "rules": [bmi, levels]})

    keep = frame["LBXGLU"].notna()
    by_bmi = keep & ~frame["bmi"].between(20, 40)  # a blank bmi is not between them
    keep &= ~by_bmi
    code = frame["DIQ010"]
    unconfirmed = keep & (code.isna() | code.isin([7, 9]).fillna(False))
    keep &= ~unconfirmed
    at_level = keep & code.eq(1).fillna(False)
    a, b, c = int(by_bmi.sum()), int(at_level.sum()), int(unconfirmed.sum())
    assert int(frame["bmi"].isna().sum()) == 9 and int(code.isna().sum()) == 11
    assert min(a, b, c) > 0
    said = (f"`{a}` rows with `bmi` outside `20`–`40` or not recorded were excluded as "
            f"implausible bmi; `{b}` rows with `DIQ010` `1` were excluded as diagnosed diabetes, "
            f"and `{c}` with `DIQ010` refused (`7`), don't know (`9`) or not recorded were "
            f"excluded as not confirmed eligible, `{a + b + c}` in all.")
    assert view["decisions"][-1]["sentence"] == said
    wait_for(client, pid, {"cohort": "fresh"}, timeout=180)
    steps = {s["key"]: s["dropped"]
             for s in client.get(f"/api/projects/{pid}/stages/cohort").json()["artifact"]["steps"]}
    assert steps["exclusion:0:not_recorded"] == int(frame["bmi"].isna().sum())  # its own line
    text = client.get(f"/api/projects/{pid}/methods").json()["text"]
    assert said in text
