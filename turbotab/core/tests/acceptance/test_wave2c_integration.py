"""Wave 2c's seam: what holds only once EXPORT runs on REPAIR-VALID (MODELING_SEQUENCE §1 row 12 (b),
§4; V2 definition of done §1, "export ends every journey").

Each package's own acceptance file tests it against its references (``test_export.py``,
``test_ms6_prediction_validation.py``, ``test_previews_*.py``). This file tests where two meet:

* **A bundle handed over is a score seen.** With no rows held out, the record of scores shown
  (``scores_seen.json``) is what refuses the winner's own score: a family whose cross-validated
  score was shown cannot be dropped. The export's performance table prints every fitted family's
  cross-validated score, so a bundle downloaded by a client that never asked for the fit stage
  shows the comparison as much as the fit does, and the record keeps it.
* **The checklist route shows no score, so it records none.** Its TRIPOD+AI 12e and 23a quote the
  performance table; before any score was shown they quote it without the declared result (the
  selection-corrected score, naming the family chosen), so reading the checklist leaves the
  choice of families open.

The references are independent of the code under test: the families are read here, with pandas,
from the keys of the bundle's own ``results/performance.csv``; the record of scores shown is read
from its file; the declared result the checklist must not show is the caption of the bundle's own
``results/performance.md``, read as text; and the refusal's message is the one REPAIR-VALID wrote,
quoted.
"""
from __future__ import annotations

import json

from turbotab.core.tests.acceptance.server_drive import local_server, press_fit
from turbotab.core.tests.acceptance.test_export import (PERFORMANCE_TITLE, UNSEEN_RESULT, _csv,
                                                         _decimals, _diet, _export, _files,
                                                         _open_diet, _performance_quotes,
                                                         _seen_on_disk, _wait_fresh)
from turbotab.core.tests.acceptance.test_ms6_prediction_validation import _refused


def test_an_exported_bundle_counts_as_the_scores_it_prints_seen(tmp_path):
    """Two families compared with no rows held out, the fit computed and never served. The
    checklist is read (no score in it, and nothing recorded), then the bundle is exported: the
    record of scores shown then names exactly the families whose cross-validated rows the bundle's
    performance table prints, and dropping one is refused with the exit that keeps them."""
    csv = tmp_path / "diet.csv"
    _diet().to_csv(csv, index=False)
    with local_server(tmp_path / "home") as client:
        drive = _open_diet(client, csv, "prediction")
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "none"})
        drive.answer("models", {"kind": "select_models", "models": ["linear", "elastic_net"]})
        _wait_fresh(drive, ("cohort", "design", "fit"))
        folder = client.app.state.service.workspace.project_dir(drive.pid)
        assert _seen_on_disk(folder) == {}  # the fit computed, never served

        checklist = drive.c.get(f"/api/projects/{drive.pid}/checklist")
        assert checklist.status_code == 200, checklist.text[:600]
        early = checklist.json()
        assert _performance_quotes(early) == {"12e": [f"{PERFORMANCE_TITLE}. {UNSEEN_RESULT}"],
                                              "23a": [f"{PERFORMANCE_TITLE}. {UNSEEN_RESULT}"]}
        assert _seen_on_disk(folder) == {}  # the checklist shows no score, and records none

        # Under Predict nothing is served before Fit (SIZING P0.8): the bundle is refused until
        # it is pressed, and pressing it marks no score seen.
        assert _export(drive).json()["error"]["code"] == "estimates_withheld"
        assert press_fit(client, drive.pid)
        assert _seen_on_disk(folder) == {}
        r = _export(drive)
        assert r.status_code == 200, r.text[:900]
        files = _files(r.content)
        table = _csv(files["results/performance.csv"])
        # each family's cross-validated rows (``<family>/cv/<metric>``); the no-predictor
        # reference's row (``baseline/cv/…``) is the floor every family is read against, not a
        # family one could keep or drop
        printed = sorted({key.split("/")[0] for key in table["key"]
                          if key.split("/")[1:2] == ["cv"] and not key.startswith("baseline/")})
        assert printed == ["elastic_net", "linear"]
        assert sorted(_seen_on_disk(folder)["dm"]) == printed
        assert set(_seen_on_disk(folder)) == {"dm"}

        # What the checklist kept out: the bundle's declared result (its performance table's
        # caption, the selection-corrected score naming the family chosen), and every number in it.
        md = files["results/performance.md"].decode("utf-8").splitlines()
        assert md[0] == f"### {PERFORMANCE_TITLE}"
        caption = md[2]
        assert " was chosen among 2 families " in caption and _decimals(caption)
        text = json.dumps(early, ensure_ascii=False)
        assert caption not in text and not _decimals(caption) & _decimals(text)
        # the bundle's own checklist quotes it, and so does the live one now that it was shown
        bundled = json.loads(files["checklist/tripod_ai.json"])
        assert _performance_quotes(bundled)["12e"] == [f"{PERFORMANCE_TITLE}. {caption}"]
        assert drive.c.get(f"/api/projects/{drive.pid}/checklist").json() == bundled

        error = _refused(client, drive.pid, {"kind": "select_models", "models": ["elastic_net"]},
                         "compared_families_stay")
        assert error["message"] == (
            "These families' cross-validated scores were shown for `dm` with no rows held out: "
            "`linear`. Dropping them would make the remaining family's own score the result, which "
            "flatters the choice (Tsamardinos et al. 2018). Keep them: the result is then the "
            "selection-corrected estimate, and the best family is still the one deployed.")
        assert sorted(error["exits"][0]["decision"]["models"]) == printed
