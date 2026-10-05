"""Wave 2c's seam: what holds only once EXPORT runs on REPAIR-VALID (MODELING_SEQUENCE §1 row 12 (b),
§4; V2 definition of done §1, "export ends every journey").

Each package's own acceptance file tests it against its references (``test_export.py``,
``test_ms6_prediction_validation.py``, ``test_previews_*.py``). This file tests where two meet:

* **A bundle handed over is a score seen.** With no rows held out, the record of scores shown
  (``scores_seen.json``) is what refuses the winner's own score: a family whose cross-validated
  score was shown cannot be dropped. The export's performance table prints every fitted family's
  cross-validated score, so a bundle downloaded by a client that never asked for the fit stage
  shows the comparison as much as the fit does, and the record keeps it. The checklist route prints
  no score, so it records none.

The reference is independent of the code under test: the families are read here, with pandas, from
the keys of the bundle's own ``results/performance.csv``, and the refusal's message is the one
REPAIR-VALID wrote, quoted.
"""
from __future__ import annotations

from turbotab.core.tests.acceptance.server_drive import local_server
from turbotab.core.tests.acceptance.test_export import (_csv, _diet, _export, _files, _open_diet,
                                                         _wait_fresh)
from turbotab.core.tests.acceptance.test_ms6_prediction_validation import _refused


def test_an_exported_bundle_counts_as_the_scores_it_prints_seen(tmp_path):
    """Two families compared with no rows held out, the fit computed and never served. The
    checklist is read (no score in it), then the bundle is exported: the record of scores shown
    then names exactly the families whose cross-validated rows the bundle's performance table
    prints, and dropping one is refused with the exit that keeps them."""
    from turbotab.core.models.selection import read_seen

    csv = tmp_path / "diet.csv"
    _diet().to_csv(csv, index=False)
    with local_server(tmp_path / "home") as client:
        drive = _open_diet(client, csv, "prediction")
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "none"})
        drive.answer("models", {"kind": "select_models", "models": ["linear", "elastic_net"]})
        _wait_fresh(drive, ("cohort", "design", "fit"))
        folder = client.app.state.service.workspace.project_dir(drive.pid)
        assert read_seen(folder) == {}  # the fit computed, never served

        checklist = drive.c.get(f"/api/projects/{drive.pid}/checklist")
        assert checklist.status_code == 200, checklist.text[:600]
        assert read_seen(folder) == {}  # the checklist prints no score

        r = _export(drive)
        assert r.status_code == 200, r.text[:900]
        table = _csv(_files(r.content)["results/performance.csv"])
        # each family's cross-validated rows (``<family>/cv/<metric>``); the no-predictor
        # reference's row (``baseline/cv/…``) is the floor every family is read against, not a
        # family one could keep or drop
        printed = sorted({key.split("/")[0] for key in table["key"]
                          if key.split("/")[1:2] == ["cv"] and not key.startswith("baseline/")})
        assert printed == ["elastic_net", "linear"]
        assert sorted(read_seen(folder)["dm"]) == printed
        assert set(read_seen(folder)) == {"dm"}

        error = _refused(client, drive.pid, {"kind": "select_models", "models": ["elastic_net"]},
                         "compared_families_stay")
        assert error["message"] == (
            "These families' cross-validated scores were shown for `dm` with no rows held out: "
            "`linear`. Dropping them would make the remaining family's own score the result, which "
            "flatters the choice (Tsamardinos et al. 2018). Keep them: the result is then the "
            "selection-corrected estimate, and the best family is still the one deployed.")
        assert sorted(error["exits"][0]["decision"]["models"]) == printed
