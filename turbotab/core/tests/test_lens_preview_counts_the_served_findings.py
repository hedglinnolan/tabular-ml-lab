"""The lens preview's count of what a pack's checks raise is the count of pack findings the
findings page serves under that lens (INBOX:1032).

The preview (``fact_previews.lens_views``) said "The clinical pack's checks raise `8` findings on
this table" where the findings page listed 7: it counted the pack's readings
(``detectors.pack_findings``), not the findings as the findings stage serves them. The stage speaks
for some of them itself (the clinical pack's numbers-stored-as-text finding, where the app's own
text-number finding reads every column it names), reads the left-censoring finding again on the
participants with the outcome set aside, and adds the pack's energy findings for an energy column
the pack's own name matcher misses. Each truth here is the findings the stage serves (its
artifact, annotated as the server serves it), on a sample fixture or a table built here.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core import detectors, fact_previews  # noqa: F401 - registers the lens preview
from turbotab.core.consequences import PreviewContext, plan
from turbotab.core.decisions import ProjectState, SetLens
from turbotab.core.repairs import annotate
from turbotab.core.stages.findings import findings_stage
from turbotab.core.tests.stage_harness import LENSES, SAMPLES, Ingested
from turbotab.core.tests.test_findings_read_the_participants import (
    N_FEATURES,
    QC_AT,
    panel,
    qc_only_blanks_table,
)

COUNT = re.compile(r"checks raise `([\d,]+)` findings? on this table")


def ingested(tmp_path: Path, frame: pd.DataFrame, name: str = "t") -> Ingested:
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return Ingested(path, tmp_path / f"{name}-ingested")


def said(table: Ingested, lens: list[str], state: ProjectState, project: str) -> int:
    """The count the lens preview says for ``lens``, with ``state`` as recorded."""
    ctx = PreviewContext(project_id=project, state=state, datastore=table.store(),
                         artifact=lambda stage: None, training_row_ids=None, cohort_row_ids=None)
    note = plan(SetLens(lenses=lens), ctx, basis="").note
    found = COUNT.search(note or "")
    assert found, note
    return int(found.group(1).replace(",", ""))


def served(table: Ingested, lens: list[str], state: ProjectState) -> list[str]:
    """The pack findings the findings page serves under ``lens``: the stage's artifact, annotated
    as the server serves it (``repairs.annotate``), its findings whose source is a pack."""
    with_lens = state.model_copy(update={"lens": lens})
    artifact = annotate(table.run(findings_stage, with_lens), with_lens, [])
    return [f["id"] for f in artifact["findings"] if f.get("source") == "pack"]


def test_the_clinical_preview_counts_the_seven_pack_findings_clinical_labs_serves(tmp_path):
    """On clinical_labs the clinical pack's readings are 8, one of them its numbers-stored-as-text
    finding, whose columns the app's own text-number finding reads (its read-numbers repair), so
    the page serves 7 pack findings and the app's one: the preview says 7, where it said 8."""
    table = Ingested(SAMPLES / "clinical_labs.csv", tmp_path / "labs")
    state = ProjectState()
    readings = detectors.pack_findings(table.frame().reset_index(drop=True), ["clinical"])
    assert len(readings) == 8
    (text,) = [f for f in readings if f["id"] == "pack::clinical::text_numeric"]
    artifact = table.run(findings_stage, ProjectState(lens=["clinical"]))
    read = {c for f in artifact["findings"] if f["id"].startswith("text_numbers")
            for c in f["affected_columns"]}
    assert set(text["affected_columns"]) <= read
    page = served(table, ["clinical"], state)
    assert len(page) == 7 and "pack::clinical::text_numeric" not in page
    assert said(table, ["clinical"], state, "labs") == len(page)


def test_left_censoring_raised_only_by_the_qc_rows_is_not_counted(tmp_path):
    """A Case/Control/QC table whose QC rows alone are blank (``qc_only_blanks_table``): the page
    serves no left-censoring finding, the participants having no blank, and the preview's count is
    the page's, where it counted the pack's reading over every row."""
    table = ingested(tmp_path, qc_only_blanks_table())
    page = served(table, ["metabolomics"], ProjectState())
    assert "pack::metabolomics::pooled_qc" in page
    assert "pack::metabolomics::left_censored" not in page
    assert said(table, ["metabolomics"], ProjectState(), "qc-only") == len(page)


def outcome_among_the_blanks() -> pd.DataFrame:
    """60 participants and 10 pooled QCs (every seventh row), 40 features whose log means run from
    3.5 to 12, the 4 least abundant blank below exp(4) and the rest complete, and `stage`, an
    outcome coded 0, 1 and 2, blank on the QC rows and on 8 participants, by construction: with
    no outcome named, `stage` is the fifth column with a blank, the least the left-censoring
    reading reads; named, there are four."""
    rng = np.random.default_rng(13)
    n = 70
    qc = np.zeros(n, dtype=bool)
    qc[QC_AT] = True
    log_mean = np.linspace(3.5, 12.0, N_FEATURES)
    log_value = log_mean + rng.normal(0.0, 1.0, (n, N_FEATURES)) * np.where(qc[:, None], 0.05, 0.8)
    values = np.round(np.exp(log_value), 2)
    values[(log_value < 4.0) & (np.arange(N_FEATURES)[None, :] < 4)] = np.nan
    values[:, 4:] = np.round(np.exp(np.maximum(log_value[:, 4:], 4.5)), 2)
    frame: dict[str, object] = {"sample_type": np.where(qc, "pooled_qc", "participant")}
    for j in range(N_FEATURES):
        frame[f"mz_{j + 1:03d}"] = values[:, j]
    stage = rng.integers(0, 3, n).astype(float)
    stage[qc] = np.nan
    stage[np.flatnonzero(~qc)[:8]] = np.nan
    frame["stage"] = stage
    return pd.DataFrame(frame)


@pytest.mark.parametrize("target", [None, "stage"])
def test_the_count_reads_the_outcome_as_the_findings_do(tmp_path, target):
    """With `stage` the outcome, the left-censoring reading sets it aside and four columns with a
    blank are too few to read, so the page serves no left-censoring finding; with no outcome named
    it is read with the features and the finding is served. Either way the preview's count is the
    page's, where with the outcome named it counted the pack's reading of `stage` as a feature."""
    frame = outcome_among_the_blanks()
    part = frame[frame["sample_type"] != "pooled_qc"]
    blank = [c for c in part.columns if part[c].isna().any()]
    assert blank == ["mz_001", "mz_002", "mz_003", "mz_004", "stage"]
    table = ingested(tmp_path, frame)
    state = ProjectState(target=target)
    page = served(table, ["metabolomics"], state)
    assert ("pack::metabolomics::left_censored" in page) == (target is None)
    assert said(table, ["metabolomics"], state, f"outcome-{target}") == len(page)


def test_the_energy_findings_the_app_adds_are_counted(tmp_path):
    """nhanes_dietary with its energy column named `DR2TKCAL` (NHANES's second-day total), a name
    the pack's matcher does not read as energy: the pack raises 2 findings, and the app runs the
    pack's energy detector on the column it reads as energy, so the page serves 3 pack findings.
    The preview says 3, where it said 2."""
    frame = pd.read_csv(SAMPLES / "nhanes_dietary.csv").rename(columns={"DR1TKCAL": "DR2TKCAL"})
    table = ingested(tmp_path, frame)
    readings = detectors.pack_findings(table.frame().reset_index(drop=True), ["dietary"])
    assert "pack::dietary::energy_adjustment" not in {f["id"] for f in readings}
    page = served(table, ["dietary"], ProjectState())
    assert "pack::dietary::energy_adjustment" in page and len(page) == len(readings) + 1
    assert said(table, ["dietary"], ProjectState(), "day-two") == len(page)


def labs(*, tntc: bool, n: int = 300, seed: int = 66) -> pd.DataFrame:
    """A labs table by construction (test_leash's ``_labs``): `crp_mg_l` written "<0.3" on about
    15% of rows, a value below a detection limit, and, with ``tntc``, `wbc` holding 6 TNTC and 4
    QNS cells beside its numbers."""
    rng = np.random.default_rng(seed)
    crp = rng.lognormal(np.log(1.5), 0.8, n).round(2).astype(object)
    crp[rng.random(n) < 0.15] = "<0.3"
    wbc = rng.normal(7, 2, n).round(1).astype(object)
    wbc[:6], wbc[6:10] = "TNTC", "QNS"
    frame = pd.DataFrame({"pid": [f"L{i:04d}" for i in range(n)], "age": rng.integers(20, 80, n),
                          "crp_mg_l": crp, "wbc": wbc, "y": rng.normal(0, 1, n).round(3)})
    return frame if tntc else frame.drop(columns=["wbc"])


@pytest.mark.parametrize("tntc", [False, True])
def test_a_censored_values_finding_left_to_the_below_detection_repair_is_not_counted(
        tmp_path, tntc):
    """`crp_mg_l`'s values below a detection limit have the app's own finding, with its repair,
    so the clinical pack's censored-values finding leaves them to it: with nothing else to say
    the page does not serve it, and with `wbc`'s TNTC and QNS it does. The preview's count is
    the page's either way, where with `crp_mg_l` alone it counted the pack's reading."""
    table = ingested(tmp_path, labs(tntc=tntc))
    page = served(table, ["clinical"], ProjectState())
    assert ("pack::clinical::censored_values" in page) == tntc
    assert said(table, ["clinical"], ProjectState(), f"labs-{tntc}") == len(page)


def test_under_prediction_the_energy_finding_the_page_leaves_out_is_not_counted(tmp_path):
    """Under prediction the page leaves out the dietary pack's energy finding (no energy model is
    asked there, ``materiality.finding_fires``), so the preview counts one fewer than under
    inference, as the page lists."""
    table = Ingested(SAMPLES / "dietary_recalls.csv", tmp_path / "recalls")
    counts = {}
    for purpose in ("inference", "prediction"):
        state = ProjectState(purpose=purpose)
        page = served(table, ["dietary"], state)
        assert ("pack::dietary::energy_adjustment" in page) == (purpose == "inference")
        counts[purpose] = said(table, ["dietary"], state, f"recalls-{purpose}")
        assert counts[purpose] == len(page)
    assert counts["prediction"] == counts["inference"] - 1


@pytest.mark.slow
def test_every_fixture_under_every_lens_it_fits_counts_what_the_page_serves(tmp_path):
    """Every sample fixture under each lens the profile stage hints for it, and under all five
    lenses: the preview's count is the page's, before an outcome is chosen."""
    for i, path in enumerate(sorted(SAMPLES.glob("*.csv"))):
        table = Ingested(path, tmp_path / str(i))
        for lens in [[h] for h in table.lenses_that_fit()] + [list(LENSES)]:
            page = served(table, lens, ProjectState())
            assert said(table, lens, ProjectState(), f"{path.stem}-{'-'.join(lens)}") == len(page), (
                path.stem, lens)


def test_the_panel_counts_what_the_page_serves_before_and_after_the_outcome(tmp_path):
    """The plasma-like panel of the participants' tests (``panel``), under the metabolomics lens,
    with no outcome and with `disease` the outcome: the preview's count is the page's."""
    table = ingested(tmp_path, panel())
    for target in (None, "disease"):
        state = ProjectState(target=target)
        assert said(table, ["metabolomics"], state, f"panel-{target}") == len(
            served(table, ["metabolomics"], state))


def test_a_confirmation_the_findings_read_is_counted_in_the_same_project(tmp_path):
    """clinical_labs under the clinical lens, previewed, then `troponin` confirmed as amounts
    through the ledger (``shape_confirmations``, a slot the findings stage reads): the page then
    serves one pack finding fewer (its mixed-result-type finding reads the column as the numbers
    the fit reads), and the preview in the same project says the page's count, where it repeated
    the count it had cached before the confirmation."""
    table = Ingested(SAMPLES / "clinical_labs.csv", tmp_path / "labs")
    before, after = ProjectState(), ProjectState(
        shape_confirmations={"code_or_count:troponin": "amount"})
    page_before, page_after = served(table, ["clinical"], before), served(table, ["clinical"], after)
    assert "pack::clinical::mixed_result_type" in set(page_before) - set(page_after)
    assert len(page_after) == len(page_before) - 1
    assert said(table, ["clinical"], before, "labs-confirmed") == len(page_before)
    assert said(table, ["clinical"], after, "labs-confirmed") == len(page_after)


def test_an_answer_the_findings_do_not_read_reuses_the_count(tmp_path, monkeypatch):
    """The guard's other side: the count is cached on what the findings stage reads (its
    ``Stage`` reads, and the purpose the page's filter reads), so an answer it reads nowhere (a
    row exclusion's label here, ``exclusions``) computes nothing again, while a purpose does."""
    from turbotab.core.decisions import ExclusionRule

    table = Ingested(SAMPLES / "clinical_labs.csv", tmp_path / "labs")
    calls: list[int] = []
    real = fact_previews.served_pack_findings

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(fact_previews, "served_pack_findings", counted)
    said(table, ["clinical"], ProjectState(), "labs-reuse")
    rule = ExclusionRule(column="age", low=0.0, high=120.0, reason="a range")
    said(table, ["clinical"], ProjectState(exclusions=[rule]), "labs-reuse")
    assert len(calls) == 1
    said(table, ["clinical"], ProjectState(purpose="prediction"), "labs-reuse")
    assert len(calls) == 2
