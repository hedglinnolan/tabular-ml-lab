"""WP1 · Data identity: values mean what the file says, and reshaping keeps rows and order honest.

The acceptance tests of docs/turbotab-next/audit/AUDIT_REPORT.md §5 (WP1), one per numbered item,
on the audit's own reproduction fixtures (``repro.tar.gz``: C/t13_dates, C-skeptic/c3c, c2, c78,
B-skeptic/s11, C/t6_agg, C/t7_transpose, C/t7c, F-skeptic/f4_orient, C/t2_dot, C-skeptic/c6,
C/t14b, C/t11). Each measured value comes from the app's real stages (``GraphRun`` runs the stage
graph as the workers do); each reference is computed another way: pandas with an explicit format,
a plain pandas groupby, ``pd.get_dummies``, numpy over the finite values, or the source table read
by pandas. Never by calling the code under test for its own expected value.

Closes MA-03, MA-04, MA-05, MA-14, MA-15, MA-16, MA-17, MA-18, MA-19; the minor C12, C14, B18,
B21, B22 and D16 are pinned at the end.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import ProjectState, Refusal, SetAggregation, validate
from turbotab.core.stages.working import FEATURES, SAMPLE_COLUMN, StructureError
from turbotab.core.tests.graph_runner import GraphRun

SAMPLES = Path(__file__).resolve().parents[3] / "sample_data"


@pytest.fixture
def graph(tmp_path):
    runs: list[GraphRun] = []

    def make(frame: pd.DataFrame | None = None, name: str = "t", *, path: Path | None = None,
             **csv: object) -> GraphRun:
        if path is None:
            path = tmp_path / f"{name}.csv"
            frame.to_csv(path, index=False, **csv)
        run = GraphRun(path, tmp_path / f"p{len(runs)}")
        runs.append(run)
        return run

    yield make
    for run in runs:
        run.close()


def state(**slots: object) -> ProjectState:
    return ProjectState.model_validate(slots)


def table(bundle) -> pd.DataFrame:
    return pd.read_parquet(bundle.files["table.parquet"])


def finding(out: dict, finding_id: str) -> dict:
    found = [f for f in out["findings"]["findings"] if f["id"] == finding_id]
    assert found, f"no finding {finding_id}: {[f['id'] for f in out['findings']['findings']]}"
    return found[0]


def applied(f: dict, option: str) -> dict:
    """The finding's own option, as the client records it (its params filled from the data)."""
    chosen = [o for o in f["repairs"] if o["key"] == option]
    assert len(chosen) == 1, [o["key"] for o in f["repairs"]]
    d = chosen[0]["decision"]
    return {d["finding_id"]: {"action": "applied", "option": d["option"], "params": d["params"]}}


# ── 1 · Ambiguous dates (MA-05) ──────────────────────────────────────────────


def month_dates() -> pd.DataFrame:
    """C/t13_dates: 2,000 US dates coarsened to the first of the month, 2019–2021."""
    rng = np.random.default_rng(0)
    months = pd.date_range("2019-01-01", "2021-12-01", freq="MS")
    d = rng.choice(months, 2000)
    return pd.DataFrame({"pid": np.arange(2000), "visit_date": pd.to_datetime(d).strftime("%m/%d/%Y"),
                         "y": rng.normal(size=2000)})


def test_1_ambiguous_dates_are_never_read_one_way_without_a_word(graph):
    """Ingest keeps the column as text (no date changes) and the app asks, with three example rows
    under each reading. Reference: pandas reading the text with each explicit format."""
    df = month_dates()
    month_first = pd.to_datetime(df["visit_date"], format="%m/%d/%Y")
    day_first = pd.to_datetime(df["visit_date"], format="%d/%m/%Y")
    would_change = int((month_first != day_first).sum())
    # Every date not in January moves under the other reading: 1,843 on this fixture (the audit
    # reports 1,840–1,843 across its two fixtures; "1,840" in §5 is the other one's count).
    assert would_change == int((month_first.dt.month != 1).sum()) == 1843

    run = graph(df, "month_dates")
    column = next(c for c in run.info["columns"] if c["name"] == "visit_date")
    assert column["physical_type"] == "VARCHAR"  # kept as text, not read day-first
    assert any("visit_date" in w and "month-first and day-first" in w for w in run.info["warnings"])
    stored = pd.read_parquet(run.raw, columns=["visit_date"])["visit_date"]
    changed = int((stored.to_numpy() != df["visit_date"].to_numpy()).sum())
    assert changed == 0  # not one of the 1,843 dates the day-first reading would have moved

    out = run.run(state(lens=["clinical"]), upto=["findings"])
    asked = finding(out, "ambiguous_dates")  # the app asks, as a finding with both readings
    assert asked["affected_columns"] == ["visit_date"]
    options = {o["key"]: o for o in asked["repairs"]}
    assert set(options) == {"month_first", "day_first"}
    assert options["month_first"]["decision"]["params"] == {"formats": {"visit_date": "%m/%d/%Y"}}
    assert options["day_first"]["decision"]["params"] == {"formats": {"visit_date": "%d/%m/%Y"}}
    # Three example rows, each under both readings, as pandas reads them with each format.
    examples = re.findall(r"`([^`]+)` is (\S+) month-first or (\S+) day-first", asked["detail"])
    assert len(examples) == 3
    for text, month, day in examples:
        assert text in set(df["visit_date"])
        assert month == pd.to_datetime(text, format="%m/%d/%Y").date().isoformat()
        assert day == pd.to_datetime(text, format="%d/%m/%Y").date().isoformat()
        assert month != day


def quarterly_visits() -> pd.DataFrame:
    """C-skeptic/c3c: 150 people, three quarterly visits each, dated on the first of the month."""
    rng = np.random.default_rng(4)
    rows = []
    for p in range(150):
        start = int(rng.integers(1, 4))
        for k in range(3):
            m = start + 3 * k
            rows.append((f"P{p}", f"{m:02d}/01/2021", f"2021-{m:02d}-01", 80 - k + rng.normal(),
                         140 - 2 * k + rng.normal()))
    return pd.DataFrame(rows, columns=["pid", "visit_us", "visit_iso", "weight", "sbp"])


def test_1_after_month_first_the_repeats_reading_matches_the_iso_copy(graph):
    """Answered "month first", quarterly visits read about 91 days apart and the reading equals
    the ISO copy's. Reference: the median gap between a person's visits, from the ISO text by pandas."""
    df = quarterly_visits()
    iso = df[["pid", "visit_iso", "weight", "sbp"]].rename(columns={"visit_iso": "visit_date"})
    us = df[["pid", "visit_us", "weight", "sbp"]].rename(columns={"visit_us": "visit_date"})
    gaps = (pd.to_datetime(iso["visit_date"], format="%Y-%m-%d").groupby(iso["pid"])
            .apply(lambda s: s.sort_values().diff().dropna().dt.days))
    reference = float(np.median(gaps))
    assert 89 <= reference <= 93

    base = dict(lens=["clinical"], target="sbp", grain={"grain": "repeated", "id_column": "pid"})
    iso_reading = graph(iso, "iso").run(state(**base), upto=["structure"])["structure"]["repeats"]
    assert iso_reading["spacing"]["median_days"] == pytest.approx(reference, abs=1)

    run = graph(us, "us")
    unanswered = run.run(state(**base), upto=["structure", "findings"])
    assert unanswered["structure"]["unread_dates"] == ["visit_date"]  # never read one way silently
    assert (unanswered["structure"]["repeats"]["spacing"] or {}).get("column") != "visit_date"
    answer = applied(finding(unanswered, "ambiguous_dates"), "month_first")
    out = run.run(state(**base, findings=answer), upto=["structure", "working"])
    answered = out["structure"]["repeats"]
    assert answered["spacing"]["median_days"] == pytest.approx(reference, abs=1)
    assert answered["reading"] == iso_reading["reading"]
    assert answered["spacing"] == iso_reading["spacing"]
    # And the table every later stage reads holds the dates the ISO copy writes.
    dated = table(out["working"])["visit_date"]
    want = pd.to_datetime(iso["visit_date"], format="%Y-%m-%d")
    assert (pd.to_datetime(dated).to_numpy() == want.to_numpy()).all()


# ── 2 · Ordering when combining (MA-03) ─────────────────────────────────────


def labeled_visits() -> pd.DataFrame:
    """B-skeptic/s11: visits stored month_12, baseline, month_6, with text labels."""
    rows = []
    for i in range(30):
        for label, val in [("month_12", 3.0), ("baseline", 1.0), ("month_6", 2.0)]:
            rows.append({"pid": f"P{i:03d}", "visit": label, "age": 40 + i, "sex_code": i % 2 + 1,
                         "ldl": val + i * 0.01, "y": float(i % 5)})
    return pd.DataFrame(rows)


# The readings ledger (BLUEPRINT §14.1): whole numbers with a few values that change within a
# person may be codes or counts, and combining them waits for the user's answer for each; these
# fixtures' smoking categories and weekday are codes, said one reading at a time.
CODES = {f"code_or_count:{c}": "code" for c in ("smoking", "smoking_cat", "day_of_week")}
COMBINE = dict(lens=["clinical"], grain={"grain": "repeated", "id_column": "pid"}, unit="unit",
               shape_confirmations=CODES)


def test_2_text_visit_labels_are_refused_until_their_order_is_declared(graph):
    """Refused with "declare the order of the levels"; once declared, the LDL change is +2.0.
    Reference: pandas, the labels mapped to their order, last minus first per person."""
    df = labeled_visits()
    order = {"baseline": 0, "month_6": 1, "month_12": 2}
    ref = (df.assign(o=df["visit"].map(order)).sort_values(["pid", "o"]).groupby("pid")["ldl"]
           .agg(lambda s: s.iloc[-1] - s.iloc[0]))
    assert np.allclose(ref.to_numpy(), 2.0)

    run = graph(df, "labels")
    repeat = {"repeat_kind": "time_points", "time_column": "visit"}
    st = state(**COMBINE, target="y", repeat_kind=repeat)
    structure = run.run(st, upto=["structure"])["structure"]
    assert structure["time_order"]["orderable"] is False
    ctx = {"state": st, "columns": list(df.columns), "target": "y",
           "artifact": lambda s: structure if s == "structure" else None}
    with pytest.raises(Refusal) as refused:
        validate(SetAggregation(method="change"), ctx)
    assert refused.value.code == "cannot_order" and "Declare the order of the levels" in refused.value.message
    declared = next(e["decision"] for e in refused.value.exits
                    if e["decision"] and e["decision"]["kind"] == "set_repeat_kind")
    assert declared["levels"] == ["baseline", "month_6", "month_12"]
    with pytest.raises(StructureError, match="Declare the order of the levels"):  # the stage, too
        run.run(state(**COMBINE, target="y", repeat_kind=repeat, aggregation={"method": "change"}),
                upto=["working"])

    repeat = {k: declared[k] for k in ("repeat_kind", "time_column", "levels")}
    out = run.run(state(**COMBINE, target="y", repeat_kind=repeat, aggregation={"method": "change"}),
                  upto=["working"])
    got = table(out["working"]).set_index("pid")["ldl"]
    assert np.allclose(got.loc[ref.index].to_numpy(), ref.to_numpy())  # +2.0, not −1.0
    receipt = out["working"].data["aggregation"]
    assert receipt["ordered_by"] == "visit" and receipt["order"] == "declared levels"


def text_dated_visits() -> pd.DataFrame:
    """C/t6_agg: visit dates written "Mar 3, 2021", not in chronological order in the file."""
    rows = [("A", "Mar 3, 2021", 3, 1, 80.0, 140.0), ("A", "Jan 5, 2021", 3, 1, 90.0, 150.0),
            ("A", "Feb 2, 2021", 2, 1, 85.0, 145.0), ("B", "Jan 9, 2021", 1, 2, 70.0, 120.0),
            ("B", "", 1, 2, 66.0, 118.0), ("B", "Feb 7, 2021", 1, 2, 68.0, 119.0),
            ("C", "Jan 1, 2021", 3, 2, 100.0, 160.0), ("C", "Feb 1, 2021", 1, 2, 95.0, 155.0),
            ("C", "Mar 1, 2021", 1, 2, 92.0, 150.0)]
    return pd.DataFrame(rows, columns=["pid", "visit_date", "smoking", "sex", "weight_kg", "sbp"])


def test_2_text_dates_order_the_records_by_their_date(graph):
    """Participant A's weight change is −10 (not +5). Reference: pandas parsing "%b %d, %Y"."""
    df = text_dated_visits()
    when = pd.to_datetime(df["visit_date"], format="%b %d, %Y", errors="coerce")
    dated = df.assign(t=when).dropna(subset=["t"]).sort_values(["pid", "t"])
    ref = dated.groupby("pid")["weight_kg"].agg(lambda s: s.iloc[-1] - s.iloc[0])
    assert ref["A"] == -10.0

    repeat = {"repeat_kind": "time_points", "time_column": "visit_date"}
    out = graph(df, "text_dates").run(state(**COMBINE, target="sbp", repeat_kind=repeat,
                                            aggregation={"method": "change", "outcome": "last"}),
                                      upto=["working"])
    got = table(out["working"]).set_index("pid")["weight_kg"]
    assert got["A"] == -10.0
    assert np.allclose(got.loc[ref.index].to_numpy(), ref.to_numpy())
    receipt = out["working"].data["aggregation"]
    assert receipt["ordered_by"] == "visit_date" and receipt["undated_records"] == 1


def undated_middle_visits() -> pd.DataFrame:
    """C-skeptic/c78: 40 people, 3 ISO-dated visits; P00–P03's middle visit has no date."""
    rng = np.random.default_rng(3)
    rows = []
    for p in range(40):
        smk = [1, 1, 3] if p % 3 == 0 else ([2, 2, 2] if p % 3 == 1 else [3, 1, 1])
        dow = rng.integers(1, 8, 3)
        for k in range(3):
            date = f"2021-0{k + 1}-1{p % 9}"
            if p < 4 and k == 1:
                date = ""
            rows.append((f"P{p:02d}", date, smk[k], int(dow[k]), 80 - 3 * k + p * 0.1, 140 - 4 * k))
    return pd.DataFrame(rows, columns=["pid", "visit_date", "smoking_cat", "day_of_week",
                                       "weight_kg", "sbp"])


def test_2_an_undated_record_is_never_the_first_or_the_last(graph):
    """"last" returns the latest dated visit (74.0, outcome 132), change is −6.0, and the receipt
    counts undated records. Reference: pandas over the dated records only."""
    df = undated_middle_visits()
    dated = df[df["visit_date"] != ""].assign(t=lambda d: pd.to_datetime(d["visit_date"], format="%Y-%m-%d"))
    dated = dated.sort_values(["pid", "t"])
    last = dated.groupby("pid").last()
    change = dated.groupby("pid")["weight_kg"].agg(lambda s: s.iloc[-1] - s.iloc[0])
    assert last.loc["P00", "weight_kg"] == pytest.approx(74.0) and last.loc["P00", "sbp"] == 132
    assert change["P00"] == pytest.approx(-6.0)

    run = graph(df, "undated")
    repeat = {"repeat_kind": "time_points", "time_column": "visit_date"}
    for method, column, want in (("last", "weight_kg", last["weight_kg"]), ("change", "weight_kg", change)):
        out = run.run(state(**COMBINE, target="sbp", repeat_kind=repeat,
                            aggregation={"method": method, "outcome": "last"}), upto=["working"])
        got = table(out["working"]).set_index("pid")
        assert np.allclose(got.loc[want.index, column].to_numpy(), want.to_numpy())
        assert np.array_equal(got.loc[last.index, "sbp"].to_numpy(), last["sbp"].to_numpy())
        receipt = out["working"].data["aggregation"]
        assert receipt["undated_records"] == int((df["visit_date"] == "").sum()) == 4
    assert got.loc["P00", "weight_kg"] == pytest.approx(-6.0)


def test_2_ordered_by_never_names_a_column_that_ordered_nothing(graph):
    """Under the mean (no order needed) a column of text labels orders nothing, and the receipt
    says file order; declared, it orders, and the receipt names it."""
    df = labeled_visits()
    run = graph(df, "labels_mean")
    repeat = {"repeat_kind": "time_points", "time_column": "visit"}
    out = run.run(state(**COMBINE, target="y", repeat_kind=repeat, aggregation={"method": "mean"}),
                  upto=["working"])
    assert out["working"].data["aggregation"]["ordered_by"] == "file order"
    declared = {**repeat, "levels": ["baseline", "month_6", "month_12"]}
    out = run.run(state(**COMBINE, target="y", repeat_kind=declared, aggregation={"method": "first"}),
                  upto=["working"])
    assert out["working"].data["aggregation"]["ordered_by"] == "visit"
    firsts = table(out["working"]).set_index("pid")["ldl"]
    want = df[df["visit"] == "baseline"].set_index("pid")["ldl"]
    assert np.allclose(firsts.loc[want.index].to_numpy(), want.to_numpy())


# ── 3 · Turning a feature table (MA-04) ─────────────────────────────────────


def metabolomics_export() -> tuple[pd.DataFrame, pd.DataFrame]:
    """C/t7_transpose: metabolite, hmdb, mz, rt, S1…S12 (60 features)."""
    rng = np.random.default_rng(5)
    nf, ns = 60, 12
    feat = pd.DataFrame({"metabolite": [f"M{i}" for i in range(nf)],
                         "hmdb": [f"HMDB{i:07d}" for i in range(nf)],
                         "mz": np.round(rng.uniform(80, 900, nf), 4),
                         "rt": np.round(rng.uniform(0.5, 15, nf), 2)})
    abund = np.exp(rng.normal(np.linspace(5, 15, nf)[:, None], 0.3, (nf, ns)))
    for j in range(ns):
        feat[f"S{j + 1}"] = np.round(abund[:, j], 2)
    return feat, feat.set_index("metabolite")[[f"S{j + 1}" for j in range(ns)]].T


def test_3_a_metabolomics_export_turns_into_its_samples_only(graph):
    """Exactly 12 sample rows, with m/z, RT and HMDB kept as feature annotations. Reference: the
    sample block transposed by pandas."""
    feat, want = metabolomics_export()
    run = graph(feat, "met")
    out = run.run(state(lens=["metabolomics"], orientation="feature_major"), upto=["oriented"])
    oriented = out["oriented"]
    turned = table(oriented)
    assert len(turned) == 12  # not 14: m/z and RT are not participants
    assert list(turned[SAMPLE_COLUMN]) == list(want.index)
    features = [c for c in turned.columns if c not in (SAMPLE_COLUMN, "__row_id")]
    assert features == list(want.columns)
    assert all(turned[c].dtype == np.float64 for c in features)  # hmdb did not make them text
    assert np.array_equal(turned[features].to_numpy(), want.to_numpy())
    kept = pd.read_parquet(oriented.files[FEATURES])
    assert list(kept.columns) == ["feature", "hmdb", "mz", "rt"]
    assert kept["feature"].tolist() == feat["metabolite"].tolist()
    assert np.array_equal(kept["mz"].to_numpy(), feat["mz"].to_numpy())
    assert kept["hmdb"].tolist() == feat["hmdb"].tolist()
    assert oriented.data["turn"]["annotations"] == ["hmdb", "mz", "rt"]


def test_3_numeric_entrez_ids_stay_the_feature_names(graph):
    """A numeric Entrez-ID matrix keeps the gene IDs as feature names (C/t7c). Reference: pandas,
    the counts indexed by gene ID and transposed."""
    rng = np.random.default_rng(5)
    g = pd.DataFrame({"entrez_id": rng.choice(np.arange(1, 100000), 80, replace=False)})
    for j in range(10):
        g[f"GSM{1000 + j}"] = rng.poisson(np.exp(rng.normal(5, 2, 80)))
    want = g.set_index("entrez_id").T
    out = graph(g, "entrez").run(state(lens=["genomics"], orientation="feature_major"),
                                 upto=["oriented"])
    turned = table(out["oriented"])
    assert len(turned) == 10 and list(turned[SAMPLE_COLUMN]) == list(want.index)
    assert [c for c in turned.columns if c not in (SAMPLE_COLUMN, "__row_id")] == [str(i) for i in want.columns]
    assert np.array_equal(turned[[str(i) for i in want.columns]].to_numpy(), want.to_numpy())
    assert out["oriented"].data["turn"]["label_column"] == "entrez_id"


def test_3_mzmine_feature_columns_are_never_samples(graph):
    """MZmine's ``row ID``, ``row m/z`` and ``row retention time`` (F-skeptic/f4_orient)."""
    df = pd.read_csv(SAMPLES / "metabolomics_untargeted.csv")
    block = df.set_index("sample_id").select_dtypes("number")
    block = block[[c for c in block.columns if c not in ("age", "bmi", "run_order")]].T
    rng = np.random.default_rng(0)
    mzmine = pd.concat([pd.DataFrame({"row ID": np.arange(1, len(block) + 1),
                                      "row m/z": rng.uniform(80, 1200, len(block)),
                                      "row retention time": rng.uniform(0.5, 20, len(block))}),
                        block.reset_index(drop=True)], axis=1)
    out = graph(mzmine, "mzmine").run(state(lens=["metabolomics"], orientation="feature_major"),
                                      upto=["oriented"])
    turned = table(out["oriented"])
    assert list(turned[SAMPLE_COLUMN]) == list(block.columns)
    assert not {"row ID", "row m/z", "row retention time"} & set(turned[SAMPLE_COLUMN])
    assert out["oriented"].data["turn"]["label_column"] == "row ID"
    assert out["oriented"].data["turn"]["annotations"] == ["row m/z", "row retention time"]


def test_3_the_preview_shows_what_the_stage_writes(graph):
    """The orientation preview's turned corner equals the stage's turned table, cell for cell."""
    from turbotab.core import structure_previews as SP
    from turbotab.core.consequences import PreviewContext
    from turbotab.core.datastore import DataStore

    feat, _ = metabolomics_export()
    run = graph(feat, "met_preview")
    plain = run.run(state(lens=["metabolomics"]), upto=["oriented"])["oriented"]
    with DataStore(run.raw, 1 << 30) as store:
        ctx = PreviewContext(project_id="x", state=state(lens=["metabolomics"]), datastore=store,
                             artifact=lambda s: plain if s == "oriented" else None,
                             training_row_ids=None, cohort_row_ids=None)
        view = SP.orientation_views(SimpleNamespace(orientation="feature_major"), ctx)[0]
    preview = view.story[1]
    stage = table(run.run(state(lens=["metabolomics"], orientation="feature_major"),
                          upto=["oriented"])["oriented"])
    assert preview.columns[0] == SAMPLE_COLUMN and len(preview.rows) == SP.TURN_SAMPLES
    for i, row in enumerate(preview.rows):
        for column in preview.columns:
            assert row.values[column] == stage.iloc[i][column], (i, column)
    assert "mz" not in [r.values[SAMPLE_COLUMN] for r in preview.rows]


def test_3_a_column_that_does_not_move_with_the_samples_is_refused(graph):
    """An unnamed annotation column (a mass under a name the partition does not know) is refused
    with an exit to keep it beside the features, never turned silently into a sample."""
    feat, _ = metabolomics_export()
    feat = feat.drop(columns=["hmdb"]).rename(columns={"mz": "obs_mass"})
    run = graph(feat, "odd")
    turn = run.run(state(lens=["metabolomics"]), upto=["oriented"])["oriented"].data["turn"]
    assert turn["code"] == "not_a_sample" and "`obs_mass`" in turn["refusal"]
    keep = turn["exits"][0]["decision"]
    assert keep["kind"] == "set_feature_table" and "obs_mass" in keep["annotations"]
    out = run.run(state(lens=["metabolomics"], orientation="feature_major",
                        feature_table={"label": keep["label"], "annotations": keep["annotations"]}),
                  upto=["oriented"])
    assert len(table(out["oriented"])) == 12


# ── 4 · Per-column combining (MA-14) ────────────────────────────────────────


def test_4_each_column_is_combined_by_what_it_holds(graph):
    """Under "change" a sex code constant within persons keeps its value (not 0); under "mean" an
    integer smoking code takes a value it has (never 1.67); the receipt lists every numeric column
    that varied within persons with its rule. Reference: plain pandas groupbys."""
    df = undated_middle_visits().assign(sex_code=lambda d: (d["pid"].str[1:].astype(int) % 2) + 1)
    repeat = {"repeat_kind": "time_points", "time_column": "visit_date"}
    run = graph(df, "rules")

    out = run.run(state(**COMBINE, target="sbp", repeat_kind=repeat,
                        aggregation={"method": "change", "outcome": "last"}), upto=["working"])
    got = table(out["working"]).set_index("pid")
    sex = df.groupby("pid")["sex_code"].first()
    assert (got.loc[sex.index, "sex_code"] == sex).all() and (got["sex_code"] != 0).all()

    out = run.run(state(**COMBINE, target="sbp", repeat_kind=repeat,
                        aggregation={"method": "mean", "outcome": "last"}), upto=["working"])
    got = table(out["working"]).set_index("pid")
    dated = df.assign(t=pd.to_datetime(df["visit_date"], format="%Y-%m-%d", errors="coerce"))
    dated = dated.assign(o=dated["t"].isna()).sort_values(["pid", "o", "t"], kind="stable")

    def mode_first_seen(s: pd.Series) -> int:
        counts = s.value_counts()
        top = counts[counts == counts.max()].index
        return int(next(v for v in s if v in top))  # ties: the earliest record

    want = dated.groupby("pid")["smoking_cat"].agg(mode_first_seen)
    assert np.array_equal(got.loc[want.index, "smoking_cat"].to_numpy(), want.to_numpy())
    assert set(np.unique(got["smoking_cat"])) <= set(df["smoking_cat"])  # a level, never 1.67
    assert np.allclose(got["weight_kg"].loc[want.index], df.groupby("pid")["weight_kg"].mean().loc[want.index])

    receipt = out["working"].data["aggregation"]
    numeric = [c for c in ("smoking_cat", "day_of_week", "weight_kg", "sex_code")]
    varied = [c for c in numeric if (df.groupby("pid")[c].nunique() > 1).any()]
    listed = {e["column"]: e["rule"] for e in receipt["columns"]}
    assert sorted(listed) == sorted(varied) == ["day_of_week", "smoking_cat", "weight_kg"]
    assert listed == {"smoking_cat": "mode", "day_of_week": "mode", "weight_kg": "mean"}
    assert receipt["constant_columns"] == 1  # sex_code


def test_4_a_column_may_be_given_its_own_rule(graph):
    """The aggregation answer's ``columns`` sets one column's rule: weight kept at baseline."""
    df = undated_middle_visits()
    repeat = {"repeat_kind": "time_points", "time_column": "visit_date"}
    out = graph(df, "override").run(state(
        **COMBINE, target="sbp", repeat_kind=repeat,
        aggregation={"method": "change", "outcome": "last", "columns": {"weight_kg": "first"}}),
        upto=["working"])
    got = table(out["working"]).set_index("pid")["weight_kg"]
    first = df[df["visit_date"] != ""].groupby("pid")["weight_kg"].first()
    assert np.allclose(got.loc[first.index].to_numpy(), first.to_numpy())
    entry = next(e for e in out["working"].data["aggregation"]["columns"] if e["column"] == "weight_kg")
    assert entry["rule"] == "first" and entry["chosen"] is True


# ── 5 · Categorical declaration (MA-15) ─────────────────────────────────────


def test_5_a_declared_categorical_enters_as_indicators():
    """RIDRETH3 declared categorical enters as five indicator columns (B/exp15_codes). Reference:
    ``pd.get_dummies`` with the first level dropped."""
    from turbotab.core.decisions import MissingSpec
    from turbotab.core.models.pipeline import design_spec, shared_steps, transformer

    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({"fat_g": rng.uniform(30, 120, n), "RIDRETH3": rng.choice([1, 2, 3, 4, 6, 7], n),
                       "DMDEDUC2": rng.choice([1, 2, 3, 4, 5], n)})
    roles = {"fat_g": "exposure", "RIDRETH3": "covariate", "DMDEDUC2": "covariate"}
    plain = ProjectState(roles=roles, missing=MissingSpec(strategy="complete_case"))
    spec = design_spec(plain, df, list(roles))
    assert "RIDRETH3" in spec.numeric  # undeclared: one straight line, as before

    declared = plain.model_copy(update={"categorical": ["RIDRETH3"]})
    spec = design_spec(declared, df, list(roles))
    matrix = transformer(shared_steps(spec)).fit_transform(df)
    made = [c for c in matrix.columns if str(c).startswith("RIDRETH3")]
    want = pd.get_dummies(df["RIDRETH3"], prefix="RIDRETH3", drop_first=True, dtype=float)
    assert len(made) == 5 == want.shape[1]
    assert np.array_equal(np.asarray(matrix[made], dtype=float), want.to_numpy())


def test_5_the_declaration_is_proposed_for_codes(graph):
    """Proposed for integer columns with few levels and for known NHANES coded variables, never
    for a continuous column. Reference: the levels counted by pandas."""
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({"SEQN": np.arange(n), "RIDRETH3": rng.choice([1, 2, 3, 4, 6, 7], n),
                       "DMDEDUC2": rng.choice([1, 2, 3, 4, 5], n), "region_code": rng.choice([1, 2, 3, 4], n),
                       "DR1TKCAL": rng.normal(2000, 400, n).round(1), "glucose": rng.normal(100, 10, n)})
    roles = graph(df, "nhanes_codes").run(state(lens=["dietary"], target="glucose"), upto=["roles"])["roles"]
    proposed = {p["column"]: p for p in roles["categorical"]}
    assert set(proposed) == {"RIDRETH3", "DMDEDUC2", "region_code"}
    for c, p in proposed.items():
        assert p["levels"] == df[c].nunique()
    assert proposed["RIDRETH3"]["confidence"] == proposed["DMDEDUC2"]["confidence"] == "high"
    assert proposed["region_code"]["confidence"] == "medium"


# ── 6 · Spellings (MA-16) ───────────────────────────────────────────────────


def sas_dots() -> pd.DataFrame:
    """C/t2_dot: 3% of age, kcal and BMI written as SAS's "." for missing; kcal also has two other
    spellings that are not numbers."""
    rng = np.random.default_rng(1)
    n = 3000
    age = rng.integers(20, 80, n).astype(object)
    kcal = np.round(rng.normal(2100, 500, n)).astype(object)
    bmi = np.round(rng.normal(27, 5, n), 1).astype(object)
    for a in (age, kcal, bmi):
        a[rng.random(n) < 0.03] = "."
    kcal[[5, 50]] = "#DIV/0!"
    kcal[77] = "n.d."
    return pd.DataFrame({"SEQN": np.arange(n), "age": age, "kcal": kcal, "bmi": bmi,
                         "glucose": np.round(rng.normal(100, 15, n), 1)})


def test_6_a_sas_dot_column_reads_as_numbers_after_the_repair(graph):
    """Source check, SAS 9.4 Language Reference: Concepts, "Missing Values": "By default, SAS
    replaces a missing numeric value with a period" (quoted in the audit, MA-16).

    The repair counts and names every value it could not parse. Reference: pandas ``to_numeric``
    with ``errors="coerce"`` on the source text, and pandas' own count of each spelling."""
    df = sas_dots()
    run = graph(df, "sas_dot")
    assert {c["name"]: c["physical_type"] for c in run.info["columns"]}["kcal"] == "VARCHAR"
    out = run.run(state(lens=["dietary"], target="glucose"), upto=["findings"])
    ids = [f["id"] for f in out["findings"]["findings"]]
    assert not any(i.startswith("numeric_as_text__kcal") for i in ids)  # one card per column
    kcal = finding(out, "text_numbers__kcal")
    bad = df["kcal"][pd.to_numeric(df["kcal"], errors="coerce").isna()].value_counts()
    assert dict(bad) == {".": int((df["kcal"] == ".").sum()), "#DIV/0!": 2, "n.d.": 1}
    for spelling, n in bad.items():  # every spelling, with its count
        assert f"`{spelling}` (`{n:,}`)" in kcal["detail"]
    option = next(o for o in kcal["repairs"] if o["key"] == "read_numbers")
    for spelling, n in bad.items():
        assert f"`{spelling}` (`{n:,}`)" in option["sentence"]

    answers = {}
    for column in ("age", "kcal", "bmi"):
        answers.update(applied(finding(out, f"text_numbers__{column}"), "read_numbers"))
    working = run.run(state(lens=["dietary"], target="glucose", findings=answers), upto=["working"])
    got = table(working["working"])
    for column in ("age", "kcal", "bmi"):
        want = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        assert got[column].dtype == np.float64
        assert np.array_equal(got[column].to_numpy(dtype=float), want, equal_nan=True), column


def test_6_a_decimal_comma_file_gives_the_dot_decimal_numbers(tmp_path):
    """A ";"-separated decimal-comma file reads as the same numbers as its dot-decimal copy.
    Reference: the dot-decimal copy read by pandas."""
    from turbotab.core.datastore import DataStore, ingest

    rng = np.random.default_rng(2)
    df = pd.DataFrame({"id": np.arange(200), "kcal": np.round(rng.normal(2000, 400, 200), 1),
                       "bmi": np.round(rng.normal(26, 4, 200), 2), "n_visits": rng.integers(1, 5, 200)})
    dot, comma = tmp_path / "dot.csv", tmp_path / "comma.csv"
    df.to_csv(dot, index=False)
    df.to_csv(comma, index=False, sep=";", decimal=",")
    assert ";" in comma.read_text().splitlines()[1] and "," in comma.read_text().splitlines()[1]
    reference = pd.read_csv(dot)
    for path in (dot, comma):
        info = ingest(path, tmp_path / f"{path.stem}.parquet")
        with DataStore(tmp_path / f"{path.stem}.parquet", 1 << 30) as store:
            got = store.materialize()
        for column in reference.columns:
            assert np.array_equal(got[column].to_numpy(dtype=float),
                                  reference[column].to_numpy(dtype=float)), (path.name, column)
        types = {c.name: c.dtype for c in info.columns}
        assert types["kcal"] == types["bmi"] == "numeric"


def test_6_below_a_detection_limit_is_its_own_family(graph):
    """"<0.2" goes to a separate "<LOD" family, never folded into "not a number". Reference:
    pandas counting the "<0.2" cells; the half-limit option checked against pandas' replacement."""
    rng = np.random.default_rng(3)
    n = 300
    crp = np.round(rng.lognormal(0, 0.8, n), 2).astype(object)
    crp[rng.random(n) < 0.1] = "<0.2"
    crp[[3, 30]] = "."
    df = pd.DataFrame({"pid": np.arange(n), "crp": crp, "y": rng.normal(size=n)})
    run = graph(df, "lod")
    out = run.run(state(lens=["clinical"], target="y"), upto=["findings"])
    from turbotab.core.repairs import family_for

    lod = finding(out, "below_detection__crp")
    numbers = finding(out, "text_numbers__crp")
    assert family_for(lod["id"]).key == "below_detection" != family_for(numbers["id"]).key
    n_lod = int((df["crp"] == "<0.2").sum())
    assert f"`{n_lod:,}`" in lod["title"]
    assert "`<0.2`" not in numbers["detail"].split("The values that are not numbers:")[-1].split(".")[0]

    answer = applied(lod, "half_limit")
    got = table(run.run(state(lens=["clinical"], target="y", findings=answer), upto=["working"])["working"])
    want = pd.to_numeric(df["crp"].replace("<0.2", "0.1"), errors="coerce").to_numpy(dtype=float)
    assert np.allclose(got["crp"].to_numpy(dtype=float), want, equal_nan=True)


# ── 7 · Parity (MA-17) ──────────────────────────────────────────────────────


def test_7_csv_and_excel_agree_on_what_is_missing(tmp_path):
    """The same table as CSV and xlsx gives identical missing counts per column, over the whole
    NULL_TOKENS list; "None" stays an answer in both. Reference: the token list itself."""
    from turbotab.core.datastore import NULL_TOKENS, ingest

    answers = ["None", "-nan", "1.#IND", "#NA", "none"]  # not missing in either reader
    columns = {f"t{i}": [token, "x", "y"] for i, token in enumerate(NULL_TOKENS)}
    columns.update({f"a{i}": [token, "x", "y"] for i, token in enumerate(answers)})
    df = pd.DataFrame(columns)
    csv_path, xlsx_path = tmp_path / "t.csv", tmp_path / "t.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    got = {}
    for path in (csv_path, xlsx_path):
        info = ingest(path, tmp_path / f"{path.suffix[1:]}.parquet")
        got[path.suffix] = {c.name: c.n_missing for c in info.columns}
    assert got[".csv"] == got[".xlsx"]
    for i, _ in enumerate(NULL_TOKENS):
        assert got[".csv"][f"t{i}"] == 1
    for i, _ in enumerate(answers):
        assert got[".xlsx"][f"a{i}"] == 0


# ── 8 · Infinite values (MA-18) ─────────────────────────────────────────────


def ratio_table() -> pd.DataFrame:
    """C/t14b: protein per 1,000 kcal, with one day recorded as 0 kcal (an infinite ratio)."""
    rng = np.random.default_rng(0)
    n = 500
    kcal = rng.normal(2000, 400, n).round()
    kcal[7] = 0
    df = pd.DataFrame({"SEQN": np.arange(n), "kcal": kcal, "protein_g": rng.normal(80, 20, n).round(1),
                       "ldl": rng.normal(120, 30, n).round(1)})
    df["protein_per_1000kcal"] = df["protein_g"] / df["kcal"] * 1000
    return df


def test_8_one_infinity_is_counted_and_offered_a_repair(graph):
    """The profile completes; the summary is over finite values; a finding counts the infinity
    and offers set-missing; the Arrow and DuckDB paths agree. Reference: numpy over the finite
    values."""
    from turbotab.core.datastore import DataStore

    df = ratio_table()
    assert np.isinf(df["protein_per_1000kcal"]).sum() == 1
    run = graph(df, "ratio")
    out = run.run(state(lens=["dietary"], target="ldl"), upto=["profile", "findings"])
    assert "profile" in out
    x = df["protein_per_1000kcal"].to_numpy()
    finite = x[np.isfinite(x)]
    want = {"mean": finite.mean(), "std": finite.std(ddof=1), "min": finite.min(), "max": finite.max(),
            "q25": np.quantile(finite, 0.25), "median": np.quantile(finite, 0.5),
            "q75": np.quantile(finite, 0.75)}
    with DataStore(run.raw, 1 << 30) as store:
        duck = store._summaries_duckdb(["protein_per_1000kcal"])["protein_per_1000kcal"]
        arrow = store._summaries_arrow(["protein_per_1000kcal"])["protein_per_1000kcal"]
    for summary in (duck, arrow):
        assert summary["n_infinite"] == 1 and summary["n"] == len(df)
        for key, value in want.items():
            assert summary[key] == pytest.approx(value, rel=1e-12), key
    # The two engines agree: counts exactly, statistics to the last digits a summation order moves
    # (DuckDB's avg and numpy's pairwise sum differ by about 1e-16 relative).
    assert set(duck) == set(arrow)
    for key in duck:
        if isinstance(duck[key], float):
            assert duck[key] == pytest.approx(arrow[key], rel=1e-12), key
        else:
            assert duck[key] == arrow[key], key

    inf = finding(out, "infinite_values")
    assert inf["affected_columns"] == ["protein_per_1000kcal"]
    assert inf["summary"].startswith("`1` cell in `protein_per_1000kcal` is infinite")
    answer = applied(inf, "set_missing")
    got = table(run.run(state(lens=["dietary"], target="ldl", findings=answer), upto=["working"])["working"])
    want_col = np.where(np.isinf(x), np.nan, x)
    assert np.array_equal(got["protein_per_1000kcal"].to_numpy(dtype=float), want_col, equal_nan=True)


# ── 9 · Flow counts (MA-19, D16) ────────────────────────────────────────────


def test_9_a_range_step_counts_only_rows_in_range(tmp_path):
    """Source check, STROBE item 13(a): "Report the numbers of individuals at each stage of the
    study—e.g., numbers potentially eligible, examined for eligibility, confirmed eligible…"
    (quoted in the audit, MA-19).

    With 20% of age missing, "`age` within 20–80" counts only rows with age in range, and a
    separate line reads "`age` not recorded: n" (C/t11). Reference: pandas counts."""
    from turbotab.core.decisions import ExclusionRule
    from turbotab.core.stages.rows import cohort_flow

    rng = np.random.default_rng(0)
    n = 1000
    age = rng.integers(10, 90, n).astype(float)
    age[rng.random(n) < 0.2] = np.nan
    kcal = rng.normal(2000, 600, n)
    kcal[rng.random(n) < 0.1] = np.nan
    kcal[:30] = 9000
    f = pd.DataFrame({"y": rng.normal(size=n), "age": age, "kcal": kcal}, index=np.arange(n))
    rules = [ExclusionRule(column="age", low=20, high=80, reason="adults 20-80"),
             ExclusionRule(column="kcal", low=500, high=5000, reason="implausible energy")]
    steps, kept = cohort_flow(f, target="y", rules=rules, missing=None, predictor_columns=[])
    by_key = {s["key"]: s for s in steps}
    assert by_key["exclusion:0:not_recorded"]["label"] == "`age` not recorded"
    assert by_key["exclusion:0:not_recorded"]["dropped"] == int(f["age"].isna().sum())
    in_range = f["age"].between(20, 80)
    assert by_key["exclusion:0"]["n"] == int(in_range.sum())
    assert by_key["exclusion:0"]["label"] == "`age` within `20`–`80`"
    both = in_range & f["kcal"].between(500, 5000)
    assert by_key["exclusion:1"]["n"] == len(kept) == int(both.sum())
    k = f.loc[kept]
    assert k["age"].notna().all() and k["kcal"].notna().all() and k["age"].between(20, 80).all()

    keep = [r.model_copy(update={"missing": "keep"}) for r in rules]
    steps, kept = cohort_flow(f, target="y", rules=keep, missing=None, predictor_columns=[])
    by_key = {s["key"]: s for s in steps}
    assert by_key["exclusion:0"]["label"] == "`age` within `20`–`80`, or not recorded"
    assert by_key["exclusion:0:not_recorded"]["dropped"] == 0
    assert f"`{int(f['age'].isna().sum()):,}` kept" in by_key["exclusion:0:not_recorded"]["label"]
    assert len(kept) == int(((f["age"].isna() | in_range)
                             & (f["kcal"].isna() | f["kcal"].between(500, 5000))).sum())


def test_9_the_by_sex_screen_reports_rows_it_could_not_screen():
    """D16: rows with missing or unrecognized sex, including 9,000 kcal days, are not silently kept."""
    from turbotab.core.decisions import ExclusionRule
    from turbotab.core.stages.rows import cohort_flow

    rng = np.random.default_rng(1)
    n = 600
    sex = rng.choice(["F", "M", "U", None], n, p=[0.45, 0.45, 0.05, 0.05])
    kcal = rng.normal(2000, 500, n)
    kcal[:20] = 9000
    f = pd.DataFrame({"y": rng.normal(size=n), "sex": sex, "kcal": kcal}, index=np.arange(n))
    rule = ExclusionRule.model_validate({"column": "kcal", "reason": "Willett, by sex",
                                         "by": {"column": "sex", "ranges": {"F": [500, 3500], "M": [800, 4200]}}})
    steps, kept = cohort_flow(f, target="y", rules=[rule], missing=None, predictor_columns=[])
    by_key = {s["key"]: s for s in steps}
    unscreened = int((~f["sex"].isin(["F", "M"])).sum())
    assert by_key["exclusion:0:not_screened"]["dropped"] == unscreened > 0
    assert (f.loc[kept, "kcal"] < 9000).all()
    ok = (((f["sex"] == "F") & f["kcal"].between(500, 3500)) | ((f["sex"] == "M") & f["kcal"].between(800, 4200)))
    assert len(kept) == int(ok.sum())


# ── the minor findings WP1 also closes ──────────────────────────────────────


def test_c12_group_keys_keep_text_identifiers_apart():
    """"007", "07" and "7" are three people, not one."""
    from turbotab.core.stages.rows import draw_split

    frame, info = draw_split(np.arange(6), holdout=0.5, seed=0, folds=2,
                             groups=["007", "7", "07", "A", "B", "C"], grouped_by="pid")
    assert info["n_groups"] == 6


def test_c14_integers_past_two_to_the_53_stay_exact(tmp_path):
    from turbotab.core.datastore import DataStore, ingest, json_safe

    path = tmp_path / "big.csv"
    path.write_text("id,pos\n9007199254740993,1\n9007199254740995,2\n12345678901234567890,3\n")
    info = ingest(path, tmp_path / "big.parquet")
    assert info.columns[0].physical_type == "VARCHAR"
    with DataStore(tmp_path / "big.parquet", 1 << 30) as store:
        assert store.materialize(["id"])["id"].tolist() == ["9007199254740993", "9007199254740995",
                                                            "12345678901234567890"]
    assert json_safe(9007199254740993) == "9007199254740993" and json_safe(2 ** 53) == 2 ** 53


def test_b18_codes_are_read_before_units_and_on_text():
    """A 99999 code in a kJ column is blanked before the conversion, and a text column's codes
    are compared as numbers instead of raising."""
    import duckdb

    from turbotab.core.repairs import column_expressions

    dispositions = {
        "pack::dietary::atwater": {"action": "applied", "option": "to_kcal",
                                   "params": {"column": "e", "factor": 4.184}},
        "sentinel_missing__e": {"action": "applied", "option": "set_missing", "params": {"values": {"e": [99999]}}},
        "sentinel_missing__x": {"action": "applied", "option": "set_missing", "params": {"values": {"x": [7, 9]}}},
    }
    exprs = column_expressions(None, dispositions)
    con = duckdb.connect()
    rows = con.execute(f"SELECT {exprs['e']}, {exprs['x']} FROM (VALUES (8368.0, '7'), (99999.0, 'refused'), "
                       f"(4184.0, '12')) v(e, x)").fetchall()
    assert rows == [(2000.0, None), (None, "refused"), (1000.0, "12")]


def test_b21_b22_no_column_is_exempt_and_a_binary_outcome_is_not_averaged(graph):
    """B21: a column read as the replicate index is combined and listed like any other. B22: a
    0/1 outcome's mean is refused; "first" is the first record's value for the outcome too."""
    rows = []
    for i in range(40):
        for v, label in enumerate([2, 10, 1]):
            rows.append({"pid": f"P{i:03d}", "visit": label, "age": 40 + i % 30 + (v == 2),
                         "event": int((i + v) % 2 == 0),
                         "sbp": (np.nan if (label == 1 and i < 5) else 120 + i + 5 * v)})
    df = pd.DataFrame(rows)
    run = graph(df, "b21")
    repeat = {"repeat_kind": "time_points", "time_column": "visit"}
    with pytest.raises(StructureError, match="two values"):
        run.run(state(**COMBINE, target="event", repeat_kind=repeat,
                      aggregation={"method": "mean", "outcome": "mean"}), upto=["working"])
    out = run.run(state(**COMBINE, target="sbp", repeat_kind=repeat,
                        aggregation={"method": "change", "outcome": "first"}), upto=["working"])
    got = table(out["working"]).set_index("pid")
    first = df.sort_values(["pid", "visit"]).groupby("pid").head(1).set_index("pid")["sbp"]
    assert np.array_equal(got.loc[first.index, "sbp"].to_numpy(), first.to_numpy(), equal_nan=True)
    assert got["sbp"].isna().sum() == 5  # the first visit's blank is not filled from a later one
    listed = {e["column"] for e in out["working"].data["aggregation"]["columns"]}
    assert "age" in listed and "event" in listed
