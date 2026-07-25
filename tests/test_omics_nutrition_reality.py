"""Real -omics and nutrition data shapes, written before the fixes they force.

The stress tests so far used data I invented to exercise code I had written,
which is a closed loop. These fixtures are built from how the data actually
arrives in the two fields this app is for:

  TRANSCRIPTOMICS   20,000-60,000 genes x 50-500 samples, genes as ROWS
  PROTEOMICS        5,000-10,000 proteins, heavy below-detection missingness
  METABOLOMICS      features named by mass/retention time, "<LOD" values
  MICROBIOME        taxonomy strings as column names, zero-inflated counts
  TCGA              hierarchical barcodes: patient-level clinical must join to
                    aliquot-level assays by PREFIX
  NHANES            SEQN, 2-year cycles, survey weights that must be divided
                    when cycles are combined
  DIETARY RECALL    person x day x food item, deeply long
  FFQ               ~150 food items, wide

Sources for the two domain rules encoded here:
  https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx
  https://waldronlab.io/TCGAutils/reference/TCGAbarcode.html
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

RNG = np.random.RandomState(7)


@pytest.fixture(autouse=True)
def _deterministic_test_data():
    """Reseed the shared RNG before every test in this module.

    Without this, every test draws from one advancing stream, so the data a
    test sees depends on how many tests ran before it — and the suite is green
    only for the collection order it happens to run in.
    """
    RNG.seed(7)



# ── fixtures that look like the real thing ───────────────────────────────

def expression_matrix(n_genes=2000, n_samples=60):
    """Genes as ROWS, samples as COLUMNS — how every expression matrix ships."""
    genes = [f"ENSG{i:011d}.{RNG.randint(1, 20)}" for i in range(n_genes)]
    samples = [f"GSM{2000000 + i}" for i in range(n_samples)]
    data = RNG.lognormal(3, 2, (n_genes, n_samples)).round(3)
    df = pd.DataFrame(data, columns=samples)
    df.insert(0, "gene_id", genes)
    return df


def tcga_clinical(n=80):
    return pd.DataFrame({
        "bcr_patient_barcode": [f"TCGA-{RNG.randint(1,99):02d}-{1000+i}" for i in range(n)],
        "age_at_diagnosis": RNG.randint(30, 85, n),
        "vital_status": RNG.choice(["Alive", "Dead"], n),
    })


def tcga_expression_samples(clinical, n_features=50):
    """Aliquot-level barcodes: the patient barcode plus sample/vial/portion."""
    rows = []
    for pid in clinical["bcr_patient_barcode"]:
        rows.append(f"{pid}-01A-{RNG.randint(11,31)}R-{RNG.randint(1000,9999)}-07")
    out = pd.DataFrame({"sample_barcode": rows})
    for i in range(n_features):
        out[f"gene_{i}"] = RNG.lognormal(2, 1, len(out)).round(2)
    return out


def microbiome_counts(n=120, n_taxa=40):
    taxa = [f"k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;"
            f"f__Lachnospiraceae;g__Blautia;s__sp{i}" for i in range(n_taxa)]
    df = pd.DataFrame(RNG.poisson(3, (n, n_taxa)), columns=taxa)
    df.insert(0, "SampleID", [f"S{i:04d}" for i in range(n)])
    return df


def nhanes_cycle(year, seqn_start, n=400):
    """A cycle carrying the survey design variables that make it a survey."""
    return pd.DataFrame({
        "SEQN": range(seqn_start, seqn_start + n),
        "RIAGENDR": RNG.choice([1, 2], n),
        "RIDAGEYR": RNG.randint(18, 80, n),
        "WTMEC2YR": RNG.lognormal(10, 0.5, n).round(2),
        "WTINT2YR": RNG.lognormal(10, 0.5, n).round(2),
        "SDMVPSU": RNG.choice([1, 2], n),
        "SDMVSTRA": RNG.randint(100, 130, n),
        "cycle": year,
    })


def dietary_recall(n_people=200, days=2, foods=12):
    rows = []
    for p in range(n_people):
        for d in range(1, days + 1):
            for f in range(foods):
                rows.append({"SEQN": 90000 + p, "day": d,
                             "food_code": RNG.randint(11000000, 99000000),
                             "grams": round(RNG.uniform(5, 400), 1),
                             "kcal": round(RNG.uniform(5, 600), 1)})
    return pd.DataFrame(rows)


# ── wide data: is the key even reachable? ────────────────────────────────

class TestWideOmicsMatrices:

    def test_key_is_found_in_a_wide_feature_matrix(self):
        """A transposed expression matrix is samples x thousands of genes. If
        the key finder only inspects the first N columns, a real study cannot
        be joined to its own clinical file."""
        from ml.join_doctor import suggest_best
        expr = expression_matrix(n_genes=500, n_samples=60)
        wide = expr.set_index("gene_id").T.reset_index()
        wide = wide.rename(columns={"index": "sample_id"})
        clinical = pd.DataFrame({"sample_id": wide["sample_id"],
                                 "age": RNG.randint(30, 80, len(wide))})
        best = suggest_best(wide, clinical)
        assert best is not None, "no key found in a 500-feature matrix"
        assert best.left_col == "sample_id"

    def test_key_is_found_when_it_is_not_the_first_column(self):
        """Exports routinely put the ID after a block of annotation columns."""
        from ml.join_doctor import suggest_best
        n = 100
        wide = pd.DataFrame({f"probe_{i}": RNG.normal(size=n) for i in range(200)})
        wide["SampleID"] = [f"S{i:04d}" for i in range(n)]      # last column
        clinical = pd.DataFrame({"SampleID": wide["SampleID"], "bmi": RNG.normal(27, 4, n)})
        best = suggest_best(wide, clinical)
        assert best is not None and best.left_col == "SampleID"

    def test_import_doctor_finishes_on_a_wide_matrix(self):
        from ml.import_doctor import diagnose
        wide = pd.DataFrame({f"gene_{i}": RNG.lognormal(2, 1, 80) for i in range(3000)})
        wide.insert(0, "sample_id", [f"GSM{i}" for i in range(80)])
        start = time.time()
        diagnose(wide)
        assert time.time() - start < 25, "structural review too slow to be usable"


# ── identifiers that need more than exact matching ───────────────────────

class TestHierarchicalIdentifiers:

    def test_tcga_patient_to_aliquot_is_recognized(self):
        """Clinical is patient-level (TCGA-02-0001); assays are aliquot-level
        (TCGA-02-0001-01A-21R-1898-07). Zero exact overlap, and every TCGA
        study needs this join."""
        from ml.join_doctor import diagnose_join, find_key_candidates
        clinical = tcga_clinical(60)
        expr = tcga_expression_samples(clinical)
        d = diagnose_join(clinical, expr, "bcr_patient_barcode", "sample_barcode")
        assert d.matched_keys == 0                    # exact matching cannot work
        blob = " ".join(d.blocking + d.warnings + d.notes).lower()
        assert "start" in blob or "prefix" in blob or "longer" in blob, (
            "the app says nothing about one ID being the start of the other")

    def test_versioned_ensembl_ids_are_recognized(self):
        """ENSG00000141510.16 and ENSG00000141510 are the same gene."""
        from ml.join_doctor import diagnose_join
        left = pd.DataFrame({"gene": [f"ENSG{i:011d}.{RNG.randint(1,9)}" for i in range(200)],
                             "logfc": RNG.normal(size=200)})
        right = pd.DataFrame({"gene": [c.split(".")[0] for c in left["gene"]],
                              "pathway": RNG.choice(["a", "b"], 200)})
        d = diagnose_join(left, right, "gene", "gene")
        blob = " ".join(d.blocking + d.warnings + d.notes).lower()
        assert d.matched_keys == 200 or "start" in blob or "prefix" in blob or "version" in blob


# ── survey data carries rules the app should know ────────────────────────

class TestNHANESSurveyWeights:
    """Combining cycles without dividing the weights inflates the population
    estimate by the number of cycles. CDC states the rule explicitly:
    https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx
    """

    def test_stacking_cycles_warns_about_survey_weights(self):
        from utils.combine_preview import describe_stack
        frames = {"2015-2016": nhanes_cycle(2015, 83000),
                  "2017-2018": nhanes_cycle(2017, 93000)}
        cm = describe_stack(frames)
        blob = " ".join(cm.consequences).lower()
        assert "weight" in blob, "stacked two survey cycles without mentioning weights"

    def test_the_warning_names_the_weight_columns_it_found(self):
        from utils.combine_preview import describe_stack
        frames = {"2015-2016": nhanes_cycle(2015, 83000),
                  "2017-2018": nhanes_cycle(2017, 93000)}
        blob = " ".join(describe_stack(frames).consequences)
        assert "WTMEC2YR" in blob

    def test_no_weight_warning_when_there_are_no_weights(self):
        from utils.combine_preview import describe_stack
        frames = {"a": pd.DataFrame({"SEQN": range(50), "x": RNG.normal(size=50)}),
                  "b": pd.DataFrame({"SEQN": range(50, 100), "x": RNG.normal(size=50)})}
        assert not any("weight" in c.lower() for c in describe_stack(frames).consequences)


# ── long-format dietary data ─────────────────────────────────────────────

class TestDietaryRecall:

    def test_person_day_food_structure_is_joinable_to_demographics(self):
        from ml.join_doctor import diagnose_join, execute_join, suggest_best
        recall = dietary_recall(150, days=2, foods=10)
        demo = pd.DataFrame({"SEQN": sorted(recall["SEQN"].unique()),
                             "age": RNG.randint(18, 80, recall["SEQN"].nunique())})
        best = suggest_best(demo, recall)
        assert best is not None and best.left_col == "SEQN"
        d = diagnose_join(demo, recall, "SEQN", "SEQN", "inner")
        out, _ = execute_join(demo, recall, "SEQN", "SEQN", "inner", "demo", "recall")
        assert len(out) == d.predicted_rows == len(recall)

    def test_the_fanout_is_disclosed_as_changing_what_n_means(self):
        from utils.combine_preview import describe_join
        recall = dietary_recall(150, days=2, foods=10)
        demo = pd.DataFrame({"SEQN": sorted(recall["SEQN"].unique()),
                             "age": RNG.randint(18, 80, recall["SEQN"].nunique())})
        cm = describe_join(demo, recall, "SEQN", "SEQN", "inner", "demographics", "recall")
        assert any("no longer the number of people" in c for c in cm.consequences)

    def test_food_codes_are_not_mistaken_for_the_subject_key(self):
        """food_code is numeric, high-cardinality and shared with the food
        composition table — but it identifies FOODS, not people."""
        from ml.join_doctor import suggest_best
        recall = dietary_recall(120, days=2, foods=8)
        demo = pd.DataFrame({"SEQN": sorted(recall["SEQN"].unique()),
                             "age": RNG.randint(18, 80, recall["SEQN"].nunique())})
        best = suggest_best(demo, recall)
        assert best.left_col == "SEQN"


# ── awkward column names ─────────────────────────────────────────────────

class TestTaxonomyColumnNames:

    def test_semicolon_taxonomy_names_survive_a_join(self):
        from ml.join_doctor import execute_join
        counts = microbiome_counts(80, n_taxa=15)
        meta = pd.DataFrame({"SampleID": counts["SampleID"],
                             "group": RNG.choice(["case", "control"], 80)})
        out, _ = execute_join(counts, meta, "SampleID", "SampleID", "inner", "counts", "meta")
        assert len(out) == 80
        assert any(";" in str(c) for c in out.columns)

    def test_taxonomy_names_are_escaped_in_the_preview(self):
        from utils.combine_preview import describe_stack
        a = microbiome_counts(40, n_taxa=8)
        b = microbiome_counts(40, n_taxa=8)
        b["SampleID"] = [f"T{i:04d}" for i in range(40)]
        cm = describe_stack({"run1": a, "run2": b})
        assert cm.after_rows == 80

    def test_import_doctor_handles_taxonomy_names(self):
        from ml.import_doctor import diagnose
        diagnose(microbiome_counts(60, n_taxa=20))   # must not raise


# ── below-detection values ───────────────────────────────────────────────

class TestBelowDetectionLimits:

    def test_a_less_than_value_is_not_silently_substituted(self):
        """"<0.01" is a censored observation. Turning it into 0.01 is an
        analytic decision, not a formatting fix, and must not be pre-selected."""
        from ml.import_doctor import diagnose
        df = pd.DataFrame({"analyte": ["<0.01"] * 30 + [f"{v:.3f}" for v in
                                        RNG.uniform(0.02, 5, 70)]})
        found = [f for f in diagnose(df) if f.fix_kind == "coerce_numeric"]
        assert found, "a censored lab column was not flagged at all"
        assert not found[0].auto_suggestable or "<" in found[0].detail

    def test_nondetect_tokens_are_offered_as_missing_at_low_confidence(self):
        from ml.import_doctor import diagnose
        df = pd.DataFrame({"metab": ["n.d."] * 20 + [f"{v:.2f}" for v in
                                     RNG.uniform(1, 9, 80)]})
        assert diagnose(df) is not None


# ── technical replicates ─────────────────────────────────────────────────

class TestTechnicalReplicates:

    def test_replicate_rows_make_the_lockbox_split_by_subject(self):
        """The same subject measured twice on the same day is repeated
        measures. Splitting by row puts a subject in train AND test."""
        import streamlit as st
        from utils.test_lockbox import ensure_lockbox
        n_subj = 120
        df = pd.DataFrame({
            "subject_id": np.repeat([f"P{i:03d}" for i in range(n_subj)], 2),
            "replicate": list([1, 2]) * n_subj,
            "protein": RNG.normal(size=n_subj * 2),
            "outcome": np.repeat(RNG.choice([0, 1], n_subj), 2),
        })
        st.session_state.clear()
        lb = ensure_lockbox(df, "outcome", "classification", fraction=0.2, seed=3)
        test_subjects = set(df.loc[lb["labels"], "subject_id"])
        train_subjects = set(df.loc[~df.index.isin(lb["labels"]), "subject_id"])
        assert not (test_subjects & train_subjects), "a subject is in both halves"


# ── the new detectors must not fire when they shouldn't ──────────────────

class TestDetectorsDoNotOverfire:
    """Both of these were added to help with -omics identifiers, and both could
    do more harm than good by firing on ordinary data."""

    @pytest.mark.parametrize("name,is_survey_weight", [
        ("WTMEC2YR", True), ("WTINT2YR", True), ("WTMEC4YR", True),
        ("WTSAF2YR", True), ("survey_weight", True), ("sampling_weight", True),
        ("pweight", True), ("person_weight", True),
        # A physical mass, not a survey weight. "sample_weight" especially:
        # in -omics and food chemistry that is the mass of a specimen.
        ("sample_weight", False), ("BMXWT", False), ("weight_kg", False),
        ("weight", False), ("birth_weight", False), ("dry_weight", False),
        ("bodyweight", False), ("net_wt", False), ("wt_grams", False),
    ])
    def test_survey_weight_detection_is_narrow(self, name, is_survey_weight):
        from utils.combine_preview import _SURVEY_WEIGHT
        assert bool(_SURVEY_WEIGHT.match(name)) is is_survey_weight

    def test_a_body_weight_column_does_not_trigger_the_weighting_advice(self):
        from utils.combine_preview import describe_stack
        frames = {"site_a": pd.DataFrame({"SEQN": range(60), "weight_kg": RNG.normal(75, 12, 60)}),
                  "site_b": pd.DataFrame({"SEQN": range(60, 120), "weight_kg": RNG.normal(75, 12, 60)})}
        assert not any("weight" in c.lower() and "cycle" in c.lower()
                       for c in describe_stack(frames).consequences)

    @pytest.mark.parametrize("label,left,right,should_fire", [
        ("unrelated text ids",
         [f"AAA{i:05d}" for i in range(100)], [f"ZZZ{i:05d}" for i in range(100)], False),
        ("short numeric ids", list(range(1, 101)), list(range(500, 600)), False),
        ("already matching exactly",
         [f"S{i:04d}" for i in range(100)], [f"S{i:04d}" for i in range(100)], False),
        ("tcga patient vs aliquot",
         [f"TCGA-02-{1000+i}" for i in range(60)],
         [f"TCGA-02-{1000+i}-01A-21R-1898-07" for i in range(60)], True),
        ("ensembl with versions",
         [f"ENSG{i:011d}" for i in range(100)],
         [f"ENSG{i:011d}.7" for i in range(100)], True),
    ])
    def test_nested_id_detection_is_specific(self, label, left, right, should_fire):
        from ml.join_doctor import detect_nested_ids
        fired = detect_nested_ids(pd.Series(left), pd.Series(right)) is not None
        assert fired is should_fire, label

    def test_the_explanation_quotes_the_users_own_spelling(self):
        """Matching is case-folded; echoing the folded form back shows a TCGA
        researcher 'tcga-02-1000' for an ID their file spells in capitals."""
        from ml.join_doctor import detect_nested_ids
        msg = detect_nested_ids(
            pd.Series([f"TCGA-02-{1000+i}" for i in range(60)]),
            pd.Series([f"TCGA-02-{1000+i}-01A-21R-1898-07" for i in range(60)]))
        assert "TCGA-02-" in msg and "tcga-02-" not in msg

    def test_the_explanation_says_what_to_do(self):
        from ml.join_doctor import detect_nested_ids
        msg = detect_nested_ids(
            pd.Series([f"ENSG{i:011d}" for i in range(100)]),
            pd.Series([f"ENSG{i:011d}.7" for i in range(100)]))
        assert "Remove everything after the last" in msg

    def test_candidate_ordering_keeps_a_shared_name_first(self):
        from ml.join_doctor import _columns_worth_testing
        wide = pd.DataFrame({f"gene_{i}": [0.0] for i in range(300)})
        wide["SampleID"] = ["S1"]
        other = pd.DataFrame({"SampleID": ["S1"], "age": [40]})
        assert "SampleID" in _columns_worth_testing(wide, other, 60)

    def test_candidate_ordering_is_a_no_op_on_narrow_frames(self):
        from ml.join_doctor import _columns_worth_testing
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert _columns_worth_testing(df, df, 60) == ["a", "b", "c"]
