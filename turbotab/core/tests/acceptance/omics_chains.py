"""The reference tables for MODELING_SEQUENCE §6 chains 3 and 5 (``test_ms7_chains.py``), each
simulated with its truth known."""
from __future__ import annotations

import numpy as np
import pandas as pd

LOD = 150.0  # the instrument's detection limit in chain 3's raw units


def metabolomics_run(seed: int = 3, batches: int = 2, per_batch: int = 50, qc_every: int = 5,
                     p: int = 120, signal: int = 10, near: int = 24) -> tuple[pd.DataFrame, dict]:
    """Chain 3's table: an untargeted LC-MS run, one row per injection.

    ``batches`` batches of ``per_batch`` injections, a pooled QC at every ``qc_every``-th and the
    last; the study injections are two samples from each participant (repeat samples), cases and
    controls half each. Every metabolite carries its level, the participant's biology (the first
    ``signal`` higher in cases by 0.6 on the log scale), the sample's dilution (SD 0.3 on the log
    scale; the pooled QCs undiluted), a batch offset, a drift over the batch, and noise (SD 0.05).
    The ``near`` least abundant metabolites sit around the instrument's detection limit and are
    blank below it (some in most QCs, some in a few study samples); the pooled-QC rows have no
    outcome."""
    rng = np.random.default_rng(seed)
    rows, order, pid = [], 0, 0
    study_slots = []
    for b in range(batches):
        for k in range(per_batch):
            order += 1
            qc = (k % qc_every == 0) or k == per_batch - 1
            rows.append({"injection_order": order, "batch": f"B{b + 1}",
                         "sample_type": "QC" if qc else "Sample", "_k": k})
            if not qc:
                study_slots.append(len(rows) - 1)
    meta = pd.DataFrame(rows)
    n_study = len(study_slots)
    n_people = n_study // 2
    people = np.repeat(np.arange(n_people), 2)[:n_study]
    rng.shuffle(people)
    case_of = np.r_[np.ones(n_people // 2), np.zeros(n_people - n_people // 2)]
    rng.shuffle(case_of)
    meta["participant_id"] = None
    meta["case"] = np.nan
    meta.loc[study_slots, "participant_id"] = [f"P{q:03d}" for q in people]
    meta.loc[study_slots, "case"] = case_of[people]
    n = len(meta)
    qc = meta["sample_type"].eq("QC").to_numpy()
    level = np.exp(rng.normal(8, 1.2, p))
    low = np.argsort(level)[:near]
    level[low] = np.exp(rng.uniform(4.4, 5.8, near))  # near the detection limit
    person_bio = rng.normal(0, 0.4, (n_people, p))
    person_bio[:, :signal] += 0.35 * case_of[:, None]
    biology = np.ones((n, p))
    biology[study_slots] = np.exp(person_bio[people] + rng.normal(0, 0.15, (n_study, p)))
    dilution = np.where(qc, 1.0, np.exp(rng.normal(0, 0.3, n)))
    position = meta["_k"].to_numpy() / (per_batch - 1)
    amp = rng.uniform(0.1, 0.35, p)
    offsets = {f"B{b + 1}": np.exp(rng.normal(0, 0.25, p)) for b in range(batches)}
    drift = np.vstack([offsets[bb] * (1 - amp * pos) for bb, pos in zip(meta["batch"], position)])
    values = level * biology * dilution[:, None] * drift * np.exp(rng.normal(0, 0.05, (n, p)))
    values[values < LOD] = np.nan
    names = [f"mz_{j:03d}" for j in range(p)]
    frame = pd.DataFrame(values, columns=names)
    frame.insert(0, "case", meta["case"])
    frame.insert(0, "participant_id", meta["participant_id"])
    frame.insert(0, "sample_type", meta["sample_type"])
    frame.insert(0, "batch", meta["batch"])
    frame.insert(0, "injection_order", meta["injection_order"])
    frame.insert(0, "sample_id", [f"{'QC' if q else 'S'}{i:03d}" for i, q in enumerate(qc)])
    truth = {"signal": names[:signal], "near_limit": [names[j] for j in low],
             "n_qc": int(qc.sum()), "n_study": n_study}
    return frame, truth


def genomics_batches(seed: int = 5, n: int = 90, p: int = 600, signal: int = 15,
                     depth_ratio: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Chain 5's table: RNA-seq counts, p ≫ n, three sequencing batches, the cases measured
    unevenly across them (Nygaard et al.'s unbalanced design). Each batch multiplies every gene
    by its own factor (SD 0.4 on the log scale); the first ``signal`` genes are 1.8-fold higher in
    cases; counts are negative binomial (dispersion 0.1)."""
    rng = np.random.default_rng(seed)
    case = np.r_[np.ones(n // 2), np.zeros(n - n // 2)].astype(int)
    rng.shuffle(case)
    probs = {1: [0.55, 0.30, 0.15], 0: [0.15, 0.30, 0.55]}
    batch = np.array([rng.choice(["B1", "B2", "B3"], p=probs[c]) for c in case])
    base = np.exp(rng.normal(np.log(80), 1.2, p))
    effect = {b: np.exp(rng.normal(0, 0.4, p)) for b in ("B1", "B2", "B3")}
    depth = np.exp(rng.normal(0, 0.2, n))
    mu = np.vstack([base * effect[b] for b in batch]) * depth[:, None]
    mu[:, :signal] *= np.where(case[:, None] == 1, 1.8, 1.0)
    r = 10.0
    counts = rng.negative_binomial(r, r / (r + mu))
    names = [f"ENSG{100000 + j:011d}" for j in range(p)]
    frame = pd.DataFrame(counts, columns=names)
    frame.insert(0, "batch", batch)
    frame.insert(0, "sample_id", [f"S{i:03d}" for i in range(n)])
    frame["case"] = case
    return frame, {"signal": names[:signal]}


__all__ = ["LOD", "genomics_batches", "metabolomics_run"]
