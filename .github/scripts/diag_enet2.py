"""Temporary (ci/elastic-net-determinism-diag only): where the WP7 elastic net's inputs part ways
between platforms: the pipeline's raw inputs, their row keys, the inner splits, the model matrix."""
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")


def esc(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def say(title: str, lines: list[str]) -> None:
    print(f"::warning title={title}::{esc(chr(10).join(lines))}")
    print("\n".join([title, *lines]), file=sys.stderr)


def h(*parts) -> str:
    m = hashlib.sha256()
    for p in parts:
        m.update(p if isinstance(p, bytes) else repr(p).encode())
    return m.hexdigest()[:12]


def main() -> None:
    from turbotab.core.models import inner_cv
    from turbotab.core.models.elastic_net import PooledElasticNetCV
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests import modeling_fixtures as mf
    from turbotab.core.tests.acceptance import test_wp7_missing_data as wp7

    log = []
    orig_keys, orig_splits, orig_fit = inner_cv.row_keys, inner_cv.inner_splits, PooledElasticNetCV.fit

    def keys_spy(X, y=None):
        out = orig_keys(X, y)
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        log.append(("keys", list(frame.columns), [str(t) for t in frame.dtypes],
                    h(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()),
                    h(np.asarray(y, dtype=float).tobytes()), h(out.tobytes()),
                    repr(frame.iloc[0].tolist())[:300], repr(np.asarray(y)[:2].tolist())))
        return out

    def splits_spy(*a, **k):
        out = orig_splits(*a, **k)
        log.append(("splits", h(b"".join(np.asarray(t).tobytes() + np.asarray(s).tobytes()
                                          for t, s in (out or [])))))
        return out

    def fit_spy(self, X, y, *a, **k):
        Xa = np.asarray(X, dtype=float)
        log.append(("matrix", Xa.shape, h(Xa.tobytes()), h(np.round(Xa, 8).tobytes())))
        return orig_fit(self, X, y, *a, **k)

    inner_cv.row_keys, inner_cv.inner_splits = keys_spy, splits_spy
    PooledElasticNetCV.fit = fit_spy
    frame = wp7.prediction_fixture()
    paths = mf.ingest_frame(frame, Path(tempfile.mkdtemp()))
    split = mf.split_bundle(np.arange(len(frame)), seed=707)
    ti = mf.target_info("regression")
    st = mf.state(**{**wp7.PREDICTION_CONFIGS["impute"], "models": ["elastic_net"]})
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    stored = pd.read_parquet(paths["data"])
    lines = [f"pandas {pd.__version__} stored {h(pd.util.hash_pandas_object(stored, index=False).to_numpy().tobytes())} "
             f"dtypes {[str(t) for t in stored.dtypes]}"]
    first = [e for e in log if e[0] == "keys"][:1]
    for e in first:
        lines.append(f"first keys call: columns {e[1]}")
        lines.append(f"  dtypes {e[2]}")
        lines.append(f"  frame-hash {e[3]} y {e[4]} keys {e[5]}")
        lines.append(f"  row0 {e[6]} y0 {e[7]}")
    seq = [e for e in log if e[0] != "keys"]
    lines.append("sequence (first 12): " + " | ".join(
        f"{e[0]} {' '.join(map(str, e[1:]))}" for e in seq[:12]))
    lines.append("all keys: " + h([e[5] for e in log if e[0] == "keys"]) + " all splits: "
                 + h([e[1] for e in log if e[0] == "splits"]) + " all matrices: "
                 + h([e[2] for e in log if e[0] == "matrix"]) + " rounded: "
                 + h([e[3] for e in log if e[0] == "matrix"]))
    say("diag2 impute", lines)


if __name__ == "__main__":
    main()
