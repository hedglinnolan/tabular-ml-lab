"""Temporary (ci/elastic-net-determinism-diag only): the WP7 elastic net's choices on this platform,
printed as annotations, to compare with macOS."""
import hashlib
import platform
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNetCV

sys.path.insert(0, ".")


def esc(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def say(title: str, lines: list[str]) -> None:
    print(f"::warning title={title}::{esc(chr(10).join(lines))}")
    print("\n".join([title, *lines]), file=sys.stderr)


def main() -> None:
    from turbotab.core.models.elastic_net import L1_RATIOS, PooledElasticNetCV
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests import modeling_fixtures as mf
    from turbotab.core.tests.acceptance import test_wp7_missing_data as wp7

    captured = []
    original = PooledElasticNetCV.fit

    def spy(self, X, y, *a, **k):
        captured.append((np.asarray(X, dtype=float).copy(), np.asarray(y, dtype=float).copy(),
                         self.cv))
        return original(self, X, y, *a, **k)

    PooledElasticNetCV.fit = spy
    frame = wp7.prediction_fixture()
    paths = mf.ingest_frame(frame, Path(tempfile.mkdtemp()))
    split = mf.split_bundle(np.arange(len(frame)), seed=707)
    ti = mf.target_info("regression")
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    import sklearn
    import scipy
    head = [f"{platform.system()} {platform.machine()} {platform.processor()} numpy {np.__version__} "
            f"scipy {scipy.__version__} sklearn {sklearn.__version__} blas {blas.get('name')} "
            f"{blas.get('version')}"]
    try:
        from threadpoolctl import threadpool_info
        head += [f"{i.get('internal_api')} {i.get('architecture')} threads {i.get('num_threads')}"
                 for i in threadpool_info()]
    except Exception as e:  # noqa: BLE001
        head.append(repr(e))
    say("diag platform", head)
    for name, slots in wp7.PREDICTION_CONFIGS.items():
        start = len(captured)
        st = mf.state(**{**slots, "models": ["elastic_net"]})
        design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
        fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
        fits = captured[start:]
        digest = hashlib.sha256(b"".join(X.tobytes() + y.tobytes() for X, y, _ in fits)).hexdigest()[:16]
        old_idx, new_idx = [], []
        for X, y, cv in fits:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a = ElasticNetCV(l1_ratio=list(L1_RATIOS), cv=cv, max_iter=5000).fit(X, y)
            mix = list(L1_RATIOS).index(a.l1_ratio_)
            old_idx.append(f"{mix}.{int(np.argmin(np.abs(a.alphas_[mix] - a.alpha_)))}")
        for X, y, cv in fits:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = PooledElasticNetCV(l1_ratio=list(L1_RATIOS), cv=cv, max_iter=5000, tol=1e-12)
                original(c, X, y)
            mix = list(L1_RATIOS).index(c.l1_ratio_)
            new_idx.append(f"{mix}.{int(np.argmin(np.abs(c.alphas_[mix] - c.alpha_)))}")
        final = c
        best = np.sort(final.pooled_loss_.ravel())[:3]
        m = next(m for m in fit.data["models"] if m["family"] == "elastic_net")
        say(f"diag {name}", [
            f"inputs digest {digest} over {len(fits)} fits",
            f"old (1e-4, mean) choices: {' '.join(old_idx)}",
            f"new (1e-12, pooled) choices: {' '.join(new_idx)}",
            f"final: alpha_ {final.alpha_!r} l1 {final.l1_ratio_!r} intercept {final.intercept_!r}",
            f"final best pooled losses {[repr(float(b)) for b in best]}",
            f"stage r2 {m['cv']['r2']['estimate']!r} rmse {m['cv']['rmse']['estimate']!r}",
        ])


if __name__ == "__main__":
    main()
