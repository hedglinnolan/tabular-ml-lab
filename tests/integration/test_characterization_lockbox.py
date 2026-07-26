"""Characterization tests for the lockbox seal/redraw signature.

`ARCHITECTURE.md` §02 states the invariant: *the lockbox is sealed before
exploration, and sealed once.* `ensure_lockbox()` freezes row labels keyed by a
signature over `(df, target, task_type, fraction, seed, group_col)`, and a
redraw sets a marker, because a silent re-draw invalidates every downstream
number.

These pin the signature's behavior: which inputs change it, which do not, and
what happens when a redraw is attempted. They run the production function
through a Streamlit script, because it reads and writes `st.session_state`.

The distinction that matters, and that the extraction must not blur:

- **Same inputs → same lockbox object.** Not merely the same labels; the *same
  seal*, so "sealed once" stays a statement about history.
- **Changed inputs → a new signature.** Otherwise a changed dataset keeps a
  quarantine drawn for a different one.

Row identity here is **index labels** (`train_row_mask` tests `lbl not in
test_set`), which is what makes `T0-ID-001`'s identity barrier load-bearing: a
repair that renumbers rows leaves these labels naming different rows.
"""
import os

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.timeout(300)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _frame(n=120, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "age": rng.randint(30, 80, n),
        "bmi": rng.normal(27, 4, n).round(1),
        "glucose": rng.normal(100, 15, n).round(1),
    })


_SCRIPT = """
import streamlit as st, sys, pickle, base64
sys.path.insert(0, {root!r})
from utils.test_lockbox import ensure_lockbox, get_lockbox, train_row_mask

df = pickle.loads(base64.b64decode(st.session_state["_df"]))
calls = st.session_state["_calls"]
out = []
for kwargs in calls:
    lb = ensure_lockbox(df, **kwargs)
    out.append(None if lb is None else {{
        "signature": lb.get("signature"),
        "labels": sorted(lb["labels"]),
        "n_test": lb.get("n_test"),
        "fraction": lb.get("fraction"),
        "seed": lb.get("seed"),
    }})
st.session_state["_out"] = out
st.session_state["_redraw_refused"] = "_lockbox_redraw_refused" in st.session_state
"""


def _seal(df, calls, session=None):
    import base64
    import pickle
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_SCRIPT.format(root=ROOT))
    at.session_state["_df"] = base64.b64encode(pickle.dumps(df)).decode()
    at.session_state["_calls"] = calls
    for k, v in (session or {}).items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at.session_state["_out"], at


# ── sealed once ──────────────────────────────────────────────────────────

def test_same_inputs_reuse_the_same_seal():
    """Not "the same labels" — the same lockbox. Sealed once is a claim about
    history, and re-deriving an identical draw would not support it."""
    df = _frame()
    out, _ = _seal(df, [
        {"target_col": "glucose", "task_type": "regression"},
        {"target_col": "glucose", "task_type": "regression"},
        {"target_col": "glucose", "task_type": "regression"},
    ])
    assert out[0] is not None, "no lockbox was drawn at all"
    assert out[0]["signature"] == out[1]["signature"] == out[2]["signature"]
    assert out[0]["labels"] == out[1]["labels"] == out[2]["labels"]


def test_the_seal_holds_labels_not_positions():
    """`train_row_mask` tests membership by label. A lockbox that stored
    positions would stop meaning the same rows after any filter — which is
    exactly `T0-ID-001`."""
    df = _frame()
    df.index = range(1000, 1000 + len(df))
    out, _ = _seal(df, [{"target_col": "glucose", "task_type": "regression"}])
    assert out[0] is not None
    assert all(lbl >= 1000 for lbl in out[0]["labels"]), (
        "the lockbox stored positions, not index labels")
    assert set(out[0]["labels"]) <= set(df.index)


# ── what changes the signature ───────────────────────────────────────────

@pytest.mark.parametrize("change", [
    pytest.param({"seed": 999}, id="seed"),
    pytest.param({"fraction": 0.30}, id="fraction"),
    pytest.param({"target_col": "bmi"}, id="target"),
    pytest.param({"task_type": "classification"}, id="task_type"),
])
def test_changing_an_input_changes_the_signature(change):
    """Each of these is part of the signature, so each must force a re-seal.
    A signature that ignored one would keep a quarantine drawn for a question
    nobody is asking any more."""
    df = _frame()
    base = {"target_col": "glucose", "task_type": "regression",
            "fraction": 0.2, "seed": 42}
    changed = {**base, **change}
    out, _ = _seal(df, [base, changed])
    assert out[0] is not None and out[1] is not None
    assert out[0]["signature"] != out[1]["signature"], (
        f"changing {list(change)[0]} did not change the seal")


def test_changing_the_data_changes_the_signature():
    """The signature hashes content. A different dataset must not inherit a
    quarantine drawn for another one — the multi-user hazard behind
    `T0-LIVE-001`, in a different subsystem."""
    df_a = _frame(seed=0)
    df_b = _frame(seed=1)
    call = {"target_col": "glucose", "task_type": "regression"}
    out_a, _ = _seal(df_a, [call])
    out_b, _ = _seal(df_b, [call])
    assert out_a[0]["signature"] != out_b[0]["signature"]


def test_a_lockbox_is_not_drawn_for_a_frame_that_is_too_small():
    """Refusing is allowed; drawing a meaningless quarantine is not."""
    out, _ = _seal(_frame(n=8), [{"target_col": "glucose", "task_type": "regression"}])
    assert out[0] is None


def test_no_lockbox_without_a_target():
    out, _ = _seal(_frame(), [{"target_col": "", "task_type": "regression"}])
    assert out[0] is None


# ── the sealed rows are usable as an identity ────────────────────────────

def test_train_row_mask_excludes_exactly_the_sealed_labels():
    """The consumer side of the seal, pinned with it: whatever the lockbox
    holds, the training mask must exclude precisely those labels."""
    import base64
    import pickle
    from streamlit.testing.v1 import AppTest

    df = _frame()
    script = _SCRIPT.format(root=ROOT) + """
lb = get_lockbox()
mask = train_row_mask(df.index)
st.session_state["_mask_true"] = sorted(df.index[mask].tolist())
st.session_state["_sealed"] = sorted(lb["labels"])
"""
    at = AppTest.from_string(script)
    at.session_state["_df"] = base64.b64encode(pickle.dumps(df)).decode()
    at.session_state["_calls"] = [{"target_col": "glucose", "task_type": "regression"}]
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    sealed = set(at.session_state["_sealed"])
    train = set(at.session_state["_mask_true"])
    assert sealed, "nothing was sealed"
    assert not (sealed & train), "a sealed row is marked trainable — this is the leak"
    assert sealed | train == set(df.index), "rows went missing between the two"
