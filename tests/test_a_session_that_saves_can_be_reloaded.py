"""`F-09` — a save that announces success must be a save the app can reopen.

`render_session_controls` built the archive, checked its length against
`_MAX_UPLOAD_BYTES`, and on the way past printed a green **Session Ready for
Download!** whatever that check said. The amber line above it named the wrong
consequence ("Session is very large … consider completing your analysis before
saving" — advice that makes the file *bigger*) and never said the thing that
mattered: the app will refuse this file on the way back in. There is no second
copy — `utils/session_projects.py` keeps projects in session state and writes
nothing to disk — so the researcher's only artifact was one the app would not
accept, and it was labeled a success.

The compressed cap is also not the only door. `_validate_zip` refuses on
uncompressed total and on member count too, and save tested neither, so the
tests below cover a member-cap archive that sails through a naive length check.

Warn-and-proceed is deliberate and is asserted here: the download button still
renders when the archive is unreloadable, because an archive that is merely
un-re-uploadable is still a readable ZIP of Parquet and JSON, while a refused
save is nothing at all.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils import session_manager                                # noqa: E402
from utils.session_manager import (                              # noqa: E402
    _MAX_UPLOAD_BYTES,
    _MAX_MEMBERS,
    _collect_session_data,
    _reload_limit_bytes,
    _reload_refusal,
    render_session_controls,
)


class _FakeSessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class _Sidebar:
    """Records every message the save branch renders, in order."""

    def __init__(self):
        self.calls = []          # list of (method, first-arg)
        self.save_clicked = True

    def markdown(self, *a, **k):
        self.calls.append(("markdown", a[0] if a else ""))

    def button(self, label, **k):
        self.calls.append(("button", label))
        return self.save_clicked

    def download_button(self, **k):
        self.calls.append(("download_button", k.get("label", "")))
        self.download_kwargs = k

    def file_uploader(self, *a, **k):
        self.calls.append(("file_uploader", a[0] if a else ""))
        return None

    def success(self, msg, **k):
        self.calls.append(("success", msg))

    def warning(self, msg, **k):
        self.calls.append(("warning", msg))

    def error(self, msg, **k):
        self.calls.append(("error", msg))

    def info(self, msg, **k):
        self.calls.append(("info", msg))

    def texts(self, kind):
        return [msg for name, msg in self.calls if name == kind]


class _FakeStreamlit:
    """The tests/test_session_manager.py fake, plus a recording sidebar.

    Deliberately still has no `get_option`: that is the production case where
    the server config cannot be read, and `_reload_limit_bytes` must fall back
    rather than raise.
    """

    def __init__(self, state):
        self.session_state = state
        self.sidebar = _Sidebar()

    def markdown(self, *a, **k):
        pass


@pytest.fixture
def fake_st(monkeypatch):
    state = _FakeSessionState()
    fake = _FakeStreamlit(state)
    monkeypatch.setattr(session_manager, "st", fake)
    from utils import session_state as ss_module
    monkeypatch.setattr(ss_module, "st", fake)
    return fake


def _ordinary_csv_session(state):
    """500 rows x 20 columns — the dataset this PR must not change at all."""
    rng = np.random.default_rng(0)
    state["raw_data"] = pd.DataFrame(
        {f"var_{i:02d}": rng.normal(size=500) for i in range(20)}
    )
    state["random_seed"] = 42
    state["workflow_step"] = "EDA"


def _zip_of(n_members: int) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(n_members):
            zf.writestr(f"datasets/{i}.parquet", b"{}")
    return buf.getvalue()


# -- the ordinary case must be untouched ------------------------------------

def test_an_ordinary_500x20_session_is_reloadable_and_still_reads_as_success(fake_st):
    _ordinary_csv_session(fake_st.session_state)
    archive_bytes, _ = _collect_session_data()

    assert _reload_refusal(archive_bytes) is None

    render_session_controls()
    successes = fake_st.sidebar.texts("success")
    assert len(successes) == 1
    assert successes[0].startswith("✅ **Session Ready for Download!**")
    # No scare text on a file that reloads fine, and the button's promise is
    # kept verbatim where it is still true — this path must be a strict no-op.
    assert fake_st.sidebar.texts("error") == []
    assert fake_st.sidebar.texts("warning") == []
    assert fake_st.sidebar.download_kwargs["help"] == (
        "Save this file to resume your work later"
    )


# -- the three gates save used to ignore ------------------------------------

def test_an_archive_over_the_upload_limit_is_named_with_its_size_and_the_limit():
    limit = _reload_limit_bytes()
    refusal = _reload_refusal(b"\x00" * (limit + 1))
    assert refusal is not None
    assert f"{limit // (1024 * 1024)} MB" in refusal      # the limit
    assert "MB" in refusal.split(",")[0]                  # the actual size


def test_an_archive_that_passes_the_size_check_still_fails_on_the_member_cap():
    """The case a single length check misses: small file, too many members."""
    archive = _zip_of(_MAX_MEMBERS + 5)
    assert len(archive) < _reload_limit_bytes()           # size alone says fine
    refusal = _reload_refusal(archive)
    assert refusal is not None
    assert str(_MAX_MEMBERS) in refusal


def test_an_archive_at_the_member_cap_is_not_refused():
    assert _reload_refusal(_zip_of(_MAX_MEMBERS)) is None


# -- the effective limit ----------------------------------------------------

def test_the_effective_limit_is_the_smaller_of_the_two_ceilings(fake_st, monkeypatch):
    monkeypatch.setattr(fake_st, "get_option", lambda name: 1, raising=False)
    assert _reload_limit_bytes() == 1 * 1024 * 1024

    monkeypatch.setattr(fake_st, "get_option", lambda name: 2000, raising=False)
    assert _reload_limit_bytes() == _MAX_UPLOAD_BYTES


def test_an_unreadable_config_falls_back_to_the_app_cap_not_to_unlimited(fake_st):
    # fake_st has no get_option at all; an unverifiable ceiling must never be
    # reported as a higher one.
    assert not hasattr(fake_st, "get_option")
    assert _reload_limit_bytes() == _MAX_UPLOAD_BYTES


# -- what the user is told, and still given ---------------------------------

def test_an_unreloadable_save_says_so_and_still_offers_the_download(fake_st, monkeypatch):
    _ordinary_csv_session(fake_st.session_state)
    monkeypatch.setattr(
        session_manager, "_reload_refusal",
        lambda b: "It is 412.0 MB, over this deployment's 100 MB limit on uploads.",
    )

    render_session_controls()
    sb = fake_st.sidebar

    # Never a green success on a file that cannot come back.
    assert sb.texts("success") == []
    errors = sb.texts("error")
    assert len(errors) == 1
    assert "cannot be loaded back" in errors[0]
    assert "412.0 MB" in errors[0] and "100 MB" in errors[0]

    # The archive is still handed over: it is the only copy that exists.
    assert [name for name, _ in sb.calls].count("download_button") == 1
    assert sb.download_kwargs["data"]
    # ...and the button no longer promises a resume it cannot deliver.
    assert "resume" not in sb.download_kwargs["help"].lower()
