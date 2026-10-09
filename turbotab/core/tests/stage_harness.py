"""Run a stage function on a real file, the way a worker does, without the engine.

The file is ingested to Parquet (the datastore's own path), and the stage gets a StageContext with
the ingest artifact as its input. Used by the voice tests; cheap enough to call per fixture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from turbotab.core.datastore import DataStore, ingest
from turbotab.core.decisions import ProjectState
from turbotab.core.graph import StageContext

REPO = Path(__file__).resolve().parents[3]
SAMPLES = REPO / "turbotab" / "sample_data"


def _main_checkout(repo: Path) -> Path:
    """The main checkout when ``repo`` is a git worktree (its ``.git`` is a file), else ``repo``."""
    dot_git = repo / ".git"
    if dot_git.is_file():
        text = dot_git.read_text("utf-8").strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text.split(":", 1)[1].strip())
            if gitdir.parent.name == "worktrees":
                return gitdir.parent.parent.parent
    return repo


NHANES_GZ = Path(__file__).resolve().parent / "fixtures" / "nhanes.csv.gz"
NHANES_NAME = "_tt_tmp_nhanes.csv"  # the export's own name, which truths.FIXTURE_TRUTHS keys on
NHANES_SHA256 = "dbb2df487d50de26a223c9b7f9f214b4a55b0399db72b2047e729920d04a15da"  # decompressed


def nhanes_fixture() -> Path:
    """The tracked fixture, decompressed to a cache keyed by its bytes, under the export's own
    name (so names, uploads and captures read as they did).

    The fixture is decompressed in memory first, so a corrupt one fails before the cache is
    touched. A cached copy is reused only when its bytes are the fixture's (a partial write or an
    edit is written again). A new copy is written to a temporary name, made readable by every
    user (0644: the cache sits in the shared temporary folder, and ``mkstemp`` creates 0600) and
    moved into place, so parallel workers never read half a file; a write that fails removes its
    temporary file. Where another user's folder can be neither read nor written, the copy goes to
    a folder of this user's own."""
    import getpass
    import gzip
    import hashlib
    import os
    import stat
    import tempfile

    packed = NHANES_GZ.read_bytes()
    data = gzip.decompress(packed)
    digest = hashlib.sha256(data).hexdigest()
    shared = Path(tempfile.gettempdir()) / f"turbotab-nhanes-{hashlib.sha256(packed).hexdigest()[:16]}"
    for folder in (shared, shared.with_name(f"{shared.name}-{getpass.getuser()}")):
        out = folder / NHANES_NAME
        try:
            if out.is_file() and hashlib.sha256(out.read_bytes()).hexdigest() == digest:
                mine = not hasattr(os, "getuid") or out.stat().st_uid == os.getuid()
                if mine and stat.S_IMODE(out.stat().st_mode) != 0o644:
                    os.chmod(out, 0o644)  # a copy an earlier harness wrote 0600
                return out
            folder.mkdir(parents=True, exist_ok=True)
            fd, part = tempfile.mkstemp(dir=folder, suffix=".part")
        except PermissionError:
            continue  # another user's folder: this user's own
        try:
            with os.fdopen(fd, "wb") as dst:
                dst.write(data)
            os.chmod(part, 0o644)
            os.replace(part, out)
        except BaseException:
            if os.path.exists(part):
                os.unlink(part)
            raise
        return out
    raise PermissionError(f"Neither {shared} nor this user's own folder beside it can be written.")


def _nhanes() -> Path:
    """The real NHANES export: $TURBOTAB_NHANES_CSV, else the tracked fixture, else the untracked
    file in this checkout or the main one."""
    import os

    given = os.environ.get("TURBOTAB_NHANES_CSV")
    if given:
        return Path(given)
    if NHANES_GZ.is_file():
        return nhanes_fixture()
    here = REPO / NHANES_NAME
    return here if here.is_file() else _main_checkout(REPO) / NHANES_NAME


NHANES = _nhanes()
BUDGET = 2 << 30
LENSES = ("metabolomics", "genomics", "dietary", "clinical", "survey")


class Ingested:
    def __init__(self, source: Path, folder: Path):
        self.source = Path(source)
        self.parquet = folder / "raw.parquet"
        self.info = ingest(self.source, self.parquet).to_dict()

    def store(self) -> DataStore:
        return DataStore(self.parquet, BUDGET)

    def frame(self, columns: list[str] | None = None) -> Any:
        with self.store() as store:
            return store.materialize(columns)

    def run(self, fn: Callable[[StageContext], Any], state: ProjectState,
            inputs: dict[str, Any] | None = None) -> Any:
        ctx = StageContext(
            project_id="test",
            state=state,
            inputs={"ingest": self.info, **(inputs or {})},
            paths={"data": str(self.parquet), "source": str(self.source)},
            settings={"memory_budget_bytes": BUDGET},
        )
        return fn(ctx)

    def lenses_that_fit(self) -> list[str]:
        """The lenses the profile stage hints for this table (every lens when none is hinted):
        ``turbotab.core.detectors.lenses.hints``, which serves in place of ``packs.suggest`` (WP14)."""
        from turbotab.core.detectors import lenses

        frame = self.frame().reset_index(drop=True)
        hints = [h["lens"] for h in lenses.hints(frame) if h.get("lens") in LENSES]
        return hints or list(LENSES)
