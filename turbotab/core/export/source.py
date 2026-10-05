"""What the export reads of a project (:class:`Source`), and the input files' hashes.

The export is a pure function of a :class:`Source`: the decision log, the state it folds to, the
Router's steps, each fresh stage's artifact as a client is served it, and the input files with
their hashes. The server builds one per request (``ProjectService.export_source``); the replay
builds one from the project it rebuilt in a fresh home, so both run the same code.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

Role = Literal["table", "joined file", "codebook"]
CHUNK = 1 << 20


@dataclass(frozen=True)
class InputFile:
    """One file the analysis read. ``changed``: its bytes are no longer those the project read
    (their BLAKE2b differs from the fingerprint recorded when it was read), so no record can name
    the data the results came from."""

    role: Role
    name: str
    path: str
    bytes: int
    sha256: str
    file_id: str | None = None
    changed: bool = False
    missing: bool = False  # the file is gone from where it was read

    def to_json(self) -> dict[str, Any]:
        return {"role": self.role, "name": self.name, "path": self.path, "bytes": self.bytes,
                "sha256": self.sha256, "file_id": self.file_id}


def hash_file(path: str | Path) -> tuple[str, str, int]:
    """``(sha256, blake2b-128, size)`` of a file's bytes, in one streamed pass. The BLAKE2b is the
    fingerprint the datastore records when it reads a file (``datastore.fingerprint_file``)."""
    sha = hashlib.sha256()
    blake = hashlib.blake2b(digest_size=16)
    size = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            sha.update(chunk)
            blake.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), blake.hexdigest(), size


def input_file(role: Role, name: str, path: str | Path, fingerprint: str | None,
               file_id: str | None = None) -> InputFile:
    """The input at ``path``, hashed; ``changed`` when ``fingerprint`` (as recorded when the project
    read it) is known and its bytes no longer match it."""
    p = Path(path)
    if not p.is_file():
        return InputFile(role=role, name=name, path=str(p), bytes=0, sha256="", file_id=file_id,
                         missing=True)
    sha, blake, size = hash_file(p)
    return InputFile(role=role, name=name, path=str(p), bytes=size, sha256=sha, file_id=file_id,
                     changed=bool(fingerprint) and blake != fingerprint)


@dataclass
class Source:
    """A project as the export reads it.

    ``artifact(stage)``: the fresh stage's public data exactly as a client is served it (held-out
    scores withheld until opened, estimates withheld until the questions they rest on are
    answered), else None. ``bundle(stage)``: the fresh stage's whole cached artifact (its frames),
    else None. ``statuses``: each stage's status (``graph.StageStatus``)."""

    name: str
    engine_version: str
    records: list[Any]
    state: Any
    interview: list[Any]
    statuses: Mapping[str, Any]
    artifact: Callable[[str], Any]
    bundle: Callable[[str], Any]
    methods: Any  # provenance.MethodsText
    inputs: list[InputFile]
    decisions_jsonl: bytes
    stage_versions: dict[str, int] = field(default_factory=dict)

    def status(self, stage: str) -> Any:
        return self.statuses.get(stage)

    @property
    def purpose(self) -> str | None:
        return getattr(self.state, "purpose", None)


__all__ = ["InputFile", "Source", "hash_file", "input_file"]
