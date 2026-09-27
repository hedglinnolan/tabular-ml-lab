"""Uploads, streamed to disk as they arrive.

The multipart body is parsed chunk by chunk (python-multipart's streaming
parser) and the file part is written straight into ``uploads/<token>/``, hashed
on the way, so neither the server's memory nor a second temporary copy ever
holds the whole file. There is no size cap in local mode (BLUEPRINT §6).
"""
from __future__ import annotations

import hashlib
import re
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from python_multipart.exceptions import FormParserError, ParseError
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.requests import Request

from turbotab.core.datastore import _source_kind as source_kind
from turbotab.server.errors import ApiError

FIELD = "file"
_UNSAFE = re.compile(r"[^A-Za-z0-9._()+ -]+")


@dataclass(frozen=True)
class StagedUpload:
    path: Path
    client_name: str
    fingerprint: str
    size: int


def client_file_name(raw: str) -> str:
    """The file's own name, without any folders a browser may have sent."""
    return raw.replace("\\", "/").rsplit("/", 1)[-1].strip()


def safe_file_name(name: str) -> str:
    """A name that is safe on every file system and keeps the type suffix."""
    cleaned = _UNSAFE.sub("_", name).strip(" .") or "upload"
    if len(cleaned) > 180:
        stem, dot, suffix = cleaned.rpartition(".")
        cleaned = (stem[: 170] + dot + suffix[:9]) if dot else cleaned[:180]
    return cleaned


class _Receiver:
    def __init__(self, staging: Path):
        self.staging = staging
        self.header_field = bytearray()
        self.header_value = bytearray()
        self.headers: dict[str, bytes] = {}
        self.out: BinaryIO | None = None
        self.path: Path | None = None
        self.client_name = ""
        self.digest = hashlib.blake2b(digest_size=16)  # the same hash as datastore.fingerprint_file
        self.size = 0
        self.done = False

    def callbacks(self) -> dict[str, Any]:
        return {
            "on_part_begin": self.part_begin,
            "on_header_field": self.header_field_data,
            "on_header_value": self.header_value_data,
            "on_header_end": self.header_end,
            "on_headers_finished": self.headers_finished,
            "on_part_data": self.part_data,
            "on_part_end": self.part_end,
        }

    def part_begin(self) -> None:
        self.headers = {}

    def header_field_data(self, data: bytes, start: int, end: int) -> None:
        self.header_field += data[start:end]

    def header_value_data(self, data: bytes, start: int, end: int) -> None:
        self.header_value += data[start:end]

    def header_end(self) -> None:
        self.headers[bytes(self.header_field).decode("latin-1").lower()] = bytes(self.header_value)
        self.header_field.clear()
        self.header_value.clear()

    def headers_finished(self) -> None:
        if self.done or self.out is not None:
            return
        _, options = parse_options_header(self.headers.get("content-disposition"))
        if options.get(b"name", b"").decode("utf-8", "replace") != FIELD:
            return
        raw_name = options.get(b"filename")
        if raw_name is None:
            raise ApiError(400, "not_a_file", f"The {FIELD!r} field must be a file.")
        name = client_file_name(raw_name.decode("utf-8", "replace"))
        safe = safe_file_name(name)
        try:
            source_kind(Path(safe))
        except ValueError as exc:
            raise ApiError(400, "unsupported_file", str(exc)) from None
        self.staging.mkdir(parents=True, exist_ok=True)
        self.client_name = name or safe
        self.path = self.staging / safe
        self.out = open(self.path, "wb")

    def part_data(self, data: bytes, start: int, end: int) -> None:
        if self.out is None:
            return
        chunk = data[start:end]
        self.out.write(chunk)
        self.digest.update(chunk)
        self.size += len(chunk)

    def part_end(self) -> None:
        if self.out is not None:
            self.out.close()
            self.out = None
            self.done = True

    def discard(self) -> None:
        if self.out is not None:
            self.out.close()
            self.out = None
        shutil.rmtree(self.staging, ignore_errors=True)


async def receive_upload(request: Request, uploads_dir: Path) -> StagedUpload:
    content_type, params = parse_options_header(request.headers.get("content-type"))
    boundary = params.get(b"boundary")
    if content_type != b"multipart/form-data" or not boundary:
        raise ApiError(400, "not_multipart", f"Send the file as multipart/form-data, in a field named {FIELD!r}.")
    receiver = _Receiver(uploads_dir / secrets.token_hex(8))
    parser = MultipartParser(boundary, callbacks=receiver.callbacks())
    try:
        try:
            async for chunk in request.stream():
                if chunk:
                    parser.write(chunk)
            parser.finalize()
        except (FormParserError, ParseError) as exc:
            raise ApiError(400, "bad_upload", f"The upload could not be read as multipart/form-data: {exc}") from None
        if not receiver.done or receiver.path is None:
            raise ApiError(400, "no_file", f"The upload has no file in a field named {FIELD!r}.")
        if receiver.size == 0:
            raise ApiError(400, "empty_file", f"{receiver.client_name!r} is empty.")
    except BaseException:
        receiver.discard()
        raise
    return StagedUpload(receiver.path, receiver.client_name, receiver.digest.hexdigest(), receiver.size)
