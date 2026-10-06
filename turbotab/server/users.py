"""Server-mode accounts: the users file and its admin command.

    python -m turbotab.server.users add <name>       # asks for the password twice
    python -m turbotab.server.users passwd <name>
    python -m turbotab.server.users remove <name>
    python -m turbotab.server.users list
    ... add <name> --password-stdin                 # scripts and CI: the password on stdin

The file is ``$TURBOTAB_USERS`` (default ``$TURBOTAB_HOME/users.toml``), or ``--file``. It is TOML,
one table per account, holding a ``hashlib.scrypt`` hash with its own random salt and the scrypt
cost it was made with, so the cost can rise later without breaking old hashes::

    [users."alice"]
    hash = "scrypt$n=131072,r=8,p=1$<salt, base64>$<key, base64>"
    created = "2026-10-05T12:00:00+00:00"

A username names a folder (``$TURBOTAB_HOME/users/<name>/``), so it is checked before it is used
anywhere: see :func:`check_username`. The server re-reads the file when it changes, so an account
added, removed or given a new password takes effect without a restart; removing an account or
changing its password ends that account's sessions.
"""
from __future__ import annotations

import argparse
import base64
import functools
import getpass
import hashlib
import hmac
import logging
import os
import re
import secrets
import sys
import tempfile
import threading
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

# OWASP's scrypt floor (Password Storage Cheat Sheet): N = 2^17, r = 8, p = 1, which is 128 MiB per
# hash. ``MAX_CONCURRENT_HASHES`` bounds how much of that a burst of sign-ins can claim at once.
SCRYPT_N = 2**17
SCRYPT_R = 8
SCRYPT_P = 1
KEY_BYTES = 32
SALT_BYTES = 16
MAX_CONCURRENT_HASHES = 2
MIN_PASSWORD_LENGTH = 12

# Lowercase so two names never share a folder on a case-insensitive disk (macOS, Windows); letters,
# digits and . _ @ - so an institutional id (jdoe, jdoe@univ.edu) fits; it starts with a letter or
# a digit, never holds "..", and never ends in a dot (Windows drops it). No separator, so it can
# never step out of users/.
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._@-]{0,63}$")
RESERVED = frozenset({"con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)),
                      *(f"lpt{i}" for i in range(10))})


class InvalidUsername(ValueError):
    pass


def check_username(name: str) -> str:
    """``name`` when it can name an account folder; else :class:`InvalidUsername`, saying why."""
    if not isinstance(name, str) or not USERNAME_RE.fullmatch(name):
        raise InvalidUsername(
            f"{name!r} cannot be a TurboTab username: use 1 to 64 lowercase letters, digits and "
            ". _ @ -, starting with a letter or a digit.")
    if ".." in name or name.endswith(".") or name.split(".")[0] in RESERVED:
        raise InvalidUsername(f"{name!r} cannot be a TurboTab username: it could not name a folder "
                              "safely on every system.")
    return name


# ── hashing ──────────────────────────────────────────────────────────────────

_hashing = threading.BoundedSemaphore(MAX_CONCURRENT_HASHES)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text + "=" * (-len(text) % 4), validate=True)


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    with _hashing:
        return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
                              maxmem=256 * r * n + (1 << 20), dklen=KEY_BYTES)


def hash_password(password: str, *, n: int | None = None, r: int | None = None,
                  p: int | None = None) -> str:
    """A new salt and the scrypt key, with the cost written into the result (default: the
    module's ``SCRYPT_N``, ``SCRYPT_R``, ``SCRYPT_P``)."""
    n, r, p = n or SCRYPT_N, r or SCRYPT_R, p or SCRYPT_P
    salt = secrets.token_bytes(SALT_BYTES)
    key = _scrypt(password, salt, n, r, p)
    return f"scrypt$n={n},r={r},p={p}${_b64(salt)}${_b64(key)}"


def _parse(stored: str) -> tuple[int, int, int, bytes, bytes]:
    scheme, params, salt, key = stored.split("$")
    if scheme != "scrypt":
        raise ValueError(f"unknown hash scheme {scheme!r}")
    values = dict(part.split("=", 1) for part in params.split(","))
    return int(values["n"]), int(values["r"]), int(values["p"]), _unb64(salt), _unb64(key)


def verify_password(password: str, stored: str) -> bool:
    try:
        n, r, p, salt, key = _parse(stored)
        if n > 2**20 or r > 32 or p > 16:  # a damaged or hostile entry must not exhaust memory
            return False
        return hmac.compare_digest(_scrypt(password, salt, n, r, p), key)
    except (ValueError, KeyError, MemoryError):
        return False


@functools.lru_cache(maxsize=1)
def _decoy() -> str:
    return hash_password(secrets.token_urlsafe(16))


def burn_time(password: str) -> None:
    """Check ``password`` against a hash no account has: an unknown name then costs what a wrong
    password does, so the time a sign-in takes does not say which accounts exist."""
    verify_password(password, _decoy())


# ── the file ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Account:
    name: str
    hash: str
    created: str = ""


def default_path(environ: dict[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    explicit = (env.get("TURBOTAB_USERS") or "").strip()
    if explicit:
        return Path(explicit).expanduser().absolute()
    home = Path(env.get("TURBOTAB_HOME") or "~/.turbotab").expanduser().absolute()
    return home / "users.toml"


def read_accounts(path: Path) -> dict[str, Account]:
    """The accounts in ``path``; an absent file has none. A name that is not a valid username is
    skipped (it could never sign in: the gate checks every name)."""
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        return {}
    accounts: dict[str, Account] = {}
    for name, entry in (data.get("users") or {}).items():
        if not isinstance(entry, dict) or not isinstance(entry.get("hash"), str):
            continue
        try:
            check_username(name)
        except InvalidUsername:
            continue
        accounts[name] = Account(name, entry["hash"], str(entry.get("created") or ""))
    return accounts


def _toml_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_accounts(path: Path, accounts: Iterable[Account]) -> None:
    """Rewrite ``path`` atomically, readable by its owner only (it holds password hashes)."""
    lines = ["# TurboTab accounts (server mode). Edit with: python -m turbotab.server.users",
             "# add | remove | list | passwd. Each hash is scrypt with its own salt.", ""]
    for account in sorted(accounts, key=lambda a: a.name):
        lines += [f"[users.{_toml_string(account.name)}]",
                  f"hash = {_toml_string(account.hash)}",
                  f"created = {_toml_string(account.created)}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class UsersFile:
    """The users file as the server reads it: re-read when it changes, checked at most every
    ``recheck`` seconds."""

    def __init__(self, path: Path, *, recheck: float = 2.0):
        self.path = Path(path)
        self.recheck = recheck
        self._lock = threading.Lock()
        self._stamp: tuple[int, int, int] | None = None
        self._checked = float("-inf")
        self._accounts: dict[str, Account] = {}

    def accounts(self) -> dict[str, Account]:
        now = time.monotonic()
        with self._lock:
            if now - self._checked < self.recheck:
                return self._accounts
            self._checked = now
            try:
                st = self.path.stat()
                stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
            except FileNotFoundError:
                stamp = None
            if stamp != self._stamp or stamp is None:
                try:
                    self._accounts = read_accounts(self.path) if stamp else {}
                except (OSError, ValueError) as exc:  # tomllib.TOMLDecodeError is a ValueError
                    # Fail closed: no one signs in on a file that cannot be read, and it says why.
                    log.error("cannot read the users file %s (%s); no one can sign in until it "
                              "is fixed", self.path, exc)
                    self._accounts = {}
                self._stamp = stamp
            return self._accounts

    def get(self, name: str) -> Account | None:
        return self.accounts().get(name)


# ── the command ──────────────────────────────────────────────────────────────


def _read_password(from_stdin: bool, name: str) -> str:
    if from_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = getpass.getpass(f"Password for {name}: ")
        if getpass.getpass("Again, to confirm: ") != password:
            raise SystemExit("The two passwords differ; nothing was changed.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise SystemExit(f"A password needs at least {MIN_PASSWORD_LENGTH} characters; nothing was "
                         "changed.")
    return password


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m turbotab.server.users",
        description="Manage the accounts of a TurboTab server (server mode).")
    parser.add_argument("--file", type=Path, default=None,
                        help="the users file (default: $TURBOTAB_USERS, else "
                             "$TURBOTAB_HOME/users.toml)")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, text in (("add", "create an account"), ("passwd", "set an account's password")):
        sub = commands.add_parser(command, help=text)
        sub.add_argument("name")
        sub.add_argument("--password-stdin", action="store_true",
                         help="read the password from the first line of standard input")
    commands.add_parser("remove", help="delete an account (its projects stay on disk)") \
        .add_argument("name")
    commands.add_parser("list", help="list the accounts")
    args = parser.parse_args(argv)

    path = (args.file.expanduser().absolute() if args.file else default_path())
    try:
        accounts = read_accounts(path)
    except (OSError, ValueError) as exc:
        print(f"Cannot read {path}: {exc}", file=sys.stderr)
        return 1

    if args.command == "list":
        if not accounts:
            print(f"No accounts in {path}.")
        for account in sorted(accounts.values(), key=lambda a: a.name):
            print(f"{account.name}\t{account.created or '-'}")
        return 0

    try:
        name = check_username(args.name)
    except InvalidUsername as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.command == "add":
        if name in accounts:
            print(f"{name} already has an account; use passwd to change its password.",
                  file=sys.stderr)
            return 1
        password = _read_password(args.password_stdin, name)
        created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        accounts[name] = Account(name, hash_password(password), created)
        write_accounts(path, accounts.values())
        print(f"Added {name} to {path}.")
        return 0

    if name not in accounts:
        print(f"There is no account named {name} in {path}.", file=sys.stderr)
        return 1
    if args.command == "passwd":
        password = _read_password(args.password_stdin, name)
        old = accounts[name]
        accounts[name] = Account(name, hash_password(password), old.created)
        write_accounts(path, accounts.values())
        print(f"Changed the password of {name}; its open sessions have ended.")
        return 0
    # remove
    del accounts[name]
    write_accounts(path, accounts.values())
    print(f"Removed {name} from {path}; its sessions have ended. Its projects stay in "
          f"<TURBOTAB_HOME>/users/{name}/ until you delete that folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
