"""Intake for the Sullivan 2021 replication: the public NHANES files, fetched from wwwn.cdc.gov.

Each file ``files.json`` names is downloaded with its codebook page into
``research/data/nhanes/<cycle>/`` (ignored by git), and ``manifest.json`` (committed) lists per
file its URL, the download date, its size and SHA-256, the dates the codebook page states
("First Published", "Last Revised") and the server's Last-Modified header. A file already on disk
is not fetched again; its manifest entry is recomputed from the bytes on disk.

No request carries an identifier: a plain user agent, no query string, no email parameter.

    scripts/py research/replication/sullivan2021/engine/intake.py [--check]

``--check`` fetches nothing and fails when a file on disk no longer matches the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DATA = REPO / "research" / "data" / "nhanes"
MANIFEST = HERE / "manifest.json"
AGENT = "Mozilla/5.0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> str | None:
    """Download ``url`` to ``dest``; returns the Last-Modified header."""
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=300) as r:
        data = r.read()
        modified = r.headers.get("Last-Modified")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return modified


def page_dates(htm: Path) -> dict[str, str | None]:
    text = htm.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str | None] = {}
    for label in ("First Published", "Last Revised"):
        m = re.search(label + r":\s*([^<]+)<", text)
        out[label.lower().replace(" ", "_")] = m.group(1).strip() if m else None
    return out


def entry(path: Path, url: str, downloaded: str | None, modified: str | None) -> dict:
    return {"file": str(path.relative_to(REPO)), "url": url, "downloaded": downloaded,
            "bytes": path.stat().st_size, "sha256": sha256(path), "last_modified_header": modified}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    spec = json.loads((HERE / "files.json").read_text())
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": []}
    known = {e["file"]: e for e in old.get("files", [])}
    out, bad = [], []
    for cycle, c in spec["cycles"].items():
        folder = DATA / cycle
        folder.mkdir(parents=True, exist_ok=True)
        for name in c["files"]:
            row: dict = {"cycle": cycle, "name": name}
            for ext in ("xpt", "htm"):
                url = spec["url"].format(year=c["year"], file=name, ext=ext)
                dest = folder / f"{name}.{ext}"
                rel = str(dest.relative_to(REPO))
                before = known.get(rel) or {}
                if args.check:
                    if not dest.exists() or sha256(dest) != before.get("sha256"):
                        bad.append(rel)
                    continue
                downloaded, modified = before.get("downloaded"), before.get("last_modified_header")
                if not dest.exists():
                    modified = fetch(url, dest)
                    downloaded = time.strftime("%Y-%m-%d")
                    time.sleep(0.5)
                row[ext] = entry(dest, url, downloaded, modified)
            if not args.check:
                row["cdc_page"] = page_dates(folder / f"{name}.htm")
                out.append(row)
                print(f"{cycle} {name}: {row['xpt']['bytes']:,} bytes", flush=True)
    if args.check:
        print("every file matches the manifest" if not bad else f"changed or missing: {bad}")
        return 1 if bad else 0
    files = [e for r in out for e in (r["xpt"], r["htm"])]
    for r in out:
        r["xpt"]["cdc_page"] = r["cdc_page"]
    MANIFEST.write_text(json.dumps({
        "about": ("The NHANES files of the Sullivan 2021 replication, as downloaded (intake.py). "
                  "`cdc_page` holds the dates the file's codebook page states; "
                  "`last_modified_header` is the server's header at download."),
        "files": files}, indent=1) + "\n")
    print(f"wrote {MANIFEST.relative_to(REPO)} ({len(files)} files, "
          f"{sum(e['bytes'] for e in files):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
