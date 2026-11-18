"""Versioned raw-source store (FR-3) and conditional-refresh state (FR-1).

Every fetched page is hashed on its extracted main content (PRD FR-3: "Hash every crawled page's extracted
content"). A new immutable raw snapshot and a new registry row are written only when that hash differs from
the URL's last stored version:

* unchanged page  -> zero new snapshot objects, zero new registry rows, and the working copy under
                     data/raw/ is not touched (its bytes and modification time stay the same);
* modified page   -> exactly one new snapshot object, one new registry row, the previous version is
                     marked superseded (and kept), and the manifest is updated.

Layout under data/raw/ (PRD deviation, stated in INGESTION_REFRESH.md: SQLite and local files stand in for
the PostgreSQL registry and MinIO object store; the rules are the same):

    <raw_name(url)>                      working copy read by ingestion (always the current version)
    versions/objects/<url-key>/<sha>.<ext>   immutable snapshots, one per stored version, never overwritten
    versions/registry.sqlite3            `versions` rows (one per stored version) and `sources` check state
    versions/manifest.json               current version of every URL, with its superseded versions

The `sources` table remembers each URL's ETag / Last-Modified so a refresh can send If-None-Match /
If-Modified-Since (FR-1); a 304 answer costs no download and no write.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from .snapshot import atomic_write_bytes, atomic_write_text

UNCHANGED, NEW, CHANGED = "unchanged", "new", "changed"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def content_hash(url: str, body: bytes) -> str:
    """SHA-256 of the page's extracted main content (HTML), or of the bytes (PDF / unparseable page).

    Hashing the extracted content means a change in the site template alone (navigation, a footer date,
    a session token in markup) is not a new version, while any change in the text that is indexed is."""
    if not body.startswith(b"%PDF"):
        try:
            from .parse import parse_html
            parsed = parse_html(url, body)
            text = "\n".join(f"{s.heading}\n{s.text}" for s in parsed.sections)
            if text.strip():
                return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        except Exception:                      # fall back to the bytes; extraction errors surface at ingest
            pass
    return "sha256:" + hashlib.sha256(body).hexdigest()


class RawVersions:
    def __init__(self, raw_dir: Path, raw_name: Callable[[str], str]):
        self.raw_dir = Path(raw_dir)
        self.raw_name = raw_name
        self.dir = self.raw_dir / "versions"
        self.objects = self.dir / "objects"
        self.manifest_path = self.dir / "manifest.json"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.dir / "registry.sqlite3", timeout=30, isolation_level=None)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS versions (
                url TEXT NOT NULL, version INTEGER NOT NULL, content_hash TEXT NOT NULL,
                object TEXT NOT NULL, bytes INTEGER NOT NULL, fetched_at TEXT NOT NULL,
                superseded_at TEXT, PRIMARY KEY (url, version));
            CREATE TABLE IF NOT EXISTS sources (
                url TEXT PRIMARY KEY, etag TEXT, last_modified TEXT, last_checked TEXT, last_status TEXT);
        """)
        self.changed: list[str] = []
        self.stats: Counter = Counter()        # this run: new / changed / unchanged / not_modified (304)

    def close(self) -> None:
        if self.changed:
            self.write_manifest()
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ conditional requests (FR-1)
    def conditional_headers(self, url: str) -> dict[str, str]:
        """If-None-Match / If-Modified-Since for a URL whose current version is stored (a 304 is only safe
        when the working copy exists)."""
        if not (self.raw_dir / self.raw_name(url)).is_file():
            return {}
        row = self.db.execute("SELECT etag, last_modified FROM sources WHERE url=?", (url,)).fetchone()
        headers = {}
        if row and row[0]:
            headers["If-None-Match"] = row[0]
        if row and row[1]:
            headers["If-Modified-Since"] = row[1]
        return headers

    def not_modified(self, url: str) -> str:
        """Record a 304 answer: nothing is downloaded or written except the check time."""
        self._check(url, None, None, "304")
        self.stats["not_modified"] += 1
        return UNCHANGED

    # ------------------------------------------------------------------ versioned writes (FR-3)
    def latest(self, url: str):
        return self.db.execute("SELECT version, content_hash, object FROM versions WHERE url=? "
                               "ORDER BY version DESC LIMIT 1", (url,)).fetchone()

    def record(self, url: str, body: bytes, *, etag: str | None = None, last_modified: str | None = None) -> str:
        """Store `body` as the URL's current version if its content hash changed. Returns NEW, CHANGED or
        UNCHANGED. An unchanged page writes nothing (no snapshot object, no registry row, no working copy)."""
        working = self.raw_dir / self.raw_name(url)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            last = self.latest(url)
            if last is None and working.is_file():
                # A raw copy saved before versioning existed (or by an older release): register it as
                # version 1, so the first refresh compares against it and keeps it if the page changed.
                old = working.read_bytes()
                self._insert(url, 1, content_hash(url, old), old, fetched_at=time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(working.stat().st_mtime)))
                last = self.latest(url)
            digest = content_hash(url, body)
            if last is not None and last[1] == digest:
                self._check(url, etag, last_modified, "unchanged")
                self.db.execute("COMMIT")
                self.stats[UNCHANGED] += 1
                return UNCHANGED
            version = 1 if last is None else last[0] + 1
            now = _now()
            if last is not None:
                self.db.execute("UPDATE versions SET superseded_at=? WHERE url=? AND version=?", (now, url, last[0]))
            self._insert(url, version, digest, body, fetched_at=now)
            self._check(url, etag, last_modified, "new" if last is None else "changed")
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        atomic_write_bytes(working, body)          # the working copy always holds the current version
        self.changed.append(url)
        status = NEW if last is None else CHANGED
        self.stats[status] += 1
        return status

    def _insert(self, url: str, version: int, digest: str, body: bytes, *, fetched_at: str) -> None:
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        ext = ".pdf" if body.startswith(b"%PDF") else ".html"
        rel = f"objects/{key}/{hashlib.sha256(body).hexdigest()}{ext}"
        path = self.dir / rel
        if not path.exists():                       # objects are immutable: written once, never replaced
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(path, body)
        self.db.execute("INSERT INTO versions (url, version, content_hash, object, bytes, fetched_at) "
                        "VALUES (?,?,?,?,?,?)", (url, version, digest, rel, len(body), fetched_at))

    def _check(self, url, etag, last_modified, status) -> None:
        self.db.execute(
            "INSERT INTO sources (url, etag, last_modified, last_checked, last_status) VALUES (?,?,?,?,?) "
            "ON CONFLICT(url) DO UPDATE SET etag=COALESCE(excluded.etag, sources.etag), "
            "last_modified=COALESCE(excluded.last_modified, sources.last_modified), "
            "last_checked=excluded.last_checked, last_status=excluded.last_status",
            (url, etag, last_modified, _now(), status))

    # ------------------------------------------------------------------ manifest and history
    def history(self, url: str) -> list[dict]:
        rows = self.db.execute("SELECT version, content_hash, object, bytes, fetched_at, superseded_at FROM versions "
                               "WHERE url=? ORDER BY version", (url,)).fetchall()
        return [dict(zip(("version", "content_hash", "object", "bytes", "fetched_at", "superseded_at"), r))
                for r in rows]

    def write_manifest(self) -> None:
        """One manifest of the current raw version of every URL, with the versions it superseded."""
        out = {}
        for (url,) in self.db.execute("SELECT DISTINCT url FROM versions ORDER BY url").fetchall():
            hist = self.history(url)
            out[url] = {"current": hist[-1], "superseded": hist[:-1]}
        text = json.dumps({"updated_at": _now(), "sources": out}, indent=1, ensure_ascii=False)
        if not self.manifest_path.is_file() or json.loads(self.manifest_path.read_text(encoding="utf-8"))["sources"] != out:
            atomic_write_text(self.manifest_path, text)


def write_if_changed(path: Path, text: str) -> bool:
    """Atomically replace `path` only when its content differs (an unchanged index file keeps its mtime)."""
    path = Path(path)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    atomic_write_text(path, text)
    return True
