"""The active snapshot: which versions are queryable, under which embedding model and graph version.
On the laptop it is a JSON file written by the stub ingestion. In deployment, swap in a provider that
reads the PostgreSQL manifest (FR-4); the retriever only calls `active()`."""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    embedding_model: str
    graph_version: str | None
    active_version_ids: frozenset[str]
    departments: tuple[str, ...] = ()
    graph_sha256: str | None = None      # the graph file this snapshot was built with (FR-4: activate together)


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temporary file in the same folder and os.replace, so a reader sees the whole old file or
    the whole new one, never a truncated one (FR-4). The text's own "\n" is written on every OS, so the
    bytes on disk (and their hash) are the same on Windows and Linux."""
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomic replace of `path` with `data` (temporary file in the same folder, fsync, os.replace)."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        deadline = time.monotonic() + 10
        while True:
            try:
                os.replace(tmp, path)
                break
            except PermissionError:               # Windows: a reader has the old file open for a moment
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.01)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class SnapshotProvider(Protocol):
    def active(self) -> Snapshot: ...


class JsonSnapshotProvider:
    def __init__(self, path: Path):
        self.path = path
        self._cached: tuple[tuple, Snapshot] | None = None

    def active(self) -> Snapshot:
        # Re-read only when the file has been replaced (new inode, size or mtime). This keeps the file
        # closed almost all the time, so an atomic replace is not blocked by readers on Windows.
        deadline = time.monotonic() + 2
        while True:
            try:
                st = os.stat(self.path)
                key = (st.st_ino, st.st_size, st.st_mtime_ns)
                if self._cached and self._cached[0] == key:
                    return self._cached[1]
                data = Path(self.path).read_bytes()
                break
            except PermissionError:
                # Windows refuses to open a file for the instant it is being replaced; the swap is atomic,
                # so wait for it rather than failing the request.
                if time.monotonic() > deadline:
                    if self._cached:
                        return self._cached[1]
                    raise
                time.sleep(0.005)
        d = json.loads(data.decode("utf-8"))
        snap = Snapshot(d["snapshot_id"], d["embedding_model"], d.get("graph_version"),
                        frozenset(d["active_version_ids"]), tuple(d.get("departments", ())), d.get("graph_sha256"))
        self._cached = (key, snap)
        return snap

    @staticmethod
    def write(path: Path, snap: Snapshot) -> None:
        body = {"snapshot_id": snap.snapshot_id, "embedding_model": snap.embedding_model,
                "graph_version": snap.graph_version, "active_version_ids": sorted(snap.active_version_ids),
                "departments": list(snap.departments)}
        if snap.graph_sha256:
            body["graph_sha256"] = snap.graph_sha256
        atomic_write_text(path, json.dumps(body, indent=2))


class StaticSnapshotProvider:
    def __init__(self, snap: Snapshot):
        self.snap = snap

    def active(self) -> Snapshot:
        return self.snap
