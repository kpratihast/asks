"""The active snapshot: which versions are queryable, under which embedding model and graph version.
On the laptop it is a JSON file written by the stub ingestion. In deployment, swap in a provider that
reads the PostgreSQL manifest (FR-4); the retriever only calls `active()`."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    embedding_model: str
    graph_version: str | None
    active_version_ids: frozenset[str]


class SnapshotProvider(Protocol):
    def active(self) -> Snapshot: ...


class JsonSnapshotProvider:
    def __init__(self, path: Path):
        self.path = path

    def active(self) -> Snapshot:
        d = json.loads(self.path.read_text(encoding="utf-8"))
        return Snapshot(d["snapshot_id"], d["embedding_model"], d.get("graph_version"),
                        frozenset(d["active_version_ids"]))

    @staticmethod
    def write(path: Path, snap: Snapshot) -> None:
        path.write_text(json.dumps({"snapshot_id": snap.snapshot_id, "embedding_model": snap.embedding_model,
                                    "graph_version": snap.graph_version,
                                    "active_version_ids": sorted(snap.active_version_ids)}, indent=2),
                        encoding="utf-8")


class StaticSnapshotProvider:
    def __init__(self, snap: Snapshot):
        self.snap = snap

    def active(self) -> Snapshot:
        return self.snap
