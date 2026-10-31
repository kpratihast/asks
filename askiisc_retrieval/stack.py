"""Builds a ready Retriever for either backend, from environment variables.

  ASKIISC_BACKEND   stub (default: laptop stand-in, data/ folder) | ingestion (the team's stores)

ingestion backend (defaults match the ingestion manual of 2026-10-22):
  QDRANT_URL          http://127.0.0.1:6333
  ASKIISC_COLLECTION  iisc_policy_passages
  ASKIISC_DEPARTMENT  CDS            ("" = no department filter)
  TEI_URL             http://127.0.0.1:8080   ("" = load BGE-M3 in-process instead)
  NEO4J_URI           bolt://127.0.0.1:7687
  NEO4J_USER          neo4j
  NEO4J_PASSWORD      (required for the graph; without it results are vector-only, marked degraded)
Shared: ASKIISC_RERANKER, ASKIISC_RERANK_MODEL, ASKIISC_RERANK_MIN, ASKIISC_PEOPLE_K (see config.py).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from . import config


@dataclass
class Stack:
    retriever: object
    store: object
    backend: str
    info: dict = field(default_factory=dict)

    def close(self):
        try:
            self.store.close()
        except Exception:
            pass


def _err(msg: str):
    print(msg, file=sys.stderr, flush=True)


def backend_name() -> str:
    return os.environ.get("ASKIISC_BACKEND", "stub").strip().lower() or "stub"


def open_stack(backend: str | None = None, embedder=None, reranker=None) -> Stack:
    backend = backend or backend_name()
    if backend == "ingestion":
        return _open_ingestion(embedder, reranker)
    if backend != "stub":
        raise ValueError(f"ASKIISC_BACKEND must be 'stub' or 'ingestion', not {backend!r}")
    from .cli import make_embedder, make_reranker, make_store
    from .retriever import Retriever
    from .snapshot import JsonSnapshotProvider
    from .stores import JsonGraphStore
    if not config.SNAPSHOT_PATH.exists():
        raise RuntimeError("No stub index yet. Run `fetch` and then `ingest` first.")
    store = make_store()
    r = Retriever(embedder or make_embedder(), store, JsonSnapshotProvider(config.SNAPSHOT_PATH),
                  JsonGraphStore.load(config.GRAPH_PATH),
                  reranker=reranker if reranker is not None else make_reranker())
    return Stack(r, store, "stub", {"vector_store": store.backend})


def _open_ingestion(embedder=None, reranker=None) -> Stack:
    from .cli import make_reranker
    from .ingestion_adapter import (IngestionQdrantStore, IngestionSnapshotProvider, UnavailableGraph,
                                    load_neo4j_graph)
    from .retriever import Retriever
    env = os.environ.get
    model = config.EMBED_MODEL
    if embedder is None:
        tei = env("TEI_URL", "http://127.0.0.1:8080")
        if tei:
            from .stores import TEIEmbedder
            embedder = TEIEmbedder(tei.replace("://localhost", "://127.0.0.1"), model)
            _err(f"Query embedder: TEI at {tei}")
        else:
            from .cli import make_embedder
            embedder = make_embedder()
    store = IngestionQdrantStore(env("QDRANT_URL", "http://127.0.0.1:6333"),
                                 env("ASKIISC_COLLECTION", "iisc_policy_passages"),
                                 env("ASKIISC_DEPARTMENT", "CDS") or None, embedder.model_id)
    info = {"vector_store": store.backend, "points": store.count(),
            "active_versions": len(store.active_version_ids)}
    _err(f"Vector store: {store.backend} ({info['points']} active chunks, {info['active_versions']} versions)")
    pwd = env("NEO4J_PASSWORD")
    if not pwd:
        graph = UnavailableGraph("NEO4J_PASSWORD not set")
        info["graph"] = "unavailable: NEO4J_PASSWORD not set"
    else:
        try:
            graph, stats = load_neo4j_graph(env("NEO4J_URI", "bolt://127.0.0.1:7687"),
                                            env("NEO4J_USER", "neo4j"), pwd, store)
            info["graph"] = f"neo4j: {len(graph.edges)} edges loaded"
            info["graph_stats"] = dict(stats)
        except Exception as e:
            graph = UnavailableGraph(f"{type(e).__name__}: {e}")
            info["graph"] = f"unavailable: {type(e).__name__}: {e}"
    _err(f"Graph: {info['graph']}")
    r = Retriever(embedder, store, IngestionSnapshotProvider(store), graph,
                  reranker=reranker if reranker is not None else make_reranker())
    return Stack(r, store, "ingestion", info)
