"""Retrieval as a service: what the end-to-end pipeline calls.

Python:
    from askiisc_retrieval.service import build_retriever
    r = build_retriever()                       # backend from ASKIISC_BACKEND (stub | ingestion)
    res = r.retrieve("Who teaches DS 252 this semester?", ["public"])

HTTP (python -m askiisc_retrieval.cli serve --port 8090):
    POST /retrieve   {"question": "...", "scopes": ["public"], "context_chars": 6000}
                     -> passages, citations [S1].., context block for the prompt, abstention signal
    POST /verify     {"answer": "... [S1] ...", "markers": ["S1","S2"]} -> used / unknown / cited
    GET  /health     -> backend, counts, active snapshot, graph state, reranker, threshold
"""
from __future__ import annotations

import logging
import sys
import time

from . import config

log = logging.getLogger("askiisc.service")


def _embedder():
    import os
    # ASKIISC_EMBEDDER, else the ingestion .env's EMBEDDER (tei | local), else tei for the ingestion backend
    choice = (os.environ.get("ASKIISC_EMBEDDER") or os.environ.get("EMBEDDER")
              or ("tei" if config.BACKEND == "ingestion" else "local")).lower()
    if choice == "hash":
        from .stores import HashEmbedder
        return HashEmbedder(dim=config.EMBED_DIM, model_id=config.EMBED_MODEL)
    if choice == "tei":
        from .stores import TEIEmbedder
        print(f"Embedder: TEI at {config.TEI_URL}", file=sys.stderr)
        return TEIEmbedder(config.TEI_URL, config.EMBED_MODEL)
    from .stores import BGEM3Embedder
    print("Loading BGE-M3 locally ...", file=sys.stderr)
    return BGEM3Embedder(config.EMBED_MODEL, config.EMBED_MAX_SEQ, config.EMBED_BATCH)


def _registry():
    if not config.PG_DSN:
        print("Registry: none (ASKIISC_PG_DSN unset) -> active versions from Qdrant flags, "
              "chunk ids = Qdrant point ids (graph evidence may not resolve)", file=sys.stderr)
        return None
    from .ingestion_adapter import PostgresRegistry
    try:
        reg = PostgresRegistry(config.PG_DSN)
        reg.active_version_ids()
        print("Registry: postgres", file=sys.stderr)
        return reg
    except Exception as e:
        print(f"Registry: unavailable ({type(e).__name__}: {e}) -> Qdrant flags", file=sys.stderr)
        return None


def _rule_supplement(store):
    """Fallback for graphs without research/lab edges (ingestion v2): edges derived from the chunks
    (graph_rules), used ONLY when Neo4j has no WORKS_ON/HEADS. ASKIISC_GRAPH_RULES=off disables it."""
    import os
    if store is None or os.environ.get("ASKIISC_GRAPH_RULES", "supplement").lower() == "off":
        return None
    from .graph_rules import build_edges
    return lambda: build_edges(store.chunks())


def _graph(store=None):
    if not config.NEO4J_PASSWORD:
        print("Graph: none (NEO4J_PASSWORD unset) -> vector-only retrieval", file=sys.stderr)
        return None
    from .graph_neo4j import Neo4jGraphStore, neo4j_fetcher
    try:
        fetch = neo4j_fetcher(config.NEO4J_URI.replace("://localhost", "://127.0.0.1"),
                              config.NEO4J_USER, config.NEO4J_PASSWORD)
    except Exception as e:                                    # driver missing / bad URI
        print(f"Graph: unavailable ({type(e).__name__}: {e})", file=sys.stderr)
        return None
    def linked_edges():
        from .link_graph import build_link_edges

        chunks = store.chunks()
        return build_link_edges(chunks) if any(c.links or c.contacts for c in chunks) else []

    g = Neo4jGraphStore(fetch, ttl_s=config.GRAPH_TTL_S, supplement=_rule_supplement(store),
                        evidence_chunks=store.get if store is not None else None,
                        linked_edges=linked_edges if store is not None else None)
    print(f"Graph: neo4j {config.NEO4J_URI} {g.stats or g.error}", file=sys.stderr)
    return g


def build_ingestion_retriever(embedder=None, reranker="default", store=None, registry="default", graph="default"):
    """Retriever over the ingestion team's stores. Every argument can be injected (tests)."""
    from .ingestion_adapter import IngestionQdrantStore, IngestionSnapshotProvider
    from .retriever import Retriever
    registry = _registry() if registry == "default" else registry
    if store is None:
        url = config.INGEST_QDRANT_URL.replace("://localhost", "://127.0.0.1")
        try:
            store = IngestionQdrantStore(config.INGEST_COLLECTION, url=url, departments=config.INGEST_DEPARTMENTS,
                                         embedding_model=config.EMBED_MODEL, registry=registry)
        except Exception as e:
            sys.exit(f"Cannot read collection {config.INGEST_COLLECTION!r} at {url} ({type(e).__name__}: {e}). "
                     "Is the ingestion Qdrant container running and restored?")
        print(f"Vector store: {store.backend} | layout {store.layout} | {store.stats}", file=sys.stderr)
    emb = embedder or _embedder()
    if embedder is None:
        try:
            emb.embed_query("health check")
        except Exception as e:
            sys.exit(f"Embedder not reachable ({type(e).__name__}: {e}). For TEI check {config.TEI_URL}/health, "
                     "or set ASKIISC_EMBEDDER=local to run BGE-M3 in-process.")
    if reranker == "default":
        from .cli import make_reranker
        reranker = make_reranker()
    graph = _graph(store) if graph == "default" else graph
    snaps = IngestionSnapshotProvider(store, registry, embedding_model=emb.model_id,
                                      graph_version="neo4j" if graph is not None else None)
    r = Retriever(emb, store, snaps, graph, reranker=reranker)
    r.store = store
    return r


def build_retriever():
    from .cli import _open_stack
    r, store = _open_stack()
    r.store = store
    return r


# ---------------------------------------------------------------- HTTP
from pydantic import BaseModel, Field  # noqa: E402  (qdrant-client already depends on pydantic)


class RetrieveIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    scopes: list[str] = ["public"]
    use_graph: bool = True
    use_rerank: bool = True
    include_text: bool = True
    context_chars: int | None = 6000            # prompt budget; lowest-ranked sources dropped first


class VerifyIn(BaseModel):
    answer: str
    markers: list[str]                          # the citation markers /retrieve returned


def create_app(retriever=None, service_key=None):
    import os
    import hmac
    from fastapi import FastAPI, HTTPException, Header
    service_key = service_key or os.environ.get("RETRIEVAL_SERVICE_KEY")

    from .citations import MARKER_RE, format_context, render_sources, result_to_dict
    from .contract import SCOPES

    app = FastAPI(title="Ask IISc retrieval", version="1.0")
    state = {"r": retriever}

    def r():
        if state["r"] is None:
            state["r"] = build_retriever()
        return state["r"]

    @app.post("/retrieve")
    def retrieve(body: RetrieveIn, authorization: str | None = Header(default=None)):
        bad = [s for s in body.scopes if s not in SCOPES]
        if bad:
            raise HTTPException(422, f"unknown scopes {bad}; allowed {list(SCOPES)}")
        if set(body.scopes) - {"public"}:
            if not service_key or not hmac.compare_digest((authorization or "").encode("utf-8"), ("Bearer " + service_key).encode("utf-8")):
                raise HTTPException(403, "Restricted retrieval requires the gateway service credential")
        t = time.perf_counter()
        try:
            res = r().retrieve(body.question, body.scopes, use_graph=body.use_graph, use_rerank=body.use_rerank)
        except Exception as e:                  # stores down: say so, never a partial silent answer
            log.exception("retrieve failed")
            raise HTTPException(503, "Retrieval is temporarily unavailable")
        out = result_to_dict(res, include_text=body.include_text)
        out["context"] = format_context(res, body.context_chars)
        out["sources_text"] = render_sources(res)
        out["timings_ms"]["service"] = round((time.perf_counter() - t) * 1000, 1)
        return out

    @app.post("/verify")
    def verify(body: VerifyIn):
        valid = set(body.markers)
        used, unknown = [], []
        for m in MARKER_RE.findall(body.answer):
            bucket = used if m in valid else unknown
            if m not in bucket:
                bucket.append(m)
        return {"used": used, "unknown": unknown, "cited": bool(used), "ok": bool(used) and not unknown}

    @app.get("/health")
    def health():
        rt = r()
        info = {"backend": config.BACKEND, "rerank_model": getattr(rt.reranker, "model_id", None),
                "rerank_min": rt.rerank_min, "embedding_model": rt.embedder.model_id}
        try:
            snap = rt.snapshots.active()
            info["snapshot"] = {"id": snap.snapshot_id, "active_versions": len(snap.active_version_ids),
                                "source": getattr(rt.snapshots, "source", "file")}
            info["chunks"] = rt.store.count()
            info["store"] = getattr(rt.store, "backend", "?")
            if hasattr(rt.store, "stats"):
                info["adapter"] = vars(rt.store.stats)
            info["status"] = "ok"
        except Exception as e:
            info["status"], info["error"] = "down", f"{type(e).__name__}: {e}"
        g = rt.graph
        info["graph"] = None if g is None else {"edges": len(g.edges), **getattr(g, "stats", {}),
                                                "error": getattr(g, "error", None)}
        return info

    return app
