"""`retrieval.run(...)`: the serving tier's view of retrieval.

    hits, trace = await retrieval.run(question, pinned, mode, programme, academic_year, scopes=[...])

hits: one per citable source, best first:
    {"marker", "title", "section", "url", "version_id", "chunk_ids", "academic_term", "academic_year",
     "programme", "location", "freshness": {"status", "snapshot"}, "instructor", "text", "excerpt",
     "rerank_score", "via"}
trace: {"mode", "snapshot_id", "target_term", "evidence_sufficient", "rerank_top", "degraded",
        "degraded_reason", "timings_ms"}

mode: "full"   = vector + graph + rerank (the full system)
      "vector" = no graph expansion (baseline B2)
      "none"   = no retrieval (baseline B0: the model answers from its own knowledge)
pinned: a snapshot id; if given and the active snapshot differs, SnapshotMismatch (benchmarks pin
        the snapshot so every run answers from the same corpus).
Retrieval runs in-process ("local") or over HTTP against the retrieval service ("remote").
"""
from __future__ import annotations

import asyncio
import re

from askiisc_retrieval.config import current_term

MODES = ("full", "vector", "none")
INSTRUCTOR_RE = re.compile(r"Instructor(?:\(s\)|s)?\s*:\s*([^;\n]+)", re.I)
TERM_RE = re.compile(r"^(\d{4})-(AUG|JAN|SUM)$")


class SnapshotMismatch(Exception):
    pass


class RetrievalUnavailable(Exception):
    pass


def academic_year(term: str | None) -> str | None:
    """IISc academic year runs Aug-Jul: 2026-AUG, 2027-JAN, 2027-SUM -> '2026-2027'."""
    m = TERM_RE.match(term or "")
    if not m:
        return None
    y = int(m.group(1))
    start = y if m.group(2) == "AUG" else y - 1
    return f"{start}-{start + 1}"


def _term_key(t: str) -> tuple[int, int]:
    y, s = t.split("-")
    return int(y), {"JAN": 0, "SUM": 1, "AUG": 2}[s]


def hits_from_result(d: dict, now_term: str | None = None) -> list[dict]:
    """Map a retrieval result (askiisc_retrieval.citations.result_to_dict) to serving hits."""
    now_term = now_term or current_term()
    texts: dict[str, list[str]] = {}
    for p in d.get("passages", []):
        if p.get("text"):
            texts.setdefault(p["marker"], []).append(p["text"].strip())
    by_marker = {}
    for p in d.get("passages", []):
        by_marker.setdefault(p["marker"], []).append(p)
    hits = []
    for c in d.get("citations", []):
        term = c.get("academic_term")
        text = "\n".join(texts.get(c["marker"], [])) or c.get("excerpt", "")
        if term and TERM_RE.match(term) and _term_key(term) < _term_key(now_term):
            status = f"past term ({term})"
        else:
            status = "current"
        m = INSTRUCTOR_RE.search(text)
        hits.append({
            "access_scope": c.get("access_scope"), "department": c.get("department", ""),
            "marker": c["marker"], "title": c.get("title", ""), "section": c.get("section_heading", ""),
            "url": c.get("source_url", ""), "version_id": c.get("version_id"), "chunk_ids": c.get("chunk_ids", []),
            "academic_term": term, "academic_year": academic_year(term) or "not term-specific",
            "programme": c.get("programme") or "not specified",
            "location": f'{c.get("section_heading", "")} ({c.get("source_url", "")})',
            "freshness": {"status": status, "snapshot": d.get("snapshot_id")},
            "instructor": m.group(1).strip() if m else None,
            "text": text, "excerpt": c.get("excerpt", ""), "rerank_score": c.get("rerank_score"), "via": c.get("via"),
            "graph_paths": [p["graph_path"] for p in by_marker.get(c["marker"], []) if p.get("graph_path")],
            "graph_support": [p["graph_support"] for p in by_marker.get(c["marker"], []) if p.get("graph_support")],
        })
    return hits


def _filter_year(hits: list[dict], year: str | None) -> list[dict]:
    if not year:
        return hits
    return [h for h in hits if not h["academic_term"] or h["academic_year"] == year]


def _trace(d: dict, mode: str) -> dict:
    return {"mode": mode, "snapshot_id": d.get("snapshot_id"),
            "departments": d.get("departments", []), "department_status": d.get("department_status"),
            "connection_status": d.get("connection_status"), "policy_trace": d.get("policy_trace", {}), "graph_proofs": d.get("graph_proofs", []), "target_term": d.get("target_term"),
            "evidence_sufficient": d.get("evidence_sufficient"), "rerank_top": d.get("rerank_top"),
            "degraded": d.get("degraded", False), "degraded_reason": d.get("degraded_reason"),
            "timings_ms": d.get("timings_ms", {})}


class LocalRetrieval:
    """In-process: the retrieval package's Retriever (models loaded once per serving process)."""

    def __init__(self, retriever):
        self.r = retriever

    def snapshot_id(self) -> str:
        return self.r.snapshots.active().snapshot_id

    async def run(self, question, pinned, mode, programme, academic_year_, *, scopes):
        from askiisc_retrieval.citations import result_to_dict
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        try:
            snap = await asyncio.to_thread(self.snapshot_id)
        except Exception as exc:
            raise RetrievalUnavailable(f"Snapshot unavailable: {type(exc).__name__}") from exc
        if pinned and pinned != snap:
            raise SnapshotMismatch(f"pinned snapshot {pinned} is not active (active: {snap})")
        if mode == "none":
            return [], {"mode": mode, "snapshot_id": snap, "evidence_sufficient": None, "timings_ms": {}}
        try:
            res = await asyncio.to_thread(self.r.retrieve, question, list(scopes), mode == "full", True)
        except Exception as e:
            raise RetrievalUnavailable(f"{type(e).__name__}: {e}") from e
        d = result_to_dict(res)
        if pinned and d["snapshot_id"] != pinned:
            raise SnapshotMismatch("Snapshot changed during retrieval")
        return [h for h in _filter_year(hits_from_result(d), academic_year_) if h["access_scope"] in scopes], _trace(d, mode)

    def source(self, chunk_id: str, scopes: list[str]) -> dict | None:
        """Full text of one cited chunk, only if it is in the active snapshot and the caller's scopes."""
        snap = self.r.snapshots.active()
        for c in self.r.store.get([chunk_id]):
            if c.version_id in snap.active_version_ids and c.access_scope in scopes:
                return {"chunk_id": c.chunk_id, "title": c.title, "section": c.section_heading,
                        "url": c.source_url, "version_id": c.version_id, "academic_term": c.academic_term,
                        "text": c.text, "access_scope": c.access_scope}
        return None

    async def ready(self) -> bool:
        try:
            await asyncio.to_thread(self.snapshot_id)
            return True
        except Exception:
            return False


class RemoteRetrieval:
    """Over HTTP: the retrieval service (`python -m askiisc_retrieval.cli serve`)."""

    def __init__(self, url: str, client=None, timeout_s: float = 60, service_key: str | None = None):
        import httpx
        import os
        self.service_key = service_key or os.environ.get("RETRIEVAL_SERVICE_KEY")
        self.url = url.rstrip("/")
        self.c = client or httpx.AsyncClient(timeout=timeout_s, trust_env=False)

    async def run(self, question, pinned, mode, programme, academic_year_, *, scopes):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "none":
            return [], {"mode": mode, "snapshot_id": None, "evidence_sufficient": None, "timings_ms": {}}
        try:
            r = await self.c.post(self.url + "/retrieve", headers={"Authorization": "Bearer " + self.service_key} if self.service_key else {},
                                  json={"question": question, "scopes": list(scopes),
                                                               "use_graph": mode == "full", "context_chars": None})
            r.raise_for_status()
        except Exception as e:
            raise RetrievalUnavailable(f"{type(e).__name__}: {e}") from e
        try:
            d = r.json()
            if pinned and d.get("snapshot_id") != pinned:
                raise SnapshotMismatch("Remote snapshot changed")
            citations = {c["marker"]: c for c in d["citations"]}
            if len(citations) != len(d["citations"]):
                raise RetrievalUnavailable("Remote citation markers are ambiguous")
            if any("access_scope" not in c for c in citations.values()):
                raise RetrievalUnavailable("Remote retrieval must supply access_scope on every citation")
            for passage in d["passages"]:
                citation = citations.get(passage.get("marker"))
                if (citation is None or passage.get("access_scope") != citation["access_scope"]
                        or passage.get("chunk_id") not in citation["chunk_ids"]
                        or passage["source_url"] != citation["source_url"]
                        or passage.get("version_id", citation["version_id"]) != citation["version_id"]
                        or passage.get("department", "") != citation.get("department", "")):
                    raise RetrievalUnavailable("Remote passage provenance does not match its citation")
            hits = [h for h in _filter_year(hits_from_result(d), academic_year_) if h["access_scope"] in scopes]
            return hits, _trace(d, mode)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise RetrievalUnavailable("Remote retrieval returned malformed evidence") from exc

    def source(self, chunk_id, scopes):
        return None

    async def ready(self) -> bool:
        try:
            return (await self.c.get(self.url + "/health")).status_code == 200
        except Exception:
            return False
