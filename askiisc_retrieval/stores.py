"""Embedder, vector store and graph store behind small interfaces, so the laptop versions
(sentence-transformers, Qdrant local mode, JSON graph) swap for TEI / Qdrant server / Neo4j later
without touching the retriever."""
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .contract import ChunkPayload


# ---------------------------------------------------------------- embedders
class Embedder(Protocol):
    model_id: str
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class BGEM3Embedder:
    """BGE-M3 dense vectors on CPU via sentence-transformers. First use downloads ~2.3 GB."""

    def __init__(self, model_id: str = "BAAI/bge-m3", max_seq_length: int = 512, batch_size: int = 8):
        from sentence_transformers import SentenceTransformer
        self.model_id = model_id
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_id, device="cpu")
        self.model.max_seq_length = max_seq_length
        import threading
        self._lock = threading.Lock()

    def embed_documents(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=True).tolist()

    def embed_query(self, text):
        with self._lock:
            return self.model.encode([text], normalize_embeddings=True)[0].tolist()


class TEIEmbedder:
    """Same model served by text-embeddings-inference (deployment)."""

    def __init__(self, endpoint: str, model_id: str = "BAAI/bge-m3"):
        import httpx
        self.model_id, self.c = model_id, httpx.Client(base_url=endpoint, timeout=30)

    def embed_documents(self, texts):
        out = []
        for i in range(0, len(texts), 32):
            r = self.c.post("/embed", json={"inputs": texts[i:i + 32], "normalize": True, "truncate": True})
            r.raise_for_status()
            out.extend(r.json())
        return out

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class HashEmbedder:
    """Deterministic bag-of-words vectors for tests only. Never use for evaluation numbers."""

    def __init__(self, dim: int = 256, model_id: str = "test/hash-bow"):
        self.dim, self.model_id = dim, model_id

    def _one(self, t: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", t.lower()):
            d = hashlib.md5(tok.encode()).digest()
            v[int.from_bytes(d[:4], "little") % self.dim] += 1.0 if d[4] & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._one(t) for t in texts]

    def embed_query(self, text):
        return self._one(text)


# ---------------------------------------------------------------- rerankers
class Reranker(Protocol):
    model_id: str
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class BGEReranker:
    """BGE cross-encoder (reads question and passage together). Scores are sigmoid(logit) in [0, 1].
    Uses transformers directly so the output does not depend on sentence-transformers' version."""

    def __init__(self, model_id: str = "BAAI/bge-reranker-base", max_length: int = 384, batch_size: int = 8,
                 device: str | None = None):
        import threading

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.model_id, self.max_length, self.batch_size = torch, model_id, max_length, batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self._lock = threading.Lock()          # HF fast tokenizers are not safe under concurrent calls

    def score(self, query, passages):
        out: list[float] = []
        with self._lock, self.torch.no_grad():
            for i in range(0, len(passages), self.batch_size):
                batch = passages[i:i + self.batch_size]
                enc = self.tok([query] * len(batch), batch, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits.view(-1).float()
                out.extend(self.torch.sigmoid(logits).tolist())
        return out


class KeywordReranker:
    """Fraction of question words found in the passage. Tests only; never for evaluation numbers."""
    model_id = "test/keyword-overlap"
    STOP = {"the", "a", "an", "of", "in", "for", "to", "is", "what", "who", "which", "does", "do", "how",
            "and", "at", "by", "this", "that", "are", "did", "have"}

    def __init__(self):
        self.fail = False

    def score(self, query, passages):
        if self.fail:
            raise RuntimeError("reranker unavailable (simulated)")
        q = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in self.STOP}
        return [len(q & set(re.findall(r"[a-z0-9]+", p.lower()))) / max(len(q), 1) for p in passages]


# ---------------------------------------------------------------- vector store
class QdrantStore:
    """Qdrant with payload filtering. `path=` gives local mode (a folder, no server); `url=` the server.
    Filters run inside the search, so top-k is computed over eligible chunks only."""

    def __init__(self, collection: str, dim: int, path: str | Path | None = None, url: str | None = None):
        from qdrant_client import QdrantClient, models
        self.m = models
        self.c = QdrantClient(path=str(path)) if path is not None else QdrantClient(url=url, timeout=30)
        self.col, self.dim, self.server = collection, dim, path is None
        self.backend = f"qdrant server {url}" if self.server else f"qdrant local {path}"
        self._ensure()

    def _ensure(self):
        m = self.m
        if self.c.collection_exists(self.col):
            return
        self.c.create_collection(self.col, vectors_config=m.VectorParams(size=self.dim, distance=m.Distance.COSINE))
        if self.server:                               # payload indexes only exist on the server
            for f in ("version_id", "access_scope", "embedding_model", "course_codes", "academic_term"):
                self.c.create_payload_index(self.col, f, m.PayloadSchemaType.KEYWORD)

    def reset(self):
        """Drop and recreate the collection (the stub ingestion rebuilds from scratch)."""
        if self.c.collection_exists(self.col):
            self.c.delete_collection(self.col)
        self._ensure()

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.UUID(hashlib.md5(chunk_id.encode()).hexdigest()))

    def upsert(self, chunks: list[ChunkPayload], vectors: list[list[float]]) -> None:
        pts = [self.m.PointStruct(id=self.point_id(c.chunk_id), vector=v, payload=c.to_dict())
               for c, v in zip(chunks, vectors)]
        for i in range(0, len(pts), 256):
            self.c.upsert(self.col, points=pts[i:i + 256])

    def _filter(self, version_ids, scopes, embedding_model, course_codes=None, doc_types=None):
        m = self.m
        must = [m.FieldCondition(key="version_id", match=m.MatchAny(any=sorted(version_ids))),
                m.FieldCondition(key="access_scope", match=m.MatchAny(any=list(scopes))),
                m.FieldCondition(key="embedding_model", match=m.MatchValue(value=embedding_model))]
        if course_codes:
            must.append(m.FieldCondition(key="course_codes", match=m.MatchAny(any=list(course_codes))))
        if doc_types:
            must.append(m.FieldCondition(key="doc_type", match=m.MatchAny(any=list(doc_types))))
        return m.Filter(must=must)

    def search(self, vector, k, version_ids, scopes, embedding_model, course_codes=None, doc_types=None):
        res = self.c.query_points(self.col, query=vector, limit=k, with_payload=True,
                                  query_filter=self._filter(version_ids, scopes, embedding_model, course_codes,
                                                            doc_types))
        return [(ChunkPayload(**p.payload), float(p.score)) for p in res.points]

    def document_chunks(self, source_url, version_ids, scopes, embedding_model):
        filt = self._filter(version_ids, scopes, embedding_model)
        filt.must.append(self.m.FieldCondition(key="source_url", match=self.m.MatchValue(value=source_url)))
        points, offset, out = [], None, []
        while True:
            points, offset = self.c.scroll(self.col, scroll_filter=filt, offset=offset,
                                           limit=128, with_payload=True, with_vectors=False)
            out.extend(ChunkPayload(**p.payload) for p in points)
            if offset is None:
                break
        return out

    def get(self, chunk_ids: list[str]) -> list[ChunkPayload]:
        if not chunk_ids:
            return []
        pts = self.c.retrieve(self.col, ids=[self.point_id(i) for i in chunk_ids], with_payload=True)
        return [ChunkPayload(**p.payload) for p in pts]

    def count(self) -> int:
        return self.c.count(self.col).count

    def close(self):
        self.c.close()


# ---------------------------------------------------------------- graph store
@dataclass(frozen=True)
class Edge:
    """A source-backed fact. Never cited itself: its supporting chunks are what get cited."""
    src: str
    rel: str
    dst: str
    chunk_ids: tuple[str, ...]
    version_id: str
    scope: str
    academic_term: str | None = None
    qualifiers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class GraphHit:
    """An edge reached during expansion. Only `evidence` edges contribute citable chunks; `bridge` edges
    exist only to connect hops (e.g. 'who teaches DS 252 now' on the way to a January course)."""
    edge: Edge
    evidence: bool
    hop: int
    seeds: frozenset[str]          # which seeds reached this edge (more seeds = more relevant)


def _term_key(term: str) -> tuple[int, int]:
    y, s = term.split("-")
    return int(y), {"JAN": 0, "SUM": 1, "AUG": 2}[s]


class GraphStore(Protocol):
    def expand(self, seeds: list[str], version_ids: frozenset[str], scopes: list[str], hops: int,
               limit: int, *, target_term: str, explicit: bool, now_term: str) -> list[GraphHit]: ...
    def entities_of_chunks(self, chunk_ids: list[str]) -> list[str]: ...
    def find_people(self, text: str) -> list[str]: ...


class JsonGraphStore:
    """In-process graph persisted as JSON. Stand-in for Neo4j on the laptop. `fail=True` simulates
    an outage for the degraded-mode tests (E6)."""

    def __init__(self, edges: list[Edge] | None = None):
        self.edges = edges or []
        self.fail = False
        self._by_node: dict[str, list[Edge]] = defaultdict(list)
        self._by_chunk: dict[str, set[str]] = defaultdict(set)
        self._terms_of_course: dict[str, set[str]] = defaultdict(set)
        self._people_by_surname: dict[str, set[str]] = defaultdict(set)
        for e in self.edges:
            self._index(e)

    def _index(self, e: Edge):
        self._by_node[e.src].append(e)
        self._by_node[e.dst].append(e)
        for c in e.chunk_ids:
            self._by_chunk[c] |= {e.src, e.dst}
        if e.academic_term:
            self._terms_of_course[e.src].add(e.academic_term)
        for n in (e.src, e.dst):
            if n.startswith("Person:"):
                self._people_by_surname[n.split(" ")[-1].split(":")[-1]].add(n)

    def add_aliases(self, node: str, names: list[str]):
        """Make other spellings of a person ('Venky', 'R. Venkatesh Babu') find `node` in questions."""
        from .graph_rules import person
        for n in names:
            key = person(n) or f"Person:{n.strip().lower()}"
            surname = key.split(" ")[-1].split(":")[-1]
            if surname:
                self._people_by_surname[surname].add(node)

    def add(self, edges: list[Edge]):
        for e in edges:
            self.edges.append(e)
            self._index(e)

    def save(self, path: Path):
        path.write_text(json.dumps([asdict(e) for e in self.edges], indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "JsonGraphStore":
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        return cls([Edge(**{**e, "chunk_ids": tuple(e["chunk_ids"]), "qualifiers": tuple(tuple(q) for q in e.get("qualifiers", ()))}) for e in raw])

    def _check(self):
        if self.fail:
            raise ConnectionError("graph store unavailable (simulated)")

    def active_edges(self):
        self._check()
        return self.edges

    def entities_of_chunks(self, chunk_ids):
        self._check()
        out: list[str] = []
        for c in chunk_ids:
            for ent in sorted(self._by_chunk.get(c, ())):
                if ent not in out:
                    out.append(ent)
        return out

    def find_people(self, text):
        """Person nodes named in free text, via the same alias keys ('Y. Simmhan' == 'Yogesh Simmhan')."""
        self._check()
        toks = re.findall(r"[a-z]+", text.lower())
        found: list[str] = []
        for i, t in enumerate(toks):
            cands = self._people_by_surname.get(t)
            if not cands:
                continue
            if i > 0:                                   # prefer the one whose initial matches the previous word
                narrowed = {c for c in cands if c.split(":")[1][0] == toks[i - 1][0]}
                cands = narrowed or cands
            found.extend(sorted(c for c in cands if c not in found))
        return found

    def _latest_term(self, course: str, now_term: str) -> str | None:
        ok = [t for t in self._terms_of_course.get(course, ()) if _term_key(t) <= _term_key(now_term)]
        return max(ok, key=_term_key) if ok else None

    def expand(self, seeds, version_ids, scopes, hops, limit, *, target_term, explicit, now_term,
               through_types=("Course", "Person")):
        """Breadth-first from each seed, over edges in the active snapshot and the user's scopes.

        Term rules (term-bound edges only; timeless edges such as prerequisites always pass; future terms
        never pass):
          * no explicit term   -> every past term is traversable and citable; the retriever then picks,
                                  per course, the term linked to the most entities named in the question
                                  (ties -> most recent). "Who teaches DS 211?" -> latest offering;
                                  "Which course have A and B taught together?" -> latest term with BOTH.
          * explicit term T    -> cite only edges in T; may also traverse current-as-of-now edges as
                                  bridges ("the DS 252 instructor" is resolved as of now, then T is asked)
        Traversal continues only through `through_types` nodes: hubs such as a term or a venue connect to
        everything and would flood the result.
        """
        self._check()
        hits: dict[Edge, GraphHit] = {}
        for seed in sorted(set(seeds)):
            frontier, seen = [seed], {seed}
            for hop in range(hops):
                nxt: list[str] = []
                for node in frontier:
                    for e in self._by_node.get(node, ()):
                        if e.version_id not in version_ids or e.scope not in scopes:
                            continue
                        t = e.academic_term
                        if t is not None and _term_key(t) > _term_key(now_term) and t != target_term:
                            continue                                    # future terms were not asked about
                        if explicit:
                            current = t is None or t == self._latest_term(e.src, now_term)
                            evidence = t is None or t == target_term
                            if not (evidence or current):
                                continue
                        else:
                            evidence = True
                        prev = hits.get(e)
                        hits[e] = GraphHit(e, evidence, min(hop, prev.hop) if prev else hop,
                                           (prev.seeds if prev else frozenset()) | {seed})
                        for n in (e.src, e.dst):
                            if n not in seen and n.split(":", 1)[0] in through_types:
                                seen.add(n)
                                nxt.append(n)
                frontier = nxt
        ranked = sorted(hits.values(), key=lambda h: (-len(h.seeds), h.hop))
        return ranked if limit is None else ranked[:limit]
