"""Read the ingestion team's stores (Smit's pipeline) as retrieval's `ChunkPayload`s.

Two ingestion layouts are supported, detected from the payload keys:

  v2 (pipeline.py + contract.py, 2026-10-24): the payload IS the retrieval contract (14 fields:
          chunk_id, document_id, version_id, source_url, title, section_heading, text, access_scope,
          doc_type, department, academic_term, course_codes, embedding_model, ordinal). No is_active
          flags in Qdrant: old points of a URL are deleted on re-ingestion, and Postgres
          document_versions.is_active says which version is current.
  v1 (integration manual, 2026-10-22): payload {id, version_id, source_url, department, chunk_index,
          text, is_active, is_superseded}; the missing fields are derived as below.
  Postgres `iisc_registry`: document_versions (is_active, is_superseded), document_chunks
          (chunk_id <-> qdrant_point_id [, page_or_section, embedding_model]); columns are discovered.
  Neo4j   edges carry chunk_id / version_id / source_url (see graph_neo4j.py)
  TEI     http://localhost:8080/embed (same BGE-M3)

Fields retrieval needs that the payload does not carry yet are DERIVED here, and every derived field is
used only if the payload lacks it, so the adapter keeps working, and gets better, as ingestion adds them:

  field            taken from payload if present, else ...
  chunk_id         Postgres document_chunks (qdrant_point_id -> chunk_id), else the Qdrant point id
  title            first markdown heading of the document's chunk 0 / of the chunk, else the URL
  section_heading  payload `page_or_section`, else the first markdown heading in the chunk
  doc_type         URL rules (crawl.TYPE_RULES + /faculty/ = people)
  access_scope     "public" (ingestion has no scopes yet)
  course_codes     codes found in heading + text
  academic_term    None (cannot be derived reliably from a 300-500 word window)
  embedding_model  Postgres document_chunks.embedding_model, else BAAI/bge-m3

Filters on fields that exist in the payload run INSIDE the Qdrant search (version_id, access_scope,
embedding_model, department, course_codes, doc_type, and is_active / is_superseded when present).
Filters on derived fields run on an over-fetched candidate list in Python, and every candidate is
re-checked in Python either way.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass
from urllib.parse import urlparse

from .contract import ChunkPayload
from .crawl import TYPE_RULES
from .parse import course_codes as find_codes, is_course_code
from .snapshot import Snapshot

log = logging.getLogger("askiisc.ingestion")

CONTRACT_FIELDS = ("chunk_id", "version_id", "source_url", "title", "section_heading", "text", "access_scope",
                   "doc_type", "department", "academic_term", "course_codes", "embedding_model")
# The ingestion manual's faculty pages live under /faculty/<name>; the stand-in crawl's under /people/.
EXTRA_TYPE_RULES = [(r"^/faculty/[^/]+/?$", "people")]
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", re.M)


def doc_type_from_url(url: str) -> str:
    path = urlparse(url).path or "/"
    for pat, t in EXTRA_TYPE_RULES + TYPE_RULES:
        if re.search(pat, path):
            return t
    return "page"


def _ordinal(pl: dict) -> int:
    v = pl.get("ordinal", pl.get("chunk_index", 0))
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def first_heading(text: str) -> str | None:
    m = HEADING_RE.search(text or "")
    return m.group(2).strip() if m else None


def _title_from_url(url: str) -> str:
    seg = [s for s in urlparse(url).path.split("/") if s]
    return seg[-1].replace("-", " ").replace("_", " ").title() if seg else urlparse(url).netloc


# ---------------------------------------------------------------- Postgres registry (optional)
class Registry:
    """What retrieval reads from the ingestion registry. `PostgresRegistry` in deployment; a dict-backed
    one in tests. Every method may raise; callers fall back to what Qdrant alone provides."""

    def active_version_ids(self) -> set[str]: ...
    def chunk_ids(self) -> dict[str, tuple[str, str | None, str | None]]:
        """qdrant_point_id -> (chunk_id, page_or_section, embedding_model)"""
        ...


class PostgresRegistry(Registry):
    def __init__(self, dsn: str):
        import psycopg2
        self.dsn, self._pg = dsn, psycopg2

    def _rows(self, sql: str):
        with self._pg.connect(self.dsn, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()

    def active_version_ids(self):
        return {str(r[0]) for r in self._rows(
            "SELECT version_id FROM document_versions WHERE is_active AND NOT COALESCE(is_superseded, FALSE)")}

    def _columns(self, table: str) -> set[str]:
        return {r[0] for r in self._rows(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'")}

    def chunk_ids(self):
        cols = self._columns("document_chunks")
        sec = "page_or_section" if "page_or_section" in cols else "NULL"
        mod = "embedding_model" if "embedding_model" in cols else "NULL"
        return {str(p): (str(c), s, m) for c, p, s, m in self._rows(
            f"SELECT chunk_id, qdrant_point_id, {sec}, {mod} FROM document_chunks")}


class DictRegistry(Registry):
    def __init__(self, active: set[str] | None = None, chunks: dict | None = None):
        self.active, self.chunks = set(active or ()), dict(chunks or {})

    def active_version_ids(self):
        return set(self.active)

    def chunk_ids(self):
        return dict(self.chunks)


# ---------------------------------------------------------------- the store
@dataclass
class AdapterStats:
    points: int = 0
    native_fields: tuple[str, ...] = ()
    mapped_chunk_ids: int = 0
    registry: str = "none"
    bad_course_codes: int = 0                  # payload course_codes that fail validation (dropped)
    bad_code_examples: tuple[str, ...] = ()


class IngestionQdrantStore:
    """Read-only, same `search` / `get` interface as `QdrantStore`, over the ingestion collection."""

    def __init__(self, collection: str = "iisc_policy_passages", url: str | None = None, *, client=None,
                 departments: tuple[str, ...] = ("CDS",), embedding_model: str = "BAAI/bge-m3",
                 registry: Registry | None = None):
        from qdrant_client import QdrantClient, models
        self.m = models
        self.c = client or QdrantClient(url=url, timeout=30)
        self.col, self.departments, self.default_model = collection, tuple(departments), embedding_model
        self.backend = f"ingestion qdrant {url or '(client)'} / {collection}"
        self.registry = registry
        self.stats = AdapterStats()
        self._by_point: dict[str, ChunkPayload] = {}
        self._point_of: dict[str, str] = {}
        self._native: set[str] = set()
        self.refresh()

    # ---- loading
    def refresh(self) -> None:
        """Read every point's payload once (no vectors) and derive the missing contract fields.
        Corpus is small (hundreds to tens of thousands of chunks); this keeps search-time work trivial."""
        reg_map: dict = {}
        if self.registry is not None:
            try:
                reg_map = self.registry.chunk_ids()
                self.stats.registry = "postgres"
            except Exception as e:                                   # registry down: Qdrant alone
                log.warning("registry unavailable, using Qdrant point ids as chunk ids: %s", e)
                self.stats.registry = f"unavailable ({type(e).__name__})"
        raw: list[tuple[str, dict]] = []
        offset = None
        while True:
            pts, offset = self.c.scroll(self.col, limit=512, offset=offset, with_payload=True, with_vectors=False)
            raw.extend((str(p.id), p.payload or {}) for p in pts)
            if offset is None:
                break
        native = sorted({k for _, pl in raw for k in pl})
        self._native = set(native)
        titles: dict[str, str] = {}                                  # version -> title from its first chunk
        for pid, pl in sorted(raw, key=lambda r: _ordinal(r[1])):
            v = pl.get("version_id", "")
            if v not in titles:
                titles[v] = pl.get("title") or first_heading(pl.get("text", "")) or _title_from_url(pl.get("source_url", ""))
        by_point, point_of, mapped = {}, {}, 0
        bad, bad_ex = 0, []
        for pid, pl in raw:
            reg = reg_map.get(pid)
            mapped += reg is not None
            c = self._to_chunk(pid, pl, reg, titles)
            extra = sorted(set(pl.get("course_codes") or []) - set(c.course_codes))
            if extra:
                bad += 1
                bad_ex.extend(x for x in extra if x not in bad_ex)
            by_point[pid] = c
            point_of[c.chunk_id] = pid
        self._by_point, self._point_of = by_point, point_of
        self.stats.points, self.stats.native_fields, self.stats.mapped_chunk_ids = len(raw), tuple(native), mapped
        self.stats.bad_course_codes, self.stats.bad_code_examples = bad, tuple(bad_ex[:6])

    def _to_chunk(self, pid: str, pl: dict, reg, titles: dict) -> ChunkPayload:
        text = pl.get("text") or pl.get("text_content") or ""
        url = pl.get("source_url", "")
        reg_chunk, reg_section, reg_model = reg if reg else (None, None, None)
        heading = pl.get("section_heading") or pl.get("page_or_section") or reg_section or first_heading(text) \
            or f"part {_ordinal(pl) + 1}"
        # Course codes: the payload's (ingestion v3 takes a row's code from its Course ID column), each one
        # validated; derived from the text only when the payload has none (v1).
        codes = pl.get("course_codes")
        codes = [c for c in codes if is_course_code(c)] if codes is not None else find_codes(f"{heading} {text}")
        return ChunkPayload(
            chunk_id=str(pl.get("chunk_id") or reg_chunk or pid),
            document_id=str(pl.get("document_id") or hashlib.sha1(url.encode()).hexdigest()[:12]),
            version_id=str(pl.get("version_id", "")),
            source_url=url,
            title=pl.get("title") or titles.get(pl.get("version_id", ""), _title_from_url(url)),
            section_heading=heading,
            text=text,
            access_scope=pl.get("access_scope") or "public",
            doc_type=pl.get("doc_type") or doc_type_from_url(url),
            department=pl.get("department") or "",
            academic_term=pl.get("academic_term"),
            course_codes=list(codes),
            embedding_model=pl.get("embedding_model") or reg_model or self.default_model,
            ordinal=_ordinal(pl),
        )

    @property
    def layout(self) -> str:
        return "full contract payload" if {"chunk_id", "access_scope", "embedding_model"} <= self._native \
            else "minimal payload (fields derived)"

    def _has_flags(self) -> bool:
        return {"is_active", "is_superseded"} <= self._native

    def active_flagged_version_ids(self) -> set[str]:
        """Fallback when Postgres is down. v1: versions of points flagged active and not superseded.
        v2 has no flags but deletes a URL's old points on re-ingestion, so every version in Qdrant is
        the current one."""
        if not self._has_flags():
            return {c.version_id for c in self._by_point.values()}
        out = set()
        offset = None
        flt = self._native_filter(None)
        while True:
            pts, offset = self.c.scroll(self.col, limit=512, offset=offset, with_payload=["version_id"],
                                        with_vectors=False, scroll_filter=flt)
            out.update(str((p.payload or {}).get("version_id")) for p in pts)
            if offset is None:
                break
        return out

    # ---- search
    def _native_filter(self, version_ids, scopes=None, embedding_model=None, course_codes=None, doc_types=None):
        """Every condition whose field the payload actually carries goes into the Qdrant filter."""
        m, have = self.m, self._native
        must = []
        if self._has_flags():
            must += [m.FieldCondition(key="is_active", match=m.MatchValue(value=True)),
                     m.FieldCondition(key="is_superseded", match=m.MatchValue(value=False))]
        if self.departments:
            must.append(m.FieldCondition(key="department", match=m.MatchAny(any=list(self.departments))))
        if version_ids is not None:
            must.append(m.FieldCondition(key="version_id", match=m.MatchAny(any=sorted(version_ids))))
        if scopes is not None and "access_scope" in have:
            must.append(m.FieldCondition(key="access_scope", match=m.MatchAny(any=list(scopes))))
        if embedding_model and "embedding_model" in have:
            must.append(m.FieldCondition(key="embedding_model", match=m.MatchValue(value=embedding_model)))
        if course_codes and "course_codes" in have:
            must.append(m.FieldCondition(key="course_codes", match=m.MatchAny(any=list(course_codes))))
        if doc_types and "doc_type" in have:
            must.append(m.FieldCondition(key="doc_type", match=m.MatchAny(any=list(doc_types))))
        return m.Filter(must=must) if must else None

    def search(self, vector, k, version_ids, scopes, embedding_model, course_codes=None, doc_types=None):
        if version_ids is not None and not version_ids:
            return []                                              # nothing active: nothing to return
        derived = not {"access_scope", "embedding_model", "doc_type"} <= self._native
        extra = bool(course_codes or doc_types)
        fetch = min(max(k * (8 if extra else 2), 50), 1000) if derived else k   # over-fetch only for derived
        res = self.c.query_points(self.col, query=vector, limit=fetch, with_payload=False,
                                  query_filter=self._native_filter(version_ids, scopes, embedding_model,
                                                                   course_codes, doc_types))
        out = []
        for p in res.points:
            c = self._by_point.get(str(p.id))
            if c is None:                                          # indexed after our last refresh
                continue
            if c.access_scope not in scopes or c.embedding_model != embedding_model:
                continue
            if course_codes and not set(course_codes) & set(c.course_codes):
                continue
            if doc_types and c.doc_type not in doc_types:
                continue
            out.append((c, float(p.score)))
            if len(out) >= k:
                break
        return out

    def document_chunks(self, source_url, version_ids, scopes, embedding_model):
        return [c for c in self._by_point.values() if c.source_url == source_url
                and c.version_id in version_ids and c.access_scope in scopes and c.embedding_model == embedding_model]

    def get(self, chunk_ids: list[str]) -> list[ChunkPayload]:
        return [self._by_point[self._point_of[c]] for c in chunk_ids if c in self._point_of]

    def count(self) -> int:
        return len(self._by_point)

    def chunks(self) -> list[ChunkPayload]:
        return list(self._by_point.values())

    def close(self):
        try:
            self.c.close()
        except Exception:
            pass


# ---------------------------------------------------------------- active snapshot
class IngestionSnapshotProvider:
    """Active versions = document_versions WHERE is_active AND NOT is_superseded (Postgres), else the
    versions of Qdrant points flagged that way. Cached for `ttl_s`, so a re-ingestion is picked up
    without restarting the service. A single manifest row would make Qdrant and Neo4j switch together
    (PRD FR-4); until ingestion has one, both stores are filtered by the same version ids."""

    def __init__(self, store: IngestionQdrantStore, registry: Registry | None = None,
                 embedding_model: str = "BAAI/bge-m3", graph_version: str | None = "neo4j", ttl_s: float = 60.0):
        self.store, self.registry, self.model, self.graph_version, self.ttl = store, registry, embedding_model, graph_version, ttl_s
        self._snap: Snapshot | None = None
        self._at = 0.0
        self.source = "none"

    def active(self) -> Snapshot:
        if self._snap is not None and time.monotonic() - self._at < self.ttl:
            return self._snap
        ids = None
        if self.registry is not None:
            try:
                ids, self.source = self.registry.active_version_ids(), "postgres"
            except Exception as e:
                log.warning("registry unavailable for active versions, using Qdrant flags: %s", e)
        if ids is None:
            ids, self.source = self.store.active_flagged_version_ids(), "qdrant flags"
        sid = "ing-" + hashlib.sha1(("|".join(sorted(ids)) + self.model).encode()).hexdigest()[:10]
        if self._snap is not None and sid != self._snap.snapshot_id:
            self.store.refresh()                                   # re-ingested: reload payloads too
        self._snap, self._at = Snapshot(sid, self.model, self.graph_version, frozenset(ids)), time.monotonic()
        return self._snap


def point_uuid(seed: str) -> str:
    """Deterministic UUID (tests build fake ingestion collections with it)."""
    return str(uuid.UUID(hashlib.md5(seed.encode()).hexdigest()))
