"""Neo4j-backed graph store for the ingestion team's knowledge graph.

The graph is READ from Neo4j and held in memory with the same traversal code that the tests cover
(`JsonGraphStore.expand`: term rules, evidence vs bridge edges, 2 hops through Course/Person only).
It is re-read every `ttl_s` seconds, so re-ingestion shows up without a restart. Why a read-through
cache instead of a Cypher query per request: the CDS graph is ~100 nodes / ~160 edges, a per-request
Bolt round trip costs milliseconds for nothing, and the traversal rules stay in one tested place.
If Neo4j is unreachable at start, graph calls raise -> the retriever falls back to vector-only results
with `degraded=True` (PRD FR-9). If it becomes unreachable later, the last good copy keeps serving.

Ingestion schema (manual 2026-10-22) -> retrieval node ids / relations:
  (:Faculty {name, aliases?})  -> Person:<initial surname>   (graph_rules.person: same alias keys as
                                                              the stand-in; `aliases` are indexed too)
  (:Course {code})             -> Course:<CODE without spaces>
  (:ResearchArea {name})       -> Topic:<name lower-case>
  (:Department {name})         -> Dept:<name>
  (:Lab {name})                -> Lab:<name>
  RESEARCHES_IN -> WORKS_ON, REQUIRES_PREREQUISITE -> REQUIRES, others keep their name
  (TAUGHT_BY, HEADS, OFFERED_IN are used as soon as ingestion provides them.)
Every edge must carry chunk_id + version_id (+ source_url): the chunk is what gets cited, never the edge.
Edges without a chunk_id are skipped and counted.

Ingestion v3 (pipeline.py 2026-10-24) writes one Faculty node per person ({key, name, aliases}),
RESEARCHES_IN and HEADS (-> ResearchLab) edges, and validated course codes. Retrieval still VALIDATES
what it reads, so an older graph degrades gracefully instead of giving wrong answers:
  - a person node whose name is several people joined with '&' / 'and' is split (v2 graphs);
  - TAUGHT_BY {academic_term: "UNKNOWN"} -> no term;
  - an edge whose Course is not a course code (a room number) is skipped and counted;
  - MEMBER_OF carries no chunk_id -> skipped (it is not evidence anyway);
  - only when Neo4j has neither WORKS_ON nor HEADS edges (v2 graphs) are rule-derived edges from the
    chunks merged in (`supplement`).
"""
from __future__ import annotations

import logging
import re
import time
from collections import Counter
from .contract import SCOPES
from typing import Callable

from .graph_rules import person
from .parse import is_course_code
from .stores import Edge, JsonGraphStore

log = logging.getLogger("askiisc.graph")

EDGE_QUERY = ("MATCH (a)-[r]->(b) RETURN labels(a) AS la, properties(a) AS pa, type(r) AS t, "
              "properties(r) AS pr, labels(b) AS lb, properties(b) AS pb")
REL_MAP = {"RESEARCHES_IN": "WORKS_ON", "REQUIRES_PREREQUISITE": "REQUIRES"}
TERM_RE = re.compile(r"^\d{4}-(AUG|JAN|SUM)$")
PERSON_LABELS = {"Faculty", "Person", "Instructor"}


def node_key(labels: list[str], props: dict) -> str:
    lab = next((l for l in labels if l in PERSON_LABELS), labels[0] if labels else "Node")
    name = str(props.get("name") or props.get("code") or props.get("id") or "").strip()
    if lab in PERSON_LABELS:
        return person(name) or f"Person:{name.lower()}"
    if lab == "Course":
        return "Course:" + re.sub(r"[\s\-]+", "", str(props.get("code") or name)).upper()
    if lab == "ResearchArea":
        return f"Topic:{name.lower()}"
    if lab == "Department":
        return f"Dept:{name}"
    if lab in ("ResearchLab", "Lab"):
        return f"Lab:{name}"
    return f"{lab}:{name}"


def valid_course(key: str) -> bool:
    """Validation: an edge whose Course node is not a course code (e.g. a room number) is skipped."""
    return not key.startswith("Course:") or is_course_code(key[7:])


CO_NAMES = re.compile(r"\s*(?:&|\band\b|;)\s*")


def _keys(labels: list[str], props: dict) -> list[tuple[str, str]]:
    """(node key, display name) pairs. A person node whose name is several people joined with '&' / 'and'
    becomes several persons."""
    if set(labels) & PERSON_LABELS:
        raw = str(props.get("name") or "")
        names = [n for n in CO_NAMES.split(raw) if n.strip()] or [raw]
        out = []
        for n in names:
            k = node_key(labels, {**props, "name": n})
            if (k, n) not in out:
                out.append((k, n))
        return out
    return [(node_key(labels, props), str(props.get("name") or props.get("code") or ""))]


def edges_from_records(records: list[dict]) -> tuple[list[Edge], dict[str, list[str]], Counter]:
    """Records (as returned by EDGE_QUERY) -> retrieval Edges, person aliases, and skip counts."""
    edges: list[Edge] = []
    aliases: dict[str, list[str]] = {}
    skipped: Counter = Counter()
    for r in records:
        la, pa, lb, pb, pr = r["la"], r["pa"] or {}, r["lb"], r["pb"] or {}, r["pr"] or {}
        if pr.get("access_scope") not in SCOPES:
            skipped["missing or invalid access_scope"] += 1
            continue
        srcs, dsts = _keys(la, pa), _keys(lb, pb)
        for labels, props, keys in ((la, pa, srcs), (lb, pb, dsts)):
            if set(labels) & PERSON_LABELS:
                for key, name in keys:
                    names = [name] + (list(props.get("aliases") or []) if len(keys) == 1 else [])
                    aliases.setdefault(key, [])
                    aliases[key].extend(n for n in names if n and n not in aliases[key])
        chunk = pr.get("chunk_id") or pr.get("chunk_ids")
        chunks = tuple(str(c) for c in (chunk if isinstance(chunk, (list, tuple)) else [chunk]) if c)
        if not chunks or not pr.get("version_id"):
            skipped["no chunk_id/version_id"] += 1
            continue
        term = pr.get("academic_term")
        term = term if isinstance(term, str) and TERM_RE.match(term) else None
        if not all(valid_course(k) for k, _ in srcs + dsts):
            skipped["not a course code (room number?)"] += 1
            continue
        for src, _ in srcs:
            for dst, _ in dsts:
                if src != dst:
                    edges.append(Edge(src, REL_MAP.get(r["t"], r["t"]), dst, chunks, str(pr["version_id"]),
                                      pr["access_scope"], term))
    return edges, aliases, skipped


def merge_edges(primary: list[Edge], extra: list[Edge]) -> tuple[list[Edge], int]:
    """primary + the extra edges that are not already there (same src, relation, dst and term)."""
    seen = {(e.src, e.rel, e.dst, e.academic_term) for e in primary}
    added = [e for e in extra if (e.src, e.rel, e.dst, e.academic_term) not in seen
             and not seen.add((e.src, e.rel, e.dst, e.academic_term))]
    return primary + added, len(added)


def neo4j_fetcher(uri: str, user: str, password: str) -> Callable[[], list[dict]]:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=5)

    def fetch() -> list[dict]:
        with driver.session() as s:
            return [rec.data() for rec in s.run(EDGE_QUERY)]
    fetch.close = driver.close                                    # type: ignore[attr-defined]
    return fetch


class Neo4jGraphStore:
    """GraphStore over Neo4j (read-through cache). `fetch` returns EDGE_QUERY records."""

    def __init__(self, fetch: Callable[[], list[dict]], ttl_s: float = 300.0, label: str = "neo4j",
                 supplement: Callable[[], list[Edge]] | None = None):
        """`supplement`: extra edges (rule-based, from the ingestion chunks) merged into what Neo4j
        returns, for relations the ingestion graph does not have yet. Neo4j edges win on duplicates."""
        self.fetch, self.ttl, self.label, self.supplement = fetch, ttl_s, label, supplement
        self._inner: JsonGraphStore | None = None
        self._at = 0.0
        self.error: str | None = None
        self.stats: dict = {}
        self.fail = False                                          # tests: simulate an outage
        self._load()

    def _load(self) -> None:
        try:
            records = self.fetch()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            log.warning("neo4j load failed (%s); %s", self.error,
                        "keeping last good graph" if self._inner else "graph unavailable -> degraded mode")
            self._at = time.monotonic()                            # do not hammer a down server
            return
        edges, aliases, skipped = edges_from_records(records)
        n_neo4j, added = len(edges), 0
        neo4j_rels = dict(Counter(e.rel for e in edges))
        if self.supplement is not None and not {"WORKS_ON", "HEADS"} & set(neo4j_rels):   # only for v2 graphs
            try:
                edges, added = merge_edges(edges, self.supplement())
            except Exception as e:                                 # rules must never take the graph down
                log.warning("graph rule supplement failed: %s", e)
        g = JsonGraphStore(edges)
        for node, names in aliases.items():
            g.add_aliases(node, names)
        self._inner, self._at, self.error = g, time.monotonic(), None
        self.stats = {"edges": len(edges), "from_neo4j": n_neo4j, "from_rules": added, "neo4j_relations": neo4j_rels,
                      "by_relation": dict(Counter(e.rel for e in edges)),
                      "nodes": len({n for e in edges for n in (e.src, e.dst)}), "skipped": dict(skipped)}

    def _g(self) -> JsonGraphStore:
        if self.fail:
            raise ConnectionError("graph store unavailable (simulated)")
        if time.monotonic() - self._at > self.ttl:
            self._load()
        if self._inner is None:
            raise ConnectionError(f"neo4j unavailable: {self.error}")
        return self._inner

    def active_edges(self):
        return self._g().active_edges()

    def expand(self, *a, **kw):
        return self._g().expand(*a, **kw)

    def entities_of_chunks(self, chunk_ids):
        return self._g().entities_of_chunks(chunk_ids)

    def find_people(self, text):
        return self._g().find_people(text)

    @property
    def edges(self):
        return self._inner.edges if self._inner else []

    def close(self):
        if hasattr(self.fetch, "close"):
            self.fetch.close()
