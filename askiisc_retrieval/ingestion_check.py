"""Checks the ingestion output against what retrieval needs (`contract.ChunkPayload`) and reports
gaps in plain words, so the report can be sent to the ingestion team as it is.

    python -m askiisc_retrieval.cli check-ingestion [--json]
"""
from __future__ import annotations

import re
from collections import Counter

from .ingestion_adapter import CONTRACT_FIELDS, IngestionQdrantStore

TERM_RE = re.compile(r"^\d{4}-(AUG|JAN|SUM)$")


def check(store: IngestionQdrantStore, graph=None, active_ids: set[str] | None = None,
          active_source: str = "?") -> dict:
    chunks = store.chunks()
    native = set(store.stats.native_fields)
    rep: dict = {"points": len(chunks), "registry": store.stats.registry, "active_source": active_source,
                 "layout": getattr(store, "layout", "?"),
                 "native_fields": sorted(native),
                 "missing_fields": [f for f in CONTRACT_FIELDS if f not in native],
                 "derived_fields": {}, "problems": [], "notes": []}
    for f in CONTRACT_FIELDS:
        if f not in native:
            how = {"chunk_id": f"Postgres map for {store.stats.mapped_chunk_ids}/{len(chunks)} points"
                   if store.stats.mapped_chunk_ids else "Qdrant point id (no Postgres map)",
                   "title": "first markdown heading / URL", "section_heading": "first markdown heading in the chunk",
                   "doc_type": "URL rules", "access_scope": "'public' for all", "course_codes": "regex on text",
                   "academic_term": "NOT derivable -> None", "embedding_model": "Postgres or default"}.get(f, "-")
            rep["derived_fields"][f] = how
    by_type = Counter(c.doc_type for c in chunks)
    rep["doc_types"] = dict(by_type.most_common())
    versions = {c.version_id for c in chunks}
    rep["versions_in_qdrant"] = len(versions)
    if active_ids is not None:
        rep["active_versions"] = len(active_ids)
        orphan = versions - active_ids
        if orphan:
            rep["notes"].append(f"{len(orphan)} version(s) in Qdrant are not active (excluded from results).")
    no_heading = sum(c.section_heading.startswith("part ") for c in chunks)
    termed = sum(1 for c in chunks if c.academic_term and TERM_RE.match(c.academic_term))
    sched = [c for c in chunks if c.doc_type == "schedule"]
    words = sorted(len(c.text.split()) for c in chunks) or [0]
    rep["chunk_words"] = {"min": words[0], "median": words[len(words) // 2], "max": words[-1]}
    rep["chunks_without_heading"] = no_heading
    rep["chunks_with_term"] = termed
    rep["schedule_chunks"] = len(sched)
    rep["people_chunks"] = by_type.get("people", 0)
    urls = sorted({c.source_url for c in chunks})
    rep["pages"] = len(urls)
    must_have = {"courses/schedule": "course schedule", "courses/descriptions": "course descriptions",
                 "/people/": "people pages", "/faculty/": "faculty pages", "research-new": "research-area pages",
                 "admissions": "admissions pages"}
    rep["coverage"] = {label: sum(key in u for u in urls) for key, label in must_have.items()}

    # pages whose prose all carries ONE section heading although the page has several (pipeline.py v2
    # flushes prose only at a table or at the end of the page, so every paragraph gets the last heading)
    prose_pages = Counter()
    heads = {}
    for c in chunks:
        if not (c.doc_type == "schedule" and c.course_codes):
            prose_pages[c.source_url] += 1
            heads.setdefault(c.source_url, set()).add(c.section_heading)
    merged = sorted(u for u, n in prose_pages.items() if n >= 2 and len(heads[u]) == 1)
    rep["pages_with_one_prose_heading"] = len(merged)

    P = rep["problems"]
    if store.stats.bad_course_codes:
        P.append(f"{store.stats.bad_course_codes} chunks list course codes that are not courses "
                 f"(e.g. {', '.join(store.stats.bad_code_examples)}), e.g. room numbers ('Venue: CDS 202'). "
                 "They are dropped. Ingestion v3 fixes this (Course ID column, no CDS prefix).")
    if merged:
        P.append(f"{len(merged)} page(s) have several prose chunks that all carry the same section heading "
                 f"(e.g. {merged[0]}): prose is flushed only at a table or at the page end, so citations "
                 "point to the wrong section and per-section terms are lost. Fix: flush prose at every h2/h3.")
    if "academic_term" not in native or termed == 0:
        P.append("No per-chunk academic_term: 'this semester' / 'in Jan 2026' questions cannot be filtered.")
    if sched and max(len(c.course_codes) for c in sched) > 2:
        P.append(f"Schedule chunks hold several courses each (up to {max(len(c.course_codes) for c in sched)}): "
                 "need one chunk per table row so each course-semester row is citable.")
    if not sched:
        P.append("No course-schedule chunks found (URL /courses/schedule/): 'who teaches X' questions have no source.")
    if "section_heading" not in native:
        P.append(f"No section_heading in payload; derived from markdown headings ({no_heading} chunks have none).")
    if "access_scope" not in native:
        P.append("No access_scope: everything is treated as public; student/faculty/admin-only content cannot be protected.")
    if "chunk_id" not in native and not store.stats.mapped_chunk_ids:
        P.append("Qdrant payload has no chunk_id and Postgres was not reachable: graph evidence (edges carry "
                 "chunk_id) cannot be matched to passages. Add chunk_id to the payload.")
    if rep["people_chunks"] == 0:
        P.append("No people/faculty chunks recognised: 'who works on X' questions lose the people search.")
    if graph is not None:
        g = {"error": getattr(graph, "error", None), **getattr(graph, "stats", {})}
        rep["graph"] = g
        rels = set((g.get("by_relation") or {}).keys())
        n4 = set((g.get("neo4j_relations") or {}).keys())
        if g.get("from_neo4j") is not None and not {"WORKS_ON", "HEADS"} & n4:
            P.append("Neo4j has no research-area or lab edges (ResearchArea nodes are created without a "
                     "Faculty -> ResearchArea edge; no lab nodes): 'who works on X' questions rely on "
                     f"retrieval's rule fallback ({g.get('from_rules', 0)} edges added). Ingestion v3 writes "
                     "RESEARCHES_IN and HEADS edges.")
        if (g.get("skipped") or {}).get("not a course code (room number?)"):
            P.append(f"{g['skipped']['not a course code (room number?)']} graph edges involve a 'course' that "
                     "is a room number (e.g. Course:CDS202); skipped.")
        if "TAUGHT_BY" not in rels:
            P.append("Graph has no TAUGHT_BY (course -> instructor, with academic_term): multi-hop course/instructor "
                     "questions cannot use the graph.")
        persons = {n for e in graph.edges for n in (e.src, e.dst) if n.startswith("Person:")}
        short = sorted(n for n in persons if " " not in n.split(":", 1)[1])
        if short:
            P.append(f"{len(short)} faculty node(s) have a one-word name (e.g. {', '.join(short[:4])}): "
                     "store full names, plus an `aliases` list.")
        ids = {c.chunk_id for c in chunks}
        dangling = sum(1 for e in graph.edges if not set(e.chunk_ids) & ids)
        if dangling:
            P.append(f"{dangling}/{len(graph.edges)} graph edges point to chunk_ids not found in Qdrant "
                     "(no citable evidence for them).")
    else:
        rep["graph"] = None
    return rep


def render(rep: dict) -> str:
    L = [f"Ingestion check ({rep.get('layout', '?')}): {rep['points']} chunks from {rep['pages']} pages, "
         f"{rep['versions_in_qdrant']} versions (active set from {rep['active_source']}).",
         f"Chunk length (words): min {rep['chunk_words']['min']}, median {rep['chunk_words']['median']}, "
         f"max {rep['chunk_words']['max']}.",
         f"Payload fields present: {', '.join(rep['native_fields'])}",
         "Fields retrieval needs that are missing (and how they are filled for now):"]
    L += [f"  - {f}: {how}" for f, how in rep["derived_fields"].items()]
    L.append("Page types (derived): " + ", ".join(f"{t} {n}" for t, n in rep["doc_types"].items()))
    L.append("Coverage: " + ", ".join(f"{k} {v}" for k, v in rep["coverage"].items()))
    if rep.get("graph"):
        g = rep["graph"]
        L.append(f"Graph: {g.get('edges', 0)} usable edges ({g.get('from_neo4j', '?')} from Neo4j, "
                 f"{g.get('from_rules', 0)} from rules) {g.get('by_relation', {})}"
                 + (f", skipped {g['skipped']}" if g.get("skipped") else "")
                 + (f", ERROR {g['error']}" if g.get("error") else ""))
    L.append("\nProblems for retrieval (most important first):" if rep["problems"] else "\nNo problems found.")
    L += [f"  {i}. {p}" for i, p in enumerate(rep["problems"], 1)]
    L += [f"Note: {n}" for n in rep["notes"]]
    return "\n".join(L)
