"""Retrieval over the ingestion team's format (manual 2026-10-22), with a fake collection built exactly
like theirs: payload {id, version_id, source_url, department, chunk_index, text, is_active, is_superseded}."""
import pytest

from askiisc_retrieval.graph_neo4j import Neo4jGraphStore, edges_from_records
from askiisc_retrieval.ingestion_adapter import (DictRegistry, IngestionQdrantStore, IngestionSnapshotProvider,
                                                 point_uuid)
from askiisc_retrieval.ingestion_check import check, render
from askiisc_retrieval.retriever import Retriever
from askiisc_retrieval.service import build_ingestion_retriever, create_app
from askiisc_retrieval.stores import HashEmbedder, KeywordReranker

B = "https://cds.iisc.ac.in"
V1, V2, V3, VOLD, VCSA = (point_uuid(f"v{i}") for i in range(5))
ROWS = [  # (key, version, url, dept, idx, text, active, superseded)
    ("venky", V1, f"{B}/faculty/venky", "CDS", 0,
     "# Faculty Profile: R. Venkatesh Babu\n\nProfessor, Department of Computational and Data Sciences.\n"
     "Research Interests: Video analytics, Computer Vision, Deep Learning.", True, False),
    ("simmhan", V2, f"{B}/faculty/simmhan", "CDS", 0,
     "# Faculty Profile: Yogesh Simmhan\n\nProfessor. Research Interests: distributed systems, cloud computing, "
     "Internet of Things.", True, False),
    ("desc", V3, f"{B}/courses/descriptions/", "CDS", 0,
     "# Course Descriptions\n\n## DS 252 Cloud Computing\n\nInstructor: Yogesh Simmhan. Prerequisites: DS 221. "
     "Covers virtualization, elastic computing and cloud storage.", True, False),
    ("old", VOLD, f"{B}/courses/m-tech/", "CDS", 0,
     "# M.Tech requirements\n\nOLD-VERSION: M.Tech students need 64 credits.", True, True),
    ("csa", VCSA, "https://csa.iisc.ac.in/courses/", "CSA", 0,
     "# CSA courses\n\nCSA-ONLY: credits and courses of another department.", True, False),
]
PID = {k: point_uuid("p-" + k) for k, *_ in ROWS}
CHUNK = {k: point_uuid("c-" + k) for k, *_ in ROWS}


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    from qdrant_client import QdrantClient, models
    emb = HashEmbedder(dim=64, model_id="BAAI/bge-m3")
    client = QdrantClient(path=str(tmp_path_factory.mktemp("ing") / "q"))
    client.create_collection("iisc_policy_passages",
                             vectors_config=models.VectorParams(size=64, distance=models.Distance.COSINE))
    pts = []
    for key, v, url, dept, idx, text, act, sup in ROWS:
        pts.append(models.PointStruct(id=PID[key], vector=emb.embed_query(text), payload={
            "id": PID[key], "version_id": v, "source_url": url, "department": dept, "chunk_index": idx,
            "text": text, "is_active": act, "is_superseded": sup}))
    client.upsert("iisc_policy_passages", points=pts)
    reg = DictRegistry(active={V1, V2, V3, VCSA},           # VOLD superseded, so not active
                       chunks={PID[k]: (CHUNK[k], None, "BAAI/bge-m3") for k in PID})
    yield {"client": client, "emb": emb, "reg": reg}
    client.close()


def records(with_alias=True):
    fac = lambda n, al=None: (["Faculty"], {"name": n, **({"aliases": al} if al else {})})
    rec = lambda a, t, b, chunk, ver, **kw: {"la": a[0], "pa": a[1], "t": t, "lb": b[0], "pb": b[1],
                                             "pr": {"chunk_id": chunk, "version_id": ver, "source_url": "x", **kw}}
    return [
        rec(fac("Venky", ["R. Venkatesh Babu"] if with_alias else None), "RESEARCHES_IN",
            (["ResearchArea"], {"name": "Computer Vision"}), CHUNK["venky"], V1, academic_year="2026-2027"),
        rec((["Course"], {"code": "DS252"}), "REQUIRES_PREREQUISITE", (["Course"], {"code": "DS221"}), CHUNK["desc"], V3),
        rec((["Course"], {"code": "DS252"}), "TAUGHT_BY", fac("Yogesh Simmhan"), CHUNK["desc"], V3,
            academic_term="2026-AUG"),
        rec(fac("Yogesh Simmhan"), "MEMBER_OF", (["Department"], {"name": "CDS"}), None, V2),   # no chunk -> skipped
    ]


def store(fake, registry="reg"):
    return IngestionQdrantStore(client=fake["client"], registry=fake["reg"] if registry == "reg" else registry)


def retriever(fake, graph=None, **kw):
    s = store(fake)
    return Retriever(fake["emb"], s, IngestionSnapshotProvider(s, fake["reg"]), graph, today_term="2026-AUG",
                     reranker=KeywordReranker(), rerank_min=None, **kw)


def test_missing_fields_are_derived(fake):
    s = store(fake)
    by = {c.source_url: c for c in s.chunks()}
    venky = by[f"{B}/faculty/venky"]
    assert venky.title == "Faculty Profile: R. Venkatesh Babu"
    assert venky.doc_type == "people" and venky.access_scope == "public"
    assert venky.chunk_id == CHUNK["venky"]                              # from the Postgres map
    desc = by[f"{B}/courses/descriptions/"]
    assert desc.doc_type == "catalogue" and desc.course_codes == ["DS252", "DS221"]
    assert desc.academic_term is None and desc.embedding_model == "BAAI/bge-m3"
    assert s.stats.mapped_chunk_ids == len(ROWS) and "text" in s.stats.native_fields


def test_only_active_current_department_chunks_are_returned(fake):
    r = retriever(fake)
    res = r.retrieve("How many credits do students need?", ["public"])
    texts = " ".join(p.chunk.text for p in res.passages)
    assert "OLD-VERSION" not in texts and "CSA-ONLY" not in texts
    assert res.passages and r.audit_violations == 0


def test_without_postgres_active_versions_come_from_qdrant_flags(fake):
    s = store(fake, registry=None)
    snaps = IngestionSnapshotProvider(s, None)
    snap = snaps.active()
    assert snaps.source == "qdrant flags" and VOLD not in snap.active_version_ids and V1 in snap.active_version_ids
    assert s.get([PID["venky"]])[0].chunk_id == PID["venky"]             # point id stands in for chunk id


def test_people_search_uses_the_derived_page_type(fake):
    res = retriever(fake).retrieve("Which professor works on computer vision?", ["public"])
    assert any("venky" in p.chunk.source_url for p in res.passages)


def test_neo4j_records_map_to_retrieval_edges():
    edges, aliases, skipped = edges_from_records(records())
    rels = {(e.src, e.rel, e.dst) for e in edges}
    assert ("Person:venky", "WORKS_ON", "Topic:computer vision") in rels
    assert ("Course:DS252", "REQUIRES", "Course:DS221") in rels
    assert ("Course:DS252", "TAUGHT_BY", "Person:y simmhan") in rels
    taught = next(e for e in edges if e.rel == "TAUGHT_BY")
    assert taught.academic_term == "2026-AUG" and taught.chunk_ids == (CHUNK["desc"],)
    assert next(e for e in edges if e.rel == "WORKS_ON").academic_term is None    # "2026-2027" is not a term
    assert skipped["no chunk_id/version_id"] == 1


def test_aliases_let_full_names_find_nickname_nodes():
    g = Neo4jGraphStore(lambda: records(with_alias=True))
    assert g.find_people("What does Venkatesh Babu work on?") == ["Person:venky"]
    assert Neo4jGraphStore(lambda: records(with_alias=False)).find_people("Venkatesh Babu") == []


def test_graph_evidence_resolves_to_ingestion_chunks(fake):
    g = Neo4jGraphStore(lambda: records())
    r = retriever(fake, graph=g, top_k=1, rerank_pool=1)
    res = r.retrieve("Who teaches DS 252 and what are its prerequisites?", ["public"])
    assert not res.degraded
    assert any("descriptions" in p.chunk.source_url for p in res.passages)


def test_neo4j_down_gives_vector_only_results_marked_degraded(fake):
    def boom():
        raise ConnectionError("bolt refused")
    g = Neo4jGraphStore(boom)
    res = retriever(fake, graph=g).retrieve("Who teaches DS 252?", ["public"])
    assert res.degraded and "neo4j unavailable" in res.degraded_reason and res.passages


def test_last_good_graph_survives_a_later_outage():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] > 1:
            raise ConnectionError("down")
        return records()
    g = Neo4jGraphStore(flaky, ttl_s=-1)            # always stale: every call tries to reload
    assert g.find_people("Venky") == ["Person:venky"]                    # reload fails, old copy serves
    assert g.error and g.stats["edges"] == 3


def test_check_report_names_the_gaps(fake):
    s = store(fake)
    rep = check(s, Neo4jGraphStore(lambda: records(with_alias=False)), {V1, V2, V3, VCSA}, "postgres")
    text = render(rep)
    assert "academic_term" in rep["missing_fields"] and "section_heading" in rep["missing_fields"]
    assert any("academic_term" in p for p in rep["problems"])
    assert any("one-word name" in p for p in rep["problems"])          # 'Venky'
    assert not any("TAUGHT_BY" in p for p in rep["problems"])          # the fake graph has it
    assert "Problems for retrieval" in text


def test_http_service(fake):
    from fastapi.testclient import TestClient
    s = store(fake)
    r = build_ingestion_retriever(embedder=fake["emb"], reranker=KeywordReranker(), store=s,
                                  registry=fake["reg"], graph=Neo4jGraphStore(lambda: records()))
    r.rerank_min = 0.3
    app = TestClient(create_app(r))
    out = app.post("/retrieve", json={"question": "What are the prerequisites for DS 252?"}).json()
    assert out["citations"] and out["citations"][0]["marker"] == "S1"
    assert "[S1]" in out["context"] and out["sources_text"].startswith("Sources:")
    assert isinstance(out["evidence_sufficient"], bool) and "service" in out["timings_ms"]
    v = app.post("/verify", json={"answer": "Needs DS 221 [S1] [S9].", "markers": ["S1", "S2"]}).json()
    assert v == {"used": ["S1"], "unknown": ["S9"], "cited": True, "ok": False}
    h = app.get("/health").json()
    assert h["status"] == "ok" and h["chunks"] == len(ROWS) and h["graph"]["edges"] == 3
    assert app.post("/retrieve", json={"question": "x", "scopes": ["superuser"]}).status_code == 422


def test_old_v2_graph_is_validated_not_trusted():
    """Ingestion v2 graphs: co-instructors in one node, room numbers as courses, no research edges."""
    from askiisc_retrieval.graph_neo4j import Neo4jGraphStore
    from askiisc_retrieval.stores import Edge
    ev = {"chunk_id": "c1", "version_id": "v1", "source_url": "u", "academic_term": "2024-AUG"}
    recs = [
        {"la": ["Course"], "pa": {"code": "DS221"}, "t": "TAUGHT_BY", "pr": ev,
         "lb": ["Faculty"], "pb": {"name": "S.Vadhiyar & Y. Simmhan"}},
        {"la": ["Course"], "pa": {"code": "CDS202"}, "t": "TAUGHT_BY", "pr": ev,
         "lb": ["Faculty"], "pb": {"name": "Yogesh Simmhan"}},
    ]
    rule_edge = Edge("Person:y simmhan", "WORKS_ON", "Topic:cloud", ("c2",), "v1", "public", None)
    g = Neo4jGraphStore(lambda: recs, ttl_s=-1, supplement=lambda: [rule_edge])
    r = {(e.src, e.rel, e.dst) for e in g.edges}
    assert ("Course:DS221", "TAUGHT_BY", "Person:s vadhiyar") in r and ("Course:DS221", "TAUGHT_BY", "Person:y simmhan") in r
    assert not any(e.src == "Course:CDS202" for e in g.edges)
    assert g.stats["from_rules"] == 1                                   # v2: no WORKS_ON/HEADS -> fallback used
    recs.append({"la": ["Faculty"], "pa": {"key": "y simmhan", "name": "Yogesh Simmhan"}, "t": "RESEARCHES_IN",
                 "pr": ev, "lb": ["ResearchArea"], "pb": {"name": "edge computing"}})
    g2 = Neo4jGraphStore(lambda: recs, ttl_s=-1, supplement=lambda: [rule_edge])
    assert g2.stats["from_rules"] == 0                                  # v3 graph: fallback off
    assert ("Person:y simmhan", "WORKS_ON", "Topic:edge computing") in {(e.src, e.rel, e.dst) for e in g2.edges}
