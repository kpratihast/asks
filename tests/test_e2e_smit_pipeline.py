"""End to end over the ingestion team's REAL code: pipeline.py (ingestion v3) ingests sample CDS-style pages
into PostgreSQL (real server) + Qdrant (in-memory, same client API) + a Neo4j stand-in that executes the
pipeline's Cypher (MERGE / SET / DELETE semantics); then retrieval (ingestion backend) and serving
(/v1/ask, fake LLM) run on top of exactly what it wrote.

Faked, and why: embeddings (no BGE-M3 download in CI; one hash embedder used by BOTH the pipeline and
retrieval, so vectors are consistent), MinIO (put_object only), the Neo4j server, the LLM.
Real: the pipeline's parser, chunker, payloads, graph extraction, Postgres writes and write order;
retrieval's adapter, filters, snapshot, graph mapping; serving.

Needs:  SMIT_PIPELINE_DIR (folder with pipeline.py, contract.py, settings.py, schema.sql;
        default ../ingestion) and ASKIISC_TEST_PG_DSN (an empty Postgres database). Skipped otherwise.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "tests" / "fixtures" / "ingestion_pages"
SMIT = Path(os.environ.get("SMIT_PIPELINE_DIR", ROOT.parent / "ingestion"))
DSN = os.environ.get("ASKIISC_TEST_PG_DSN")

pytestmark = pytest.mark.skipif(not (DSN and (SMIT / "pipeline.py").exists() and (SMIT / "settings.py").exists()),
                                reason="needs ASKIISC_TEST_PG_DSN and the ingestion v3 folder")
BASE = "https://cds.iisc.ac.in"
URLS = {"schedule.html": f"{BASE}/courses/schedule", "faculty_simmhan.html": f"{BASE}/faculty/simmhan",
        "admissions.html": f"{BASE}/admissions/research-degrees"}


# ---------------------------------------------------------------- Neo4j stand-in
EDGE_RE = re.compile(r"MERGE \(a:(\w+) \{(\w+): \$a\}\) MERGE \(b:(\w+) \{(\w+): \$b\}\) "
                     r"MERGE \(a\)-\[r:(\w+)(?: \{[^}]*\})?\]->\(b\) SET r \+= \$props")


class FakeNeo4j:
    """Executes the statements pipeline.py v3 issues. Nodes keyed by (label, key value); edges by
    (src node, rel, dst node, identity props)."""

    def __init__(self):
        self.nodes: dict[tuple, dict] = {}
        self.edges: dict[tuple, dict] = {}
        self.fail_on: str | None = None             # raise when a statement contains this text

    def session(self):
        return self

    def begin_transaction(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        pass

    def close(self):
        pass

    def run(self, q: str, **p):
        if self.fail_on and self.fail_on in q:
            raise RuntimeError("neo4j unavailable (simulated)")
        if q.startswith("MERGE (d:Department"):
            self.nodes.setdefault(("Department", p["name"]), {"name": p["name"]})
        elif q.startswith("MERGE (f:Faculty {key: $key})"):
            n = self.nodes.get(("Faculty", p["key"]))
            if n is None:
                self.nodes[("Faculty", p["key"])] = {"key": p["key"], "name": p["name"], "aliases": [p["name"]]}
            else:
                if p["name"] not in n["aliases"]:
                    n["aliases"].append(p["name"])
                if len(p["name"]) > len(n["name"]):
                    n["name"] = p["name"]
        elif (m := EDGE_RE.match(q)):
            la, ka, lb, kb, rel = m.groups()
            for lab, key, val in ((la, ka, p["a"]), (lb, kb, p["b"])):
                self.nodes.setdefault((lab, val), {key: val})
            ident = tuple(sorted((k[2:], v) for k, v in p.items() if k.startswith("i_")))
            k = ((la, p["a"]), rel, (lb, p["b"]), ident)
            self.edges.setdefault(k, dict(ident)).update(p["props"])
        elif "r.version_id <> $vid DELETE r" in q:
            self.edges = {k: v for k, v in self.edges.items()
                          if not (v.get("source_url") == p["url"] and v.get("version_id") != p["vid"])}
        elif "r.version_id = $vid DELETE r" in q:
            self.edges = {k: v for k, v in self.edges.items()
                          if not (v.get("source_url") == p["url"] and v.get("version_id") == p["vid"])}
        elif "AND NOT (n)--() DELETE n" in q:
            used = {n for k in self.edges for n in (k[0], k[2])}
            self.nodes = {k: v for k, v in self.nodes.items() if k[0] == "Department" or k in used}
        else:
            raise AssertionError(f"FakeNeo4j does not know this statement: {q[:80]}")

    def records(self) -> list[dict]:
        return [{"la": [s[0]], "pa": dict(self.nodes.get(s, {})), "t": rel, "pr": dict(pr),
                 "lb": [d[0]], "pb": dict(self.nodes.get(d, {}))}
                for (s, rel, d, _), pr in self.edges.items()]


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[Key] = Body


# ---------------------------------------------------------------- fixtures
def load_pipeline():
    sys.path.insert(0, str(SMIT))
    try:
        spec = importlib.util.spec_from_file_location("ingestion_pipeline", SMIT / "pipeline.py")
        pl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pl)
    finally:
        sys.path.remove(str(SMIT))
        for m in ("contract", "settings"):                 # do not shadow retrieval's own modules
            sys.modules.pop(m, None)
    return pl


@pytest.fixture(scope="module")
def pl():
    return load_pipeline()


@pytest.fixture(scope="module")
def stack(pl):
    import psycopg2
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    from askiisc_retrieval.stores import HashEmbedder

    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS document_chunks, document_versions, active_snapshots CASCADE")
        cur.execute((SMIT / "schema.sql").read_text())
    emb = HashEmbedder(dim=1024, model_id="BAAI/bge-m3")
    qd = QdrantClient(":memory:")
    qd.create_collection(pl.COLLECTION_NAME, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
    d = psycopg2.extensions.parse_dsn(DSN)
    pl.PG_CONFIG.clear()
    pl.PG_CONFIG.update(dbname=d["dbname"], user=d.get("user"), password=d.get("password"),
                        host=d.get("host", "127.0.0.1"), port=int(d.get("port", 5432)))
    import tempfile
    snap_dir = tempfile.mkdtemp(prefix="snapshots-")
    pl.QDRANT_CLIENT, pl.s3 = qd, pl.LocalSnapshotStore(snap_dir)      # the real default snapshot store

    async def embed(texts, batch_size=16):
        return emb.embed_documents(texts)
    pl.generate_embeddings_batched = embed
    neo = FakeNeo4j()

    def ingest(name, html=None):
        html = html or (PAGES / name).read_text(encoding="utf-8")
        from bs4 import BeautifulSoup
        title = BeautifulSoup(html, "html.parser").title.get_text(strip=True)
        asyncio.run(pl.process_and_index_document(URLS[name], html, title, neo, "CDS", "M.Tech/PhD", "html"))

    for name in URLS:
        ingest(name)
    pl.record_active_snapshot_manifest()
    return {"pl": pl, "qd": qd, "neo": neo, "emb": emb, "ingest": ingest, "snap_dir": Path(snap_dir)}


def retriever_for(stack, rules=False):
    """rules=False: Neo4j alone (no rule fallback), to prove the ingestion graph is enough."""
    from askiisc_retrieval.graph_neo4j import Neo4jGraphStore
    from askiisc_retrieval.graph_rules import build_edges
    from askiisc_retrieval.ingestion_adapter import IngestionQdrantStore, PostgresRegistry
    from askiisc_retrieval.service import build_ingestion_retriever
    from askiisc_retrieval.stores import KeywordReranker
    reg = PostgresRegistry(DSN)
    store = IngestionQdrantStore(stack["pl"].COLLECTION_NAME, client=stack["qd"], registry=reg)
    graph = Neo4jGraphStore(stack["neo"].records, ttl_s=-1,
                            supplement=(lambda: build_edges(store.chunks())) if rules else None)
    r = build_ingestion_retriever(embedder=stack["emb"], reranker=KeywordReranker(), store=store,
                                  registry=reg, graph=graph)
    r.snapshots.ttl = -1
    return r, store, graph


def graph_of(r):
    return r.graph


def rels(graph):
    return {(e.src, e.rel, e.dst) for e in graph.edges}


# ---------------------------------------------------------------- pure functions (no stores)
def test_course_codes_rooms_and_e0_codes(pl):
    assert pl.extract_course_codes("Course ID: DS 252 | Venue: CDS 202 | E0 251 and DS-221") == \
        ["DS252", "E0251", "DS221"]
    assert pl.row_course_codes({"Course ID": "DS 252", "Course Title": "Cloud", "Venue": "CDS 202"}) == ["DS252"]
    assert pl.row_course_codes({"Code": "E0 251", "Room": "CDS 102"}) == ["E0251"]
    assert pl.row_course_codes({"Course": "DS 307 Ethics in AI", "Venue": "CDS 419"}) == ["DS307"]


def test_terms_from_every_schedule_heading_format(pl):
    cases = {"Course Schedule for Aug - Dec 2026 term": "2026-AUG", "Jan 2024 - Apr 2024": "2024-JAN",
             "Feb-May (2021)": "2021-JAN", "Oct 2020 -Jan 2021": "2020-AUG", "May-June (summer) 2019": "2019-SUM",
             "(Aug-Dec, 2016)": "2016-AUG", "August 2026 semester": "2026-AUG", "Spring 2025": "2025-JAN",
             "Mid-term examination 2026": None, "Notes": None, "Long term plan 2030": None}
    assert {t: pl.normalize_academic_term(t) for t in cases} == cases


def test_people_split_and_keys_match_retrieval(pl):
    from askiisc_retrieval.graph_rules import person
    assert pl.split_people("S.Vadhiyar & Y. Simmhan") == ["S.Vadhiyar", "Y. Simmhan"]
    assert pl.split_people("Danish Pruthi, Aditya Gopalan and Prof. Sashikumaar Ganesan") == \
        ["Danish Pruthi", "Aditya Gopalan", "Sashikumaar Ganesan"]
    assert pl.split_people("TBA") == []
    for name in ["Yogesh Simmhan", "Y. Simmhan", "Prof. Y Simmhan", "S.Vadhiyar", "Venkatesh Babu R",
                 "R. Venkatesh Babu", "Deepak N. Subramani", "Konduri Aditya"]:
        assert "Person:" + pl.faculty_key(name) == person(name), name


def test_sections_terms_and_nesting_in_the_parser(pl):
    chunks, page = pl.parse_html_to_chunks((PAGES / "faculty_simmhan.html").read_text(), URLS["faculty_simmhan.html"],
                                           "Yogesh Simmhan – CDS")
    heads = [c["section_heading"] for c in chunks]
    assert heads == ["Yogesh Simmhan", "Yogesh Simmhan > Research Interests", "Yogesh Simmhan > Teaching"]
    assert page["person"] == "Yogesh Simmhan"
    sched, _ = pl.parse_html_to_chunks((PAGES / "schedule.html").read_text(), URLS["schedule.html"], "Course Schedule")
    notes = [c for c in sched if c["section_heading"].endswith("Notes")]
    assert len(notes) == 1 and notes[0]["academic_term"] is None              # term reset after the 2024 table
    assert notes[0]["text"].count("Audit registrations") == 1                 # <p> inside <li> read once
    rows = {(tuple(c["course_codes"]), c["academic_term"]) for c in sched if c["is_table_row"]}
    assert (("DS252",), "2026-AUG") in rows and (("DS252",), "2024-AUG") in rows and (("DS221",), "2024-AUG") in rows


# ---------------------------------------------------------------- what the pipeline wrote
def test_raw_snapshots_are_stored_once_per_content(stack):
    files = list((stack["snap_dir"] / stack["pl"].BUCKET_NAME / "snapshots").glob("*.md"))
    assert len(files) == len(URLS) and all(f.stat().st_size > 100 for f in files)
    before = {f: f.stat().st_mtime_ns for f in files}
    stack["pl"].s3.put_object(Bucket=stack["pl"].BUCKET_NAME, Key=f"snapshots/{files[0].name}", Body=b"x")
    assert files[0].stat().st_mtime_ns == before[files[0]] and files[0].read_bytes() != b"x"   # immutable


def test_payload_is_the_full_contract_with_clean_codes(stack):
    _, store, _ = retriever_for(stack)
    assert store.layout.startswith("full contract") and store.stats.bad_course_codes == 0
    rows = [c for c in store.chunks() if c.doc_type == "schedule" and c.course_codes == ["DS252"]]
    assert sorted(c.academic_term for c in rows) == ["2024-AUG", "2026-AUG"]
    assert not any("CDS" in code for c in store.chunks() for code in c.course_codes)


def test_graph_has_one_node_per_person_research_and_labs(stack):
    _, _, graph = retriever_for(stack)
    r = rels(graph)
    assert ("Course:DS221", "TAUGHT_BY", "Person:s vadhiyar") in r and ("Course:DS221", "TAUGHT_BY", "Person:y simmhan") in r
    assert ("Person:y simmhan", "WORKS_ON", "Topic:cloud computing") in r
    assert ("Person:y simmhan", "HEADS", "Lab:DREAM:Lab (Distributed systems Research on Emerging Applications & Machines)") in r
    assert ("Person:d subramani", "HEADS", "Lab:QUEST Lab") in r
    assert not any(n.startswith("Course:CDS") for e in graph.edges for n in (e.src, e.dst))
    fac = stack["neo"].nodes[("Faculty", "y simmhan")]
    assert fac["name"] == "Yogesh Simmhan" and {"Y. Simmhan", "Yogesh Simmhan"} <= set(fac["aliases"])
    assert graph.stats["from_rules"] == 0 and graph.stats["skipped"].get("not a course code (room number?)", 0) == 0


# ---------------------------------------------------------------- retrieval on the stores
def test_current_term_row_is_cited_first(stack):
    r, _, _ = retriever_for(stack)
    res = r.retrieve("Who teaches DS 252 this semester?", ["public"])
    top = res.passages[0].chunk
    assert top.course_codes == ["DS252"] and top.academic_term == "2026-AUG" and "Simmhan" in top.text
    assert not any(p.chunk.course_codes == ["DS252"] and p.chunk.academic_term == "2024-AUG" for p in res.passages)


def test_taught_together_from_the_neo4j_graph_alone(stack):
    r, _, _ = retriever_for(stack, rules=False)
    res = r.retrieve("Which course have Sathish Vadhiyar and Yogesh Simmhan taught together?", ["public"])
    assert any(p.chunk.course_codes == ["DS221"] for p in res.passages)
    # in this tiny corpus vector search finds the row too, so check the graph path itself: one course
    # linked to BOTH people, with the row as evidence
    hits = graph_of(r).expand(["Person:s vadhiyar", "Person:y simmhan"], r.snapshots.active().active_version_ids,
                              ["public"], 2, None, target_term=None, explicit=False, now_term="2026-AUG")
    both = [h for h in hits if len(h.seeds) == 2 and h.edge.src == "Course:DS221"]
    assert both and set(both[0].edge.chunk_ids) <= {c.chunk_id for c in r.store.chunks() if c.course_codes == ["DS221"]}


def test_research_interest_section_is_cited_with_its_heading(stack):
    r, _, _ = retriever_for(stack)
    res = r.retrieve("Who works on edge computing?", ["public"])
    assert any(p.chunk.section_heading.endswith("Research Interests") for p in res.passages)


def test_unrelated_question_is_refused(stack):
    r, _, _ = retriever_for(stack)
    assert r.retrieve("What is the monthly mess fee at the IISc hostels?", ["public"]).evidence_sufficient is False


def test_check_ingestion_finds_no_problems_in_v3(stack):
    from askiisc_retrieval.ingestion_check import check, render
    r, store, graph = retriever_for(stack)
    rep = check(store, graph, set(r.snapshots.active().active_version_ids), "postgres")
    print(render(rep))
    assert rep["missing_fields"] == [] and rep["pages_with_one_prose_heading"] == 0
    assert not any("room number" in p or "research-area or lab" in p for p in rep["problems"]), rep["problems"]


# ---------------------------------------------------------------- versions and failures (run last: they re-ingest)
def test_zz_reingestion_replaces_the_old_version_everywhere(stack):
    import psycopg2
    url = URLS["schedule.html"]
    html = (PAGES / "schedule.html").read_text(encoding="utf-8").replace(
        "<td>Danish Pruthi</td>", "<td>Partha Pratim Talukdar</td>")
    stack["ingest"]("schedule.html", html)
    r, store, graph = retriever_for(stack)
    texts = " ".join(p.chunk.text for p in r.retrieve("Who is the instructor for Ethics in AI DS 307?", ["public"]).passages)
    assert "Talukdar" in texts and "Pruthi" not in texts
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT version_id FROM document_versions WHERE source_url = %s AND is_active", (url,))
        (active,), = cur.fetchall()
        cur.execute("SELECT count(*) FROM document_versions WHERE source_url = %s", (url,))
        assert cur.fetchone()[0] == 2
    assert {c.version_id for c in store.chunks() if c.source_url == url} == {str(active)}       # old vectors removed
    assert {v["version_id"] for v in stack["neo"].edges.values() if v.get("source_url") == url} == {str(active)}
    assert ("Course:DS307", "TAUGHT_BY", "Person:d pruthi") not in rels(graph)


def test_zz_failed_reingestion_leaves_the_old_version_intact(stack):
    import psycopg2
    url = URLS["schedule.html"]
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT version_id FROM document_versions WHERE source_url = %s AND is_active", (url,))
        (before,), = cur.fetchall()
    html = (PAGES / "schedule.html").read_text(encoding="utf-8").replace("<td>CDS 202</td>", "<td>CDS 102</td>")
    stack["neo"].fail_on = "TAUGHT_BY"                       # graph write fails half-way through the page
    try:
        with pytest.raises(RuntimeError):
            stack["ingest"]("schedule.html", html)
    finally:
        stack["neo"].fail_on = None
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT version_id FROM document_versions WHERE source_url = %s AND is_active", (url,))
        assert cur.fetchall() == [(before,)]                  # still the old version
    _, store, _ = retriever_for(stack)
    assert {c.version_id for c in store.chunks() if c.source_url == url} == {str(before)}   # new points removed
    r, _, _ = retriever_for(stack)
    assert r.retrieve("Who teaches DS 252 this semester?", ["public"]).passages[0].chunk.academic_term == "2026-AUG"


# ---------------------------------------------------------------- serving on top
def test_serving_answers_from_the_stores_and_refuses_off_topic(stack):
    import httpx
    from fastapi.testclient import TestClient

    from askiisc_serving.app import create_app
    from askiisc_serving.auth import KeyStore
    from askiisc_serving.generator import Generator
    from askiisc_serving.retrieval_client import LocalRetrieval
    from askiisc_serving.settings import Settings
    from tests.test_serving import FakeModel

    r, _, _ = retriever_for(stack)
    model = FakeModel("DS 252 is taught by Yogesh Simmhan [S1].")
    s = Settings(vllm_url="http://model", served_model="test-model")
    gen = Generator(s, client=httpx.AsyncClient(transport=httpx.MockTransport(model.handler)))
    key = "integration-guest-key-0001"
    app = create_app(s, retrieval=LocalRetrieval(r), generator=gen,
                     keys=KeyStore({key: {"tenant": "applicants", "role": "guest"}}))
    with TestClient(app) as c:
        h = {"Authorization": f"Bearer {key}"}
        ok = c.post("/v1/ask", json={"question": "Who teaches DS 252 this semester?"}, headers=h).json()
        assert ok["status"] == "answered" and ok["citations"][0]["url"].endswith("/courses/schedule")
        assert ok["citations"][0]["academic_term"] == "2026-AUG"
        assert "Yogesh Simmhan" in model.requests[-1]["messages"][1]["content"]
        n = len(model.requests)
        off = c.post("/v1/ask", json={"question": "What is the monthly mess fee at the IISc hostels?"}, headers=h).json()
        assert off["status"] == "abstained" and off["reason"] == "insufficient_evidence" and len(model.requests) == n
        src = c.get(f"/v1/sources/{ok['citations'][0]['chunk_ids'][0]}", headers=h)
        assert src.status_code == 200 and "Simmhan" in src.json()["text"]


def test_validate_live_report_sections_run(stack):
    """tools/validate_live.py sections 3-4 on this stack (the real-model run happens on the laptop)."""
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import validate_live
    finally:
        sys.path.remove(str(ROOT / "tools"))
    r, _, _ = retriever_for(stack)
    lines = validate_live.report_retrieval(r, run_eval=True, dev_csv=None)
    text = "\n".join(lines)
    assert "## 3. Spot questions" in text and "## 4. Dev questions" in text and "Summary:" in text
    assert "REFUSE" in text                                   # the mess-fee question


def test_embedder_local_runs_bge_m3_in_process(monkeypatch):
    """EMBEDDER=local: the pipeline encodes with sentence-transformers (normalised), no TEI call."""
    import types
    pl = load_pipeline()
    calls = {}

    class FakeST:
        def __init__(self, name, device):
            calls["model"], calls["device"] = name, device

        def encode(self, texts, batch_size, normalize_embeddings):
            calls["normalize"] = normalize_embeddings
            import numpy as np
            return np.ones((len(texts), 4))
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    monkeypatch.setattr(pl.settings, "EMBEDDER", "local")
    pl._LOCAL_MODEL = None
    out = asyncio.run(pl.generate_embeddings_batched(["a", "b"]))
    assert out == [[1.0] * 4] * 2 and calls == {"model": "BAAI/bge-m3", "device": "cpu", "normalize": True}
