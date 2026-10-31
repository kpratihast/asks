"""Command line entry point.

    python -m askiisc_retrieval.cli fetch            # save the CDS pages once (needs internet)
    python -m askiisc_retrieval.cli ingest           # parse, chunk, embed, index (offline)
    python -m askiisc_retrieval.cli ask "Who teaches DS 252?" [--scope public student] [--no-graph]
    python -m askiisc_retrieval.cli eval [--k 5] [--no-graph]
    python -m askiisc_retrieval.cli show-chunks [--grep DS252]   # inspect what got indexed
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys

from . import config


def make_embedder():
    """ASKIISC_EMBEDDER=hash gives a tiny fake embedder for checking the plumbing without the
    2.3 GB model download. Its retrieval quality is meaningless; re-run `ingest` without it."""
    import os
    if os.environ.get("ASKIISC_EMBEDDER", "").lower() == "hash":
        from .stores import HashEmbedder
        print("!! Using the fake hash embedder (plumbing check only, results are not meaningful)", file=sys.stderr)
        return HashEmbedder(dim=config.EMBED_DIM)
    from .stores import BGEM3Embedder
    print(f"Loading {config.EMBED_MODEL} on CPU ...", flush=True, file=sys.stderr)
    return BGEM3Embedder(config.EMBED_MODEL, config.EMBED_MAX_SEQ, config.EMBED_BATCH)


def make_reranker():
    """ASKIISC_RERANKER=none disables reranking. With the fake hash embedder a fake reranker is used too."""
    import os
    choice = os.environ.get("ASKIISC_RERANKER", "").lower()
    if choice == "none":
        return None
    if os.environ.get("ASKIISC_EMBEDDER", "").lower() == "hash":
        from .stores import KeywordReranker
        return KeywordReranker()
    from .stores import BGEReranker
    print(f"Loading reranker {config.RERANK_MODEL} (first run downloads it) ...", flush=True, file=sys.stderr)
    return BGEReranker(config.RERANK_MODEL, config.RERANK_MAX_LEN, device=config.DEVICE)


def make_store(reset: bool = False):
    """Qdrant server when QDRANT_URL is set (Docker), otherwise local mode in data/qdrant/."""
    from .stores import QdrantStore
    if config.QDRANT_URL:
        # Windows tries IPv6 (::1) first for "localhost"; the container listens on IPv4 only, so
        # every request waited ~2 s for the fallback. Use the IPv4 address directly.
        url = config.QDRANT_URL.replace("://localhost", "://127.0.0.1")
        try:
            store = QdrantStore(config.COLLECTION, config.EMBED_DIM, url=url)
        except Exception as e:
            sys.exit(f"Cannot reach Qdrant at {url} ({type(e).__name__}). "
                     "Is the container running? `docker compose up -d qdrant`")
        if reset:
            store.reset()
    else:
        if reset and config.QDRANT_PATH.exists():
            shutil.rmtree(config.QDRANT_PATH)
        config.DATA.mkdir(exist_ok=True)
        store = QdrantStore(config.COLLECTION, config.EMBED_DIM, path=config.QDRANT_PATH)
    print(f"Vector store: {store.backend}", file=sys.stderr)
    return store


def _open_stack():
    if config.BACKEND == "ingestion":
        from .service import build_ingestion_retriever
        r = build_ingestion_retriever()
        return r, r.store
    from .retriever import Retriever
    from .snapshot import JsonSnapshotProvider
    from .stores import JsonGraphStore
    if not config.SNAPSHOT_PATH.exists():
        sys.exit("No index yet. Run `fetch` and then `ingest` first.")
    emb = make_embedder()
    store = make_store()
    graph = JsonGraphStore.load(config.GRAPH_PATH)
    return Retriever(emb, store, JsonSnapshotProvider(config.SNAPSHOT_PATH), graph,
                     reranker=make_reranker()), store


def cmd_fetch(a):
    from .fetch import fetch_all
    res = fetch_all(refresh=a.refresh)
    listed = [r for r in res if r["url"] != "(crawl)"]
    print(f"\nListed pages: {sum(r['status'] in ('saved', 'cached') for r in listed)}/{len(listed)} available")
    for r in res:
        if r["url"] == "(crawl)":
            print(f"Crawl: {r['status']}")
    print(f"Raw pages are in {config.RAW_DIR}")


def cmd_ingest(a):
    from .ingest_stub import build_index, load_page_inputs
    pages = load_page_inputs()
    if not pages:
        sys.exit("Nothing to ingest. Run `fetch` first.")
    config.DATA.mkdir(exist_ok=True)
    emb = make_embedder()
    store = make_store(reset=True)                       # stub rebuilds from scratch each time
    rep = build_index(pages, emb, store, config.GRAPH_PATH, config.SNAPSHOT_PATH, config.CHUNKS_PATH)
    store.close()
    print()
    for p in rep["pages"]:
        if "version_id" in p:
            flag = "" if p["active"] else "  (inactive: superseded test version)"
            print(f"  {p['chunks']:4d} chunks  terms {p['terms']:<22} {p['url']}{flag}")
        else:
            print(f"  {p['status']}: {p['url']}")
    if rep.get("template_removed"):
        parts = ", ".join(f"'{h}' x{n}" for h, n in sorted(rep["template_removed"].items(), key=lambda kv: -kv[1]))
        print(f"\nRemoved repeated site-template sections: {parts}")
    if rep["untermed_schedule"]:
        print(f"\n!! {len(rep['untermed_schedule'])} schedule chunk(s) have NO academic_term (term logic will not "
              "apply to them). Headings, e.g.:")
        for hd in sorted(set(rep["untermed_schedule"]))[:8]:
            print("     ", hd)
    print(f"\n{rep['chunks']} chunks, {rep['edges']} graph edges, snapshot {rep['snapshot']}")
    print(f"Inspect chunks in {config.CHUNKS_PATH}")


def _print_result(res, full: bool):
    print(f"\nsnapshot {res.snapshot_id} | term {res.target_term} | "
          f"{'DEGRADED: ' + res.degraded_reason if res.degraded else 'graph ok'} | timings ms {res.timings_ms}")
    if res.rerank_error:
        print(f"rerank FAILED ({res.rerank_error}); vector order kept")
    elif res.rerank_top is not None:
        verdict = {True: "SUFFICIENT", False: "INSUFFICIENT -> abstain", None: "threshold not calibrated"}
        print(f"best rerank score {res.rerank_top:.3f} | evidence: {verdict[res.evidence_sufficient]}")
    for i, p in enumerate(res.passages, 1):
        c = p.chunk
        extra = f"  [{p.graph_path}]" if p.graph_path else ""
        rr = f" rerank {p.rerank_score:.3f} |" if p.rerank_score is not None else ""
        print(f"\n#{i} cited as [{p.marker}] |{rr} vector {p.score:.3f} via {p.via}{extra}"
              f"\n     {c.title} > {c.section_heading}"
              f"\n     {c.source_url}  term={c.academic_term} scope={c.access_scope} version={c.version_id}")
        text = c.text if full else (c.text[:300] + (" ..." if len(c.text) > 300 else ""))
        print("     " + text.replace("\n", "\n     "))
    from .citations import render_sources
    print("\n" + render_sources(res))


def cmd_ask(a):
    r, store = _open_stack()
    res = r.retrieve(a.question, a.scope, use_graph=not a.no_graph, use_rerank=not a.no_rerank)
    if a.json:
        from .citations import result_to_dict
        print(json.dumps(result_to_dict(res), indent=2, ensure_ascii=False))
    elif a.context:
        from .citations import format_context
        print(format_context(res))
    else:
        _print_result(res, a.full)
    store.close()


def cmd_eval(a):
    from .eval_recall import run
    r, store = _open_stack()
    from pathlib import Path
    qpath = Path(a.questions) if a.questions else config.ROOT / "dev_questions.yaml"
    if not qpath.exists():
        sys.exit(f"Question file not found: {qpath}")
    print(f"Questions: {qpath.name}")
    run(r, k=a.k, use_graph=not a.no_graph, use_rerank=not a.no_rerank, out=a.out, questions_path=qpath)
    store.close()


def cmd_rebuild_graph(a):
    """Rebuild graph.json from the saved chunks, without re-embedding (needed after graph-rule changes)."""
    from collections import Counter

    from .contract import ChunkPayload
    from .graph_rules import GRAPH_RULES_VERSION, build_edges
    from .snapshot import JsonSnapshotProvider, Snapshot
    from .stores import JsonGraphStore
    if not config.CHUNKS_PATH.exists():
        sys.exit("Run `ingest` first.")
    chunks = [ChunkPayload(**json.loads(line))
              for line in config.CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    graph = JsonGraphStore(build_edges(chunks))
    graph.save(config.GRAPH_PATH)
    old = JsonSnapshotProvider(config.SNAPSHOT_PATH).active()
    JsonSnapshotProvider.write(config.SNAPSHOT_PATH, Snapshot(old.snapshot_id, old.embedding_model,
                                                              GRAPH_RULES_VERSION, old.active_version_ids))
    by_rel = Counter(e.rel for e in graph.edges)
    people = {n for e in graph.edges for n in (e.src, e.dst) if n.startswith("Person:")}
    print(f"{len(graph.edges)} edges from {len(chunks)} chunks ({GRAPH_RULES_VERSION}): "
          + ", ".join(f"{r} {n}" for r, n in by_rel.most_common()) + f" | {len(people)} people")
    labs = sorted({(e.src, e.dst) for e in graph.edges if e.rel == "HEADS"})
    for src, dst in labs:
        print(f"  {src} -HEADS-> {dst}")


def cmd_serve(a):
    import uvicorn

    from .service import build_retriever, create_app
    app = create_app(build_retriever())                 # load models once, before taking requests
    print(f"Serving retrieval ({config.BACKEND} backend) on http://{a.host}:{a.port}  "
          f"(POST /retrieve, POST /verify, GET /health)", file=sys.stderr)
    uvicorn.run(app, host=a.host, port=a.port, workers=1, log_level="warning")


def cmd_check_ingestion(a):
    """Compare the ingestion team's stores with what retrieval needs; print a report to send them."""
    from .ingestion_adapter import IngestionQdrantStore, IngestionSnapshotProvider
    from .ingestion_check import check, render
    from .service import _graph, _registry
    reg = _registry()
    url = config.INGEST_QDRANT_URL.replace("://localhost", "://127.0.0.1")
    try:
        store = IngestionQdrantStore(config.INGEST_COLLECTION, url=url, departments=config.INGEST_DEPARTMENTS,
                                     embedding_model=config.EMBED_MODEL, registry=reg)
    except Exception as e:
        sys.exit(f"Cannot read {config.INGEST_COLLECTION} at {url}: {type(e).__name__}: {e}")
    snaps = IngestionSnapshotProvider(store, reg, embedding_model=config.EMBED_MODEL)
    snap = snaps.active()
    rep = check(store, _graph(store), set(snap.active_version_ids), snaps.source)
    print(json.dumps(rep, indent=2) if a.json else render(rep))


def cmd_show(a):
    if not config.CHUNKS_PATH.exists():
        sys.exit("Run `ingest` first.")
    n = 0
    for line in config.CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        blob = (c["section_heading"] + " " + c["text"]).upper().replace(" ", "")
        if a.grep and a.grep.upper().replace(" ", "") not in blob:
            continue
        n += 1
        print(f"- [{c['doc_type']}/{c['access_scope']}/{c['academic_term']}] {c['section_heading']}\n  {c['text'][:200]}")
    print(f"\n{n} chunks")


def cmd_diagnose(a):
    """What did the parser keep from one fetched page, and what did each boilerplate rule remove?"""
    from .fetch import raw_name
    from .parse import drop_report, parse_html
    f = config.RAW_DIR / raw_name(a.url)
    if not f.exists():
        sys.exit(f"Not fetched: {a.url}")
    html = f.read_bytes()
    page = parse_html(a.url, html)
    kept = sum(len(s.text) for s in page.sections)
    print(f"{a.url}\n  raw HTML {len(html) // 1024} KB | content read from <{page.root}> | "
          f"kept {len(page.sections)} sections, {kept} chars | footer-phrase lines removed: {page.boiler_lines}")
    print("  (site-template sections such as sidebars are removed later, across all pages, by `ingest`)")
    print("  removed by boilerplate rules (chars):")
    for sel, n in drop_report(html):
        print(f"    {n:7d}  {sel}")
    for s in page.sections[:12]:
        print(f"  - [{s.academic_term}] {s.heading[:70]} :: {s.text[:90]!r}")


def cmd_explain(a):
    """Where does a page rank for a question? Vector rank among ALL eligible chunks, and reranker score.
    For diagnosing misses: tells 'never a candidate' (vector) apart from 'candidate but scored low' (rerank)."""
    from .rerank_text import rerank_input
    r, store = _open_stack()
    snap = r.snapshots.active()
    qvec = r.embedder.embed_query(a.question)
    hits = store.search(qvec, 5000, snap.active_version_ids, a.scope, snap.embedding_model)
    want = a.url.rstrip("/")
    mine = [(i + 1, c, s) for i, (c, s) in enumerate(hits) if c.source_url.rstrip("/") == want]
    print(f"\n{len(hits)} eligible chunks. Vector candidates passed to ranking: top {r.candidates}; "
          f"reranker pool: top {r.rerank_pool}.")
    if not mine:
        sys.exit(f"No eligible chunk from {a.url} (not indexed, wrong scope, or not in the active snapshot).")
    for rank, c, s in mine:
        line = f"  vector rank {rank:4d}  score {s:.3f}  {c.section_heading[:70]}"
        if r.reranker is not None:
            texts = list(dict.fromkeys([c.embedding_input(), rerank_input(c)]))
            line += f"  | rerank {max(r.reranker.score(a.question, texts)):.3f}"
        print(line)
    print("\n  for comparison, vector top 5:")
    for i, (c, s) in enumerate(hits[:5], 1):
        print(f"  {i:4d}  {s:.3f}  {c.source_url}  {c.section_heading[:50]}")
    store.close()


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")         # Windows consoles default to cp1252
    p = argparse.ArgumentParser(prog="askiisc_retrieval")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--refresh", action="store_true"); f.set_defaults(fn=cmd_fetch)
    sub.add_parser("ingest").set_defaults(fn=cmd_ingest)
    q = sub.add_parser("ask"); q.add_argument("question")
    q.add_argument("--scope", nargs="+", default=["public"], help="user scopes, e.g. public student")
    q.add_argument("--no-graph", action="store_true"); q.add_argument("--full", action="store_true")
    q.add_argument("--no-rerank", action="store_true")
    q.add_argument("--context", action="store_true", help="print only the [S1]-labelled block sent to the generator")
    q.add_argument("--json", action="store_true", help="print the full result (citations, passages) as JSON")
    q.set_defaults(fn=cmd_ask)
    e = sub.add_parser("eval"); e.add_argument("--k", type=int, default=5)
    e.add_argument("--no-graph", action="store_true"); e.add_argument("--out", default=None)
    e.add_argument("--no-rerank", action="store_true")
    e.add_argument("--questions", default=None, help="question file (default: dev_questions.yaml)")
    e.set_defaults(fn=cmd_eval)
    sv = sub.add_parser("serve", help="HTTP service: POST /retrieve, POST /verify, GET /health")
    sv.add_argument("--host", default="127.0.0.1"); sv.add_argument("--port", type=int, default=8090)
    sv.set_defaults(fn=cmd_serve)
    ci = sub.add_parser("check-ingestion", help="report how the ingestion output matches retrieval's needs")
    ci.add_argument("--json", action="store_true"); ci.set_defaults(fn=cmd_check_ingestion)
    sub.add_parser("rebuild-graph", help="rebuild graph edges from saved chunks (no re-embedding)") \
        .set_defaults(fn=cmd_rebuild_graph)
    s = sub.add_parser("show-chunks"); s.add_argument("--grep", default=None); s.set_defaults(fn=cmd_show)
    ex = sub.add_parser("explain", help="where a page ranks for a question (vector rank + rerank score)")
    ex.add_argument("question"); ex.add_argument("url")
    ex.add_argument("--scope", nargs="+", default=["public"]); ex.set_defaults(fn=cmd_explain)
    dg = sub.add_parser("diagnose", help="show what the parser kept/removed for one fetched page")
    dg.add_argument("url"); dg.set_defaults(fn=cmd_diagnose)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
