"""Stub ingestion: saved HTML -> sections -> chunks -> embeddings -> Qdrant (+ graph edges, snapshot).
Just enough to give retrieval a realistic index. No crawling, staleness or registry; that is the
ingestion team's pipeline, which will write the same ChunkPayload records."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .chunking import chunk_page, h
from .config import ROOT
from .contract import ChunkPayload, SCOPES
from .parse import parse_html, remove_template_sections

# Pages with less main content than this are treated as empty shells (landing pages whose content is
# images/links, or pages that only link to a PDF). Was 300 while sidebar copies padded every page; after
# template removal, real faculty profiles have 118-287 chars and empty pages 0-79 (crawl, 2026-10-08).
MIN_CONTENT_CHARS = 100
from .snapshot import JsonSnapshotProvider, Snapshot
from .stores import Embedder, JsonGraphStore, QdrantStore
from .graph_rules import GRAPH_RULES_VERSION, build_edges


@dataclass
class PageInput:
    url: str
    html: str | bytes
    doc_type: str
    scope: str
    department: str = "CDS"
    active: bool = True           # False = indexed but superseded (tests version filtering)
    version_tag: str = ""         # distinguishes a synthetic older version of the same URL
    tagged_departments: bool = False  # opt in for reviewed, normalized institute pages


def build_index(pages: list[PageInput], embedder: Embedder, store: QdrantStore, graph_path: Path,
                snapshot_path: Path, chunks_path: Path | None = None) -> dict:
    for page in pages:
        if page.scope not in SCOPES:
            raise ValueError(f"Unknown access scope: {page.scope!r}")
    all_chunks: list[ChunkPayload] = []
    active: set[str] = set()
    report = []
    parsed_all = [parse_html(p.url, p.html, tagged_departments=p.tagged_departments) for p in pages]
    template_removed = remove_template_sections(parsed_all)
    for p, parsed in zip(pages, parsed_all):
        body = "\n".join(s.text for s in parsed.sections)
        if len(body) < MIN_CONTENT_CHARS:
            report.append({"url": p.url, "status": f"NOT INDEXED: only {len(body)} chars of main content "
                           "(the real content is probably in a linked PDF/iframe; tell the ingestion team)"})
            continue
        ownership = "|".join(f"{s.heading}:{s.department or p.department}" for s in parsed.sections)
        version_id = h(f"{p.url}|{h(body)}|{p.version_tag}|scope:{p.scope}|{h(ownership)}")
        chunks = chunk_page(parsed, version_id=version_id, document_id=h(p.url, 12), scope=p.scope,
                            doc_type=p.doc_type, department=p.department, embedding_model=embedder.model_id)
        all_chunks.extend(chunks)
        if p.active:
            active.add(version_id)
        from .query_analysis import term_key
        terms = sorted({c.academic_term for c in chunks if c.academic_term}, key=term_key)
        report.append({"url": p.url, "version_id": version_id, "active": p.active, "sections": len(parsed.sections),
                       "chunks": len(chunks), "departments": sorted({c.department for c in chunks}), "terms": f"{terms[0]}..{terms[-1]} ({len(terms)})" if terms else "-"})

    print(f"Embedding {len(all_chunks)} chunks with {embedder.model_id} ...")
    vectors = embedder.embed_documents([c.embedding_input() for c in all_chunks])
    store.upsert(all_chunks, vectors)

    graph = JsonGraphStore(build_edges(all_chunks))
    graph.save(graph_path)
    snap = Snapshot(snapshot_id="snap-" + h("|".join(sorted(active)) + embedder.model_id + GRAPH_RULES_VERSION, 10),
                    embedding_model=embedder.model_id, graph_version=GRAPH_RULES_VERSION,
                    active_version_ids=frozenset(active),
                    departments=tuple(sorted({c.department for c in all_chunks if c.version_id in active})))
    JsonSnapshotProvider.write(snapshot_path, snap)
    if chunks_path:
        with chunks_path.open("w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    untermed = [c.section_heading for c in all_chunks if c.doc_type == "schedule" and c.academic_term is None]
    return {"pages": report, "chunks": len(all_chunks), "edges": len(graph.edges), "snapshot": snap.snapshot_id,
            "untermed_schedule": untermed, "template_removed": template_removed}


def load_page_inputs(include_fixtures: bool = True) -> list[PageInput]:
    """Real pages from data/raw/ plus the synthetic fixtures."""
    from .fetch import load_sources, raw_name
    from .config import RAW_DIR
    src = load_sources()
    pages, missing = [], []
    for pg in src["pages"]:
        f = RAW_DIR / raw_name(pg["url"])
        if not f.exists():
            missing.append(pg["url"])
            continue
        pages.append(PageInput(pg["url"], f.read_bytes(), pg["doc_type"], pg["scope"],
                               pg.get("department", src["department"])))
    if missing:
        print(f"WARNING: {len(missing)} page(s) not fetched yet (run `fetch` first): {missing}")
    crawl_index = RAW_DIR / "crawl.json"
    if src["crawls"] and crawl_index.exists():
        from urllib.parse import urlsplit
        hosts = {urlsplit(seed).netloc: group["department"]
                 for group in src["crawls"] for seed in group["seeds"]}
        listed = {pg["url"] for pg in src["pages"]}
        for pg in json.loads(crawl_index.read_text(encoding="utf-8")):
            f = RAW_DIR / raw_name(pg["url"])
            if pg["url"] in listed or not f.exists():
                continue
            host = urlsplit(pg["url"]).netloc
            owner = next((hosts[root] for root in sorted(hosts, key=len, reverse=True)
                          if host == root or host.endswith("." + root)), None)
            if owner and pg.get("department", owner) == owner:
                pages.append(PageInput(pg["url"], f.read_bytes(), pg["doc_type"], "public", owner))
    fx_path = ROOT / src.get("fixtures", "fixtures/fixtures.yaml")
    if include_fixtures and fx_path.exists():
        for fx in yaml.safe_load(fx_path.read_text(encoding="utf-8"))["fixtures"]:
            html = (fx_path.parent / fx["file"]).read_text(encoding="utf-8")
            pages.append(PageInput(fx["url"], html, fx["doc_type"], fx["scope"], "CDS",
                                   active=fx.get("active", True), version_tag=fx.get("version_tag", "")))
    admissions_cache = RAW_DIR.parent / 'admissions'
    if (admissions_cache / 'sources.json').is_file():
        from .admissions import cached_pages
        approved = json.loads((admissions_cache / 'sources.json').read_text())
        admissions, _ = cached_pages(approved, admissions_cache)
        refreshed = {p.url for p in admissions}
        pages = [p for p in pages if p.url not in refreshed] + admissions
    from .private_documents import DocumentCatalog
    pages.extend(DocumentCatalog().page_inputs())
    return pages
