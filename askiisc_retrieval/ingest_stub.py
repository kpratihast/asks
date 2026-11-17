"""Stub ingestion: saved HTML -> sections -> chunks -> embeddings -> Qdrant (+ graph edges, snapshot).
Just enough to give retrieval a realistic index. No crawling, staleness or registry; that is the
ingestion team's pipeline, which will write the same ChunkPayload records."""
from __future__ import annotations

import json
import logging
import re
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

log = logging.getLogger("askiisc.ingest")


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
    pdf_title: str = ""           # set for a crawled PDF: `html` holds the PDF bytes (FR-1)


def build_index(pages: list[PageInput], embedder: Embedder, store: QdrantStore, graph_path: Path,
                snapshot_path: Path, chunks_path: Path | None = None) -> dict:
    for page in pages:
        if page.scope not in SCOPES:
            raise ValueError(f"Unknown access scope: {page.scope!r}")
    all_chunks: list[ChunkPayload] = []
    active: set[str] = set()
    report, failures = [], []

    def failed(p, exc):
        # FR-2: a page whose parser raises is an extraction failure, logged with URL and reason; the
        # rest of the corpus is still indexed and the failed page produces no chunk or graph edge.
        reason = f"{type(exc).__name__}: {exc}"[:300]
        failures.append({"url": p.url, "reason": reason})
        report.append({"url": p.url, "status": f"EXTRACTION FAILED ({reason})"})
        log.error("extraction failed: %s: %s", p.url, reason)

    parsed_ok = []
    for p in pages:
        try:
            html = p.html
            if p.pdf_title:                       # a crawled PDF goes through the PDF extractor (FR-1)
                from .admissions import pdf_html
                html = pdf_html(p.html, p.pdf_title, max_pages=32)
            parsed_ok.append((p, parse_html(p.url, html, tagged_departments=p.tagged_departments)))
        except Exception as exc:
            failed(p, exc)
    template_removed = remove_template_sections([parsed for _, parsed in parsed_ok])
    for p, parsed in parsed_ok:
        try:
            body = "\n".join(s.text for s in parsed.sections)
            if len(body) < MIN_CONTENT_CHARS:
                report.append({"url": p.url, "status": f"NOT INDEXED: only {len(body)} chars of main content "
                               "(the real content is probably in a linked PDF/iframe; tell the ingestion team)"})
                continue
            ownership = "|".join(f"{s.heading}:{s.department or p.department}" for s in parsed.sections)
            links = json.dumps([s.links for s in parsed.sections], ensure_ascii=False)
            version_id = h(f"{p.url}|{h(body)}|{p.version_tag}|scope:{p.scope}|{h(ownership)}|{h(links)}")
            chunks = chunk_page(parsed, version_id=version_id, document_id=h(p.url, 12), scope=p.scope,
                                doc_type=p.doc_type, department=p.department, embedding_model=embedder.model_id)
            from .query_analysis import term_key
            terms = sorted({c.academic_term for c in chunks if c.academic_term}, key=term_key)
            report.append({"url": p.url, "version_id": version_id, "active": p.active, "sections": len(parsed.sections),
                           "chunks": len(chunks), "departments": sorted({c.department for c in chunks}), "terms": f"{terms[0]}..{terms[-1]} ({len(terms)})" if terms else "-"})
            all_chunks.extend(chunks)          # only once the whole page has been processed
            if p.active:
                active.add(version_id)
        except Exception as exc:
            failed(p, exc)

    if not active:
        # Nothing indexable: keep the previous snapshot rather than activating an empty corpus (FR-4).
        raise RuntimeError(f"no page could be indexed ({len(failures)} extraction failures); "
                           "the previous snapshot is still active")
    print(f"Embedding {len(all_chunks)} chunks with {embedder.model_id} ...")
    vectors = embedder.embed_documents([c.embedding_input() for c in all_chunks])
    store.upsert(all_chunks, vectors)

    # Activation order (FR-4): new vectors are added alongside the old ones, then the graph and chunk
    # list are replaced atomically, and the snapshot - which names the graph by hash - is replaced last.
    # Until that final rename, readers keep the previous snapshot, its versions and its graph.
    from .snapshot import atomic_write_text
    graph = JsonGraphStore(build_edges(all_chunks))
    graph_sha256 = graph.save(graph_path)
    if chunks_path:
        atomic_write_text(chunks_path, "".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in all_chunks))
    snap = Snapshot(snapshot_id="snap-" + h("|".join(sorted(active)) + embedder.model_id + GRAPH_RULES_VERSION, 10),
                    embedding_model=embedder.model_id, graph_version=GRAPH_RULES_VERSION,
                    active_version_ids=frozenset(active),
                    departments=tuple(sorted({c.department for c in all_chunks if c.version_id in active})),
                    graph_sha256=graph_sha256)
    JsonSnapshotProvider.write(snapshot_path, snap)
    untermed = [c.section_heading for c in all_chunks if c.doc_type == "schedule" and c.academic_term is None]
    return {"pages": report, "failures": failures, "built_versions": sorted({c.version_id for c in all_chunks}),
            "chunks": len(all_chunks), "edges": len(graph.edges), "snapshot": snap.snapshot_id,
            "untermed_schedule": untermed, "template_removed": template_removed}


def load_page_inputs(include_fixtures: bool = True) -> list[PageInput]:
    """Real pages from data/raw/ plus the synthetic fixtures."""
    from .fetch import load_sources, raw_name
    from .config import RAW_DIR
    from .departments import SHARED, source_departments
    src = load_sources()
    departments = source_departments(src["department"])
    pages, missing = [], []
    normalized_seeds = set()
    for pg in src["pages"]:
        f = RAW_DIR / raw_name(pg["url"])
        if not f.exists():
            missing.append(pg["url"])
            continue
        department = pg.get("department", src["department"])
        mode = pg.get("normalize")
        pdf = pg.get("pdf")
        if pdf:
            from .admissions import pdf_html
            body = pdf_html(f, pdf["title"], max_pages=32, page_ranges=pdf["page_ranges"], departments=departments)
        elif mode:
            from .admissions import normalize_html
            body = normalize_html(f.read_bytes(), pg["url"], mode, departments)
            normalized_seeds.add(pg["url"])
        else:
            body = f.read_bytes()
        pages.append(PageInput(pg["url"], body, pg["doc_type"], pg["scope"], department,
                               tagged_departments=bool(pdf or (mode and department == SHARED))))
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
                title = ""
                if pg.get("format") == "pdf":
                    from urllib.parse import unquote
                    stem = unquote(urlsplit(pg["url"]).path.rstrip("/").rsplit("/", 1)[-1]).rsplit(".", 1)[0]
                    title = re.sub(r"[-_]+", " ", stem).strip() or "PDF document"
                pages.append(PageInput(pg["url"], f.read_bytes(), pg["doc_type"], "public", owner, pdf_title=title))
    from .linked_fetch import INDEX
    linked_index = RAW_DIR / INDEX
    if linked_index.is_file():
        from urllib.parse import urlsplit
        have = {page.url for page in pages}
        for pg in json.loads(linked_index.read_text(encoding="utf-8")):
            settings = src.get("linked_pages", {}).get(pg["department"])
            path = RAW_DIR / raw_name(pg["url"])
            if (settings and urlsplit(pg["url"]).netloc in settings["hosts"]
                    and pg["scope"] == "public" and pg["url"] not in have and path.is_file()):
                pages.append(PageInput(pg["url"], path.read_bytes(), pg["doc_type"], pg["scope"], pg["department"]))
                have.add(pg["url"])
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
        admissions = [page for page in admissions if page.url not in normalized_seeds]
        refreshed = {p.url for p in admissions}
        pages = [p for p in pages if p.url not in refreshed] + admissions
    from .private_documents import DocumentCatalog
    pages.extend(DocumentCatalog().page_inputs())
    return pages
