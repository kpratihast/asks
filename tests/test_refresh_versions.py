"""#21 (FR-1 conditional refresh) and #22 (FR-3 hash-based versioning) on a fake site, no network.

Replays the red-team procedure: `fetch`, then `fetch --refresh`, recording every request, every file's bytes
and modification time, and the registry rows."""
import hashlib
import json
import os
import sqlite3

import httpx
import pytest

from askiisc_retrieval import config, crawl as crawl_mod, fetch as fetch_mod
from askiisc_retrieval.fetch import raw_name
from askiisc_retrieval.versions import RawVersions, content_hash

HOME = "https://cds.iisc.ac.in/"
LISTED = "https://www.iisc.ac.in/admissions/fees/"


def page(title, text, stamp="10:01", links=""):
    return (f'<html><body><nav><a href="/">Home</a> visited {stamp}</nav><main><h1>{title}</h1><p>{text}</p>'
            f'{links}</main><footer>Generated {stamp}</footer></body></html>').encode()


class Site:
    """A fake institute site. Pages with an ETag answer If-None-Match with 304; others never send one."""

    def __init__(self):
        self.pages = {
            LISTED: [page("Fees", "The PhD application fee is Rs 500 for all applicants."), '"fee-1"'],
            HOME: [page("CDS", "Department of Computational and Data Sciences at IISc.",
                        links='<a href="/academics/">Academics</a> <a href="/people/">People</a>'), '"home-1"'],
            HOME + "academics/": [page("Academics", "CDS offers MTech, MTech Research and PhD programmes."), None],
            HOME + "people/": [page("People", "Faculty of the department are listed here."), '"people-1"'],
        }
        self.requests = []

    def handler(self, request):
        url = str(request.url)
        if url.endswith("/robots.txt"):
            return httpx.Response(404)
        self.requests.append((url, dict(request.headers)))
        if url not in self.pages:
            return httpx.Response(404)
        body, etag = self.pages[url]
        if etag and request.headers.get("if-none-match") == etag:
            return httpx.Response(304, headers={"etag": etag})
        headers = {"content-type": "text/html"}
        if etag:
            headers["etag"] = etag
        return httpx.Response(200, content=body, headers=headers)

    def conditional(self):
        return [u for u, h in self.requests if "if-none-match" in h or "if-modified-since" in h]


@pytest.fixture
def env(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    sources = tmp_path / "sources.yaml"
    sources.write_text(f"""department: CDS
pages:
  - {{url: '{LISTED}', department: CDS, doc_type: admissions, scope: public}}
crawls:
  - {{seeds: ['{HOME}'], department: CDS, max_depth: 1, max_pages: 10, delay_s: 0}}
""")
    monkeypatch.setenv("ASKIISC_SOURCES_FILE", str(sources))
    monkeypatch.setenv("ASKIISC_SOURCE_DEPARTMENTS", "CDS")
    for module in (config, fetch_mod, crawl_mod):
        monkeypatch.setattr(module, "RAW_DIR", raw)
    site = Site()
    real_client = httpx.Client
    monkeypatch.setattr(fetch_mod.httpx, "Client",
                        lambda **kw: real_client(transport=httpx.MockTransport(site.handler), **kw))
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(crawl_mod.time, "sleep", lambda s: None)
    return raw, site


def state(raw):
    """sha256 and mtime of every file under raw/, excluding the version store itself."""
    out = {}
    for path in sorted(raw.rglob("*")):
        if path.is_file() and "versions" not in path.relative_to(raw).parts:
            out[str(path.relative_to(raw))] = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    return out


def rows(raw):
    with sqlite3.connect(raw / "versions" / "registry.sqlite3") as db:
        return db.execute("SELECT url, version, content_hash, object, superseded_at FROM versions ORDER BY url, version").fetchall()


def objects(raw):
    return sorted(p for p in (raw / "versions" / "objects").rglob("*") if p.is_file())


def test_refresh_of_unchanged_site_writes_nothing_and_sends_conditional_requests(env):
    """#22 acceptance (PRD.md:87): re-crawling unchanged pages writes zero snapshot objects and zero registry
    rows; #21: the refresh uses If-None-Match and the crawl is re-checked too."""
    raw, site = env
    fetch_mod.fetch_all(refresh=False, delay_s=0)
    before, before_rows, before_objects = state(raw), rows(raw), objects(raw)
    assert len(before_rows) == 4 and len(before_objects) == 4          # one version per page

    site.requests.clear()
    fetch_mod.fetch_all(refresh=True, delay_s=0)
    requested = [u for u, _ in site.requests]
    assert set(requested) == set(site.pages)                            # crawled pages are re-requested too
    assert set(site.conditional()) == {LISTED, HOME, HOME + "people/"}  # every page that had an ETag
    assert state(raw) == before                                         # no file rewritten: same bytes and mtime
    assert rows(raw) == before_rows and objects(raw) == before_objects  # zero new rows, zero new objects


def test_modified_page_writes_exactly_one_new_version_and_keeps_the_old_one(env):
    raw, site = env
    fetch_mod.fetch_all(refresh=False, delay_s=0)
    old_body = site.pages[LISTED][0]
    site.pages[LISTED] = [page("Fees", "The PhD application fee is Rs 800 for all applicants."), '"fee-2"']
    site.pages[HOME + "academics/"][0] = page("Academics", "CDS offers MTech and PhD programmes only.")
    before = state(raw)

    fetch_mod.fetch_all(refresh=True, delay_s=0)
    after = state(raw)
    changed = {name for name in after if after[name] != before.get(name)}
    assert changed == {raw_name(LISTED), raw_name(HOME + "academics/"), "index.json"}
    assert (raw / raw_name(LISTED)).read_bytes() == site.pages[LISTED][0]          # working copy is current

    by_url = {}
    for url, version, digest, obj, superseded in rows(raw):
        by_url.setdefault(url, []).append((version, obj, superseded))
    assert len(rows(raw)) == 6 and len(objects(raw)) == 6                # exactly one new version per change
    (v1, obj1, sup1), (v2, obj2, sup2) = by_url[LISTED]
    assert (v1, v2) == (1, 2) and sup1 and sup2 is None                  # previous version marked superseded
    assert (raw / "versions" / obj1).read_bytes() == old_body            # ... and kept, byte for byte
    manifest = json.loads((raw / "versions" / "manifest.json").read_text())["sources"][LISTED]
    assert manifest["current"]["version"] == 2 and [v["version"] for v in manifest["superseded"]] == [1]


def test_page_without_etag_is_compared_by_content_hash_and_template_changes_are_not_versions(env):
    raw, site = env
    fetch_mod.fetch_all(refresh=False, delay_s=0)
    url = HOME + "academics/"
    site.pages[url][0] = page("Academics", "CDS offers MTech, MTech Research and PhD programmes.", stamp="23:59")
    before, before_rows = state(raw), rows(raw)
    fetch_mod.fetch_all(refresh=True, delay_s=0)
    assert state(raw) == before and rows(raw) == before_rows            # only nav/footer changed: not a version


def test_copy_saved_before_versioning_is_registered_and_kept(tmp_path):
    """An existing data/raw/ from rt-baseline: the first refresh records the saved copy as version 1."""
    raw = tmp_path / "raw"
    raw.mkdir()
    old = page("Fees", "The PhD application fee is Rs 500 for all applicants.")
    (raw / raw_name(LISTED)).write_bytes(old)
    with RawVersions(raw, raw_name) as versions:
        assert versions.record(LISTED, old) == "unchanged"
        assert versions.record(LISTED, page("Fees", "The PhD application fee is Rs 800.")) == "changed"
        history = versions.history(LISTED)
    assert [h["version"] for h in history] == [1, 2]
    assert (raw / "versions" / history[0]["object"]).read_bytes() == old


def test_content_hash_uses_extracted_content():
    a = page("Fees", "The PhD application fee is Rs 500 for all applicants.")
    assert content_hash(LISTED, a) == content_hash(LISTED, a.replace(b"10:01", b"11:45"))
    assert content_hash(LISTED, a) != content_hash(LISTED, a.replace(b"Rs 500", b"Rs 800"))
    pdf = b"%PDF-1.4 fake"
    assert content_hash(LISTED, pdf) == "sha256:" + hashlib.sha256(pdf).hexdigest()


# ------------------------------------------------------------------ index side (Cloudhack's evidence on #22)
class CountingEmbedder:
    def __init__(self):
        from askiisc_retrieval.stores import HashEmbedder
        self.inner, self.embedded = HashEmbedder(), 0
        self.dim, self.model_id = self.inner.dim, self.inner.model_id

    def embed_documents(self, texts):
        self.embedded += len(texts)
        return self.inner.embed_documents(texts)

    def embed_query(self, text):
        return self.inner.embed_query(text)


def test_rebuild_reembeds_only_changed_pages_and_serves_only_the_current_version(tmp_path):
    """No source change -> nothing re-embedded and the same snapshot; one edited page -> only its chunks are
    embedded, and the superseded version is not active (NFR-2). Its raw version stays in the version store."""
    pytest.importorskip("qdrant_client")
    from askiisc_retrieval.ingest_stub import PageInput, build_index
    from askiisc_retrieval.snapshot import JsonSnapshotProvider
    from askiisc_retrieval.stores import QdrantStore
    from conftest import MTECH, PAGES, SCHEDULE

    def pages(mtech_html):
        return [PageInput(SCHEDULE, (PAGES / "schedule.html").read_text(), "schedule", "public"),
                PageInput(MTECH, mtech_html, "policy", "public")]

    emb = CountingEmbedder()
    store = QdrantStore("refresh", emb.dim, path=tmp_path / "qdrant")
    args = (store, tmp_path / "graph.json", tmp_path / "snapshot.json", tmp_path / "chunks.jsonl")
    try:
        mtech = (PAGES / "mtech.html").read_text()
        first = build_index(pages(mtech), emb, *args)
        assert first["embedded"] == first["chunks"] > 0

        emb.embedded = 0
        again = build_index(pages(mtech), emb, *args)
        assert emb.embedded == 0 and again["embedded"] == 0                   # unchanged corpus: no re-embedding
        assert again["snapshot"] == first["snapshot"]

        edited = mtech.replace("a minimum of 64 credits", "a minimum of 66 credits", 1)     # content edit
        assert edited != mtech
        changed = build_index(pages(edited), emb, *args)
        old_mtech = {p["version_id"] for p in first["pages"] if p["url"] == MTECH}
        new_mtech = {p["version_id"] for p in changed["pages"] if p["url"] == MTECH}
        mtech_chunks = next(p["chunks"] for p in changed["pages"] if p["url"] == MTECH)
        assert old_mtech != new_mtech
        assert emb.embedded == changed["embedded"] == mtech_chunks          # only the edited page
        active = JsonSnapshotProvider(tmp_path / "snapshot.json").active().active_version_ids
        assert new_mtech <= active and not old_mtech & active                 # superseded version never served
    finally:
        store.close()


# ------------------------------------------------------------------ #21 scheduler and maintainer trigger
def test_scheduler_runs_every_interval_and_a_maintainer_trigger_runs_it_early(tmp_path, monkeypatch):
    from askiisc_retrieval import refresh
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    now = [0.0]
    runs = []

    def run():
        runs.append(now[0])
        refresh.trigger_path().unlink(missing_ok=True)
        return {"ok": True, "finished_at": str(now[0])}

    def sleep(seconds):
        now[0] += seconds
        if now[0] == 600:                         # ten minutes in, a maintainer asks for an urgent refresh
            refresh.trigger()

    assert refresh.schedule(24, poll_s=60, max_runs=3, run=run, sleep=sleep, clock=lambda: now[0]) == 3
    assert runs == [0.0, 600.0, 600.0 + 24 * 3600]   # at start, on the trigger, then 24 h after the last run


def test_refresh_records_status_and_keeps_the_previous_snapshot_on_failure(tmp_path, monkeypatch):
    from askiisc_retrieval import refresh
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")

    ok = refresh.run_refresh(ingest=True, fetch=lambda: [{"url": "(versions)", "status": {"changed": 1, "unchanged": 5}}],
                             rebuild=lambda: "rebuilt")
    assert ok["ok"] and ok["stats"] == {"changed": 1, "unchanged": 5} and ok["index"] == "rebuilt"

    quiet = refresh.run_refresh(ingest=True, fetch=lambda: [{"url": "(versions)", "status": {"not_modified": 6}}],
                                rebuild=lambda: pytest.fail("nothing changed: no rebuild"))
    assert quiet["ok"] and quiet["index"].startswith("unchanged")

    def locked():
        raise RuntimeError("Storage folder is already accessed by another instance")
    behind = refresh.run_refresh(ingest=True, fetch=lambda: [{"url": "(versions)", "status": {"changed": 2}}],
                                 rebuild=locked)
    assert behind["index"].startswith("rebuild failed") and "freshness_warning" in behind

    def offline():
        raise OSError("network unreachable")
    failed = refresh.run_refresh(fetch=offline)
    assert not failed["ok"] and "freshness_warning" in failed
    assert json.loads(refresh.status_path().read_text()) == failed


def test_only_one_refresh_runs_at_a_time(tmp_path, monkeypatch):
    from askiisc_retrieval import refresh
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    with refresh.RefreshLock():
        status = refresh.run_refresh(fetch=lambda: pytest.fail("must not run concurrently"))
    assert not status["ok"] and "another refresh is running" in status["error"]
    assert refresh.run_refresh(fetch=lambda: [])["ok"]                 # lock released afterwards


def test_cli_has_refresh_scheduler_and_trigger():
    from askiisc_retrieval import cli
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["refresh", "--help"])
    assert exit_info.value.code == 0


def test_page_both_listed_and_crawled_is_requested_once_per_refresh(tmp_path, monkeypatch, env):
    """#44 (E4): a page listed in sources.yaml that is also a crawl seed was requested twice per refresh."""
    raw, site = env
    sources = tmp_path / "sources.yaml"
    sources.write_text(f"""department: CDS
pages:
  - {{url: '{LISTED}', department: CDS, doc_type: admissions, scope: public}}
  - {{url: '{HOME}', department: CDS, doc_type: page, scope: public}}
crawls:
  - {{seeds: ['{HOME}'], department: CDS, max_depth: 1, max_pages: 10, delay_s: 0}}
""")
    fetch_mod.fetch_all(refresh=False, delay_s=0)
    for refresh in (False, True):
        site.requests.clear()
        fetch_mod.fetch_all(refresh=refresh, delay_s=0)
        requested = [u for u, _ in site.requests]
        assert len(requested) == len(set(requested)), requested        # no URL requested twice in one run
    assert requested.count(HOME) == 1 and set(requested) == set(site.pages)
    crawled = {p["url"] for p in json.loads((raw / "crawl.json").read_text())}
    assert HOME in crawled and HOME + "people/" in crawled             # the crawl still follows the seed's links
