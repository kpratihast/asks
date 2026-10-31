"""Crawl rules on a fake site (httpx.MockTransport, no network)."""
import httpx

from askiisc_retrieval import crawl as crawl_mod
from askiisc_retrieval.fetch import raw_name

SITE = {
    "https://cds.iisc.ac.in/robots.txt": "User-agent: *\nDisallow: /private/\n",
    "https://cds.iisc.ac.in/": '<a href="/people-all/faculty/">Faculty</a> <a href="/forms/">Forms</a> '
                               '<a href="/files/handbook.pdf">PDF</a> <a href="https://example.com/">Elsewhere</a> '
                               '<a href="/private/x/">Hidden</a> <a href="mailto:a@b.c">Mail</a> '
                               '<a href="https://dsl.cds.iisc.ac.in/">DSL lab</a> <a href="/feed/">RSS</a>',
    "https://cds.iisc.ac.in/people-all/faculty/": '<a href="/people/yogesh-simmhan/">YS</a> <a href="/">Home</a>',
    "https://cds.iisc.ac.in/forms/": '<a href="/forms/ta-bill.docx">TA bill</a>',
    "https://cds.iisc.ac.in/people/yogesh-simmhan/": '<a href="/deeper/">Too deep</a>',
    "https://cds.iisc.ac.in/deeper/": "depth 3",
    "https://dsl.cds.iisc.ac.in/": '<a href="https://dsl.cds.iisc.ac.in/members/">Members</a>',
    "https://dsl.cds.iisc.ac.in/members/": "should not be reached",
}


def _client(requested):
    def handler(req):
        url = str(req.url)
        requested.append(url)
        body = SITE.get(url)
        if body is None:
            return httpx.Response(404)
        ctype = "text/plain" if url.endswith("robots.txt") else "text/html"
        return httpx.Response(200, text=body, headers={"content-type": ctype})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_crawl_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(crawl_mod, "RAW_DIR", tmp_path)
    requested = []
    pages = crawl_mod.crawl(["https://cds.iisc.ac.in/"], 2, 50, _client(requested), raw_name, delay_s=0, log=lambda *a: None)
    urls = {p["url"] for p in pages}
    assert urls == {"https://cds.iisc.ac.in/", "https://cds.iisc.ac.in/people-all/faculty/",
                    "https://cds.iisc.ac.in/forms/", "https://cds.iisc.ac.in/people/yogesh-simmhan/",
                    "https://dsl.cds.iisc.ac.in/"}
    assert not any(u.endswith((".pdf", ".docx")) or "example.com" in u or "/private/" in u or "/feed/" in u
                   for u in requested)                              # skipped without even downloading
    assert "https://dsl.cds.iisc.ac.in/members/" not in requested   # subdomain is a leaf
    assert "https://cds.iisc.ac.in/deeper/" not in requested        # depth limit
    types = {p["url"]: p["doc_type"] for p in pages}
    assert types["https://cds.iisc.ac.in/people/yogesh-simmhan/"] == "people"
    assert types["https://cds.iisc.ac.in/forms/"] == "resources"
    assert (tmp_path / "crawl.json").exists()


def test_crawl_reuses_saved_pages_and_respects_max_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(crawl_mod, "RAW_DIR", tmp_path)
    (tmp_path / raw_name("https://cds.iisc.ac.in/")).write_text(SITE["https://cds.iisc.ac.in/"])
    requested = []
    pages = crawl_mod.crawl(["https://cds.iisc.ac.in/"], 2, 2, _client(requested), raw_name, delay_s=0, log=lambda *a: None)
    assert len(pages) == 2
    assert "https://cds.iisc.ac.in/" not in requested                # frozen copy used, not re-downloaded
