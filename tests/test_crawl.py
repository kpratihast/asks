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
    assert not any(u.endswith(".docx") or "example.com" in u or "/private/" in u or "/feed/" in u
                   for u in requested)                              # skipped without even downloading
    assert "https://cds.iisc.ac.in/files/handbook.pdf" in requested  # FR-1: a followed PDF link is downloaded
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


PDF_BYTES = b"%PDF-1.4\n% fake but well-formed header\n" + b"0" * 200


def test_fr1_pdfs_are_stored_and_chrome_links_are_not_traversed(tmp_path, monkeypatch):
    """Red-team #10: a followed PDF lands in the raw store (not traversed); nav/header/footer links are skipped."""
    monkeypatch.setattr(crawl_mod, "RAW_DIR", tmp_path)
    site = {
        "https://cds.iisc.ac.in/robots.txt": ("text/plain", b""),
        "https://cds.iisc.ac.in/": ("text/html", b'<header><a href="/header-home/">Home</a></header>'
                                    b'<nav><a href="/nav-menu-item/">Menu</a></nav>'
                                    b'<div role="navigation"><a href="/aria-menu/">Menu</a></div>'
                                    b'<main><a href="/courses/ds252_syllabus.pdf">Syllabus</a>'
                                    b'<a href="/courses/">Courses</a></main>'
                                    b'<footer><a href="/footer-copyright/">Legal</a></footer>'),
        "https://cds.iisc.ac.in/courses/": ("text/html", b"<main>Course list</main>"),
        "https://cds.iisc.ac.in/courses/ds252_syllabus.pdf": ("application/pdf", PDF_BYTES),
    }
    requested = []

    def handler(req):
        requested.append(str(req.url))
        ctype, body = site.get(str(req.url), ("text/html", None))
        return httpx.Response(200, content=body, headers={"content-type": ctype}) if body is not None \
            else httpx.Response(404)
    pages = crawl_mod.crawl(["https://cds.iisc.ac.in/"], 2, 50, httpx.Client(transport=httpx.MockTransport(handler)),
                            raw_name, delay_s=0, log=lambda *a: None)
    urls = {p["url"] for p in pages}
    pdf = "https://cds.iisc.ac.in/courses/ds252_syllabus.pdf"
    assert pdf in urls and (tmp_path / raw_name(pdf)).read_bytes() == PDF_BYTES
    assert next(p for p in pages if p["url"] == pdf)["format"] == "pdf"
    chrome = ("/header-home/", "/nav-menu-item/", "/aria-menu/", "/footer-copyright/")
    assert not any(u.endswith(chrome) for u in urls) and not any(u.endswith(chrome) for u in requested)
    log = (tmp_path / "crawl.json").read_text()
    assert not any(c in log for c in chrome)
    assert crawl_mod.links_in(site["https://cds.iisc.ac.in/"][1], "https://cds.iisc.ac.in/") == \
        [pdf, "https://cds.iisc.ac.in/courses/"]


def test_fr1_html_disguised_as_pdf_is_not_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(crawl_mod, "RAW_DIR", tmp_path)
    def handler(req):
        if str(req.url).endswith("robots.txt"):
            return httpx.Response(404)
        if str(req.url).endswith(".pdf"):
            return httpx.Response(200, content=b"<html>login wall</html>", headers={"content-type": "application/pdf"})
        return httpx.Response(200, content=b'<main><a href="/x.pdf">x</a></main>', headers={"content-type": "text/html"})
    pages = crawl_mod.crawl(["https://cds.iisc.ac.in/"], 2, 50, httpx.Client(transport=httpx.MockTransport(handler)),
                            raw_name, delay_s=0, log=lambda *a: None)
    assert [p["url"] for p in pages] == ["https://cds.iisc.ac.in/"]
