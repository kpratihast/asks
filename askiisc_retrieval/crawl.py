"""Bounded crawl of the CDS site. Defines the corpus by the site's own links, NOT by any question set,
so the held-out test set cannot leak into what gets indexed.

Rules (sources.yaml -> crawl):
  * breadth-first from the seeds, up to max_depth clicks, at most max_pages pages
  * hosts: exactly the seed host (e.g. cds.iisc.ac.in) is traversed; subdomains of it (lab sites such as
    dsl.cds.iisc.ac.in) are fetched as LEAVES: their linked page is indexed, their links are not followed
  * HTML pages are traversed; a linked PDF on the seed host (or a lab subdomain) within the depth limit is
    downloaded into the raw store as a LEAF and handed to the PDF parser at ingestion (FR-1); images and
    office files are skipped; robots.txt is respected; pages already saved are reused
  * links inside the shared site chrome (<nav>, <header>, <footer>, and ARIA navigation/banner/contentinfo
    regions) are not traversed, so the crawl follows each page's own content links only (FR-1)
"""
from __future__ import annotations

import json
import re
import time
import urllib.robotparser
from collections import deque
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .config import RAW_DIR
from .fetch import MAX_SOURCE_BYTES

PDF_EXT = re.compile(r"\.pdf$", re.I)
SKIP_EXT = re.compile(r"\.(docx?|xlsx?|pptx?|zip|rar|tar|gz|jpe?g|png|gif|svg|webp|ico|mp4|mp3|css|js|xml|ics)$", re.I)
SKIP_PATH = re.compile(r"/(wp-admin|wp-login|wp-json|wp-content|feed|xmlrpc|tag|author|comments)(/|$)|/page/\d+/?$", re.I)
CRAWL_INDEX = "crawl.json"

# doc_type from the URL path (first match wins); everything else is a generic "page"
TYPE_RULES = [
    (r"^/people/", "people"), (r"^/people-all/", "people"), (r"^/courses/schedule", "schedule"),
    (r"^/courses/descriptions", "catalogue"), (r"^/courses/", "policy"), (r"^/admissions/", "admissions"),
    (r"^/research", "research"), (r"^/academics/calendar", "calendar"), (r"^/academics/", "policy"),
    (r"^/events", "events"), (r"^/forms", "resources"), (r"^/internal-resources/", "resources"),
    (r"^/contact", "contact"), (r"^/(serc|about|opportunities)", "about"),
]


def normalise(url: str) -> str:
    url, _ = urldefrag(url.strip())
    u = urlparse(url)
    path = u.path or "/"
    if not SKIP_EXT.search(path) and not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
        path += "/"                                     # WordPress canonical form: trailing slash
    return u._replace(scheme="https", netloc=u.netloc.lower(), path=path, query="", params="").geturl()


def doc_type_for(url: str, home_host: str) -> str:
    u = urlparse(url)
    if u.netloc != home_host:
        return "research"                               # lab subdomain sites
    for pat, t in TYPE_RULES:
        if re.search(pat, u.path):
            return t
    return "page"


CHROME_TAGS = ("nav", "header", "footer")
CHROME_ROLES = {"navigation", "banner", "contentinfo"}


def in_site_chrome(a) -> bool:
    """True for a link inside the shared navigation, header or footer (repeated on every page)."""
    for parent in a.parents:
        if parent.name in CHROME_TAGS or (parent.get("role") or "").lower() in CHROME_ROLES:
            return True
    return False


def links_in(html: bytes | str, base: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.find_all("a", href=True):
        if in_site_chrome(a):
            continue
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        out.append(normalise(urljoin(base, href)))
    return out


def crawl(seeds: list[str], max_depth: int, max_pages: int, client: httpx.Client, raw_name,
          delay_s: float = 0.5, respect_robots: bool = True, log=print, *, department: str | None = None) -> list[dict]:
    home = urlparse(normalise(seeds[0])).netloc
    if any(urlparse(normalise(seed)).netloc != home for seed in seeds):
        raise ValueError("Use a separate crawl group for each department host")
    robots = None
    if respect_robots:
        robots = urllib.robotparser.RobotFileParser()
        try:
            r = client.get(f"https://{home}/robots.txt")
            robots.parse(r.text.splitlines() if r.status_code == 200 else [])
        except httpx.HTTPError:
            robots.parse([])

    def allowed(url: str) -> tuple[bool, bool]:
        """(fetch it?, follow its links?)"""
        u = urlparse(url)
        if u.scheme not in ("http", "https") or SKIP_EXT.search(u.path) or SKIP_PATH.search(u.path):
            return False, False
        pdf = bool(PDF_EXT.search(u.path))
        if u.netloc == home:
            ok = robots.can_fetch("*", url) if robots else True
            return ok, ok and not pdf                   # a PDF is downloaded, never traversed
        if u.netloc.endswith("." + home):
            return True, False                          # lab subdomain: leaf only
        return False, False

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    queue = deque((normalise(s), 0) for s in seeds)
    seen: set[str] = set()
    pages: list[dict] = []
    while queue and len(pages) < max_pages:
        url, depth = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        fetch_ok, follow = allowed(url)
        if not fetch_ok:
            continue
        path = RAW_DIR / raw_name(url)
        if path.exists():
            body = path.read_bytes()                    # frozen snapshot: never re-download
        else:
            try:
                for _ in range(10):
                    seen.add(url)
                    r = client.get(url, follow_redirects=False)
                    time.sleep(delay_s)
                    if r.status_code not in {301, 302, 303, 307, 308}:
                        break
                    target = normalise(urljoin(url, r.headers.get("location", "")))
                    if target in seen or not allowed(target)[0]:
                        r = None
                        break
                    url = target
                else:
                    r = None
                if r is None:
                    continue
                path = RAW_DIR / raw_name(url)
                _, follow = allowed(url)
            except httpx.HTTPError as e:
                log(f"  FAILED  {url} ({type(e).__name__})")
                continue
            ctype = r.headers.get("content-type", "html").lower()
            is_pdf = bool(PDF_EXT.search(urlparse(url).path)) or "application/pdf" in ctype
            if r.status_code != 200 or not ("html" in ctype or is_pdf):
                continue
            if is_pdf and (not r.content.startswith(b"%PDF") or len(r.content) > MAX_SOURCE_BYTES):
                log(f"  SKIPPED {url} (not a PDF or over the size limit)")
                continue
            final = normalise(str(r.url))
            if not allowed(final)[0]:
                continue
            if final != url and final in seen:
                continue
            seen.add(final)
            body = r.content
            path.write_bytes(body)
            log(f"  saved   d{depth} {url}")
        entry = {"url": url, "depth": depth, "doc_type": doc_type_for(url, home)}
        if body.startswith(b"%PDF"):
            entry["format"] = "pdf"                     # parsed by the PDF extractor at ingestion
            follow = False
        pages.append(entry)
        if follow and depth < max_depth:
            for link in links_in(body, url):
                if link not in seen:
                    queue.append((link, depth + 1))
    if department:
        for page in pages:
            page["department"] = department
    index = RAW_DIR / CRAWL_INDEX
    previous = json.loads(index.read_text(encoding="utf-8")) if index.exists() and department else []
    def belongs_to_host(page):
        host = urlparse(page["url"]).netloc
        return host == home or host.endswith("." + home)
    retained = [page for page in previous if not belongs_to_host(page)]
    index.write_text(json.dumps(retained + pages, indent=1), encoding="utf-8")
    return pages
