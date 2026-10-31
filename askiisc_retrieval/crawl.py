"""Bounded crawl of the CDS site. Defines the corpus by the site's own links, NOT by any question set,
so the held-out test set cannot leak into what gets indexed.

Rules (sources.yaml -> crawl):
  * breadth-first from the seeds, up to max_depth clicks, at most max_pages pages
  * hosts: exactly the seed host (e.g. cds.iisc.ac.in) is traversed; subdomains of it (lab sites such as
    dsl.cds.iisc.ac.in) are fetched as LEAVES: their linked page is indexed, their links are not followed
  * HTML only (no PDFs, images, office files); robots.txt is respected; pages already saved are reused
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

SKIP_EXT = re.compile(r"\.(pdf|docx?|xlsx?|pptx?|zip|rar|tar|gz|jpe?g|png|gif|svg|webp|ico|mp4|mp3|css|js|xml|ics)$", re.I)
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


def links_in(html: bytes | str, base: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        out.append(normalise(urljoin(base, href)))
    return out


def crawl(seeds: list[str], max_depth: int, max_pages: int, client: httpx.Client, raw_name,
          delay_s: float = 0.5, respect_robots: bool = True, log=print) -> list[dict]:
    home = urlparse(normalise(seeds[0])).netloc
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
        if u.netloc == home:
            ok = robots.can_fetch("*", url) if robots else True
            return ok, ok
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
                r = client.get(url)
                time.sleep(delay_s)
            except httpx.HTTPError as e:
                log(f"  FAILED  {url} ({type(e).__name__})")
                continue
            if r.status_code != 200 or "html" not in r.headers.get("content-type", "html"):
                continue
            final = normalise(str(r.url))
            if final != url and final in seen:          # redirect to a page we already have
                continue
            body = r.content
            path.write_bytes(body)
            log(f"  saved   d{depth} {url}")
        pages.append({"url": url, "depth": depth, "doc_type": doc_type_for(url, home)})
        if follow and depth < max_depth:
            for link in links_in(body, url):
                if link not in seen:
                    queue.append((link, depth + 1))
    (RAW_DIR / CRAWL_INDEX).write_text(json.dumps(pages, indent=1), encoding="utf-8")
    return pages
