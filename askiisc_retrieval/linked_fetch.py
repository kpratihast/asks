"""Fetch a bounded set of approved links during ingestion."""
from __future__ import annotations

import json
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from . import config
from .fetch import UA, raw_name
from .parse import parse_html
from .web_links import page_key

INDEX = "linked.json"
MAX_BYTES = 2 * 1024 * 1024


def _approved(url, hosts):
    parts = urlsplit(url)
    return (parts.scheme == "https" and parts.netloc.lower() in hosts
            and not parts.query and not parts.username)


NOT_MODIFIED = object()


def _download(client, url, hosts, robots, headers=None):
    """Body of an approved page, NOT_MODIFIED for a 304 to a conditional request, or None."""
    for _ in range(4):
        if not _approved(url, hosts):
            return None
        host = urlsplit(url).netloc.lower()
        if host not in robots:
            response = client.get(f"https://{host}/robots.txt")
            if response.status_code not in {200, 404}:
                robots[host] = None
            else:
                policy = RobotFileParser()
                policy.parse(response.text.splitlines() if response.status_code == 200 else [])
                robots[host] = policy
        if not robots[host] or not robots[host].can_fetch(UA, url):
            return None
        with client.stream("GET", url, headers=headers or None) as response:
            if response.status_code == 304 and headers:
                return NOT_MODIFIED
            if response.is_redirect and "location" in response.headers:
                url = urljoin(url, response.headers["location"])
                continue
            if response.status_code != 200 or "html" not in response.headers.get("content-type", ""):
                return None
            data = bytearray()
            for block in response.iter_bytes():
                if len(data) + len(block) > MAX_BYTES:
                    return None
                data.extend(block)
            return bytes(data)
    return None


def fetch_linked(sources, *, refresh=False, versions=None):
    """Fetch approved linked pages. On refresh each saved page is re-checked conditionally and stored only
    when its content changed (FR-1, FR-3)."""
    if versions is None:
        from .versions import RawVersions
        with RawVersions(config.RAW_DIR, raw_name) as own:
            return fetch_linked(sources, refresh=refresh, versions=own)
    records = []
    robots = {}
    known = {page_key(page["url"]) for page in sources["pages"]}
    crawl = config.RAW_DIR / "crawl.json"
    if crawl.is_file():
        known.update(page_key(row["url"]) for row in json.loads(crawl.read_text()))
    with httpx.Client(timeout=15, follow_redirects=False, headers={"User-Agent": UA}) as client:
        for department, options in sources["linked_pages"].items():
            hosts = set(options["hosts"])
            attempts, seen = 0, set()
            for page in sources["pages"]:
                if page.get("department", sources["department"]) != department or page["scope"] != "public":
                    continue
                if page["doc_type"] not in options["from_types"]:
                    continue
                raw = config.RAW_DIR / raw_name(page["url"])
                if not raw.exists():
                    continue
                parsed = parse_html(page["url"], raw.read_bytes())
                for target in dict.fromkeys(
                        link for section in parsed.sections
                        if page["doc_type"] not in {"schedule", "catalogue"} or section.course_codes
                        for _, link in section.links):
                    if attempts >= options["max_pages"]:
                        break
                    if target in known or target in seen or not _approved(target, hosts):
                        continue
                    seen.add(target)
                    attempts += 1
                    # Keep the approved alias as the indexed URL, even when its host redirects the request.
                    path = config.RAW_DIR / raw_name(target)
                    if refresh or not path.is_file():
                        try:
                            body = _download(client, target, hosts, robots, versions.conditional_headers(target))
                        except (httpx.HTTPError, ValueError):
                            body = None
                        if body is None:
                            print(f"Skipped linked source: {target}", flush=True)
                            continue
                        if body is NOT_MODIFIED:
                            versions.not_modified(target)
                        else:
                            versions.record(target, body)
                    records.append({"url": target, "department": department, "doc_type": options["doc_type"],
                                    "scope": "public"})
    from .versions import write_if_changed
    write_if_changed(config.RAW_DIR / INDEX, json.dumps(records, indent=2))
    return len(records)
