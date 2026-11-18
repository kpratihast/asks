"""Cache configured pages and documents under data/raw/ before crawling or following links.
Re-running skips pages already saved unless --refresh is given, so the dev corpus stays stable."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit
import re

import httpx
import yaml

from .config import RAW_DIR, ROOT

UA = "AskIISc-dev/0.1 (DS252 course project; retrieval dev snapshot)"
MAX_SOURCE_BYTES = 8 * 1024 * 1024


def raw_name(url: str) -> str:
    suffix = ".pdf" if urlsplit(url).path.lower().endswith(".pdf") else ".html"
    return hashlib.sha256(url.encode()).hexdigest()[:16] + suffix


def download_source(client: httpx.Client, url: str) -> bytes:
    status, body, _ = fetch_source(client, url)
    return body


def fetch_source(client: httpx.Client, url: str, headers: dict | None = None) -> tuple[int, bytes | None, dict]:
    """GET `url` (optionally conditional). Returns (status, body or None for 304, validators) where
    validators holds the response's ETag / Last-Modified for the next conditional request (FR-1)."""
    body = bytearray()
    with client.stream("GET", url, headers=headers or None) as response:
        validators = {"etag": response.headers.get("etag"), "last_modified": response.headers.get("last-modified")}
        if response.status_code == 304:
            return 304, None, validators
        response.raise_for_status()
        for block in response.iter_bytes():
            if len(body) + len(block) > MAX_SOURCE_BYTES:
                raise ValueError("Source exceeds the download size limit")
            body.extend(block)
    return response.status_code, bytes(body), validators


def refresh_source(client: httpx.Client, versions, url: str) -> str:
    """Conditionally re-fetch one page and store it only if its content changed (FR-1, FR-3).
    Returns "unchanged" (304 or same content hash: nothing written), "new" or "changed"."""
    status, body, validators = fetch_source(client, url, versions.conditional_headers(url))
    if status == 304:
        return versions.not_modified(url)
    return versions.record(url, body, **validators)


def load_sources() -> dict:
    from .departments import DEFAULT, SHARED, source_departments, validate_codes

    path = Path(os.environ.get("ASKIISC_SOURCES_FILE", ROOT / "sources.yaml")).expanduser()
    sources = yaml.safe_load(path.read_text(encoding="utf-8"))
    default = sources.get("department", DEFAULT)
    groups = sources.get("crawls", [])
    if sources.get("crawl"):
        groups = [{"department": default, **sources["crawl"]}, *groups]
    pages = sources["pages"]
    if len({page["url"] for page in pages}) != len(pages):
        raise ValueError("A source URL must have exactly one configured owner")
    hosts = {}
    for group in groups:
        group_hosts = {urlsplit(seed).netloc for seed in group["seeds"]}
        if len(group_hosts) != 1 or "" in group_hosts:
            raise ValueError("Each crawl group requires seeds on one host")
        host = group_hosts.pop()
        if host in hosts:
            raise ValueError(f"Duplicate crawl host: {host}")
        hosts[host] = group["department"]
    validate_codes([default, *(page.get("department", default) for page in pages),
                    *(group["department"] for group in groups)])
    enabled = {*source_departments(default), SHARED}
    linked = sources.get("linked_pages", {})
    validate_codes(linked)
    for owner, options in linked.items():
        if (not isinstance(options["max_pages"], int) or not 0 <= options["max_pages"] <= 50
                or not options["hosts"]
                or any(not re.fullmatch(r"(?:[a-z0-9-]+\.)+[a-z][a-z0-9-]+", host) for host in options["hosts"])):
            raise ValueError(f"Invalid linked page limits or approved hosts for {owner}")
    return {**sources, "department": default,
            "pages": [page for page in pages if page.get("department", default) in enabled],
            "crawls": [group for group in groups if group["department"] in enabled],
            "linked_pages": {owner: opts for owner, opts in linked.items() if owner in enabled}}


def fetch_all(refresh: bool = False, delay_s: float = 1.0) -> list[dict]:
    """Fetch the listed pages, run the configured crawls and fetch approved linked pages.

    Without `refresh`, pages already saved are reused. With `refresh`, every page - listed, crawled and
    linked - is re-checked with a conditional request (If-None-Match / If-Modified-Since); a page is
    written, as a new immutable version, only when its extracted content changed (FR-1, FR-3)."""
    from .versions import RawVersions, write_if_changed
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    index_path = RAW_DIR / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    results = []
    sources = load_sources()
    with RawVersions(RAW_DIR, raw_name) as versions, \
            httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=30) as client:
        for page in sources["pages"]:
            url, path = page["url"], RAW_DIR / raw_name(page["url"])
            if path.exists() and not refresh:
                results.append({"url": url, "status": "cached"})
                continue
            try:
                status = refresh_source(client, versions, url)
            except (httpx.HTTPError, ValueError) as e:
                print(f"  FAILED  {url}  ({e})")
                results.append({"url": url, "status": f"error: {e}"})
                continue
            if status != "unchanged":
                index[url] = {"file": path.name, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "bytes": path.stat().st_size, "content_hash": versions.latest(url)[1]}
            print(f"  {status:9} {url}")
            results.append({"url": url, "status": "saved" if status != "unchanged" else "unchanged"})
            time.sleep(delay_s)                  # be polite to the institute server
        for crawl_cfg in sources["crawls"]:
            from .crawl import crawl
            print(f"\nCrawling {crawl_cfg['seeds']} (depth {crawl_cfg['max_depth']}, max {crawl_cfg['max_pages']} pages) ...")
            found = crawl(crawl_cfg["seeds"], crawl_cfg["max_depth"], crawl_cfg["max_pages"], client, raw_name,
                          delay_s=crawl_cfg.get("delay_s", 0.5),
                          department=crawl_cfg["department"], refresh=refresh, versions=versions)
            results.append({"url": "(crawl)", "status": f"{len(found)} pages"})
            print(f"Crawl: {len(found)} pages")
        if sources["linked_pages"]:
            from .linked_fetch import fetch_linked
            results.append({"url": "(approved links)",
                            "status": f"{fetch_linked(sources, refresh=refresh, versions=versions)} pages"})
        results.append({"url": "(versions)", "status": dict(versions.stats)})
    write_if_changed(index_path, json.dumps(index, indent=2))
    return results
