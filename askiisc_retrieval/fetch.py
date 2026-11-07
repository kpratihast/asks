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
    body = bytearray()
    with client.stream("GET", url) as response:
        response.raise_for_status()
        for block in response.iter_bytes():
            if len(body) + len(block) > MAX_SOURCE_BYTES:
                raise ValueError("Source exceeds the download size limit")
            body.extend(block)
    return bytes(body)


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
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    index_path = RAW_DIR / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    results = []
    sources = load_sources()
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=30) as client:
        for page in sources["pages"]:
            url, path = page["url"], RAW_DIR / raw_name(page["url"])
            if path.exists() and not refresh:
                results.append({"url": url, "status": "cached"})
                continue
            try:
                body = download_source(client, url)
            except (httpx.HTTPError, ValueError) as e:
                print(f"  FAILED  {url}  ({e})")
                results.append({"url": url, "status": f"error: {e}"})
                continue
            path.write_bytes(body)
            index[url] = {"file": path.name, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "bytes": len(body)}
            print(f"  saved   {url}  ({len(body) // 1024} KB)")
            results.append({"url": url, "status": "saved"})
            time.sleep(delay_s)                  # be polite to the institute server
        for crawl_cfg in sources["crawls"]:
            from .crawl import crawl
            print(f"\nCrawling {crawl_cfg['seeds']} (depth {crawl_cfg['max_depth']}, max {crawl_cfg['max_pages']} pages) ...")
            found = crawl(crawl_cfg["seeds"], crawl_cfg["max_depth"], crawl_cfg["max_pages"], client, raw_name,
                          delay_s=crawl_cfg.get("delay_s", 0.5),
                          department=crawl_cfg["department"])
            results.append({"url": "(crawl)", "status": f"{len(found)} pages"})
            print(f"Crawl: {len(found)} pages")
    if sources["linked_pages"]:
        from .linked_fetch import fetch_linked
        results.append({"url": "(approved links)", "status": f"{fetch_linked(sources, refresh=refresh)} pages"})
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return results
