"""Fetch the fixed page list once and save raw HTML to data/raw/. Not a crawler: no link following.
Re-running skips pages already saved unless --refresh is given, so the dev corpus stays stable."""
from __future__ import annotations

import hashlib
import json
import time

import httpx
import yaml

from .config import RAW_DIR, ROOT

UA = "AskIISc-dev/0.1 (DS252 course project; retrieval dev snapshot)"


def raw_name(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16] + ".html"


def load_sources() -> dict:
    return yaml.safe_load((ROOT / "sources.yaml").read_text(encoding="utf-8"))


def fetch_all(refresh: bool = False, delay_s: float = 1.0) -> list[dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    index_path = RAW_DIR / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    results = []
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=30) as client:
        for page in load_sources()["pages"]:
            url, path = page["url"], RAW_DIR / raw_name(page["url"])
            if path.exists() and not refresh:
                results.append({"url": url, "status": "cached"})
                continue
            try:
                r = client.get(url)
                r.raise_for_status()
            except httpx.HTTPError as e:
                print(f"  FAILED  {url}  ({e})")
                results.append({"url": url, "status": f"error: {e}"})
                continue
            path.write_bytes(r.content)
            index[url] = {"file": path.name, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "bytes": len(r.content)}
            print(f"  saved   {url}  ({len(r.content) // 1024} KB)")
            results.append({"url": url, "status": "saved"})
            time.sleep(delay_s)                  # be polite to the institute server
        crawl_cfg = load_sources().get("crawl")
        if crawl_cfg:
            from .crawl import crawl
            print(f"\nCrawling {crawl_cfg['seeds']} (depth {crawl_cfg['max_depth']}, max {crawl_cfg['max_pages']} pages) ...")
            found = crawl(crawl_cfg["seeds"], crawl_cfg["max_depth"], crawl_cfg["max_pages"], client, raw_name,
                          delay_s=crawl_cfg.get("delay_s", 0.5))
            results.append({"url": "(crawl)", "status": f"{len(found)} pages"})
            print(f"Crawl: {len(found)} pages")
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return results
