"""Scheduled source refresh (FR-1) with a maintainer trigger.

    python -m askiisc_retrieval.cli refresh                  # one conditional refresh now (maintainer trigger)
    python -m askiisc_retrieval.cli refresh --ingest         # ... and rebuild the index if any page changed
    python -m askiisc_retrieval.cli refresh --every 24       # scheduler: re-check every 24 hours
    python -m askiisc_retrieval.cli refresh --trigger        # ask a running scheduler to refresh now

Each run re-checks every listed, crawled and linked source with If-None-Match / If-Modified-Since and stores
a page only when its extracted content changed (see versions.py). The outcome of the last run is written to
data/raw/versions/last_refresh.json. A failed refresh or rebuild leaves the previous raw versions and the
previous index snapshot serving, and the status file records the failure (freshness warning, PRD Sec. 6).
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

from . import config

STALE_LOCK_S = 6 * 3600


def _versions_dir() -> Path:
    path = config.RAW_DIR / "versions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def trigger_path() -> Path:
    return _versions_dir() / "REFRESH_NOW"


def status_path() -> Path:
    return _versions_dir() / "last_refresh.json"


def trigger() -> Path:
    """Maintainer trigger for urgent notices: a running scheduler starts a refresh within its poll interval."""
    path = trigger_path()
    path.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8")
    return path


class RefreshLock:
    """One refresh at a time (scheduler, trigger and manual runs share the raw store)."""

    def __init__(self):
        self.path = _versions_dir() / "refresh.lock"

    def __enter__(self):
        if self.path.exists() and time.time() - self.path.stat().st_mtime > STALE_LOCK_S:
            self.path.unlink(missing_ok=True)                 # a crashed run's lock
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f"another refresh is running ({self.path})") from None
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def run_refresh(ingest: bool = False, fetch=None, rebuild=None) -> dict:
    """One refresh run. `fetch` / `rebuild` are injectable for tests."""
    from .snapshot import atomic_write_text
    if fetch is None:
        from .fetch import fetch_all
        fetch = lambda: fetch_all(refresh=True)                # noqa: E731
    status = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ok": False}
    try:
        with RefreshLock():
            trigger_path().unlink(missing_ok=True)
            results = fetch()
            stats = next((r["status"] for r in results if r.get("url") == "(versions)"), {})
            status["stats"] = stats
            status["errors"] = [r for r in results if str(r.get("status", "")).startswith("error")]
            changed = stats.get("new", 0) + stats.get("changed", 0)
            if ingest and changed:
                try:
                    status["index"] = (rebuild or _rebuild)()
                except Exception as exc:
                    # e.g. a local-folder Qdrant locked by the running server: the new raw versions are
                    # stored, the previous snapshot keeps serving, and the warning says the index is behind.
                    status["index"] = f"rebuild failed: {type(exc).__name__}: {exc}"
                    status["freshness_warning"] = (f"{changed} source(s) changed but the index was not rebuilt; "
                                                   "the previous snapshot is still active.")
            elif ingest:
                status["index"] = "unchanged: no source changed, nothing re-embedded"
            status["ok"] = True
    except Exception as exc:                                   # keep the previous snapshot; record a warning
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["freshness_warning"] = ("Refresh failed; the previous raw versions and index snapshot are "
                                       "still active and may be stale.")
        traceback.print_exc()
    status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_text(status_path(), json.dumps(status, indent=2))
    return status


def _rebuild() -> str:
    from .cli import make_embedder, safe_rebuild
    from .ingest_stub import load_page_inputs
    report = safe_rebuild(load_page_inputs(), make_embedder())
    return f"rebuilt: snapshot {report['snapshot']}, {report.get('embedded', report['chunks'])} chunks embedded"


def schedule(every_hours: float, ingest: bool = False, *, poll_s: float = 30, max_runs: int | None = None,
             run=None, sleep=time.sleep, clock=time.monotonic) -> int:
    """Run a refresh now and then every `every_hours` hours; a maintainer trigger (`refresh --trigger`)
    starts one early. Returns the number of runs (only reached when `max_runs` is set, for tests)."""
    run = run or (lambda: run_refresh(ingest=ingest))
    runs, due = 0, clock()
    while max_runs is None or runs < max_runs:
        if clock() >= due or trigger_path().exists():
            status = run()
            runs += 1
            print(f"[refresh] {status.get('finished_at')} ok={status.get('ok')} {status.get('stats', status.get('error'))}",
                  flush=True)
            due = clock() + every_hours * 3600
            continue
        sleep(poll_s)
    return runs
