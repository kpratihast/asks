# Source refresh and raw-version store (FR-1, FR-3)

Sources are re-checked every 24 hours with conditional requests and content hashes, and a maintainer can
trigger a refresh at any time. A page is stored, as a new immutable version, only when its extracted content
changed; earlier versions are kept.

## Commands

```bash
python -m askiisc_retrieval.cli refresh               # refresh now (maintainer trigger for urgent notices)
python -m askiisc_retrieval.cli refresh --ingest      # ... and rebuild the index if any page changed
python -m askiisc_retrieval.cli refresh --every 24    # scheduler: refresh now, then every 24 hours
python -m askiisc_retrieval.cli refresh --trigger     # make a running scheduler refresh within 30 s
python -m askiisc_retrieval.cli fetch --refresh       # the same conditional refresh, without the status file
```

`./run-local.sh --refresh` refreshes the listed and approved linked pages the same way (it does not crawl;
use `cli refresh` for listed, crawled and linked pages together).

## What a refresh does

1. Every listed, crawled and linked page that is already saved is requested with `If-None-Match` (its last
   `ETag`) and `If-Modified-Since` (its last `Last-Modified`). A `304 Not Modified` costs no download and
   writes nothing.
2. A downloaded page is hashed on its **extracted main content** (FR-3; PDFs on their bytes). If the hash
   equals the URL's last stored version, nothing is written: no snapshot object, no registry row, and the
   working copy under `data/raw/` keeps its bytes and modification time. A change in the site template alone
   (navigation, footer date) is not a new version.
3. A changed page writes exactly one new snapshot object and one registry row, marks the previous version
   superseded (it is kept), replaces the working copy atomically and updates the manifest.
4. With `--ingest`, the index is rebuilt only if a page changed. Unchanged pages keep their stored vectors,
   so only changed pages are re-embedded. The index serves only the current version of each page
   (NFR-2: every indexed passage maps to an active, non-superseded version); superseded versions stay in
   the raw-version store for audit and rollback.
5. The outcome is written to `data/raw/versions/last_refresh.json`. A failed refresh or rebuild leaves the
   previous raw versions and index snapshot serving and records a `freshness_warning`.

## Layout under `data/raw/`

| Path | Content |
|---|---|
| `<hash>.html` / `.pdf` | working copy of each page = its current version (read by ingestion) |
| `versions/objects/<url-key>/<sha256>.<ext>` | immutable snapshots, one per stored version, never overwritten |
| `versions/registry.sqlite3` | `versions` (url, version, content_hash, object, bytes, fetched_at, superseded_at) and `sources` (ETag, Last-Modified, last check) |
| `versions/manifest.json` | current version of every URL with the versions it superseded |
| `versions/last_refresh.json` | outcome of the last refresh |

Inspect the history of one page:

```bash
sqlite3 data/raw/versions/registry.sqlite3 "SELECT version, content_hash, fetched_at, superseded_at FROM versions WHERE url='https://www.iisc.ac.in/admissions/'"
```

**Deviation from the PRD architecture (Sec. 6):** the PRD names PostgreSQL for the registry and MinIO for raw
versions. This repository uses SQLite and local files with the same rules (hash-gated writes, immutable
versions, one manifest). An existing `data/raw/` from an earlier release needs no migration: the first refresh
registers each saved page as version 1.

## Scheduling

**systemd (server):** copy `deploy/refresh/askiisc-refresh.service` and `askiisc-refresh.timer` to
`/etc/systemd/system/`, edit the three paths in the service, then

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now askiisc-refresh.timer
systemctl list-timers askiisc-refresh.timer             # next run
sudo systemctl start askiisc-refresh.service            # urgent refresh now
```

The timer runs daily at 03:00 IST (21:30 UTC) with up to 20 minutes of random delay, and catches up after a
host restart (`Persistent=true`).

**Without systemd:** run `python -m askiisc_retrieval.cli refresh --every 24 --ingest` in a terminal or
`tmux`; `refresh --trigger` from another terminal starts an urgent refresh.

**Local-folder Qdrant:** the local index folder can be opened by one process at a time. While the chat server
is running, `--ingest` cannot rebuild it: the refresh still stores the new versions and records a
`freshness_warning` that the index is behind; run `./run-local.sh --rebuild` with serving stopped. A Qdrant
server (`QDRANT_URL`) has no such restriction.
