# Local public admissions snapshot

The local app combines the existing CDS corpus with a bounded snapshot of public IISc postgraduate admissions information. The source allowlist is `config/admissions_sources.json`. Content fetched on 29 October 2026 remains a dated snapshot; it is not a live admissions feed.

The boundary includes the admissions announcement page, common PG application rules, CDS entries in central M.Tech./research/CFTI pages, fees and scholarships, FAQ, admissions/CDS contacts, important dates, the current four-page mid-year notice and its one-page corrigendum, and the two CDS admissions pages. Central programme tabs for other departments are excluded. The image-only entry-mode flowchart is recorded as skipped because its text requires reviewed OCR. UG programme detail pages, previous cycles, lab subsite crawls, cutoff reports, and the application portal are outside this import.

The importer allows at most 14 source URLs, 8 MiB total source data, six PDF pages and 180 new chunks. It raises an error at a bound instead of truncating a document or partially publishing an index. Redirects during future fetches require review. It respects robots.txt and pauses between requests. Original bytes, SHA-256 checksums, normalized HTML and an ingestion report are stored in the index's `admissions/` directory. HTML panel selection uses programme content rather than positional tab labels, because the current M.Tech. page has one malformed panel. Fee tables retain the programme and category for each amount, including merged cells.

The current import preserves all unrelated active CDS chunks and all 10 student and 19 faculty chunks. It replaces the old versions of the two CDS admissions URLs. It embeds only the new admissions chunks, rebuilds the combined graph, and leaves the old index available for rollback. Old Qdrant points may remain physically present in the candidate copy; the active snapshot excludes superseded versions.

## Refresh and evaluate

Run these commands from the active project directory. The existing local launcher selects the configured local LLM. The importer does not change that endpoint or switch the serving index.

```bash
PYTHON="$(cat .local/python)"
BASE="$($PYTHON -c 'import json; print(json.load(open("config/local.json"))["ASKIISC_DATA_DIR"])')"
TAG="$(date +%Y%m%d-%H%M%S)"
$PYTHON tools/import_admissions.py \
  --base "$BASE" \
  --out "$PWD/data/admissions-$TAG" \
  --cache "$PWD/.local/admissions-fetch-$TAG" \
  --fetch
$PYTHON tools/compare_admissions.py \
  --base "$BASE" \
  --candidate "$PWD/data/admissions-$TAG" \
  --out "$PWD/output/admissions-$TAG"
```

A cached import uses `--cache <index>/admissions` without `--fetch`. The import writes a new output directory and refuses to overwrite an existing one. A failed build has `IMPORT_INCOMPLETE`; do not activate it. Refreshing after the admissions cycle changes may require updating the explicit PDF URLs in the source profile; the importer never follows arbitrary links.

Read `admissions/ingestion.json`, the comparison report and the raw answers before activating a candidate. The development set combines institute rules with CDS details and includes contradictory-source and unavailable-future-cycle cases. It compares the previous corpus with the expanded corpus and separately compares vector RAG with Graph RAG on the expanded corpus. Source coverage gains do not demonstrate a graph traversal advantage. Regex fact and evidence checks are proxies and do not establish semantic accuracy.

To activate, stop only the chat process, copy the latest SQLite auth database into the candidate with SQLite's backup API, and set `ASKIISC_DATA_DIR` in `config/local.json` to the candidate's absolute path. Start `./run-local.sh --port 8100`. The importer already copies sessions consistently, but repeat the auth backup at activation if users signed in after the import. Keep `VLLM_URL`, `SERVED_MODEL` and the auth feature flag unchanged. Rollback consists of restoring the previous data-directory value and restarting chat. Never copy a live SQLite database with an ordinary file copy.

`./run-local.sh --rebuild` now retains the full cached CDS crawl and the verified admissions sources. It does not refetch the admissions snapshot. Rebuilding source text may change chunk IDs; use the incremental importer to preserve the existing non-admissions chunk IDs. `--refresh` refreshes the seed CDS pages; a fresh bounded admissions fetch uses the importer above.

The OTP hint distinguishes public entrance tests (GATE, NET/JRF and admissions tests) from protected course/final exams. Per-chunk permissions and document-download permissions still apply to every query, including mixed-topic questions and when authentication is disabled.

## Source conflicts and limits

The central M.Tech. selection description and CDS August 2026 page disagree about an oral interview. The original January 2027 mid-year PDF and Important Dates page give different interview ranges. These are separate evidence sources; an answer should disclose a conflict and direct the applicant to official clarification. CDS August admissions must not be treated as proof of a January intake, and Brain, Computation, and Data Science (BCD) must not be conflated with CDS.

The current graph rules also cover source-backed programme and policy relationships; see `POLICY_GRAPH.md` for traversal, rebuilding and matched-corpus evaluation. Compound questions can still miss a necessary fact or exceed the six-source prompt budget. The earlier ingestion comparison measures corpus coverage; use the newer policy-graph comparison to assess traversal improvements.

For AWS, create a new immutable release from the chosen index using `AWS_DEPLOYMENT.md`. The previously prepared release remains unchanged and contains the older corpus. No AWS resources are created by local ingestion or comparison.
