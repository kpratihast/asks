# Report errata

Corrections to `ASKS_Project_Report.pdf` found during red-team testing. Each entry names the issue, the report text, what is wrong and what replaces it. Numbers here come only from scripts or measurements cited in the entry.

## E2 and the 429 policy (#55)

**Report text.** Section 4.4.1: "Every rejection is an HTTP 429 response with a Retry-After header and a JSON body carrying the reason". Section 8.4, below Table 10: "HTTP 429 responses first appeared at 30 concurrent users".

**What is wrong.**

1. A pre-authentication request guard (180 requests/min per client address, 1,200/min across all addresses) ran before the five admission parameters of Section 4.4.1. Its 429 had no `reason`, it was missing from the per-request log, and neither the report nor `LOAD_TESTING.md` described it. One load-generator address could therefore send about 3 requests/s sustained, whatever R, C, Q and G were.
2. The run configuration behind Table 10 was not recorded: the per-key rate, the number of load-generator addresses and the run length. With the documented defaults (R = 30/min per key, three tenant keys) the per-key limiter alone rejects requests well below 30 users; Cloudhack measured 429s from 10 users (#55). The Section 8.4 statement cannot be reproduced from the documented configuration and is withdrawn.

**Correction.** Since the #55 fix, guard 429s carry `reason` (`address_rate_limited`, `all_addresses_rate_limited` or `request_bodies_busy`) and `limiter: request_guard`, are logged and counted like admission rejections, are reported to operators in `/readyz`, are configurable (`GUARD_PER_ADDRESS_PER_MIN`, `GUARD_ALL_ADDRESSES_PER_MIN`) and are documented in `LOAD_TESTING.md`. Table 10's latencies stand as measured, but they describe an unrecorded configuration. A re-run of E2 must record R, C, Q, W, G, the guard limits, the number of load-generator addresses and the run length, and report 429s by `reason`.

## Evaluation gaps (#44)

Cloudhack's #44 lists six gaps between the report's §8 and the PRD's evaluation plan. All six were confirmed against rt-baseline (563c864). Sections 1 to 3 below are new measurements made on 19 Nov 2026 with the scripts named in each section.

**Run environment for sections 1 to 3.** Windows laptop, Python 3.13, CPU only; embedder `BAAI/bge-m3`; reranker `BAAI/bge-reranker-base`; reranker cutoff `ASKIISC_RERANK_MIN` left at its code default (`none`, no refusal gate); private temporary index, so the deployed CDS index was not used. These runs measure retrieval, ingestion and gateway behaviour, which do not depend on the GPU. They do not re-measure GPU latency or throughput.

### 1. Stale-version questions (PRD.md:258, FR-9)

New set `test_stale12.yaml`: 12 questions, kept separate from the held-out `test_cds120.yaml`, which is not changed. Each question is asked against an index that holds the current CDS schedule, course-description and M.Tech-requirements pages (`tests/pages/`) **and** an older, superseded version of the same three URLs (`fixtures/superseded_*.html`) that gives a conflicting answer. The superseded versions are embedded but not active. Script: `python tools/eval_stale.py --out eval_stale`, with k = 5.

| Mode | Superseded passages returned | Current answer in top 5 | Current answer at rank 1 | MRR |
|---|---:|---:|---:|---:|
| Vector | 0 | 12/12 | 11/12 | 0.958 |
| Full (graph) | 0 | 12/12 | 11/12 | 0.958 |

The control search ran the same queries with the active-version filter removed. The superseded answer then ranked in the top 5 for 10 of the 12 questions, with 20 superseded passages in total. The zero above is therefore the version filter at work: the old text is in the index and would otherwise be retrieved.

### 2. E4: ingestion and freshness (PRD.md:278)

Script: `python tools/eval_e4.py --out eval_e4`. It runs the whole ingestion pipeline (fetch, versioned raw store, parse/chunk/embed, snapshot activation, query) against a fake copy of the CDS site served in process. The fake site uses real page content; no institute server is contacted. As in `sources.yaml`, the schedule and M.Tech pages are listed pages and crawl seeds. The descriptions page is found only by the crawl and sends no ETag, so it is checked by content hash.

| Step | Requests | Duplicate requests | Conditional | 304 | 200 | Outcome |
|---|---:|---:|---:|---:|---:|---|
| 1. Initial fetch and ingest | 5 | 0 | 0 | 0 | 4 | 18 chunks indexed, 4 registry rows, 0 extraction failures |
| 2. Refresh, nothing changed | 5 | 0 | 3 | 3 | 1 | 0 new or changed; index unchanged, nothing re-embedded; 0 new registry rows |
| 3. M.Tech page changed ("minimum of 64 credits" to "72"), refresh with ingest | 5 | 0 | 3 | 2 | 2 | 1 changed; 2 of 18 chunks re-embedded; snapshot `snap-ff5637b2fc` to `snap-01c07fca84`; 0 extraction failures |

In each run, the fifth request is the site root, which the M.Tech page links to and the fake site answers with 404. It is requested once per run.

- **Version correctness:** the M.Tech page has version 1 (superseded) and version 2 (current); the version-1 object is kept byte for byte. Every other page keeps one version.
- **Queryable:** after the change, the M.Tech passages returned for "What is the minimum number of credits an M.Tech (CDS) student must earn?" contain 72 and not 64.
- **Write-to-queryable delay: 0.953 s.** This is fetch 0.077 s, rebuild 0.725 s, and opening the index plus the query 0.144 s, on a 4-page corpus. The delay grows with the corpus, because only changed chunks are re-embedded but every page is re-checked. With the 24-hour refresh schedule, the worst case is 24 h plus this delay. The maintainer trigger (`cli refresh --trigger`) gives this delay plus up to 30 s for the scheduler to notice the trigger.
- **Duplicate fetches:** before this fix, a page that was both a listed page and a crawl seed was requested twice in every refresh. Fixed in the commit that added `eval_e4.py` (regression test `tests/test_refresh_versions.py::test_page_both_listed_and_crawled_is_requested_once_per_refresh`). The table above shows 0 duplicates.

### 3. E6: generation-service and graph outage (PRD.md:280)

Script: `python tools/eval_e6.py --out eval_e6`. It runs the real gateway (`askiisc_serving.create_app`) in process, against the same index as section 1, with the 12 questions of `test_stale12.yaml`.

| Induced failure | Expected | Observed | Pass |
|---|---|---|---|
| Generation service unavailable (model URL on a closed port), 12 questions × non-streaming and streaming | HTTP 200, `fallback` / `generation_unavailable`, source excerpts, no generated text | 24/24 as expected; 0 answers generated; latency p50 3.47 s, max 10.34 s (CPU retrieval and reranking plus the refused connection) | Yes |
| Graph store unavailable: retriever | vector fallback, marked degraded | 12/12 returned passages with `degraded = true` | Yes |
| Graph store unavailable: through the gateway | HTTP 200, degraded, excerpts | 12/12 | Yes |

### 4. Cost and GPU-hours (NFR-10)

Derived from the report's measured throughput (Table 8 for the full system, Table 18 for B1) as GPU-hours per 1,000 completed queries = 1000 / (queries per second × 3600). This assumes throughput at the load used for those tables (E1), which the report does not fully record (see the #55 entry above).

| Model | Full system (q/s) | GPU-hours / 1,000 queries | B1 (q/s) | GPU-hours / 1,000 queries |
|---|---:|---:|---:|---:|
| Llama 3.2 3B | 3.39 | 0.082 | 1.39 | 0.200 |
| Llama 3.1 8B (BF16) | 1.72 | 0.161 | 0.72 | 0.386 |
| Llama 3.1 8B-AWQ | 3.24 | 0.086 | 0.69 | 0.403 |

Cost per 1,000 queries = GPU-hours per 1,000 queries × the GPU host's hourly price. No hourly price has been stated for the assessment GPU, so no money figure is given.

### 5. Abstention verdict (NFR-7, FR-11)

Table 20, row "Unsupported Queries · Abstain · Refusal accuracy 85–100% · Pass: Yes" is corrected to **Pass: No**. NFR-7 (PRD.md:197) defines abstention on unanswerable queries as a 100% invariant, so any measured value below 100% fails it. The live shortfall is tracked in #4.

### 6. B1 launch variants (Table 18)

Table 18 reports B1 throughput for 3B, 8B and 8B-AWQ, but `deploy/vllm_variants.sh` at rt-baseline could launch only `b1-3b`. Fixed in `91e7be5`: `deploy/vllm_variants.sh` now has `b1-8b` and `b1-8b-awq` (single sequence, no prefix cache), tested by `tests/test_issue_followups.py::test_b1_launch_variants_are_single_sequence_without_prefix_cache`. The Table 18 figures themselves were not re-measured.
