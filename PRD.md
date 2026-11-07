# Ask IISc: A Retrieval-Augmented QA Assistant over Self-Hosted Open-Weight Small Language Models

**Project Requirements Document (PRD) — DS252 Cloud Computing, Aug 2026**

- **Team Name:** ASKS
- **Date:** 30/09/2026
- **Version:** 0.2

**Team Members**

| Name | ID | Email |
| --- | --- | --- |
| Kshitij Pratihast | 28768 | kshitijprat1@iisc.ac.in |
| Aditya Barodia | 28351 | adityabarod1@iisc.ac.in |
| Smit Patel | 26782 | smitpatel@iisc.ac.in |
| Sejadri Banik | 26387 | sejadribanik@iisc.ac.in |

**Summary.** A privacy-preserving RAG assistant answering IISc policy and admissions queries on self-hosted open-weight models, evaluated for accuracy, GPU efficiency and cost.

## Sec. 1 Problem statement and motivation

Prospective applicants, enrolled students, faculty advisors, and departmental staff at IISc need fast, accurate answers to questions scattered across academic-policy documents, admissions bulletins, course catalogues, and handbooks [6]. Today they search PDFs and wiki pages across portals, or email staff and wait days for a reply. This worsens during admissions and registration, when query volume spikes and documents are revised.

The technical gap is twofold. First, no institute-wide question-answering interface exists over this corpus, and general-purpose LLM chatbots would send institutional data – including administratively sensitive handbook content – to third-party API providers, compromising confidentiality. Second, self-hosting is itself a systems problem: serving a 3–8B model to many concurrent users at acceptable latency requires GPU-aware batching, KV-cache management [2], and quantization choices, while ingestion must continuously crawl, re-chunk, and re-embed a changing corpus. A single workstation cannot provide this, motivating a multi-tier cloud deployment: a GPU serving cluster, a scalable vector store, and an ingestion pipeline independent of user traffic.

## Sec. 2 Scope: in-scope, out-of-scope, assumptions

### In scope

- An ingestion pipeline that crawls policy, admissions, catalogue and handbook sources; parses, chunks, embeds and indexes them; and detects staleness.
- A self-hosted vector store for dense retrieval with metadata filtering (version, source, staleness flag) as the primary path.
- A self-hosted graph store for bounded relational expansion, built by rule- and local-SLM-based extraction with alias resolution and evidence validation; each edge carries supporting chunk IDs, version and scope, and is never a citation by itself.
- A shared snapshot manifest activating vector and graph versions together after both validate; failed refreshes keep the previous snapshot with warnings.
- Explicit fallbacks: graph failure degrades to filtered vector retrieval; generation failure returns source excerpts and links.
- A vLLM-class GPU serving tier with continuous batching, KV-cache reuse, and 3B, 8B and quantized (AWQ/GPTQ) variants served sequentially on one GPU.
- Admission control: per-user rate limits, per-tenant concurrency limits, bounded queues and token caps, degrading via HTTP 429 with retry guidance.
- An evaluation harness for accuracy, citation correctness, throughput, GPU-hours and latency across model variants and vector-only vs. graph-assisted retrieval (graph-indexing cost reported separately), and a minimal CLI/thin web query interface.

### Out of scope

- Training or fine-tuning models; only pretrained open-weight checkpoints are used.
- Multi-lingual support (English only) and a production mobile front end.
- Campus SSO; access uses per-tenant scoped tokens (guest/applicant for the public corpus, institute-issued otherwise); Keycloak is optional.
- Unrestricted anonymous access and third-party tunnels; databases and model endpoints are never publicly reachable.
- Open-domain or inferred relationships, and traversal beyond a configured bound (every edge traces to an ingested version; depth is bounded to hold NFR-4 latency).
- A live user-feedback/re-ranking loop; refresh is via scheduled re-crawl only.

### Assumptions

- A fixed, enumerable set of public-facing IISc policy, admissions, and handbook documents is accessible within the private cloud boundary [6], re-validated every 24 hours, with maintainer-triggered refresh for urgent notices.
- Serving and evaluation run on one institute-hosted NVIDIA RTX 4090 (24 GB VRAM); concurrency ceilings, context bounds and token caps are calibrated against it, so variants run sequentially, not as simultaneous replicas.
- The team can build a held-out gold set (100–200 QA pairs with citations) [6], including multi-hop questions not resolvable from a single passage.
- Relationships are extractable at sufficient precision to aid multi-hop retrieval; extraction quality is measured, and unvalidated edges are excluded from the active manifest.
- GPU-hour pricing comes from public listings for comparable 24 GB accelerators [7]; absolute costs are illustrative, cross-configuration comparisons are the result.
- The private inter-tier network is trusted; no encryption beyond TLS is needed for the MVP.

## Sec. 3 Stakeholders and usage scenarios

**Roles.** Prospective Student, Current Student, Faculty Advisor, Administrative Staff, and System Administrator (curation, monitoring, re-indexing).

### Usage scenarios

1. **Admissions query.** A prospective student asks “What is the minimum CGPA to reapply to the PhD program after a rejected application?”. The system answers with an explicit citation to the source document and section.
2. **Time-sensitive registration query.** A current student asks about the current semester’s add/drop deadline shortly after the registrar updates the handbook. Staleness detection must already have triggered re-ingestion so the answer reflects the new deadline.
3. **Bursty concurrent load.** During orientation week, dozens of faculty advisors and students query concurrently. The serving tier must hold latency targets via batching and admission control, not unbounded queueing.

These scenarios anchor Sec. 4: scenario 1 drives FR-1/FR-2, scenario 2 FR-3, and scenario 3 FR-5/FR-6.

## Sec. 4 Functional requirements

### FR-1 (M) — Bounded crawl, PDF handling & scheduled refresh

Crawl the approved seed list (policy, admissions, course-catalogue root pages) to a per-source-type configurable depth, visiting each URL once and excluding shared nav/header/footer links from traversal. A followed PDF link is downloaded and handed to the parser rather than traversed further. Sources are checked every 24 hours using ETags and content hashes, with maintainer-triggered refreshes for urgent notices.

**Acceptance.** No URL is fetched more than once per crawl run; zero nav/footer-only links appear in the crawl log; every in-depth-limit PDF is present in the raw snapshot store.

### FR-2 (M) — Main-content & entity extraction

Extract only the main-content region of each crawled HTML page and PDF, discarding boilerplate; extract source-backed facts and entities for graph indexing. Record a page as an extraction failure — not indexed silently — if the region is empty or the parser raises.

**Acceptance.** Every extraction failure is logged with URL and reason; no failed extraction ever produces an indexed chunk or graph node. (Failure-rate bound: NFR-1.)

### FR-3 (M) — Hash-based versioning & unified manifest

Hash every crawled page’s extracted content. Write a new raw snapshot to object storage and a new version row (hash, source URL, applicability date) to the PostgreSQL registry only when the hash differs from that URL’s last stored version. Link Qdrant passages and Neo4j entities through one validated manifest.

**Acceptance.** Re-crawling an unchanged page writes zero new snapshot objects and zero new registry rows; a modified page writes exactly one new snapshot and updates the manifest.

### FR-4 (M) — Freshness/staleness gating & atomic activation

Before a new document version is queryable, evaluate it against refresh-failure, age, and supersession rules; block activation of any version that fails. Updated vector and graph versions are activated through one validated manifest; failed refreshes preserve the previous snapshot with freshness warnings.

**Acceptance.** A version failing validation is never returned; passing versions become queryable within one ingestion cycle of being written to the registry; vector and graph representations never activate out of sync.

### FR-5 (M) — Embedding-model consistency

Use one pinned embedding model (BGE-M3, justified in Sec. 6) for both document chunks and incoming queries. Never allow vectors from two different embedding-model versions to coexist in the active index.

**Acceptance.** Index metadata reports a single embedding-model version for all active vectors; changing the pinned model requires re-embedding the full corpus before new vectors are activated, and old/new vectors are never queried together.

### FR-6 (M) — Controlled public access & scope extraction

Authenticate incoming requests at the gateway. Prospective students access an institute-approved public HTTPS gateway using guest credentials or an applicant account, without requiring campus SSO, restricted strictly to the approved public corpus. Enrolled users authenticate via bearer credentials for role-specific scopes. Databases and model endpoints remain strictly private.

**Acceptance.** Requests without valid credentials are rejected with HTTP 401; guest requests never return non-public documents; unrestricted anonymous access and third-party tunnels are rejected.

### FR-7 (M) — Latency-conscious admission control & load shedding

Enforce per-user request-rate limits, per-tenant concurrency limits, bounded queues, and token caps calibrated on the RTX 4090. Overload returns an explicit HTTP 429 with retry guidance.

**Acceptance.** Under a configured ceiling/quota, surplus requests receive HTTP 429; already-admitted requests run to completion uninterrupted. (Rejection-latency target: NFR-9.)

### FR-8 (M) — Model-variant configuration on single RTX 4090

Serve designated model variants (Llama 3.2-3B-Instruct, Llama 3.1-8B-Instruct in BF16, and a four-bit AWQ version of the same 8B checkpoint) sequentially within the 24 GB VRAM footprint. An 8B BF16 model requires approximately 16 GB for weights alone, so context length and concurrency must leave space for runtime memory and KV cache. Simultaneous model replicas are deferred.

**Acceptance.** Every response’s metadata records the exact checkpoint and precision that generated it; the actively serving variant always matches the current deployment configuration.

### FR-9 (M) — Version-filtered hybrid retrieval & explicit fallback

Qdrant retrieves passages, while Neo4j connects source-backed facts, filtered to active (non-superseded) document versions and user scope from the validated snapshot. If graph retrieval fails, fall back to filtered Qdrant retrieval from that snapshot and report degraded operation.

**Acceptance.** 100% of passages and facts returned for any query satisfy active-version and scope filters; zero superseded, stale, or unauthorized passages ever appear; graph failures trigger automated fallback without dropping requests.

### FR-10 (M) — Deterministic prompt assembly for prefix reuse

Assemble each prompt as a fixed, deterministic invariant-instruction prefix, followed by retrieved passages and the query, bounded to one configured max length — truncating lowest-ranked passages first if exceeded.

**Acceptance.** Consecutive requests sharing identical invariant instructions produce identical token IDs across the fixed prefix (tokenizer comparison), enabling prefix/KV-cache reuse at the serving tier. The length bound is enforced exactly once, at this stage. (Bounds: NFR-5/6.)

### FR-11 (M) — Grounded generation, abstention & excerpt fallback

Generate answers using only authorized evidence. Missing or conflicting evidence triggers explicit abstention (“insufficient evidence”) or clarification. If generation is unavailable, return retrieved source excerpts and links without a generated answer. For every answer generated, attribute each factual claim to a passage and attach a citation resolving to its document ID, version, and location. External-model fallback is rejected.

**Acceptance.** On a fixed out-of-corpus query set, the system abstains — never fabricates — for 100% of cases; on a fixed adversarial set, zero unsupported claims reach the user; every citation resolves to a currently active, validated document version; generation outages cleanly return source excerpts and links.

### FR-12 (S) — Streaming & citation resolution

Stream generated tokens incrementally to the client, and resolve each citation marker, on selection, to its source document’s metadata and text excerpt.

**Acceptance.** The client begins receiving tokens before generation completes; selecting a citation tag displays the corresponding source metadata and excerpt. (Time-to-first-token target: NFR-5.)

### FR-13 (C) — Optional reranking / factuality re-check

If MVP milestones (Sec. 8) are met early, add cross-encoder reranking of retrieved passages and/or an automated post-generation factual-consistency check.

**Acceptance.** When enabled, either mechanism only prunes or flags content — it never silently passes a low-scoring passage or claim through unchanged.

## Sec. 5 Non-functional requirements

Targets are hard invariants (security/correctness properties holding exactly, not statistically) or findings calibrated against a pilot/baseline run.

### NFR-1 — Ingestion reliability

(Verifies FR-1, FR-2.) Extraction-failure rate & content-boundary accuracy, per crawl run. Bar fixed from the Wk1–2 pilot crawl’s observed rate + a documented margin — not preset.

- **Workload.** Full seed-list crawl; accuracy on a 30-page labeled subsample.
- **Method.** Per-URL ingestion logs; one-time labeled-sample audit.

### NFR-2 — Freshness & version consistency

(Verifies FR-3, FR-4.) Version queryable <1 ingestion cycle after registry write; 100% of indexed passages map to an active, non-superseded version (invariant).

- **Workload.** Controlled single-document edit, re-ingested end-to-end on the pilot corpus.
- **Method.** Write→first-retrieval timestamp diff; automated version-metadata audit.

### NFR-3 — Retrieval quality & isolation

(Verifies FR-9, FR-11.) Recall@k (k per FR-9), value set from a baseline run, not preset; 0 unauthorized/inactive passages returned (invariant).

- **Workload.** Held-out multi-scope query set, incl. queries against superseded versions.
- **Method.** Passage-ID match vs. labels; automated scope/status audit of every result.

### NFR-4 — Retrieval-path latency

(Verifies FR-9.) p95 retrieval latency, from a baseline load test — kept independent of NFR-9’s rejection latency unless deliberately unified.

- **Workload.** Sustained concurrent load at the FR-7 concurrency ceiling, 10 min, full corpus.
- **Method.** Orchestration telemetry, query-issue→result-received.

### NFR-5 — Response latency by model config

(Verifies FR-8, FR-10.) p95 time-to-first-token & p95 end-to-end latency, reported per config (3B/8B/8B-AWQ); the comparison itself is the deliverable, no shared bar.

- **Workload.** Identical fixed concurrent-client load, run sequentially per config on the allocated hardware.
- **Method.** Client timestamps, submit→first/final token.

### NFR-6 — Throughput & resource efficiency

(Verifies FR-8, FR-10.) Queries/s, tokens/s, GPU utilization %, and prefix/KV-cache hit rate % — per config across a concurrency sweep; comparison findings, no fixed bar.

- **Workload.** 10-min steady-state run at a concurrency sweep (e.g. 1/4/8/16, per actual hardware), once per config.
- **Method.** Request logs + serving/cache telemetry.

### NFR-7 — Answer & citation quality

(Verifies FR-11.) Correctness reported as a measured outcome (baseline- or benchmark-calibrated, not preset); abstention = 100% on unanswerable queries (invariant, matches FR-11).

- **Workload.** >50 held-out questions, answerable + deliberately unanswerable.
- **Method.** Eval harness vs. reference answers & labeled sources.

### NFR-8 — Privacy & network isolation

(Verifies project constraint: no institutional data egress.) 0 bytes of institutional content, embeddings, or query data leave the private-cloud boundary (structural invariant).

- **Workload.** Full ingestion-retrieval-generation pipeline under active multi-user load.
- **Method.** Network/egress-log inspection during the run.

### NFR-9 — Availability & overload handling

(Verifies FR-6, FR-7.) Availability >99% over admitted requests (shed requests excluded); rejection latency from a baseline run, example config C=20/Q=10/quota 30 per min (tunable).

- **Workload.** 60-min run incl. a controlled overload period past admission capacity.
- **Method.** Success/admitted ratio; shed-request & health/restart logs.

### NFR-10 — Cost per useful work

(Verifies FR-8.) Cost & GPU-hours per 1,000 completed queries, per config (3B/8B/8B-AWQ), one accounting method reported as findings.

- **Workload.** 1,000 representative queries, identical definition, across all 3 configs.
- **Method.** Normalized compute/runtime/storage cost + measured GPU-hours.

## Sec. 6 System architecture and technology stack

### Component topology and data flow

Ingestion, online answering, and evaluation are separated. Ingestion follows approved sources → crawl/parse (Crawl4AI + Docling) → change detection (ETags/content hashes every 24 hours, maintainer webhooks) → chunk/embed → Qdrant and Neo4j. MinIO retains raw document versions; PostgreSQL records hashes, locations, dates, and refresh status. One validated manifest activates vector and graph versions together; failed refreshes keep the previous snapshot with freshness warnings.

The request path is browser → gateway → FastAPI → query embedding → hybrid retrieval (Qdrant passages + Neo4j facts) → optional reranking → vLLM → evidence/citation checks → browser. Queries and documents share one pinned embedding model, and retrieval enforces programme/year and access filters against one snapshot. Graph failure falls back to filtered Qdrant retrieval (degraded mode reported); generation failure returns source excerpts and links; missing or conflicting evidence prompts abstention.

### Tier placement and boundary data

- **Device.** The browser hosts the chat interface; HTTPS carries questions, filters, and credentials out, and answers, citations, and excerpts back.
- **Edge.** An institute-hosted gateway authenticates users and applies per-user rate limits, per-tenant concurrency ceilings (C = 20), bounded queues (Q = 10), and token caps calibrated on the RTX 4090; overload returns HTTP 429 with retry guidance. Prospective students use a public HTTPS gateway with guest/applicant credentials (no campus SSO), restricted to the public corpus.
- **Cloud (CPU sub-cluster).** Ingestion, FastAPI, and storage/metadata services (MinIO, PostgreSQL, Qdrant, Neo4j) over private traffic.
- **Cloud (GPU sub-cluster).** The RTX 4090 workstation (24 GB VRAM) hosts vLLM and local embeddings/reranking. Documents, vectors, graphs, weights, and KV state stay inside IISc.

### Hardware allocation and model memory budgeting

Self-hosted vLLM provides the continuous batching, KV-cache management, and prefix reuse the serving experiments need; hosted APIs are rejected because prompts and evidence would leave IISc. The RTX 4090’s 24 GB VRAM runs Llama 3.2-3B-Instruct, Llama 3.1-8B-Instruct, and a four-bit AWQ version of the same 8B checkpoint sequentially [4]. Llama offers instruction-tuned checkpoints at the required sizes and continuity with the prototype. An 8B BF16 model needs about 16 GB for weights alone, so context length and concurrency must leave room for runtime memory and KV cache.

### Frameworks and services

Crawl4AI and Docling handle crawling and structured extraction [5]; BGE-M3 via TEI supplies local embeddings [3, 5]; Llama 3.2 (3B) and 3.1 (8B) generate answers [4]. Qdrant provides retrieval with metadata filtering and Neo4j stores source-backed facts [5]. FastAPI coordinates requests; MinIO and PostgreSQL separate documents from version metadata [5]. vLLM supplies continuous batching and prefix caching [2, 5], and Kubernetes [5] schedules services/jobs. A Locust [5] client drives the gateway; client timings, Prometheus/DCGM [5], and scheduler records measure latency, throughput, and GPU-hours. All services are self-hosted; no external managed AI or database service is used.

### Design decisions and rejected alternatives

1. **Self-hosted vLLM over hosted LLM APIs.** Enables batching and KV-cache experiments while keeping prompts and evidence within IISc; hosted inference violates privacy.
2. **Single validated manifest over independently activated indexes.** Independent Qdrant/Neo4j activation would compromise version consistency.
3. **Sequential variants over simultaneous replicas.** Replicas (≈16 GB for 8B weights alone) would starve KV-cache space within the 24 GB budget.
4. **Controlled public gateway over anonymous access or tunnels.** Applicants reach the public corpus without SSO while GPU resources stay protected.
5. **Internal excerpt fallback over external-model fallback.** Preserves data sovereignty during outages.

## Sec. 7 Evaluation plan

### 7.1 Workload Generator

Held-out query set of ≈120 questions, human-verified against source documents: 70% in-corpus (gold answer plus gold citation known), 20% out-of-corpus/adversarial (no genuine answer — tests abstention), and 10% stale-version queries (a superseded document version contains a conflicting answer — tests version filtering) [6]. Traffic replay: Locust-driven client [5], Poisson arrivals, 3 simulated tenants (separate API keys), concurrency swept 1→5→10→20→30 to evaluate both normal operation and overload scenarios.

### 7.2 Baseline Configurations

- **B0 (no retrieval).** Model answers from parametric knowledge only — used as a basic quality baseline.
- **B1 (basic RAG).** Retrieval is enabled, but serving optimizations such as continuous batching and prefix/KV-cache reuse are disabled — used as a systems baseline.
- **B2 (vector-only RAG, full serving optimizations).** Qdrant retrieval is enabled but graph-assisted retrieval is disabled — used to measure the benefit of graph-assisted retrieval.
- **Full system.** Retrieval, graph-assisted retrieval, citation generation, admission control, version filtering, and vLLM serving optimizations are enabled.

### 7.3 Metrics

- **Quality.** Answer accuracy, Recall@k, citation correctness, and abstention accuracy.
- **Systems.** TTFT, p50/p95/p99 end-to-end latency, throughput (queries/sec), GPU utilization, GPU-hours per 1,000 queries, and HTTP 429 rate.
- **Graph retrieval.** Accuracy and citation correctness for vector-only versus graph-assisted retrieval.

### 7.4 Experiments

- **E1 (Model Variant Comparison validates FR-5, FR-8, FR-10; NFR-5, NFR-6, NFR-10).** Run the workload using three model configurations: Llama 3.2 3B, Llama 3.1 8B, and 8B-AWQ. Measure answer accuracy, citation correctness, TTFT, p95 end-to-end latency, throughput, GPU utilization, and GPU-hours. Objective: compare model quality, serving performance, and resource consumption.
- **E2 (Concurrency and Admission Control Evaluation validates FR-7, FR-12; NFR-9).** Increase concurrency from 1 to 30 users using the Full System and B1. Measure p50/p95/p99 latency, throughput, HTTP 429 rate, and admitted-request success ratio. Verify that requests beyond the configured capacity are rejected while admitted requests complete successfully, and that streaming responses and citation resolution work correctly.
- **E3 (Retrieval and Grounding Evaluation validates FR-6, FR-9, FR-11, FR-13; NFR-3, NFR-4, NFR-7).** Run the answerable, unanswerable, and stale-version subsets against the Full System. Measure Recall@k, answer accuracy, citation correctness, abstention accuracy, stale/unauthorized retrieval rate, and p50/p95/p99 retrieval/response latency. Where enabled, compare retrieval with and without reranking/factuality verification. Objective: verify that answers are grounded in valid evidence and that stale or unauthorized information is not returned.
- **E4 (Ingestion and Freshness Validation validates FR-1, FR-2, FR-3, FR-4, NFR-1, NFR-2).** Modify one document and run it through the complete ingestion pipeline. Measure extraction failures, duplicate fetches, document-version correctness, and write-to-queryable delay. Objective: verify change detection, versioning, freshness and consistency of the updated indexes.
- **E5 (Graph-Assisted Retrieval Evaluation — validates FR-9’s hybrid-retrieval design decision).** Run a small multi-hop subset against B2 and the Full System. Measure answer accuracy and citation correctness for both configurations. Report graph-indexing time/GPU-hours separately. Objective: determine whether graph-assisted retrieval provides useful improvements for questions requiring information from multiple documents.
- **E6 (Fallback and Degraded-Mode Behavior — validates FR-9’s fallback clause and FR-11’s excerpt-fallback clause).** Make the graph store unavailable during a test run and verify that the system falls back to filtered vector retrieval. Separately make the generation service unavailable and verify that the system returns the available source excerpts instead of failing silently or fabricating an answer.

### 7.5 Cost Model (per unit of useful work)

Define 1 unit of useful work = 1 correctly-answered, correctly-cited query.

$$
\text{Cost per useful answer} =
\frac{\text{GPU-hours} \times (\text{INR} / \text{GPU-hour})}{\text{Count of correct \& correctly-cited answers}}
$$

The metric is computed for the 3B, 8B, and 8B-AWQ configurations over the same workload used in E1. Evaluation is performed on the institute’s RTX 4090 (24 GB VRAM); therefore, public pricing for comparable 24 GB-class accelerators such as L4/A10G [7] is used only as a reference for relative cost comparison. Graph-indexing GPU-hours are reported separately as a one-time ingestion cost.

### 7.6 Expected Form of Results

- **Plot.** Model Variant (3B, 8B, 8B-AWQ) vs. Answer Accuracy (%)
- **Plot.** Model Variant vs. Throughput (Queries/s)
- **Plot.** Concurrent Users vs. p50/p95/p99 latency, with C = 20 marked.
- **Plot.** GPU-Hours per Query vs. Answer Accuracy
- **Bar chart.** Vector-only (B2) vs. Graph-assisted (Full system) accuracy and citation correctness.
- **Table.** Peak VRAM usage per model config vs. the 24 GB budget.
- **Table.** Degraded-mode test results — induced failure, expected behavior, observed behavior, pass/fail.
- Summary table comparing model variants on accuracy, citation correctness, TTFT, p50/p95/p99 latency, throughput, GPU-hours and cost per useful answer.

## Sec. 8 Work plan, milestones and risks

| Week / phase | Tasks and owners |
| --- | --- |
| 1 — Foundation | **Kshitij:** Finalise API contracts and local prototype.<br>**Aditya:** Approve source list and test extraction.<br>**Sejadri:** Confirm GPU access, vLLM compatibility.<br>**Smit:** Define scoring rubric and dev questions. |
| 2 — Development | **Smit:** Connect crawling, MinIO, PostgreSQL and Qdrant.<br>**Sejadri:** Deploy internal BGE-M3/TEI and vLLM.<br>**Kshitij:** Integrate cited answers and UI.<br>**Aditya:** Prepare and separate development/held-out sets. |
| 3 — Development and testing | **Aditya:** Implement refresh, staleness and atomic snapshots.<br>**Kshitij:** Implement tenant quotas, bounded queues and cancellation.<br>**Sejadri:** Configure batching/cache metrics.<br>**Smit:** Test retrieval, citations and abstention. |
| 4 — Integration testing | **All:** Test source replacement, rollback, restarts, access restrictions and blocked-egress serving.<br>**Sejadri/Smit:** Pilot all three model variants; freeze corpus, prompts and configurations. |
| 5 — Experiments | **Sejadri/Smit:** Repeat model, batching, caching, overload experiments; score held-out outputs.<br>**Kshitij/Aditya:** Fix critical defects; audit logs, versions and citations. |
| 6 — Delivery | **Smit:** Final plots and quality/cost analysis.<br>**Kshitij:** Reproducible deployment and demo.<br>**Aditya:** Corpus documentation.<br>**Sejadri:** Configuration manifests and reproduction checks.<br>**All:** Report and viva preparation. |

*Table 1. Six-week task breakdown with owners across development, testing, and experiment phases.*

### Demonstrable milestones

Weekly demos and twice-weekly integration checks; scripts and results retained for spot checks.

- **M1 (end of Week 2).** End-to-end cited QA over all four document categories.
- **M2 (end of Week 4).** Source updates, rollback, concurrent serving, tenant limits and three model configurations demonstrated.
- **M3 (end of Week 6).** Held-out scores, throughput/GPU-hours/p95 plots and a repeatable deployment.

### Principal risks and mitigations

1. **Limited GPU access.** Reserve slots early, run variants sequentially, reduce context/concurrency if needed.
2. **Poor extraction or outdated evidence.** Preserve locations and versions, test tables, flag stale sources, abstain when evidence is insufficient.
3. **Integration or evaluation delays.** Agree interfaces in Week 1, pilot evaluation early, freeze features after Week 4.

**MVP.** Bounded approved corpus; versioned ingestion and freshness; self-hosted embeddings/Qdrant; cited QA; tenant admission; vLLM batching/cache experiments; 3B, 8B and quantized-8B evaluation.

**Stretch.** Keycloak SSO, hybrid retrieval, reranking, MiniCheck, FP8 KV cache and multi-GPU routing.

## Sec. 9 References

1. Lewis, P. et al. “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.” NeurIPS, 2020.
2. Kwon, W. et al. “Efficient Memory Management for LLM Serving with PagedAttention.” SOSP, 2023.
3. Chen, J. et al. “BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings,” 2024. Weights: MIT Licence.
4. Meta AI. “The Llama 3 Herd of Models,” 2024. Llama 3.1/3.2 weights: Llama Community License.
5. Reused software: vLLM, TEI, Qdrant, Crawl4AI, Kubernetes, Prometheus, DCGM — Apache-2.0; Docling, FastAPI — MIT; Locust — BSD-2; MinIO — AGPL-3.0; PostgreSQL — PostgreSQL Licence.
6. Dataset: approved IISc policy, admissions, course and handbook documents; URLs, versions and access dates are in the source registry. Original rights retained (no open licence assumed); evaluation questions are team-authored.
7. AWS EC2 on-demand GPU pricing: NVIDIA L4 24 GB (G6, \$0.804/GPU-hr, INR 67 equiv.) and A10G 24 GB (G5, \$1.006/GPU-hr), accessed Sep. 2026. [AWS EC2 on-demand pricing](https://aws.amazon.com/ec2/pricing/on-demand/)
8. LLM disclosure: Claude/ChatGPT used to stress-test scope, NFRs and architecture, and to draft this PRD.
