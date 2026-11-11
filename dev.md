# Ask IISc developer guide

## Code map

| Area | Files | Responsibility |
|---|---|---|
| Sources and downloads | `sources.yaml`, `config/departments.yaml`, `askiisc_retrieval/fetch.py`, `crawl.py`, `linked_fetch.py` | Select departments, cache pages and fetch a bounded set of links from approved hosts. |
| Parsing and indexing | `askiisc_retrieval/parse.py`, `admissions.py`, `chunking.py`, `ingest_stub.py`, `snapshot.py` | Keep headings, PDF page numbers and table rows, preserve hyperlinks and published email addresses, embed passages and record active versions. |
| Graph construction | `askiisc_retrieval/link_graph.py`, `graph_rules.py`, `connections.py`, `policy_graph.py` | Add edges from actual page links and supported facts. Keep the passage that supports each edge. |
| Search | `askiisc_retrieval/retriever.py`, `stores.py`, `ingestion_adapter.py` | Search active, authorized passages. Add bounded graph evidence when it helps answer the question. |
| API and chat | `askiisc_serving/app.py`, `generator.py`, `evidence_selection.py`, `static/index.html` | Receive requests, fit evidence into the model prompt, check citations and show answers. |
| Accounts and access | `askiisc_serving/email_auth.py`, `auth.py`, `security.py`, `askiisc_retrieval/private_documents.py` | Verify OTP, set roles on the server, and check access before returning passages, graph paths or files. |
| Operations | `tools/`, `deploy/aws/`, `bench/` | Build and inspect indexes, run comparisons and package releases. |

## Follow a question through the code

1. The API verifies the session and passes the permitted document scopes to retrieval.
2. Both modes search active, authorized passages. General admissions and fee questions use institute evidence. A named department can use its own evidence and shared IISc evidence.
3. In full mode the retriever can follow source links or graph relationships. The source-link search reads at most two hops and eight target pages from the index. The service makes no web requests while answering.
4. The prompt builder keeps the supporting passages for a connected answer together. The app abstains if they cannot fit. An email must appear in the cited evidence. For a linked instructor lookup, the email must also match the selected contact evidence.

When a course page links to an instructor’s profile, ingestion saves the URL even if the link is labelled `WWW`. It also reads published addresses behind `mailto:` links. Graph RAG can follow the stored URL to a profile in the index, or match the instructor’s full name against the department directory. It needs evidence for both the instructor and the address. The graph also records lab leadership when a research table names the lab and the faculty in charge. These relationships come from source content and do not require a rule per course or person.

A missing page, inactive version or unauthorized passage breaks the path. The retriever can still use the ordinary vector candidates. Joint offerings and unknown site layouts should be reviewed before ingesting a new department.

If a question names a document and department, retrieval searches that source within the permitted departments before selecting the prompt passages. Catalogue entries prefer the chunk containing the course title over a continuation containing prerequisites. Fee evidence keeps the published programme, category and reporting period. The response can quote an annual figure when a semester breakdown is unavailable.

A reranker score is specific to the model and indexed corpus. `ASKIISC_RERANK_MIN` defaults to `none`, which lets reranking order the passages while the model checks whether they answer the question. Set a numeric cutoff only after evaluating a separate dev set covering the enabled departments. Access, active-version and graph-proof checks apply independently of that cutoff.

To diagnose a refusal, inspect the response's `reason`, `trace.evidence_sufficient`, `trace.rerank_top`, `trace.snapshot_id` and `budget`. A model abstention, a score cutoff and missing prompt evidence need different fixes. Confirm that the active index contains the configured pages before changing a cutoff.

A partial numeric reply can survive a model's refusal sentence when its cited amount matches a labelled field in policy evidence covering the requested fee or scholarship metric. Refusal explanations with incidental citations retain their refusal status.

## Work on ingestion

- The default source list lives in `sources.yaml`. The quickstart enables CDS, CSA and ECE. `ASKIISC_SOURCE_DEPARTMENTS` filters the department pages and sections at build time.
- The fees normalizer retains the programme and category with the fee rows. Keep those labels when adding fee tables so a quoted amount stays tied to the right group.
- For a PDF seed, set `pdf.title` and inclusive `pdf.page_ranges` keyed by source owner. The reader extracts at most 32 selected pages, keeps the physical PDF page number in each citation, and carries a course heading across consecutive pages in the same chapter. For a new edition, check the section boundaries before rebuilding.
- Keep source access scope explicit. A page link gives no permissions to its destination.
- Add a crawl or linked-page profile in `sources.yaml`. Keep host lists and page limits small. A linked source must pass the host and robots checks, including redirects.
- Rebuild from raw HTML to add link and contact metadata to an existing corpus. A graph-only rebuild of older chunks cannot recover HTML links that were discarded.
- Run `./run-local.sh --rebuild` with a new data directory and copied raw caches when testing parser changes. Use `--prepare` on a new index to fetch approved links, or `--refresh` to download configured sources again.
- Deploy new indexes with a full bundle. The code-only pipeline retains the previous corpus.
- External ingestion can supply `links` and `contacts` in Qdrant payloads. `links` holds normalized absolute URLs. `contacts` holds the source lines that publish addresses. Older payloads still load, with empty link and contact lists.

## Test and measure

```bash
python -m pytest -q
python -m pytest -q tests/test_admissions.py tests/test_departments.py tests/test_link_graph.py tests/test_email_access.py
```

Test new links on documents and questions outside the cases used to write the parser. Include a fee lookup with a named programme and category, a course lookup that cites a PDF page, and a question that needs two sources. Check the stated fact and cited passages. Record whether the prompt contained all the evidence separately from whether the answer was correct.

Use `python -m askiisc_retrieval.cli eval --questions graph_connected.yaml` to measure source recall on the checked-in evaluation questions with gold sources. Add `--no-graph` for the vector run. Use a snapshot that contains the gold sources listed in the question file.

For a paired model run, send each question with `mode=vector` and `mode=full` against the same snapshot. Use `pinned_snapshot` in both requests. Keep model, role, prompt budget and source limit fixed. Report where each mode wins or abstains. A link helps when it points to the missing evidence. A direct lookup can work equally well in both modes.
