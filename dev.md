# Ask IISc developer guide

## Code map

| Area | Files | Responsibility |
|---|---|---|
| Sources and downloads | `sources.yaml`, `config/departments.yaml`, `askiisc_retrieval/fetch.py`, `crawl.py`, `linked_fetch.py` | Select departments, cache pages and fetch a bounded set of links from approved hosts. |
| Parsing and indexing | `askiisc_retrieval/parse.py`, `chunking.py`, `ingest_stub.py`, `snapshot.py` | Keep headings and table rows, preserve hyperlinks and published mail addresses, embed passages and record active versions. |
| Graph construction | `askiisc_retrieval/link_graph.py`, `graph_rules.py`, `connections.py`, `policy_graph.py` | Add edges from actual page links and supported facts. Keep the passage that supports each edge. |
| Search | `askiisc_retrieval/retriever.py`, `stores.py`, `ingestion_adapter.py` | Search active, authorized passages. Add bounded graph evidence when it helps answer the question. |
| API and chat | `askiisc_serving/app.py`, `generator.py`, `evidence_selection.py`, `static/index.html` | Receive requests, fit evidence into the model prompt, check citations and show answers. |
| Accounts and access | `askiisc_serving/email_auth.py`, `auth.py`, `security.py`, `askiisc_retrieval/private_documents.py` | Verify OTP, set roles on the server, and check access before returning passages, graph paths or files. |
| Operations | `tools/`, `deploy/aws/`, `bench/` | Build and inspect indexes, run comparisons and package releases. |

## Follow a question through the code

1. The API verifies the session and passes the allowed roles to retrieval.
2. Retrieval selects active source versions, routes named departments and ranks passages. Both retrieval modes use these filters.
3. In full mode the retriever can follow links from those passages. It reads at most two link hops and eight target pages from the index. The service makes no web requests while answering.
4. The prompt builder keeps a connected path together. If its sources cannot fit, the app abstains. The answer must cite the supporting pages. An email must appear in the cited evidence. For a linked instructor lookup it must also match the selected contact evidence.

A `WWW` link is useful because its target identifies a page even when the anchor says nothing about the instructor. The parser also reads the address behind a `mailto:` link. The index joins an outgoing link to a target page already in the corpus. If a course link is broken, Graph RAG can also connect an instructor named in the schedule to an exact full-name match in the same department’s faculty directory. Both source passages must be present. This works with new URLs and names without adding a course-specific rule. The schedule and policy extractors still have rules for the types of facts they read. Those rules describe fields and relations.

A missing page, inactive version or unauthorized passage breaks the path. The retriever can still use the ordinary vector candidates. Joint offerings and unknown site layouts should be reviewed before ingesting a new department.

## Work on ingestion

- Keep source access scope explicit. A page link gives no permissions to its destination.
- Add a crawl or linked-page profile in `sources.yaml`. Keep host lists and page limits small. A linked source must pass the host and robots checks, including redirects.
- Rebuild from raw HTML to add link and contact metadata to an existing corpus. A graph-only rebuild of older chunks cannot recover HTML links that were discarded.
- Run `./run-local.sh --rebuild` with a new data directory and copied raw caches when testing parser changes. Use `--prepare` to fetch missing linked pages.
- Deploy new indexes with a full bundle. The code-only pipeline retains the previous corpus.
- External ingestion can supply `links` and `contacts` in Qdrant payloads. `links` holds normalized absolute URLs. `contacts` holds the source lines that publish addresses. Older payloads still load, with empty link and contact lists.

## Test and measure

```bash
python -m pytest -q
python -m pytest -q tests/test_link_graph.py tests/test_email_access.py
```

Test new links on documents and questions outside the cases used to write the parser. Include direct lookups, questions that need two sources, and questions with missing or conflicting evidence. For each answer, check the stated fact and cited passages. Record complete evidence in the prompt separately from final answer accuracy.

For a paired model run, send each question with `mode=vector` and `mode=full` against the same snapshot. Use `pinned_snapshot` in both requests. Keep model, role, prompt budget and source limit fixed. Report where each mode wins or abstains. A link helps when it points to the missing evidence. A direct lookup can work equally well in both modes.
