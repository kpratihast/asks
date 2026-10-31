# Programme and policy Graph RAG

The local graph connects a department's degree offering to its programme and institution, then follows typed links to source chunks about fees, funding, duration, selection and admissions. A connected answer can combine a department curriculum with an institute policy without relying on both chunks ranking highly for one embedding query.

For example, `Offering:CDS/MTech → IS_A → Programme:MTech → HAS_FUNDING_AMOUNT → Evidence` retrieves the regular M.Tech. scholarship. M.Tech. (Research) has a separate programme node. Every edge records a source chunk, version, access scope and applicable qualifiers. The generator receives original source text, with compact applicability labels; the graph never stores a generated answer.

## Retrieval and safeguards

The planner recognizes policy facets in the question, traverses forward through programme, department, institution and cycle links, and selects a compact set of source chunks that covers those facets. Bridge evidence is included with endpoint evidence. It does not traverse backward into other departments. Entry route, fee category and explicit cycle constrain the path. Questions comparing selection sources retain institute and department origins separately. Date comparisons retain notice and calendar sources separately.

Traversal is bounded to three bridge hops, 512 states, 100 fact candidates and six supporting chunks. Every selected edge and chunk must agree on active version, embedding model and access scope. The existing course–teaching-slot–exam path remains separate. Public, student and faculty permissions, OTP validation and the professor test override are unchanged.

Graph errors fall back to the normal retrieval path and appear as degradation in the trace. A partial policy graph reports missing facets; it does not certify the answer. The generator preserves complete evidence bundles within the shared context limit. It may abstain if required evidence cannot fit. The UI displays source connections, and the API trace includes `policy_trace` and policy paths in `graph_proofs`.

The extraction rules use programme names, source headings and fact patterns. They do not contain benchmark question IDs, expected answers, fixed fees or exam dates. This is a curated domain schema, not unrestricted graph discovery. New institutional layouts, additional programmes and differently phrased questions may require extraction and intent coverage work. Arbitrary comparisons across several degree programmes and exhaustive negative claims over long documents remain limitations; a graph trace marked complete means the requested facets found candidates, not that every answer claim is correct.

## Rebuild without re-embedding

Run from the active project directory:

```bash
PYTHON="$(cat .local/python)"
BASE="$($PYTHON -c 'import json; print(json.load(open("config/local.json"))["ASKIISC_DATA_DIR"])')"
TAG="$(date +%Y%m%d-%H%M%S)"
$PYTHON -m tools.rebuild_graph --base "$BASE" --out "$PWD/data/policy-$TAG"
$PYTHON -m pytest -q
$PYTHON -m tools.compare_policy_graph \
  --index "$PWD/data/policy-$TAG" --out "$PWD/output/policy-$TAG"
$PYTHON -m tools.ablate_policy_graph \
  --index "$PWD/data/policy-$TAG" \
  --questions "$PWD/output/policy-$TAG/questions.json" \
  --out "$PWD/output/policy-$TAG/ablation.json"
```

The rebuild refuses an existing destination. It clones Qdrant consistently, preserves chunk bytes and active versions, regenerates the graph, and writes `graph-build.json`. Do not activate a directory containing `BUILD_INCOMPLETE`. Refresh source content separately using `ADMISSIONS_INGESTION.md`.

Before activation, stop chat, repeat the SQLite backup of the current auth database into the candidate, change only `ASKIISC_DATA_DIR` in `config/local.json`, and restart `./run-local.sh --port 8100`. Keep the old index for rollback. The local model endpoint stays unchanged. If the chat runs directly inside tmux, stopping it closes the session; start a new session instead of sending commands to the old one.

Future AWS releases should package the selected code and candidate index together using `AWS_DEPLOYMENT.md`. Previously created immutable release bundles do not acquire these changes automatically. No cloud resources are provisioned by these tools.

## Evaluation design

`tools.compare_policy_graph` runs 12 existing questions and a separately frozen 16-question transfer set against the same corpus and model. The transfer set is author-written with knowledge of the domain schema; it is not an independent blind benchmark. It includes eight connected questions, four direct controls, three unavailable-information questions and one conditional-eligibility question.

Three arms share the system prompt, original corpus, model, 4096-token context limit, 512-token output limit and six-source cap:

- Vector RAG uses the original query.
- Decomposed vector RAG uses the same facet vocabulary, up to four additional queries, facet diversity and reciprocal-rank fusion. It receives no graph or expected answers.
- Graph RAG adds typed, qualified evidence traversal and dependency-aware packing.

Arm positions are balanced across questions using a fixed seed. Local prefix caching still affects latency, so the report does not claim a speed advantage. Raw answers, selected prompt evidence, paths, queries, token estimates and implementation hashes are saved. Resuming with different implementation hashes is refused. One API evaluation sample is recorded per question and arm; the existing citation validator may retry generation once. Attempts and combined token usage are retained.

Report all three outcomes separately: required evidence in the prompt, regex fact checks, and review of complete source-supported answers. Regex checks can reject valid paraphrases and can pass an incomplete or unsupported answer. Negative-case review must also distinguish a grounded denial of a false premise from a literal abstention. The bridge-removal experiment measures retrieval only; it is not a separately generated answer-accuracy result.
