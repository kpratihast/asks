# Relationship evidence selection

The local chat keeps its six-source cap and current model/context settings. Graph evidence now reaches prompt selection as structured provenance, rather than being appended behind six vector sources and then excluded by position.

## Retrieval

`requested_relations()` maps the requested answer type to the graph relations already available in the project. Lab questions prefer `HEADS`; teaching relationships prefer `TAUGHT_BY`; prerequisite questions prefer `REQUIRES`; venue questions prefer `HELD_IN`. This is a deterministic intent heuristic, not a new model or semantic reranker.

Graph candidate ranking considers the requested relation, reachability from the question entities, direct entity coverage, and the instructor associated with a named course. When a term is explicit, the instructor anchor prefers that term; a current offering can still anchor the existing cross-term traversal when the named course was not offered in the requested term. Scope/version audits and the existing rule that bridge edges themselves are never cited remain in force.

Each graph passage includes `graph_support` metadata: relation, endpoints, query coverage, focused people, originating question seeds and hop count. Serving preserves that metadata and graph paths. It is used for selection only; actual source text remains the evidence sent to the model.

## Prompt selection

For a recognized relationship question with relevant graph support, `evidence_bundle()` chooses a course anchor and the graph endpoints answering the requested relation. It prefers a short, applicable schedule row over the full syllabus when both support the anchor. Shared facts covering multiple named people take precedence over facts about only one. A direct named-lab question does not promote unrelated neighbours.

The prompt builder considers the required sources first and stops when the relationship bundle is complete, avoiding unrelated neighbours added only to fill six slots. It then enforces the same source cap and estimated token budget. Whole source text is preserved in the tested relationship bundles; the change does not insert graph edges as factual evidence or paraphrase source text.

For ordinary ranked retrieval and the vector baseline, source limits apply to successfully packed blocks. An oversized candidate no longer prevents a later short source from being considered. Required relationship blocks are never silently truncated. If a required bundle cannot fit, the API abstains with `insufficient_context_budget` instead of generating from a known incomplete relationship. The response budget records the selection method and required/selected source counts.

This heuristic does not solve arbitrary graph queries or establish that every graph neighbour is relevant. Queries it does not recognize retain ordinary ranked retrieval. The citation checker still validates markers rather than the semantic truth of every claim; answer quality is separately evaluated against source excerpts.

## Validation

`tests/test_evidence_selection.py` covers late graph evidence, irrelevant graph neighbours, source-budget gaps, metadata transfer, relation intent, anchored lab ranking and incomplete-bundle abstention. The full local suite passed with 146 tests and 17 Postgres-dependent tests skipped.

The rerun uses the same 30-question development pilot and all four original variants, with identical model, questions, source snapshot, embedding model and context/output limits. This is a regression evaluation informed by the first pilot; it is not held-out generalization evidence. Implementation hashes and source copies are archived alongside raw outputs.

Original outputs: `output/graph-comparison/2026-10-28/`.
Rerun outputs: `output/graph-comparison/2026-10-28-evidence-fix/`.
