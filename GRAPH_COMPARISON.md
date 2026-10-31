# Graph RAG comparison

Run a frozen graph ablation against the existing local MLX model:

```bash
cd /Users/kshitij/mvps/askiisc/askiisc-retrieval
/Users/kshitij/Desktop/askiisc/.venv/bin/python tools/compare_graph.py \
  --out output/graph-comparison/my-run
```

The model configured in `config/local.json` must already be serving. The runner checks its identity, uses a consistent SQLite backup of the real Qdrant index, and leaves the live chat and index running. No pages are fetched, no synthetic documents are inserted, and nothing is re-embedded. The query encoder is the actual cached BGE-small model.

The 30-question pilot in `config/graph_questions.json` is tied to snapshot `snap-618aabd36c`. Questions and gold evidence groups were frozen before retrieval outputs were inspected. It includes existing/reworded development examples and related cases; report it as an exploratory pilot, not a held-out benchmark. Gold groups identify required facts; any listed chunk can satisfy a group, and all groups must be found for complete-evidence success.

Four variants run in seeded, interleaved order, with a warmup excluded from latency:

| Arm | Vector passages | Graph extras | Evidence-block cap |
|---|---:|---:|---:|
| `vector_current` | 6 | 0 | 6 |
| `graph_current` | 6 | up to 3 | 6 |
| `vector_equal` | 9 | 0 | 9 |
| `graph_equal` | 6 | up to 3 | 9 |

All variants share the same model, temperature, thinking setting, prompt instructions, context limit and output limit. The equal-budget comparison has the same maximum number of evidence blocks and context limit, rather than exactly equal input-token counts. Actual token counts are reported. The vector baseline retains semester logic and identifier handling; it isolates the contribution of graph expansion.

Every completed request is checkpointed. Reusing an output directory resumes only if the corpus, questions, runner, model and principal settings match. Use a new directory for a fresh run.

Outputs:

- `comparison.html`: offline report with metrics, all answers, selected evidence and graph paths.
- `REPORT.md`: metrics, paired differences, methodology, limitations, and every answer.
- `comparison.csv`: per-question scores, answers, timings and graph paths.
- `metrics.json`: aggregate and category metrics, paired wins/losses, bootstrap intervals and exact paired tests.
- `raw.jsonl`: complete retrieved passages, actual selected evidence, prompts, model responses and token usage.
- `metadata.json`: configuration, model identity and corpus/question/runner hashes.
- `review.json`: explicit review decisions, if a source-grounded answer review was performed. These are distinct from automated fact-pattern checks.

The automatic rubric checks expected answer patterns and abstention status. It does not detect all unsupported claims, incorrect logical qualifiers, or citation-placement errors. Review the actual answer against its cited excerpts before calling this semantic accuracy. Record `correct`, `supported` and `note` under keys such as `R01/graph_equal` in `review.json`, then rebuild reports without inference:

```bash
/Users/kshitij/Desktop/askiisc/.venv/bin/python tools/compare_graph.py \
  --out output/graph-comparison/my-run --report-only
```

The initial execution and source-grounded agent review are saved in [the 28 October 2026 report](output/graph-comparison/2026-10-28/REPORT.md). No retrieval changes were made after seeing its outcomes.
