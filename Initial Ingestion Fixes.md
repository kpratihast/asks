# Ingestion pipeline v2 (pipeline.py, 2026-10-24): findings from the integration

> **Status: all 14 are fixed in ingestion v3** (`ingestion-v3.zip`, see its `CHANGES.md` and
> `git diff HEAD~1`). Kept here as the record of why.

The new payload matches the retrieval contract exactly (all 14 fields), Postgres versioning works
(re-ingestion supersedes the old version), and retrieval + serving run on it end to end. These came out of
running `pipeline.py` itself on sample CDS pages under `tests/test_e2e_smit_pipeline.py`, and out of
`python -m askiisc_retrieval.cli check-ingestion`. Retrieval works around 1, 2 and 5 already. The others
need a change in ingestion.

## High: affects answers

1. **Room numbers become course codes.** `extract_course_codes` (`[A-Z]{2,3}\s*[0-9]{3}`) matches
   `Venue: CDS 202`, so schedule rows get `course_codes = ["CDS202", "DS252"]`, and Neo4j gets `Course:CDS202`
   with TAUGHT_BY edges ("Simmhan teaches CDS 202"). The same pattern misses codes like `E0 251`.
   Fix: for table rows, take the code from the Course ID column only. Otherwise use
   `\b(?!CDS\b)([A-Z]{1,3}[0-9]?)\s?-?\s?(\d{3})\b`.

2. **Co-instructors joined with `&` stay one person.** The split regex `[,/]|\band\b` leaves
   `"S.Vadhiyar & Y. Simmhan"` as a single Faculty node, so "which course did Vadhiyar and Simmhan teach
   together" has no graph path. Fix: add `&` and `;` to the split.

3. **Prose is flushed only at a table or at the end of the page.** In `parse_html_to_chunks`, h2/h3 change
   `current_h2` but do not call `flush_prose`, so every paragraph on a page is labelled with the **last**
   heading and the last semester seen. Citations then point to the wrong section, and per-section facts
   (e.g. "Research Interests") lose their heading. Fix: call `flush_prose(...)` before updating the
   heading in the h1/h2/h3 branches.

4. **The semester carries over into sections that are not about a semester.** `current_academic_term` is
   never reset, so a "Notes" section after "Schedule … 2024" is tagged 2024-AUG. Fix: when a new h2/h3
   has no term, set `current_academic_term = None` (or keep it only inside the same table block).

5. **No research-area or lab edges in Neo4j.** `ResearchArea` nodes are created without any edge to a
   faculty member, and there are no lab nodes, so "who works on X" cannot use the graph. Fix:
   `(:Faculty)-[:RESEARCHES_IN {chunk_id, version_id, source_url}]->(:ResearchArea)` from the profile
   chunk, and `(:Faculty)-[:HEADS {...}]->(:ResearchLab)` from the "List of labs" entries.
   Related: `cleanup_url.py` ends with `MATCH (n) WHERE NOT (n)--() DELETE n`, which deletes every
   ResearchArea node (they never have edges).

6. **Coverage: 50 pages at depth 2.** The 120-question test set cites faculty pages, course
   descriptions and lab pages that a 50-page crawl may not reach, and every page not crawled is a miss.
   Suggest `max_pages` around 300, or extra seeds for `/courses/`, `/people-all/`, `/faculty/`,
   `/admissions/`.

## Medium: robustness

7. **Order of writes in `process_and_index_document`.** Old Qdrant points and Neo4j edges are deleted
   *before* Postgres commits and before the new points are upserted. If anything fails in between, the
   page disappears from search, or Postgres says "active" with no vectors behind it. Safer order:
   upsert the new points, commit Postgres (new version active), then delete the old version's points and
   edges.
8. **Nested elements are duplicated.** `find_all(["p", "li", "table", ...])` also returns `<p>` inside
   `<li>`, and `<p>`/`<li>` inside table cells, so the same text lands in a prose chunk and a row chunk.
   Fix: skip elements that have a `li` or `table` ancestor.
9. **Faculty names are not normalised.** "Y. Simmhan", "Yogesh Simmhan" and "Prof. Yogesh Simmhan" become
   different nodes. Retrieval merges them by initial + surname, but a full name plus an `aliases` list on
   the node would be cleaner.
10. **TEI batch size 2** makes ingestion slow on CPU. 16–32 is fine for BGE-M3.

## Setup and security

11. **No Postgres table script was shared.** The tables `document_versions`, `document_chunks` and
    `active_snapshots` are needed before `pipeline.py` runs. I reconstructed them from the code in
    `deploy/ingestion_schema.sql`; please share yours if it differs.
12. **`image: iisc-minio:latest`** is a locally built image, so it fails on any other machine. Use
    `minio/minio:latest` or share the Dockerfile.
13. **Passwords in code:** `cleanup_url.py`, `setup_graph.py` and `setup_stores.py` hard-code them, and
    `pipeline.py` has them as defaults. Read them from `.env` like `pipeline.py` does, with no defaults,
    and keep `.env` out of git (only `.env.example` committed).
14. **`requirements.txt`** lists crawl4ai, playwright and docling, which `pipeline.py` does not import.
    These are heavy installs.
