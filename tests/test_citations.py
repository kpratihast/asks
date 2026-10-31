"""Citations: markers, grouping, prompt block, answer checking, and that only audited passages are cited."""
from askiisc_retrieval.citations import build_citations, check_answer, format_context, render_sources
from askiisc_retrieval.contract import ChunkPayload, Passage, RetrievalResult


def _p(url, heading, text, cid, version="v1", score=0.5, term=None):
    c = ChunkPayload(cid, "d", version, url, "Page", heading, text, "public", "policy", "CDS", term, [], "m", 0)
    return Passage(c, 0.5, rerank_score=score)


def _result(passages):
    return RetrievalResult("q", passages, "snap", citations=build_citations(passages))


def test_markers_follow_rank_and_same_section_shares_one():
    ps = [_p("u1", "A", "first part", "c1", score=0.9), _p("u2", "B", "other", "c2", score=0.8),
          _p("u1", "A", "second part", "c3", score=0.7)]
    cits = build_citations(ps)
    assert [c.marker for c in cits] == ["S1", "S2"]
    assert cits[0].chunk_ids == ["c1", "c3"] and [p.marker for p in ps] == ["S1", "S2", "S1"]
    assert cits[0].rerank_score == 0.9


def test_same_page_different_sections_or_versions_are_separate_sources():
    ps = [_p("u1", "A", "x", "c1"), _p("u1", "B", "y", "c2"), _p("u1", "A", "z", "c3", version="v2")]
    assert len(build_citations(ps)) == 3


def test_context_block_labels_and_budget():
    res = _result([_p("u1", "A", "alpha text", "c1"), _p("u2", "B", "beta text", "c2", term="2026-AUG")])
    ctx = format_context(res)
    assert ctx.startswith("[S1] Page > A\nalpha text") and "[S2] Page > B | term 2026-AUG\nbeta text" in ctx
    assert "[S2]" not in format_context(res, max_chars=25)               # lowest-ranked source dropped first


def test_answer_citation_check():
    res = _result([_p("u1", "A", "x", "c1"), _p("u2", "B", "y", "c2")])
    good = check_answer("DS 252 is taught by Yogesh Simmhan [S1].", res)
    assert good.ok and good.used == ["S1"] and good.sources[0].source_url == "u1"
    fake = check_answer("It is in CDS 202 [S1][S7].", res)
    assert not fake.ok and fake.unknown == ["S7"]
    none = check_answer("It is in CDS 202.", res)
    assert not none.ok and not none.cited
    assert "[S2]" not in render_sources(res, good.used) and "u1" in render_sources(res, good.used)


def test_retriever_citations_come_only_from_audited_passages(retriever):
    res = retriever.retrieve("minimum CGPA M.Tech", ["public"])
    assert res.citations and all(p.marker for p in res.passages)
    assert {m for p in res.passages for m in [p.marker]} == {c.marker for c in res.citations}
    assert not any("SUPERSEDED-FIXTURE" in c.excerpt for c in res.citations)
    assert all(c.version_id in retriever.snapshots.active().active_version_ids for c in res.citations)


def test_result_to_dict_is_json_ready(retriever):
    import json
    from askiisc_retrieval.citations import result_to_dict
    res = retriever.retrieve("Who teaches DS 252?", ["public"])
    d = json.loads(json.dumps(result_to_dict(res)))
    assert d["citations"][0]["marker"] == "S1" and d["citations"][0]["source_url"].startswith("https://")
    assert {p["marker"] for p in d["passages"]} == {c["marker"] for c in d["citations"]}
    assert "version_id" in d["citations"][0] and "text" in d["passages"][0]
