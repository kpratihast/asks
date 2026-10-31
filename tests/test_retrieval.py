"""Retrieval behaviour on synthetic CDS-shaped pages (hash embedder, Qdrant local mode, no network)."""
import pytest

from askiisc_retrieval.parse import parse_html, parse_term
from askiisc_retrieval.query_analysis import analyse
from askiisc_retrieval.retriever import EmbeddingModelMismatch, Retriever
from askiisc_retrieval.snapshot import JsonSnapshotProvider
from askiisc_retrieval.stores import HashEmbedder, JsonGraphStore
from conftest import DESCRIPTIONS, MTECH, PAGES, SCHEDULE


# ---------------------------------------------------------------- parsing / analysis
def test_term_parsing():
    assert parse_term("Course Schedule for Aug - Dec 2026 term") == "2026-AUG"
    assert parse_term("Course Schedule Jan 2022 - Apr 2022") == "2022-JAN"
    assert parse_term("Course Schedule for Jan - Apr  2026 term") == "2026-JAN"
    assert parse_term("Cloud Computing") is None


def test_schedule_rows_carry_their_section_term():
    page = parse_html(SCHEDULE, (PAGES / "schedule.html").read_text())
    rows = {(s.course_codes[0], s.academic_term) for s in page.sections if s.kind == "row"}
    assert ("DS252", "2026-AUG") in rows and ("DS252", "2025-AUG") in rows
    assert ("DS216", "2026-JAN") in rows
    assert not any("Home" in s.text for s in page.sections)          # nav stripped


def test_course_headings_in_paragraphs_become_sections():
    page = parse_html(DESCRIPTIONS, (PAGES / "descriptions.html").read_text())
    heads = [s.heading for s in page.sections]
    assert any(h.startswith("DS 393") for h in heads)
    assert all(s.academic_term is None for s in page.sections)       # (AUG) is an offering, not a term


def test_query_analysis():
    q = analyse("who teaches ds252?", today_term="2026-AUG")
    assert q.course_codes == ["DS252"] and not q.term_is_explicit and q.target_term == "2026-AUG"
    assert analyse("Where was DS 252 in Aug-Dec 2025?", "2026-AUG").explicit_term == "2025-AUG"
    assert analyse("DS 216 in January 2026", "2026-AUG").explicit_term == "2026-JAN"
    assert analyse("DS 252 last semester", "2026-AUG").explicit_term == "2026-JAN"
    assert analyse("Which room is CDS 202?", "2026-AUG").course_codes == []   # venues are not courses


# ---------------------------------------------------------------- invariants (FR-9, NFR-3)
def test_test_corpus_is_fully_indexed(built):
    """Guards against tests passing vacuously because a synthetic page fell under the thin-page rule."""
    assert all("version_id" in p for p in built["report"]["pages"]), built["report"]["pages"]


def test_superseded_version_never_returned(retriever):
    for q in ["minimum CGPA M.Tech", "How many credits M.Tech", "M.Tech project credits"]:
        res = retriever.retrieve(q, ["public", "student", "staff"])
        assert res.passages, q
        assert not any("SUPERSEDED-FIXTURE" in p.chunk.text for p in res.passages)
    assert retriever.audit_violations == 0


def test_scope_filter(retriever):
    guest = retriever.retrieve("lab access cards renew CDS office", ["public"])
    assert not any("STUDENT-ONLY-FIXTURE" in p.chunk.text for p in guest.passages)
    student = retriever.retrieve("lab access cards renew CDS office", ["public", "student"])
    assert "STUDENT-ONLY-FIXTURE" in student.passages[0].chunk.text


def test_embedding_model_mismatch_is_an_error(built):
    r = Retriever(HashEmbedder(model_id="some/other-model"), built["store"],
                  JsonSnapshotProvider(built["dir"] / "snapshot.json"))
    with pytest.raises(EmbeddingModelMismatch):
        r.retrieve("DS 252", ["public"])


# ---------------------------------------------------------------- term handling
def test_current_term_preferred_and_old_term_dropped(retriever):
    res = retriever.retrieve("Who teaches DS 252?", ["public"], use_graph=False)
    rows = [p.chunk for p in res.passages if p.chunk.doc_type == "schedule" and "DS252" in p.chunk.course_codes]
    assert [c.academic_term for c in rows] == ["2026-AUG"]
    assert "CDS 202" in rows[0].text


def test_explicit_term_selects_that_term(retriever):
    res = retriever.retrieve("Where was DS 252 held in Aug-Dec 2025?", ["public"], use_graph=False)
    rows = [p.chunk for p in res.passages if p.chunk.doc_type == "schedule"]
    assert rows and all(c.academic_term == "2025-AUG" for c in rows)
    assert res.passages[0].chunk.academic_term == "2025-AUG" and "CDS 419" in res.passages[0].chunk.text


def test_course_not_offered_this_term_falls_back_to_latest_past(retriever):
    res = retriever.retrieve("Who teaches DS 211?", ["public"], use_graph=False)
    rows = [p.chunk for p in res.passages if p.chunk.doc_type == "schedule" and "DS211" in p.chunk.course_codes]
    assert [c.academic_term for c in rows] == ["2025-AUG"]


def test_asked_course_ranked_first(retriever):
    res = retriever.retrieve("What are the prerequisites for DS 393?", ["public"], use_graph=False)
    assert "DS 393" in res.passages[0].chunk.section_heading


# ---------------------------------------------------------------- graph expansion + fallback (FR-9, E5, E6)
def test_graph_adds_multihop_evidence(built):
    # top_k=2 leaves no room for DS 256 among vector hits, so it can only arrive through the graph
    retriever = Retriever(built["emb"], built["store"], JsonSnapshotProvider(built["dir"] / "snapshot.json"),
                          JsonGraphStore.load(built["dir"] / "graph.json"), today_term="2026-AUG", top_k=2)
    q = "What other courses does the DS 252 instructor teach this semester?"
    no_graph = retriever.retrieve(q, ["public"], use_graph=False)
    assert not any("DS256" in p.chunk.course_codes for p in no_graph.passages)
    with_graph = retriever.retrieve(q, ["public"])
    has_256 = lambda r: any("DS256" in p.chunk.course_codes and p.chunk.doc_type == "schedule" for p in r.passages)
    graph_ps = [p for p in with_graph.passages if p.via == "graph"]
    assert not with_graph.degraded and graph_ps
    assert all(p.graph_path for p in graph_ps)                         # explains why it was added
    assert has_256(with_graph)
    assert all(p.chunk.academic_term in (None, "2026-AUG") for p in graph_ps)
    assert len(with_graph.passages) <= len(no_graph.passages) + retriever.graph_extra


def test_graph_failure_falls_back_to_vector_only(built):
    graph = JsonGraphStore.load(built["dir"] / "graph.json")
    graph.fail = True
    r = Retriever(built["emb"], built["store"], JsonSnapshotProvider(built["dir"] / "snapshot.json"),
                  graph, today_term="2026-AUG")
    res = r.retrieve("Who teaches DS 252?", ["public"])
    assert res.degraded and "unavailable" in res.degraded_reason
    assert res.passages and all(p.via != "graph" for p in res.passages)


def test_graph_timeout_falls_back(built):
    import time

    class SlowGraph(JsonGraphStore):
        def entities_of_chunks(self, chunk_ids):
            time.sleep(1.0)
            return []

    r = Retriever(built["emb"], built["store"], JsonSnapshotProvider(built["dir"] / "snapshot.json"),
                  SlowGraph(), today_term="2026-AUG", graph_timeout_s=0.1)
    res = r.retrieve("Who teaches DS 252?", ["public"])
    assert res.degraded and "timeout" in res.degraded_reason and res.passages


def test_graph_edges_respect_scope_and_version(built):
    graph = JsonGraphStore.load(built["dir"] / "graph.json")
    snap = JsonSnapshotProvider(built["dir"] / "snapshot.json").active()
    hits = graph.expand(["Course:DS252"], snap.active_version_ids, ["public"], 2, 100,
                        target_term="2026-AUG", explicit=False, now_term="2026-AUG")
    assert hits and all(h.edge.version_id in snap.active_version_ids and h.edge.scope == "public" for h in hits)


# ---------------------------------------------------------------- every real CDS term heading (2014-2026)
REAL_HEADINGS = {
    "Course Schedule for Aug - Dec 2026 term": "2026-AUG", "Course Schedule Jan 2024 - Apr 2024": "2024-JAN",
    "Course Schedule Aug 2023 - Dec 2023": "2023-AUG", "Course Schedule Feb-May (2021)": "2021-JAN",
    "Course Schedule Oct 2020 -Jan 2021": "2020-AUG", "Course Schedule Jan-Apr (2020)": "2020-JAN",
    "Course Schedule Aug-Dec (2019)": "2019-AUG", "Course Schedule for May-June (summer) 2019": "2019-SUM",
    "Course Schedule for Aug-Dec 2018": "2018-AUG", "Course Schedule - I Term (Jan-Apr, 2017)": "2017-JAN",
    "Course Schedule II Term (Jan-May 2016)": "2016-JAN", "Course Schedule - II Term (Jan-May, 2014)": "2014-JAN",
}


@pytest.mark.parametrize("heading,term", REAL_HEADINGS.items())
def test_real_cds_term_headings(heading, term):
    assert parse_term(heading) == term


def test_term_order_and_current_term():
    from datetime import date
    from askiisc_retrieval.config import current_term
    from askiisc_retrieval.query_analysis import previous_term, term_key
    assert term_key("2019-JAN") < term_key("2019-SUM") < term_key("2019-AUG") < term_key("2020-JAN")
    assert current_term(date(2026, 10, 5)) == "2026-AUG" and current_term(date(2026, 6, 1)) == "2026-SUM"
    assert previous_term("2026-AUG") == "2026-JAN" and previous_term("2026-JAN") == "2025-AUG"


# ---------------------------------------------------------------- boilerplate + aliases (found on real pages)
def test_theme_footer_and_menu_divs_are_stripped():
    page = parse_html(MTECH, (PAGES / "mtech.html").read_text())
    blob = " ".join(s.heading + " " + s.text for s in page.sections)
    assert "Locate us" not in blob and "Surfzone" not in blob and "About CDS" not in blob
    assert "64 credits" in blob


def test_instructor_name_variants_resolve_to_one_person():
    from askiisc_retrieval.graph_rules import person
    keys = {person(n) for n in ["Yogesh Simmhan", "Y. Simmhan", "Prof. Y Simmhan"]}
    assert keys == {"Person:y simmhan"}
    assert person("S.Vadhiyar") == person("Sathish Vadhiyar") == "Person:s vadhiyar"
    assert person("TBA") is None and person("Multiple faculty") is None


# ---------------------------------------------------------------- graph term rules + name seeds (q18, q19)
def _small(built, k=1):
    return Retriever(built["emb"], built["store"], JsonSnapshotProvider(built["dir"] / "snapshot.json"),
                     JsonGraphStore.load(built["dir"] / "graph.json"), today_term="2026-AUG", top_k=k)


def test_this_semester_is_an_explicit_current_term():
    q = analyse("What other courses does the DS 252 instructor teach this semester?", "2026-AUG")
    assert q.explicit_term == "2026-AUG"
    assert analyse("Who teaches DS 211?", "2026-AUG").explicit_term is None


def test_names_in_question_seed_the_graph(built):
    g = JsonGraphStore.load(built["dir"] / "graph.json")
    people = g.find_people("Which course have Sathish Vadhiyar and Yogesh Simmhan taught together?")
    assert set(people) == {"Person:s vadhiyar", "Person:y simmhan"}


def test_co_taught_course_from_an_older_term_is_found(built):
    """q18: no course code and no term in the question; the answer (DS 221) was last taught in 2024."""
    res = _small(built).retrieve("Which course have Sathish Vadhiyar and Yogesh Simmhan taught together?", ["public"])
    graph_ps = [p for p in res.passages if p.via == "graph"]
    # Real-data trap: DS 221's LATEST offering (2026) is Vadhiyar + Jain. The answer is the latest term
    # linked to BOTH named people (2024), not the latest offering.
    assert graph_ps and "DS221" in graph_ps[0].chunk.course_codes            # linked to BOTH people: first
    assert graph_ps[0].chunk.academic_term == "2024-AUG" and "Simmhan" in graph_ps[0].chunk.text


def test_cross_term_bridge_cites_only_the_asked_term(built):
    """q19: 'the DS 252 instructor' is resolved as of now (2026-AUG, a bridge), the answer is in 2026-JAN."""
    res = _small(built).retrieve("Which course did the DS 252 instructor teach in the Jan-Apr 2026 term?", ["public"])
    graph_ps = [p for p in res.passages if p.via == "graph"]
    assert any("DS256" in p.chunk.course_codes and p.chunk.academic_term == "2026-JAN" for p in graph_ps)
    assert all(p.chunk.academic_term in (None, "2026-JAN") for p in graph_ps)   # bridge rows are not cited


def test_single_entity_question_still_gets_latest_offering(built):
    """'Who teaches DS 221?' (no term): every offering links the one named course, so the latest wins."""
    res = _small(built, k=1).retrieve("Who teaches DS 221?", ["public"])
    ds221 = [p.chunk for p in res.passages if p.chunk.doc_type == "schedule" and "DS221" in p.chunk.course_codes]
    assert ds221 and ds221[0].academic_term == "2026-AUG"


def test_this_semester_graph_does_not_pull_other_terms(built):
    res = _small(built, k=2).retrieve("What other courses does the DS 252 instructor teach this semester?", ["public"])
    assert all(p.chunk.academic_term in (None, "2026-AUG") for p in res.passages if p.via == "graph")


def test_page_builder_content_survives_boilerplate_rules():
    """Regression: a broad [class*=widget] rule deleted whole Elementor-built pages (research pages)."""
    page = parse_html("https://cds.iisc.ac.in/research-new/software-and-runtime-systems/",
                      (PAGES / "elementor_research.html").read_text())
    blob = " ".join(s.heading + " " + s.text for s in page.sections)
    assert "MARS" in blob and "DREAM" in blob and "runtime frameworks" in blob
    assert "Locate us" not in blob and "Surfzone" not in blob and "About" not in blob


# ---------------------------------------------------------------- reranker + abstention signal (FR-11, FR-13)
from askiisc_retrieval.stores import KeywordReranker


def _rr(built, reranker=None, **kw):
    return Retriever(built["emb"], built["store"], JsonSnapshotProvider(built["dir"] / "snapshot.json"),
                     JsonGraphStore.load(built["dir"] / "graph.json"), today_term="2026-AUG",
                     reranker=reranker or KeywordReranker(), **kw)


def test_rerank_scores_and_orders_passages(built):
    res = _rr(built, rerank_min=None).retrieve("M.Tech project credits second year", ["public"])
    scores = [p.rerank_score for p in res.passages]
    assert all(s is not None for s in scores)
    assert scores == sorted(scores, reverse=True)                     # no course codes -> pure rerank order
    assert "project" in res.passages[0].chunk.text.lower()
    assert res.rerank_top == max(scores) and res.evidence_sufficient is None   # threshold not calibrated


def test_threshold_gives_abstention_signal(built):
    good = _rr(built, rerank_min=0.5).retrieve("M.Tech project credits", ["public"])
    bad = _rr(built, rerank_min=0.5).retrieve("hostel mess fee monthly", ["public"])
    assert good.evidence_sufficient is True and bad.evidence_sufficient is False


def test_reranker_failure_keeps_vector_order(built):
    broken = KeywordReranker()
    broken.fail = True
    res = _rr(built, reranker=broken, rerank_min=0.5).retrieve("Who teaches DS 252?", ["public"])
    plain = _rr(built).retrieve("Who teaches DS 252?", ["public"], use_rerank=False)
    assert res.rerank_error and res.passages and res.evidence_sufficient is None
    assert [p.chunk.chunk_id for p in res.passages[:3]] == [p.chunk.chunk_id for p in plain.passages[:3]]


def test_graph_passages_are_reranked_and_can_move_up(built):
    class PrefersDS221:                     # stands in for a cross-encoder that understands "together"
        model_id = "test/prefers-ds221"
        def score(self, query, passages):
            return [0.9 if "DS 221" in p else 0.2 for p in passages]
    res = _rr(built, reranker=PrefersDS221(), top_k=1, rerank_pool=1).retrieve(
        "Which course have Sathish Vadhiyar and Yogesh Simmhan taught together?", ["public"])
    graph_ps = [p for p in res.passages if p.via == "graph"]
    assert graph_ps and all(p.rerank_score is not None for p in graph_ps)
    assert res.passages[0].via == "graph" and "DS221" in res.passages[0].chunk.course_codes


def test_rerank_off_means_no_scores(built):
    res = _rr(built).retrieve("Who teaches DS 252?", ["public"], use_rerank=False)
    assert res.rerank_top is None and all(p.rerank_score is None for p in res.passages)


def test_graph_expansion_is_not_truncated_before_question_seed_ranking(built):
    """Regression (real data, v7): with many extra seeds, a 200-edge cut ranked by ALL seeds dropped the
    edge linking both people named in the question."""
    from askiisc_retrieval.stores import Edge
    g = JsonGraphStore.load(built["dir"] / "graph.json")
    snap = JsonSnapshotProvider(built["dir"] / "snapshot.json").active()
    vid = next(iter(snap.active_version_ids))
    noise = [Edge(f"Course:XX{i:03d}", "TAUGHT_BY", f"Person:n{i} noise", ("c",), vid, "public", None)
             for i in range(300)]
    noise += [Edge("Course:HUB", "REQUIRES", f"Course:XX{i:03d}", ("c",), vid, "public", None) for i in range(300)]
    g.add(noise)
    hits = g.expand(["Person:s vadhiyar", "Person:y simmhan", "Course:HUB"], snap.active_version_ids,
                    ["public"], 2, None, target_term="2026-AUG", explicit=False, now_term="2026-AUG")
    assert len(hits) > 200
    assert any(h.edge.src == "Course:DS221" for h in hits)


# ---------------------------------------------------------------- reranker input for schedule rows
from askiisc_retrieval.contract import ChunkPayload
from askiisc_retrieval.rerank_text import rerank_input, schedule_sentence


def _row(text, term="2024-AUG", doc_type="schedule"):
    return ChunkPayload("c", "d", "v", SCHEDULE, "Course Schedule", "x | DS 221", text, "public", doc_type,
                        "CDS", term, ["DS221"], "m", 0)


def test_schedule_row_becomes_a_sentence_for_the_reranker():
    c = _row("Course ID: DS 221; Credits: 3:1; Course Title: Introduction to Scalable Systems; "
             "Instructor: S.Vadhiyar & Y. Simmhan; Venue: CDS 202; First Class: 06th Aug 2024")
    assert schedule_sentence(c) == ("In the Aug-Dec 2024 term, DS 221 Introduction to Scalable Systems was taught "
                                    "by S.Vadhiyar & Y. Simmhan in CDS 202 (credits 3:1, first class 06th Aug 2024).")
    assert c.text.startswith("Course ID")                                  # stored/cited text unchanged


def test_older_column_names_are_understood():
    c = _row("Course ID: DS 221; Credits: 3:0; Name: Introduction to Scalable Systems; Instructor: S. Vadhiyar, "
             "Y. Simmhan; Days of Week: TTh; Time: 11:30-13.00; Room: CDS 202; Start date: Aug 3, 2017", "2017-AUG")
    s = schedule_sentence(c)
    assert "Aug-Dec 2017" in s and "Introduction to Scalable Systems" in s and "in CDS 202" in s and "days TTh" in s


def test_non_rows_and_incomplete_rows_are_left_alone():
    prose = _row("Students take 36 credits of coursework.", None, doc_type="admissions")
    assert rerank_input(prose) == prose.embedding_input()
    no_instructor = _row("Course ID: DS 221; Credits: 3:1; Course Title: Introduction to Scalable Systems")
    assert rerank_input(no_instructor) == no_instructor.embedding_input()


def test_reranker_receives_sentences(built):
    seen = []

    class Capture:
        model_id = "test/capture"
        def score(self, query, passages):
            seen.extend(passages)
            return [0.5] * len(passages)

    _rr(built, reranker=Capture()).retrieve("Who teaches DS 252?", ["public"])
    assert any("was taught by Yogesh Simmhan" in p for p in seen)          # sentence form scored
    assert any("Course ID: DS 252" in p for p in seen)                      # raw form scored too


def test_row_keeps_the_better_of_raw_and_sentence_score(built):
    """Regression (real data, v10): the sentence form scored q04's answer 0.134 vs 0.839 raw."""
    class PrefersRaw:
        model_id = "test/prefers-raw"
        def score(self, query, passages):
            return [0.9 if "Course ID:" in p else 0.1 for p in passages]

    class PrefersSentence:
        model_id = "test/prefers-sentence"
        def score(self, query, passages):
            return [0.9 if "was taught by" in p else 0.1 for p in passages]

    for rr in (PrefersRaw(), PrefersSentence()):
        res = _rr(built, reranker=rr, rerank_min=0.45).retrieve("Who teaches DS 252?", ["public"])
        rows = [p for p in res.passages if p.chunk.doc_type == "schedule"]
        assert rows and all(p.rerank_score == 0.9 for p in rows)
        assert res.evidence_sufficient is True


# ---------------------------------------------------------------- site template + footer lines (crawl, 2026-10-08)
from askiisc_retrieval.parse import remove_template_sections


def _page(i, news):
    return f"""<html><head><title>Post {i}</title></head><body><div class="entry-content">
    <h1>Post {i}</h1><p>Unique body text number {i} about topic {i * 7} with enough words to be a section.</p>
    <h3>Recent News</h3><ul>{''.join(f'<li>{n}</li>' for n in news)}</ul>
    <h3>Contact Us</h3><p>Department of Computational and Data Sciences, Indian Institute of Science, Bangalore 560012</p>
    </div></body></html>"""


def test_repeated_sidebar_sections_are_removed_across_the_corpus():
    news = [f"News item {k} about awards and events" for k in range(8)]
    pages = [parse_html(f"https://x/{i}", _page(i, news[i % 2: i % 2 + 6])) for i in range(12)]  # lists shift
    removed = remove_template_sections(pages)
    assert removed == {"Recent News": 12, "Contact Us": 12}
    for i, p in enumerate(pages):
        assert len(p.sections) == 1 and f"Unique body text number {i}" in p.sections[0].text


def test_template_removal_skips_small_corpora_and_unique_sections():
    news = ["a", "b", "c"]
    few = [parse_html(f"https://x/{i}", _page(i, news)) for i in range(4)]
    assert remove_template_sections(few) == {} and len(few[0].sections) == 3


def test_footer_phrase_removes_lines_not_the_whole_section():
    """Regression: a heading-less page (faculty list) lost ALL content because one footer line shared its section."""
    html = """<html><head><title>Faculty</title></head><body><div class="entry-content">
      <p>Konduri Aditya Associate Professor</p><p>Yogesh Simmhan Professor</p><p>Sathish Vadhiyar Professor</p>
      <p>&copy 2026 Department of Computational and Data Sciences. Website by: Surfzone Technologies</p>
    </div></body></html>"""
    page = parse_html("https://cds.iisc.ac.in/people-all/faculty/", html)
    text = " ".join(s.text for s in page.sections)
    assert "Yogesh Simmhan Professor" in text and "Surfzone" not in text
    assert page.boiler_lines == 1 and page.root == ".entry-content"


def test_template_block_is_kept_on_its_own_page():
    """Regression: 'Contact Us' was removed from the Contact Us page itself."""
    news = [f"News item {k}" for k in range(6)]
    pages = [parse_html(f"https://x/{i}", _page(i, news)) for i in range(12)]
    contact = parse_html("https://x/contact-us/", """<html><head><title>Contact Us - CDS</title></head><body>
      <div class="entry-content"><h1>Contact Us</h1><p>Department of Computational and Data Sciences,
      Indian Institute of Science, Bangalore 560012</p></div></body></html>""")
    remove_template_sections(pages + [contact])
    assert contact.sections and "Bangalore 560012" in contact.sections[0].text
    assert all(len(p.sections) == 1 for p in pages)


def test_short_but_real_pages_are_indexed(tmp_path):
    from askiisc_retrieval.ingest_stub import MIN_CONTENT_CHARS, PageInput, build_index
    from askiisc_retrieval.stores import HashEmbedder, QdrantStore
    profile = """<html><head><title>Yogesh Simmhan</title></head><body><div class="entry-content">
      <h1>Yogesh Simmhan</h1><p>Professor. Research interests: distributed systems, cloud computing,
      Internet of Things (IoT), big data platforms.</p></div></body></html>"""
    empty = """<html><head><title>Research</title></head><body><div class="entry-content"><h1>Research</h1>
      </div></body></html>"""
    emb = HashEmbedder()
    store = QdrantStore("t", emb.dim, path=tmp_path / "q")
    rep = build_index([PageInput("https://x/p/", profile, "people", "public"),
                       PageInput("https://x/r/", empty, "research", "public")],
                      emb, store, tmp_path / "g.json", tmp_path / "s.json")
    status = {r["url"]: r for r in rep["pages"]}
    assert "version_id" in status["https://x/p/"]                    # ~150 chars of real content: indexed
    assert "NOT INDEXED" in status["https://x/r/"]["status"]
    assert MIN_CONTENT_CHARS <= 120
    store.close()


def test_rerank_sentences_for_ingestion_v3_formats():
    """v3 rows use ' | ' and profile headings start with the page title (2026-10-25 live run: DS 221 at rank 7)."""
    from askiisc_retrieval.contract import ChunkPayload
    from askiisc_retrieval.rerank_text import profile_sentence, schedule_sentence

    def chunk(heading, text, doc_type, term=None):
        return ChunkPayload(chunk_id="c", document_id="d", version_id="v", source_url="u", title="t",
                            section_heading=heading, text=text, access_scope="public", doc_type=doc_type,
                            department="CDS", academic_term=term, course_codes=[], embedding_model="m", ordinal=0)
    row = chunk("Course Schedule > Course Schedule for Aug - Dec 2024 term | DS221",
                "Course ID: DS 221 | Credits: 3:1 | Course Title: Introduction to Scalable Systems | "
                "Instructor: S.Vadhiyar & Y. Simmhan | Venue: CDS 202", "schedule", "2024-AUG")
    assert schedule_sentence(row) == ("In the Aug-Dec 2024 term, DS 221 Introduction to Scalable Systems was taught by "
                                      "S.Vadhiyar & Y. Simmhan in CDS 202 (credits 3:1).")
    prof = chunk("Yogesh Simmhan - Computational and Data Sciences - IISc > Research Interests",
                 "Distributed systems, Cloud computing", "people")
    assert profile_sentence(prof) == "Yogesh Simmhan works on research in Distributed systems, Cloud computing."
