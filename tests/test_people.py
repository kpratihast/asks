"""People / lab questions: the people pass, research-interest and lab edges, name aliases."""
import pytest

from askiisc_retrieval.contract import ChunkPayload
from askiisc_retrieval.graph_rules import edges_for_chunk, person, split_topics
from askiisc_retrieval.ingest_stub import PageInput, build_index
from askiisc_retrieval.query_analysis import analyse
from askiisc_retrieval.retriever import Retriever
from askiisc_retrieval.snapshot import JsonSnapshotProvider
from askiisc_retrieval.stores import HashEmbedder, JsonGraphStore, KeywordReranker, QdrantStore

PEOPLE = """<html><head><title>People</title></head><body><main><h1>People</h1>
<h3>Anirban Chakraborty</h3><h4>Research Interests</h4>
<p>Visual analytics, Data association over graphs, Data fusion and consistency, video surveillance.</p>
<h3>Ratikanta Behera</h3><h4>Research Interests</h4>
<p>Tensor Decompositions, Neural Networks, Numerical Linear Algebra, Wavelets in Scientific Computing.</p>
<h3>Venkatesh Babu R</h3><h4>Research Interests</h4>
<p>Video Analytics, Multimedia, Image/Video Processing, Compression, Computer Vision and Machine Learning.</p>
</main></body></html>"""

LABS = """<html><head><title>Research Admissions 2026</title></head><body><main><h1>Research Admissions 2026</h1>
<h2>List of labs that are admitting students</h2>
<h3>5. Visual Computing Lab (VCL)</h3>
<p>Faculty: Anirban Chakraborty Website: http://visual-computing.in/ Lab Description: We develop new
algorithms for images and video, with students working on recognition and tracking problems.</p>
</main></body></html>"""

ABOUT = """<html><head><title>About CDS</title></head><body><main><h1>About CDS</h1>
<p>The Department of Computational and Data Sciences works across scientific computing, systems, data
science and machine learning. The department offers PhD, M.Tech (Research) and M.Tech programmes and hosts
many seminars, workshops and outreach events every year for students and the public.</p>
</main></body></html>"""

BASE = "https://cds.iisc.ac.in/"


@pytest.fixture(scope="module")
def people_index(tmp_path_factory):
    d = tmp_path_factory.mktemp("people")
    pages = [PageInput(BASE + "people/", PEOPLE, "people", "public"),
             PageInput(BASE + "admissions/research/", LABS, "admissions", "public"),
             PageInput(BASE + "about/", ABOUT, "about", "public")]
    emb = HashEmbedder()
    store = QdrantStore("people", emb.dim, path=d / "qdrant")
    build_index(pages, emb, store, d / "graph.json", d / "snapshot.json", d / "chunks.jsonl")
    yield {"dir": d, "emb": emb, "store": store}
    store.close()


def _r(ix, **kw):
    kw.setdefault("rerank_min", None)
    return Retriever(ix["emb"], ix["store"], JsonSnapshotProvider(ix["dir"] / "snapshot.json"),
                     JsonGraphStore.load(ix["dir"] / "graph.json"), today_term="2026-AUG",
                     reranker=KeywordReranker(), **kw)


def test_people_intent_detection():
    assert analyse("Which professor works on tensor decompositions?").wants_people
    assert analyse("I like GPU computing; whom should I approach?").wants_people
    assert analyse("Which lab should I join?").wants_people
    assert not analyse("What are the prerequisites for DS 393?").wants_people
    assert not analyse("What is the last date of classes?").wants_people


def test_split_topics_keeps_brackets_and_drops_prose():
    assert split_topics("Artificial Intelligence (AI), GenAI, and High Performance Computing (HPC).") == \
        ["artificial intelligence (ai)", "genai", "high performance computing (hpc)"]
    long = "x" * 90
    assert split_topics(f"{long}, Cloud computing") == ["cloud computing"]


def test_trailing_initial_names_match_the_initial_first_spelling():
    assert person("Venkatesh Babu R") == person("R. Venkatesh Babu") == "Person:r babu"
    assert person("Sekar K") == person("K. Sekar") == "Person:k sekar"
    assert person("Jayant R Haritsa") == "Person:j haritsa"


def _chunk(heading, text, doc_type):
    return ChunkPayload("c1", "d", "v1", "u", "t", heading, text, "public", doc_type, "CDS", None, [], "m", 0)


def test_research_interest_and_lab_edges():
    e = edges_for_chunk(_chunk("People > Danish Pruthi > Research Interests",
                               "Natural Language Processing (NLP), Deep Learning", "people"))
    assert {(x.src, x.rel, x.dst) for x in e} == {
        ("Person:d pruthi", "WORKS_ON", "Topic:natural language processing (nlp)"),
        ("Person:d pruthi", "WORKS_ON", "Topic:deep learning")}
    assert all(x.chunk_ids == ("c1",) for x in e)                     # every edge cites its chunk
    lab = edges_for_chunk(_chunk("Admissions > List of labs that are admitting students > 5. Visual Computing Lab (VCL)",
                                 "Faculty: Anirban Chakraborty Website: http://x/ Lab Description: ...", "admissions"))
    assert [(x.src, x.rel, x.dst) for x in lab] == [("Person:a chakraborty", "HEADS", "Lab:Visual Computing Lab (VCL)")]
    # a research-interest heading on a non-people page is not trusted as a profile
    assert edges_for_chunk(_chunk("Lab > Research Interests", "Graphs, Vision", "page")) == []


def test_people_pass_brings_profiles_to_the_reranker(people_index):
    q = "Which professor works on tensor decompositions?"
    res = _r(people_index, top_k=1, rerank_pool=1).retrieve(q, ["public"])
    assert "Ratikanta Behera" in res.passages[0].chunk.section_heading
    off = _r(people_index, top_k=1, rerank_pool=1, people_k=0).retrieve(q, ["public"])
    assert all(p.via != "people" for p in off.passages)


def test_people_pass_needs_people_intent(people_index):
    res = _r(people_index).retrieve("Tell me about the department", ["public"])
    assert all(p.via != "people" for p in res.passages)


def test_graph_adds_the_lab_of_a_matching_person(people_index):
    res = _r(people_index, top_k=1, rerank_pool=1).retrieve("Who works on visual analytics?", ["public"])
    assert "Anirban Chakraborty" in res.passages[0].chunk.section_heading
    labs = [p for p in res.passages if p.via == "graph" and "Visual Computing Lab" in p.chunk.section_heading]
    assert labs and "HEADS" in labs[0].graph_path
    assert any(c.source_url == BASE + "admissions/research/" for c in res.citations)


from askiisc_retrieval.rerank_text import rerank_input


def test_profiles_and_lab_listings_get_a_sentence_for_the_reranker():
    prof = _chunk("People > Danish Pruthi > Research Interests", "Natural Language Processing (NLP), Deep Learning.", "people")
    assert rerank_input(prof) == "Danish Pruthi works on research in Natural Language Processing (NLP), Deep Learning."
    lab = _chunk("Admissions > List of labs that are admitting students > 5. Visual Computing Lab (VCL)",
                 "Faculty: Anirban Chakraborty Website: http://x/ Lab Description: We build vision systems.", "admissions")
    assert rerank_input(lab) == "The Visual Computing Lab (VCL) is led by Anirban Chakraborty. We build vision systems."
    other = _chunk("About CDS", "Some text", "about")
    assert rerank_input(other) == other.embedding_input()             # everything else is unchanged


def test_anyone_and_someone_are_people_questions():
    assert analyse("Is anyone in the department doing research on quantum computing?").wants_people
    assert analyse("Is there someone working on graphs?").wants_people
