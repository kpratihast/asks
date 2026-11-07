"""Exercise mixed department data through the real parsers, Qdrant and serving route."""
from dataclasses import replace

import httpx
import pytest

from askiisc_retrieval import config, crawl, fetch
from askiisc_retrieval.admissions import normalize_html
from askiisc_retrieval.departments import mentioned
from askiisc_retrieval.ingest_stub import PageInput, build_index, load_page_inputs
from askiisc_retrieval.private_documents import DocumentCatalog
from askiisc_retrieval.query_analysis import analyse, institute_admissions
from askiisc_retrieval.retriever import Retriever
from askiisc_retrieval.snapshot import JsonSnapshotProvider, StaticSnapshotProvider
from askiisc_retrieval.stores import HashEmbedder, JsonGraphStore, QdrantStore
from askiisc_serving.retrieval_client import LocalRetrieval

from test_connections import TERM, SCOPES, course, exam, make as make_connections
from test_serving import FakeModel, ask, make

PANELS = """
<div class="col-md-9"><h1>M.Tech. admissions</h1>
<p>Common policy: all applicants use the IISc application portal.</p>
<ul class="resp-tabs-list"><li>Computer Science and Automation</li>
<li>Electrical Communication Engineering</li><li>Computational and Data Sciences</li></ul>
<div class="resp-tabs-container">
<div class="fruitful_tab">Discipline: Data Science
Department/Centre/Unit: Computational and Data Sciences
<p>Selection procedure: A written test is mandatory for admission to the Computational and Data Sciences programme.
Shortlisted applicants should bring identification for the written test.</p></div>
<div class="fruitful_tab">Discipline: Electronics and Communication Engineering
Department/Centre/Unit: Electrical Communication Engineering
<p>Selection procedure: An oral interview is required for admission to the Electrical Communication Engineering programme.
Shortlisted applicants should attend the oral interview at the department office.</p></div>
<div class="fruitful_tab">Discipline: Computer Science &amp; Engineering
Department/Centre/Unit: Computer Science and Automation
<p>Selection procedure: A GATE paper in CS is required for the Computer Science and Automation programme.
Applicants use their GATE score for this programme selection procedure.</p></div>
</div><h2>Common rules</h2><p>For all candidates, submit the application through the institute admissions portal.
The institute portal confirms submission and lists the payment status for your application.</p></div>
"""


def test_registry_allows_a_new_department_without_code_changes(tmp_path, monkeypatch):
    registry = tmp_path / 'departments.yaml'
    registry.write_text('departments:\n  BIO:\n    name: Department of Biology\n    aliases: [Life Sciences]\n')
    monkeypatch.setenv('ASKIISC_DEPARTMENT_REGISTRY', str(registry))
    assert mentioned('Life Sciences and BIO research') == ('BIO',)
    assert mentioned('biological topics') == ()


def test_cached_and_crawled_pages_retain_owner_and_respect_rollout_switch(tmp_path, monkeypatch):
    profile = tmp_path / 'sources.yaml'
    profile.write_text("""department: CDS
pages:
  - {url: 'https://csa.example.edu/admissions/', department: CSA, doc_type: admissions, scope: public}
crawls:
  - {seeds: ['https://cds.example.edu/'], department: CDS, max_depth: 0, max_pages: 1}
  - {seeds: ['https://ece.example.edu/'], department: ECE, max_depth: 0, max_pages: 1}
""")
    monkeypatch.setenv('ASKIISC_SOURCES_FILE', str(profile))
    monkeypatch.setenv('ASKIISC_SOURCE_DEPARTMENTS', 'CDS,CSA,ECE')
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path)
    monkeypatch.setattr(crawl, 'RAW_DIR', tmp_path)
    monkeypatch.setattr(DocumentCatalog, 'page_inputs', lambda self: [])
    source = 'https://csa.example.edu/admissions/'
    (tmp_path / fetch.raw_name(source)).write_text('<main>CSA source</main>')
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(
            200, headers={'content-type': 'text/html'}, text='<main>Example source</main>'))) as client:
        for group in fetch.load_sources()['crawls']:
            crawl.crawl(group['seeds'], group['max_depth'], group['max_pages'], client, fetch.raw_name,
                        delay_s=0, respect_robots=False, department=group['department'])
    assert {page.department for page in load_page_inputs(False)} == {'CDS', 'CSA', 'ECE'}
    monkeypatch.setenv('ASKIISC_SOURCE_DEPARTMENTS', 'CDS')
    assert [page.department for page in load_page_inputs(False)] == ['CDS']


@pytest.fixture
def mixed_retriever(tmp_path):
    url = 'https://www.iisc.ac.in/admissions/m-tech-m-des/'
    normalized = normalize_html(PANELS, url, 'programmes', ('CDS', 'CSA', 'ECE'))
    embedder = HashEmbedder()
    store = QdrantStore('mixed', embedder.dim, path=tmp_path/'q')
    graph_path, snapshot_path = tmp_path/'graph.json', tmp_path/'snapshot.json'
    build_index([PageInput(url, normalized, 'admissions', 'public', 'IISc', tagged_departments=True)],
                embedder, store, graph_path, snapshot_path)
    retriever = Retriever(embedder, store, JsonSnapshotProvider(snapshot_path),
                          JsonGraphStore.load(graph_path), today_term=TERM)
    try:
        yield retriever
    finally:
        store.close()


@pytest.mark.parametrize('name,code,fact', [
    ('Computer Science and Automation', 'CSA', 'GATE paper in CS'),
    ('ECE', 'ECE', 'oral interview'),
    ('CDS', 'CDS', 'written test is mandatory'),
])
def test_department_selection_filters_shared_document_sections(mixed_retriever, name, code, fact):
    result = mixed_retriever.retrieve(f'What is the selection procedure for M.Tech. at {name}?', ['public'])
    assert set(mixed_retriever.snapshots.active().departments) == {'CDS', 'CSA', 'ECE', 'IISc'}
    assert {p.chunk.department for p in result.passages} <= {code, 'IISc'}
    assert any(fact in p.chunk.text for p in result.passages if p.chunk.department == code)
    assert all(c.department in (code, 'IISc') for c in result.citations)


def test_unqualified_programme_requests_clarification(mixed_retriever):
    model = FakeModel('This would be an ungrounded answer [S1].')
    with make(model, retrieval=LocalRetrieval(mixed_retriever))[0] as client:
        response = ask(client, q='What is the M.Tech. selection process?')
    assert response.json()['reason'] == 'department_required'
    assert not model.requests


@pytest.mark.parametrize('query', [
    'How do I apply for a research programme at IISc?',
    'Where is the application portal?',
    'What are the M.Tech admission routes at the Indian Institute of Science?',
])
@pytest.mark.parametrize('use_graph', [False, True])
def test_institute_admissions_only_uses_institute_passages(mixed_retriever, query, use_graph):
    result = mixed_retriever.retrieve(query, ['public'], use_graph=use_graph)
    assert result.department_status == 'institute_admissions'
    assert result.passages and {p.chunk.department for p in result.passages} == {'IISc'}
    assert all(c.department == 'IISc' for c in result.citations)


def test_institute_question_does_not_fall_back_to_the_only_indexed_department(tmp_path):
    embedder = HashEmbedder()
    store = QdrantStore('single-dept', embedder.dim, path=tmp_path/'q')
    graph_path, snapshot_path = tmp_path/'graph.json', tmp_path/'snapshot.json'
    body = '<main><h1>CDS research admissions</h1><p>' + ('Applicants apply to the CDS programme. ' * 4) + '</p></main>'
    try:
        build_index([PageInput('https://cds.example.test/research/', body, 'admissions', 'public')],
                    embedder, store, graph_path, snapshot_path)
        retriever = Retriever(embedder, store, JsonSnapshotProvider(snapshot_path), JsonGraphStore.load(graph_path))
        model = FakeModel('An answer should never be generated [S1].')
        with make(model, retrieval=LocalRetrieval(retriever))[0] as client:
            for mode in ('full', 'vector'):
                result = ask(client, q='How do I apply for a research programme?', mode=mode).json()
                assert result['status'] == 'abstained' and not result['citations']
        assert not model.requests
    finally:
        store.close()


def test_non_admissions_and_named_department_questions_keep_their_scope():
    for question in ('How do I apply to ECE at IISc?', 'Which IISc labs work on applications of AI?'):
        assert not institute_admissions(analyse(question))


def test_seed_admissions_normalization_keeps_shared_and_department_sections_separate(tmp_path, monkeypatch):
    from askiisc_retrieval.parse import parse_html

    url = 'https://example.test/admissions/'
    profile = tmp_path / 'sources.yaml'
    profile.write_text(f'''department: CDS
pages:
  - {{url: '{url}', department: IISc, doc_type: admissions, scope: public, normalize: programmes}}
''')
    monkeypatch.setenv('ASKIISC_SOURCES_FILE', str(profile))
    monkeypatch.setenv('ASKIISC_SOURCE_DEPARTMENTS', 'ECE')
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path)
    monkeypatch.setattr(DocumentCatalog, 'page_inputs', lambda self: [])
    (tmp_path / fetch.raw_name(url)).write_text(PANELS)
    [page] = load_page_inputs(False)
    parsed = parse_html(url, page.html, tagged_departments=page.tagged_departments)
    assert {section.department or page.department for section in parsed.sections} == {'IISc', 'ECE'}
    assert 'written test is mandatory' not in page.html


def test_compare_uses_both_departments_and_rejects_one_sided_citation(mixed_retriever):
    query = 'Compare the M.Tech. selection procedure in CSA and ECE.'
    model = FakeModel('CSA requires a GATE CS paper [S1].',
                      'CSA requires a GATE CS paper [S1]. ECE requires an oral interview [S2].')
    with make(model, retrieval=LocalRetrieval(mixed_retriever), regenerate_on_bad_citations=False)[0] as client:
        first = ask(client, q=query).json()
        second = ask(client, q=query).json()
    assert first['status'] == 'fallback' and first['verification']['missing_department_citations'] == ['ECE']
    assert second['status'] == 'answered'
    assert {c['department'] for c in second['citations'] if c['used']} == {'CSA', 'ECE'}
    assert second['trace']['department_status'] == 'comparison'


def test_public_question_does_not_hit_another_departments_private_gate():
    catalog = DocumentCatalog()
    assert catalog.denied_scopes('What is the CSA course schedule?', ['public']) == []
    assert 'student' in catalog.denied_scopes('What is the CDS course schedule?', ['public'])
    assert 'faculty' in catalog.denied_scopes('When is the DS252 final exam?', ['public'])


def test_scoped_exam_graph_ignores_another_departments_identical_teaching_slot(tmp_path):
    chunks = [course('DS252'), exam('exam-cds'),
              replace(course('EC701'), department='ECE'),
              replace(exam('exam-ece', when='2026-12-02'), department='ECE')]
    retriever, store, snapshot = make_connections(tmp_path, chunks)
    try:
        retriever.snapshots = StaticSnapshotProvider(replace(snapshot, departments=('CDS', 'ECE')))
        cds = retriever.retrieve('When is the DS252 final exam?', SCOPES)
        ece = retriever.retrieve('When is the ECE EC701 final exam?', SCOPES)
        assert cds.connection_status == ece.connection_status == 'complete'
        assert set(cds.graph_proofs[0]['required_chunk_ids']) == {'course-DS252', 'exam-cds'}
        assert set(ece.graph_proofs[0]['required_chunk_ids']) == {'course-EC701', 'exam-ece'}
    finally:
        store.close()


def test_divi_body_footer_flag_keeps_the_department_content():
    from askiisc_retrieval.parse import parse_html

    body = '''<body class="et-tb-has-footer"><div class="entry-content">
    <h2>Admission interview</h2><p>There is no written test for this programme.</p>
    <div class="actual-footer">Irrelevant footer</div></div></body>'''
    page = parse_html('https://example.edu/selection/', body)
    assert page.sections[0].text == 'There is no written test for this programme.'
