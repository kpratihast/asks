"""Link ingestion and retrieval using unrelated course names and HTML layouts."""
from dataclasses import replace

import httpx
import pytest

from askiisc_retrieval import config
from askiisc_retrieval.ingest_stub import PageInput, build_index
from askiisc_retrieval.linked_fetch import _download
from askiisc_retrieval.parse import parse_html
from askiisc_retrieval.retriever import Retriever
from askiisc_retrieval.snapshot import JsonSnapshotProvider, StaticSnapshotProvider
from askiisc_retrieval.stores import HashEmbedder, JsonGraphStore, QdrantStore
from askiisc_serving.retrieval_client import LocalRetrieval
from test_serving import FakeModel, ask, make

BASE = 'https://example.edu'


def test_mailto_is_preserved_with_its_context_and_other_links():
    page = parse_html(BASE+'/classes', '''<main><p>Ada Mora
        <a href="mailto:Ada%2EMora@example.edu?subject=class">mail</a>
        <a href="/profiles/ada/">WWW</a></p>
        <footer><a href="mailto:office@example.edu">help</a></footer></main>''')
    sec, = page.sections
    assert sec.contacts == ['Ada Mora mail (email: ada.mora@example.edu) WWW']
    assert sec.links == [(sec.text, BASE+'/profiles/ada')]


def test_link_fetch_does_not_follow_an_unapproved_redirect():
    requests = []

    def serve(req):
        requests.append(str(req.url))
        if req.url.path == '/robots.txt':
            return httpx.Response(200, text='User-agent: *\nAllow: /')
        return httpx.Response(302, headers={'location': 'https://unapproved.example/secret'})

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        assert _download(client, BASE+'/course', {'example.edu'}, {}) is None
    assert requests == [BASE+'/robots.txt', BASE+'/course']


@pytest.fixture
def contact_index(tmp_path):
    pages = [
        PageInput(BASE+'/schedule/', '''<main><h2>Aug-Dec 2026</h2><table>
            <tr><th>Course ID</th><th>Title</th><th>Instructor</th><th>Website</th></tr>
            <tr><td>EC 731</td><td>Coherent Optics and Imaging Systems Laboratory</td>
            <td>Ada Mora</td><td><a href="/classes/optics/">WWW</a></td></tr></table></main>''',
                  'schedule', 'public', 'ECE'),
        PageInput(BASE+'/classes/optics/', '''<main><h1>Optics course notes</h1>
            <p>Refer to the manual for guidelines on equipment. The lab groups are listed in the first edition.
            Instructor contact details are on the <a href="../staff/mora/">instructor profile</a>.</p></main>''',
                  'page', 'public', 'ECE'),
        PageInput(BASE+'/classes/staff/mora/', '''<main><h1>Ada Mora</h1>
            <p>My lab studies imaging and lasers for remote optical systems. You can get in touch through the
            <a href="mailto:ada.mora@example.edu">published mailbox</a> during office hours.</p></main>''',
                  'people', 'public', 'ECE'),
    ]
    for number in range(5):
        pages.append(PageInput(f'{BASE}/notice/{number}', f'''<main><h2>EC 731 email questions</h2>
            <p>The course instructor email is described in the course schedule and not given in this notice.
            Notice {number} applies to the course EC 731. Email queries to instructors are common.</p></main>''',
                               'page', 'public', 'ECE'))
    embedder = HashEmbedder()
    store = QdrantStore('links', embedder.dim, path=tmp_path/'q')
    graph, snapshot = tmp_path/'graph.json', tmp_path/'snapshot.json'
    build_index(pages, embedder, store, graph, snapshot)
    r = Retriever(embedder, store, JsonSnapshotProvider(snapshot), JsonGraphStore.load(graph),
                  top_k=4, today_term='2026-AUG', rerank_min=None)
    try:
        yield r, store
    finally:
        store.close()


QUESTION = 'What is the email address of the instructor for ECE EC 731?'


def test_link_walk_finds_instructor_contact_in_two_hops(contact_index):
    r, _ = contact_index
    result = r.retrieve(QUESTION, ['public'])
    proof, = result.graph_proofs
    assert [s['relation'] for s in proof['steps']] == ['LINKS_TO', 'CONTAINS', 'LINKS_TO', 'CONTAINS']
    assert {c.source_url for c in result.citations if set(c.chunk_ids) & set(proof['required_chunk_ids'])} == {
        BASE+'/schedule/', BASE+'/classes/optics/', BASE+'/classes/staff/mora/'
    }
    assert any(p.graph_support.get('contact_addresses') == ['ada.mora@example.edu'] for p in result.passages)
    baseline = r.retrieve(QUESTION, ['public'], use_graph=False)
    assert not baseline.graph_proofs
    assert all('ada.mora@example.edu' not in p.chunk.text for p in baseline.passages)


def test_stale_link_target_is_not_used(contact_index):
    r, store = contact_index
    stale = next(c for c in store.document_chunks(BASE+'/classes/staff/mora/', r.snapshots.active().active_version_ids,
                                                  ['public'], r.embedder.model_id))
    snapshot = r.snapshots.active()
    r.snapshots = StaticSnapshotProvider(replace(snapshot, active_version_ids=snapshot.active_version_ids - {stale.version_id}))
    result = r.retrieve(QUESTION, ['public'])
    assert not result.graph_proofs
    assert all(stale.chunk_id != p.chunk.chunk_id for p in result.passages)


def test_guessed_address_is_rejected_even_with_citations(contact_index):
    r, _ = contact_index
    model = FakeModel('The email is guessed@example.edu [S1][S2][S3].',
                      'The email is ada.mora@example.edu [S1][S2][S3].')
    with make(model, LocalRetrieval(r), regenerate_on_bad_citations=False)[0] as client:
        denied = ask(client, q=QUESTION).json()
        accepted = ask(client, q=QUESTION).json()
    assert denied['status'] == 'fallback' and denied['verification']['unsupported_email']
    assert accepted['status'] == 'answered'


def test_directory_contact_requires_exact_instructor_name(contact_index):
    r, store = contact_index
    anchor = next(c for c in store.document_chunks(BASE+'/schedule/', r.snapshots.active().active_version_ids,
                                                    ['public'], r.embedder.model_id))
    directory = replace(anchor, chunk_id='directory', doc_type='people', links=[],
                        section_heading='Faculty directory', source_url=BASE+'/directory',
                        text='Ada Mora, Assistant Professor, ada.mora@example.edu',
                        contacts=['Ada Mora, Assistant Professor, ada.mora@example.edu'], course_codes=[])
    unrelated = replace(directory, chunk_id='unrelated', text='Arun Mora, wrong@example.edu',
                        contacts=['Arun Mora, wrong@example.edu'])
    anchor = replace(anchor, links=[])
    store.upsert([anchor, directory, unrelated], r.embedder.embed_documents(
        [c.embedding_input() for c in (anchor, directory, unrelated)]))
    from askiisc_retrieval.graph_rules import build_edges
    r.graph = JsonGraphStore(build_edges([anchor, directory, unrelated]))
    result = r.retrieve(QUESTION, ['public'])
    proof, = result.graph_proofs
    assert [s['relation'] for s in proof['steps']] == ['INSTRUCTOR', 'PUBLISHED_EMAIL']
    assert 'directory' in proof['required_chunk_ids']
    assert 'unrelated' not in proof['required_chunk_ids']


def test_single_source_policy_evidence_does_not_displace_other_candidates():
    from askiisc_serving.evidence_selection import evidence_bundle

    graph = {'marker': 'S1', 'chunk_ids': ['summary'], 'graph_support': [
        {'policy_bundle_id': 'eligibility', 'required_chunk_ids': ['summary']},
    ]}
    detail = {'marker': 'S2', 'chunk_ids': ['criteria']}
    ranked, required = evidence_bundle('What are the admission requirements?', [graph, detail])
    assert ranked == [graph, detail] and not required


def test_links_in_course_headings_keep_their_destination():
    parsed = parse_html(BASE+'/catalogue', '''<main>
    <h2>EE 731 <a href="/courses/sensors/">WWW</a></h2>
    <p>Course description for the sensor systems laboratory.</p></main>''')
    assert parsed.sections[0].links == [('WWW', BASE+'/courses/sensors')]
