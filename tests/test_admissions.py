from pathlib import Path
import json
from types import SimpleNamespace
import pytest
from bs4 import BeautifulSoup
from askiisc_retrieval.admissions import normalize_html,table_grid,fees_table,pdf_html
from askiisc_retrieval.parse import parse_html
from askiisc_retrieval.query_analysis import course_exam_intent,requested_relations
from askiisc_retrieval.private_documents import DocumentCatalog


def test_pdf_seed_keeps_pages_and_course_continuations_in_scope(tmp_path, monkeypatch):
    import pypdf
    from askiisc_retrieval import config, fetch
    from askiisc_retrieval.ingest_stub import load_page_inputs

    url = 'https://example.test/scheme.pdf'
    filename = fetch.raw_name(url)
    assert filename.endswith('.pdf')
    (tmp_path / filename).touch()
    sample = [
        'Common rules for the institute. Doctoral students take the prescribed research training credits.',
        'E2 202 ( AUG ) 3 : 0\nRandom Processes\nThe first page covers probability spaces and random variables.',
        'The continuation covers stationary processes and spectral representation for time series.',
        'Unselected department page. This material is deliberately outside the configured source scope.',
    ]
    pages = [SimpleNamespace(extract_text=lambda text=text, **_: text) for text in sample]
    monkeypatch.setattr(pypdf, 'PdfReader', lambda _: SimpleNamespace(pages=pages))
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path)
    monkeypatch.setenv('ASKIISC_SOURCE_DEPARTMENTS', 'ECE')
    monkeypatch.setattr(DocumentCatalog, 'page_inputs', lambda self: [])
    spec = {'url': url, 'scope': 'public', 'department': 'IISc', 'doc_type': 'catalogue',
            'pdf': {'title': 'Scheme of Instruction', 'page_ranges': {'IISc': [1, 1], 'ECE': [2, 3], 'CSA': [4, 4]}}}
    monkeypatch.setattr(fetch, 'load_sources', lambda: {'department': 'CDS', 'pages': [spec], 'crawls': []})

    [page] = load_page_inputs(False)
    sections = parse_html(url, page.html, tagged_departments=page.tagged_departments).sections
    assert [s.department for s in sections] == ['IISc', 'ECE', 'ECE']
    assert sections[2].course_codes == ['E2202']
    assert 'PDF page 3' in sections[2].heading and 'stationary processes' in sections[2].text
    with pytest.raises(ValueError, match='Overlapping PDF'):
        pdf_html(tmp_path / filename, 'Scheme', page_ranges={'IISc': [1, 2], 'ECE': [2, 3]})
    with pytest.raises(ValueError, match='page boundary'):
        pdf_html(tmp_path / filename, 'Scheme', max_pages=1, page_ranges={'IISc': [1, 1], 'ECE': [2, 2]})

@pytest.mark.parametrize('query',['What GATE exam is accepted for CDS admissions?','When is the entrance examination timetable published?','Which qualifying exam does Phani Motamarri accept for admissions?'])
def test_public_admission_tests_do_not_request_private_timetable(query):
    assert not course_exam_intent(query)
    assert 'EXAM_ON' not in requested_relations(query)
    assert DocumentCatalog().denied_scopes(query,['public'])==[]

@pytest.mark.parametrize('query',['When is DS252 final exam?','What is the semester exam after admissions?','Show professor-only exam details for GATE candidates','Show the attached pdf for admissions'])
def test_protected_exams_remain_protected(query):
    assert 'faculty' in DocumentCatalog().denied_scopes(query,['public'])

def test_programme_panels_use_content_not_positional_labels():
    body='''<div class="col-md-9"><h1>M.Tech.</h1><ul class="resp-tabs-list"><li>Other</li><li>Broken</li><li>Computational &amp; Data Sciences</li></ul><div class="resp-tabs-container"><div class="fruitful_tab">Discipline: Other Department/Centre/Unit: Other <p>Wrong eligibility</p></div><p>Discipline: Broken unwrapped panel</p><div class="fruitful_tab">Discipline: Computational and Data Sciences Department/Centre/Unit: CDS <p>Correct eligibility</p></div></div></div>'''
    normalized=normalize_html(body,'https://example.test','programmes')
    assert 'Correct eligibility' in normalized and 'Wrong eligibility' not in normalized and 'unwrapped panel' not in normalized

def test_fee_spans_preserve_programme_category():
    table=BeautifulSoup('''<table><tr><th colspan=3>Fee structure for Course Programme (INR per annum)</th></tr><tr><td></td><td>GN/OBC/EWS Category</td><td>SC/ST/PwD Category</td></tr><tr><td>Particulars</td><td>M.Tech./M.Des.</td><td>M.Tech./M.Des.</td></tr><tr><td>Tuition</td><td>15000</td><td>Fully waived</td></tr><tr><td>Refundable deposit</td><td colspan=2>7500</td></tr></table>''','html.parser').table
    assert table_grid(table)[-1]==['Refundable deposit','7500','7500']
    sections=parse_html('https://test', '<main>'+fees_table(table)+'</main>').sections
    assert len(sections)==2
    assert '15000' in sections[0].text and 'Fully waived' not in sections[0].text
    assert 'Fully waived' in sections[1].text and '15000' not in sections[1].text
    assert all('7500' in s.text for s in sections)

def test_unreviewed_source_container_fails_closed():
    with pytest.raises(ValueError,match='container'):normalize_html('<article>New site layout</article>','https://test','general')

def test_missing_cds_panel_fails_instead_of_ingesting_only_common_rules():
    body='''<div class="col-md-9"><h1>Admissions</h1><ul class="resp-tabs-list"><li>Computational &amp; Data Sciences</li></ul><div class="resp-tabs-container"><div class="fruitful_tab">Department/Centre/Unit: Other department</div></div></div>'''
    with pytest.raises(ValueError,match='CDS programme panel missing'):
        normalize_html(body,'https://example.test','programmes')

def test_source_and_checksum_bounds(tmp_path):
    from askiisc_retrieval.admissions import cached_pages
    profile={'sources':[{'url':'https://example.test','mode':'general'}],'limits':{'sources':0}}
    with pytest.raises(ValueError,match='Source boundary'):cached_pages(profile,tmp_path)
    profile['limits']['sources']=1
    (tmp_path/'download.json').write_text(json.dumps([{'url':'https://example.test','file':'source.html','sha256':'incorrect'}]))
    (tmp_path/'raw').mkdir();(tmp_path/'raw/source.html').write_text('Changed bytes')
    with pytest.raises(ValueError,match='checksum'):cached_pages(profile,tmp_path)

def test_rebuild_retains_cached_crawl_and_admissions(tmp_path,monkeypatch):
    from askiisc_retrieval import ingest_stub,config,fetch
    import askiisc_retrieval.admissions as admissions
    raw=tmp_path/'raw';raw.mkdir();(tmp_path/'admissions').mkdir()
    (tmp_path/'admissions/sources.json').write_text('{}')
    seed='https://example.test/seed';crawl='https://example.test/crawl';updated='https://example.test/admissions'
    fees='https://example.test/fees'
    for url in (seed,crawl,updated,fees):(raw/fetch.raw_name(url)).write_text('<main>Original content</main>')
    (raw/'crawl.json').write_text(json.dumps([{'url':crawl,'doc_type':'research'},{'url':updated,'doc_type':'admissions'}]))
    monkeypatch.setattr(config,'RAW_DIR',raw)
    monkeypatch.setattr(fetch,'load_sources',lambda:{'pages':[
        {'url':seed,'doc_type':'page','scope':'public'},
        {'url':fees,'doc_type':'admissions','scope':'public','department':'IISc','normalize':'fees'},
    ],'department':'CDS','crawls':[{'department':'CDS','seeds':['https://example.test/']}]})
    monkeypatch.setattr(admissions,'normalize_html',lambda *args: 'Current fees')
    monkeypatch.setattr(admissions,'cached_pages',lambda *args:([
        ingest_stub.PageInput(updated,'Updated content','admissions','public','IISc'),
        ingest_stub.PageInput(fees,'Older fees','admissions','public','IISc'),
    ],{}))
    monkeypatch.setattr(DocumentCatalog,'page_inputs',lambda self:[])
    pages=ingest_stub.load_page_inputs(include_fixtures=False)
    assert {p.url for p in pages}=={seed,crawl,updated,fees}
    assert next(p for p in pages if p.url==updated).html=='Updated content'
    assert [p.html for p in pages if p.url==fees]==['Current fees']
