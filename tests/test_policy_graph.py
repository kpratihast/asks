"""Policy joins use source identity, provenance and qualifications, not known benchmark answers."""
from dataclasses import replace
import pytest
from askiisc_retrieval.contract import ChunkPayload
from askiisc_retrieval.policy_graph import build_policy_edges,retrieve_policy,programmes
from askiisc_retrieval.stores import JsonGraphStore
from askiisc_retrieval.snapshot import Snapshot

def c(cid,text,heading='M.Tech.',department='XYZ',scope='public'):
    return ChunkPayload(cid,cid,'v-'+cid,'https://example.test/'+cid,'Admissions',heading,text,scope,'admissions',department,None,[],'test',0)

def fixture():
    return [c('offering','The programme is designed to be completed in 19 months.'),
            c('grant','Programme: M.Tech.; Scholarship/fellowship per month: Rs. 9876','M.Tech.','IISc'),
            c('research','Scholarship/fellowship per month: Rs. 5432','M.Tech. (Research)','IISc')]

def run(cs,query='For XYZ M.Tech., how long is the programme and what is the monthly scholarship?',edges=None,scopes=('public',),snap=None,**kw):
    class Store:
        def get(self,ids):return [x for x in cs if x.chunk_id in ids]
    graph=JsonGraphStore(build_policy_edges(cs) if edges is None else edges)
    snap=snap or Snapshot('test','test','test',frozenset(x.version_id for x in cs))
    return retrieve_policy(query,graph,Store(),snap,list(scopes),**kw)

def test_unseen_department_amount_duration_and_real_path():
    ps,proofs,t=run(fixture())
    assert t['status']=='complete'
    assert {p.chunk.chunk_id for p in ps}=={'offering','grant'}
    assert any([s['relation'] for s in p['steps']]==['IS_A','HAS_FUNDING_AMOUNT'] for p in proofs)
    assert all(p.graph_support['required_chunk_ids'] for p in ps)

def test_programme_identity_not_substring():
    assert programmes('M.Tech. (Research)')==['MTechResearch']
    assert programmes('M.Tech. and M.Tech. (Res)')==['MTechResearch','MTech']
    ps,_,t=run(fixture(),'What monthly scholarship is listed for M.Tech. (Research)?')
    assert t['status']=='complete' and {p.chunk.chunk_id for p in ps}=={'research'}

def test_removing_programme_bridge_loses_policy_evidence():
    cs=fixture();edges=[e for e in build_policy_edges(cs) if e.rel!='IS_A']
    ps,_,t=run(cs,edges=edges)
    assert 'FUNDING_AMOUNT' in t['missing'] and all(p.chunk.chunk_id!='grant' for p in ps)

@pytest.mark.parametrize('fault',['scope','inactive','embedding','missing','version'])
def test_every_hop_validates_provenance(fault):
    cs=fixture();edges=build_policy_edges(cs);snap=None
    if fault=='scope':
        cs[0]=replace(cs[0],access_scope='faculty');edges=build_policy_edges(cs)
    elif fault=='inactive':snap=Snapshot('s','test','test',frozenset({'v-grant','v-research'}))
    elif fault=='embedding':cs[0]=replace(cs[0],embedding_model='different')
    elif fault=='missing':cs=cs[1:]
    elif fault=='version':cs[0]=replace(cs[0],version_id='changed')
    ps,proofs,t=run(cs,edges=edges,snap=snap)
    assert not any(p.chunk.chunk_id=='grant' for p in ps)
    assert not proofs

@pytest.mark.parametrize('category', ['SC/ST/PwD', 'SC', 'ST'])
def test_fees_respect_category_and_no_other_department_reverse_traversal(category):
    cs=fixture()+[c('reserved','Tuition Fee (Annual): 0 TOTAL: 2345','Fees — M.Tech. — SC/ST/PwD','IISc'),c('general','Tuition Fee (Annual): 7890 TOTAL: 9000','Fees — M.Tech. — GN/OBC/EWS','IISc'),c('other','The programme is completed in 88 months.','M.Tech.','OTHER')]
    ps,_,_=run(cs,f'For XYZ M.Tech., how long is the programme and what is the {category} annual fee?')
    ids={p.chunk.chunk_id for p in ps}
    assert 'reserved' in ids and not ids&{'general','other'}

def test_uncategorized_fees_keep_multiple_published_categories():
    cs = [
        c('reserved', 'Tuition Fee (Annual): 0 TOTAL: 2345', 'Fees — M.Tech. — SC/ST/PwD', 'IISc'),
        c('general', 'Tuition Fee (Annual): 7890 TOTAL: 9000', 'Fees — M.Tech. — GN/OBC/EWS', 'IISc'),
        c('research', 'Tuition Fee (Annual): 6789 TOTAL: 8899', 'Fees — M.Tech. (Res) — GN/OBC/EWS', 'IISc'),
    ]
    ps, proofs, trace = run(cs, 'What is the M.Tech. tuition fee?')
    assert trace['status'] == 'complete'
    assert {p.chunk.chunk_id for p in ps} == {'general', 'reserved'}
    assert {proof['steps'][-1]['qualifiers']['category'] for proof in proofs} == {'SC/ST/PwD', 'GN/OBC/EWS'}


def test_route_and_cycle_qualifications():
    cs=fixture()+[replace(c('route','Scholarship/fellowship per month: Rs. 9999','M.Tech.','IISc'),title='CFTI admissions'),c('old','Events: Interview dates; Date: November 3, 2030','January 2031','IISc'),c('new','Events: Interview dates; Date: November 8, 2031','January 2032','IISc')]
    ps,_,_=run(cs,'What are the interview dates for January 2032?')
    assert {p.chunk.chunk_id for p in ps}=={'new'}
    ps,_,_=run(cs)
    assert all(p.chunk.chunk_id!='route' for p in ps)

def test_conflicting_origins_both_selected_and_qualified():
    cs=[c('dept','There is no oral interview.','M.Tech.'),c('central','Selection Procedure: written and interview.','M.Tech.','IISc')]
    ps,proofs,t=run(cs,'Compare central and XYZ department selection for M.Tech. about an oral interview.')
    assert t['status']=='complete' and {p.chunk.chunk_id for p in ps}=={'dept','central'}
    assert {s['qualifiers']['origin'] for p in proofs for s in p['steps']}=={'department','institute'}

def test_bounded_and_json_roundtrip(tmp_path):
    edges=build_policy_edges(fixture());g=JsonGraphStore(edges);g.save(tmp_path/'graph.json')
    assert JsonGraphStore.load(tmp_path/'graph.json').edges==edges
    with pytest.raises(ValueError,match='state budget'):run(fixture(),max_states=1)

def test_policy_dependencies_survive_prompt_packing():
    from askiisc_retrieval.contract import RetrievalResult
    from askiisc_retrieval.citations import build_citations,result_to_dict
    from askiisc_serving.retrieval_client import hits_from_result
    from askiisc_serving.generator import Generator
    from askiisc_serving.settings import Settings
    ps,proofs,_=run(fixture());r=RetrievalResult('question',ps,'test',citations=build_citations(ps),graph_proofs=proofs)
    _,selected,budget=Generator(Settings()).prepare('What is the XYZ M.Tech. duration and monthly scholarship?',hits_from_result(result_to_dict(r)))
    assert budget['required_sources']==budget['required_selected']==2
    assert len(selected)==2

def test_no_policy_facet_does_not_change_retrieval():
    assert run(fixture(),'Who teaches CS777?')==([],[],{})

def test_graph_free_multi_query_preserves_access_scopes():
    from askiisc_retrieval.multi_query import retrieve_multi
    from askiisc_retrieval.contract import Passage,RetrievalResult
    class Retriever:
        calls=[]
        def retrieve(self,q,scopes,graph,rerank):
            self.calls.append((q,scopes,graph,rerank))
            return RetrievalResult(q,[Passage(x,.8) for x in fixture()],'test')
    retriever=Retriever();result,variants=retrieve_multi(retriever,'XYZ M.Tech. duration and monthly scholarship',['public'],max_passages=2)
    assert len(variants)<=5 and len(result.passages)==2
    assert all(scope==['public'] and not graph for _,scope,graph,_ in retriever.calls)
    assert result.citations
