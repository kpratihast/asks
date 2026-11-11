"""Behavioral regressions for complete relationship evidence and source-budget packing."""
from askiisc_serving.evidence_selection import evidence_bundle
from askiisc_serving.generator import Generator
from askiisc_serving.settings import Settings
from askiisc_serving.retrieval_client import hits_from_result
from askiisc_retrieval.query_analysis import requested_relations
from askiisc_retrieval.contract import ChunkPayload, Passage
from askiisc_retrieval.retriever import Retriever
from askiisc_retrieval.stores import Edge, JsonGraphStore
from askiisc_retrieval.snapshot import Snapshot
from askiisc_retrieval.query_analysis import analyse


def hit(i,section,text,**kw):
    return {'access_scope':'public','marker':f'S{i}','title':'Source','section':section,'url':'https://example.org/source',
            'version_id':'v','chunk_ids':[f'c{i}'],'academic_term':None,'academic_year':'not term-specific',
            'programme':'not specified','location':section,'freshness':{'status':'current','snapshot':'s'},
            'instructor':None,'text':text,'excerpt':text,'rerank_score':None,'via':'vector',**kw}


def test_late_graph_endpoint_and_short_anchor_survive_six_source_cap():
    anchor=hit(1,'DS 901','Course ID: DS 901; Instructor: Ada North',instructor='Ada North')
    noise=[hit(i,f'DS {910+i}','An unrelated course description.') for i in range(2,7)]
    lab=hit(7,'Discovery Lab','Faculty: Ada North. The lab studies distributed computing.',via='graph',
            graph_support=[{'relation':'HEADS','coverage':0,'focus':1,'question_seeds':['Course:DS901']}])
    g=Generator(Settings(max_model_len=2000,max_output_tokens=200))
    msgs,selected,budget=g.prepare('Which lab does the DS 901 instructor lead?',[anchor,*noise,lab])
    assert [h['chunk_ids'] for h in selected]==[['c1'],['c7']]
    assert budget['required_selected']==budget['required_sources']==2
    assert 'unrelated' not in msgs[1]['content']
    assert budget['prompt_tokens']<=budget['limit']


def test_a_direct_lab_question_does_not_promote_unrelated_graph_neighbours():
    direct=hit(1,'NLP Lab','Faculty: Ada North')
    unrelated=hit(2,'Astronomy Lab','Faculty: Bea West',via='graph',graph_support=[{'relation':'HEADS','coverage':0,'focus':0,'question_seeds':[]}])
    ordered,required=evidence_bundle('Who leads the NLP lab?',[direct,unrelated])
    assert ordered[0] is direct and not required


def test_unfitting_high_ranked_source_does_not_hide_later_short_evidence():
    a=hit(1,'First','A short fact.')
    b=hit(2,'Huge','irrelevant '*1000)
    c=hit(3,'Last','The answer is supported here.')
    _,selected,budget=Generator(Settings(max_evidence=2,max_model_len=1500,max_output_tokens=200)).prepare('Question?',[a,b,c],mode='vector')
    assert [h['title'] for h in selected]==['Source','Source']
    assert [h['chunk_ids'] for h in selected]==[['c1'],['c3']]
    assert budget['dropped']==1 and budget['prompt_tokens']<=budget['limit']


def test_structured_graph_support_survives_serving_mapping():
    support={'relation':'HEADS','coverage':1,'focus':1,'question_seeds':['Person:a north']}
    raw={'snapshot_id':'s','citations':[{'marker':'S1','source_url':'https://example.org','section_heading':'Discovery Lab','chunk_ids':['c'],'via':'graph'}],
         'passages':[{'marker':'S1','chunk_id':'c','text':'Faculty: Ada North','graph_path':'Person:a north -HEADS-> Lab:Discovery','graph_support':support}]}
    h=hits_from_result(raw)[0]
    assert h['graph_support']==[support] and h['graph_paths']


def test_lab_relation_has_priority_over_course_word_in_question():
    assert requested_relations('Which lab does the instructor of DS 901 lead?')==('HEADS',)
    assert requested_relations('Which course did the DS 901 instructor teach?')==('TAUGHT_BY',)


def chunk(cid,heading,text,code=None,term=None):
    return ChunkPayload(cid,'d','v','https://example.org','Source',heading,text,'public',
                        'schedule' if term else 'admissions','CDS',term,[code] if code else [],'test',0)


def test_graph_expansion_prioritizes_the_anchored_lab_over_other_courses():
    course=chunk('anchor','DS 901','Instructor: Ada North','DS901','2026-AUG')
    other=chunk('other','DS 902','Instructor: Ada North','DS902','2026-AUG')
    lab=chunk('lab','Discovery Lab','Faculty: Ada North')
    noise=chunk('wrong','Unrelated Lab','Faculty: Bea West')
    graph=JsonGraphStore([
        Edge('Course:DS901','TAUGHT_BY','Person:a north',('anchor',),'v','public','2026-AUG'),
        Edge('Course:DS902','TAUGHT_BY','Person:a north',('other',),'v','public','2026-AUG'),
        Edge('Person:a north','HEADS','Lab:Discovery',('lab',),'v','public'),
        Edge('Person:b west','HEADS','Lab:Other',('wrong',),'v','public')])
    class Store:
        def get(self,ids):return [x for x in [course,other,lab,noise] if x.chunk_id in ids]
    r=Retriever(None,Store(),None,graph,graph_extra=1,today_term='2026-AUG')
    snap=Snapshot('s','test','rules-v2',['v'])
    extra=r._graph_passages([Passage(course,.9)],analyse('Which lab does the DS 901 instructor lead?','2026-AUG'),snap,['public'])
    assert len(extra)==1 and extra[0].chunk.chunk_id=='lab'
    assert extra[0].graph_support['relation']=='HEADS'
    assert extra[0].graph_support['focus']==1


def test_incomplete_relationship_bundle_abstains_without_calling_model():
    from test_serving import FakeRetrieval, FakeModel, make, ask
    anchor=hit(1,'DS 901','Course ID: DS 901; Instructor: Ada North',instructor='Ada North')
    lab=hit(2,'Discovery Lab','Faculty: Ada North',via='graph',graph_support=[
        {'relation':'HEADS','coverage':0,'focus':1,'question_seeds':['Course:DS901']}])
    model=FakeModel('This must not be generated [S1].')
    client,_=make(model,FakeRetrieval(hits=[anchor,lab]),max_evidence=1)
    with client:
        d=ask(client,q='Which lab does the DS 901 instructor lead?').json()
    assert d['status']=='abstained' and d['reason']=='insufficient_context_budget'
    assert not model.requests


def test_required_relationship_source_is_not_silently_truncated():
    lab=hit(1,'Discovery Lab','Faculty: Ada North. '+('long source text '*1000),via='graph',
            graph_support=[{'relation':'HEADS','coverage':1,'focus':1,'question_seeds':['Person:a north']}])
    _,selected,budget=Generator(Settings(max_model_len=1000,max_output_tokens=200)).prepare(
        'Which lab does Ada North lead?',[lab])
    assert not selected and not budget['truncated']
    assert budget['required_sources']==1 and budget['required_selected']==0


def test_category_specific_fact_excludes_nearby_unqualified_passages():
    fee = hit(1, "ST category", "Annual tuition is waived.")
    fee["graph_support"] = [{
        "policy_bundle_id": "fee",
        "required_chunk_ids": fee["chunk_ids"],
        "policy_qualifiers": {"category": "SC/ST/PwD"},
        "policy_facets": ["FEES"],
    }]
    others = hit(2, "Application fees", "Application fee is 400.")
    generator = Generator(Settings())
    _, selected, budget = generator.prepare("Does IISc charge ST students tuition?", [fee, others])
    assert [item["marker"] for item in selected] == ["S1"]
    assert budget["required_sources"] == 1
