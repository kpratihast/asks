"""General source-to-source joins, with unseen courses, clocks, versions and access levels."""
from dataclasses import replace
from pathlib import Path

import pytest

from askiisc_retrieval.contract import ChunkPayload
from askiisc_retrieval.connections import ConnectionPlan, bounded_paths, resolve_connections
from askiisc_retrieval.graph_rules import build_edges
from askiisc_retrieval.snapshot import StaticSnapshotProvider, Snapshot
from askiisc_retrieval.stores import HashEmbedder, JsonGraphStore, QdrantStore, Edge
from askiisc_retrieval.retriever import Retriever
from askiisc_retrieval.citations import result_to_dict
from askiisc_serving.retrieval_client import hits_from_result
from askiisc_serving.generator import Generator, verify_connected
from askiisc_serving.settings import Settings
from askiisc_retrieval.scheduling import slot,pattern_slot,exam_table_html,weekdays,time_range

TERM='2026-AUG'
SCOPES=['public','student','faculty']


def chunk(cid,text,*,codes=(),scope='student',term=TERM,version=None,title=None):
    return ChunkPayload(cid,cid,version or 'v-'+cid,'https://example.test/'+cid,title or cid,cid,text,
                        scope,'schedule','CDS',term,list(codes),'test/hash-bow',0)


def course(code='CS777',days='T/Th',time='10:00-11:30',*,term=TERM,scope='student',person='Ada North'):
    return chunk('course-'+code, f'Course ID: {code}; Instructor: {person}; Days of Week: {days}; Timings: {time}; Venue: X',codes=[code],scope=scope,term=term)


def exam(cid='exam-1',patterns='TTh 10 / TTh 10-11.30 / TTh 10.30-12',when='2026-12-01',*,term=TERM,scope='faculty'):
    return chunk(cid,f'Course time slots: {patterns}; Final examination date: {when}; Examination session: Forenoon; Examination time: 09:00-12:00',scope=scope,term=term)


def make(tmp_path,chunks,edges=None):
    emb=HashEmbedder();store=QdrantStore('connected',emb.dim,path=tmp_path/'q')
    store.upsert(chunks,emb.embed_documents([c.embedding_input() for c in chunks]))
    snap=Snapshot('snap-test',emb.model_id,'test',frozenset(c.version_id for c in chunks))
    graph=JsonGraphStore(build_edges(chunks) if edges is None else edges)
    r=Retriever(emb,store,StaticSnapshotProvider(snap),graph,today_term=TERM,rerank_min=None)
    return r,store,snap


@pytest.mark.parametrize('days',['T/Th','TTh','Tue, Thu','Tuesday and Thursday'])
def test_weekday_aliases_connect_exact_times(days):
    assert slot(days,'10:00-11:30').key(TERM)==pattern_slot('TTh 10-11.30').key(TERM)


@pytest.mark.parametrize('raw,expected',[('TTh 2-3.30',(840,930)),('MWF 12.30-2',(750,840)),
    ('TTh 11.30-1',(690,780)),('MWF 8',(480,540)),('TTh 3.30-5',(930,1020))])
def test_pdf_academic_clock_normalization(raw,expected):
    s=pattern_slot(raw);assert (s.start,s.end)==expected


def test_family_policy_is_explicit_and_does_not_invent_weekend_match():
    assert slot('M/W','14:00-15:00').key(TERM)!=pattern_slot('MWF 2').key(TERM)
    assert slot('Saturday','11:00-12:00').key(TERM)!=pattern_slot('TTh 11').key(TERM)
    assert weekdays('not announced') is None and time_range('TBA') is None


def test_exam_pdf_rows_are_citable_without_invented_course_ids():
    layout='''Forenoon: 09:00 AM – 12:00 PM Afternoon: 02:00 PM – 05:00 PM
13. TTh 10; TTh 10-11.30; TTh 10.30-12 Tuesday, December 01 Forenoon
16. TTh 8; TTh 8-9.30; TTh 8.30-10 Wednesday, December 02 Afternoon'''
    html=exam_table_html([layout],'Exam timetable','Aug-Dec 2026',TERM)
    assert html.count('<h3>Exam lookup row')==2
    assert '2026-12-01' in html and '14:00-17:00' in html
    assert 'CS777' not in html and 'DS252' not in html
    assert exam_table_html(['unrecognized layout'],'x','Aug-Dec 2026',TERM) is None
    with pytest.raises(ValueError,match='weekday'):
        exam_table_html([layout.replace('Tuesday','Monday')],'x','Aug-Dec 2026',TERM)


def test_new_course_two_source_join_and_complete_prompt(tmp_path):
    r,store,snap=make(tmp_path,[course(),exam()])
    result=r.retrieve('When is CS777 final exam?',SCOPES)
    assert result.connection_status=='complete' and len(result.graph_proofs)==1
    assert {p.chunk.chunk_id for p in result.passages if p.graph_support.get('proof_id')}=={'course-CS777','exam-1'}
    assert [s['relation'] for s in result.graph_proofs[0]['steps']]==['SCHEDULED_AT','EXAM_ON']
    hits=hits_from_result(result_to_dict(result))
    messages,selected,budget=Generator(Settings()).prepare('When is CS777 final exam?',hits)
    assert len(selected)==2 and budget['required_sources']==budget['required_selected']==2
    assert '2026-12-01' in messages[1]['content'] and 'Verified source connection' in messages[1]['content']
    assert not verify_connected('The exam is on December 1 [S1].',selected)['ok']
    assert verify_connected('The exam is on December 1 [S1][S2].',selected)['ok']
    store.close()


@pytest.mark.parametrize('scopes',[['public'],['public','student'],['public','faculty']])
def test_every_hop_requires_its_own_scope(tmp_path,scopes):
    r,store,_=make(tmp_path,[course(),exam()])
    result=r.retrieve('When is CS777 final exam?',scopes)
    assert result.connection_status=='missing_path' and result.evidence_sufficient is False
    assert not result.graph_proofs
    assert all(p.chunk.access_scope in scopes for p in result.passages)
    store.close()


@pytest.mark.parametrize('variant',['other_term','different_time','no_course','inactive_exam','inactive_course','conflict','bad_provenance'])
def test_missing_ambiguous_or_stale_links_do_not_become_answers(tmp_path,variant):
    chunks=[course(),exam()]
    if variant=='other_term':chunks[1]=exam(term='2027-JAN')
    if variant=='different_time':chunks[1]=exam(patterns='TTh 11-12.30')
    if variant=='no_course':chunks=chunks[1:]
    if variant=='conflict':chunks.append(exam('exam-2',when='2026-12-02'))
    r,store,snap=make(tmp_path,chunks)
    if variant.startswith('inactive_'):
        cid='exam-1' if variant=='inactive_exam' else 'course-CS777'
        snap=replace(snap,active_version_ids=snap.active_version_ids-{'v-'+cid});r.snapshots=StaticSnapshotProvider(snap)
    if variant=='bad_provenance':
        r.graph=JsonGraphStore([replace(e,scope='public') if e.rel=='EXAM_ON' else e for e in r.graph.edges])
    result=r.retrieve('When is CS777 final exam?',SCOPES)
    assert result.connection_status==('ambiguous_path' if variant=='conflict' else 'missing_path')
    assert result.evidence_sufficient is False and not result.graph_proofs
    store.close()


def test_instructor_to_course_to_slot_to_exam_is_three_hops(tmp_path):
    r,store,_=make(tmp_path,[course(),exam()])
    result=r.retrieve('When is the final examination of the course taught by Ada North?',SCOPES)
    assert result.connection_status=='complete'
    assert len(result.graph_proofs[0]['steps'])==3
    store.close()


def test_multi_course_and_explicit_term_do_not_cross_join(tmp_path):
    chunks=[course(),course('EE888',time='14:00-15:30'),exam(),exam('exam-2',patterns='TTh 2-3.30',when='2026-11-27')]
    r,store,_=make(tmp_path,chunks)
    result=r.retrieve('When are CS777 and EE888 final examinations?',SCOPES)
    assert len(result.graph_proofs)==2
    assert r.retrieve('When is CS777 final exam in Jan 2027?',SCOPES).evidence_sufficient is False
    store.close()


def test_vector_does_not_receive_graph_proof_but_keeps_same_sources(tmp_path):
    r,store,_=make(tmp_path,[course(),exam()])
    result=r.retrieve('When is CS777 final exam?',SCOPES,use_graph=False)
    assert result.connection_status is None and not result.graph_proofs
    assert not any(p.graph_support.get('proof_id') for p in result.passages)
    assert any(p.chunk.chunk_id=='course-CS777' for p in result.passages)
    assert any(p.chunk.chunk_id=='exam-1' for p in result.passages)
    store.close()


def test_graph_failure_abstains_for_connected_question_and_source_cap_cannot_drop_hop(tmp_path):
    r,store,_=make(tmp_path,[course(),exam()])
    result=r.retrieve('When is CS777 final exam?',SCOPES)
    _,_,budget=Generator(Settings(max_evidence=1)).prepare('When is CS777 final exam?',hits_from_result(result_to_dict(result)))
    assert budget['required_sources']>budget['required_selected']
    r.graph.fail=True
    result=r.retrieve('When is CS777 final exam?',SCOPES)
    assert result.degraded and result.evidence_sufficient is False and result.connection_status=='unavailable'
    store.close()


def test_path_engine_reuses_ordered_joins_for_other_relation_types():
    edges=[Edge('Course:A','REQUIRES','Course:B',('c1',),'v','public'),
           Edge('Course:B','REQUIRES','Course:C',('c2',),'v','public'),
           Edge('Course:C','TAUGHT_BY','Person:Q',('c3',),'v','faculty')]
    assert len(bounded_paths(edges,['Course:A'],['REQUIRES','REQUIRES','TAUGHT_BY'],{'v'},['public','faculty'],TERM))==1
    assert not bounded_paths(edges,['Person:Q'],['^TAUGHT_BY','^REQUIRES'],{'v'},['public'],TERM)
    with pytest.raises(ValueError,match='budget'):
        bounded_paths(edges,['Course:A'],['REQUIRES'],{'v'},['public'],TERM,max_states=0)


def test_classroom_is_not_an_exam_venue(tmp_path):
    r,store,_=make(tmp_path,[course(),exam()])
    result=r.retrieve('Where is CS777 final exam?',SCOPES)
    assert result.evidence_sufficient is False and not result.graph_proofs
    store.close()


def test_instructor_question_does_not_omit_an_unmatched_second_course(tmp_path):
    r,store,_=make(tmp_path,[course(),course('EE888',days='Saturday',time='11:00-12:00'),exam()])
    result=r.retrieve('When are the final exams for the courses Ada North teaches?',SCOPES)
    assert result.connection_status=='partial_path' and result.evidence_sufficient is False
    store.close()


def test_graph_failure_preserves_vector_fallback_for_direct_pdf_question(tmp_path):
    r,store,_=make(tmp_path,[course(),exam()]);r.graph.fail=True
    result=r.retrieve('What are the forenoon examination hours?',SCOPES)
    assert result.degraded and result.connection_status is None and result.evidence_sufficient is None
    store.close()
