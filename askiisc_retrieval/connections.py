"""Bounded, ordered relation joins with per-hop provenance, scope and snapshot checks.

Plans describe relation types, never answers. Each completed path resolves all
supporting chunks before it can contribute any evidence to generation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from .contract import Passage

from .query_analysis import course_exam_intent


@dataclass(frozen=True)
class ConnectionPlan:
    starts: tuple[str,...]
    relations: tuple[str,...]
    term: str
    answer_relation: str


def plan_for(info, graph):
    if not course_exam_intent(info.query):return None
    venue = bool(re.search(r'\b(?:where|venue|room|location)\b', info.query, re.I))
    tail = ('SCHEDULED_AT','EXAM_ON') + (('HELD_IN',) if venue else ())
    answer_relation = 'HELD_IN' if venue else 'EXAM_ON'
    if info.course_codes:
        return ConnectionPlan(tuple('Course:'+c for c in info.course_codes),tail,info.target_term,answer_relation)
    if graph is not None:
        try:
            people=graph.find_people(info.query)
        except Exception:
            if re.search(r'taught|teach|instructor|professor', info.query, re.I):
                return ConnectionPlan(('UnresolvedPerson',),('^TAUGHT_BY','SCHEDULED_AT','EXAM_ON'),info.target_term,'EXAM_ON')
            return None
        if people:
            return ConnectionPlan(tuple(people),('^TAUGHT_BY',)+tail,info.target_term,answer_relation)
    return None


def bounded_paths(edges, starts, relations, version_ids, scopes, term, *, max_states=1024, max_paths=128):
    """Reusable exact-key joins. A ^relation traverses an edge in reverse.

    Every hop must be current, authorized and citable. No cross-term join,
    hidden bridge, fuzzy match or partial path can establish an answer.
    """
    if not 1<=len(relations)<=4:raise ValueError('Connection plans allow one to four hops')
    eligible=[e for e in edges if e.version_id in version_ids and e.scope in scopes and e.chunk_ids
              and (e.academic_term is None or e.academic_term==term)]
    states=[(start,start,()) for start in starts]
    for relation in relations:
        reverse=relation.startswith('^');name=relation.lstrip('^')
        index={}
        for e in eligible:
            if e.rel==name:index.setdefault(e.dst if reverse else e.src,[]).append(e)
        nxt=[]
        for start,node,path in states:
            for e in index.get(node,[]):
                other=e.src if reverse else e.dst
                if e in path:continue
                nxt.append((start,other,path+(e,)))
                if len(nxt)>max_states:raise ValueError('Connection search exceeded its bounded state budget')
        states=nxt
    if len(states)>max_paths:raise ValueError('Connection search exceeded its bounded path budget')
    return states


def resolve_connections(graph, store, snapshot, scopes, plan):
    edges=graph.active_edges() if hasattr(graph,'active_edges') else graph.edges
    paths=bounded_paths(edges,plan.starts,plan.relations,snapshot.active_version_ids,scopes,plan.term)
    ids=sorted({cid for _,_,path in paths for e in path for cid in e.chunk_ids})
    chunks={c.chunk_id:c for c in store.get(ids)}
    valid=[]
    for start,end,path in paths:
        if not all(cid in chunks and chunks[cid].version_id==e.version_id
                   and chunks[cid].access_scope==e.scope and chunks[cid].access_scope in scopes
                   and chunks[cid].embedding_model==snapshot.embedding_model
                   and chunks[cid].version_id in snapshot.active_version_ids
                   for e in path for cid in e.chunk_ids):continue
        courses=sorted({n for e in path for n in (e.src,e.dst) if n.startswith('Course:')})
        if len(courses)!=1:continue
        valid.append((start,courses[0],end,path))
    if set(plan.starts)-{start for start,_,_,_ in valid}:
        return [],[], 'missing_path'
    # A branching instructor query must not silently omit a course with a missing lookup.
    expected_courses = {start for start in plan.starts if start.startswith('Course:')}
    if plan.relations[0] == '^TAUGHT_BY':
        expected_courses |= {end for _,end,_ in bounded_paths(edges,plan.starts,plan.relations[:1],snapshot.active_version_ids,scopes,plan.term) if end.startswith('Course:')}
    if expected_courses - {course for _,course,_,_ in valid}:
        return [], [], 'partial_path'
    by_course={}
    for item in valid:by_course.setdefault(item[1],[]).append(item)
    # Conflicting final exam dates/times for one course must not be resolved by ranking.
    if any(len({p[2] for p in candidates})!=1 for candidates in by_course.values()):
        return [],[], 'ambiguous_path'
    passages=[];proofs=[];seen=set()
    for course,candidates in sorted(by_course.items()):
        start,_,end,path=min(candidates,key=lambda p:tuple(cid for e in p[3] for cid in e.chunk_ids))
        required=list(dict.fromkeys(cid for e in path for cid in e.chunk_ids))
        proof_id=hashlib.sha256('|'.join([course,end,*required]).encode()).hexdigest()[:16]
        steps=[{'from':e.src,'relation':e.rel,'to':e.dst,'chunk_ids':list(e.chunk_ids),
                'version_id':e.version_id,'access_scope':e.scope,'academic_term':e.academic_term} for e in path]
        text=' ; '.join(f'{e.src} -{e.rel}-> {e.dst}' for e in path)
        proof={'id':proof_id,'course':course,'endpoint':end,'steps':steps,'required_chunk_ids':required,'term':plan.term}
        proofs.append(proof)
        for cid in required:
            if cid in seen:continue
            seen.add(cid)
            support={'relation':plan.answer_relation,'proof_id':proof_id,'required_chunk_ids':required,
                     'question_seeds':list(plan.starts),'coverage':len(plan.starts),'focus':1,'hop':len(path)}
            passages.append(Passage(chunks[cid],1.0,'graph',text,graph_support=support))
    # Shared chunks need the union of every path's dependencies, so all queried courses survive packing.
    for p in passages:
        own=[proof for proof in proofs if p.chunk.chunk_id in proof['required_chunk_ids']]
        p.graph_support['required_chunk_ids']=sorted({cid for proof in own for cid in proof['required_chunk_ids']})
    return passages,proofs,'complete' if proofs else 'missing_path'
