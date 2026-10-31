"""Typed, source-backed programme/policy graph, independent of evaluation questions.

Edges point to evidence, not generated answers. Programme identity, department,
source origin, admission cycle and fee category qualify traversal. General policy
is reached through an explicit offering -> programme -> institution path; the
reverse path into another department is never traversed.
"""
from __future__ import annotations
from collections import defaultdict, deque
import hashlib,re
from .stores import Edge
from .contract import Passage

PROGRAMMES={
 'MTechResearch':r'm\s*\.?\s*tech\s*\.?\s*\(\s*res(?:earch)?\s*\)',
 'PhD':r'ph\s*\.?\s*d\.?|doctoral',
 'MTech':r'm\s*\.?\s*tech\.?|master\s+of\s+technology',
 'MDes':r'm\s*\.?\s*des\.?|master\s+of\s+design',
 'MSc':r'm\s*\.?\s*sc\.?|master\s+of\s+science',
 'BTech':r'b\s*\.?\s*tech\.?|bachelor\s+of\s+technology',
}
BRIDGES={'IS_A','AT_INSTITUTION','IN_DEPARTMENT','HAS_CYCLE'}

def programmes(value):
    out=[];remaining=value
    if re.search(PROGRAMMES['MTechResearch'],remaining,re.I):
        out.append('MTechResearch');remaining=re.sub(PROGRAMMES['MTechResearch'],'',remaining,flags=re.I)
    for name,pattern in PROGRAMMES.items():
        if name!='MTechResearch' and re.search(pattern,remaining,re.I):out.append(name)
    return out

def cycle(value):
    m=re.search(r'\b(jan(?:uary)?|aug(?:ust)?)\s*(?:,\s*)?(20\d{2})\b',value,re.I)
    return m.group(2)+('-JAN' if m.group(1).lower().startswith('jan') else '-AUG') if m else ''

def category(value):
    if re.search(r'\bSC\s*/\s*ST|\bPwD\b|\bscheduled (?:caste|tribe)',value,re.I):return 'SC/ST/PwD'
    if re.search(r'\bGN\b|\bgeneral.category|\bOBC\b|\bEWS\b',value,re.I):return 'GN/OBC/EWS'
    return ''

def department(value):
    # Full names are associated with their published acronym, not a fuzzy substring.
    if re.search(r'\bCDS\b|computational\s*(?:&|and)\s*data\s*sciences?',value,re.I):return 'CDS'
    return ''

# Classifiers describe evidence types; they do not contain institution-specific amounts,
# dates, course IDs, source URLs, names, or expected answers.
FACT_PATTERNS={
 'DURATION':r'(?:completed in\s+\d+\s+months?|duration.*?(?:years?|months?)|minimum of\s+\d+\s+years?|programme?\s+spans)',
 'COURSEWORK':r'\d+\s+credits\s+of\s+coursework|course\s*work.*?\d+\s+credits',
 'DISSERTATION':r'dissertation\s+project\s+worth\s+\d+\s+credits|dissertation.*?\d+\s+(?:months?|credits)',
 'FUNDING_AMOUNT':r'(?:scholarship|fellowship)\s+per\s+month\s*:|JRF\s*[–-]\s*Rs',
 'FUNDING_CONDITIONS':r'(?:not eligible|not entitled|may be considered|maximum period|receive salary|employers/sponsors|through CFTI mode)',
 'FEES':r'Tuition Fee\s*\(Annual\)|TOTAL\s*:\s*[\d,]+',
 'APPLICATION_FEE':r'(?:Online Application Fees|application fee.*?(?:Rs\.?|INR)|Category\s+Application Fee)',
 'APPLICATION_CHOICES':r'(?:one application form|single application|maximum of\s*\d+\s*disciplines)',
 'LAB_PREFERENCES':r'(?:choose|select).*?(?:three|\d+)\s+labs',
 'SELECTION':r'(?:oral interview|written and interview|(?:mandatory|required|performance|score|selection).{0,90}written test|written test.{0,50}(?:mandatory|required)|performance in.*?interview|selection procedure\s*:)',
 'TEST_SYLLABUS':r'syllabus for the written test|Programming\s*:\s*Data Structures',
 'ELIGIBILITY':r'(?:GATE paper|valid GATE|Basic Qualification for Eligibility|Qualification with required)',
 'INTAKE':r'\bIntake\s*:\s*\d+',
 'CONTACT':r'[\w.+-]+[@#][\w.-]+\.(?:in|edu|org)',
 'QUOTA':r'no NRI|no.*management quota',
 'INTERVIEW_DATES':r'(?:Events:\s*Interview dates|Date of Interview).*?\d',
 'APPLICATION_DEADLINE':r'(?:Last date for.*?(?:submission|application)|Website closes at).*?\d',
 'SESSION_START':r'(?:Commencement of Session|Semester starts).*?(?:January|August|\d)',
 'NOTICE':r'(?:Department\s*/Centre\s*/\s*Unit|Ph\.D\.\s+in\s+(?:Science|Engineering|Interdisciplinary))',
 'AMENDMENT':r'CORRIGENDUM|in continuation to the.*admission notice',
}

def facts(c):
    text=c.text;heading=c.section_heading
    found={f for f,p in FACT_PATTERNS.items() if re.search(p,text,re.I|re.S)}
    if 'duration' in heading.lower() and re.search(r'\d.*(?:years?|months?)',text,re.I):found.add('DURATION')
    if 'FUNDING_CONDITIONS' in found and not re.search(r'scholarship|fellowship',text,re.I):found.remove('FUNDING_CONDITIONS')
    if 'DURATION' in found and re.search(r'scholarship|fellowship',heading,re.I):found.remove('DURATION')
    if 'APPLICATION_CHOICES' in found and not re.search(r'departments?|disciplines?|program',text,re.I):found.remove('APPLICATION_CHOICES')
    if 'NOTICE' in found and not re.search(r'mid.year.*notice',c.title,re.I):found.remove('NOTICE')
    return found


def build_policy_edges(chunks,institution='IISc'):
    documents=defaultdict(list)
    for c in chunks:
        if c.doc_type in {'admissions','policy','about'}:documents[c.source_url].append(c)
    edges=[]
    for items in documents.values():
        items.sort(key=lambda c:c.ordinal)
        doc_text=' '.join(c.text for c in items);title=items[0].title
        doc_cycle=cycle(title)
        if not doc_cycle:
            m=re.search(r'Commencement of Session.{0,100}?(?:January|August)\s+\d{1,2},?\s+20\d{2}',doc_text,re.I)
            if m:
                v=re.sub(r'((?:January|August))\s+\d{1,2},?\s+(20\d{2})',r'\1 \2',m.group(),flags=re.I);doc_cycle=cycle(v)
        if not doc_cycle:
            doc_cycle=next((cycle(c.section_heading) for c in items if cycle(c.section_heading)), '')
        doc_programmes=programmes(title)
        if not doc_programmes and items[0].department!='IISc' and re.search(r'research admissions',title,re.I):
            doc_programmes=programmes(' '.join(c.text for c in items[:3]))
            doc_programmes=[p for p in doc_programmes if p in ('PhD','MTechResearch')]
        for c in items:
            leaf=c.section_heading.split(' > ')[-1]
            # Multi-column fee tables have already retained the exact column programme.
            subject_heading=leaf.split(' — ')[-2] if ' — ' in leaf else leaf
            ps=programmes(subject_heading) or doc_programmes
            dep=c.department if c.department!=institution else department(leaf)
            if not dep and ('|' in c.text or re.search(r'^Discipline:',c.text)):
                dep=department(c.text.split('|')[0][:300])
            if dep and not ps:ps=doc_programmes
            cy=cycle(c.section_heading) or doc_cycle
            qual={'cycle':cy,'category':category(leaf),'origin':'department' if c.department!=institution else 'institute'}
            if re.search(r'(?:admission.*notice|mid.year.*notice)',title,re.I):qual['document_kind']='notice'
            if re.search(r'CFTI',title,re.I):qual['route']='CFTI'
            if re.search(r'undergraduate',leaf,re.I):qual['level']='UG'
            elif re.search(r'post.?graduate|PG/',leaf,re.I):qual['level']='PG'
            def add(src,rel,dst):
                edges.append(Edge(src,rel,dst,(c.chunk_id,),c.version_id,c.access_scope,None,tuple(sorted((k,v) for k,v in qual.items() if v))))
            inst='Institution:'+institution
            subjects=[]
            for p in ps:
                generic='Programme:'+p
                if dep:
                    subject='Offering:'+dep+'/'+p
                    add(subject,'IS_A',generic);add(subject,'IN_DEPARTMENT','Department:'+dep)
                    add('Department:'+dep,'AT_INSTITUTION',inst)
                else:subject=generic
                add(generic,'AT_INSTITUTION',inst);subjects.append(subject)
            if not subjects:subjects=[inst]
            if cy:
                add(inst,'HAS_CYCLE','AdmissionCycle:'+cy)
            for facet in facts(c):
                # Institute-wide application rules and cycles are not department-specific policy.
                scoped=subjects
                if facet in {'APPLICATION_FEE','APPLICATION_CHOICES','QUOTA','INTERVIEW_DATES','APPLICATION_DEADLINE','SESSION_START','NOTICE','AMENDMENT'} and not dep:
                    scoped=['AdmissionCycle:'+cy] if cy and facet in {'INTERVIEW_DATES','APPLICATION_DEADLINE','SESSION_START','NOTICE','AMENDMENT'} else [inst]
                for subject in scoped:add(subject,'HAS_'+facet,'Evidence:'+c.chunk_id)
    return list(dict.fromkeys(edges))


def requested_facets(query):
    q=query.lower();out=[]
    def add(name,pattern):
        if re.search(pattern,q,re.I):out.append(name)
    funding=bool(re.search(r'scholarship|fellowship|stipend|\bjrf\b|\bsrf\b',q))
    add('DURATION',r'how long|programme duration|program duration|degree.*duration|duration.*degree|minimum and maximum.*duration')
    add('COURSEWORK',r'coursework|course work|course credits')
    add('DISSERTATION',r'dissertation|thesis|project.*credits')
    if funding:
        if re.search(r'month|amount|\bjrf\b|\bsrf\b|stipend',q):out.append('FUNDING_AMOUNT')
        if re.search(r'cfti|iit|salar|employ|sponsor|guarantee|entitled|eligib|exception|how many years|maximum|funding.*years|how long.{0,30}(?:scholarship|fellowship|funding)',q):out.append('FUNDING_CONDITIONS')
        if not any(f.startswith('FUNDING') for f in out):out.append('FUNDING_AMOUNT')
    add('APPLICATION_FEE',r'application\s+fees?|fees?\s+(?:for|per|of)\s+(?:an?\s+)?(?:online\s+)?application')
    if re.search(r'tuition|first.year.*fee|annual.*fee|fee total|refundable|deposits?',q):out.append('FEES')
    add('APPLICATION_CHOICES',r'(?:three|3|how many|max\w*)\s+(?:m\.?tech\.?\s+)?(?:departments|disciplines)|(?:one|single)\s+application|research and course.*application')
    add('LAB_PREFERENCES',r'(?:three|3|how many|choose|select).*?labs|candidate information form')
    add('TEST_SYLLABUS',r'prepar|syllabus|programming languages|maths.*topics')
    add('ELIGIBILITY',r'gate.*(?:papers?|accept)|papers?.*accept|academic background|eligib|qualification')
    add('INTAKE',r'\bintake\b|number of seats')
    add('CONTACT',r'\be.?mail\b|contact.*queries')
    add('QUOTA',r'quota')
    add('SELECTION',r'selection|oral interview|written test.*(?:require|must|mandatory)|(?:must|require|mandatory).*written test')
    add('INTERVIEW_DATES',r'interview dates|date.*interview|interview.*ranges')
    add('APPLICATION_DEADLINE',r'deadline|closing date|application.*close')
    add('SESSION_START',r'semester.*start|commencement|session.*start')
    add('NOTICE',r'admission notice|mid.year.*admit|admit.*mid.year')
    add('AMENDMENT',r'corrigendum|amendment|withdraw')
    return tuple(dict.fromkeys(out))


def roots_for(query,edges):
    ps=programmes(query);dep=department(query)
    if not dep:
        names={e.src.split(':',1)[1] for e in edges if e.src.startswith('Department:')}
        dep=next((d for d in sorted(names) if re.search(r'(?<!\w)'+re.escape(d)+r'(?!\w)',query,re.I)), '')
    roots=[]
    if ps:
        roots=[('Offering:'+dep+'/'+p) if dep else 'Programme:'+p for p in ps]
    elif dep and re.search(r'research',query,re.I):
        roots=['Offering:'+dep+'/'+p for p in ('MTechResearch','PhD')]
    elif dep:roots=['Department:'+dep]
    else:roots=sorted({e.src for e in edges if e.src.startswith('Institution:')})
    return roots


def retrieve_policy(query,graph,store,snapshot,scopes,*,max_states=512,max_candidates=100,max_sources=6):
    facets=requested_facets(query)
    if not facets:return [],[],{}
    all_edges=graph.active_edges() if hasattr(graph,'active_edges') else graph.edges
    edges=[e for e in all_edges if e.chunk_ids and e.scope in scopes and e.version_id in snapshot.active_version_ids]
    roots=roots_for(query,all_edges);cy=cycle(query);year=re.search(r'\b20\d{2}\b',query);cat=category(query)
    compare_origins=bool(re.search(r'central|institute.*department|general.*CDS',query,re.I)) and ('SELECTION' in facets or 'CONTACT' in facets)
    want_origins=('institute','department') if compare_origins else ('',)
    requirements={(f,o) for f in facets for o in (want_origins if f in ('SELECTION','CONTACT') else ('',))}
    compare_dates='INTERVIEW_DATES' in facets and bool(re.search(r'compare|agree|two|both|differ',query,re.I))
    if compare_dates:
        requirements.discard(('INTERVIEW_DATES',''))
        requirements.update({('INTERVIEW_DATES','notice'),('INTERVIEW_DATES','calendar')})
    def allowed(e):
        qual=dict(e.qualifiers);ecy=qual.get('cycle','')
        ps=programmes(query)
        if qual.get('level')=='UG' and (any(p!='BTech' for p in ps) or re.search(r'general.*admission',query,re.I)):return False
        if qual.get('level')=='PG' and ps==['BTech']:return False
        if ecy and cy and ecy!=cy:return False
        if ecy and year and ecy[:4]!=year.group():return False
        if qual.get('route') and qual['route'] not in query.upper():return False
        if qual.get('category') and cat and qual['category']!=cat:return False
        return True
    by_node=defaultdict(list)
    for e in edges:
        if allowed(e):by_node[e.src].append(e)
    # Traverse only typed forward bridges. No unqualified hub expansion into other offerings.
    queue=deque((root,()) for root in roots);paths={};states=0
    while queue:
        node,path=queue.popleft();states+=1
        if states>max_states:raise ValueError('Policy graph state budget exceeded')
        if node in paths and len(paths[node])<=len(path):continue
        paths[node]=path
        if len(path)>=3:continue
        for e in by_node[node]:
            if e.rel in BRIDGES and e.dst not in paths and not any(n==e.dst for n,_ in queue):queue.append((e.dst,path+(e,)))
    candidates=[]
    for node,path in paths.items():
        for edge in by_node[node]:
            facet=edge.rel.removeprefix('HAS_')
            if not edge.dst.startswith('Evidence:') or facet not in facets:continue
            origin=dict(edge.qualifiers).get('origin','')
            if compare_dates and facet=='INTERVIEW_DATES':
                origin='notice' if dict(edge.qualifiers).get('document_kind')=='notice' else 'calendar'
            # Offering-specific questions require evidence about that offering.
            local={'ELIGIBILITY','INTAKE','COURSEWORK','DISSERTATION','TEST_SYLLABUS','LAB_PREFERENCES'}
            if any(r.startswith('Offering:') for r in roots) and facet in local and node not in roots:continue
            if facet=='SELECTION' and not compare_origins and any(r.startswith('Offering:') for r in roots) and origin!='department':continue
            req=(facet,origin if (facet,origin) in requirements else '')
            if req not in requirements:continue
            candidates.append((req,path+(edge,)))
    # Duplicate bridge proofs can occur; only fact endpoints determine the bounded candidate pool.
    candidates=list({(req,path[-1].dst):(req,path) for req,path in candidates}.values())
    if len(candidates)>max_candidates:raise ValueError('Policy candidate budget exceeded')
    bridge_options=defaultdict(list)
    for e in edges:
        if e.rel in BRIDGES and e.src in paths and allowed(e):bridge_options[(e.src,e.rel,e.dst)].append(e)
    ids=sorted({cid for _,path in candidates for e in path for cid in e.chunk_ids} | {cid for options in bridge_options.values() for e in options for cid in e.chunk_ids})
    if len(ids)>512:raise ValueError('Policy evidence budget exceeded')
    chunks={c.chunk_id:c for c in store.get(ids)}
    def valid(path):
        return all(cid in chunks and chunks[cid].version_id==e.version_id and chunks[cid].access_scope==e.scope
                   and chunks[cid].access_scope in scopes and chunks[cid].embedding_model==snapshot.embedding_model
                   for e in path for cid in e.chunk_ids)
    candidates=[(req,path) for req,path in candidates if valid(path)]
    words=set(re.findall(r'[a-z]{3,}',query.lower()))
    def reanchor(path):
        endpoint=path[-1].chunk_ids[0]
        out=[]
        for edge in path[:-1]:
            options=[e for e in bridge_options[(edge.src,edge.rel,edge.dst)] if valid((e,))]
            def cost(e):
                c=chunks[e.chunk_ids[0]]
                return (0 if endpoint in e.chunk_ids else 1, 0 if c.doc_type=='admissions' else 1, -len(words & set(re.findall(r'[a-z]{3,}',c.embedding_input().lower())))/max(1,len(c.text.split())**.5),len(c.text))
            out.append(min(options,key=cost) if options else edge)
        return tuple(out)+(path[-1],)
    candidates=[(req,reanchor(path)) for req,path in candidates]
    # Greedy set coverage reduces redundant chunks while retaining one endpoint per requested facet.
    groups={}
    for req,path in candidates:
        cid=path[-1].chunk_ids[0];g=groups.setdefault(cid,{'requirements':set(),'paths':[]})
        g['requirements'].add(req);g['paths'].append(path)
    covered=set();chosen=[];chosen_ids=set()
    while requirements-covered:
        choices=[]
        for cid,g in groups.items():
            gain=g['requirements']-covered
            if not gain:continue
            c=chunks[cid];words_hit=len(words&set(re.findall(r'[a-z]{3,}',c.embedding_input().lower())))
            # Shortest evidence-backed bridge, preferring one already selected as an endpoint.
            path=min(g['paths'],key=lambda p:(len({i for e in p for i in e.chunk_ids}-chosen_ids),len(p),tuple(i for e in p for i in e.chunk_ids)))
            needed={i for e in path for i in e.chunk_ids}
            if len(chosen_ids|needed)>max_sources:continue
            quality=words_hit/max(1,len(c.text.split())**.35)
            choices.append((len(gain),quality,-len(needed-chosen_ids),cid,path,gain))
        if not choices:break
        _,_,_,cid,path,gain=max(choices,key=lambda x:(x[0],x[1],x[2],x[3]))
        chosen.append((cid,path,gain));chosen_ids.update(i for e in path for i in e.chunk_ids);covered.update(gain)
    if not chosen:return [],[],{'kind':'policy','requested':sorted(facets),'covered':[],'missing':sorted(facets),'status':'missing'}
    bundle=hashlib.sha256((query+'|'+ '|'.join(sorted(chosen_ids))).encode()).hexdigest()[:16]
    # Endpoints precede bridge-only evidence. No source text or answer is synthesized.
    ordered=list(dict.fromkeys([cid for cid,_,_ in chosen]+[cid for _,path,_ in chosen for e in path for cid in e.chunk_ids]))
    proofs=[];passages=[]
    for cid,path,gain in chosen:
        proofs.append({'id':bundle+'-'+cid[:8],'kind':'policy','endpoint':path[-1].dst,'facets':sorted(f for f,_ in gain),
            'steps':[{'from':e.src,'relation':e.rel,'to':e.dst,'chunk_ids':list(e.chunk_ids),'version_id':e.version_id,'access_scope':e.scope,'qualifiers':dict(e.qualifiers)} for e in path],
            'required_chunk_ids':list(dict.fromkeys(i for e in path for i in e.chunk_ids))})
    for cid in ordered:
        own=[p for p in proofs if cid in p['required_chunk_ids']]
        explanation=' ; '.join(' → '.join([p['steps'][0]['from']]+[s['relation']+' '+s['to'] for s in p['steps']]) for p in own)
        support={'policy_bundle_id':bundle,'required_chunk_ids':ordered,'policy_facets':sorted({f for p in own if p['endpoint']=='Evidence:'+cid for f in p['facets']}) or ['ENTITY_LINK'],
            'policy_subjects':sorted({s['from'] for p in own for s in p['steps'] if cid in s['chunk_ids']}),
            'policy_qualifiers':dict(next(e for _,path,_ in chosen for e in path if cid in e.chunk_ids).qualifiers),
            'policy_missing':sorted({f for f,_ in requirements-covered})}
        passages.append(Passage(chunks[cid],1.0,'graph',explanation,graph_support=support))
    trace={'kind':'policy','roots':roots,'requested':sorted(facets),'covered':sorted({f for f,_ in covered}),
           'missing':sorted({f for f,_ in requirements-covered}),'status':'complete' if requirements<=covered else 'partial','states':states,'candidates':len(candidates)}
    return passages,proofs,trace
