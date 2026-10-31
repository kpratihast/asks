"""A bounded graph-free baseline sharing the policy intent vocabulary.

One original query plus at most four facet queries. Each facet's highest-ranked
eligible result gets a slot, then reciprocal-rank fusion fills remaining slots.
It receives no graph, gold answers, URLs, or evaluation labels.
"""
from dataclasses import replace
from .policy_graph import requested_facets,programmes,department
from .citations import build_citations

LABELS={
 'DURATION':'programme duration minimum maximum years months',
 'COURSEWORK':'coursework credits required', 'DISSERTATION':'dissertation project credits duration months',
 'FUNDING_AMOUNT':'monthly scholarship fellowship amount JRF SRF',
 'FUNDING_CONDITIONS':'scholarship eligibility exceptions salary sponsors CFTI maximum support years',
 'FEES':'annual tuition fees first year total refundable deposits',
 'APPLICATION_FEE':'online application fee category',
 'APPLICATION_CHOICES':'single application maximum departments disciplines programmes',
 'LAB_PREFERENCES':'candidate information form choose labs mandatory',
 'TEST_SYLLABUS':'written test syllabus programming languages mathematics',
 'ELIGIBILITY':'eligibility qualifications accepted GATE papers',
 'INTAKE':'admission intake seats','CONTACT':'admissions email contact', 'QUOTA':'NRI management quota',
 'SELECTION':'selection procedure written test oral interview',
 'INTERVIEW_DATES':'interview dates','APPLICATION_DEADLINE':'application deadline',
 'SESSION_START':'semester commencement','NOTICE':'mid year admission notice departments',
 'AMENDMENT':'admissions corrigendum amendment',
}
NAMES={'MTech':'M.Tech.','MTechResearch':'M.Tech. (Research)','PhD':'Ph.D.','MDes':'M.Des.','MSc':'M.Sc.','BTech':'B.Tech.'}

def queries(question):
    context=' '.join([department(question)]+[NAMES[p] for p in programmes(question)])
    return [question]+[context+' '+LABELS[f] for f in requested_facets(question)[:4]]

def retrieve_multi(retriever,question,scopes,max_passages=6):
    variants=queries(question);results=[retriever.retrieve(q,list(scopes),False,False) for q in variants]
    fused={};items={}
    for result in results:
        for rank,p in enumerate(result.passages):
            cid=p.chunk.chunk_id;items[cid]=p;fused[cid]=fused.get(cid,0)+1/(60+rank+1)
    order=[]
    for result in results[1:]:
        if result.passages:
            cid=result.passages[0].chunk.chunk_id
            if cid not in order:order.append(cid)
    order.extend(cid for cid in sorted(fused,key=lambda c:(-fused[c],c)) if cid not in order)
    ps=[replace(items[c],score=fused[c],marker=None) for c in order[:max_passages]]
    return replace(results[0],query=question,passages=ps,citations=build_citations(ps)),variants
