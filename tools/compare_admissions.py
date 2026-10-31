#!/usr/bin/env python3
"""Compare the preserved CDS-only index with a bounded admissions+CDS candidate.

Three arms: previous corpus Graph RAG; expanded corpus vector RAG; expanded corpus Graph RAG.
Same production serving path, source cap, 4096 context and actual configured local model.
Checkpointed raw outputs and exact selected evidence; regex rubrics are not semantic judging.
"""
import argparse,dataclasses,json,os,re,sys,time,statistics,hashlib,html
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))

def group_hits(groups,selected):
    return [any(any(g['url'] in h['url'] for g in group.get('alternatives',[group])) and any(re.search(g['pattern'],h['text'],re.I) for g in group.get('alternatives',[group]) if g['url'] in h['url']) for h in selected) for group in groups]

def report(out,qs):
    rows=[json.loads(l) for l in (out/'raw.jsonl').read_text().splitlines()]
    reviews=json.loads((out/'review.json').read_text()) if (out/'review.json').exists() else {}
    arms=['previous_graph','expanded_vector','expanded_graph'];metrics={}
    for arm in arms:
        rs=[r for r in rows if r['arm']==arm];ans=[r for r in rs if not r['expected'].get('abstain')];unknown=[r for r in rs if r['expected'].get('abstain')]
        metrics[arm]={'n':len(rs),'answerable_n':len(ans),'all_evidence_in_prompt':sum(all(r['group_hits']) for r in ans),'fact_rubric_pass':sum(r['fact_pass'] for r in ans),'correct_abstentions':sum(r['response'].get('status')=='abstained' for r in unknown),'unknown_n':len(unknown),'http_errors':sum(r['http']!=200 for r in rs),'fallbacks':sum(r['response'].get('status')=='fallback' for r in rs),'median_seconds':statistics.median(r['seconds'] for r in rs)}
        metrics[arm]['reviewed_complete_supported']=sum(reviews.get(r['id']+'/'+arm,{}).get('complete_and_supported',False) for r in ans) if reviews else None
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    lines=['# Public admissions + CDS: local comparison','', 'Exploratory development checks, not a held-out benchmark. Same local model and generation settings in all arms. The previous-corpus arm uses the revised code too, so the corpus comparison isolates ingestion; it does not measure the old entrance-exam login-hint bug.','', 'All required evidence is checked against source URLs and text patterns in the actual prompt. Fact checks are regex proxies, not semantic accuracy. The report includes every raw answer and selected evidence for review.','', '| Arm | Evidence complete / 11 | Fact rubric / 11 | Reviewed complete / 11 | Unknown abstention / 1 | HTTP errors | Fallbacks | Median seconds |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,m in metrics.items():lines.append(f"| {arm} | {m['all_evidence_in_prompt']} | {m['fact_rubric_pass']} | {m['reviewed_complete_supported']} | {m['correct_abstentions']} | {m['http_errors']} | {m['fallbacks']} | {m['median_seconds']:.2f} |")
    lines+=['','Manual review requires a complete answer whose claims are supported by the selected sources. Partial answers fail. Two regex false negatives concern the form name and email obfuscation; review.json records every decision. Arms run in fixed previous/vector/graph order, with one generation per arm/question. Local prefix caching benefits later arms, so latency is diagnostic, not evidence that Graph RAG is faster.','','These questions mostly require combining public text, not traversing existing course–instructor–lab or course–slot–exam graph edges. Gains over the previous corpus are coverage gains. A tie between expanded vector and graph arms must not be described as a Graph RAG win. Admission cycles and conflicting sources require explicit qualification.','']
    cards=[]
    esc=lambda x:html.escape(str(x))
    for q in qs:
        lines += ['## '+q['id'],'',q['question'],'']
        card='<section><h2>'+esc(q['question'])+'</h2><div class="grid">'
        for arm in arms:
            r=next((r for r in rows if r['id']==q['id'] and r['arm']==arm),None)
            if not r:continue
            d=r['response'];answer=d.get('answer') or d.get('detail') or d.get('error')
            lines+=['**'+arm+'** — '+str(d.get('status'))+'; fact rubric '+str(r['fact_pass'])+'; evidence groups '+str(r['group_hits']),'',str(answer),'']
            card+='<article><h3>'+esc(arm)+'</h3><p>'+esc(d.get('status'))+' · rubric '+str(r['fact_pass'])+' · '+str(r['seconds'])+'s</p><p class="answer">'+esc(answer)+'</p><details><summary>Actual prompt evidence</summary>'
            for h in r['selected']:card+='<h4>'+esc(h['prompt_marker']+' '+h['section'])+'</h4><p>'+esc(h['text'])+'</p><a href="'+esc(h['url'])+'">Official source</a>'
            review=reviews.get(q['id']+'/'+arm,{})
            if review:
                lines+=['Review: '+review['note'],'']
                card+='<p><strong>Review:</strong> '+esc(review['note'])+'</p>'
            card+='</details></article>'
        cards.append(card+'</div></section>')
    (out/'REPORT.md').write_text('\n'.join(lines))
    table='<table><tr><th>Arm</th><th>All evidence /11</th><th>Fact rubric /11</th><th>Reviewed complete /11</th><th>Unknown abstention /1</th></tr>'+''.join('<tr><td>'+esc(a)+'</td><td>'+str(m['all_evidence_in_prompt'])+'</td><td>'+str(m['fact_rubric_pass'])+'</td><td>'+str(m['reviewed_complete_supported'])+'</td><td>'+str(m['correct_abstentions'])+'</td></tr>' for a,m in metrics.items())+'</table>'
    (out/'comparison.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Admissions + CDS comparison</title><style>body{font:15px/1.6 system-ui;background:#f5f7f8;color:#1b3238;margin:0}main{max-width:1500px;padding:32px;margin:auto}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}article{background:white;padding:20px;border:1px solid #d7e2e6;border-radius:12px;overflow-wrap:anywhere}section{margin-top:32px}h2{font-size:21px}.answer{white-space:pre-wrap}details{font-size:13px}summary{cursor:pointer}td,th{padding:12px;border-bottom:1px solid #cbd9de;text-align:left}table{background:white;width:100%;border-collapse:collapse}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><main><h1>Public admissions + CDS</h1><p>12 exploratory questions · actual local Qwen outputs · same generation settings. Corpus expansion and graph ablation are measured separately.</p><p>Manual review requires complete, supported answers. Regex fact rubrics are separate proxies. Fixed arm order and prefix caching make the timing figures diagnostic. Expand the evidence to inspect source support. Conflicts and missing sources remain visible; data-coverage gains do not establish graph-reasoning gains.</p>'''+table+''.join(cards)+'</main></html>')
    print(json.dumps(metrics,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--report-only',action='store_true');a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    qs=json.loads((ROOT/'config/admissions_questions.json').read_text())['questions']
    if a.report_only:report(out,qs);return
    from tools.local import configure
    configure();os.environ['HF_HUB_OFFLINE']='1';os.environ['AUTH_DB_PATH']=str(out/'auth-test.sqlite3')
    from tools.import_admissions import clone_store
    from askiisc_retrieval.cli import make_embedder
    from askiisc_retrieval.stores import QdrantStore,JsonGraphStore
    from askiisc_retrieval.snapshot import JsonSnapshotProvider
    from askiisc_retrieval.retriever import Retriever
    from askiisc_retrieval.citations import result_to_dict
    from askiisc_serving.retrieval_client import LocalRetrieval,hits_from_result,_trace
    from askiisc_serving.generator import Generator
    from askiisc_serving.settings import Settings
    from askiisc_serving.app import create_app
    from askiisc_serving.email_auth import EmailAuth,AuthConfig
    from fastapi.testclient import TestClient
    import httpx,asyncio
    emb=make_embedder();retrievers={};stores=[]
    for label,path in [('previous',a.base.resolve()),('expanded',a.candidate.resolve())]:
        target=out/('.index-'+label)
        if not target.exists():clone_store(path/'qdrant',target)
        st=QdrantStore('askiisc_chunks',384,path=target);stores.append(st)
        retrievers[label]=Retriever(emb,st,JsonSnapshotProvider(path/'snapshot.json'),JsonGraphStore.load(path/'graph.json'),rerank_min=None,today_term='2026-AUG')
    class RecordingRetrieval(LocalRetrieval):
        async def run(self,question,pinned,mode,programme,academic_year_,*,scopes):
            res=await asyncio.to_thread(self.r.retrieve,question,list(scopes),mode=='full',False)
            self.last=result_to_dict(res)
            return hits_from_result(self.last),_trace(self.last,mode)
    class RecordingGenerator(Generator):
        def prepare(self,*args,**kwargs):
            result=super().prepare(*args,**kwargs);self.last_messages,self.last_selected,self.last_budget=result;return result
    settings=Settings(public_chat=True,rate_per_min=100000,global_concurrency=1)
    with httpx.Client(timeout=10,trust_env=False) as c:
        ids=[m['id'] for m in c.get(settings.vllm_url+'/v1/models').json()['data']]
        if settings.served_model not in ids:raise RuntimeError('Configured local model not running')
    metadata={'recorded_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'model':settings.served_model,'max_model_len':settings.max_model_len,'max_output_tokens':settings.max_output_tokens,'max_evidence':settings.max_evidence,'temperature':0,'chat_template_kwargs':settings.chat_template_kwargs,'snapshots':{k:r.snapshots.active().snapshot_id for k,r in retrievers.items()},'questions_sha256':hashlib.sha256((ROOT/'config/admissions_questions.json').read_bytes()).hexdigest()}
    meta=out/'metadata.json'
    if meta.exists():
        old=json.loads(meta.read_text())
        if any(old.get(k)!=v for k,v in metadata.items() if k!='recorded_at'):raise RuntimeError('Cannot resume a changed evaluation')
    else:meta.write_text(json.dumps(metadata,indent=2))
    retrieval=RecordingRetrieval(retrievers['previous']);gen=RecordingGenerator(settings)
    done=set()
    raw=out/'raw.jsonl'
    if raw.exists():done={(r['id'],r['arm']) for r in map(json.loads,raw.read_text().splitlines())}
    try:
        with TestClient(create_app(settings,retrieval=retrieval,generator=gen)) as client,raw.open('a') as f:
            if client.get('/readyz').status_code!=200:raise RuntimeError('Readiness failed')
            for q in qs:
                for arm,index,mode in [('previous_graph','previous','full'),('expanded_vector','expanded','vector'),('expanded_graph','expanded','full')]:
                    if (q['id'],arm) in done:continue
                    retrieval.r=retrievers[index];retrieval.last={};gen.last_selected=[];gen.last_messages=[];gen.last_budget={}
                    t=time.perf_counter();resp=client.post('/v1/ask',json={'question':q['question'],'mode':mode});d=resp.json();secs=round(time.perf_counter()-t,3)
                    sel=gen.last_selected;groups=group_hits(q['groups'],sel)
                    answer=d.get('answer') or '';facthits=[bool(re.search(p,answer,re.I)) for p in q['facts']]
                    passed=(d.get('status')=='abstained') if q.get('abstain') else d.get('status')=='answered' and all(facthits)
                    rec={'id':q['id'],'arm':arm,'expected':q,'http':resp.status_code,'seconds':secs,'response':d,'group_hits':groups,'fact_hits':facthits,'fact_pass':passed,'selected':sel,'retrieval':retrieval.last,'messages':gen.last_messages,'budget':gen.last_budget}
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
                    print(q['id'],arm,resp.status_code,d.get('status'),'rubric',passed,'evidence',groups,secs,answer[:100],flush=True)
            client.portal.call(gen.c.aclose)
    finally:
        for st in stores:st.close()
    report(out,qs)
if __name__=='__main__':main()
