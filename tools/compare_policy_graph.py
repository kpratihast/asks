"""Matched-corpus vector / decomposed vector / policy Graph RAG evaluation.

Uses production auth, context packing, generation and citation validation.
Freeze questions and implementation before generation. Resume only identical runs.
"""
import argparse,json,os,re,time,hashlib,random,sys,asyncio
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))
ARMS=['vector','multi_query','graph']

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--index',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    from tools.local import configure
    configure();os.environ['HF_HUB_OFFLINE']='1';os.environ['AUTH_DB_PATH']=str(out/'.auth-test.sqlite3')
    from tools.import_admissions import clone_store
    from tools.compare_admissions import group_hits
    from askiisc_retrieval.cli import make_embedder
    from askiisc_retrieval.stores import QdrantStore,JsonGraphStore
    from askiisc_retrieval.snapshot import JsonSnapshotProvider
    from askiisc_retrieval.retriever import Retriever
    from askiisc_retrieval.citations import result_to_dict
    from askiisc_retrieval.multi_query import retrieve_multi
    from askiisc_serving.retrieval_client import LocalRetrieval,hits_from_result,_trace
    from askiisc_serving.generator import Generator
    from askiisc_serving.settings import Settings
    from askiisc_serving.app import create_app
    from fastapi.testclient import TestClient
    import httpx
    qs=[]
    for name,cohort in [('admissions_questions.json','existing'),('graph_transfer_questions.json','fresh')]:
        content=json.loads((ROOT/'config'/name).read_text());qs.extend(dict(q,cohort=cohort) for q in content['questions'])
    (out/'questions.json').write_text(json.dumps(qs,indent=2))
    files=[p for d in ('askiisc_retrieval','askiisc_serving') for p in (ROOT/d).rglob('*') if p.suffix in ('.py','.html')]+[Path(__file__),ROOT/'config/admissions_questions.json',ROOT/'config/graph_transfer_questions.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    settings=Settings(public_chat=True,rate_per_min=100000,global_concurrency=1)
    with httpx.Client(timeout=10,trust_env=False) as c:
        ids=[m['id'] for m in c.get(settings.vllm_url+'/v1/models').json()['data']]
        if settings.served_model not in ids:raise RuntimeError('Configured model is not running')
    metadata={'model':settings.served_model,'max_model_len':settings.max_model_len,'max_output_tokens':settings.max_output_tokens,'max_evidence':settings.max_evidence,'temperature':0,'chat_template_kwargs':settings.chat_template_kwargs,'snapshot':json.loads((a.index/'snapshot.json').read_text()),'chunks_sha256':hashlib.sha256((a.index/'chunks.jsonl').read_bytes()).hexdigest(),'implementation_sha256':hashes,'order_seed':20261029,'limitations':'Fresh author-written schema-aware transfer set, not independent blind holdout. One generation per cell. Balanced arm positions, prefix caching still affects latency.'}
    meta=out/'metadata.json'
    if meta.exists() and json.loads(meta.read_text())!=metadata:raise RuntimeError('Cannot resume changed experiment')
    meta.write_text(json.dumps(metadata,indent=2))
    target=out/'.index'
    if not target.exists():clone_store(a.index/'qdrant',target)
    st=QdrantStore('askiisc_chunks',384,path=target)
    r=Retriever(make_embedder(),st,JsonSnapshotProvider(a.index/'snapshot.json'),JsonGraphStore.load(a.index/'graph.json'),rerank_min=None,today_term='2026-AUG')
    class RecordingRetrieval(LocalRetrieval):
        arm='vector'
        async def run(self,question,pinned,mode,programme,academic_year_,*,scopes):
            self.variants=[question]
            if self.arm=='multi_query':res,self.variants=await asyncio.to_thread(retrieve_multi,self.r,question,scopes)
            else:res=await asyncio.to_thread(self.r.retrieve,question,list(scopes),self.arm=='graph',False)
            self.last=result_to_dict(res)
            return hits_from_result(self.last),_trace(self.last,mode)
    class RecordingGenerator(Generator):
        def prepare(self,*args,**kwargs):
            result=super().prepare(*args,**kwargs);self.last_messages,self.last_selected,self.last_budget=result;return result
    retrieval=RecordingRetrieval(r);gen=RecordingGenerator(settings);raw=out/'raw.jsonl'
    done={(x['id'],x['arm']) for x in map(json.loads,raw.read_text().splitlines())} if raw.exists() else set()
    rng=random.Random(metadata['order_seed']);permutations=[['vector','multi_query','graph'],['multi_query','graph','vector'],['graph','vector','multi_query'],['graph','multi_query','vector'],['multi_query','vector','graph'],['vector','graph','multi_query']]
    rng.shuffle(permutations)
    try:
        with TestClient(create_app(settings,retrieval=retrieval,generator=gen)) as client,raw.open('a') as f:
            assert client.get('/readyz').status_code==200
            for i,q in enumerate(qs):
                for arm in permutations[i%len(permutations)]:
                    if (q['id'],arm) in done:continue
                    retrieval.arm=arm;retrieval.last={};gen.last_selected=[];gen.last_messages=[];gen.last_budget={}
                    t=time.perf_counter();resp=client.post('/v1/ask',json={'question':q['question'],'mode':'full' if arm=='graph' else 'vector'});d=resp.json();seconds=round(time.perf_counter()-t,3)
                    sel=gen.last_selected;groups=group_hits(q['groups'],sel);answer=d.get('answer') or '';facthits=[bool(re.search(p,answer,re.I)) for p in q['facts']]
                    passed=d.get('status')=='abstained' if q.get('abstain') else d.get('status')=='answered' and all(facthits)
                    rec={'id':q['id'],'cohort':q['cohort'],'arm':arm,'expected':q,'http':resp.status_code,'seconds':seconds,'response':d,'group_hits':groups,'fact_hits':facthits,'fact_pass':passed,'selected':sel,'retrieval':retrieval.last,'queries':retrieval.variants,'messages':gen.last_messages,'budget':gen.last_budget}
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
                    print(q['id'],arm,d.get('status'),'facts',passed,'evidence',groups,seconds,answer[:110],flush=True)
            client.portal.call(gen.c.aclose)
    finally:st.close()
if __name__=='__main__':main()
