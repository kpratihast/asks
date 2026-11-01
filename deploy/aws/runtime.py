"""AWS-only entry point. Never reads or changes config/local.json or local SMTP files."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent.parent))
from release_lib import MODEL,runtime_config

def environment(release,config,state,domain):
    release=Path(release).resolve();state=Path(state).resolve()
    manifest=json.loads((release/'release.json').read_text())
    raw=runtime_config(config)
    env={k:('1' if v else '0') if isinstance(v,bool) else json.dumps(v) if isinstance(v,dict) else str(v) for k,v in raw.items()}
    env.update({'ASKIISC_BACKEND':'stub','ASKIISC_EMBED_MODEL':MODEL,'ASKIISC_EMBED_DIM':'384',
        'ASKIISC_RERANKER':'none','ASKIISC_RERANK_MIN':'none','ASKIISC_DEVICE':'cpu',
        'ASKIISC_DATA_DIR':str(state/'indexes'/manifest['release_id']),
        'AUTH_DB_PATH':str(state/'auth/sessions.sqlite3'),'AUTH_COOKIE_SECURE':'1',
        'AUTH_PUBLIC_ORIGIN':'https://'+domain,'PUBLIC_CHAT':'1','SERVING_RETRIEVAL':'local',
        'ASKIISC_PRIVATE_MANIFEST':str(release/'config/restricted_documents.json'),
        'FACULTY_DIRECTORY':str(release/'config/faculty_directory.json'),
        'HF_HOME':str(release/'assets/hf'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'GLOBAL_CONCURRENCY':str(raw.get('GLOBAL_CONCURRENCY',1)),
        'TENANT_CONCURRENCY':str(raw.get('TENANT_CONCURRENCY',1)),
        'TENANT_QUEUE':str(raw.get('TENANT_QUEUE',4)),
        'QUEUE_WAIT_S':str(raw.get('QUEUE_WAIT_S',120)),'GEN_TIMEOUT_S':'180','REGENERATE':'0',
        'FORWARDED_ALLOW_IPS':'127.0.0.1','PYTHONDONTWRITEBYTECODE':'1','TZ':'Asia/Kolkata'})
    return env

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8100)
    p.add_argument('--release',type=Path,default=HERE.parent.parent)
    p.add_argument('--state',type=Path,default=Path('/var/lib/askiisc'))
    p.add_argument('--config',type=Path);p.add_argument('--domain');p.add_argument('--check',action='store_true');a=p.parse_args()
    release=a.release.resolve();rid=json.loads((release/'release.json').read_text())['release_id']
    host=json.loads(Path('/etc/askiisc/host.json').read_text()) if not a.domain else {'domain':a.domain}
    cfg=json.loads((a.config or Path('/etc/askiisc/releases')/(rid+'.json')).read_text())
    # AWS starts from an explicit allowlist. Inherited local profiles cannot override paths or roles.
    for key in list(os.environ):
        if key.startswith(('ASKIISC_','AUTH_','SMTP_','HF_','TRANSFORMERS_','SERVING_')) or key in {'QDRANT_URL','NEO4J_PASSWORD','MODEL_API_KEY','OPENROUTER_API_KEY','VLLM_URL'}:
            os.environ.pop(key,None)
    os.environ.update(environment(release,cfg,a.state,host['domain']))
    if hasattr(time,'tzset'):time.tzset()
    if a.check:
        from askiisc_retrieval.private_documents import DocumentCatalog
        from askiisc_retrieval.service import build_retriever
        from askiisc_retrieval.graph_rules import GRAPH_RULES_VERSION
        DocumentCatalog().page_inputs()
        r=build_retriever()
        try:
            snap=r.snapshots.active();manifest=json.loads((release/'release.json').read_text())
            if snap.snapshot_id!=manifest['snapshot'] or snap.graph_version!=GRAPH_RULES_VERSION:raise ValueError('Snapshot/graph rules mismatch')
            if not r.store.count():raise ValueError('Index is empty')
            print(json.dumps({'snapshot':snap.snapshot_id,'graph':snap.graph_version,'chunks':manifest['chunks'],'status':'validated'}))
        finally:r.store.close()
        return
    import uvicorn
    from askiisc_serving.app import create_app
    uvicorn.run(create_app(),host='127.0.0.1',port=a.port,workers=1,log_level='warning',proxy_headers=True,forwarded_allow_ips='127.0.0.1')
if __name__=='__main__':main()
