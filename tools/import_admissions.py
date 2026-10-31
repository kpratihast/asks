#!/usr/bin/env python3
"""Bounded public admissions import into a NEW local index; never edits the serving index.

Use --fetch or a verified --cache containing raw/ and download.json. Switch config/local.json
only after evaluating the candidate. Auth sessions are copied consistently with SQLite backup.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,sqlite3,sys,time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def sqlite_copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True) as reader:
        with sqlite3.connect(target) as writer:reader.backup(writer)

def clone_store(source,target):
    target.mkdir(parents=True)
    shutil.copy2(source/'meta.json',target/'meta.json')
    for src in source.rglob('storage.sqlite'):sqlite_copy(src,target/src.relative_to(source))

def fetch_sources(profile,cache):
    import httpx
    from askiisc_retrieval.fetch import UA
    raw=cache/'raw';raw.mkdir(parents=True,exist_ok=True);records=[];robots={};total=0
    with httpx.Client(timeout=45,follow_redirects=False,trust_env=False,headers={'User-Agent':UA}) as client:
        for spec in profile['sources']:
            url=spec['url'];host=urlparse(url).netloc
            if host not in {'www.iisc.ac.in','cds.iisc.ac.in'}:raise ValueError('Unapproved source host')
            if host not in robots:
                response=client.get('https://'+host+'/robots.txt');response.raise_for_status()
                parser=RobotFileParser();parser.parse(response.text.splitlines());robots[host]=parser
            if not robots[host].can_fetch(UA,url):raise ValueError('robots.txt disallows '+url)
            body=bytearray()
            with client.stream('GET',url) as response:
                response.raise_for_status()
                for part in response.iter_bytes():
                    body.extend(part)
                    if len(body)>5*1024*1024:raise ValueError('Per-source byte limit exceeded')
            total+=len(body)
            if total>profile['limits']['total_bytes']:raise ValueError('Total byte boundary exceeded')
            name=hashlib.sha256(url.encode()).hexdigest()[:16]+('.pdf' if spec['mode']=='pdf' else '.html')
            (raw/name).write_bytes(body)
            records.append({'url':url,'file':name,'bytes':len(body),'sha256':digest(raw/name),'fetched_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
            time.sleep(.5)
    (cache/'download.json').write_text(json.dumps(records,indent=2))

from askiisc_retrieval.admissions import cached_pages


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--fetch',action='store_true')
    ap.add_argument('--sources',type=Path,default=ROOT/'config/admissions_sources.json')
    a=ap.parse_args();base=a.base.resolve();out=a.out.resolve();cache=a.cache.resolve()
    if out.exists():raise SystemExit('Output must be new; existing index is never overwritten')
    if not (base/'snapshot.json').is_file():raise SystemExit('Base snapshot missing')
    from tools.local import configure
    configure();os.environ.setdefault('HF_HUB_OFFLINE','1')
    profile=json.loads(a.sources.read_text());cache.mkdir(parents=True,exist_ok=True)
    if a.fetch:fetch_sources(profile,cache)
    pages,report=cached_pages(profile,cache)
    from askiisc_retrieval.cli import make_embedder
    from askiisc_retrieval.ingest_stub import build_index
    from askiisc_retrieval.contract import ChunkPayload
    from askiisc_retrieval.stores import QdrantStore,JsonGraphStore
    from askiisc_retrieval.snapshot import JsonSnapshotProvider,Snapshot
    from askiisc_retrieval.graph_rules import build_edges,GRAPH_RULES_VERSION
    from askiisc_retrieval.chunking import h
    from askiisc_retrieval import config
    old=json.loads((base/'snapshot.json').read_text());replaced={p.url for p in pages}
    kept=[ChunkPayload(**json.loads(line)) for line in (base/'chunks.jsonl').read_text().splitlines() if line.strip()]
    kept=[c for c in kept if c.version_id in old['active_version_ids'] and c.source_url not in replaced]
    emb=make_embedder()
    if emb.model_id!=old['embedding_model']:raise ValueError('Embedding model differs from base')
    out.mkdir();(out/'IMPORT_INCOMPLETE').touch();clone_store(base/'qdrant',out/'qdrant')
    store=QdrantStore(config.COLLECTION,config.EMBED_DIM,path=out/'qdrant')
    try:added=build_index(pages,emb,store,out/'graph.json',out/'snapshot.json',out/'chunks.jsonl')
    finally:store.close()
    new=[ChunkPayload(**json.loads(line)) for line in (out/'chunks.jsonl').read_text().splitlines() if line.strip()]
    combined=kept+new;versions=frozenset(c.version_id for c in combined)
    snap=Snapshot('snap-'+h('|'.join(sorted(versions))+emb.model_id+GRAPH_RULES_VERSION,10),emb.model_id,GRAPH_RULES_VERSION,versions)
    JsonSnapshotProvider.write(out/'snapshot.json',snap);JsonGraphStore(build_edges(combined)).save(out/'graph.json')
    (out/'chunks.jsonl').write_text(''.join(json.dumps(c.to_dict(),ensure_ascii=False)+'\n' for c in combined))
    if (base/'raw').exists():shutil.copytree(base/'raw',out/'raw')
    dest=out/'admissions';dest.mkdir();shutil.copytree(cache/'raw',dest/'raw');shutil.copytree(cache/'normalized',dest/'normalized')
    shutil.copy2(cache/'download.json',dest/'download.json');shutil.copy2(a.sources,dest/'sources.json')
    # The local auth default follows DATA; retain current signed-in sessions when switching.
    auth=base/'auth'
    if auth.exists():
        for src in auth.glob('*.sqlite3'):sqlite_copy(src,out/'auth'/src.name)
    report.update({'base_snapshot':old['snapshot_id'],'snapshot_id':snap.snapshot_id,'preserved_chunks':len(kept),'added_chunks':len(new),'total_chunks':len(combined),'scopes':{scope:sum(c.access_scope==scope for c in combined) for scope in ('public','student','faculty')},'build':added})
    (out/'admissions/ingestion.json').write_text(json.dumps(report,indent=2));(out/'IMPORT_INCOMPLETE').unlink()
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','build')},indent=2))
if __name__=='__main__':main()
