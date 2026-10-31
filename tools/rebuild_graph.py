"""Clone a local index and rebuild only graph edges; chunks/vectors remain identical."""
import argparse,hashlib,json,shutil,sqlite3
from pathlib import Path

def rebuild(base,out):
    from tools.import_admissions import clone_store
    from askiisc_retrieval.contract import ChunkPayload
    from askiisc_retrieval.graph_rules import build_edges,GRAPH_RULES_VERSION
    from askiisc_retrieval.stores import JsonGraphStore
    from askiisc_retrieval.snapshot import JsonSnapshotProvider,Snapshot
    base=Path(base).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('Destination must not exist; active index is never overwritten')
    old=JsonSnapshotProvider(base/'snapshot.json').active()
    raw=(base/'chunks.jsonl').read_bytes();cs=[ChunkPayload(**json.loads(l)) for l in raw.splitlines() if l]
    active=[c for c in cs if c.version_id in old.active_version_ids]
    out.mkdir(parents=True);(out/'BUILD_INCOMPLETE').touch()
    clone_store(base/'qdrant',out/'qdrant')
    for name in ('raw','admissions'):
        if (base/name).exists():shutil.copytree(base/name,out/name)
    for p in base.glob('*.json'):
        if p.name not in ('graph.json','snapshot.json'):shutil.copy2(p,out/p.name)
    (out/'chunks.jsonl').write_bytes(raw)
    edges=build_edges(active);JsonGraphStore(edges).save(out/'graph.json')
    digest=hashlib.sha256(raw+GRAPH_RULES_VERSION.encode()).hexdigest()
    snap=Snapshot('snap-'+digest[:10],old.embedding_model,GRAPH_RULES_VERSION,old.active_version_ids)
    JsonSnapshotProvider.write(out/'snapshot.json',snap)
    # Session data is copied again immediately before an operator activates the candidate.
    auth=base/'auth/sessions.sqlite3'
    if auth.exists():
        (out/'auth').mkdir()
        with sqlite3.connect(f'file:{auth}?mode=ro',uri=True) as src,sqlite3.connect(out/'auth/sessions.sqlite3') as dst:src.backup(dst)
    report={'base_snapshot':old.snapshot_id,'snapshot':snap.snapshot_id,'chunks':len(active),'edges':len(edges),'chunks_sha256':hashlib.sha256(raw).hexdigest(),'graph_version':GRAPH_RULES_VERSION,'scopes':{s:sum(c.access_scope==s for c in active) for s in ('public','student','faculty')}}
    (out/'graph-build.json').write_text(json.dumps(report,indent=2));(out/'BUILD_INCOMPLETE').unlink()
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    print(json.dumps(rebuild(a.base,a.out),indent=2))
if __name__=='__main__':main()
