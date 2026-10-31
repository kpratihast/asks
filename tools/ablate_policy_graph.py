"""Retrieval-only causal check; removes typed bridges while retaining all chunks."""
import json,argparse
from pathlib import Path
from askiisc_retrieval.contract import ChunkPayload
from askiisc_retrieval.policy_graph import retrieve_policy
from askiisc_retrieval.stores import JsonGraphStore
from askiisc_retrieval.snapshot import JsonSnapshotProvider
from tools.compare_admissions import group_hits

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=Path,required=True);ap.add_argument('--questions',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    cs=[ChunkPayload(**json.loads(l)) for l in (a.index/'chunks.jsonl').read_text().splitlines()];s=JsonSnapshotProvider(a.index/'snapshot.json').active();edges=JsonGraphStore.load(a.index/'graph.json').edges
    class Store:
        def get(self,ids):return [c for c in cs if c.chunk_id in ids]
    rows=[]
    for q in json.loads(a.questions.read_text()):
        for label,removed in [('full',set()),('no_programme_bridge',{'IS_A'}),('no_institution_bridge',{'AT_INSTITUTION'})]:
            graph=JsonGraphStore([e for e in edges if e.rel not in removed]);ps,proofs,trace=retrieve_policy(q['question'],graph,Store(),s,['public'])
            selected=[{'url':p.chunk.source_url,'text':p.chunk.text} for p in ps]
            rows.append({'id':q['id'],'cohort':q['cohort'],'category':q.get('category','existing'),'arm':label,'group_hits':group_hits(q['groups'],selected),'expected_abstain':q.get('abstain',False),'trace':trace,'paths':len(proofs),'chunk_ids':[p.chunk.chunk_id for p in ps]})
    a.out.write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
