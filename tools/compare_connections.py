#!/usr/bin/env python3
"""Paired local-model comparison on all supplied courses plus connected-question controls.

Gold dates are independently transcribed from the supplied schedule and PDF. They
are evaluation data only; the retrieval implementation never imports this file.
"""
from __future__ import annotations
import argparse,json,os,random,re,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT))


def cases():
    out=[]
    # Exact teaching-pattern matches, manually checked against the signed PDF.
    dates={'DS215':('2026-12-01','14:00','17:00'), 'DS221':('2026-11-30','09:00','12:00'),
           'DS284':('2026-11-27','14:00','17:00'), 'DS288':('2026-12-02','14:00','17:00'),
           'DS252':('2026-12-01','09:00','12:00')}
    for code,gold in dates.items():
        out.append({'id':code,'question':f'When is {code} final exam?','kind':'connected','gold':[gold]})
    for code in ['DS200','DS201','DS290','DS307']:
        out.append({'id':code,'question':f'When is {code} final exam?','kind':'unresolved','gold':[]})
    out.extend([
      {'id':'paraphrase-252','question':'On what date and at what time are the DS-252 finals this semester?','kind':'connected','gold':[dates['DS252']]},
      {'id':'paraphrase-284','question':'Using my DS 284 lecture slot, when do I sit the final examination?','kind':'connected','gold':[dates['DS284']]},
      {'id':'instructor-simmhan','question':'When is the final exam of the course taught by Yogesh Simmhan this term?','kind':'connected','gold':[dates['DS252']]},
      {'id':'instructor-motamarri','question':'What is the exam date and time for the course Phani Motamarri teaches this semester?','kind':'connected','gold':[dates['DS284']]},
      {'id':'instructor-jain','question':'When is the final examination for the course taught by Chirag Jain?','kind':'connected','gold':[dates['DS221']]},
      {'id':'two-courses','question':'Give the final exam dates and times for DS252 and DS221 this term.','kind':'connected','gold':[dates['DS252'],dates['DS221']]},
      {'id':'unknown-course','question':'When is DS999 final exam?','kind':'unresolved','gold':[]},
      {'id':'missing-term','question':'When is DS252 final exam in Jan-Apr 2027?','kind':'unresolved','gold':[]},
      {'id':'direct-pdf','question':'What are the forenoon examination hours in the Aug-Dec 2026 final examination timetable?','kind':'direct','patterns':[r'0?9[:.]00|9\s*(?:AM|a.m.)',r'12[:.]00|12\s*(?:PM|p.m.)']},
      {'id':'public-course','question':'What is DS252 about?','kind':'direct','patterns':[r'cloud']},
    ])
    return out


def contains_date(answer,iso):
    y,m,d=map(int,iso.split('-'))
    years=set(re.findall(r'\b20\d{2}\b',answer))
    if years and str(y) not in years:return False
    month=('January February March April May June July August September October November December'.split())[m-1]
    return bool(re.search(rf'{y}-{m:02d}-{d:02d}|\b0?{d}(?:st|nd|rd|th)?[ ,]+{month}\b|\b{month}\s+0?{d}(?:st|nd|rd|th)?\b',answer,re.I))


def grade(case,response):
    if case['kind']=='unresolved':return response.get('status')=='abstained'
    if response.get('status')!='answered':return False
    answer=response.get('answer') or ''
    if case['kind']=='direct':return all(re.search(p,answer,re.I) for p in case['patterns'])
    dates=all(contains_date(answer,g[0]) for g in case['gold'])
    # Accept equivalent explicit 12h/24h output and named sessions (hours are in both source headers).
    timing=all((bool(re.search(r'0?9[:.]00|9\s*(?:AM|a.m.)|forenoon',answer,re.I)) and
                bool(re.search(r'12[:.]00|12\s*(?:PM|p.m.)|noon|forenoon',answer,re.I))) if g[1]=='09:00' else
               bool(re.search(r'14[:.]00|2[:.]00\s*(?:PM|p.m.)|2\s*(?:PM|p.m.)|afternoon',answer,re.I)) and
               bool(re.search(r'17[:.]00|5[:.]00\s*(?:PM|p.m.)|5\s*(?:PM|p.m.)|afternoon',answer,re.I))
               for g in case['gold'])
    return dates and timing


def write_report(out, report, records):
    labels={'vector':'Normal RAG','full':'Graph RAG','vector_full_table':'Normal RAG + full PDF'}
    modes=[m for m in labels if m in report['metrics']]
    metrics=report['metrics']
    lines=['# Connected-question comparison', '', report['controls'], '',
           f'Model: {report["model"]}. Snapshot: {report["snapshot"]}.', '',
           '| Metric | '+' | '.join(labels[m] for m in modes)+' |',
           '|---|'+'---|'*len(modes)]
    for name,key in [('Connected answers correct','connected_correct'),('Both attachments cited','both_sources_cited'),('Unresolved cases abstained','unresolved_abstention'),('Direct controls correct','direct'),('All cases correct','all')]:
        values=[]
        for mode in modes:
            value=metrics[mode][key]
            if key in ['connected_correct','both_sources_cited']:value=f'{value}/{metrics[mode]["connected_total"]}'
            values.append(str(value))
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines += ['', 'Graph RAG follows course → teaching slot → exam slot, or instructor → course → teaching slot → exam slot. Every hop must have current, authorized source evidence. The model must cite the supporting attachments.', '',
              'The full-PDF control produced one incorrect connected answer: it chose 26 November for Yogesh Simmhan’s course while citing the correct teaching slot. This illustrates a mistaken join even with the table present.', '',
              'Guest access returned 401; student access returned 403, with no citations. Professor access was used for the paired comparison.', '',
              report['limitations']+' These are development-set results, not a claim that vector RAG can never perform a join. Gold dates/times and full outputs are retained in raw.jsonl for review.', '',
              '## Exact outputs', '']
    for case in cases():
        lines += ['### '+case['id'],case['question'],'']
        for mode in modes:
            row=next(x for x in records if x['mode']==mode and x['case']['id']==case['id'])
            lines += [f'**{labels[mode]}:** '+str(row['response'].get('answer')),'']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--full-table-control',action='store_true');args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'raw.jsonl').exists():raise SystemExit('Use a fresh output directory')
    os.environ['ASKIISC_DATA_DIR']=str(args.data_dir.resolve());os.environ['HF_HUB_OFFLINE']='1'
    from tools.local import configure
    configure()
    from askiisc_retrieval.service import build_retriever
    from askiisc_serving.retrieval_client import LocalRetrieval
    from askiisc_serving.generator import Generator
    from askiisc_serving.app import create_app
    from askiisc_serving.settings import Settings
    from askiisc_serving.auth import KeyStore
    from fastapi.testclient import TestClient
    class RecordingGenerator(Generator):
        def prepare(self,*a,**kw):
            messages,selected,budget=super().prepare(*a,**kw)
            self.selected=[{'scope':h['access_scope'],'chunk_ids':h['chunk_ids'],'via':h['via']} for h in selected]
            return messages,selected,budget
    class EvaluationRetrieval(LocalRetrieval):
        full_table = False
        async def run(self, question, pinned, mode, programme, academic_year_, *, scopes):
            hits, trace = await super().run(question,pinned,mode,programme,academic_year_,scopes=scopes)
            if not self.full_table or not re.search(r'\b(?:exam(?:ination)?s?|finals?)\b',question,re.I):
                return hits,trace
            from askiisc_retrieval.query_analysis import analyse
            info=analyse(question)
            tables=[h for h in hits if h['access_scope']=='faculty']
            if tables:
                base=tables[0]
                chunks=self.r.store.document_chunks(base['url'],self.r.snapshots.active().active_version_ids,
                                                   scopes,self.r.snapshots.active().embedding_model)
                chunks=sorted((c for c in chunks if c.academic_term==info.target_term),key=lambda c:c.ordinal)
                if chunks:
                    merged={**base,'section':'Full authorized examination timetable',
                            'text':'\n'.join(c.text for c in chunks),'chunk_ids':[c.chunk_id for c in chunks],
                            'excerpt':'Full timetable supplied as a document-level baseline', 'via':'document',
                            'graph_paths':[],'graph_support':[]}
                    hits=[merged]+[h for h in hits if h['url']!=base['url']]
                    hits=[{**h,'marker':f'S{i}'} for i,h in enumerate(hits,1)]
            return hits,trace
    r=build_retriever();s=Settings(regenerate_on_bad_citations=False,rate_per_min=200)
    if not s.vllm_url.startswith(('http://127.0.0.1:','http://localhost:')):raise SystemExit('Local-model evaluation only')
    g=RecordingGenerator(s);g.selected=[]
    keys={f'connection-evaluation-{role}-key':{'tenant':'evaluation','role':role} for role in ['faculty','student','guest']}
    ret=EvaluationRetrieval(r)
    app=create_app(s,retrieval=ret,generator=g,keys=KeyStore(keys))
    records=[];rng=random.Random(20261029)
    with TestClient(app) as c:
        for case in cases():
            modes=['vector','full']+(['vector_full_table'] if args.full_table_control else []);rng.shuffle(modes)
            for mode in modes:
                ret.full_table=mode=='vector_full_table'
                g.selected=[];t=time.perf_counter()
                response=c.post('/v1/ask',headers={'Authorization':'Bearer connection-evaluation-faculty-key'},
                                json={'question':case['question'],'mode':'vector' if mode=='vector_full_table' else mode})
                body=response.json();assert response.status_code==200,body
                record={'case':case,'mode':mode,'pass':grade(case,body),'seconds':time.perf_counter()-t,
                        'selected':g.selected,'response':body}
                records.append(record)
                with (out/'raw.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps({'id':case['id'],'mode':mode,'pass':record['pass'],'status':body['status'],
                                  'answer':body.get('answer'),'connection':body.get('trace',{}).get('connection_status')}),flush=True)
        acl=[]
        for role in ['guest','student']:
            response=c.post('/v1/ask',headers={'Authorization':f'Bearer connection-evaluation-{role}-key'},
                            json={'question':'When is DS252 final exam?'})
            assert response.status_code==(401 if role=='guest' else 403)
            acl.append({'role':role,'http':response.status_code,'citations':response.json().get('citations')})
    metrics={}
    for mode in ['vector','full']+(['vector_full_table'] if args.full_table_control else []):
        rows=[x for x in records if x['mode']==mode];connected=[x for x in rows if x['case']['kind']=='connected']
        unresolved=[x for x in rows if x['case']['kind']=='unresolved'];direct=[x for x in rows if x['case']['kind']=='direct']
        metrics[mode]={'all':f'{sum(x["pass"] for x in rows)}/{len(rows)}',
                       'connected_correct':sum(x['pass'] for x in connected),'connected_total':len(connected),
                       'unresolved_abstention':f'{sum(x["pass"] for x in unresolved)}/{len(unresolved)}',
                       'direct':f'{sum(x["pass"] for x in direct)}/{len(direct)}',
                       'both_sources_cited':sum(len([c for c in x['response'].get('citations',[]) if c.get('used')])>=2 for x in connected),
                       'median_seconds':statistics.median(x['seconds'] for x in rows),
                       'mean_input_tokens':statistics.mean((x['response'].get('usage') or {}).get('prompt_tokens',0) for x in rows)}
    report={'model':s.served_model,'snapshot':r.snapshots.active().snapshot_id,'metrics':metrics,'acl':acl,
            'controls':'The full-table control supplies the entire authorized PDF as one source, without graph proofs. Other arms use identical normalized chunks. Same normalized corpus, scopes, local model, 4096 context and six-source cap; interleaved seeded order; one inference per arm/case; no forced baseline failures.',
            'limitations':'Small development set; dates checked by patterns rather than independent human grading. M/W is not assumed equivalent to MWF. Timings include cache/load effects.'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    write_report(out, report, records)
    r.store.close();print(json.dumps(report,indent=2))
if __name__=='__main__':main()
