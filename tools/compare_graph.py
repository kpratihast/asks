#!/usr/bin/env python3
"""Reproducible graph ablation on a consistent copy of the local index and the live model.

Run from the project: python tools/compare_graph.py --out output/graph-comparison/RUN
Results checkpoint after every question/arm. --report-only rebuilds reports without inference.
Uses the frozen development pilot. No synthetic documents or changes to the source index.
"""
from __future__ import annotations
import argparse
import collections
import dataclasses
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ARMS = {
    'vector_current': {'label': 'RAG · current', 'mode': 'vector', 'top_k': 6, 'max_evidence': 6},
    'graph_current': {'label': 'Graph RAG · current', 'mode': 'full', 'top_k': 6, 'max_evidence': 6},
    'vector_equal': {'label': 'RAG · nine-block budget', 'mode': 'vector', 'top_k': 9, 'max_evidence': 9},
    'graph_equal': {'label': 'Graph RAG · nine-block budget', 'mode': 'full', 'top_k': 6, 'max_evidence': 9},
}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def evidence_score(groups, ids):
    if not groups: return {'recall': None, 'all': None, 'precision': None}
    ids=set(ids)
    hit=[bool(ids.intersection(g)) for g in groups]
    allowed=set().union(*(set(g) for g in groups))
    return {'recall': sum(hit)/len(hit), 'all': all(hit),
            'precision': len(ids & allowed)/len(ids) if ids else 0.0}

def row_scores(q, raw, selected, response):
    retrieved=evidence_score(q['gold_groups'],[p['chunk_id'] for p in raw['passages']])
    prompt_ids=[i for h in selected for i in h['chunk_ids']]
    prompt=evidence_score(q['gold_groups'],prompt_ids)
    used=[c for c in response.get('citations',[]) if c.get('used')]
    cited=evidence_score(q['gold_groups'],[i for c in used for i in c['chunk_ids']])
    answer=response.get('answer') or ''
    patterns=q['required_fact_patterns']
    fact_hits=[bool(re.search(p,answer,re.I)) for p in patterns]
    answerable=bool(q['gold_groups'])
    expected_pass=(response.get('status')=='answered' and all(fact_hits)) if answerable else response.get('status')=='abstained'
    selected_ids=set(prompt_ids)
    graph_ids={p['chunk_id'] for p in raw['passages'] if p['via']=='graph'}
    return {'retrieved_recall':retrieved['recall'],'retrieved_all':retrieved['all'],
            'prompt_recall':prompt['recall'],'prompt_all':prompt['all'],
            'prompt_precision':prompt['precision'],'cited_all':cited['all'],
            'cited_precision':cited['precision'],
            'fact_coverage':sum(fact_hits)/len(fact_hits) if fact_hits else None,
            'expected_fact_pass':expected_pass,'fact_hits':fact_hits,
            'correct_abstention':response.get('status')=='abstained' if not answerable else None,
            'graph_added':len(graph_ids),'graph_in_prompt':len(graph_ids & selected_ids)}

def clone_index(source, dest):
    """SQLite backup takes a consistent read snapshot while chat owns the original Qdrant lock."""
    if dest.exists(): return
    dest.mkdir(parents=True)
    shutil.copy2(source/'meta.json',dest/'meta.json')
    for src in sorted(source.rglob('storage.sqlite')):
        target=dest/src.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(src.as_uri()+'?mode=ro',uri=True) as reader:
            with sqlite3.connect(target) as writer: reader.backup(writer)


def pct(xs,q):
    xs=sorted(x for x in xs if x is not None)
    return xs[min(len(xs)-1,math.ceil(q*len(xs))-1)] if xs else None

def avg(xs):
    xs=[x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None

def aggregate(rows):
    a=[r for r in rows if r['category']!='unanswerable']
    u=[r for r in rows if r['category']=='unanswerable']
    return {'n':len(rows),'answerable_n':len(a),'unanswerable_n':len(u),
            'retrieved_all':avg([r['scores']['retrieved_all'] for r in a]),
            'retrieved_recall':avg([r['scores']['retrieved_recall'] for r in a]),
            'prompt_all':avg([r['scores']['prompt_all'] for r in a]),
            'prompt_recall':avg([r['scores']['prompt_recall'] for r in a]),
            'prompt_precision':avg([r['scores']['prompt_precision'] for r in a]),
            'expected_fact_pass':avg([r['scores']['expected_fact_pass'] for r in a]),
            'overall_rubric_pass':avg([r['scores']['expected_fact_pass'] for r in rows]),
            'reviewed_accuracy':avg([r.get('review',{}).get('correct') for r in rows]),
            'answerable_reviewed_accuracy':avg([r.get('review',{}).get('correct') for r in a]),
            'supported_answer_rate':avg([r.get('review',{}).get('supported') for r in a]),
            'correct_abstention':avg([r['scores']['correct_abstention'] for r in u]),
            'cited_all':avg([r['scores']['cited_all'] for r in a]),
            'cited_precision':avg([r['scores']['cited_precision'] for r in a]),
            'retrieval_p50_ms':pct([r['response'].get('trace',{}).get('timings_ms',{}).get('retrieval_ms') for r in rows],.5),
            'retrieval_p95_ms':pct([r['response'].get('trace',{}).get('timings_ms',{}).get('retrieval_ms') for r in rows],.95),
            'total_p50_ms':pct([r['wall_ms'] for r in rows],.5),
            'total_p95_ms':pct([r['wall_ms'] for r in rows],.95),
            'prompt_tokens_mean':avg([r['response'].get('usage',{}).get('prompt_tokens') for r in rows]),
            'completion_tokens_mean':avg([r['response'].get('usage',{}).get('completion_tokens') for r in rows]),
            'graph_added_mean':avg([r['scores']['graph_added'] for r in rows]),
            'graph_in_prompt_mean':avg([r['scores']['graph_in_prompt'] for r in rows]),
            'degraded':sum(r['retrieval']['degraded'] for r in rows),
            'errors':sum(r['http']!=200 or r['response'].get('status')=='fallback' for r in rows)}

def paired(rows,first,second,metric='expected_fact_pass'):
    a={r['id']:r for r in rows if r['arm']==first and r['category']!='unanswerable'}
    b={r['id']:r for r in rows if r['arm']==second and r['category']!='unanswerable'}
    ids=sorted(a.keys() & b.keys())
    def value(r):return r.get('review',{}).get('correct') if metric=='reviewed_accuracy' else r['scores'][metric]
    ids=[i for i in ids if value(a[i]) is not None and value(b[i]) is not None]
    diffs=[int(value(b[i]))-int(value(a[i])) for i in ids]
    wins=[i for i,d in zip(ids,diffs) if d>0]; losses=[i for i,d in zip(ids,diffs) if d<0]
    if not ids:return {}
    rng=random.Random(1729)
    boot=sorted(statistics.mean(rng.choices(diffs,k=len(diffs))) for _ in range(5000))
    discordant=len(wins)+len(losses)
    p=min(1,2*sum(math.comb(discordant,k) for k in range(min(len(wins),len(losses))+1))/2**discordant) if discordant else 1
    return {'metric':metric,'n':len(ids),'wins':wins,'losses':losses,'ties':len(ids)-len(wins)-len(losses),
            'delta_pp':100*statistics.mean(diffs),'bootstrap_95_pp':[100*pct(boot,.025),100*pct(boot,.975)],'exact_p':p}

def reports(out,questions):
    import csv
    rows=[json.loads(x) for x in (out/'raw.jsonl').read_text().splitlines() if x.strip()]
    review_path=out/'review.json'
    reviews=json.loads(review_path.read_text()) if review_path.exists() else {}
    for r in rows:
        r['review']=reviews.get(r['id']+'/'+r['arm'],{})
    summaries={arm:aggregate([r for r in rows if r['arm']==arm]) for arm in ARMS}
    cats={cat:{arm:aggregate([r for r in rows if r['arm']==arm and r['category']==cat]) for arm in ARMS}
          for cat in sorted({q['category'] for q in questions['questions']})}
    comparisons={label:{metric:paired(rows,v,g,metric) for metric in ('expected_fact_pass','prompt_all','retrieved_all','reviewed_accuracy')}
                 for label,v,g in [('current','vector_current','graph_current'),('equal_budget','vector_equal','graph_equal')]}
    result={'metadata':json.loads((out/'metadata.json').read_text()),'arms':summaries,'categories':cats,'paired':comparisons}
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    csvrows=[]
    for r in rows:
        resp=r['response'];timings=resp.get('trace',{}).get('timings_ms',{})
        csvrows.append({'id':r['id'],'category':r['category'],'arm':r['arm'],'question':r['question'],
            'status':resp.get('status'),'answer':resp.get('answer'),'expected':r['expected'],**{k:v for k,v in r['scores'].items() if k!='fact_hits'},
            **r['review'],'retrieval_ms':timings.get('retrieval_ms'),'total_ms':r['wall_ms'],
            'graph_paths':' ; '.join(p['graph_path'] or '' for p in r['retrieval']['passages'] if p['via']=='graph')})
    fields=list(dict.fromkeys(k for r in csvrows for k in r))
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(csvrows)
    fmt=lambda x:'—' if x is None else f'{x:.1%}'
    lines=['# Graph RAG comparison — local pilot','',f"Snapshot: `{questions['snapshot_id']}`. Model: `{result['metadata']['model']}`.",'',
        '30 questions: 8 direct, 10 relationship, 8 temporal, 4 unanswerable. Four interleaved arms; same frozen corpus, model, prompt instructions, 4,096 context limit, 512 output tokens, temperature 0, thinking disabled. This is a regression rerun after evidence-selection changes informed by the earlier development pilot. The frozen questions and corpus are unchanged; no changes are made during the rerun.','',
        '| Metric | RAG current | Graph current | RAG nine blocks | Graph nine blocks |','|---|---:|---:|---:|---:|']
    metric_labels=[('retrieved_all','All required evidence retrieved (26 answerable)'),('prompt_all','All required evidence reached prompt (26)'),
        ('expected_fact_pass','Expected-fact rubric pass (26)'),('answerable_reviewed_accuracy','Reviewed answer accuracy (26)'),
        ('correct_abstention','Correct abstention (4)'),('reviewed_accuracy','Reviewed overall accuracy (30)'),
        ('cited_all','All required evidence cited (26)'),('cited_precision','Gold-chunk citation precision (answered cases)'),
        ('prompt_precision','Gold-chunk prompt precision (26)')]
    for key,label in metric_labels:lines.append('| '+label+' | '+' | '.join(fmt(summaries[a][key]) for a in ARMS)+' |')
    for key,label,scale in [('retrieval_p50_ms','Retrieval p50 (ms)',1),('retrieval_p95_ms','Retrieval p95 (ms)',1),('total_p50_ms','End-to-end p50 (s)',1000),('total_p95_ms','End-to-end p95 (s)',1000),('prompt_tokens_mean','Mean actual prompt tokens',1),('graph_added_mean','Mean graph passages retrieved',1),('graph_in_prompt_mean','Mean graph passages reaching prompt',1)]:
        lines.append('| '+label+' | '+' | '.join('—' if summaries[a][key] is None else f'{summaries[a][key]/scale:.2f}' for a in ARMS)+' |')
    lines+=['','## By category','', '| Category | RAG current | Graph current | RAG nine blocks | Graph nine blocks |','|---|---:|---:|---:|---:|']
    for cat,arms in cats.items():
        key='correct_abstention' if cat=='unanswerable' else 'expected_fact_pass'
        lines.append('| '+cat+' (rubric) | '+' | '.join(fmt(arms[a][key]) for a in ARMS)+' |')
    lines+=['','## Paired changes','']
    for name,metrics in comparisons.items():
        for key,p in metrics.items():
            if p:lines.append(f"- {name}, {key}: {p['delta_pp']:+.1f} percentage points; wins {len(p['wins'])}, losses {len(p['losses'])}, ties {p['ties']}; paired bootstrap 95% interval {p['bootstrap_95_pp'][0]:+.1f} to {p['bootstrap_95_pp'][1]:+.1f} pp; exact paired p={p['exact_p']:.4f}.")
    lines+=['','## Interpretation and limits','',
        '- The current arms reproduce the local top-k=6 / max-evidence=6 configuration. Graph mode may retrieve three extras, but the prompt can drop them. The equal-budget arms both allow at most nine evidence blocks within the same context limit; vector retrieval gets nine candidates and graph retrieval gets six plus up to three graph additions. Actual token usage and evidence reaching the prompt are reported.',
        '- RAG baseline retains course-code and semester handling. This is an ablation of graph expansion, not a comparison with pure dense search.',
        '- Gold groups describe the required facts, allowing alternative source chunks. All-required-evidence recall requires every group. These are strict chunk-level labels, not exhaustive relevance judgments; precision is a conservative gold-match proxy.',
        '- The automatic expected-fact rubric checks required answer patterns plus answered status, or abstention for unavailable facts. It is not semantic accuracy. review.json contains explicit per-answer review decisions against the answer and selected source text; marker validity alone does not establish factual support.',
        '- An exploratory pilot, with existing/reworded development examples and related questions, not an untouched held-out study. Correlated cases and the small sample limit generalization. Bootstrap intervals are descriptive question-level estimates.',
        '- One measured inference per arm/question after a warmup; arms are interleaved in seeded order. Latencies include the in-process production serving pipeline against a consistent backup of the actual Qdrant index and the live MLX endpoint. Other local workloads and model prompt caching affect latency.',
        '- The model, stored graph, questions and source index are unchanged. Retrieval and prompt selection use the implementation archived with this run. See metadata.json for hashes and raw.jsonl for complete retrieved evidence, selected prompt evidence, prompts, graph paths, model outputs, token usage and timings.','', '## Every answer','']
    for q in questions['questions']:
        lines += [f"### {q['id']} · {q['question']}",'',f"Expected: {q['expected_answer']}",'']
        for arm in ARMS:
            r=next((r for r in rows if r['id']==q['id'] and r['arm']==arm),None)
            if not r:continue
            lines += [f"**{ARMS[arm]['label']}** — {r['response'].get('status')}; rubric {'pass' if r['scores']['expected_fact_pass'] else 'fail'}; {r['wall_ms']/1000:.2f}s",'',str(r['response'].get('answer') or r['response'].get('detail') or '(no answer)'), '']
            if r['review']:lines += ['Review: '+r['review'].get('note',''),'']
    (out/'REPORT.md').write_text('\n'.join(lines))
    # Offline, dependency-free report: real outputs with expandable evidence and source links.
    esc=lambda s:html.escape(str(s))
    table='<table><thead><tr><th>Metric</th>'+''.join('<th>'+esc(v['label'])+'</th>' for v in ARMS.values())+'</tr></thead><tbody>'
    for key,label in metric_labels:table+='<tr><td>'+esc(label)+'</td>'+''.join('<td>'+fmt(summaries[a][key])+'</td>' for a in ARMS)+'</tr>'
    for key,label,scale in [('retrieval_p50_ms','Retrieval p50 (ms)',1),('total_p50_ms','Total p50 (s)',1000),('total_p95_ms','Total p95 (s)',1000)]:
        table+='<tr><td>'+label+'</td>'+''.join('<td>'+('—' if summaries[a][key] is None else f'{summaries[a][key]/scale:.2f}')+'</td>' for a in ARMS)+'</tr>'
    table+='</tbody></table>'
    cards=[]
    for q in questions['questions']:
        body=f'<section data-category="{esc(q["category"])}"><h2>{esc(q["id"])} · {esc(q["question"])}</h2><p class="expected">Expected: {esc(q["expected_answer"])}</p><div class="grid">'
        for arm in ARMS:
            r=next((r for r in rows if r['id']==q['id'] and r['arm']==arm),None)
            if not r:continue
            s=r['scores'];resp=r['response']
            body+=f'<article><h3>{esc(ARMS[arm]["label"])}</h3><p class="tag">{esc(resp.get("status"))} · rubric {"PASS" if s["expected_fact_pass"] else "FAIL"} · {r["wall_ms"]/1000:.2f}s</p><p class="answer">{esc(resp.get("answer") or resp.get("detail") or "No answer")}</p>'
            if r['review']:body+='<p class="review">Review: '+esc(r['review'].get('note',''))+'</p>'
            body+=f'<p>Evidence: {fmt(s["retrieved_recall"])} retrieved · {fmt(s["prompt_recall"])} in prompt<br>Graph passages: {s["graph_added"]} retrieved / {s["graph_in_prompt"]} in prompt</p><details><summary>Prompt evidence and graph paths</summary>'
            for h in r['selected']:
                paths=[p['graph_path'] for p in r['retrieval']['passages'] if p['chunk_id'] in h['chunk_ids'] and p['graph_path']]
                body+=f'<h4>[{esc(h.get("prompt_marker"))}] {esc(h["section"])} · {esc(h["via"])}</h4><p>'+esc(h['text'])+'</p>'
                if h['url'].startswith(('http://','https://')):body+=f'<a href="{esc(h["url"])}">Source page</a>'
                for p in paths:body+='<pre>'+esc(p)+'</pre>'
            body+='</details></article>'
        cards.append(body+'</div></section>')
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ask IISc · Graph RAG comparison</title><style>body{font:15px/1.55 system-ui;background:#f4f6f3;color:#18302a;margin:0}main{max-width:1450px;margin:auto;padding:32px}h1{font-size:32px}h2{font-size:20px}h3{font-size:16px}.sub,.expected{color:#52655e}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:11px;border-bottom:1px solid #dce5df}th{background:#dce9df}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}article{background:white;padding:18px;border:1px solid #dce5df;border-radius:12px;overflow-wrap:anywhere}.answer{white-space:pre-wrap}.tag{font-size:12px;color:#205a45}.review{padding:8px;background:#edf3ed;font-size:13px}section{margin-top:32px}summary{cursor:pointer;font-weight:600}pre{font-size:12px;white-space:pre-wrap}select{padding:10px;margin:20px 0}details{font-size:13px}header{padding-bottom:20px}@media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){main{padding:15px}.grid{grid-template-columns:1fr}table{font-size:11px}th,td{padding:6px}}</style><main><header><h1>Graph RAG comparison</h1>'''
    doc+=f'<p class="sub">30-question local pilot · {esc(result["metadata"]["model"])} · snapshot {esc(questions["snapshot_id"])}</p><p>Same documents, model, prompt instructions and context limit. Current and equal-budget ablations. Expand each answer to inspect the actual evidence sent to the model.</p></header>'+table
    doc+='<p><strong>Reading these numbers:</strong> automatic expected-fact checks are distinct from reviewed answer accuracy. This pilot includes development examples and related questions; it is not a held-out benchmark. Gold-match precision is a conservative proxy. Latency is one measured run per question/arm.</p><label>Show questions: <select id="filter"><option value="all">All</option>'+''.join(f'<option value="{c}">{c}</option>' for c in cats)+'</select></label>'+''.join(cards)
    doc+='<script>document.getElementById("filter").addEventListener("change",e=>document.querySelectorAll("section").forEach(s=>s.hidden=e.target.value!=="all"&&s.dataset.category!==e.target.value));</script></main></html>'
    (out/'comparison.html').write_text(doc)
    print(json.dumps({'arms':summaries,'paired':comparisons},indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--questions',type=Path,default=ROOT/'config/graph_questions.json')
    ap.add_argument('--report-only',action='store_true')
    a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    questions=json.loads(a.questions.read_text())
    if a.report_only:reports(out,questions);return
    from tools.local import configure
    configure();os.environ.setdefault('HF_HUB_OFFLINE','1')
    from askiisc_retrieval import config
    from askiisc_retrieval.cli import make_embedder
    from askiisc_retrieval.stores import QdrantStore,JsonGraphStore
    from askiisc_retrieval.snapshot import JsonSnapshotProvider
    from askiisc_retrieval.retriever import Retriever
    from askiisc_retrieval.citations import result_to_dict
    from askiisc_serving.settings import Settings
    from askiisc_serving.generator import Generator
    from askiisc_serving.retrieval_client import LocalRetrieval
    from askiisc_serving.app import create_app
    from fastapi.testclient import TestClient
    import httpx
    snapshot=json.loads(config.SNAPSHOT_PATH.read_text())
    if snapshot['snapshot_id']!=questions['snapshot_id']:raise SystemExit('Question set snapshot differs from current index')
    settings=Settings(public_chat=True,rate_per_min=100000,global_concurrency=1)
    with httpx.Client(timeout=10,trust_env=False) as client:
        models=client.get(settings.vllm_url+'/v1/models');models.raise_for_status()
        ids=[m['id'] for m in models.json()['data']]
        if settings.served_model not in ids:raise SystemExit(f'Configured model not served: {settings.served_model}')
    metadata={'model':settings.served_model,'model_url':settings.vllm_url,'snapshot':snapshot,
        'questions_sha256':sha(a.questions),'chunks_sha256':sha(config.CHUNKS_PATH),'graph_sha256':sha(config.GRAPH_PATH),
        'snapshot_sha256':sha(config.SNAPSHOT_PATH),'context_limit':settings.max_model_len,'max_output_tokens':settings.max_output_tokens,
        'chat_template_kwargs':settings.chat_template_kwargs,'temperature':0,'reranker':None,'arms':ARMS,
        'seed':1729,'started_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'runner_sha256':sha(__file__),
        'implementation_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(
            list((ROOT/'askiisc_retrieval').glob('*.py'))+list((ROOT/'askiisc_serving').glob('*.py')))},
        'evaluation_note':'Development regression set; implementation informed by the previous pilot, not held-out evidence.'}
    mp=out/'metadata.json'
    if mp.exists():
        old=json.loads(mp.read_text())
        for k in ('model','questions_sha256','chunks_sha256','graph_sha256','context_limit','max_output_tokens','runner_sha256','implementation_sha256'):
            if old.get(k)!=metadata[k]:raise SystemExit(f'Cannot resume: {k} changed')
    else:mp.write_text(json.dumps(metadata,indent=2))
    for rel in metadata['implementation_sha256']:
        archived=out/'implementation'/rel
        if not archived.exists():
            archived.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/rel,archived)
    index=out/'.index';clone_index(config.QDRANT_PATH,index)
    store=QdrantStore(config.COLLECTION,config.EMBED_DIM,path=index)
    if store.count()!=sum(1 for x in config.CHUNKS_PATH.read_text().splitlines() if x.strip()):raise SystemExit('Index/chunks count mismatch')
    chunks={c['chunk_id']:c for c in [json.loads(x) for x in config.CHUNKS_PATH.read_text().splitlines()]}
    for q in questions['questions']:
        for group in q['gold_groups']:
            if not group or not all(i in chunks for i in group):raise SystemExit(f'Invalid gold: {q["id"]}')
    retriever=Retriever(make_embedder(),store,JsonSnapshotProvider(config.SNAPSHOT_PATH),JsonGraphStore.load(config.GRAPH_PATH),
        reranker=None,rerank_min=None,today_term='2026-AUG')
    class RecordingRetrieval(LocalRetrieval):
        def __init__(self,r):super().__init__(r);self.last=None
        async def run(self,question,pinned,mode,programme,academic_year_,*,scopes):
            import asyncio
            from askiisc_serving.retrieval_client import hits_from_result,_trace
            res=await asyncio.to_thread(self.r.retrieve,question,list(scopes),mode=='full',False)
            self.last=result_to_dict(res)
            if res.snapshot_id!=questions['snapshot_id']:raise RuntimeError('Snapshot changed during run')
            return hits_from_result(self.last,now_term='2026-AUG'),_trace(self.last,mode)
    class RecordingGenerator(Generator):
        def prepare(self,*args,**kwargs):
            result=super().prepare(*args,**kwargs)
            self.last_messages,self.last_selected,self.last_budget=result
            return result
    retrieval=RecordingRetrieval(retriever);gen=RecordingGenerator(settings)
    done=set()
    rawfile=out/'raw.jsonl'
    if rawfile.exists():done={(r['id'],r['arm']) for r in [json.loads(x) for x in rawfile.read_text().splitlines() if x.strip()]}
    try:
        with TestClient(create_app(settings,retrieval=retrieval,generator=gen)) as client:
            if not client.get('/readyz').is_success:raise RuntimeError('Model readiness failed')
            # Warm both CPU embeddings and the live generator; do not include warmup in metrics.
            if not done:
                warm=client.post('/v1/ask',json={'question':'Who teaches DS 252 in Aug-Dec 2026?','mode':'vector'})
                if warm.status_code!=200 or warm.json().get('status')=='fallback':raise RuntimeError(f'Warmup failed: {warm.text[:500]}')
                print('Warmup complete',flush=True)
            rng=random.Random(1729)
            with rawfile.open('a') as f:
                for q in questions['questions']:
                    arms=list(ARMS);rng.shuffle(arms)
                    for arm in arms:
                        if (q['id'],arm) in done:continue
                        spec=ARMS[arm];retriever.top_k=spec['top_k'];settings.max_evidence=spec['max_evidence']
                        gen.last_messages=[];gen.last_selected=[];gen.last_budget={}
                        start=time.perf_counter()
                        resp=client.post('/v1/ask',json={'question':q['question'],'mode':spec['mode'],'pinned_snapshot':questions['snapshot_id']})
                        wall=round((time.perf_counter()-start)*1000,2);d=resp.json()
                        if resp.status_code!=200:raise RuntimeError(f'{q["id"]}/{arm} HTTP {resp.status_code}: {d}')
                        if d.get('trace',{}).get('model')!=metadata['model']:raise RuntimeError('Model identity changed')
                        raw=retrieval.last;sel=gen.last_selected
                        scores=row_scores(q,raw,sel,d)
                        rec={'id':q['id'],'category':q['category'],'arm':arm,'question':q['question'],'expected':q['expected_answer'],
                            'http':resp.status_code,'wall_ms':wall,'scores':scores,'response':d,'retrieval':raw,
                            'selected':sel,'messages':gen.last_messages,'gold_groups':q['gold_groups']}
                        f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
                        print(f'{q["id"]} {arm:16} {d.get("status"):10} fact={int(scores["expected_fact_pass"])} evidence={scores["prompt_all"]} graph={scores["graph_in_prompt"]}/{scores["graph_added"]} {wall/1000:6.2f}s {(d.get("answer") or d.get("detail") or "")[:110]}',flush=True)
                        if d.get('status')=='fallback':print('FALLBACK DETAIL '+json.dumps(d),flush=True)
            client.portal.call(gen.c.aclose)
        if sha(config.SNAPSHOT_PATH)!=metadata['snapshot_sha256'] or sha(config.CHUNKS_PATH)!=metadata['chunks_sha256'] or sha(config.GRAPH_PATH)!=metadata['graph_sha256']:raise RuntimeError('Corpus changed during benchmark')
    finally:
        import asyncio
        if not gen.c.is_closed:
            try:asyncio.run(gen.c.aclose())
            except RuntimeError:pass
        store.close()
    reports(out,questions)

if __name__=='__main__':main()
