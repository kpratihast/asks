"""Benchmark validity: complete evidence, actual prompt inclusion, and paired result accounting."""
import importlib.util
from pathlib import Path

p=Path(__file__).resolve().parents[1]/'tools/compare_graph.py'
spec=importlib.util.spec_from_file_location('graph_comparison',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_every_gold_group_is_required_and_alternatives_are_allowed():
    assert m.evidence_score([['a','a2'],['b']],['a2'])['all'] is False
    assert m.evidence_score([['a','a2'],['b']],['a2'])['recall']==.5
    assert m.evidence_score([['a','a2'],['b']],['a2','b'])['all'] is True

def test_retrieval_success_does_not_imply_evidence_reached_model():
    q={'gold_groups':[['a'],['b']],'required_fact_patterns':['answer']}
    raw={'passages':[{'chunk_id':'a','via':'vector'},{'chunk_id':'b','via':'graph'}]}
    selected=[{'chunk_ids':['a']}]
    d={'status':'answered','answer':'answer [S1]','citations':[{'used':True,'chunk_ids':['a']}]}
    s=m.row_scores(q,raw,selected,d)
    assert s['retrieved_all'] is True and s['prompt_all'] is False
    assert s['graph_added']==1 and s['graph_in_prompt']==0
    assert s['cited_all'] is False

def test_fallback_is_not_correct_abstention():
    q={'gold_groups':[],'required_fact_patterns':[]}
    s=m.row_scores(q,{'passages':[]},[],{'status':'fallback','answer':None})
    assert s['correct_abstention'] is False and s['expected_fact_pass'] is False

def test_paired_comparison_counts_wins_losses_and_ties():
    rows=[]
    for i,(a,b) in enumerate([(0,1),(1,0),(1,1),(0,1)]):
        for arm,v in [('v',a),('g',b)]:
            rows.append({'id':str(i),'arm':arm,'category':'relationship','scores':{'expected_fact_pass':bool(v)}})
    r=m.paired(rows,'v','g')
    assert r['wins']==['0','3'] and r['losses']==['1'] and r['ties']==1
    assert r['delta_pp']==25 and r['exact_p']==1

def test_nearest_rank_percentiles():
    assert m.pct(list(range(1,101)),.95)==95
