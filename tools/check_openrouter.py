#!/usr/bin/env python3
"""Check the free Liquid model through the real local RAG pipeline; optionally leave chat running.

python tools/check_openrouter.py --out output/openrouter/RUN --serve --port 8101
The key is read from OPENROUTER_API_KEY or a hidden prompt, and is never saved.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
from decimal import Decimal
import getpass
import html
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL = 'liquid/lfm-2.5-2.6b:free'
BASE = 'https://openrouter.ai/api/v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default=MODEL, help='Explicit zero-priced :free model ID')
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--port', type=int, default=8101)
    args = parser.parse_args()
    model_id = args.model
    if not model_id.endswith(':free'):
        raise SystemExit('This test runner only permits an explicit :free model.')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        raise SystemExit('Choose a fresh output directory to preserve the previous check.')
    key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter key (hidden): ')
    if not key:
        raise SystemExit('An OpenRouter key is required.')
    import httpx
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    with httpx.Client(timeout=90, trust_env=False, headers={'Authorization': 'Bearer ' + key}) as client:
        catalog = client.get(BASE + '/models')
        catalog.raise_for_status()
        model = next((m for m in catalog.json()['data'] if m['id'] == model_id), None)
        if not model or not all(Decimal(model['pricing'][k]) == 0 for k in ('prompt', 'completion')):
            raise SystemExit('The requested model is not currently listed with free input/output pricing.')
        quota_response = client.get(BASE + '/key')
        quota_response.raise_for_status()
        before = quota_response.json().get('data', {}).get('free_model_daily_requests')
        print(json.dumps({'model': model_id, 'quota_before': before}), flush=True)
        smoke = []
        for attempt, delay in enumerate((0, 5, 15), 1):
            if delay:
                time.sleep(delay)
            response = client.post(BASE + '/chat/completions', json={
                'model': model_id, 'messages': [{'role': 'user', 'content':
                    'How many r`s are in the word `strawberry?`'}],
                'reasoning': {'enabled': True}, 'max_tokens': 1024,
            })
            body = response.json()
            choice = (body.get('choices') or [{}])[0]
            error = body.get('error') or {}
            metadata = error.get('metadata') or {}
            record = {'attempt': attempt, 'http': response.status_code,
                      'model': body.get('model'), 'answer': choice.get('message', {}).get('content'),
                      'finish_reason': choice.get('finish_reason'), 'usage': body.get('usage'),
                      'error_code': error.get('code'), 'provider': metadata.get('provider_name'),
                      'limit_source': metadata.get('limit_source')}
            smoke.append(record)
            print(json.dumps({'smoke': record}, ensure_ascii=False), flush=True)
            if response.status_code != 429:
                break

    # Configure the same retrieval stack, using a consistent independent copy of the live index.
    from tools.local import configure
    os.environ['ASKIISC_LOCAL_CONFIG'] = str(ROOT / 'config/openrouter.json')
    configure()
    source_data = Path(os.environ['ASKIISC_DATA_DIR'])
    data = out / '.data'
    data.mkdir()
    from tools.compare_graph import clone_index, row_scores
    clone_index(source_data / 'qdrant', data / 'qdrant')
    for name in ('graph.json', 'snapshot.json', 'chunks.jsonl'):
        shutil.copy2(source_data / name, data / name)
    os.environ['ASKIISC_DATA_DIR'] = str(data)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from askiisc_retrieval.service import build_retriever
    from askiisc_retrieval.citations import result_to_dict
    from askiisc_serving.app import create_app
    from askiisc_serving.settings import Settings
    from askiisc_serving.generator import Generator
    from askiisc_serving.retrieval_client import LocalRetrieval, hits_from_result, _trace
    from fastapi.testclient import TestClient

    settings = Settings(vllm_url=BASE, served_model=model_id, model_api_key=key,
                        model_reasoning={'enabled': True}, chat_template_kwargs={}, tokenizer=None,
                        max_model_len=4096, max_output_tokens=1024,
                        regenerate_on_bad_citations=False, public_chat=True,
                        global_concurrency=1, tenant_concurrency=1)
    retriever = build_retriever()

    class RecordingRetrieval(LocalRetrieval):
        async def run(self, question, pinned, mode, programme, academic_year, *, scopes):
            result = await asyncio.to_thread(self.r.retrieve, question, list(scopes), mode == 'full', False)
            self.last = result_to_dict(result)
            return hits_from_result(self.last, now_term='2026-AUG'), _trace(self.last, mode)

    class RecordingGenerator(Generator):
        def prepare(self, *args, **kwargs):
            result = super().prepare(*args, **kwargs)
            self.messages, self.selected, self.budget = result
            return result

    retrieval = RecordingRetrieval(retriever)
    generator = RecordingGenerator(settings)
    questions = json.loads((ROOT / 'config/graph_questions.json').read_text())['questions']
    requested = ['R03', 'R02', 'R08', 'D01', 'R04', 'T03', 'U01', 'R01']
    # During an outage, prove the real app's fallback for both modes without exhausting free quota.
    ids = requested if smoke[-1]['http'] == 200 else ['R03']
    records = []
    failures = 0
    with TestClient(create_app(settings, retrieval=retrieval, generator=generator)) as app:
        assert app.get('/').status_code == 200
        assert app.get('/ui-config').json()['model'] == model_id
        assert app.get('/readyz').status_code == 200
        rng = random.Random(1729)
        for qid in ids:
            question = next(q for q in questions if q['id'] == qid)
            modes = ['vector', 'full']
            rng.shuffle(modes)
            for mode in modes:
                start = time.perf_counter()
                response = app.post('/v1/ask', json={'question': question['question'], 'mode': mode})
                payload = response.json()
                record = {'id': qid, 'mode': mode, 'question': question['question'],
                          'http': response.status_code, 'wall_ms': round((time.perf_counter()-start)*1000, 2),
                          'response': payload, 'selected': generator.selected,
                          'retrieval': retrieval.last, 'messages': generator.messages,
                          'scores': row_scores(question, retrieval.last, generator.selected, payload)}
                records.append(record)
                failures = failures + 1 if payload.get('status') == 'fallback' else 0
                print(json.dumps({'id': qid, 'mode': mode, 'status': payload.get('status'),
                                  'answer': payload.get('answer'), 'reason': payload.get('reason'),
                                  'wall_ms': record['wall_ms']}, ensure_ascii=False), flush=True)
            if failures >= 2:
                break
        app.portal.call(generator.c.aclose)
    retriever.store.close()
    with httpx.Client(timeout=20, trust_env=False, headers={'Authorization': 'Bearer ' + key}) as client:
        response = client.get(BASE + '/key')
        after = response.json().get('data', {}).get('free_model_daily_requests') if response.is_success else None
    results = {'model': model_id, 'endpoint': BASE, 'reasoning': {'enabled': True},
               'context_limit': settings.max_model_len, 'max_output_tokens': settings.max_output_tokens,
               'quota_before': before, 'quota_after': after, 'smoke': smoke, 'records': records,
               'planned_questions': requested, 'completed_cases': len(records),
               'note': 'Small development smoke test; expected-fact matches are not an independent correctness review.'}
    serialized = json.dumps(results, ensure_ascii=False, indent=2)
    assert key not in serialized
    (out / 'results.json').write_text(serialized)
    md = ['# Local OpenRouter check', '', f'Model: `{model_id}`. Reasoning enabled. No paid fallback configured.', '',
          f'Strawberry check HTTP status: {smoke[-1]["http"]}.',
          f'Completed local application cases: {len(records)}.', '',
          '| Question | Mode | Status | Actual answer |', '|---|---|---|---|']
    for record in records:
        answer = (record['response'].get('answer') or record['response'].get('reason') or '').replace('|', '\\|').replace('\n', ' ')
        md.append(f'| {record["question"]} | {record["mode"]} | {record["response"].get("status")} | {answer} |')
    md += ['', 'An upstream rate limit is an availability failure, not an answer-quality result.', '', results['note']]
    (out / 'REPORT.md').write_text('\n'.join(md) + '\n')
    with (out / 'comparison.csv').open('w', newline='') as stream:
        fields = ['id', 'mode', 'question', 'http', 'status', 'answer', 'wall_ms']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({**{k: record[k] for k in ('id', 'mode', 'question', 'http', 'wall_ms')},
                             'status': record['response'].get('status'), 'answer': record['response'].get('answer')})
    print(json.dumps({'report': str(out / 'REPORT.md'), 'quota_after': after,
                      'completed_cases': len(records)}), flush=True)
    if args.serve:
        import uvicorn
        print(f'Chat UI: http://127.0.0.1:{args.port} (key held in process memory only)', flush=True)
        uvicorn.run(create_app(settings), host='127.0.0.1', port=args.port, workers=1, log_level='warning')


if __name__ == '__main__':
    main()
