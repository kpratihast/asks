"""Integration boundaries: public scope, served-model identity, follow-ups, broken streams."""
import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from askiisc_serving.app import create_app
from askiisc_serving.generator import GenerationUnavailable, Generator
from askiisc_serving.settings import Settings
from test_serving import FakeModel, FakeRetrieval, make, ask


def test_public_browser_is_guest_and_cannot_bypass_retrieval():
    retrieval = FakeRetrieval()
    model = FakeModel('Course description [S1].')
    settings = Settings(public_chat=True, served_model='test-model')
    generator = Generator(settings, client=httpx.AsyncClient(transport=httpx.MockTransport(model.handler)))
    with TestClient(create_app(settings, retrieval=retrieval, generator=generator)) as client:
        assert client.get('/').status_code == 200
        assert client.get('/ui-config').json()['public_chat'] is True
        result = client.post('/v1/ask', json={'question': 'Course?'})
        assert result.json()['status'] == 'answered'
        assert retrieval.calls[0]['scopes'] == ['public']
        assert client.post('/v1/ask', json={'question': 'Course?', 'mode': 'none'}).status_code == 403
        assert client.post('/v1/ask', json={'question': 'Course?'}, headers={'Authorization': 'Bearer invalid'}).status_code == 401
        assert client.post('/v1/ask', json={'question': '   '}).status_code == 422


def test_discovery_uses_alias_and_actual_context_and_health_checks_identity():
    async def check():
        def handler(request):
            return httpx.Response(200, json={'data': [{'id': 'glance-vision', 'max_model_len': 4096}]})
        s = Settings(served_model='auto', max_model_len=8192, vllm_url='http://model/v1/')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            gen = Generator(s, client=client)
            await gen.discover()
            assert s.served_model == 'glance-vision' and s.max_model_len == 4096
            assert s.vllm_url == 'http://model'
            assert await gen.ready()
            s.served_model = 'not-loaded'
            assert not await gen.ready()
    asyncio.run(check())


def test_ambiguous_model_discovery_fails_with_configuration_hint():
    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={
            'data': [{'id': 'one'}, {'id': 'two'}]}))) as client:
            with pytest.raises(GenerationUnavailable, match='SERVED_MODEL'):
                await Generator(Settings(served_model='auto', vllm_url='http://model'), client=client).discover()
    asyncio.run(check())


def test_followup_resolves_references_then_retrieves_with_original_access_scope():
    c, retrieval = make(FakeModel('Who teaches DS 252?', 'Yogesh Simmhan teaches DS 252 [S1].'))
    with c:
        result = ask(c, q='Who teaches it?', history=[
            {'role': 'user', 'content': 'What is DS 252?'},
            {'role': 'assistant', 'content': 'A course on cloud computing.'}]).json()
    assert result['status'] == 'answered'
    assert retrieval.calls[0]['q'] == 'Who teaches DS 252?'
    assert retrieval.calls[0]['scopes'] == ['public']


def test_premature_model_stream_cannot_be_treated_as_a_finished_answer():
    async def check():
        body = 'data: ' + json.dumps({'choices': [{'delta': {'content': 'Incomplete claim [S1]'}}]}) + '\n\n'
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=body))) as client:
            gen = Generator(Settings(vllm_url='http://model'), client=client)
            with pytest.raises(GenerationUnavailable, match='ended'):
                await gen.complete([{'role': 'user', 'content': 'q'}])
    asyncio.run(check())
