"""Hosted model authentication, reasoning options and safe streaming failure handling."""
import asyncio
import json

import httpx
import pytest

from askiisc_serving.generator import GenerationUnavailable, Generator
from askiisc_serving.settings import Settings
from test_serving import make, ask, sse_body

MODEL = 'liquid/lfm-2.5-2.6b:free'
SECRET = 'test-hosted-secret-value'


def test_hosted_auth_reasoning_catalog_and_stream():
    async def check():
        requests = []
        def handler(request):
            requests.append(request)
            assert request.headers['authorization'] == 'Bearer ' + SECRET
            if request.url.path == '/api/v1/models':
                return httpx.Response(200, json={'data': [{'id': MODEL, 'context_length': 4096}]})
            assert request.url.path == '/api/v1/chat/completions'
            body = json.loads(request.content)
            assert body['model'] == MODEL and body['reasoning'] == {'enabled': True}
            assert 'chat_template_kwargs' not in body
            return httpx.Response(200, content=sse_body('Yogesh Simmhan [S1].'))
        settings = Settings(vllm_url='https://openrouter.ai/api/v1/', served_model=MODEL,
                            model_api_key=SECRET, model_reasoning={'enabled': True})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = Generator(settings, client=client)
            await generator.discover()
            assert settings.max_model_len == 4096
            assert await generator.ready()
            answer, usage = await generator.complete([{'role': 'user', 'content': 'Question?'}])
            assert answer == 'Yogesh Simmhan [S1].' and usage['finish_reason'] == 'stop'
        assert len(requests) == 3 and SECRET not in repr(settings)
    asyncio.run(check())


def test_upstream_rate_limit_returns_excerpt_fallback_and_redacts_key():
    class Limited:
        def handler(self, request):
            return httpx.Response(429, json={'error': {'message': 'Rate limited ' + SECRET}})
    client, _ = make(Limited(), model_api_key=SECRET)
    with client:
        result = ask(client).json()
    assert result['status'] == 'fallback' and result['excerpts']
    assert result['detail'] == 'The model is temporarily unavailable.' and SECRET not in json.dumps(result)


@pytest.mark.parametrize('event', [
    {'error': {'code': 429}, 'choices': [{'delta': {}, 'finish_reason': 'error'}]},
    {'choices': [{'delta': {}, 'finish_reason': 'error'}]},
])
def test_midstream_error_cannot_become_a_successful_answer(event):
    async def check():
        body = ('data: ' + json.dumps({'choices': [{'delta': {'content': 'Partial [S1].'}}]}) +
                '\n\ndata: ' + json.dumps(event) + '\n\n')
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=body))) as client:
            with pytest.raises(GenerationUnavailable, match='error'):
                await Generator(Settings(), client=client).complete([{'role': 'user', 'content': 'Q?'}])
    asyncio.run(check())


def test_reasoning_requires_an_object():
    with pytest.raises(ValueError, match='MODEL_REASONING'):
        Settings(model_reasoning=[])


def test_local_requests_do_not_add_hosted_options_or_auth():
    async def check():
        def handler(request):
            assert 'authorization' not in request.headers
            assert 'reasoning' not in json.loads(request.content)
            assert 'provider' not in json.loads(request.content)
            return httpx.Response(200, content=sse_body('Answer [S1].'))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = Generator(Settings(model_api_key=None, model_reasoning={}), client=client)
            assert (await generator.complete([{'role': 'user', 'content': 'Q?'}]))[0] == 'Answer [S1].'
    asyncio.run(check())


def test_openrouter_privacy_and_price_constraints_are_sent_unchanged():
    options = {'data_collection': 'deny', 'zdr': True,
               'max_price': {'prompt': 0, 'completion': 0, 'request': 0}}
    async def check():
        def handler(request):
            assert json.loads(request.content)['provider'] == options
            return httpx.Response(200, content=sse_body('Answer [S1].'))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = Generator(Settings(model_provider_preferences=options), client=client)
            await generator.complete([{'role': 'user', 'content': 'Q?'}])
    asyncio.run(check())
    with pytest.raises(ValueError, match='MODEL_PROVIDER_PREFERENCES'):
        Settings(model_provider_preferences=[])
