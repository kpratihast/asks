"""#48: SERVED_MODEL=auto with the model endpoint down must not stop the gateway; the model is discovered
on first use once it is reachable."""
import asyncio

import httpx
import pytest

from askiisc_serving.generator import GenerationUnavailable, Generator
from askiisc_serving.settings import Settings

ANSWER = ('data: {"choices":[{"delta":{"content":"ok [S1]"},"finish_reason":null}]}\n\n'
          'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')


class Endpoint:
    def __init__(self):
        self.up = False

    def handler(self, request):
        if not self.up:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "llama-3.1-8b", "max_model_len": 8192}]})
        return httpx.Response(200, content=ANSWER.encode(), headers={"content-type": "text/event-stream"})


def test_auto_model_is_discovered_on_first_use_after_an_outage():
    endpoint = Endpoint()
    g = Generator(Settings(served_model="auto", vllm_url="http://model"),
                  client=httpx.AsyncClient(transport=httpx.MockTransport(endpoint.handler)))

    async def run():
        assert await g.ready() is False                                   # down: not ready, no exception
        with pytest.raises(GenerationUnavailable):                        # answers take the excerpt fallback
            await g.complete([{"role": "user", "content": "hi"}])
        endpoint.up = True
        assert await g.ready() is True and g.s.served_model == "llama-3.1-8b"
        assert (await g.complete([{"role": "user", "content": "hi"}]))[0] == "ok [S1]"

    asyncio.run(run())
