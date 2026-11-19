"""#55: the pre-authentication request guard is documented, reports a reason, is logged and configurable."""
import asyncio
import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from askiisc_serving.metrics import Metrics
from askiisc_serving.security import RequestGuard, guard_limits
from askiisc_serving.settings import Settings
from test_redteam_fixes import ADMIN, _app, _tenant_keys

ROOT = Path(__file__).resolve().parents[1]


def _hit(guard, n, path='/healthz', peer='10.0.0.1', app=None):
    """Send n GET requests through the guard; return the (status, body, headers) of each."""
    async def run():
        out = []
        async def endpoint(scope, receive, send):
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'{}'})
        async def receive():
            return {'type': 'http.request', 'body': b'', 'more_body': False}
        guard.app = endpoint
        for _ in range(n):
            sent = []
            async def send(message):
                sent.append(message)
            scope = {'type': 'http', 'method': 'GET', 'path': path, 'scheme': 'http',
                     'headers': [(b'host', b'testserver')], 'client': (peer, 1)}
            if app is not None:
                scope['app'] = app
            await guard(scope, receive, send)
            body = b''.join(m.get('body', b'') for m in sent if m['type'] == 'http.response.body')
            out.append((sent[0]['status'], json.loads(body or b'{}'), dict(sent[0]['headers'])))
        return out
    return asyncio.run(run())


def test_default_limits_are_the_documented_ones():
    limits = guard_limits()
    assert limits['http']['per_address_per_min'] == 180 and limits['http']['all_addresses_per_min'] == 1200


def test_guard_429_carries_a_reason_and_the_limiter(monkeypatch):
    # Cloudhack's probe: 180 x 200 from one address, then 429 with no reason.
    monkeypatch.delenv('GUARD_PER_ADDRESS_PER_MIN', raising=False)
    monkeypatch.delenv('GUARD_ALL_ADDRESSES_PER_MIN', raising=False)
    results = _hit(RequestGuard(None), 181)
    assert [s for s, _, _ in results[:180]] == [200] * 180
    status, body, headers = results[180]
    assert status == 429 and body['reason'] == 'address_rate_limited' and body['limiter'] == 'request_guard'
    assert body['retry_after_s'] >= 1 and headers[b'retry-after'] == str(body['retry_after_s']).encode()


def test_shared_bucket_rejection_names_its_own_reason(monkeypatch):
    monkeypatch.setenv('GUARD_ALL_ADDRESSES_PER_MIN', '3')
    guard = RequestGuard(None)
    results = [_hit(guard, 1, peer=f'10.0.0.{i}')[0] for i in range(4)]
    assert [s for s, _, _ in results] == [200, 200, 200, 429]
    assert results[3][1]['reason'] == 'all_addresses_rate_limited'


def test_limits_are_configurable_for_controlled_load_tests(monkeypatch):
    monkeypatch.setenv('GUARD_PER_ADDRESS_PER_MIN', '5')
    results = _hit(RequestGuard(None), 6)
    assert [s for s, _, _ in results] == [200] * 5 + [429]
    monkeypatch.setenv('GUARD_PER_ADDRESS_PER_MIN', '6000')
    assert [s for s, _, _ in _hit(RequestGuard(None), 300)] == [200] * 300


@pytest.mark.parametrize('value', ['0', '-1', 'abc', '1.5'])
def test_invalid_limit_is_rejected_at_startup(monkeypatch, value):
    monkeypatch.setenv('GUARD_PER_ADDRESS_PER_MIN', value)
    with pytest.raises(ValueError, match='GUARD_PER_ADDRESS_PER_MIN'):
        guard_limits()


def test_guard_rejections_are_in_the_request_log_and_metrics(monkeypatch, caplog):
    # Cloudhack: the structured log had 492 rejected:rate_limited lines against 1,405 client 429s.
    monkeypatch.setenv('GUARD_PER_ADDRESS_PER_MIN', '2')
    app = type('App', (), {})()
    app.state = type('State', (), {'metrics': Metrics()})()
    with caplog.at_level(logging.INFO, logger='askiisc.requests'):
        results = _hit(RequestGuard(None), 4, app=app)
    assert [s for s, _, _ in results] == [200, 200, 429, 429]
    logged = [json.loads(r.getMessage()) for r in caplog.records if r.name == 'askiisc.requests']
    assert [r['status'] for r in logged] == ['rejected:address_rate_limited'] * 2 and all(r['http'] == 429 for r in logged)
    assert app.state.metrics.requests[('-', 'rejected:address_rate_limited')] == 2
    assert 'status="rejected:address_rate_limited"} 2' in app.state.metrics.prometheus()


def test_operators_see_the_guard_limits_in_readyz(monkeypatch):
    monkeypatch.delenv('GUARD_PER_ADDRESS_PER_MIN', raising=False)
    s = Settings(email_auth_enabled=False, public_chat=False, served_model='test-model')
    with TestClient(_app(s, keys=_tenant_keys())) as c:
        full = c.get('/readyz', headers={'Authorization': f'Bearer {ADMIN}'}).json()
        assert full['request_guard']['http'] == guard_limits()['http']
        assert full['admission']['limits']['rate_per_min'] == s.rate_per_min
        assert 'request_guard' not in c.get('/readyz').json()          # anonymous callers: liveness only


def test_load_testing_doc_describes_the_guard():
    doc = (ROOT / 'LOAD_TESTING.md').read_text(encoding='utf-8')
    for text in ('address_rate_limited', 'all_addresses_rate_limited', 'request_bodies_busy',
                 'GUARD_PER_ADDRESS_PER_MIN', 'GUARD_ALL_ADDRESSES_PER_MIN', '180', '1,200'):
        assert text in doc, text
