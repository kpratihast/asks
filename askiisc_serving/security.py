"""Small, bounded HTTP protections independent of retrieval and model instructions."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import posixpath
import re
import time
import unicodedata
from collections import OrderedDict
from pathlib import Path
from urllib.parse import unquote, urlsplit

import anyio

from starlette.responses import JSONResponse


class RequestGuard:
    """Bound work before JSON parsing/authentication; never trust client IP headers here.

    AWS Uvicorn trusts only the loopback Caddy peer, which replaces forwarded headers.
    Per-peer state is bounded; a separate global bucket still applies after eviction.
    Chat's per-principal queue/rate limit remains independently enforced.
    """
    max_body = 32768
    body_timeout = 10
    max_peers = 2048

    def __init__(self, app):
        self.app = app
        self.buckets = OrderedDict()
        self.global_buckets = {}
        self.readers = 0
        html = (Path(__file__).parent / 'static/index.html').read_text()
        hashes = ["'sha256-" + base64.b64encode(hashlib.sha256(s.encode()).digest()).decode() + "'"
                  for s in re.findall(r'<script>(.*?)</script>', html, re.S)]
        self.headers = {
            'content-security-policy': "default-src 'none'; script-src " + ' '.join(hashes) +
                "; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
                "base-uri 'none'; form-action 'self'; frame-ancestors 'none'; object-src 'none'",
            'x-frame-options': 'DENY', 'x-content-type-options': 'nosniff',
            'referrer-policy': 'same-origin', 'cache-control': 'private, no-store',
            'permissions-policy': 'camera=(), microphone=(), geolocation=()',
            'cross-origin-opener-policy': 'same-origin',
            'vary': 'Cookie, Authorization',
        }

    @staticmethod
    def take(table, key, capacity, now):
        tokens, at = table.get(key, (capacity, now))
        tokens = min(capacity, tokens + max(0, now-at)*capacity/60)
        wait = 0 if tokens >= 1 else max(1, math.ceil((1-tokens)*60/capacity))
        table[key] = (tokens-1 if not wait else tokens, now)
        return wait

    def throttle(self, peer, path):
        group = 'otp' if path == '/auth/request-otp' else 'verify' if path == '/auth/verify-otp' else 'http'
        per_peer, global_limit = {'otp': (10, 60), 'verify': (30, 120), 'http': (180, 1200)}[group]
        now = time.monotonic()
        wait = self.take(self.global_buckets, group, global_limit, now)
        if wait:
            return wait
        key = (peer, group)
        wait = self.take(self.buckets, key, per_peer, now)
        self.buckets.move_to_end(key)
        while len(self.buckets) > self.max_peers:
            self.buckets.popitem(last=False)
        return wait

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        async def secured_send(message):
            if message['type'] == 'http.response.start':
                headers = [(k,v) for k,v in message.get('headers', []) if k.lower().decode() not in self.headers]
                headers.extend((k.encode(),v.encode()) for k,v in self.headers.items())
                message = {**message, 'headers': headers}
            await send(message)

        async def reject(status, error, **extra):
            headers = {'Retry-After': str(extra['retry_after_s'])} if 'retry_after_s' in extra else None
            await JSONResponse({'error': error, **extra}, status_code=status, headers=headers)(scope, receive, secured_send)

        raw_headers = scope.get('headers', [])
        headers = {k.lower():v.decode('latin1') for k,v in raw_headers}
        host = headers.get(b'host','')
        if not re.fullmatch(r'(?:[a-zA-Z0-9.-]+|\[[0-9a-fA-F:]+\])(?::[0-9]{1,5})?', host):
            return await reject(400, 'Invalid host header')
        try:
            authority = urlsplit('http://' + host)
            if authority.port == 0:
                return await reject(400, 'Invalid host header')
        except ValueError:
            return await reject(400, 'Invalid host header')
        origin = os.environ.get('AUTH_PUBLIC_ORIGIN')
        if origin and authority.hostname not in {urlsplit(origin).hostname,'127.0.0.1','::1','localhost'}:
            return await reject(400, 'Unrecognized host')
        # Reject ambiguous framing even on development/direct ASGI deployments.
        for name in [b'host', b'content-length', b'content-type', b'authorization', b'origin', b'transfer-encoding']:
            if sum(k.lower()==name for k,v in raw_headers) > 1:
                return await reject(400, 'Duplicate request header')
        if b'content-length' in headers and b'transfer-encoding' in headers:
            return await reject(400, 'Ambiguous request framing')
        path = scope.get('path','/')
        wait = self.throttle((scope.get('client') or ('unknown',0))[0], path)
        if wait:
            return await reject(429, 'Too many requests. Try again later.', retry_after_s=wait)
        if scope['method'] in {'POST','PUT','PATCH','DELETE'}:
            expected = origin or (scope.get('scheme','http')+'://'+host)
            if headers.get(b'sec-fetch-site') == 'cross-site' or (b'origin' in headers and headers[b'origin'] != expected):
                return await reject(403, 'Cross-origin requests are not allowed')
            if headers.get(b'content-encoding','identity').lower() != 'identity':
                return await reject(415, 'Encoded request bodies are not accepted')
            length = headers.get(b'content-length')
            if length is not None:
                if not re.fullmatch(r'[0-9]{1,10}',length):
                    return await reject(400, 'Invalid content length')
                if int(length) > self.max_body:
                    return await reject(413, 'Request body is too large')
            content_type = headers.get(b'content-type','').split(';')[0].strip().lower()
            if content_type != 'application/json' and not (path=='/auth/logout' and length=='0'):
                return await reject(415, 'Use application/json')
            if self.readers >= 64:
                return await reject(429, 'Server is busy. Try again later.', retry_after_s=2)
            body = bytearray()
            self.readers += 1
            try:
                async with asyncio.timeout(self.body_timeout):
                    while True:
                        event = await receive()
                        if event['type'] == 'http.disconnect':
                            return
                        body.extend(event.get('body',b''))
                        if len(body)>self.max_body:
                            return await reject(413, 'Request body is too large')
                        if not event.get('more_body',False):
                            break
            except TimeoutError:
                return await reject(408, 'Request body timed out')
            finally:
                self.readers -= 1
            if length is not None and len(body) != int(length):
                return await reject(400, 'Content length does not match request body')
            if body:
                try:
                    def pairs(items):
                        result={}
                        for k,v in items:
                            if k in result: raise ValueError('duplicate key')
                            result[k]=v
                        return result
                    data=json.loads(body, object_pairs_hook=pairs)
                    def validate(value, depth=0):
                        if depth>12: raise ValueError('nesting')
                        if isinstance(value,str):
                            value.encode('utf-8')  # reject isolated UTF-16 surrogates
                            if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]',value): raise ValueError('control character')
                        elif isinstance(value,dict):
                            for k,v in value.items(): validate(k,depth+1);validate(v,depth+1)
                        elif isinstance(value,list):
                            for v in value: validate(v,depth+1)
                        elif isinstance(value,float) and not math.isfinite(value): raise ValueError('nonfinite')
                    validate(data)
                except (ValueError, RecursionError, UnicodeError):
                    return await reject(400, 'Invalid JSON request')
            sent = False
            async def replay():
                nonlocal sent
                if not sent:
                    sent = True
                    return {'type':'http.request','body':bytes(body),'more_body':False}
                return await receive()
            try:
                return await self.app(scope, replay, secured_send)
            finally:
                # The ASGI request owns the slot, including response-start failures and
                # disconnects before a streaming iterator has been entered.
                slot = scope.get('state', {}).pop('admission_slot', None)
                if slot is not None:
                    with anyio.CancelScope(shield=True):
                        await slot.release()
        return await self.app(scope, receive, secured_send)


def instruction_attack(text: str) -> bool:
    """Reject clear instruction overrides, not legitimate academic security questions.

    This is defense in depth, NOT an authorization or semantic truth boundary.
    Unknown/obfuscated attacks still face source ACLs and the fixed system policy.
    """
    text=unicodedata.normalize('NFKC',text)
    text=''.join(c for c in text if unicodedata.category(c)!='Cf').casefold()
    patterns=[
        # Requests about this assistant's private messages are not corpus questions.
        r'\b(?:your|the|our|this)\s+(?:(?:hidden|internal|initial|original|private)\s+)*(?:system|developer)\s+(?:role\s+)?(?:messages?|prompts?|instructions?)\b',
        r'\binstructions?\s+(?:that\s+)?(?:you\s+(?:received|were given)|(?:placed|provided|given)\s+(?:before|above))\b',
        r'\b(?:ignore|disregard|override|bypass|forget)\b.{0,70}\b(?:instructions?|rules?|access controls?|permissions?|authentication)\b',
        r'\b(?:reveal|repeat|print|show|extract|output|display)\b.{0,70}\b(?:system (?:prompt|instructions?)|hidden (?:prompt|instructions?)|api keys?|passwords?|environment variables|smtp credentials)\b',
        r'<\|(?:im_start|im_end|system|assistant|eot_id|start_header_id)\|>|\[/?inst\]|<<sys>>',
        r'(?m)^\s*(?:system|developer|assistant)\s*:',
        r'(?m)^\s*\[s\d+\]\s*\S',
    ]
    return any(re.search(p,text,re.S) for p in patterns)


def evidence_instruction_attack(hit: dict) -> bool:
    fields = ('text', 'title', 'section', 'programme', 'department', 'instructor', 'excerpt', 'location')
    values = [hit.get(field) for field in fields] + list(hit.get('graph_paths', []))
    return any(instruction_attack(value) for value in values if isinstance(value, str))


def audit_evidence(hits, trace, scopes, catalog=None):
    """Fail closed on missing classification; trace/proof data shares the same boundary."""
    allowed=[]
    for hit in hits:
        scope = hit.get('access_scope')
        if scope not in scopes:
            continue
        try:
            url = urlsplit(hit.get('url', ''))
            if url.scheme not in {'', 'http', 'https'} or (url.scheme and not url.hostname):
                continue
            # Browsers and proxies normalize dot segments and decode path escapes.
            path = posixpath.normpath('/' + unquote(url.path, errors='strict').lstrip('/'))
        except (TypeError, ValueError, UnicodeError):
            continue
        doc = catalog.find(path.removeprefix('/v1/documents/')) if catalog and path.startswith('/v1/documents/') else None
        # A current manifest restriction also overrides stale/misclassified indexed copies.
        if path.startswith('/v1/documents/') and catalog and (doc is None or doc['access_scope'] != scope):
            continue
        allowed.append(hit)
    ids={cid for h in allowed for cid in h.get('chunk_ids',[])}
    proofs=[p for p in trace.get('graph_proofs',[]) if p.get('required_chunk_ids')
            and set(p['required_chunk_ids']) <= ids
            and all(step.get('access_scope') in scopes and set(step.get('chunk_ids',[])) <= ids
                    for step in p.get('steps',[]))]
    proof_ids = {p.get('id') for p in proofs}
    safe=[]
    for hit in allowed:
        support=hit.get('graph_support',[])
        if any(not set(s.get('required_chunk_ids', [])) <= ids
               or (s.get('proof_id') and s['proof_id'] not in proof_ids) for s in support):
            hit={**hit,'graph_support':[],'graph_paths':[]}
        safe.append(hit)
    stripped = len(allowed)!=len(hits) or len(proofs)!=len(trace.get('graph_proofs',[]))
    trace={**trace,'graph_proofs':proofs}
    if stripped:
        trace['policy_trace']={}
        if trace.get('connection_status') == 'complete' and not proofs:
            trace.update(connection_status='missing_path', evidence_sufficient=False)
    # Backend errors can contain URLs, paths or credentials; never return raw diagnostics.
    if trace.get('degraded_reason'): trace['degraded_reason']='retrieval_component_unavailable'
    return safe,trace
