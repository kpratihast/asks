#!/usr/bin/env python3
"""Manage the private AWS model configuration. API keys use a hidden prompt, never CLI arguments."""
from __future__ import annotations
import argparse, getpass, json, os, tempfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
from release_lib import runtime_config

OPENROUTER = 'https://openrouter.ai/api/v1'
MODEL = 'xingchen-agi/xing4.0-29b-a4b:free'
FREE_PRIVATE = {'data_collection': 'deny', 'zdr': True,
                'max_price': {'prompt': 0, 'completion': 0, 'request': 0}}
SMTP_KEYS = {'SMTP_HOST', 'SMTP_PORT', 'SMTP_USERNAME', 'SMTP_PASSWORD', 'SMTP_FROM'}


def configure(path, *, url=None, model=None, key=None, email_path=None):
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError('Refusing a symlink for the private configuration')
    raw = json.loads(path.read_text()) if path.exists() else {
        'VLLM_URL': OPENROUTER, 'SERVED_MODEL': MODEL, 'MODEL_API_KEY': '',
        'MODEL_PROVIDER_PREFERENCES': FREE_PRIVATE, 'MODEL_REASONING': {'enabled': False}, 'EMAIL_AUTH_ENABLED': True,
        'AUTH_PROFESSOR_TEST_EMAILS': '',
    }
    if not path.exists():
        email = Path(email_path or Path.home()/'.config/askiisc/email.json').expanduser()
        if email.is_file():
            smtp = json.loads(email.read_text())
            if not isinstance(smtp, dict) or set(smtp)-SMTP_KEYS:
                raise ValueError('Invalid private SMTP configuration')
            raw.update(smtp)
    if url is not None:
        endpoint = url.rstrip('/')
        if endpoint != raw.get('VLLM_URL', '').rstrip('/'):
            # A credential for one provider must never be sent to a replacement endpoint.
            raw.update(MODEL_API_KEY='', MODEL_PROVIDER_PREFERENCES={}, MODEL_REASONING={}, CHAT_TEMPLATE_KWARGS={})
        raw['VLLM_URL'] = endpoint
        if endpoint == OPENROUTER:
            raw['MODEL_PROVIDER_PREFERENCES'] = FREE_PRIVATE
    if model is not None:
        raw['SERVED_MODEL'] = model
    if key is not None:
        if not key.strip():
            raise ValueError('API key cannot be empty')
        raw['MODEL_API_KEY'] = key.strip()
    config = runtime_config(raw)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix='.aws-runtime-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(config, handle, indent=2); handle.write('\n')
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return config


def summary(config):
    return {'endpoint': config['VLLM_URL'], 'model': config['SERVED_MODEL'],
            'api_key_present': bool(config.get('MODEL_API_KEY')),
            'provider_preferences': config['MODEL_PROVIDER_PREFERENCES'],
            'email_auth_enabled': config['EMAIL_AUTH_ENABLED'],
            'smtp_configured': all(config.get(k) for k in SMTP_KEYS-{'SMTP_PORT'})}


def check(config):
    """One public-only completion. Never prints the credential or an upstream error body."""
    api_key = config.get('MODEL_API_KEY')
    if urlsplit(config['VLLM_URL']).hostname == 'openrouter.ai' and not api_key:
        raise ValueError('API key is missing. Run this command with --ask-key in your terminal.')
    url = config['VLLM_URL'].rstrip('/').removesuffix('/v1')+'/v1/chat/completions'
    body = {'model': config['SERVED_MODEL'], 'messages': [{'role': 'user', 'content': 'Reply with the word ready.'}],
            'max_tokens': 128, 'temperature': 0, 'stream': False}
    for key, field in [('MODEL_PROVIDER_PREFERENCES', 'provider'), ('MODEL_REASONING', 'reasoning'), ('CHAT_TEMPLATE_KWARGS', 'chat_template_kwargs')]:
        if config.get(key): body[field] = config[key]
    headers = {'Content-Type': 'application/json'}
    if api_key: headers['Authorization'] = 'Bearer '+api_key
    request = Request(url, data=json.dumps(body).encode(), headers=headers)
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs): return None
    try:
        with build_opener(NoRedirect).open(request, timeout=60) as response: result = json.load(response)
    except HTTPError as exc:
        raise ValueError(f'Model probe returned HTTP {exc.code}. Check the key, quota, model and privacy-compatible provider availability; no fallback was enabled.') from None
    except Exception:
        raise ValueError('Model probe failed; check endpoint connectivity. No configuration fallback was enabled.') from None
    choices = result.get('choices') or []
    if not choices or not choices[0].get('message', {}).get('content') or choices[0].get('finish_reason') == 'error':
        raise ValueError('Model probe returned no usable completion')
    return {'public_probe': 'passed', 'model': result.get('model'), 'usage': result.get('usage', {})}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--file', type=Path, default=Path.home()/'.config/askiisc/aws-runtime.json')
    p.add_argument('--url', help='HTTPS OpenAI-compatible API base URL')
    p.add_argument('--model', help='Exact provider model ID')
    p.add_argument('--ask-key', action='store_true', help='Read the API key with a hidden terminal prompt')
    p.add_argument('--show', action='store_true', help='Show only non-secret settings')
    p.add_argument('--check', action='store_true', help='Send one short public-only completion')
    a = p.parse_args()
    key = getpass.getpass('Model API key (hidden): ') if a.ask_key else None
    config = configure(a.file, url=a.url, model=a.model, key=key)
    print(json.dumps({'file': str(a.file), **summary(config)}, indent=2))
    if a.check: print(json.dumps(check(config), indent=2))


if __name__ == '__main__':
    try: main()
    except ValueError as exc: raise SystemExit(str(exc))
