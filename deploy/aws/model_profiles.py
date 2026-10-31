#!/usr/bin/env python3
"""One-command model profiles: public-only test, then immutable AWS deployment."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

import awsctl
from deploy_latest import check_account, verify_artifact
from release_lib import runtime_config, sqlite_copy, valid_id

OPENROUTER = 'https://openrouter.ai/api/v1'
MODEL_KEYS = {'VLLM_URL', 'SERVED_MODEL', 'MODEL_API_KEY', 'MODEL_REASONING',
              'MODEL_PROVIDER_PREFERENCES', 'CHAT_TEMPLATE_KWARGS', 'MAX_MODEL_LEN', 'MAX_OUTPUT_TOKENS'}
PROFILE_KEYS = {'provider', 'endpoint', 'model', 'api_key_env', 'api_key_file', 'provider_preferences',
                'reasoning', 'chat_template_kwargs', 'max_model_len', 'max_output_tokens', 'test_budget_usd'}


def atomic_json(path, value, private=True):
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError('Refusing a symlink for a configuration file')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        if not private:
            Path(temporary).chmod(0o644)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_profiles(path):
    data = json.loads(Path(path).expanduser().read_text())
    if set(data)-{'active', 'aws', 'profiles', 'runtime_file'} or not isinstance(data.get('profiles'), dict):
        raise ValueError('Invalid model profiles configuration')
    if set(data.get('aws', {})) != {'profile', 'account', 'region', 'stack'}:
        raise ValueError('Set AWS profile, account, region and stack in the model profiles configuration')
    for name, profile in data['profiles'].items():
        valid_id(name)
        if len(name)>45:
            raise ValueError('Profile names must be at most 45 characters')
        if not isinstance(profile, dict) or set(profile)-PROFILE_KEYS:
            raise ValueError('Unknown profile fields; keep API keys in environment/private key files')
        if not all(isinstance(profile.get(k),str) and profile[k].strip() for k in ['endpoint','model']):
            raise ValueError('Set an endpoint and exact model ID for every profile')
        if profile.get('provider') not in {'local', 'openrouter'}:
            raise ValueError('Profile provider must be local or openrouter')
    return data


def endpoint(value):
    value = value.rstrip('/')
    if not urlsplit(value).path.rstrip('/').endswith('/v1'):
        value += '/v1'
    return value


def resolve_key(profile, ask=False, optional=False):
    if ask:
        if profile['provider'] != 'openrouter':
            raise ValueError('--ask-key is available only for the OpenRouter profile')
        key = getpass.getpass('OpenRouter API key (hidden): ').strip()
        if not key:
            raise ValueError('API key cannot be empty')
        return key
    env = profile.get('api_key_env')
    if env and os.environ.get(env):
        return os.environ[env].strip()
    key_path = Path(profile.get('api_key_file', '~/.config/askiisc/openrouter.key')).expanduser()
    if profile['provider'] == 'openrouter' and key_path.exists():
        if key_path.is_symlink() or key_path.stat().st_mode & 0o077:
            raise ValueError('Private key file must be a regular file with permissions 600')
        key = key_path.read_text().strip()
        if key:
            return key
    if profile['provider'] == 'openrouter' and not optional:
        raise ValueError('OpenRouter key missing: set OPENROUTER_API_KEY or run this command with --ask-key once')
    return ''


def save_key(profile, key):
    path = Path(profile.get('api_key_file', '~/.config/askiisc/openrouter.key')).expanduser()
    if path.is_symlink():
        raise ValueError('Refusing a symlink for the private key file')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix='.openrouter-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(key+'\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def candidate(base, profile, key):
    url = endpoint(profile['endpoint'])
    if profile['provider'] == 'openrouter' and url != OPENROUTER:
        raise ValueError('OpenRouter credentials may be sent only to https://openrouter.ai/api/v1')
    if profile['provider'] == 'local' and urlsplit(url).hostname == 'openrouter.ai':
        raise ValueError('Use an OpenRouter profile for the OpenRouter endpoint')
    raw = {**base, 'VLLM_URL': url, 'SERVED_MODEL': profile['model'], 'MODEL_API_KEY': key,
           'MODEL_PROVIDER_PREFERENCES': profile.get('provider_preferences', {}) if profile['provider']=='openrouter' else {},
           'MODEL_REASONING': profile.get('reasoning', {}) if profile['provider']=='openrouter' else {},
           'CHAT_TEMPLATE_KWARGS': profile.get('chat_template_kwargs', {}) if profile['provider']=='local' else {},
           'MAX_MODEL_LEN': profile.get('max_model_len', base.get('MAX_MODEL_LEN',4096)),
           'MAX_OUTPUT_TOKENS': profile.get('max_output_tokens', base.get('MAX_OUTPUT_TOKENS',512))}
    result = runtime_config(raw)
    if not 1<=int(result['MAX_OUTPUT_TOKENS'])<int(result['MAX_MODEL_LEN']):
        raise ValueError('Set a positive output token limit below the context limit')
    if {k:v for k,v in result.items() if k not in MODEL_KEYS} != {k:v for k,v in base.items() if k not in MODEL_KEYS}:
        raise ValueError('A model switch must preserve authentication and non-model runtime settings')
    if profile['provider'] == 'openrouter':
        prefs = result['MODEL_PROVIDER_PREFERENCES']
        if prefs.get('data_collection')!='deny' or prefs.get('zdr') is not True:
            raise ValueError('OpenRouter profiles must retain data_collection=deny and zdr=true')
        prices = prefs.get('max_price', {})
        if not {'prompt','completion','request'} <= set(prices) or Decimal(str(prices['request'])) != 0:
            raise ValueError('Set explicit prompt/completion price ceilings and zero per-request fee')
        values=[Decimal(str(prices[k])) for k in prices]
        if any(not value.is_finite() or value<0 for value in values):
            raise ValueError('Price ceilings must be finite and non-negative')
        bound = (Decimal(str(prices['prompt']))*int(result['MAX_MODEL_LEN']) +
                 Decimal(str(prices['completion']))*int(result['MAX_OUTPUT_TOKENS']))/1000000
        if bound > Decimal(str(profile.get('test_budget_usd', '0.001'))):
            raise ValueError('Configured token/price ceilings exceed the single-test cost budget')
    return result


def public_probe(config, root, data_dir=None):
    """One real public-only RAG completion, with the app's prompt, streaming and citation checks."""
    root = Path(root)
    if data_dir is None:
        local = root/'config/local.json'
        data_dir = Path(json.loads(local.read_text())['ASKIISC_DATA_DIR']) if local.exists() else root/'data'
    data_dir = Path(data_dir)
    with tempfile.TemporaryDirectory(prefix='askiisc-public-probe-') as td:
        data = Path(td)/'data'; data.mkdir()
        for name in ['graph.json','snapshot.json','chunks.jsonl','qdrant/meta.json']:
            dest = data/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(data_dir/name,dest)
        for db in (data_dir/'qdrant/collection').glob('*/storage.sqlite'):
            sqlite_copy(db,data/db.relative_to(data_dir))
        os.environ.update({'ASKIISC_BACKEND':'stub','ASKIISC_DATA_DIR':str(data),
            'ASKIISC_EMBED_MODEL':'BAAI/bge-small-en-v1.5','ASKIISC_EMBED_DIM':'384',
            'ASKIISC_RERANKER':'none','ASKIISC_RERANK_MIN':'none','ASKIISC_DEVICE':'cpu',
            'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
        sys.path.insert(0,str(root))
        from askiisc_retrieval.service import build_retriever
        from askiisc_serving.retrieval_client import LocalRetrieval
        from askiisc_serving.generator import Generator, GenerationUnavailable, verify_connected, citations_for
        from askiisc_serving.settings import Settings
        retriever = build_retriever()
        async def run():
            generator = Generator(Settings(vllm_url=config['VLLM_URL'],served_model=config['SERVED_MODEL'],
                model_api_key=config.get('MODEL_API_KEY') or None,
                model_provider_preferences=config['MODEL_PROVIDER_PREFERENCES'],model_reasoning=config['MODEL_REASONING'],
                chat_template_kwargs=config['CHAT_TEMPLATE_KWARGS'],max_model_len=int(config['MAX_MODEL_LEN']),
                max_output_tokens=int(config['MAX_OUTPUT_TOKENS']),regenerate_on_bad_citations=False))
            try:
                question = 'Who teaches DS 252?'
                hits, trace = await LocalRetrieval(retriever).run(question,None,'full',None,None,scopes=['public'])
                if not hits or any(h.get('access_scope','public')!='public' for h in hits):
                    raise ValueError('The model test requires public-only source evidence')
                messages, selected, budget = generator.prepare(question,hits,None,None,mode='full')
                try:
                    answer, metadata = await generator.complete(messages)
                except GenerationUnavailable as exc:
                    raise RuntimeError('Model test failed: '+str(exc)) from None
                checked = verify_connected(answer,selected)
                if not checked['ok'] or checked['abstained']:
                    raise RuntimeError('Model test did not return a verified cited answer; no retry or fallback was used')
                return {'status':'passed','model':config['SERVED_MODEL'],'question':question,'answer':answer,
                        'citations':citations_for(selected,checked['used']),'usage':metadata.get('usage'),
                        'snapshot':trace.get('snapshot_id'),'generation_calls':1,'evidence_scopes':['public']}
            finally:
                await generator.c.aclose()
        try:
            return asyncio.run(run())
        finally:
            retriever.store.close()


def runtime_metadata(region, instance):
    # The deployer may write this parameter but cannot read its secret value. The
    # existing instance role returns only its version/digest, plus the active release.
    code = """import hashlib,json,subprocess
from pathlib import Path
host=json.loads(Path('/etc/askiisc/host.json').read_text())
p=json.loads(subprocess.check_output(['aws','--region',host['region'],'ssm','get-parameter','--name',host['parameter'],'--with-decryption','--output','json'],text=True))['Parameter']
cfg=json.loads(p['Value'])
print(json.dumps({'version':p['Version'],'sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),'release':Path('/var/lib/askiisc/current').resolve().name}))
"""
    result = awsctl.remote(region,instance,"python3.12 - <<'PY'\n"+code+"PY")
    return json.loads(result['StandardOutputContent'])


def put_runtime(region, parameter, config):
    if len(json.dumps(config).encode()) > 4096:
        raise ValueError('Standard SecureString configuration must fit within 4 KB')
    return awsctl.json_call(region,['ssm','put-parameter'],{'Name':parameter,'Type':'SecureString','Tier':'Standard',
                           'Value':json.dumps(config),'Overwrite':True})['Version']


def deploy_profile(config_path, profiles, name, profile, base, config, root, bundles, data_dir=None):
    settings = profiles['aws']; region=settings['region']; stack=settings['stack']
    for key in ['AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_SESSION_TOKEN']:
        if os.environ.get(key):
            raise ValueError('Unset exported AWS access/session keys so the named profile is used')
    os.environ.update(AWS_PROFILE=settings['profile'],AWS_DEFAULT_PROFILE=settings['profile'],AWS_PAGER='')
    check_account(region,settings['account'])
    outputs = awsctl.outputs(region,stack)
    before = runtime_metadata(region,outputs['InstanceId'])
    if before['sha256'] != hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest():
        raise ValueError('AWS runtime differs from the protected local base; reconcile it before switching models')
    data = data_dir or Path(json.loads((Path(root)/'config/local.json').read_text())['ASKIISC_DATA_DIR'])
    rid = datetime.now(timezone.utc).strftime('model-%Y%m%dT%H%M%SZ-')+name
    release = awsctl.package(root,data,bundles,release_id=rid)
    verify_artifact(release/'artifact.json')
    version = before['version']
    changed = config != base
    if changed:
        version = put_runtime(region,outputs['RuntimeParameter'],config)
    try:
        # Old-model readiness is deliberately not a prerequisite: its tunnel may
        # have expired. The immutable candidate must pass readiness AND generation.
        awsctl.deploy(region,stack,release/'artifact.json')
    except BaseException:
        # Reconcile the actual host before changing the parameter back; the SSM
        # response may have been lost after successful activation.
        try:
            actual = runtime_metadata(region,outputs['InstanceId'])
            if actual['release'] == rid:
                atomic_json(profiles['runtime_file'],config)
                profiles['profiles'][name]=profile; profiles['active']=name
                atomic_json(config_path,profiles,private=False)
            elif changed:
                put_runtime(region,outputs['RuntimeParameter'],base)
                print('Candidate did not activate; previous runtime parameter restored.',file=sys.stderr)
        except Exception:
            print('Could not confirm activation state; inspect AWS status before retrying.',file=sys.stderr)
        raise
    atomic_json(profiles['runtime_file'],config)
    profiles['profiles'][name] = profile; profiles['active'] = name
    atomic_json(config_path,profiles,private=False)
    result={'release':rid,'previous_release':before['release'],'runtime_parameter_version':version,
            'profile':name,'endpoint':config['VLLM_URL'],'model':config['SERVED_MODEL'],'url':outputs['ApplicationUrl']}
    # Save activation before public health checks, so a transient network failure
    # cannot leave the local base pointing at a different captured runtime.
    atomic_json(release/'model-switch.json',{**result,'status':'activated_pending_public_checks'})
    awsctl.smoke(outputs['ApplicationUrl'])
    atomic_json(release/'model-switch.json',{**result,'status':'verified'})
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('profile',nargs='?')
    p.add_argument('--config',type=Path,default=awsctl.HERE/'model-profiles.example.json')
    p.add_argument('--root',type=Path,default=awsctl.ROOT)
    p.add_argument('--bundles',type=Path)
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--url');p.add_argument('--model');p.add_argument('--ask-key',action='store_true')
    modes=p.add_mutually_exclusive_group();modes.add_argument('--check-only',action='store_true');modes.add_argument('--dry-run',action='store_true')
    p.add_argument('--report',type=Path)
    a=p.parse_args(argv)
    profiles=load_profiles(a.config);name=a.profile or profiles.get('active')
    if name not in profiles['profiles']:
        raise ValueError('Choose a configured profile: '+', '.join(profiles['profiles']))
    profile=deepcopy(profiles['profiles'][name])
    if a.url:profile['endpoint']=a.url
    if a.model:profile['model']=a.model
    if a.ask_key and a.dry_run:raise ValueError('--ask-key cannot be combined with --dry-run')
    base=runtime_config(json.loads(Path(profiles['runtime_file']).expanduser().read_text()))
    key=resolve_key(profile,ask=a.ask_key,optional=a.dry_run)
    config=candidate(base,profile,key)
    print(json.dumps({'profile':name,'endpoint':config['VLLM_URL'],'model':config['SERVED_MODEL'],
                      'api_key_present':bool(key),'dry_run':a.dry_run,'check_only':a.check_only},indent=2),flush=True)
    if a.dry_run:return
    report=public_probe(config,a.root,a.data_dir)
    print(json.dumps({'public_rag_test':report},indent=2),flush=True)
    if a.ask_key:save_key(profile,key)
    if a.report:atomic_json(a.report,report)
    if a.check_only:return
    result=deploy_profile(a.config,profiles,name,profile,base,config,a.root,a.bundles or a.root/'.local/aws-releases',a.data_dir)
    print(json.dumps({'deployment':result},indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError,KeyError,subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
