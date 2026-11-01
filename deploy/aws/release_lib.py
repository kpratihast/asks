"""Portable release integrity, packaging and deployment validation (standard library only)."""
from __future__ import annotations
import hashlib,json,math,re,sqlite3,tarfile
from pathlib import Path,PurePosixPath
from urllib.parse import urlsplit

MODEL='BAAI/bge-small-en-v1.5'
CONFIG_KEYS={'VLLM_URL','SERVED_MODEL','MODEL_API_KEY','MODEL_REASONING','MODEL_PROVIDER_PREFERENCES','CHAT_TEMPLATE_KWARGS',
             'MAX_MODEL_LEN','MAX_OUTPUT_TOKENS','EMAIL_AUTH_ENABLED','AUTH_PROFESSOR_TEST_EMAILS',
             'SMTP_HOST','SMTP_PORT','SMTP_USERNAME','SMTP_PASSWORD','SMTP_FROM','RATE_PER_MIN','GLOBAL_CONCURRENCY','TENANT_CONCURRENCY','TENANT_QUEUE','QUEUE_WAIT_S'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def valid_id(value):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}',value):raise ValueError('Invalid release ID')
    return value

def hostname(value):
    if len(value)>253 or not re.fullmatch(r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,63}',value):
        raise ValueError('Use a public DNS hostname you control, without scheme or path')
    return value.lower()

def runtime_config(raw):
    if not isinstance(raw,dict) or set(raw)-CONFIG_KEYS:raise ValueError('Runtime config contains unsupported keys')
    out={'MAX_MODEL_LEN':4096,'MAX_OUTPUT_TOKENS':512,'EMAIL_AUTH_ENABLED':True,
         'AUTH_PROFESSOR_TEST_EMAILS':'','MODEL_REASONING':{},'MODEL_PROVIDER_PREFERENCES':{},'CHAT_TEMPLATE_KWARGS':{},'RATE_PER_MIN':10,**raw}
    u=urlsplit(out.get('VLLM_URL',''))
    if u.scheme!='https' or not u.hostname or u.username or u.password or u.query or u.fragment:
        raise ValueError('AWS model endpoint must be an HTTPS base URL without credentials/query')
    if not isinstance(out.get('SERVED_MODEL'),str) or not out['SERVED_MODEL'].strip():raise ValueError('Set SERVED_MODEL explicitly')
    for k in ['MODEL_REASONING','MODEL_PROVIDER_PREFERENCES','CHAT_TEMPLATE_KWARGS']:
        if not isinstance(out[k],dict):raise ValueError(k+' must be an object')
    if not isinstance(out['EMAIL_AUTH_ENABLED'],bool):raise ValueError('EMAIL_AUTH_ENABLED must be a JSON boolean')
    if out['EMAIL_AUTH_ENABLED'] and any(not out.get(k) for k in ['SMTP_HOST','SMTP_USERNAME','SMTP_PASSWORD','SMTP_FROM']):
        raise ValueError('Email authentication requires the SMTP host, username, password and sender')
    if int(out['MAX_MODEL_LEN'])<=int(out['MAX_OUTPUT_TOKENS'])+128:raise ValueError('Context must exceed output and safety budget')
    for name,low,high in [('RATE_PER_MIN',1,100),('GLOBAL_CONCURRENCY',1,32),('TENANT_CONCURRENCY',1,32),('TENANT_QUEUE',0,100)]:
        if name not in out:continue
        value=out[name]
        if isinstance(value,bool) or not isinstance(value,(int,str)) or not re.fullmatch(r'[0-9]+',str(value)) or not low<=int(value)<=high:
            raise ValueError(f'{name} must be an integer from {low} to {high}')
    if 'QUEUE_WAIT_S' in out:
        try:value=float(out['QUEUE_WAIT_S'])
        except (ValueError,TypeError):raise ValueError('QUEUE_WAIT_S must be a finite number from 1 to 300') from None
        if isinstance(out['QUEUE_WAIT_S'],bool) or not math.isfinite(value) or not 1<=value<=300:
            raise ValueError('QUEUE_WAIT_S must be a finite number from 1 to 300')
    for v in out.values():
        if isinstance(v,str) and ('\x00' in v or '\n' in v or '\r' in v):raise ValueError('Config strings must be single-line values')
    return out

def verify_tree(root):
    root=Path(root).resolve();manifest=json.loads((root/'release.json').read_text())
    if manifest.get('schema')!=1:raise ValueError('Unsupported release schema')
    valid_id(manifest['release_id'])
    expected=set(manifest['files'])|{'release.json'}
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if expected!=actual:raise ValueError('Release contains missing or unexpected files')
    for name,digest in manifest['files'].items():
        rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name:raise ValueError('Unsafe release path')
        p=root/name
        if p.is_symlink() or any(x.is_symlink() for x in p.parents if x!=root.parent):raise ValueError('Release links are not allowed')
        if sha(p)!=digest:raise ValueError('Release checksum mismatch: '+name)
    return manifest

def unpack(archive,target):
    target=Path(target);target.mkdir(mode=0o700)
    with tarfile.open(archive,'r:gz') as t:
        members=t.getmembers()
        if len(members)>10000 or sum(m.size for m in members)>4*1024**3:raise ValueError('Release exceeds extraction limits')
        names=set()
        for m in members:
            rel=PurePosixPath(m.name)
            if rel.is_absolute() or '..' in rel.parts or '\\' in m.name or not(m.isfile() or m.isdir()):raise ValueError('Unsafe archive member')
            if m.name in names:raise ValueError('Duplicate archive member')
            names.add(m.name)
        t.extractall(target,filter='data')
    return verify_tree(target)

def sqlite_copy(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(Path(source).resolve().as_uri()+'?mode=ro',uri=True) as a,sqlite3.connect(target) as b:
        a.backup(b)
        if b.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('SQLite backup failed integrity check')
