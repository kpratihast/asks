#!/usr/bin/env python3.12
"""Root controller: combine CI code with an existing verified AWS corpus, then deploy."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse,fcntl,json,os,shutil,tempfile
from pathlib import Path
import host,ci_bundle
from release_lib import sha,unpack,valid_id,verify_tree


def prune(keep=5):
    protected={host.pointer('current'),host.pointer('previous')}
    candidates=sorted((p for p in (host.STATE/'releases').glob('ci-*') if p.is_dir() and not p.is_symlink()),key=lambda p:p.stat().st_mtime,reverse=True)
    for release in candidates[keep:]:
        if release in protected or (release/'.keep').exists():continue
        # Delete only verified, CI-created releases recoverable from the private S3 artifact.
        try:m=verify_tree(release)
        except (ValueError,OSError,KeyError):continue
        if not m.get('git_sha') or not m.get('base_release'):continue
        shutil.rmtree(release)
        index=host.STATE/'indexes'/release.name
        if index.exists() and not index.is_symlink():shutil.rmtree(index)
        host.cfg_for(release).unlink(missing_ok=True)


def deploy(bucket,key,checksum):
    h=json.loads(host.HOST.read_text())
    if bucket!=h['bucket'] or not key.startswith('releases/ci/') or not key.endswith('/code.tar.gz'):raise ValueError('Unexpected CI bucket/key')
    if len(checksum)!=64 or any(c not in '0123456789abcdef' for c in checksum):raise ValueError('Invalid checksum')
    with tempfile.TemporaryDirectory(prefix='.ci-',dir=host.STATE) as tmp:
        work=Path(tmp);source=work/'code.tar.gz'
        host.run('aws','--region',h['region'],'s3','cp','s3://'+bucket+'/'+key,source,'--only-show-errors')
        if sha(source)!=checksum:raise ValueError('Code checksum mismatch')
        code=work/'code';m=unpack(source,code);ci_bundle.validate_code(code);rid=valid_id(m['release_id'])
        if key!=f'releases/ci/{rid}/code.tar.gz':raise ValueError('CI release ID does not match key')
        base=host.pointer('current')
        if not base:raise ValueError('A full initial release must be installed before code-only deployments')
        # Retry after an SSM polling interruption must not reconstruct this ID against a new base.
        if base.name==rid:
            if verify_tree(base).get('code_sha256')!=m['source_sha256']:raise ValueError('Immutable CI release changed')
            host.health(8100,json.loads(host.cfg_for(base).read_text())['SERVED_MODEL']);return
        release=work/'release';full=ci_bundle.compose(base,code,release)
        bundle=work/'bundle.tar.gz';ci_bundle.archive(release,bundle)
        prefix=f'releases/ci/{rid}/';digest=sha(bundle)
        # Node permissions allow only this CI prefix, never overwriting credentials or corpus inputs.
        host.aws('s3api','put-object','--bucket',bucket,'--key',prefix+'bundle.tar.gz','--body',str(bundle),
                 '--server-side-encryption','AES256','--if-none-match','*','--metadata','sha256='+digest)
        for name in ['host.py','release_lib.py']:
            source=release/'deploy/aws'/name
            host.aws('s3api','put-object','--bucket',bucket,'--key',prefix+'recovery-'+name,'--body',str(source),
                     '--server-side-encryption','AES256','--if-none-match','*','--metadata','sha256='+sha(source))
        # host.install intentionally runs inside our shared deploy lock; it validates the candidate
        # before switching, uses the existing SSM model config, and restores the old app on failure.
        host.install(bucket,prefix+'bundle.tar.gz',digest)
        print(json.dumps({'git_sha':full['git_sha'],'release':rid,'snapshot':full['snapshot'],'artifact_sha256':digest}))
        prune()


def main():
    os.umask(0o077)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bucket',required=True);p.add_argument('--key',required=True);p.add_argument('--sha256',required=True);a=p.parse_args()
    if os.geteuid()!=0 or not os.path.ismount(host.STATE):raise SystemExit('Run as root with the persistent data volume mounted')
    with (host.STATE/'deploy.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);deploy(a.bucket,a.key,a.sha256)
if __name__=='__main__':main()
