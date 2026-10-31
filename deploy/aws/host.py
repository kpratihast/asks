#!/usr/bin/env python3.12
"""Root-only EC2 release controller. SSM invokes a checksum-pinned copy of this file."""
from __future__ import annotations
import argparse,fcntl,json,os,pwd,shutil,sqlite3,stat,subprocess,sys,tarfile,tempfile,time,uuid
from pathlib import Path
from urllib.request import urlopen,Request
sys.path.insert(0,str(Path(__file__).resolve().parent))
from release_lib import sha,unpack,verify_tree,valid_id,runtime_config,sqlite_copy
STATE=Path('/var/lib/askiisc')
HOST=Path('/etc/askiisc/host.json')

def run(*args,capture=False,as_app=False):
    return subprocess.run([str(x) for x in args],check=True,text=True,stdout=subprocess.PIPE if capture else None,
                          **(identity() if as_app else {})).stdout

def aws(*args):
    return json.loads(run('aws','--region',json.loads(HOST.read_text())['region'],*args,'--output','json',capture=True) or '{}')

def pointer(name):
    p=STATE/name
    if not p.is_symlink():return None
    target=p.resolve()
    if target.parent!=STATE/'releases':raise ValueError('Invalid release pointer')
    return target

def link(target,name):
    tmp=STATE/('.link-'+uuid.uuid4().hex);tmp.symlink_to(target);tmp.replace(STATE/name)

def readable_tree(root):
    # Never follow venv Python symlinks into the operating system.
    for path in [root,*root.rglob('*')]:
        if path.is_symlink():continue
        executable=bool(path.stat().st_mode & stat.S_IXUSR)
        shutil.chown(path,user='root',group='askiisc')
        path.chmod(0o750 if path.is_dir() or executable else 0o640)

def venv_for(release):return STATE/'venvs'/sha(release/'deploy/aws/requirements.lock')

def health(port,model,timeout=180):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        try:
            with urlopen(f'http://127.0.0.1:{port}/readyz',timeout=15) as r:body=json.load(r)
            if body.get('retrieval') and body.get('model_endpoint') and body.get('model')==model:return
        except Exception:pass
        time.sleep(3)
    raise RuntimeError('Readiness check failed; inspect local application logs and model credentials')

def inference_probe(port):
    request=Request(f'http://127.0.0.1:{port}/v1/ask',data=json.dumps({'question':'What is DS252 about?'}).encode(),headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=200) as response:body=json.load(response)
    if body.get('status')!='answered' or not any(c.get('used') for c in body.get('citations',[])):
        raise RuntimeError('Public generation smoke failed; check model quota, key and answer capability')

def identity():
    u=pwd.getpwnam('askiisc');return {'user':u.pw_uid,'group':u.pw_gid,'extra_groups':[]}

def cfg_for(release):return Path('/etc/askiisc/releases')/(release.name+'.json')

def check(release):
    # Qdrant creates its persistent lock here; it must belong to the service user.
    run(venv_for(release)/'bin/python',release/'deploy/aws/runtime.py','--release',release,'--check',as_app=True)
    with (STATE/'candidate.log').open('w') as log:
        child=subprocess.Popen([str(venv_for(release)/'bin/python'),str(release/'deploy/aws/runtime.py'),
                                '--release',str(release),'--port','8101'],cwd=release,stdout=log,stderr=log,
                               env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},**identity())
        try:
            health(8101,json.loads(cfg_for(release).read_text())['SERVED_MODEL'])
            inference_probe(8101)
        finally:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()

def activate(release):
    old=pointer('current')
    if old==release:
        health(8100,json.loads(cfg_for(release).read_text())['SERVED_MODEL']);return
    # Preserve sign-ins before any change; auth state is independent of every release/index.
    db=STATE/'auth/sessions.sqlite3'
    if db.exists():
        backup=STATE/'backups'/('pre-release-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.sqlite3')
        sqlite_copy(db,backup);backup.chmod(0o600)
    run('systemctl','stop','askiisc.service')
    link(release,'current')
    try:
        run('systemctl','start','askiisc.service')
        health(8100,json.loads(cfg_for(release).read_text())['SERVED_MODEL'])
    except Exception:
        run('systemctl','stop','askiisc.service')
        if old:
            link(old,'current');run('systemctl','start','askiisc.service')
            health(8100,json.loads(cfg_for(old).read_text())['SERVED_MODEL'])
        else:(STATE/'current').unlink(missing_ok=True)
        raise
    if old:link(old,'previous')
    # Keep a root-owned controller for rollback/backup independent of the current symlink.
    for name in ['host.py','release_lib.py']:
        shutil.copy2(release/'deploy/aws'/name,Path('/usr/local/lib/askiisc')/name)
    run('systemctl','enable','--now','askiisc-backup.timer')
    print(json.dumps({'release':release.name,'previous':old.name if old else None,'status':'active'}))

def install(bucket,key,checksum):
    if len(checksum)!=64 or any(c not in '0123456789abcdef' for c in checksum):raise ValueError('Invalid artifact checksum')
    host=json.loads(HOST.read_text())
    if bucket!=host['bucket'] or not key.startswith('releases/'):raise ValueError('Unexpected release bucket/key')
    work=Path(tempfile.mkdtemp(prefix='.deploy-',dir=STATE))
    try:
        archive=work/'bundle.tar.gz'
        run('aws','--region',host['region'],'s3','cp','s3://'+bucket+'/'+key,archive,'--only-show-errors')
        if sha(archive)!=checksum:raise ValueError('Artifact checksum mismatch')
        incoming=work/'release';manifest=unpack(archive,incoming);rid=valid_id(manifest['release_id'])
        release=STATE/'releases'/rid
        if release.exists():
            if verify_tree(release)!=manifest:raise ValueError('Release ID already exists with different content')
        else:incoming.rename(release)
        secret=aws('ssm','get-parameter','--name',host['parameter'],'--with-decryption')['Parameter']
        config=runtime_config(json.loads(secret['Value']));cfg=cfg_for(release)
        if cfg.exists() and json.loads(cfg.read_text())!=config:raise ValueError('Configuration changed; create a new release ID')
        cfg.write_text(json.dumps(config)+'\n');cfg.chmod(0o640);shutil.chown(cfg,group='askiisc')
        envdir=venv_for(release)
        if not (envdir/'.complete').exists():
            if envdir.exists():shutil.rmtree(envdir) # only an incomplete, hash-addressed venv
            run('python3.12','-m','venv',envdir)
            run(envdir/'bin/pip','install','--only-binary=:all:','--require-hashes','--extra-index-url',
                'https://download.pytorch.org/whl/cpu','-r',release/'deploy/aws/requirements.lock')
            (envdir/'.complete').touch()
        readable_tree(envdir)
        index=STATE/'indexes'/rid
        if not index.exists():
            shutil.copytree(release/'data',index)
            for p in [index,*index.rglob('*')]:shutil.chown(p,user='askiisc',group='askiisc')
            index.chmod(0o700)
        # Code and weights are readable by the app but remain owned by root.
        for p in [release,*release.rglob('*')]:
            shutil.chown(p,user='root',group='askiisc');p.chmod(0o750 if p.is_dir() else 0o640)
        check(release);activate(release)
        (STATE/'deployment.json').write_text(json.dumps({'release':rid,'artifact_sha256':checksum,
            'parameter_version':secret['Version'],'snapshot':manifest['snapshot'],'at':time.time()},indent=2)+'\n')
    finally:shutil.rmtree(work)

def rollback(target=None):
    release=STATE/'releases'/valid_id(target) if target else pointer('previous')
    if not release or not release.is_dir():raise ValueError('No previous release is available')
    verify_tree(release);check(release);activate(release)

def backup():
    current=pointer('current')
    if not current:raise ValueError('No active release')
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8]
    work=Path(tempfile.mkdtemp(prefix='.backup-',dir=STATE))
    try:
        db=STATE/'auth/sessions.sqlite3'
        if db.exists():sqlite_copy(db,work/'sessions.sqlite3')
        (work/'state.json').write_text(json.dumps({'current':current.name,'previous':pointer('previous').name if pointer('previous') else None}))
        archive=STATE/'backups'/(stamp+'.tar.gz')
        with tarfile.open(archive,'w:gz') as t:
            for p in work.iterdir():t.add(p,arcname=p.name)
        archive.chmod(0o600);host=json.loads(HOST.read_text());key='backups/'+stamp+'.tar.gz'
        run('aws','--region',host['region'],'s3','cp',archive,'s3://'+host['bucket']+'/'+key,'--sse','AES256','--only-show-errors')
        print(json.dumps({'backup':'s3://'+host['bucket']+'/'+key,'sha256':sha(archive)}))
    finally:shutil.rmtree(work)

def main():
    os.umask(0o077)
    if os.geteuid()!=0:raise SystemExit('Run this controller through SSM as root')
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('install');d.add_argument('--bucket',required=True);d.add_argument('--key',required=True);d.add_argument('--sha256',required=True)
    r=sub.add_parser('rollback');r.add_argument('--release-id');sub.add_parser('backup');sub.add_parser('status');a=p.parse_args()
    if not os.path.ismount(STATE):raise SystemExit('Persistent data volume is not mounted')
    with (STATE/'deploy.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if a.command=='install':install(a.bucket,a.key,a.sha256)
        elif a.command=='rollback':rollback(a.release_id)
        elif a.command=='backup':backup()
        else:print(json.dumps({'current':pointer('current').name if pointer('current') else None,'previous':pointer('previous').name if pointer('previous') else None}))
if __name__=='__main__':main()
