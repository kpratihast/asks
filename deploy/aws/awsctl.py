#!/usr/bin/env python3
"""Prepare private releases and operate the AWS stack. No AWS call is made by package/verify."""
from __future__ import annotations
import argparse,hashlib,json,os,re,shlex,shutil,sqlite3,subprocess,sys,tarfile,tempfile,time
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import urlopen,Request
from urllib.error import HTTPError
sys.path.insert(0,str(Path(__file__).resolve().parent))
from release_lib import MODEL,sha,sqlite_copy,verify_tree,unpack,valid_id,hostname,runtime_config
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
DEPLOY_FILES={'ci.py','ci_bundle.py','ci_host.py','ci_setup.py','GITHUB_DEPLOYMENT.md','model_profiles.py','model-profiles.example.json','deploy_latest.py','awsctl.py','release_lib.py','host.py','runtime.py','model_config.py','iam_bootstrap.py','render_stack.py','bootstrap.sh','stack.json','requirements.in','requirements.lock','README.md','runtime.example.json','deployer-policy.example.json'}
SOURCE_SUFFIXES={'.py','.html','.css','.js','.svg','.png','.webp','.ico','.woff2'}

def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)

def package(root,data,out,model_dir=None,release_id=None):
    root=Path(root).resolve();data=Path(data).resolve();out=Path(out).resolve()
    snap=json.loads((data/'snapshot.json').read_text())
    if snap['embedding_model']!=MODEL:raise ValueError('This deployment profile requires the current BGE-small index')
    frozen={n:sha(data/n) for n in ['snapshot.json','chunks.jsonl','graph.json','qdrant/meta.json']}
    private=json.loads((root/'config/restricted_documents.json').read_text())
    if model_dir is None:
        cache=Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'hub/models--BAAI--bge-small-en-v1.5'
        model_dir=cache/'snapshots'/(cache/'refs/main').read_text().strip()
    model_dir=Path(model_dir).resolve();revision=model_dir.name
    if not re.fullmatch(r'[0-9a-f]{40}',revision) or not (model_dir/'model.safetensors').is_file():raise ValueError('Supply a complete cached model snapshot ending in its 40-character revision')
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    with tempfile.TemporaryDirectory(prefix='.aws-package-',dir=out) as tmp:
        tree=Path(tmp)/'release';tree.mkdir()
        for directory in ['askiisc_retrieval','askiisc_serving']:
            for p in (root/directory).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix in SOURCE_SUFFIXES:copy(p,tree/p.relative_to(root))
        for p in HERE.iterdir():
            if p.is_file() and p.name in DEPLOY_FILES:
                copy(p,tree/'deploy/aws'/p.name)
        for name in ['faculty_directory.json','restricted_documents.json']:copy(root/'config'/name,tree/'config'/name)
        for doc in private['documents']:
            for key,digest in [('original','sha256'),('extracted','extracted_sha256')]:
                source=(root/doc[key]).resolve()
                if not source.is_relative_to((root/'.local/private').resolve()) or sha(source)!=doc[digest]:raise ValueError('Private input changed or escaped its approved directory')
                copy(source,tree/doc[key])
        for name in ['snapshot.json','chunks.jsonl','graph.json','qdrant/meta.json']:copy(data/name,tree/'data'/name)
        databases=list((data/'qdrant/collection').glob('*/storage.sqlite'))
        if not databases:raise ValueError('Expected the embedded Qdrant SQLite layout')
        for db in databases:sqlite_copy(db,tree/'data'/db.relative_to(data))
        if frozen!={n:sha(data/n) for n in frozen}:raise ValueError('Index changed while packaging; retry from a stable snapshot')
        hf=tree/'assets/hf/hub/models--BAAI--bge-small-en-v1.5'
        for p in model_dir.rglob('*'):
            if p.is_file():copy(p,hf/'snapshots'/revision/p.relative_to(model_dir))
        (hf/'refs').mkdir();(hf/'refs/main').write_text(revision)
        chunks=[json.loads(line) for line in (data/'chunks.jsonl').read_text().splitlines() if line.strip()]
        for doc in private['documents']:
            matches=[c for c in chunks if c.get('source_url')=='/v1/documents/'+doc['id']
                     and c.get('version_id') in snap.get('active_version_ids', [])]
            if not matches or any(c['access_scope']!=doc['access_scope'] for c in matches):
                raise ValueError('Private catalog and selected snapshot disagree; use --data-dir from the successful document import')
        files={str(p.relative_to(tree)):sha(p) for p in sorted(tree.rglob('*')) if p.is_file()}
        content=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
        rid=valid_id(release_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+content[:10])
        destination=out/rid
        if destination.exists():raise ValueError('Use a new release ID/output directory')
        manifest={'schema':1,'release_id':rid,'source_sha256':content,'snapshot':snap['snapshot_id'],
                  'embedding_model':MODEL,'embedding_revision':revision,'chunks':len(chunks),
                  'scopes':{s:sum(c['access_scope']==s for c in chunks) for s in ['public','student','faculty']},'files':files}
        (tree/'release.json').write_text(json.dumps(manifest,indent=2)+'\n');verify_tree(tree)
        destination.mkdir(mode=0o700);archive=destination/'bundle.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=3) as t:
            for p in sorted(tree.rglob('*')):
                if p.is_file():t.add(p,arcname=str(p.relative_to(tree)),recursive=False)
        archive.chmod(0o600)
        for name in ['host.py','release_lib.py']:copy(HERE/name,destination/name)
        artifact={'release_id':rid,'archive':'bundle.tar.gz','sha256':sha(archive),'bytes':archive.stat().st_size,
                  'snapshot':manifest['snapshot'],'embedding_revision':revision,
                  'controls':{n:sha(destination/n) for n in ['host.py','release_lib.py']}}
        (destination/'artifact.json').write_text(json.dumps(artifact,indent=2)+'\n')
    print(json.dumps({'artifact':str(destination/'artifact.json'),**artifact},indent=2));return destination

def aws(region,*args):
    cmd=['aws','--region',region,*map(str,args),'--output','json','--no-cli-pager']
    return json.loads(subprocess.run(cmd,check=True,text=True,stdout=subprocess.PIPE).stdout or '{}')

def outputs(region,stack):
    data=aws(region,'cloudformation','describe-stacks','--stack-name',stack)['Stacks'][0]
    return {x['OutputKey']:x['OutputValue'] for x in data.get('Outputs',[])}

def json_call(region,args,payload):
    with tempfile.TemporaryDirectory() as td:
        f=Path(td)/'input.json';f.write_text(json.dumps(payload));f.chmod(0o600)
        return aws(region,*args,'--cli-input-json','file://'+str(f))

def remote(region,instance,command):
    data=json_call(region,['ssm','send-command'],{'DocumentName':'AWS-RunShellScript','InstanceIds':[instance],
        'TimeoutSeconds':1800,'Parameters':{'commands':['bash -c '+shlex.quote('set -euo pipefail\n'+command)],'executionTimeout':['1800']},'Comment':'Ask IISc release operation'})
    cid=data['Command']['CommandId'];print('SSM command:',cid,flush=True);last=None
    until=time.monotonic()+1900
    while time.monotonic()<until:
        time.sleep(5)
        try:result=aws(region,'ssm','get-command-invocation','--command-id',cid,'--instance-id',instance)
        except subprocess.CalledProcessError:continue
        status=result['Status']
        if status!=last:print('SSM status:',status,flush=True);last=status
        if status=='Success':print(result.get('StandardOutputContent',''));return result
        if status in {'Failed','Cancelled','TimedOut','Cancelling'}:
            print(result.get('StandardErrorContent',''),file=sys.stderr)
            raise RuntimeError('Remote operation failed; inspect the SSM command and journal logs')
    raise TimeoutError('Remote operation still pending; inspect SSM before running another deployment')

def upload(region,bucket,key,path):
    # Immutable keys: rerunning the same upload is safe; an existing different artifact is rejected.
    try:head=aws(region,'s3api','head-object','--bucket',bucket,'--key',key)
    except subprocess.CalledProcessError:head=None
    if head is not None:
        if head.get('Metadata',{}).get('sha256')!=sha(path):raise ValueError('S3 release key already contains different content')
        return
    aws(region,'s3api','put-object','--bucket',bucket,'--key',key,'--body',path,'--server-side-encryption','AES256',
        '--if-none-match','*','--metadata','sha256='+sha(path))

def deploy(region,stack,descriptor):
    descriptor=Path(descriptor).resolve();a=json.loads(descriptor.read_text());rid=valid_id(a['release_id']);folder=descriptor.parent
    if a.get('archive')!='bundle.tar.gz' or sha(folder/'bundle.tar.gz')!=a['sha256']:raise ValueError('Artifact checksum mismatch')
    for name in ['host.py','release_lib.py']:
        if sha(folder/name)!=a['controls'][name]:raise ValueError('Controller checksum mismatch')
    o=outputs(region,stack);bucket=o['ReleaseBucket'];prefix='releases/'+rid+'/'
    for name in ['bundle.tar.gz','artifact.json','host.py','release_lib.py']:upload(region,bucket,prefix+name,folder/name)
    control='/var/lib/askiisc/control/'+a['controls']['host.py']
    commands=['install -d -m 0700 '+shlex.quote(control)]
    for name in ['host.py','release_lib.py']:
        path=control+'/'+name
        commands+=['aws --region '+shlex.quote(region)+' s3 cp '+shlex.quote('s3://'+bucket+'/'+prefix+name)+' '+shlex.quote(path)+' --only-show-errors',
                   "printf '%s\\n' "+shlex.quote(a['controls'][name]+'  '+path)+' | sha256sum -c -']
    commands += ['python3.12 '+shlex.quote(control+'/host.py')+' install --bucket '+shlex.quote(bucket)+' --key '+shlex.quote(prefix+'bundle.tar.gz')+' --sha256 '+a['sha256'],
                 'systemctl start caddy.service']
    remote(region,o['InstanceId'],'\n'.join(commands));print('Application activated:',o['ApplicationUrl'])
    print('Run the smoke command after the trusted HTTPS certificate is ready.')

def smoke(url):
    url=url.rstrip('/');u=__import__('urllib.parse',fromlist=['urlsplit']).urlsplit(url)
    if u.scheme!='https' and not(u.scheme=='http' and u.hostname in {'localhost','127.0.0.1'}):raise ValueError('Use HTTPS, or HTTP on loopback for local testing')
    results={}
    for path in ['/','/healthz','/readyz','/auth/me']:
        with urlopen(url+path,timeout=30) as response:
            results[path]=response.status
            if path=='/readyz':
                body=json.load(response)
                if not(body.get('retrieval') and body.get('model_endpoint')):raise ValueError('Model/index not ready')
    for path,body in [('/v1/ask',{'question':'When is DS252 final exam?'}),('/v1/documents/faculty-exam-timetable-ad2026',None)]:
        request=Request(url+path,data=json.dumps(body).encode() if body else None,headers={'Content-Type':'application/json'})
        try:
            with urlopen(request,timeout=30):raise ValueError('Restricted source was accessible without login')
        except HTTPError as exc:
            if exc.code not in {401,403}:raise
            data=exc.read()
            if b'2026-12-01' in data or b'09:00-12:00' in data:raise ValueError('Denied response leaked restricted facts')
            results[path]=exc.code
    print(json.dumps({'url':url,'checks':results,'generation_calls':0},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    pkg=sub.add_parser('package');pkg.add_argument('--root',type=Path,default=ROOT);pkg.add_argument('--data-dir',type=Path)
    pkg.add_argument('--out',type=Path,required=True);pkg.add_argument('--embedding-dir',type=Path);pkg.add_argument('--release-id')
    v=sub.add_parser('verify');v.add_argument('archive',type=Path)
    s=sub.add_parser('smoke');s.add_argument('--url',required=True)
    for name in ['provision','put-secret','deploy','status','rollback','backup','logs']:
        cmd=sub.add_parser(name);cmd.add_argument('--region',default='ap-south-1');cmd.add_argument('--stack',required=True)
        if name=='provision':
            cmd.add_argument('--domain',default='',help='Optional hostname; omit for trusted HTTPS on the Elastic IP');cmd.add_argument('--instance-type',choices=['t3.small','t3.medium'],default='t3.small')
            cmd.add_argument('--execution-role',required=True,help='App-scoped CloudFormation service role ARN');cmd.add_argument('--instance-profile',required=True);cmd.add_argument('--ami-id');cmd.add_argument('--parameter',default='/askiisc/prod/runtime');cmd.add_argument('--budget-email',default='')
            cmd.add_argument('--monthly-budget-usd',type=float,default=20);cmd.add_argument('--client-cidr',default='0.0.0.0/0');cmd.add_argument('--plan',action='store_true')
            cmd.add_argument('--preserve-on-failure',action='store_true',help='Keep failed host for SSM diagnostics; resources continue consuming credits until recovered or removed')
        if name=='put-secret':cmd.add_argument('--file',type=Path,required=True)
        if name=='deploy':cmd.add_argument('--artifact',type=Path,required=True)
        if name=='rollback':cmd.add_argument('--release-id')
    a=p.parse_args()
    if a.command=='package':
        data=a.data_dir or Path(json.loads((a.root/'config/local.json').read_text())['ASKIISC_DATA_DIR'])
        package(a.root,data,a.out,a.embedding_dir,a.release_id)
    elif a.command=='verify':
        with tempfile.TemporaryDirectory() as td:
            m=unpack(a.archive,Path(td)/'release');print(json.dumps({k:v for k,v in m.items() if k!='files'},indent=2))
    elif a.command=='smoke':smoke(a.url)
    elif a.command=='provision':
        domain=hostname(a.domain) if a.domain else ''
        if domain.endswith(('.example.com','.example.org','.example.net')):raise ValueError('Replace the example hostname with a real domain')
        ami=a.ami_id
        if not ami:
            try:ami=outputs(a.region,a.stack).get('AmiId')
            except subprocess.CalledProcessError:pass
        if not ami:ami=aws(a.region,'ssm','get-parameter','--name','/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64')['Parameter']['Value']
        params={'DomainName':domain,'InstanceProfileName':a.instance_profile,'InstanceType':a.instance_type,'AmiId':ami,'RuntimeParameter':a.parameter,
                'BudgetEmail':a.budget_email,'MonthlyBudgetUsd':str(a.monthly_budget_usd),'ClientCidr':a.client_cidr}
        cmd=['aws','--region',a.region,'cloudformation','deploy','--stack-name',a.stack,'--template-file',str(HERE/'stack.json'),
             '--role-arn',a.execution_role,'--no-fail-on-empty-changeset','--parameter-overrides',*[k+'='+v for k,v in params.items()]]
        if a.plan:cmd.append('--no-execute-changeset')
        if a.preserve_on_failure:cmd.append('--disable-rollback')
        subprocess.run(cmd,check=True)
        if not a.plan:print(json.dumps(outputs(a.region,a.stack),indent=2))
    elif a.command=='put-secret':
        raw=json.loads(a.file.read_text());config=runtime_config(raw)
        if any(isinstance(v,str) and v.startswith('<') for v in config.values()):raise ValueError('Replace all secret placeholders')
        if len(json.dumps(config).encode())>4096:raise ValueError('Standard Parameter Store JSON must fit within 4 KB')
        o=outputs(a.region,a.stack)
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'runtime.json';f.write_text(json.dumps(config));f.chmod(0o600)
            print(json.dumps(aws(a.region,'ssm','put-parameter','--name',o['RuntimeParameter'],'--type','SecureString',
                                 '--tier','Standard','--value','file://'+str(f),'--overwrite')))
    elif a.command=='deploy':deploy(a.region,a.stack,a.artifact)
    else:
        o=outputs(a.region,a.stack);command='python3.12 /usr/local/lib/askiisc/host.py '+a.command
        if a.command=='rollback' and a.release_id:command+=' --release-id '+shlex.quote(valid_id(a.release_id))
        if a.command=='logs':command='journalctl -u askiisc.service -u caddy.service -n 80 --no-pager'
        if a.command=='status':command='if [ -f /usr/local/lib/askiisc/host.py ]; then '+command+'; else echo Host-provisioned-no-release; fi'
        remote(a.region,o['InstanceId'],command)
if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError) as exc:raise SystemExit(str(exc))
