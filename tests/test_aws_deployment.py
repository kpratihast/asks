"""Offline deployment safety tests; no account, credentials, network or Docker required."""
import hashlib,io,json,sqlite3,sys,tarfile
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'deploy/aws'))
import awsctl,release_lib,runtime,render_stack,host,model_config,iam_bootstrap


def config():return {'VLLM_URL':'https://models.example.org/v1','SERVED_MODEL':'test/model','EMAIL_AUTH_ENABLED':False}

@pytest.mark.parametrize('url',['http://127.0.0.1:8000','http://models.example.org','https://user:key@models.example.org','https://models.example.org/?key=secret'])
def test_remote_endpoint_requires_https_without_embedded_secrets(url):
    with pytest.raises(ValueError):release_lib.runtime_config({**config(),'VLLM_URL':url})


def test_cloud_defaults_have_no_professor_override():
    c=release_lib.runtime_config(config())
    assert c['AUTH_PROFESSOR_TEST_EMAILS']==''
    with pytest.raises(ValueError):release_lib.runtime_config({**config(),'EMAIL_AUTH_ENABLED':True})
    with pytest.raises(ValueError):release_lib.runtime_config({**config(),'AUTH_DB_PATH':'/tmp/unsafe'})

@pytest.mark.parametrize('value',['bad; command','../release','/absolute','x\ncommand'])
def test_release_id_rejects_shell_and_path_inputs(value):
    with pytest.raises(ValueError):release_lib.valid_id(value)


def test_sqlite_backup_reads_committed_wal_without_stopping_writer(tmp_path):
    source=tmp_path/'source.db';target=tmp_path/'backup.db'
    with sqlite3.connect(source) as db:
        db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE records (value TEXT)');db.execute("INSERT INTO records VALUES ('keep')");db.commit()
        release_lib.sqlite_copy(source,target)
        db.execute("INSERT INTO records VALUES ('after')");db.commit()
    with sqlite3.connect(target) as db:assert db.execute('SELECT value FROM records').fetchall()==[('keep',)]


def fake_project(tmp_path):
    root=tmp_path/'project';data=root/'data/current';root.mkdir();data.mkdir(parents=True)
    for package in ['askiisc_retrieval','askiisc_serving']:
        (root/package).mkdir();(root/package/'__init__.py').write_text('# source\n')
    (root/'config').mkdir();(root/'config/faculty_directory.json').write_text('{"faculty":[]}')
    private=root/'.local/private';private.mkdir(parents=True)
    (private/'original.pdf').write_bytes(b'private source');(private/'reviewed.html').write_text('private extraction')
    doc={'id':'test-document','access_scope':'student','original':'.local/private/original.pdf','extracted':'.local/private/reviewed.html',
         'sha256':release_lib.sha(private/'original.pdf'),'extracted_sha256':release_lib.sha(private/'reviewed.html')}
    (root/'config/restricted_documents.json').write_text(json.dumps({'documents':[doc]}))
    (root/'config/local.json').write_text(json.dumps({'private':'must not package'}))
    (root/'.local/secret.json').write_text('SMTP_SECRET_NEVER_PACKAGE')
    (data/'auth').mkdir();(data/'auth/sessions.sqlite3').write_bytes(b'LOGIN_SESSION_NEVER_PACKAGE')
    (data/'snapshot.json').write_text(json.dumps({'embedding_model':release_lib.MODEL,'snapshot_id':'test-snapshot','active_version_ids':['v1']}))
    (data/'chunks.jsonl').write_text('{"access_scope":"student","source_url":"/v1/documents/test-document","version_id":"v1"}\n')
    (data/'graph.json').write_text('[]');(data/'qdrant/collection/askiisc_chunks').mkdir(parents=True)
    (data/'qdrant/meta.json').write_text('{}')
    with sqlite3.connect(data/'qdrant/collection/askiisc_chunks/storage.sqlite') as db:db.execute('CREATE TABLE test (n INTEGER)')
    model=tmp_path/('a'*40);model.mkdir();(model/'model.safetensors').write_bytes(b'model');(model/'config.json').write_text('{}')
    return root,data,model


def test_package_is_self_contained_and_excludes_machine_credentials_sessions_and_local_profile(tmp_path):
    root,data,model=fake_project(tmp_path);before=(root/'config/local.json').read_bytes()
    out=awsctl.package(root,data,tmp_path/'output',model,'test-release')
    target=tmp_path/'unpacked';manifest=release_lib.unpack(out/'bundle.tar.gz',target)
    assert manifest['scopes']=={'public':0,'student':1,'faculty':0}
    assert (target/'.local/private/original.pdf').read_bytes()==b'private source'
    assert not (target/'config/local.json').exists() and not (target/'data/auth').exists()
    assert not (target/'.local/secret.json').exists()
    assert (root/'config/local.json').read_bytes()==before
    assert (data/'auth/sessions.sqlite3').read_bytes()==b'LOGIN_SESSION_NEVER_PACKAGE'
    assert manifest['embedding_revision']=='a'*40
    (target/'askiisc_retrieval/__init__.py').write_text('changed')
    with pytest.raises(ValueError,match='checksum'):release_lib.verify_tree(target)


@pytest.mark.parametrize('name,kind',[('../escape','file'),('/absolute','file'),('link','symlink'),('fifo','fifo')])
def test_release_extraction_rejects_unsafe_members(tmp_path,name,kind):
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        m=tarfile.TarInfo(name)
        if kind=='symlink':m.type=tarfile.SYMTYPE;m.linkname='/etc/passwd'
        elif kind=='fifo':m.type=tarfile.FIFOTYPE
        else:m.size=1
        t.addfile(m,io.BytesIO(b'x') if kind=='file' else None)
    with pytest.raises(ValueError,match='Unsafe archive'):release_lib.unpack(archive,tmp_path/'tree')


def test_cloud_runtime_uses_stable_auth_and_explicit_aws_paths(tmp_path):
    release=tmp_path/'release';release.mkdir();(release/'release.json').write_text('{"release_id":"one"}')
    env=runtime.environment(release,config(),tmp_path/'state','chat.example.org')
    assert env['AUTH_DB_PATH']==str(tmp_path/'state/auth/sessions.sqlite3')
    assert env['ASKIISC_DATA_DIR']==str(tmp_path/'state/indexes/one')
    assert env['AUTH_COOKIE_SECURE']=='1' and env['AUTH_PUBLIC_ORIGIN']=='https://chat.example.org'
    assert env['EMAIL_AUTH_ENABLED']=='0' and env['AUTH_PROFESSOR_TEST_EMAILS']==''
    assert env['HF_HUB_OFFLINE']=='1' and env['REGENERATE']=='0'
    assert env['TZ']=='Asia/Kolkata'


def test_provisioning_enforces_private_storage_and_only_web_ingress():
    t=render_stack.template();r=t['Resources']
    assert json.loads((ROOT/'deploy/aws/stack.json').read_text())==t
    assert not any('NatGateway' in x['Type'] or 'LoadBalancer' in x['Type'] for x in r.values())
    assert {x['FromPort'] for x in r['SecurityGroup']['Properties']['SecurityGroupIngress']}=={80,443}
    assert r['DataVolume']['Properties']['Encrypted'] and r['DataVolume']['DeletionPolicy']=='Retain'
    assert r['ReleaseBucket']['Properties']['PublicAccessBlockConfiguration']['BlockPublicPolicy']
    assert r['AppInstance']['Properties']['MetadataOptions']['HttpTokens']=='required'
    assert r['AppInstance']['Properties']['CreditSpecification']['CPUCredits']=='standard'
    assert not any(x['Type'].startswith('AWS::IAM::') for x in r.values())
    assert r['AppInstance']['Properties']['IamInstanceProfile']=={'Ref':'InstanceProfileName'}
    docs=iam_bootstrap.documents('123456789012','ap-south-1','askiisc-prod')
    policies=docs['instance-policy']['Statement']
    assert any(x['Effect']=='Deny' and 'NotResource' in x for x in policies)
    assert any(x['Action']=='s3:GetObject' and '/releases/*' in x['Resource'] for x in policies)
    for name in ['execution-policy','deployer-policy']:
        allowed=[a for statement in docs[name]['Statement'] if statement['Effect']=='Allow'
                 for a in ([statement['Action']] if isinstance(statement['Action'],str) else statement['Action'])]
        assert not any(a.startswith(('iam:Create','iam:Put','iam:Attach','iam:Update')) for a in allowed)
        assert 'freetier:UpgradeAccountPlan' not in allowed
    guard=next(s for s in docs['execution-policy']['Statement'] if s['Effect']=='Deny' and s['Action']=='ec2:RunInstances')
    assert guard['Condition']['StringNotEquals']['ec2:InstanceType']=='t3.small'
    # RunInstances also authorizes AMI, subnet, ENI and volume resources, which lack this context key.
    assert guard['Resource']=='arn:aws:ec2:ap-south-1:123456789012:instance/*'


def test_activation_failure_restores_previous_release_without_replacing_auth(tmp_path,monkeypatch):
    state=tmp_path/'state';(state/'releases/old').mkdir(parents=True);(state/'releases/new').mkdir();(state/'auth').mkdir();(state/'backups').mkdir()
    old=state/'releases/old';new=state/'releases/new';(state/'current').symlink_to(old)
    db=state/'auth/sessions.sqlite3'
    with sqlite3.connect(db) as c:c.execute('CREATE TABLE sessions (n INTEGER)');c.execute('INSERT INTO sessions VALUES (123)')
    for release in [old,new]:(release/'cfg.json').write_text(json.dumps({'SERVED_MODEL':release.name}))
    monkeypatch.setattr(host,'STATE',state);monkeypatch.setattr(host,'cfg_for',lambda r:r/'cfg.json')
    commands=[];monkeypatch.setattr(host,'run',lambda *args,**kw:commands.append(args))
    def health(port,model,**kw):
        if model=='new':raise RuntimeError('candidate passed, final process failed')
    monkeypatch.setattr(host,'health',health)
    with pytest.raises(RuntimeError):host.activate(new)
    assert (state/'current').resolve()==old
    with sqlite3.connect(db) as c:assert c.execute('SELECT n FROM sessions').fetchone()==(123,)
    assert list((state/'backups').glob('*.sqlite3'))
    assert commands[-1]==('systemctl','start','askiisc.service')


def test_remote_controller_shell_checksums_keep_the_newline(tmp_path,monkeypatch):
    folder=tmp_path/'artifact';folder.mkdir()
    for name in ['bundle.tar.gz','host.py','release_lib.py']:(folder/name).write_bytes(b'x')
    descriptor={'release_id':'release-1','archive':'bundle.tar.gz','sha256':release_lib.sha(folder/'bundle.tar.gz'),
                'controls':{n:release_lib.sha(folder/n) for n in ['host.py','release_lib.py']}}
    (folder/'artifact.json').write_text(json.dumps(descriptor))
    monkeypatch.setattr(awsctl,'outputs',lambda *a:{'ReleaseBucket':'test-bucket','InstanceId':'i-test','ApplicationUrl':'https://chat.example.org'})
    monkeypatch.setattr(awsctl,'upload',lambda *a:None)
    commands=[];monkeypatch.setattr(awsctl,'remote',lambda *a:commands.append(a[-1]))
    awsctl.deploy('ap-south-1','test',folder/'artifact.json')
    checksum_line=next(x for x in commands[0].splitlines() if x.startswith('printf'))
    # Execute the exact format against a real local checksum, without AWS calls.
    actual=checksum_line.replace('/var/lib/askiisc/control/'+descriptor['controls']['host.py']+'/host.py',str(folder/'host.py'))
    import subprocess
    subprocess.run(['bash','-c',actual],check=True,capture_output=True)

@pytest.mark.parametrize('body,valid',[({'status':'answered','citations':[{'used':True}]},True),({'status':'unavailable','citations':[]},False),({'status':'answered','citations':[]},False)])
def test_candidate_probe_requires_a_cited_public_completion(monkeypatch,body,valid):
    import contextlib
    requests=[]
    def open_(request,**kwargs):
        requests.append(json.loads(request.data))
        return contextlib.closing(io.BytesIO(json.dumps(body).encode()))
    monkeypatch.setattr(host,'urlopen',open_)
    if valid:host.inference_probe(8101)
    else:
        with pytest.raises(RuntimeError):host.inference_probe(8101)
    assert requests==[{'question':'What is DS252 about?'}]


def test_runtime_preflight_suppresses_bytecode_in_immutable_release():
    source=(ROOT/'deploy/aws/runtime.py').read_text()
    assert source.index('sys.dont_write_bytecode=True')<source.index('from release_lib import')


def test_root_owned_environment_becomes_readable_without_chmod_of_os_python(tmp_path,monkeypatch):
    env=tmp_path/'venv';env.mkdir(mode=0o700);(env/'lib.py').write_text('# lib');(env/'lib.py').chmod(0o600)
    binary=tmp_path/'os-python';binary.write_text('os');binary.chmod(0o700)
    (env/'python').symlink_to(binary)
    script=env/'pip';script.write_text('#!/bin/sh\n');script.chmod(0o700)
    monkeypatch.setattr(host.shutil,'chown',lambda *a,**kw:None)
    host.readable_tree(env)
    assert env.stat().st_mode&0o777==0o750
    assert (env/'lib.py').stat().st_mode&0o777==0o640
    assert script.stat().st_mode&0o777==0o750
    assert binary.stat().st_mode&0o777==0o700


def test_release_verification_resolves_mac_temporary_directory_aliases(tmp_path):
    root,data,model=fake_project(tmp_path)
    out=awsctl.package(root,data,tmp_path/'output',model,'alias-test')
    tree=tmp_path/'physical-root';expected=release_lib.unpack(out/'bundle.tar.gz',tree)
    alias=tmp_path/'mac-style-link';alias.symlink_to(tree,target_is_directory=True)
    assert release_lib.verify_tree(alias)==expected


def test_public_ip_https_bootstrap_has_public_issuer_and_no_resource_cycle(tmp_path):
    import re,subprocess
    t=render_stack.template();r=t['Resources']
    assert t['Parameters']['DomainName']['Default']==''
    assert t['Outputs']['ApplicationUrl']['Value']['Fn::If'][2]=={'Fn::Sub':'https://${ElasticIp}'}
    deps={name:set() for name in r}
    def refs(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k=='Ref' and v in r:yield v
                elif k=='Fn::GetAtt' and v[0] in r:yield v[0]
                elif k=='Fn::Sub' and isinstance(v,str):
                    for ref in re.findall(r'\$\{([^}]+)\}',v):
                        if ref.split('.')[0] in r:yield ref.split('.')[0]
                else:yield from refs(v)
        elif isinstance(value,list):
            for item in value:yield from refs(item)
    for name,value in r.items():
        deps[name].update(refs(value));explicit=value.get('DependsOn',[])
        deps[name].update([explicit] if isinstance(explicit,str) else explicit)
    visited=set()
    def visit(name,path):
        assert name not in path,'CloudFormation dependency cycle'
        if name in visited:return
        for dep in deps[name]:visit(dep,path|{name})
        visited.add(name)
    for name in r:visit(name,set())
    source=(ROOT/'deploy/aws/bootstrap.sh').read_text()
    # Execute only the host/TLS selection and Caddyfile rendering, never host installation.
    selection=source[source.index("SITE_HOST='"):source.index('cat > /etc/askiisc/host.json')]
    block=source[source.index('cat > /etc/caddy/Caddyfile'):source.index('chmod 0644 /etc/caddy/Caddyfile')]
    for domain in ['', 'chat.example.org']:
        target=tmp_path/('ip.Caddyfile' if not domain else 'dns.Caddyfile')
        script=(selection+block).replace('@@DOMAIN@@',domain).replace('@@PUBLIC_IP@@','203.0.113.10').replace('/etc/caddy/Caddyfile',str(target))
        subprocess.run(['bash','-eu','-c',script],check=True)
        rendered=target.read_text()
        assert 'tls internal' not in rendered
        assert ('profile shortlived' in rendered)==(not domain)
        assert ('https://acme-v02.api.letsencrypt.org/directory' in rendered)==(not domain)
        assert 'default_sni '+(domain or '203.0.113.10') in rendered
        assert '\n'+(domain or '203.0.113.10')+' {' in rendered


def test_private_model_config_protects_secrets_and_clears_key_on_endpoint_change(tmp_path):
    smtp=tmp_path/'email.json';smtp.write_text(json.dumps({k:'private-smtp' for k in model_config.SMTP_KEYS}))
    path=tmp_path/'aws-runtime.json'
    first=model_config.configure(path,email_path=smtp,key='private-model-key')
    assert first['MODEL_PROVIDER_PREFERENCES']==model_config.FREE_PRIVATE
    assert first['AUTH_PROFESSOR_TEST_EMAILS']==''
    assert path.stat().st_mode&0o777==0o600
    public=json.dumps(model_config.summary(first))
    assert 'private-model-key' not in public and 'private-smtp' not in public
    changed=model_config.configure(path,url='https://other.example.org/v1',model='other/model')
    assert changed['MODEL_API_KEY']=='' and changed['MODEL_PROVIDER_PREFERENCES']=={}
    assert changed['SMTP_PASSWORD']==first['SMTP_PASSWORD']
    before=path.read_bytes()
    with pytest.raises(ValueError):model_config.configure(path,url='http://unsafe.example.org')
    assert path.read_bytes()==before
    with pytest.raises(ValueError):release_lib.runtime_config({**config(),'MODEL_PROVIDER_PREFERENCES':[]})


def test_model_probe_accepts_keyless_self_hosted_endpoint_without_auth_header(monkeypatch):
    import contextlib
    seen=[]
    class Opener:
        def open(self,request,**kwargs):
            seen.append(request)
            return contextlib.closing(io.BytesIO(json.dumps({'model':'local/model','choices':[{'message':{'content':'ready'},'finish_reason':'stop'}]}).encode()))
    monkeypatch.setattr(model_config,'build_opener',lambda *args:Opener())
    c=release_lib.runtime_config({**config(),'VLLM_URL':'https://tunnel.example.org/v1','MODEL_API_KEY':''})
    assert model_config.check(c)['public_probe']=='passed'
    assert 'Authorization' not in seen[0].headers
    assert seen[0].full_url=='https://tunnel.example.org/v1/chat/completions'
    with pytest.raises(ValueError,match='API key is missing'):
        model_config.check({**c,'VLLM_URL':'https://openrouter.ai/api/v1'})


def test_runtime_preflight_runs_as_service_user_to_avoid_root_owned_qdrant_lock(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(host.pwd,'getpwnam',lambda name:SimpleNamespace(pw_uid=123,pw_gid=456))
    assert host.identity()=={'user':123,'group':456,'extra_groups':[]}
    calls=[]
    monkeypatch.setattr(host,'identity',lambda:{'user':123,'group':456})
    class Result:stdout=None
    monkeypatch.setattr(host.subprocess,'run',lambda args,**kwargs:(calls.append(kwargs) or Result()))
    host.run('python','runtime.py','--check',as_app=True)
    assert calls[0]['user']==123 and calls[0]['group']==456
    host.run('systemctl','start','askiisc.service')
    assert 'user' not in calls[1]
    import inspect
    assert 'as_app=True' in inspect.getsource(host.check)


@pytest.mark.parametrize('scope', ['faculty', None])
def test_packaging_refuses_stale_private_catalog(tmp_path, scope):
    root, data, model = fake_project(tmp_path)
    rows = [] if scope is None else [{'access_scope':scope, 'source_url':'/v1/documents/test-document','version_id':'v1'}]
    (data/'chunks.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match='catalog and selected snapshot'):
        awsctl.package(root, data, tmp_path/'output', model, 'stale')
