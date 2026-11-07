"""Code-only packaging, asset preservation, OIDC scope and pipeline safety."""
import json,re,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deploy/aws'))
import ci_bundle,ci_setup,ci,ci_host,release_lib


def project(root):
    for package in ['askiisc_retrieval','askiisc_serving']:
        (root/package).mkdir(parents=True);(root/package/'__init__.py').write_text('# app\n')
    (root/'askiisc_retrieval/graph_rules.py').write_text('GRAPH_RULES_VERSION = "test-rules"\n')
    (root/'config').mkdir();(root/'config/faculty_directory.json').write_text('{}')
    (root/'config/restricted_documents.json').write_text('{"documents":[]}')
    (root/'deploy/aws').mkdir(parents=True)
    for name in ['host.py','release_lib.py','runtime.py','requirements.lock']:(root/'deploy/aws'/name).write_text('# runtime\n')
    return root


def test_code_package_excludes_secrets_indexes_model_and_private_inputs(tmp_path):
    root=project(tmp_path/'project')
    for name in ['.env','config/local.json','config/aws-models.json','.local/private/restricted.pdf','data/chunks.jsonl','assets/model.safetensors','output/bundle.tar.gz']:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('SENSITIVE')
    tree=tmp_path/'code';m=ci_bundle.package(root,tree,'a'*40,'ci-test')
    assert all(ci_bundle.allowed(n) for n in m['files'])
    assert not any(b'SENSITIVE' in p.read_bytes() for p in tree.rglob('*') if p.is_file())
    (root/'askiisc_serving/link.py').symlink_to(root/'.env')
    with pytest.raises(ValueError,match='links'):ci_bundle.package(root,tmp_path/'other','a'*40,'ci-other')


@pytest.mark.parametrize('path',['data/chunks.jsonl','assets/model.py','.local/private/a.py','config/local.json','../escape.py','/tmp/escape.py','askiisc_serving/.hidden.py','askiisc_serving/__pycache__/x.py','deploy/aws/unknown.py'])
def test_host_rejects_code_paths_outside_allowlist(path):
    assert not ci_bundle.allowed(path)


def test_compose_retains_frozen_private_assets_and_removes_old_code(tmp_path):
    base=project(tmp_path/'base');(base/'askiisc_serving/obsolete.py').write_text('old')
    retained={'data/chunks.jsonl':'frozen-index','data/snapshot.json':'{"graph_version":"test-rules"}',
              'assets/model.safetensors':'frozen-model','.local/private/exam.pdf':'private-document'}
    for name,body in retained.items():
        p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)
    document={'original':'.local/private/exam.pdf','extracted':'.local/private/exam.pdf','sha256':release_lib.sha(base/'.local/private/exam.pdf'),'extracted_sha256':release_lib.sha(base/'.local/private/exam.pdf')}
    (base/'config/restricted_documents.json').write_text(json.dumps({'documents':[document]}))
    ci_bundle.manifest(base,dict(release_id='base',snapshot='snap',embedding_model='BGE',embedding_revision='b'*40,chunks=1,scopes={'faculty':1}))
    source=project(tmp_path/'source');(source/'config/restricted_documents.json').write_bytes((base/'config/restricted_documents.json').read_bytes())
    (source/'askiisc_serving/__init__.py').write_text('# updated\n')
    code=tmp_path/'code';ci_bundle.package(source,code,'a'*40,'ci-new')
    target=tmp_path/'full';m=ci_bundle.compose(base,code,target)
    assert m['base_release']=='base' and m['git_sha']=='a'*40 and m['snapshot']=='snap'
    assert (target/'askiisc_serving/__init__.py').read_text()=='# updated\n'
    assert not (target/'askiisc_serving/obsolete.py').exists()
    assert {name:(target/name).read_text() for name in retained}==retained
    (source/'config/restricted_documents.json').write_text(json.dumps({'documents':[{**document,'sha256':'0'*64}]}))
    other=tmp_path/'bad-code';ci_bundle.package(source,other,'c'*40,'ci-other')
    with pytest.raises(ValueError,match='full, reviewed'):ci_bundle.compose(base,other,tmp_path/'bad-full')
    assert release_lib.verify_tree(base)['release_id']=='base'
    (source/'askiisc_retrieval/graph_rules.py').write_text('GRAPH_RULES_VERSION = "new-rules"\n')
    newer=tmp_path/'new-code';ci_bundle.package(source,newer,'d'*40,'ci-new-rules')
    incompatible=tmp_path/'bad-graph'
    with pytest.raises(ValueError,match='Rebuild'):ci_bundle.compose(base,newer,incompatible)
    assert not incompatible.exists()


def test_oidc_trust_pins_immutable_repository_ids_and_branch_refs():
    docs=ci_setup.documents('123456789012','ap-south-1','askiisc-prod','i-0123456789abcdef0','owner/repo','123','456')
    trust=docs['github-trust']['Statement'][0]
    assert trust['Condition']['StringLike']=={'token.actions.githubusercontent.com:sub':'repo:owner@123/repo@456:ref:refs/heads/*'}
    assert trust['Condition']['StringEquals']=={'token.actions.githubusercontent.com:aud':'sts.amazonaws.com'}
    statements=docs['github-permissions']['Statement']
    allowed={a for s in statements if s['Effect']=='Allow' for a in ([s['Action']] if isinstance(s['Action'],str) else s['Action'])}
    assert not any(a.startswith(('iam:','ec2:','ssm:GetParameter','ssm:PutParameter','cloudformation:Create')) for a in allowed)
    assert any(s['Effect']=='Deny' and s['Action']=='freetier:UpgradeAccountPlan' for s in statements)
    assert docs['instance-ci-upload']['Statement'][0]['Resource'].endswith('/releases/ci/*')
    node=next(s for s in statements if s['Action']=='ssm:SendCommand' and ':instance/' in s['Resource'])
    assert node['Resource'].endswith('/i-0123456789abcdef0')


def test_pruning_keeps_manual_current_previous_and_recent_releases(tmp_path,monkeypatch):
    state=tmp_path/'state';(state/'releases').mkdir(parents=True);(state/'indexes').mkdir();configs=tmp_path/'configs';configs.mkdir()
    import os
    for i in range(8):
        release=state/'releases'/f'ci-{i}';release.mkdir()
        ci_bundle.manifest(release,{'release_id':release.name,'git_sha':'a'*40,'base_release':'manual'})
        os.utime(release,(100+i,100+i));(state/'indexes'/release.name).mkdir();(configs/(release.name+'.json')).write_text('{}')
    (state/'releases/manual').mkdir();(state/'current').symlink_to(state/'releases/ci-0');(state/'previous').symlink_to(state/'releases/ci-1')
    monkeypatch.setattr(ci_host.host,'STATE',state);monkeypatch.setattr(ci_host.host,'cfg_for',lambda r:configs/(r.name+'.json'))
    ci_host.prune(keep=3)
    assert {p.name for p in (state/'releases').iterdir()}=={'manual','ci-0','ci-1','ci-5','ci-6','ci-7'}
    assert not (state/'indexes/ci-2').exists() and not (configs/'ci-2.json').exists()


def test_workflow_pins_actions_and_deploys_branches_after_tests_with_one_lock():
    source=(ROOT/'.github/workflows/aws-validation.yml').read_text()
    import yaml
    workflow=yaml.load(source,Loader=yaml.BaseLoader)
    assert set(workflow['on'])=={'push','pull_request','workflow_dispatch'}
    assert workflow['jobs']['deploy']['needs']=='validate'
    assert "branches: ['**']" in source and 'pull_request_target' not in source
    assert "github.event_name != 'pull_request'" in source and "startsWith(github.ref, 'refs/heads/')" in source
    assert 'needs: validate' in source and "vars.AWS_DEPLOY_ENABLED == 'true'" in source
    assert 'group: askiisc-production-deploy' in source and 'cancel-in-progress: false' in source
    assert source.count('id-token: write')==1 and 'persist-credentials: false' in source
    actions=re.findall(r'uses: (\S+)',source)
    assert actions and all(re.fullmatch(r'[\w/-]+@[0-9a-f]{40}',a) for a in actions)
    assert '--require-hashes' in source and 'AWS_ACCESS_KEY_ID' not in source


def test_ci_uses_shared_deploy_lock_and_never_runs_ingestion():
    source=(ROOT/'deploy/aws/ci_host.py').read_text()
    assert "host.STATE/'deploy.lock'" in source and 'fcntl.LOCK_EX' in source
    assert 'host.install(' in source and 'ssm:get-parameter' not in source and 'build_index' not in source
