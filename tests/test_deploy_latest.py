import json
import os
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'deploy/aws'))
import deploy_latest
from test_aws_deployment import fake_project
import awsctl


def test_latest_uses_descriptor_time_not_lexical_release_name_and_never_silently_skips_corruption(tmp_path):
    old = tmp_path/'z-release/artifact.json';old.parent.mkdir();old.write_text('{}')
    new = tmp_path/'a-release/artifact.json';new.parent.mkdir();new.write_text('bad json')
    os.utime(old, ns=(10,10));os.utime(new, ns=(20,20))
    assert deploy_latest.latest_artifact(tmp_path) == new
    with pytest.raises(ValueError):deploy_latest.verify_artifact(new)
    with pytest.raises(ValueError, match='No bundles'):deploy_latest.latest_artifact(tmp_path/'missing')


def test_dry_run_verifies_every_archive_file_and_makes_no_aws_calls(tmp_path, monkeypatch):
    root,data,model=fake_project(tmp_path)
    out=awsctl.package(root,data,tmp_path/'out',model,'release-test')
    monkeypatch.setattr(awsctl,'aws',lambda *args:pytest.fail('Dry run called AWS'))
    monkeypatch.setattr(awsctl,'deploy',lambda *args:pytest.fail('Dry run deployed'))
    deploy_latest.main(['--artifact',str(out/'artifact.json'),'--dry-run'])
    (out/'host.py').write_text('corrupt controller')
    with pytest.raises(ValueError, match='checksum'):
        deploy_latest.main(['--artifact',str(out/'artifact.json'),'--dry-run'])


@pytest.mark.parametrize('account,arn,plan,credits', [
    ('wrong','arn:aws:iam::800168266746:user/deployer','FREE',100),
    ('800168266746','arn:aws:iam::800168266746:root','FREE',100),
    ('800168266746','arn:aws:iam::800168266746:user/deployer','PAID',100),
    ('800168266746','arn:aws:iam::800168266746:user/deployer','FREE',0),
])
def test_account_guard_refuses_wrong_account_root_paid_plan_or_no_credits(monkeypatch,account,arn,plan,credits):
    def call(region,service,*args):
        if service=='sts':return {'Account':account,'Arn':arn}
        return {'accountId':'800168266746','accountPlanType':plan,'accountPlanStatus':'ACTIVE',
                'accountPlanRemainingCredits':{'unit':'USD','amount':credits}}
    monkeypatch.setattr(awsctl,'aws',call)
    with pytest.raises(ValueError):deploy_latest.check_account('ap-south-1','800168266746')


def test_release_path_preflights_before_mutation_and_smokes_after_activation(tmp_path,monkeypatch):
    root,data,model=fake_project(tmp_path)
    out=awsctl.package(root,data,tmp_path/'out',model,'release-test')
    for key in ['AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_SESSION_TOKEN']:monkeypatch.delenv(key,raising=False)
    calls=[]
    monkeypatch.setattr(deploy_latest,'check_account',lambda *args:calls.append('account'))
    monkeypatch.setattr(awsctl,'outputs',lambda *args:{'ApplicationUrl':'https://app.example.org'})
    monkeypatch.setattr(awsctl,'smoke',lambda *args:calls.append('smoke'))
    monkeypatch.setattr(awsctl,'deploy',lambda *args:calls.append('deploy'))
    deploy_latest.main(['--artifact',str(out/'artifact.json'),'--profile','test-deployer'])
    assert calls==['account','smoke','deploy','smoke']
    assert os.environ['AWS_PROFILE']=='test-deployer'
    calls.clear();monkeypatch.setenv('AWS_ACCESS_KEY_ID','redacted-test-key')
    with pytest.raises(ValueError,match='Unset exported'):
        deploy_latest.main(['--artifact',str(out/'artifact.json')])
    assert not calls
