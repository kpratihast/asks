"""Model switching preserves auth, isolates credentials and deploys transactional candidates."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'deploy/aws'))
import model_profiles as models


def base():
    return models.runtime_config({'VLLM_URL':'https://old.example.org/v1','SERVED_MODEL':'old-model',
        'MODEL_API_KEY':'old-provider-key','EMAIL_AUTH_ENABLED':True,'AUTH_PROFESSOR_TEST_EMAILS':'prof@iisc.ac.in',
        'SMTP_HOST':'smtp.example.org','SMTP_USERNAME':'smtp-user','SMTP_PASSWORD':'private-smtp-secret','SMTP_FROM':'sender@example.org'})


def configuration(tmp_path):
    data=json.loads((Path(models.__file__).parent/'model-profiles.example.json').read_text())
    runtime=tmp_path/'runtime.json';models.atomic_json(runtime,base());data['runtime_file']=str(runtime)
    data['profiles']['local']['endpoint']='https://new.example.org'
    data['profiles']['openrouter']['api_key_file']=str(tmp_path/'openrouter.key')
    path=tmp_path/'profiles.json';models.atomic_json(path,data)
    return path,data


def test_switch_clears_other_provider_options_and_preserves_all_identity_settings(tmp_path):
    _,cfg=configuration(tmp_path)
    original=base();original['MODEL_REASONING']={'enabled':True};original['CHAT_TEMPLATE_KWARGS']={'old':True}
    local=models.candidate(original,cfg['profiles']['local'],'')
    assert local['VLLM_URL']=='https://new.example.org/v1' and local['MODEL_API_KEY']==''
    assert local['MODEL_REASONING']=={} and local['MODEL_PROVIDER_PREFERENCES']=={}
    assert local['CHAT_TEMPLATE_KWARGS']=={'enable_thinking':False}
    remote=models.candidate(local,cfg['profiles']['openrouter'],'new-provider-key')
    assert remote['CHAT_TEMPLATE_KWARGS']=={} and remote['MODEL_API_KEY']=='new-provider-key'
    assert remote['MODEL_PROVIDER_PREFERENCES']['zdr']
    for key in original.keys()-models.MODEL_KEYS:
        assert local[key]==remote[key]==original[key]


@pytest.mark.parametrize('url',['http://openrouter.ai/api/v1','https://openrouter.ai.evil.org/api/v1',
    'https://user:secret@openrouter.ai/api/v1','https://other.example.org/v1'])
def test_openrouter_key_never_goes_to_a_replacement_host(tmp_path,url):
    _,cfg=configuration(tmp_path);profile=cfg['profiles']['openrouter'];profile['endpoint']=url
    with pytest.raises(ValueError):models.candidate(base(),profile,'private-provider-key')


@pytest.mark.parametrize('alter',[
    lambda p:p['provider_preferences'].pop('zdr'),
    lambda p:p['provider_preferences'].update(data_collection='allow'),
    lambda p:p['provider_preferences'].update(max_price={'prompt':100,'completion':100,'request':0}),
    lambda p:p['provider_preferences']['max_price'].update(request=1),
])
def test_privacy_and_bounded_test_price_cannot_be_silently_relaxed(tmp_path,alter):
    _,cfg=configuration(tmp_path);profile=cfg['profiles']['openrouter'];alter(profile)
    with pytest.raises(ValueError):models.candidate(base(),profile,'key')


def test_private_key_storage_and_explicit_env_precedence(tmp_path,monkeypatch):
    _,cfg=configuration(tmp_path);profile=cfg['profiles']['openrouter']
    monkeypatch.delenv('OPENROUTER_API_KEY',raising=False)
    with pytest.raises(ValueError,match='key missing'):models.resolve_key(profile)
    assert models.resolve_key(cfg['profiles']['local'])==''
    models.save_key(profile,'private-key');path=Path(profile['api_key_file'])
    assert path.stat().st_mode&0o777==0o600 and models.resolve_key(profile)=='private-key'
    monkeypatch.setenv('OPENROUTER_API_KEY','environment-key')
    assert models.resolve_key(profile)=='environment-key'
    monkeypatch.delenv('OPENROUTER_API_KEY');path.chmod(0o644)
    with pytest.raises(ValueError,match='permissions 600'):models.resolve_key(profile)


def test_plain_config_rejects_embedded_credentials(tmp_path):
    path,cfg=configuration(tmp_path);cfg['profiles']['openrouter']['api_key']='must-not-store'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='API keys'):models.load_profiles(path)


def test_dry_run_has_no_network_no_aws_no_secret_output_or_config_mutation(tmp_path,monkeypatch,capsys):
    path,cfg=configuration(tmp_path);before=path.read_bytes();runtime=Path(cfg['runtime_file']).read_bytes()
    monkeypatch.setenv('OPENROUTER_API_KEY','secret-environment-key')
    monkeypatch.setattr(models,'public_probe',lambda *a:pytest.fail('Dry run called model'))
    monkeypatch.setattr(models,'deploy_profile',lambda *a:pytest.fail('Dry run deployed'))
    models.main(['openrouter','--config',str(path),'--dry-run'])
    printed=capsys.readouterr().out
    assert 'secret-environment-key' not in printed and 'private-smtp-secret' not in printed
    assert path.read_bytes()==before and Path(cfg['runtime_file']).read_bytes()==runtime


def test_one_time_model_test_makes_one_probe_and_never_changes_aws_or_active_profile(tmp_path,monkeypatch):
    path,cfg=configuration(tmp_path);before=path.read_bytes();runtime=Path(cfg['runtime_file']).read_bytes();calls=[]
    monkeypatch.setenv('OPENROUTER_API_KEY','test-key')
    monkeypatch.setattr(models,'public_probe',lambda *a:(calls.append(a) or {'status':'passed','generation_calls':1}))
    monkeypatch.setattr(models,'deploy_profile',lambda *a:pytest.fail('Check-only deployed'))
    report=tmp_path/'report.json'
    models.main(['openrouter','--config',str(path),'--check-only','--report',str(report)])
    assert len(calls)==1 and json.loads(report.read_text())['generation_calls']==1
    assert path.read_bytes()==before and Path(cfg['runtime_file']).read_bytes()==runtime


def setup_deploy(tmp_path,monkeypatch,fail=False,active_new=False):
    path,cfg=configuration(tmp_path);profile=cfg['profiles']['local'];original=base()
    new=models.candidate(original,profile,'');calls=[]
    for key in ['AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','AWS_SESSION_TOKEN']:monkeypatch.delenv(key,raising=False)
    monkeypatch.setattr(models,'check_account',lambda *a:calls.append('account'))
    monkeypatch.setattr(models.awsctl,'outputs',lambda *a:{'InstanceId':'instance','RuntimeParameter':'param','ApplicationUrl':'https://app.example.org'})
    def package(root,data,bundles,release_id):
        result=tmp_path/'out'/release_id;result.mkdir(parents=True);(result/'artifact.json').write_text('{}');calls.append('package');return result
    monkeypatch.setattr(models.awsctl,'package',package)
    monkeypatch.setattr(models,'verify_artifact',lambda *a:calls.append('verify'))
    count=[0]
    def metadata(*a):
        count[0]+=1
        current=next((tmp_path/'out').glob('*'),None) if (tmp_path/'out').exists() else None
        return {'version':3,'sha256':hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest(),
                'release':current.name if active_new and count[0]>1 else 'old-release'}
    monkeypatch.setattr(models,'runtime_metadata',metadata)
    def put(region,param,config):calls.append('put_new' if config==new else 'put_old');return 4
    monkeypatch.setattr(models,'put_runtime',put)
    def deploy(*a):
        calls.append('deploy')
        if fail:raise RuntimeError('candidate rejected')
    monkeypatch.setattr(models.awsctl,'deploy',deploy)
    monkeypatch.setattr(models.awsctl,'smoke',lambda *a:calls.append('smoke'))
    def invoke():return models.deploy_profile(path,cfg,'local',profile,original,new,tmp_path,tmp_path/'out',tmp_path/'data')
    return invoke,calls,path,cfg,new


def test_normal_switch_saves_profile_only_after_success_and_checks_new_not_old_endpoint(tmp_path,monkeypatch):
    invoke,calls,path,cfg,new=setup_deploy(tmp_path,monkeypatch)
    result=invoke()
    assert calls==['account','package','verify','put_new','deploy','smoke']
    assert json.loads(path.read_text())['active']=='local'
    assert json.loads(Path(cfg['runtime_file']).read_text())==new
    assert result['previous_release']=='old-release' and result['runtime_parameter_version']==4


def test_failed_candidate_restores_parameter_without_changing_profile_or_local_runtime(tmp_path,monkeypatch):
    invoke,calls,path,cfg,new=setup_deploy(tmp_path,monkeypatch,fail=True)
    before=path.read_bytes();runtime=Path(cfg['runtime_file']).read_bytes()
    with pytest.raises(RuntimeError,match='candidate rejected'):invoke()
    assert calls[-1]=='put_old' and 'smoke' not in calls
    assert path.read_bytes()==before and Path(cfg['runtime_file']).read_bytes()==runtime


def test_lost_ssm_response_after_activation_does_not_restore_old_parameter(tmp_path,monkeypatch):
    invoke,calls,path,cfg,new=setup_deploy(tmp_path,monkeypatch,fail=True,active_new=True)
    with pytest.raises(RuntimeError):invoke()
    assert 'put_old' not in calls
    assert json.loads(Path(cfg['runtime_file']).read_text())==new


def test_drifted_runtime_stops_before_packaging_or_putting_secrets(tmp_path,monkeypatch):
    invoke,calls,*_=setup_deploy(tmp_path,monkeypatch)
    monkeypatch.setattr(models,'runtime_metadata',lambda *a:{'version':99,'sha256':'different','release':'old'})
    with pytest.raises(ValueError,match='differs'):invoke()
    assert calls==['account']
