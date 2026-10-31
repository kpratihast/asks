#!/usr/bin/env python3
"""GitHub Actions client: upload code only and deploy to the existing stack using OIDC."""
from __future__ import annotations
import argparse,json,re,shlex,tempfile
from pathlib import Path
import awsctl,ci_bundle
from deploy_latest import check_account
from release_lib import sha

CONTROLS=['ci_host.py','ci_bundle.py','host.py','release_lib.py']


def deploy(root,region,account,stack,commit,run_id,attempt):
    if not re.fullmatch('[0-9a-f]{40}',commit) or not re.fullmatch(r'\d+',run_id) or not re.fullmatch(r'\d+',attempt):raise ValueError('Invalid CI run identity')
    rid=f'ci-{commit[:12]}-{run_id}-{attempt}';check_account(region,account)
    out=awsctl.outputs(region,stack);bucket=out['ReleaseBucket'];prefix='releases/ci/'+rid+'/'
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp);tree=work/'code';ci_bundle.package(root,tree,commit,rid)
        archive=work/'code.tar.gz';ci_bundle.archive(tree,archive)
        awsctl.upload(region,bucket,prefix+'code.tar.gz',archive)
        for name in CONTROLS:awsctl.upload(region,bucket,prefix+name,tree/'deploy/aws'/name)
        control='/var/lib/askiisc/control/'+rid
        commands=['install -d -m 0700 '+shlex.quote(control)]
        for name in CONTROLS:
            source=tree/'deploy/aws'/name;target=control+'/'+name
            commands+=['aws --region '+shlex.quote(region)+' s3 cp '+shlex.quote('s3://'+bucket+'/'+prefix+name)+' '+shlex.quote(target)+' --only-show-errors',
                       "printf '%s\\n' "+shlex.quote(sha(source)+'  '+target)+' | sha256sum -c -']
        commands+=['PYTHONDONTWRITEBYTECODE=1 python3.12 '+shlex.quote(control+'/ci_host.py')+' --bucket '+shlex.quote(bucket)+' --key '+shlex.quote(prefix+'code.tar.gz')+' --sha256 '+sha(archive)]
        awsctl.remote(region,out['InstanceId'],'\n'.join(commands))
        awsctl.smoke(out['ApplicationUrl'])
        print(json.dumps({'release':rid,'commit':commit,'url':out['ApplicationUrl'],'new_infrastructure':False}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--account',required=True);p.add_argument('--region',default='ap-south-1');p.add_argument('--stack',default='askiisc-prod');p.add_argument('--commit',required=True);p.add_argument('--run-id',required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
    deploy(Path(__file__).resolve().parents[2],a.region,a.account,a.stack,a.commit,a.run_id,a.attempt)
if __name__=='__main__':main()
