#!/usr/bin/env python3
"""Prepare exact GitHub OIDC policies; --apply requires an explicit administrator profile."""
from __future__ import annotations
import argparse,json,re,subprocess
from pathlib import Path
from iam_bootstrap import policy,allow


def documents(account,region,stack,instance,repo,owner_id,repo_id):
    if not re.fullmatch(r'\d{12}',account) or not re.fullmatch(r'[a-z]{2}-[a-z]+-\d',region):raise ValueError('Invalid AWS account/region')
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,25}',stack) or not re.fullmatch(r'i-[0-9a-f]{8,17}',instance):raise ValueError('Invalid stack/instance')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repo) or not owner_id.isdigit() or not repo_id.isdigit():raise ValueError('Expected exact repository and immutable numeric IDs')
    owner,name=repo.split('/');issuer='token.actions.githubusercontent.com';provider=f'arn:aws:iam::{account}:oidc-provider/{issuer}'
    subject=f'repo:{owner}@{owner_id}/{name}@{repo_id}:ref:refs/heads/*'
    trust=policy([{'Effect':'Allow','Principal':{'Federated':provider},'Action':'sts:AssumeRoleWithWebIdentity',
        'Condition':{'StringEquals':{issuer+':aud':'sts.amazonaws.com'},'StringLike':{issuer+':sub':subject}}}])
    objects=f'arn:aws:s3:::{stack}-{account}-{region}-releases/releases/ci/*'
    scoped=policy([
        {'Effect':'Deny','Action':'freetier:UpgradeAccountPlan','Resource':'*'},
        allow('freetier:GetAccountPlanState','*'),
        allow('cloudformation:DescribeStacks',f'arn:aws:cloudformation:{region}:{account}:stack/{stack}/*'),
        allow(['s3:GetObject','s3:PutObject'],objects),
        allow('ssm:SendCommand',f'arn:aws:ec2:{region}:{account}:instance/{instance}'),
        allow('ssm:SendCommand',f'arn:aws:ssm:{region}::document/AWS-RunShellScript'),
        allow('ssm:GetCommandInvocation','*')])
    node=policy([allow('s3:PutObject',objects)])
    return {'github-trust':trust,'github-permissions':scoped,'instance-ci-upload':node}


def apply(args,docs):
    def aws(*parts,missing=False):
        r=subprocess.run(['aws','--profile',args.admin_profile,'--region',args.region,*parts,'--output','json','--no-cli-pager'],text=True,capture_output=True)
        if r.returncode:
            if missing and 'NoSuchEntity' in r.stderr:return None
            raise RuntimeError(r.stderr.strip())
        return json.loads(r.stdout or '{}')
    identity=aws('sts','get-caller-identity')
    if identity['Account']!=args.account:raise ValueError('Wrong administrator account')
    provider=f'arn:aws:iam::{args.account}:oidc-provider/token.actions.githubusercontent.com'
    existing=aws('iam','get-open-id-connect-provider','--open-id-connect-provider-arn',provider,missing=True)
    if existing:
        if existing['Url'] not in {'token.actions.githubusercontent.com','https://token.actions.githubusercontent.com'} or 'sts.amazonaws.com' not in existing['ClientIDList']:raise ValueError('Existing GitHub identity provider has unexpected settings; review it first')
    else:aws('iam','create-open-id-connect-provider','--url','https://token.actions.githubusercontent.com','--client-id-list','sts.amazonaws.com')
    role=args.stack+'-github';existing=aws('iam','get-role','--role-name',role,missing=True)
    if existing:
        if existing['Role']['AssumeRolePolicyDocument']!=docs['github-trust']:raise ValueError('Existing role trust differs; review before changing it')
        current=aws('iam','get-role-policy','--role-name',role,'--policy-name','AskIIScGitHub',missing=True)
        if current and current['PolicyDocument']!=docs['github-permissions']:raise ValueError('Existing deployment permissions differ; review before changing them')
    else:aws('iam','create-role','--role-name',role,'--assume-role-policy-document','file://'+str(args.out/'github-trust.json'),'--max-session-duration','3600','--description','Code deployment from the private AskIISc GitHub repository')
    aws('iam','put-role-policy','--role-name',role,'--policy-name','AskIIScGitHub','--policy-document','file://'+str(args.out/'github-permissions.json'))
    current=aws('iam','get-role-policy','--role-name',args.stack+'-instance','--policy-name','AskIIScCIUpload',missing=True)
    if current and current['PolicyDocument']!=docs['instance-ci-upload']:raise ValueError('Existing instance CI upload policy differs')
    aws('iam','put-role-policy','--role-name',args.stack+'-instance','--policy-name','AskIIScCIUpload','--policy-document','file://'+str(args.out/'instance-ci-upload.json'))
    print(json.dumps({'role_arn':f'arn:aws:iam::{args.account}:role/{role}','oidc_setup':'complete','provisioned_compute':False},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--account',required=True);p.add_argument('--region',default='ap-south-1');p.add_argument('--stack',default='askiisc-prod');p.add_argument('--instance',required=True);p.add_argument('--repo',required=True);p.add_argument('--owner-id',required=True);p.add_argument('--repo-id',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--admin-profile');p.add_argument('--apply',action='store_true');a=p.parse_args()
    docs=documents(a.account,a.region,a.stack,a.instance,a.repo,a.owner_id,a.repo_id);a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    for name,document in docs.items():(a.out/(name+'.json')).write_text(json.dumps(document,indent=2)+'\n')
    if a.apply:
        if not a.admin_profile:raise ValueError('--apply requires an explicit --admin-profile')
        apply(a,docs)
    else:print('Prepared policies only; no AWS calls:',a.out)
if __name__=='__main__':main()
