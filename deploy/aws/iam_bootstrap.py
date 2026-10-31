#!/usr/bin/env python3
"""Prepare or apply one-time app IAM setup. Routine deployment cannot modify these roles.

Run without --apply to write reviewable policies only. Uses an explicitly selected
administrator profile for IAM setup; no credentials or application data are copied.
"""
from __future__ import annotations
import argparse,json,re,subprocess,time
from pathlib import Path


def policy(statements): return {'Version':'2012-10-17','Statement':statements}
def allow(actions,resource,**kw): return {'Effect':'Allow','Action':actions,'Resource':resource,**kw}


def documents(account,region,stack,parameter='/askiisc/prod/runtime'):
    if not re.fullmatch(r'\d{12}',account) or not re.fullmatch(r'[a-z]{2}-[a-z]+-\d',region):raise ValueError('Invalid account/region')
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,25}',stack):raise ValueError('Use a short lowercase stack name')
    if not re.fullmatch(r'/[a-zA-Z0-9_./-]+',parameter):raise ValueError('Invalid parameter name')
    iam=f'arn:aws:iam::{account}';stackarn=f'arn:aws:cloudformation:{region}:{account}:stack/{stack}/*'
    bucket=f'{stack}-{account}-{region}-releases';s3=f'arn:aws:s3:::{bucket}'
    param=f'arn:aws:ssm:{region}:{account}:parameter{parameter}'
    instance=f'{stack}-instance';execution=f'{stack}-cloudformation'
    kms={'StringEquals':{'kms:ViaService':f'ssm.{region}.amazonaws.com','kms:EncryptionContext:PARAMETER_ARN':param}}
    node=policy([
        {'Effect':'Deny','Action':['ssm:GetParameter','ssm:GetParameters','ssm:GetParametersByPath'],'NotResource':param},
        allow('ssm:GetParameter',param),allow('kms:Decrypt','*',Condition=kms),
        allow('s3:GetObject',s3+'/releases/*'),allow('s3:PutObject',s3+'/backups/*'),
        allow('cloudformation:SignalResource',stackarn)])
    ec2=['Describe*','CreateVpc','ModifyVpcAttribute','DeleteVpc','CreateSubnet','ModifySubnetAttribute','DeleteSubnet',
         'CreateInternetGateway','AttachInternetGateway','DetachInternetGateway','DeleteInternetGateway',
         'CreateRouteTable','DeleteRouteTable','AssociateRouteTable','DisassociateRouteTable','CreateRoute','DeleteRoute',
         'CreateSecurityGroup','DeleteSecurityGroup','AuthorizeSecurityGroupIngress','AuthorizeSecurityGroupEgress',
         'RevokeSecurityGroupIngress','RevokeSecurityGroupEgress','CreateVolume','DeleteVolume','AttachVolume','DetachVolume',
         'RunInstances','TerminateInstances','ModifyInstanceAttribute','ModifyInstanceCreditSpecification',
         'CreateTags','DeleteTags','AllocateAddress','AssociateAddress','DisassociateAddress','ReleaseAddress']
    infrastructure=policy([
        allow(['ec2:'+x for x in ec2],'*',Condition={'StringEquals':{'aws:RequestedRegion':region}}),
        {'Effect':'Deny','Action':'ec2:RunInstances','Resource':f'arn:aws:ec2:{region}:{account}:instance/*','Condition':{'StringNotEquals':{'ec2:InstanceType':'t3.small'}}},
        {'Effect':'Deny','Action':'ec2:CreateVolume','Resource':'*','Condition':{'NumericGreaterThan':{'ec2:VolumeSize':20}}},
        {'Effect':'Deny','Action':'ec2:CreateVolume','Resource':'*','Condition':{'Bool':{'ec2:Encrypted':'false'}}},
        allow('s3:*',[s3,s3+'/*']),
        allow('iam:PassRole',iam+':role/'+instance,Condition={'StringEquals':{'iam:PassedToService':'ec2.amazonaws.com'}}),
        allow(['iam:GetRole','iam:GetInstanceProfile'],[iam+':role/'+instance,iam+':instance-profile/'+instance])])
    operator=policy([
        {'Effect':'Deny','Action':'freetier:UpgradeAccountPlan','Resource':'*'},
        allow(['cloudformation:CreateChangeSet','cloudformation:ExecuteChangeSet','cloudformation:DeleteChangeSet','cloudformation:DeleteStack',
               'cloudformation:DescribeChangeSet','cloudformation:DescribeStacks','cloudformation:DescribeStackEvents',
               'cloudformation:GetTemplate','cloudformation:ListStackResources'],stackarn),
        allow('iam:PassRole',iam+':role/'+execution,Condition={'StringEquals':{'iam:PassedToService':'cloudformation.amazonaws.com'}}),
        allow('ssm:PutParameter',param),allow('kms:Encrypt','*',Condition=kms),
        allow('s3:ListBucket',s3,Condition={'StringLike':{'s3:prefix':['releases/*']}}),
        allow(['s3:GetObject','s3:PutObject'],s3+'/releases/*'),
        allow('ssm:SendCommand',f'arn:aws:ec2:{region}:{account}:instance/*',Condition={'StringEquals':{'ssm:resourceTag/aws:cloudformation:stack-name':stack}}),
        allow('ssm:SendCommand',f'arn:aws:ssm:{region}::document/AWS-RunShellScript'),
        allow('ssm:GetCommandInvocation','*')])
    def trust(service):return policy([{'Effect':'Allow','Principal':{'Service':service},'Action':'sts:AssumeRole'}])
    return {'instance-policy':node,'execution-policy':infrastructure,'deployer-policy':operator,
            'instance-trust':trust('ec2.amazonaws.com'),'execution-trust':trust('cloudformation.amazonaws.com')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--account',required=True);p.add_argument('--region',default='ap-south-1');p.add_argument('--stack',default='askiisc-prod')
    p.add_argument('--user',default='askiisc-deployer');p.add_argument('--admin-profile',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
    docs=documents(a.account,a.region,a.stack);a.out.mkdir(parents=True,exist_ok=True,mode=0o700)
    for name,doc in docs.items():(a.out/(name+'.json')).write_text(json.dumps(doc,indent=2)+'\n')
    if not a.apply:print('Policies prepared; no AWS calls made:',a.out);return
    def aws(*args,optional=False):
        run=subprocess.run(['aws','--profile',a.admin_profile,'--region',a.region,*args,'--output','json','--no-cli-pager'],capture_output=True,text=True)
        if run.returncode:
            if optional and 'NoSuchEntity' in run.stderr:return None
            raise RuntimeError(run.stderr.strip())
        return json.loads(run.stdout or '{}')
    caller=aws('sts','get-caller-identity')
    if caller['Account']!=a.account:raise ValueError('Administrator profile belongs to a different account')
    for kind,suffix in [('instance','instance'),('execution','cloudformation')]:
        name=a.stack+'-'+suffix;existing=aws('iam','get-role','--role-name',name,optional=True)
        if existing is not None:
            if existing['Role']['AssumeRolePolicyDocument']!=docs[kind+'-trust']:raise ValueError('Existing role trust differs: '+name)
        else:aws('iam','create-role','--role-name',name,'--assume-role-policy-document','file://'+str(a.out/(kind+'-trust.json')))
        aws('iam','put-role-policy','--role-name',name,'--policy-name','AskIIScScoped','--policy-document','file://'+str(a.out/(kind+'-policy.json')))
        if kind=='instance':aws('iam','attach-role-policy','--role-name',name,'--policy-arn','arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore')
    name=a.stack+'-instance';profile=aws('iam','get-instance-profile','--instance-profile-name',name,optional=True)
    if profile is None:aws('iam','create-instance-profile','--instance-profile-name',name);roles=[]
    else:roles=[r['RoleName'] for r in profile['InstanceProfile']['Roles']]
    if roles and roles!=[name]:raise ValueError('Existing instance profile contains a different role')
    if not roles:aws('iam','add-role-to-instance-profile','--instance-profile-name',name,'--role-name',name)
    # IAM user inline policies share a 2 KiB limit; a scoped managed policy has 6 KiB.
    policy_arn=f'arn:aws:iam::{a.account}:policy/{a.stack}-deployer'
    existing=aws('iam','get-policy','--policy-arn',policy_arn,optional=True)
    if existing is None:
        aws('iam','create-policy','--policy-name',a.stack+'-deployer','--policy-document','file://'+str(a.out/'deployer-policy.json'))
    else:
        version=aws('iam','get-policy-version','--policy-arn',policy_arn,'--version-id',existing['Policy']['DefaultVersionId'])
        if version['PolicyVersion']['Document']!=docs['deployer-policy']:
            raise ValueError('Existing deployment policy differs; review a policy version update explicitly')
    aws('iam','attach-user-policy','--user-name',a.user,'--policy-arn',policy_arn)
    print(json.dumps({'account':a.account,'instance_profile':name,'execution_role':f'arn:aws:iam::{a.account}:role/{a.stack}-cloudformation',
                      'user':a.user,'infrastructure_created':False,'iam_setup':'complete'},indent=2))


if __name__=='__main__':main()
