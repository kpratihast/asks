"""Generate stack.json from the readable bootstrap script. Does not call AWS."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
R=lambda n:{'Ref':n}
S=lambda s:{'Fn::Sub':s}
G=lambda n,a:{'Fn::GetAtt':[n,a]}
AZ={'Fn::Select':[0,{'Fn::GetAZs':''}]}

def template():
    params={
      'DomainName':{'Type':'String','Default':'','AllowedPattern':r'(?:|(?:[a-zA-Z0-9][a-zA-Z0-9-]*\.)+[a-zA-Z]{2,63})','Description':'Optional hostname you control. Leave empty for HTTPS on the Elastic IP.'},
      'InstanceProfileName':{'Type':'String','AllowedPattern':r'[A-Za-z0-9+=,.@_-]{1,128}','Description':'Existing app-only EC2 instance profile, created once by iam_bootstrap.py.'},
      'InstanceType':{'Type':'String','Default':'t3.small','AllowedValues':['t3.small','t3.medium'],'Description':'Start with 2 GiB; choose t3.medium if memory or load requires it.'},
      'AmiId':{'Type':'AWS::EC2::Image::Id','Description':'Resolved and pinned AL2023 x86_64 AMI. The CLI preserves this value on updates.'},
      'RuntimeParameter':{'Type':'String','Default':'/askiisc/prod/runtime','AllowedPattern':r'/[a-zA-Z0-9_./-]+','Description':'Existing or subsequently created Standard SecureString JSON parameter.'},
      'DataVolumeGiB':{'Type':'Number','Default':20,'MinValue':20,'MaxValue':100},
      'ClientCidr':{'Type':'String','Default':'0.0.0.0/0','Description':'Who can access HTTPS. Port 80 stays public for ACME validation.'},
      'BudgetEmail':{'Type':'String','Default':'','Description':'Optional email for account-wide monthly budget alerts.'},
      'MonthlyBudgetUsd':{'Type':'Number','Default':20,'MinValue':1,'Description':'AWS Budgets uses USD; alert only, not a spending cap.'}}
    resources={
      'Vpc':{'Type':'AWS::EC2::VPC','Properties':{'CidrBlock':'10.78.0.0/16','EnableDnsSupport':True,'EnableDnsHostnames':True}},
      'Gateway':{'Type':'AWS::EC2::InternetGateway'},
      'GatewayAttachment':{'Type':'AWS::EC2::VPCGatewayAttachment','Properties':{'VpcId':R('Vpc'),'InternetGatewayId':R('Gateway')}},
      'Subnet':{'Type':'AWS::EC2::Subnet','Properties':{'VpcId':R('Vpc'),'CidrBlock':'10.78.1.0/24','AvailabilityZone':AZ,'MapPublicIpOnLaunch':True}},
      'RouteTable':{'Type':'AWS::EC2::RouteTable','Properties':{'VpcId':R('Vpc')}},
      'InternetRoute':{'Type':'AWS::EC2::Route','DependsOn':'GatewayAttachment','Properties':{'RouteTableId':R('RouteTable'),'DestinationCidrBlock':'0.0.0.0/0','GatewayId':R('Gateway')}},
      'SubnetRoute':{'Type':'AWS::EC2::SubnetRouteTableAssociation','Properties':{'SubnetId':R('Subnet'),'RouteTableId':R('RouteTable')}},
      'SecurityGroup':{'Type':'AWS::EC2::SecurityGroup','Properties':{'GroupDescription':'HTTPS chat; SSM administration; no SSH or database listeners','VpcId':R('Vpc'),
        'SecurityGroupIngress':[{'IpProtocol':'tcp','FromPort':80,'ToPort':80,'CidrIp':'0.0.0.0/0'},{'IpProtocol':'tcp','FromPort':443,'ToPort':443,'CidrIp':R('ClientCidr')}] }},
      'ReleaseBucket':{'Type':'AWS::S3::Bucket','DeletionPolicy':'Retain','UpdateReplacePolicy':'Retain','Properties':{
        'BucketName':S('${AWS::StackName}-${AWS::AccountId}-${AWS::Region}-releases'),
        'BucketEncryption':{'ServerSideEncryptionConfiguration':[{'ServerSideEncryptionByDefault':{'SSEAlgorithm':'AES256'}}]},
        'PublicAccessBlockConfiguration':{'BlockPublicAcls':True,'IgnorePublicAcls':True,'BlockPublicPolicy':True,'RestrictPublicBuckets':True},
        'OwnershipControls':{'Rules':[{'ObjectOwnership':'BucketOwnerEnforced'}]},'VersioningConfiguration':{'Status':'Enabled'},
        'LifecycleConfiguration':{'Rules':[{'Id':'BackupRetention','Status':'Enabled','Prefix':'backups/','ExpirationInDays':30,'NoncurrentVersionExpiration':{'NoncurrentDays':7}},
          {'Id':'IncompleteUploads','Status':'Enabled','AbortIncompleteMultipartUpload':{'DaysAfterInitiation':1}}]}}},
      'BucketPolicy':{'Type':'AWS::S3::BucketPolicy','Properties':{'Bucket':R('ReleaseBucket'),'PolicyDocument':{'Version':'2012-10-17','Statement':[{
        'Effect':'Deny','Principal':'*','Action':'s3:*','Resource':[G('ReleaseBucket','Arn'),S('${ReleaseBucket.Arn}/*')],
        'Condition':{'Bool':{'aws:SecureTransport':'false'}}}]}}},
      'DataVolume':{'Type':'AWS::EC2::Volume','DeletionPolicy':'Retain','UpdateReplacePolicy':'Retain','Properties':{'AvailabilityZone':AZ,'Size':R('DataVolumeGiB'),'VolumeType':'gp3','Encrypted':True,'Tags':[{'Key':'Name','Value':S('${AWS::StackName}-state')}]}},
      'AppInstance':{'Type':'AWS::EC2::Instance','DependsOn':['InternetRoute','SubnetRoute'],
        'CreationPolicy':{'ResourceSignal':{'Timeout':'PT20M'}},'Properties':{
        'ImageId':R('AmiId'),'InstanceType':R('InstanceType'),'IamInstanceProfile':R('InstanceProfileName'),
        'SubnetId':R('Subnet'),'SecurityGroupIds':[R('SecurityGroup')],'MetadataOptions':{'HttpTokens':'required','HttpEndpoint':'enabled','HttpPutResponseHopLimit':1},
        'CreditSpecification':{'CPUCredits':'standard'},
        'BlockDeviceMappings':[{'DeviceName':'/dev/xvda','Ebs':{'VolumeSize':16,'VolumeType':'gp3','Encrypted':True,'DeleteOnTermination':True}}],
        'Volumes':[{'Device':'/dev/sdf','VolumeId':R('DataVolume')}],
        'Tags':[{'Key':'Name','Value':S('${AWS::StackName}-app')}]}},
      'ElasticIp':{'Type':'AWS::EC2::EIP','DependsOn':'GatewayAttachment','Properties':{'Domain':'vpc'}},
      'IpAssociation':{'Type':'AWS::EC2::EIPAssociation','Properties':{'InstanceId':R('AppInstance'),'AllocationId':G('ElasticIp','AllocationId')}},
      'Budget':{'Type':'AWS::Budgets::Budget','Condition':'HasBudgetEmail','Properties':{
        'Budget':{'BudgetName':S('${AWS::StackName}-account-monthly'),'BudgetType':'COST','TimeUnit':'MONTHLY','BudgetLimit':{'Amount':R('MonthlyBudgetUsd'),'Unit':'USD'},
          'CostTypes':{'IncludeCredit':False,'IncludeRefund':False}},
        'NotificationsWithSubscribers':[{'Notification':{'NotificationType':'ACTUAL','ComparisonOperator':'GREATER_THAN','Threshold':n,'ThresholdType':'PERCENTAGE'},
          'Subscribers':[{'SubscriptionType':'EMAIL','Address':R('BudgetEmail')}]} for n in [50,80,100]]}}}
    script=(HERE/'bootstrap.sh').read_text()
    for token,ref in {'STACK':'AWS::StackName','REGION':'AWS::Region','DOMAIN':'DomainName','PUBLIC_IP':'ElasticIp','BUCKET':'ReleaseBucket','PARAMETER':'RuntimeParameter','VOLUME':'DataVolume'}.items():
        script=script.replace('@@'+token+'@@','${'+ref+'}')
    resources['AppInstance']['Properties']['UserData']={'Fn::Base64':S(script)}
    return {'AWSTemplateFormatVersion':'2010-09-09','Description':'Ask IISc single-node MVP: HTTPS, private releases, durable state, SSM operations. External LLM required.',
      'Parameters':params,'Conditions':{'HasDomain':{'Fn::Not':[{'Fn::Equals':[R('DomainName'),'']}]},'HasBudgetEmail':{'Fn::Not':[{'Fn::Equals':[R('BudgetEmail'),'']}]}},'Resources':resources,
      'Outputs':{'InstanceId':{'Value':R('AppInstance')},'PublicIp':{'Value':R('ElasticIp')},'Domain':{'Value':{'Fn::If':['HasDomain',R('DomainName'),R('ElasticIp')]}},
        'ApplicationUrl':{'Value':{'Fn::If':['HasDomain',S('https://${DomainName}'),S('https://${ElasticIp}')]}},'ReleaseBucket':{'Value':R('ReleaseBucket')},'DataVolumeId':{'Value':R('DataVolume')},
        'RuntimeParameter':{'Value':R('RuntimeParameter')},'AmiId':{'Value':R('AmiId')}}}
if __name__=='__main__':(HERE/'stack.json').write_text(json.dumps(template(),indent=2)+'\n')
