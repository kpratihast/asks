"""Code-only CI archives; private corpus and model assets never leave the AWS host."""
from __future__ import annotations
import hashlib,json,re,shutil,tarfile
from pathlib import Path,PurePosixPath
from release_lib import sha,unpack,verify_tree,valid_id

SUFFIXES={'.py','.html','.css','.js','.svg','.png','.webp','.ico','.woff2'}
DEPLOY_NAMES={'awsctl.py','release_lib.py','host.py','runtime.py','model_config.py','model_profiles.py',
 'deploy_latest.py','iam_bootstrap.py','render_stack.py','bootstrap.sh','stack.json','requirements.in',
 'requirements.lock','README.md','runtime.example.json','deployer-policy.example.json',
 'model-profiles.example.json','ci.py','ci_bundle.py','ci_host.py','ci_setup.py','GITHUB_DEPLOYMENT.md'}
CONFIG_NAMES={'faculty_directory.json','restricted_documents.json','departments.yaml'}


def allowed(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '\\' in name or any(x.startswith('.') or x=='__pycache__' for x in p.parts):return False
    return (len(p.parts)>=2 and p.parts[0] in {'askiisc_retrieval','askiisc_serving'} and p.suffix in SUFFIXES or
            len(p.parts)==3 and p.parts[:2]==('deploy','aws') and p.name in DEPLOY_NAMES or
            len(p.parts)==2 and p.parts[0]=='config' and p.name in CONFIG_NAMES)


def validate_code(tree):
    m=verify_tree(tree)
    if m.get('kind')!='code' or not re.fullmatch('[0-9a-f]{40}',m.get('git_sha','')):raise ValueError('Expected a code-only commit archive')
    if not all(allowed(n) for n in m['files']):raise ValueError('Code archive contains a forbidden path')
    required={'askiisc_retrieval/__init__.py','askiisc_serving/__init__.py',
              'config/restricted_documents.json','config/faculty_directory.json',
              'deploy/aws/host.py','deploy/aws/release_lib.py','deploy/aws/runtime.py','deploy/aws/requirements.lock'}
    if not required<=m['files'].keys():raise ValueError('Code archive is incomplete')
    return m


def manifest(tree,metadata):
    files={str(p.relative_to(tree)):sha(p) for p in sorted(tree.rglob('*')) if p.is_file() and p.name!='release.json'}
    content=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    m={**metadata,'schema':1,'source_sha256':content,'files':files}
    (tree/'release.json').write_text(json.dumps(m,indent=2)+'\n');verify_tree(tree)
    return m


def archive(tree,destination):
    with tarfile.open(destination,'w:gz',compresslevel=3) as t:
        for p in sorted(tree.rglob('*')):
            if p.is_file():
                info=t.gettarinfo(str(p),str(p.relative_to(tree)))
                info.uid=info.gid=0;info.uname=info.gname='root';info.mtime=0
                with p.open('rb') as f:t.addfile(info,f)
    destination.chmod(0o600)


def package(root,tree,git_sha,release_id):
    root=Path(root);tree=Path(tree);tree.mkdir(mode=0o700)
    if not re.fullmatch('[0-9a-f]{40}',git_sha):raise ValueError('Expected a full commit SHA')
    for p in sorted(root.rglob('*')):
        name=str(p.relative_to(root))
        if not allowed(name):continue
        if p.is_symlink() or any(q.is_symlink() for q in p.parents if q!=root.parent):raise ValueError('Code links are forbidden')
        if p.is_file():
            target=tree/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    if sum(p.stat().st_size for p in tree.rglob('*') if p.is_file())>10*1024**2:raise ValueError('Code archive exceeds 10 MiB')
    manifest(tree,{'kind':'code','release_id':valid_id(release_id),'git_sha':git_sha})
    return validate_code(tree)


def compose(base,code,target):
    base=Path(base);code=Path(code);target=Path(target)
    old=verify_tree(base);new=validate_code(code);target.mkdir(mode=0o700)
    retained=[n for n in old['files'] if n.startswith(('data/','assets/','.local/private/'))]
    if not any(n.startswith('data/') for n in retained) or not any(n.startswith('assets/') for n in retained):raise ValueError('Base release lacks corpus/model assets')
    for source,names in [(base,retained),(code,list(new['files']))]:
        for name in names:
            destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,destination)
    # A changed ACL manifest must still describe the private originals/extractions retained on AWS.
    for doc in json.loads((target/'config/restricted_documents.json').read_text())['documents']:
        for field,digest in [('original','sha256'),('extracted','extracted_sha256')]:
            path=target/doc[field]
            if not path.resolve().is_relative_to((target/'.local/private').resolve()) or sha(path)!=doc[digest]:raise ValueError('Private document metadata requires a full, reviewed data deployment')
    metadata={k:old[k] for k in ['snapshot','embedding_model','embedding_revision','chunks','scopes']}
    metadata.update(release_id=new['release_id'],git_sha=new['git_sha'],base_release=old['release_id'],code_sha256=new['source_sha256'])
    return manifest(target,metadata)
