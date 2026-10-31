#!/usr/bin/env python3
"""Verify and deploy an application bundle to the existing AWS stack; never provision resources."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import awsctl
from release_lib import sha, unpack, valid_id


def latest_artifact(bundles: Path) -> Path:
    candidates = list(bundles.glob('*/artifact.json'))
    if not candidates:
        raise ValueError(f'No bundles found in {bundles}. Use --build, --bundles DIR or --artifact FILE.')
    # Select by creation descriptor time, never by release-name lexical ordering.
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p))).resolve()


def verify_artifact(descriptor: Path) -> dict:
    a = json.loads(descriptor.read_text())
    valid_id(a['release_id'])
    folder = descriptor.parent
    if a.get('archive') != 'bundle.tar.gz' or sha(folder/'bundle.tar.gz') != a['sha256']:
        raise ValueError('Artifact checksum mismatch')
    for name in ['host.py', 'release_lib.py']:
        if sha(folder/name) != a['controls'][name]:
            raise ValueError('Controller checksum mismatch')
    with tempfile.TemporaryDirectory(prefix='askiisc-verify-') as td:
        manifest = unpack(folder/'bundle.tar.gz', Path(td)/'release')
        if manifest['release_id'] != a['release_id'] or manifest['snapshot'] != a['snapshot']:
            raise ValueError('Descriptor does not match the bundled release')
        for name in ['host.py', 'release_lib.py']:
            if manifest['files']['deploy/aws/'+name] != a['controls'][name]:
                raise ValueError('Controller does not match the bundled release')
    return a


def check_account(region: str, expected: str) -> dict:
    identity = awsctl.aws(region, 'sts', 'get-caller-identity')
    if identity['Account'] != expected:
        raise ValueError('Wrong AWS account; no deployment was attempted')
    if identity['Arn'].endswith(':root'):
        raise ValueError('Root credentials are not allowed; sign in with the deployer profile')
    plan = awsctl.aws('us-east-1', 'freetier', 'get-account-plan-state')
    if plan.get('accountId') != expected or plan.get('accountPlanType') != 'FREE' or plan.get('accountPlanStatus') != 'ACTIVE':
        raise ValueError('An ACTIVE FREE account plan is required; no account upgrade will be attempted')
    credits = plan.get('accountPlanRemainingCredits', {})
    if credits.get('unit') != 'USD' or float(credits.get('amount', 0)) <= 0:
        raise ValueError('Available promotional credits could not be confirmed')
    return plan


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--artifact', type=Path, help='Explicit artifact.json; otherwise choose the newest bundle')
    selection.add_argument('--build', action='store_true', help='Package current application/index before deployment')
    p.add_argument('--root', type=Path, default=awsctl.ROOT)
    p.add_argument('--bundles', type=Path, help='Bundle directory; default ROOT/.local/aws-releases')
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--profile', default='askiisc-deploy')
    p.add_argument('--account', default='800168266746')
    p.add_argument('--region', default='ap-south-1')
    p.add_argument('--stack', default='askiisc-prod')
    p.add_argument('--dry-run', action='store_true', help='Verify locally and print the plan; make no AWS calls')
    a = p.parse_args(argv)
    bundles = (a.bundles or a.root/'.local/aws-releases').expanduser().resolve()
    if a.build:
        data = a.data_dir or Path(json.loads((a.root/'config/local.json').read_text())['ASKIISC_DATA_DIR'])
        descriptor = awsctl.package(a.root, data, bundles)/'artifact.json'
    else:
        descriptor = a.artifact.expanduser().resolve() if a.artifact else latest_artifact(bundles)
    release = verify_artifact(descriptor)
    print(json.dumps({'action': 'deploy_existing_stack', 'artifact': str(descriptor),
                      'release': release['release_id'], 'profile': a.profile, 'account': a.account,
                      'region': a.region, 'stack': a.stack, 'dry_run': a.dry_run}, indent=2), flush=True)
    if a.dry_run:
        return
    # A named profile must not be silently overridden by exported access keys.
    if any(os.environ.get(k) for k in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']):
        raise ValueError('Unset exported AWS access/session keys so the named deployer profile is used')
    os.environ['AWS_PROFILE'] = a.profile
    os.environ['AWS_DEFAULT_PROFILE'] = a.profile
    os.environ['AWS_PAGER'] = ''
    check_account(a.region, a.account)
    outputs = awsctl.outputs(a.region, a.stack)
    # Preflight verifies the currently selected model, HTTPS and ACLs without generation.
    awsctl.smoke(outputs['ApplicationUrl'])
    awsctl.deploy(a.region, a.stack, descriptor)
    awsctl.smoke(outputs['ApplicationUrl'])
    print('Deployment verified:', outputs['ApplicationUrl'], flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        raise SystemExit(str(exc))
