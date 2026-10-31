# AWS deployment files

Start with [the deployment guide](../../AWS_DEPLOYMENT.md). All AWS mutations require running an explicit CLI subcommand; `package`, `verify` and local tests do not call AWS.

| File | Purpose |
|---|---|
| `iam_bootstrap.py` | One-time app IAM setup; review policies before applying with an administrator profile |
| `stack.json` | CloudFormation resources and embedded host bootstrap |
| `bootstrap.sh` | Readable source for the host setup |
| `render_stack.py` | Regenerate the template after a bootstrap/resource change |
| `requirements.in`, `requirements.lock` | Runtime input versions and complete Linux CPU dependency hashes |
| `deploy_latest.py` | Verify the newest bundle, enforce the account/FREE-plan checks, deploy to the existing stack and smoke test |
| `awsctl.py` | Package, provision, configure, deploy, verify, smoke, status, logs, backup and rollback |
| `release_lib.py` | Checksum, archive, configuration and SQLite-copy validation |
| `host.py` | Root controller on EC2; candidate checks, activation and rollback |
| `runtime.py` | Isolated AWS settings and single-worker application entry point |
| `model_profiles.py`, `model-profiles.example.json` | One-command exposed-local/OpenRouter profiles, bounded public RAG test and transactional deployment |
| `model_config.py` | Private endpoint/model/key settings; hidden key prompt and public-only probe |
| `runtime.example.json` | Private runtime JSON template; fill it outside the repository |
| `deployer-policy.example.json` | Narrow policy example for trusted release operators |

To regenerate the dependency lock deliberately, run from the project root with `uv` installed:

```bash
uv pip compile deploy/aws/requirements.in \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
  --index https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match --generate-hashes \
  --no-header --no-annotate --output-file deploy/aws/requirements.lock
python deploy/aws/render_stack.py
python -m pytest -q
cfn-lint deploy/aws/stack.json
bash -n deploy/aws/bootstrap.sh
```

Review updated packages and hashes before deployment. The CPU index is an explicit upstream PyTorch source; the installation requires hashes and binary wheels. On the host, a changed lock creates a new environment without modifying the old one.
