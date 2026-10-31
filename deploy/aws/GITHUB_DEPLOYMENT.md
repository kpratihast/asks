# GitHub → AWS deployment

Repository: https://github.com/kpratihast/asks (private).

A push to any branch runs offline application/security tests. After one-time AWS OIDC setup and enabling `AWS_DEPLOY_ENABLED`, a successful branch build deploys to the existing `askiisc-prod` app. Pull requests run tests only. The Actions page also provides a manual run for a selected branch. One deployment runs at a time; the most recent pending run can replace an older pending run, so each intermediate commit is not guaranteed a separate deployment. All branches share one app: the last successful deployment becomes live. Treat anyone with branch push access as a production deployer.

## First-time AWS setup

The administrator step creates an IAM GitHub identity provider if absent, a narrowly scoped `askiisc-prod-github` role and one additional instance-role permission to upload full recovery bundles under the existing private bucket's `releases/ci/` prefix. It does not create compute, disks, network services or an AWS build service. It does not change the account plan. A normal deployer profile intentionally cannot make these IAM changes.

Generate the reviewable policies without AWS access:

```bash
python3 deploy/aws/ci_setup.py \
  --account 800168266746 --region ap-south-1 --stack askiisc-prod \
  --instance i-04324b3a4662b318d --repo kpratihast/asks \
  --owner-id 20087001 --repo-id 1429859022 \
  --out .local/github-iam
```

Inspect all three JSON files. The trust pins GitHub's immutable owner/repository IDs, AWS audience `sts.amazonaws.com` and `refs/heads/*`. The actual repository OIDC settings were verified to use immutable subjects. Pull-request subjects, tags, other repositories and environments do not match. If migrating to another repository, inspect `gh api repos/OWNER/REPO/actions/oidc/customization/sub` and prepare a matching trust policy; do not broaden the wildcard.

Apply once with an explicitly selected administrator profile:

```bash
python3 deploy/aws/ci_setup.py \
  --account 800168266746 --region ap-south-1 --stack askiisc-prod \
  --instance i-04324b3a4662b318d --repo kpratihast/asks \
  --owner-id 20087001 --repo-id 1429859022 \
  --out .local/github-iam --admin-profile YOUR_ADMIN_PROFILE --apply
```

The script refuses to replace an existing, different trust/permissions policy. GitHub uses only temporary OIDC sessions afterward. Do not put root or IAM access keys in GitHub secrets.

Set these repository **Actions variables** (not secrets):

```bash
gh variable set AWS_ACCOUNT_ID --repo kpratihast/asks --body 800168266746
gh variable set AWS_REGION --repo kpratihast/asks --body ap-south-1
gh variable set AWS_STACK_NAME --repo kpratihast/asks --body askiisc-prod
gh variable set AWS_DEPLOY_ROLE_ARN --repo kpratihast/asks --body arn:aws:iam::800168266746:role/askiisc-prod-github
gh variable set AWS_DEPLOY_ENABLED --repo kpratihast/asks --body true
gh workflow run aws-validation.yml --repo kpratihast/asks --ref main
```

Before enabling, set the GitHub account's Actions spending budget to stop usage at the included allowance (or zero additional spend). Private repositories have a finite included runner allowance; runner minutes are not unlimited free. This workflow uses standard Ubuntu runners, installs no LLM/weights, uses no Actions artifact storage, and provisions no additional AWS infrastructure. Existing EC2, storage, public IP and private S3 release storage continue using account credits. Every deployment requires an ACTIVE FREE plan with positive confirmed credits, and its role explicitly denies account upgrades. Failed checks stop deployment; they cannot guarantee no incidental storage/network usage.

## What happens during a deployment

1. Hash-locked lightweight dependencies run the tests against synthetic fixtures without private inputs, model calls or credentials. Third-party Actions are pinned to immutable commits; checkout does not retain Git credentials.
2. The deployment job alone requests an AWS OIDC token. Its role can inspect the one stack, upload/read `releases/ci/*` and run SSM commands on the one existing EC2 instance. It cannot provision resources, change IAM, or read/write runtime Parameter Store secrets directly. Because it can deploy code through SSM, trusted branch writers can execute code on this app server; that is a deployment privilege, not a secret-isolation boundary against malicious maintainers.
3. The runner uploads an allowlisted code archive, not the corpus, model weights, restricted originals, OTP database or local configuration. Checksums validate each controller and file; symlinks and paths into private data are rejected.
4. On EC2, the controller takes the same lock used by local/manual deployments. It verifies the current release and combines its frozen corpus/weights/private originals with the new code. A changed private-document manifest must match the retained bytes; ingestion changes need a full deployment. It uploads a full private recovery bundle to S3, then runs the existing release controller.
5. The controller reuses the hash-addressed Python environment when requirements are unchanged, creates a separate candidate index, checks retrieval/model readiness and a cited public response, and then switches the app. A failed candidate leaves the current app live; failed activation restores the previous release. OTP state stays on the persistent volume. The model endpoint/key stay in the existing SSM configuration.
6. The final smoke verifies HTTPS, readiness and anonymous access denial for protected content without sending OTP email. The candidate public answer uses one request to the selected model. If the model is a local ngrok tunnel, the local server and tunnel must be running for deployment to pass.

At least the five newest CI release directories plus the current/previous releases are retained on EC2. Only older verified CI releases and their matching indexes/config files are pruned after success. Manual releases, model environments, auth state, and S3 recovery bundles are retained. Private S3 versions/bundles grow over time; review storage usage and remove old artifacts deliberately. Never delete the active or previous recovery bundle.

## Future changes

```bash
git add <changed-source-files>
git commit -m "Describe the change"
git push origin YOUR_BRANCH
# Follow the run:
gh run list --repo kpratihast/asks --limit 5
```

Pushes deploy source code, UI, public faculty metadata and auth document metadata. They do not ingest fresh website/PDF content or change the selected LLM. Do those explicitly with the existing full-bundle/model commands. New runtime dependencies require regenerating the Linux Python 3.12 hash-locked `deploy/aws/requirements.lock`. Tests have a separate `ci-requirements.lock` to avoid loading Torch or downloading models.

To pause automatic deployments while retaining tests:

```bash
gh variable set AWS_DEPLOY_ENABLED --repo kpratihast/asks --body false
```

To roll back, use the named deployer profile:

```bash
AWS_PROFILE=askiisc-deploy python3 deploy/aws/awsctl.py rollback --stack askiisc-prod --region ap-south-1
```

Do not stop/replace an in-progress deployment because its web page appears quiet. Inspect the Actions and SSM command status first. When the EC2 instance is stopped overnight, disable deployments or leave changes unpushed until it is online; the workflow never starts a stopped server automatically.

## Local development

The existing local source tree, `config/local.json`, native MLX process and local chat UI remain usable. The Git repository excludes machine-specific configuration, private documents, datasets, keys and generated reports. For a fresh clone, follow `LOCAL_CHAT.md`; copy `config/local.example.json` to `config/local.json`, set the local endpoint/model and ingest the public sources. For the model-switch wrapper, copy `deploy/aws/model-profiles.example.json` to ignored `config/aws-models.json`, supply your private runtime/key file paths there, and use `./model-aws.sh local --url https://YOUR-TUNNEL.ngrok-free.app` or `./model-aws.sh openrouter --model provider/model-id`. For full ingestion releases use `./deploy-aws.sh --build`; ordinary source pushes use GitHub Actions. The existing Desktop wrappers also remain available. Restricted originals must be transferred through your approved private process and verified against `config/restricted_documents.json`. A clean clone can run the entire offline test suite without those inputs:

```bash
python3 -m venv .venv
.venv/bin/pip install --only-binary=:all: --require-hashes -r deploy/aws/ci-requirements.lock
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q
```

Official references: [GitHub AWS OIDC](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws), [immutable subjects](https://docs.github.com/en/actions/reference/security/oidc), [Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions).
