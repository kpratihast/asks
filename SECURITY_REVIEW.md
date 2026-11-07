# Security review and regression checks

This review hardens the public chat and document-access boundaries. It is not a certification that the application is free of vulnerabilities. The browser, questions, history, model output, and ingested text are untrusted. A verified server-side principal determines document permissions; model instructions never grant a role.

## Changes

| Boundary | Previous gap | Protection |
|---|---|---|
| Retrieval | Missing labels could default to public; filtering passages did not independently filter trace proofs | Explicit scope required at serving; provenance and graph-step scopes checked; current private-document manifest overrides stale labels; source resolution checked again |
| Optional ingestion adapters | Legacy Qdrant/Neo4j data without classification was treated as public | Unclassified chunks are unavailable to supported roles; unclassified relationships and their aliases are skipped. Ingestion must explicitly classify legacy data |
| Error handling | Retrieval/model failures and validation errors could return raw diagnostic or input text | Generic client errors; upstream error bodies are not read or forwarded |
| Streaming | Draft tokens reached the client before citation validation | Buffer bounded model output and release answer tokens only after validation; failed drafts never leave the server |
| Prompt instructions | Explicit overrides and fabricated evidence could reach the model | Normalize and reject clear instruction overrides in questions/relevant history; refuse contaminated sources; encode question as an untrusted JSON string and strengthen system instructions |
| HTTP input | Development/direct access relied on proxy body limits; parsing accepted ambiguous JSON | 32 KiB body limit including chunked input, ten-second body deadline, 64 simultaneous body readers, JSON-only writes, nesting/duplicate-key/control-character checks and bounded fields |
| Authentication abuse | Verification had challenge limits but no endpoint rate limit; rotating IPs could exhaust email quota | Peer/global pre-authentication limits, bounded peer state, and persistent limits of 50 OTP sends/hour and 200/day; existing per-email/IP and five-attempt limits remain |
| Browser | No CSP or frame protection in the application | Hashed script CSP, no framing/plugins/external connections, no-store responses, MIME protection and attachment downloads |
| Diagnostics | Public protection depended on exact reverse-proxy routes; readiness contacted the model on every call | Public Swagger/OpenAPI disabled in the application, metrics require authorization, readiness probes coalesced and cached for five seconds |
| Upstream resource use | A trickling model could evade individual read timeouts | Overall generation deadline and bounded output size |
| Dependencies | pypdf 6.18.1 had published denial-of-service advisories | Upgrade local, AWS and CI requirements to 6.21.4 with PyPI SHA-256 release hashes |

The full/vector modes, public chat, email OTP flow, role hierarchy, faculty test override and configured admission limits remain available. Streaming now shows the verified answer after generation rather than displaying unverified draft tokens. Protected originals download instead of rendering inline.

## Repeatable tests

From the repository root, using its Python environment:

```bash
python -m pytest tests/test_security_redteam.py tests/test_email_access.py tests/test_admission_limits.py -q
python -m pytest -q
```

Tests use a fake mail inbox/model and synthetic private canaries. They cover full/vector retrieval and SSE/JSON responses; guest/student/faculty scope combinations; forged client roles, cookies and headers; source-ID access; OTP replay, expiry, lockout, rotation and distributed send limits; malformed requests; source contamination; graph metadata; cached health probes; stream buffering and deadlines. Existing GitHub validation runs these tests with every code change. Optional external ingestion integrations require their separate services and are skipped without them.

Audit the pinned AWS dependencies in an isolated tool environment before the assessment:

```bash
uv tool run --from pip-audit==2.10.1 pip-audit \
  --no-deps --disable-pip -r deploy/aws/requirements.lock
```

On 6 November 2026 the audit found 21 advisory entries for pypdf 6.18.1. After the update it reported zero known vulnerabilities for 59 checked packages. The CPU-specific `torch==2.14.0+cpu` distribution was not found in the scanner's PyPI feed and was skipped; this is not a clean bill of health for that artifact or for the OS/proxy. Dependency auditing does not prove exploitability or absence of unknown flaws. [Upstream PDF security fix](https://github.com/py-pdf/pypdf/security/advisories/GHSA-5849-834g-hvw5).

## Assessment boundaries and remaining risks

- The explicitly requested professor test override remains enabled on AWS. That address gains faculty access only after OTP verification. It is an intentional exception to directory-based role assignment and should be disclosed to the evaluator.
- IISc email ownership is not proof of current enrollment. Non-faculty IISc accounts receive the student role under the requested design; stronger verification requires an institute-managed identity directory.
- Prompt filters are defense in depth. A novel/obfuscated injection or a factually wrong answer with valid citation markers can still evade these checks. Citation validation checks references and required connected sources, not semantic entailment. No secrets, tools, shell or arbitrary URL-fetch capability are provided to the model. [OWASP prompt-injection guidance](https://genai.owasp.org/llmrisk/llm01-prompt-injection/).
- Document classification is an operator responsibility. Publicly published facts may overlap private attachments; protecting the attachment cannot make independently public facts secret. Rebuild and validate a new immutable snapshot when changing documents or permissions.
- Rate limits cap resource consumption but cannot guarantee availability against distributed denial-of-service or prevent an attacker from using the permitted anonymous/OTP quota. No paid WAF or new AWS infrastructure is introduced.
- Authorized restricted-document questions send their selected evidence to the configured model provider. Use the local model for workloads that must not leave the institution. Provider privacy settings are not a substitute for institutional approval.
- SMTP/model keys previously pasted into chat should be rotated through their providers before formal testing and updated through the private runtime configuration. Do not store replacement secrets in Git or reports.
- The named deployer cannot read security-group rules or the S3 public-access-block configuration. Host inspection and bounded external probes provide partial evidence only; an administrator should verify the actual policy settings and account MFA. Do not use root credentials for routine deployment.

Operational test evidence is saved outside Git so it cannot accidentally publish corpus contents or runtime data. AWS deployment uses the existing FREE-plan guard, immutable bundle checks, candidate validation and rollback. Keep the previous release for recovery; a rollback also restores that release's old security behavior and dependency versions.
