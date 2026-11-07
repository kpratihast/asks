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

## Submission review

| Finding | Fix |
|---|---|
| A malformed citation URL returned 500 and occupied the only generation slot indefinitely. A queued client disconnect could also strand capacity until timeout. | Invalid source URLs are dropped. The HTTP request releases the slot on completion, failure or disconnect, including before an SSE response starts. |
| An OpenRouter key from the environment was sent to a different model endpoint. Authorized restricted evidence could also reach the hosted model. | The OpenRouter environment key is sent only to the OpenRouter HTTPS origin. Restricted generation requires loopback or an exact trusted self-hosted endpoint. Hosted endpoints can serve public evidence. |
| A remote citation could claim a different URL or version from its passage. A local snapshot change during a pinned read could return a different snapshot. | The gateway checks passage provenance before formatting it and verifies the snapshot pin again after retrieval. Encoded and normalized private paths are checked against the current catalogue. |
| A real vector answer cited both exam documents but joined the wrong teaching slot to the exam date. | A slot-based exam answer needs a verified connection unless a source states the exam details against the course directly. |
| Upstream error or reasoning events could consume memory outside the answer limit. A token-limit finish could pass citation checks and expose a truncated answer. | The gateway caps the complete SSE response and requires a normal stop before sending answer text. |
| A full release could copy the contents of a source symlink. Code-only packaging could combine a new graph implementation with an incompatible snapshot. | Both package paths reject source links. Full packaging and code-only composition check the snapshot against the graph rules before producing a release. |
| The CUDA launcher bound vLLM to every interface by default. | The launcher defaults to loopback. Other interfaces require an API key and operator-controlled private networking. |
| pypdf 6.21.4 is affected by newly published issues. | The local, AWS and CI requirements now use pypdf 7.0.0 or later within major version 7. The AWS and CI pins use the published 7.0.0 hashes. |

The focused regression suite covers the cases above using synthetic protected evidence and local model fixtures. The public model checks use published IISc pages. [pypdf advisory on cross-reference streams](https://github.com/py-pdf/pypdf/security/advisories/GHSA-59jv-x5x2-gg4c) · [pypdf advisory on XFA forms](https://github.com/py-pdf/pypdf/security/advisories/GHSA-cmpx-464x-82c3)

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

- The repository defaults to an empty professor test override. Existing AWS releases keep their captured runtime settings until a fresh release is activated. An explicitly configured test address gains faculty access after OTP verification.
- IISc email ownership is not proof of current enrollment. Non-faculty IISc accounts receive the student role under the requested design; stronger verification requires an institute-managed identity directory.
- Prompt filters are defense in depth. A novel/obfuscated injection or a factually wrong answer with valid citation markers can still evade these checks. Citation validation checks references and required connected sources, not semantic entailment. No secrets, tools, shell or arbitrary URL-fetch capability are provided to the model. [OWASP prompt-injection guidance](https://genai.owasp.org/llmrisk/llm01-prompt-injection/).
- Document classification is an operator responsibility. Publicly published facts may overlap private attachments; protecting the attachment cannot make independently public facts secret. Rebuild and validate a new immutable snapshot when changing documents or permissions.
- Rate limits cap resource consumption but cannot guarantee availability against distributed denial-of-service or prevent an attacker from using the permitted anonymous/OTP quota. No paid WAF or new AWS infrastructure is introduced.
- Restricted evidence stays on loopback or the exact private endpoint named by `ASKIISC_TRUSTED_MODEL_URL`. The app returns authorized excerpts and document links when the configured model is untrusted. Hosted models still receive the public evidence and question for public answers.
- SMTP/model keys previously pasted into chat should be rotated through their providers before formal testing and updated through the private runtime configuration. Do not store replacement secrets in Git or reports.
- The named deployer cannot read security-group rules or the S3 public-access-block configuration. Host inspection and bounded external probes provide partial evidence only; an administrator should verify the actual policy settings and account MFA. Do not use root credentials for routine deployment.

Operational test evidence is saved outside Git so it cannot accidentally publish corpus contents or runtime data. AWS deployment uses the existing FREE-plan guard, immutable bundle checks, candidate validation and rollback. Keep the previous release for recovery; a rollback also restores that release's old security behavior and dependency versions.
