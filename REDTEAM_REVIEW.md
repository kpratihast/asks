Local red-team issue review

Branch fix/reported-issues-20261117 was cut from origin/main at 563c864d0a76a9a85f36948d126c98a8cc456e22. The remote has no master branch. This review describes the approved first fix batch.

Reviewed all 38 open issues. Reused and reviewed the existing origin/fix/redteam-issues patch through 16345d7, then added the follow-up fixes described below. Reused changes were applied and tested together with the follow-up fixes.

Validation

- After integrating `origin/fix/redteam-issues` through `923a634`, 660 tests passed and 17 optional tests were skipped on the local Python 3.14 environment, using Soup Sieve 4.1 in an isolated temporary dependency directory.
- Shell syntax and whitespace checks passed. Python compilation passed. Soup Sieve 4.1 was installed with hash verification for testing. The full Linux production lock was not reinstalled on this Mac.
- Tests use local fixtures and fake model/services. Live GPU/Metal execution, fresh model quality benchmarks, real SMTP delivery and AWS deployment are outside this validation run.

Latest branch integration

Integrated `fix/redteam-issues` through `923a634` into `fix/reported-issues-20261117`. Resolved six conflicted files, retaining the destination's stricter checks that question text is not evidence and its source-backed partial numeric answer handling. Added upstream list-count contradiction checks, key-generation tooling and a shared rebuild path for CLI ingestion and the local launcher. A subprocess regression confirms the old snapshot and vectors survive a rebuild killed during embedding. The 17 skipped tests require the separate pipeline checkout and database configuration.

This integration was tested locally before the user authorized committing and pushing it. AWS deployment is outside this integration's scope. The earlier approval recorded below applies to the previously published batch.

Important limits

- #21 and #22 still require the freshness scheduler and immutable raw-version implementation.
- #3 and #4 remain partial because lexical overlap cannot demonstrate zero unsupported semantic claims or a 100% live-model refusal invariant.
- #15 still needs concurrent-writer and cross-store activation work.
- #26 retains buffered release for security. An earlier token cannot be retracted after disclosure.
- #44 needs measured E4/E6, stale-version, GPU-hour and cost results. Historical report figures have not been replaced with invented results.

Changes requiring review

The strict local launcher can create a private guest key. The example local profile explicitly enables public chat, preserving anonymous access to public information when chosen. Host allowlists now need configuration for non-loopback development hosts. No-retrieval evaluation requires an admin key and an explicit flag. Claim checks can conservatively abstain on paraphrases. Proved partial numeric answers retain a source-backed exception for missing reporting-period breakdowns.

| Issue | Status | Review |
| --- | --- | --- |
| [#1](https://github.com/kpratihast/asks/issues/1) | Coordination | Baseline notice. GPU host and private reporting contact remain team-supplied information. |
| [#2](https://github.com/kpratihast/asks/issues/2) | Implemented; runtime retest needed | Model identity discovery and response metadata, with explicit unknowns when a provider supplies no checkpoint/precision. Offline vLLM and Ollama fixtures pass. |
| [#3](https://github.com/kpratihast/asks/issues/3) | Partial | Claim-level lexical checks reject invented terms/numbers and padded refusals. Question text no longer supplies evidence. This does not establish semantic entailment or catch every wrong label attached to a real number. |
| [#4](https://github.com/kpratihast/asks/issues/4) | Partial | Padded-refusal and unsupported-claim regression tests pass. The 100% live-model invariant has not been demonstrated on the assessment configuration or independent sets. |
| [#5](https://github.com/kpratihast/asks/issues/5) | Implemented | Missing/corrupt graphs degrade to vector retrieval with explicit status. Offline failure tests pass. |
| [#6](https://github.com/kpratihast/asks/issues/6) | Implemented; broader attacks pending | Explicit general-knowledge overrides are refused before generation, and claim checks run on generated content. This is not a proof against all prompt injections. |
| [#7](https://github.com/kpratihast/asks/issues/7) | Configuration distinction | Invalid presented keys are rejected. Strict mode requires credentials. Explicit PUBLIC_CHAT=1 remains available because the product requires anonymous access to public material. The example local profile opts into public mode. |
| [#9](https://github.com/kpratihast/asks/issues/9) | Implemented | Anonymous readiness omits tenant names, admission counters and limits. Operator keys can inspect them. |
| [#10](https://github.com/kpratihast/asks/issues/10) | Implemented | Crawler ignores template navigation and routes followed PDFs to extraction. Offline fixtures cover downloadable and scanned PDFs. |
| [#11](https://github.com/kpratihast/asks/issues/11) | Implemented | Remote retrieval exposes authorized source resolution and the gateway uses it. |
| [#12](https://github.com/kpratihast/asks/issues/12) | Implemented | Failed SMTP delivery removes its challenge and send accounting, allowing a legitimate retry. |
| [#13](https://github.com/kpratihast/asks/issues/13) | Implemented | Anonymous document access is checked before identifier lookup. |
| [#14](https://github.com/kpratihast/asks/issues/14) | Implemented | Per-page extraction failures are recorded while valid pages proceed. All-failed builds preserve the prior active snapshot. |
| [#15](https://github.com/kpratihast/asks/issues/15) | Partial | Removed destructive reset, atomic snapshot writes, retained old versions during rebuild and graph-digest checks. A full concurrent-writer, cross-store activation protocol remains outside this patch. |
| [#16](https://github.com/kpratihast/asks/issues/16) | Implemented | Programme filtering applies in local/remote retrieval and is reported. Unknown programme values are handled explicitly. |
| [#17](https://github.com/kpratihast/asks/issues/17) | Implemented | CSP hashes use the UTF-8 bytes served to the browser. Non-UTF-8 locale regression passes. |
| [#18](https://github.com/kpratihast/asks/issues/18) | Docs added; coordination pending | Added claude/GPU_RUNBOOK.md with explicit assessment settings and no external-ingestion dependency. GPU allocation and private contact placeholders in GitHub still need team information. |
| [#19](https://github.com/kpratihast/asks/issues/19) | Implemented for reported invalid ranges | Academic-year input requires positive consecutive years. Reversed, equal and multi-year ranges return 422. |
| [#21](https://github.com/kpratihast/asks/issues/21) | Remaining, high priority | Add conditional source checks, a bounded 24-hour scheduler and a maintainer trigger. No scheduler has been installed or claimed in this batch. |
| [#22](https://github.com/kpratihast/asks/issues/22) | Remaining, high priority | Add immutable raw-source history, content-hash no-op handling and refresh of crawled sources. Existing raw-cache overwrite behavior still needs replacement. |
| [#24](https://github.com/kpratihast/asks/issues/24) | Implemented | Hostnames default to a loopback/test allowlist, with AUTH_PUBLIC_ORIGIN or ALLOWED_HOSTS for configured hosts. Foreign hosts cannot control redirect locations. |
| [#26](https://github.com/kpratihast/asks/issues/26) | Intentional security tradeoff; acceptance unresolved | Preserved validation before answer release. Documentation distinguishes upstream TTFT from client-visible latency. Incremental safe streaming or a PRD amendment still needs a decision. |
| [#27](https://github.com/kpratihast/asks/issues/27) | Source fixed; rotation pending | Compose reads NEO4J_PASSWORD from a private environment. Existing credentials and historical repository exposure require operator rotation. History has not been rewritten. |
| [#28](https://github.com/kpratihast/asks/issues/28) | Implemented | Both deployment and CI locks use Soup Sieve 4.1 with official PyPI hashes. Tests ran with that version in an isolated dependency directory. |
| [#30](https://github.com/kpratihast/asks/issues/30) | Docs/profile fixed | Local instructions explain copying the example profile, model discovery, one default directory and explicit directory overrides, and --port. |
| [#31](https://github.com/kpratihast/asks/issues/31) | Docs improved; platform retest pending | Corrected clone-folder references, added upstream Metal installation guidance, and identified Python 3.12 as the CI-tested version. No new GPU/Metal installation was performed. |
| [#32](https://github.com/kpratihast/asks/issues/32) | Code fixed; device retest pending | Metal launcher passes --gpu-memory-utilization and clears obsolete overrides. Command-generation tests pass. Actual 16 GiB startup remains to be measured. |
| [#33](https://github.com/kpratihast/asks/issues/33) | Implemented | Redirects are followed explicitly with a bounded hop count and visited-target checks before fetching. Both link orders fetch/store the canonical target once. |
| [#34](https://github.com/kpratihast/asks/issues/34) | Implemented | Instruction filtering focuses on directive positions and retains actual override detection. Academic security and registration questions pass regression tests. |
| [#35](https://github.com/kpratihast/asks/issues/35) | Implemented | KeyStore rejects published placeholder keys, including the example admin key. |
| [#36](https://github.com/kpratihast/asks/issues/36) | Implemented | Whitespace is normalized before term matching. A 100,000-space regression completes within its bound. |
| [#37](https://github.com/kpratihast/asks/issues/37) | Implemented | Malformed terms have an invalid sort sentinel and are excluded from passage selection/auditing. |
| [#38](https://github.com/kpratihast/asks/issues/38) | Implemented | Oversized fixed prompts and oversized rebuilt blocks are rejected. Final assembled messages are checked against the configured token counter. Empty evidence does not reach generation. |
| [#39](https://github.com/kpratihast/asks/issues/39) | Docs fixed | Corrected the refusal sentence and documented validation-buffered delivery. |
| [#41](https://github.com/kpratihast/asks/issues/41) | Implemented | Empty logout requests work without Content-Length and with chunked framing. Nonempty non-JSON bodies still fail. |
| [#42](https://github.com/kpratihast/asks/issues/42) | Implemented | Non-rate admission failure/cancellation refunds the taken rate token. Queue-full retry and concurrency regressions pass. |
| [#43](https://github.com/kpratihast/asks/issues/43) | Implemented | B0 requires both an explicit ALLOW_NO_RETRIEVAL flag and an admin bearer key. Guest credentials remain denied. |
| [#44](https://github.com/kpratihast/asks/issues/44) | Partial; evaluation/report work remaining | Added B1 8B and AWQ launch variants. GPU runbook identifies failed invariants and required E4/E6/cost evidence. Historical PDF results and held-out taxonomy remain unchanged pending fresh measured evidence. |

Upstream verification

Soup Sieve versions below 4.0.0 are affected by the reported selector-matching issue. Both locks use 4.1. [Advisory](https://github.com/advisories/GHSA-mmhr-3c5m-f64f).

Metal memory sizing uses the explicit GPU memory utilization option. [Upstream configuration](https://github.com/vllm-project/vllm-metal/blob/main/docs/configuration.md).

Commit approval

The user authorized committing, pushing and deploying this batch. Remaining issues above retain their stated status. GitHub issue comments and closure are separate from publishing this release.
