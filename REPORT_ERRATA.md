# Report errata

Corrections to `ASKS_Project_Report.pdf` found during red-team testing. Each entry names the issue, the report text, what is wrong and what replaces it. Numbers here come only from scripts or measurements cited in the entry.

## E2 and the 429 policy (#55)

**Report text.** Section 4.4.1: "Every rejection is an HTTP 429 response with a Retry-After header and a JSON body carrying the reason". Section 8.4, below Table 10: "HTTP 429 responses first appeared at 30 concurrent users".

**What is wrong.**

1. A pre-authentication request guard (180 requests/min per client address, 1,200/min across all addresses) ran before the five admission parameters of Section 4.4.1. Its 429 had no `reason`, it was missing from the per-request log, and neither the report nor `LOAD_TESTING.md` described it. One load-generator address could therefore send about 3 requests/s sustained, whatever R, C, Q and G were.
2. The run configuration behind Table 10 was not recorded: the per-key rate, the number of load-generator addresses and the run length. With the documented defaults (R = 30/min per key, three tenant keys) the per-key limiter alone rejects requests well below 30 users; Cloudhack measured 429s from 10 users (#55). The Section 8.4 statement cannot be reproduced from the documented configuration and is withdrawn.

**Correction.** Since the #55 fix, guard 429s carry `reason` (`address_rate_limited`, `all_addresses_rate_limited` or `request_bodies_busy`) and `limiter: request_guard`, are logged and counted like admission rejections, are reported to operators in `/readyz`, are configurable (`GUARD_PER_ADDRESS_PER_MIN`, `GUARD_ALL_ADDRESSES_PER_MIN`) and are documented in `LOAD_TESTING.md`. Table 10's latencies stand as measured, but they describe an unrecorded configuration. A re-run of E2 must record R, C, Q, W, G, the guard limits, the number of load-generator addresses and the run length, and report 429s by `reason`.
