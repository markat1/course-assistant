# Handoff from the Codespace, 30 September 2026

Work done in a GitHub Codespace on branch `codespace-work`, without a GPU. The
local Claude Code session history was not available there, so this file and
`git log eacc11b..codespace-work` are the record.

## Done

- `DESIGN.md`: answers to Parts 0–8 of the final project, built from the code
  and the existing `metrics/` files. Includes the Part 1 capacity math
  (144 KiB/token for Qwen3-8B BF16; 115,299 tokens per 38,000 MiB HAMi slice;
  14 sequences at 8,192, ~57 at a ~2,000-token agent turn) and the hypothesis
  check (the scheduler cap was the first limiter, not KV).
- Every missing piece is marked **GAP** in `DESIGN.md` and collected under
  "Open work" at the end.

- Tenant token window: `gateway/policies/tenant_window.py`, wired in
  `gateway/main.py` after the guard (429 `tenant_tokens`, stays local).
  Settings `GATEWAY_TENANT_MAX_TOKENS` (200,000) / `GATEWAY_TENANT_WINDOW_S` (60).
- Interactive before batch: per-worker `asyncio.PriorityQueue`,
  `X-Request-Class` header, `gateway/policies/priority.py`.
- The app sends `X-Request-Class: interactive` and `X-Tenant` (`APP_TENANT`) on
  every agent step; Locust labels interactive/agent users as `student-N` and
  batch users as one tenant `revision-batch`.
- Working agreement: Claude writes the tests, Mark writes the production code.
  No Claude co-author lines on commits.

## Needs checking by Mark

- Part 2 GPU row: why an H100 and not a cheaper card (cost/availability).
- Part 3/7: the overflow model is only proposed, not decided.
- Part 5 "Client gone": the claim that SGLang aborts and frees KV on disconnect
  has not been proven on the GPU.
- Compare with the local (unpushed) handoff file and merge any next steps it
  had that are missing here.

## Next (no GPU needed)

Done: `orch_replica_queue_depth{worker}`, `orch_replica_in_flight{worker}` (set on
each `/metrics` scrape) and `orch_hop_evictions_total{cause=capacity|worker_lost}`.
Done: `prefix_then_load` placement (`GATEWAY_PREFIX_LOAD_SLACK`, default 4) and
a hop ledger that keeps the set of workers per prefix.
Then: ramp for a returning worker, four Prometheus alert rules, notebook skeleton.
On the next GPU session, re-run the labelled Locust mix (tenant sheds, p99 spread).
