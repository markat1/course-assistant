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

## Needs checking by Mark

- Part 2 GPU row: why an H100 and not a cheaper card (cost/availability).
- Part 3/7: the overflow model is only proposed, not decided.
- Part 5 "Client gone": the claim that SGLang aborts and frees KV on disconnect
  has not been proven on the GPU.
- Compare with the local (unpushed) handoff file and merge any next steps it
  had that are missing here.

## Next (no GPU needed)

In the order of "Open work" in `DESIGN.md`: tenant token window, interactive
priority, `prefix_then_load` + per-prefix worker set, queue-depth gauge and
evict counter, ramp for a returning worker, four Prometheus alert rules,
notebook skeleton.
