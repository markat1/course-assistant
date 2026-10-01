# Presentation map - 30 minutes

For each of the brief's 13 questions: which diagram to point at, where the code is,
which number proves it, and the term to use. Full answers are in
[`DESIGN.md`](../DESIGN.md); this file is only the route through them.

Line numbers are for the current `main`. They move when `gateway/main.py` changes.

## The one sentence

We built the control plane in front of two engines on one GPU. The gateway decides
what enters and where it goes; the engine decides what runs next. Every decision is
proven with scrapes from the real cluster.

## Timeline

| Minutes | Topic | Diagram | Questions |
|---|---|---|---|
| 0-2 | The one sentence, the whole picture | `architecture.excalidraw` | - |
| 2-6 | The app: one question = three calls | `agent-turn.excalidraw` | 1 |
| 6-11 | The cluster and capacity on paper | `cluster.excalidraw`, `capacity.excalidraw` | 9, 10 |
| 11-21 | The gateway, step by step | `shipping-pipeline.excalidraw` | 2-8 |
| 21-25 | Proof: alerts and the five numbers | `plots/`, Grafana | 11 |
| 25-28 | Scaling, and what is still open | `cluster.excalidraw` (red box) | 12, 13 |
| 28-30 | Questions | - | - |

## The 13 questions

| # | Question | Point at | Code | Number | Term |
|---|---|---|---|---|---|
| 1 | What is the app; shared vs unique tokens? | `agent-turn`: the four boxes in APP, the green and blue notes | [`app/agents/router.py:4`](../app/agents/router.py#L4), [`app/agents/tutor.py:5`](../app/agents/tutor.py#L5), [`app/tools/course_lookup.py:8`](../app/tools/course_lookup.py#L8), headers in [`app/lifespan.py:24`](../app/lifespan.py#L24) | ~97 % prefix-cache hits | shared prefix, prefix cache |
| 2 | What dies at guard / admit / place / queue? | `shipping-pipeline`: right column, steps 1-6 | [`gateway/main.py:60`](../gateway/main.py#L60) (the whole order); guard [`:66`](../gateway/main.py#L66), tenant [`:88`](../gateway/main.py#L88), admit [`:96`](../gateway/main.py#L96), place [`:110`](../gateway/main.py#L110) | 400 / 429 / 503 / 504 | shed, admission |
| 3 | Where do I prevent work that will time out? | `shipping-pipeline`: step 6 | [`gateway/execution/waiting.py:8`](../gateway/execution/waiting.py#L8), [`gateway/execution/dispatch.py:31`](../gateway/execution/dispatch.py#L31) | 102 x 504 on batch | queue deadline |
| 4 | Where do I protect KV? | `shipping-pipeline`: step 3; `capacity`: red box | [`gateway/policies/admission.py:6`](../gateway/policies/admission.py#L6), dispatch cap in [`gateway/execution/lifecycle.py:32`](../gateway/execution/lifecycle.py#L32) | admit below 0.90; KV peak 8.8 % | KV pressure, dispatch cap |
| 5 | Where do I prioritise interactive traffic? | `architecture`: gateway step 5 | [`gateway/policies/priority.py:1`](../gateway/policies/priority.py#L1), [`gateway/models/queued_request.py:21`](../gateway/models/queued_request.py#L21), [`gateway/lifespan.py:54`](../gateway/lifespan.py#L54) | interactive 0 failures | priority queue, request class |
| 6 | Where do I stop one tenant owning the GPU? | `shipping-pipeline`: step 2 | [`gateway/policies/tenant_window.py:27`](../gateway/policies/tenant_window.py#L27) | 76 x 429, all on `revision-batch` | tenant window, Retry-After |
| 7 | Where do I hop, and what is not copied? | `architecture`: the HOP box between the workers | [`gateway/execution/hops.py:14`](../gateway/execution/hops.py#L14), [`gateway/policies/hop_ledger.py:22`](../gateway/policies/hop_ledger.py#L22), placement in [`gateway/policies/routing.py:43`](../gateway/policies/routing.py#L43) | hops 399 -> 2 | hop, recompute, prefix_then_load |
| 8 | Where do I evict; what becomes a ghost? | `architecture`: the HOP box | [`gateway/policies/hop_ledger.py:46`](../gateway/policies/hop_ledger.py#L46), called from [`gateway/monitoring/polling.py:41`](../gateway/monitoring/polling.py#L41) and [`gateway/execution/dispatch.py:31`](../gateway/execution/dispatch.py#L31) | `orch_hop_evictions_total` | ghost, eviction |
| 9 | Engine scheduler vs my admit / place / queue? | `cluster`: Compose box vs k3s box | ours: [`gateway/main.py:60`](../gateway/main.py#L60); theirs: flags in [`cluster/workers/sglang.yaml:41`](../cluster/workers/sglang.yaml#L41) | engine queue ~0, backlog in ours | two boxes, continuous batching |
| 10 | What limited concurrency on this GPU? | `capacity`: step 3 and the red box | [`cluster/workers/sglang.yaml:44`](../cluster/workers/sglang.yaml#L44), [`gateway/execution/lifecycle.py:32`](../gateway/execution/lifecycle.py#L32) | 3.3x from cap 2 -> 8; KV 8.8 % | scheduler cap |
| 11 | Four production alerts? | Grafana / `plots/grafana-2026-09-29/` | [`monitoring/alerts.yaml:4`](../monitoring/alerts.yaml#L4) | 2 fired, 1 pending, 1 never | SLO, p99 TTFT |
| 12 | If I scale, which pool? | `cluster`: red box | DESIGN Part 8 | decode slots, not KV | TPOT, decode slot |
| 13 | 10x traffic; three wrong knobs? | `cluster`: red box | DESIGN Part 8 | - | backpressure |

## Readiness and warmup (asked as "hop or warmup proof")

| Point at | Code | Number |
|---|---|---|
| `architecture`: "startup" box under each worker | [`gateway/monitoring/readiness.py:11`](../gateway/monitoring/readiness.py#L11), prefix in [`gateway/monitoring/warmup_request.py:10`](../gateway/monitoring/warmup_request.py#L10), ramp in [`gateway/policies/ramp.py:1`](../gateway/policies/ramp.py#L1) | first request on a fresh worker 54-73 ms -> 22-23 ms |

## The five numbers

| Number | Says | File |
|---|---|---|
| 3.3x | Our own dispatch cap (2) was the first limiter | `metrics/load-gateway-2026-09-29.txt` |
| 8.8 % | KV was never the limiter | `metrics/locust-labelled-a100-2026-09-30.txt` |
| 76 x 429, interactive 0 failures | The tenant window and the priority queue work | same file |
| 54-73 ms -> 22-23 ms | Warm before ready | `metrics/warmup-first-token-2026-09-29.txt` |
| 399 -> 2 hops | The prefix stays on the worker that holds it | `metrics/locust-labelled-a100-2026-09-30.txt` |

## Say these before anyone asks

- Batch ends as 504 after 5 s (102 times). Fix designed, tests written: a 30 s
  deadline for batch and a cap of half the queue.
- Overflow is decided (Qwen3-8B on an owned 24 GB GPU) but not wired: 503 is
  counted as `leave_disabled`.
- The tokenizer counter is new and proven without a GPU only (6/6 against the
  engine's recorded counts); the A/B run on the GPU is owed.
- Latency on the H100 and the A100 is not compared.

## Terms

| Term | Meaning here |
|---|---|
| KV cache | The keys and values the model keeps for every earlier token so it does not recompute them. 144 KiB per token for Qwen3-8B. |
| Prefill | Processing the prompt, all tokens in parallel. Cost grows with prompt length. |
| Decode | Generating the answer one token at a time. |
| TTFT | Time to first token: queue wait + prefill + the first decode step. |
| TPOT | Time per output token during decode. |
| Prefix cache (radix cache) | The engine reuses the KV of a prompt start it has seen before. Our shared prefix is the system prompt and the tool schema. |
| GQA | Grouped-query attention: several attention heads share one K and V. 8 KV heads instead of 32, so 4x less KV per token. |
| Continuous batching | The engine adds and removes sequences from the running batch at every step. Not ours. |
| Chunked prefill | A long prompt is prefilled in pieces (2,048 tokens) so it does not block the others. |
| Retraction (preempt) | The engine takes a running sequence out of the batch when KV runs out. Never happened in our runs. |
| Guard | The first no, before any GPU work: a request the engine could never serve (400). |
| Admission | Is any worker ready, with fresh metrics and KV below 0.90? Otherwise 503. |
| Shed | To refuse a request on purpose, with a status code and a reason. |
| Placement | Choosing the worker: least loaded, but the holder of the prefix wins within a slack of 4 (`prefix_then_load`). |
| Hop | A prefix lands on a worker that does not hold it. Nothing is copied; the worker recomputes it. |
| Ghost | A prefix the ledger believes a worker holds after that worker lost its cache. |
| Readiness, warmup | Ready = health + model + 5 warmup requests on the shared prefix + fresh metrics. |
| Ramp | A returning worker gets less traffic at first: its limit starts at 1 and doubles while the engine keeps up. |
| Tenant window | A sliding token budget per `X-Tenant` (200,000 per 60 s). Over budget: 429 + Retry-After. |
| Stay or leave | 429 and 500 stay; 503 and 529 may go to another provider (overflow). |
| Dispatch cap | At most 8 requests in flight per worker, equal to the engine's `--max-running-requests`. |
| In flight | Sent to the engine, not finished. |
| HAMi slice | A part of the GPU's memory given to one pod (38,000 MiB). Memory is split, compute is shared. |
| NodePort | A fixed port on the host for one pod: 30001 = worker A, 30002 = worker B. |
| p99, SLO | The latency 99 % of requests stay under; the target we alert on (p99 TTFT above 1 s). |
