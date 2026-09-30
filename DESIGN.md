# Course Assistant: design

The answers to the final project questions, each pointing at code or a scrape.
Items marked **GAP** are not implemented or not yet proven on the GPU; they are
listed together in [Open work](#open-work).

## Part 0. The application

**Track B, tool-using agent, with retrieval as its tool.** Students ask
questions about the inference engineering course in Open WebUI.

```
Open WebUI -> app (OpenAI Agents SDK) -> gateway -> SGLang worker A | B
                Router --handoff--> Tutor --lookup_course--> corpus
```

- `app/agents/router.py`: the Router hands course questions to the Tutor.
- `app/agents/tutor.py`: the Tutor calls `lookup_course`, then answers with a
  check question.
- `app/tools/course_lookup.py`, `app/knowledge/`: a toy corpus with keyword search.
- `app/llm.py`: the only model client. It points at the gateway, so every agent
  step (Router, Tutor tool call, Tutor answer) goes through
  guard -> admit -> place -> queue.

One user turn means several engine calls with a growing context: Router, then
the Tutor's tool call, then the Tutor's answer over the tool result. Evidence of
the full path is in `metrics/h100-sxm5-hami-workers-2026-09-29.txt` §13 and
`metrics/app-streaming-2026-09-30.txt`.

**Shared vs unique tokens**

| Tokens | Shared? | Where |
|---|---|---|
| Router instructions + SDK handoff prompt + handoff tool schema | shared by every turn | `app/agents/router.py` |
| Tutor instructions + `lookup_course` schema | shared by every turn | `app/agents/tutor.py` |
| Retrieved passages | shared between questions on the same topic (small corpus) | `app/knowledge/corpus.py` |
| User question, conversation history, tool call and tool result | unique | per turn |

The gateway keys the prefix on the first message (the system prompt):
`gateway/execution/hops.py:prefix_key`.

Load generators use the same shape: a shared prefix plus a unique question.
`experiments/locustfile.py` has the Class 7 mix (interactive Tutor prefix,
agent tool-schema prefix, batch unique documents), and `experiments/load.py`
and `experiments/first_token.py` use the shared course prefix.

## Part 1. Capacity on paper

**GPU as run:** 1x NVIDIA H100 80GB HBM3 (SXM5) on Lambda, split by HAMi into
two 38,000 MiB memory slices, one per SGLang worker
(`cluster/workers/sglang.yaml`, `nvidia.com/gpumem: 38000`).

**Model:** Qwen3-8B, BF16 (`--dtype=bfloat16`), 36 layers, 8 KV heads (GQA),
head dim 128.

```
kv_bytes_per_token = 2 (K,V) x 36 layers x 8 kv_heads x 128 dim x 2 bytes
                   = 147,456 B = 144 KiB/token
```

The engine's own startup log agrees: 15.84 GiB KV / 115,299 tokens ≈ 147.5 KB/token
(`metrics/h100-sxm5-hami-workers-2026-09-29.txt` §7).

**Per worker (one 38,000 MiB slice, measured by SGLang):**

| | GiB |
|---|---|
| Visible to the pod | 36.6 |
| Weights | 15.3 |
| KV pool (`--mem-fraction-static=0.85`) | 15.8 → 115,299 tokens |
| CUDA graphs, activations, reserve | ~5.4 |

```
max_concurrent_seqs ≈ (HBM − weights − activations) / (kv_bytes_per_token × len)
                    = 115,299 tokens / len
```

| Length | Sequences per worker | Both workers |
|---|---|---|
| max_len = 8,192 (`--context-length`) | 14 | 28 |
| Batch document, ~4,500 (Locust batch, worst case) | 25 | 51 |
| Agent turn, ~2,000 (prefix + tool result + answer) | 57 | 115 |
| Load test, ~1,800 (1,670 prefix + 128 out) | 64 | 128 |

With the radix prefix cache, the shared prefix is stored once per worker rather
than once per sequence. That makes the real KV capacity higher than this table.

**Model choice and bytes/token:** An 8B model with GQA (8 KV heads) costs
144 KiB/token. A model without GQA and the same shape (32 KV heads) would cost
4x, 576 KiB/token, which is only 3.5 full-length sequences per slice.

**Hypothesis: the first limiter is the scheduler, not KV.**
`--max-running-requests=8` caps each worker at 8 sequences, below the 14–64
that KV allows. **Result: confirmed.** Under overload the engines ran exactly
their cap, their own queue was empty and KV usage stayed at ~0–1 %
(`metrics/load-gateway-2026-09-29.txt`, `plots/grafana-2026-09-29/03-*.png`,
`04-*.png`). Before that, the gateway's dispatch cap of 2 was an even earlier
limiter; raising it to 8 gave 3.3x throughput (same file). After the scheduler,
the next limiter is **compute/HBM bandwidth**, because HAMi slices the memory,
not the SMs. Both workers share one H100's compute. A single worker alone
reached ~1,000 generated tok/s at 8 concurrent requests
(`metrics/kill-worker-under-load-2026-09-29.txt`).

## Part 2. The cluster

| Decision | Choice | Why |
|---|---|---|
| GPU | H100 80GB SXM5 (Lambda, $4.29/h, rented per session and terminated between sessions) | The cheapest single GPU that holds **two** Qwen3-8B BF16 replicas (the brief requires at least two workers) with a useful KV pool each: 2 x 38,000 MiB HAMi slices give 115,299 KV tokens per worker (~14 sequences at 8,192). Rejected (Lambda list, 29-30 Sep 2026): A10 24 GB ($1.29/h) fits one replica with ~4-5 GB KV (~30k tokens), so two workers would need two machines or a smaller model; A100 40 GB ($1.99/h) cannot hold two 15.3 GiB replicas plus KV; A100 80 GB only as 8x ($22.32/h); GH200 ($2.29/h) is ARM64 while our pinned SGLang image is amd64. H100 PCIe ($3.29/h) was the first choice but unavailable on 29 Sep. Session cost: ~$4.29 per hour of experiments. |
| Model | Qwen3-8B BF16, thinking disabled | Reliable structured tool calls with SGLang's `qwen25` parser (`metrics/h100-sxm5-hami-workers-2026-09-29.txt` §12); GQA keeps KV at 144 KiB/token. |
| Engine | SGLang v0.5.20 (pinned digest) | Radix prefix cache for the shared agent prefix; Prometheus `/metrics`. |
| Topology | 2 colocated replicas (prefill + decode on each), HAMi memory slices on one GPU | The workload is a short, highly shared prefix (~97 % cache hit) with multi-step decode. A prefill/decode split would move KV for little gain. HAMi enforces the memory split inside the pod (`metrics/h100-sxm5-hami-workers-2026-09-29.txt` §4). |
| Concurrency | `--max-running-requests=8`, `--context-length=8192`, `--chunked-prefill-size=2048` | See Part 5. |
| Hop backend | `recompute`, named and counted by the gateway; no Mooncake | Same GPU, colocated replicas: moving KV would need a transfer engine for a prefix that recomputes in ~54–73 ms cold. See Part 6. |
| Overflow | **Qwen3-8B BF16 on the same pinned SGLang image, on one 24 GB GPU we own** (demo: the local RTX 4090 over a reverse SSH tunnel; production: a 24 GB instance in the same region). Interactive only, 503/529 only. Currently switched off: a leave is counted as `leave_disabled` | Same weights, template and `qwen25` tool parser, so Router/Tutor tool calls behave identically, and we keep the KV, the warmup and the metrics. First limiter: the KV pool on 24 GB (~4-5 GB after weights). See Part 3. |
| Orchestration | k3s StatefulSet `sglang` (2 replicas) + Compose for gateway, app, UI, Prometheus, Grafana | `cluster/`, `compose.*.yaml`, `docs/kubernetes-runbook.md` |

**Two boxes:**
- **Gateway** (`gateway/`, FastAPI, port 8780): guard, admit, place, queue,
  hop record, warmup/readiness and stay-or-leave. It decides *what enters and where*.
- **Engine** (SGLang pods, NodePorts 30001/30002): waiting queue, token pool,
  radix cache, retraction (preemption), chunked prefill, continuous batching
  and kernels. It decides *what runs next inside the batch*.

**Scaling:** see Part 8, "If I scale".

## Part 3. Guardrails, admit, stay vs leave

**Guard:** `gateway/policies/guard.py:inspect`. This is the first "no", before any queue slot
or GPU work:
- `bad_max_tokens` (400): `max_tokens` > `max_output_tokens` (1024).
- `prompt_too_long` (400): estimated prompt + output > `context_length` (8192).
  The estimate is 4 chars/token because the gateway has no tokenizer.

Evidence: `orch_guard_rejected_total{reason="bad_max_tokens"}` in
`metrics/guard-overflow-hops-2026-09-30.txt`.

**Admit:** `gateway/policies/admission.py:admit`. A worker is eligible only if it
is ready (warmed), its metrics are fresh (≤ `metrics_max_age_s`) and its KV usage is
below `kv_usage_limit` (0.90):
- `kv_pressure` (503 + Retry-After): workers are up but all above the KV limit.
- `no_eligible_workers` (503 + Retry-After): no worker is ready or fresh.

After placement, the per-worker queue can shed:
- `queue_full` (503 + Retry-After): the chosen worker's queue (16) is full.
- `timeout_queue` (504): the request waited longer than `queue_timeout_s` (5 s)
  before dispatch.

All sheds are counted in `orch_shed_total{reason,code}`.

**Stay vs leave:** `gateway/policies/overflow.py:stay_or_leave`. 429/500/slice OOM
stay: they are the client's or our own fault, and another provider would not fix
them. 503/529 may leave. Overflow is disabled, so a leave is counted as
`orch_overflow_total{decision="leave_disabled"}` and returned to the client with
Retry-After. Evidence: 252 capacity 503s → 252 `leave_disabled`
(`metrics/guard-overflow-hops-2026-09-30.txt`).

**Overflow recipient (decision).** Who receives a 503/529: **Qwen3-8B BF16 on the
same pinned SGLang v0.5.20 image and flags, on one 24 GB GPU that we own**, as a
third worker outside the H100 cluster. Only interactive requests (`X-Request-Class:
interactive`) and only 503/529 may leave; 429 (`tenant_tokens`), 500 and slice OOM
never leave; batch is shed with Retry-After instead.

- *Why this model:* it is the same model, chat template and `qwen25` tool-call
  parser as the cluster, so the Router's handoff and the Tutor's `lookup_course`
  calls work unchanged and answers do not change character under load. We own the
  engine, so the gateway's warmup (five requests on the shared prefix), readiness
  and `/metrics` apply to it exactly as to the cluster workers.
- *Where it runs:* for the demo, the local RTX 4090 (24 GB), reached from the Lambda
  gateway through a reverse SSH tunnel from the laptop; in production, a 24 GB
  instance (e.g. an A10) in the same region, which removes the home network.
- *Its limiter (hits first):* the KV pool. 24 GB minus ~15.3 GiB of weights and
  runtime leaves roughly 4-5 GiB for KV, i.e. ~30,000 tokens at 144 KiB/token:
  ~3-4 sequences at 8,192 tokens or ~15 agent turns at ~2,000 tokens. The engine
  runs with `--max-running-requests=4` and the gateway caps overflow at 4 requests
  in flight, so the overflow cannot become a second queue. The exact pool is read
  from SGLang's `max_total_num_tokens` at startup. Second limiter: network - the
  reverse tunnel from the US to Denmark adds ~100 ms or more to TTFT (vs 18-35 ms
  in the cluster), acceptable for a request that would otherwise get a 503.
- *Alternatives rejected (checked 30 September 2026):* Qwen3-8B on OpenRouter has
  one provider and is deprecated on 9 October 2026; Qwen3-8B on Alibaba Cloud Model
  Studio supports function calling only in the Beijing region, which would break
  the agent's tool calls; Qwen3-32B on Groq was shut down on 17 July 2026; Qwen3-8B
  on Fireworks is offered only as a dedicated GPU deployment (~$8/h), i.e. a second
  cluster rather than overflow. The best hosted alternative is **Qwen3-32B on
  DeepInfra** (same Qwen3 family and template, function calling, 40,960 context,
  ~$0.10/$0.28 per 1M tokens, 200 concurrent requests per model); it was not chosen
  because we would own neither its KV, its warmup nor its metrics, data would leave
  the cluster, and a larger model would answer differently under load.
- *Status:* decided and documented; the gateway still counts leaves as
  `leave_disabled`. Sending them to the overflow worker (a URL setting plus the
  in-flight cap of 4) and a live overflow run are not implemented. **GAP**.

**Tenant window (`tenant_tokens`):** `gateway/policies/tenant_window.py`, called
in `gateway/main.py` after the guard and before `admit`. Each tenant (`X-Tenant`
header, default `default`) may spend `tenant_max_tokens` (200,000) per sliding
`tenant_window_s` (60 s). A request costs its estimated prompt tokens plus
`max_tokens` (or `max_output_tokens` if missing). Over budget → **429**
`tenant_tokens` with Retry-After (seconds until enough old tokens leave the
window), counted in `orch_shed_total{reason="tenant_tokens",code="429"}` and as
`stay`: a 429 never overflows. A rejected request does not use the budget; a
guard rejection does not reach the window. Tests: `tests/unit/test_tenant_window.py`,
`tests/integration/test_tenant_admission.py`.

The app sends `X-Tenant` from `APP_TENANT` (default `course-assistant`) on every
agent step (`app/lifespan.py`, `default_headers`). In `experiments/locustfile.py`
each interactive and agent user is its own `student-N` tenant (~20k tokens/min),
while all batch users share `revision-batch` (~600k tokens/min offered), so the
batch tenant is expected to hit `tenant_tokens` while students pass. **GAP**: not
yet run on the GPU.

**`should_shed(req, snap) -> (shed, code, reason, retry_after_seconds)`.** The brief's
single function is implemented as an ordered pipeline in `gateway/main.py:chat_completions`,
where `snap` is the gateway's view of the workers (`WorkerState`: ready, metric age,
engine running/waiting, KV usage, gateway queue and in-flight). The first stage that
says "shed" decides the tuple; nothing later runs:

| Order | Stage (file) | shed? | code | reason | Retry-After |
|---|---|---|---|---|---|
| 1 | Guard `inspect` (`policies/guard.py`) | request cannot be served | 400 | `bad_max_tokens`, `prompt_too_long` | none (client must change the request) |
| 2 | Tenant window (`policies/tenant_window.py`) | tenant over its token budget | 429 | `tenant_tokens` | seconds until enough budget frees up |
| 3 | Admission `admit` (`policies/admission.py`) | no ready worker with fresh metrics / all above the KV limit | 503 | `no_eligible_workers`, `kv_pressure` | `capacity_retry_after_s` (7) |
| 4 | Enqueue on the placed worker (`execution/queueing.py`) | worker queue full | 503 | `queue_full` | 7 |
| 5 | Queue deadline (`execution/waiting.py`) | waited longer than `queue_timeout_s` | 504 | `timeout_queue` | none |

Stages 1-3 decide before placement from the request and the snapshot; 4-5 depend on
the chosen worker's queue. Every 503/429 is also classified stay-or-leave. Known
limit: `timeout_queue` is detected after the wait, not predicted before enqueue; a
predictive check (queue depth x recent service time > deadline -> 503 before
enqueue) is the next step.

## Part 4. Place

`gateway/policies/routing.py:select_worker`: **`prefix_then_load`**, i.e.
least-loaded with queue depth as a scorer, then prefix affinity within a slack.

```
load(worker) = gateway_queue_depth + gateway_in_flight + engine_waiting
```

- Workers whose gateway queue has room are preferred; queue depth is part of the
  score, not only an admission input.
- Ties are broken at random. Before this, ties always went to the first worker,
  which herded ~5:1 onto worker-a (`metrics/load-gateway-2026-09-29.txt`) and
  evened out to ~1:1 after the fix (`metrics/after-routing-fix-2026-09-29.txt`,
  `plots/grafana-2026-09-29/10-*.png`).
- Passive health check: a refused connection marks the worker not ready
  immediately (`gateway/execution/dispatch.py`). Before: a dead worker drained
  instantly, looked least loaded and burnt 735/800 requests. After: 800/800 OK
  (`metrics/kill-worker-under-load-2026-09-29.txt`,
  `metrics/after-routing-fix-2026-09-29.txt`).
- **No bounce:** a request is placed once. If its queue is full it is shed
  (503), not re-placed.

- **Prefix affinity:** among eligible workers with queue room, a worker that
  already holds the request's shared prefix (`gateway/execution/hops.py:prefix_holders`)
  wins if its load is at most `lowest + prefix_load_slack` (default 4 = half the
  dispatch cap of 8); otherwise the least-loaded worker wins and a hop is
  recorded. Slack 0 would use the prefix only on ties; a high slack would pin a
  popular prefix to one worker while the other idles. Why it is worth waiting a
  little: a cold shared prefix costs 54–73 ms, a warm one 15–17 ms
  (`metrics/warmup-first-token-2026-09-29.txt`). Before (random ties, no
  affinity) half of all placements moved the prefix (~200 hops each way in
  `metrics/guard-overflow-hops-2026-09-30.txt`). Tests: `tests/unit/test_routing.py`,
  `tests/integration/test_prefix_placement.py`. **GAP**: re-measure hops and
  p99 on the GPU.
- Prefill and decode use the same scorer because the replicas are colocated.

## Part 5. Queue: what runs next, and what does not

```
admit -> place -> gateway queue (per worker, FIFO, 16) -> dispatch (8 per worker)
      -> SGLang waiting -> running (8) -> retracted
```

| Question | Answer |
|---|---|
| Who sits in my queue vs the engine's? | Gateway queue: requests admitted but not dispatched (`gateway_queue_depth`). Engine waiting: dispatched but not yet running (`sglang:num_queue_reqs`). Because the dispatch cap (8) equals `--max-running-requests` (8), the engine queue stays ~0 and the backlog is visible in the gateway, where it can still be shed. |
| Waiting / running / retracted? | `sglang:num_queue_reqs`, `sglang:num_running_reqs`, `sglang:num_retracted_reqs`, all on the "Engine and Queues" dashboard. |
| Queue depth per pod? | `orch_replica_queue_depth{worker}` (waiting in our queue) and `orch_replica_in_flight{worker}` (dispatched, not finished), set from `WorkerState` on every `/metrics` scrape (`gateway/main.py:metrics`). Next to `sglang:num_queue_reqs` this shows who waits where. Tests: `tests/integration/test_queue_depth_metrics.py`. **GAP**: Grafana panel and GPU scrape under the labelled mix. |
| 32k RAG retrieve vs short agent decode? | A 32k prompt never gets in: guard rejects it (`prompt_too_long`, context 8192). Inside 8k, **we** decide the order in our queue: interactive before batch (below). Once dispatched, **the engine** decides: `--chunked-prefill-size=2048` splits a long prefill so running decodes keep stepping. |
| Interactive vs batch in my queue? | Each worker queue is an `asyncio.PriorityQueue` (`gateway/lifespan.py:create_queues`). `X-Request-Class: interactive\|batch` (default `interactive`; unknown → 400 `bad_request_class` at the guard) maps to a priority in `gateway/policies/priority.py`; `QueuedRequest` orders by (priority, arrival) so a class stays FIFO (`gateway/models/queued_request.py`). Batch can starve while interactive keeps arriving and then expires as `timeout_queue` (504): a deliberate "what we do not run". This orders only our queue; it is not the engine scheduler. Tests: `tests/unit/test_request_priority.py`, `tests/integration/test_request_class.py`. |
| PagedAttention vs radix cache: which saved memory on the shared-prefix mix? | The radix cache. SGLang's token pool avoids fragmentation, but the savings on our mix come from reuse: ~97 % cache hit in Grafana, and a warm request recomputes 15–17 tokens instead of ~1,670 (`metrics/warmup-first-token-2026-09-29.txt`, `#cached-token: 1656`). |
| Chunked prefill / batching flags | `--chunked-prefill-size=2048` (limits prefill per step, protects decode TPOT); `--max-running-requests=8` (decode CUDA graphs captured for bs 1, 2, 4, 8); `max_prefill_tokens=16384` (default). |
| KV full after admit? | The gateway admits only below 90 % KV; after that the engine retracts (`num_retracted_reqs`). Not reached in any run (KV ≤ 1 %). The gateway does not fix OOM; it only avoids adding work above the limit. |
| Client gone? | Streaming: a client disconnect closes the gateway→engine stream, SGLang aborts the request and frees its KV. The shared prefix stays in the radix cache as evictable cache. Queue: `forward_until_done` cancels the upstream task when the request's result is already done (e.g. expired). **GAP**: prove the abort on the GPU (engine running count drops). |
| Worker returns: slam or ramp? | Ramp. Before, it slammed: 8 requests within 2 s of warmup (`metrics/kill-worker-under-load-2026-09-29.txt`, recovery timeline). Now a worker that becomes ready starts at `ramp_limit = 1` (`gateway/monitoring/polling.py:refresh_worker`); each metrics poll (5 s) doubles it up to the dispatch cap (1 → 2 → 4 → 8 in 15 s) while the engine queue is empty, and halves it when `sglang:num_queue_reqs > 0` (`gateway/policies/ramp.py`). The engine queue is our proxy for "p99 holds": once SGLang makes requests wait, TTFT is rising. Routing adds `dispatch_max − ramp_limit` to the worker's load (`gateway/policies/routing.py:gateway_load`), so a returning worker gets traffic only when the warm one is busier. It is a soft cap: if the warm worker is full, the ramping one still serves rather than shedding. Tests: `tests/unit/test_ramp.py`, `tests/integration/test_worker_ramp.py`, `tests/integration/test_prefix_placement.py`. **GAP**: repeat the kill-worker run on the GPU. |

**Notebook:** `notebook/part5_queue.ipynb` answers these from a live Prometheus
scrape (queue table per worker, queue depth by pod, door sheds vs engine
retractions, radix cache vs KV usage, hops and sheds, ramp after a kill) and
saves its plots to `plots/`. Query helpers: `experiments/prometheus_snapshot.py`
(tested in `tests/unit/test_prometheus_snapshot.py`). **GAP**: run it with
outputs during the GPU session.

## Part 6. Hop and warmup

**Hop record:** `gateway/execution/hops.py`, `gateway/policies/hop_ledger.py`.
- Same worker (src == dst): no hop, nothing recorded.
- Different worker: `Hop(src, dst, prefix, tokens, backend="recompute")`, counted in
  `orch_hop_total{src,dst,backend}` and `orch_hop_tokens_total`.
- **What is not copied:** KV tensors. Nothing moves between workers; `dst`
  recomputes the prefix (or finds it in its own radix cache).
- **Ghosts:** when a worker fails or restarts, `forget_worker` drops its prefixes
  so the ledger does not believe a lost cache is still warm
  (`gateway/execution/dispatch.py`, `gateway/monitoring/polling.py`).
- **Evict:** the ledger keeps 1,024 prefixes (LRU). Every forgotten prefix is
  counted in `orch_hop_evictions_total{cause}`: `capacity` when the ledger is
  full (`HopLedger.evictions`), `worker_lost` when a worker's cache is gone
  (`gateway/execution/hops.py`). Tests: `tests/unit/test_hop_ledger.py`,
  `tests/unit/test_hops.py`. The ledger remembers the **set** of workers holding each prefix
  (`HopLedger.holders`), so returning to a worker that still holds it is not a
  hop. The earlier last-owner ledger over-counted (431,331 tokens was an upper
  bound, `metrics/guard-overflow-hops-2026-09-30.txt`). SGLang may still evict
  a prefix from its radix cache without telling us, so a counted non-hop can
  in rare cases be a recompute.

**Is it actually warm?** A replica is not ready when the weights are loaded.
The gateway's `prepare_worker` (`gateway/monitoring/readiness.py`) checks
health and the model, then sends **5 warmup requests on the shared application prefix**,
checks that metrics are fresh, and only then marks the worker ready.

| First user request on a fresh replica | TTFT proxy (prefill + 1 token) |
|---|---|
| Before app-prefix warmup (4 restarts) | 54, 72, 73, 72 ms, plus one ~100 ms outlier in 4/4 restarts |
| After app-prefix warmup (2 restarts) | 22, 23 ms, no outlier |
| Warm steady state | 15–17 ms |

Source: `metrics/warmup-first-token-2026-09-29.txt`. Through the whole serve path
(gateway, streaming): TTFT 18–35 ms warm (`metrics/stream-ttft-gateway-2026-09-29.txt`).
The SLO is quoted from warm replicas only.

## Part 7. Wire the app to the cluster

```
app turn (Router / Tutor tool call / Tutor answer)   app/llm.py -> GATEWAY
  -> guard.inspect                                   gateway/policies/guard.py
  -> admit                                           gateway/policies/admission.py
  -> place (select_worker) + hop record              gateway/policies/routing.py, gateway/execution/hops.py
  -> per-worker queue -> dispatch                    gateway/execution/queueing.py, dispatch.py
  -> SGLang /v1/chat/completions (stream)            gateway/execution/forwarding.py, gateway/api/streaming.py
  -> stay or leave                                   gateway/policies/overflow.py
```

Smoke before serve: direct worker checks in `metrics/h100-sxm5-hami-workers-2026-09-29.txt`
§6–8, then the first gateway completion (§9) and the first app turn (§11–13).

## Part 8. Proof

**Dashboards** (`monitoring/grafana/dashboards/`; `engine.json` also has gateway
queue depth and in-flight per worker, KV hops and hop-ledger evictions, and
stay-or-leave decisions):
- `overview.json`: scrape status (gateway, Prometheus, both workers).
- `gateway.json`: requests, sheds and placements since start.
- `engine.json`: sheds by reason, placements per worker, engine running and
  queued per worker, KV usage, TTFT p50/p99, generated tok/s, prefix cache hit
  rate and retracted requests.

Screenshots: `plots/grafana-2026-09-29/`.

| Question | Answer and evidence |
|---|---|
| What is the app; shared vs unique tokens? | Part 0 |
| What dies at guard / admit / place / queue? | Guard: `bad_max_tokens`, `prompt_too_long` (400). Admit: `kv_pressure`, `no_eligible_workers` (503). Place: `no_eligible_workers` (503). Queue: `queue_full` (503), `timeout_queue` (504). `orch_shed_total`, `orch_guard_rejected_total` |
| Where do I prevent work that will time out? | Queue deadline in `gateway/execution/waiting.py` and `dispatch.py`: an expired request is never dispatched (504 `timeout_queue`). Seen as 24 × 504 at dispatch cap 2 and 0 after the cap fix (`metrics/load-gateway-2026-09-29.txt`) |
| Where do I protect KV? | Admission `kv_usage_limit` 0.90 on fresh metrics; dispatch cap = `max_running_requests`; guard caps prompt + output ≤ 8192 |
| Where do I prioritize interactive traffic? | Gateway priority queue per worker (Part 5). Before, with FIFO: interactive p99 6.0 s vs batch 4.4 s, spread +1.6 s (`metrics/locust-class7-mix-2026-09-29.txt`). **GAP**: re-run the labelled Locust mix on the GPU and compare the spread. |
| Where do I stop one tenant owning the GPU? | `gateway/policies/tenant_window.py` (Part 3), 429 `tenant_tokens`. **GAP**: GPU evidence. |
| Where do I hop; what is not copied? | Part 6 |
| Where do I evict; what becomes a ghost? | Part 6: `forget_worker` on a lost worker (else its prefixes are ghosts: the ledger would call an empty cache warm) and ledger LRU; both counted in `orch_hop_evictions_total{cause}`. SGLang's own radix-cache eviction is inside the engine and not visible to the ledger. |
| Engine scheduler vs my admit/place/queue? | Part 2, "Two boxes"; Part 5 |
| What limited concurrency? | `max_running_requests=8` (and before that the gateway dispatch cap 2), not KV: Part 1 |
| Four production alerts | `monitoring/alerts.yaml`, loaded by both Prometheus configs: **KvCachePressure** (`sglang:full_token_usage > 0.85` for 2 m: just under the 0.90 admission limit, so `kv_pressure` sheds or engine retractions are next), **TtftSloBreach** (engine p99 TTFT > 1 s for 5 m per worker; warm TTFT is 18–35 ms, so 1 s means queueing), **HighShedRate** (> 5 % of chat requests shed, per reason, for 5 m: names *which* decision is refusing work), **WorkerDown** (`up{job="sglang"} == 0` for 1 m: capacity halved). Unit-tested with `promtool test rules monitoring/tests/alerts_test.yaml`. |
| If I scale, which pool? | Colocated replicas, so there is one pool, and on our mix the pressure is **decode slots** (97 % prefix hits make uncached prefill small). The first step is raising `--max-running-requests` while KV is at ~1 % and TPOT holds; after that, more GPU compute. Not "add a replica of the same size on the same GPU": that splits the same SMs and duplicates the prefix KV. |
| 10× traffic; three wrong knobs | 10×: tenant windows and interactive priority first; overflow for 503 only; raise engine concurrency against a TPOT SLO; a second physical GPU for compute. Wrong knobs: (1) raise `queue_max_size`/`queue_timeout_s` (hides overload as TTFT); (2) raise `kv_usage_limit` toward 1.0 (turns sheds into engine retractions); (3) add more HAMi replicas on the same GPU (same compute, less KV each, more hops). |

## Open work

Can be done without a GPU (code and tests), then proven on the GPU:

1. ~~Tenant token window~~: done in code (Part 3); needs GPU evidence.
2. ~~Interactive before batch~~: done in code, app and Locust traffic labelled
   (Part 5); needs GPU evidence (p99 spread).
3. ~~`prefix_then_load` placement + per-prefix worker set~~: done in code
   (Part 4, Part 6); needs GPU evidence (hops and p99 before/after).
4. ~~Export `orch_replica_queue_depth{worker}` and an evict counter~~: done in
   code (Part 5, Part 6); needs a Grafana panel and a GPU scrape.
5. ~~Ramp for a returning worker~~: done in code (Part 5); needs the kill-worker run on the GPU.
6. ~~Four alerts~~ and Grafana panels for queue depth, hops/evictions and overflow: done (`monitoring/alerts.yaml`, `engine.json`).
7. ~~Name the overflow model~~: decided (Part 3, "Overflow recipient"): Qwen3-8B on
   one owned 24 GB GPU; forwarding leaves to it is not implemented.
8. ~~Notebook~~: written (`notebook/part5_queue.ipynb`); run it with outputs on the GPU.
9. GPU cost/availability reasoning in Part 2.

Needs the GPU: live evidence for items 1–5 (tenant sheds on `revision-batch`,
p99 spread with priority), capacity at `max_running_requests`
> 8, a client-abort proof and the notebook run against live Prometheus.
