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
| GPU | H100 80GB SXM5 (Lambda, hourly) | Room for two replicas of an 8B BF16 model, each with a KV pool above max_len × 8. A 24 GB card fits one replica with ~4 GB KV. **GAP**: add cost/availability reasoning. |
| Model | Qwen3-8B BF16, thinking disabled | Reliable structured tool calls with SGLang's `qwen25` parser (`metrics/h100-sxm5-hami-workers-2026-09-29.txt` §12); GQA keeps KV at 144 KiB/token. |
| Engine | SGLang v0.5.20 (pinned digest) | Radix prefix cache for the shared agent prefix; Prometheus `/metrics`. |
| Topology | 2 colocated replicas (prefill + decode on each), HAMi memory slices on one GPU | The workload is a short, highly shared prefix (~97 % cache hit) with multi-step decode. A prefill/decode split would move KV for little gain. HAMi enforces the memory split inside the pod (`metrics/h100-sxm5-hami-workers-2026-09-29.txt` §4). |
| Concurrency | `--max-running-requests=8`, `--context-length=8192`, `--chunked-prefill-size=2048` | See Part 5. |
| Hop backend | `recompute`, named and counted by the gateway; no Mooncake | Same GPU, colocated replicas: moving KV would need a transfer engine for a prefix that recomputes in ~54–73 ms cold. See Part 6. |
| Overflow | **Disabled**; 503/529 are counted as `leave_disabled` | See Part 3. **GAP**: name the overflow model. |
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

**GAP: tenant admission (`tenant_tokens`)**. There is no per-tenant token window
yet, so one tenant can fill both queues.

**GAP: `should_shed(req, snap)` signature.** Admission is split between `admit`
(workers) and the queue (`queue_full`, `timeout_queue`). It could be presented as
one function returning `(shed, code, reason, retry_after)`.

## Part 4. Place

`gateway/policies/routing.py:select_worker`: **least-loaded with queue depth as a
scorer**.

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

**GAP: prefix-aware placement (`prefix_then_load`).** Half of all placements
move a shared prefix to the other worker (~200 hops each way in
`metrics/guard-overflow-hops-2026-09-30.txt`). Prefill and decode use the same
scorer because the replicas are colocated.

## Part 5. Queue: what runs next, and what does not

```
admit -> place -> gateway queue (per worker, FIFO, 16) -> dispatch (8 per worker)
      -> SGLang waiting -> running (8) -> retracted
```

| Question | Answer |
|---|---|
| Who sits in my queue vs the engine's? | Gateway queue: requests admitted but not dispatched (`gateway_queue_depth`). Engine waiting: dispatched but not yet running (`sglang:num_queue_reqs`). Because the dispatch cap (8) equals `--max-running-requests` (8), the engine queue stays ~0 and the backlog is visible in the gateway, where it can still be shed. |
| Waiting / running / retracted? | `sglang:num_queue_reqs`, `sglang:num_running_reqs`, `sglang:num_retracted_reqs`, all on the "Engine and Queues" dashboard. |
| Queue depth per pod? | **GAP**: `gateway_queue_depth` is used for routing but is not exported as a Prometheus gauge (`orch_replica_queue_depth{worker}`). |
| 32k RAG retrieve vs short agent decode? | A 32k prompt never gets in: guard rejects it (`prompt_too_long`, context 8192). Inside 8k the gateway is FIFO per worker (**GAP**: no interactive priority). The engine then interleaves: `--chunked-prefill-size=2048` splits a long prefill so running decodes keep stepping. |
| PagedAttention vs radix cache: which saved memory on the shared-prefix mix? | The radix cache. SGLang's token pool avoids fragmentation, but the savings on our mix come from reuse: ~97 % cache hit in Grafana, and a warm request recomputes 15–17 tokens instead of ~1,670 (`metrics/warmup-first-token-2026-09-29.txt`, `#cached-token: 1656`). |
| Chunked prefill / batching flags | `--chunked-prefill-size=2048` (limits prefill per step, protects decode TPOT); `--max-running-requests=8` (decode CUDA graphs captured for bs 1, 2, 4, 8); `max_prefill_tokens=16384` (default). |
| KV full after admit? | The gateway admits only below 90 % KV; after that the engine retracts (`num_retracted_reqs`). Not reached in any run (KV ≤ 1 %). The gateway does not fix OOM; it only avoids adding work above the limit. |
| Client gone? | Streaming: a client disconnect closes the gateway→engine stream, SGLang aborts the request and frees its KV. The shared prefix stays in the radix cache as evictable cache. Queue: `forward_until_done` cancels the upstream task when the request's result is already done (e.g. expired). **GAP**: prove the abort on the GPU (engine running count drops). |
| Worker returns: slam or ramp? | Today it slams: 8 requests within 2 s of warmup (`metrics/kill-worker-under-load-2026-09-29.txt`, recovery timeline). **GAP**: ramp 1→2→4→8 per poll while p99 holds. |

**GAP: notebook** (`notebook/`) that answers these from a live Prometheus scrape.

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
- **Evict:** the ledger keeps 1,024 prefixes (LRU). **GAP**: no evict counter,
  and the ledger remembers only the *last* worker per prefix, so hops onto a
  worker that still holds the prefix are over-counted (431,331 tokens is an
  upper bound, `metrics/guard-overflow-hops-2026-09-30.txt`).

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

**Dashboards** (`monitoring/grafana/dashboards/`):
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
| Where do I prioritize interactive traffic? | **GAP**. Measured need: interactive p99 6.0 s vs batch 4.4 s, spread +1.6 s (`metrics/locust-class7-mix-2026-09-29.txt`) |
| Where do I stop one tenant owning the GPU? | **GAP** (tenant token window) |
| Where do I hop; what is not copied? | Part 6 |
| Where do I evict; what becomes a ghost? | Part 6 (`forget_worker`; ledger LRU). **GAP**: evict count |
| Engine scheduler vs my admit/place/queue? | Part 2, "Two boxes"; Part 5 |
| What limited concurrency? | `max_running_requests=8` (and before that the gateway dispatch cap 2), not KV: Part 1 |
| Four production alerts | **GAP**: proposed in [Open work](#open-work) |
| If I scale, which pool? | Colocated replicas, so there is one pool, and on our mix the pressure is **decode slots** (97 % prefix hits make uncached prefill small). The first step is raising `--max-running-requests` while KV is at ~1 % and TPOT holds; after that, more GPU compute. Not "add a replica of the same size on the same GPU": that splits the same SMs and duplicates the prefix KV. |
| 10× traffic; three wrong knobs | 10×: tenant windows and interactive priority first; overflow for 503 only; raise engine concurrency against a TPOT SLO; a second physical GPU for compute. Wrong knobs: (1) raise `queue_max_size`/`queue_timeout_s` (hides overload as TTFT); (2) raise `kv_usage_limit` toward 1.0 (turns sheds into engine retractions); (3) add more HAMi replicas on the same GPU (same compute, less KV each, more hops). |

## Open work

Can be done without a GPU (code and tests), then proven on the GPU:

1. **Tenant token window** (`tenant_tokens` shed). Part 3.
2. **Interactive before batch** in the gateway queue, measured as p99 spread. Part 5, Part 8.
3. **`prefix_then_load` placement** and a ledger that remembers the set of
   workers per prefix; hops are counted only onto a cold worker. Part 4, Part 6.
4. **Export `orch_replica_queue_depth{worker}`** and an evict counter. Part 5, Part 6.
5. **Ramp for a returning worker** (1→2→4→8 while p99 holds). Part 5.
6. **Four alerts** (Prometheus rules), proposed:
   - `sglang:full_token_usage > 0.85` for 2 m: KV pressure; sheds are imminent.
   - p99 TTFT above the SLO for 5 m (warm replicas).
   - `rate(orch_shed_total[5m]) / rate(orch_requests_total[5m]) > 0.05`, by reason.
   - A worker unready or `up == 0` for 1 m (capacity halved; the other is at risk).
7. **Name the overflow model** (e.g. the same Qwen3-8B on a hosted provider, so
   the tool parser and prompts behave identically; interactive only, 503/529 only).
8. **Notebook** (`notebook/`) for Part 5 from a live Prometheus scrape.
9. GPU cost/availability reasoning in Part 2.

Needs the GPU: live evidence for items 1–5, capacity at `max_running_requests`
> 8, a client-abort proof and the notebook run against live Prometheus.
