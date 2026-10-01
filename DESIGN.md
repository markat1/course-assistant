# Course Assistant: design

The answers to the final project questions, each pointing at code or a scrape.
Items marked **GAP** are not implemented or not yet proven on the GPU; they are
listed together in [Open work](#open-work).

**Hardware per session.** 29 Sep: Lambda 1x H100 80GB SXM5. 30 Sep: Lambda 1x
A100-SXM4-80GB (no H100 was free), with the same two 38,000 MiB HAMi slices and
engine flags. Behavioural results (sheds, priority, hops, ramp, abort, alerts)
are valid on both; **latencies are only compared within the same GPU**.

## What we ship

Diagrams: `docs/shipping-pipeline.excalidraw` (the brief's pipeline next to ours),
`docs/architecture.excalidraw` (the whole cluster), `docs/agent-turn.excalidraw`
(one question as three calls to the gateway), `docs/capacity.excalidraw` (the Part 1
calculation step by step).

**The brief's pipeline, mapped to `gateway/main.py:chat_completions`:**

| Brief | Ours | Code |
|---|---|---|
| user / agent / retriever | every agent step, headers `X-Tenant`, `X-Request-Class` | `app/llm.py`, `app/lifespan.py:25` |
| guardrails (never reaches a GPU) | `inspect()` -> 400, before anything else | `main.py:66`, `policies/guard.py` |
| overflow gate (stay or leave, after the local result) | on every refusal and every non-200 engine answer | `api/chat.py:43` `record_overflow_decision`, `policies/overflow.py` |
| should we accept it? (admit) | tenant window (429), then `admit()` (503) | `main.py:88-107` |
| which worker? (place) | `select_worker`, `prefix_then_load` | `main.py:110`, `policies/routing.py` |
| your queue (same pod: no hop; two ids: hop) | `record_placement` records a hop when the prefix moves worker; then the worker's priority queue | `main.py:127`, `execution/hops.py`, `execution/queueing.py` |
| engine (prefill, decode, KV, waiting/running/preempt) | SGLang, unchanged, configured by flags | `cluster/workers/sglang.yaml` |
| 429 / 500 / slice_oom stay; 503 / 529 may leave | `LEAVE_STATUSES = {503, 529}`, counted in `orch_overflow_total{decision,code}` | `policies/overflow.py` |

The brief draws the overflow gate right after guardrails, but it decides on the
**local result**, which exists only after admit, the queue or the engine has
answered. Ours therefore runs where each result appears: in `reject_before_dispatch`
and after the engine's answer (`api/chat.py:104`). Part 7 of the brief also puts it last.
Evidence: 76 x 429 and 73 x 504 counted `stay`, 2 x 503 `leave_disabled`
(`metrics/locust-labelled-a100-2026-09-30.txt`). Two limits: `slice_oom` has no
separate code (an SGLang OOM returns 500 and stays), and upstream 502/504 (worker
dead, upstream timeout) are not counted by the gate.

**The four resources:**

| Resource | What controls it | Scarce? (measured) |
|---|---|---|
| Decode slots | `--max-running-requests=8` per worker; gateway dispatch cap 8 | **Yes, binds first**: engines ran exactly 8, the rest waited in our queue (Part 1) |
| KV blocks | 115,299 tokens per worker; admission stops at 90 % | No: peak 8.8 %, 0 retractions |
| Hop bandwidth | backend `recompute`: no bytes move | Paid as prefill time instead (cold prefix 54-73 ms vs warm 15-17 ms), so `prefix_then_load` keeps hops low: 2 in a 3-minute mix |
| Warmup time | 5 warmup requests on the shared prefix before ready, then a ramp | Yes, on restart: first request 54-73 ms without warmup, 22-23 ms with it |

**The six decisions we own:**

| Decision | Where |
|---|---|
| guard | `gateway/policies/guard.py` |
| admit | `gateway/policies/tenant_window.py`, `gateway/policies/admission.py` |
| place | `gateway/policies/routing.py:select_worker` |
| queue (hop) | `gateway/execution/queueing.py`, `gateway/execution/hops.py` |
| declare warm | `gateway/monitoring/readiness.py:prepare_worker` |
| stay vs leave | `gateway/policies/overflow.py`, `gateway/api/chat.py` |

**Not the scheduler.** FCFS, chunked prefill, batching and preemption stay inside
SGLang. We only set engine arguments (`--max-running-requests=8`,
`--chunked-prefill-size=2048`, `--context-length=8192`), which the brief allows.
The gateway decides what enters and where; SGLang decides what runs in which batch.

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
On the A100 80GB (30 Sep, same slice and flags) SGLang reported
`max_total_num_tokens=115916` (`metrics/a100-sglang-capacity-2026-09-30.txt`), so the
table below holds for both GPUs used.

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
(`metrics/kill-worker-under-load-2026-09-29.txt`). On the A100 80GB (same slices)
the labelled Class 7 mix with unique batch documents peaked at **8.8 % KV** with
**0 engine retractions** (Part 5 notebook, `metrics/locust-labelled-a100-2026-09-30.txt`):
KV is still far from being the limiter.

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
- `prompt_too_long` (400): prompt + output > `context_length` (8192). The prompt
  is counted as 4 chars/token by default, or with Qwen's tokenizer when
  `GATEWAY_TOKEN_COUNTER=tokenizer` ("Real token counts" at the end of this part).

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
`tenant_window_s` (60 s). A request costs its prompt tokens (the guard's count) plus
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
batch tenant is expected to hit `tenant_tokens` while students pass.

**Evidence (A100, 3-minute labelled mix):** 76 x 429 `tenant_tokens`, all on
`revision-batch`; 28 interactive and 4 agent students had **0 failures**. All 76
were classified `stay` (`orch_overflow_total{code="429",decision="stay"}`). With
the window effectively off (a second run), the priority queue alone pushed batch
to its deadline: 72 of 99 batch requests ended as 504 after waiting. The window
refuses the heavy tenant **early and cheaply** (429 + Retry-After before it takes
a queue slot) instead of letting it wait for a 504
(`metrics/locust-labelled-a100-2026-09-30.txt`).

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

### Real token counts in the gateway, switchable (1 October)

**Status: implemented and tested without a GPU; the default is unchanged.** The
numbers below were measured on 1 October in a Codespace (2 vCPU Xeon 8370C), with
no engine call: `experiments/token_count.py`, output in
`metrics/token-count-2026-10-01.txt`. The A/B run on the GPU is still owed (proof
plan, steps 2 and 3).

**Problem.** The gateway had no tokenizer and estimated tokens as characters / 4
(`CHARS_PER_TOKEN = 4`). Three decisions use the count:

| Where | Code | Effect of a wrong count |
|---|---|---|
| Guard `prompt_too_long` | `policies/guard.py:inspect` | Under-count: the request passes, takes a queue slot, and SGLang rejects it with its own 400. Over-count: we refuse a request that would have fit. |
| Tenant token window | `main.py:84` (`guard.prompt_tokens`, the guard's own number) | A tenant is charged more or less than it used. |
| Hop record `tokens` | `execution/hops.py:prefix_key` (first message only) | `orch_hop_tokens_total` is an estimate, not the recomputed amount. |

**Measured: the estimate over-counts our own prefix by 25 %.** Same six prompts as
`experiments/first_token.py` (the warmup prefix + one question each), compared with
the engine's own `usage.prompt_tokens` in `metrics/warmup-first-token-2026-09-29.txt`:

| | Tokens | Difference from the engine |
|---|---|---|
| Engine (`prompt_tokens`) | 1,671-1,674 | - |
| Qwen's chat template rendered locally, then tokenized | 1,671-1,674 | **0 on all six** |
| Gateway, `tokenizer` mode (contents + template tokens) | 1,671-1,674 | **0 on all six** |
| Qwen tokenizer on the message contents only | 1,654-1,657 | exactly 17 too few on all six |
| Characters / 4 (`estimate` mode) | 2,085-2,088 | 413-415 too many |

- The prefix is 8,307 characters and 1,648 tokens: 5.04 characters per token, not 4.
- The 17 is the chat template: 5 tokens per message (`<|im_start|>`, the role, a
  newline, `<|im_end|>`, a newline) and 7 for the start of the reply
  (`<|im_start|>assistant\n` plus the empty think block that
  `enable_thinking=false` adds). 2 x 5 + 7 = 17.
- Because the locally rendered template reproduces the engine's number, it is the
  reference for request shapes that have no engine measurement yet:

  | Request shape | Template | `tokenizer` | `estimate` |
  |---|---|---|---|
  | 1-8 plain messages | 20-137 | 0 | -9 to -53 |
  | system + user with 1-2 tools | 182-253 | 0 | -167 to -226 |
  | tool calls and tool results (3 shapes) | 255-341 | +2 to +6 | -199 to -267 |

- The estimate is wrong in both directions. It over-counts long prose by 25 %, so
  the guard refuses prose from about 6,500 real tokens, not 8,192. It counts only
  string `content`, so it does not see the `tools` schema or `tool_calls` at all and
  under-counts every agent step: the tenant window charges those too little.

**Decision.** Qwen's own tokenizer through the `tokenizers` library (Rust, no
torch) is a second counter, selected by a setting. The estimate stays the default,
so nothing changes until the switch is flipped.

| Option | Verdict |
|---|---|
| `tokenizers` + `tokenizer.json` from `Qwen/Qwen3-8B` | **Chosen.** Small dependency, in-process, no engine call. First load 2.0 s including the download, 0.55 s from the cache. |
| `transformers` `apply_chat_template(..., tools=...)` | Exact for every request shape, because it renders the same template as the engine, but a much heavier dependency for the gateway image. Next step if the gap measured on the GPU is too large. |
| Ask the engine to tokenize | Rejected: the guard must run before any engine work, and it adds a network call per request. |
| Replace the estimate outright | Rejected: no way back during a demo, and no before/after on the same GPU. |

**The switch.**

```text
GATEWAY_TOKEN_COUNTER=estimate    # default: characters / 4, as before
GATEWAY_TOKEN_COUNTER=tokenizer   # Qwen tokenizer + chat template tokens
```

- `Settings.token_counter: Literal["estimate", "tokenizer"] = "estimate"`.
- `.env.example` sets `tokenizer`, so a new deployment starts with the real
  count; without the variable the gateway uses the estimate.
- `make counter COUNTER=tokenizer` edits `.env`, restarts the gateway and prints
  the active counter, like `make slack`.
- The active counter is exported as `orch_token_counter_info{counter="..."} 1` and
  `make counters` saves it, so every saved scrape states which counter produced
  its sheds.

**Design.**

- `gateway/policies/token_count.py`:
  - `TokenCounter = Callable[[str], int]`, and `estimate_tokens(text)`.
  - `build_token_counter(settings)`: returns the estimate, or loads the tokenizer
    named by `model_name`. `tokenizers` is imported inside this function, so
    `estimate` mode never touches the library.
  - `count_prompt_tokens(payload, counter)`: with the estimate, exactly the old
    number. With any other counter: every string `content`, the `tools` schemas
    and `tool_calls` as the template renders them, plus the template tokens
    (5 per message, 7 for the reply, 76 for the tools instructions, 4 around each
    tool call and each tool result).
- `inspect()` and `record_placement()` take the counter as a parameter that
  defaults to `estimate_tokens`. **All 356 existing tests pass unchanged.**
- The prompt is counted **once** per request: `inspect()` returns the number in
  `Guard.prompt_tokens` and the tenant window is charged that number (before, it
  was computed twice).
- `gateway/lifespan.py` builds the counter once and stores it on `app.state`.
- **If the tokenizer cannot be loaded, the gateway falls back to the estimate and
  says so**: a warning in the log and `orch_token_counter_info{counter="estimate"}`.
  Checked by starting the gateway with the Hugging Face hub offline.
- The counter caches by text digest (LRU, 1,024 entries), so the shared prefix is
  tokenized once, not on every request: 5.5 ms uncached, 0.011 ms cached.
- A prompt above `MAX_CHARS_PER_TOKEN` (16) x `context_length` characters is not
  tokenized; it gets the estimate, which is over the limit by construction. A
  10 MB prompt is refused in 0.01 ms.
- `orch_prompt_token_difference{request_class}` is a histogram of the gateway's
  count minus the engine's `usage.prompt_tokens`, on non-streaming 200 answers
  (`api/chat.py:observe_engine_count`).

**What changed besides the code.**

- `tokenizers` is in `pyproject.toml` and `uv.lock`; the gateway image builds with
  `uv sync --locked` and loads the tokenizer inside the container.
- In `tokenizer` mode the gateway container needs internet at startup to fetch
  `tokenizer.json` (true on the Lambda host). Hardening, not done: fetch it in the
  Docker build, and pin the same revision as the workers.
- New tests only (`tests/unit/test_token_count.py`), with a fake counter and no
  network. One test uses the real tokenizer and asserts the six engine numbers
  above; it is skipped when the tokenizer cannot be loaded.
  `test_missing_output_budget_counts_as_the_limit_when_checking_length` stays an
  estimate-mode test: `"x" * 29600` is 7,401 by the estimate and 3,700 real tokens.

**Proof plan.**

1. Without a GPU (done, and kept as the unit test above): the tables above, and the
   real gateway in both modes with no worker: `"x" * 29600` is refused by the
   estimate and passes the guard with the tokenizer; 45,000 characters of prose
   (about 8,900 tokens) is refused by both.
2. On the GPU, same Locust mix twice, `make counter COUNTER=estimate` then
   `COUNTER=tokenizer`: compare `orch_guard_rejected_total{reason="prompt_too_long"}`,
   the 429 `tenant_tokens` count and `orch_hop_tokens_total`.
3. On the GPU: read `orch_prompt_token_difference` per request class. The claim to
   make is "the gateway's count is within N tokens of the engine's", with N
   measured on the engine for agent turns with tools, not only for system + user.

**Honest limits.**

- Only system + user is confirmed against the engine. The tools and tool-call rows
  compare with the template rendered locally, on hand-written requests, not with
  SGLang on real agent turns. Step 3 turns that into a measured number.
- The template tokens are constants for Qwen3 with thinking disabled. Another
  model or `enable_thinking=true` needs new constants.
- Streaming answers carry no `usage`, so they are not in the difference histogram.
- Tokenizing runs in the request handler, on the event loop: 5.5 ms for the
  prefix (once, then cached), 17 ms at 8,236 tokens, 59 ms worst case at the
  character cap. Measured on a 2 vCPU Codespace, not on the Lambda host.
- With the real count the guard loosens by about 25 % on prose, so more reaches the
  engine, while agent steps with tools are charged more than before. The 76 x 429
  and the shed counts in the saved runs were produced by the estimate and are not
  comparable with a `tokenizer` run without saying so.

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
  `tests/integration/test_prefix_placement.py`.
  **Evidence (A100):** a 3-minute labelled mix gave **2 hops in total** (one each
  way): each shared prefix (Tutor, agent) was recomputed once on the second
  worker, after which both held it; unique batch documents are not hops. The
  earlier 399 hops came from the last-owner ledger and over-counted. After two
  kill tests: 5 recompute hops and `orch_hop_evictions_total{cause="worker_lost"} 2`
  (`metrics/locust-labelled-a100-2026-09-30.txt`).
- Prefill and decode use the same scorer because the replicas are colocated.

**`pick(req, workers, *, policy)`.** `select_worker` is our `pick`; it returns a
worker, or `None`, which becomes the shed 503 `no_eligible_workers`. The policy is
set by configuration rather than a parameter: `prefix_load_slack = 0` gives
least-loaded with random tie-break (prefix only breaks ties), `> 0` gives
`prefix_then_load`. **p2c** (sample two workers, take the less loaded) is
identical to least-loaded with two workers, because the sample is always both;
it only pays off with many replicas, where scoring all of them is costly and
stale metrics make everyone pick the same "least loaded" one.

## Part 5. Queue: what runs next, and what does not

```
admit -> place -> gateway queue (per worker, priority: interactive first, 16) -> dispatch (8 per worker)
      -> SGLang waiting -> running (8) -> retracted
```

| Question | Answer |
|---|---|
| Who sits in my queue vs the engine's? | Gateway queue: requests admitted but not dispatched (`gateway_queue_depth`). Engine waiting: dispatched but not yet running (`sglang:num_queue_reqs`). Because the dispatch cap (8) equals `--max-running-requests` (8), the engine queue stays ~0 and the backlog is visible in the gateway, where it can still be shed. |
| Waiting / running / retracted? | `sglang:num_queue_reqs`, `sglang:num_running_reqs`, `sglang:num_retracted_reqs`, all on the "Engine and Queues" dashboard. |
| Queue depth per pod? | `orch_replica_queue_depth{worker}` (waiting in our queue) and `orch_replica_in_flight{worker}` (dispatched, not finished), set from `WorkerState` on every `/metrics` scrape (`gateway/main.py:metrics`). Next to `sglang:num_queue_reqs` this shows who waits where. Tests: `tests/integration/test_queue_depth_metrics.py`. Grafana panel "Gateway queue depth and in-flight per worker"; notebook plot `plots/part5-queue-depth-by-pod.png` over the labelled mix on the A100. |
| 32k RAG retrieve vs short agent decode? | A 32k prompt never gets in: guard rejects it (`prompt_too_long`, context 8192). Inside 8k, **we** decide the order in our queue: interactive before batch (below). Once dispatched, **the engine** decides: `--chunked-prefill-size=2048` splits a long prefill so running decodes keep stepping. |
| Interactive vs batch in my queue? | Each worker queue is an `asyncio.PriorityQueue` (`gateway/lifespan.py:create_queues`). `X-Request-Class: interactive\|batch` (default `interactive`; unknown → 400 `bad_request_class` at the guard) maps to a priority in `gateway/policies/priority.py`; `QueuedRequest` orders by (priority, arrival) so a class stays FIFO (`gateway/models/queued_request.py`). Batch can starve while interactive keeps arriving and then expires as `timeout_queue` (504): a deliberate "what we do not run". This orders only our queue; it is not the engine scheduler. Tests: `tests/unit/test_request_priority.py`, `tests/integration/test_request_class.py`. **Evidence (A100):** interactive and agent 0 failures; batch 102 x 504 (with the tenant window) and 72 of 99 x 504 (without). **Lesson:** one 5 s queue deadline for every class does not fit batch (Abi's Class 7 trace gives batch 30 s); a per-class deadline is the next change. |
| PagedAttention vs radix cache: which saved memory on the shared-prefix mix? | The radix cache. SGLang's token pool avoids fragmentation, but the savings on our mix come from reuse: ~97 % cache hit in Grafana, and a warm request recomputes 15–17 tokens instead of ~1,670 (`metrics/warmup-first-token-2026-09-29.txt`, `#cached-token: 1656`). |
| Chunked prefill / batching flags | `--chunked-prefill-size=2048` (limits prefill per step, protects decode TPOT); `--max-running-requests=8` (decode CUDA graphs captured for bs 1, 2, 4, 8); `max_prefill_tokens=16384` (default). |
| KV full after admit? | The gateway admits only below 90 % KV; after that the engine retracts (`num_retracted_reqs`). Not reached in any run: peak KV 8.8 %, 0 retractions (A100 labelled mix, notebook). The gateway does not fix OOM; it only avoids adding work above the limit. |
| Client gone? | Streaming: a client disconnect closes the gateway→engine stream, SGLang aborts the request and frees its KV. The shared prefix stays in the radix cache as evictable cache. Queue: `forward_until_done` cancels the upstream task when the request's result is already done (e.g. expired). **Evidence (A100):** a streaming request with `max_tokens` 1000 (~15-25 s of decode); the client leaves after 2 s -> 3 s after the start `sglang:num_running_reqs` is 0 on both workers; control with the client staying -> 1.0 (`metrics/client-abort-a100-2026-09-30.txt`, `make abort`). |
| Worker returns: slam or ramp? | Ramp. Before, it slammed: 8 requests within 2 s of warmup (`metrics/kill-worker-under-load-2026-09-29.txt`, recovery timeline). Now a worker that becomes ready starts at `ramp_limit = 1` (`gateway/monitoring/polling.py:refresh_worker`); each metrics poll (5 s) doubles it up to the dispatch cap (1 → 2 → 4 → 8 in 15 s) while the engine queue is empty, and halves it when `sglang:num_queue_reqs > 0` (`gateway/policies/ramp.py`). The engine queue is our proxy for "p99 holds": once SGLang makes requests wait, TTFT is rising. Routing adds `dispatch_max − ramp_limit` to the worker's load (`gateway/policies/routing.py:gateway_load`), so a returning worker gets traffic only when the warm one is busier. It is a soft cap: if the warm worker is full, the ramping one still serves rather than shedding. Tests: `tests/unit/test_ramp.py`, `tests/integration/test_worker_ramp.py`, `tests/integration/test_prefix_placement.py`. **Evidence (A100, 1,200 requests, worker-b killed 10 s in):** 1,200/1,200 completed; on return: health, models, five warmup completions, then **0 user requests for 19 s** (ramp limit 1 + penalty, and prefix affinity to the warm holder), then back to its normal share (~4 per 3 s) within ~7 s. With 8 clients each worker carries ~4 in flight, so the ramp shows as a delayed re-entry, not a visible 1->2->4->8 staircase (`metrics/kill-ramp-a100-2026-09-30.txt`). |

**Notebook:** `notebook/part5_queue.ipynb` answers these from a live Prometheus
scrape (queue table per worker, queue depth by pod, door sheds vs engine
retractions, radix cache vs KV usage, hops and sheds, ramp after a kill) and
saves its plots to `plots/`. Query helpers: `experiments/prometheus_snapshot.py`
(tested in `tests/unit/test_prometheus_snapshot.py`). **Run with outputs** against the
live A100 cluster (all four targets up): plots `plots/part5-queue-depth-by-pod.png`,
`part5-shared-prefix-kv.png`, `part5-hops-and-sheds.png`, `part5-ramp-after-return.png`.
Saved run: window 22:30-00:00 on 30 Sep (`WINDOW_MIN=90`), covering the kill tests and three Locust runs. Run it with `PROM_URL=http://127.0.0.1:29090 WINDOW_MIN=90 uv run --with jupyter
--with matplotlib jupyter nbconvert --to notebook --execute --inplace notebook/part5_queue.ipynb`.

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

**Why recompute and not Mooncake: what a KV transfer would save on this mix.**
Estimated from our own measurements; no Mooncake was run.

1. A hop costs the cold prefill once: 54-73 ms cold vs 15-17 ms warm, so ~40-55 ms extra
   (`metrics/warmup-first-token-2026-09-29.txt`).
2. Hops are rare: 2 in a 3-minute labelled run of ~680 requests, 0 in the rehearsal
   run (`metrics/locust-labelled-a100-2026-09-30.txt`). Upper bound of the saving:
   2 x ~55 ms ≈ 0.1 s per 3 minutes.
3. Moving the KV is not free: the hop record's prefix (922 tokens x 144 KiB) is
   ≈ 136 MB, the full app prefix (~1,670 tokens) ≈ 246 MB.

| Path | Bandwidth (assumed) | 246 MB takes | vs recompute (~40-55 ms) |
|---|---|---|---|
| TCP 10 Gbit/s (no RDMA on one Lambda host) | ~1.25 GB/s | ~200 ms | slower |
| Host RAM (both workers on one machine) | ~10+ GB/s | ~25 ms | slightly faster |
| RDMA 200 Gbit/s | ~25 GB/s | ~10 ms | faster |

Break-even: prefill ran ~1,670 tokens in ~45 ms ≈ 37,000 tokens/s, so a transfer must
deliver 147 KB x 37,000/s ≈ 5.5 GB/s to beat recompute. On this mix Mooncake would
save at most ~0.1 s per 3 minutes, and over TCP it would be slower than recomputing.
It pays off for long prefixes (prefill grows faster than linearly with length) or
with RDMA. The bandwidths are assumptions, not measurements. A switch
(`GATEWAY_HOP_BACKEND=recompute|mooncake`, default recompute, refused unless the
workers really use a Mooncake store) is planned but not implemented.

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

Screenshots: `plots/grafana-2026-09-29/`; notebook plots `plots/part5-*.png`.
- **Success and failures:** failures per reason are on the sheds and stay-or-leave
  panels; there is no dedicated completions-by-status panel yet (Open work).
- **Pods / replicas / KEDA:** no autoscaling. Two fixed replicas on one GPU; the
  scaling answer is below ("If I scale, which pool?").

| Question | Answer and evidence |
|---|---|
| What is the app; shared vs unique tokens? | Part 0 |
| What dies at guard / admit / place / queue? | Guard: `bad_max_tokens`, `prompt_too_long` (400). Admit: `kv_pressure`, `no_eligible_workers` (503). Place: `no_eligible_workers` (503). Queue: `queue_full` (503), `timeout_queue` (504). `orch_shed_total`, `orch_guard_rejected_total` |
| Where do I prevent work that will time out? | Queue deadline in `gateway/execution/waiting.py` and `dispatch.py`: an expired request is never dispatched (504 `timeout_queue`). Seen as 24 × 504 at dispatch cap 2 and 0 after the cap fix (`metrics/load-gateway-2026-09-29.txt`) |
| Where do I protect KV? | Admission `kv_usage_limit` 0.90 on fresh metrics; dispatch cap = `max_running_requests`; guard caps prompt + output ≤ 8192 |
| Where do I prioritize interactive traffic? | Gateway priority queue per worker (Part 5). Before, with FIFO (H100): interactive p99 6.0 s vs batch 4.4 s (`metrics/locust-class7-mix-2026-09-29.txt`). With priority (A100): interactive and agent 0 failures while batch is pushed to its deadline (102 x 504). A clean same-GPU p99 spread needs a FIFO baseline on the A100: not measured (Open work). |
| Where do I stop one tenant owning the GPU? | `gateway/policies/tenant_window.py` (Part 3): 76 x 429 `tenant_tokens`, all on `revision-batch`, 0 failures for students (`metrics/locust-labelled-a100-2026-09-30.txt`). |
| Where do I hop; what is not copied? | Part 6 |
| Where do I evict; what becomes a ghost? | Part 6: `forget_worker` on a lost worker (else its prefixes are ghosts: the ledger would call an empty cache warm) and ledger LRU; both counted in `orch_hop_evictions_total{cause}`. SGLang's own radix-cache eviction is inside the engine and not visible to the ledger. |
| Engine scheduler vs my admit/place/queue? | Part 2, "Two boxes"; Part 5 |
| What limited concurrency? | `max_running_requests=8` (and before that the gateway dispatch cap 2), not KV: Part 1 |
| Four production alerts | `monitoring/alerts.yaml`, loaded by both Prometheus configs: **KvCachePressure** (`sglang:full_token_usage > 0.85` for 2 m: just under the 0.90 admission limit, so `kv_pressure` sheds or engine retractions are next), **TtftSloBreach** (engine p99 TTFT > 1 s for 5 m per worker; warm TTFT is 18–35 ms, so 1 s means queueing), **HighShedRate** (> 5 % of chat requests shed, per reason, for 5 m: names *which* decision is refusing work), **WorkerDown** (`up{job="sglang"} == 0` for 1 m: capacity halved). Unit-tested with `promtool test rules monitoring/tests/alerts_test.yaml`. **Live (A100):** TtftSloBreach fired on both workers and HighShedRate fired under the labelled mix; WorkerDown went pending when a worker was killed and did not fire because the pod returned within its 1 m `for` window; KvCachePressure did not fire (KV ≤ 8.8 %). |
| If I scale, which pool? | Colocated replicas, so there is one pool, and on our mix the pressure is **decode slots** (97 % prefix hits make uncached prefill small). The first step is raising `--max-running-requests` while KV is at ~1 % and TPOT holds; after that, more GPU compute. Not "add a replica of the same size on the same GPU": that splits the same SMs and duplicates the prefix KV. |
| 10× traffic; three wrong knobs | 10×: tenant windows and interactive priority first; overflow for 503 only; raise engine concurrency against a TPOT SLO; a second physical GPU for compute. Wrong knobs: (1) raise `queue_max_size`/`queue_timeout_s` (hides overload as TTFT); (2) raise `kv_usage_limit` toward 1.0 (turns sheds into engine retractions); (3) add more HAMi replicas on the same GPU (same compute, less KV each, more hops). |

## Open work

Done and proven on the GPU (see the parts above): tenant window, interactive
priority, `prefix_then_load` with the per-prefix holder set, queue depth per
worker, evictions against ghosts, ramp for a returning worker, client abort,
four alerts (three observed live), the overflow decision, the Part 5 notebook with
outputs, GPU cost reasoning.

Still open, in order of value:

1. **Per-class queue deadline** (batch ~30 s, interactive 5 s): today batch ends
   as 504 after waiting.
2. **Predictive `timeout_queue`**: refuse before enqueue when queue depth x recent
   service time exceeds the deadline (503) instead of 504 after the wait.
3. **Overflow forwarding**: the recipient is decided (Qwen3-8B on an owned 24 GB
   GPU) but 503/529 leaves are only counted (`leave_disabled`).
4. **Clean p99 spread on one GPU**: a FIFO baseline on the same hardware as the
   priority run (priority cannot be switched off by configuration today).
5. **Completions-by-status panel** ("success and failures") in Grafana.
6. **Capacity above `--max-running-requests=8`** (e.g. 16) against a TPOT SLO.
7. **Streaming status while tools run**: the first visible word comes after
   Router, handoff, Tutor and the lookup.
8. **Token counter A/B on the GPU**: the same Locust mix with
   `GATEWAY_TOKEN_COUNTER=estimate` and `tokenizer`, and the difference from the
   engine's `prompt_tokens` for agent turns (Part 3, "Real token counts").
9. `pyproject.toml` has no `notebook` dependency group; the notebook runs with
   `uv run --with jupyter --with matplotlib` (Part 5).
