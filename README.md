# Course Assistant - design the cluster and serve an app

InferenceOps final project. A tool-using course agent (Track B) is served by our
own control plane in front of two SGLang workers on one GPU, and every decision
is proven with scrapes from the real cluster.

- **Answers with evidence:** [`DESIGN.md`](DESIGN.md) (Parts 0-8, the 13 questions).
- **How to bring it up and demo it:** [`docs/DEMO.md`](docs/DEMO.md).
- **Part 5 notebook:** [`notebook/part5_queue.ipynb`](notebook/part5_queue.ipynb), run against live Prometheus.
- **Raw evidence:** [`metrics/`](metrics) (command output, scrapes, counters), [`plots/`](plots) (Grafana from the cluster).

```text
Open WebUI -> app: Router -> handoff -> Tutor -> lookup_course     (Agents SDK, streaming)
   -> GATEWAY (box 1)  guard -> tenant window -> admit -> place -> per-worker
                       priority queue -> dispatch (cap 8) -> stay or leave
                       + warmup/readiness, ramp, hop ledger, /metrics
   -> ENGINE (box 2)   SGLang worker-a / worker-b, Qwen3-8B BF16, k3s StatefulSet,
                       one 38,000 MiB HAMi slice each on one 80 GB GPU
Prometheus scrapes gateway + both engines -> Grafana dashboards + 4 alert rules
```

## Pass conditions of the brief

| Condition | Met | Evidence |
|---|---|---|
| Real NVIDIA GPU | Lambda H100 80GB SXM5 (29 Sep), A100 80GB (30 Sep) | `metrics/h100-sxm5-*`, `metrics/*a100*` |
| At least two workers | 2 SGLang replicas, HAMi memory slices | `metrics/h100-sxm5-hami-workers-2026-09-29.txt` |
| Live `/metrics` scrape | Prometheus: gateway, worker-a, worker-b all `up` | same file §10; notebook cell 0 |
| Hop-or-warmup proof | cold vs warm first token, before/after gateway warmup; recompute hops counted | `metrics/warmup-first-token-2026-09-29.txt`, `metrics/locust-labelled-a100-2026-09-30.txt` |
| The app calls our serve path | Open WebUI -> app -> gateway -> SGLang, with tool calls | `metrics/app-streaming-2026-09-30.txt` |

## Part by part

| Part | What we built | Where | Proof |
|---|---|---|---|
| **0 App** | Track B agent: Router hands off to Tutor, Tutor calls `lookup_course` over a small course corpus and cites sources; OpenAI-compatible API with streaming; shared tokens = system prompt + tool schema, unique = the question | `app/` | Answer cites "KV memory and concurrency capacity"; streams in Open WebUI |
| **1 Capacity** | KV = 2 x 36 layers x 8 KV heads x 128 x 2 B = **144 KiB/token**; measured **115,299 tokens** per 38,000 MiB slice = ~14 seqs at 8,192, ~57 at a ~2,000-token turn. Hypothesis "the scheduler cap binds before KV": confirmed, and the very first limiter was our own gateway cap (2) | `DESIGN.md` Part 1 | SGLang startup log; KV peak 8.8 % under load |
| **2 Cluster** | 80 GB GPU (cheapest single GPU for two 8B BF16 replicas); 2 colocated replicas, HAMi slices; `--max-running-requests 8`, `--context-length 8192`, `--chunked-prefill-size 2048`; hop backend **recompute** (no Mooncake); overflow **Qwen3-8B on one owned 24 GB GPU** (decided, forwarding not wired); gateway and engine are two boxes | `cluster/`, `compose*.yaml` | HAMi limit: 30,000 MiB pod sees 29.3 GiB |
| **3 Guard, admit, stay/leave** | `inspect()` guard (400), tenant token window (429 `tenant_tokens`), admission on readiness + fresh metrics + KV < 0.90 (503), queue_full (503), timeout_queue (504); `stay_or_leave`: 429/500 stay, 503/529 may leave; Retry-After on capacity sheds; prompt tokens counted as characters / 4 or, with `GATEWAY_TOKEN_COUNTER=tokenizer`, by Qwen's tokenizer (off by default, GPU A/B owed) | `gateway/policies/` | 76 x 429 only on tenant `revision-batch`; all 429/504 "stay"; tokenizer count = engine `prompt_tokens` on 6/6 prompts (`metrics/token-count-2026-10-01.txt`) |
| **4 Place** | `pick` = `select_worker`: least-loaded on queue + in-flight + engine waiting (queue depth is a scorer), random tie-break, **prefix_then_load** (slack 4), ramp penalty; p2c equals least-loaded with two workers | `gateway/policies/routing.py` | Herding 5:1 -> even; hops 399 (old) -> 2 |
| **5 Queue** | Per-worker priority queue (interactive before batch), dispatch cap = engine cap, passive health, ramp for a returning worker, streaming relay that aborts the engine request when the client leaves | `gateway/execution/` | Interactive 0 failures vs batch to deadline; client abort frees the slot; ramp: no slam |
| **6 Hop and warmup** | Hop ledger per shared prefix (set of holders, evict on lost worker); warmup = 5 requests on the shared prefix before "ready" | `gateway/policies/hop_ledger.py`, `gateway/monitoring/` | First request on a fresh replica 54-73 ms -> **22-23 ms**; TTFT through the gateway 18-35 ms |
| **7 Wiring** | UI -> app -> guard -> tenant -> admit -> place -> queue -> engine -> stay/leave | `gateway/main.py` | Full agent turn through the gateway on the GPU |
| **8 Proof** | Grafana: sheds by reason, placements, queue depth by worker, engine running/queued, KV, TTFT, tokens/s, prefix hit rate + retractions, hops/evictions, stay/leave; 4 alerts | `monitoring/` | `TtftSloBreach`, `HighShedRate` fired under load; `WorkerDown` pending on a kill |

## The brief's questions

| # | Question | Short answer | Evidence |
|---|---|---|---|
| 1 | What is the app; shared vs unique tokens? | Course agent; shared = system prompt + tool schema (~97 % prefix-cache hits), unique = question / documents | DESIGN Part 0 |
| 2 | What dies at guard / admit / place / queue? | Guard 400 (`bad_max_tokens`, `prompt_too_long`); tenant 429; admit 503 (`no_eligible_workers`, `kv_pressure`); queue 503 `queue_full`, 504 `timeout_queue` | DESIGN Part 3 table |
| 3 | Where do I prevent work that will time out? | Queue deadline before dispatch: interactive 5 s, batch 30 s with a cap of half the queue (the 102 x 504 were measured with one 5 s deadline; not yet re-run) | `gateway/execution/waiting.py`; 102 x 504 on batch |
| 4 | Where do I protect KV? | Admission refuses above 0.90 KV on fresh metrics; dispatch cap = engine running cap | `gateway/policies/admission.py` |
| 5 | Where do I prioritise interactive traffic? | Per-worker priority queue on `X-Request-Class` | `gateway/policies/priority.py`; interactive 0 failures |
| 6 | Where do I stop one tenant owning the GPU? | Sliding token window per `X-Tenant`, 429 + Retry-After, never overflows | `gateway/policies/tenant_window.py`; 76 x 429 |
| 7 | Where do I hop, and what is not copied? | When a prefix lands on a worker that does not hold it; backend "recompute": no KV bytes move, the destination recomputes the prefix | `gateway/policies/hop_ledger.py`; 2 hops |
| 8 | Where do I evict; what becomes a ghost? | `forget_worker` when a worker is lost; otherwise the gateway would route to a "warm" prefix on a restarted, empty worker | `orch_hop_evictions_total` |
| 9 | Engine scheduler vs my admit/place/queue? | We decide what enters and where; SGLang decides batching, chunked prefill, KV allocation and retraction inside the batch | DESIGN Part 2 "two boxes"; notebook |
| 10 | What limited concurrency on this GPU? | First our gateway cap (2): 3.3x faster at 8; then the engine cap 8; KV never above 8.8 % | `metrics/load-gateway-2026-09-29.txt`, `metrics/locust-labelled-a100-2026-09-30.txt` |
| 11 | Four production alerts? | KvCachePressure, TtftSloBreach, HighShedRate, WorkerDown | `monitoring/alerts.yaml` (+ promtool tests) |
| 12 | If I scale, which pool? | Decode slots (prefill is mostly cached): raise the engine running cap while TPOT holds, then more GPU compute - not another replica on the same GPU | DESIGN Part 8 |
| 13 | 10x traffic; three wrong knobs? | Tenant windows + priority + overflow for 503 only + more compute; wrong: bigger queues/timeouts, KV limit toward 1.0, more replicas on the same GPU | DESIGN Part 8 |

## Bad answers we avoid

| Bad answer in the brief | Our position |
|---|---|
| Bench latency at batch 8 as the production SLO | SLOs from the app's own mix (Locust Class 7 shares) through the serve path |
| Cache full, so add a replica of the same size | KV peaked at 8.8 %; the limiter was concurrency caps, not KV |
| NCCL/NIXL in the repo moves KV | Hop backend is "recompute"; we state that nothing is copied |
| Ready when the weights are on the GPU | Ready = health + model + 5 validated warmup completions + fresh metrics |
| Overflow is "another API" with no model or limiter | Named model (Qwen3-8B, owned 24 GB GPU) and limiter (its KV pool) |
| Reimplemented the engine scheduler in the gateway | Gateway admits, places and queues; batching stays in SGLang |
| A 429 that overflowed | 429 is always "stay" (`orch_overflow_total{decision="stay"}`) |
| The gateway fixed OOM | HAMi limits memory; the gateway only refuses work |
| TTFT from a cold replica as the SLO | Warmup before ready; cold vs warm measured separately |
| Comparing wall seconds across models without token counts | One model throughout; runs on different GPUs are not compared |

## Layout (the brief's submission layout)

| Brief | Here |
|---|---|
| `app/` | [`app/`](app) - agents, tool, HTTP API |
| `control/` | [`gateway/`](gateway) - guard, tenant window, admit, place, queue, hop, readiness, stay/leave |
| `cluster/` | [`cluster/`](cluster), [`setup/`](setup), `compose*.yaml`, [`Makefile`](Makefile) (experiments) |
| `DESIGN.md` | [`DESIGN.md`](DESIGN.md) |
| `plots/` | [`plots/`](plots) |
| `metrics/` | [`metrics/`](metrics) |
| `notebook/` | [`notebook/`](notebook) |

Also: [`experiments/`](experiments) (Locust mix, load, first-token and TTFT tools),
[`monitoring/`](monitoring) (Prometheus, alerts, Grafana), [`tests/`](tests)
(pytest, simulated engines: `.venv/bin/python -m pytest -q`).
