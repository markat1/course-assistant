# Course Assistant - setup and demo walkthrough

Step-by-step guide to bring the cluster up on a Lambda GPU and show, question by
question, how the design answers the final-project brief. Answers with evidence are
in [`DESIGN.md`](../DESIGN.md); this file is the order to run and show them.

Lambda's firewall only allows SSH. Keep the Step 2 session open - it tunnels the
ports. On the laptop open only `127.0.0.1` URLs:

- Gateway (our control plane) - http://127.0.0.1:18780
- App (Router/Tutor agents, OpenAI-compatible) - http://127.0.0.1:18781
- Open WebUI - http://127.0.0.1:23001 - after Step 8. Model `course-assistant`.
- Grafana - http://127.0.0.1:23000 - user `admin`, password `admin`. Dashboard
  "Course Assistant Engine and Queues".
- Prometheus - http://127.0.0.1:29090

## The setup in one picture

```text
Open WebUI -> app (Router -> Tutor -> lookup_course, streaming)
           -> GATEWAY (box 1): guard -> tenant window -> admit -> place -> per-worker
              priority queue -> dispatch (cap 8) -> stay-or-leave; warmup/readiness,
              ramp, hop ledger
           -> ENGINE (box 2): SGLang worker-a / worker-b (Qwen3-8B BF16), k3s
              StatefulSet, one 38,000 MiB HAMi slice each on one 80 GB GPU
Prometheus scrapes gateway + both engines -> Grafana, 4 alert rules
```

| Piece | Where | Why |
|---|---|---|
| GPU | Lambda H100 80GB SXM5 (or A100 80GB when no H100 is free) | Cheapest single GPU that holds two 8B BF16 replicas with a useful KV pool |
| Slicing | HAMi, 2 x 38,000 MiB (`cluster/workers/sglang.yaml`) | Memory split enforced inside CUDA (29.3 GiB visible for a 30,000 MiB pod) |
| Engine | SGLang v0.5.20, `--max-running-requests=8`, `--context-length=8192`, `--mem-fraction-static=0.85` | Radix prefix cache for the shared agent prefix; 115,299 KV tokens per worker |
| Gateway + app + monitoring | Docker Compose on the same host (`compose*.yaml`) | Gateway and engine are two separate boxes |
| Overflow | Decided: Qwen3-8B on one owned 24 GB GPU; forwarding not implemented | Same model and tool parser; we own KV and warmup |

---

## Bring the cluster up

### Step 1

Laptop terminal, in `course-assistant`. Everything must be committed.

```
bash setup/sync_to_gpu.sh ubuntu@<host> /home/markt/.ssh/id_ed25519
```

### Step 2

Laptop terminal. Keep this session open: it forwards all ports above.

```
bash setup/ssh_gpu.sh ubuntu@<host> /home/markt/.ssh/id_ed25519
```

All following steps run on the GPU host unless they say "laptop".

### Step 3

```
bash setup/check_gpu_host.sh; bash setup/install_k3s.sh
```

Expect the GPU name, a Ready node and `RuntimeClass nvidia`.

### Step 4

```
nohup sudo k3s ctr images pull docker.io/lmsysorg/sglang:v0.5.20-cu130@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f > ~/image-pull.log 2>&1 &
```

About 2 minutes, 14 GB. Check with `tail -1 ~/image-pull.log`.

### Step 5

```
bash setup/install_hami.sh
```

Expect `hami-scheduler` and `hami-device-plugin` rolled out.

### Step 6

```
bash setup/deploy_workers.sh
```

Both `sglang-0` and `sglang-1` must be `1/1 Running`.

### Step 7

```
cp .env.example .env && printf '\n' >> .env && sed -i 's#http://worker-a:8000/v1#http://host.docker.internal:30001/v1#; s#http://worker-b:8000/v1#http://host.docker.internal:30002/v1#' .env
```

### Step 8

```
sudo docker compose -p course-assistant -f compose.yaml -f compose.monitoring.yaml -f compose.k3s.yaml -f compose.app.yaml -f compose.ui.yaml up -d --build
```

### Step 9

```
make check
```

Expect both pods Ready, `max_total_num_tokens`, four Prometheus targets `up` and
four alert rules. Output is saved in `evidence/`.

---

## Demo: what to show, and which question it answers

Every `make` target writes its output to `evidence/<timestamp>-<name>.txt`.

### Step 10 - the application through the serve path (Part 0, Part 7)

Browser: Open WebUI, model `course-assistant`, a new chat. Enable "Stream Chat
Response" in Settings -> General -> Advanced Parameters if the answer arrives in
one piece. Ask: *"How does KV cache size limit concurrent requests?"*

Show: the answer streams; it cites the passage "KV memory and concurrency
capacity" (only reachable through the `lookup_course` tool). Path: UI -> app ->
Router -> handoff -> Tutor -> tool -> gateway -> SGLang.

### Step 11 - capacity on paper vs measured (Part 1)

Show the `make check` output: `max_total_num_tokens=115299` on the H100, 115916 on the
A100 80GB (`metrics/a100-sglang-capacity-2026-09-30.txt`). 15.84 GiB / 115,299
tokens = 144 KiB/token = 2 x 36 layers x 8 KV heads x 128 x 2 bytes. ~14
sequences at 8,192 tokens, ~57 at a ~2,000-token turn, so the engine's running
cap (8) binds before KV. First measured limiter was our own gateway cap (2).

### Step 12 - the labelled Class 7 mix (Part 3, Part 5, Part 8)

```
make locust
make counters
```

Grafana while it runs. Show:
- 28 interactive + 4 agent users (own tenant each), 8 batch users sharing tenant
  `revision-batch`.
- `orch_shed_total{reason="tenant_tokens",code="429"}` only on batch: **one tenant
  cannot own the GPU**; 429 is "stay", never overflow.
- Interactive and agent with 0 failures, batch pushed to its queue deadline
  (`timeout_queue` 504): **interactive is prioritised** in our per-worker queue.
- `orch_hop_total` small: `prefix_then_load` keeps shared prefixes on the worker
  that holds them.

### Step 13 - a worker dies and returns (Part 5 "slam or ramp", Part 6, alerts)

`experiments/load.py` sends no `X-Tenant`, so all its requests share tenant `default`
and hit the tenant window (429) instead of testing the kill. Raise the budget first
(and remove the line again before a Locust run that should show `tenant_tokens`):

```
echo "GATEWAY_TENANT_MAX_TOKENS=100000000" >> .env && sudo docker compose -p course-assistant -f compose.yaml -f compose.monitoring.yaml -f compose.k3s.yaml -f compose.app.yaml -f compose.ui.yaml up -d gateway
make kill
```

`make kill` sends 800 requests; the ramp was clearly visible with 1,200 (see
`metrics/kill-ramp-a100-2026-09-30.txt`).

Show: 0 or a handful of 502s (dead worker marked unready at once, no black hole);
gateway log "Worker refresh failed for worker-b"; on return `/health`, `/v1/models`
and five warmup completions on the shared prefix before the worker is ready; then
the ramp 1 -> 2 -> 4 -> 8. Prometheus alerts:

```
curl -s localhost:19090/api/v1/alerts | python3 -c "import json,sys; [print(a['labels']['alertname'], a['state'], a['labels'].get('worker','')) for a in json.load(sys.stdin)['data']['alerts']]"
```

### Step 14 - a client leaves (Part 5 "client gone: who frees the KV?")

```
make abort
```

Show: after the client drops, `sglang:num_running_reqs` is back to 0: the gateway
closed the engine stream and SGLang aborted the request and freed its KV.

### Step 15 - the Part 5 notebook

Laptop terminal:

```
PROM_URL=http://127.0.0.1:29090 WINDOW_MIN=90 uv run --with jupyter --with matplotlib jupyter lab notebook/part5_queue.ipynb
```

Run all cells: who waits in our queue vs the engine's queue, per worker; plots.
`WINDOW_MIN` is how far back the plots look - make it cover the runs you want to show.
To execute and save the outputs without opening Jupyter:

```
PROM_URL=http://127.0.0.1:29090 WINDOW_MIN=90 uv run --with jupyter --with matplotlib jupyter nbconvert --to notebook --execute --inplace notebook/part5_queue.ipynb
```

### Step 16 - warmup and hop evidence (Part 6)

Not a live step: show `metrics/warmup-first-token-2026-09-29.txt`. Cold shared
prefix 54-73 ms vs warm 15-17 ms; a fresh replica paid one ~100 ms request in 4/4
restarts; with the gateway's warmup (five requests on the shared prefix) the first
user request took 22-23 ms. Streaming TTFT through the gateway: 18-35 ms.

---

## The brief's questions - where to point

| Question | Show |
|---|---|
| What is the app; shared vs unique tokens? | Step 10; DESIGN Part 0 (shared Tutor/tool prefix, unique question/documents) |
| What dies at guard / admit / place / queue? | DESIGN Part 3 table: guard 400, tenant 429, admit 503, queue_full 503, timeout_queue 504 |
| Where do I prevent work that will time out? | Queue deadline (`gateway/execution/waiting.py`); Step 12 504s on batch |
| Where do I protect KV? | Admission `kv_usage_limit` 0.90; dispatch cap = engine running cap |
| Where do I prioritise interactive traffic? | Per-worker priority queue (`gateway/policies/priority.py`); Step 12 |
| Where do I stop one tenant owning the GPU? | `gateway/policies/tenant_window.py`; Step 12 429s |
| Where do I hop; what is not copied? | `gateway/policies/hop_ledger.py`: backend "recompute", no KV bytes move; Step 12 counters |
| Where do I evict; what becomes a ghost? | `forget_worker` on a lost worker; `orch_hop_evictions_total` |
| Engine scheduler vs my admit/place/queue? | DESIGN Part 2 "two boxes"; notebook Step 15 |
| What limited concurrency? | Step 11: first our gateway cap 2, then the engine cap 8, not KV |
| Four production alerts | `monitoring/alerts.yaml`; Step 13 |
| If I scale, which pool? | DESIGN Part 8: decode slots on this mix |
| 10x traffic; wrong knobs | DESIGN Part 8 |

## If the GPU is not running

Every step above has saved evidence: `metrics/*.txt` (raw output) and
`plots/grafana-*` (screenshots). Walk through them in the order of Steps 10-16.

## Shut down

Laptop: commit `evidence`-derived files you want to keep, then terminate the
instance in the Lambda console and check that it shows "terminated".
