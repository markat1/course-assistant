# H100 deployment and smoke tests

Status: local browser-to-gateway rejection smoke passed; no remote deployment
or real GPU result recorded yet. App and UI packaging is committed through
968e395. The app/UI SSH forwards are implemented in setup/ssh_gpu.sh and
shell syntax validation passes; remote connectivity remains untested.
Run commands one at a time and check their output before continuing.

## What the two SSH scripts do

- `setup/sync_to_gpu.sh` runs on the laptop. It sends the latest local commit to
  `~/course-assistant/releases/<commit>` on the GPU host. It does not start services.
  Uncommitted changes are not sent. Local `.env` and private documents are absent
  from the current Git archive. The remote snapshot has no `.git` directory.
- `setup/ssh_gpu.sh` also starts on the laptop. It opens a shell on the GPU host,
  enters that same commit directory, and forwards service ports to the laptop.
  Keep that terminal open while using the forwarded addresses.
- Neither script provisions a cloud instance. Sync a commit before opening its
  SSH session. Commit new setup files before syncing them.

## 1. Choose the GPU session

Use one H100 80 GB with two colocated SGLang replicas. They share GPU memory and
compute; the configured memory fractions are not hardware isolation. The image
uses CUDA 13, so the host driver and NVIDIA Container Toolkit must be compatible.

Before renting, record the instance type, console price, spending cap and intended
termination time. No instance creation is authorized by this document.
[Lambda's public prices](https://lambda.ai/pricing), checked 27 September 2026,
list single-GPU H100 PCIe at $3.29/hour and H100 SXM at $4.29/hour, before applicable
taxes. A proposed two-hour smoke session would cost $6.58 or $8.58 in instance
charges at those rates; actual availability, price and extra charges must be
checked in the console. This is a proposed timebox, not an approved budget.

## 2. Transfer and connect — laptop terminal

Replace both example values with the new instance's SSH address and your key path.
Do not put the private key in the repository or copy it onto the GPU host.

```bash
bash setup/sync_to_gpu.sh ubuntu@YOUR_GPU_IP /absolute/path/to/your/key
```

```bash
bash setup/ssh_gpu.sh ubuntu@YOUR_GPU_IP /absolute/path/to/your/key
```

Verify the new host's fingerprint when SSH asks. Subsequent commands run in the
remote shell. The SSH script sets `COMPOSE_PROJECT_NAME=course-assistant`; commands
below also use `-p course-assistant` explicitly so volumes retain a stable name.

## 3. Check the host and prepare remote settings — GPU terminal

```bash
bash setup/check_gpu_host.sh
```

Record GPU name, memory and driver plus Docker/Compose versions. If a prerequisite
is missing, resolve that specific host issue before proceeding. See the official
[NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
if Docker cannot expose the GPU; do not reinstall a working host blindly.

Create settings from the public example if this release has no `.env` yet:

```bash
test -e .env || cp .env.example .env
```

The example model must remain `Qwen/Qwen3-8B`, with worker URLs
`http://worker-a:8000/v1` and `http://worker-b:8000/v1`. Localhost browser ports are
not worker URLs inside Docker. Copy any deliberate setting changes into each new
release's remote `.env`; the sync script never sends the laptop's `.env`.

```bash
docker compose -p course-assistant -f compose.yaml -f compose.monitoring.yaml -f compose.workers.yaml -f compose.app.yaml -f compose.ui.yaml config --quiet
```

## 4. Test CUDA inside the pinned image — GPU terminal

This may download the SGLang image. It starts a temporary container using worker A's
GPU configuration, but does not start the model server or download model weights.

```bash
docker compose -p course-assistant -f compose.workers.yaml run --rm --no-deps -T --entrypoint python3 worker-a - < setup/check_gpu_container.py
```

Require `PASS: CUDA allocation and BF16 matrix multiplication.` Save the reported
versions and GPU identity. A passed `nvidia-smi` or Python syntax check is insufficient.

## 5. Start both engines — GPU terminal

Start A first and wait for startup. Initial model download/kernel preparation can
be slow; inspect logs rather than interpreting every initial connection refusal
as a broken gateway.

```bash
docker compose -p course-assistant -f compose.workers.yaml up -d worker-a
```

```bash
docker compose -p course-assistant -f compose.workers.yaml logs --tail=80 worker-a
```

```bash
curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8001/health
```

After A's health check succeeds, start B and check its startup in the same way:

```bash
docker compose -p course-assistant -f compose.workers.yaml up -d worker-b
```

```bash
docker compose -p course-assistant -f compose.workers.yaml logs --tail=80 worker-b
```

```bash
curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8002/health
```

These commands intentionally select worker services. Do not use `--remove-orphans`
when operating with a subset of the project's Compose files.

## 6. Smoke-test each engine directly — GPU terminal

Build the gateway image to obtain our probe code and locked Python dependencies:

```bash
docker compose -p course-assistant -f compose.yaml -f compose.workers.yaml build gateway
```

```bash
docker compose -p course-assistant -f compose.yaml -f compose.workers.yaml run --rm --no-deps -T --entrypoint /app/.venv/bin/python gateway - < setup/smoke_workers.py
```

The script requires at least two worker URLs and checks health, served model,
a valid bounded completion and fresh parsed SGLang metrics. Require a PASS for
both worker-a and worker-b. Errors exit unsuccessfully. The temporary probe does
not start the gateway server or mark workers ready in a running gateway.

This smoke warms the workers. It does not demonstrate cold/warm latency, tool
compatibility, app execution through the gateway, or stability under load.

## 7. Start gateway and monitoring — GPU terminal

```bash
docker compose -p course-assistant -f compose.yaml -f compose.monitoring.yaml -f compose.workers.yaml up -d gateway prometheus grafana
```

```bash
docker compose -p course-assistant -f compose.yaml -f compose.monitoring.yaml -f compose.workers.yaml ps -a
```

On the laptop, with the SSH terminal still open:

| Service | Laptop address |
| --- | --- |
| Gateway | http://127.0.0.1:18780 |
| Grafana | http://127.0.0.1:23000 |
| Prometheus | http://127.0.0.1:29090 |
| Worker A | http://127.0.0.1:28001 |
| Worker B | http://127.0.0.1:28002 |

In Prometheus, check `up{job="sglang"}`: both worker targets should be 1 after a
scrape. Check `up{job="gateway"}` too. Then verify the actual required engine
metrics; target health alone does not prove the parser accepts the scrape.

## 8. Start the agent app and browser UI - GPU terminal

Build the app from the uploaded commit and start it with Open WebUI:

```bash
docker compose -p course-assistant -f compose.yaml -f compose.monitoring.yaml -f compose.workers.yaml -f compose.app.yaml -f compose.ui.yaml up -d --build app open-webui
```

Open WebUI connects to http://app:8781/v1. The app connects to the gateway;
its configured inference model comes from GATEWAY_MODEL_NAME. The browser model
alias is course-assistant. UI defaults disable streaming and the changelog popup;
Compose is authoritative because ENABLE_PERSISTENT_CONFIG=false. Local demo
login is disabled; all host publications remain bound to loopback.

First startup can take time for database setup and UI embedding-model download.
Check readiness on the GPU host:

```bash
curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:13001/health
```

The laptop SSH script includes these additional forwards:

```text
127.0.0.1:18781 -> GPU host 127.0.0.1:8781 (app)
127.0.0.1:23001 -> GPU host 127.0.0.1:13001 (Open WebUI)
```

With that SSH session open, visit http://127.0.0.1:23001 on the laptop. Send
"Explain KV cache capacity." Save the response and evidence of the agent's
handoff/tool requests through the gateway. A visible answer alone does not prove
the lookup tool ran. no_eligible_workers indicates that gateway worker readiness
must be investigated; it is not a successful GPU smoke. Buffered responses do
not provide a true time-to-first-token measurement.

## Evidence and remaining work

Save the commit ID (release directory name), launch settings, GPU/driver/image
versions, both startup logs, smoke outputs and raw engine scrapes before teardown.
Keep secrets and personal connection details out of public evidence.

The next application proof must send actual agent/tool steps through the gateway.
A direct worker smoke is only a setup check. Remaining assignment work includes
capacity measurements, cold/warm comparison, queue/admission/fairness evidence,
stay/leave policy, notebook and four alerts. Do not mark these complete based on
this runbook. Prepare the application locally while a GPU session is unavailable.

## End the paid session

After copying the required evidence back to the laptop, stop the project:

```bash
docker compose -p course-assistant -f compose.yaml -f compose.monitoring.yaml -f compose.workers.yaml -f compose.app.yaml -f compose.ui.yaml down
```

Do not add `-v` unless deliberately deleting cached models and monitoring data.
Stopping Docker does not terminate the rented instance. Terminate it in the
provider console and confirm its final state. Download needed files first; do not
rely on instance-local data surviving termination.

For a later code revision, sync a new commit, prepare its remote `.env`, and
recreate services from that release using the same Compose project name. Running
containers keep their old bind mounts until recreated; uploading new code alone
is not a deployment.
