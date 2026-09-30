# Course Assistant: serving an agent on our own cluster

InferenceOps final project. A Track B tool-using agent (Router -> Tutor ->
`lookup_course`) is served by our own control plane in front of two SGLang
workers (Qwen3-8B BF16) on one H100, sliced with HAMi under k3s.

```text
Open WebUI -> app (agents) -> gateway (guard, tenant window, admit, place, queue,
hop record, warmup/readiness, stay-or-leave) -> SGLang worker-a / worker-b (k3s)
                                  \-> Prometheus -> Grafana
```

**Read [`DESIGN.md`](DESIGN.md) first**: it answers Parts 0-8 and the 13 questions
of the brief, each with a pointer to code, a scrape or a plot. Items marked
**GAP** are not implemented or not yet proven on the GPU.

## Layout (mapped to the brief's submission layout)

| Brief | Here | Contents |
|---|---|---|
| `app/` | [`app/`](app) | Agents SDK Router and Tutor, course lookup tool, OpenAI-compatible HTTP API with streaming; calls only the gateway |
| `control/` | [`gateway/`](gateway) | Guard (`policies/guard.py`), tenant window, admission, placement (`policies/routing.py`, `prefix_then_load`), per-worker priority queue and dispatch (`execution/`), hop ledger, warmup/readiness and polling (`monitoring/`), stay-or-leave (`policies/overflow.py`) |
| `cluster/` | [`cluster/`](cluster), [`setup/`](setup), `compose*.yaml` | k3s config, HAMi Helm chart, SGLang StatefulSet and NodePorts; install and deploy scripts; Compose for gateway, app, UI and monitoring |
| `DESIGN.md` | [`DESIGN.md`](DESIGN.md) | Answers with evidence |
| `plots/` | [`plots/`](plots) | Grafana screenshots from the cluster |
| `metrics/` | [`metrics/`](metrics) | Raw scrapes and command output: KV capacity, warmup, load, kill-worker, Locust, streaming TTFT, guard/overflow/hops |
| `notebook/` | [`notebook/part5_queue.ipynb`](notebook/part5_queue.ipynb) | Part 5 queue analysis against live Prometheus |
| - | [`experiments/`](experiments) | Traffic and measurement tools: `locustfile.py` (Class 7 mix), `load.py`, `first_token.py`, `stream_ttft.py`, `prometheus_snapshot.py` |
| - | [`monitoring/`](monitoring) | Prometheus configs, four alert rules with promtool tests, Grafana dashboards |
| - | [`tests/`](tests) | pytest suite (unit and integration, simulated engines) |
| - | [`docs/`](docs) | Runbooks and handoff notes |

## Run the tests

```bash
uv sync
.venv/bin/python -m pytest -q
promtool test rules monitoring/tests/alerts_test.yaml   # needs promtool
```

## Bring the cluster up on a GPU host

Summary of the steps used in the GPU sessions (details in `docs/`):

```bash
# laptop
bash setup/sync_to_gpu.sh ubuntu@<host> <ssh-key>
bash setup/ssh_gpu.sh ubuntu@<host> <ssh-key>        # also forwards Grafana, Prometheus, UI

# GPU host, in the release directory
bash setup/install_k3s.sh
sudo k3s ctr images pull docker.io/lmsysorg/sglang:v0.5.20-cu130@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f
bash setup/install_hami.sh
bash setup/deploy_workers.sh
cp .env.example .env && printf '\n' >> .env
sed -i 's#http://worker-a:8000/v1#http://host.docker.internal:30001/v1#; s#http://worker-b:8000/v1#http://host.docker.internal:30002/v1#' .env
sudo docker compose -p course-assistant -f compose.yaml -f compose.monitoring.yaml \
  -f compose.k3s.yaml -f compose.app.yaml -f compose.ui.yaml up -d --build
```

Local ports through `ssh_gpu.sh`: gateway 18780, app 18781, Grafana 23000,
Open WebUI 23001, Prometheus 29090. In Open WebUI, enable "Stream Chat Response"
in the user settings if answers arrive in one piece.
