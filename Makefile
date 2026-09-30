# GPU-session experiments. Run on the GPU host in the release directory.
# Every target tees its output to evidence/<timestamp>-<name>.txt.
COMPOSE = sudo docker compose -p course-assistant -f compose.yaml -f compose.monitoring.yaml -f compose.k3s.yaml
EVIDENCE = evidence/$(shell date +%Y%m%d-%H%M%S)

.PHONY: check locust counters kill abort slack

check: ## pods, KV capacity, Prometheus targets and alert rules
	mkdir -p evidence && { sudo k3s kubectl get pods -l app=sglang; sudo k3s kubectl logs sglang-0 | grep -i -E "max_total_num_tokens|load weight end"; curl -s localhost:19090/api/v1/targets | python3 -c "import json,sys; [print(t['labels']['job'], t['labels'].get('worker','-'), t['health']) for t in json.load(sys.stdin)['data']['activeTargets']]"; curl -s localhost:19090/api/v1/rules | grep -o '"name":"[A-Za-z]*"' | sort -u; } | tee $(EVIDENCE)-check.txt

locust: ## labelled Class 7 mix; USERS (default 40), MINUTES (default 3)
	mkdir -p evidence && $(COMPOSE) -f compose.load.yaml run --rm --no-deps locust -f /mnt/experiments/locustfile.py --host http://gateway:8780 --headless -u $(or $(USERS),40) -r 2 -t $(or $(MINUTES),3)m --only-summary 2>&1 | tee $(EVIDENCE)-locust.txt

counters: ## gateway decision counters (sheds, hops, evictions, overflow, guard)
	mkdir -p evidence && curl -s localhost:8780/metrics | grep -E "^orch_(shed_total|hop_total|hop_evictions_total|overflow_total|guard)" | tee $(EVIDENCE)-counters.txt

kill: ## kill worker-b 10 s into an 8-concurrent load (ramp, WorkerDown alert)
	mkdir -p evidence && { (sleep 10 && sudo k3s kubectl delete pod sglang-1 --wait=false) & $(COMPOSE) run --rm --no-deps -T -e CONCURRENCY=8 -e REQUESTS=800 --entrypoint /app/.venv/bin/python gateway - < experiments/load.py 2>/dev/null; } | tee $(EVIDENCE)-kill.txt

abort: ## client leaves a streaming request after 2 s; engine running must drop to 0
	mkdir -p evidence && { timeout 2 curl -N -s localhost:8780/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Write a long essay about GPUs."}],"max_tokens":1000,"stream":true}' >/dev/null; sleep 1; curl -s localhost:30001/metrics localhost:30002/metrics | grep "^sglang:num_running_reqs"; } | tee $(EVIDENCE)-abort.txt

slack: ## set GATEWAY_PREFIX_LOAD_SLACK=$(SLACK) in .env and restart the gateway
	sed -i '/^GATEWAY_PREFIX_LOAD_SLACK=/d' .env && echo "GATEWAY_PREFIX_LOAD_SLACK=$(SLACK)" >> .env && $(COMPOSE) -f compose.app.yaml -f compose.ui.yaml up -d gateway && grep PREFIX_LOAD_SLACK .env
