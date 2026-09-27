from prometheus_client.parser import text_string_to_metric_families
from gateway.models.workers.worker_metrics import WorkerMetrics

METRIC_FIELDS = {
    "vllm:num_requests_running": "engine_running",
    "vllm:num_requests_waiting": "engine_waiting",
    "vllm:kv_cache_usage_perc": "kv_cache_usage_ratio",
}

def parse_worker_metrics(text: str) -> WorkerMetrics:
    """Parse required metrics from a single-engine worker endpoint."""
    values: dict[str, float] = {}

    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            field = METRIC_FIELDS.get(sample.name)

            if field is None:
                continue

            if field in values:
                raise ValueError(f"Duplicate metric: {sample.name}.")

            values[field] = sample.value

    return WorkerMetrics.model_validate(values)