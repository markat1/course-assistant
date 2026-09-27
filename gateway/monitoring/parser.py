from collections.abc import Mapping

from prometheus_client.parser import text_string_to_metric_families
from gateway.models.workers.worker_metrics import WorkerMetrics

def parse_worker_metrics(text: str,*,fields: Mapping[str, str]) -> WorkerMetrics:
    """Parse required metrics from a single-engine worker endpoint."""
    values: dict[str, float] = {}

    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            field = fields.get(sample.name)

            if field is None:
                continue

            if field in values:
                raise ValueError(f"Duplicate metric: {sample.name}.")

            values[field] = sample.value

    return WorkerMetrics.model_validate(values)