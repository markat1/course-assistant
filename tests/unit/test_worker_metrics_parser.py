import pytest
from pydantic import ValidationError

from gateway.monitoring.parser import parse_worker_metrics


# Synthetic single-worker samples; names verified against SGLang v0.5.20:
# https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/observability/metrics_collector.py
# These are not captured GPU measurements or proof of deployment compatibility.
@pytest.fixture
def fields():
    return {
        "sglang:num_running_reqs": "engine_running",
        "sglang:num_queue_reqs": "engine_waiting",
        "sglang:full_token_usage": "kv_cache_usage_ratio",
    }


@pytest.fixture
def sample():
    return (
        'sglang:num_running_reqs{model_name="test-model"} 2\n'
        'sglang:num_queue_reqs{model_name="test-model"} 3\n'
        'sglang:full_token_usage{model_name="test-model"} 0.25\n'
    )


def test_selected_metrics_are_normalized_and_unrelated_metrics_ignored(fields, sample):
    metrics = parse_worker_metrics(
        sample + "sglang:token_usage 0.9\nvllm:kv_cache_usage_perc 0.8\n",
        fields=fields,
    )

    assert metrics.engine_running == 2
    assert metrics.engine_waiting == 3
    assert metrics.kv_cache_usage_ratio == 0.25


def test_parser_accepts_another_mapping_without_engine_specific_logic():
    metrics = parse_worker_metrics(
        "example:active 1\nexample:queued 0\nexample:memory_ratio 0.5\n",
        fields={
            "example:active": "engine_running",
            "example:queued": "engine_waiting",
            "example:memory_ratio": "kv_cache_usage_ratio",
        },
    )

    assert metrics.engine_running == 1
    assert metrics.engine_waiting == 0
    assert metrics.kv_cache_usage_ratio == 0.5


@pytest.mark.parametrize("missing_index", [0, 1, 2])
def test_incomplete_metrics_are_rejected(fields, sample, missing_index):
    lines = sample.splitlines()
    del lines[missing_index]

    with pytest.raises(ValidationError):
        parse_worker_metrics("\n".join(lines) + "\n", fields=fields)


def test_multiple_series_for_one_field_are_rejected(fields, sample):
    ambiguous = sample + 'sglang:num_running_reqs{model_name="another-model"} 4\n'

    with pytest.raises(ValueError, match="Duplicate metric"):
        parse_worker_metrics(ambiguous, fields=fields)


@pytest.mark.parametrize("usage", ["-0.1", "1.1", "NaN", "+Inf"])
def test_invalid_kv_ratio_is_rejected(fields, sample, usage):
    with pytest.raises(ValidationError):
        parse_worker_metrics(sample.replace("0.25", usage), fields=fields)


@pytest.mark.parametrize("running", ["-1", "1.5"])
def test_invalid_request_count_is_rejected(fields, sample, running):
    with pytest.raises(ValidationError):
        parse_worker_metrics(sample.replace('} 2\n', f'}} {running}\n'), fields=fields)
