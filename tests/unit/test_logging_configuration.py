import os
import re
import subprocess
import sys

import pytest
from pydantic import ValidationError

from gateway.models.settings import Settings


@pytest.fixture
def settings_values(monkeypatch):
    monkeypatch.delenv("GATEWAY_LOG_LEVEL", raising=False)
    return {
        "_env_file": None,
        "model_name": "test-model",
        "worker_urls": {"worker-a": "http://worker-a:8000/v1"},
    }


def test_log_level_defaults_to_info(settings_values):
    assert Settings(**settings_values).log_level == "INFO"


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_log_level_can_be_set_from_environment(settings_values, monkeypatch, level):
    monkeypatch.setenv("GATEWAY_LOG_LEVEL", level)
    assert Settings(**settings_values).log_level == level


def test_invalid_log_level_is_rejected(settings_values):
    with pytest.raises(ValidationError) as caught:
        Settings(**settings_values, log_level="VERBOSE")
    assert any(error["loc"] == ("log_level",) for error in caught.value.errors())


def run_logging_script(script):
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=5,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
    assert result.returncode == 0, result.stderr
    return result.stderr


def test_logging_has_timestamp_level_name_and_no_duplicate_records():
    output = run_logging_script('''
import logging
from gateway.monitoring.logging_config import configure_logging
configure_logging("INFO")
configure_logging("INFO")
logger = logging.getLogger("gateway.monitoring.polling")
logger.debug("hidden diagnostic")
logger.info("Worker ready")
try:
    raise RuntimeError("engine failure")
except RuntimeError:
    logger.exception("Worker failed")
''')
    assert "hidden diagnostic" not in output
    assert output.count("Worker ready") == 1
    assert output.count("Worker failed") == 1
    assert re.search(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*INFO.*gateway.monitoring.polling.*Worker ready",
        output,
    )
    assert "ERROR" in output
    assert "Traceback (most recent call last)" in output
    assert "RuntimeError: engine failure" in output


def test_reconfiguration_changes_gateway_level():
    output = run_logging_script('''
import logging
from gateway.monitoring.logging_config import configure_logging
configure_logging("INFO")
configure_logging("DEBUG")
logging.getLogger("gateway.monitoring.polling").debug("diagnostic detail")
configure_logging("ERROR")
logging.getLogger("gateway.monitoring.polling").warning("hidden warning")
''')
    assert output.count("diagnostic detail") == 1
    assert "hidden warning" not in output
