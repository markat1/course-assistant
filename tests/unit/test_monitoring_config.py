import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MONITORING = ROOT / "monitoring"


def dashboard_expressions(name: str) -> str:
    """All PromQL expressions in a provisioned Grafana dashboard, joined."""
    dashboard = json.loads((MONITORING / "grafana" / "dashboards" / name).read_text())
    expressions = []

    def collect(panels):
        for panel in panels:
            expressions.extend(target.get("expr", "") for target in panel.get("targets", []))
            collect(panel.get("panels", []))

    collect(dashboard.get("panels", []))
    return "\n".join(expressions)


def test_engine_dashboard_shows_queue_depth_and_in_flight_per_worker():
    expressions = dashboard_expressions("engine.json")

    assert "orch_replica_queue_depth" in expressions
    assert "orch_replica_in_flight" in expressions


def test_engine_dashboard_shows_hops_evictions_and_overflow_decisions():
    expressions = dashboard_expressions("engine.json")

    assert "orch_hop_total" in expressions
    assert "orch_hop_evictions_total" in expressions
    assert "orch_overflow_total" in expressions


def test_alert_rules_file_exists_next_to_its_tests():
    assert (MONITORING / "alerts.yaml").is_file()


def test_both_prometheus_configs_load_the_alert_rules():
    for config in ("prometheus.yaml", "prometheus.k3s.yaml"):
        text = (MONITORING / config).read_text()

        assert "rule_files:" in text, config
        assert "/etc/prometheus/alerts.yaml" in text, config


def test_prometheus_container_mounts_the_alert_rules():
    compose = (ROOT / "compose.monitoring.yaml").read_text()

    assert "./monitoring/alerts.yaml:/etc/prometheus/alerts.yaml:ro" in compose
