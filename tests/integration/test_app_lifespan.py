import importlib
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI


@pytest.fixture
def runtime(monkeypatch):
    from app.main import create_runtime_app
    from app.models.settings import AppSettings

    lifecycle = importlib.import_module("app.lifespan")
    real_http_client = httpx.AsyncClient

    def build(handler):
        requests = []
        http_clients = []
        sdk_clients = []
        http_options = []

        def gateway(request):
            requests.append(request)
            return handler(request)

        def create_http_client(**kwargs):
            http_options.append(kwargs)
            client = real_http_client(transport=httpx.MockTransport(gateway), **kwargs)
            http_clients.append(client)
            return client

        def create_sdk_client(**kwargs):
            client = AsyncOpenAI(**kwargs)
            sdk_clients.append(client)
            return client

        settings = AppSettings(
            _env_file=None,
            model_name="configured-model",
            gateway_url="http://configured-gateway:8780/v1",
            request_timeout_s=45,
            max_turns=4,
        )
        monkeypatch.setattr(lifecycle, "AppSettings", lambda: settings)
        monkeypatch.setattr(lifecycle, "httpx", SimpleNamespace(AsyncClient=create_http_client))
        monkeypatch.setattr(lifecycle, "AsyncOpenAI", create_sdk_client)
        return SimpleNamespace(
            app=create_runtime_app(), requests=requests, http_clients=http_clients,
            sdk_clients=sdk_clients, http_options=http_options, lifecycle=lifecycle,
        )

    return build


@pytest.mark.asyncio
@pytest.mark.parametrize("gateway_status", [200, 503])
async def test_runtime_uses_configured_gateway_and_closes_clients(runtime, gateway_status):
    def gateway(request):
        if gateway_status == 503:
            return httpx.Response(503, json={"detail": "no_eligible_workers"})
        return httpx.Response(200, json={
            "id": "completion-runtime",
            "object": "chat.completion",
            "created": 1,
            "model": "configured-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Test answer."},
                "finish_reason": "stop",
            }],
        })

    state = runtime(gateway)
    async with state.app.router.lifespan_context(state.app):
        assert state.app.state.max_turns == 4
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=state.app), base_url="http://app"
        ) as browser:
            response = await browser.post("/v1/chat/completions", json={
                "model": "course-assistant",
                "messages": [{"role": "user", "content": "Explain KV memory."}],
            })
        assert response.status_code == gateway_status
        if gateway_status == 200:
            assert response.json()["choices"][0]["message"]["content"] == "Test answer."
        assert not state.sdk_clients[0].is_closed()
        assert not state.http_clients[0].is_closed

    assert len(state.requests) == 1
    assert str(state.requests[0].url) == "http://configured-gateway:8780/v1/chat/completions"
    assert json.loads(state.requests[0].content)["model"] == "configured-model"
    assert state.sdk_clients[0].max_retries == 0
    assert state.sdk_clients[0].timeout == 45
    assert state.http_options[0]["trust_env"] is False
    assert len(state.http_clients) == len(state.sdk_clients) == 1
    assert state.sdk_clients[0].is_closed()
    assert state.http_clients[0].is_closed


@pytest.mark.asyncio
async def test_runtime_closes_clients_when_lifespan_body_raises(runtime):
    state = runtime(lambda request: pytest.fail("No inference expected"))

    with pytest.raises(RuntimeError, match="test lifecycle failure"):
        async with state.app.router.lifespan_context(state.app):
            raise RuntimeError("test lifecycle failure")

    assert state.sdk_clients[0].is_closed()
    assert state.http_clients[0].is_closed


@pytest.mark.asyncio
async def test_model_setup_failure_closes_created_clients(runtime, monkeypatch):
    state = runtime(lambda request: pytest.fail("No inference expected"))

    def fail_setup(*args, **kwargs):
        raise RuntimeError("test model setup failure")

    monkeypatch.setattr(state.lifecycle, "create_model", fail_setup)
    with pytest.raises(RuntimeError, match="test model setup failure"):
        async with state.app.router.lifespan_context(state.app):
            pytest.fail("Startup should have failed")

    assert state.sdk_clients[0].is_closed()
    assert state.http_clients[0].is_closed
