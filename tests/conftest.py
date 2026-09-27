import asyncio
import importlib
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.models.chat.chat_request import ChatRequest
from gateway.models.queued_request import QueuedRequest
from gateway.models.workers.worker_state import WorkerState


@pytest.fixture
def chat_payload():
    return ChatRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=32,
    )


@pytest.fixture
def worker():
    return WorkerState(id="worker-a", base_url="http://worker-a:8000/v1")


@pytest.fixture
def request_queue():
    return asyncio.Queue[QueuedRequest](maxsize=4)


@pytest_asyncio.fixture
async def deadline_timer(monkeypatch):
    waiting_module = importlib.import_module("gateway.execution.waiting")
    loop = asyncio.get_running_loop()
    registered = asyncio.Event()
    fired = asyncio.Event()
    handles = []
    delays = []

    def schedule(delay, callback, *args):
        def invoke():
            callback(*args)
            fired.set()

        handle = loop.call_later(delay, invoke)
        handles.append(handle)
        delays.append(delay)
        registered.set()
        return handle

    proxy = SimpleNamespace(get_running_loop=lambda: SimpleNamespace(call_later=schedule))
    monkeypatch.setattr(waiting_module, "asyncio", proxy)
    return SimpleNamespace(
        handles=handles, delays=delays, registered=registered, fired=fired
    )


