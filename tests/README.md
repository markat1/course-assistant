# Gateway tests

## Run

```bash
uv run python -m pytest -q
uv run python -m pytest tests/unit -q
uv run python -m pytest tests/integration -q
```

Alert rules have their own promtool tests (needs Docker):

```bash
docker run --rm -v "$PWD/monitoring:/monitoring" --entrypoint promtool \
  prom/prometheus:v3.4.1 test rules /monitoring/tests/alerts_test.yaml
```

All current pytest tests run locally without Docker, network access or a GPU; one
test in `unit/test_token_count.py` compares Qwen's real tokenizer with the engine's
recorded counts and is skipped when the tokenizer cannot be loaded. Integration
tests connect gateway components in process; HTTPX transports simulate the worker
boundary. Passing these tests does not establish real worker readiness, GPU memory
release, streaming latency or client-disconnect behavior.

## Framework choice

We retain pytest and pytest-asyncio. Rob Myers recommends Python's unittest in
*Essential Test-Driven Development* (2026), printed page 24 (PDF page 59). That is
his framework recommendation, not a requirement of the testing principles.

Our existing suite uses pytest fixtures and asynchronous tests. A framework
migration would add work without addressing the test smells found in this review.
The refactor uses pytest's existing features; it adds no dependencies.

- [pytest fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
- [pytest-asyncio concepts](https://pytest-asyncio.readthedocs.io/en/stable/concepts.html)
- [HTTPX transports](https://www.python-httpx.org/advanced/transports/)

## Test boundaries

- `unit/`: focused behaviors such as deadline decisions, queue cleanup, response
  waiting and HTTP forwarding. External HTTP calls are replaced at the transport.
- `integration/`: interactions between queues, dispatch, worker-task lifecycle,
  HTTP endpoints and metrics. Keep shutdown and recovery scenarios here.
- `conftest.py`: fresh request, worker and queue fixtures, plus the timer observer
  shared by waiting tests. Request fixtures explicitly; avoid global mutable data.

Test modules describe behavior, not a one-to-one copy of production files. Unit
tests may exercise several functions when those functions implement one behavior.

## Applying the book

The following rules paraphrase Myers; the PDF remains a private local reference.
The printed page numbers below are 35 less than the PDF page numbers.

| Principle | How we apply it | Printed pages |
| --- | --- | --- |
| Given, when, then | Separate setup, action and expected outcome with whitespace. | 23–24 |
| Clear, focused, independent and repeatable tests | Give scenarios descriptive names and fresh state. Use events to coordinate concurrent work. | 80–83 |
| Test behavior | Assert observable results and resource cleanup. Avoid exact task-name strings as a substitute for checking worker execution. | 88–89 |
| Reduce repeated setup | Share small fixtures; keep scenario-specific data next to the test. | 102–105 |
| Avoid branching scenarios | Give success, errors and cancellation separate tests. Parametrize values only when the behavior and assertions are the same. | 120 |
| Keep integration tests focused | Preserve HTTP and shutdown coverage; test deadline decisions independently. | 123–124, 161 |

Multiple assertions are appropriate when they describe one outcome. A shutdown
test may need to verify cancellation, an empty queue and a closed client. Do not
split away the ordering guarantee that makes the scenario meaningful.

Use a controlled clock at deadline boundaries. Short asyncio wait timeouts are
hang guards, not performance expectations. The HTTP queue-timeout test deliberately
exercises a real timer; it does not assert elapsed milliseconds.

Cleanup belongs in fixtures or `finally` blocks where tasks and clients are
allocated. Tests of cleanup itself still need to observe the production cleanup.
Do not hide unexpected exceptions or remove failure cases just to get a green run.

## Next feature: readiness

Use the red–green–refactor cycle for new behavior:

1. Describe one readiness outcome with a failing test.
2. Confirm it fails because the behavior is missing.
3. The student implements the smallest change that satisfies it.
4. Run the tests, then simplify the code with tests still passing.

The existing tests were largely written after implementation. Refactoring them
improves the regression suite; it does not retroactively make that work TDD.

Next distinguish unavailable, loaded and warmed workers, then prove the path with
a real worker. Keep measured GPU evidence separate from simulated test results.
