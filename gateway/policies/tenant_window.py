import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True)
class TenantDecision:
    ok: bool
    status: int = 200
    reason: str = "ok"
    retry_after_s: int | None = None


class TenantWindow:
    """Limit each tenant to max_tokens within a sliding window_s seconds"""
    def __init__(self,
                 max_tokens: int,
                 window_s: float,
                 clock: Callable[[], float] = monotonic) -> None:
        self._max_tokens = max_tokens
        self._window_s = window_s
        self._clock = clock
        self._spent: dict[str, deque[tuple[float, int]]] = {}

    def admit(self, tenant: str, tokens: int) -> TenantDecision:
        now = self._clock()
        spent = self._spent.setdefault(tenant, deque())

        while spent and now - spent[0][0] >= self._window_s:
            spent.popleft()

        used = sum(amount for _, amount in spent)
        if used + tokens <= self._max_tokens:
            spent.append((now, tokens))
            return TenantDecision(ok=True)

        return TenantDecision(
            ok=False,
            status=429,
            reason="tenant_tokens",
            retry_after_s=self._retry_after(spent, now, used + tokens - self._max_tokens)
        )

    def _retry_after(self, spent: deque[tuple[float, int]], now: float, excess: int) -> int:
        """Seconds until enough of the oldest tokens leave the window."""
        freed = 0
        for admitted_at, amount in spent:
            freed += amount
            if freed >= excess:
                return math.ceil(admitted_at + self._window_s - now)
        return math.ceil(self._window_s)
