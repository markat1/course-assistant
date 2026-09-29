from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class Hop:
    src: str
    dst: str
    prefix: str
    tokens: int
    backend: str = "recompute"


class HopLedger:
    """Remember where each shared prefix was last placed; nothing is copied between workers."""

    def __init__(self, max_prefixes: int = 1024) -> None:
        self._owners: OrderedDict[str, str] = OrderedDict()
        self._max_prefixes = max_prefixes

    def record(self, prefix: str, dst: str, *, tokens: int) -> Hop | None:
        src = self._owners.pop(prefix, None)
        self._owners[prefix] = dst
        if len(self._owners) > self._max_prefixes:
            self._owners.popitem(last=False)
        if src is None or src == dst:
            return None
        return Hop(src=src, dst=dst, prefix=prefix, tokens=tokens)

    def forget_worker(self, worker_id: str) -> int:
        """Drop prefixes of a worker whoe cache is gone, so they cannot become ghosts"""
        lost = [prefix for prefix, owner in self._owners.items() if owner == worker_id]
        for prefix in lost:
            del self._owners[prefix]
        return len(lost)