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
    """Remember which workers hold each shared prefix; nothing is copied between workers."""

    def __init__(self, max_prefixes: int = 1024) -> None:
        self._holders: OrderedDict[str, list[str]] = OrderedDict()
        self._max_prefixes = max_prefixes
        self.evictions = 0

    def record(self, prefix: str, dst: str, *, tokens: int) -> Hop | None:
        holders = self._holders.pop(prefix, [])
        self._holders[prefix] = holders
        
        if len(self._holders) > self._max_prefixes:
            self._holders.popitem(last=False)
            self.evictions += 1
        
        if not holders:
            holders.append(dst)
            return None

        if dst in holders:
            holders.remove(dst)
            holders.append(dst)
            return None
        
        src = holders[-1]
        holders.append(dst)
        return Hop(src=src, dst=dst, prefix=prefix, tokens=tokens)
    
    def holders(self, prefix: str) -> set[str]:
        return set(self._holders.get(prefix,()))

    def forget_worker(self, worker_id: str) -> int:
        """Drop a worker whose cache is gone from every prefix, so it cannot become a ghost."""
        lost = 0
        for prefix in list(self._holders):
            holders = self._holders[prefix]
            if worker_id in holders:
                holders.remove(worker_id)
                lost += 1
                if not holders:
                    del self._holders[prefix]
        return lost
   