def next_dispatch_limit(current: int, *, maximum: int, engine_waiting: int) -> int:
    """Double while engine keeps up, halve when it queues; stay within 1..maximum."""
    if engine_waiting > 0:
        return max(1, current // 2)
    return min(maximum, current * 2)