from gateway.policies.ramp import next_dispatch_limit


def test_limit_doubles_while_the_engine_keeps_up():
    assert next_dispatch_limit(1, maximum=8, engine_waiting=0) == 2
    assert next_dispatch_limit(2, maximum=8, engine_waiting=0) == 4


def test_limit_never_exceeds_the_dispatch_cap():
    assert next_dispatch_limit(4, maximum=6, engine_waiting=0) == 6
    assert next_dispatch_limit(8, maximum=8, engine_waiting=0) == 8


def test_limit_halves_when_the_engine_builds_a_queue():
    assert next_dispatch_limit(8, maximum=8, engine_waiting=3) == 4


def test_limit_never_drops_below_one():
    assert next_dispatch_limit(1, maximum=8, engine_waiting=5) == 1
