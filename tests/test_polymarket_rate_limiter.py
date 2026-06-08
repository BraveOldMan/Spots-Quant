"""Tests for Polymarket client-side rate limiting."""

from __future__ import annotations

from polymarket_rate_limiter import (
    OFFICIAL_RATE_LIMITS,
    PolymarketRateLimiter,
    RateLimitRule,
    SlidingWindowRateLimiter,
)


class FakeClock:
    """Controllable monotonic clock used by limiter tests."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_official_polymarket_rate_limits_are_configured() -> None:
    """Local quota names should match the Polymarket endpoint families we call."""
    assert OFFICIAL_RATE_LIMITS["gamma_markets"] == RateLimitRule(
        "gamma_markets", 300, 10.0
    )
    assert OFFICIAL_RATE_LIMITS["gamma_events"] == RateLimitRule(
        "gamma_events", 500, 10.0
    )
    assert OFFICIAL_RATE_LIMITS["data_trades"] == RateLimitRule(
        "data_trades", 200, 10.0
    )
    assert OFFICIAL_RATE_LIMITS["data_positions"] == RateLimitRule(
        "data_positions", 150, 10.0
    )
    assert OFFICIAL_RATE_LIMITS["clob_book"] == RateLimitRule("clob_book", 1_500, 10.0)
    assert OFFICIAL_RATE_LIMITS["clob_auth_api_key"] == RateLimitRule(
        "clob_auth_api_key", 100, 10.0
    )
    assert OFFICIAL_RATE_LIMITS["relayer_submit"] == RateLimitRule(
        "relayer_submit", 25, 60.0
    )


def test_sliding_window_limiter_waits_after_quota_is_full() -> None:
    """The next request should wait until the oldest event leaves the window."""
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(
        RateLimitRule("test", 2, 10.0),
        clock=clock,
        sleeper=clock.sleep,
    )

    assert limiter.acquire() == 0.0
    assert limiter.acquire() == 0.0
    assert limiter.acquire() == 10.0
    assert clock.sleeps == [10.0]


def test_disabled_polymarket_limiter_does_not_sleep() -> None:
    """The env-controlled off switch should bypass local throttling."""
    clock = FakeClock()
    limiter = PolymarketRateLimiter(
        enabled=False,
        clock=clock,
        sleeper=clock.sleep,
    )

    for _ in range(3):
        assert limiter.acquire("relayer_submit") == 0.0

    assert clock.sleeps == []
