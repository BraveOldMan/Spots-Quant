"""Local Polymarket API rate limits.

The limits mirror Polymarket's published sliding-window quotas. They are used
as client-side guardrails before network calls and do not retry failed requests.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class RateLimitRule:
    """A sliding-window quota for one Polymarket endpoint family."""

    name: str
    max_requests: int
    window_seconds: float

    def __post_init__(self) -> None:
        """Reject invalid quota definitions at import or test time."""
        if self.max_requests <= 0:
            raise ValueError("max_requests must be positive.")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")


Clock = Callable[[], float]
Sleeper = Callable[[float], None]


OFFICIAL_RATE_LIMITS: dict[str, RateLimitRule] = {
    "general": RateLimitRule("general", 15_000, 10.0),
    "gamma_general": RateLimitRule("gamma_general", 4_000, 10.0),
    "gamma_events": RateLimitRule("gamma_events", 500, 10.0),
    "gamma_markets": RateLimitRule("gamma_markets", 300, 10.0),
    "gamma_listing": RateLimitRule("gamma_listing", 900, 10.0),
    "gamma_comments": RateLimitRule("gamma_comments", 200, 10.0),
    "gamma_tags": RateLimitRule("gamma_tags", 200, 10.0),
    "gamma_search": RateLimitRule("gamma_search", 350, 10.0),
    "data_general": RateLimitRule("data_general", 1_000, 10.0),
    "data_trades": RateLimitRule("data_trades", 200, 10.0),
    "data_positions": RateLimitRule("data_positions", 150, 10.0),
    "data_closed_positions": RateLimitRule("data_closed_positions", 150, 10.0),
    "clob_general": RateLimitRule("clob_general", 9_000, 10.0),
    "clob_book": RateLimitRule("clob_book", 1_500, 10.0),
    "clob_books": RateLimitRule("clob_books", 500, 10.0),
    "clob_price": RateLimitRule("clob_price", 1_500, 10.0),
    "clob_prices": RateLimitRule("clob_prices", 500, 10.0),
    "clob_midpoint": RateLimitRule("clob_midpoint", 1_500, 10.0),
    "clob_midpoints": RateLimitRule("clob_midpoints", 500, 10.0),
    "clob_prices_history": RateLimitRule("clob_prices_history", 1_000, 10.0),
    "clob_tick_size": RateLimitRule("clob_tick_size", 200, 10.0),
    "clob_ledger": RateLimitRule("clob_ledger", 900, 10.0),
    "clob_auth_api_key": RateLimitRule("clob_auth_api_key", 100, 10.0),
    "clob_post_order_burst": RateLimitRule("clob_post_order_burst", 5_000, 10.0),
    "clob_post_order_sustained": RateLimitRule(
        "clob_post_order_sustained", 120_000, 600.0
    ),
    "bridge_general": RateLimitRule("bridge_general", 50, 10.0),
    "relayer_submit": RateLimitRule("relayer_submit", 25, 60.0),
    "relayer_api_key": RateLimitRule("relayer_api_key", 100, 10.0),
}


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window limiter for a single quota."""

    def __init__(
        self,
        rule: RateLimitRule,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        self.rule = rule
        self._clock = clock
        self._sleeper = sleeper
        self._events: deque[float] = deque()
        self._lock = Lock()

    def acquire(self) -> float:
        """Block until one request slot is available and return seconds slept."""
        wait_seconds = self._reserve_or_wait()
        if wait_seconds <= 0:
            return 0.0

        self._sleeper(wait_seconds)
        with self._lock:
            now = self._clock()
            self._prune(now)
            self._events.append(now)
        return wait_seconds

    def _reserve_or_wait(self) -> float:
        with self._lock:
            now = self._clock()
            self._prune(now)
            if len(self._events) < self.rule.max_requests:
                self._events.append(now)
                return 0.0
            return max(
                self._events[0] + self.rule.window_seconds - now,
                0.0,
            )

    def _prune(self, now: float) -> None:
        cutoff = now - self.rule.window_seconds
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()


class PolymarketRateLimiter:
    """Named Polymarket sliding-window limiters with an environment-controlled gate."""

    def __init__(
        self,
        enabled: bool = True,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        self.enabled = enabled
        self._limiters = {
            name: SlidingWindowRateLimiter(rule, clock=clock, sleeper=sleeper)
            for name, rule in OFFICIAL_RATE_LIMITS.items()
        }

    def acquire(self, limit_name: str) -> float:
        """Acquire a slot for a named Polymarket endpoint family."""
        if not self.enabled:
            return 0.0
        try:
            limiter = self._limiters[limit_name]
        except KeyError as exc:
            raise KeyError(f"Unknown Polymarket rate limit: {limit_name}") from exc
        return limiter.acquire()
