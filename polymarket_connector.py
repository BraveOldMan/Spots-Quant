"""Polymarket CLOB connector with dry-run defaults and live fail-closed guards."""

import os
import random
import re
import time
from typing import Any

from config import get_settings
from polymarket_rate_limiter import PolymarketRateLimiter

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        """Fallback when python-dotenv is not installed."""
        return None

try:
    from py_clob_client_v2 import ApiCreds, ClobClient

    HAS_CLOB_CLIENT = True
except ImportError:
    ApiCreds = None
    ClobClient = None
    HAS_CLOB_CLIENT = False


ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


class PolymarketConnector:
    """Read Polymarket prices and simulate orders unless live trading is enabled."""

    def __init__(self) -> None:
        load_dotenv()
        self.is_live = False
        self.client: Any = None
        settings = get_settings(load_env=False)
        self.rate_limiter = PolymarketRateLimiter(
            enabled=settings.polymarket.rate_limit_enabled
        )

        allow_real_money = os.getenv("ALLOW_REAL_MONEY", "0")
        private_key = os.getenv("POLYGON_PRIVATE_KEY")
        funder = os.getenv("FUNDER_ADDRESS")
        host = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
        chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))
        signature_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))
        api_key = os.getenv("POLYMARKET_CLOB_API_KEY") or os.getenv("CLOB_API_KEY")
        api_secret = os.getenv("POLYMARKET_CLOB_API_SECRET") or os.getenv("CLOB_SECRET")
        api_passphrase = os.getenv("POLYMARKET_CLOB_API_PASSPHRASE") or os.getenv(
            "CLOB_PASS_PHRASE"
        )

        if allow_real_money == "1":
            if not HAS_CLOB_CLIENT or ClobClient is None:
                raise RuntimeError(
                    "ALLOW_REAL_MONEY=1 but py-clob-client-v2 is not installed."
                )
            missing = [
                name
                for name, value in (
                    ("POLYGON_PRIVATE_KEY", private_key),
                    ("FUNDER_ADDRESS", funder),
                )
                if not value
            ]
            if missing:
                raise RuntimeError(
                    "ALLOW_REAL_MONEY=1 but required Polymarket credentials are missing: "
                    + ",".join(missing)
                )
            if funder is None or not ADDRESS_RE.match(funder):
                raise RuntimeError(
                    "FUNDER_ADDRESS must be a 0x-prefixed 40-byte address."
                )
            creds = self._api_creds(api_key, api_secret, api_passphrase)
            self.is_live = True
            client_kwargs = {
                "host": host,
                "chain_id": chain_id,
                "key": private_key,
                "funder": funder,
                "signature_type": signature_type,
            }
            if creds is not None:
                client_kwargs["creds"] = creds
            self.client = ClobClient(**client_kwargs)
            print("[Polymarket] live CLOB client initialized.")
            return

        print("[Polymarket] dry-run mode initialized.")

    @staticmethod
    def _api_creds(
        api_key: str | None,
        api_secret: str | None,
        api_passphrase: str | None,
    ) -> Any:
        """Build optional V2 API credentials, failing closed on partial config."""
        values = (api_key, api_secret, api_passphrase)
        if not any(values):
            return None
        if not all(values):
            raise RuntimeError("Incomplete Polymarket CLOB API credentials.")
        if ApiCreds is None:
            raise RuntimeError("py-clob-client-v2 ApiCreds is unavailable.")
        return ApiCreds(
            api_key=str(api_key),
            api_secret=str(api_secret),
            api_passphrase=str(api_passphrase),
        )

    @staticmethod
    def _best_ask(book: Any) -> float | None:
        """Extract best ask from a py-clob-client order book or dict."""
        if book is None:
            return None
        asks = getattr(book, "asks", None)
        if asks is None and isinstance(book, dict):
            asks = book.get("asks")
        if not asks:
            return None

        prices = []
        for level in asks:
            price = getattr(level, "price", None)
            if price is None and isinstance(level, dict):
                price = level.get("price")
            if price is None:
                continue
            try:
                prices.append(float(price))
            except (TypeError, ValueError):
                continue
        return min(prices) if prices else None

    def get_order_book(self, market_token: str) -> Any:
        """Return the live order book for a token, or None on failure."""
        if not (self.is_live and self.client):
            return None
        try:
            self._rate_limit("clob_book")
            return self.client.get_order_book(market_token)
        except Exception as exc:
            print(f"[orderbook] read failed: {exc}")
            return None

    def _rate_limit(self, limit_name: str) -> float:
        """Apply a configured Polymarket limiter before a live network call."""
        limiter = getattr(self, "rate_limiter", None)
        if limiter is None:
            return 0.0
        return limiter.acquire(limit_name)

    def fetch_market_odds(
        self, match_name: str, market_tokens: dict[str, str] | None = None
    ) -> dict[str, float]:
        """Fetch share prices; live mode fails closed when tokens/books are missing."""
        if self.is_live and self.client:
            if not market_tokens:
                print("[orderbook] live mode requires real market tokens; fail-closed.")
                return {}

            prices = {}
            for key, token in market_tokens.items():
                ask = self._best_ask(self.get_order_book(token))
                if ask is None:
                    print("[orderbook] live order book is incomplete; fail-closed.")
                    return {}
                prices[key] = ask
            return prices

        home_base = random.uniform(0.2, 0.8)
        draw_base = random.uniform(0.1, 0.3)
        away_base = max(1.0 - home_base - draw_base, 0.05)
        total = home_base + draw_base + away_base
        return {
            "home_price": round((home_base / total) * 1.02, 3),
            "draw_price": round((draw_base / total) * 1.02, 3),
            "away_price": round((away_base / total) * 1.02, 3),
        }

    def place_order(
        self, market_token: str, side: str, size: float, price: float
    ) -> bool:
        """Place an order in dry-run only; live network submission is disabled."""
        if self.is_live and self.client:
            print(
                "[live blocked] create_and_post_order is disabled; "
                f"not sent: {side.upper()} {size:.2f} shares @ ${price:.3f}."
            )
            return False

        print(
            f"[dry-run] virtual order: {side.upper()} {size:.2f} shares "
            f"@ ${price:.3f} on {market_token}."
        )
        return True

    def close_position(
        self, market_token: str, current_size: float, current_price: float
    ) -> bool:
        """Close a position through the same guarded order path."""
        print(
            f"[close] preparing SELL {current_size:.2f} shares "
            f"@ ${current_price:.3f}."
        )
        time.sleep(0.5)
        return self.place_order(
            market_token, side="SELL", size=current_size, price=current_price
        )
