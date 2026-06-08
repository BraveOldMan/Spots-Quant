"""Tests for the central Spots-Quant configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest

import config


def test_default_settings_preserve_current_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default settings must keep the current strategy and risk baseline."""
    for name in (
        "SPOTS_EV_THRESHOLD",
        "SPOTS_INITIAL_CAPITAL",
        "SPOTS_MAX_DRAWDOWN_LIMIT",
        "SPOTS_BACKTEST_CSV_PATHS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = config.get_settings(load_env=False)

    assert settings.strategy.ev_threshold == 1.05
    assert settings.risk.initial_capital == 10000.0
    assert settings.risk.max_drawdown_limit == 0.15
    assert settings.backtest.csv_paths == (
        "data_seasons/E0_2223.csv",
        "data_seasons/E0_2324.csv",
        "data_seasons/E0_2425.csv",
    )


def test_settings_read_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit environment values should override defaults in typed settings."""
    monkeypatch.setenv("SPOTS_EV_THRESHOLD", "1.12")
    monkeypatch.setenv("SPOTS_INITIAL_CAPITAL", "25000")
    monkeypatch.setenv("SPOTS_MAX_MATCH_EXPOSURE", "none")
    monkeypatch.setenv("SPOTS_BACKTEST_CSV_PATHS", "a.csv;b.csv")
    monkeypatch.setenv("ALLOW_REAL_MONEY", "0")
    monkeypatch.setenv("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com")
    monkeypatch.setenv("POLYMARKET_DATA_URL", "https://data-api.polymarket.com")
    monkeypatch.setenv("POLYMARKET_RATE_LIMIT_ENABLED", "0")

    settings = config.get_settings(load_env=False)

    assert settings.strategy.ev_threshold == 1.12
    assert settings.risk.initial_capital == 25000.0
    assert settings.risk.max_match_exposure is None
    assert settings.backtest.csv_paths == ("a.csv", "b.csv")
    assert settings.polymarket.allow_real_money is False
    assert settings.polymarket.gamma_url == "https://gamma-api.polymarket.com"
    assert settings.polymarket.data_url == "https://data-api.polymarket.com"
    assert settings.polymarket.rate_limit_enabled is False


def test_invalid_numeric_environment_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid numeric config should raise instead of silently using defaults."""
    monkeypatch.setenv("SPOTS_KELLY_MULT", "not-a-number")

    with pytest.raises(ValueError, match="SPOTS_KELLY_MULT"):
        config.get_settings(load_env=False)


def test_invalid_funder_address_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed Polymarket funder addresses should not enter settings."""
    monkeypatch.setenv("FUNDER_ADDRESS", "not-an-address")

    with pytest.raises(ValueError, match="FUNDER_ADDRESS"):
        config.get_settings(load_env=False)


def test_load_env_file_does_not_override_existing_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Env file loading should preserve existing process variables by default."""
    env_path = tmp_path / ".env"
    env_path.write_text("SPOTS_EV_THRESHOLD=1.20\n", encoding="utf-8")
    monkeypatch.setenv("SPOTS_EV_THRESHOLD", "1.10")

    config.load_env_file(env_path)

    assert config.get_settings(load_env=False).strategy.ev_threshold == 1.10


def test_redacted_settings_masks_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings rendered for reports must not expose complete secrets."""
    monkeypatch.setenv("API_FOOTBALL_KEY", "abcdefghijklmnopqrstuvwxyz")
    monkeypatch.setenv("POLYMARKET_CLOB_API_KEY", "clobabcdefghijklmnopqrstuvwxyz")
    monkeypatch.setenv("POLYMARKET_CLOB_API_SECRET", "secretabcdefghijklmnopqrstuvwxyz")
    monkeypatch.setenv("POLYMARKET_CLOB_API_PASSPHRASE", "passabcdefghijklmnopqrstuvwxyz")
    monkeypatch.setenv("FUNDER_ADDRESS", "0x3333333333333333333333333333333333333333")
    monkeypatch.setenv("POLYMARKET_API_ADDRESS", "0x1111111111111111111111111111111111111111")
    monkeypatch.setenv("POLYMARKET_RELAYER_API_KEY", "019e9d67-159d-7760-9a1c-test")
    monkeypatch.setenv(
        "POLYMARKET_RELAYER_API_KEY_ADDRESS",
        "0x2222222222222222222222222222222222222222",
    )

    settings = config.get_settings(load_env=False)
    redacted = config.redacted_settings(settings)

    assert redacted["api"]["api_football_key"] == "abcd...wxyz"
    assert redacted["polymarket"]["clob_api_key"] == "clob...wxyz"
    assert redacted["polymarket"]["clob_api_secret"] == "secr...wxyz"
    assert redacted["polymarket"]["clob_api_passphrase"] == "pass...wxyz"
    assert redacted["polymarket"]["funder_address"] == "0x33...3333"
    assert redacted["polymarket"]["api_address"] == "0x11...1111"
    assert redacted["polymarket"]["gamma_url"] == "https://gamma-api.polymarket.com"
    assert redacted["polymarket"]["data_url"] == "https://data-api.polymarket.com"
    assert redacted["polymarket"]["relayer_api_key"] == "019e...test"
    assert redacted["polymarket"]["relayer_api_key_address"] == "0x22...2222"
