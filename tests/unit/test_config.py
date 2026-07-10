from __future__ import annotations

import sys
import types

import pytest

from telegram_whisper_bot import config


def _install_fake_secrets(monkeypatch: pytest.MonkeyPatch, **attrs: object) -> None:
    mod = types.ModuleType("secrets")
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, "secrets", mod)


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "TELEGRAM_BOT_TOKEN",
        "WHISPER_BIN",
        "WHISPER_MODEL",
        "WHISPER_LANGUAGE",
        "MAX_CONCURRENT",
        "TRANSCRIBE_TIMEOUT",
        "ADMIN_ID",
    ):
        monkeypatch.delenv(key, raising=False)


def test_load_settings_uses_env_token_over_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env_token")
    monkeypatch.setenv("ADMIN_ID", "1")
    _install_fake_secrets(monkeypatch, TELEGRAM_BOT_TOKEN="secret_token")

    settings = config.load_settings()

    assert settings.bot_token == "env_token"


def test_load_settings_uses_secrets_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("ADMIN_ID", "1")
    _install_fake_secrets(monkeypatch, TELEGRAM_BOT_TOKEN="secret_token")

    settings = config.load_settings()

    assert settings.bot_token == "secret_token"


def test_load_settings_accepts_whisper_telegram_token_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("ADMIN_ID", "1")
    _install_fake_secrets(monkeypatch, WHISPER_TELEGRAM_TOKEN="fallback_token")

    settings = config.load_settings()

    assert settings.bot_token == "fallback_token"


def test_load_settings_accepts_bot_token_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("ADMIN_ID", "1")
    _install_fake_secrets(monkeypatch, BOT_TOKEN="fallback_bot_token")

    settings = config.load_settings()

    assert settings.bot_token == "fallback_bot_token"


def test_load_settings_raises_if_no_token_anywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("ADMIN_ID", "1")
    _install_fake_secrets(monkeypatch)

    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN is required"):
        config.load_settings()


def test_admin_id_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("ADMIN_ID", "321")
    _install_fake_secrets(monkeypatch, ADMIN_ID=999)

    settings = config.load_settings()

    assert settings.admin_id == 321


def test_admin_id_from_secrets_admin_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    _install_fake_secrets(monkeypatch, ADMIN_ID=654)

    settings = config.load_settings()

    assert settings.admin_id == 654


def test_admin_id_from_secrets_my_chat_id_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    _install_fake_secrets(monkeypatch, MY_CHAT_ID=777)

    settings = config.load_settings()

    assert settings.admin_id == 777


def test_admin_id_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    _install_fake_secrets(monkeypatch)

    with pytest.raises(ValueError, match="ADMIN_ID is required"):
        config.load_settings()


def test_admin_id_invalid_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("ADMIN_ID", "abc")
    _install_fake_secrets(monkeypatch)

    with pytest.raises(ValueError):
        config.load_settings()


def test_max_concurrent_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("ADMIN_ID", "1")
    monkeypatch.setenv("MAX_CONCURRENT", "0")
    _install_fake_secrets(monkeypatch)

    with pytest.raises(ValueError, match="MAX_CONCURRENT"):
        config.load_settings()


def test_transcribe_timeout_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("ADMIN_ID", "1")
    monkeypatch.setenv("TRANSCRIBE_TIMEOUT", "0")
    _install_fake_secrets(monkeypatch)

    with pytest.raises(ValueError, match="TRANSCRIBE_TIMEOUT"):
        config.load_settings()


def test_default_paths_and_language(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("ADMIN_ID", "1")
    _install_fake_secrets(monkeypatch)

    settings = config.load_settings()

    assert str(settings.whisper_bin).endswith("whisper.cpp/build/bin/whisper-cli")
    assert str(settings.whisper_model).endswith("whisper.cpp/models/ggml-medium.bin")
    assert settings.whisper_language == "auto"
