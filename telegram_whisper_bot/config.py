from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    bot_token: str
    whisper_bin: Path
    whisper_model: Path
    whisper_language: str
    max_concurrent: int
    transcribe_timeout: int
    admin_id: int


def _token_from_secrets() -> str | None:
    try:
        import secrets as local_secrets
    except Exception:
        return None

    for key in ("TELEGRAM_BOT_TOKEN", "WHISPER_TELEGRAM_TOKEN", "BOT_TOKEN"):
        value = getattr(local_secrets, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _parse_admin_id(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        return int(value.strip())
    return None


def _admin_id_from_secrets() -> int | None:
    try:
        import secrets as local_secrets
    except Exception:
        return None

    for key in ("ADMIN_ID", "MY_CHAT_ID"):
        try:
            parsed = _parse_admin_id(getattr(local_secrets, key, None))
        except ValueError:
            continue
        if parsed is not None:
            return parsed
    return None


def load_settings() -> Settings:
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or _token_from_secrets()
    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN is required (env var or secrets.py TELEGRAM_BOT_TOKEN)"
        )

    max_concurrent = int(os.environ.get("MAX_CONCURRENT", "1"))
    if max_concurrent < 1:
        raise ValueError("MAX_CONCURRENT must be >= 1")

    transcribe_timeout = int(os.environ.get("TRANSCRIBE_TIMEOUT", "300"))
    if transcribe_timeout < 1:
        raise ValueError("TRANSCRIBE_TIMEOUT must be >= 1")

    whisper_bin = Path(os.environ.get("WHISPER_BIN", "./whisper.cpp/build/bin/whisper-cli"))
    whisper_model = Path(os.environ.get("WHISPER_MODEL", "./whisper.cpp/models/ggml-medium.bin"))
    whisper_language = os.environ.get("WHISPER_LANGUAGE", "auto")

    env_admin_id = os.environ.get("ADMIN_ID")
    admin_id = _parse_admin_id(env_admin_id) if env_admin_id is not None else _admin_id_from_secrets()
    if admin_id is None:
        raise ValueError("ADMIN_ID is required (env var or secrets.py ADMIN_ID/MY_CHAT_ID)")

    return Settings(
        bot_token=token,
        whisper_bin=whisper_bin,
        whisper_model=whisper_model,
        whisper_language=whisper_language,
        max_concurrent=max_concurrent,
        transcribe_timeout=transcribe_timeout,
        admin_id=admin_id,
    )
