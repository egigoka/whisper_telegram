from __future__ import annotations

import pytest

from telegram_whisper_bot.bot import build_dispatcher
from telegram_whisper_bot.config import Settings


@pytest.mark.asyncio
async def test_build_dispatcher_registers_handlers(tmp_path):
    settings = Settings(
        bot_token="tok",
        whisper_bin=tmp_path / "whisper-cli",
        whisper_model=tmp_path / "ggml-medium.bin",
        whisper_language="auto",
        max_concurrent=1,
        transcribe_timeout=5,
        admin_id=123,
    )
    settings.whisper_bin.write_text("x", encoding="utf-8")
    settings.whisper_model.write_text("x", encoding="utf-8")

    dp = build_dispatcher(settings)

    # aiogram stores handlers in observer registry; this asserts we at least have the expected routes.
    observers = dp.observers
    assert "message" in observers
    assert len(observers["message"].handlers) >= 5
