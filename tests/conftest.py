from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_aiogram_stubs() -> None:
    aiogram = types.ModuleType("aiogram")
    exc = types.ModuleType("aiogram.exceptions")
    filters = types.ModuleType("aiogram.filters")
    types_mod = types.ModuleType("aiogram.types")

    class TelegramBadRequest(Exception):
        pass

    class Bot:  # pragma: no cover
        pass

    class _DummyObserver:
        def __init__(self) -> None:
            self.handlers: list[object] = []

    class Dispatcher:
        def __init__(self) -> None:
            self.observers = {"message": _DummyObserver()}

        def message(self, *args: object, **kwargs: object):
            def decorator(func):
                self.observers["message"].handlers.append(func)
                return func

            return decorator

    class _Field:
        def __init__(self, name: str) -> None:
            self.name = name

    class F:
        voice = _Field("voice")
        audio = _Field("audio")
        video_note = _Field("video_note")

    class Command:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    class Message:  # pragma: no cover
        pass

    aiogram.Bot = Bot
    aiogram.Dispatcher = Dispatcher
    aiogram.F = F
    exc.TelegramBadRequest = TelegramBadRequest
    filters.Command = Command
    types_mod.Message = Message

    sys.modules["aiogram"] = aiogram
    sys.modules["aiogram.exceptions"] = exc
    sys.modules["aiogram.filters"] = filters
    sys.modules["aiogram.types"] = types_mod


try:
    import aiogram  # type: ignore # noqa: F401
except ModuleNotFoundError:
    _install_aiogram_stubs()


from telegram_whisper_bot.config import Settings


@pytest.fixture
def base_settings(tmp_path: Path) -> Settings:
    bin_path = tmp_path / "whisper-cli"
    model_path = tmp_path / "ggml-medium.bin"
    bin_path.write_text("bin", encoding="utf-8")
    model_path.write_text("model", encoding="utf-8")
    return Settings(
        bot_token="token",
        whisper_bin=bin_path,
        whisper_model=model_path,
        whisper_language="auto",
        max_concurrent=2,
        transcribe_timeout=5,
        admin_id=123,
    )
