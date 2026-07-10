from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from telegram_whisper_bot.bot import _process_media_message
from telegram_whisper_bot.transcriber import TranscriptionError, TranscriptionTimeout


class FakeStatusMessage:
    def __init__(self) -> None:
        self.deleted = False
        self.edits: list[str] = []

    async def delete(self) -> None:
        self.deleted = True

    async def edit_text(self, text: str) -> None:
        self.edits.append(text)


class FakeMessage:
    def __init__(self, *, voice=None, audio=None, video_note=None, user=None) -> None:
        self.voice = voice
        self.audio = audio
        self.video_note = video_note
        self.from_user = user
        self.replies: list[str] = []
        self.status = FakeStatusMessage()

    async def reply(self, text: str):
        self.replies.append(text)
        if text == "Transcribing...":
            return self.status
        return SimpleNamespace()


class FakeBot:
    def __init__(self) -> None:
        self.admin_messages: list[tuple[int, str]] = []
        self.downloaded_to: Path | None = None

    async def send_message(self, chat_id: int, text: str) -> None:
        self.admin_messages.append((chat_id, text))

    async def get_file(self, file_id: str):
        return SimpleNamespace(file_path=f"files/{file_id}")

    async def download_file(self, file_path: str, destination: Path) -> None:
        destination.write_text("binary", encoding="utf-8")
        self.downloaded_to = destination


class FakeTranscriber:
    def __init__(self, text: str = "ok", exc: Exception | None = None) -> None:
        self.text = text
        self.exc = exc

    async def transcribe(self, wav_path: Path) -> str:
        if self.exc:
            raise self.exc
        return self.text


@pytest.mark.asyncio
async def test_process_media_sends_inbound_metadata_to_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = FakeBot()
    msg = FakeMessage(
        voice=SimpleNamespace(file_id="v1", file_unique_id="uq", file_size=123),
        user=SimpleNamespace(full_name="User One", username="u1", id=42),
    )
    tr = FakeTranscriber(text="hello")

    async def fake_ffmpeg(input_path: Path, output_path: Path) -> None:
        output_path.write_text("wav", encoding="utf-8")

    monkeypatch.setattr("telegram_whisper_bot.bot._run_ffmpeg", fake_ffmpeg)

    await _process_media_message(msg, bot, tr, admin_id=777, model_name="medium")

    assert bot.admin_messages[0][0] == 777
    assert bot.admin_messages[0][1].startswith("IN: from User One (@u1, id=42) | type=voice | file=voice_uq.ogg")


@pytest.mark.asyncio
async def test_process_media_rejects_too_large_and_reports_fail_to_admin() -> None:
    bot = FakeBot()
    msg = FakeMessage(
        voice=SimpleNamespace(file_id="v1", file_unique_id="uq", file_size=21 * 1024 * 1024),
        user=SimpleNamespace(full_name="User", username="u", id=1),
    )

    await _process_media_message(msg, bot, FakeTranscriber(), admin_id=777, model_name="medium")

    assert "File is too large" in msg.replies[-1]
    texts = [t for _, t in bot.admin_messages]
    assert any("IN:" in t for t in texts)
    assert any("FAIL:" in t and "file too large" in t for t in texts)
    assert "Transcribing..." not in texts


@pytest.mark.asyncio
async def test_process_media_success_replies_with_transcript_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = FakeBot()
    long_text = "a" * 5000
    msg = FakeMessage(
        voice=SimpleNamespace(file_id="v1", file_unique_id="uq", file_size=1000),
        user=SimpleNamespace(full_name="User", username="u", id=1),
    )

    async def fake_ffmpeg(input_path: Path, output_path: Path) -> None:
        output_path.write_text("wav", encoding="utf-8")

    monkeypatch.setattr("telegram_whisper_bot.bot._run_ffmpeg", fake_ffmpeg)

    await _process_media_message(msg, bot, FakeTranscriber(text=long_text), admin_id=777, model_name="medium")

    assert msg.status.deleted
    user_chunks = [t for t in msg.replies if t != "Transcribing..."]
    assert len(user_chunks) == 2
    texts = [t for _, t in bot.admin_messages]
    assert any(t.startswith("SUCCESS:") for t in texts)
    admin_chunks = [t for t in texts if set(t) == {"a"}]
    assert len(admin_chunks) == 0
    assert "Transcribing..." not in texts


@pytest.mark.asyncio
async def test_process_media_success_no_speech_message_mirrored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = FakeBot()
    msg = FakeMessage(
        voice=SimpleNamespace(file_id="v1", file_unique_id="uq", file_size=1000),
        user=SimpleNamespace(full_name="User", username="u", id=1),
    )

    async def fake_ffmpeg(input_path: Path, output_path: Path) -> None:
        output_path.write_text("wav", encoding="utf-8")

    monkeypatch.setattr("telegram_whisper_bot.bot._run_ffmpeg", fake_ffmpeg)

    await _process_media_message(msg, bot, FakeTranscriber(text=""), admin_id=777, model_name="medium")

    assert "No speech detected." in msg.replies
    texts = [t for _, t in bot.admin_messages]
    assert any("SUCCESS:" in t and "no speech detected" in t for t in texts)
    assert not any(t == "No speech detected." for t in texts)


@pytest.mark.asyncio
async def test_process_media_timeout_edits_status_and_mirrors_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = FakeBot()
    msg = FakeMessage(
        voice=SimpleNamespace(file_id="v1", file_unique_id="uq", file_size=1000),
        user=SimpleNamespace(full_name="User", username="u", id=1),
    )

    async def fake_ffmpeg(input_path: Path, output_path: Path) -> None:
        output_path.write_text("wav", encoding="utf-8")

    monkeypatch.setattr("telegram_whisper_bot.bot._run_ffmpeg", fake_ffmpeg)

    await _process_media_message(
        msg,
        bot,
        FakeTranscriber(exc=TranscriptionTimeout("timeout")),
        admin_id=777,
        model_name="medium",
    )

    assert msg.status.edits[-1] == "Transcription timed out. Try a shorter file."
    texts = [t for _, t in bot.admin_messages]
    assert any("FAIL:" in t and "timeout" in t for t in texts)


@pytest.mark.asyncio
async def test_process_media_transcription_error_edits_status_and_mirrors_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = FakeBot()
    msg = FakeMessage(
        voice=SimpleNamespace(file_id="v1", file_unique_id="uq", file_size=1000),
        user=SimpleNamespace(full_name="User", username="u", id=1),
    )

    async def fake_ffmpeg(input_path: Path, output_path: Path) -> None:
        output_path.write_text("wav", encoding="utf-8")

    monkeypatch.setattr("telegram_whisper_bot.bot._run_ffmpeg", fake_ffmpeg)

    await _process_media_message(
        msg,
        bot,
        FakeTranscriber(exc=TranscriptionError("boom")),
        admin_id=777,
        model_name="medium",
    )

    assert msg.status.edits
    assert msg.status.edits[-1].startswith("Transcription failed: boom")
    texts = [t for _, t in bot.admin_messages]
    assert any("FAIL:" in t and "transcription error" in t for t in texts)


@pytest.mark.asyncio
async def test_process_media_without_admin_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = FakeBot()
    msg = FakeMessage(
        voice=SimpleNamespace(file_id="v1", file_unique_id="uq", file_size=1000),
        user=SimpleNamespace(full_name="User", username="u", id=1),
    )

    async def fake_ffmpeg(input_path: Path, output_path: Path) -> None:
        output_path.write_text("wav", encoding="utf-8")

    monkeypatch.setattr("telegram_whisper_bot.bot._run_ffmpeg", fake_ffmpeg)

    await _process_media_message(msg, bot, FakeTranscriber(text="hello"), admin_id=None, model_name="medium")

    assert "hello" in msg.replies
    assert bot.admin_messages == []


@pytest.mark.asyncio
async def test_process_media_from_admin_does_not_mirror_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = FakeBot()
    msg = FakeMessage(
        voice=SimpleNamespace(file_id="v1", file_unique_id="uq", file_size=1000),
        user=SimpleNamespace(full_name="Admin", username="admin", id=777),
    )

    async def fake_ffmpeg(input_path: Path, output_path: Path) -> None:
        output_path.write_text("wav", encoding="utf-8")

    monkeypatch.setattr("telegram_whisper_bot.bot._run_ffmpeg", fake_ffmpeg)

    await _process_media_message(msg, bot, FakeTranscriber(text="hello"), admin_id=777, model_name="medium")

    assert "hello" in msg.replies
    assert bot.admin_messages == []
