from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from telegram_whisper_bot.config import Settings
from telegram_whisper_bot.transcriber import (
    TranscriptionError,
    TranscriptionTimeout,
    WhisperTranscriber,
)


class FakeStream:
    def __init__(self, chunks: list[bytes] | None = None) -> None:
        self._chunks = list(chunks or [])

    async def read(self, _: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class FakeProc:
    def __init__(
        self,
        returncode: int = 0,
        stderr: bytes = b"",
        stdout_chunks: list[bytes] | None = None,
        stderr_chunks: list[bytes] | None = None,
    ) -> None:
        self.returncode = returncode
        self._stderr = stderr
        self.killed = False
        self.communicate_calls = 0
        self.stdout = FakeStream(stdout_chunks)
        self.stderr = FakeStream(stderr_chunks if stderr_chunks is not None else [stderr])

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicate_calls += 1
        return b"", self._stderr

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True


def _settings(bin_path: Path, model_path: Path, max_concurrent: int = 2, timeout: int = 5) -> Settings:
    return Settings(
        bot_token="tok",
        whisper_bin=bin_path,
        whisper_model=model_path,
        whisper_language="auto",
        max_concurrent=max_concurrent,
        transcribe_timeout=timeout,
        admin_id=1,
    )


@pytest.mark.asyncio
async def test_transcribe_fails_when_binary_missing(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_text("x", encoding="utf-8")
    tr = WhisperTranscriber(_settings(tmp_path / "missing-bin", model))

    with pytest.raises(TranscriptionError, match="whisper binary was not found"):
        await tr.transcribe(tmp_path / "audio.wav")


@pytest.mark.asyncio
async def test_transcribe_fails_when_model_missing(tmp_path: Path) -> None:
    binary = tmp_path / "whisper-cli"
    binary.write_text("x", encoding="utf-8")
    tr = WhisperTranscriber(_settings(binary, tmp_path / "missing-model"))

    with pytest.raises(TranscriptionError, match="whisper model was not found"):
        await tr.transcribe(tmp_path / "audio.wav")


@pytest.mark.asyncio
async def test_transcribe_invokes_whisper_with_expected_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "whisper-cli"
    model = tmp_path / "model.bin"
    wav = tmp_path / "input.wav"
    txt = tmp_path / "input.txt"
    binary.write_text("x", encoding="utf-8")
    model.write_text("x", encoding="utf-8")
    wav.write_text("x", encoding="utf-8")
    txt.write_text("hello", encoding="utf-8")
    tr = WhisperTranscriber(_settings(binary, model))

    calls: list[tuple[str, ...]] = []

    async def fake_create(*cmd: str, **_: object) -> FakeProc:
        calls.append(tuple(cmd))
        return FakeProc(returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    out = await tr.transcribe(wav)

    assert out == "hello"
    cmd = calls[0]
    assert cmd[0] == str(binary)
    assert "-m" in cmd and str(model) in cmd
    assert "-f" in cmd and str(wav) in cmd
    assert "--no-timestamps" in cmd
    assert "--print-progress" in cmd
    assert "-otxt" in cmd
    assert "-of" in cmd


@pytest.mark.asyncio
async def test_transcribe_wraps_filenotfound_as_transcription_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "whisper-cli"
    model = tmp_path / "model.bin"
    wav = tmp_path / "audio.wav"
    binary.write_text("x", encoding="utf-8")
    model.write_text("x", encoding="utf-8")
    wav.write_text("x", encoding="utf-8")

    tr = WhisperTranscriber(_settings(binary, model))

    async def fake_create(*_: object, **__: object) -> FakeProc:
        raise FileNotFoundError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    with pytest.raises(TranscriptionError, match="failed to start whisper binary"):
        await tr.transcribe(wav)


@pytest.mark.asyncio
async def test_transcribe_timeout_kills_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "whisper-cli"
    model = tmp_path / "model.bin"
    wav = tmp_path / "audio.wav"
    binary.write_text("x", encoding="utf-8")
    model.write_text("x", encoding="utf-8")
    wav.write_text("x", encoding="utf-8")

    tr = WhisperTranscriber(_settings(binary, model, timeout=1))
    proc = FakeProc(returncode=0)

    async def fake_create(*_: object, **__: object) -> FakeProc:
        return proc

    async def fake_wait_for(awaitable, **__: object):
        awaitable.cancel()
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    with pytest.raises(TranscriptionTimeout, match="timed out"):
        await tr.transcribe(wav)
    assert proc.killed
    assert proc.communicate_calls >= 1


@pytest.mark.asyncio
async def test_transcribe_nonzero_exit_raises_with_stderr_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "whisper-cli"
    model = tmp_path / "model.bin"
    wav = tmp_path / "audio.wav"
    binary.write_text("x", encoding="utf-8")
    model.write_text("x", encoding="utf-8")
    wav.write_text("x", encoding="utf-8")

    tr = WhisperTranscriber(_settings(binary, model))
    proc = FakeProc(returncode=42, stderr=b"boom")

    async def fake_create(*_: object, **__: object) -> FakeProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    with pytest.raises(TranscriptionError, match="code 42: boom"):
        await tr.transcribe(wav)


@pytest.mark.asyncio
async def test_transcribe_raises_when_output_txt_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "whisper-cli"
    model = tmp_path / "model.bin"
    wav = tmp_path / "audio.wav"
    binary.write_text("x", encoding="utf-8")
    model.write_text("x", encoding="utf-8")
    wav.write_text("x", encoding="utf-8")

    tr = WhisperTranscriber(_settings(binary, model))

    async def fake_create(*_: object, **__: object) -> FakeProc:
        return FakeProc(returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    with pytest.raises(TranscriptionError, match="did not produce a transcript file"):
        await tr.transcribe(wav)


@pytest.mark.asyncio
async def test_transcribe_returns_stripped_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "whisper-cli"
    model = tmp_path / "model.bin"
    wav = tmp_path / "audio.wav"
    txt = tmp_path / "audio.txt"
    binary.write_text("x", encoding="utf-8")
    model.write_text("x", encoding="utf-8")
    wav.write_text("x", encoding="utf-8")
    txt.write_text("\n  hello world  \n", encoding="utf-8")

    tr = WhisperTranscriber(_settings(binary, model))

    async def fake_create(*_: object, **__: object) -> FakeProc:
        return FakeProc(returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    out = await tr.transcribe(wav)

    assert out == "hello world"


@pytest.mark.asyncio
async def test_transcribe_respects_max_concurrent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "whisper-cli"
    model = tmp_path / "model.bin"
    wav1 = tmp_path / "a1.wav"
    wav2 = tmp_path / "a2.wav"
    txt1 = tmp_path / "a1.txt"
    txt2 = tmp_path / "a2.txt"
    binary.write_text("x", encoding="utf-8")
    model.write_text("x", encoding="utf-8")
    wav1.write_text("x", encoding="utf-8")
    wav2.write_text("x", encoding="utf-8")
    txt1.write_text("one", encoding="utf-8")
    txt2.write_text("two", encoding="utf-8")

    tr = WhisperTranscriber(_settings(binary, model, max_concurrent=1))

    active = 0
    max_seen = 0

    class Proc(FakeProc):
        async def wait(self) -> int:
            nonlocal active, max_seen
            active += 1
            max_seen = max(max_seen, active)
            await asyncio.sleep(0.05)
            active -= 1
            return self.returncode

    async def fake_create(*_: object, **__: object) -> Proc:
        return Proc(returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    out1, out2 = await asyncio.gather(tr.transcribe(wav1), tr.transcribe(wav2))

    assert sorted([out1, out2]) == ["one", "two"]
    assert max_seen == 1
