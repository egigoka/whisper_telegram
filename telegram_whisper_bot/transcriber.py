from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from .config import Settings


class TranscriptionError(RuntimeError):
    pass


class TranscriptionTimeout(TranscriptionError):
    pass


@dataclass(frozen=True)
class TranscriptionProgress:
    percent: float | None = None
    audio_total_sec: float | None = None


class WhisperTranscriber:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sem = asyncio.Semaphore(settings.max_concurrent)

    async def transcribe(
        self,
        wav_path: Path,
        progress_callback: Callable[[TranscriptionProgress], Awaitable[None]] | None = None,
    ) -> str:
        if not self._settings.whisper_bin.exists():
            raise TranscriptionError(f"whisper binary was not found: {self._settings.whisper_bin}")
        if not self._settings.whisper_model.exists():
            raise TranscriptionError(f"whisper model was not found: {self._settings.whisper_model}")

        output_prefix = wav_path.with_suffix("")
        output_txt = output_prefix.with_suffix(".txt")

        cmd = [
            str(self._settings.whisper_bin),
            "-m",
            str(self._settings.whisper_model),
            "-f",
            str(wav_path),
            "-l",
            self._settings.whisper_language,
            "--no-timestamps",
            "--print-progress",
            "-otxt",
            "-of",
            str(output_prefix),
        ]

        async with self._sem:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as exc:
                raise TranscriptionError("failed to start whisper binary") from exc

            stderr_data = bytearray()
            percent_re = re.compile(r"progress\s*=\s*([0-9]+(?:\.[0-9]+)?)%")
            total_sec_re = re.compile(r"\(\s*\d+\s+samples,\s*([0-9]+(?:\.[0-9]+)?)\s+sec\)")

            async def consume_stream(stream: asyncio.StreamReader | None) -> None:
                if stream is None:
                    return
                while True:
                    chunk = await stream.read(4096)
                    if not chunk:
                        break
                    stderr_data.extend(chunk)
                    if progress_callback is None:
                        continue
                    text = chunk.decode("utf-8", errors="replace")
                    total_match = total_sec_re.search(text)
                    if total_match:
                        await progress_callback(
                            TranscriptionProgress(audio_total_sec=float(total_match.group(1)))
                        )
                    for match in percent_re.finditer(text):
                        pct = float(match.group(1))
                        if pct < 0:
                            continue
                        await progress_callback(TranscriptionProgress(percent=min(100.0, pct)))

            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        consume_stream(proc.stdout),
                        consume_stream(proc.stderr),
                        proc.wait(),
                    ),
                    timeout=self._settings.transcribe_timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                raise TranscriptionTimeout(
                    f"transcription timed out after {self._settings.transcribe_timeout}s"
                )

        if proc.returncode != 0:
            stderr = bytes(stderr_data)
            error_tail = stderr.decode("utf-8", errors="replace").strip()[-1000:]
            raise TranscriptionError(f"whisper failed with code {proc.returncode}: {error_tail}")

        if not output_txt.exists():
            raise TranscriptionError("whisper did not produce a transcript file")

        text = output_txt.read_text(encoding="utf-8", errors="replace").strip()
        return text
