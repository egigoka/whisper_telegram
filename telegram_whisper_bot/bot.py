from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import tempfile
import time
from contextlib import suppress
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import Message

from .config import Settings, load_settings
from .transcriber import (
    TranscriptionError,
    TranscriptionProgress,
    TranscriptionTimeout,
    WhisperTranscriber,
)

MAX_MESSAGE_LEN = 4096


class AuthorizationStore:
    def __init__(self, path: Path, admin_id: int | None) -> None:
        self._path = path
        self._admin_id = admin_id
        self._lock = asyncio.Lock()
        self._authorized: set[int] = set()
        self._pending: set[int] = set()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._authorized = {int(x) for x in data.get("authorized_user_ids", [])}
                self._pending = {int(x) for x in data.get("pending_user_ids", [])}
            except Exception:
                logging.exception("Failed to load auth store: %s", self._path)
                self._authorized = set()
                self._pending = set()

        if self._admin_id is not None:
            self._authorized.add(self._admin_id)
            self._pending.discard(self._admin_id)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "authorized_user_ids": sorted(self._authorized),
            "pending_user_ids": sorted(self._pending),
        }
        self._path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    async def is_authorized(self, user_id: int) -> bool:
        async with self._lock:
            return user_id in self._authorized

    async def add_pending_if_needed(self, user_id: int) -> bool:
        async with self._lock:
            if user_id in self._authorized or user_id in self._pending:
                return False
            self._pending.add(user_id)
            self._save()
            return True

    async def authorize(self, user_id: int) -> bool:
        async with self._lock:
            new = user_id not in self._authorized
            self._authorized.add(user_id)
            self._pending.discard(user_id)
            self._save()
            return new


def _chunk_text(text: str, size: int = MAX_MESSAGE_LEN) -> list[str]:
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]


def _sender_label(message: Message) -> str:
    user = message.from_user
    if user is None:
        return "unknown"
    full_name = user.full_name.strip() if user.full_name else "unknown"
    username = f"@{user.username}" if user.username else "no_username"
    return f"{full_name} ({username}, id={user.id})"


def _media_kind_and_filename(message: Message) -> tuple[str, str]:
    if message.audio is not None:
        file_name = message.audio.file_name or f"audio_{message.audio.file_unique_id}.bin"
        return "audio", file_name
    if message.voice is not None:
        return "voice", f"voice_{message.voice.file_unique_id}.ogg"
    if message.video_note is not None:
        return "video_note", f"video_note_{message.video_note.file_unique_id}.mp4"
    return "unknown", "unknown"


async def _send_admin_message(bot: Bot, admin_id: int | None, text: str) -> None:
    if admin_id is None:
        return
    try:
        await bot.send_message(admin_id, text)
    except Exception:
        logging.exception("Failed to send admin mirror message")


def _fmt_mmss(seconds: float) -> str:
    sec = max(0, int(seconds))
    mm, ss = divmod(sec, 60)
    return f"{mm:02d}:{ss:02d}"


def _model_name_from_path(model_path: Path) -> str:
    name = model_path.name.lower()
    match = re.match(r"ggml-(.+?)\\.bin$", name)
    if match:
        return match.group(1)
    return model_path.stem


async def _transcribe_with_optional_progress(
    transcriber: WhisperTranscriber,
    wav_path: Path,
    progress_callback,
) -> str:
    try:
        return await transcriber.transcribe(wav_path, progress_callback=progress_callback)
    except TypeError:
        # Backward compatibility for tests/mocks that implement transcribe(wav_path) only.
        return await transcriber.transcribe(wav_path)


async def _stop_progress_task(stop_event: asyncio.Event, task: asyncio.Task[None]) -> None:
    stop_event.set()
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def _ensure_user_is_authorized(
    message: Message,
    bot: Bot,
    auth_store: AuthorizationStore,
    admin_id: int | None,
) -> bool:
    user = message.from_user
    if user is None:
        logging.warning("Auth check failed: message has no from_user")
        await message.reply("Unable to identify user.")
        return False

    if admin_id is not None and user.id == admin_id:
        logging.info("Auth bypass for admin user_id=%s", user.id)
        return True

    if await auth_store.is_authorized(user.id):
        logging.info("Auth accepted for user_id=%s", user.id)
        return True

    logging.info("Auth pending for user_id=%s", user.id)
    await message.reply("Access pending admin approval.")
    if await auth_store.add_pending_if_needed(user.id):
        logging.info("Auth request sent to admin for user_id=%s", user.id)
        await _send_admin_message(
            bot,
            admin_id,
            (
                f"AUTH REQUEST: {_sender_label(message)}\n"
                f"Approve with: /approve {user.id}"
            ),
        )
    return False


async def _run_ffmpeg(input_path: Path, output_path: Path) -> None:
    logging.info("Running ffmpeg conversion input=%s output=%s", input_path, output_path)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        error = stderr.decode("utf-8", errors="replace").strip()[-1000:]
        logging.error("ffmpeg conversion failed: %s", error)
        raise RuntimeError(f"ffmpeg conversion failed: {error}")
    logging.info("ffmpeg conversion completed output=%s", output_path)


def _check_file_size(msg: Message) -> bool:
    if msg.voice and msg.voice.file_size:
        return msg.voice.file_size <= 20 * 1024 * 1024
    if msg.audio and msg.audio.file_size:
        return msg.audio.file_size <= 20 * 1024 * 1024
    if msg.video_note and msg.video_note.file_size:
        return msg.video_note.file_size <= 20 * 1024 * 1024
    return True


async def _process_media_message(
    message: Message,
    bot: Bot,
    transcriber: WhisperTranscriber,
    admin_id: int | None,
    model_name: str,
) -> None:
    sender_id = message.from_user.id if message.from_user is not None else None
    mirror_admin_id = admin_id
    if admin_id is not None and sender_id == admin_id:
        mirror_admin_id = None

    sender = _sender_label(message)
    media_type, filename = _media_kind_and_filename(message)
    logging.info(
        "Received media message sender=%s type=%s file=%s admin_mirror=%s",
        sender,
        media_type,
        filename,
        mirror_admin_id is not None,
    )

    await _send_admin_message(
        bot,
        mirror_admin_id,
        f"IN: from {sender} | type={media_type} | file={filename}",
    )

    if not _check_file_size(message):
        logging.warning("Rejected oversized file sender=%s file=%s", sender, filename)
        user_text = "File is too large. Telegram bot download limit is 20 MB."
        await message.reply(user_text)
        await _send_admin_message(
            bot,
            mirror_admin_id,
            f"FAIL: from {sender} | file={filename} | reason=file too large",
        )
        return

    media = message.voice or message.audio or message.video_note
    if media is None:
        logging.warning("Unsupported media type sender=%s file=%s", sender, filename)
        user_text = "Unsupported message type."
        await message.reply(user_text)
        await _send_admin_message(
            bot,
            mirror_admin_id,
            f"FAIL: from {sender} | file={filename} | reason=unsupported message type",
        )
        return

    status_message = await message.reply("Transcribing...")
    logging.info("Transcription started sender=%s file=%s model=%s", sender, filename, model_name)
    progress_state: dict[str, float | None] = {"percent": None, "audio_total_sec": None}
    start_ts = time.monotonic()
    stop_progress = asyncio.Event()

    async def on_progress(progress: TranscriptionProgress) -> None:
        if progress.percent is not None:
            progress_state["percent"] = progress.percent
        if progress.audio_total_sec is not None:
            progress_state["audio_total_sec"] = progress.audio_total_sec

    async def progress_updater() -> None:
        last_text: str | None = None
        while not stop_progress.is_set():
            await asyncio.sleep(10)
            if stop_progress.is_set():
                break

            elapsed = time.monotonic() - start_ts
            pct = progress_state["percent"]
            total_sec = progress_state["audio_total_sec"]
            prefix = f"Transcribing ({model_name})..."
            if pct is not None:
                eta = _fmt_mmss(0) if pct >= 100 else _fmt_mmss((elapsed * (100.0 - pct) / pct) if pct > 0 else 0)
                if total_sec is not None and total_sec > 0:
                    done_sec = min(total_sec, total_sec * (pct / 100.0))
                    text = f"{prefix} {pct:.1f}% ({_fmt_mmss(done_sec)}/{_fmt_mmss(total_sec)}) ETA {_fmt_mmss((total_sec - done_sec) if done_sec < total_sec else 0)}"
                else:
                    text = f"{prefix} {pct:.1f}% ETA {eta}"
            else:
                if total_sec is not None and total_sec > 0:
                    text = f"{prefix} {_fmt_mmss(elapsed)} elapsed / {_fmt_mmss(total_sec)} audio ETA n/a"
                else:
                    text = f"{prefix} {_fmt_mmss(elapsed)} elapsed ETA n/a"

            if text != last_text:
                try:
                    logging.info("Progress sender=%s file=%s status=%s", sender, filename, text)
                    await status_message.edit_text(text)
                except TelegramBadRequest:
                    return
                except Exception:
                    logging.exception("Failed to update progress message")
                    return
                last_text = text

    progress_task = asyncio.create_task(progress_updater())

    with tempfile.TemporaryDirectory(prefix="tg-whisper-") as tmp_dir:
        tmp = Path(tmp_dir)
        input_path = tmp / "input"
        wav_path = tmp / "input.wav"

        try:
            file = await bot.get_file(media.file_id)
            logging.info("Telegram file metadata fetched sender=%s path=%s", sender, file.file_path)
            await bot.download_file(file.file_path, destination=input_path)
            logging.info("Telegram file downloaded sender=%s to=%s", sender, input_path)

            await _run_ffmpeg(input_path, wav_path)
            text = await _transcribe_with_optional_progress(transcriber, wav_path, on_progress)

            await _stop_progress_task(stop_progress, progress_task)
            await status_message.delete()
            if not text:
                user_text = "No speech detected."
                await message.reply(user_text)
                await _send_admin_message(
                    bot,
                    mirror_admin_id,
                    f"SUCCESS: from {sender} | file={filename} | no speech detected",
                )
                logging.info("Transcription finished with no speech sender=%s file=%s", sender, filename)
                return

            chunks = _chunk_text(text)
            if not chunks:
                user_text = "No speech detected."
                await message.reply(user_text)
                await _send_admin_message(
                    bot,
                    mirror_admin_id,
                    f"SUCCESS: from {sender} | file={filename} | empty transcript chunks",
                )
                logging.info("Transcription finished with empty chunks sender=%s file=%s", sender, filename)
                return

            for chunk in chunks:
                await message.reply(chunk)

            await _send_admin_message(
                bot,
                mirror_admin_id,
                f"SUCCESS: from {sender} | file={filename}",
            )
            logging.info(
                "Transcription success sender=%s file=%s chunks=%d chars=%d",
                sender,
                filename,
                len(chunks),
                len(text),
            )
        except TranscriptionTimeout:
            await _stop_progress_task(stop_progress, progress_task)
            user_text = "Transcription timed out. Try a shorter file."
            await status_message.edit_text(user_text)
            await _send_admin_message(
                bot,
                mirror_admin_id,
                f"FAIL: from {sender} | file={filename} | reason=timeout",
            )
            await _send_admin_message(bot, mirror_admin_id, user_text)
            logging.warning("Transcription timeout sender=%s file=%s", sender, filename)
        except TranscriptionError as exc:
            await _stop_progress_task(stop_progress, progress_task)
            logging.exception("Transcription error")
            user_text = f"Transcription failed: {exc}"
            await status_message.edit_text(user_text)
            await _send_admin_message(
                bot,
                mirror_admin_id,
                f"FAIL: from {sender} | file={filename} | reason=transcription error",
            )
            await _send_admin_message(bot, mirror_admin_id, user_text)
            logging.error("Transcription error sender=%s file=%s: %s", sender, filename, exc)
        except TelegramBadRequest:
            await _stop_progress_task(stop_progress, progress_task)
            logging.exception("Telegram API error while replying")
            await _send_admin_message(
                bot,
                mirror_admin_id,
                f"FAIL: from {sender} | file={filename} | reason=telegram API error",
            )
            logging.warning("TelegramBadRequest while replying sender=%s file=%s", sender, filename)
        except Exception:
            await _stop_progress_task(stop_progress, progress_task)
            logging.exception("Unexpected processing error")
            user_text = "Unexpected error while transcribing."
            await status_message.edit_text(user_text)
            await _send_admin_message(
                bot,
                mirror_admin_id,
                f"FAIL: from {sender} | file={filename} | reason=unexpected error",
            )
            await _send_admin_message(bot, mirror_admin_id, user_text)


def _extract_approve_target_id(message: Message) -> int | None:
    text = (message.text or "").strip()
    parts = text.split(maxsplit=1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[1].strip())
    except ValueError:
        return None


def build_dispatcher(settings: Settings) -> Dispatcher:
    dp = Dispatcher()
    transcriber = WhisperTranscriber(settings)
    model_name = _model_name_from_path(settings.whisper_model)
    auth_file = Path(os.environ.get("AUTHORIZED_USERS_FILE", "./authorized_users.json"))
    auth_store = AuthorizationStore(auth_file, settings.admin_id)

    @dp.message(Command("approve"))
    async def approve_handler(message: Message, bot: Bot) -> None:
        sender = message.from_user
        if settings.admin_id is None:
            await message.reply("ADMIN_ID is not configured.")
            return
        if sender is None or sender.id != settings.admin_id:
            await message.reply("Only admin can approve users.")
            return

        target_id = _extract_approve_target_id(message)
        if target_id is None:
            await message.reply("Usage: /approve <user_id>")
            return

        is_new = await auth_store.authorize(target_id)
        logging.info("Admin approve command sender=%s target_user_id=%s new=%s", sender.id if sender else None, target_id, is_new)
        if is_new:
            await message.reply(f"User {target_id} approved.")
        else:
            await message.reply(f"User {target_id} is already approved.")

        try:
            await bot.send_message(target_id, "Access granted. You can now use this bot.")
        except Exception:
            logging.exception("Failed to notify approved user")

    @dp.message(Command("start"))
    async def start_handler(message: Message, bot: Bot) -> None:
        if not await _ensure_user_is_authorized(message, bot, auth_store, settings.admin_id):
            return
        await message.reply(
            "Send a voice message, audio file, or video note and I will transcribe it."
        )

    @dp.message(Command("help"))
    async def help_handler(message: Message, bot: Bot) -> None:
        if not await _ensure_user_is_authorized(message, bot, auth_store, settings.admin_id):
            return
        await message.reply(
            "Supported types: voice, audio, video note.\n"
            "Max file size for bot download is 20 MB."
        )

    @dp.message(F.voice)
    async def voice_handler(message: Message, bot: Bot) -> None:
        if not await _ensure_user_is_authorized(message, bot, auth_store, settings.admin_id):
            return
        await _process_media_message(message, bot, transcriber, settings.admin_id, model_name)

    @dp.message(F.audio)
    async def audio_handler(message: Message, bot: Bot) -> None:
        if not await _ensure_user_is_authorized(message, bot, auth_store, settings.admin_id):
            return
        await _process_media_message(message, bot, transcriber, settings.admin_id, model_name)

    @dp.message(F.video_note)
    async def video_note_handler(message: Message, bot: Bot) -> None:
        if not await _ensure_user_is_authorized(message, bot, auth_store, settings.admin_id):
            return
        await _process_media_message(message, bot, transcriber, settings.admin_id, model_name)

    return dp


async def _run() -> None:
    logging.basicConfig(level=logging.INFO)

    settings = load_settings()
    dp = build_dispatcher(settings)
    bot = Bot(token=settings.bot_token)

    logging.info(
        "Starting bot with whisper binary=%s model=%s language=%s admin_id=%s",
        shlex.quote(str(settings.whisper_bin)),
        shlex.quote(str(settings.whisper_model)),
        settings.whisper_language,
        settings.admin_id,
    )

    await dp.start_polling(bot)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
