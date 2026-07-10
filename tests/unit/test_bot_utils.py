from __future__ import annotations

from types import SimpleNamespace

from telegram_whisper_bot.bot import (
    _check_file_size,
    _chunk_text,
    _media_kind_and_filename,
    _sender_label,
)


def test_chunk_text_empty_returns_empty_list() -> None:
    assert _chunk_text("") == []


def test_chunk_text_under_limit_single_chunk() -> None:
    text = "hello"
    assert _chunk_text(text, size=4096) == [text]


def test_chunk_text_over_limit_multiple_chunks() -> None:
    text = "a" * 9
    assert _chunk_text(text, size=4) == ["aaaa", "aaaa", "a"]


def test_chunk_text_exact_boundary() -> None:
    text = "a" * 8
    assert _chunk_text(text, size=4) == ["aaaa", "aaaa"]


def test_sender_label_with_username() -> None:
    msg = SimpleNamespace(from_user=SimpleNamespace(full_name="John Doe", username="john", id=10))
    assert _sender_label(msg) == "John Doe (@john, id=10)"


def test_sender_label_without_username() -> None:
    msg = SimpleNamespace(from_user=SimpleNamespace(full_name="John Doe", username=None, id=10))
    assert _sender_label(msg) == "John Doe (no_username, id=10)"


def test_sender_label_without_user() -> None:
    msg = SimpleNamespace(from_user=None)
    assert _sender_label(msg) == "unknown"


def test_media_kind_and_filename_audio_uses_audio_filename() -> None:
    audio = SimpleNamespace(file_name="song.mp3", file_unique_id="u1")
    msg = SimpleNamespace(audio=audio, voice=None, video_note=None)
    assert _media_kind_and_filename(msg) == ("audio", "song.mp3")


def test_media_kind_and_filename_voice_generates_ogg_name() -> None:
    voice = SimpleNamespace(file_unique_id="v1")
    msg = SimpleNamespace(audio=None, voice=voice, video_note=None)
    assert _media_kind_and_filename(msg) == ("voice", "voice_v1.ogg")


def test_media_kind_and_filename_video_note_generates_mp4_name() -> None:
    video_note = SimpleNamespace(file_unique_id="n1")
    msg = SimpleNamespace(audio=None, voice=None, video_note=video_note)
    assert _media_kind_and_filename(msg) == ("video_note", "video_note_n1.mp4")


def test_check_file_size_voice_ok() -> None:
    msg = SimpleNamespace(voice=SimpleNamespace(file_size=1024), audio=None, video_note=None)
    assert _check_file_size(msg)


def test_check_file_size_voice_too_large() -> None:
    msg = SimpleNamespace(voice=SimpleNamespace(file_size=20 * 1024 * 1024 + 1), audio=None, video_note=None)
    assert not _check_file_size(msg)


def test_check_file_size_audio_too_large() -> None:
    msg = SimpleNamespace(voice=None, audio=SimpleNamespace(file_size=20 * 1024 * 1024 + 1), video_note=None)
    assert not _check_file_size(msg)


def test_check_file_size_video_note_too_large() -> None:
    msg = SimpleNamespace(voice=None, audio=None, video_note=SimpleNamespace(file_size=20 * 1024 * 1024 + 1))
    assert not _check_file_size(msg)


def test_check_file_size_missing_size_defaults_true() -> None:
    msg = SimpleNamespace(voice=SimpleNamespace(file_size=None), audio=None, video_note=None)
    assert _check_file_size(msg)
