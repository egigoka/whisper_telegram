# Test Suite: Telegram Whisper Bot

## Scope
This suite covers:
- Configuration loading (`telegram_whisper_bot/config.py`)
- Whisper wrapper behavior (`telegram_whisper_bot/transcriber.py`)
- Bot message pipeline and admin mirroring (`telegram_whisper_bot/bot.py`)
- End-to-end sanity checks with real `ffmpeg`/`whisper.cpp` (optional smoke tests)

Out of scope:
- Telegram platform reliability itself
- Whisper model quality benchmarking (accuracy/WER)

## Recommended Stack
- `pytest`
- `pytest-asyncio`
- `unittest.mock` / `pytest-mock`
- `tmp_path` fixture
- `monkeypatch` fixture

## Directory Layout

```text
whisper_telegram/
  tests/
    unit/
      test_config.py
      test_transcriber.py
      test_bot_utils.py
      test_bot_flow.py
    integration/
      test_bot_with_fake_whisper.py
    smoke/
      test_real_stack.md  # manual scripted smoke steps
```

---

## 1. Unit Tests: `config.py`

### 1.1 Token Source Priority
1. `test_load_settings_uses_env_token_over_secrets`
- Arrange: env `TELEGRAM_BOT_TOKEN=env_token`, secrets has another token.
- Assert: `Settings.bot_token == env_token`.

2. `test_load_settings_uses_secrets_when_env_missing`
- Arrange: no env token, secrets exposes `TELEGRAM_BOT_TOKEN`.
- Assert: token loaded from secrets.

3. `test_load_settings_accepts_whisper_telegram_token_fallback`
- Arrange: no env token, secrets has `WHISPER_TELEGRAM_TOKEN` only.
- Assert: token loaded.

4. `test_load_settings_accepts_bot_token_fallback`
- Arrange: no env token, secrets has `BOT_TOKEN` only.
- Assert: token loaded.

5. `test_load_settings_raises_if_no_token_anywhere`
- Arrange: no env token, secrets import fails or lacks token fields.
- Assert: `ValueError` with clear message.

### 1.2 Admin ID Loading
1. `test_admin_id_from_env`
- Arrange: env `ADMIN_ID=123`.
- Assert: `settings.admin_id == 123`.

2. `test_admin_id_from_secrets_admin_id`
- Arrange: no env `ADMIN_ID`, secrets has `ADMIN_ID=123`.
- Assert: `settings.admin_id == 123`.

3. `test_admin_id_from_secrets_my_chat_id_fallback`
- Arrange: no env `ADMIN_ID`, secrets has `MY_CHAT_ID=456`.
- Assert: `settings.admin_id == 456`.

4. `test_admin_id_none_when_absent`
- Arrange: no env/admin in secrets.
- Assert: `settings.admin_id is None`.

5. `test_admin_id_invalid_env_raises`
- Arrange: `ADMIN_ID=abc`.
- Assert: `ValueError`.

### 1.3 Numeric Constraints
1. `test_max_concurrent_must_be_positive`
- Arrange: `MAX_CONCURRENT=0`.
- Assert: `ValueError`.

2. `test_transcribe_timeout_must_be_positive`
- Arrange: `TRANSCRIBE_TIMEOUT=0`.
- Assert: `ValueError`.

3. `test_default_paths_and_language`
- Arrange: no path/language envs.
- Assert defaults match code.

---

## 2. Unit Tests: `transcriber.py`

Use temporary files and mock `asyncio.create_subprocess_exec`.

### 2.1 Preconditions
1. `test_transcribe_fails_when_binary_missing`
- Assert `TranscriptionError` contains missing binary path.

2. `test_transcribe_fails_when_model_missing`
- Assert `TranscriptionError` contains missing model path.

### 2.2 Process Launch and Command
1. `test_transcribe_invokes_whisper_with_expected_args`
- Assert command includes: `-m`, `-f`, `-l`, `--no-timestamps`, `-otxt`, `-of`.

2. `test_transcribe_wraps_filenotfound_as_transcription_error`
- Mock `create_subprocess_exec` raising `FileNotFoundError`.
- Assert `TranscriptionError("failed to start whisper binary")`.

### 2.3 Result Handling
1. `test_transcribe_timeout_kills_process`
- Mock `proc.communicate` to block and trigger `asyncio.TimeoutError` via `wait_for`.
- Assert `proc.kill()` called and raises `TranscriptionTimeout`.

2. `test_transcribe_nonzero_exit_raises_with_stderr_tail`
- Mock return code non-zero and stderr bytes.
- Assert raised `TranscriptionError` contains exit code and tail text.

3. `test_transcribe_raises_when_output_txt_missing`
- Mock successful process but no output txt file.
- Assert `TranscriptionError`.

4. `test_transcribe_returns_stripped_output`
- Create output txt with leading/trailing whitespace.
- Assert returned text is stripped.

### 2.4 Concurrency Control
1. `test_transcribe_respects_max_concurrent`
- Set `max_concurrent=1`, run two transcribe coroutines with controllable fake subprocess.
- Assert second invocation does not enter process section before first releases.

---

## 3. Unit Tests: `bot.py` Helpers

### 3.1 Text Chunking
1. `test_chunk_text_empty_returns_empty_list`
2. `test_chunk_text_under_limit_single_chunk`
3. `test_chunk_text_over_limit_multiple_chunks`
4. `test_chunk_text_exact_boundary`

### 3.2 Sender/Media Metadata
1. `test_sender_label_with_username`
2. `test_sender_label_without_username`
3. `test_sender_label_without_user`
4. `test_media_kind_and_filename_audio_uses_audio_filename`
5. `test_media_kind_and_filename_voice_generates_ogg_name`
6. `test_media_kind_and_filename_video_note_generates_mp4_name`

### 3.3 File Size Gate
1. `test_check_file_size_voice_ok`
2. `test_check_file_size_voice_too_large`
3. `test_check_file_size_audio_too_large`
4. `test_check_file_size_video_note_too_large`
5. `test_check_file_size_missing_size_defaults_true`

### 3.4 Admin Send Wrapper
1. `test_send_admin_message_no_admin_id_noop`
2. `test_send_admin_message_swallow_exceptions_and_log`

---

## 4. Unit/Behavior Tests: Message Processing Flow

Mock Telegram `Message`, `Bot`, `status_message`, transcriber, and ffmpeg helper.

### 4.1 Inbound Mirroring
1. `test_process_media_sends_inbound_metadata_to_admin`
- Assert admin receives `IN: from ... | type=... | file=...` once per incoming media.

### 4.2 Early Rejections
1. `test_process_media_rejects_too_large_and_reports_fail_to_admin`
- Assert user gets size error.
- Assert admin gets `FAIL ... reason=file too large`.
- Assert no status message posted.

2. `test_process_media_rejects_unsupported_and_reports_fail_to_admin`

### 4.3 Happy Path
1. `test_process_media_success_replies_with_transcript_chunks`
- Arrange transcript > 4096 chars.
- Assert user receives chunked replies.
- Assert admin receives `SUCCESS` then identical chunk sequence.
- Assert admin does not receive `Transcribing...`.

2. `test_process_media_success_no_speech_message_mirrored`
- Transcriber returns empty text.
- Assert user gets `No speech detected.`.
- Assert admin gets `SUCCESS ... no speech detected` + mirrored final message.

### 4.4 Failure Paths
1. `test_process_media_timeout_edits_status_and_mirrors_fail`
- Transcriber raises `TranscriptionTimeout`.
- Assert status edited to timeout text.
- Assert admin gets `FAIL ... reason=timeout` and mirrored final text.

2. `test_process_media_transcription_error_edits_status_and_mirrors_fail`
- Transcriber raises `TranscriptionError`.

3. `test_process_media_telegram_bad_request_logs_and_mirrors_fail`
- Simulate `message.reply` or status edit raising `TelegramBadRequest`.

4. `test_process_media_unexpected_exception_edits_status_and_mirrors_fail`

### 4.5 Download/Conversion Wiring
1. `test_process_media_downloads_file_before_ffmpeg`
- Assert call order: `get_file` -> `download_file` -> `_run_ffmpeg` -> `transcribe`.

2. `test_process_media_uses_tempdir_cleanup`
- Ensure temporary paths are inside ephemeral dir and removed after completion.

---

## 5. Integration Tests (No Real Telegram)

Build dispatcher and invoke handlers with fake update objects.

1. `test_build_dispatcher_registers_start_help_voice_audio_video_note_handlers`
2. `test_voice_handler_passes_admin_id_from_settings`
3. `test_audio_handler_passes_admin_id_from_settings`
4. `test_video_note_handler_passes_admin_id_from_settings`

---

## 6. Optional Real-Stack Smoke Tests

Run manually when deploying to a server with real token + whisper binaries.

### 6.1 Startup
1. Start bot process.
2. Verify startup log includes whisper paths and `admin_id` value.

### 6.2 Functional
1. Send short voice message (<20MB).
2. Expect:
- User receives transcript.
- Admin receives inbound metadata + `SUCCESS` + same transcript.
- Admin does not receive `Transcribing...`.

3. Send non-speech clip.
- Expect user/admin `No speech detected.` behavior.

4. Send large file (>20MB).
- Expect user size rejection and admin `FAIL ... file too large`.

5. Induce timeout (very long audio + low `TRANSCRIBE_TIMEOUT`).
- Expect timeout user text and admin `FAIL ... timeout`.

---

## 7. Negative and Security Tests

1. `test_no_shell_injection_surface`
- Validate subprocess calls use argument lists (no `shell=True`).

2. `test_admin_id_not_required_for_processing`
- With `admin_id=None`, user flow still works.

3. `test_transcriber_error_does_not_leak_full_internal_state`
- Assert user-facing error remains bounded and readable.

4. `test_large_stderr_truncated`
- Ensure error message contains only tail (bounded length).

---

## 8. Regression Checklist

Run this checklist on every change touching `config.py`, `transcriber.py`, or `bot.py`:
1. All unit tests pass.
2. At least one happy-path integration test passes.
3. Timeout and transcription-error tests pass.
4. Admin mirroring tests pass for inbound metadata and final replies.
5. No test asserts on transient exact temp paths.

---

## 9. Minimum Coverage Targets

- `config.py`: 95%+
- `transcriber.py`: 95%+
- `bot.py`: 90%+ (high branch coverage prioritized)

Suggested command:

```bash
pytest -q --maxfail=1 --disable-warnings --cov=telegram_whisper_bot --cov-report=term-missing
```

---

## 10. Priority Execution Order

1. `config.py` unit tests
2. `transcriber.py` unit tests
3. `bot.py` helper tests
4. behavior flow tests
5. integration tests
6. manual smoke tests on host

This order catches logic regressions early and leaves environment-dependent checks for the end.
