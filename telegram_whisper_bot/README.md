# Telegram Whisper Bot (whisper.cpp)

Async Telegram bot using `aiogram` + `whisper.cpp` to transcribe voice/audio/video-note messages.

## 1. Build whisper.cpp + model

```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
bash models/download-ggml-model.sh medium
cmake -B build -DWHISPER_OPENBLAS=ON
cmake --build build --config Release -j"$(nproc)"
```

## 2. Install Python deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r telegram_whisper_bot/requirements.txt
```

## 3. Configure env

```bash
export TELEGRAM_BOT_TOKEN='YOUR_BOT_TOKEN'
export WHISPER_BIN='./whisper.cpp/build/bin/whisper-cli'
export WHISPER_MODEL='./whisper.cpp/models/ggml-medium.bin'
export WHISPER_LANGUAGE='auto'
export MAX_CONCURRENT='1'  # one transcription at a time; others queue
export TRANSCRIBE_TIMEOUT='300'
```

## 4. Run

```bash
python3 -m telegram_whisper_bot.bot
```

## Notes

- `ffmpeg` must be installed and available in `PATH`.
- Telegram bot file download limit is 20 MB for these message types.
- The bot does not persist transcripts to disk.
