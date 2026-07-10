from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from telegram_whisper_bot.bot import AuthorizationStore, _ensure_user_is_authorized


class FakeMessage:
    def __init__(self, user_id: int, full_name: str = "User", username: str | None = "u") -> None:
        self.from_user = SimpleNamespace(id=user_id, full_name=full_name, username=username)
        self.replies: list[str] = []
        self.text = ""

    async def reply(self, text: str):
        self.replies.append(text)
        return SimpleNamespace()


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))


@pytest.mark.asyncio
async def test_auth_store_authorize_and_persist(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    store = AuthorizationStore(auth_file, admin_id=100)

    assert await store.is_authorized(100)
    assert not await store.is_authorized(200)

    assert await store.add_pending_if_needed(200)
    assert not await store.add_pending_if_needed(200)
    assert await store.authorize(200)
    assert not await store.authorize(200)
    assert await store.is_authorized(200)

    data = json.loads(auth_file.read_text(encoding="utf-8"))
    assert 100 in data["authorized_user_ids"]
    assert 200 in data["authorized_user_ids"]
    assert 200 not in data["pending_user_ids"]


@pytest.mark.asyncio
async def test_ensure_user_is_authorized_allows_admin(tmp_path: Path) -> None:
    store = AuthorizationStore(tmp_path / "auth.json", admin_id=999)
    msg = FakeMessage(user_id=999)
    bot = FakeBot()

    ok = await _ensure_user_is_authorized(msg, bot, store, admin_id=999)

    assert ok
    assert msg.replies == []
    assert bot.sent == []


@pytest.mark.asyncio
async def test_ensure_user_is_authorized_requests_approval_once(tmp_path: Path) -> None:
    store = AuthorizationStore(tmp_path / "auth.json", admin_id=999)
    msg = FakeMessage(user_id=111, full_name="New User", username="new")
    bot = FakeBot()

    ok1 = await _ensure_user_is_authorized(msg, bot, store, admin_id=999)
    ok2 = await _ensure_user_is_authorized(msg, bot, store, admin_id=999)

    assert not ok1
    assert not ok2
    assert msg.replies == ["Access pending admin approval.", "Access pending admin approval."]
    assert len(bot.sent) == 1
    assert bot.sent[0][0] == 999
    assert "AUTH REQUEST" in bot.sent[0][1]
    assert "/approve 111" in bot.sent[0][1]


@pytest.mark.asyncio
async def test_ensure_user_is_authorized_after_approve(tmp_path: Path) -> None:
    store = AuthorizationStore(tmp_path / "auth.json", admin_id=999)
    msg = FakeMessage(user_id=111)
    bot = FakeBot()

    await store.authorize(111)
    ok = await _ensure_user_is_authorized(msg, bot, store, admin_id=999)

    assert ok
    assert msg.replies == []
    assert bot.sent == []
