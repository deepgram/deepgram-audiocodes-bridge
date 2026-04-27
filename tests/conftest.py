"""Shared test fixtures and fake WebSocket helpers."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest


class FakeSocket:
    """Fake WebSocket that lets tests push inbound messages and capture outbound ones.

    Usage::

        sock = FakeSocket()
        sock.push_text(json.dumps({"type": "session.initiate", ...}))
        handler = AudioCodesHandler(sock, ("raw/mulaw",))
        await handler.run()
        sent = sock.sent_texts()
    """

    def __init__(self) -> None:
        self._inbound: asyncio.Queue[str | bytes | Exception] = asyncio.Queue()
        self._sent: list[str | bytes] = []
        self._closed = False

    # ── Inbound helpers ──────────────────────────────────────────────────────

    def push_text(self, msg: str) -> None:
        """Enqueue a text frame to be yielded by the async-for loop."""
        self._inbound.put_nowait(msg)

    def push_binary(self, data: bytes) -> None:
        """Enqueue a binary frame."""
        self._inbound.put_nowait(data)

    def push_json(self, obj: Any) -> None:
        """Enqueue a JSON-serialised text frame."""
        self._inbound.put_nowait(json.dumps(obj))

    def push_close(self) -> None:
        """Signal end-of-stream (stops the async-for loop)."""
        self._inbound.put_nowait(StopAsyncIteration())

    def push_error(self, exc: Exception) -> None:
        """Cause the async-for loop to raise ``exc``."""
        self._inbound.put_nowait(exc)

    # ── WebSocket protocol interface ─────────────────────────────────────────

    def __aiter__(self) -> "FakeSocket":
        return self

    async def __anext__(self) -> str | bytes:
        item = await self._inbound.get()
        if isinstance(item, StopAsyncIteration):
            raise StopAsyncIteration
        if isinstance(item, Exception):
            raise item
        return item

    async def send(self, data: str | bytes) -> None:
        self._sent.append(data)

    async def recv(self) -> str | bytes:
        item = await self._inbound.get()
        if isinstance(item, StopAsyncIteration):
            raise StopAsyncIteration
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self._closed = True

    # ── Inspection helpers ───────────────────────────────────────────────────

    def sent_texts(self) -> list[str]:
        return [m for m in self._sent if isinstance(m, str)]

    def sent_json(self) -> list[Any]:
        return [json.loads(m) for m in self.sent_texts()]

    def sent_binaries(self) -> list[bytes]:
        return [m for m in self._sent if isinstance(m, bytes)]

    @property
    def closed(self) -> bool:
        return self._closed

    # Allow attribute access that ServerConnection has (e.g. remote_address)
    remote_address: tuple[str, int] = ("127.0.0.1", 12345)


@pytest.fixture
def fake_socket() -> FakeSocket:
    return FakeSocket()


def make_session_initiate(
    conversation_id: str = "conv-123",
    caller: str = "+15551234567",
    bot_name: str = "test_bot",
    supported_formats: list[str] | None = None,
    expect_audio: bool = True,
) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "type": "session.initiate",
        "conversationId": conversation_id,
        "caller": caller,
        "botName": bot_name,
        "supportedMediaFormats": supported_formats or ["raw/mulaw", "raw/lpcm16"],
        "expectAudioMessages": expect_audio,
    }
    return msg
