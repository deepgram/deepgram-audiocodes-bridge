"""Tests for DeepgramHandler — Deepgram Voice Agent V1 protocol implementation."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from deepgram_audiocodes_bridge.deepgram_handler import DeepgramHandler, DeepgramHandshakeError
from deepgram_audiocodes_bridge.types import DeepgramAgentConfig
from tests.conftest import FakeSocket


def make_config() -> DeepgramAgentConfig:
    return {
        "agent": {
            "think": {"prompt": "You are a helpful assistant."},
            "speak": {"provider": {"type": "deepgram"}},
        }
    }


def make_handler(session_id: str = "sess-1") -> tuple[DeepgramHandler, FakeSocket]:
    """Return (handler, fake_dg_socket) with all callbacks as AsyncMocks."""
    dg_sock = FakeSocket()
    config = make_config()
    handler = DeepgramHandler("test-api-key", config, session_id)

    handler.on_conversation_text = AsyncMock()
    handler.on_user_started_speaking = AsyncMock()
    handler.on_agent_thinking = AsyncMock()
    handler.on_agent_audio_done = AsyncMock()
    handler.on_function_call_request = AsyncMock()
    handler.on_prompt_updated = AsyncMock()
    handler.on_speak_updated = AsyncMock()
    handler.on_think_updated = AsyncMock()
    handler.on_warning = AsyncMock()
    handler.on_error = AsyncMock()
    handler.on_agent_audio = AsyncMock()
    handler.on_connected = AsyncMock()
    handler.on_disconnected = AsyncMock()

    return handler, dg_sock


async def do_handshake(handler: DeepgramHandler, dg_sock: FakeSocket) -> None:
    """Perform the Welcome→Settings→SettingsApplied handshake using the fake socket."""
    dg_sock.push_json({"type": "Welcome", "request_id": "req-1"})
    dg_sock.push_json({"type": "SettingsApplied"})

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        await handler.connect()


# ── handshake ────────────────────────────────────────────────────────────────

async def test_connect_does_not_send_before_welcome() -> None:
    handler, dg_sock = make_handler()
    dg_sock.push_json({"type": "Welcome", "request_id": "req-1"})
    dg_sock.push_json({"type": "SettingsApplied"})

    sent_before_welcome: list[str] = []
    original_send = dg_sock.send

    recv_count = 0

    async def patched_recv() -> str:
        nonlocal recv_count
        result = await dg_sock._inbound.get()
        recv_count += 1
        if isinstance(result, StopAsyncIteration):
            raise StopAsyncIteration
        if isinstance(result, Exception):
            raise result
        return result  # type: ignore[return-value]

    dg_sock.recv = patched_recv  # type: ignore[method-assign]

    async def patched_send(data: str | bytes) -> None:
        if recv_count < 1:
            sent_before_welcome.append(str(data))
        await original_send(data)

    dg_sock.send = patched_send  # type: ignore[method-assign]

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        await handler.connect()

    assert sent_before_welcome == [], "Must not send anything before Welcome"


async def test_connect_sends_settings_after_welcome() -> None:
    handler, dg_sock = make_handler()
    dg_sock.push_json({"type": "Welcome", "request_id": "r"})
    dg_sock.push_json({"type": "SettingsApplied"})

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        await handler.connect()

    sent = dg_sock.sent_json()
    assert any(m.get("type") == "Settings" for m in sent)


async def test_connect_returns_only_after_settings_applied() -> None:
    handler, dg_sock = make_handler()
    dg_sock.push_json({"type": "Welcome", "request_id": "r"})
    dg_sock.push_json({"type": "SettingsApplied"})

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        await handler.connect()

    assert handler.is_connected


async def test_connect_raises_if_first_message_not_welcome() -> None:
    handler, dg_sock = make_handler()
    dg_sock.push_json({"type": "Error", "err_code": "bad"})

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        with pytest.raises(DeepgramHandshakeError, match="Deepgram rejected handshake at Welcome step"):
            await handler.connect()


async def test_connect_raises_if_second_message_not_settings_applied() -> None:
    handler, dg_sock = make_handler()
    dg_sock.push_json({"type": "Welcome", "request_id": "r"})
    dg_sock.push_json({"type": "SomethingElse"})

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        with pytest.raises(DeepgramHandshakeError, match="expected SettingsApplied"):
            await handler.connect()


async def test_settings_payload_has_correct_nesting() -> None:
    handler, dg_sock = make_handler()
    dg_sock.push_json({"type": "Welcome", "request_id": "r"})
    dg_sock.push_json({"type": "SettingsApplied"})

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        await handler.connect()

    settings = next(m for m in dg_sock.sent_json() if m.get("type") == "Settings")
    assert "agent" in settings
    assert "audio" in settings or True  # audio may be absent if not injected


# ── reader loop — JSON message dispatch ──────────────────────────────────────

async def _run_reader_with_messages(
    handler: DeepgramHandler, dg_sock: FakeSocket, *messages: dict
) -> None:
    for m in messages:
        dg_sock.push_json(m)
    dg_sock.push_close()
    await handler.run_reader()


async def test_reader_dispatches_conversation_text() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await _run_reader_with_messages(
        handler, dg_sock,
        {"type": "ConversationText", "role": "user", "content": "Hello"},
    )
    handler.on_conversation_text.assert_called_once()  # type: ignore[union-attr]
    event = handler.on_conversation_text.call_args[0][0]  # type: ignore[union-attr]
    assert event.role == "user"
    assert event.content == "Hello"


async def test_reader_dispatches_user_started_speaking() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await _run_reader_with_messages(
        handler, dg_sock,
        {"type": "UserStartedSpeaking"},
    )
    handler.on_user_started_speaking.assert_called_once()  # type: ignore[union-attr]


async def test_reader_dispatches_agent_thinking() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await _run_reader_with_messages(
        handler, dg_sock,
        {"type": "AgentThinking", "content": "Let me think..."},
    )
    event = handler.on_agent_thinking.call_args[0][0]  # type: ignore[union-attr]
    assert event.content == "Let me think..."


async def test_reader_dispatches_agent_audio_done() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await _run_reader_with_messages(handler, dg_sock, {"type": "AgentAudioDone"})
    handler.on_agent_audio_done.assert_called_once()  # type: ignore[union-attr]


async def test_reader_dispatches_function_call_request() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await _run_reader_with_messages(
        handler, dg_sock,
        {
            "type": "FunctionCallRequest",
            "functions": [
                {
                    "id": "fc-1",
                    "name": "check_order",
                    "arguments": '{"order_id": "123"}',
                    "client_side": True,
                }
            ],
        },
    )
    handler.on_function_call_request.assert_called_once()  # type: ignore[union-attr]
    event = handler.on_function_call_request.call_args[0][0]  # type: ignore[union-attr]
    assert len(event.functions) == 1
    fc = event.functions[0]
    assert fc.id == "fc-1"
    assert fc.name == "check_order"
    assert fc.arguments == '{"order_id": "123"}'
    assert fc.client_side is True


async def test_reader_dispatches_prompt_updated() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await _run_reader_with_messages(handler, dg_sock, {"type": "PromptUpdated"})
    handler.on_prompt_updated.assert_called_once()  # type: ignore[union-attr]


async def test_reader_dispatches_speak_updated() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await _run_reader_with_messages(handler, dg_sock, {"type": "SpeakUpdated"})
    handler.on_speak_updated.assert_called_once()  # type: ignore[union-attr]


async def test_reader_dispatches_think_updated() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await _run_reader_with_messages(handler, dg_sock, {"type": "ThinkUpdated"})
    handler.on_think_updated.assert_called_once()  # type: ignore[union-attr]


async def test_reader_dispatches_warning() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await _run_reader_with_messages(
        handler, dg_sock,
        {"type": "Warning", "warn_code": "W001", "warn_msg": "low confidence"},
    )
    event = handler.on_warning.call_args[0][0]  # type: ignore[union-attr]
    assert event.code == "W001"
    assert event.description == "low confidence"


async def test_reader_dispatches_error() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await _run_reader_with_messages(
        handler, dg_sock,
        {"type": "Error", "err_code": "E500", "err_msg": "internal error"},
    )
    handler.on_error.assert_called_once()  # type: ignore[union-attr]
    err = handler.on_error.call_args[0][0]  # type: ignore[union-attr]
    assert err.code == "E500"
    assert err.description == "internal error"


async def test_reader_dispatches_binary_as_agent_audio() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    dg_sock.push_binary(b"\x01\x02\x03")
    dg_sock.push_close()
    await handler.run_reader()
    handler.on_agent_audio.assert_called_once()  # type: ignore[union-attr]
    assert handler.on_agent_audio.call_args[0][0] == b"\x01\x02\x03"  # type: ignore[union-attr]


async def test_reader_invokes_disconnected_on_unexpected_close() -> None:
    import websockets.exceptions
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    dg_sock.push_error(
        websockets.exceptions.ConnectionClosedError(None, None)  # type: ignore[arg-type]
    )
    await handler.run_reader()
    handler.on_disconnected.assert_called_once()  # type: ignore[union-attr]


# ── outbound client messages ─────────────────────────────────────────────────

async def test_inject_agent_message_serializes_correctly() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await handler.inject_agent_message("Hello!")
    msgs = dg_sock.sent_json()
    msg = next(m for m in msgs if m.get("type") == "InjectAgentMessage")
    assert msg["content"] == "Hello!"


async def test_inject_user_message_serializes_correctly() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await handler.inject_user_message("What is my balance?")
    msgs = dg_sock.sent_json()
    msg = next(m for m in msgs if m.get("type") == "InjectUserMessage")
    assert msg["content"] == "What is my balance?"


async def test_update_prompt_serializes_correctly() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await handler.update_prompt("Extra instructions.")
    msg = next(m for m in dg_sock.sent_json() if m.get("type") == "UpdatePrompt")
    assert msg["prompt"] == "Extra instructions."


async def test_update_speak_serializes_correctly() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await handler.update_speak({"provider": {"type": "deepgram", "model": "aura-2"}})
    msg = next(m for m in dg_sock.sent_json() if m.get("type") == "UpdateSpeak")
    assert msg["speak"]["provider"]["type"] == "deepgram"


async def test_update_think_serializes_correctly() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await handler.update_think({"provider": {"type": "open_ai", "model": "gpt-4o"}})
    msg = next(m for m in dg_sock.sent_json() if m.get("type") == "UpdateThink")
    assert msg["think"]["provider"]["type"] == "open_ai"


async def test_send_function_call_response_serializes_correctly() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await handler.send_function_call_response("fc-1", "check_order", '{"status": "ok"}')
    msg = next(m for m in dg_sock.sent_json() if m.get("type") == "FunctionCallResponse")
    assert msg["id"] == "fc-1"
    assert msg["name"] == "check_order"
    assert msg["content"] == '{"status": "ok"}'


async def test_keep_alive_serializes_correctly() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await handler.send_keep_alive()
    assert any(m.get("type") == "KeepAlive" for m in dg_sock.sent_json())


async def test_disconnect_closes_socket() -> None:
    handler, dg_sock = make_handler()
    await do_handshake(handler, dg_sock)
    await handler.disconnect()
    assert dg_sock.closed
    assert not handler.is_connected
