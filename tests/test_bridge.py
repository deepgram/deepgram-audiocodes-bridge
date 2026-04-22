"""Tests for DeepgramBridge — event wiring, state machine, and public API."""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepgram_audiocodes_bridge.bridge import DeepgramBridge, SessionState, _inject_audio_format
from deepgram_audiocodes_bridge.types import (
    BridgeConfig,
    BridgeErrorEvent,
    ConversationTextEvent,
    DeepgramAgentConfig,
    SessionEndEvent,
    SessionStartEvent,
)
from tests.conftest import FakeSocket, make_session_initiate


def make_config(
    ac_token: str = "secret",
    preferred: tuple[str, ...] = ("raw/mulaw",),
) -> BridgeConfig:
    return BridgeConfig(
        deepgram_api_key="dg-key",
        deepgram_config={"agent": {"think": {"prompt": "You are helpful."}}},
        ac_token=ac_token,
        preferred_media_formats=preferred,
    )


def make_bridge(**kwargs: Any) -> DeepgramBridge:
    return DeepgramBridge(make_config(**kwargs))


# ── Audio format injection ────────────────────────────────────────────────────

@pytest.mark.parametrize("fmt,encoding,sample_rate", [
    ("raw/mulaw", "mulaw", 8000),
    ("wav/mulaw", "mulaw", 8000),
    ("raw/lpcm16_8", "linear16", 8000),
    ("wav/lpcm16_8", "linear16", 8000),
    ("raw/lpcm16", "linear16", 16000),
    ("wav/lpcm16", "linear16", 16000),
    ("raw/lpcm16_24", "linear16", 24000),
    ("wav/lpcm16_24", "linear16", 24000),
])
def test_inject_audio_format(fmt: str, encoding: str, sample_rate: int) -> None:
    config: DeepgramAgentConfig = {}
    _inject_audio_format(config, fmt)  # type: ignore[arg-type]
    assert config["audio"]["input"]["encoding"] == encoding  # type: ignore[index]
    assert config["audio"]["input"]["sample_rate"] == sample_rate  # type: ignore[index]
    assert config["audio"]["output"]["encoding"] == encoding  # type: ignore[index]
    assert config["audio"]["output"]["sample_rate"] == sample_rate  # type: ignore[index]
    assert config["audio"]["output"]["container"] == "none"  # type: ignore[index]


# ── session_start event emission ─────────────────────────────────────────────

async def _simulate_session(
    bridge: DeepgramBridge,
    ac_sock: FakeSocket,
    dg_sock: FakeSocket,
    extra_ac_messages: list[dict] | None = None,
) -> None:
    """Pump one full session through the bridge using fake sockets.

    Pushes session.initiate through the AC socket, performs the Deepgram
    handshake on the DG socket, optionally sends extra AC messages, then
    closes both sockets cleanly.
    """
    # AudioCodes side
    ac_sock.push_json(make_session_initiate())
    if extra_ac_messages:
        for m in extra_ac_messages:
            ac_sock.push_json(m)
    ac_sock.push_close()

    # Deepgram side
    dg_sock.push_json({"type": "Welcome", "request_id": "r"})
    dg_sock.push_json({"type": "SettingsApplied"})
    dg_sock.push_close()

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        await bridge._handle_connection(ac_sock)  # type: ignore[arg-type]


async def test_session_start_emitted_with_correct_fields() -> None:
    bridge = make_bridge()
    events: list[SessionStartEvent] = []
    bridge.on("session_start")(lambda e: events.append(e))  # type: ignore[arg-type]

    ac_sock, dg_sock = FakeSocket(), FakeSocket()
    await _simulate_session(bridge, ac_sock, dg_sock)

    assert len(events) == 1
    e = events[0]
    assert e.conversation_id == "conv-123"
    assert e.caller == "+15551234567"
    assert e.bot_name == "test_bot"
    assert e.media_format == "raw/mulaw"
    assert e.session_id != ""


async def test_session_start_conversation_id_is_from_initiate() -> None:
    bridge = make_bridge()
    events: list[SessionStartEvent] = []

    @bridge.on("session_start")
    async def _(e: SessionStartEvent) -> None:
        events.append(e)

    ac_sock, dg_sock = FakeSocket(), FakeSocket()
    ac_sock.push_json(make_session_initiate(conversation_id="my-conv-id"))
    ac_sock.push_close()
    dg_sock.push_json({"type": "Welcome", "request_id": "r"})
    dg_sock.push_json({"type": "SettingsApplied"})
    dg_sock.push_close()

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        await bridge._handle_connection(ac_sock)  # type: ignore[arg-type]

    assert events[0].conversation_id == "my-conv-id"


# ── deepgram config injection ────────────────────────────────────────────────

async def test_bridge_injects_audio_format_into_deepgram_config() -> None:
    """Bridge must inject audio.input/output into the Settings payload sent to Deepgram."""
    bridge = make_bridge()
    ac_sock, dg_sock = FakeSocket(), FakeSocket()
    ac_sock.push_json(make_session_initiate())
    ac_sock.push_close()
    dg_sock.push_json({"type": "Welcome", "request_id": "r"})
    dg_sock.push_json({"type": "SettingsApplied"})
    dg_sock.push_close()

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        await bridge._handle_connection(ac_sock)  # type: ignore[arg-type]

    # The Settings message is sent from DeepgramHandler.connect() to dg_sock
    sent = dg_sock.sent_json()
    settings = next((m for m in sent if m.get("type") == "Settings"), None)
    assert settings is not None, "No Settings message was sent to Deepgram"
    audio = settings.get("audio", {})
    assert audio.get("input", {}).get("encoding") == "mulaw"
    assert audio.get("input", {}).get("sample_rate") == 8000
    assert audio.get("output", {}).get("encoding") == "mulaw"
    assert audio.get("output", {}).get("container") == "none"


# ── transcript accumulation ──────────────────────────────────────────────────

async def test_conversation_text_accumulated_in_order() -> None:
    bridge = make_bridge()
    ac_sock, dg_sock = FakeSocket(), FakeSocket()
    ac_sock.push_json(make_session_initiate())
    ac_sock.push_close()

    dg_sock.push_json({"type": "Welcome", "request_id": "r"})
    dg_sock.push_json({"type": "SettingsApplied"})
    dg_sock.push_json({"type": "ConversationText", "role": "user", "content": "Hi"})
    dg_sock.push_json({"type": "ConversationText", "role": "assistant", "content": "Hello!"})
    dg_sock.push_close()

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        await bridge._handle_connection(ac_sock)  # type: ignore[arg-type]

    transcript = bridge.get_transcript()
    assert len(transcript) == 2
    assert transcript[0].role == "user"
    assert transcript[0].content == "Hi"
    assert transcript[1].role == "assistant"
    assert transcript[1].content == "Hello!"


async def test_get_transcript_returns_copy() -> None:
    bridge = make_bridge()
    ac_sock, dg_sock = FakeSocket(), FakeSocket()
    ac_sock.push_json(make_session_initiate())
    ac_sock.push_close()
    dg_sock.push_json({"type": "Welcome", "request_id": "r"})
    dg_sock.push_json({"type": "SettingsApplied"})
    dg_sock.push_json({"type": "ConversationText", "role": "user", "content": "Hi"})
    dg_sock.push_close()

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        await bridge._handle_connection(ac_sock)  # type: ignore[arg-type]

    t1 = bridge.get_transcript()
    t1.clear()
    t2 = bridge.get_transcript()
    assert len(t2) == 1


# ── session_end ────────────────────────────────────────────────────────────────

async def test_caller_hangup_emits_session_end_with_hangup_reason() -> None:
    bridge = make_bridge()
    end_events: list[SessionEndEvent] = []
    bridge.on("session_end")(lambda e: end_events.append(e))  # type: ignore[arg-type]

    ac_sock, dg_sock = FakeSocket(), FakeSocket()
    await _simulate_session(
        bridge, ac_sock, dg_sock,
        extra_ac_messages=[{"type": "session.end"}],
    )

    assert any(e.reason == "hangup" for e in end_events)


async def test_session_end_emitted_before_cleanup() -> None:
    """session_end must be emitted in every termination path."""
    bridge = make_bridge()
    emitted = []
    bridge.on("session_end")(lambda e: emitted.append(e))  # type: ignore[arg-type]

    ac_sock, dg_sock = FakeSocket(), FakeSocket()
    await _simulate_session(bridge, ac_sock, dg_sock)

    assert len(emitted) >= 1


# ── barge-in ────────────────────────────────────────────────────────────────

async def test_user_started_speaking_triggers_barge_in() -> None:
    bridge = make_bridge()
    flush_called = []

    ac_sock, dg_sock = FakeSocket(), FakeSocket()
    ac_sock.push_json(make_session_initiate())
    ac_sock.push_close()
    dg_sock.push_json({"type": "Welcome", "request_id": "r"})
    dg_sock.push_json({"type": "SettingsApplied"})
    dg_sock.push_json({"type": "UserStartedSpeaking"})
    dg_sock.push_close()

    original_flush = None

    async def mock_flush() -> None:
        flush_called.append(True)

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ), patch(
        "deepgram_audiocodes_bridge.audio_router.AudioRouter.flush_audiocodes_playback",
        side_effect=mock_flush,
    ):
        await bridge._handle_connection(ac_sock)  # type: ignore[arg-type]

    assert flush_called


# ── speech lifecycle mirroring ───────────────────────────────────────────────

async def test_user_conversation_text_sends_speech_recognition_and_committed() -> None:
    bridge = make_bridge()
    ac_sock, dg_sock = FakeSocket(), FakeSocket()
    ac_sock.push_json(make_session_initiate())
    ac_sock.push_close()
    dg_sock.push_json({"type": "Welcome", "request_id": "r"})
    dg_sock.push_json({"type": "SettingsApplied"})
    dg_sock.push_json({"type": "ConversationText", "role": "user", "content": "Tell me more"})
    dg_sock.push_close()

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        await bridge._handle_connection(ac_sock)  # type: ignore[arg-type]

    msgs = ac_sock.sent_json()
    recog = next(
        (m for m in msgs if m.get("type") == "userStream.speech.recognition"), None
    )
    committed = next(
        (m for m in msgs if m.get("type") == "userStream.speech.committed"), None
    )
    assert recog is not None
    assert recog["text"] == "Tell me more"
    assert recog["confidence"] == 1.0
    assert committed is not None


async def test_assistant_conversation_text_does_not_send_speech_messages() -> None:
    bridge = make_bridge()
    ac_sock, dg_sock = FakeSocket(), FakeSocket()
    ac_sock.push_json(make_session_initiate())
    ac_sock.push_close()
    dg_sock.push_json({"type": "Welcome", "request_id": "r"})
    dg_sock.push_json({"type": "SettingsApplied"})
    dg_sock.push_json({"type": "ConversationText", "role": "assistant", "content": "Hello!"})
    dg_sock.push_close()

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        await bridge._handle_connection(ac_sock)  # type: ignore[arg-type]

    msgs = ac_sock.sent_json()
    assert not any(m.get("type") == "userStream.speech.recognition" for m in msgs)
    assert not any(m.get("type") == "userStream.speech.committed" for m in msgs)


async def test_user_started_speaking_sends_speech_started_to_vaic() -> None:
    bridge = make_bridge()
    ac_sock, dg_sock = FakeSocket(), FakeSocket()
    ac_sock.push_json(make_session_initiate())
    ac_sock.push_close()
    dg_sock.push_json({"type": "Welcome", "request_id": "r"})
    dg_sock.push_json({"type": "SettingsApplied"})
    dg_sock.push_json({"type": "UserStartedSpeaking"})
    dg_sock.push_close()

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        await bridge._handle_connection(ac_sock)  # type: ignore[arg-type]

    msgs = ac_sock.sent_json()
    assert any(m.get("type") == "userStream.speech.started" for m in msgs)


# ── inbound activities bubble up ─────────────────────────────────────────────

async def test_inbound_activities_bubble_as_activity_event() -> None:
    bridge = make_bridge()
    activity_events = []
    bridge.on("activity")(lambda e: activity_events.append(e))  # type: ignore[arg-type]

    ac_sock, dg_sock = FakeSocket(), FakeSocket()
    await _simulate_session(
        bridge, ac_sock, dg_sock,
        extra_ac_messages=[{
            "type": "activities",
            "activities": [{"type": "event", "name": "dtmf", "value": "9"}],
        }],
    )

    assert any(
        e.activity.get("name") == "dtmf"  # type: ignore[union-attr]
        for e in activity_events
    )


# ── state-gating: ENDED state raises RuntimeError ────────────────────────────

async def _build_ended_bridge() -> DeepgramBridge:
    bridge = make_bridge()
    bridge._state = SessionState.ENDED
    bridge._dg_handler = MagicMock()
    bridge._ac_handler = MagicMock()
    return bridge


@pytest.mark.parametrize("method,args", [
    ("send_agent_message", ("hello",)),
    ("send_user_message", ("hi",)),
    ("update_prompt", ("prompt",)),
    ("update_speak", ({"provider": {}},)),
    ("update_think", ({"provider": {}},)),
    ("respond_to_function_call", ("id", "name", "content")),
    ("send_activity", ({"type": "event", "name": "hangup"},)),
    ("play_url", ("https://example.com/x.wav",)),
    ("send_dtmf", ("123",)),
    ("send_meta_data", ({"foo": "bar"},)),
    ("abort_prompts", ()),
    ("expect_another_bot_message", ()),
    ("apply_config", ({"language": "es-ES"},)),
    ("start_call_recording", ()),
    ("stop_call_recording", ()),
    ("pause_call_recording", ()),
    ("resume_call_recording", ()),
])
async def test_ended_state_raises_runtime_error(method: str, args: tuple) -> None:
    bridge = await _build_ended_bridge()
    with pytest.raises(RuntimeError, match="session is ended"):
        await getattr(bridge, method)(*args)


# ── application handler exception firewall ────────────────────────────────────

async def test_application_handler_exception_emits_recoverable_error() -> None:
    bridge = make_bridge()
    error_events: list[BridgeErrorEvent] = []

    @bridge.on("session_start")
    async def bad_handler(e: SessionStartEvent) -> None:
        raise ValueError("crash in handler")

    @bridge.on("error")
    async def on_err(e: BridgeErrorEvent) -> None:
        error_events.append(e)

    ac_sock, dg_sock = FakeSocket(), FakeSocket()
    await _simulate_session(bridge, ac_sock, dg_sock)

    assert any(e.recoverable for e in error_events)
    assert any("crash in handler" in e.description for e in error_events)


# ── send_activity envelope format ────────────────────────────────────────────

async def test_send_activity_single_element_envelope() -> None:
    bridge = make_bridge()
    bridge._state = SessionState.ACTIVE
    ac_sock = FakeSocket()
    mock_ac = MagicMock()
    mock_ac.send_activity = AsyncMock()
    bridge._ac_handler = mock_ac

    await bridge.send_activity({"type": "event", "name": "playUrl", "activityParams": {"playUrlUrl": "x"}})
    mock_ac.send_activity.assert_called_once()  # type: ignore[union-attr]


async def test_send_activity_list_two_elements() -> None:
    bridge = make_bridge()
    bridge._state = SessionState.ACTIVE
    mock_ac = MagicMock()
    mock_ac.send_activity = AsyncMock()
    bridge._ac_handler = mock_ac

    acts: list[Any] = [
        {"type": "event", "name": "hangup"},
        {"type": "event", "name": "abortPrompts"},
    ]
    await bridge.send_activity(acts)
    mock_ac.send_activity.assert_called_once_with(acts)  # type: ignore[union-attr]


# ── typed helper delegates to send_activity ──────────────────────────────────

@pytest.mark.parametrize("helper,ac_method", [
    ("play_url", "send_play_url"),
    ("send_dtmf", "send_dtmf"),
    ("send_meta_data", "send_meta_data"),
    ("abort_prompts", "send_abort_prompts"),
    ("expect_another_bot_message", "send_expect_another_bot_message"),
    ("apply_config", "send_config"),
    ("start_call_recording", "send_start_call_recording"),
    ("stop_call_recording", "send_stop_call_recording"),
    ("pause_call_recording", "send_pause_call_recording"),
    ("resume_call_recording", "send_resume_call_recording"),
])
async def test_typed_helpers_delegate_to_ac_handler(helper: str, ac_method: str) -> None:
    bridge = make_bridge()
    bridge._state = SessionState.ACTIVE
    mock_ac = MagicMock()
    setattr(mock_ac, ac_method, AsyncMock())
    bridge._ac_handler = mock_ac

    method = getattr(bridge, helper)
    # Build appropriate args per helper
    arg_map: dict[str, tuple] = {
        "play_url": ("https://x.com/file.wav",),
        "send_dtmf": ("123",),
        "send_meta_data": ({"k": "v"},),
        "abort_prompts": (),
        "expect_another_bot_message": (),
        "apply_config": ({"language": "en"},),
        "start_call_recording": (),
        "stop_call_recording": (),
        "pause_call_recording": (),
        "resume_call_recording": (),
    }
    await method(*arg_map.get(helper, ()))
    getattr(mock_ac, ac_method).assert_called_once()  # type: ignore[union-attr]


# ── transfer emits session_end with 'transfer' reason ────────────────────────

async def test_transfer_emits_session_end_with_transfer_reason() -> None:
    bridge = make_bridge()
    end_events: list[SessionEndEvent] = []
    bridge.on("session_end")(lambda e: end_events.append(e))  # type: ignore[arg-type]

    bridge._state = SessionState.ACTIVE
    bridge._session_id = "sess-xyz"
    mock_ac = MagicMock()
    mock_ac.send_transfer = AsyncMock()
    mock_ac.close = AsyncMock()
    bridge._ac_handler = mock_ac
    mock_dg = MagicMock()
    mock_dg.disconnect = AsyncMock()
    bridge._dg_handler = mock_dg

    await bridge.transfer("sip:queue@x.com")

    assert any(e.reason == "transfer" for e in end_events)


# ── end_session emits session_end with 'ended' reason ────────────────────────

async def test_end_session_emits_session_end_with_ended_reason() -> None:
    bridge = make_bridge()
    end_events: list[SessionEndEvent] = []
    bridge.on("session_end")(lambda e: end_events.append(e))  # type: ignore[arg-type]

    bridge._state = SessionState.ACTIVE
    bridge._session_id = "sess-abc"
    mock_ac = MagicMock()
    mock_ac.send_hangup = AsyncMock()
    mock_ac.close = AsyncMock()
    bridge._ac_handler = mock_ac
    mock_dg = MagicMock()
    mock_dg.disconnect = AsyncMock()
    bridge._dg_handler = mock_dg

    await bridge.end_session()
    assert any(e.reason == "ended" for e in end_events)


# ── expectAudioMessages: false ────────────────────────────────────────────────

async def test_expect_audio_messages_false_never_emits_session_start() -> None:
    bridge = make_bridge()
    start_events = []
    bridge.on("session_start")(lambda e: start_events.append(e))  # type: ignore[arg-type]

    ac_sock = FakeSocket()
    ac_sock.push_json(make_session_initiate(expect_audio=False))
    ac_sock.push_close()

    dg_sock = FakeSocket()
    dg_sock.push_close()

    with patch(
        "deepgram_audiocodes_bridge.deepgram_handler.websockets.asyncio.client.connect",
        new=AsyncMock(return_value=dg_sock),
    ):
        await bridge._handle_connection(ac_sock)  # type: ignore[arg-type]

    assert start_events == []
