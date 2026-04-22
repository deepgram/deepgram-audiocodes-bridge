"""Tests for AudioCodesHandler — AudioCodes Bot API protocol implementation."""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from deepgram_audiocodes_bridge.audiocodes_handler import AudioCodesHandler, PlayStream
from tests.conftest import FakeSocket, make_session_initiate


def make_handler(
    socket: FakeSocket,
    preferred: tuple[str, ...] = ("raw/mulaw", "raw/lpcm16"),
) -> AudioCodesHandler:
    handler = AudioCodesHandler(socket, preferred)  # type: ignore[arg-type]
    handler.on_session_initiate = AsyncMock()
    handler.on_session_resumed = AsyncMock()
    handler.on_user_audio = AsyncMock()
    handler.on_activity = AsyncMock()
    handler.on_session_end = AsyncMock()
    handler.on_error = AsyncMock()
    return handler


async def run_with_messages(handler: AudioCodesHandler, sock: FakeSocket, *msgs: Any) -> None:
    """Push messages, close stream, then run handler to completion."""
    for m in msgs:
        sock.push_json(m) if isinstance(m, dict) else sock.push_text(m)
    sock.push_close()
    await handler.run()


# ── session.initiate ─────────────────────────────────────────────────────────

async def test_session_initiate_sends_accepted_before_callback() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)

    order: list[str] = []
    sent_before_callback: list[Any] = []

    async def on_initiate(msg: Any) -> None:
        order.append("callback")
        sent_before_callback.extend(sock.sent_json())

    handler.on_session_initiate = on_initiate
    await run_with_messages(handler, sock, make_session_initiate())

    assert order == ["callback"]
    assert any(m.get("type") == "session.accepted" for m in sent_before_callback)


async def test_session_initiate_chooses_first_preferred_format() -> None:
    sock = FakeSocket()
    handler = make_handler(sock, preferred=("raw/mulaw", "raw/lpcm16"))
    await run_with_messages(
        handler, sock,
        make_session_initiate(supported_formats=["raw/lpcm16", "raw/mulaw"]),
    )
    accepted = next(m for m in sock.sent_json() if m.get("type") == "session.accepted")
    assert accepted["mediaFormat"] == "raw/mulaw"


async def test_session_initiate_no_common_format_sends_error() -> None:
    sock = FakeSocket()
    handler = make_handler(sock, preferred=("raw/mulaw",))
    await run_with_messages(
        handler, sock,
        make_session_initiate(supported_formats=["raw/lpcm16"]),
    )
    msgs = sock.sent_json()
    assert any(m.get("type") == "session.error" for m in msgs)
    handler.on_session_initiate.assert_not_called()  # type: ignore[union-attr]
    assert sock.closed


async def test_session_initiate_expect_audio_false_sends_error() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(
        handler, sock,
        make_session_initiate(expect_audio=False),
    )
    msgs = sock.sent_json()
    err = next(m for m in msgs if m.get("type") == "session.error")
    assert "expectAudioMessages" in err["reason"]
    handler.on_session_initiate.assert_not_called()  # type: ignore[union-attr]
    assert sock.closed


async def test_session_initiate_expect_audio_true_proceeds() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(
        handler, sock,
        make_session_initiate(expect_audio=True),
    )
    handler.on_session_initiate.assert_called_once()  # type: ignore[union-attr]


async def test_session_initiate_expect_audio_omitted_proceeds() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    msg = make_session_initiate()
    del msg["expectAudioMessages"]
    await run_with_messages(handler, sock, msg)
    handler.on_session_initiate.assert_called_once()  # type: ignore[union-attr]


async def test_session_initiate_captures_metadata() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(
        handler, sock,
        make_session_initiate(
            conversation_id="c-999",
            caller="+15550001234",
            bot_name="my_bot",
        ),
    )
    assert handler.conversation_id == "c-999"
    assert handler.caller == "+15550001234"
    assert handler.bot_name == "my_bot"
    assert handler.media_format == "raw/mulaw"


# ── userStream lifecycle ──────────────────────────────────────────────────────

async def test_user_stream_start_sends_started() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(
        handler, sock,
        make_session_initiate(),
        {"type": "userStream.start"},
    )
    msgs = sock.sent_json()
    assert any(m.get("type") == "userStream.started" for m in msgs)


async def test_user_stream_started_has_no_participant_field() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(
        handler, sock,
        make_session_initiate(),
        {"type": "userStream.start", "participant": "caller"},
    )
    started = next(m for m in sock.sent_json() if m.get("type") == "userStream.started")
    assert "participant" not in started


async def test_user_stream_chunk_decoded_and_forwarded() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    raw_audio = b"\x00\x01\x02\x03"
    encoded = base64.b64encode(raw_audio).decode()
    await run_with_messages(
        handler, sock,
        make_session_initiate(),
        {"type": "userStream.start"},
        {"type": "userStream.chunk", "audioChunk": encoded},
    )
    handler.on_user_audio.assert_called_once()  # type: ignore[union-attr]
    call_args = handler.on_user_audio.call_args  # type: ignore[union-attr]
    assert call_args[0][0] == raw_audio


async def test_user_audio_meta_has_participant() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    encoded = base64.b64encode(b"\xff").decode()
    await run_with_messages(
        handler, sock,
        make_session_initiate(),
        {"type": "userStream.start", "participant": "caller-leg"},
        {"type": "userStream.chunk", "audioChunk": encoded},
    )
    call_args = handler.on_user_audio.call_args  # type: ignore[union-attr]
    meta = call_args[0][1]
    assert meta.get("participant") == "caller-leg"


async def test_user_stream_stop_sends_stopped() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(
        handler, sock,
        make_session_initiate(),
        {"type": "userStream.stop"},
    )
    assert any(m.get("type") == "userStream.stopped" for m in sock.sent_json())


async def test_user_stream_stopped_has_no_participant_field() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(
        handler, sock,
        make_session_initiate(),
        {"type": "userStream.start", "participant": "caller"},
        {"type": "userStream.stop"},
    )
    stopped = next(m for m in sock.sent_json() if m.get("type") == "userStream.stopped")
    assert "participant" not in stopped


# ── activities ───────────────────────────────────────────────────────────────

async def test_activities_invokes_on_activity_per_element() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    activities = [
        {"type": "event", "name": "dtmf", "value": "5"},
        {"type": "event", "name": "start"},
    ]
    await run_with_messages(
        handler, sock,
        make_session_initiate(),
        {"type": "activities", "activities": activities},
    )
    assert handler.on_activity.call_count == 2  # type: ignore[union-attr]


async def test_activities_preserves_all_base_fields() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    activity = {
        "type": "event",
        "name": "start",
        "activityParams": {"foo": "bar"},
        "sessionParams": {"s": 1},
        "parameters": {"p": 2},
        "value": "v",
        "delay": 100,
        "id": "act-1",
        "timestamp": "2024-01-01T00:00:00Z",
        "participant": "p1",
    }
    await run_with_messages(
        handler, sock,
        make_session_initiate(),
        {"type": "activities", "activities": [activity]},
    )
    received = handler.on_activity.call_args[0][0]  # type: ignore[union-attr]
    assert received["activityParams"] == {"foo": "bar"}
    assert received["participant"] == "p1"
    assert received["id"] == "act-1"


# ── connection.validate ───────────────────────────────────────────────────────

async def test_connection_validate_replies_validated() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(
        handler, sock,
        make_session_initiate(),
        {"type": "connection.validate"},
    )
    msgs = sock.sent_json()
    validated = next(m for m in msgs if m.get("type") == "connection.validated")
    assert validated["success"] is True


# ── session.end ───────────────────────────────────────────────────────────────

async def test_session_end_invokes_callback() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(
        handler, sock,
        make_session_initiate(),
        {"type": "session.end"},
    )
    handler.on_session_end.assert_called_once()  # type: ignore[union-attr]


# ── session.resume ────────────────────────────────────────────────────────────

async def test_session_resume_matching_id_sends_accepted() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(
        handler, sock,
        make_session_initiate(conversation_id="conv-123"),
        {"type": "session.resume", "conversationId": "conv-123"},
    )
    msgs = sock.sent_json()
    accepted_msgs = [m for m in msgs if m.get("type") == "session.accepted"]
    assert len(accepted_msgs) == 2  # one for initiate, one for resume


async def test_session_resume_matching_id_invokes_callback() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(
        handler, sock,
        make_session_initiate(conversation_id="conv-123"),
        {"type": "session.resume", "conversationId": "conv-123"},
    )
    handler.on_session_resumed.assert_called_once()  # type: ignore[union-attr]


async def test_session_resume_mismatched_id_sends_error() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(
        handler, sock,
        make_session_initiate(conversation_id="conv-123"),
        {"type": "session.resume", "conversationId": "conv-WRONG"},
    )
    msgs = sock.sent_json()
    err = next(m for m in msgs if m.get("type") == "session.error")
    assert "conversation not found" in err["reason"]
    handler.on_session_resumed.assert_not_called()  # type: ignore[union-attr]


# ── playStream lifecycle ──────────────────────────────────────────────────────

async def test_start_play_stream_sends_start_message() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    stream = await handler.start_play_stream()
    msgs = sock.sent_json()
    start = next(m for m in msgs if m.get("type") == "playStream.start")
    assert start["streamId"] == stream.stream_id
    assert "mediaFormat" in start


async def test_start_play_stream_with_participant() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await handler.start_play_stream(participant="caller-leg")
    start = next(m for m in sock.sent_json() if m.get("type") == "playStream.start")
    assert start.get("participant") == "caller-leg"


async def test_start_play_stream_with_alt_text() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await handler.start_play_stream(alt_text="Hello!")
    start = next(m for m in sock.sent_json() if m.get("type") == "playStream.start")
    assert start.get("altText") == "Hello!"


async def test_play_stream_write_chunk_base64_encodes() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    stream = await handler.start_play_stream()
    raw = b"\xde\xad\xbe\xef"
    await stream.write_chunk(raw)
    chunk_msg = next(m for m in sock.sent_json() if m.get("type") == "playStream.chunk")
    assert base64.b64decode(chunk_msg["audioChunk"]) == raw
    assert chunk_msg["streamId"] == stream.stream_id


async def test_play_stream_end_sends_stop() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    stream = await handler.start_play_stream()
    await stream.end()
    msgs = sock.sent_json()
    stop = next(m for m in msgs if m.get("type") == "playStream.stop")
    assert stop["streamId"] == stream.stream_id


async def test_successive_play_streams_get_monotonic_ids() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    s0 = await handler.start_play_stream()
    s1 = await handler.start_play_stream()
    s2 = await handler.start_play_stream()
    assert s0.stream_id == "play-0"
    assert s1.stream_id == "play-1"
    assert s2.stream_id == "play-2"


async def test_cancel_current_play_stream_sends_stop() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await handler.start_play_stream()
    await handler.cancel_current_play_stream()
    msgs = sock.sent_json()
    assert any(m.get("type") == "playStream.stop" for m in msgs)


# ── outbound activities ───────────────────────────────────────────────────────

async def test_send_activity_single_wraps_in_envelope() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await handler.send_activity({"type": "event", "name": "hangup"})
    msgs = sock.sent_json()
    env = next(m for m in msgs if m.get("type") == "activities")
    assert env["activities"] == [{"type": "event", "name": "hangup"}]


async def test_send_activity_list_wraps_atomically() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    acts: list[Any] = [
        {"type": "event", "name": "hangup"},
        {"type": "event", "name": "abortPrompts"},
    ]
    await handler.send_activity(acts)
    env = next(m for m in sock.sent_json() if m.get("type") == "activities")
    assert len(env["activities"]) == 2


async def test_send_transfer_serializes_correctly() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await handler.send_transfer("sip:queue@company.com", handover_reason="userRequest")
    env = next(m for m in sock.sent_json() if m.get("type") == "activities")
    act = env["activities"][0]
    assert act["name"] == "transfer"
    assert act["activityParams"]["transferTarget"] == "sip:queue@company.com"
    assert act["activityParams"]["handoverReason"] == "userRequest"


async def test_send_hangup_serializes_correctly() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await handler.send_hangup()
    env = next(m for m in sock.sent_json() if m.get("type") == "activities")
    assert env["activities"][0] == {"type": "event", "name": "hangup"}


async def test_send_play_url_serializes_correctly() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await handler.send_play_url(
        "https://example.com/file.wav",
        options={"playUrlMediaFormat": "audio/wav"},
    )
    env = next(m for m in sock.sent_json() if m.get("type") == "activities")
    act = env["activities"][0]
    assert act["name"] == "playUrl"
    assert act["activityParams"]["playUrlUrl"] == "https://example.com/file.wav"
    assert act["activityParams"]["playUrlMediaFormat"] == "audio/wav"


async def test_send_dtmf_serializes_correctly() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await handler.send_dtmf("1234#")
    env = next(m for m in sock.sent_json() if m.get("type") == "activities")
    act = env["activities"][0]
    assert act["name"] == "sendDtmf"
    assert act["activityParams"]["dtmf"] == "1234#"


async def test_send_meta_data_serializes_correctly() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await handler.send_meta_data({"foo": "bar"})
    env = next(m for m in sock.sent_json() if m.get("type") == "activities")
    act = env["activities"][0]
    assert act["name"] == "sendMetaData"
    assert act["activityParams"]["foo"] == "bar"


async def test_send_abort_prompts() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await handler.send_abort_prompts()
    env = next(m for m in sock.sent_json() if m.get("type") == "activities")
    assert env["activities"][0]["name"] == "abortPrompts"


async def test_send_expect_another_bot_message() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await handler.send_expect_another_bot_message()
    env = next(m for m in sock.sent_json() if m.get("type") == "activities")
    assert env["activities"][0]["name"] == "expectAnotherBotMessage"


async def test_send_config_serializes_params() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await handler.send_config({"language": "es-ES"})
    env = next(m for m in sock.sent_json() if m.get("type") == "activities")
    act = env["activities"][0]
    assert act["name"] == "config"
    assert act["activityParams"]["language"] == "es-ES"


@pytest.mark.parametrize("helper,expected_name", [
    ("send_start_call_recording", "startCallRecording"),
    ("send_stop_call_recording", "stopCallRecording"),
    ("send_pause_call_recording", "pauseCallRecording"),
    ("send_resume_call_recording", "resumeCallRecording"),
])
async def test_call_recording_helpers(helper: str, expected_name: str) -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await getattr(handler, helper)()
    env = next(m for m in sock.sent_json() if m.get("type") == "activities")
    assert env["activities"][0]["name"] == expected_name


# ── speech lifecycle helpers ─────────────────────────────────────────────────

async def test_send_speech_started_serializes_correctly() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await handler.send_speech_started()
    msgs = sock.sent_json()
    speech = next(m for m in msgs if m.get("type") == "userStream.speech.started")
    assert list(speech.keys()) == ["type"]


async def test_send_speech_recognition_serializes_correctly() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await handler.send_speech_recognition("Hello there", 1.0)
    msgs = sock.sent_json()
    recog = next(m for m in msgs if m.get("type") == "userStream.speech.recognition")
    assert recog["text"] == "Hello there"
    assert recog["confidence"] == 1.0
    assert set(recog.keys()) == {"type", "text", "confidence"}


async def test_send_speech_committed_serializes_correctly() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    await run_with_messages(handler, sock, make_session_initiate())
    await handler.send_speech_committed()
    msgs = sock.sent_json()
    committed = next(m for m in msgs if m.get("type") == "userStream.speech.committed")
    assert list(committed.keys()) == ["type"]


# ── malformed input ───────────────────────────────────────────────────────────

async def test_malformed_json_invokes_on_error_without_crash() -> None:
    sock = FakeSocket()
    handler = make_handler(sock)
    sock.push_text("NOT VALID JSON")
    sock.push_close()
    await handler.run()
    handler.on_error.assert_called_once()  # type: ignore[union-attr]
