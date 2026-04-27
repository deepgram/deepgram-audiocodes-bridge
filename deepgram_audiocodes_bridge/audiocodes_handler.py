from __future__ import annotations

import base64
import json
import logging
from collections.abc import Awaitable, Callable
from collections.abc import Sequence
from typing import TypedDict, cast

from websockets.asyncio.server import ServerConnection

from .types import (
    AudioCodesEventActivity,
    AudioCodesMediaFormat,
    CallRecordingActivityParams,
    PlayUrlActivityParams,
    SessionInitiateMessage,
)

logger = logging.getLogger(__name__)

# Internal callback types — not exposed to application code
_OnSessionInitiate = Callable[[SessionInitiateMessage], Awaitable[None]]
_OnSessionResumed = Callable[[dict[str, object]], Awaitable[None]]
_OnUserAudio = Callable[[bytes, "UserAudioChunkMeta"], Awaitable[None]]
_OnActivity = Callable[[AudioCodesEventActivity], Awaitable[None]]
_OnSessionEnd = Callable[[dict[str, object]], Awaitable[None]]
_OnError = Callable[[Exception], Awaitable[None]]


class UserAudioChunkMeta(TypedDict, total=False):
    """Metadata surfaced alongside each decoded user-audio chunk.

    Attributes:
        participant: The ``participant`` field from the most recent
            ``userStream.start`` if present.
    """
    participant: str


class PlayStream:
    """Writer returned by :meth:`AudioCodesHandler.start_play_stream`.

    Attributes:
        stream_id: The monotonically-assigned ``streamId`` (e.g. ``"play-0"``).
    """

    def __init__(
        self,
        stream_id: str,
        socket: ServerConnection,
        media_format: AudioCodesMediaFormat,
    ) -> None:
        self.stream_id = stream_id
        self._socket = socket
        self._media_format = media_format
        self._ended = False

    async def write_chunk(self, audio_bytes: bytes) -> None:
        """Send one ``playStream.chunk`` envelope with base64-encoded bytes."""
        if self._ended:
            return
        msg = {
            "type": "playStream.chunk",
            "streamId": self.stream_id,
            "audioChunk": base64.b64encode(audio_bytes).decode(),
        }
        await self._socket.send(json.dumps(msg))

    async def end(self) -> None:
        """Send ``playStream.stop`` for this stream."""
        if self._ended:
            return
        self._ended = True
        msg = {"type": "playStream.stop", "streamId": self.stream_id}
        await self._socket.send(json.dumps(msg))


class AudioCodesHandler:
    """AudioCodes Bot API protocol handler for one call.

    Args:
        socket: The ``websockets`` server connection. Authentication has
            already been validated in the bridge's ``process_request``
            callback before this class is constructed.
        preferred_media_formats: Ordered preference list. The handler picks
            the first entry that also appears in the client's
            ``supportedMediaFormats`` during ``session.initiate`` negotiation.
    """

    # Callbacks set by DeepgramBridge after construction.
    on_session_initiate: _OnSessionInitiate | None
    on_session_resumed: _OnSessionResumed | None
    on_user_audio: _OnUserAudio | None
    on_activity: _OnActivity | None
    on_session_end: _OnSessionEnd | None
    on_error: _OnError | None

    def __init__(
        self,
        socket: ServerConnection,
        preferred_media_formats: tuple[AudioCodesMediaFormat, ...],
    ) -> None:
        self._socket = socket
        self._preferred_media_formats = preferred_media_formats
        self._conversation_id: str | None = None
        self._caller: str | None = None
        self._bot_name: str | None = None
        self._media_format: AudioCodesMediaFormat | None = None
        self._expect_audio_messages: bool = True
        self._stream_counter = -1
        self._current_play_stream: PlayStream | None = None
        self._current_participant: str | None = None

        self.on_session_initiate = None
        self.on_session_resumed = None
        self.on_user_audio = None
        self.on_activity = None
        self.on_session_end = None
        self.on_error = None

    @property
    def conversation_id(self) -> str | None:
        return self._conversation_id

    @property
    def caller(self) -> str | None:
        return self._caller

    @property
    def bot_name(self) -> str | None:
        return self._bot_name

    @property
    def media_format(self) -> AudioCodesMediaFormat | None:
        return self._media_format

    @property
    def expect_audio_messages(self) -> bool:
        return self._expect_audio_messages

    async def run(self) -> None:
        """Read and dispatch inbound messages until the socket closes."""
        try:
            async for raw in self._socket:
                if isinstance(raw, bytes):
                    logger.warning("unexpected binary frame from AudioCodes")
                    continue
                try:
                    msg: dict[str, object] = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "malformed JSON from AudioCodes",
                        exc_info=exc,
                        extra={"conversation_id": self._conversation_id},
                    )
                    if self.on_error:
                        await self.on_error(exc)
                    continue

                msg_type = msg.get("type")
                try:
                    await self._dispatch(msg_type, msg)
                except Exception as exc:
                    logger.exception(
                        "error dispatching AudioCodes message",
                        extra={"msg_type": msg_type, "conversation_id": self._conversation_id},
                    )
                    if self.on_error:
                        await self.on_error(exc)
        except Exception as exc:
            logger.exception(
                "AudioCodes socket error",
                extra={"conversation_id": self._conversation_id},
            )
            if self.on_error:
                await self.on_error(exc)

    async def _dispatch(self, msg_type: object, msg: dict[str, object]) -> None:
        match msg_type:
            case "session.initiate":
                await self._handle_session_initiate(msg)  # type: ignore[arg-type]
            case "session.resume":
                await self._handle_session_resume(msg)
            case "session.end":
                if self.on_session_end:
                    await self.on_session_end(msg)
            case "connection.validate":
                await self._socket.send(
                    json.dumps({"type": "connection.validated", "success": True})
                )
            case "userStream.start":
                participant = msg.get("participant")
                self._current_participant = str(participant) if participant is not None else None
                await self._socket.send(json.dumps({"type": "userStream.started"}))
            case "userStream.chunk":
                audio_b64 = str(msg.get("audioChunk", ""))
                audio_bytes = base64.b64decode(audio_b64)
                meta: UserAudioChunkMeta = {}
                if self._current_participant is not None:
                    meta["participant"] = self._current_participant
                if self.on_user_audio:
                    await self.on_user_audio(audio_bytes, meta)
            case "userStream.stop":
                await self._socket.send(json.dumps({"type": "userStream.stopped"}))
            case "activities":
                activities = msg.get("activities", [])
                if isinstance(activities, list) and self.on_activity:
                    for activity in activities:
                        await self.on_activity(activity)
            case _:
                logger.debug(
                    "unhandled AudioCodes message type",
                    extra={"msg_type": msg_type},
                )

    async def _handle_session_initiate(self, msg: SessionInitiateMessage) -> None:
        expect_audio = msg.get("expectAudioMessages")
        if expect_audio is False:
            reason = (
                "expectAudioMessages must be true — this SDK bridges streaming audio to "
                "the Deepgram Voice Agent API and is incompatible with VAIC-side TTS mode"
            )
            await self._socket.send(json.dumps({"type": "session.error", "reason": reason}))
            await self._socket.close()
            if self.on_session_end:
                await self.on_session_end(cast(dict[str, object], msg))
            return

        supported: Sequence[object] = ()
        raw_supported = msg.get("supportedMediaFormats")
        if isinstance(raw_supported, list):
            supported = raw_supported

        chosen: AudioCodesMediaFormat | None = None
        for fmt in self._preferred_media_formats:
            if fmt in supported:
                chosen = fmt
                break

        if chosen is None:
            await self._socket.send(
                json.dumps({"type": "session.error", "reason": "No common media format"})
            )
            await self._socket.close()
            if self.on_session_end:
                await self.on_session_end(cast(dict[str, object], msg))
            return

        self._conversation_id = str(msg.get("conversationId", ""))
        caller_raw = msg.get("caller")
        self._caller = str(caller_raw) if caller_raw is not None else None
        bot_name_raw = msg.get("botName")
        self._bot_name = str(bot_name_raw) if bot_name_raw is not None else None
        self._media_format = chosen
        self._expect_audio_messages = True

        await self._socket.send(
            json.dumps({"type": "session.accepted", "mediaFormat": chosen})
        )

        if self.on_session_initiate:
            await self.on_session_initiate(msg)

    async def _handle_session_resume(self, msg: dict[str, object]) -> None:
        resume_id = str(msg.get("conversationId", ""))
        if resume_id != self._conversation_id:
            await self._socket.send(
                json.dumps({"type": "session.error", "reason": "conversation not found"})
            )
            await self._socket.close()
            return

        if self._media_format:
            await self._socket.send(
                json.dumps({"type": "session.accepted", "mediaFormat": self._media_format})
            )

        if self.on_session_resumed:
            await self.on_session_resumed(msg)

    # ─── playStream lifecycle ────────────────────────────────────────────────

    async def start_play_stream(
        self,
        *,
        participant: str | None = None,
        alt_text: str | None = None,
    ) -> PlayStream:
        """Begin a new TTS playback stream.

        Returns:
            A :class:`PlayStream` with ``stream_id``, ``write_chunk``, and
            ``end`` methods.
        """
        self._stream_counter += 1
        stream_id = f"play-{self._stream_counter}"

        start_msg: dict[str, object] = {
            "type": "playStream.start",
            "streamId": stream_id,
            "mediaFormat": self._media_format,
        }
        if participant is not None:
            start_msg["participant"] = participant
        if alt_text is not None:
            start_msg["altText"] = alt_text

        await self._socket.send(json.dumps(start_msg))

        stream = PlayStream(stream_id, self._socket, self._media_format)  # type: ignore[arg-type]
        self._current_play_stream = stream
        return stream

    async def cancel_current_play_stream(self) -> None:
        """Immediately stop the currently-active play stream (if any).

        Used for barge-in — the bridge calls this when Voice Agent emits
        ``UserStartedSpeaking``.
        """
        if self._current_play_stream is not None:
            stream = self._current_play_stream
            self._current_play_stream = None
            await stream.end()

    # ─── Outbound activity primitive ─────────────────────────────────────────

    async def send_activity(
        self, activity: AudioCodesEventActivity | list[AudioCodesEventActivity]
    ) -> None:
        """Wrap one or more activities in an ``activities`` envelope and send."""
        activities = activity if isinstance(activity, list) else [activity]
        msg = {"type": "activities", "activities": activities}
        await self._socket.send(json.dumps(msg))

    # ─── Typed activity helpers ──────────────────────────────────────────────

    async def send_transfer(
        self,
        target: str,
        *,
        handover_reason: str | None = None,
        transfer_sip_headers: list[dict[str, str]] | None = None,
    ) -> None:
        """Send an ``activities`` message with a single ``transfer`` event."""
        params: dict[str, object] = {"transferTarget": target}
        if handover_reason is not None:
            params["handoverReason"] = handover_reason
        if transfer_sip_headers is not None:
            params["transferSipHeaders"] = transfer_sip_headers
        await self.send_activity({"type": "event", "name": "transfer", "activityParams": params})

    async def send_hangup(self) -> None:
        """Send a ``hangup`` event."""
        await self.send_activity({"type": "event", "name": "hangup"})

    async def send_play_url(
        self,
        url: str,
        *,
        options: PlayUrlActivityParams | None = None,
    ) -> None:
        """Send a ``playUrl`` event to play a pre-recorded audio file via VAIC."""
        params: dict[str, object] = {"playUrlUrl": url}
        if options:
            params.update(options)
        await self.send_activity({"type": "event", "name": "playUrl", "activityParams": params})

    async def send_dtmf(
        self,
        digits: str,
        *,
        options: dict[str, object] | None = None,
    ) -> None:
        """Send a ``sendDtmf`` event to emit DTMF digits downstream."""
        params: dict[str, object] = {"dtmf": digits}
        if options:
            params.update(options)
        await self.send_activity({"type": "event", "name": "sendDtmf", "activityParams": params})

    async def send_meta_data(self, data: dict[str, object]) -> None:
        """Send arbitrary metadata via ``sendMetaData``."""
        await self.send_activity({"type": "event", "name": "sendMetaData", "activityParams": data})

    async def send_abort_prompts(self) -> None:
        """Cancel any VAIC-managed prompts currently playing."""
        await self.send_activity({"type": "event", "name": "abortPrompts"})

    async def send_expect_another_bot_message(self) -> None:
        """Tell VAIC to keep the turn open — another bot utterance is coming."""
        await self.send_activity({"type": "event", "name": "expectAnotherBotMessage"})

    async def send_config(self, params: dict[str, object]) -> None:
        """Dynamically change session-level configuration."""
        await self.send_activity({"type": "event", "name": "config", "activityParams": params})

    async def send_start_call_recording(
        self, params: CallRecordingActivityParams | None = None
    ) -> None:
        """Start call recording on the VAIC / SBC side."""
        activity: dict[str, object] = {"type": "event", "name": "startCallRecording"}
        if params:
            activity["activityParams"] = dict(params)
        await self.send_activity(activity)  # type: ignore[arg-type]

    async def send_stop_call_recording(self) -> None:
        """Stop call recording."""
        await self.send_activity({"type": "event", "name": "stopCallRecording"})

    async def send_pause_call_recording(self) -> None:
        """Pause call recording."""
        await self.send_activity({"type": "event", "name": "pauseCallRecording"})

    async def send_resume_call_recording(self) -> None:
        """Resume a paused call recording."""
        await self.send_activity({"type": "event", "name": "resumeCallRecording"})

    # ─── userStream.speech.* helpers ─────────────────────────────────────────

    async def send_speech_started(self) -> None:
        """Send ``userStream.speech.started``.

        Invoked by ``DeepgramBridge`` when Deepgram emits
        ``UserStartedSpeaking``.
        """
        await self._socket.send(json.dumps({"type": "userStream.speech.started"}))

    async def send_speech_recognition(self, text: str, confidence: float) -> None:
        """Send ``userStream.speech.recognition`` with the final ASR text.

        Invoked by ``DeepgramBridge`` when Deepgram emits ``ConversationText``
        with ``role: "user"``. Uses ``confidence=1.0`` because the Voice Agent
        ``ConversationText`` event does not carry per-word confidence.
        """
        await self._socket.send(
            json.dumps({
                "type": "userStream.speech.recognition",
                "text": text,
                "confidence": confidence,
            })
        )

    async def send_speech_committed(self) -> None:
        """Send ``userStream.speech.committed``.

        Invoked by ``DeepgramBridge`` immediately after
        :meth:`send_speech_recognition`.
        """
        await self._socket.send(json.dumps({"type": "userStream.speech.committed"}))

    async def close(self) -> None:
        """Close the underlying WebSocket cleanly."""
        try:
            await self._socket.close()
        except Exception:
            pass
