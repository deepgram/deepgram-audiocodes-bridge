from __future__ import annotations

import asyncio
import copy
import logging
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

import websockets.asyncio.server
from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from .audio_router import AudioRouter
from .audiocodes_handler import AudioCodesHandler
from .deepgram_handler import DeepgramHandler, DeepgramHandshakeError
from .types import (
    AgentAudioDoneEvent,
    AgentThinkingEvent,
    AudioCodesEventActivity,
    AudioCodesMediaFormat,
    BridgeConfig,
    BridgeErrorEvent,
    CallRecordingActivityParams,
    ConversationTextEvent,
    DeepgramAgentConfig,
    DeepgramAudio,
    DeepgramAudioInput,
    DeepgramAudioOutput,
    FunctionCallRequestEvent,
    InboundActivityEvent,
    PlayUrlActivityParams,
    PromptUpdatedEvent,
    SessionEndEvent,
    SessionEndReason,
    SessionInitiateMessage,
    SessionStartEvent,
    SpeakUpdatedEvent,
    ThinkUpdatedEvent,
    UserStartedSpeakingEvent,
    WarningEvent,
)

logger = logging.getLogger(__name__)


class SessionState(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ENDING = "ending"
    ENDED = "ended"


# Narrow literal for the encodings this bridge actually produces. Both values
# are valid for DeepgramAudioInput.encoding and DeepgramAudioOutput.encoding,
# so the same literal can flow into either TypedDict without a cast.
_BridgeEncoding = Literal["linear16", "mulaw"]

# Maps AudioCodes media format → (Deepgram encoding, sample_rate)
_MEDIA_FORMAT_MAP: dict[str, tuple[_BridgeEncoding, int]] = {
    "raw/mulaw": ("mulaw", 8000),
    "wav/mulaw": ("mulaw", 8000),
    "raw/lpcm16_8": ("linear16", 8000),
    "wav/lpcm16_8": ("linear16", 8000),
    "raw/lpcm16": ("linear16", 16000),
    "wav/lpcm16": ("linear16", 16000),
    "raw/lpcm16_24": ("linear16", 24000),
    "wav/lpcm16_24": ("linear16", 24000),
}


def _inject_audio_format(
    config: DeepgramAgentConfig, media_format: AudioCodesMediaFormat
) -> None:
    """Populate ``audio.input`` and ``audio.output`` in ``config`` to match
    ``media_format``. Sets ``container: "none"`` on output so TTS bytes can
    be forwarded verbatim to AudioCodes ``playStream.chunk``.
    """
    encoding, sample_rate = _MEDIA_FORMAT_MAP.get(media_format, ("mulaw", 8000))
    audio_input: DeepgramAudioInput = {
        "encoding": encoding,
        "sample_rate": sample_rate,
    }
    audio_output: DeepgramAudioOutput = {
        "encoding": encoding,
        "sample_rate": sample_rate,
        "container": "none",
    }
    config["audio"] = DeepgramAudio(input=audio_input, output=audio_output)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ms() -> int:
    return int(time.time() * 1000)


class AsyncEventEmitter:
    """Minimal async event emitter with decorator-based handler registration.

    Handlers are stored on the bridge; each :class:`Session` reads this dict
    when emitting and invokes handlers with ``(session, event)``.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., Any]]] = {}

    def on(self, event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a handler for ``event``.

        Usage::

            @bridge.on("session_start")
            async def _(session: Session, ev: SessionStartEvent) -> None: ...
        """
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._handlers.setdefault(event, []).append(fn)
            return fn
        return decorator


class Session:
    """One AudioCodes call + its paired Deepgram Voice Agent connection.

    A ``Session`` owns all per-call state — sockets, audio router, transcript,
    lifecycle — and exposes the control methods (``send_agent_message``,
    ``transfer``, ``end_session``, ``play_url``, ...). Concurrent calls each
    get their own ``Session``, so there is no global state to collide.

    Event handlers registered on the parent :class:`DeepgramBridge` receive the
    ``Session`` as their first argument, so application code always has a
    handle to the specific call that triggered the event.

    Two ways to use it:

    - **One-liner server.** ``DeepgramBridge(config).run()`` owns the listener
      and instantiates one ``Session`` per inbound connection. Most users want
      this.

    - **Embedded.** For dropping the bridge into an existing server built on
      the ``websockets`` library, call
      ``await Session.serve(socket, bridge)`` from your WebSocket route after
      you've accepted the upgrade and done your own authentication.
    """

    def __init__(
        self, socket: ServerConnection, bridge: "DeepgramBridge"
    ) -> None:
        self._bridge = bridge
        self._socket = socket
        self._session_id = uuid.uuid4().hex
        self._state: SessionState = SessionState.INITIALIZING
        self._conversation_id: str | None = None
        self._media_format_current: AudioCodesMediaFormat | None = None
        self._transcript: list[ConversationTextEvent] = []
        self._session_start_ms: int = 0

        self._ac_handler: AudioCodesHandler = AudioCodesHandler(
            socket, bridge._config.preferred_media_formats
        )
        self._dg_handler: DeepgramHandler = DeepgramHandler(
            bridge._config.deepgram_api_key,
            bridge._config.deepgram_config,
            self._session_id,
        )
        self._audio_router: AudioRouter = AudioRouter(
            self._ac_handler, self._dg_handler
        )

    @classmethod
    async def serve(
        cls, socket: ServerConnection, bridge: "DeepgramBridge"
    ) -> "Session":
        """Drive one AudioCodes call on an already-accepted WebSocket.

        For embedding the bridge inside an external server built on the
        ``websockets`` library. Construct the :class:`DeepgramBridge` once at
        startup (it holds config and event handlers), then call this from your
        WebSocket route after you've accepted the upgrade and done your own
        authentication. Returns the completed ``Session`` when both sockets
        have closed.

        Authentication is the caller's responsibility in this mode — the
        built-in ``ac_token`` / ``authenticate`` checks on ``BridgeConfig``
        apply only to :meth:`DeepgramBridge.run`.
        """
        session = cls(socket, bridge)
        await session.run()
        return session

    async def run(self) -> None:
        """Wire the AC and Deepgram handlers together and pump them to
        completion. Returns when both underlying sockets have closed.
        """
        self._wire_callbacks()

        logger.info(
            "new AudioCodes session",
            extra={"session_id": self._session_id},
        )

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._ac_handler.run())
                tg.create_task(self._dg_handler.run_reader())
        except* Exception as eg:
            for exc in eg.exceptions:
                logger.exception(
                    "unhandled exception in session task group",
                    exc_info=exc,
                    extra={"session_id": self._session_id},
                )
        finally:
            await self._handle_session_end("ended")

    # ─── Callback wiring ──────────────────────────────────────────────────────

    def _wire_callbacks(self) -> None:
        ac = self._ac_handler
        dg = self._dg_handler
        router = self._audio_router
        session_id = self._session_id

        async def on_session_initiate(initiate_msg: SessionInitiateMessage) -> None:
            cloned: DeepgramAgentConfig = copy.deepcopy(
                self._bridge._config.deepgram_config
            )
            _inject_audio_format(cloned, ac.media_format)  # type: ignore[arg-type]
            self._media_format_current = ac.media_format
            self._conversation_id = str(initiate_msg.get("conversationId", ""))

            try:
                await dg.connect(cloned)
            except DeepgramHandshakeError as exc:
                logger.error(
                    "Deepgram handshake rejected",
                    extra={
                        "session_id": session_id,
                        "code": exc.code,
                        "description": exc.description,
                        "payload": exc.payload,
                    },
                )
                err = BridgeErrorEvent(
                    session_id=session_id,
                    description=exc.description,
                    code=exc.code,
                    cause=exc,
                    recoverable=False,
                )
                await self._emit("error", err)
                await self._handle_session_end("error")
                return
            except Exception as exc:
                logger.exception(
                    "Deepgram connect failed",
                    extra={"session_id": session_id},
                )
                err = BridgeErrorEvent(
                    session_id=session_id,
                    description=f"Deepgram connect failed: {exc}",
                    code="connect_failed",
                    cause=exc,
                    recoverable=False,
                )
                await self._emit("error", err)
                await self._handle_session_end("error")
                return

            self._state = SessionState.ACTIVE
            self._session_start_ms = _now_ms()

            event = SessionStartEvent(
                session_id=session_id,
                conversation_id=str(initiate_msg.get("conversationId", "")),
                caller=str(initiate_msg["caller"]) if "caller" in initiate_msg else None,
                bot_name=str(initiate_msg["botName"]) if "botName" in initiate_msg else None,
                media_format=ac.media_format,  # type: ignore[arg-type]
                started_at=_now_iso(),
            )
            await self._emit("session_start", event)

        async def on_session_resumed(resume_msg: dict[str, object]) -> None:
            logger.info(
                "session resumed",
                extra={
                    "session_id": session_id,
                    "conversation_id": resume_msg.get("conversationId"),
                },
            )

        async def on_user_audio(pcm_bytes: bytes, meta: Any) -> None:
            await router.forward_to_deepgram(pcm_bytes)

        async def on_activity(activity: AudioCodesEventActivity) -> None:
            await self._emit(
                "activity",
                InboundActivityEvent(session_id=session_id, activity=activity),
            )

        async def on_session_end(end_msg: dict[str, object]) -> None:
            await self._handle_session_end("hangup")

        async def on_ac_error(exc: BaseException) -> None:
            await self._emit(
                "error",
                BridgeErrorEvent(
                    session_id=session_id,
                    description=str(exc),
                    code="audiocodes_error",
                    cause=exc,
                    recoverable=True,
                ),
            )

        ac.on_session_initiate = on_session_initiate
        ac.on_session_resumed = on_session_resumed
        ac.on_user_audio = on_user_audio
        ac.on_activity = on_activity
        ac.on_session_end = on_session_end
        ac.on_error = on_ac_error

        async def on_conversation_text(event: ConversationTextEvent) -> None:
            self._transcript.append(event)
            if event.role == "user":
                await ac.send_speech_recognition(
                    text=event.content, confidence=1.0
                )
                await ac.send_speech_committed()
            await self._emit("conversation_text", event)

        async def on_user_started_speaking(event: UserStartedSpeakingEvent) -> None:
            await router.flush_audiocodes_playback()
            await ac.send_speech_started()
            await self._emit("user_started_speaking", event)

        async def on_agent_thinking(event: AgentThinkingEvent) -> None:
            await self._emit("agent_thinking", event)

        async def on_agent_audio_done(event: AgentAudioDoneEvent) -> None:
            await router.end_audiocodes_playback()
            await self._emit("agent_audio_done", event)

        async def on_function_call_request(event: FunctionCallRequestEvent) -> None:
            await self._emit("function_call_request", event)

        async def on_prompt_updated(event: PromptUpdatedEvent) -> None:
            await self._emit("prompt_updated", event)

        async def on_speak_updated(event: SpeakUpdatedEvent) -> None:
            await self._emit("speak_updated", event)

        async def on_think_updated(event: ThinkUpdatedEvent) -> None:
            await self._emit("think_updated", event)

        async def on_warning(event: WarningEvent) -> None:
            await self._emit("warning", event)

        async def on_agent_audio(chunk: bytes) -> None:
            await router.forward_to_audiocodes(chunk)

        async def on_dg_error(err: BridgeErrorEvent) -> None:
            await self._emit("error", err)
            await self._handle_session_end("error")

        async def on_dg_disconnected(reason: str) -> None:
            err = BridgeErrorEvent(
                session_id=session_id,
                description=f"Deepgram disconnected: {reason}",
                code="disconnected",
                cause=None,
                recoverable=False,
            )
            await self._emit("error", err)
            await self._handle_session_end("error")

        dg.on_conversation_text = on_conversation_text
        dg.on_user_started_speaking = on_user_started_speaking
        dg.on_agent_thinking = on_agent_thinking
        dg.on_agent_audio_done = on_agent_audio_done
        dg.on_function_call_request = on_function_call_request
        dg.on_prompt_updated = on_prompt_updated
        dg.on_speak_updated = on_speak_updated
        dg.on_think_updated = on_think_updated
        dg.on_warning = on_warning
        dg.on_agent_audio = on_agent_audio
        dg.on_error = on_dg_error
        dg.on_disconnected = on_dg_disconnected

    # ─── Event dispatch ───────────────────────────────────────────────────────

    async def _emit(self, event: str, arg: Any) -> None:
        for handler in self._bridge._handlers.get(event, []):
            try:
                result = handler(self, arg)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                await self._emit_error_from_handler(event, exc, arg)

    async def _emit_error_from_handler(
        self, event: str, exc: Exception, event_arg: Any = None
    ) -> None:
        logger.exception(
            "application handler raised",
            extra={"event": event, "session_id": self._session_id},
        )
        if event == "error":
            # Avoid recursion — log only.
            return
        error_event = BridgeErrorEvent(
            session_id=self._session_id,
            description=f"handler for '{event}' raised: {exc}",
            code="handler_error",
            cause=exc,
            recoverable=True,
        )
        # Directly invoke error handlers without going through _emit to avoid recursion.
        for handler in self._bridge._handlers.get("error", []):
            try:
                result = handler(self, error_event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "exception inside error handler",
                    extra={"session_id": self._session_id},
                )

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def _handle_session_end(self, reason: SessionEndReason) -> None:
        if self._state in (SessionState.ENDING, SessionState.ENDED):
            return
        self._state = SessionState.ENDING

        ended_at = _now_iso()
        duration_ms = _now_ms() - self._session_start_ms

        await self._emit(
            "session_end",
            SessionEndEvent(
                session_id=self._session_id,
                reason=reason,
                duration_ms=duration_ms,
                ended_at=ended_at,
            ),
        )

        try:
            await self._dg_handler.disconnect()
        except Exception:
            pass
        try:
            await self._ac_handler.close()
        except Exception:
            pass

        self._state = SessionState.ENDED
        logger.info(
            "session ended",
            extra={
                "session_id": self._session_id,
                "reason": reason,
                "duration_ms": duration_ms,
            },
        )

    def _require_active(self, method: str) -> None:
        if self._state in (SessionState.ENDING, SessionState.ENDED):
            raise RuntimeError(
                f"cannot call {method}: session is {self._state.value}"
            )

    # ─── Deepgram Voice Agent control methods ────────────────────────────────

    async def send_agent_message(self, content: str) -> None:
        """Force the agent to immediately speak ``content``.

        Wraps the Voice Agent ``InjectAgentMessage`` client message.
        """
        self._require_active("send_agent_message")
        await self._dg_handler.inject_agent_message(content)

    async def send_user_message(self, content: str) -> None:
        """Inject a text utterance as if the user had spoken it.

        Wraps the Voice Agent ``InjectUserMessage`` client message.
        """
        self._require_active("send_user_message")
        await self._dg_handler.inject_user_message(content)

    async def update_prompt(self, prompt: str) -> None:
        """Add to the agent's system prompt mid-conversation.

        Wraps the Voice Agent ``UpdatePrompt`` client message.
        """
        self._require_active("update_prompt")
        await self._dg_handler.update_prompt(prompt)

    async def update_speak(self, speak: dict[str, object]) -> None:
        """Swap the TTS provider/model mid-conversation.

        Wraps the Voice Agent ``UpdateSpeak`` client message.
        """
        self._require_active("update_speak")
        await self._dg_handler.update_speak(speak)

    async def update_think(self, think: dict[str, object]) -> None:
        """Replace the entire Think (LLM) configuration mid-conversation.

        Wraps the Voice Agent ``UpdateThink`` client message.
        """
        self._require_active("update_think")
        await self._dg_handler.update_think(think)

    async def respond_to_function_call(
        self, id: str, name: str, content: str
    ) -> None:
        """Send the result of a client-side function call back to Deepgram.

        Wraps the Voice Agent ``FunctionCallResponse`` client message.
        """
        self._require_active("respond_to_function_call")
        await self._dg_handler.send_function_call_response(id, name, content)

    # ─── AudioCodes Bot API control methods ──────────────────────────────────

    async def send_activity(
        self, activity: AudioCodesEventActivity | list[AudioCodesEventActivity]
    ) -> None:
        """Generic AudioCodes activity sender.

        Accepts a single activity dict or a list and wraps them in an
        ``activities`` envelope.

        Raises:
            RuntimeError: If the session is not in the ``Active`` state.
        """
        self._require_active("send_activity")
        await self._ac_handler.send_activity(activity)

    async def transfer(
        self,
        destination: str,
        *,
        handover_reason: str | None = None,
        transfer_sip_headers: list[dict[str, str]] | None = None,
    ) -> None:
        """Signal AudioCodes to perform a SIP REFER transfer to ``destination``.

        Emits ``session_end`` with reason ``'transfer'`` after initiating.
        """
        self._require_active("transfer")
        await self._ac_handler.send_transfer(
            destination,
            handover_reason=handover_reason,
            transfer_sip_headers=transfer_sip_headers,
        )
        await self._handle_session_end("transfer")

    async def end_session(self, reason: str | None = None) -> None:
        """Gracefully end the session.

        Sends a ``hangup`` event and closes both WebSocket connections.
        Emits ``session_end`` with reason ``'ended'``.
        """
        self._require_active("end_session")
        await self._ac_handler.send_hangup()
        await self._handle_session_end("ended")

    async def play_url(
        self, url: str, *, options: PlayUrlActivityParams | None = None
    ) -> None:
        """Play a pre-recorded audio file via VAIC's own audio engine."""
        self._require_active("play_url")
        await self._ac_handler.send_play_url(url, options=options)

    async def send_dtmf(
        self, digits: str, *, options: dict[str, object] | None = None
    ) -> None:
        """Send DTMF ``digits`` downstream via VAIC."""
        self._require_active("send_dtmf")
        await self._ac_handler.send_dtmf(digits, options=options)

    async def send_meta_data(self, data: dict[str, object]) -> None:
        """Push arbitrary metadata to VAIC."""
        self._require_active("send_meta_data")
        await self._ac_handler.send_meta_data(data)

    async def abort_prompts(self) -> None:
        """Tell VAIC to cancel any VAIC-managed prompts currently playing."""
        self._require_active("abort_prompts")
        await self._ac_handler.send_abort_prompts()

    async def expect_another_bot_message(self) -> None:
        """Tell VAIC to keep the turn open — another bot utterance is coming."""
        self._require_active("expect_another_bot_message")
        await self._ac_handler.send_expect_another_bot_message()

    async def apply_config(self, params: dict[str, object]) -> None:
        """Dynamically change session-level configuration mid-call."""
        self._require_active("apply_config")
        await self._ac_handler.send_config(params)

    # ─── Call recording ───────────────────────────────────────────────────────

    async def start_call_recording(
        self, params: CallRecordingActivityParams | None = None
    ) -> None:
        """Start call recording on the VAIC / SBC side."""
        self._require_active("start_call_recording")
        await self._ac_handler.send_start_call_recording(params)

    async def stop_call_recording(self) -> None:
        """Stop call recording."""
        self._require_active("stop_call_recording")
        await self._ac_handler.send_stop_call_recording()

    async def pause_call_recording(self) -> None:
        """Pause call recording."""
        self._require_active("pause_call_recording")
        await self._ac_handler.send_pause_call_recording()

    async def resume_call_recording(self) -> None:
        """Resume a paused call recording."""
        self._require_active("resume_call_recording")
        await self._ac_handler.send_resume_call_recording()

    # ─── Read-only state ──────────────────────────────────────────────────────

    def get_transcript(self) -> list[ConversationTextEvent]:
        """Return a copy of all ``ConversationText`` events accumulated during
        the session, in order.

        Safe to call at any point including inside a ``session_end`` handler.
        """
        return list(self._transcript)

    @property
    def session_id(self) -> str:
        """The unique session ID for this call."""
        return self._session_id

    @property
    def is_active(self) -> bool:
        """``True`` if the session is still active."""
        return self._state == SessionState.ACTIVE

    @property
    def conversation_id(self) -> str | None:
        """The AudioCodes conversation ID from ``session.initiate``."""
        return self._conversation_id

    @property
    def media_format(self) -> AudioCodesMediaFormat | None:
        """The media format negotiated in ``session.accepted``."""
        return self._media_format_current


class DeepgramBridge(AsyncEventEmitter):
    """WebSocket server + event handler registry.

    Holds configuration and application-level event handlers; instantiates one
    :class:`Session` per inbound AudioCodes connection. The per-call state and
    control methods (``send_agent_message``, ``transfer``, ``end_session``, ...)
    live on the ``Session``, which is passed as the first argument to every
    registered event handler.

    Usage (one-liner server)::

        bridge = DeepgramBridge(BridgeConfig(...))

        @bridge.on("session_start")
        async def on_start(session: Session, event: SessionStartEvent) -> None:
            print(event.session_id)

        asyncio.run(bridge.run())

    Usage (embedded in an existing ``websockets``-based server)::

        bridge = DeepgramBridge(BridgeConfig(...))

        @bridge.on("session_start")
        async def on_start(session: Session, event: SessionStartEvent) -> None:
            ...

        # inside your framework's WebSocket route, after auth + upgrade:
        await Session.serve(socket, bridge)

    Public events (register with ``@bridge.on("<name>")``):

    - ``"session_start"`` → :class:`SessionStartEvent`
    - ``"session_end"`` → :class:`SessionEndEvent`
    - ``"conversation_text"`` → :class:`ConversationTextEvent`
    - ``"user_started_speaking"`` → :class:`UserStartedSpeakingEvent`
    - ``"agent_thinking"`` → :class:`AgentThinkingEvent`
    - ``"agent_audio_done"`` → :class:`AgentAudioDoneEvent`
    - ``"function_call_request"`` → :class:`FunctionCallRequestEvent`
    - ``"prompt_updated"`` → :class:`PromptUpdatedEvent`
    - ``"speak_updated"`` → :class:`SpeakUpdatedEvent`
    - ``"think_updated"`` → :class:`ThinkUpdatedEvent`
    - ``"warning"`` → :class:`WarningEvent`
    - ``"activity"`` → :class:`InboundActivityEvent`
    - ``"error"`` → :class:`BridgeErrorEvent`
    """

    def __init__(self, config: BridgeConfig) -> None:
        super().__init__()
        self._config = config

    async def run(self) -> None:
        """Start the AudioCodes Bot API WebSocket server and accept connections.

        Resolves once the server is listening, then awaits forever — returns
        only when the server is cancelled or the process exits. Each inbound
        connection is handled by its own :class:`Session` running concurrently
        on the event loop.
        """
        server = await websockets.asyncio.server.serve(
            self._handle_connection,
            host=self._config.host,
            port=self._config.port,
            ping_interval=30,
            ping_timeout=30,
            process_request=self._verify_authorization,
        )
        logger.info(
            "AudioCodes Bridge server listening",
            extra={"host": self._config.host, "port": self._config.port},
        )
        await server.serve_forever()

    async def _verify_authorization(
        self, connection: ServerConnection, request: Request
    ) -> Response | None:
        # Built-in modes cover AudioCodes "No Auth" (ac_token=None) and
        # "Permanent Token". For OAuth 2.0 or any other scheme (DB-backed
        # tokens, IP allowlists, etc.) pass a custom ``authenticate`` callback
        # on BridgeConfig — when set it fully owns the accept/reject decision
        # and the built-in ac_token check is skipped.
        if self._config.authenticate is not None:
            return await self._config.authenticate(connection, request)
        if self._config.ac_token is None:
            return None
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {self._config.ac_token}":
            logger.warning(
                "rejected unauthorized AudioCodes upgrade",
                extra={"remote": getattr(connection, "remote_address", None)},
            )
            return Response(401, "Unauthorized", Headers([]), b"Unauthorized\n")
        return None

    async def _handle_connection(self, socket: ServerConnection) -> Session:
        session = Session(socket, self)
        await session.run()
        return session
