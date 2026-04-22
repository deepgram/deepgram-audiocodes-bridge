from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from collections.abc import Awaitable, Callable

import websockets
import websockets.asyncio.client
import websockets.exceptions

from .types import (
    AgentAudioDoneEvent,
    AgentThinkingEvent,
    BridgeErrorEvent,
    ConversationTextEvent,
    DeepgramAgentConfig,
    FunctionCall,
    FunctionCallRequestEvent,
    PromptUpdatedEvent,
    SpeakUpdatedEvent,
    ThinkUpdatedEvent,
    UserStartedSpeakingEvent,
    WarningEvent,
)

logger = logging.getLogger(__name__)

_DEEPGRAM_AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"
_KEEPALIVE_INTERVAL = 8.0


class DeepgramHandshakeError(Exception):
    """Raised when the Voice Agent handshake fails.

    Carries the structured fields needed to build a ``BridgeErrorEvent``
    that preserves Deepgram's own error code and message, rather than
    collapsing everything into a generic ``connect_failed`` string.

    Attributes:
        code: Deepgram's ``err_code`` if the server sent an ``Error`` frame,
            otherwise a bridge-defined code (e.g. ``handshake_unexpected_type``,
            ``handshake_binary_frame``).
        description: Human-readable description. Contains Deepgram's
            ``err_msg`` when available, otherwise describes the protocol
            violation we observed.
        payload: The raw JSON message Deepgram sent, if any. ``None`` for
            binary-frame or parse errors.
    """

    def __init__(
        self,
        code: str,
        description: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        super().__init__(description)
        self.code = code
        self.description = description
        self.payload = payload

# Internal callback types — not exposed to application code.
# Names mirror Voice Agent V1 server message types in snake_case.
_OnConversationText = Callable[[ConversationTextEvent], Awaitable[None]]
_OnUserStartedSpeaking = Callable[[UserStartedSpeakingEvent], Awaitable[None]]
_OnAgentThinking = Callable[[AgentThinkingEvent], Awaitable[None]]
_OnAgentAudioDone = Callable[[AgentAudioDoneEvent], Awaitable[None]]
_OnFunctionCallRequest = Callable[[FunctionCallRequestEvent], Awaitable[None]]
_OnPromptUpdated = Callable[[PromptUpdatedEvent], Awaitable[None]]
_OnSpeakUpdated = Callable[[SpeakUpdatedEvent], Awaitable[None]]
_OnThinkUpdated = Callable[[ThinkUpdatedEvent], Awaitable[None]]
_OnWarning = Callable[[WarningEvent], Awaitable[None]]
_OnError = Callable[[BridgeErrorEvent], Awaitable[None]]
_OnAgentAudio = Callable[[bytes], Awaitable[None]]
_OnConnected = Callable[[], Awaitable[None]]
_OnDisconnected = Callable[[str], Awaitable[None]]


class DeepgramHandler:
    """Deepgram Voice Agent V1 WebSocket handler.

    Args:
        api_key: Your Deepgram API key. Sent as ``Authorization: Token <key>``
            on the HTTP upgrade request.
        config: Deepgram Voice Agent configuration. The bridge has already
            injected ``audio.input`` / ``audio.output`` based on the
            negotiated AudioCodes media format before calling
            :meth:`connect`.
        session_id: The bridge-assigned session ID for this call.
    """

    # Callbacks set by DeepgramBridge after construction.
    on_conversation_text: _OnConversationText | None
    on_user_started_speaking: _OnUserStartedSpeaking | None
    on_agent_thinking: _OnAgentThinking | None
    on_agent_audio_done: _OnAgentAudioDone | None
    on_function_call_request: _OnFunctionCallRequest | None
    on_prompt_updated: _OnPromptUpdated | None
    on_speak_updated: _OnSpeakUpdated | None
    on_think_updated: _OnThinkUpdated | None
    on_warning: _OnWarning | None
    on_error: _OnError | None
    on_agent_audio: _OnAgentAudio | None
    on_connected: _OnConnected | None
    on_disconnected: _OnDisconnected | None

    def __init__(
        self,
        api_key: str,
        config: DeepgramAgentConfig,
        session_id: str,
    ) -> None:
        self._api_key = api_key
        self._config: DeepgramAgentConfig = copy.deepcopy(config)
        self._session_id = session_id
        self._socket: websockets.asyncio.client.ClientConnection | None = None
        self._connected = False
        self._last_audio_time: float = 0.0
        # Signals run_reader() that connect() has completed.
        self._connected_event = asyncio.Event()

        self.on_conversation_text = None
        self.on_user_started_speaking = None
        self.on_agent_thinking = None
        self.on_agent_audio_done = None
        self.on_function_call_request = None
        self.on_prompt_updated = None
        self.on_speak_updated = None
        self.on_think_updated = None
        self.on_warning = None
        self.on_error = None
        self.on_agent_audio = None
        self.on_connected = None
        self.on_disconnected = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self, config: DeepgramAgentConfig | None = None) -> None:
        """Open the socket and run the handshake.

        Args:
            config: If provided, replaces the config supplied at construction
                time. The bridge uses this to inject the negotiated
                ``audio.input`` / ``audio.output`` fields after learning the
                AudioCodes media format.

        Raises:
            DeepgramHandshakeError: If Deepgram rejects the handshake with
                an ``Error`` frame, or sends an unexpected frame type. The
                exception carries Deepgram's own ``err_code`` / ``err_msg``
                when available, so the caller can emit a
                :class:`BridgeErrorEvent` with the real server-side code.
        """
        if config is not None:
            self._config = config

        socket = await websockets.asyncio.client.connect(
            _DEEPGRAM_AGENT_URL,
            additional_headers={"Authorization": f"Token {self._api_key}"},
            compression=None,
        )
        self._socket = socket

        # Step 2: wait for Welcome — server speaks first
        first = await socket.recv()
        if isinstance(first, bytes):
            raise DeepgramHandshakeError(
                code="handshake_binary_frame",
                description="expected Welcome text frame, got binary",
            )
        welcome = json.loads(first)
        if welcome.get("type") != "Welcome":
            raise self._handshake_error("Welcome", welcome)

        # Step 3: send Settings
        settings: dict[str, object] = copy.deepcopy(self._config)  # type: ignore[assignment]
        settings["type"] = "Settings"
        await socket.send(json.dumps(settings))

        # Step 4: wait for SettingsApplied
        second = await socket.recv()
        if isinstance(second, bytes):
            raise DeepgramHandshakeError(
                code="handshake_binary_frame",
                description="expected SettingsApplied text frame, got binary",
            )
        applied = json.loads(second)
        if applied.get("type") != "SettingsApplied":
            raise self._handshake_error("SettingsApplied", applied)

        self._connected = True
        self._connected_event.set()
        logger.info(
            "Deepgram Voice Agent connected",
            extra={"session_id": self._session_id},
        )
        if self.on_connected:
            await self.on_connected()

    async def run_reader(self) -> None:
        """Read inbound frames until the socket closes.

        Waits for :meth:`connect` to complete before entering the read loop.
        Should be awaited inside an :class:`asyncio.TaskGroup` alongside the
        AudioCodes handler task.
        """
        await self._connected_event.wait()
        if not self._socket:
            return

        keepalive_task: asyncio.Task[None] | None = None
        try:
            keepalive_task = asyncio.create_task(self._keepalive_loop())
            await self._read_loop()
        except websockets.exceptions.ConnectionClosedError as exc:
            self._connected = False
            err = BridgeErrorEvent(
                session_id=self._session_id,
                description=f"Deepgram connection closed: {exc}",
                code="connection_closed",
                cause=exc,
                recoverable=False,
            )
            if self.on_error:
                await self.on_error(err)
            if self.on_disconnected:
                await self.on_disconnected(str(exc))
        except Exception as exc:
            self._connected = False
            err = BridgeErrorEvent(
                session_id=self._session_id,
                description=str(exc),
                code="reader_error",
                cause=exc,
                recoverable=False,
            )
            if self.on_error:
                await self.on_error(err)
            if self.on_disconnected:
                await self.on_disconnected(str(exc))
        finally:
            self._connected = False
            if keepalive_task and not keepalive_task.done():
                keepalive_task.cancel()
                try:
                    await keepalive_task
                except asyncio.CancelledError:
                    pass

    async def _read_loop(self) -> None:
        assert self._socket is not None
        async for frame in self._socket:
            if isinstance(frame, bytes):
                if self.on_agent_audio:
                    await self.on_agent_audio(frame)
            else:
                await self._dispatch_text(frame)

    async def _keepalive_loop(self) -> None:
        while self._connected and self._socket:
            await asyncio.sleep(_KEEPALIVE_INTERVAL)
            if time.monotonic() - self._last_audio_time >= _KEEPALIVE_INTERVAL:
                try:
                    await self.send_keep_alive()
                except Exception:
                    pass

    async def _dispatch_text(self, raw: str) -> None:
        try:
            msg: dict[str, object] = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "malformed JSON from Deepgram",
                exc_info=exc,
                extra={"session_id": self._session_id},
            )
            return

        msg_type = msg.get("type")
        match msg_type:
            case "ConversationText":
                if self.on_conversation_text:
                    event = ConversationTextEvent(
                        session_id=self._session_id,
                        role=msg.get("role", "user"),  # type: ignore[arg-type]
                        content=str(msg.get("content", "")),
                    )
                    await self.on_conversation_text(event)

            case "UserStartedSpeaking":
                if self.on_user_started_speaking:
                    await self.on_user_started_speaking(
                        UserStartedSpeakingEvent(session_id=self._session_id)
                    )

            case "AgentThinking":
                if self.on_agent_thinking:
                    await self.on_agent_thinking(
                        AgentThinkingEvent(
                            session_id=self._session_id,
                            content=str(msg.get("content", "")),
                        )
                    )

            case "AgentAudioDone":
                if self.on_agent_audio_done:
                    await self.on_agent_audio_done(
                        AgentAudioDoneEvent(session_id=self._session_id)
                    )

            case "FunctionCallRequest":
                if self.on_function_call_request:
                    funcs_raw = msg.get("functions", [])
                    functions = tuple(
                        FunctionCall(
                            id=str(f.get("id", "")),
                            name=str(f.get("name", "")),
                            arguments=str(f.get("arguments", "{}")),
                            client_side=bool(f.get("client_side", True)),
                        )
                        for f in (funcs_raw if isinstance(funcs_raw, list) else [])
                    )
                    await self.on_function_call_request(
                        FunctionCallRequestEvent(
                            session_id=self._session_id,
                            functions=functions,
                        )
                    )

            case "PromptUpdated":
                if self.on_prompt_updated:
                    await self.on_prompt_updated(PromptUpdatedEvent(session_id=self._session_id))

            case "SpeakUpdated":
                if self.on_speak_updated:
                    await self.on_speak_updated(SpeakUpdatedEvent(session_id=self._session_id))

            case "ThinkUpdated":
                if self.on_think_updated:
                    await self.on_think_updated(ThinkUpdatedEvent(session_id=self._session_id))

            case "Warning":
                if self.on_warning:
                    await self.on_warning(
                        WarningEvent(
                            session_id=self._session_id,
                            code=str(msg.get("warn_code", msg.get("code", ""))),
                            description=str(msg.get("warn_msg", msg.get("description", ""))),
                        )
                    )

            case "Error":
                err = BridgeErrorEvent(
                    session_id=self._session_id,
                    description=str(msg.get("err_msg", msg.get("description", "unknown error"))),
                    code=str(msg.get("err_code", msg.get("code", "unknown"))),
                    cause=None,
                    recoverable=False,
                )
                if self.on_error:
                    await self.on_error(err)

            case _:
                logger.debug(
                    "unhandled Deepgram message type",
                    extra={"msg_type": msg_type, "session_id": self._session_id},
                )

    async def send_audio(self, chunk: bytes) -> None:
        """Send a raw binary audio frame.

        Must only be called after :meth:`connect` returns.
        """
        if self._socket and self._connected:
            self._last_audio_time = time.monotonic()
            await self._socket.send(chunk)

    async def inject_agent_message(self, content: str) -> None:
        """Wraps Voice Agent ``InjectAgentMessage``."""
        await self._send_json({"type": "InjectAgentMessage", "content": content})

    async def inject_user_message(self, content: str) -> None:
        """Wraps Voice Agent ``InjectUserMessage``."""
        await self._send_json({"type": "InjectUserMessage", "content": content})

    async def update_prompt(self, prompt: str) -> None:
        """Wraps Voice Agent ``UpdatePrompt``."""
        await self._send_json({"type": "UpdatePrompt", "prompt": prompt})

    async def update_speak(self, speak: dict[str, object]) -> None:
        """Wraps Voice Agent ``UpdateSpeak``."""
        await self._send_json({"type": "UpdateSpeak", "speak": speak})

    async def update_think(self, think: dict[str, object]) -> None:
        """Wraps Voice Agent ``UpdateThink``."""
        await self._send_json({"type": "UpdateThink", "think": think})

    async def send_function_call_response(
        self, id: str, name: str, content: str
    ) -> None:
        """Wraps Voice Agent ``FunctionCallResponse``."""
        await self._send_json(
            {"type": "FunctionCallResponse", "id": id, "name": name, "content": content}
        )

    async def send_keep_alive(self) -> None:
        """Wraps Voice Agent ``KeepAlive``."""
        await self._send_json({"type": "KeepAlive"})

    async def disconnect(self) -> None:
        """Close the Deepgram WebSocket cleanly."""
        self._connected = False
        if self._socket:
            try:
                await self._socket.close()
            except Exception:
                pass
            self._socket = None
        # Unblock run_reader() in case connect() never completed.
        self._connected_event.set()

    async def _send_json(self, payload: dict[str, object]) -> None:
        if self._socket and self._connected:
            await self._socket.send(json.dumps(payload))

    @staticmethod
    def _handshake_error(
        expected: str, msg: dict[str, object]
    ) -> DeepgramHandshakeError:
        """Build a structured exception for an unexpected handshake frame.

        When Deepgram rejects a handshake it sends an ``Error`` frame with
        ``err_code`` / ``err_msg`` (or ``code`` / ``description``) fields.
        Those are preserved verbatim on the exception so the bridge can
        forward them into a :class:`BridgeErrorEvent` without string
        round-tripping.
        """
        msg_type = msg.get("type")
        if msg_type == "Error":
            code = str(msg.get("err_code") or msg.get("code") or "unknown")
            description = str(
                msg.get("err_msg")
                or msg.get("description")
                or msg.get("message")
                or "no description"
            )
            return DeepgramHandshakeError(
                code=code,
                description=(
                    f"Deepgram rejected handshake at {expected} step: {description}"
                ),
                payload=msg,
            )
        return DeepgramHandshakeError(
            code="handshake_unexpected_type",
            description=f"expected {expected}, got {msg_type!r}",
            payload=msg,
        )
