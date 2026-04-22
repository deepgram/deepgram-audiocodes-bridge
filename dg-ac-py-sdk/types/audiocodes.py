from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from .deepgram import DeepgramAgentConfig

# ─── Bridge configuration ─────────────────────────────────────────────────────

AudioCodesMediaFormat = Literal[
    "raw/mulaw",
    "wav/mulaw",
    "raw/lpcm16",
    "wav/lpcm16",
    "raw/lpcm16_8",
    "wav/lpcm16_8",
    "raw/lpcm16_24",
    "wav/lpcm16_24",
]


@dataclass(frozen=True, slots=True)
class BridgeConfig:
    """Top-level bridge configuration.

    Attributes:
        deepgram_api_key: Your Deepgram API key.
        deepgram_config: Deepgram Voice Agent configuration.
        ac_token: AudioCodes Bot API Permanent Token. Must match the token
            configured on the LiveHub / VAIC Bot Connection. Pass ``None`` to
            disable authentication (AudioCodes "No Auth" mode) — intended for
            local development only.
        host: Host the WebSocket server listens on. Defaults to ``'0.0.0.0'``.
        port: Port the WebSocket server listens on. Defaults to ``8081``.
        preferred_media_formats: Ordered preference list of AudioCodes media
            formats the bridge is willing to accept.
    """
    deepgram_api_key: str
    deepgram_config: DeepgramAgentConfig
    ac_token: str | None
    host: str = "0.0.0.0"
    port: int = 8081
    preferred_media_formats: tuple[AudioCodesMediaFormat, ...] = (
        "raw/mulaw",
        "raw/lpcm16_8",
        "raw/lpcm16",
    )


# ─── AudioCodes Bot API — Activity wire types ─────────────────────────────────

AudioCodesEventName = Literal[
    "start",
    "hangup",
    "transfer",
    "config",
    "playUrl",
    "playFinished",
    "dtmf",
    "sendDtmf",
    "sendMetaData",
    "abortPrompts",
    "expectAnotherBotMessage",
    "startCallRecording",
    "stopCallRecording",
    "pauseCallRecording",
    "resumeCallRecording",
]


class AudioCodesBaseActivity(TypedDict, total=False):
    """Fields present on every ``Activity``, regardless of type.

    All keys are the on-wire camelCase names — this dict is serialized
    verbatim to JSON.
    """
    activityParams: dict[str, object]
    sessionParams: dict[str, object]
    parameters: dict[str, object]
    value: object
    delay: int
    id: str
    timestamp: str
    participant: str


class AudioCodesMessageActivity(AudioCodesBaseActivity, total=False):
    """A text activity — carries ``text``."""
    type: Literal["message"]
    text: str


class AudioCodesEventActivity(AudioCodesBaseActivity, total=False):
    """A control event — carries a ``name``."""
    type: Literal["event"]
    name: str


AudioCodesActivity = AudioCodesMessageActivity | AudioCodesEventActivity


# ─── Inbound session messages ─────────────────────────────────────────────────

class SessionInitiateMessage(TypedDict, total=False):
    """AudioCodes ``session.initiate`` message.

    All keys are the on-wire camelCase names — this dict is received verbatim
    from JSON.
    """
    type: Literal["session.initiate"]
    conversationId: str
    expectAudioMessages: NotRequired[bool]
    supportedMediaFormats: NotRequired[list[AudioCodesMediaFormat]]
    caller: NotRequired[str]
    botName: NotRequired[str]


# ─── Typed parameter shapes for common outbound activities ────────────────────

class TransferActivityParams(TypedDict, total=False):
    transferTarget: str
    handoverReason: str
    transferSipHeaders: list[dict[str, str]]


class PlayUrlActivityParams(TypedDict, total=False):
    playUrlUrl: str
    playUrlMediaFormat: str
    playUrlCache: bool


class SendDtmfActivityParams(TypedDict, total=False):
    dtmf: str


class CallRecordingActivityParams(TypedDict, total=False):
    recordingName: str
    recordingFormat: str


# ─── Events ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SessionStartEvent:
    """Emitted once per call after the AudioCodes handshake is accepted and
    the Deepgram Voice Agent ``SettingsApplied`` has landed.
    """
    session_id: str
    conversation_id: str
    caller: str | None
    bot_name: str | None
    media_format: AudioCodesMediaFormat
    started_at: str


SessionEndReason = Literal["transfer", "hangup", "error", "ended"]


@dataclass(frozen=True, slots=True)
class SessionEndEvent:
    """Emitted exactly once per call, in every termination path.

    Attributes:
        session_id: Session ID.
        reason: Reason the session ended.
        duration_ms: Total duration of the session in milliseconds.
        ended_at: ISO 8601 timestamp of session end.
    """
    session_id: str
    reason: SessionEndReason
    duration_ms: int
    ended_at: str


@dataclass(frozen=True, slots=True)
class InboundActivityEvent:
    """Fires once for every inbound ``Activity`` that AudioCodes sends inside
    an ``activities`` envelope.
    """
    session_id: str
    activity: AudioCodesActivity


@dataclass(frozen=True, slots=True)
class BridgeErrorEvent:
    """Union of bridge-originated errors and the Voice Agent ``Error`` server
    message.

    Attributes:
        session_id: Session ID, if session was established; ``None`` otherwise.
        description: Human-readable error description.
        code: Voice Agent error code, or a bridge-defined code for local failures.
        cause: The underlying exception, if available.
        recoverable: If ``False``, the session has ended and a
            ``SessionEndEvent`` will also be emitted.
    """
    session_id: str | None
    description: str
    code: str
    cause: Exception | None
    recoverable: bool
