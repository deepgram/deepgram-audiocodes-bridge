from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

# ─── Configuration ────────────────────────────────────────────────────────────


class DeepgramAudioInput(TypedDict, total=False):
    """Input half of Deepgram ``audio.input``. Keys are wire-format exactly as
    the Voice Agent V1 API expects — do not rename them.
    """
    encoding: Literal[
        "linear16", "linear32", "flac", "alaw",
        "mulaw", "amr-nb", "amr-wb", "opus",
        "ogg-opus", "speex", "g729"]
    sample_rate: int  # Most common values are 16000, 24000, 44100, 48000


class DeepgramAudioOutput(TypedDict, total=False):
    """Output half of Deepgram ``audio.output``. ``container`` must be
    ``"none"`` for raw streams — AudioCodes ``playStream.chunk`` cannot
    carry WAV or OGG containers.
    """
    encoding: Literal["linear16", "mulaw", "alaw"]
    sample_rate: int
    container: Literal["none", "wav", "ogg"]
    bitrate: int


class DeepgramAudio(TypedDict, total=False):
    input: DeepgramAudioInput
    output: DeepgramAudioOutput


class DeepgramListen(TypedDict, total=False):
    provider: dict[str, object]


# ─── Shared types ─────────────────────────────────────────────────────────────


class CustomEndpoint(TypedDict, total=False):
    """Custom provider endpoint. Used for both LLM (think) and TTS (speak)."""
    url: str
    headers: dict[str, str]


class AwsCredentials(TypedDict):
    """AWS credentials for Polly and Bedrock. ``session_token`` is required for STS auth."""
    type: Literal["sts", "iam"]
    region: str
    access_key_id: str
    secret_access_key: str
    session_token: NotRequired[str]  # Required for STS, optional for IAM


# ─── Function types ───────────────────────────────────────────────────────────


class DeepgramFunctionParameterProperty(TypedDict, total=False):
    """JSON Schema property definition for a single parameter within a
    function's ``parameters.properties`` map.
    """
    type: str
    description: str
    enum: list[str]


class DeepgramFunctionParameters(TypedDict, total=False):
    """JSON Schema ``object`` describing a function's input parameters."""
    type: Literal["object"]
    properties: dict[str, DeepgramFunctionParameterProperty]
    required: list[str]


class DeepgramFunction(TypedDict):
    """A single function definition in the Voice Agent ``think.functions`` list."""
    name: str
    description: str
    parameters: NotRequired[DeepgramFunctionParameters]
    endpoint: NotRequired[CustomEndpoint]


# ─── Think provider types ─────────────────────────────────────────────────────
# Each provider has its own TypedDict discriminated by the ``type`` field.
#
# Enum-like aliases use ``Literal[...] | str`` so known values get IDE
# autocomplete but the SDK doesn't break when providers ship new models.

DeepgramThinkOpenAIModel = Literal[
    "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4",
    "gpt-5.3-chat-latest",
    "gpt-5.2-chat-latest", "gpt-5.2",
    "gpt-5.1-chat-latest", "gpt-5.1",
    "gpt-5-nano", "gpt-5-mini", "gpt-5",
    "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1",
    "gpt-4o-mini", "gpt-4o",
] | str

DeepgramThinkAnthropicModel = Literal[
    "claude-4-5-haiku-latest",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-3-5-haiku-latest",
    "claude-sonnet-4-20250514",
] | str

DeepgramThinkGoogleModel = Literal[
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
] | str

DeepgramThinkGroqModel = Literal["openai/gpt-oss-20b"] | str

DeepgramThinkBedrockModel = Literal[
    "anthropic/claude-3-5-sonnet-20240620-v1:0",
    "anthropic/claude-3-5-haiku-20240307-v1:0",
] | str


class DeepgramThinkProviderOpenAI(TypedDict):
    type: Literal["open_ai"]
    model: DeepgramThinkOpenAIModel
    version: NotRequired[Literal["v1"]]
    temperature: NotRequired[float]  # 0–2


class DeepgramThinkProviderAnthropic(TypedDict):
    type: Literal["anthropic"]
    model: DeepgramThinkAnthropicModel
    version: NotRequired[Literal["v1"]]
    temperature: NotRequired[float]  # 0–1


class DeepgramThinkProviderGoogle(TypedDict):
    type: Literal["google"]
    model: DeepgramThinkGoogleModel
    version: NotRequired[Literal["v1beta"]]
    temperature: NotRequired[float]  # 0–2


class DeepgramThinkProviderGroq(TypedDict):
    type: Literal["groq"]
    model: DeepgramThinkGroqModel
    version: NotRequired[Literal["v1"]]
    temperature: NotRequired[float]  # 0–2


class DeepgramThinkProviderAwsBedrock(TypedDict):
    type: Literal["aws_bedrock"]
    model: DeepgramThinkBedrockModel
    temperature: NotRequired[float]  # 0–2
    credentials: AwsCredentials


DeepgramThinkProvider = (
    DeepgramThinkProviderOpenAI
    | DeepgramThinkProviderAnthropic
    | DeepgramThinkProviderGoogle
    | DeepgramThinkProviderGroq
    | DeepgramThinkProviderAwsBedrock
)


class DeepgramThink(TypedDict, total=False):
    provider: DeepgramThinkProvider
    prompt: str
    context_length: int | Literal["max"]
    endpoint: CustomEndpoint
    functions: list[DeepgramFunction]


# ─── Speak provider types ─────────────────────────────────────────────────────
# Each provider has its own TypedDict discriminated by the ``type`` field.
#
# Enum-like aliases use ``Literal[...] | str`` so known values get IDE
# autocomplete but the SDK doesn't break when Deepgram ships new models.

DeepgramSpeakModel = Literal[
    # Aura (v1)
    "aura-asteria-en", "aura-luna-en", "aura-stella-en", "aura-athena-en",
    "aura-hera-en", "aura-orion-en", "aura-arcas-en", "aura-perseus-en",
    "aura-angus-en", "aura-orpheus-en", "aura-helios-en", "aura-zeus-en",
    # Aura 2 — English
    "aura-2-amalthea-en", "aura-2-andromeda-en", "aura-2-apollo-en",
    "aura-2-arcas-en", "aura-2-aries-en", "aura-2-asteria-en",
    "aura-2-athena-en", "aura-2-atlas-en", "aura-2-aurora-en",
    "aura-2-callista-en", "aura-2-cora-en", "aura-2-cordelia-en",
    "aura-2-delia-en", "aura-2-draco-en", "aura-2-electra-en",
    "aura-2-harmonia-en", "aura-2-helena-en", "aura-2-hera-en",
    "aura-2-hermes-en", "aura-2-hyperion-en", "aura-2-iris-en",
    "aura-2-janus-en", "aura-2-juno-en", "aura-2-jupiter-en",
    "aura-2-luna-en", "aura-2-mars-en", "aura-2-minerva-en",
    "aura-2-neptune-en", "aura-2-odysseus-en", "aura-2-ophelia-en",
    "aura-2-orion-en", "aura-2-orpheus-en", "aura-2-pandora-en",
    "aura-2-phoebe-en", "aura-2-pluto-en", "aura-2-saturn-en",
    "aura-2-selene-en", "aura-2-thalia-en", "aura-2-theia-en",
    "aura-2-vesta-en", "aura-2-zeus-en",
    # Aura 2 — Spanish
    "aura-2-sirio-es", "aura-2-nestor-es", "aura-2-carina-es",
    "aura-2-celeste-es", "aura-2-alvaro-es", "aura-2-diana-es",
    "aura-2-aquila-es", "aura-2-selena-es", "aura-2-estrella-es",
    "aura-2-javier-es",
] | str

ElevenLabsSpeakModelId = Literal[
    "eleven_turbo_v2_5", "eleven_monolingual_v1", "eleven_multilingual_v2"
] | str

CartesiaSpeakModelId = Literal["sonic-2", "sonic-multilingual"] | str

OpenAISpeakModel = Literal["tts-1", "tts-1-hd"] | str
OpenAISpeakVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] | str

AwsPollySpeakVoice = Literal[
    "Matthew", "Joanna", "Amy", "Emma", "Brian", "Arthur", "Aria", "Ayanda"
] | str
AwsPollySpeakEngine = Literal["generative", "long-form", "standard", "neural"] | str


class CartesiaSpeakVoice(TypedDict):
    """Cartesia voice selection."""
    mode: str  # Cartesia voice mode (e.g. "id")
    id: str    # Cartesia voice ID


class DeepgramSpeakProviderDeepgram(TypedDict):
    type: Literal["deepgram"]
    model: DeepgramSpeakModel
    version: NotRequired[Literal["v1"]]
    speed: NotRequired[float]  # Speaking rate multiplier; default 1.0. Allowed values 0.7-1.5


class DeepgramSpeakProviderElevenLabs(TypedDict):
    type: Literal["eleven_labs"]
    model_id: ElevenLabsSpeakModelId
    version: NotRequired[Literal["v1"]]
    language: NotRequired[str]
    language_code: NotRequired[str]  # Deprecated; prefer ``language``


class DeepgramSpeakProviderCartesia(TypedDict):
    type: Literal["cartesia"]
    model_id: CartesiaSpeakModelId
    voice: CartesiaSpeakVoice
    version: NotRequired[Literal["2025-03-17"]]
    language: NotRequired[str]
    volume: NotRequired[float]  # Range 0.5–2.0


class DeepgramSpeakProviderOpenAI(TypedDict):
    type: Literal["open_ai"]
    model: OpenAISpeakModel
    voice: OpenAISpeakVoice
    version: NotRequired[Literal["v1"]]


class DeepgramSpeakProviderAwsPolly(TypedDict):
    type: Literal["aws_polly"]
    voice: AwsPollySpeakVoice
    language: str
    engine: AwsPollySpeakEngine
    credentials: AwsCredentials
    language_code: NotRequired[str]  # Deprecated; prefer ``language``


DeepgramSpeakProvider = (
    DeepgramSpeakProviderDeepgram
    | DeepgramSpeakProviderElevenLabs
    | DeepgramSpeakProviderCartesia
    | DeepgramSpeakProviderOpenAI
    | DeepgramSpeakProviderAwsPolly
)


class DeepgramSpeak(TypedDict):
    """A single TTS provider configuration.

    ``agent.speak`` accepts one ``DeepgramSpeak`` or a list of them
    (primary + fallback providers).
    """
    provider: DeepgramSpeakProvider
    endpoint: NotRequired[CustomEndpoint]


class DeepgramAgent(TypedDict, total=False):
    language: str
    context: dict[str, object]
    listen: DeepgramListen
    think: DeepgramThink | list[DeepgramThink]
    speak: DeepgramSpeak | list[DeepgramSpeak]
    greeting: str


class DeepgramAgentConfig(TypedDict, total=False):
    """Deepgram Voice Agent V1 Settings payload.

    Serialized verbatim and sent as the ``Settings`` JSON payload immediately
    after the server's ``Welcome`` message. The ``type`` field is injected by
    ``DeepgramHandler`` — application code provides only ``audio`` and
    ``agent``.
    """
    tags: list[str]
    experimental: bool
    mip_opt_out: bool
    flags: dict[str, object]
    audio: DeepgramAudio
    agent: DeepgramAgent


DeepgramConfig = DeepgramAgentConfig


# ─── Events ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ConversationTextEvent:
    """Mirrors the Voice Agent ``ConversationText`` server message."""
    session_id: str
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class UserStartedSpeakingEvent:
    """Mirrors the Voice Agent ``UserStartedSpeaking`` server message."""
    session_id: str


@dataclass(frozen=True, slots=True)
class AgentThinkingEvent:
    """Mirrors the Voice Agent ``AgentThinking`` server message."""
    session_id: str
    content: str


@dataclass(frozen=True, slots=True)
class AgentAudioDoneEvent:
    """Mirrors the Voice Agent ``AgentAudioDone`` server message."""
    session_id: str


@dataclass(frozen=True, slots=True)
class FunctionCall:
    """One entry in the ``functions`` list of a ``FunctionCallRequestEvent``."""
    id: str
    name: str
    arguments: str
    client_side: bool


@dataclass(frozen=True, slots=True)
class FunctionCallRequestEvent:
    """Mirrors the Voice Agent ``FunctionCallRequest`` server message."""
    session_id: str
    functions: tuple[FunctionCall, ...]


@dataclass(frozen=True, slots=True)
class WarningEvent:
    """Mirrors the Voice Agent ``Warning`` server message."""
    session_id: str
    code: str
    description: str


@dataclass(frozen=True, slots=True)
class PromptUpdatedEvent:
    """Mirrors the Voice Agent ``PromptUpdated`` server message."""
    session_id: str


@dataclass(frozen=True, slots=True)
class SpeakUpdatedEvent:
    """Mirrors the Voice Agent ``SpeakUpdated`` server message."""
    session_id: str


@dataclass(frozen=True, slots=True)
class ThinkUpdatedEvent:
    """Mirrors the Voice Agent ``ThinkUpdated`` server message."""
    session_id: str
