import asyncio
import logging
from dotenv import load_dotenv
import os

load_dotenv()

from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from deepgram_audiocodes_bridge import (
    DeepgramBridge,
    Session,
    BridgeConfig,
    SessionStartEvent,
)
from deepgram_audiocodes_bridge.types import (
    DeepgramAgent,
    DeepgramAgentConfig,
    DeepgramListen,
    DeepgramSpeak,
    DeepgramSpeakProviderDeepgram,
    DeepgramThink,
    DeepgramThinkProviderOpenAI,
    CartesiaSpeakVoice,
    DeepgramSpeakProviderCartesia,
    DeepgramThinkProviderAnthropic,
    BridgeErrorEvent,
    WarningEvent,
    SessionEndEvent,
    ConversationTextEvent,
    UserStartedSpeakingEvent,
    AgentThinkingEvent,
    AgentAudioDoneEvent,
    FunctionCallRequestEvent,
    PromptUpdatedEvent,
    SpeakUpdatedEvent,
    ThinkUpdatedEvent,
    InboundActivityEvent
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

deepgram_config: DeepgramAgentConfig = DeepgramAgentConfig(
    # You don't need to configure or set the Audio input/output settings. 
    # It will be pulled dynamically from AudioCodes Bot API.

    agent=DeepgramAgent(
        listen=DeepgramListen(
            provider={"type": "deepgram", "model": "flux-general-en"},
        ),
    # It's highly recommended to configure multiple LLM providers so you have a fallback mechanism
    # https://developers.deepgram.com/docs/voice-agent-llm-models#using-multiple-llm-providers
        think=[
            DeepgramThink(
                provider=DeepgramThinkProviderOpenAI(
                    type="open_ai",
                    model="gpt-5.4-mini",
                ),
                prompt="You are a helpful assistant.",
            ),
            DeepgramThink(
                provider=DeepgramThinkProviderAnthropic(
                    type="anthropic",
                    model="claude-sonnet-4-6"
                )
            )
        ],
    # It's also highly recommnded to configure multiple providers for TTS, for the same reasons.
    # https://developers.deepgram.com/docs/voice-agent-tts-models#using-multiple-tts-providers
        speak=[
            DeepgramSpeak(
                provider=DeepgramSpeakProviderDeepgram(
                    type="deepgram",
                    model="aura-2-helena-en"
                )
            ),
            DeepgramSpeak(
                provider=DeepgramSpeakProviderCartesia(
                    type="cartesia",
                    model_id="sonic-2",
                    voice=CartesiaSpeakVoice(
                        mode="id",
                        id="e07c00bc-4134-4eae-9ea4-1a55fb45746b"
                    )
                )
            )
        ],
        greeting="Hello from the otter slide!"
    ),
)

# The Deepgram___* config classes are helper types. 
# You can also pass a generic JSON object, for example:
"""
deepgram_config = {
  "agent": {
    "listen": {
        "provider": {
            "type": "deepgram",
            "model": "flux-general-en"
        }
    },
    "think" {
    ...[etc.]...
    }
  }
}
"""

# Optional: custom authentication (OAuth 2.0, DB-backed tokens, IP allowlists,
# etc.). When set on BridgeConfig, this fully replaces the built-in ac_token
# check — you own validation. Return None to accept, or a Response to reject.
async def authenticate(
    connection: ServerConnection, request: Request
) -> Response | None:
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    print(f"[auth] got Bearer token: {token!r}")
    # ... validate JWT signature ...
    # if not is_valid(token):
    #     return Response(401, "Unauthorized", Headers([]), b"Unauthorized\n")
    return None

bridge = DeepgramBridge(BridgeConfig(
    deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
    deepgram_config=deepgram_config,
    # ac_token="your-audiocodes-token",  # Validates the Header Authentication configured in LiveHub / VAIC Bot Connection
    ac_token=None,
    authenticate=authenticate,  # Uncomment to use custom auth (OAuth 2.0, etc.)
    port=8000,  # default is port 8081. You can change it here.
))

# Event handlers receive (session, event). The session is unique per call,
# so with multiple concurrent calls each handler invocation targets the right
# one — use session.send_agent_message(...), session.transfer(...), etc.

@bridge.on("session_start")
async def on_start(session: Session, event: SessionStartEvent) -> None:
    print(f"Call started: {session.session_id}")
    print(event)

@bridge.on("error")
async def on_error(session: Session, event: BridgeErrorEvent) -> None:
    print(event)

@bridge.on("warning")
async def on_warning(session: Session, event: WarningEvent) -> None:
    print(event)

@bridge.on("session_end")
async def on_session_end(session: Session, event: SessionEndEvent) -> None:
    print(event)

@bridge.on("conversation_text")
async def on_conversation_text(session: Session, event: ConversationTextEvent) -> None:
    print(event)

@bridge.on("user_started_speaking")
async def on_user_started_speaking(session: Session, event: UserStartedSpeakingEvent) -> None:
    print(event)

@bridge.on("agent_thinking")
async def on_agent_thinking(session: Session, event: AgentThinkingEvent) -> None:
    print(event)

@bridge.on("agent_audio_done")
async def on_agent_audio_done(session: Session, event: AgentAudioDoneEvent) -> None:
    print(event)

@bridge.on("function_call_request")
async def on_function_call_request(session: Session, event: FunctionCallRequestEvent) -> None:
    print(event)

@bridge.on("prompt_updated")
async def on_prompt_updated(session: Session, event: PromptUpdatedEvent) -> None:
    print(event)

@bridge.on("speak_updated")
async def on_speak_updated(session: Session, event: SpeakUpdatedEvent) -> None:
    print(event)

@bridge.on("think_updated")
async def on_think_updated(session: Session, event: ThinkUpdatedEvent) -> None:
    print(event)

@bridge.on("activity")
async def on_activity(session: Session, event: InboundActivityEvent) -> None:
    print(event)

asyncio.run(bridge.run())