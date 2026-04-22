# deepgram-audiocodes-bridge

Python SDK that bridges the AudioCodes Bot API to the Deepgram Voice Agent API.

## Overview

Runs a WebSocket server implementing the AudioCodes Bot API protocol, opens and manages a Deepgram Voice Agent API connection per call, routes audio bidirectionally in real time, and emits typed higher-level events to application code.

## Quick Start

```python
import asyncio
import logging

from deepgram_audiocodes_bridge import (
    DeepgramBridge,
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
    WarningEvent
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
                provider=DeepgramSpeakProviderDeepgram(å
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
                        id="<your-desired-voice-id>"
                    # Sign up for a Cartesia account and then see the Voice Library for the Voice ID values.
                    # https://play.cartesia.ai/voices
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


bridge = DeepgramBridge(BridgeConfig(
    deepgram_api_key="your-api-key",
    deepgram_config=deepgram_config,
    ac_token="your-audiocodes-token",  # Validates the Header Authentication configured in LiveHub / VAIC Bot Connection
    port=8000,  # default is port 8081. You can change it here.
))

@bridge.on("session_start")
async def on_start(event: SessionStartEvent) -> None:
    print(f"Call started: {event.session_id}")
    print(event)

@bridge.on("error")
async def on_error(event: BridgeErrorEvent) -> None:
    print(event)

@bridge.on("warning")
async def on_warning(event: WarningEvent) -> None:
    print(event)

## Register your other bridge event handlers here!

asyncio.run(bridge.run())
```

## Events

Register handlers with `@bridge.on("<name>")`. The full set of events emitted by the bridge:

| Event name                | Payload type               |
| ------------------------- | -------------------------- |
| `"session_start"`         | `SessionStartEvent`        |
| `"session_end"`           | `SessionEndEvent`          |
| `"conversation_text"`     | `ConversationTextEvent`    |
| `"user_started_speaking"` | `UserStartedSpeakingEvent` |
| `"agent_thinking"`        | `AgentThinkingEvent`       |
| `"agent_audio_done"`      | `AgentAudioDoneEvent`      |
| `"function_call_request"` | `FunctionCallRequestEvent` |
| `"prompt_updated"`        | `PromptUpdatedEvent`       |
| `"speak_updated"`         | `SpeakUpdatedEvent`        |
| `"think_updated"`         | `ThinkUpdatedEvent`        |
| `"warning"`               | `WarningEvent`             |
| `"activity"`              | `InboundActivityEvent`     |
| `"error"`                 | `BridgeErrorEvent`         |

## Installation

```bash
pip install deepgram-audiocodes-bridge
```

## Environment Variables

`.env.example` shows the required environment variables. One is your Deepgram API Key. The other is the AudioCodes Token that is configured in the LiveHub / VAIC bot connection.
