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

# The Deepgram___* config classes are helper types (autocomplete, etc.)
# You can also just pass a generic JSON object if you prefer, for example:
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


### This is how you initiate your bridge.
bridge = DeepgramBridge(BridgeConfig(
    deepgram_api_key="your-api-key",
    deepgram_config=deepgram_config,
    ac_token="your-audiocodes-token",  # Validates the Header Authentication configured in LiveHub / VAIC Bot Connection
    port=8000,  # default is port 8081. You can change it here.
))

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

## Register your other bridge event handlers here!

asyncio.run(bridge.run())
```

## Concurrency and the `Session` object

The bridge handles multiple concurrent calls out of the box — `bridge.run()` spawns a new asyncio task per inbound WebSocket; you don't need threads.

Each call is represented by a `Session` object, passed as the first argument to every event handler. It carries the per-call state and exposes all the control methods (`session.send_agent_message(...)`, `session.transfer(...)`, `session.end_session()`, `session.play_url(...)`, `session.get_transcript()`, and so on). Concurrent calls each get their own `Session`, so there is no global state to collide.

```python
@bridge.on("conversation_text")
async def on_text(session: Session, event: ConversationTextEvent) -> None:
    if "transfer me" in event.content.lower():
        await session.transfer("sip:queue@example.com")
```

### Embedding the bridge in an existing server

If you already run a WebSocket server (FastAPI, Starlette, or any framework built on the `websockets` library) and want the bridge to live as one route alongside health checks, admin APIs, and the rest of your app, use `Session.serve(socket, bridge)` directly. Construct the `DeepgramBridge` once at startup — it holds config and event handlers — and call `Session.serve` from your WebSocket handler after you've accepted the upgrade and performed your own authentication.

```python
bridge = DeepgramBridge(BridgeConfig(...))

@bridge.on("session_start")
async def on_start(session: Session, event: SessionStartEvent) -> None:
    ...

# inside your framework's WebSocket route:
# await Session.serve(websocket, bridge)
```

## Authentication

AudioCodes LiveHub / VAIC Bot Connections support three authentication modes on the upgrade request to your bridge. Pick one on the LiveHub side and match it on the bridge side via `BridgeConfig`. See the [AudioCodes Bot API documentation](https://techdocs.audiocodes.com/livehub/#LiveHub/AudiocodesAPI-framework.htm#Create2) for the LiveHub-side configuration details.

### 1. No Authentication

LiveHub opens the WebSocket with no `Authorization` header. Intended for local development only.

**LiveHub:** set Authentication to _None_.
**Bridge:**

```python
bridge = DeepgramBridge(BridgeConfig(
    deepgram_api_key="your-api-key",
    deepgram_config=deepgram_config,
    ac_token=None,
))
```

### 2. Permanent Token

LiveHub sends a static shared secret as `Authorization: Bearer <token>` on every upgrade. The bridge compares byte-for-byte and rejects mismatches with 401. Simple to operate — rotation means updating the value in both places.

**LiveHub:** set Authentication to _Header Authentication_ and paste the token.
**Bridge:**

```python
bridge = DeepgramBridge(BridgeConfig(
    deepgram_api_key="your-api-key",
    deepgram_config=deepgram_config,
    ac_token="your-audiocodes-token",
))
```

### 3. OAuth 2.0 (or anything else) — custom `authenticate` callback

For OAuth 2.0 LiveHub does a client-credentials grant against your identity provider, gets back an access token (usually a JWT), and presents it as `Authorization: Bearer <jwt>` on the upgrade. Validating that token — JWKS fetch, signature verification, `iss` / `aud` / `exp` checks — is application-specific, so the SDK exposes a callback and lets you own it.

When `authenticate` is set it fully replaces the built-in `ac_token` check. Return `None` to accept the upgrade or a `Response` to reject it.

**LiveHub:** set Authentication to _OAuth 2.0_ and fill in the token URL, Client ID, and Client Secret.
**Bridge:**

```python
from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

async def authenticate(
    connection: ServerConnection, request: Request
) -> Response | None:
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    if not is_valid_jwt(token):  # your validator
        return Response(401, "Unauthorized", Headers([]), b"Unauthorized\n")
    return None

bridge = DeepgramBridge(BridgeConfig(
    deepgram_api_key="your-api-key",
    deepgram_config=deepgram_config,
    ac_token=None,
    authenticate=authenticate,
))
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

### Handling inbound AudioCodes activities

Everything AudioCodes sends inside an `activities` envelope arrives on the `"activity"` event as an `InboundActivityEvent`. The raw activity dict is on `event.activity` — branch on `activity["name"]` to handle specific events like DTMF digits or silence-timeout notifications.

The most common inbound event activities are:

| Activity `name` | When it fires                                                        | Where the data lives                                           |
| --------------- | -------------------------------------------------------------------- | -------------------------------------------------------------- |
| `"dtmf"`        | Caller pressed digits on their phone keypad                          | `activity["value"]` — a string like `"1"`, `"42#"`, `"*"`      |
| `"noUserInput"` | VAIC's silence timer expired without caller speech (see note below)  | `activity["value"]` — int count of how many times it has fired |
| `"start"`       | VAIC sends this once per session; already handled by `session_start` | —                                                              |
| `"hangup"`      | VAIC sends this at call end; already handled by `session_end`        | `activity["activityParams"]["hangupReason"]`                   |

> **`noUserInput` requires VAIC-side configuration.** The AudioCodes gateway only forwards it to the bot when `sendEventsToBot` is set to include `noUserInput` on the VAIC bot connection. If it isn't configured, you will never see this activity — this can be helpful because the Deepgram Voice Agent API has no context as to silence and does not keep track of it natively.

```python
from deepgram_audiocodes_bridge import InboundActivityEvent

@bridge.on("activity")
async def on_activity(event: InboundActivityEvent) -> None:
    name = event.activity.get("name")

    if name == "dtmf":
        digits = str(event.activity.get("value", ""))
        print(f"Caller pressed: {digits}")
        # e.g. route to a menu handler, append to an account-number buffer, etc.

    elif name == "noUserInput":
        count = event.activity.get("value", 0)
        print(f"Silence timeout fired (#{count})")
        # e.g. reprompt the caller, or hang up after N timeouts.

    else:
        # Unknown / future event — log it so you notice new activity types.
        print(f"unhandled activity: {name} {event.activity}")
```

For the complete list of activities VAIC can send, see the "Receiving notifications" section under the ["Bot integration" AudioCodes docs](https://techdocs.audiocodes.com/voice-ai-connect/#VAIG_Combined/bot-integration.htm?TocPath=Bot%2520integration%257C_____0).

## Methods

Every method below is called on the per-call `Session` object. Inside an event handler the session arrives as the first argument:

```python
@bridge.on("conversation_text")
async def on_text(session: Session, event: ConversationTextEvent) -> None:
    if "agent" in event.content.lower():
        await session.send_agent_message("I'm here, how can I help?")
```

All control methods are coroutines — `await` them. Calling one after the session has ended raises `RuntimeError`; gate on `session.is_active` if you've handed the session off to a background task.

### Deepgram Voice Agent control

Methods that drive the agent itself — inject turns, swap providers, return function-call results. Each one wraps a [Voice Agent client message](https://developers.deepgram.com/docs/voice-agent-inputs).

| Method                                        | What it does                                                   | Wire message           |
| --------------------------------------------- | -------------------------------------------------------------- | ---------------------- |
| `send_agent_message(content)`                 | Force the agent to immediately speak `content`.                | `InjectAgentMessage`   |
| `send_user_message(content)`                  | Inject text as if the user had spoken it.                      | `InjectUserMessage`    |
| `update_prompt(prompt)`                       | Append to the agent's system prompt mid-conversation.          | `UpdatePrompt`         |
| `update_speak(speak)`                         | Swap the TTS provider/model mid-conversation.                  | `UpdateSpeak`          |
| `update_think(think)`                         | Replace the entire Think (LLM) configuration mid-conversation. | `UpdateThink`          |
| `respond_to_function_call(id, name, content)` | Return the result of a client-side function call to Deepgram.  | `FunctionCallResponse` |

```python
# Push the agent to speak something specific (e.g. after a long DB lookup).
await session.send_agent_message("Thanks for waiting — I found your account.")

# Swap to a different LLM mid-call (e.g. on user request, or as a fallback).
await session.update_think({
    "provider": {"type": "anthropic", "model": "claude-sonnet-4-6"},
    "prompt": "You are a billing specialist."
})

# Handle a function call from the agent.
@bridge.on("function_call_request")
async def on_function_call(session: Session, event: FunctionCallRequestEvent) -> None:
    for fc in event.functions:
        if fc.name == "get_order_status":
            result = await lookup_order(fc.arguments)
            await session.respond_to_function_call(fc.id, fc.name, result)
```

### AudioCodes Bot API control

Methods that drive the telephony layer — transfers, hangups, DTMF, audio prompts. Each one wraps an [AudioCodes outbound activity](https://techdocs.audiocodes.com/voice-ai-connect/#VAIG_Combined/sending-activities.htm?TocPath=Bot%2520integration%257CControlling%2520the%2520call%257C_____1).

| Method                                                                      | What it does                                                                                                                                                                           |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `transfer(destination, *, handover_reason=None, transfer_sip_headers=None)` | SIP REFER transfer to `destination`. Emits `session_end` with reason `'transfer'`.                                                                                                     |
| `end_session(reason=None)`                                                  | Send `hangup` and close both sockets. Emits `session_end` with reason `'ended'`.                                                                                                       |
| `play_url(url, *, options=None)`                                            | Play a pre-recorded audio file via VAIC's own audio engine.                                                                                                                            |
| `send_dtmf(digits, *, options=None)`                                        | Send DTMF digits downstream via VAIC.                                                                                                                                                  |
| `send_meta_data(data)`                                                      | Push arbitrary metadata to VAIC.                                                                                                                                                       |
| `abort_prompts()`                                                           | Cancel any VAIC-managed prompts currently playing.                                                                                                                                     |
| `expect_another_bot_message()`                                              | Tell VAIC to keep the turn open — another bot utterance is coming.                                                                                                                     |
| `apply_config(params)`                                                      | Dynamically change session-level configuration mid-call.                                                                                                                               |
| `send_activity(activity)`                                                   | Generic escape hatch — accepts a single activity dict or a list and wraps it in an `activities` envelope. Use this for any AudioCodes event the SDK doesn't expose a typed helper for. |

```python
# Warm transfer to a queue.
await session.transfer(
    "sip:queue@example.com",
    handover_reason="caller asked for a human",
)

# End the call from the bot side.
await session.end_session()

# Play a pre-recorded prompt instead of TTS.
await session.play_url("https://cdn.example.com/hold-music.wav")

# Send DTMF (e.g. navigating an IVR after transfer).
await session.send_dtmf("1234#")

# Generic activity for anything the SDK doesn't wrap.
await session.send_activity({
    "type": "event",
    "name": "customEvent",
    "activityParams": {"foo": "bar"},
})
```

### Call recording

Recording is handled by VAIC / the SBC, not by this SDK. These methods just toggle it on and off.

| Method                              | What it does               |
| ----------------------------------- | -------------------------- |
| `start_call_recording(params=None)` | Start recording.           |
| `stop_call_recording()`             | Stop recording.            |
| `pause_call_recording()`            | Pause recording.           |
| `resume_call_recording()`           | Resume a paused recording. |

```python
@bridge.on("session_start")
async def on_start(session: Session, event: SessionStartEvent) -> None:
    await session.start_call_recording({"recordingName": event.session_id})
```

### Read-only state

Properties and accessors on `Session`. Safe to read at any time, including inside a `session_end` handler.

| Accessor           | Returns                                                                                       |
| ------------------ | --------------------------------------------------------------------------------------------- |
| `session_id`       | `str` — unique ID assigned by the bridge for this call.                                       |
| `conversation_id`  | `str \| None` — AudioCodes conversation ID from `session.initiate`.                           |
| `media_format`     | `AudioCodesMediaFormat \| None` — format negotiated in `session.accepted`.                    |
| `is_active`        | `bool` — `True` while the session is in the `Active` state.                                   |
| `get_transcript()` | `list[ConversationTextEvent]` — every `conversation_text` event accumulated so far, in order. |

```python
async def archive_call(session_id, conversation_id, transcript):
    # Do something here
    # For example, POST to external CRM
    pass

@bridge.on("session_end")
async def on_end(session: Session, event: SessionEndEvent) -> None:
    transcript = session.get_transcript()
    await archive_call(session.session_id, session.conversation_id, transcript)
```

## A note on Barge-In

The Deepgram Voice Agent API has support for barge in and interruptions, and this SDK handles that natively (when a UserStartedSpeaking event is received from Deepgram, the SDK tells VAIC to stop playing the TTS audio). However, VAIC has an option to disable it if you want (it will ignore the stop message).

In VAIC / LiveHub, simply toggle on or off the "Barge-in" setting in the Bot Connection. Note, the default when creating a Bot Connection is for this setting to be off. For LiveHub, to toggle this setting on, and allow the user to interrupt the agent, see here - [Edit your Bot Connection](https://techdocs.audiocodes.com/livehub/#LiveHub/Editing%20your%20bot.htm?TocPath=Bot%2520connectivity%257C_____6)

## Local Development / Testing

```bash
git clone <ssh-or-https-url-for-this-repo>
cd audiocodes-deepgram-bridge
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Installation

```bash
pip install deepgram-audiocodes-bridge
```

NOTE: not published yet.

## Environment Variables

`.env.example` shows some example environment variables.

- `DEEPGRAM_API_KEY` is the only requirement for using this SDK.
- `AC_TOKEN` is optional and only needed if you are using Permanent Token authentication in the LiveHub / VAIC bot connection. It must match the value configured in LiveHub / VAIC.
