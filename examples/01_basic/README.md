# Basic Example

A minimal end-to-end bridge server: accepts AudioCodes Bot API WebSocket connections, proxies them to a Deepgram Voice Agent with failover providers configured for both Think (LLM) and Speak (TTS), and prints every bridge event to stdout.

Use this as the starting point for your own integration — clone `server.py`, swap in your own handlers, and go.

## What's in `server.py`

- **`deepgram_config`** — a `DeepgramAgentConfig` with a Deepgram STT model, two Think providers (OpenAI primary, Anthropic fallback), and two Speak providers (Deepgram primary, Cartesia fallback). Audio format is negotiated automatically from AudioCodes at session start, so you don't configure it here.
- **`BridgeConfig`** — wires the Deepgram key, agent config, and port together. Port defaults to 8081; this example overrides to 8000.
- **`@bridge.on(...)` handlers** — one for every event the bridge emits, each just prints the payload. Replace the bodies with your own logic.

## Run it

```bash
# from the repo root
cp .env.example .env     # then fill in DEEPGRAM_API_KEY
python examples/01_basic/server.py
```

The server listens on `ws://localhost:8000`. Expose that to the public internet and point your AudioCodes LiveHub / VAIC Bot Connection to your public URL.

## What to change first

1. **Models and greeting** — edit the `DeepgramAgentConfig` to use your preferred LLM, voice, and opening line.
2. **Event handlers** — replace the `print(event)` bodies with real logic. See the [Events](../../README.md#events) and [Methods](../../README.md#methods) sections of the top-level README for the full surface area.
