# Multi-Agent Example

Run multiple Deepgram Voice Agents from a single Python process — each with its own `DeepgramBridge`, its own `BridgeConfig`, and its own port.

This is useful when you want to host more than one bot persona on the same host (sales vs. support, English vs. Spanish, demo vs. production, multiple tenants, etc.) without spinning up separate processes or containers.

## How it works

Every `DeepgramBridge` instance owns its own configuration and event-handler registry. `bridge.run()` is just an awaitable that opens its own WebSocket listener on the port from its `BridgeConfig`. Because there is no shared global state between bridges, you can construct as many as you want and drive them all on the same event loop with `asyncio.gather(...)`:

```python
sales_bridge   = DeepgramBridge(BridgeConfig(..., port=8001))
support_bridge = DeepgramBridge(BridgeConfig(..., port=8002))

async def main() -> None:
    await asyncio.gather(
        sales_bridge.run(),
        support_bridge.run(),
    )

asyncio.run(main())
```

The bridges run independently:

- Each accepts inbound AudioCodes connections on **its own port**.
- Each opens **its own Deepgram Voice Agent connection** per call, using the agent config it was constructed with.
- Each has **its own handler registry** — a `@sales_bridge.on("session_start")` handler will only fire for calls that arrive on the sales bridge's port.
- Each `Session` is scoped to the bridge that created it, so handlers always have an unambiguous handle to the right call and the right agent.

## What's in `server.py`

- **`sales_bridge`** on port `8001` — upbeat sales prompt, Helena voice.
- **`support_bridge`** on port `8002` — patient support prompt, Aurora voice.
- A small `register_common_handlers(bridge, label)` helper that attaches the same set of logging handlers to both bridges. In a real app you would normally diverge here — different routing rules, different function-call handlers, different transfer destinations per persona.
- An `async def main()` that runs both bridges with `asyncio.gather(...)`.

## Run it

```bash
# from the repo root
cp .env.example .env     # then fill in DEEPGRAM_API_KEY
python examples/04_multi_agent/server.py
```

You should see two `AudioCodes Bridge server listening` log lines — one for `:8001` and one for `:8002`. Configure two AudioCodes Bot Connections in LiveHub / VAIC, point each one at the corresponding port, and calls will route to the right agent.

## Scaling out

`asyncio.gather` happily takes any number of coroutines, so the same pattern scales to N agents:

```python
bridges = [
    DeepgramBridge(BridgeConfig(..., port=8001)),
    DeepgramBridge(BridgeConfig(..., port=8002)),
    DeepgramBridge(BridgeConfig(..., port=8003)),
    # ...
]

async def main() -> None:
    await asyncio.gather(*(b.run() for b in bridges))

asyncio.run(main())
```

If you'd rather route many tenants over a single port (e.g. distinguish them by Authorization header or URL path), use the embedded-server pattern with `Session.serve(socket, bridge)` instead — see the top-level [README's "Embedding the bridge in an existing server"](../../README.md#embedding-the-bridge-in-an-existing-server) section. The two patterns can also be combined: one shared front-door server plus N port-bound bridges for tenants that want their own listener.
