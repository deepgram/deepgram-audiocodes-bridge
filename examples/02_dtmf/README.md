# DTMF Example

Shows how to handle **inbound DTMF** — digits the caller presses on their phone keypad — alongside a running Deepgram Voice Agent. The example buffers digits per-call, treats `#` as "submit", `*` as "clear", and uses `session.send_agent_message(...)` to have the agent acknowledge the input in its own voice.

## Required AudioCodes configuration

DTMF forwarding is **off by default** on AudioCodes Bot Connections — you must opt in, otherwise the bridge will never see an activity when the caller presses keys.

On your LiveHub / VAIC Bot Connection, open the **Advanced JSON settings** and add:

```json
{
  "sendDTMF": true
}
```

Merge it with any other advanced settings already present. Without this flag the call will connect fine but no `dtmf` activity will ever reach the bridge, and the demo will appear to do nothing when you press digits.

Always consult the [AudioCodes docs](https://techdocs.audiocodes.com/voice-ai-connect/#VAIG_Combined/receive-dtmf.htm?TocPath=Bot%2520integration%257CReceiving%2520notifications%257C_____3) for the most up to date information.

## How DTMF arrives

The Deepgram Voice Agent API has no concept of DTMF — the digits come directly from AudioCodes, not from the audio stream. They show up on the bridge's `"activity"` event as an `InboundActivityEvent` with `activity["name"] == "DTMF"` and the pressed key in `activity["value"]` (a string like `"1"`, `"#"`, or occasionally multiple digits like `"42#"`).

```python
@bridge.on("activity")
async def on_activity(session: Session, event: InboundActivityEvent) -> None:
    if event.activity.get("name") == "DTMF":
        digits = str(event.activity.get("value", ""))
        ...
```

## Per-session state

Each concurrent call gets its own `Session`, so any per-call state (like a digit buffer) needs to be keyed by `session.session_id`. This example uses a module-level `dict[str, str]` initialized in `session_start` and cleaned up in `session_end`.

## Run it

```bash
# from the repo root
cp .env.example .env     # then fill in DEEPGRAM_API_KEY
python examples/02_dtmf/server.py
```

The server listens on `ws://localhost:8000`. Expose that to the public internet and point your AudioCodes LiveHub / VAIC Bot Connection to your public URL. Then, place a call through your LiveHub / VAIC Bot Connection and press keys on the keypad — the server logs each digit and the agent speaks back when you press `#` or `*`.

## Adapting it

- **IVR menu routing** — branch on the first digit (`1` → sales, `2` → billing, etc.) and call `session.transfer(...)` to hand off.
- **Account number capture** — keep the buffer-and-submit pattern; on `#`, look up the account in your backend and use `session.update_prompt(...)` to give the agent caller context.
- **PIN entry** — same buffer pattern, but validate length and compare against a stored hash before acknowledging.

## Related: `noUserInput`

DTMF isn't the only activity VAIC forwards. If you enable `noUserInput` in the VAIC Bot Connection's `sendEventsToBot`, you'll also see `activity["name"] == "noUserInput"` when the caller is silent past VAIC's timeout — handy for reprompting or hanging up after N strikes. See the [top-level README](../../README.md#handling-inbound-audiocodes-activities) for the full list of activities.
