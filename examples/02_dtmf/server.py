import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from deepgram_audiocodes_bridge import (
    DeepgramBridge,
    Session,
    BridgeConfig,
    SessionStartEvent,
    SessionEndEvent,
    InboundActivityEvent,
    BridgeErrorEvent,
    WarningEvent,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("dtmf-example")

# Minimal Deepgram Voice Agent config — just enough for the agent to greet the
# caller and speak replies. Audio format is negotiated from AudioCodes.
deepgram_config = {
    "agent": {
        "listen": {"provider": {"type": "deepgram", "model": "flux-general-en"}},
        "think": {
            "provider": {"type": "open_ai", "model": "gpt-5.4-mini"},
            "prompt": "You are a helpful assistant. The caller may enter digits on their keypad.",
        },
        "speak": {"provider": {"type": "deepgram", "model": "aura-2-aurora-en"}},
        "greeting": "Hi! Enter digits on your keypad, then press pound to submit or star to clear.",
    },
}

# Per-session DTMF buffers. Keyed by session_id so concurrent calls don't collide.
dtmf_buffers: dict[str, str] = {}

bridge = DeepgramBridge(BridgeConfig(
    deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
    deepgram_config=deepgram_config,
    ac_token=None,
    port=8000,
))


@bridge.on("session_start")
async def on_start(session: Session, event: SessionStartEvent) -> None:
    dtmf_buffers[session.session_id] = ""
    log.info("call started: %s", session.session_id)


@bridge.on("session_end")
async def on_end(session: Session, event: SessionEndEvent) -> None:
    dtmf_buffers.pop(session.session_id, None)
    log.info("call ended: %s (%s)", session.session_id, event.reason)


@bridge.on("error")
async def on_error(session: Session, event: BridgeErrorEvent) -> None:
    log.error("bridge error [%s]: %s", event.code, event.description)


@bridge.on("warning")
async def on_warning(session: Session, event: WarningEvent) -> None:
    log.warning("bridge warning [%s]: %s", event.code, event.description)


# DTMF digits arrive as an inbound activity with name="dtmf" and the pressed
# key(s) in activity["value"]. VAIC typically sends one digit per activity,
# but this handler is tolerant of multi-character values too.
@bridge.on("activity")
async def on_activity(session: Session, event: InboundActivityEvent) -> None:
    if event.activity.get("name") != "DTMF":
        return

    digits = str(event.activity.get("value", ""))
    log.info("DTMF received: %r", digits)

    for digit in digits:
        if digit == "#":
            entered = dtmf_buffers.get(session.session_id, "")
            dtmf_buffers[session.session_id] = ""
            if entered:
                log.info("submitted: %s", entered)
                await session.send_agent_message(
                    f"I got {' '.join(entered)}. Thanks!"
                )
            else:
                await session.send_agent_message("You haven't entered anything yet.")

        elif digit == "*":
            dtmf_buffers[session.session_id] = ""
            log.info("buffer cleared")
            await session.send_agent_message("Cleared. Try again.")

        else:
            dtmf_buffers[session.session_id] = (
                dtmf_buffers.get(session.session_id, "") + digit
             )


asyncio.run(bridge.run())
