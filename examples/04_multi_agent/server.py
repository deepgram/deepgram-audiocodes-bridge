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
    ConversationTextEvent,
    BridgeErrorEvent,
    WarningEvent,
)
from deepgram_audiocodes_bridge.types import (
    DeepgramAgent,
    DeepgramAgentConfig,
    DeepgramListen,
    DeepgramSpeak,
    DeepgramSpeakProviderDeepgram,
    DeepgramThink,
    DeepgramThinkProviderOpenAI,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# ─── Agent #1: Sales ──────────────────────────────────────────────────────────
# Cheerful Aura voice, sales-flavored prompt. Listens on port 8001.

sales_config = DeepgramAgentConfig(
    agent=DeepgramAgent(
        listen=DeepgramListen(
            provider={"type": "deepgram", "model": "flux-general-en"},
        ),
        think=[
            DeepgramThink(
                provider=DeepgramThinkProviderOpenAI(
                    type="open_ai",
                    model="gpt-5.4-mini",
                ),
                prompt=(
                    "You are an upbeat sales rep for Acme Otter Slides. "
                    "Help callers pick a slide model and book a demo."
                ),
            ),
        ],
        speak=[
            DeepgramSpeak(
                provider=DeepgramSpeakProviderDeepgram(
                    type="deepgram",
                    model="aura-2-helena-en",
                ),
            ),
        ],
        greeting="Hi, thanks for calling Acme Otter Slides sales — how can I help?",
    ),
)

sales_bridge = DeepgramBridge(BridgeConfig(
    deepgram_api_key=DEEPGRAM_API_KEY,
    deepgram_config=sales_config,
    ac_token=None,
    port=8001,
))

# ─── Agent #2: Support ────────────────────────────────────────────────────────
# Calmer Aura voice, support-flavored prompt. Listens on port 8002.

support_config = DeepgramAgentConfig(
    agent=DeepgramAgent(
        listen=DeepgramListen(
            provider={"type": "deepgram", "model": "flux-general-en"},
        ),
        think=[
            DeepgramThink(
                provider=DeepgramThinkProviderOpenAI(
                    type="open_ai",
                    model="gpt-5.4-mini",
                ),
                prompt=(
                    "You are a patient technical support agent for Acme Otter "
                    "Slides. Diagnose issues and walk callers through fixes."
                ),
            ),
        ],
        speak=[
            DeepgramSpeak(
                provider=DeepgramSpeakProviderDeepgram(
                    type="deepgram",
                    model="aura-2-aurora-en",
                ),
            ),
        ],
        greeting="Acme Otter Slides support — what's going on with your slide today?",
    ),
)

support_bridge = DeepgramBridge(BridgeConfig(
    deepgram_api_key=DEEPGRAM_API_KEY,
    deepgram_config=support_config,
    ac_token=None,
    port=8002,
))


# ─── Handlers ─────────────────────────────────────────────────────────────────
# Each bridge has its own independent handler registry. You can either register
# handlers separately per bridge (different behavior per agent) or share one
# function across both — shown here for brevity.

def register_common_handlers(bridge: DeepgramBridge, label: str) -> None:
    log = logging.getLogger(label)

    @bridge.on("session_start")
    async def on_start(session: Session, event: SessionStartEvent) -> None:
        log.info("call started: %s (caller=%s)", session.session_id, event.caller)

    @bridge.on("conversation_text")
    async def on_text(session: Session, event: ConversationTextEvent) -> None:
        log.info("[%s] %s: %s", session.session_id, event.role, event.content)

    @bridge.on("session_end")
    async def on_end(session: Session, event: SessionEndEvent) -> None:
        log.info("call ended: %s (%s)", session.session_id, event.reason)

    @bridge.on("error")
    async def on_error(session: Session, event: BridgeErrorEvent) -> None:
        log.error("bridge error [%s]: %s", event.code, event.description)

    @bridge.on("warning")
    async def on_warning(session: Session, event: WarningEvent) -> None:
        log.warning("bridge warning [%s]: %s", event.code, event.description)


register_common_handlers(sales_bridge, "sales")
register_common_handlers(support_bridge, "support")


# ─── Run both bridges concurrently ────────────────────────────────────────────
# Each bridge.run() awaits its own serve_forever(); asyncio.gather drives them
# in parallel on the same event loop. Add as many bridges as you need — just
# give each one a unique port.

async def main() -> None:
    await asyncio.gather(
        sales_bridge.run(),
        support_bridge.run(),
    )


asyncio.run(main())
