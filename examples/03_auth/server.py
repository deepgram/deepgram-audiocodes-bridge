import asyncio
import logging
import os

from dotenv import load_dotenv

from deepgram_audiocodes_bridge import (
    DeepgramBridge,
    Session,
    BridgeConfig,
    SessionStartEvent,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("auth-example")

# Minimal Deepgram Voice Agent config — the focus of this example is auth, not agent configuration.
deepgram_config = {
    "agent": {
        "listen": {"provider": {"type": "deepgram", "model": "flux-general-en"}},
        "think": {
            "provider": {"type": "open_ai", "model": "gpt-5.4-mini"},
            "prompt": "You are a helpful assistant.",
        },
        "speak": {"provider": {"type": "deepgram", "model": "aura-2-helena-en"}},
        "greeting": "Hello! You successfully authenticated.",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Pick ONE of the three auth modes below and delete (or comment out) the others.
# See README.md in this directory for when to use each.
# ─────────────────────────────────────────────────────────────────────────────


# ── Mode 1: No Authentication ───────────────────────────────────────────────
# LiveHub opens the WebSocket with no Authorization header. Local dev only.
# LiveHub setting: Authentication = "None"
#
# bridge = DeepgramBridge(BridgeConfig(
#     deepgram_api_key=deepgram_api_key,
#     deepgram_config=deepgram_config,
#     ac_token=None,
#     port=8000,
# ))


# ── Mode 2: Permanent Token (shared secret) ─────────────────────────────────
# LiveHub sends `Authorization: Bearer <AC_TOKEN>` on every upgrade. The bridge
# compares byte-for-byte and rejects mismatches with HTTP 401.
# LiveHub setting: Authentication = "Header Authentication", then paste the token.

deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
if not deepgram_api_key:
    raise SystemExit("DEEPGRAM_API_KEY environment variable is required")

bridge = DeepgramBridge(BridgeConfig(
    deepgram_api_key=deepgram_api_key,
    deepgram_config=deepgram_config,
    ac_token=os.getenv("AC_TOKEN"),  # must match the token set in LiveHub
    port=8000,
))


# ── Mode 3: OAuth 2.0 / custom callback ─────────────────────────────────────
# LiveHub does a client-credentials grant against your identity provider and
# presents the resulting access token (typically a JWT) as `Authorization:
# Bearer <jwt>`. Validation is application-specific, so you own it here.
#
# Return None to accept the upgrade, or a Response(401, ...) to reject.
# When `authenticate` is set it fully replaces the `ac_token` check.
# LiveHub setting: Authentication = "OAuth 2.0", then fill in token URL,
# Client ID, and Client Secret.
#

# async def authenticate(
#     connection: ServerConnection, request: Request
# ) -> Response | None:
#     auth = request.headers.get("Authorization", "")
#     token = auth.removeprefix("Bearer ").strip()

#     # Your validator — JWKS fetch, signature check, iss / aud / exp checks, etc.
#     if not is_valid_jwt(token):
#         return Response(401, "Unauthorized", Headers([]), b"Unauthorized\n")
#     return None

# def is_valid_jwt(token):
#     return True

# bridge = DeepgramBridge(BridgeConfig(
#     deepgram_api_key=deepgram_api_key,
#     deepgram_config=deepgram_config,
#     ac_token=None,
#     authenticate=authenticate,
#     port=8000,
# ))


@bridge.on("session_start")
async def on_start(session: Session, event: SessionStartEvent) -> None:
    log.info("authenticated call started: %s", session.session_id)


asyncio.run(bridge.run())
