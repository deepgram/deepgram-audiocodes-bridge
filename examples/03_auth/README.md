# Authentication Example

AudioCodes LiveHub / VAIC Bot Connections support three authentication modes on the upgrade request to your bridge. Pick the mode on the LiveHub side and match it on the bridge side via `BridgeConfig`. This example walks through all three and when to use each.

For the LiveHub-side configuration, see the AudioCodes docs:

- [LiveHub — Create Bot Connection](https://techdocs.audiocodes.com/livehub/#LiveHub/AudiocodesAPI-framework.htm#Create2)
- [VAIC — Bot Integration](https://techdocs.audiocodes.com/voice-ai-connect/#Bot-API/ac-bot-api-mode-websocket.htm?TocPath=AudioCodes%2520Bot%2520API%257CBot%2520API%257C_____2)

The bridge evaluates auth in this order: if `authenticate` is set it runs and fully replaces the `ac_token` check; otherwise if `ac_token` is set the incoming `Authorization: Bearer <token>` is compared byte-for-byte; otherwise no check is performed.

## 1. No Authentication

LiveHub opens the WebSocket with no `Authorization` header. Intended for **local development only** — never ship this to production.

**LiveHub:** set Authentication to _None_.

```python
bridge = DeepgramBridge(BridgeConfig(
    deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
    deepgram_config=deepgram_config,
    ac_token=None,
))
```

## 2. Permanent Token (shared secret)

LiveHub sends a static shared secret as `Authorization: Bearer <token>` on every upgrade. The bridge compares byte-for-byte and rejects mismatches with HTTP 401. Simple to operate — rotation means updating the value in both places at once.

**LiveHub:** set Authentication to _Header Authentication_ and paste the token.

```python
bridge = DeepgramBridge(BridgeConfig(
    deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
    deepgram_config=deepgram_config,
    ac_token=os.getenv("AC_TOKEN"),
))
```

This is the default mode shown in `server.py` — set `AC_TOKEN` in your `.env` to the value you pasted in LiveHub.

## 3. OAuth 2.0 (or anything else) — custom `authenticate` callback

For OAuth 2.0, LiveHub does a client-credentials grant against your identity provider, gets back an access token (usually a JWT), and presents it as `Authorization: Bearer <jwt>` on the upgrade. Validating that token — JWKS fetch, signature verification, `iss` / `aud` / `exp` checks — is application-specific, so the SDK exposes a callback and lets you own it.

Same callback shape works for anything else you want to gate on: DB-backed tokens, IP allowlists, mTLS peer certs, API-gateway-signed headers, etc.

When `authenticate` is set it **fully replaces** the built-in `ac_token` check. Return `None` to accept the upgrade or a `Response` to reject it.

**LiveHub:** set Authentication to _OAuth 2.0_ and fill in the token URL, Client ID, and Client Secret.

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
    deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
    deepgram_config=deepgram_config,
    ac_token=None,
    authenticate=authenticate,
))
```

## Run it

```bash
# from the repo root
cp .env.example .env     # fill in DEEPGRAM_API_KEY, and AC_TOKEN if using Mode 2
python examples/03_auth/server.py
```

The server listens on `ws://localhost:8000`. Expose that to the public internet and point your AudioCodes LiveHub / VAIC Bot Connection to your public URL. Open `server.py` and pick the mode that matches your LiveHub Bot Connection — Mode 2 (Permanent Token) is active by default; Modes 1 and 3 are commented blocks you can swap in.

## Picking a mode

| Situation                                                     | Mode                  |
| ------------------------------------------------------------- | --------------------- |
| Running the bridge on your laptop for a dev call              | 1. No Auth            |
| Production deploy, single tenant, you rotate secrets manually | 2. Permanent Token    |
| Production deploy, multi-tenant or compliance-driven rotation | 3. OAuth 2.0 / custom |
| You need to validate JWT claims or gate by IP / mTLS          | 3. OAuth 2.0 / custom |
