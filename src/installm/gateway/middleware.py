"""Authentication middleware for the InstaLLM gateway.

When auth is enabled (via --require-auth flag or INSTALLM_REQUIRE_AUTH env),
all requests except /health must include a valid Bearer token in the
Authorization header. Mirrors the OpenAI API auth pattern.
"""

import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from installm.auth import validate_key

# Paths that never require auth
OPEN_PATHS = {"/health", "/docs", "/openapi.json"}


def is_auth_enabled() -> bool:
    """Check if auth enforcement is active."""
    return os.environ.get("INSTALLM_REQUIRE_AUTH", "").lower() in ("1", "true", "yes")


class AuthMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid API key when auth is enabled."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth for open paths and OPTIONS (CORS preflight)
        if request.url.path in OPEN_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        if not is_auth_enabled():
            return await call_next(request)

        # Extract Bearer token
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Missing API key. Include 'Authorization: Bearer sk-installm-...' header.",
                        "type": "authentication_error",
                        "code": "missing_api_key",
                    }
                },
            )

        token = auth_header[7:]  # Strip "Bearer "
        if not validate_key(token):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid API key.",
                        "type": "authentication_error",
                        "code": "invalid_api_key",
                    }
                },
            )

        return await call_next(request)
