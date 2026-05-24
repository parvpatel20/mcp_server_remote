"""Lambda entry point — wraps the FastMCP ASGI app via Mangum."""
from mangum import Mangum
from server import app


class FixAcceptMiddleware:
    """
    Patch Accept header for MCP clients that don't send both required types.

    FastMCP requires: Accept: application/json, text/event-stream
    Cursor and many MCP clients omit one or both — this middleware fills the gap.
    """

    def __init__(self, wrapped_app):
        self._app = wrapped_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            accept = headers.get(b"accept", b"").decode()
            if "text/event-stream" not in accept or "application/json" not in accept:
                headers[b"accept"] = b"application/json, text/event-stream"
                scope["headers"] = list(headers.items())
        await self._app(scope, receive, send)


handler = Mangum(FixAcceptMiddleware(app), lifespan="auto")
