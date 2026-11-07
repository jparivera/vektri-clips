"""Bound request bytes before JSON buffering or multipart spooling.

The ASGI receive wrapper forwards each allowed message unchanged. It neither
buffers a complete body nor drains an oversized request. Raising HTTPException
lets FastAPI preserve the 413 and Starlette's multipart parser close partially
written temporary files while unwinding. A fake disconnect would obscure the
reason for failure and is deliberately not used.
"""
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import config


JSON_BODY_LIMIT = 8 * 1024**2
MULTIPART_OVERHEAD = 1024**2


class RequestBodyTooLarge(HTTPException):
    def __init__(self, limit: int, headers: dict[str, str]) -> None:
        super().__init__(
            status_code=413,
            detail=f'Request body exceeds the {limit} byte limit. Use bounded resumable uploads for larger media.',
            headers=headers,
        )


class RequestBodyLimitMiddleware:
    """Pure ASGI limits, including bodies without a trustworthy Content-Length.

    Only the actual multipart import route receives the larger media allowance.
    JSON, raw resumable chunks, and all other request bodies use the 8 MiB bound,
    so changing Content-Type cannot expand a JSON endpoint's memory budget.
    Optional limits exist for isolated tests; production reads MAX_UPLOAD live.
    """

    def __init__(self, app: ASGIApp, *, json_limit: int = JSON_BODY_LIMIT,
                 media_limit: int | None = None) -> None:
        if json_limit < 0 or (media_limit is not None and media_limit < 0):
            raise ValueError('Request body limits cannot be negative')
        self.app = app
        self.json_limit = json_limit
        self.media_limit = media_limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        is_media = (
            scope.get('method') == 'POST'
            and scope.get('path') == '/api/v1/imports'
            and headers.get('content-type', '').split(';', 1)[0].strip().lower()
            == 'multipart/form-data'
        )
        limit = (self.media_limit if self.media_limit is not None else config.MAX_UPLOAD + MULTIPART_OVERHEAD) if is_media else self.json_limit
        response_headers = {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'}
        # The unread remainder must not become another HTTP/1 request. HTTP/2
        # forbids Connection headers; its server owns the independent stream.
        if scope.get('http_version', '1.1') in {'1.0', '1.1'}:
            response_headers['Connection'] = 'close'
        failure = RequestBodyTooLarge(limit, response_headers)

        declared = headers.get('content-length')
        if declared is not None:
            try:
                valid = 0 <= int(declared) <= limit
            except ValueError:
                valid = False
            if not valid:
                await JSONResponse({'detail': failure.detail}, status_code=413,
                                   headers=response_headers)(scope, receive, send)
                return

        consumed = 0
        exceeded = False
        response_started = False

        async def bounded_receive() -> Message:
            nonlocal consumed, exceeded
            if exceeded:
                raise failure
            message = await receive()
            if message['type'] == 'http.request':
                consumed += len(message.get('body', b''))
                if consumed > limit:
                    exceeded = True
                    raise failure
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message['type'] == 'http.response.start':
                response_started = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, tracked_send)
        except RequestBodyTooLarge:
            # FastAPI normally handles this inside its exception middleware.
            # This fallback also supports direct ASGI consumers. Never send a
            # second response if a streaming consumer started its response.
            if response_started:
                raise
            await JSONResponse({'detail': failure.detail}, status_code=413,
                               headers=response_headers)(scope, receive, send)
