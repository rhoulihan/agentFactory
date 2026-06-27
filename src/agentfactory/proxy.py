from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .backends import (
    call_openai_backend, stream_openai_backend, stream_upstream,
    filter_request_headers,
)
from .config import FactoryConfig
from .errors import anthropic_error
from .streaming import format_sse, openai_stream_to_anthropic_events
from .translate import anthropic_to_openai_request, openai_to_anthropic_response


def create_app(config: FactoryConfig) -> FastAPI:
    app = FastAPI(title="agentFactory proxy")
    app.state.config = config

    @app.get("/v1/models")
    async def list_models() -> dict:
        return {"data": [{"type": "model", "id": name}
                         for name in config.discovery_models()]}

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "models": list(config.models.keys())}

    @app.post("/v1/messages")
    async def messages(request: Request):
        raw = await request.body()
        body = await request.json()
        model = body.get("model", "")
        stream = bool(body.get("stream"))

        if config.is_passthrough(model):
            url = config.proxy.upstream_anthropic.rstrip("/") + "/v1/messages"
            headers = filter_request_headers(request.headers)
            status, up_headers, body_iter = await stream_upstream(
                "POST", url, headers, raw)
            media = up_headers.get("content-type", "application/json")
            return StreamingResponse(body_iter, status_code=status, media_type=media)

        mc = config.resolve_model(model)
        if mc is None:
            return JSONResponse(
                status_code=404,
                content=anthropic_error(f"no such model: {model}", "not_found_error"))

        backend = config.backend_for(mc)
        payload = anthropic_to_openai_request(body, mc)

        if stream:
            payload["stream"] = True

            async def event_stream():
                chunks = stream_openai_backend(
                    backend.base_url, backend.api_key, payload)
                async for event in openai_stream_to_anthropic_events(chunks, model):
                    yield format_sse(event)

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        status, oai = await call_openai_backend(
            backend.base_url, backend.api_key, payload)
        if status >= 400:
            return JSONResponse(
                status_code=status,
                content=anthropic_error(f"backend error: {oai}", "api_error"))
        return JSONResponse(openai_to_anthropic_response(oai, model))

    return app
