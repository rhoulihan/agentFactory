# tests/test_backends.py
import httpx
import respx
from agentfactory.backends import (
    call_openai_backend, stream_openai_backend, filter_request_headers,
)


def test_filter_request_headers_drops_hop_by_hop():
    out = filter_request_headers({"Host": "x", "Content-Length": "3",
                                  "x-api-key": "secret", "anthropic-version": "2023-06-01"})
    assert "host" not in {k.lower() for k in out}
    assert "content-length" not in {k.lower() for k in out}
    assert out["x-api-key"] == "secret"
    assert out["anthropic-version"] == "2023-06-01"


@respx.mock
async def test_call_openai_backend():
    route = respx.post("http://be:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
    status, body = await call_openai_backend("http://be:8000/v1", "k", {"model": "m"})
    assert status == 200
    assert body["choices"][0]["message"]["content"] == "ok"
    assert route.calls.last.request.headers["authorization"] == "Bearer k"


@respx.mock
async def test_stream_openai_backend_parses_sse():
    sse = (b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
           b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
           b'data: [DONE]\n\n')
    respx.post("http://be:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=sse,
                                    headers={"content-type": "text/event-stream"}))
    out = [c async for c in stream_openai_backend("http://be:8000/v1", "k", {"stream": True})]
    assert out[0]["choices"][0]["delta"]["content"] == "hi"
    assert out[-1]["choices"][0]["finish_reason"] == "stop"
