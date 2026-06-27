# src/agentfactory/backends.py
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

_HOP_BY_HOP = {"host", "content-length", "connection", "accept-encoding"}


def filter_request_headers(headers) -> dict:
    return {k: v for k, v in dict(headers).items() if k.lower() not in _HOP_BY_HOP}


async def stream_upstream(
    method: str, url: str, headers: dict, content: bytes
) -> tuple[int, dict, AsyncIterator[bytes]]:
    client = httpx.AsyncClient(timeout=None)
    req = client.build_request(method, url, headers=headers, content=content)
    resp = await client.send(req, stream=True)

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return resp.status_code, dict(resp.headers), body()


async def call_openai_backend(base_url: str, api_key: str, payload: dict) -> tuple[int, dict]:
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        return resp.status_code, resp.json()


async def stream_openai_backend(
    base_url: str, api_key: str, payload: dict
) -> AsyncIterator[dict]:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if data == "[DONE]":
                    return
                yield json.loads(data)
