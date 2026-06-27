# src/agentfactory/smoke.py
from __future__ import annotations

import httpx

from .config import FactoryConfig

_PROBE_TOOL = {
    "name": "factory_probe",
    "description": "Acknowledge readiness. Call this tool with ok=true.",
    "input_schema": {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    },
}


async def smoke_test_model(
    config: FactoryConfig, alias: str, base_url: str | None = None
) -> tuple[bool, str]:
    base = base_url or f"http://{config.proxy.host}:{config.proxy.port}"
    request = {
        "model": alias,
        "max_tokens": 256,
        "tools": [_PROBE_TOOL],
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user",
                      "content": "Call the factory_probe tool with ok=true."}],
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{base}/v1/messages", json=request)
    except httpx.HTTPError as exc:
        return False, f"request failed: {exc}"

    if resp.status_code != 200:
        return False, f"http {resp.status_code}: {resp.text[:200]}"

    blocks = resp.json().get("content", [])
    for block in blocks:
        if block.get("type") == "tool_use" and block.get("name") == "factory_probe":
            return True, "tool_use returned"
    return False, "no tool_use in response"
