# src/agentfactory/translate.py
from __future__ import annotations

import json
import uuid

from .config import ModelConfig


def _system_to_text(system) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    return "\n".join(b.get("text", "") for b in system if b.get("type") == "text")


def _result_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def _convert_message(msg: dict) -> list[dict]:
    """Convert one Anthropic message into one or more OpenAI messages."""
    role = msg["role"]
    content = msg.get("content")
    if isinstance(content, str):
        return [{"role": role, "content": content}]

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    for block in content or []:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block.get("input", {})),
                },
            })
        elif btype == "tool_result":
            tool_results.append({
                "role": "tool",
                "tool_call_id": block["tool_use_id"],
                "content": _result_content_to_text(block.get("content", "")),
            })

    out: list[dict] = []
    if role == "assistant" and tool_calls:
        out.append({
            "role": "assistant",
            "content": "\n".join(text_parts) or None,
            "tool_calls": tool_calls,
        })
    elif text_parts:
        out.append({"role": role, "content": "\n".join(text_parts)})
    out.extend(tool_results)
    return out


def _convert_tools(tools: list[dict]) -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
        },
    } for t in tools]


def anthropic_to_openai_request(body: dict, mc: ModelConfig) -> dict:
    messages: list[dict] = []
    system = _system_to_text(body.get("system"))
    if system is not None:
        messages.append({"role": "system", "content": system})
    for msg in body.get("messages", []):
        messages.extend(_convert_message(msg))

    payload: dict = {"model": mc.model, "messages": messages}
    if "max_tokens" in body:
        payload["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        payload["temperature"] = body["temperature"]
    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]

    tools = body.get("tools")
    if tools:
        payload["tools"] = _convert_tools(tools)
        # Guided-decoding constraint: when tools are present on a local request the
        # proxy must ALWAYS set tool_choice so vLLM engages its tool parser. This is
        # independent of mc.guided_decoding.
        if "tool_choice" in body and isinstance(body["tool_choice"], dict):
            tc = body["tool_choice"]
            if tc.get("type") == "tool" and tc.get("name"):
                payload["tool_choice"] = {"type": "function",
                                          "function": {"name": tc["name"]}}
            elif tc.get("type") == "any":
                payload["tool_choice"] = "required"
            elif tc.get("type") == "none":
                payload["tool_choice"] = "none"
            else:
                payload["tool_choice"] = "auto"
        else:
            payload["tool_choice"] = "auto"
    return payload


_STOP_MAP = {"tool_calls": "tool_use", "length": "max_tokens", "stop": "end_turn"}


def map_stop_reason(finish_reason: str | None) -> str:
    return _STOP_MAP.get(finish_reason or "", "end_turn")


def openai_to_anthropic_response(resp: dict, requested_model: str) -> dict:
    choice = resp["choices"][0]
    message = choice.get("message", {})
    content: list[dict] = []

    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        content.append({
            "type": "tool_use",
            "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
            "name": fn.get("name", ""),
            "input": args,
        })

    usage = resp.get("usage", {})
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": map_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }
