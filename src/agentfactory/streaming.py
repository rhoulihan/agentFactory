# src/agentfactory/streaming.py
from __future__ import annotations

import json
import uuid
from typing import AsyncIterator

from .translate import map_stop_reason


def format_sse(event: dict) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode("utf-8")


async def openai_stream_to_anthropic_events(
    chunks: AsyncIterator[dict], requested_model: str
) -> AsyncIterator[dict]:
    yield {
        "type": "message_start",
        "message": {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": requested_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }

    next_index = 0
    text_index: int | None = None
    tool_block_index: dict[int, int] = {}   # openai tool index -> anthropic block index
    open_blocks: set[int] = set()
    finish_reason: str | None = None

    async for chunk in chunks:
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

        if delta.get("content"):
            if text_index is None:
                text_index = next_index
                next_index += 1
                open_blocks.add(text_index)
                yield {"type": "content_block_start", "index": text_index,
                       "content_block": {"type": "text", "text": ""}}
            yield {"type": "content_block_delta", "index": text_index,
                   "delta": {"type": "text_delta", "text": delta["content"]}}

        for tc in delta.get("tool_calls") or []:
            oai_idx = tc.get("index", 0)
            if oai_idx not in tool_block_index:
                idx = next_index
                next_index += 1
                tool_block_index[oai_idx] = idx
                open_blocks.add(idx)
                fn = tc.get("function", {})
                yield {"type": "content_block_start", "index": idx,
                       "content_block": {
                           "type": "tool_use",
                           "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                           "name": fn.get("name", ""),
                           "input": {},
                       }}
            args = (tc.get("function") or {}).get("arguments")
            if args:
                yield {"type": "content_block_delta",
                       "index": tool_block_index[oai_idx],
                       "delta": {"type": "input_json_delta", "partial_json": args}}

    for idx in sorted(open_blocks):
        yield {"type": "content_block_stop", "index": idx}

    yield {"type": "message_delta",
           "delta": {"stop_reason": map_stop_reason(finish_reason),
                     "stop_sequence": None},
           "usage": {"output_tokens": 0}}
    yield {"type": "message_stop"}
