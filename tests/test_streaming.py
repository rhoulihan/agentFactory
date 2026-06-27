# tests/test_streaming.py
import json
from agentfactory.streaming import openai_stream_to_anthropic_events, format_sse


async def _gen(items):
    for it in items:
        yield it


async def _collect(chunks, model="local/x"):
    return [e async for e in openai_stream_to_anthropic_events(_gen(chunks), model)]


def _types(events):
    return [e["type"] for e in events]


async def test_text_stream_event_order():
    chunks = [
        {"choices": [{"delta": {"content": "He"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "llo"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    events = await _collect(chunks)
    assert _types(events) == [
        "message_start", "content_block_start",
        "content_block_delta", "content_block_delta",
        "content_block_stop", "message_delta", "message_stop",
    ]
    deltas = [e["delta"]["text"] for e in events if e["type"] == "content_block_delta"]
    assert deltas == ["He", "llo"]
    assert events[0]["message"]["model"] == "local/x"
    assert events[-2]["delta"]["stop_reason"] == "end_turn"


async def test_tool_call_stream_reassembles_arguments():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "bash", "arguments": ""}}]},
            "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"comm'}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'and": "ls"}'}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    events = await _collect(chunks)
    start = next(e for e in events if e["type"] == "content_block_start")
    assert start["content_block"]["type"] == "tool_use"
    assert start["content_block"]["name"] == "bash"
    assert start["content_block"]["id"] == "call_1"
    partials = "".join(
        e["delta"]["partial_json"] for e in events
        if e["type"] == "content_block_delta"
    )
    assert json.loads(partials) == {"command": "ls"}
    assert events[-2]["delta"]["stop_reason"] == "tool_use"


def test_format_sse():
    raw = format_sse({"type": "message_stop"})
    assert raw == b'event: message_stop\ndata: {"type": "message_stop"}\n\n'
