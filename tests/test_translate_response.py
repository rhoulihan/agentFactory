# tests/test_translate_response.py
from agentfactory.translate import openai_to_anthropic_response, map_stop_reason

def test_map_stop_reason():
    assert map_stop_reason("tool_calls") == "tool_use"
    assert map_stop_reason("length") == "max_tokens"
    assert map_stop_reason("stop") == "end_turn"
    assert map_stop_reason(None) == "end_turn"

def test_text_response():
    resp = {"choices": [{"message": {"role": "assistant", "content": "hello"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    out = openai_to_anthropic_response(resp, "local/x")
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["model"] == "local/x"
    assert out["content"] == [{"type": "text", "text": "hello"}]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert out["id"].startswith("msg_")

def test_tool_call_response_parses_arguments():
    resp = {"choices": [{"message": {"role": "assistant", "content": None,
              "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
              "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9}}
    out = openai_to_anthropic_response(resp, "local/x")
    assert out["stop_reason"] == "tool_use"
    block = out["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "call_1"
    assert block["name"] == "bash"
    assert block["input"] == {"command": "ls"}

def test_text_and_tool_call_both_present():
    resp = {"choices": [{"message": {"role": "assistant", "content": "running it",
              "tool_calls": [{"id": "call_2", "type": "function",
                "function": {"name": "bash", "arguments": "{}"}}]},
              "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    out = openai_to_anthropic_response(resp, "local/x")
    assert out["content"][0] == {"type": "text", "text": "running it"}
    assert out["content"][1]["type"] == "tool_use"
