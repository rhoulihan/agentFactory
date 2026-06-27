# tests/test_translate_request.py
from agentfactory.config import ModelConfig
from agentfactory.translate import anthropic_to_openai_request

MC = ModelConfig(alias="local/x", backend="b", model="Qwen/X", guided_decoding=True)

def test_system_lifted_to_message():
    body = {"model": "local/x", "max_tokens": 100, "system": "be brief",
            "messages": [{"role": "user", "content": "hi"}]}
    out = anthropic_to_openai_request(body, MC)
    assert out["model"] == "Qwen/X"
    assert out["messages"][0] == {"role": "system", "content": "be brief"}
    assert out["messages"][1] == {"role": "user", "content": "hi"}
    assert out["max_tokens"] == 100

def test_system_block_list_concatenated():
    body = {"messages": [], "system": [{"type": "text", "text": "a"},
                                       {"type": "text", "text": "b"}]}
    out = anthropic_to_openai_request(body, MC)
    assert out["messages"][0] == {"role": "system", "content": "a\nb"}

def test_tool_use_and_result_roundtrip_into_openai():
    body = {"messages": [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "bash",
             "input": {"command": "ls"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.txt"}]},
    ]}
    out = anthropic_to_openai_request(body, MC)
    asst = out["messages"][0]
    assert asst["role"] == "assistant"
    assert asst["tool_calls"][0]["id"] == "toolu_1"
    assert asst["tool_calls"][0]["function"]["name"] == "bash"
    assert asst["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'
    tool_msg = out["messages"][1]
    assert tool_msg == {"role": "tool", "tool_call_id": "toolu_1", "content": "a.txt"}

def test_tools_mapped_and_tool_choice_defaulted():
    body = {"messages": [{"role": "user", "content": "go"}],
            "tools": [{"name": "bash", "description": "run",
                       "input_schema": {"type": "object",
                                        "properties": {"command": {"type": "string"}}}}]}
    out = anthropic_to_openai_request(body, MC)
    fn = out["tools"][0]["function"]
    assert out["tools"][0]["type"] == "function"
    assert fn["name"] == "bash"
    assert fn["parameters"]["properties"]["command"]["type"] == "string"
    assert out["tool_choice"] == "auto"

def test_tool_choice_defaulted_even_without_guided_decoding():
    """Guided-decoding constraint: tools present + no tool_choice always yields
    tool_choice='auto', independent of the per-model guided_decoding flag."""
    mc = ModelConfig(alias="local/x", backend="b", model="Qwen/X",
                     guided_decoding=False)
    body = {"messages": [{"role": "user", "content": "go"}],
            "tools": [{"name": "bash", "description": "run",
                       "input_schema": {"type": "object", "properties": {}}}]}
    out = anthropic_to_openai_request(body, mc)
    assert out["tool_choice"] == "auto"

def test_tool_choice_none_mapped():
    body = {"messages": [{"role": "user", "content": "go"}],
            "tool_choice": {"type": "none"},
            "tools": [{"name": "bash", "description": "run",
                       "input_schema": {"type": "object", "properties": {}}}]}
    out = anthropic_to_openai_request(body, MC)
    assert out["tool_choice"] == "none"

def test_stop_sequences_mapped():
    body = {"messages": [], "stop_sequences": ["STOP"]}
    out = anthropic_to_openai_request(body, MC)
    assert out["stop"] == ["STOP"]
