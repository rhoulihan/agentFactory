from agentfactory.errors import anthropic_error

def test_anthropic_error_shape():
    err = anthropic_error("no such model: local/foo", "not_found_error")
    assert err == {
        "type": "error",
        "error": {"type": "not_found_error", "message": "no such model: local/foo"},
    }

def test_anthropic_error_default_type():
    assert anthropic_error("bad")["error"]["type"] == "invalid_request_error"
