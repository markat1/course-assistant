import pytest
from pydantic import ValidationError


def test_browser_request_preserves_full_text_history():
    from app.models.browser.chat_request import BrowserChatRequest

    messages = [
        {"role": "system", "content": "Answer in Danish."},
        {"role": "user", "content": "What is KV memory?"},
        {"role": "assistant", "content": "Stored attention keys and values."},
        {"role": "user", "content": "Why does it grow?"},
    ]

    request = BrowserChatRequest.model_validate({
        "model": "course-assistant", "messages": messages,
    })

    assert request.model == "course-assistant"
    assert request.stream is False
    assert [message.model_dump() for message in request.messages] == messages


@pytest.mark.parametrize("changes", [
    {"model": "unknown-model"},
    {"stream": True},
    {"messages": []},
    {"messages": [{"role": "user", "content": "   "}]},
    {"messages": [{"role": "user", "content": None}]},
    {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
    {"messages": [{"role": "tool", "content": "Unsolicited tool result"}]},
])
def test_browser_request_rejects_unsupported_input(changes):
    from app.models.browser.chat_request import BrowserChatRequest

    payload = {
        "model": "course-assistant",
        "messages": [{"role": "user", "content": "Explain warmup."}],
        **changes,
    }

    with pytest.raises(ValidationError):
        BrowserChatRequest.model_validate(payload)
