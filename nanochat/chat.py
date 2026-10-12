"""Shared chat-conversation preparation utilities."""

import copy


def message_content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part["text"] for part in content)
    raise TypeError(f"Unsupported message content type: {type(content)}")


def prepare_completion_messages(conversation):
    """Normalize a conversation for assistant completion generation."""
    conversation = copy.deepcopy(conversation)
    messages = conversation["messages"]
    assert messages, "Conversation has no messages"
    if messages[-1]["role"] == "assistant":
        messages.pop()
    assert messages[-1]["role"] == "user", "Last message must be from the Assistant or user"
    if messages[0]["role"] == "system":
        assert len(messages) >= 2 and messages[1]["role"] == "user", \
            "System message must be followed by a user message"
        messages[1]["content"] = (
            message_content_to_text(messages[0]["content"])
            + "\n\n"
            + message_content_to_text(messages[1]["content"])
        )
        messages = messages[1:]
    return messages
