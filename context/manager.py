"""
Conversation context management.

Manages history between user and AI agent, tracks token counts.
"""
import os
from dataclasses import dataclass
from typing import Any, List
from prompts.system import get_system_prompt
from utils.text import count_tokens


@dataclass
class MessageItem:
    """Single message with role, content, and token count."""
    role: str
    content: str
    token_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LLM APIs."""
        result: dict[str, Any] = {"role": self.role}
        if self.content:
            result['content'] = self.content
        return result


class ContextManager:
    """Manages conversation context, system prompt, and message history."""
    
    def __init__(self) -> None:
        """Initialize with system prompt and empty history."""
        self._system_prompt = get_system_prompt()
        self._messages: List[MessageItem] = []
        self._model_name = os.environ.get("MODEL_NAME", 'openai/gpt-oss-20b:free')
    
    def add_user_message(self, content: str) -> None:
        """Add user message to history."""
        item = MessageItem(
            role='user',
            content=content,
            token_count=count_tokens(content, self._model_name),
        )
        self._messages.append(item)

    def add_assistant_message(self, content: str) -> None:
        """Add assistant message to history."""
        item = MessageItem(
            role='assistant',
            content=content,
            token_count=count_tokens(content, self._model_name),
        )
        self._messages.append(item)
    
    def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages formatted for LLM API."""
        messages = []

        if self._system_prompt:
            messages.append({
                'role': 'system',
                'content': self._system_prompt,
            })
        
        for item in self._messages:
            messages.append(item.to_dict())
        
        return messages
