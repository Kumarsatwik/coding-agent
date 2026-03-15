"""
Response structures for LLM client streaming.

Defines data structures for streaming responses: text chunks, token usage, errors.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


@dataclass
class TextDelta:
    """Chunk of text from streaming response."""
    content: str

    def __str__(self) -> str:
        return self.content


@dataclass
class TokenUsage:
    """Token usage statistics for API requests."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: TokenUsage):
        """Combine two TokenUsage objects."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        ) 


class StreamEventType(str, Enum):
    """Event types in streaming response."""
    TEXT_DELTA = "text_delta"
    MESSAGE_COMPLETE = "message_complete"
    ERROR = "error"
    DONE = "done"


@dataclass
class StreamEvent:
    """Single event in streaming response."""
    type: StreamEventType
    text_delta: TextDelta | None = None
    error: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None

    @classmethod
    def stream_error(cls, error: str, *, usage: TokenUsage | None = None, finish_reason: str | None = "error"):
        """Create ERROR event."""
        return cls(
            type=StreamEventType.ERROR,
            error=error,
            usage=usage,
            finish_reason=finish_reason
        )
