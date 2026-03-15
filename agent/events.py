"""
Event types and structures for agent system.

Defines events for real-time communication between components and UI.
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Any 

from client.response import TokenUsage


class AgentEventType(str, Enum):
    """Possible event types in agent system."""
    
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"
    
    TEXT_DELTA = "text_delta"
    TEXT_COMPLETE = "text_complete"


@dataclass
class AgentEvent:
    """Represents an event during agent processing."""
    type: AgentEventType
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def agent_start(cls, message: str) -> AgentEvent:
        """Create AGENT_START event."""
        return cls(type=AgentEventType.AGENT_START, data={"message": message})

    @classmethod
    def agent_end(cls, response: str | None = None, usage: TokenUsage | None = None) -> AgentEvent:
        """Create AGENT_END event."""
        return cls(
            type=AgentEventType.AGENT_END,
            data={
                "response": response,
                "usage": usage.__dict__ if usage else None
            }
        )

    @classmethod
    def agent_error(cls, error: str, details: dict[str, Any] | None = None) -> AgentEvent:
        """Create AGENT_ERROR event."""
        return cls(
            type=AgentEventType.AGENT_ERROR,
            data={
                "error": error,
                "details": details or {}
            }
        )
    
    @classmethod
    def text_delta(cls, delta: str) -> AgentEvent:
        """Create TEXT_DELTA event."""
        return cls(type=AgentEventType.TEXT_DELTA, data={"content": delta})

    @classmethod
    def text_complete(cls, content: str) -> AgentEvent:
        """Create TEXT_COMPLETE event."""
        return cls(type=AgentEventType.TEXT_COMPLETE, data={"content": content})
    
