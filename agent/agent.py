"""
Core Agent implementation.

Orchestrates interaction between user, LLM, and context.
"""
from __future__ import annotations
from typing import AsyncGenerator
from client.llm_client import LLMClient
from client.response import StreamEventType
from agent.events import AgentEvent, AgentEventType
from context.manager import ContextManager


class Agent:
    """Main AI agent for processing messages and generating responses."""
    
    def __init__(self):
        """Initialize agent with LLM client and context manager."""
        self.client = LLMClient()
        self.context_manager = ContextManager()

    
    async def run(self, message: str):
        """Run agent with user message and yield events."""
        yield AgentEvent.agent_start(message)
        
        self.context_manager.add_user_message(message)
        
        final_response: str | None = None
        
        async for event in self._agentic_loop(message):
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")
        
        yield AgentEvent.agent_end(final_response)

    async def _agentic_loop(self, message: str) -> AsyncGenerator[AgentEvent, None]:
        """Main loop for communicating with LLM and streaming responses."""
        response_text = ""

        async for event in self.client.chat_completion(self.context_manager.get_messages(), True):
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:
                    content = event.text_delta.content
                    response_text += content
                    yield AgentEvent.text_delta(content)
                    
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(event.error or "Unknown error occured")

        if response_text:
            self.context_manager.add_assistant_message(response_text)

        if response_text:
            yield AgentEvent.text_complete(response_text)

    async def __aenter__(self) -> Agent:
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit and cleanup."""
        if self.client:
            await self.client.close()
            self.client = None
