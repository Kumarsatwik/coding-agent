from openai import AsyncOpenAI
from client.response import StreamEvent, EventType, TextDelta, TokenUsage
from typing import AsyncGenerator
from dotenv import load_dotenv
import os

load_dotenv()

class LLMClient:
    def __init__(self)->None:
        self._client: AsyncOpenAI | None = None

    def get_client(self)->AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key= os.environ["API_KEY"], 
                base_url="https://openrouter.ai/api/v1"
            )
        return self._client

    async def close(self)->None:
        if self._client is not None:
            await self._client.close()
            self._client = None 

    async def chat_completion(self, messages: list[dict[str, any]],stream:bool=True)->AsyncGenerator[StreamEvent, None]:
        client = self.get_client()
        kwargs={
            "model":"qwen/qwen3-235b-a22b-2507",
            "messages":messages,
            "stream":stream
        }
        if stream :
            await self._stream_response(client, kwargs)
        else:
            event = await self._non_stream_response(client, kwargs)
            yield event
        return
    
    async def _stream_response(self,client: AsyncOpenAI, kwargs: dict[str, any]):
        pass
    
    async def _non_stream_response(self,client: AsyncOpenAI, kwargs: dict[str, any]):
        response = await client.chat.completions.create(**kwargs)
        print(response.choices[0].message.content)
        # return response
        choice = response.choices[0]
        message=choice.message

        text_delta=None
        if message.content:
            text_delta= TextDelta(content=message.content)
        
        if response.usage:
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens,
            )

        return StreamEvent(type=EventType.TEXT_DELTA, text_delta=text_delta, usage=usage, finish_reason=choice.finish_reason) 
