from openai import APIConnectionError, AsyncOpenAI, RateLimitError, APIError
from client.response import StreamEvent, StreamEventType, TextDelta, TokenUsage
from typing import Any, AsyncGenerator
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

class LLMClient:
    def __init__(self)->None:
        self._client: AsyncOpenAI | None = None
        self._max_retries:int = 3

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

    async def chat_completion(self, messages: list[dict[str, Any]],stream:bool=True)->AsyncGenerator[StreamEvent, None]:
        client = self.get_client()
        kwargs={
            "model":"openai/gpt-oss-120b:free",
            "messages":messages,
            "stream":stream
        }
        for attempt in range(self._max_retries+1):
            try:
                if stream :
                    async for event in self._stream_response(client, kwargs):
                        yield event
                else:
                    event = await self._non_stream_response(client, kwargs)
                    yield event
                return
            except RateLimitError as e:
                if attempt < self._max_retries:
                    wait_time = (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent.stream_error(f"RateLimitError: {self._format_error(e)}")
                    return
             
            except APIConnectionError as e:
                if attempt < self._max_retries:
                    wait_time = (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent.stream_error(f"APIConnectionError: {self._format_error(e)}")
                    return

            except APIError as e:
                yield StreamEvent.stream_error(f"APIError: {self._format_error(e)}")
                return
            except Exception as e:
                yield StreamEvent.stream_error(f"Exception: {self._format_error(e)}")
                return
    
    async def _stream_response(self,client: AsyncOpenAI, kwargs: dict[str, Any])->AsyncGenerator[StreamEvent, None]:
        usage : TokenUsage | None = None
        finish_reason :str | None = None
        had_error = False

        try:
            response = await client.chat.completions.create(**kwargs)

            async for chunk in response:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    prompt_tokens_details = getattr(chunk_usage, "prompt_tokens_details", None)
                    cached_tokens = getattr(prompt_tokens_details, "cached_tokens", 0) if prompt_tokens_details else 0

                    usage = TokenUsage(
                        prompt_tokens=chunk_usage.prompt_tokens,
                        completion_tokens=chunk_usage.completion_tokens,
                        total_tokens=chunk_usage.total_tokens,
                        cached_tokens=cached_tokens,
                    )

                if not getattr(chunk, "choices", None):
                    continue

                choice = chunk.choices[0]

                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason

                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    yield StreamEvent(
                        type=StreamEventType.TEXT_DELTA,
                        text_delta=TextDelta(content=content),
                        usage=usage,
                        finish_reason=finish_reason,
                    )
        except Exception as err:
            had_error = True
            yield StreamEvent.stream_error(self._format_error(err), usage=usage, finish_reason="error")

        yield StreamEvent(type=StreamEventType.MESSAGE_COMPLETE, usage=usage, finish_reason=finish_reason or ("error" if had_error else "stop"))
    
    async def _non_stream_response(self,client: AsyncOpenAI, kwargs: dict[str, Any]):
        usage: TokenUsage | None = None

        try:
            response = await client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message

            text_delta = None
            if message.content:
                text_delta = TextDelta(content=message.content)

            if response.usage:
                prompt_tokens_details = getattr(response.usage, "prompt_tokens_details", None)
                cached_tokens = getattr(prompt_tokens_details, "cached_tokens", 0) if prompt_tokens_details else 0

                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    cached_tokens=cached_tokens,
                )

            return StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                text_delta=text_delta,
                usage=usage,
                finish_reason=choice.finish_reason,
            )
        except Exception as err:
            return StreamEvent.stream_error(self._format_error(err), usage=usage, finish_reason="error")

    def _format_error(self, err: Any)->str:
        if isinstance(err, str):
            return err

        parts: list[str] = [f"{err.__class__.__name__}: {err}"]

        status_code = getattr(err, "status_code", None)
        if status_code is not None:
            parts.append(f"status_code={status_code}")

        request_id = getattr(err, "request_id", None)
        if request_id:
            parts.append(f"request_id={request_id}")

        response = getattr(err, "response", None)
        if response is not None:
            try:
                text = getattr(response, "text", None)
                if text:
                    parts.append(text)
            except Exception:
                pass

        body = getattr(err, "body", None)
        if body is not None:
            try:
                parts.append(str(body))
            except Exception:
                pass

        message = " | ".join(parts)
        if len(message) > 2000:
            return message[:2000] + "…"
        return message
