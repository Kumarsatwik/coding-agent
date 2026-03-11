from client.llm_client import LLMClient
import asyncio

async def main():
    client = LLMClient()
    messages = [{"role": "user", "content": "hello , how are you?"}]
    async for event in client.chat_completion(messages,False):
        print(event)

if __name__ == "__main__":
    asyncio.run(main())
