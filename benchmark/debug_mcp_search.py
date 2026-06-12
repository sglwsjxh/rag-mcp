"""Quick MCP debug: check what search() returns."""
import asyncio, json
from mcp import ClientSession, StdioServerParameters, stdio_client

SERVER_PARAMS = StdioServerParameters(command="python", args=["-m", "rag_mcp"])

async def main():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Search single collection
            r = await session.call_tool("search", {
                "query": "test",
                "collection_name": "高数",
                "top_k": 3,
            })
            raw_text = r.content[0].text
            data = json.loads(raw_text)
            print(f"Type: {type(data).__name__}")
            print(f"Repr: {repr(data)[:300]}")

asyncio.run(main())
