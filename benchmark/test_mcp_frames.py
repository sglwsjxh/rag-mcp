"""Test RAG-MCP features via real MCP protocol (stdio transport).

Tests:
1. list_knowledge_bases
2. add_database + add_file (with hash dedup)
3. search (full = grouped dict, single = flat list)
4. confidence annotation on results

Usage:
    pip install -e ".[benchmark]"
    python benchmark/test_mcp_frames.py
"""
import json
import asyncio
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client


SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["-m", "rag_mcp"],
)

# Sample texts mimicking different reasoning categories
SAMPLE_TEXTS = {
    "numerical": [
        "Tokyo has a population of 13.9 million people as of 2023. "
        "It is the most populous city in Japan and one of the largest in the world. "
        "The population density is approximately 6,158 people per square kilometer. "
        "New York City has 8.4 million people with a density of 10,194 per square kilometer.",
        "The GDP of France was $2.63 trillion in 2019. "
        "By 2022, it had grown to $2.78 trillion. "
        "The GDP per capita in 2022 was $41,000. "
        "France ranks 7th globally by nominal GDP.",
    ],
    "temporal": [
        "World War II began in 1939 and ended in 1945. "
        "The Cold War started in 1947 and continued until 1991. "
        "The Berlin Wall fell in 1989. "
        "The Soviet Union dissolved on December 26, 1991.",
        "The iPhone was first released in 2007. "
        "By 2010, smartphones had become mainstream. "
        "As of 2023, over 6 billion people use smartphones worldwide.",
    ],
    "multi_constraint": [
        "The Eiffel Tower was built in 1889 and is located in Paris, France. "
        "It is 330 meters tall and was the tallest structure in the world until 1930. "
        "The tower has three levels for visitors, with restaurants on each level.",
        "Python is a high-level programming language created by Guido van Rossum in 1991. "
        "It emphasizes code readability and supports multiple programming paradigms. "
        "Python 3.0 was released in 2008 and is not backward compatible with Python 2.",
    ],
    "tabular": [
        "The periodic table organizes elements by atomic number. "
        "Hydrogen (H) is element 1, Helium (He) is element 2. "
        "Oxygen (O) is element 8, and Gold (Au) is element 79. "
        "There are 118 confirmed elements as of 2024.",
        "In baseball, a home run is scored when the batter hits the ball and "
        "successfully rounds all bases. The record for most home runs in a season "
        "is 73, set by Barry Bonds in 2001. Hank Aaron holds the career record with 755.",
    ],
    "postprocessing": [
        "The top 5 highest-grossing films of all time are: "
        "1. Avatar ($2.92B), 2. Avengers: Endgame ($2.79B), "
        "3. Avatar: The Way of Water ($2.32B), 4. Titanic ($2.26B), "
        "5. Star Wars: The Force Awakens ($2.07B). "
        "All numbers are in USD and include worldwide box office.",
        "Sorting algorithms arrange data in a specific order. "
        "Bubble sort has O(n^2) complexity. "
        "Quick sort averages O(n log n). "
        "Merge sort also has O(n log n) but uses more memory.",
    ],
}

# Test queries that target each category
TEST_QUERIES = [
    ("numerical", "What is the population of Tokyo compared to New York?"),
    ("temporal", "When did World War II start and end?"),
    ("multi_constraint", "Tell me about the Eiffel Tower"),
    ("tabular", "What is element 1 on the periodic table?"),
    ("postprocessing", "What are the highest grossing films?"),
    ("all", "What is a sorting algorithm?"),  # cross-category query
]


def parse_result(result) -> any:
    """FastMCP may split list[dict] into multiple TextContent items.
    Collect and re-join them into the expected Python type."""
    items = []
    for content in result.content:
        try:
            data = json.loads(content.text)
            items.append(data)
        except (json.JSONDecodeError, AttributeError):
            items.append(content.text)
    if len(items) == 1:
        return items[0]
    # If all items are dicts, return them as a list
    if all(isinstance(i, dict) for i in items):
        return items
    return items


async def call_tool(session, tool_name: str, args: dict) -> any:
    result = await session.call_tool(tool_name, args)
    return parse_result(result)


async def run_tests():
    print("=" * 60)
    print("RAG-MCP Benchmark Test via MCP Protocol")
    print("=" * 60)

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ── 1. List tools ──
            print("\n[1] Listing available tools...")
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"  Tools: {tool_names}")
            assert "list_knowledge_bases" in tool_names
            assert "add_database" in tool_names
            assert "add_file" in tool_names
            assert "search" in tool_names
            assert "delete_file" in tool_names
            assert "delete_database" in tool_names
            print("  PASS")

            # ── 2. List KBs (initial) ──
            print("\n[2] Listing knowledge bases...")
            result = await session.call_tool("list_knowledge_bases", {})
            content = result.content[0].text
            print(f"  KBs: {content}")

            # ── 3. Create KBs from sample texts ──
            print("\n[3] Creating knowledge bases and adding files...")
            for category, texts in SAMPLE_TEXTS.items():
                # Create KB
                await session.call_tool("add_database", {
                    "name": category,
                    "description": f"Test KB for {category} reasoning",
                })
                print(f"  Created KB: {category}")

                # Add files
                for idx, text in enumerate(texts):
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".txt", delete=False, encoding="utf-8"
                    ) as f:
                        f.write(text)
                        tmp_path = f.name

                    result = await session.call_tool("add_file", {
                        "collection_name": category,
                        "file_path": tmp_path,
                    })
                    chunk_info = result.content[0].text
                    print(f"    Added doc {idx+1}: {chunk_info}")
                    Path(tmp_path).unlink(missing_ok=True)

            # ── 4. Test hash dedup ──
            print("\n[4] Testing hash dedup (add same file twice)...")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write("This is duplicate content for testing hash dedup.")
                dup_path = f.name

            r1 = await call_tool(session, "add_file", {
                "collection_name": "numerical",
                "file_path": dup_path,
            })
            r2 = await call_tool(session, "add_file", {
                "collection_name": "numerical",
                "file_path": dup_path,
            })
            print(f"  First add: {r1}")
            print(f"  Second add: {r2}")
            assert "0 个片段" in r2, "Dedup should return 0 chunks"
            print("  PASS: dedup works")
            Path(dup_path).unlink(missing_ok=True)

            # ── 5. Test single-collection search ──
            print("\n[5] Testing single-collection search (flat list)...")
            data = await call_tool(session, "search", {
                "query": "population of Tokyo",
                "collection_name": "numerical",
            })
            is_list = isinstance(data, list)
            print(f"  Result type: {'list' if is_list else 'dict'} (len={len(data) if is_list else 'N/A'})")
            assert is_list, "Single-collection search should return a list"
            if len(data) > 0:
                assert "confidence" in data[0], "Results should have confidence"
                assert "score" in data[0], "Results should have score"
                print(f"  First result confidence: {data[0].get('confidence')}")
                print(f"  First result score: {data[0].get('score')}")
                print("  PASS: single search returned flat list with confidence")

            # ── 6. Test full search (grouped dict) ──
            print("\n[6] Testing full search (grouped dict)...")
            data = await call_tool(session, "search", {
                "query": "population",
                "collection_name": "",
            })
            is_dict = isinstance(data, dict)
            print(f"  Result type: {'dict' if is_dict else 'list'}")
            if is_dict:
                print(f"  Keys ({len(data)}): {list(data.keys())}")
                for k, v in data.items():
                    print(f"    {k}: {len(v)} results")
            assert is_dict, "Full search should return a dict"
            print("  PASS: full search returned grouped dict")

            # ── 7. Test confidence annotation ──
            print("\n[7] Checking confidence annotation...")
            data = await call_tool(session, "search", {
                "query": "Tokyo population",
                "collection_name": "numerical",
            })
            if isinstance(data, list) and len(data) > 0:
                first = data[0]
                has_conf = "confidence" in first
                has_score = "score" in first
                print(f"  Has confidence: {has_conf} (value: {first.get('confidence', 'N/A')})")
                print(f"  Has score: {has_score} (value: {first.get('score', 'N/A')})")
                if has_conf and has_score:
                    print("  PASS: confidence annotation present")

            # ── 8. Test search result shapes ──
            print("\n[8] Testing search result shapes...")
            # Full search with nonsense query — should still return dict type
            data_full = await call_tool(session, "search", {
                "query": "xyznonexistent12345",
                "collection_name": "",
            })
            print(f"  Full search type: {type(data_full).__name__}, keys={len(data_full) if isinstance(data_full, dict) else 'N/A'}")
            assert isinstance(data_full, dict), "Full search should return dict"
            print("  PASS: full search returns dict")

            # Single-collection empty search — should return list type
            data_single = await call_tool(session, "search", {
                "query": "xyznonexistent12345",
                "collection_name": "numerical",
            })
            print(f"  Single search type: {type(data_single).__name__}, len={len(data_single) if isinstance(data_single, list) else 'N/A'}")
            assert isinstance(data_single, list), "Single search should return list"
            print("  PASS: single search returns list")

            # ── 9. Test delete_file + re-add ──
            print("\n[9] Testing delete_file + re-add...")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write("Temporary document for delete test.")
                del_path = f.name

            add_r = await call_tool(session, "add_file", {
                "collection_name": "numerical",
                "file_path": del_path,
            })
            print(f"  Added: {add_r}")

            del_r = await call_tool(session, "delete_file", {
                "collection_name": "numerical",
                "file_name": Path(del_path).name,
            })
            print(f"  Deleted: {del_r}")

            add2_r = await call_tool(session, "add_file", {
                "collection_name": "numerical",
                "file_path": del_path,
            })
            print(f"  Re-added: {add2_r}")
            assert "已导入" in add2_r and "0 个片段" not in add2_r, \
                f"Re-add should succeed after delete, got: {add2_r}"
            print("  PASS: delete+re-add cycle works (hash was cleaned)")
            Path(del_path).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_tests())
