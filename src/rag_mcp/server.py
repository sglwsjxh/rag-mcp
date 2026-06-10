"""MCP server — register tools and entry point."""

from mcp.server.fastmcp import FastMCP

from .knowledge_manager import KnowledgeManager

mcp = FastMCP("rag-mcp")
km = KnowledgeManager()


@mcp.tool()
def list_knowledge_bases() -> list[dict]:
    """列出所有知识库"""
    return km.list()


@mcp.tool()
def add_database(name: str, description: str = "") -> str:
    """新建知识库"""
    km.add_database(name, description)
    return f"知识库 [{name}] 已创建"


@mcp.tool()
def add_file(collection_name: str, file_path: str) -> str:
    """往知识库里加文件"""
    n = km.add_file(collection_name, file_path)
    return f"已导入 {n} 个片段到 [{collection_name}]"


@mcp.tool()
def search(query: str, collection_name: str = "") -> list[dict]:
    """搜索知识库。不指定 collection_name 则搜索全部"""
    return km.search(query, collection_name)


@mcp.tool()
def delete_file(collection_name: str, file_name: str) -> str:
    """删除知识库中的某个文件"""
    km.delete_file(collection_name, file_name)
    return f"已从 [{collection_name}] 删除 {file_name}"


@mcp.tool()
def delete_database(name: str) -> str:
    """删除整个知识库（危险操作！会删除所有数据）"""
    km.delete_database(name)
    return f"知识库 [{name}] 已删除"


def main() -> None:
    """启动 MCP server（stdio 传输）"""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
