__version__ = "1.0.0"


def main() -> None:
    """Entry point for ``rag-mcp`` console script."""
    from rag_mcp.server import main as server_main

    server_main()
