"""Clean benchmark KBs for a fresh test run."""
from rag_mcp.knowledge_manager import KnowledgeManager

km = KnowledgeManager()
keep = {"高数", "线性代数", "childfluencer"}

for kb in list(km.list()):
    name = kb["name"]
    if name not in keep:
        print(f"delete: {name}")
        km.delete_database(name)

print("remaining:")
for kb in km.list():
    print(f"  {kb['name']}")
