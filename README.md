# RAG MCP Server

一个 RAG 知识库的 MCP server，给 opencode 加上**向量检索**和**重排**能力

> 本项目开发目的是为了在 opencode 中使用 rag 文档检索功能，但是使用的是标准的 mcp 格式，理论上在任何支持 mcp 的平台上都能用

## 设计架构

- **多模态支持**：可以采用 VL 的向量和重排模型
- **多知识库**：可以建多个知识库，分类存放不同类型的文档
- **本地存储**：向量数据和源文件都保存在本地
- **图片 URL 支持**：得益于 `openai` 统一接口，base64 和 url 都支持且可以混用

```txt
┌──────────┐         ┌──────────────────────────────────────────────────┐
│ Opencode │ ──────> │                  RAG MCP Server                  │
│ (Client) │         │                                                  │
└──────────┘         │  ┌────────┐      ┌──────────┐     ┌───────────┐  │
                     │  │ Router │ ───> │ Embedder │ ──> │ Chroma DB │  │
                     │  └────────┘      └──────────┘     └───────────┘  │
                     │      │                │                 │        │
                     │      │                └─────────────────┘        │
                     │      │                         │                 │
                     │      │                    ┌──────────┐           │
                     │      └──────────────────> │  Search  │           │
                     │                           └──────────┘           │
                     │                                │                 │
                     │                           ┌──────────┐           │
                     │                           │  Rerank  │           │
                     │                           └──────────┘           │
                     └────────────────────────────────┼─────────────────┘
                                                      │
                                           ┌─────────────────────┐
                                           │  Matched Docs       │
                                           │  + Context + Answer │
                                           └─────────────────────┘
```

## 安装

```bash
git clone https://github.com/sglwsjxh/rag-mcp.git
cd rag-mcp
python -m venv venv
.\venv\Scripts\activate
pip install -e .
```

## 配置

复制 `.env.example` 到 `.env`

```bash
cp .env.example .env
```

填上 API key：

```txt
EMBEDDING_BASEURL=https://openrouter.ai/api/v1
EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
EMBEDDING_API_KEY=sk-or-...
RERANK_BASEURL=https://ai.api.nvidia.com/v1
RERANK_MODEL=nvidia/llama-nemotron-rerank-vl-1b-v2
RERANK_API_KEY=nvapi-...
DATABASE_PATH=./database
```

> 推荐使用 `OpenRouter` 和 `NVIDIA` 的免费模型，但是任何 `openai` 兼容的模型都可使用

## 注册到 opencode

在 `~/.config/opencode/opencode.json` 里加这段：

```json
{
  "mcp": {
    "rag": {
      "enabled": true,
      "type": "local",
      "command": [
        "C:/path/to/rag-mcp/venv/Scripts/python.exe",
        "-m",
        "rag_mcp"
      ]
    }
  }
}
```

重启 opencode，就能直接调这六个工具：

- `rag_list_knowledge_bases` - 列出所有知识库
- `rag_add_database` - 新建知识库
- `rag_add_file` - 往知识库里加文件
- `rag_search` - 搜索知识库
- `rag_delete_file` - 删除某个文件
- `rag_delete_database` - 删除整个知识库

## 数据目录结构

```
database/
├── index.json       # 知识库列表和元信息
├── python/          # 单文件夹知识库
│   ├── chroma_db/
│   └── assets/
├── cpp/
│   ├── chroma_db/
│   └── assets/
└── ...
```

## 许可证

[MIT License](LICENSE)
