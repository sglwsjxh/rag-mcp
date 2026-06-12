# google/frames-benchmark 评测教程

> 用这个评测集来测试 RAG-MCP 的4个新功能：分组搜索、置信度标注、文件去重、整体检索质量。

---

## 0. 先装依赖

```bash
# 激活虚拟环境后
pip install -e ".[benchmark]"
```

这行会装 `datasets>=3.0`，HuggingFace 的数据集加载库。

顺便确认 pytest 也能用（评测里会用到断言来验证返回形状）：

```bash
pip install -e ".[test]"
```

---

## 1. 数据长什么样

```python
from datasets import load_dataset

ds = load_dataset("google/frames-benchmark")
print(ds)
# DatasetDict({
#     train: Dataset({ features: ['Question', 'Answer', 'Answer_type', ...], num_rows: 659 })
#     test:  Dataset({ features: ['Question', 'Answer', ...], num_rows: 165 })
#     corpus: Dataset({ features: ['doc_id', 'title', 'text', 'url'], num_rows: 6836 })
# })
```

三个分集：

| 分集 | 行数 | 内容 |
|------|------|------|
| `train` | 659 | 训练查询 + 答案 + 推理类型标签 |
| `test` | 165 | 测试查询 + 答案（没有推理类型标签） |
| `corpus` | 6,836 | Wikipedia 文章，用作检索库 |

**train 的一条数据示例：**

```python
{
  "Question": "Which city had a higher population in 2020, Tokyo or New York?",
  "Answer": "Tokyo",
  "Answer_type": "numerical",
  "Answer_Type_Key": "comparison",
  "complexity_level": "medium",
  "relevant_docs": ["url:https://en.wikipedia.org/wiki/Tokyo",
                     "url:https://en.wikipedia.org/wiki/New_York_City"],
  "Reasoning_Steps": ["Tokyo population: 13.9 million",
                      "New York population: 8.4 million",
                      "13.9 > 8.4, so Tokyo has higher population"]
}
```

**推理类型（Answer_type）：**

| 类型 | 说明 | 示例 |
|------|------|------|
| `numerical` | 数值比较/计算 | GDP、人口对比 |
| `tabular` | 表格推理 | 从表格找最大值 |
| `multi_constraint` | 多条件约束 | 2020年后成立、员工>1000的公司 |
| `temporal` | 时序推理 | 2010 vs 2020 变化趋势 |
| `postprocessing` | 后处理推理 | 排序、前N个、百分比计算 |
| `other` | 其他 | 综合推理 |

---

## 2. 按推理类型建知识库

推荐把 6 种推理类型分别建库，这样能测 grouped search。

创建一个 Python 脚本 `benchmark/load_frames.py`：

```python
"""加载 google/frames-benchmark，按推理类型建知识库。"""
import json
import tempfile
from pathlib import Path
from datasets import load_dataset
from rag_mcp.knowledge_manager import KnowledgeManager


def setup_knowledge_bases(ds, km):
    """按推理类型建 6 个知识库，导入相关 Wikipedia 文章。"""
    # 先收集每个类型涉及哪些 doc_id
    answer_types = ["numerical", "tabular", "multi_constraint",
                    "temporal", "postprocessing", "other"]
    type_to_docs = {t: set() for t in answer_types}

    for row in ds["train"]:
        at = row["Answer_type"]
        if at in type_to_docs:
            for url in row["relevant_docs"]:
                type_to_docs[at].add(url)

    # 建 KB + 导入文章
    corpus_map = {row["url"]: row["text"]
                  for row in ds["corpus"]}

    for at, urls in type_to_docs.items():
        print(f"\n📦 建知识库: {at}（{len(urls)} 篇文章）")
        km.add_database(at, f"Frames benchmark - {at}")

        for url in urls:
            text = corpus_map.get(url, "")
            if not text:
                continue

            # 写成临时 txt 文件再导入
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write(text)
                tmp_path = f.name

            n = km.add_file(at, tmp_path)
            Path(tmp_path).unlink()  # 删临时文件
            print(f"  {url.split('/')[-1][:30]:30s} → {n} chunks")


if __name__ == "__main__":
    print("🔄 加载 google/frames-benchmark...")
    ds = load_dataset("google/frames-benchmark")

    km = KnowledgeManager()
    setup_knowledge_bases(ds, km)

    print("\n✅ 完成！知识库列表：")
    for kb in km.list():
        print(f"  - {kb['name']}: {kb['desc']}")
```

运行：

```bash
python benchmark/load_frames.py
```

> **注意**：第一次运行会下载数据集（~几十MB），之后会缓存到 `~/.cache/huggingface/datasets/`。
> 6 个知识库共约 6,836 篇文章，导入需要一些时间。

---

## 3. 跑查询测试

### 3.1 测试 grouped search（全库搜索）

```python
from rag_mcp.knowledge_manager import KnowledgeManager

km = KnowledgeManager()

# 全库搜索（不指定 collection_name）
query = "Which city had a higher population, Tokyo or New York?"
result = km.search(query)

print(type(result))  # 应该是 dict[str, list[dict]]
for kb_name, docs in result.items():
    print(f"\n📁 {kb_name}: {len(docs)} 条结果")
    for d in docs[:3]:
        print(f"  [{d['confidence']:7s}] score={d['score']:.3f}  {d['text'][:60]}...")
```

**预期结果：**

```
<class 'dict'>
📁 numerical: 5 条结果
  [high  ] score=0.921  Tokyo is the capital of Japan with a population...
  [high  ] score=0.887  New York City is the most populous city in...
  [low   ] score=0.234  The GDP of Japan in 2020 was...
```

### 3.2 测试单库搜索（保持兼容）

```python
# 指定 collection_name → 返回 flat list
result = km.search(query, collection_name="numerical")
print(type(result))  # 应该是 list[dict]
print(f"{len(result)} 条结果")
for d in result[:3]:
    print(f"  [{d['confidence']:7s}] score={d['score']:.3f}")
```

### 3.3 测试空结果

```python
# 搜一个明显不相关的问题
result = km.search("quantum physics string theory", collection_name="numerical")
print(type(result))  # 应该是 list
print(result)        # 应该是 []
```

### 3.4 测试 hash 去重

```python
# 同一个文件加两次
n1 = km.add_file("numerical", "/path/to/tokyo_article.txt")
n2 = km.add_file("numerical", "/path/to/tokyo_article.txt")
print(f"第一次: {n1} chunks")   # > 0
print(f"第二次: {n2} chunks")   # 应该是 0
```

---

## 4. 跑自动化评测（评估检索质量）

创建一个评测脚本 `benchmark/evaluate_frames.py`：

```python
"""用 frames-benchmark 的 train 集评估检索质量。"""
from datasets import load_dataset
from rag_mcp.knowledge_manager import KnowledgeManager


def evaluate():
    ds = load_dataset("google/frames-benchmark")
    km = KnowledgeManager()

    # 建立 URL → answer_type 的映射
    url_to_type = {}
    for row in ds["train"]:
        for url in row["relevant_docs"]:
            url_to_type[url] = row["Answer_type"]

    results = {"total": 0, "correct_kb": 0, "has_high_confidence": 0}

    for row in ds["train"]:
        query = row["Question"]
        expected_type = row["Answer_type"]
        expected_urls = set(row["relevant_docs"])

        # 全库搜索
        grouped = km.search(query)

        results["total"] += 1

        # 检查正确答案是否出现在对应的 KB 里
        if expected_type in grouped and len(grouped[expected_type]) > 0:
            results["correct_kb"] += 1

        # 检查 top-1 是否 high confidence
        all_docs = [d for docs in grouped.values() for d in docs]
        if all_docs and all_docs[0].get("confidence") == "high":
            results["has_high_confidence"] += 1

    print(f"\n📊 评测结果（{results['total']} 条查询）")
    print(f"  ✅ 命中正确 KB:     {results['correct_kb']}/{results['total']}"
          f" ({results['correct_kb']/results['total']*100:.1f}%)")
    print(f"  ✅ Top-1 高置信度:  {results['has_high_confidence']}/{results['total']}"
          f" ({results['has_high_confidence']/results['total']*100:.1f}%)")


if __name__ == "__main__":
    evaluate()
```

运行：

```bash
python benchmark/evaluate_frames.py
```

> **注意**：评测需要所有 KB 已经建好，先跑 `load_frames.py` 再跑评测。

---

## 5. 各功能验证清单

| # | 功能 | 验证方法 | 预期 |
|---|------|---------|------|
| 1 | **全库搜索 → grouped dict** | `km.search("query")` | 返回 `{kb_name: [docs]}` |
| 2 | **单库搜索 → flat list** | `km.search("query", "numerical")` | 返回 `[docs]` |
| 3 | **空全库 → {}** | `km.search("nonsense query")` | 返回 `{}` |
| 4 | **空单库 → []** | `km.search("nonsense", "numerical")` | 返回 `[]` |
| 5 | **confidence 字段** | 看返回结果 | 每个 doc 有 `confidence: "high"/"low"/"unknown"` |
| 6 | **hash 去重** | 同文件加两次 | 第二次返回 `0` |
| 7 | **delete 清理 hash** | 删文件再重加 | 能加进去，返回 `> 0` |
| 8 | **confidence 类型分布** | 跑 `evaluate_frames.py` | 数值类比多约束类有更多 high |

---

## 6. 补充：用 BEIR 格式做标准评测

如果你想和学术界标准对标（`nDCG@10`, `Recall@100` 等），frames-benchmark 也可以用 BEIR 格式跑：

```python
# 转换成 BEIR 格式
corpus = []
for row in ds["corpus"]:
    corpus.append({
        "_id": row["doc_id"],
        "title": row["title"],
        "text": row["text"]
    })

queries = []
qrels = {}
for i, row in enumerate(ds["train"]):
    qid = f"q_{i}"
    queries.append({"_id": qid, "text": row["Question"]})
    for url in row["relevant_docs"]:
        doc_id = f"doc_{hash(url)}"
        qrels.setdefault(qid, {})[doc_id] = 1
```

然后用 `beir` 库的标准评测函数算指标。不过这步比较进阶，初期先跑前面的功能测试就够了。

---

## 常见问题

**Q: 下载数据集很慢怎么办？**
第一次要下 ~几十 MB，之后就缓存了。也可以去 [HuggingFace 页面](https://huggingface.co/datasets/google/frames-benchmark) 手动下载。

**Q: 6,836 篇文章全导入会不会太久？**
会。可以只选部分推理类型先测，或者限制每个 KB 的文章数量：

```python
for at, urls in type_to_docs.items():
    limited = list(urls)[:50]  # 每类只取前 50 篇
    ...
```

**Q: 怎么只测 grouped search 不看具体结果？**
用 `assert` 做自动验证：

```python
result = km.search("test query")
assert isinstance(result, dict), "全库搜索应该返回 dict"
assert len(result) > 0, "应该有结果"
```
