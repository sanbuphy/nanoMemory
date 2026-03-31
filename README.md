# nanoMemory

**If you can read ~200 lines of Python, you understand agent memory.**
如果你能读懂 ~200 行 Python，你就能理解 Agent 记忆。

A minimal, progressive demonstration of 9 different agent memory architectures — each self-contained in ~80-150 lines of Python.

9 种 Agent 记忆架构的渐进式演示——每种方法独立成文件，约 80-150 行 Python。

---

## What It Is / 这是什么

Most agent memory tutorials show one approach. This project shows **nine**, each a fundamentally different way to build memory:

大多数 Agent 记忆教程只展示一种方法。这个项目展示了**九种**，每种都是构建记忆的本质不同方式：

| Level | Method / 方法 | File / 文件 | Lines |
|:-----:|---------------|-------------|:-----:|
| 0 | No memory / 无记忆 | `agent.py` | ~30 |
| 1 | Text + Keyword / 文本+关键词 | `memory_file.py` | ~80 |
| 2 | Vector Embedding / 向量嵌入 | `memory_vector.py` | ~100 |
| 3 | Cognitive Scoring / 认知评分 | `memory_scored.py` | ~120 |
| 4 | Knowledge Graph / 知识图谱 | `memory_graph.py` | ~150 |
| 5 | Summary Compression / 摘要压缩 | `memory_summary.py` | ~80 |
| 6 | Hierarchical Memory / 层次化记忆 | `memory_hierarchical.py` | ~100 |
| 7 | Agentic Lifecycle / 自主生命周期 | `memory_lifecycle.py` | ~120 |
| 8 | Production Tools / 生产工具集成 | `memory_production.py` | ~120 |

---

## Why It Exists / 为什么做这个

Agent memory systems are rapidly evolving but often presented as black boxes. This project takes the opposite stance:

Agent 记忆系统正在快速发展，但通常以黑盒形式呈现。这个项目采取相反的立场：

- **Each level is one file.** No imports between levels. Read top to bottom.
  **每个 Level 就一个文件。** Level 之间没有互相导入。从头读到尾。
- **Each level is a different paradigm.** Not feature stacking — fundamentally different ways to store and retrieve memory.
  **每个 Level 是不同的范式。** 不是功能堆叠——是存储和检索记忆的本质不同方式。
- **Every file cites its sources.** Academic papers, GitHub repos, and survey references in the docstring.
  **每个文件都标注了参考来源。** 学术论文、GitHub 仓库和综述引用都在文件头。

---

## Quick Start / 快速开始

```bash
# Install / 安装
pip install -r requirements.txt

# Set API key / 设置 API 密钥
export OPENAI_API_KEY="your-key"
# Optional: 自定义 endpoint / custom endpoint
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_EMBED_MODEL="text-embedding-3-small"

# Run any level / 运行任意 Level
python agent.py "Hello, I'm Alice"
python memory_file.py "I prefer dark mode"
python memory_vector.py "What do you know about me?"
python memory_scored.py "Remind me what we discussed"
python memory_graph.py "Alice moved to Tokyo"
python memory_summary.py "Let me tell you about my project"
python memory_hierarchical.py "I've been learning Rust"
python memory_lifecycle.py "Forget my old address, I moved"
python memory_production.py  # prints comparison table
```

---

## Level Details / 各级详解

### Level 0: No Memory / 无记忆
Baseline agent with tool use. No state persists between conversations.
基线 Agent，支持工具调用。对话之间无状态持久化。

### Level 1: Text + Keyword / 文本+关键词
LLM extracts facts → JSONL file → keyword matching for retrieval. The simplest cross-session memory.
LLM 提取事实 → JSONL 文件 → 关键词匹配检索。最简单的跨会话记忆。

### Level 2: Vector Embedding / 向量嵌入
OpenAI embeddings + numpy cosine similarity replace keyword matching. Semantic search, not just string matching.
OpenAI embedding + numpy 余弦相似度替代关键词匹配。语义搜索，而非字符串匹配。

### Level 3: Cognitive Scoring / 认知评分
Three-factor retrieval: `α·similarity + β·recency + γ·importance`. Ebbinghaus-inspired decay. Reflection mechanism distills memories into insights.
三因子检索：`α·相似度 + β·时效性 + γ·重要性`。艾宾浩斯遗忘曲线。反思机制将记忆提炼为洞察。

### Level 4: Knowledge Graph / 知识图谱
(SPO) triples in SQLite. Temporal reasoning with `valid_from`/`valid_until`. Entity normalization. Contradiction detection.
SQLite 中的主谓宾三元组。支持时序推理、实体归一化、矛盾检测。

### Level 5: Summary Compression / 摘要压缩
LLM compresses conversations into summaries. Periodic consolidation merges old summaries. Trades fidelity for capacity.
LLM 将对话压缩为摘要。定期合并旧摘要。用保真度换容量。

### Level 6: Hierarchical Memory / 层次化记忆
3-level hierarchy: raw → episode → theme. Top-down retrieval mirrors human recall: gist first, then details.
三层层次结构：原始 → 情节 → 主题。自顶向下检索模拟人类回忆：先要点，后细节。

### Level 7: Agentic Lifecycle / 自主生命周期
The agent itself controls memory via tool calls: `save`, `delete`, `update`, `search`. No passive auto-extraction.
Agent 自己通过工具调用控制记忆：`save`、`delete`、`update`、`search`。没有被动的自动提取。

### Level 8: Production Tools / 生产工具集成
Side-by-side comparison of Mem0, Zep, and Graphiti SDKs. Install instructions, usage examples, and a comparison table.
Mem0、Zep、Graphiti SDK 的并排对比。安装说明、用法示例和比较表。

```bash
pip install mem0ai         # Mem0: fact extraction + vector/graph
pip install zep-cloud      # Zep: temporal knowledge graph
pip install graphiti-core  # Graphiti: SPO triples + Neo4j
```

---

## Architecture Principles / 架构原则

```
┌─────────────────────────────────────────────────┐
│  Storage Paradigms (what you store)             │
│  存储范式（存什么）                               │
│  Text → Vector → Graph → Summary → Hierarchy   │
├─────────────────────────────────────────────────┤
│  Retrieval Strategies (how you search)          │
│  检索策略（怎么查）                               │
│  Keyword → Cosine → Scoring → Graph → Top-down │
├─────────────────────────────────────────────────┤
│  Control Model (who decides)                    │
│  控制模型（谁决定）                               │
│  Auto-extract (L0-6) → Agent-controlled (L7)   │
│  → Production SDKs (L8)                        │
└─────────────────────────────────────────────────┘
```

---

## Key References / 核心参考文献

- Park et al. (2023) "Generative Agents" — [arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)
- "Memory in the Age of AI Agents" Survey — [arxiv.org/abs/2512.13564](https://arxiv.org/abs/2512.13564)
- MemGPT (Packer et al., 2023) — [arxiv.org/abs/2310.08560](https://arxiv.org/abs/2310.08560)
- MemoryBank (Zhong et al., 2024) — [arxiv.org/abs/2401.10917](https://arxiv.org/abs/2401.10917)
- xMemory (2025) Hierarchical Retrieval — [arxiv.org/abs/2502.13743](https://arxiv.org/abs/2502.13743)

---

## License / 许可证

MIT
