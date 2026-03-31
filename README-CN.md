# nanoMemory

**如果你能读懂 ~200 行 Python，你就能理解 Agent 记忆。**

9 种 Agent 记忆架构的渐进式演示——每种方法独立成文件，约 80-150 行 Python。

[English](README.md)

---

## 这是什么

大多数 Agent 记忆教程只展示一种方法。这个项目展示了**九种**，每种都是构建记忆的本质不同方式：

| Level | 方法 | 文件 | 行数 |
|:-----:|------|------|:----:|
| 0 | 无记忆 | `agent.py` | ~30 |
| 1 | 文本+关键词 | `memory_file.py` | ~80 |
| 2 | 向量嵌入 | `memory_vector.py` | ~100 |
| 3 | 认知评分 | `memory_scored.py` | ~120 |
| 4 | 知识图谱 | `memory_graph.py` | ~150 |
| 5 | 摘要压缩 | `memory_summary.py` | ~80 |
| 6 | 层次化记忆 | `memory_hierarchical.py` | ~100 |
| 7 | 自主生命周期 | `memory_lifecycle.py` | ~120 |
| 8 | 生产工具集成 | `memory_production.py` | ~120 |

---

## 为什么做这个

Agent 记忆系统正在快速发展，但通常以黑盒形式呈现。这个项目采取相反的立场：

- **每个 Level 就一个文件。** Level 之间没有互相导入。从头读到尾。
- **每个 Level 是不同的范式。** 不是功能堆叠——是存储和检索记忆的本质不同方式。
- **每个文件都标注了参考来源。** 学术论文、GitHub 仓库和综述引用都在文件头。

---

## 快速开始

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="your-key"
# 可选：
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_EMBED_MODEL="text-embedding-3-small"

python agent.py "你好，我是 Alice"
python memory_file.py "我喜欢深色模式"
python memory_vector.py "你知道关于我的什么？"
python memory_scored.py "提醒我之前讨论了什么"
python memory_graph.py "Alice 搬到了东京"
python memory_summary.py "让我告诉你我的项目"
python memory_hierarchical.py "我一直在学 Rust"
python memory_lifecycle.py "忘掉我的旧地址，我搬家了"
python memory_production.py  # 打印对比表
```

---

## 各级详解

### Level 0: 无记忆
基线 Agent，支持工具调用。对话之间无状态持久化。

### Level 1: 文本+关键词
LLM 提取事实存入 JSONL 文件，关键词匹配检索。最简单的跨会话记忆。

### Level 2: 向量嵌入
OpenAI embedding + numpy 余弦相似度替代关键词匹配。语义搜索，而非字符串匹配。

### Level 3: 认知评分
三因子检索：`alpha*相似度 + beta*时效性 + gamma*重要性`。艾宾浩斯遗忘曲线。反思机制将记忆提炼为洞察。

### Level 4: 知识图谱
SQLite 中的主谓宾三元组。支持时序推理（`valid_from`/`valid_until`）、实体归一化、矛盾检测。

### Level 5: 摘要压缩
LLM 将对话压缩为摘要。定期合并旧摘要。用保真度换容量。

### Level 6: 层次化记忆
三层层次结构：原始消息 → 情节摘要 → 主题洞察。自顶向下检索模拟人类回忆：先要点，后细节。

### Level 7: 自主生命周期
Agent 自己通过工具调用控制记忆：`save`、`delete`、`update`、`search`。没有被动的自动提取——Agent 自己决定记住什么、忘记什么。

### Level 8: 生产工具集成
Mem0、Zep、Graphiti SDK 的并排对比。安装说明、用法示例和比较表。

```bash
pip install mem0ai         # Mem0: 事实提取 + 向量/图存储
pip install zep-cloud      # Zep: 时序知识图谱
pip install graphiti-core  # Graphiti: SPO 三元组 + Neo4j
```

---

## 架构原则

```
存储范式（存什么）
  文本 → 向量 → 图 → 摘要 → 层次结构

检索策略（怎么查）
  关键词 → 余弦相似度 → 评分 → 图遍历 → 自顶向下

控制模型（谁决定）
  自动提取 (L0-6) → Agent 自主控制 (L7) → 生产级 SDK (L8)
```

---

## 核心参考文献

- Park et al. (2023) "Generative Agents" — [arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)
- "Memory in the Age of AI Agents" 综述 — [arxiv.org/abs/2512.13564](https://arxiv.org/abs/2512.13564)
- MemGPT (Packer et al., 2023) — [arxiv.org/abs/2310.08560](https://arxiv.org/abs/2310.08560)
- MemoryBank (Zhong et al., 2024) — [arxiv.org/abs/2401.10917](https://arxiv.org/abs/2401.10917)
- xMemory (2025) 层次化检索 — [arxiv.org/abs/2502.13743](https://arxiv.org/abs/2502.13743)

---

## 许可证

MIT
