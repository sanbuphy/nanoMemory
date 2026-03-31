[English](README.md) | 中文

> _"记忆是心灵的宝库。"_ — 托马斯·富勒

如果你能读懂 ~200 行 Python，你就能理解 Agent 记忆。

9 种 Agent 记忆架构的渐进式演示。每个 Level 独立一个文件，约 80-150 行。

## 安装

```
pip install -r requirements.txt
```

设置环境变量：

__macOS/Linux:__

```
export OPENAI_API_KEY='your-key-here'
export OPENAI_BASE_URL='https://api.openai.com/v1'  # 可选
export OPENAI_MODEL='gpt-4o-mini'  # 可选
export OPENAI_EMBED_MODEL='text-embedding-3-small'  # 可选
```

__Windows (PowerShell):__

```
$env:OPENAI_API_KEY='your-key-here'
$env:OPENAI_BASE_URL='https://api.openai.com/v1'
$env:OPENAI_MODEL='gpt-4o-mini'
```

## 快速开始

```
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

## 原理

每个 Level 展示一种本质不同的记忆构建方式——不是功能堆叠，而是不同的范式。

| Level | 文件 | 存储 | 检索 |
|:-----:|------|------|------|
| 0 | [`agent.py`](agent.py) | 无 | 无 |
| 1 | [`memory_file.py`](memory_file.py) | JSONL | 字符串匹配 |
| 2 | [`memory_vector.py`](memory_vector.py) | JSONL + embedding | 余弦相似度 |
| 3 | [`memory_scored.py`](memory_scored.py) | JSONL + embedding | alpha*相似度 + beta*时效 + gamma*重要性 |
| 4 | [`memory_graph.py`](memory_graph.py) | SQLite SPO 三元组 | 图遍历 + 时序推理 |
| 5 | [`memory_summary.py`](memory_summary.py) | 压缩后的摘要 | 关键词匹配摘要 |
| 6 | [`memory_hierarchical.py`](memory_hierarchical.py) | 三层：原始 → 情节 → 主题 | 自顶向下逐层 |
| 7 | [`memory_lifecycle.py`](memory_lifecycle.py) | JSONL + CRUD | Agent 通过工具调用控制 |
| 8 | [`memory_production.py`](memory_production.py) | Mem0 / Zep / Graphiti | SDK 提供 |

所有 Level 共享同一个 agent 循环：检索记忆 → 注入提示词 → LLM 响应 → 提取/保存记忆。不同的是*怎么存*和*怎么查*。

```python
# 所有 Level 的核心模式：
for _ in range(max_iterations):
    response = client.chat.completions.create(model=model, messages=messages)
    if not response.choices[0].message.tool_calls:
        return response.choices[0].message.content
    # 执行工具调用，追加结果，重复
```

## 各级详解

### Level 0: 无记忆 — [`agent.py`](agent.py)

```
    User --> Agent --> LLM --> Response
                  ^                |
                  |___无状态_______|

    每次对话从零开始。
```

基线 Agent，支持工具调用。~30 行。

### Level 1: 文本+关键词 — [`memory_file.py`](memory_file.py)

```
    User --> Agent --> LLM --> Response
      |                            |
      |  ┌──────────────────┐      |
      +->│  memory_facts.jsonl│<----+  提取事实
         │  {"text": "..."}   │       关键词匹配
         │  {"text": "..."}   │       下次检索
         └──────────────────┘

    存原文，搜关键词。
```

LLM 提取事实存入 JSONL。关键词匹配检索。~80 行。

### Level 2: 向量嵌入 — [`memory_vector.py`](memory_vector.py)

```
    User --> Agent --> LLM --> Response
      |                            |
      |  ┌──────────────────┐      |
      +->│  memory_vector.jsonl│<---+  提取事实
         │  {"text", "embed"}  │       嵌入查询
         │  cosine(query, db)  │       top-k 检索
         └──────────────────┘

    同样是 JSONL，但每条多了 embedding 向量。
    用语义相似度替代字符串匹配。
```

OpenAI embedding + numpy 余弦相似度。~100 行。

### Level 3: 认知评分 — [`memory_scored.py`](memory_scored.py)

```
    Score = alpha * 相似度 + beta * 时效性 + gamma * 重要性
                                |                |
                         艾宾浩斯衰减        LLM 打分 1-10
                         (半衰期=30天)

    ┌──────────────────────────────────────────────┐
    │  记忆条目                                    │
    │  text: "Alice 喜欢 Python"                  │
    │  embedding: [0.12, -0.34, ...]               │
    │  importance: 8                               │
    │  timestamp: 2025-03-31                       │
    │  access_count: 3                             │
    └──────────────────────────────────────────────┘

    + 反思：定期将记忆提炼为更高层的洞察
```

Park 风格三因子评分 + 反思机制。~120 行。

### Level 4: 知识图谱 — [`memory_graph.py`](memory_graph.py)

```
    ┌─────────┐  "moved_to"   ┌────────┐
    │  alice   │ ────────────> │ tokyo   │  valid_from: 2025-03
    └─────────┘                └────────┘
         |                           ^
         | "works_as"                | (旧事实，已失效)
         v                           |
    ┌─────────┐  "moved_to"   ┌────────┐
    │ engineer │               │ NYC     │  valid_until: 2025-03
    └─────────┘               └────────┘

    SQLite 存 (主, 谓, 宾) 三元组。
    自动检测矛盾。旧事实自动失效。
    时序推理：什么时间 什么是真的。
```

SQLite 中的 SPO 三元组 + 时序推理。~150 行。

### Level 5: 摘要压缩 — [`memory_summary.py`](memory_summary.py)

```
    轮次1: "我喜欢 Python 和深色模式"      ─┐
    轮次2: "我在一家叫 X 的创业公司工作"     ─┤
    轮次3: "我们前端用 React"               ─┤  满 5 条时
    轮次4: "我的狗叫 Buddy"                 ─┤  压缩
    轮次5: "我周末在学 Rust"                ─┘
                        |
                        v
    "用户是 Python 开发者，在创业公司 X 工作，
     前端用 React，有一只叫 Buddy 的狗，
     正在学 Rust。喜欢深色模式。"

    一条摘要替代 N 条原始记录。
```

LLM 将对话压缩为摘要。~80 行。

### Level 6: 层次化记忆 — [`memory_hierarchical.py`](memory_hierarchical.py)

```
    ┌─────────────────────────────────┐
    │  主题 (level 2)                  │  "用户是多语言开发者"
    │  高层模式和洞察                   │
    ├─────────────────────────────────┤
    │  情节 (level 1)                  │  "用户三月在学 Rust"
    │  压缩后的事件摘要                 │
    ├─────────────────────────────────┤
    │  原始 (level 0)                  │  "我今天写了个 Rust CLI 工具"
    │  原始对话片段                     │
    └─────────────────────────────────┘

    检索：先搜主题（便宜），再下钻到情节，最后原始（贵）。
    像人类回忆：要点 → 背景 → 细节。
```

三层层次结构，自顶向下检索。~100 行。

### Level 7: 自主生命周期 — [`memory_lifecycle.py`](memory_lifecycle.py)

```
    ┌───────────────────────────────────────────────┐
    │  Agent 拥有 4 个记忆工具：                     │
    │                                               │
    │  memory_save(fact)     -- 这个值得记住        │
    │  memory_delete(id)     -- 这个过时了          │
    │  memory_update(id, ..) -- 这个变了            │
    │  memory_search(query)  -- 让我查一下          │
    │                                               │
    │  Agent 自己决定何时使用每个工具。               │
    │  没有自动提取。没有被动存储。                   │
    └───────────────────────────────────────────────┘

    Level 0-6: agent 每轮自动保存。
    Level 7:   agent 自己选择存，或者不存。
```

Agent 通过函数调用自主控制记忆。~120 行。

### Level 8: 生产工具 — [`memory_production.py`](memory_production.py)

```
    ┌─────────────┬──────────────┬──────────────┐
    │    Mem0      │     Zep      │   Graphiti   │
    │  pip install │  pip install │  pip install │
    │   mem0ai     │  zep-cloud   │ graphiti-core│
    ├─────────────┼──────────────┼──────────────┤
    │ 事实提取     │ 时序知识图谱  │ SPO + Neo4j  │
    │ 向量/图存储  │ 自动消矛盾    │ 矛盾检测     │
    │ 快速上手     │ 生产级        │ 知识密集型   │
    └─────────────┴──────────────┴──────────────┘

    不用从零造轮子。选对工具就行。
    或者继续造——这就是 Level 0-7 的意义。
```

Mem0、Zep、Graphiti SDK 并排对比。~120 行。

## 参考文献

每个文件头都标注了参考来源。核心论文：

- Park et al. (2023) "Generative Agents" — [arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)
- "Memory in the Age of AI Agents" 综述 — [arxiv.org/abs/2512.13564](https://arxiv.org/abs/2512.13564)
- MemGPT (Packer et al., 2023) — [arxiv.org/abs/2310.08560](https://arxiv.org/abs/2310.08560)
- MemoryBank (Zhong et al., 2024) — [arxiv.org/abs/2401.10917](https://arxiv.org/abs/2401.10917)
- xMemory (2025) — [arxiv.org/abs/2502.13743](https://arxiv.org/abs/2502.13743)

---

## 许可证

MIT

────────────────────────────────────────

⏺ _如同记忆本身，每一行都很微小——但汇聚在一起，便能记住一切。_
