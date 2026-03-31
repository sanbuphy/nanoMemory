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

| Level | 方法 | 存储 | 检索 |
|:-----:|------|------|------|
| 0 | 无记忆 | 无 | 无 |
| 1 | 文本+关键词 | JSONL | 字符串匹配 |
| 2 | 向量嵌入 | JSONL + embedding | 余弦相似度 |
| 3 | 认知评分 | JSONL + embedding | alpha*相似度 + beta*时效 + gamma*重要性 |
| 4 | 知识图谱 | SQLite SPO 三元组 | 图遍历 + 时序推理 |
| 5 | 摘要压缩 | 压缩后的摘要 | 关键词匹配摘要 |
| 6 | 层次化记忆 | 三层：原始 → 情节 → 主题 | 自顶向下逐层 |
| 7 | 自主生命周期 | JSONL + CRUD | Agent 通过工具调用控制 |
| 8 | 生产工具 | Mem0 / Zep / Graphiti | SDK 提供 |

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

### Level 0: 无记忆
基线 Agent。对话之间无状态。~30 行。

### Level 1: 文本+关键词
LLM 提取事实存入 JSONL。关键词匹配检索。上限 200 条。~80 行。

### Level 2: 向量嵌入
OpenAI embedding 替代关键词匹配。numpy 余弦相似度实现语义搜索。~100 行。

### Level 3: 认知评分
Park 风格三因子评分：相似度 x 时效性 x 重要性。艾宾浩斯遗忘曲线。反思机制提炼洞察。~120 行。

### Level 4: 知识图谱
SQLite 中的主谓宾三元组。时序推理。实体归一化。矛盾检测——新事实自动使旧矛盾失效。~150 行。

### Level 5: 摘要压缩
LLM 将对话压缩为摘要，不存原文。定期合并旧摘要。~80 行。

### Level 6: 层次化记忆
三层结构：原始消息 → 情节 → 主题。自顶向下检索模拟人类回忆——先要点，后细节。~100 行。

### Level 7: 自主生命周期
Agent 自己通过工具调用控制记忆：`save`、`delete`、`update`、`search`。没有被动提取。~120 行。

### Level 8: 生产工具
开源记忆框架的并排对比：

```
pip install mem0ai         # 事实提取 + 向量/图存储
pip install zep-cloud      # 时序知识图谱
pip install graphiti-core  # SPO 三元组 + Neo4j
```

包含安装说明、用法示例和对比表。~120 行。

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
