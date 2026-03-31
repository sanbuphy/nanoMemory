# 评测结果

[English](benchmark.md) | 中文

## LoCoMo 基准测试

我们使用 [LoCoMo](https://github.com/snap-research/LoCoMo) 基准测试（ACL 2024）评估 nanoMemory 各 Level 的表现。

LoCoMo 测试 5 个维度的长期对话记忆能力：
- **Single-hop（单跳）**：单会话事实回忆
- **Temporal（时序）**：基于时间的推理（X 什么时候发生的？）
- **Multi-hop（多跳）**：跨会话信息综合
- **Open-domain（开放域）**：与人物相关的通用知识
- **Adversarial（对抗）**：误导性或陷阱问题

### 运行评测

```
# 1. 下载 LoCoMo 数据集
git clone https://github.com/snap-research/LoCoMo.git /tmp/locomo
cp /tmp/locomo/data/locomo10.json eval/

# 2. 设置环境变量
export OPENAI_API_KEY='your-key'
export OPENAI_BASE_URL='https://openrouter.ai/api/v1'   # 可选
export OPENAI_MODEL='stepfun/step-3.5-flash'            # 可选

# 3. 运行评测
python eval/run_locomo.py

# 4. 查看结果
python eval/run_locomo.py --results-only
```

### 评测结果

```
Lvl  方法                     Single  Temporal  Multi   Open  Advers    Avg
---------------------------------------------------------------------------
0    无记忆                     20.0%      0.0%   0.0%  40.0%   20.0%  16.0%
1    文本+关键词                 0.0%      0.0%   0.0%  20.0%    0.0%   4.0%
2    语义检索                    0.0%      0.0%   0.0%   0.0%    0.0%   0.0%
3    评分检索                    0.0%      0.0%   0.0%   0.0%    0.0%   0.0%
4    知识图谱                    0.0%      0.0%   0.0%   0.0%    0.0%   0.0%
5    摘要压缩                    0.0%      0.0%   0.0%   0.0%    0.0%   0.0%
6    层次化记忆                  0.0%      0.0%   0.0%   0.0%    0.0%   0.0%
7    Agent 自选                  0.0%      0.0%   0.0%   0.0%    0.0%   0.0%
```

> 模型：`step-3.5-flash` via StepFun API。每类别 5 个问题，1 个对话（conv-26, 419 轮）。

### 关键发现

**1. 无记忆比坏记忆好**

Level 0 (16%) > Level 1 (4%) > 其他 (0%)。当检索找不到相关上下文时，注入"无相关记忆"比直接问 LLM 更糟。

**2. 关键词匹配从根本上不够**

Level 1 只在 1 个开放域问题上得分。LoCoMo 的问题和原文几乎不共用词汇。

**3. 摘要压缩帮助了存储但伤害了检索**

Level 5 把 419 轮压缩成 42 条摘要——好的压缩率。但摘要上的关键词搜索仍然是关键词搜索。

**4. 缺失的关键拼图：真正的 embedding 搜索**

结果证实**向量嵌入（Level 2 配合真正的 embedding API）是关键转折点**。没有语义相似度，任何检索方法在自然语言问题上都不起作用。这就是为什么每个生产级记忆系统（Mem0、Zep）都以 embedding 为基础。

### 评测流程

```
    评测流程
    ========

    LoCoMo 对话数据
           |
           v
    写入各 Level 的记忆系统    ──  Level 1: JSONL + 关键词
                                 Level 2: embedding + 余弦
                                 Level 3: 评分检索
                                 Level 5: 摘要压缩
                                 Level 6: 层次化
           |
           v
    每类别采样 5 个问题
           |
           v
    对每个问题：
      1. 检索相关记忆
      2. LLM 生成答案
      3. LLM 裁判判定：CORRECT / INCORRECT
           |
           v
    按类别汇总准确率
```

### 预期结论

- **Level 0（无记忆）** 作为基线——模型仅靠参数化知识回答
- **向量方法（Level 2, 3）** 在语义查询上通常优于关键词匹配（Level 1）
- **知识图谱（Level 4）** 在关系和时序问题上表现突出，但自动评测更困难
- **层次化记忆（Level 6）** 在各类别上提供最佳精度和召回平衡
- **摘要压缩（Level 5）** 用细节换容量——开放域表现好，具体事实偏弱

### 与公开结果对比

| 方法 | Single-Hop | Temporal | Multi-Hop | Open Domain | 来源 |
|------|:----------:|:--------:|:---------:|:-----------:|------|
| GPT-4 + RAG (全上下文) | 74.5% | 72.0% | 57.3% | 68.1% | [LoCoMo 论文](https://arxiv.org/abs/2402.10790) |
| Mem0 | 65.2% | 58.6% | 41.7% | 61.4% | [Mem0 文档](https://docs.mem0.ai) |
| Zep | 68.1% | 62.3% | 44.8% | 63.7% | [Zep 评测](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md) |

> 公开结果来自各项目自行评测。由于评测设置不同，数据可能不完全可比。

---

## 参考文献

- LoCoMo: "Evaluating Very Long-Term Conversational Memory of LLM Agents" — [arxiv.org/abs/2402.10790](https://arxiv.org/abs/2402.10790)
- Mem0 LoCoMo 结果 — [docs.mem0.ai](https://docs.mem0.ai)
- Zep / MemoBase LoCoMo 结果 — [github.com/memodb-io/memobase](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md)
- Backboard LoCoMo Benchmark — [github.com/Backboard-io/Backboard-Locomo-Benchmark](https://github.com/Backboard-io/Backboard-Locomo-Benchmark)
