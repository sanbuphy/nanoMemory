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

<!-- 结果将在运行评测后自动填入 -->

> 运行 `python eval/run_locomo.py` 后，结果将保存到 `docs/benchmark_results.json`。

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
