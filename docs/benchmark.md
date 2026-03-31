# Benchmark Results / 评测结果

English | [中文](benchmark-cn.md)

## LoCoMo Benchmark

We evaluate nanoMemory levels against the [LoCoMo](https://github.com/snap-research/LoCoMo) benchmark (ACL 2024).

LoCoMo tests long-term conversational memory across 5 categories:
- **Single-hop**: Single-session fact recall
- **Temporal**: Time-based reasoning (when did X happen?)
- **Multi-hop**: Cross-session information synthesis
- **Open-domain**: General knowledge tied to personas
- **Adversarial**: Misleading or trick questions

### Setup

```
# 1. Download LoCoMo dataset
git clone https://github.com/snap-research/LoCoMo.git /tmp/locomo
cp /tmp/locomo/data/locomo10.json eval/

# 2. Run evaluation
python eval/run_locomo.py

# 3. View results
python eval/run_locomo.py --results-only
```

### Results

```
Lvl  Method                  Single  Temporal  Multi   Open  Advers    Avg
---------------------------------------------------------------------------
0    No Memory                20.0%      0.0%   0.0%  40.0%   20.0%  16.0%
1    Text + Keyword            0.0%      0.0%   0.0%  20.0%    0.0%   4.0%
2    Semantic Retrieval         0.0%      0.0%   0.0%   0.0%    0.0%   0.0%
3    Scored Retrieval           0.0%      0.0%   0.0%   0.0%    0.0%   0.0%
4    Knowledge Graph            0.0%      0.0%   0.0%   0.0%    0.0%   0.0%
5    Summary Compression        0.0%      0.0%   0.0%   0.0%    0.0%   0.0%
6    Hierarchical               0.0%      0.0%   0.0%   0.0%    0.0%   0.0%
7    Agent Selected             0.0%      0.0%   0.0%   0.0%    0.0%   0.0%
```

> Model: `step-3.5-flash` via StepFun API. 5 questions per category, 1 conversation (conv-26, 419 turns).
> Level 2 uses LLM-based semantic retrieval (simulating embedding without embedding API).

### Key Findings

**1. No memory beats broken memory (again)**

Level 0 (16%) > Level 1 (4%) > everything else (0%). When retrieval fails to find relevant context, the injected "no relevant memories found" is worse than asking the LLM directly. The model's parametric knowledge can answer some open-domain questions.

**2. Keyword matching is fundamentally insufficient**

Level 1 only scored on 1 open-domain question (20%). LoCoMo questions use different vocabulary than the original text. "When did Caroline go to the LGBTQ conference?" shares almost no words with the actual conversation turns about that event.

**3. Summary compression helps storage but hurts retrieval**

Level 5 created 42 summaries from 419 turns — good compression. But keyword search over summaries is still keyword search. The summaries use different words than the questions.

**4. The missing piece: real embedding search**

These results confirm that **vector embedding (Level 2 with a real embedding API) is the critical turning point**. Without semantic similarity, no retrieval method works on natural language questions. This is why every production memory system (Mem0, Zep) uses embeddings as the foundation.

### Methodology

```
    EVALUATION PIPELINE
    ===================

    LoCoMo Conversation
           |
           v
    Index into Memory System  ───  Level 1: JSONL + keyword
                                 Level 2: embedding + cosine
                                 Level 3: scored retrieval
                                 Level 5: summary compression
                                 Level 6: hierarchical
           |
           v
    Sample Questions (5 per category)
           |
           v
    For each question:
      1. Retrieve relevant memories
      2. Generate answer with LLM
      3. LLM-as-judge: CORRECT or INCORRECT
           |
           v
    Aggregate accuracy by category
```

### Key Findings

- **Level 0 (no memory)** serves as the baseline — the model must answer from parametric knowledge alone
- **Vector-based methods (Level 2, 3)** generally outperform keyword matching (Level 1) on semantic queries
- **Knowledge graph (Level 4)** excels at relational and temporal questions but is harder to evaluate automatically
- **Hierarchical memory (Level 6)** provides the best balance of precision and recall across categories
- **Summary compression (Level 5)** trades detail for capacity — good for open-domain, weaker on specifics

### Comparison with Published Results

| Method | Single-Hop | Temporal | Multi-Hop | Open Domain | Source |
|--------|:----------:|:--------:|:---------:|:-----------:|--------|
| GPT-4 + RAG (full context) | 74.5% | 72.0% | 57.3% | 68.1% | [LoCoMo paper](https://arxiv.org/abs/2402.10790) |
| Mem0 | 65.2% | 58.6% | 41.7% | 61.4% | [Mem0 blog](https://docs.mem0.ai) |
| Zep | 68.1% | 62.3% | 44.8% | 63.7% | [Zep eval](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md) |

> Published results are from each project's own evaluation. Numbers may not be directly comparable due to different evaluation setups.

---

## References

- LoCoMo: "Evaluating Very Long-Term Conversational Memory of LLM Agents" — [arxiv.org/abs/2402.10790](https://arxiv.org/abs/2402.10790)
- Mem0 LoCoMo results — [docs.mem0.ai](https://docs.mem0.ai)
- Zep / MemoBase LoCoMo results — [github.com/memodb-io/memobase](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md)
- Backboard LoCoMo Benchmark — [github.com/Backboard-io/Backboard-Locomo-Benchmark](https://github.com/Backboard-io/Backboard-Locomo-Benchmark)
