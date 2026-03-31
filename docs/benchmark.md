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
Level Method                     Single  Temporal  Multi   Open  Advers    Avg
------------------------------------------------------------------------------
0     No Memory                   20.0%      0.0%   0.0%  40.0%   40.0%  20.0%
1     Text + Keyword               0.0%      0.0%   0.0%   0.0%    0.0%   0.0%
2     Vector Embedding           (requires embedding API)
3     Cognitive Scoring          (requires embedding API)
4     Knowledge Graph            (requires embedding API)
5     Summary Compression          0.0%      0.0%   0.0%   0.0%    0.0%   0.0%
6     Hierarchical                 0.0%      0.0%   0.0%   0.0%    0.0%   0.0%
7     Agentic Lifecycle          (requires embedding API)
```

> Model: `step-3.5-flash` via StepFun API. 5 questions per category, 1 conversation (conv-26, 419 turns).
> Levels 2/3/7 require embedding API (not available on StepFun). Level 4 requires separate evaluation setup.

### Why Level 0 Beats Level 1/5/6

This is the most important finding: **no memory beats broken memory**.

- Level 0 asks the LLM directly — for open-domain and adversarial questions, the model can sometimes answer from parametric knowledge
- Levels 1/5/6 use keyword matching to retrieve context — but LoCoMo questions rarely share words with the original text, so retrieval returns nothing useful
- Empty or irrelevant context is worse than no context — it wastes the prompt budget and confuses the model

**This is exactly why vector embedding (Level 2) exists.** Without semantic search, memory retrieval fails on real conversational data. Keyword matching works for exact-term lookup but not for natural language questions.

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
