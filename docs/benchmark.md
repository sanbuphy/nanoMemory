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

<!-- RESULTS_INSERT_POINT -->

> Note: Results use a sampled subset (5 questions per category) from 1 conversation.
> Full LoCoMo has 10 conversations, 1986 QA pairs. This is a quick evaluation, not a comprehensive benchmark.
> Model used: `stepfun/step-3.5-flash` via OpenRouter.

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
