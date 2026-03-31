English | [中文](README-CN.md)

> _"Memory is the treasure house of the mind."_ — Thomas Fuller

If you can read ~200 lines of Python, you understand agent memory.

A minimal, progressive demonstration of 9 different agent memory architectures. Each level is one self-contained file, ~80-150 lines.

## install

```
pip install -r requirements.txt
```

Set your environment variables:

__macOS/Linux:__

```
export OPENAI_API_KEY='your-key-here'
export OPENAI_BASE_URL='https://api.openai.com/v1'  # optional
export OPENAI_MODEL='gpt-4o-mini'  # optional
export OPENAI_EMBED_MODEL='text-embedding-3-small'  # optional
```

__Windows (PowerShell):__

```
$env:OPENAI_API_KEY='your-key-here'
$env:OPENAI_BASE_URL='https://api.openai.com/v1'
$env:OPENAI_MODEL='gpt-4o-mini'
```

## quick start

```
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

## how it works

Each level demonstrates a fundamentally different way to build memory — not feature stacking, but different paradigms.

| level | file | storage | retrieval |
|:-----:|------|---------|-----------|
| 0 | [`agent.py`](agent.py) | none | none |
| 1 | [`memory_file.py`](memory_file.py) | JSONL | string matching |
| 2 | [`memory_vector.py`](memory_vector.py) | JSONL + embeddings | cosine similarity |
| 3 | [`memory_scored.py`](memory_scored.py) | JSONL + embeddings | alpha*sim + beta*recency + gamma*importance |
| 4 | [`memory_graph.py`](memory_graph.py) | SQLite SPO triples | graph traversal + temporal |
| 5 | [`memory_summary.py`](memory_summary.py) | compressed summaries | keyword over summaries |
| 6 | [`memory_hierarchical.py`](memory_hierarchical.py) | 3-level: raw -> episode -> theme | top-down drill |
| 7 | [`memory_lifecycle.py`](memory_lifecycle.py) | JSONL with CRUD | agent-controlled tool calls |
| 8 | [`memory_production.py`](memory_production.py) | Mem0 / Zep / Graphiti | SDK-provided |

The agent loop is the same everywhere: retrieve memories -> inject into prompt -> LLM response -> extract/save memories. What changes is *how* you store and *how* you retrieve.

```python
# The core pattern across all levels:
for _ in range(max_iterations):
    response = client.chat.completions.create(model=model, messages=messages)
    if not response.choices[0].message.tool_calls:
        return response.choices[0].message.content
    # execute tool calls, append results, repeat
```

## levels

### level 0: no memory
Baseline agent. No state between conversations. ~30 lines.

### level 1: text + keyword
LLM extracts facts into JSONL. Keyword matching for retrieval. Capped at 200 memories. ~80 lines.

### level 2: vector embedding
OpenAI embeddings replace keyword matching. Semantic search via numpy cosine similarity. ~100 lines.

### level 3: cognitive scoring
Park-style three-factor scoring: similarity x recency x importance. Ebbinghaus decay curve. Reflection mechanism distills memories into insights. ~120 lines.

### level 4: knowledge graph
(subject, predicate, object) triples in SQLite. Temporal reasoning. Entity normalization. Contradiction detection — new facts invalidate conflicting old ones. ~150 lines.

### level 5: summary compression
LLM compresses conversations into summaries instead of storing raw text. Periodic consolidation merges old summaries into one. ~80 lines.

### level 6: hierarchical memory
3-level hierarchy: raw messages -> episodes -> themes. Top-down retrieval mirrors human recall — you remember the gist, then the details. ~100 lines.

### level 7: agentic lifecycle
The agent itself controls memory via tool calls: `save`, `delete`, `update`, `search`. No passive auto-extraction. The agent decides what to remember and what to forget. ~120 lines.

### level 8: production tools
Side-by-side comparison of open-source memory frameworks:

```
pip install mem0ai         # fact extraction + vector/graph
pip install zep-cloud      # temporal knowledge graph
pip install graphiti-core  # SPO triples + Neo4j
```

Includes install instructions, usage examples, and a comparison table. ~120 lines.

## references

Every file cites its sources in the docstring. Key papers:

- Park et al. (2023) "Generative Agents" — [arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)
- "Memory in the Age of AI Agents" survey — [arxiv.org/abs/2512.13564](https://arxiv.org/abs/2512.13564)
- MemGPT (Packer et al., 2023) — [arxiv.org/abs/2310.08560](https://arxiv.org/abs/2310.08560)
- MemoryBank (Zhong et al., 2024) — [arxiv.org/abs/2401.10917](https://arxiv.org/abs/2401.10917)
- xMemory (2025) — [arxiv.org/abs/2502.13743](https://arxiv.org/abs/2502.13743)

---

## license

MIT

────────────────────────────────────────

⏺ _Like memory itself, each line is small — but together they remember everything._
