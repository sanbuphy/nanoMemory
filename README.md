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

### level 0: no memory — [`agent.py`](agent.py)

```
    User --> Agent --> LLM --> Response
                  ^                |
                  |___no state_____|

    Every conversation starts from zero.
```

Baseline agent with tool use. ~30 lines.

### level 1: text + keyword — [`memory_file.py`](memory_file.py)

```
    User --> Agent --> LLM --> Response
      |                            |
      |  ┌──────────────────┐      |
      +->│  memory_facts.jsonl│<----+  extract facts
         │  {"text": "..."}   │       keyword match
         │  {"text": "..."}   │       next turn
         └──────────────────┘

    Store raw text. Search by keyword overlap.
```

LLM extracts facts into JSONL. Keyword matching for retrieval. ~80 lines.

### level 2: vector embedding — [`memory_vector.py`](memory_vector.py)

```
    User --> Agent --> LLM --> Response
      |                            |
      |  ┌──────────────────┐      |
      +->│  memory_vector.jsonl│<---+  extract facts
         │  {"text", "embed"}  │       embed query
         │  cosine(query, db)  │       top-k retrieval
         └──────────────────┘

    Same JSONL, but now each entry has an embedding vector.
    Replace string matching with semantic similarity.
```

OpenAI embeddings + numpy cosine similarity. ~100 lines.

### level 3: cognitive scoring — [`memory_scored.py`](memory_scored.py)

```
    Score = alpha * similarity + beta * recency + gamma * importance
                                     |              |
                              Ebbinghaus decay    LLM rates 1-10
                              (half-life=30d)

    ┌──────────────────────────────────────────────┐
    │  Memory Entry                                │
    │  text: "Alice prefers Python"                │
    │  embedding: [0.12, -0.34, ...]               │
    │  importance: 8                               │
    │  timestamp: 2025-03-31                       │
    │  access_count: 3                             │
    └──────────────────────────────────────────────┘

    + Reflection: periodically distill memories into insights
```

Park-style three-factor scoring + reflection. ~120 lines.

### level 4: knowledge graph — [`memory_graph.py`](memory_graph.py)

```
    ┌─────────┐   "moved_to"   ┌────────┐
    │  alice   │ ────────────>  │ tokyo   │  valid_from: 2025-03
    └─────────┘                 └────────┘
         |                            ^
         | "works_as"                 | (old, invalidated)
         v                            |
    ┌─────────┐   "moved_to"   ┌────────┐
    │ engineer │               │ NYC     │  valid_until: 2025-03
    └─────────┘               └────────┘

    SQLite stores (subject, predicate, object) triples.
    Contradictions auto-detected. Old facts invalidated.
    Temporal reasoning: what was true WHEN.
```

SPO triples in SQLite with temporal reasoning. ~150 lines.

### level 5: summary compression — [`memory_summary.py`](memory_summary.py)

```
    Turn 1: "I like Python and dark mode"     ─┐
    Turn 2: "I work at a startup called X"     ─┤  compress
    Turn 3: "We use React for the frontend"    ─┤  when >= 5
    Turn 4: "My dog's name is Buddy"           ─┤  summaries
    Turn 5: "I'm learning Rust on weekends"    ─┘
                        |
                        v
    "User is a Python developer at startup X,
     uses React, has a dog named Buddy,
     learning Rust. Prefers dark mode."

    One summary replaces N raw entries.
```

LLM compresses conversations into summaries. ~80 lines.

### level 6: hierarchical memory — [`memory_hierarchical.py`](memory_hierarchical.py)

```
    ┌─────────────────────────────────┐
    │  THEMES (level 2)               │  "User is a polyglot developer"
    │  high-level patterns            │
    ├─────────────────────────────────┤
    │  EPISODES (level 1)             │  "User studied Rust in March"
    │  compressed event summaries     │
    ├─────────────────────────────────┤
    │  RAW (level 0)                  │  "I wrote a Rust CLI tool today"
    │  original conversation snippets │
    └─────────────────────────────────┘

    Retrieval: search themes first (cheap),
    drill down to episodes, then raw (expensive).
    Like human recall: gist -> context -> details.
```

3-level hierarchy with top-down retrieval. ~100 lines.

### level 7: agentic lifecycle — [`memory_lifecycle.py`](memory_lifecycle.py)

```
    ┌───────────────────────────────────────────────┐
    │  Agent has 4 memory TOOLS:                    │
    │                                               │
    │  memory_save(fact)     -- I should remember   │
    │  memory_delete(id)     -- this is outdated    │
    │  memory_update(id, ..) -- this changed        │
    │  memory_search(query)  -- let me check        │
    │                                               │
    │  The agent DECIDES when to use each tool.     │
    │  No auto-extraction. No passive storage.      │
    └───────────────────────────────────────────────┘

    Levels 0-6: agent saves automatically after every turn.
    Level 7:    agent chooses to save, or not.
```

Agent-controlled memory via function calling. ~120 lines.

### level 8: production tools — [`memory_production.py`](memory_production.py)

```
    ┌─────────────┬──────────────┬──────────────┐
    │    Mem0      │     Zep      │   Graphiti   │
    │  pip install │  pip install │  pip install │
    │   mem0ai     │  zep-cloud   │ graphiti-core│
    ├─────────────┼──────────────┼──────────────┤
    │ fact extract │ temporal KG  │ SPO + Neo4j  │
    │ vector/graph │ auto-resolve │ contradictions│
    │ quick start  │ production   │ knowledge-heavy│
    └─────────────┴──────────────┴──────────────┘

    Stop building from scratch. Use the right tool for the job.
    Or keep building — that's what levels 0-7 are for.
```

Side-by-side SDK comparison: Mem0, Zep, Graphiti. ~120 lines.

## references

Every file cites its sources in the docstring. Key papers:

- Park et al. (2023) "Generative Agents" — [arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)
- "Memory in the Age of AI Agents" survey — [arxiv.org/abs/2512.13564](https://arxiv.org/abs/2512.13564)
- MemGPT (Packer et al., 2023) — [arxiv.org/abs/2310.08560](https://arxiv.org/abs/2310.08560)
- MemoryBank (Zhong et al., 2024) — [arxiv.org/abs/2401.10917](https://arxiv.org/abs/2401.10917)
- xMemory (2025) — [arxiv.org/abs/2502.13743](https://arxiv.org/abs/2502.13743)

## benchmarks

How do you know if a memory system actually works? These benchmarks measure different aspects:

| Benchmark | Year | What It Tests | Link |
|-----------|------|---------------|------|
| **bAbI** | 2015 | 20 toy reasoning tasks (basic fact recall, deduction) | [facebookresearch/bAbI](https://github.com/facebookarchive/bAbI) |
| **LoCoMo** | 2024 | Long-term conversational memory across multi-session dialogues (~26K tokens each) | [snap-research/locomo](https://snap-research.github.io/locomo/) |
| **LongMemEval** | 2024 | Long-term interactive memory for chat assistants (4 categories: knowledge, preference, event, session) | [arxiv.org/abs/2410.10876](https://arxiv.org/abs/2410.10876) |
| **MemBench** | 2025 | Comprehensive evaluation: effectiveness, efficiency, capacity of LLM agent memory | [arxiv.org/abs/2507.05257](https://arxiv.org/abs/2507.05257) |
| **MemoryAgentBench** | 2025 | Memory retention, update, retrieval, conflict resolution in multi-turn settings | [HUST-AI-HYZ/MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) (ICLR 2026) |
| **MemoryArena** | 2026 | Interdependent multi-session agentic tasks (web nav, planning, search) | [memoryarena.github.io](https://memoryarena.github.io/) |

```
    BENCHMARK LANDSCAPE
    ===================

    bAbI (2015)           ── basic reasoning
         |
    LoCoMo (2024)         ── long conversations
    LongMemEval (2024)    ── chat assistant memory
         |
    MemBench (2025)       ── comprehensive: efficiency + capacity
    MemoryAgentBench (2025) ── CRUD + conflict resolution
         |
    MemoryArena (2026)    ── real agentic workflows

    Trend: toy tasks → real conversations → full agent evaluation
```

---

## benchmark

Full LoCoMo evaluation results — [`docs/benchmark.md`](docs/benchmark.md)

---

## license

MIT

────────────────────────────────────────

⏺ _Like memory itself, each line is small — but together they remember everything._
