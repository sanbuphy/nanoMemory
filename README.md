# nanoMemory

**If you can read ~200 lines of Python, you understand agent memory.**

A minimal, progressive demonstration of 9 different agent memory architectures — each self-contained in ~80-150 lines of Python.

[中文文档](README-CN.md)

---

## What It Is

Most agent memory tutorials show one approach. This project shows **nine**, each a fundamentally different way to build memory:

| Level | Method | File | Lines |
|:-----:|--------|------|:-----:|
| 0 | No memory | `agent.py` | ~30 |
| 1 | Text + Keyword | `memory_file.py` | ~80 |
| 2 | Vector Embedding | `memory_vector.py` | ~100 |
| 3 | Cognitive Scoring | `memory_scored.py` | ~120 |
| 4 | Knowledge Graph | `memory_graph.py` | ~150 |
| 5 | Summary Compression | `memory_summary.py` | ~80 |
| 6 | Hierarchical Memory | `memory_hierarchical.py` | ~100 |
| 7 | Agentic Lifecycle | `memory_lifecycle.py` | ~120 |
| 8 | Production Tools | `memory_production.py` | ~120 |

---

## Why It Exists

Agent memory systems are rapidly evolving but often presented as black boxes. This project takes the opposite stance:

- **Each level is one file.** No imports between levels. Read top to bottom.
- **Each level is a different paradigm.** Not feature stacking — fundamentally different ways to store and retrieve memory.
- **Every file cites its sources.** Academic papers, GitHub repos, and survey references in the docstring.

---

## Quick Start

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="your-key"
# Optional:
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_EMBED_MODEL="text-embedding-3-small"

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

---

## Level Details

### Level 0: No Memory
Baseline agent with tool use. No state persists between conversations.

### Level 1: Text + Keyword
LLM extracts facts into a JSONL file. Keyword matching for retrieval. The simplest cross-session memory.

### Level 2: Vector Embedding
OpenAI embeddings + numpy cosine similarity replace keyword matching. Semantic search, not just string matching.

### Level 3: Cognitive Scoring
Three-factor retrieval: `alpha*similarity + beta*recency + gamma*importance`. Ebbinghaus-inspired decay. Reflection mechanism distills memories into higher-level insights.

### Level 4: Knowledge Graph
(SPO) triples in SQLite. Temporal reasoning with `valid_from`/`valid_until`. Entity normalization. Contradiction detection.

### Level 5: Summary Compression
LLM compresses conversations into summaries. Periodic consolidation merges old summaries. Trades fidelity for capacity.

### Level 6: Hierarchical Memory
3-level hierarchy: raw messages -> episodes -> themes. Top-down retrieval mirrors human recall: gist first, then details.

### Level 7: Agentic Lifecycle
The agent itself controls memory via tool calls: `save`, `delete`, `update`, `search`. No passive auto-extraction — the agent decides what to remember and what to forget.

### Level 8: Production Tools
Side-by-side comparison of Mem0, Zep, and Graphiti SDKs. Install instructions, usage examples, and a comparison table.

```bash
pip install mem0ai         # Mem0: fact extraction + vector/graph
pip install zep-cloud      # Zep: temporal knowledge graph
pip install graphiti-core  # Graphiti: SPO triples + Neo4j
```

---

## Architecture Principles

```
Storage Paradigms (what you store)
  Text -> Vector -> Graph -> Summary -> Hierarchy

Retrieval Strategies (how you search)
  Keyword -> Cosine -> Scoring -> Graph -> Top-down

Control Model (who decides)
  Auto-extract (L0-6) -> Agent-controlled (L7) -> Production SDKs (L8)
```

---

## Key References

- Park et al. (2023) "Generative Agents" — [arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)
- "Memory in the Age of AI Agents" Survey — [arxiv.org/abs/2512.13564](https://arxiv.org/abs/2512.13564)
- MemGPT (Packer et al., 2023) — [arxiv.org/abs/2310.08560](https://arxiv.org/abs/2310.08560)
- MemoryBank (Zhong et al., 2024) — [arxiv.org/abs/2401.10917](https://arxiv.org/abs/2401.10917)
- xMemory (2025) Hierarchical Retrieval — [arxiv.org/abs/2502.13743](https://arxiv.org/abs/2502.13743)

---

## License

MIT
