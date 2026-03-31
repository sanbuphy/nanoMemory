# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanoMemory is an educational project demonstrating progressive levels of agent memory systems, each implemented in ~100-150 lines of Python. The tagline: "If you can read ~200 lines of Python, you understand agent memory."

## Running

Each level is a standalone script that can be run directly:
```bash
python agent.py <task>                    # Level 0: no memory
python memory_file.py <task>              # Level 1: text + keyword
python memory_vector.py <task>            # Level 2: vector embedding
python memory_scored.py <task>            # Level 3: cognitive scoring
python memory_graph.py <task>             # Level 4: knowledge graph
python memory_summary.py <task>           # Level 5: summary compression
python memory_hierarchical.py <task>      # Level 6: hierarchical memory
python memory_lifecycle.py <task>         # Level 7: agentic memory lifecycle
python memory_production.py               # Level 8: production tools (Mem0/Zep/Graphiti)
```

Requires `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_EMBED_MODEL`) environment variables.

Install dependencies: `pip install -r requirements.txt`

No test or eval framework exists yet — `tests/` and `eval/` directories are empty placeholders.

## Architecture: Progressive Memory Levels

Each level demonstrates a fundamentally different **memory construction method** (not just feature stacking):

| Level | File | Method | Storage | Retrieval |
|-------|------|--------|---------|-----------|
| 0 | `agent.py` | No memory | None | None |
| 1 | `memory_file.py` | Text + keyword | JSONL | String matching |
| 2 | `memory_vector.py` | Vector embedding | JSONL + embeddings | Cosine similarity |
| 3 | `memory_scored.py` | Cognitive scoring | JSONL + embeddings | α·sim + β·recency + γ·importance |
| 4 | `memory_graph.py` | Knowledge graph | SQLite SPO triples | Graph traversal + temporal |
| 5 | `memory_summary.py` | Summary compression | Compressed summaries | Keyword over summaries |
| 6 | `memory_hierarchical.py` | Hierarchical layers | 3-level: raw→episode→theme | Top-down drill |
| 7 | `memory_lifecycle.py` | Agentic lifecycle | JSONL with CRUD | Agent-controlled tool calls |
| 8 | `memory_production.py` | Production tools | Mem0/Zep/Graphiti | SDK-provided |

## Key Patterns

- All files share the same OpenAI client setup pattern: `api_key` + optional `base_url` from env vars
- Each level is self-contained — no imports between files
- Storage formats: JSONL for levels 1-3/5-7, SQLite for level 4, external DBs for level 8
- Levels 0-6: auto-extract memories after each turn (passive storage)
- Level 7: agent decides when to save/delete/update/search (active management)
- Level 8: production SDK demos, not self-built implementations
- LLM-based extraction prompts always request JSON array output with `temperature=0`
- Every file includes reference sources with links in its docstring
