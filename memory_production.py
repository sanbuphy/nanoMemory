"""
nanoMemory Level 8: Production-Grade Memory Tools Integration
~120 lines. Compare self-built memory vs production open-source tools.

Instead of building from scratch, this level demonstrates how to use
established memory frameworks via their Python SDKs. Each tool has a
different philosophy — compare and choose for your use case.

Tools covered:
  1. Mem0  — "The Memory Layer for AI" — fact extraction + vector/graph storage
  2. Zep   — "Memory Infrastructure for AI" — temporal knowledge graph + structured memory
  3. Graphiti — "Knowledge Graph Memory" — SPO triples with temporal reasoning

Install:
  pip install mem0ai          # Mem0: https://github.com/mem0ai/mem0
  pip install zep-cloud       # Zep Cloud: https://github.com/getzep/zep
  pip install graphiti-core   # Graphiti: https://github.com/getzep/graphiti

References:
  - Mem0: Open-source persistent memory layer, 25k+ GitHub stars
    https://github.com/mem0ai/mem0
    Docs: https://docs.mem0.ai/open-source/python-quickstart
  - Zep: Temporal knowledge graph for AI agents
    https://github.com/getzep/zep
    Docs: https://help.getzep.com
  - Graphiti: Knowledge graph memory with temporal awareness (by Zep team)
    https://github.com/getzep/graphiti
    Blog: https://www.getzep.com/blog/graphiti
  - OpenMemory by Mem0: Self-hosted MCP server for shared agent memory
    https://github.com/mem0ai/mem0/tree/main/openmemory
  - Survey comparison: "Memory in the Age of AI Agents" §5 frameworks
    https://arxiv.org/abs/2512.13564
  - "AI Agent Memory Systems in 2026: Mem0, Zep, Hindsight, MemVid
    and Everything In Between — Compared"
    https://medium.com/@yogeshyadav/ai-agent-memory-systems-in-2026
"""
import os
import json
from datetime import datetime

# All tools are optional — each section works independently.
# Install only what you need: pip install mem0ai / zep-cloud / graphiti-core


# ═══════════════════════════════════════════════════════════════════════
# 1. Mem0 — Simple fact-based memory
# ═══════════════════════════════════════════════════════════════════════
# pip install mem0ai
# GitHub: https://github.com/mem0ai/mem0 (25k+ stars)
# Philosophy: LLM extracts facts → vector store + optional graph → auto-dedup
#
# Best for: User preferences, personalization, simple fact storage
# Trade-off: Black-box extraction; less control over what gets stored

def demo_mem0():
    """Mem0: add/search/delete memories for a user."""
    try:
        from mem0 import Memory
    except ImportError:
        print("Install: pip install mem0ai")
        return

    m = Memory()  # uses Qdrant (local) by default; config for Postgres/Neo4j available

    # Add memories from conversation
    m.add("I prefer dark mode and use VS Code for Python", user_id="alice")
    m.add("I'm allergic to shellfish", user_id="alice")
    # Or from message list:
    # m.add(messages, user_id="alice")

    # Search
    results = m.search("what editor does alice use", user_id="alice")
    for r in results:
        print(f"[Mem0] {r['memory']}")

    # Get all
    all_mem = m.get_all(user_id="alice")

    # Delete
    if all_mem:
        m.delete(all_mem[0]["id"])


# ═══════════════════════════════════════════════════════════════════════
# 2. Zep — Temporal Knowledge Graph Memory
# ═══════════════════════════════════════════════════════════════════════
# pip install zep-cloud
# GitHub: https://github.com/getzep/zep
# Philosophy: Structured memory with temporal awareness — facts have timestamps,
#   contradictions auto-resolved, entity relationships tracked over time
#
# Best for: Complex agent workflows, multi-session, temporal reasoning
# Trade-off: More setup; cloud API key required (self-hosted also available)

def demo_zep():
    """Zep Cloud: structured memory with temporal knowledge graph."""
    try:
        from zep_cloud.client import Zep
    except ImportError:
        print("Install: pip install zep-cloud")
        return

    client = Zep(api_key=os.environ.get("ZEP_API_KEY", ""))

    # Add memory from conversation
    client.memory.add(
        user_id="alice",
        messages=[
            {"role": "user", "content": "I just moved to Tokyo"},
            {"role": "assistant", "content": "Welcome to Tokyo! How can I help?"},
        ],
    )

    # Search memories (hybrid: vector + graph)
    results = client.memory.search(user_id="alice", text="where does alice live")
    for r in results:
        print(f"[Zep] {r}")

    # Graph search — relationships and temporal facts
    graph_results = client.graph.search(user_id="alice", query="alice location")


# ═══════════════════════════════════════════════════════════════════════
# 3. Graphiti — Knowledge Graph Memory (by Zep team, standalone)
# ═══════════════════════════════════════════════════════════════════════
# pip install graphiti-core
# GitHub: https://github.com/getzep/graphiti (23k+ stars)
# Requires: Neo4j database for graph storage
# Philosophy: Extract (subject, predicate, object) triples from conversations,
#   store as temporal knowledge graph, detect contradictions automatically
#
# Best for: Knowledge-intensive agents, temporal reasoning, relationship tracking
# Trade-off: Requires Neo4j; more complex setup; best for graph-heavy use cases

async def demo_graphiti():
    """Graphiti: knowledge graph memory with temporal awareness."""
    try:
        from graphiti_core import Graphiti
        from graphiti_core.llm_client import OpenAIClient
    except ImportError:
        print("Install: pip install graphiti-core")
        return

    # Requires Neo4j running (e.g., docker run neo4j)
    graphiti = Graphiti(uri="bolt://localhost:7687", user="neo4j", password="test")

    # Add episode (raw conversation)
    await graphiti.add_episode(
        name="chat-001",
        episode_body="Alice moved from NYC to Tokyo last week. She works as a data engineer.",
        reference_id="session-1",
    )

    # Search — hybrid semantic + graph traversal
    results = await graphiti.search("where does alice live", num_results=5)
    for r in results:
        print(f"[Graphiti] {r}")

    await graphiti.close()


# ═══════════════════════════════════════════════════════════════════════
# Comparison Table
# ═══════════════════════════════════════════════════════════════════════

COMPARISON = """
| Feature        | Self-built (Level 1-7) | Mem0          | Zep           | Graphiti       |
|----------------|------------------------|---------------|---------------|----------------|
| Storage        | JSONL / SQLite         | Qdrant/Neo4j  | Postgres+Graph| Neo4j          |
| Extraction     | Manual LLM prompt      | Auto (LLM)    | Auto          | Auto (LLM)     |
| Temporal       | Manual (Level 4)       | No            | Yes           | Yes            |
| Contradictions | Manual (Level 4)       | Dedup only    | Auto-resolve  | Auto-invalidate|
| Graph          | SQLite (Level 4)       | Optional      | Built-in      | Core           |
| Setup          | Zero                   | pip install   | API key       | Neo4j + pip    |
| Best for       | Learning / custom      | Quick start   | Production    | Knowledge-heavy|
"""


if __name__ == "__main__":
    print(COMPARISON)
    print("\n--- Mem0 Demo ---")
    demo_mem0()
    # print("\n--- Zep Demo ---")
    # demo_zep()
    # print("\n--- Graphiti Demo ---")
    # import asyncio; asyncio.run(demo_graphiti())
