"""
nanoMemory Level 7: Agentic Memory — Autonomous Memory Lifecycle
~120 lines. The agent itself decides when to save, delete, update, or search memories.

Key shift: Levels 1-6 auto-extract memories after every turn. This level gives
the agent explicit memory TOOLS and lets it decide when to use them via function calling.
The agent can choose to remember, forget, correct, or look up — just like a human.

This mirrors the "agentic memory" paradigm where memory is not a passive store
but an active resource the agent manages as part of its reasoning loop.

References:
  - Agentic Memory (Yu et al., 2025) "Omni-Memory: Towards Eternal Life
    for Large Language Model Agents" — memory as first-class operable objects
    with CRUD operations controlled by the agent
    https://arxiv.org/abs/2502.12110
  - MemGPT (Packer et al., 2023) "MemGPT: Towards LLMs as Operating Systems"
    — virtual context management with implicit/explicit memory tiers,
    agent self-manages memory pages via function calls
    https://arxiv.org/abs/2310.08560 (ACL 2024 Outstanding Paper)
  - M^p (2025) "Exploring Agent Procedural Memory" — build/retrieval/update
    lifecycle with continuous correction and deprecation
    https://arxiv.org/abs/2508.06433
  - Survey: "Memory in the Age of AI Agents" §4 Dynamics — formation,
    evolution (consolidation, abstraction, forgetting), retrieval
    https://arxiv.org/abs/2512.13564
"""
import os
import json
from datetime import datetime
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MEMORY_FILE = "memory_lifecycle.jsonl"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_save",
            "description": "Save an important fact or preference to long-term memory. Use when you learn something worth remembering.",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string", "description": "The fact to remember"}},
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "Delete an outdated or incorrect memory. Use when you realize a past memory is wrong or no longer relevant.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Memory ID to delete"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_update",
            "description": "Update an existing memory with new information. Use when a fact has changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Memory ID to update"},
                    "new_fact": {"type": "string", "description": "Updated fact text"},
                },
                "required": ["id", "new_fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search long-term memories for relevant information. Use before answering if you think past context might help.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to search for"}},
                "required": ["query"],
            },
        },
    },
]


# ── Memory Store (simple JSONL) ─────────────────────────────────────────

def load_memories():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE) as f:
        memories = {}
        for line in f:
            m = json.loads(line.strip())
            if not m.get("deleted"):
                memories[m["id"]] = m
        return memories


def save_to_store(fact: str) -> str:
    import uuid
    mid = str(uuid.uuid4())[:8]
    entry = {"id": mid, "text": fact, "timestamp": datetime.now().isoformat()}
    with open(MEMORY_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[Memory:SAVE] #{mid} {fact[:60]}")
    return mid


def delete_from_store(mid: str):
    entry = {"id": mid, "deleted": True, "timestamp": datetime.now().isoformat()}
    with open(MEMORY_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[Memory:DELETE] #{mid}")


def update_in_store(mid: str, new_fact: str):
    entry = {"id": mid, "text": new_fact, "updated": True, "timestamp": datetime.now().isoformat()}
    with open(MEMORY_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[Memory:UPDATE] #{mid} → {new_fact[:60]}")


def search_store(query: str, top_k: int = 5) -> list[dict]:
    keywords = set(query.lower().split())
    memories = load_memories()
    scored = []
    for m in memories.values():
        words = set(m.get("text", "").lower().split())
        overlap = len(keywords & words)
        if overlap > 0:
            scored.append((m, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scored[:top_k]]


# ── Tool Execution ─────────────────────────────────────────────────────

def execute_tool(name: str, args: dict) -> str:
    if name == "memory_save":
        mid = save_to_store(args["fact"])
        return f"Saved memory #{mid}: {args['fact']}"
    elif name == "memory_delete":
        delete_from_store(args["id"])
        return f"Deleted memory #{args['id']}"
    elif name == "memory_update":
        update_in_store(args["id"], args["new_fact"])
        return f"Updated memory #{args['id']}: {args['new_fact']}"
    elif name == "memory_search":
        results = search_store(args["query"])
        if not results:
            return "No relevant memories found."
        return "\n".join(f"#{m['id']}: {m['text']}" for m in results)
    return "Unknown tool"


# ── Agent with Autonomous Memory Control ─────────────────────────────────

def run_agent_with_memory(user_message: str, max_iterations: int = 10) -> str:
    messages = [
        {"role": "system", "content": """You are a helpful assistant with long-term memory.
You have tools to save, delete, update, and search your memories.
- Use memory_save when you learn an important fact about the user.
- Use memory_delete when a past memory is outdated.
- Use memory_update when a fact has changed.
- Use memory_search when past context might be relevant.
Be proactive about memory management. Be concise."""},
        {"role": "user", "content": user_message},
    ]

    ai_response = ""
    for _ in range(max_iterations):
        response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
        message = response.choices[0].message
        messages.append(message)
        if message.content:
            ai_response = message.content
        if not message.tool_calls:
            break
        for tc in message.tool_calls:
            import json as _json
            args = _json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return ai_response


if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello"
    print(run_agent_with_memory(task))
